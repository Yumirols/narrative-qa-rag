# -*- coding: utf-8 -*-
"""
第 15 讲 · 推理优化:让模型跑得更便宜(可离线复现的最小演示)
================================================================

两个阶段的瓶颈完全不同:
    Prefill(预填充):整段提示一次并行算完 —— 计算密集,TTFT(首 token 延迟)受它影响
    Decode(解码):每步只算一个新 token,但要读全部历史 KV —— 显存带宽密集,TPOT 受它影响

KV Cache(拿显存换计算):不缓存则每步重算全部历史(计算量平方级增长);
缓存后每步只算新 token 的 Q/K/V,代价是要存每一层 × 每个头 × 每个位置的 K 和 V:

    KV Cache 大小 = 2(K与V) × 层数 × KV头数 × 每头维度 × 序列长度 × 批大小 × 每元素字节数

PagedAttention 借操作系统分页:把缓存切成固定大小的块,不连续存放 + 块表记录,
碎片浪费从 60–80% 降到 4% 以下(vLLM 的核心)。

量化:把连续浮点映射到少数档位。x_q = round(x / s),反量化 x ≈ s·x_q。
FP16(2B)→INT8(1B)→INT4(0.5B);decode 是带宽密集,权重小 4 倍 ≈ 搬运快 4 倍。

投机解码:小草稿模型猜 k 个 token,大模型一次前向并行验证;按 min(1, p/q) 接受,
输出分布与直接用大模型采样完全相同 —— 无损加速,期望每步产出 (1−α^(k+1))/(1−α) 个 token。

IVF(倒排文件索引):先用聚类把向量分成 K 个簇,查询只在最相近的 nprobe 个簇里搜
—— 搜索范围缩小几十倍,nprobe 是「更准 vs 更快」的旋钮。
"""
import sys
import time

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------- 一、KV Cache 显存账(考点) ----------
def kv_cache_bytes(layers, kv_heads, d_head, seq_len, batch, bytes_per=2):
    """2(K与V) × 层数 × KV头数 × 每头维度 × 序列长 × 批大小 × 每元素字节。"""
    return 2 * layers * kv_heads * d_head * seq_len * batch * bytes_per


def demo_kv_cache():
    print("── [1] KV Cache 显存账:上下文越长、并发越高,越贵(课件第 12 页)──")
    rows = [(32, 32, 128, 4096, 1, "单条请求 4K 上下文(MHA)"),
            (32, 32, 128, 4096, 8, "同上,批 8"),
            (32, 32, 128, 4096, 16, "同上,批 16"),
            (32, 32, 128, 32768, 4, "长上下文 32K + 批 4"),
            (32, 4, 128, 4096, 16, "GQA:KV 头 32→4(Qwen2.5-0.5B 同思路)")]
    for L, H, dh, S, B, tag in rows:
        gb = kv_cache_bytes(L, H, dh, S, B) / 1e9
        print(f"  层{L}·KV头{H}·d_head{dh}·len{S:>6}·batch{B:>2}: {gb:6.1f} GB   {tag}")
    print("  结论:缓存 ∝ 序列长 × 批大小 → 长上下文与高并发是一对矛盾;")
    print("  省显存四招:KV 量化 / 分页(PagedAttention) / 前缀共享 / GQA-MQA 少存 KV")


# ---------- 二、量化:精度换显存与带宽 ----------
def quantize(x, bits):
    """对称均匀量化:x_q = round(x/s),s = max|x|/(2^(b-1)−1);反量化 x̂ = s·x_q。"""
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(x).max() / qmax
    x_q = np.clip(np.round(x / s), -qmax, qmax).astype(np.int32)
    return s * x_q, s


def demo_quantize():
    print("── [2] 量化:INT8 几乎无损,INT4 粗一档(课件第 14 页)──")
    rng = np.random.default_rng(0)
    w = rng.standard_normal(10000) * 0.02          # 一层典型权重分布
    for bits in [8, 4]:
        x_hat, s = quantize(w, bits)
        rel = np.linalg.norm(w - x_hat) / np.linalg.norm(w)
        print(f"  FP16 → INT{bits:<2} 缩放因子 s={s:.5f}  相对误差 = {rel:.4f}  "
              f"显存 {2}B → {bits / 8:.2f}B/参数")
    print("  权重大多集中在 0 附近,小档位误差在多层累加后并不显著放大;")
    print("  decode 是显存带宽密集:权重小 4 倍 → 搬运少 4 倍 → 解码加速(INT4 时)")


# ---------- 三、投机解码:无损换并行 ----------
def demo_speculative():
    print("── [3] 投机解码:草稿猜 k 个,大模型并行验证,按 min(1,p/q) 接受 ──")
    print("  期望每步产出 token 数 = (1 − α^(k+1)) / (1 − α),α = 草稿与目标的一致率:")
    for alpha in [0.5, 0.7, 0.9]:
        for k in [3, 5]:
            expect = (1 - alpha ** (k + 1)) / (1 - alpha)
            print(f"  一致率 α={alpha},草稿步长 k={k}: 期望加速 ≈ {expect:.2f}×")
    print("  与量化的对比记忆:量化是「有损换资源」,投机解码是「无损换并行」,可叠加;")
    print("  草稿太差(α 低)反而更慢 —— 加速比完全取决于草稿模型猜得准不准")


# ---------- 四、IVF:近似最近邻,拿一点召回换几十倍速度 ----------
def kmeans(X, K, iters=10, seed=0):
    """内存安全的 k-means:向量已 L2 归一化,欧氏最近 = 余弦最大,直接用内积。"""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=K, replace=False)]
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(K):
            m = assign == j
            if m.any():
                c = X[m].mean(0)
                C[j] = c / np.linalg.norm(c)
    return C


def demo_ivf(n=100_000, K=64):
    print(f"── [4] IVF:先分簇,查询只搜最近的 nprobe 个簇(课件第 24 页)──")
    rng = np.random.default_rng(0)
    # 模拟真实嵌入:32 个语义簇(同类文本向量互相靠近),查询取自簇内邻近点
    centers = rng.standard_normal((32, 32)) * 2.0
    X = np.concatenate([c + 0.15 * rng.standard_normal((n // 32, 32)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    src = X[rng.choice(len(X), size=50, replace=False)]
    Q = src + 0.05 * rng.standard_normal(src.shape)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    centroids = kmeans(X, K)
    cluster_of = np.argmax(X @ centroids.T, axis=1)

    t0 = time.perf_counter()
    gt = np.argsort(-X @ Q.T, axis=0)[:10]             # 精确检索的真 top-10
    t_exact = time.perf_counter() - t0
    print(f"  {n} 条向量、{K} 个簇,50 条查询,指标为 Recall@10 / 耗时(精确检索={t_exact * 1000:.0f}ms):")
    for nprobe in [1, 2, 4]:
        t0 = time.perf_counter()
        hits = 0
        qc = np.argmax(Q @ centroids.T, axis=1)
        probe = np.argsort(-(Q @ centroids.T), axis=1)[:, :nprobe]   # 最近的 nprobe 个簇
        for qi in range(len(Q)):
            cand = np.where(np.isin(cluster_of, probe[qi]))[0]       # 只在簇内暴力搜
            local = np.argsort(-(X[cand] @ Q[qi]))[:10]
            hits += len(set(cand[local]) & set(gt[:, qi]))
        t = time.perf_counter() - t0
        print(f"  nprobe={nprobe}: Recall@10 = {hits / (10 * len(Q)):.3f}   耗时 {t / t_exact:.3f}×")
    print("  nprobe 调大→更准更慢,调小→更快可能漏 —— 没有免费的午餐,参数要按延迟要求调")


if __name__ == "__main__":
    demo_kv_cache()
    demo_quantize()
    demo_speculative()
    demo_ivf()
    print("\n所有自检通过 ✔")
