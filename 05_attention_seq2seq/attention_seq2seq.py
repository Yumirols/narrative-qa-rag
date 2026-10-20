# -*- coding: utf-8 -*-
"""
第 10 讲 · Seq2Seq 与注意力机制(Attention)
==================================================

编码器-解码器把「变长进、变长出」变成现实:
    编码器:  h_t = f_enc(h_{t-1}, x_t)              读完源句,输出隐状态序列 h_1..h_n
    解码器:  s_t = f_dec(s_{t-1}, e(y_{t-1}), c)    以上下文为条件自回归逐词生成

无注意力版本把整句压成一个定长向量 c = h_n:c 的容量由维度决定(常数),
而源句信息量随手长 n 线性增长 —— 「容量常数 vs 需求线性」写在架构里,
是长句 BLEU 断崖式下跌的根因(Bahdanau et al., 2015, Figure 2)。
加深加宽 / 双向编码 / 倒序输入都只是缓解,治不了本。

注意力(Attention)的解法是不压缩:把全部 h_1..h_n 留着,解码每步按需加权读取。
三步(必背):

    ① 打分    e_tj = a(s_{t-1}, h_j)                 j = 1..n,原始相关性分数
    ② 归一化  α_tj = exp(e_tj) / Σ_k exp(e_tk)       softmax 对「源位置 j」做
    ③ 加权    c_t  = Σ_j α_tj · h_j                  动态上下文,与 h_j 同维

关键认知:注意力不增加任何信息(用的还是那批 h_j),它改变的是「读取方式」——
从「一次压缩、全程共用」变成「全部保留、按需加权读取」。α 是加权混合比例,
不是「选中某个词的概率」;对齐热图是诊断线索,不是因果解释。

四种打分函数(Luong 2015 / Vaswani 2017):
    dot             e = sᵀ h                无参数、可批量矩阵乘,现代主流
    general         e = sᵀ W h              允许 s、h 维度不同
    concat/additive e = vᵀ tanh(W[s; h])    Bahdanau 原版,表达力最强、参数最多
    scaled dot      e = sᵀ h / √d           大维度下防 softmax 饱和(第 11 讲主角)

本文件用 numpy 复现课件第 24-25 页的两道手算例题,并演示软对齐与信息瓶颈。
"""
import sys

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------- 通用工具 ----------
def softmax(x, axis=-1):
    """数值稳定的 softmax:先减最大值,防止 exp 上溢。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------- 四种打分函数 ----------
def score_dot(s, H):
    """e_j = sᵀ h_j。无参数,一次矩阵乘算完 —— 工程默认。"""
    return H @ s


def score_scaled_dot(s, H):
    """e_j = sᵀ h_j / √d。点积的改良:防大维度下 softmax 饱和。"""
    return score_dot(s, H) / np.sqrt(H.shape[1])


def score_general(s, H, W):
    """e_j = sᵀ W h_j。允许 s 与 h 维度不同,表达力更强。"""
    return H @ W @ s


def score_concat(s, H, W, v):
    """e_j = vᵀ tanh(W [s; h_j])。Bahdanau 原版:拼接过一层再打分。"""
    n = H.shape[0]
    sh = np.concatenate([np.tile(s, (n, 1)), H], axis=1)   # n × (dim_s + dim_h)
    return np.tanh(sh @ W) @ v


# ---------- 注意力三步 ----------
def attention(s, H, score_fn=score_dot):
    """给定解码状态 s 与编码器隐状态序列 H(n×d),返回 (α, c_t)。

    ① 打分 → ② 对源位置做 softmax → ③ 加权求和。
    """
    e = score_fn(s, H)                 # ① 打分:n 维,可正可负、未归一化
    alpha = softmax(e)                 # ② 归一化:非负、和为 1
    context = alpha @ H                # ③ 加权求和:d 维,与 h_j 同维
    return alpha, context


# ---------- 演示 ----------
if __name__ == "__main__":
    print("── [1] 手算复现:打分 → softmax → 加权求和(课件第 24-25 页)──")
    H = np.array([[1.0, 0.0],      # h_1
                  [0.0, 1.0],      # h_2
                  [1.0, 1.0]])     # h_3
    for s, a_ref, c_ref in [(np.array([1.0, 0.5]), [0.3072, 0.1863, 0.5065], [0.8137, 0.6928]),
                            (np.array([0.5, 1.0]), [0.1863, 0.3072, 0.5065], [0.6928, 0.8137])]:
        e = score_dot(s, H)
        alpha, c = attention(s, H)
        print(f"  s = {s} -> e = {e} -> α = {np.round(alpha, 4)} -> c = {np.round(c, 4)}")
        assert np.allclose(alpha, a_ref, atol=2e-4)
        assert np.allclose(c, c_ref, atol=2e-4)
        assert abs(alpha.sum() - 1.0) < 1e-9
    print("  第一遍 α=[0.31,0.19,0.51];第二遍 s 变了,α=[0.19,0.31,0.51]:")
    print("  第 2 位反超第 1 位 —— 权重随查询变化,这就是「动态上下文 c_t」")
    print("  且 α 是软的:三个位置都拿到非零权重(软对齐,不是硬选一个词)")

    print("── [2] 四种打分函数对比(同一组 H、s)──")
    s = np.array([1.0, 0.5])
    W_g = np.array([[1.0, 0.2], [0.2, 1.0]])               # general 的 W
    W_c = np.array([[0.5, 0.4, 0.1], [0.4, 0.5, 0.1],       # concat 的 W:作用于 [s; h](4 维)
                    [0.3, 0.3, 0.4], [0.1, 0.1, 0.3]])      #   → 3 维隐层
    v_c = np.array([0.5, 0.5, 0.5])                         # concat 的 v
    for name, fn in [("dot          ", lambda: score_dot(s, H)),
                     ("scaled dot   ", lambda: score_scaled_dot(s, H)),
                     ("general      ", lambda: score_general(s, H, W_g)),
                     ("concat       ", lambda: score_concat(s, H, W_c, v_c))]:
        e = fn()
        a, _ = attention(s, H, score_fn=lambda _, _H: e)
        print(f"  {name} e = {np.round(e, 3)}  α = {np.round(a, 3)}")
    print("  排序可能不同,但三步流程完全一致;多数任务上最终 BLEU 差别不大,")
    print("  工程上优先选无参数、可批量矩阵乘的 dot / 缩放点积")

    print("── [3] 软对齐热图:注意力矩阵就是「对齐副产品」──")
    src = ["我", "爱", "自然", "语言", "处理"]
    tgt = ["I", "love", "NLP"]                              # NLP ↔ 自然+语言(一对多)
    Hx = np.array([[1.0, 0.1, 0.0],      # 我
                   [0.1, 1.0, 0.0],      # 爱
                   [0.0, 0.2, 1.0],      # 自然
                   [0.0, 0.2, 1.0],      # 语言
                   [0.1, 0.0, 1.2]])     # 处理
    sx = np.array([[1.0, 0.2, 0.0],      # 解码 I 时的状态(贴近「我」)
                   [0.2, 1.0, 0.0],      # 解码 love 时的状态
                   [0.0, 0.1, 1.0]])     # 解码 NLP 时的状态(贴近「自然/语言」)
    A = np.stack([attention(si, Hx)[0] for si in sx])       # 3×5 注意力矩阵
    print(f"  {'':6}" + "".join(f"{w:>7}" for w in src))
    for w, row in zip(tgt, A):
        bar = "".join("█" if a > 0.45 else ("▒" if a > 0.12 else "░") for a in row)
        print(f"  {w:6}{bar}   {np.round(row, 3)}")
    print(f"  每行和 = {A.sum(axis=1).round(6)}(行 = 目标词,列 = 源词)")
    print("  读图:I/love 近似单点对齐;NLP 一行摊在 自然/语言/处理 上 ——")
    print("  α 可以一对多、多对一;某列整列很浅 ≈ 该源词可能被漏译(调试线索)")

    print("── [4] 信息瓶颈:定长向量装不下长句(容量常数 vs 需求线性)──")
    rng = np.random.default_rng(0)
    for n in [2, 4, 8, 16, 32, 64]:
        Hn = rng.standard_normal((n, 16))                   # n 个词的隐状态
        Hn /= np.linalg.norm(Hn, axis=1, keepdims=True)
        c = Hn.mean(axis=0)                                 # 无注意力:整句压成均值向量
        c /= np.linalg.norm(c)
        cos = Hn @ c                                        # 各词与压缩向量的相似度
        print(f"  句长 n={n:3}  平均 cos(h_i, c) = {cos.mean():.3f}   最弱词 = {cos.min():.3f}")
    print("  n 越大,压缩向量与每个词越「不像」—— 定长 c 的信息随句长被摊薄;")
    print("  注意力的出路:不压缩,保留全部 h_j 按需读取(长句 BLEU 不再断崖)")

    print("\n所有自检通过 ✔")
