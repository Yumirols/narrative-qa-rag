# -*- coding: utf-8 -*-
"""
第 11 讲 · Transformer 原理精讲(最小实现 + 数值实验)
========================================================

一个公式:Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
四步流程:投影 → 打分并缩放 → 按行 softmax → 加权 V。
「多头」「自注意力 / 交叉注意力」用的都是这四步,区别只在 Q/K/V 从哪来、
有没有 mask —— 理解「从哪来」比记公式更重要。

为什么要除以 √d_k(论文脚注 4):q、k 各分量独立、均值 0、方差 1 时,
点积 q·k = Σ q_i k_i 的均值为 0、方差为 d_k,标准差 √d_k;分数过大会让
softmax 近似 one-hot(饱和),雅可比趋近 0、梯度消失。除以 √d_k 把方差拉回 1。

为什么必须有位置编码:自注意力是「置换等变」的 —— 打乱输入行序,输出跟着
打乱,公式里没有任何绝对位置项。而「猫追鱼」≠「鱼追猫」,语序必须显式注入。

残差给通路(out = x + Sublayer(x),∂out/∂x = I + ∂Sublayer/∂x),
LayerNorm 给稳定(逐样本、对 d_model 维特征归一化)—— 两者合起来
才让 30 个子层的深堆可以训练。base 配方:N=6,d_model=512,h=8,
d_k=d_v=64,d_ff=2048(FFN 占每层参数约 2/3 —— 被低估的大户)。

本文件用 numpy 复现课件第 19-21 页的手算例题,并完成课后三组数值实验
(√d_k 饱和 / causal mask / 置换等变)。
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
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """LN(x) = γ ⊙ (x − μ)/√(σ²+ε) + β。对最后一个维(特征)统计,
    不是对 batch —— 这是 LN 与 BatchNorm 的关键区别。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    g = np.ones_like(x) if gamma is None else gamma
    b = np.zeros_like(x) if beta is None else beta
    return g * xhat + b


def rms_norm(x, gamma=None, eps=1e-5):
    """RMSNorm = γ ⊙ x / RMS(x),RMS = √(mean(x²))。去掉 re-centering,
    只保留 re-scaling(Llama / Qwen / GLM 采用)。"""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    g = np.ones_like(x) if gamma is None else gamma
    return g * x / rms


# ---------- 核心:缩放点积注意力(四步) ----------
def scaled_dot_attention(Q, K, V, mask=None):
    """Q(n×d_k), K(n×d_k), V(n×d_v) → 输出(n×d_v)与注意力矩阵 P(n×n)。

    ① 打分 S = QKᵀ  ② 缩放 / √d_k  ③ 按行 softmax  ④ 加权 P·V
    mask:True 处为禁止位置(加 −1e9,必须在 softmax 之前)。
    """
    d_k = Q.shape[1]
    S = Q @ K.T / np.sqrt(d_k)                     # ①②
    if mask is not None:
        S = np.where(mask, -1e9, S)                # 禁止位 = −∞(实现用 −1e9)
    P = softmax(S, axis=-1)                        # ③ 按行:行=query,列=key
    return P @ V, P                                # ④


def multi_head_attention(X, WQs, WKs, WVs, WO, mask=None):
    """多头 = 分空间,不是分模型:把 d_model 切成 h 个 d_k 子空间,各自做一次
    完整注意力,拼接后过 W^O 融合。参数量等于一个 d_model×d_model 投影。"""
    heads = [scaled_dot_attention(X @ WQ, X @ WK, X @ WV, mask)[0]
             for WQ, WK, WV in zip(WQs, WKs, WVs)]
    return np.concatenate(heads, axis=-1) @ WO


def positional_encoding(max_len, d_model):
    """PE(pos,2i)=sin(pos/10000^(2i/d)),PE(pos,2i+1)=cos(同)。
    偶数维 sin、奇数维 cos;i 越大频率越低 —— 像 256 个转速不同的表针,
    组合出每个位置唯一的「指纹」;相对位置可由线性(旋转)变换表示。"""
    pos = np.arange(max_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (i // 2) * 2 / d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 0::2])
    return pe


# ---------- 演示与实验 ----------
if __name__ == "__main__":
    print("── [1] 手算复现:单头自注意力四步(课件第 19-21 页,猫 追 鱼,d=2)──")
    X = np.array([[1.0, 0.0],      # 猫
                  [1.0, 1.0],      # 追
                  [0.0, 1.0]])     # 鱼
    WQ = np.eye(2)                                  # Q = X
    WK = np.array([[0.0, 1.0], [1.0, 1.0]])         # [a,b] -> [b, a+b]
    WV = np.array([[2.0, 0.0], [1.0, 3.0]])         # [a,b] -> [2a+b, 3b]
    Q, K, V = X @ WQ, X @ WK, X @ WV
    O, P = scaled_dot_attention(Q, K, V)
    print(f"  Q = {Q.tolist()}  K = {K.tolist()}  V = {V.tolist()}")
    print(f"  S = QKᵀ/√2 =\n{np.round(Q @ K.T / np.sqrt(2), 3)}")
    print(f"  P(逐行 softmax,每行和=1) =\n{np.round(P, 3)}")
    print(f"  O = P·V =\n{np.round(O, 3)}")
    assert np.allclose(P, [[0.198, 0.401, 0.401],
                           [0.140, 0.576, 0.284],
                           [0.248, 0.503, 0.248]], atol=2e-3)
    # 课件 O 第三行印作 2.253,是用四舍五入后的 P 算的;精确值 2.2551
    assert np.allclose(O, [[2.0000, 2.4066], [2.2917, 2.5797], [2.2551, 2.2551]], atol=1e-3)
    print("  注意:「猫·猫」= 0(两向量正交)—— 对角线不一定最大,自注意力不是只看自己")

    print("── [2] 实验①√d_k:不缩放会 softmax 饱和、梯度趋近 0(课后必做)──")
    rng = np.random.default_rng(0)
    n, d_k = 64, 64
    Qr, Kr, Vr = rng.standard_normal((n, d_k)), rng.standard_normal((n, d_k)), rng.standard_normal((n, 64))
    S = Qr @ Kr.T
    for tag, Pm in [("不缩放   ", softmax(S)), ("除以 √64 ", softmax(S / 8))]:
        p_max = Pm.max(axis=1).mean()
        # softmax 雅可比 J = diag(p) − p pᵀ,其 Frobenius 范数反映梯度规模
        jac = np.stack([np.diag(p) - np.outer(p, p) for p in Pm])
        print(f"  {tag}: 平均最大权重 = {p_max:.4f}   雅可比范数(梯度规模) = {np.linalg.norm(jac):.2f}")
    print("  不缩放:分布近似 one-hot(最大权重→1)、梯度→0,模型学不动;")
    print("  点积方差 = d_k → 标准差 √d_k → 除以 √d_k 拉回方差 1。不是为快,也不为变概率")

    print("── [3] 实验②causal mask:先遮分数,再归一化(课后必做)──")
    n = 4
    causal = np.triu(np.ones((n, n), dtype=bool), k=1)   # 上三角 = 未来 = 禁止
    _, Pc = scaled_dot_attention(rng.standard_normal((n, 8)),
                                 rng.standard_normal((n, 8)),
                                 rng.standard_normal((n, 8)), mask=causal)
    print(f"  上三角权重之和 = {Pc[np.triu_indices(n, k=1)].sum():.2e}(应为 0)")
    print(f"  每行之和 = {Pc.sum(axis=1).round(6)}(仍为 1,只在可见位置内分配)")
    assert np.allclose(Pc[np.triu_indices(n, k=1)], 0.0)
    assert np.allclose(Pc.sum(axis=1), 1.0)
    print("  mask 必须加在 softmax 之前的分数上(−1e9);作用在输出上会破坏归一化")

    print("── [4] 多头:同一份输入,不同子空间,不同的注意力分布 ──")
    d_model, h, dkh = 16, 4, 4
    Xh = rng.standard_normal((5, d_model))
    WQs = [rng.standard_normal((d_model, dkh)) / np.sqrt(d_model) for _ in range(h)]
    WKs = [rng.standard_normal((d_model, dkh)) / np.sqrt(d_model) for _ in range(h)]
    WVs = [rng.standard_normal((d_model, dkh)) / np.sqrt(d_model) for _ in range(h)]
    WO = rng.standard_normal((h * dkh, d_model)) / np.sqrt(h * dkh)
    Oh = multi_head_attention(Xh, WQs, WKs, WVs, WO)
    P0 = scaled_dot_attention(Xh @ WQs[0], Xh @ WKs[0], Xh @ WVs[0])[1]
    P1 = scaled_dot_attention(Xh @ WQs[1], Xh @ WKs[1], Xh @ WVs[1])[1]
    print(f"  输出形状 = {Oh.shape}(n×d_model);头 0 与头 1 的注意力矩阵相同吗? "
          f"{np.allclose(P0, P1)}")
    print(f"  参数账:4 个 {d_model}×{d_model} 等效投影 + FFN 2×{d_model}×{4 * d_model}"
          f" → FFN 约占每层参数 2/3(注意力是招牌,FFN 才是参数大户)")

    print("── [5] 实验③置换等变:没有 PE 的注意力是「位置盲」的 ──")
    Xw = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])   # 猫 追 鱼
    Xr = Xw[::-1]                                          # 鱼 追 猫
    Wq = rng.standard_normal((2, 2))
    O1, _ = scaled_dot_attention(Xw @ Wq, Xw @ Wq, Xw)
    O2, _ = scaled_dot_attention(Xr @ Wq, Xr @ Wq, Xr)
    same_wo_pe = np.allclose(O2, O1[::-1])
    print(f"  不加 PE:「鱼追猫」的输出是「猫追鱼」输出的行交换吗? {same_wo_pe}")
    assert same_wo_pe
    pe3 = positional_encoding(3, 2)                        # 位置固定 0,1,2
    O3, _ = scaled_dot_attention((Xw + pe3) @ Wq, (Xw + pe3) @ Wq, Xw + pe3)
    O4, _ = scaled_dot_attention((Xr + pe3) @ Wq, (Xr + pe3) @ Wq, Xr + pe3)
    same_w_pe = np.allclose(O4, O3[::-1])
    print(f"  加上 PE 后,两句话的输出还是简单的行交换吗?         {same_w_pe}")
    assert not same_w_pe
    print("  不加 PE:两句话在模型眼里是同一「袋词」;语序信息全部来自 PE —— 不可省")

    print("── [6] 正弦位置编码:频率阶梯 + 相对位置的线性(旋转)表示 ──")
    PE = positional_encoding(6, 8)
    print(f"  PE[0:3, 0:4] =\n{np.round(PE[:3, :4], 3)}")
    p, k, i = 3, 2, 1                                       # 验证 PE(p+k) = R(k)·PE(p)
    theta = 1.0 / 10000 ** (2 * i / 8)
    R = np.array([[np.cos(k * theta), np.sin(k * theta)],
                  [-np.sin(k * theta), np.cos(k * theta)]])
    lhs = PE[p + k, 2 * i:2 * i + 2]
    rhs = R @ PE[p, 2 * i:2 * i + 2]
    print(f"  PE({p}+{k}, 维{i}) = 旋转矩阵 · PE({p}, 维{i}):{np.allclose(lhs, rhs)}")
    assert np.allclose(lhs, rhs)
    print("  三角恒等式保证相对偏移 = 一次旋转 —— RoPE 的直接思想源头")

    print("── [7] LayerNorm / RMSNorm 手算(课件第 34 页:x = [1, 2, 3])──")
    x = np.array([1.0, 2.0, 3.0])
    print(f"  LayerNorm({x.tolist()}) = {np.round(layer_norm(x), 4)}"
          f"   [参考: μ=2, σ≈0.816 → [-1.225, 0, 1.225]]")
    print(f"  RMSNorm  ({x.tolist()}) = {np.round(rms_norm(x), 4)}   (不减均值,少一半统计量)")
    assert np.allclose(layer_norm(x), [-1.2247, 0.0, 1.2247], atol=1e-3)

    print("\n所有自检通过 ✔")
