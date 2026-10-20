# -*- coding: utf-8 -*-
"""
第 12 讲 · 预训练语言模型:两类自监督目标(AR vs MLM)
==========================================================

范式:预训练 + 微调。「昂贵的事只做一次,便宜的事做一万次」——
两个前提:参数可迁移;标签可自动生成(自监督)。自监督用规则把无标注序列
x 变成 (输入 g(x), 标签 f(x)),于是「有文本就能学」,NLP 才吃到了互联网规模数据。

两类目标(目标函数决定能力边界):

  自回归 AR(GPT 路线)   P(x) = Π_t P(x_t | x_{<t}),因果掩码,严格单向
                         信号密度 100%(每个位置都产生损失),天然能生成
  自编码 AE(BERT 路线)  MLM:随机选 15% 位置,按 80/10/10 替换:
                         80% → [MASK];10% → 随机词;10% → 保持不变
                         预测时看除自身外的左右全部上下文(双向全可见)
                         信号密度只有 ~15%;输入含 [MASK],不能直接生成

本文件实现一个单头自注意力小模型(d 维、无 FFN),手推反向传播、
梯度检查,然后在同一小语料上分别按两种目标训练,直观对比:
  · 因果掩码 vs 双向全可见的注意力矩阵
  · 每个epoch 的监督信号数量(100% vs ~15%)
  · AR 训练完能「续写」,MLM 训练完能「完形填空」—— 训练什么就会什么
最后演示同一底座接四种任务头(分类 / 序列标注 / 句对 / 抽取式问答)。
"""
import sys

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BOS, EOS, MASK = "<BOS>", "<EOS>", "[MASK]"

SENTENCES = [
    "我爱东北大学", "东北大学在沈阳", "我爱自然语言处理", "我们学习语言模型",
    "东北大学学生爱学习", "沈阳有东北大学", "语言模型处理自然语言", "我们爱学习",
]


# ---------- 数据 ----------
def build_vocab():
    chars = sorted({c for s in SENTENCES for c in s})
    vocab = [BOS, EOS, MASK] + chars
    return vocab, {w: i for i, w in enumerate(vocab)}


# ---------- 模型:单头自注意力语言模型 ----------
class TinyAttnLM:
    """E(嵌入) + 单头自注意力 + 输出投影。手推反向传播,公式见 backward。"""

    def __init__(self, V, d=24, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.d = V, d
        self.E = 0.1 * rng.standard_normal((V, d))
        self.WQ = 0.1 * rng.standard_normal((d, d))
        self.WK = 0.1 * rng.standard_normal((d, d))
        self.WV = 0.1 * rng.standard_normal((d, d))
        self.Wo = 0.01 * rng.standard_normal((d, V))
        self.params = ["E", "WQ", "WK", "WV", "Wo"]

    def forward(self, ids_in, causal=False):
        """返回 logits(T×V)与缓存。causal=True 时位置 i 只能看 ≤ i。"""
        X = self.E[ids_in]                                  # T×d
        QX, KX, VX = X @ self.WQ, X @ self.WK, X @ self.WV
        S = QX @ KX.T / np.sqrt(self.d)
        if causal:                                          # 因果掩码:上三角 = −∞
            S = S + np.triu(np.full(S.shape, -1e9), k=1)
        P = np.exp(S - S.max(axis=-1, keepdims=True))
        P /= P.sum(axis=-1, keepdims=True)
        h = P @ VX                                          # T×d
        logits = h @ self.Wo                                # T×V
        return logits, (ids_in, X, QX, KX, VX, P, h)

    def backward(self, dlogits, cache, causal=False):
        """链式法则回传。softmax 的反向:dS = P ⊙ (dP − Σ(dP⊙P))。"""
        ids_in, X, QX, KX, VX, P, h = cache
        grads = {}
        grads["Wo"] = h.T @ dlogits
        dh = dlogits @ self.Wo.T
        dVX = P.T @ dh
        dP = dh @ VX.T
        dS = P * (dP - (dP * P).sum(-1, keepdims=True))
        dQX = dS @ KX / np.sqrt(self.d)
        dKX = dS.T @ QX / np.sqrt(self.d)
        dX = dQX @ self.WQ.T + dKX @ self.WK.T + dVX @ self.WV.T
        grads["WQ"], grads["WK"], grads["WV"] = X.T @ dQX, X.T @ dKX, X.T @ dVX
        dE = np.zeros_like(self.E)
        np.add.at(dE, ids_in, dX)                           # 同一下标重复出现要累加
        grads["E"] = dE
        return grads

    def loss_and_grads(self, ids_in, positions, targets, causal):
        """positions=None 表示所有位置都计损失(AR);否则只算被掩码位置(MLM)。"""
        logits, cache = self.forward(ids_in, causal)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        n = len(positions) if positions is not None else len(targets)
        idx = positions if positions is not None else range(len(targets))
        loss = -np.mean([np.log(probs[t, targets[t]] + 1e-12) for t in idx])
        dlogits = probs.copy()
        for t in idx:
            dlogits[t, targets[t]] -= 1.0
        dlogits /= n
        return float(loss), self.backward(dlogits, cache, causal)

    def sgd_step(self, grads, lr, clip=1.0):
        """SGD + 全局范数裁剪。朴素 SGD 下 logits 会无界漂移(过拟合后一旦
        翻转 argmax,交叉熵瞬间爆炸),裁剪 + 学习率衰减是教学实现的最小稳定器。"""
        total = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / max(total, 1e-12))
        for name in self.params:
            setattr(self, name, getattr(self, name) - lr * scale * grads[name])


# ---------- 两种目标 ----------
def ar_example(ids, id_map):
    """AR:输入 = 右移一位的真值序列,标签 = 真值序列(教师强制)。
    长度 T 的句子产生 T 个监督信号 —— 信号密度 100%。"""
    seq = [id_map[BOS]] + ids + [id_map[EOS]]
    return seq[:-1], None, seq[1:]                          # 输入, 位置, 标签


def mlm_example(ids, rng, id_map, p=0.15):
    """MLM:随机选 ~15% 位置,80/10/10 替换;只对被选位置计损失(密度 ~15%)。"""
    n_mask = max(1, int(round(p * len(ids))))
    positions = sorted(rng.choice(len(ids), size=n_mask, replace=False))
    inputs = list(ids)
    for t in positions:
        r = rng.random()
        if r < 0.8:                                         # 80% 换成 [MASK]
            inputs[t] = id_map[MASK]
        elif r < 0.9:                                       # 10% 换成随机词
            inputs[t] = rng.integers(3, len(id_map))        # 跳过三个特殊符号
        # 剩下 10% 保持不变(缓解预训练-微调的输入分布差异)
    return inputs, positions, list(ids)


# ---------- 演示 ----------
if __name__ == "__main__":
    vocab, id_map = build_vocab()
    V = len(vocab)
    corpus_ids = [[id_map[c] for c in s] for s in SENTENCES]

    print("── [1] 两张掩码矩阵:单向(GPT) vs 双向(BERT)(课件第 15 页)──")
    toks = "东北大学好"
    n = len(toks)
    causal_m = np.tril(np.ones((n, n), dtype=int))
    print(f"  因果掩码(GPT,行=当前字,列=可见)      双向全可见(BERT)")
    for r in range(n):
        print(f"  {' '.join(map(str, causal_m[r]))}   <- {toks[r]}          "
              f"{' '.join('1' for _ in range(n))}")
    print("  实现:softmax 之前给 0 的位置加 −∞,而不是把权重置零(那样破坏归一化)")

    print("── [2] MLM 的 15% 与 80/10/10(课件第 24 页)──")
    rng = np.random.default_rng(1)
    sent = SENTENCES[2]                                     # 我爱自然语言处理
    ids = [id_map[c] for c in sent]
    inputs, positions, targets = mlm_example(ids, rng, id_map)
    show = lambda l: " ".join(vocab[i] for i in l)
    print(f"  原句: {sent}")
    print(f"  输入: {show(inputs)}   被选位置 {positions}(~15%)")
    print(f"  标签: 只对被选位置算损失:{[vocab[ids[t]] for t in positions]}")
    print("  口诀:80 挖 [MASK]、10 换随机词(防捷径)、10 留(缓解输入不一致)")

    print("── [3] 梯度检查:手推反向传播 vs 数值微分 ──")
    m = TinyAttnLM(V, d=8, seed=0)
    ids_in, _, targets = ar_example(corpus_ids[0], id_map)
    _, grads = m.loss_and_grads(ids_in, None, targets, causal=True)
    eps, worst = 1e-6, 0.0
    for name in m.params:
        A = getattr(m, name)
        flat_ix = rng.choice(A.size, size=8, replace=False)
        for ix in flat_ix:
            iu = np.unravel_index(ix, A.shape)
            old = A[iu]
            A[iu] = old + eps; lp = m.loss_and_grads(ids_in, None, targets, causal=True)[0]
            A[iu] = old - eps; lm = m.loss_and_grads(ids_in, None, targets, causal=True)[0]
            A[iu] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name][iu]
            # 组合误差:梯度本来就接近 0 的项按绝对误差衡量,其余按相对误差
            worst = max(worst, abs(num - ana) / max(1e-4, abs(num) + abs(ana)))
    print(f"  因果掩码下 5 组参数 40 个抽样点的最大组合误差 = {worst:.2e}(< 1e-4 为通过)")
    assert worst < 1e-4

    print("── [4] 同一小模型、同一语料,分别按两种目标训练 ──")
    trained = {}
    for tag, causal, lr0, epochs in [("AR(因果掩码,预测下一字)", True, 0.05, 400),
                                     ("MLM(双向全可见,完形填空)", False, 0.03, 600)]:
        model = TinyAttnLM(V, d=24, seed=0)
        trng = np.random.default_rng(0)
        losses, step, total = [], 0, epochs * len(corpus_ids)
        for epoch in range(epochs):
            order = trng.permutation(len(corpus_ids))
            ep_loss, n_sig = 0.0, 0
            for si in order:
                ids = corpus_ids[si]
                if causal:
                    inp, pos, tgt = ar_example(ids, id_map)
                else:
                    inp, pos, tgt = mlm_example(ids, trng, id_map)
                loss, grads = model.loss_and_grads(inp, pos, tgt, causal)
                # 朴素 SGD 的最小稳定器:梯度裁剪 + 线性学习率衰减(退火到 ~5%)
                lr = lr0 * max(0.1 if causal else 0.05, 1 - step / total)
                model.sgd_step(grads, lr=lr)
                step += 1
                n_t = len(tgt) if pos is None else len(pos)
                ep_loss += loss * n_t
                n_sig += n_t
            losses.append(ep_loss / n_sig)
        mode = "每个位置都产生损失(密度 100%)" if causal else "只有被掩码位置产生损失(密度 ~15%)"
        print(f"  {tag}:损失 {losses[0]:.3f} → {losses[-1]:.3f};"
              f"每 epoch 监督信号 ≈ {n_sig} 个({mode})")
        trained[causal] = model

    print("── [5] 目标函数决定能力边界:AR 续写,MLM 填空 ──")
    model_ar = trained[True]
    out = "东北大学"
    for _ in range(4):
        ids_in = [id_map[BOS]] + [id_map[c] for c in out]
        logits, _ = model_ar.forward(ids_in, causal=True)
        logits[-1, :3] = -1e9                         # 不生成 BOS/EOS/MASK 等特殊符号
        out += vocab[int(np.argmax(logits[-1]))]
    print(f"  AR 续写:「东北大学」→ 「{out}」(训练目标 = 给前缀预测下一字,所以会续写;")
    print(f"          小语料 + 贪心解码出现复读,正是第 10 讲说的暴露偏差症状)")

    model_mlm = trained[False]
    probe = [id_map["我"], id_map[MASK], id_map["东"], id_map["北"], id_map["大"], id_map["学"]]
    logits, _ = model_mlm.forward(probe, causal=False)
    fill = vocab[int(np.argmax(logits[probe.index(id_map[MASK])]))]
    print(f"  MLM 填空:「我[MASK]东北大学」→ 我[{fill}]东北大学(双向上下文,只会完形填空)")
    print("  双向注意力在生成时能看到右边的词 —— 抄答案;这就是 BERT「不会说话」的原因")

    print("── [6] 同一底座,四种任务头(课件第 27 页,看形状)──")
    B, T, d, K = 2, 5, 8, 3
    H = np.random.default_rng(0).standard_normal((B, T, d))   # 底座输出
    c = H[:, 0, :]                                            # 取 [CLS] 位置
    print(f"  ① 文本分类:  [CLS] {c.shape} → 线性(d×K) → logits {(c @ np.ones((d, K))).shape}")
    print(f"  ② 序列标注:  全部位置 {H.shape} → 共享线性 → logits {(H @ np.ones((d, K))).shape}")
    print(f"  ③ 句对关系:  [CLS](输入含两句){c.shape} → 线性 → logits {(c @ np.ones((d, K))).shape}")
    S, E = np.ones(d), np.ones(d)
    print(f"  ④ 抽取式问答: 每位置与 S/E 点积 → start {(H @ S).shape} / end {(H @ E).shape}")
    print("  差别只在『取哪个位置的张量 + 接什么输出层』;微调 lr 2e-5~5e-5,epochs 2~4")

    print("\n所有自检通过 ✔")
