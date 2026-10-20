# -*- coding: utf-8 -*-
"""
第 5 讲 · 词向量与语义表示(Word2Vec:Skip-gram + 负采样)
============================================================

核心思想(Firth, 1957):「观其伴而知其义」—— 上下文相似的词,语义相近。

两条技术路线:
  · 计数派:直接数共现次数,做 PPMI 变换(点互信息截断到 0),矩阵行即词向量。
  · 预测派:Word2Vec 把「相似上下文」变成可训练的分类任务 —— 本文件主角。

Skip-gram:用中心词 c 预测上下文词 o。对每个正样本 (c, o) 采 K 个负样本
(按 unigram^0.75 分布),做 K+1 个二分类:

    L = -log σ(v_c · u_o)  -  Σ_{k=1..K} log σ(- v_c · u_k)

其中 v 是中心词的「输入向量」(即最终词向量),u 是上下文的「输出向量」。
梯度非常干净,记 s = σ(v_c·u_o)、s_k = σ(v_c·u_k):

    ∂L/∂u_o = (s - 1) · v_c        ∂L/∂u_k = s_k · v_c
    ∂L/∂v_c = (s - 1) · u_o + Σ_k s_k · u_k

训练完用余弦相似度做最近邻检索,「猫」的邻居应当是「狗/小猫」而不是「电脑」。
"""
import sys
from collections import Counter

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 小语料:三组语义簇(动物 / 校园 / 电子设备),空格分词
SENTENCES = [
    "猫 追 老鼠", "狗 追 老鼠", "小猫 追 蝴蝶", "小狗 追 皮球",
    "猫 吃 鱼", "狗 吃 肉", "小猫 吃 鱼", "小狗 吃 骨头",
    "猫 是 可爱 的 动物", "狗 是 忠诚 的 动物", "动物 园 里 有 猫 和 狗",
    "学生 学习 知识", "老师 讲授 知识", "学生 阅读 书籍", "老师 推荐 书籍",
    "学生 完成 作业", "老师 批改 作业", "学生 在 教室 上课", "老师 在 教室 讲课",
    "学生 去 图书馆 借书", "老师 去 图书馆 查资料",
    "老师 指导 学生", "老师 帮助 学生", "老师 和 学生 讨论 问题",
    "电脑 运行 软件", "手机 运行 应用", "电脑 处理 数据", "手机 处理 照片",
    "工程师 编写 程序", "工程师 修复 缺陷", "电脑 和 手机 都是 设备",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------- 1. 计数派:共现 + PPMI ----------
def ppmi_matrix(sentences, window=2):
    """滑动窗口共现计数 -> PPMI 矩阵。PMI = log( p(i,j) / (p(i)·p(j)) ),负值截为 0。"""
    tokens = [w for s in sentences for w in s.split()]
    vocab = sorted(set(tokens))
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V, V))
    for s in sentences:
        ids = [idx[w] for w in s.split()]
        for i, ci in enumerate(ids):
            for j in range(i + 1, min(i + 1 + window, len(ids))):
                C[ci, ids[j]] += 1
                C[ids[j], ci] += 1
    total = C.sum()
    pw = C.sum(axis=1) / total                      # p(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / total) / (pw[:, None] * pw[None, :]))
    return np.maximum(pmi, 0), vocab


# ---------- 2. 预测派:Skip-gram + 负采样 ----------
class SkipGram:
    def __init__(self, sentences, dim=24, window=2, neg_k=4, seed=0):
        tokens = [w for s in sentences for w in s.split()]
        self.vocab = sorted(set(tokens))
        self.id = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        counts = np.array([tokens.count(w) for w in self.vocab], dtype=float)
        self.noise = counts ** 0.75                 # 负采样分布:unigram^0.75
        self.noise /= self.noise.sum()
        self.K, self.window = neg_k, window

        rng = np.random.default_rng(seed)
        self.W_in = (rng.random((V, dim)) - 0.5) / dim    # 中心词向量
        self.W_out = np.zeros((V, dim))                   # 上下文向量

        self.pairs = []
        for s in sentences:
            ids = [self.id[w] for w in s.split()]
            for i, c in enumerate(ids):
                for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                    if j != i:
                        self.pairs.append((c, ids[j]))
        self.rng = rng

    def _negatives(self, pos_id):
        """采 K 个负样本;极少数撞上正样本就重采一次(教学实现,从简)。"""
        for _ in range(10):
            negs = self.rng.choice(len(self.vocab), size=self.K, p=self.noise)
            if pos_id not in negs:
                return negs
        return negs[negs != pos_id]


def train_skipgram(sentences, dim=24, window=2, neg_k=4, epochs=400, lr=0.05, seed=0):
    """过程式写法,梯度更新一目了然(比类封装更适合教学)。"""
    model = SkipGram(sentences, dim, window, neg_k, seed)
    rng, V = model.rng, len(model.vocab)
    W_in, W_out = model.W_in, model.W_out
    loss_hist = []
    for ep in range(epochs):
        cur_lr = max(lr * (1 - ep / epochs), lr * 0.01)
        total_loss = 0.0
        for c, o in model.pairs:
            v = W_in[c]                                        # (d,)
            negs = model._negatives(o)
            uo = W_out[o].copy()                               # 先拷贝再更新,保证梯度用旧值
            s_pos = sigmoid(v @ uo)
            s_neg = sigmoid(W_out[negs] @ v)
            total_loss += -np.log(s_pos + 1e-9) - np.log(1 - s_neg + 1e-9).sum()

            g_pos = s_pos - 1.0                                # ∂L/∂u_o
            g_neg = s_neg                                      # ∂L/∂u_k
            W_in[c] -= cur_lr * (g_pos * uo + W_out[negs].T @ g_neg)
            W_out[o] -= cur_lr * g_pos * v
            np.add.at(W_out, negs, -cur_lr * np.outer(g_neg, v))  # 处理重复下标
        loss_hist.append(total_loss / len(model.pairs))
    return W_in, model, loss_hist


def nearest(W_in, vocab, word, k=5):
    """余弦相似度最近邻。"""
    ids = {w: i for i, w in enumerate(vocab)}
    v = W_in[ids[word]]
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = W_in @ v / (np.linalg.norm(W_in, axis=1) + 1e-9)
    sims[ids[word]] = -np.inf
    top = np.argsort(-sims)[:k]
    return [(vocab[i], round(float(sims[i]), 3)) for i in top]


if __name__ == "__main__":
    print("── [1] 计数派:PPMI 矩阵(截断展示)──")
    P, vocab = ppmi_matrix(SENTENCES, window=2)
    for a, b in [("猫", "狗"), ("老师", "学生"), ("电脑", "手机"), ("猫", "学生")]:
        print(f"  PPMI({a},{b}) = {P[vocab.index(a), vocab.index(b)]:.3f}")
    print("  ^ 同簇共现词的 PPMI 高;猫-学生 ≈ 0(几乎不同框出现)")

    print("── [2] 预测派:训练 Skip-gram + 负采样(约 10 秒)──")
    W_in, model, loss_hist = train_skipgram(SENTENCES, epochs=400)
    print(f"  初始平均损失 {loss_hist[0]:.3f} -> 最终 {loss_hist[-1]:.3f}(在降,说明在学)")

    print("── [3] 最近邻检索(余弦相似度)──")
    for q in ["猫", "学生", "电脑", "吃"]:
        print(f"  {q:3} -> {nearest(W_in, model.vocab, q)}")
    print("\n观察:每个查询词的邻居都在同一语义簇 —— 这就是「分布式语义」")
