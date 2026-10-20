# -*- coding: utf-8 -*-
"""
第 7 讲 · 统计语言模型:N-gram 与评估(Statistical Language Models)
====================================================================

语言模型给句子打「合理性」分数。链式法则太苛刻:
    P(w1..wn) = Π P(wi | w1..wi-1)        <- 上下文无限长,没法估计
N-gram 用马尔可夫假设砍断上下文(n=2 即 bigram:只看前 1 个词):
    P(w1..wn) ≈ Π P(wi | wi-1)

估计 = 数数(极大似然):
    P(wi | h) = count(h, wi) / count(h)

问题:数据稀疏,没见过的组合概率为 0,整句 perplexity 直接爆炸(演示可见)。
解决:平滑(Smoothing)
    Laplace / Add-k:P(wi|h) = (count(h,wi) + k) / (count(h) + k·V)
    k=1 是教科书解,k 取小值(如 0.01)实践上通常远好于 1。

评估:困惑度 Perplexity = P(测试句) ^ (-1/N) = exp(平均每个词的负对数似然)
    直觉:「模型平均在多少个词里犹豫」,越低越好。

生成:从 <s> 出发,按 P(w | 前文) 采样下一个词,直到 </s> —— 「语言模型接龙」。
"""
import math
import sys
import random
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class NgramLM:
    def __init__(self, n=2, k=1.0):
        self.n, self.k = n, k

    def fit(self, sentences):
        """sentences: list[list[str]],无需手动加边界符,这里统一加。"""
        tokens = []
        for s in sentences:
            tokens += ["<s>"] * (self.n - 1) + list(s) + ["</s>"]
        self.vocab = sorted(set(tokens) - {"<s>"})
        self.V = len(self.vocab)
        self.ngrams = Counter(tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1))
        # 每个位置(含 </s> 位置)的前 n-1 个词,即上下文计数
        self.contexts = Counter(
            tuple(tokens[i - self.n + 1:i]) for i in range(self.n - 1, len(tokens))
        )
        return self

    def prob(self, word, context):
        """Add-k 平滑概率。context 是 n-1 元组。"""
        num = self.ngrams.get(context + (word,), 0)
        den = self.contexts.get(context, 0)
        if self.k == 0 and (num == 0 or den == 0):
            return 0.0                                # 不平滑 = 未登录组合概率为 0
        return (num + self.k) / (den + self.k * self.V)

    def perplexity(self, sentence):
        toks = ["<s>"] * (self.n - 1) + list(sentence) + ["</s>"]
        logp, N = 0.0, 0
        for i in range(self.n - 1, len(toks)):
            p = self.prob(toks[i], tuple(toks[i - self.n + 1:i]))
            if p <= 0.0:
                return float("inf")                   # 见到 0 概率 -> 困惑度无穷大
            logp += math.log(p)
            N += 1
        return math.exp(-logp / N)

    def generate(self, max_len=12, seed=0):
        """从 <s> 接龙采样。tiny 语料下 bigram 往往已能生成通顺短语。"""
        rng = random.Random(seed)
        toks = ["<s>"] * (self.n - 1)
        for _ in range(max_len):
            ctx = tuple(toks[-(self.n - 1):]) if self.n > 1 else ()
            words, weights = zip(*[(w, self.prob(w, ctx)) for w in self.vocab])
            w = rng.choices(words, weights=weights)[0]
            if w == "</s>":
                break
            toks.append(w)
        return toks[self.n - 1:]


if __name__ == "__main__":
    TRAIN = [
        "我 喜欢 自然 语言 处理".split(),
        "我 学习 自然 语言 处理".split(),
        "我们 学习 机器 学习".split(),
        "我 喜欢 机器 学习".split(),
        "语言 模型 很 有趣".split(),
        "机器 学习 很 有趣".split(),
        "我 在 东北大学 读书".split(),
        "我们 在 东北大学 学习".split(),
    ]
    TEST_A = "我 喜欢 机器 学习".split()      # 所有 bigram 都在训练语料里见过
    TEST_B = "我们 喜欢 语言 模型".split()    # 「我们 喜欢」等是未见组合

    print("── [1] bigram 估计:不平滑的爆零问题 ──")
    lm0 = NgramLM(n=2, k=0.0).fit(TRAIN)
    print(f"  P(喜欢|我们) = {lm0.prob('喜欢', ('我们',))}  <- 训练中从没出现过『我们 喜欢』")
    pp0 = lm0.perplexity(TEST_B)
    print(f"  测试句 B 的困惑度 = {'inf(爆零!)' if pp0 == float('inf') else pp0}")
    print("  ^ 一个未见组合就让整句概率归零 —— 必须平滑")

    print("── [2] 平滑强度 k 的权衡(困惑度越低越好)──")
    lm1 = NgramLM(n=2, k=1.0).fit(TRAIN)
    lm01 = NgramLM(n=2, k=0.01).fit(TRAIN)
    for name, sent in [("全见过 ", TEST_A), ("有未见 ", TEST_B)]:
        print(f"  {name}: k=1 -> {lm1.perplexity(sent):7.2f}   k=0.01 -> {lm01.perplexity(sent):7.2f}")
    print("  ^ 全见过的句子:k=0.01 接近真实分布,远好于 k=1 的过度均匀")
    print("    多未见的句子:k=1 反而兜底更强 —— k 是在「保真」与「兜底」间权衡,实践用验证集调")

    print("── [3] bigram vs trigram 生成 ──")
    bigram, trigram = lm01, NgramLM(n=3, k=0.01).fit(TRAIN)
    print("  bigram :", *(" ".join(bigram.generate(seed=s)) for s in range(3)), sep="\n          ")
    print("  trigram:", *(" ".join(trigram.generate(seed=s)) for s in range(3, 6)), sep="\n          ")
    print("  ^ 语料极小,trigram 基本在背诵原句 —— 这正是神经语言模型(下一讲)要解决的泛化问题")

    # ---- 自检 ----
    assert pp0 == float("inf")
    assert lm01.perplexity(TEST_A) < lm1.perplexity(TEST_A)   # 全见过:k 小者胜
    assert lm1.perplexity(TEST_B) < lm01.perplexity(TEST_B)   # 多未见:k=1 兜底强
    print("\n所有自检通过 ✔")
