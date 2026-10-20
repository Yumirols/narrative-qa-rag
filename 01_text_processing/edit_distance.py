# -*- coding: utf-8 -*-
"""
第 3 讲 · 最小编辑距离与拼写纠错(Minimum Edit Distance & Spelling Correction)
==============================================================================

最小编辑距离:把字符串 a 变成 b 所需的最少「插入 / 删除 / 替换」操作数。
用动态规划求解,子问题是「两个前缀之间的距离」:

    dp[i][j] = a[:i] -> b[:j] 的最小编辑距离
    dp[i][j] = min(
        dp[i-1][j] + 1,        # 删除 a[i-1]
        dp[i][j-1] + 1,        # 插入 b[j-1]
        dp[i-1][j-1] + cost,   # 替换:a[i-1] != b[j-1] 时 cost=1,相等时 0
    )
    边界:dp[i][0] = i(a 的前 i 个字符全删),dp[0][j] = j(全插入)

复杂度 O(m·n)。对 dp 表从右下角回溯可还原每一步操作(对齐),
这是拼写纠错、DNA 序列比对、机器翻译词对齐的共同基础。

拼写纠错思路(Norvig 式):
    候选 = 词表中与错误词编辑距离 <= 2 的词;
    排序 = 先按编辑距离升序,再按词频降序 —— 「错得少的优先,常见的优先」。
"""
import sys
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------- 1. 最小编辑距离(DP) ----------
def edit_distance(a: str, b: str) -> int:
    """dp 表法,返回最小编辑距离。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i          # 删除 i 次
    for j in range(n + 1):
        dp[0][j] = j          # 插入 j 次
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # 删
                           dp[i][j - 1] + 1,      # 插
                           dp[i - 1][j - 1] + cost)  # 替/匹配
    return dp[m][n]


def edit_ops(a: str, b: str) -> list:
    """回溯 dp 表,返回操作序列 [(op, a字符, b字符), ...],op ∈ match/replace/insert/delete。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", a[i - 1], b[j - 1]))       # 原地不动
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("replace", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("insert", "-", b[j - 1]))
            j -= 1
        else:  # i > 0 且 dp[i][j] == dp[i-1][j] + 1
            ops.append(("delete", a[i - 1], "-"))
            i -= 1
    return ops[::-1]


# ---------- 2. 拼写纠错 ----------
class SpellCorrector:
    """基于「编辑距离 + 词频」的拼写纠错。

    语料只决定词频;候选生成对整个词表按编辑距离过滤,
    并用「长度差 > 最大距离必不可能是候选」提前剪枝。
    """

    def __init__(self, corpus_words):
        self.freq = Counter(w.lower() for w in corpus_words)
        self.vocab = sorted(self.freq)            # 排序保证可复现

    def candidates(self, word: str, max_dist: int = 2):
        word = word.lower()
        out = []
        for w in self.vocab:
            if abs(len(w) - len(word)) > max_dist:   # 剪枝:长度差已超上限
                continue
            d = edit_distance(word, w)
            if d <= max_dist:
                out.append((d, -self.freq[w], w))    # 距离小者优先,词频大者优先
        return sorted(out)

    def correct(self, word: str, max_dist: int = 2) -> str:
        cands = self.candidates(word, max_dist)
        return cands[0][2] if cands else word


if __name__ == "__main__":
    print("── [1] 最小编辑距离 ──")
    for a, b in [("kitten", "sitting"), ("flaw", "lawn"), ("intention", "execution"), ("自然语言", "自然生成")]:
        print(f"  d({a!r}, {b!r}) = {edit_distance(a, b)}")
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("intention", "execution") == 5   # 课件经典例子

    print("── [2] 回溯对齐(还原操作序列)──")
    for op, x, y in edit_ops("kitten", "sitting"):
        print(f"  {op:8} {x:>2} -> {y}")

    print("── [3] 拼写纠错 ──")
    corpus = (
        "the cat sat on the mat the dog ate the bone the cat is cute "
        "apple banana cherry grape orange apple apple juice apple pie "
        "hello world machine learning natural language processing"
    ).split()
    sc = SpellCorrector(corpus)
    for bad in ["aple", "apll", "mat", "hwlo", "bananna", "langage"]:
        top = sc.candidates(bad, max_dist=2)[:3]
        print(f"  {bad!r:10} -> {sc.correct(bad)!r:12}  候选(距离,-词频): {top}")
    assert sc.correct("aple") == "apple"
    assert sc.correct("bananna") == "banana"
    assert sc.correct("mat") == "mat"          # 本身是合法词,距离 0 优先
    print("\n所有自检通过 ✔")
