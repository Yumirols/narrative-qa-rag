# -*- coding: utf-8 -*-
"""
第 4 讲 · 分词与子词表示(Tokenization & Subword)
==================================================

1. 正向最大匹配 FMM(Forward Maximum Matching)
   中文没有天然的空格边界。FMM 从句首开始,每轮贪心切出词典里能匹配的
   最长前缀,切不动就切单字。复杂度 O(n · max_len),是「词典分词」的最小实现。
   缺点:贪心不看全局,遇到歧义必错切(见演示「研究生命起源」),
   这正是后来统计分词 / 神经分词要解决的问题。

2. BPE(Byte Pair Encoding)子词算法 —— GPT / LLaMA 分词的基石
   训练:把每个词拆成字符序列(词尾加 '</w>' 标记边界),
         每轮统计语料中最高频的「相邻符号对」并合并为新符号,重复 k 轮。
   效果:常见词(如 low)很快合并成整体,罕见词拆成有意义的子词(如 est)。
   编码:把学到的合并规则按训练顺序重放即可切分新词。
"""
import sys
from collections import Counter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------- 1. 正向最大匹配 ----------
def fmm_segment(sentence: str, dictionary: set, max_word_len: int = 5) -> list:
    """从左到右,每次取「词典可匹配的最长前缀」;都不匹配则切单字。"""
    words, i, n = [], 0, len(sentence)
    while i < n:
        for L in range(min(max_word_len, n - i), 1, -1):   # 由长到短尝试
            if sentence[i:i + L] in dictionary:
                words.append(sentence[i:i + L])
                i += L
                break
        else:                     # 没有任何多字词命中 -> 切单字
            words.append(sentence[i])
            i += 1
    return words


# ---------- 2. BPE:训练 ----------
def train_bpe(corpus: str, num_merges: int):
    """返回 (merges, word_freqs)。

    merges 是按顺序学到的合并规则 [(a, b), ...],编码新词时必须按此顺序重放。
    """
    # 词频统计;词内符号用元组存,末尾加 </w> 标记词边界
    words = Counter()
    for w in corpus.split():
        words[tuple(w) + ("</w>",)] += 1

    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym_seq, freq in words.items():
            for i in range(len(sym_seq) - 1):
                pairs[(sym_seq[i], sym_seq[i + 1])] += freq
        if not pairs:
            break
        best, _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))  # 频率最高(并列取字典序,保证可复现)
        merges.append(best)
        # 把序列里所有的 best 相邻对合并掉
        new_words = Counter()
        for sym_seq, freq in words.items():
            merged, i = [], 0
            while i < len(sym_seq):
                if i < len(sym_seq) - 1 and (sym_seq[i], sym_seq[i + 1]) == best:
                    merged.append(sym_seq[i] + sym_seq[i + 1])
                    i += 2
                else:
                    merged.append(sym_seq[i])
                    i += 1
            new_words[tuple(merged)] += freq
        words = new_words
    return merges, words


def bpe_encode(word: str, merges: list) -> list:
    """对任意新词应用合并规则(按训练时的顺序重放),得到子词序列。"""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def bpe_decode(tokens: list) -> str:
    """子词序列 -> 原词:拼接后去掉词尾标记。"""
    return "".join(tokens).replace("</w>", "")


if __name__ == "__main__":
    print("── [1] 正向最大匹配(中文分词)──")
    dictionary = {"研究", "研究生", "生命", "起源", "命运", "自然语言处理",
                  "语言", "处理", "课程", "大学生", "学生", "活"}
    for sent in ["研究生命起源", "自然语言处理课程", "研究生命运"]:
        print(f"  {sent} -> {fmm_segment(sent, dictionary, max_word_len=8)}")
    print("  ^ 注意:『研究生命起源』被错切成 研究生/命/起源 —— 贪心不看全局,引出统计分词")

    print("── [2] BPE 训练(经典教材语料)──")
    corpus = "low low low low low lower lower newest newest newest newest newest widest widest widest"
    merges, final_words = train_bpe(corpus, num_merges=6)
    for a, b in merges:
        print(f"  合并规则: {a!r} + {b!r} -> {a + b!r}")
    print("  训练结束后的词表示:")
    for sym_seq, freq in sorted(final_words.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {list(sym_seq)} × {freq}")

    print("── [3] BPE 编码 / 解码 ──")
    for w in ["lowest", "newer", "low", "newly"]:
        toks = bpe_encode(w, merges)
        assert bpe_decode(toks) == w
        print(f"  {w:8} -> {toks}")
    print("\n所有自检通过 ✔")
