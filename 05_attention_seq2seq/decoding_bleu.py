# -*- coding: utf-8 -*-
"""
第 10 讲 · 解码策略与 BLEU(Greedy / Beam Search / 评测)
==========================================================

一、解码:在 |V|^m 的搜索空间里找整句
    ŷ = argmax_y P(y | x) = argmax_y Π_t P(y_t | y_{<t}, x)

    greedy   每步取局部最大,O(m·V),最快最稳,但局部最优 ≠ 全局最优
    beam     每步保留 k 条最优前缀,O(m·k·V),工业标配;仍是近似搜索
             三个实现要点:
               ① 累加对数概率(概率连乘会下溢成 0)
               ② 出现 <EOS> 的前缀移入「完成池」,不再扩展但保留参与最终比较
               ③ 必须做长度归一化,否则系统性偏好短句:
                     score(y) = ( Σ_t log P(y_t | y_{<t}, x) ) / L^α
                  α=0 原始 logP(偏短);α=1 完全除长;工业常用 α≈0.6-0.7(GNMT 0.6)
             局部剪枝不可逆:开头概率不高、后续极好的序列可能第一步就出局,
             所以 beam search 不保证全局最优(与第 9 讲 Viterbi 的本质区别)。

二、BLEU(Papineni et al., 2002)
    BLEU = BP × exp( Σ_{n=1..4} (1/4)·ln p_n )
    p_n:修正的 n-gram 精确率,截断计数防「重复刷分」:
        Count_clip(w) = min( 候选计数 , 参考中的最大计数 )
    BP:简短惩罚,治「越短越占便宜」:
        BP = 1              若 c > r
        BP = exp(1 − r/c)   若 c ≤ r        (c 候选长,r 最接近的参考长)
    注意:任一 p_n = 0 会使整个 BLEU 归零(几何平均的性质);
    BLEU 偏精确率、只看字面,抓不住漏译与同义改写 —— 指标是工具,不是判决。

本文件用课件第 33-42 页的概率表与例句复现全部数字。
"""
import sys
from collections import Counter
from math import exp, log

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

EOS = "<EOS>"


# ---------- 一、解码 ----------
def greedy_decode(model, max_len=8):
    """贪心:每步取条件概率最大的词,直到 <EOS> 或达到上限。
    model:dict{前缀 tuple -> {词: 概率}} 的概率表(教学用)。返回 (序列, 概率)。"""
    seq, p = [], 1.0
    while len(seq) < max_len:
        dist = model[tuple(seq)]
        w = max(dist, key=lambda k: (dist[k], k))
        seq.append(w)
        p *= dist[w]
        if w == EOS:
            break
    return seq, p


def beam_decode(model, k, alpha=0.0, max_len=8):
    """束搜索:每步对束中前缀做一步扩展,全局排序保留总分(logP)最高的 k 条;
    其中已生成 <EOS> 的候选移入「完成池」不再扩展,但保留参与最终比较
    (选中时占束宽名额,移走后名额让给未完成前缀 —— 与课件第 34 页演算一致)。
    最终在「完成池 + 未完成前缀」上按 logP / L^α 排序返回。"""
    beam = [((), 0.0)]                      # (前缀, 累计 logP);log 空间,避免下溢
    completed = []                          # 完成池
    for _ in range(max_len):
        cands = []
        for prefix, logp in beam:
            for w, p in model.get(prefix, {}).items():   # 无后继的前缀自然终止
                cands.append((prefix + (w,), logp + log(p)))
        if not cands:
            break
        cands.sort(key=lambda c: c[1], reverse=True)     # 剪枝:只留 top-k
        beam = []
        for pfx, lp in cands[:k]:
            (completed if pfx[-1] == EOS else beam).append((pfx, lp))
        if not beam:
            break
    pool = completed + beam
    # 长度 L 与课件口径一致:不含结尾的 <EOS>(A<EOS> → L=1,A x y → L=3)
    pool.sort(key=lambda c: c[1] / (len(c[0]) - (c[0][-1] == EOS)) ** alpha,
              reverse=True)
    return pool                                   # 按长度归一化分数从高到低


def demo_decoding():
    print("── [1] 贪心的短视:局部最优 ≠ 全局最优 ──")
    # 课件第 33 页的两分支示意(Ax=0.05 vs Bp=0.27)。这里给可归一化的版本:
    # 第 1 步 P(A)=0.50 > P(B)=0.30,贪心选 A;但 A 的最优后继只有 0.12,
    # 而 B 的最优后继高达 0.90 —— 全局最优是 Bp = 0.30×0.90 = 0.27。
    model1 = {
        (): {"A": 0.50, "B": 0.30, "C": 0.20},
        ("A",): {"x": 0.12, **{f"z{i}": 0.0978 for i in range(9)}},
        ("B",): {"p": 0.90, "q": 0.10},
    }
    seq, p = greedy_decode(model1, max_len=2)
    print(f"  greedy 输出 {' '.join(seq)}  联合概率 = {p:.3f}")
    best = beam_decode(model1, k=2, max_len=2)[0]
    print(f"  beam(k=2) 输出 {' '.join(best[0])}  联合概率 = {exp(best[1]):.3f}")
    assert seq[0] == "A" and best[0] == ("B", "p")
    print("  贪心第一步就把 B 分支丢了;beam 保留两条前缀才看得到全局更优")

    print("── [2] beam search 逐步演算(k=2)+ 长度归一化改写答案(课件第 34-35 页)──")
    model2 = {
        (): {"A": 0.50, "B": 0.30, "C": 0.20},
        ("A",): {"x": 0.40, "p": 0.10, EOS: 0.50},
        ("B",): {"x": 0.60, "p": 0.10, EOS: 0.30},
        ("A", "x"): {"y": 0.90, EOS: 0.10},
    }
    seq_g, p_g = greedy_decode(model2)
    print(f"  greedy: {' '.join(seq_g)}  P = {p_g:.3f}")
    for alpha, tag in [(0.0, "α=0(原始 logP,偏短)"), (0.7, "α=0.7(工业常用)")]:
        pool = beam_decode(model2, k=2, alpha=alpha)
        top = pool[0]
        print(f"  beam k=2 {tag}:")
        for pfx, lp in pool[:3]:
            L = len(pfx) - (pfx[-1] == EOS)   # 长度不含 <EOS>,与课件第 35 页一致
            print(f"    {' '.join(pfx):14} P={exp(lp):.3f}  logP={lp:.3f}  "
                  f"L={L}  score={lp / L ** alpha:.3f}")
        if alpha == 0.0:
            assert top[0] == ("A", EOS)          # 短句「A<EOS>」占尽便宜
        else:
            assert top[0] == ("A", "x", "y")     # 归一化后 3 词句反超
    print("  长度归一化把答案从「A<EOS>」改写为「A x y」—— 不做归一化,")
    print("  模型会学会『早出 <EOS> 拿高分』;剪枝不可逆(C 第 1 步出局就再无机会)")


# ---------- 二、BLEU ----------
def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(cand, refs, n):
    """截断计数后的 n-gram 精确率:每个 n-gram 的计数按「参考中的最大计数」截断。"""
    c_counts = ngram_counts(cand, n)
    r_counts = Counter()
    for r in refs:
        r_counts |= ngram_counts(r, n)          # 多参考:取每个 n-gram 的最大计数
    match = sum(min(cnt, r_counts[g]) for g, cnt in c_counts.items())
    total = max(sum(c_counts.values()), 1)
    return match / total, match, total


def bleu(cand_str, ref_strs, max_n=4):
    cand = cand_str.split()
    refs = [r.split() for r in ref_strs]
    c = len(cand)
    r = min((len(x) for x in refs), key=lambda L: abs(L - c))   # 最接近候选长的参考
    bp = 1.0 if c > r else exp(1.0 - r / max(c, 1))
    ps, log_sum = [], 0.0
    for n in range(1, max_n + 1):
        p, _, _ = modified_precision(cand, refs, n)
        ps.append(p)
        log_sum += (log(p) if p > 0 else -float("inf")) / max_n
    return (bp * exp(log_sum) if all(p > 0 for p in ps) else 0.0), ps, bp


def demo_bleu():
    print("── [3] 截断计数:重复刷不出分(课件论文反例)──")
    cand = "the the the the the the the"
    ref = "the cat is on the mat"
    raw = ngram_counts(cand.split(), 1)["the", ]
    clipped, match, total = modified_precision(cand.split(), [ref.split()], 1)
    print(f"  候选 {cand!r}(7 词) vs 参考 {ref!r}(the 出现 2 次)")
    print(f"  未修正 p1 = {raw}/7 = 1.000(垃圾句满分!);截断后 p1 = {match}/{total} = {clipped:.3f}")

    print("── [4] 完整 BLEU 例题(课件第 42 页,参考答案 ≈ 0.81)──")
    cand = "the cat is on the mat today"            # c = 7
    ref = "the cat is on the mat"                   # r = 6
    bleu_v, ps, bp = bleu(cand, [ref])
    for n, p in enumerate(ps, 1):
        print(f"  p{n} = {p:.4f}   (ln = {log(p):.4f})")
    print(f"  BP = {bp:.3f}(c=7 > r=6,不惩罚)   BLEU = {bleu_v:.3f}")
    assert abs(bleu_v - 0.809) < 0.005
    print("  BLEU = BP × exp(mean(ln p_n)) ≈ 0.81;注意任一 p_n=0 会令 BLEU=0")

    print("── [5] 再看截断计数:把 today 换成 the ──")
    cand2 = "the cat is on the mat the"
    ngram_counts2 = ngram_counts(cand2.split(), 1)
    clipped, match, total = modified_precision(cand2.split(), [ref.split()], 1)
    print(f"  候选 {cand2!r}:the 出现 {ngram_counts2[('the',)]} 次(参考只有 2 次)")
    print(f"  不截断 p1 = 7/7 = 1.000;截断后 p1 = {match}/{total} = {clipped:.3f}")
    print("  —— 截断计数防『重复刷分』,BP 防『越短越占便宜』,都是被钻过的漏洞逼出来的补丁")


if __name__ == "__main__":
    demo_decoding()
    print()
    demo_bleu()
    print("\n所有自检通过 ✔")
