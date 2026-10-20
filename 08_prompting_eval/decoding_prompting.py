# -*- coding: utf-8 -*-
"""
第 14 讲 · 提示工程与 LLM 评估(可离线复现的最小演示)
==========================================================

一、解码策略:提示与解码正交
    提示改「条件」(影响下一个 token 的分布形状),解码改「从这个分布里怎么挑」。
    Temperature: P_i ∝ exp(z_i / T)。T<1 锐化(确定保守),T>1 平坦化(多样易跑偏),
                 T→0 退化为贪心。
    Top-k:      只保留概率最高的 k 个,重归一化后采样 —— 截断的是「数量」。
    Top-p:      按概率降序累加,取刚超过 p 的最小集合 —— 截断的是「概率质量」,
                候选数随分布自适应:分布越尖候选越少。生产组合常为温度 0.7 + top-p 0.9。

二、提示结构(六大要素):角色 / 指令 / 上下文 / 输入数据 / 示例 / 输出格式。
    RAG 式提示的三条约束必须写死:①只依据资料作答 ②资料未提及时必须回答
    「资料未提及」③结论标注来源编号。第②条是最便宜的抗幻觉手段。

三、Self-Consistency(自洽性):同一提示在 temperature>0 下独立采样 N 条推理链,
    对「答案层」多数投票。前提:错误是随机的 —— 若错误来自系统性偏差,
    N 条链会稳定地给出同一个错误答案:一致性 ≠ 正确性。

四、LLM-as-a-Judge 三偏差:位置偏差(偏爱某个位置)、冗长偏差、自我增强偏差。
    位置偏差的校正:交换 A/B 顺序各评一次,只采纳两次一致的结论,不一致送人工。

本文件不调用任何 API:用「人为给定的下一个 token 分布」扮演 LLM 的输出层,
把课件里每个公式跑出真实数字;评委用「带位置加成的噪声打分器」模拟。
"""
import sys
from collections import Counter

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------- 一、解码策略 ----------
def softmax_temperature(z, T=1.0):
    """P_i ∝ exp(z_i / T):除以 T 再 softmax。T→0 = 贪心,T=1 = 原分布。"""
    z = np.asarray(z, dtype=float) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def top_k_mask(p, k):
    """只保留概率最高的 k 个 token,重归一化(截「数量」)。"""
    keep = np.argsort(-p)[:k]
    m = np.zeros_like(p, dtype=bool)
    m[keep] = True
    q = np.where(m, p, 0.0)
    return m, q / q.sum()


def top_p_mask(p, p_threshold):
    """核采样:按概率降序累加,取「累计刚超过 p」的最小集合(截「质量」,自适应)。"""
    order = np.argsort(-p)
    m = np.zeros_like(p, dtype=bool)
    cum = 0.0
    for i in order:
        m[i] = True
        cum += p[i]
        if cum > p_threshold:            # 严格超过才停 —— 与课件口径一致
            break
    q = np.where(m, p, 0.0)
    return m, q / q.sum()


def demo_decoding():
    print("── [1] Temperature:分布的锐化与平坦化(课件第 35 页)──")
    z = np.array([4.0, 2.2, 1.8, 1.5, 1.2])          # 人为给定的 5 个 token 的 logits
    for T in [0.5, 1.0, 2.0]:
        p = softmax_temperature(z, T)
        print(f"  T={T:<4} P = {np.round(p, 3)}   首位概率 = {p[0]:.3f}")
    print("  T<1 变尖(几乎必选首位)、T>1 摊平(低概率 token 也上台)—— 连续旋钮")

    print("── [2] Top-k 与 Top-p:固定数量 vs 固定质量(课件第 36 页)──")
    p = np.array([0.50, 0.30, 0.15, 0.05])            # [好, 吃, 睡, 打]
    for name, fn in [("top-k  k=2", lambda: top_k_mask(p, 2)),
                     ("top-p  p=0.8", lambda: top_p_mask(p, 0.8)),
                     ("top-p  p=0.8(尖分布)", lambda: top_p_mask(np.array([0.97, 0.02, 0.01, 0.0]), 0.8))]:
        m, q = fn()
        kept = np.where(m)[0].tolist()
        print(f"  {name:22} 候选集 = {kept}  重归一化后 = {np.round(q, 3)}")
    print("  top-p 截「概率质量」:分布越尖候选越少(尖分布只留 1 个),比 top-k 稳健")

    print("── [3] 贪心可复现 vs 采样有多样(评测为什么必须固定解码参数)──")
    rng = np.random.default_rng(0)
    for tag, T in [("T=0(贪心)      ", 0.05), ("T=0.7+top-p 0.9", 0.7), ("T=1.2          ", 1.2)]:
        picks = []
        for _ in range(20):
            q = softmax_temperature(z, T)
            if T <= 0.4:                              # 贪心:直接取 argmax
                picks.append(int(np.argmax(q)))
            else:
                m, _ = top_p_mask(q, 0.9)
                q = np.where(m, q, 0.0); q /= q.sum()
                picks.append(int(rng.choice(len(q), p=q)))
        print(f"  {tag} 20 次输出去重后 {len(set(picks))} 种,众数占比 {picks.count(max(set(picks), key=picks.count)) / 20:.0%}")
    print("  T=0 完全可复现;温度越高越多样 —— 不写解码参数,评测结果不可复现")


# ---------- 二、提示结构(六大要素 + 三条约束) ----------
def build_prompt(role, instruction, context, examples, query, sources):
    """按六大要素拼一个 RAG 式提示(结构分隔:规则 / 资料 / 问题三块)。"""
    parts = [f"【角色】{role}", f"【指令】{instruction}"]
    if context:
        parts.append(f"【上下文】{context}")
    parts.append("【资料】")
    parts += [f"[{i + 1}] {s}" for i, s in enumerate(sources)]
    if examples:
        parts.append("【示例】" + " ".join(examples))
    parts.append(f"【问题】{query}")
    parts.append("【要求】① 只依据上面的资料作答,不要使用你自己的知识;\n"
                 "        ② 资料中没有提到的内容,必须回答「资料未提及」;\n"
                 "        ③ 每个结论后面标注来源编号,如 [1]。")
    return "\n".join(parts)


def demo_prompt():
    print("── [4] 六大要素拼一个 RAG 式提示(课件第 29 页,三条约束写死)──")
    prompt = build_prompt(
        role="财务报销审核助手,只依据公司制度回答",
        instruction="按【依据】【计算】【结论】三步回答报销问题",
        context="职级 P6 及以下适用住宿标准;P7 及以上按实际发生额审批",
        examples=["杭州属于「其他城市」,标准 450 元/晚"],
        query="P5 员工去杭州出差 3 天,住宿每晚 700 元,有提前审批,能报销多少?",
        sources=["一线城市(北京、上海、广州、深圳)每晚不超过 600 元;其他城市每晚不超过 450 元。",
                 "超标部分:出差前提交《超标审批单》并获批的,凭发票按实际金额报销;未提前审批的不予报销。"],
    )
    print("\n".join(f"  {line}" for line in prompt.split("\n")))
    print("  第②条(允许说「不知道」)是最便宜的抗幻觉手段,很多团队恰恰忘了写")


# ---------- 三、Self-Consistency:投票的威力与边界 ----------
def self_consistency_demo(trials=2000, n_chains=20):
    print("── [5] 自洽性:随机错误能投票救回,系统性错误不能(课件第 23 页)──")
    rng = np.random.default_rng(0)

    def run(p_correct, biased):
        """采样 N 条链并对「答案层」多数投票。
        p_correct:单链正确率;biased=False 时错误在两个干扰答案间随机分配,
        True 时错误系统性倒向同一个错误答案(模型「自信地错」)。"""
        vote_ok = consist = 0
        for _ in range(trials):
            answers = ["11" if rng.random() < p_correct
                       else ("21" if biased or rng.random() < 0.5 else "7")
                       for _ in range(n_chains)]
            top, cnt = Counter(answers).most_common(1)[0]
            vote_ok += top == "11"
            consist += cnt >= int(0.9 * n_chains)     # ≥90% 链一致 = 「高一致性」
        return vote_ok / trials, consist / trials

    for tag, pc, b in [("错误随机(两类干扰各半)", 0.55, False), ("错误系统性(自信地错)", 0.05, True)]:
        v, c = run(pc, b)
        print(f"  {tag}: 单链 ≈{pc:.0%} → 投票正确率 ≈ {v:.1%},高一致性比例 ≈ {c:.1%}")
    print("  错误随机:投票把 55% 救回 ~85%,且链间分歧提示「模型不确定」;")
    print("  错误系统性:20 条链一致满分、答案全错 —— 一致性 ≠ 正确性,投票治不了偏差")
    print("  且必须 temperature>0:贪心下 N 条链完全一样,投票毫无意义")


# ---------- 四、LLM-as-a-Judge 的位置偏差与换序校正 ----------
def judge_demo(trials=3000):
    print("── [6] 评委位置偏差:交换顺序 + 只采纳一致结论(课件第 47 页)──")
    rng = np.random.default_rng(0)
    q_a, q_b, pos_bonus, noise = 0.55, 0.50, 0.30, 0.12   # A 略优于 B;先出现者加分

    def judge(first, second):
        """评委打分 = 真实质量 + 位置加成(若在首位) + 噪声,分高者胜。"""
        s = {first: q_a + pos_bonus if first == "A" else q_b + pos_bonus,
             second: q_a if second == "A" else q_b}
        s[first] += rng.normal(0, noise)
        s[second] += rng.normal(0, noise)
        return first if s[first] > s[second] else second

    naive_err = swap_err = disagree = 0
    for _ in range(trials):
        v1 = judge("A", "B")
        v2 = judge("B", "A")
        if rng.random() < 0.5:          # 不校正的基线:随机选一个顺序,只评一次
            naive_err += (v1 != "A")
        else:
            naive_err += (v2 != "A")
        if v1 == v2:
            swap_err += (v1 != "A")     # 两次一致才采纳
        else:
            disagree += 1               # 不一致 → 弃权,送人工
    print(f"  真实情况:A 略优于 B({q_a} vs {q_b});评委对先出现者加 {pos_bonus} 位置分")
    print(f"  不校正(随机顺序评一次):判错比例          = {naive_err / trials:.1%}")
    print(f"  换序校正:判错 / 弃权送人工          = {swap_err / trials:.1%} / {disagree / trials:.1%}")
    assert swap_err <= naive_err
    print("  校正后错误更少,代价是分歧结论弃权送人工 —— 用少量人工换判定可靠")
    print("  纪律:抽 50~100 条人工标注算评委与人的一致率;评分细则写成可逐条判断的清单")


if __name__ == "__main__":
    demo_decoding()
    demo_prompt()
    self_consistency_demo()
    judge_demo()
    print("\n所有自检通过 ✔")
