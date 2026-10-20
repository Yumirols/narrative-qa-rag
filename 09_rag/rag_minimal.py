# -*- coding: utf-8 -*-
"""
第 15 讲 · RAG 全链路最小实现(六步:加载 → 切块 → 向量化 → 存储 → 检索 → 生成)
==================================================================================

RAG 的定位:把模型从「知识源」降级为「阅读理解器」—— 它只需要理解给定材料作答,
不需要记住事实。缓解的是事实性幻觉(靠「查」),治不了忠实性幻觉(要靠「管」)。

本文件把六步链路完整跑通,并实现课件的四个考点:
  ② 切块    固定长度+重叠 / 递归切分(段落→句子),关键参数:块大小、重叠、元数据
  ⑤ 检索    混合检索(向量 ∥ 关键词 BM25)+ RRF 融合(Σ 1/(60+rank))+ 重排漏斗
            「召回要快要全,重排要准可以慢」;召回是 RAG 的上限
  ⑥ 生成    三条约束写死(只依据资料 / 资料未提及时必须说明 / 结论标注来源编号)
  评估      上下文召回率 / 上下文精确率(检索端),忠实度 / 拒答正确(生成端)
            测试集必须包含「资料里没有答案」的问题

零依赖约定:嵌入用字符 bigram 哈希词袋(生产环境换 bge 等上下文相关模型,
第 12 讲),生成用「抽取式假 LLM」(选与问题词面重叠最高的句子 + 引用),
因此整条链路离线可跑、确定可复现 —— 链路结构与真实系统完全同构。
"""
import re
import sys
import zlib
from collections import Counter

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------- ① 加载:语料(公司差旅制度,含干扰文档) ----------
DOCS = {
    "制度-住宿标准": "差旅住宿标准：一线城市（北京、上海、广州、深圳）住宿每晚不超过600元，其他城市每晚不超过450元。",
    "制度-超标审批": "超标部分处理：出差前提交《超标审批单》并获得批准的，可凭发票按实际金额报销；未提前审批的，超标部分不予报销。",
    "制度-适用范围": "适用范围：本规定适用于职级P6及以下员工，P7及以上按实际发生额另行审批。",
    "制度-市内交通": "市内交通费凭票据实报销，每天上限80元，打车需注明事由。",
    "制度-年假": "年假规定：员工入职满一年可享受5天带薪年假，满十年增加到10天。",
    "制度-加班餐费": "加班餐费：工作日晚间加班超过两小时可报销餐费，标准每餐不超过30元。",
}

PUNCT = re.compile(r'[，。；：、？！《》()（）\s]')


# ---------- 文本项切分:ASCII 词整存,汉字取相邻 bigram(零分词依赖) ----------
def terms(text):
    text = PUNCT.sub("", text)
    out, buf, cjk = [], [], []
    for ch in text:
        if ch.isascii() and ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            cjk.append(ch)
    if buf:
        out.append("".join(buf))
    out += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return out


def dice(a, b):
    """词面重叠系数:|A∩B|·2 / (|A|+|B|)。这里充当玩具重排器与拒答信号。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return 2 * len(sa & sb) / (len(sa) + len(sb))


# ---------- ② 切块:固定长度+重叠 vs 递归切分 ----------
def chunk_fixed(text, size, overlap):
    """按字数机械切,相邻块保留 overlap 重叠 —— 简单可预测,但可能切断句子。"""
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size]]


def chunk_recursive(text, max_len=60):
    """递归切块:先按句号/分号切,超长再拼 —— 尽量在自然边界切,最常用。"""
    sentences = [s for s in re.split(r"(?<=[。；])", text) if s]
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > max_len:
            chunks.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        chunks.append(cur)
    return chunks


# ---------- ③ 向量化:字符 bigram 哈希词袋(生产换 bge 等,这里保证零依赖) ----------
DIM = 128


def embed(text):
    v = np.zeros(DIM)
    for t in terms(text):
        v[zlib.crc32(t.encode("utf-8")) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def cosine_topk(M, q, k):
    sims = M @ q                       # 已 L2 归一化,内积即余弦相似度
    return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]


# ---------- BM25 关键词检索(第 4 讲公式的最小实现) ----------
class BM25:
    def __init__(self, corpus_terms, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_terms)
        self.avgdl = sum(len(t) for t in corpus_terms) / self.N
        self.tf = [Counter(t) for t in corpus_terms]
        df = Counter()
        for t in corpus_terms:
            df.update(set(t))
        self.idf = {w: np.log((self.N - d + 0.5) / (d + 0.5) + 1) for w, d in df.items()}

    def score(self, q_terms):
        out = []
        for tf in self.tf:
            dl = sum(tf.values())
            s = sum(self.idf.get(w, 0.0) * tf[w] * (self.k1 + 1) /
                    (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    for w in q_terms)
            out.append(s)
        return out


def rrf_fuse(*ranked_lists, k=60):
    """RRF 倒数排名融合:score(d) = Σ_路 1/(k + rank)。两路量纲不同也能融合。"""
    scores = {}
    for ranked in ranked_lists:
        for rank, (doc, _) in enumerate(ranked):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# ---------- 主流程 ----------
def main():
    print("── [1] 切块:固定长度+重叠 vs 递归切分(检索质量的第一道关卡)──")
    doc = DOCS["制度-超标审批"]
    for c in chunk_fixed(doc, size=24, overlap=4):
        print(f"  固定块({len(c):2}字): {c}")
    for c in chunk_recursive(doc):
        print(f"  递归块({len(c):2}字): {c}")
    print("  固定块把「报销；」切成两半;递归块尊重句界。生产还要带来源/章节等元数据")

    # ② 切块(递归) → ③ 向量化 → ④ 存储(矩阵即向量库,规模小不上 FAISS)
    chunk_ids, chunk_texts = [], []
    for doc_id, text in DOCS.items():
        for j, c in enumerate(chunk_recursive(text)):
            chunk_ids.append(f"{doc_id}#chunk{j}")
            chunk_texts.append(c)
    M = np.stack([embed(t) for t in chunk_texts])          # 向量库:n×dim
    bm25 = BM25([terms(t) for t in chunk_texts])
    print(f"── [2] 索引就绪:{len(chunk_texts)} 个块(向量矩阵 {M.shape} + BM25 倒排)──")

    def retrieve(query, k_vec=5, topk=3, verbose=False):
        """⑤ 混合检索:向量 ∥ BM25 → RRF 融合 → 玩具重排(dice)→ top-k。"""
        q_t = terms(query)
        vec_rank = cosine_topk(M, embed(query), k_vec)
        kw_score = bm25.score(q_t)
        kw_rank = sorted(enumerate(kw_score), key=lambda x: -x[1])[:k_vec]
        fused = rrf_fuse(vec_rank, kw_rank)[:8]            # 召回要快要全
        reranked = sorted(fused, key=lambda kv: -dice(q_t, terms(chunk_texts[kv[0]])))
        top = reranked[:topk]                              # 重排要准可以慢
        if verbose:
            print(f"    向量路 top3: {[chunk_ids[i] for i, _ in vec_rank[:3]]}")
            print(f"    关键词路 top3: {[chunk_ids[i] for i, _ in kw_rank[:3]]}")
            print(f"    RRF 融合 top3: {[(chunk_ids[i], round(s, 4)) for i, s in fused[:3]]}")
            print(f"    重排后 top3: {[(chunk_ids[i], round(dice(q_t, terms(chunk_texts[i])), 3)) for i, _ in top]}")
        return top

    def answer(query, sources, refuse_threshold=0.05):
        """⑥ 生成(玩具 LLM):三条约束 + 抽取式作答;词面全无交集 → 拒答。
        真实系统在这里调 API/本地模型,并做引用校验(rfid/来源查表)。"""
        q_t = terms(query)
        best, best_s, src_no = None, 0.0, None
        for no, (ci, _) in enumerate(sources, 1):
            for sent in [s for s in re.split(r"(?<=[。；])", chunk_texts[ci]) if s]:
                s = dice(q_t, terms(sent))
                if s > best_s:
                    best, best_s, src_no = sent.strip(), s, no
        if best_s < refuse_threshold:
            return "资料未提及（制度库中没有与该问题相关的条款）。", None
        return f"{best} [来源{src_no}]", src_no

    print("── [3] 全链路问答:P5 员工杭州出差(课件第 6 页的例题)──")
    q1 = "P5员工去杭州出差3天，住宿每晚700元，有提前审批，能报销多少？"
    top1 = retrieve(q1, verbose=True)
    a1, _ = answer(q1, top1)
    print(f"  问: {q1}")
    print(f"  答: {a1}")
    print("  (真 LLM 在此基础上按三条约束计算:其他城市 450 门槛 + 已审批按实际 700×3=2100)")

    print("── [4] 拒答测试:问一个库里根本没有的问题 ──")
    q_bad = "公司食堂的午饭多少钱一份？"
    top_bad = retrieve(q_bad, verbose=True)
    a_bad, _ = answer(q_bad, top_bad)
    print(f"  问: {q_bad}")
    print(f"  答: {a_bad}")
    assert "资料未提及" in a_bad
    print("  「测试集必须包含无答案问题」—— 这是检验会不会瞎编的唯一方式")

    print("── [5] RAG 四类评估指标:检索端 2 个 + 生成端 2 个 ──")
    qset = [  # (问题, gold 块集合, 期望拒答)
        (q1, {"制度-住宿标准", "制度-超标审批", "制度-适用范围"}, False),
        ("出差住宿没有提前审批，超标的部分能给报吗？", {"制度-超标审批"}, False),
        ("晚上加班超过两小时，吃饭能报销多少钱？", {"制度-加班餐费"}, False),
        (q_bad, set(), True),
    ]
    print(f"  {'问题':24} {'召回率@3':>8} {'精确率@3':>8}  拒答  忠实度")
    ok = True
    for q, gold, should_refuse in qset:
        top = retrieve(q)
        got = {chunk_ids[i].split("#")[0] for i, _ in top}
        recall = len(got & gold) / len(gold) if gold else 1.0
        precision = len(got & gold) / len(top)
        a, _ = answer(q, top)
        refused = "资料未提及" in a
        # 玩具忠实度:答案中的数字必须真的出现在所引用的块里(规则回查;
        # 先剥掉 [来源n] 标记,引用编号本身不是事实数字)
        body = re.sub(r"\[来源\d+\]", "", a)
        nums = re.findall(r"\d+", body) if not refused else []
        cited_texts = [chunk_texts[i] for i, _ in top]
        faithful = all(any(n in t for t in cited_texts) for n in nums)
        ok &= (recall == 1.0) and (refused == should_refuse) and faithful
        rec_s = f"{recall:>8.2f}" if gold else f"{'—':>8}"
        print(f"  {q[:22]:24} {rec_s} {precision:>8.2f}  {'√' if refused == should_refuse else '×'}"
              f"     {'√' if faithful else '×'}")
    assert ok
    print("  召回低 → 检索端问题(切块/嵌入/检索策略);召回高但忠实度低 → 生成端没管住")

    print("── [6] 组装最终提示:三条约束写死(生成阶段的抗幻觉手段)──")
    print("  【资料】")
    for no, (ci, _) in enumerate(top1, 1):
        print(f"  [{no}] ({chunk_ids[ci]}) {chunk_texts[ci]}")
    print(f"  【问题】{q1}")
    print("  【要求】① 只依据上面的资料作答;② 资料未提及时回答「资料未提及」;")
    print("          ③ 每个结论后面标注来源编号 —— 然后做引用校验(规则回查 + judge)")


if __name__ == "__main__":
    main()
    print("\n所有自检通过 ✔")
