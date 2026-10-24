# -*- coding: utf-8 -*-
"""qaset 语义落差分析(validate_qaset.py 与 calibrate_qaset.py 共用的唯一实现,防两套口径漂移)。

内容词 = 分词后去停用词/去实体/长度≥2/非纯数字(分词臂用 jieba+自定义词典,与 bm25_jieba
臂同口径,自带全角→半角归一化)。
实体集 = data/user_dict.txt 词条 ∪ lines.jsonl 全部 speaker_norm(专名仅有的两个来源)。

重叠词只在「可区分词」上统计:chunk 级文档频率 df ≤ DF_MAX 的词才有锚定检索的能力,
高频词的表层重叠不构成检索可达性——这是 Q02 类题(题面与金标 chunk 表层有重叠、
BM25 仍全脱靶)不被误判为 literal 的关键。
新词离散度:题面有而金标无的内容词,若其全语料 chunk 级 df ≥ NOVEL_DF_MAX,疑假落差/歧义
(换进去的说法指向了别处剧情,检索器会召回别处)。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

DF_MAX = 20          # 可区分词的 chunk 级 df 上限(483 chunk 语料)
NOVEL_DF_MAX = 30    # 新词 df 达到该值 → 疑假落差/歧义警告

_NUM_RE = re.compile(r"^[0-9０-９.．,%％:：]+$")

# 仅收虚词/问句架构词;实义词的"泛化性"交给 DF_MAX 过滤,不靠停用词表堆砌
STOPWORDS = frozenset("""
的 了 是 在 有 和 不 都 也 就 还 被 把 让 使 对 与 及 或 等 这 那 哪 吗 呢 吧 啊 么
之后 之前 以前 以后 然后 接着 于是 因此 所以 但是 不过 虽然 然而 如果 因为 由于
什么 怎么 怎样 如何 为什么 多少 是否 有没有 可以 会 要 能 得 地 着 过 很 更 最 较
其中 之一 一些 每个 每次 该 此 本 某 各 进行 出来 起来 时候 时 就是 还是 只是 还有
自己 最后 最终 发生 出现 开始 已经 曾经 可能 应该 确实 一起 一样
""".split())


def load_entities(user_dict_path: str | Path | None, lines: list[dict]) -> set[str]:
    """实体词集:自定义词典词条(『瑚太朗 1642 nr』取第一列) ∪ 说话人归一名。"""
    ent: set[str] = set()
    if user_dict_path and Path(user_dict_path).exists():
        for raw in Path(user_dict_path).read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith((";", "#")):
                ent.add(line.split()[0])
    for d in lines:
        s = d.get("speaker_norm") or ""
        if s:
            ent.add(s)
    return ent


def content_tokens(text: str, entities: set[str], tokenize_fn) -> list[str]:
    out = []
    for t in tokenize_fn(text):
        t = t.strip()
        if len(t) < 2 or t in entities or t in STOPWORDS or _NUM_RE.match(t):
            continue
        out.append(t)
    return out


class CorpusStats:
    """全语料 chunk 级内容词 df 表(重叠可区分性与新词离散度共用,构建一次)。"""

    def __init__(self, chunks: list[dict], entities: set[str], tokenize_fn):
        self.entities = entities
        self.df: Counter = Counter()
        for c in chunks:
            self.df.update(set(content_tokens(c["text"], entities, tokenize_fn)))

    def df_of(self, tok: str) -> int:
        return self.df.get(tok, 0)


def overlap_stats(question: str, gold_text: str, stats: CorpusStats,
                  tokenize_fn) -> dict:
    """题面×金标 chunk 的词汇落差测量。

    返回:
      q_tokens      题面内容词
      overlap       可区分重叠词(∩金标 且 df≤DF_MAX)——层标签的判据用它的数量
      overlap_all   表层重叠词(不过 df 过滤,打印供人工核对)
      novel_high_df 新词中 df≥NOVEL_DF_MAX 者(假落差/歧义嫌疑)
    """
    q_toks = content_tokens(question, stats.entities, tokenize_fn)
    gold_toks = set(content_tokens(gold_text, stats.entities, tokenize_fn))
    overlap_all = [t for t in q_toks if t in gold_toks]
    overlap = [t for t in overlap_all if stats.df_of(t) <= DF_MAX]
    novel_high_df = [t for t in q_toks
                     if t not in gold_toks and stats.df_of(t) >= NOVEL_DF_MAX]
    return {"q_tokens": q_toks, "overlap": overlap,
            "overlap_all": overlap_all, "novel_high_df": novel_high_df}
