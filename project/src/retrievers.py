# -*- coding: utf-8 -*-
"""检索器工厂(T1 检索器化重构): config 的 retrievers 条目 → 统一接口的检索器实例。

统一接口(= 报告对比表的"列"抽象,评测代码零改动的前提):
    search(query, k) -> [{rank, idx, chunk_id, score, rfid_start, rfid_end}, ...]
    stats()          -> 索引统计 dict(落 run 日志)

按 type 分发,依赖**延迟导入**:bm25-only 的 run 全程不触发 torch /
sentence_transformers 导入——保持"零 LLM、轻量、可复现"的运行性质。
新增检索器类型 = 在 build_retriever 加一个分支 + 对应 src/index_*.py 实现。

用法(run_all.py):
    for entry in cfg["retrievers"]:
        retriever = build_retriever(entry, chunks, paths, index_cfg=cfg["index"])
"""
from __future__ import annotations


def build_retriever(entry: dict, chunks: list[dict], paths: dict,
                    index_cfg: dict | None = None):
    """按 config 条目构建检索器。

    entry: {"name", "type", ...各自超参}; name 作 metrics 键(报告列名)。
    index_cfg: bm25 型的共享默认超参(config 的 index: 节),条目内同名字段可逐臂覆盖。
    """
    rtype = entry["type"]

    if rtype == "bm25":
        from src.index_bm25 import BM25Index
        from src.tokenization import get_tokenizer

        params = {"k1": 1.5, "b": 0.75, "index_speakers": False}
        params.update(index_cfg or {})
        params.update({k: entry[k] for k in ("k1", "b", "index_speakers") if k in entry})
        return BM25Index(chunks,
                         get_tokenizer(entry["tokenizer"], paths.get("user_dict")),
                         k1=params["k1"], b=params["b"],
                         index_speakers=params["index_speakers"])

    if rtype == "w2v":
        from src.index_w2v import W2VAvgIndex
        from src.tokenization import get_tokenizer

        return W2VAvgIndex(chunks,
                           get_tokenizer(entry["tokenizer"], paths.get("user_dict")),
                           dim=entry.get("dim", 100), window=entry.get("window", 2),
                           neg_k=entry.get("neg_k", 4), epochs=entry.get("epochs", 15),
                           lr=entry.get("lr", 0.05), seed=entry.get("seed", 0),
                           min_count=entry.get("min_count", 1), name=entry["name"])

    raise ValueError(f"未知检索器类型: {rtype!r} (name={entry.get('name')!r})——"
                     f"在 src/retrievers.py 的 build_retriever 注册新分支")
