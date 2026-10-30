# -*- coding: utf-8 -*-
"""检索器工厂: config 的 retrievers 条目 → 统一接口的检索器实例。

统一接口(= 结果对比表的"列"抽象,评测代码零改动的前提):
    search(query, k) -> [{rank, idx, chunk_id, score, rfid_start, rfid_end}, ...]
    stats()          -> 索引统计 dict(落 run 日志)

按 type 分发,依赖**延迟导入**:bm25-only 的 run 全程不触发 torch /
sentence_transformers 导入——保持"零 LLM、轻量、可复现"的运行性质。
新增检索器类型 = 在 build_retriever 加一个分支 + 对应 src/index_*.py 实现。

用法(run_all.py):
    for entry in cfg["retrievers"]:
        retriever = build_retriever(entry, chunks, paths, index_cfg=cfg["index"], built=built)
"""
from __future__ import annotations


def build_retriever(entry: dict, chunks: list[dict], paths: dict,
                    index_cfg: dict | None = None, built: dict | None = None):
    """按 config 条目构建检索器。

    entry: {"name", "type", ...各自超参}; name 作 metrics 键(结果表列名)。
    index_cfg: bm25 型的共享默认超参(config 的 index: 节),条目内同名字段可逐臂覆盖。
    built: {name: 检索器实例} 已构建实例表(hybrid 型按名引用臂, 复用实例不重建)。
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

    if rtype == "bge":
        # 稠密臂: 只 import src.index_vector(模块级仅 numpy); torch /
        # sentence_transformers 由 BGEIndex.__init__ 内部延迟导入——
        # bm25-only 运行零 torch 依赖的性质不变(同文件头约束)。
        # 缺省值与 BGEIndex.__init__ 签名一致(device="cpu"/batch_size=32/use_cache=True)。
        from src.index_vector import BGEIndex

        return BGEIndex(chunks, entry["model"], paths["models"],
                        device=entry.get("device", "cpu"),
                        batch_size=entry.get("batch_size", 32),
                        use_cache=entry.get("use_cache", True),
                        name=entry.get("name", "bge"))

    if rtype == "hybrid":
        # 融合臂: 按名引用同级 retrievers 条目对应的已建实例(单一配置源,
        # 臂超参不内联复制);被引用臂必须排在 hybrid 条目之前, 违反即显式报错。
        # 实现见 src/index_hybrid.py, 零 LLM、零新依赖。
        from src.index_hybrid import HybridRRFIndex

        return HybridRRFIndex(entry["arms"], built or {},
                              depth=entry.get("depth", 20),
                              rrf_k=entry.get("rrf_k", 60),
                              name=entry["name"])

    if rtype == "graph":
        # 图扩展臂: 实体路(实体-场景索引, 多实体交集门槛)+ 卡片路(事件卡
        # 小 BM25)+ 扩展路(基座 seed + 图 1 跳邻居)三路臂内 RRF。base 按
        # 名引用同级已建臂(机制同 hybrid 的 arms 引用, 单一配置源);实现见
        # src/index_graph.py。卡片 BM25 超参缺省取 index 节(与 bm25 型同源);
        # seed_top 缺省 20(depth 先验: ≥ 评测/生成侧一切 k)。
        from src.index_graph import GraphExpandIndex

        base_name = entry.get("base", "bm25_bigram")
        if base_name not in (built or {}):
            raise ValueError(
                f"graph 检索器 {entry['name']!r} 的基座臂 {base_name!r} 不在已构建"
                f"实例表中——config 的 retrievers 列表里, 被引用臂必须排在 graph "
                f"条目之前(按 name 复用实例)")
        params = {"k1": 1.5, "b": 0.75}
        params.update(index_cfg or {})
        params.update({key: entry[key] for key in ("k1", "b") if key in entry})
        return GraphExpandIndex(chunks, built[base_name], paths,
                                base_name=base_name,
                                k1=params["k1"], b=params["b"],
                                entity_top=entry.get("entity_top", 10),
                                card_top=entry.get("card_top", 10),
                                neighbor_top=entry.get("neighbor_top", 5),
                                hop=entry.get("hop", 1),
                                seed_top=entry.get("seed_top", 20),
                                rrf_k=entry.get("rrf_k", 60),
                                name=entry["name"])

    raise ValueError(f"未知检索器类型: {rtype!r} (name={entry.get('name')!r})——"
                     f"在 src/retrievers.py 的 build_retriever 注册新分支")
