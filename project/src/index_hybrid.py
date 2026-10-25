# -*- coding: utf-8 -*-
"""混合召回 + RRF 融合(T6)——多路检索结果的确定性融合层。

动机(T5 分层互补实证, runs/20261025-154043/metrics.json):
    literal 层      bm25_bigram R@3=0.8889 碾压 bge_base 0.5833(字面匹配);
    semantic_gap 层 bge_base   R@3=0.4318 反超 bigram 0.1818 约 2.4 倍(同义改写);
    两范式强项落在不同题层 → 各取 top-depth 后 RRF 融合, 应同时保住两层。
    本模块回答"互补性是否转化为总体增益"(T6 核心问题, 数字见阶段报告)。

RRF(reciprocal rank fusion, Cormack et al. 2009):
    score(d) = Σ_arms 1 / (rrf_k + rank_d_in_arm)     rank 从 1 起, rrf_k=60
    只用排名不用原始分数 → BM25 分数与 bge 余弦天然不可比的问题被绕开,
    无量纲、无超参缩放;rrf_k=60 为文献默认, 融合对 k 的取值不敏感(报告讨论点)。

确定性(阶段纪律: 同配置重跑 metrics 逐字节一致):
    各臂自身确定性(BM25 稳定排序 / bge CPU 推理 / 固定 seed w2v)+ 融合层
    纯算术 → 同分 tie-break 按 idx 升序(与 BM25/W2V/BGE 的"同分按 idx"同一契约)。

设计决策:
  1. 臂实例复用而非重建——config 的 hybrid 条目按 name 引用同级 retrievers
     条目(单一配置源, 不内联复制臂超参), 构建时由调用方(run_all)传入已建好的
     臂实例表。被引用臂必须排在 hybrid 条目**之前**(否则 w2v 重训练 3.7min /
     bge 重载模型, 且顺序约束给了显式报错)。融合层零自身索引、零新依赖。
  2. depth(每臂取 top-depth 进融合)默认 20 ≥ 评测/生成侧一切 k(mrr_depth=10,
     generation top_k=5);search(query, k) 传入 k > depth 时自动抬到 k, 不截断。
  3. 输出 dict 与单臂完全同构(rank/idx/chunk_id/score/rfid 起止), score 换为
     RRF 融合分(量纲 1/(rrf_k+rank), 与单臂分数不可比, 仅作排序展示)——
     eval_retrieval 零改动, hybrid 自动成为主表第六列。

用法(run_all 自动; 也可独立 demo):
    python src/index_hybrid.py "路易斯是什么国籍？" -k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.index_bm25 import load_chunks
from src.retrievers import build_retriever


class HybridRRFIndex:
    """RRF 融合检索器: 持有 N 个已建好的臂实例, search 时各取 top-depth 后融合。

    接口与 BM25Index / W2VAvgIndex / BGEIndex 完全同构(search/stats), 另有:
      extra_logs -> runs/<ts>/hybrid_config_<name>.json(臂构成/depth/rrf_k/各臂 stats)
    """

    def __init__(self, arm_names: list[str], built: dict, depth: int = 20,
                 rrf_k: int = 60, name: str = "hybrid"):
        missing = [a for a in arm_names if a not in built]
        if missing:
            raise ValueError(
                f"hybrid 检索器 {name!r} 引用的臂 {missing} 不在已构建实例表中——"
                f"config 的 retrievers 列表里, 被引用臂必须排在 hybrid 条目之前"
                f"(按 name 复用实例, 避免 w2v 重训练/bge 重载模型)")
        if len(set(arm_names)) != len(arm_names):
            raise ValueError(f"hybrid 检索器 {name!r} 的 arms 存在重复: {arm_names}")

        self.arm_names = list(arm_names)
        self.arms = [built[a] for a in arm_names]
        self.chunks = self.arms[0].chunks   # 各臂共用同一语料列表, idx 对齐
        self.depth, self.rrf_k, self.name = depth, rrf_k, name

    def search(self, query: str, k: int = 5) -> list[dict]:
        depth = max(self.depth, k)
        fused: dict[int, float] = {}
        for arm in self.arms:
            for hit in arm.search(query, depth):   # 臂的 rank 从 1 起
                fused[hit["idx"]] = fused.get(hit["idx"], 0.0) + 1.0 / (self.rrf_k + hit["rank"])
        # 同分按 idx 升序(与各单臂的稳定排序契约一致)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        out = []
        for rank, i in enumerate(order, 1):
            c = self.chunks[i]
            out.append({"rank": rank, "idx": i, "chunk_id": c["chunk_id"],
                        "score": round(fused[i], 6),
                        "rfid_start": c["rfid_start"], "rfid_end": c["rfid_end"]})
        return out

    def stats(self) -> dict:
        # n_tokens 对各臂求和(单位随臂: 词数或字符数, 仅满足 run_all 的打印键,
        # 语义定义见 extra_logs); vocab 标注融合构成, 不编造整数词表。
        arm_stats = [a.stats() for a in self.arms]
        n_tokens = sum(s["n_tokens"] for s in arm_stats)
        return {"n_chunks": len(self.chunks), "n_tokens": n_tokens,
                "avg_tokens": round(n_tokens / len(self.arms), 1),
                "vocab": f"rrf({'+'.join(self.arm_names)})",
                "n_arms": len(self.arms)}

    @property
    def extra_logs(self) -> dict:
        return {f"hybrid_config_{self.name}.json": {
            "arms": self.arm_names,
            "depth": self.depth,
            "rrf_k": self.rrf_k,
            "arm_stats": dict(zip(self.arm_names, (a.stats() for a in self.arms))),
        }}


def main() -> None:
    ap = argparse.ArgumentParser(description="hybrid RRF 检索 demo(臂构成读 config)")
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])

    # 与 run_all 同一构建路径, 但只建 hybrid 条目引用的臂(demo 不触发 w2v 训练)
    hyb_entry = next((e for e in cfg["retrievers"] if e["type"] == "hybrid"), None)
    if hyb_entry is None:
        sys.exit("config 的 retrievers 列表无 type: hybrid 条目")
    built: dict = {}
    for entry in cfg["retrievers"]:
        if entry["name"] in hyb_entry["arms"]:
            built[entry["name"]] = build_retriever(
                entry, chunks, paths, index_cfg=cfg["index"], built=built)
    hyb = build_retriever(hyb_entry, chunks, paths, index_cfg=cfg["index"], built=built)

    st = hyb.stats()
    print(f"[{hyb.name}] arms={'+'.join(hyb.arm_names)} depth={hyb.depth} rrf_k={hyb.rrf_k} "
          f"chunks={st['n_chunks']}")
    for r in hyb.search(args.query, args.k):
        text = hyb.chunks[r["idx"]]["text"][:60].replace("\n", " ")
        print(f"  #{r['rank']} {r['chunk_id']} score={r['score']} "
              f"rfid[{r['rfid_start']},{r['rfid_end']}] | {text}")


if __name__ == "__main__":
    main()
