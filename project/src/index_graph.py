# -*- coding: utf-8 -*-
"""图扩展召回臂(阶段四 T11,框架 §6.3c)——实体维/转述维/结构维三路进检索。

GraphExpandIndex(base): 第七列检索臂,统一 search(query,k) 接口。臂内三路,
各产出一个 rank 从 1 起的命中列表后 RRF 融合(融合函数复用 index_hybrid.rrf_fuse,
机制与 T6 完全同源——本臂 = 把"多检索器融合"下沉到图机制层):

  实体路(graph_entities.json,T9 零 LLM)——T9.3 门槛:仅高精度时投票。
      识别实体后,仅当 **多实体命中且非空交集 ≤ entity_top** 时,交集内按基座臂
      全量打分重排发全权重票(交集小而具体,如 洲崎∩加岛∩樱→6 块,才是高精度
      证据);单实体命中(瑢太朗 313 chunk 星型)/空交集并集回退/宽交集均
      **不发独立票**(实体集退居过滤角色)。机制迭代依据(消融诊断,2026-10-26):
      无门槛版本把基座候选二次加权,而 gold 常为第一人称叙述、不在实体候选内
      (T9 gold 覆盖诊断:Q24 交集 6 块含 gold 1/5、Q63 交集 0/3),字面 gold
      被系统性压出 top-5(Q31 gold 基座第 2 → 融合第 6);且实体票×邻居票的
      "辅助信号互相叠加"(Q33 融合第 1 = 实体 1/64 + 邻居 1/88)进一步放大毒化。
  卡片路(graph_cards.jsonl,T10 LLM 离线产物):
      事件卡文本 = events[].event 规范化转述 + chunk_summary 建小 BM25
      (483 卡,轻量;分词沿基座臂 bigram——阶段二已证 bigram 臂优于 jieba 臂,
      实测 jieba 卡片路总体回退;leads_to_hint 不入索引,已定决策②)
      → 命中卡回带原 chunk,零分卡过滤。卡片是 LLM 规范化转述,弥合换说法
      lexical gap,是 semantic_gap 层的主增益引擎(消融实测)。
      已知失败面(如实记录,报告第⑥节素材):转述抽签可能再换一套词
      (Q46 铺子→古董店/Q49 供电未抽出)或用规范名替换查询别名(Q10 键之少女
      →篝,单字名 bigram 不可匹配)——此类三向词面失配需稠密卡片向量
      (T13/future work),词法层不可解。
  扩展路(graph_events.json,T10.3 确定性边表)——基座强度载体 + 邻接补充:
      基座臂 top-seed_top 为种子(seed_top=20 先验沿 T6 depth:≥ 评测/生成侧
      一切 k,即 mrr_depth=10 / generation top_k=5 / hybrid depth=20),种子
      按基座排名在前;图上 hop(=1) 跳邻居在后,邻居三源:
        时序前后件(next 边双向,已含分支段互斥跳过)、
        同参与者(has_participant 二部,按共享参与者数加权取 top-neighbor_top,
          即 T10.3 移作检索期的 same_participant_top 截断——主角星型防稀释)、
        同 label(scenes 元数据 label,按 chunk 序距离取 top-neighbor_top),
      总序 = (层, 源优先级[时序>共参与>同label], seed 排名, 源内序, idx)。
      邻居在 RRF 中拿到的是固定的尾部小分值 1/(rrf_k+m+i),即计划所述
      "固定小分值并入融合":邻居只与基座/卡片票叠加时起强化作用,单凭邻居票
      进不了 top-20;种子承载基座字面强度——图信号全空时臂输出 = 基座 top-k
      (literal 不回退的机制保证)。

降级(无 Key 环境红线,与 T12.3 run_all 口径一致):entities 文件缺失报错
(零 LLM 产物,先跑 graph_build.py);cards / events 缺失则对应路静默降级
(stats 与 extra_logs 记录 degraded),实体路 + 扩展路种子仍可用,臂永不全断。

确定性:全部排序键显式固定,同输入同输出(实测重建实例逐指标一致)。
先验默认参数(敏感性纪律沿 T6:不据评测集调参;机制迭代为消融归因驱动,
非参数网格搜索,过程留痕于阶段报告):entity_top/card_top/neighbor_top/hop
取计划 §4 先验值;实体路门槛 cap = entity_top(交集超过路线可承载的条数即
不具体,零新增超参);seed_top=20 沿 T6 depth 先验;rrf_k=60 文献默认;
卡片 BM25 的 k1/b 缺省取 index 节(与 bm25 型同一默认源)。

用法:
    python src/index_graph.py "路易斯是什么国籍？" -k 5    # 三路明细 + 融合结果
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.graph_build import EntityIndex
from src.index_bm25 import load_chunks
from src.index_hybrid import rrf_fuse
from src.retrievers import build_retriever
from src.tokenization import bigram_tokenize

# 扩展路邻居源优先级:时序邻接是最强结构信号,共参与者次之,同 label 为场景上下文
NEIGHBOR_SOURCES = ("temporal", "same_participant", "same_label")


class GraphExpandIndex:
    """图扩展检索器: 基座臂 + 实体/卡片/扩展三路臂内 RRF。

    接口与 BM25Index / HybridRRFIndex 完全同构(search/stats),另有:
      search_routes(query,k) -> 各路命中明细(T11 验收诊断用)
      extra_logs -> runs/<ts>/graph_config_<name>.json(参数/图统计/降级状态/
                    实体路触发计数——确定性计数器, 重跑一致)
    """

    def __init__(self, chunks: list[dict], base, paths: dict,
                 base_name: str = "bm25_bigram", k1: float = 1.5, b: float = 0.75,
                 entity_top: int = 10, card_top: int = 10, neighbor_top: int = 5,
                 hop: int = 1, seed_top: int = 20, rrf_k: int = 60,
                 name: str = "graph_expand"):
        self.chunks = chunks
        self.base = base
        self.base_name = base_name
        self.entity_top, self.card_top = entity_top, card_top
        self.neighbor_top, self.hop = neighbor_top, max(int(hop), 0)
        self.seed_top, self.rrf_k, self.name = seed_top, rrf_k, name
        self._idx_of = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self._cid_of = [c["chunk_id"] for c in chunks]
        # 卡片路分词沿基座臂(bm25 型有 tokenize_fn;基座若为稠密臂则回退 bigram)
        self.tokenize_fn = getattr(base, "tokenize_fn", None) or bigram_tokenize
        # 实体路触发计数(诊断口径: 多实体查询数 / 实际发票数;确定性计数器)
        self.n_entity_multi = 0
        self.n_entity_fired = 0

        # ---- 实体路数据(T9,必需) ----
        ent_path = paths.get("entities")
        if not ent_path or not Path(ent_path).exists():
            raise FileNotFoundError(
                f"graph 检索器 {name!r} 需要 {ent_path or 'paths.entities'}(T9 实体-场景索引)——"
                f"先运行 python src/graph_build.py(零 LLM,确定性)")
        self.entity_index = EntityIndex.from_file(ent_path)

        # ---- 卡片路数据(T10,可选: 缺失降级) ----
        self.card_ok, self.card_bm25 = False, None
        self._card_rows: list[int] = []      # 卡片序 → 语料 idx
        self._card_tokens = 0
        cards_path = paths.get("cards")
        self._cards_path = str(cards_path) if cards_path else None
        if cards_path and Path(cards_path).exists():
            self._build_card_bm25(Path(cards_path), k1, b)

        # ---- 扩展路图数据(T10.3,可选: 缺失降级为纯 seed) ----
        self.graph_ok = False
        self._succ: dict[str, list[str]] = defaultdict(list)
        self._pred: dict[str, list[str]] = defaultdict(list)
        self._parts_of: dict[str, set[str]] = defaultdict(set)   # cid → 实体名集
        self._cards_of: dict[str, set[str]] = defaultdict(set)   # 实体名 → cid 集
        self._label_of = [c.get("label") for c in chunks]
        self._by_label: dict = defaultdict(list)
        for i, lab in enumerate(self._label_of):
            self._by_label[lab].append(i)
        events_path = paths.get("events")
        if events_path and Path(events_path).exists():
            self._load_graph(Path(events_path))

    # ---------- 数据加载 ----------

    def _build_card_bm25(self, cards_path: Path, k1: float, b: float) -> None:
        rows = [json.loads(l) for l in cards_path.read_text(encoding="utf-8").splitlines() if l]
        docs = []
        for r in rows:
            i = self._idx_of.get(r["chunk_id"])
            if i is None:
                continue                     # 防御: 非本语料的卡(冒烟/异档)不入索引
            # 规范化转述 + 概要;leads_to_hint 决策②不入召回依据
            text = " ".join(e["event"] for e in r.get("events", [])) + " " + r.get("chunk_summary", "")
            docs.append((i, text))
        if not docs:
            return
        corpus = [self.tokenize_fn(t) for _, t in docs]
        self._card_rows = [i for i, _ in docs]
        self._card_tokens = sum(len(t) for t in corpus)
        self.card_bm25 = BM25Okapi(corpus, k1=k1, b=b)
        self.card_ok = True

    def _load_graph(self, events_path: Path) -> None:
        """graph_events.json 边表 → 邻接结构(只取 next/has_participant;
        choice_at/branch_to 为交互结构展示, 不入 1 跳扩展——branch_to 去向 chunk
        与挂载 chunk 通常时序相邻, 已被 next 边覆盖)。"""
        doc = json.loads(events_path.read_text(encoding="utf-8"))
        for e in doc["edges"]:
            if e["type"] == "next":
                self._succ[e["src"]].append(e["dst"])
                self._pred[e["dst"]].append(e["src"])
            elif e["type"] == "has_participant":
                self._parts_of[e["src"]].add(e["dst"])
                self._cards_of[e["dst"]].add(e["src"])
        self.graph_ok = True

    # ---------- 三路实现 ----------

    def _pack(self, idxs: list[int], scores: list[float]) -> list[dict]:
        out = []
        for r, (i, s) in enumerate(zip(idxs, scores), 1):
            c = self.chunks[i]
            out.append({"rank": r, "idx": i, "chunk_id": c["chunk_id"],
                        "score": round(s, 4),
                        "rfid_start": c["rfid_start"], "rfid_end": c["rfid_end"]})
        return out

    def _entity_route(self, query: str, base_full: list[dict]) -> list[dict]:
        """T9.3 门槛实体路: 多实体非空交集 ≤ entity_top 才发全权重票;
        交集内按基座全量打分重排(实体不独立排序)。单实体/空交集/宽交集
        → 不发独立票(过滤角色),计数供诊断。"""
        hits = self.entity_index.recognize(query)
        if len(hits) < 2:
            return []
        self.n_entity_multi += 1
        sets_ = [set(self.entity_index.entities[h["entity"]]["chunk_ids"]) for h in hits]
        inter = set.intersection(*sets_)
        if not inter or len(inter) > self.entity_top:
            return []
        self.n_entity_fired += 1
        picked = [h for h in base_full if h["chunk_id"] in inter][:self.entity_top]
        return [dict(h, rank=r) for r, h in enumerate(picked, 1)]

    def _card_route(self, query: str) -> list[dict]:
        """卡片小 BM25 → 命中卡回带原 chunk(零分卡过滤, 防注入 idx 序噪声)。"""
        if not self.card_ok:
            return []
        scores = self.card_bm25.get_scores(self.tokenize_fn(query))
        order = sorted((j for j in range(len(scores)) if scores[j] > 0),
                       key=lambda j: (-scores[j], j))[:self.card_top]
        return self._pack([self._card_rows[j] for j in order], [scores[j] for j in order])

    def _neighbors_of(self, cid: str) -> dict[str, list[str]]:
        """单节点 1 跳邻居, 三源各自定序(截断 neighbor_top)。"""
        out: dict[str, list[str]] = {s: [] for s in NEIGHBOR_SOURCES}
        i = self._idx_of[cid]
        if self.graph_ok:
            temporal = sorted(set(self._pred[cid]) | set(self._succ[cid]),
                              key=lambda c: self._idx_of[c])
            out["temporal"] = temporal
            weight: Counter = Counter()      # 候选 chunk → 共享参与者数(共现加权)
            for p in self._parts_of.get(cid, ()):        # 主角星型在此截断
                for c in self._cards_of[p]:
                    if c != cid:
                        weight[c] += 1
            out["same_participant"] = sorted(weight, key=lambda c: (-weight[c], self._idx_of[c]))[:self.neighbor_top]
        same_label = sorted((j for j in self._by_label[self._label_of[i]] if j != i),
                            key=lambda j: (abs(j - i), j))[:self.neighbor_top]
        out["same_label"] = [self._cid_of[j] for j in same_label]
        return out

    def _expand_neighbors(self, seed_hits: list[dict]) -> list[int]:
        """seed 的 hop 跳邻居 idx 列表, 总序 = (层, 源优先级, seed 序, 源内序, idx)。"""
        visited = {h["idx"] for h in seed_hits}
        ordered: list[int] = []
        frontier = [(h["idx"], h["rank"]) for h in seed_hits]
        for _level in range(self.hop):
            by_source: dict[str, list[tuple[int, int, int]]] = {s: [] for s in NEIGHBOR_SOURCES}
            for seed_idx, seed_rank in frontier:
                for src, nbs in self._neighbors_of(self._cid_of[seed_idx]).items():
                    for pos, nb in enumerate(nbs):
                        j = self._idx_of.get(nb)
                        if j is None or j in visited:
                            continue
                        visited.add(j)
                        by_source[src].append((seed_rank, pos, j))
            frontier = []
            for src in NEIGHBOR_SOURCES:
                for _, _, j in sorted(by_source[src]):
                    ordered.append(j)
                    frontier.append((j, len(ordered)))   # 下一层的展开序
        return ordered

    def search_routes(self, query: str, k: int = 5) -> dict[str, list[dict]]:
        """三路命中明细(验收诊断/CLI 演示);search 即对其 RRF 融合。"""
        seed_m = min(max(self.seed_top, k), len(self.chunks))
        base_full = self.base.search(query, len(self.chunks))    # 基座全量排名
        seeds = base_full[:seed_m]
        nb_idx = self._expand_neighbors(seeds)
        expansion = self._pack([h["idx"] for h in seeds] + nb_idx,
                               [h["score"] for h in seeds] + [0.0] * len(nb_idx))
        return {"entity": self._entity_route(query, base_full),
                "card": self._card_route(query),
                "expansion": expansion}

    def search(self, query: str, k: int = 5) -> list[dict]:
        routes = self.search_routes(query, k)
        return rrf_fuse(list(routes.values()), self.rrf_k, k, self.chunks)

    # ---------- 统计与日志 ----------

    def stats(self) -> dict:
        base_st = self.base.stats()
        # 口径沿 hybrid(对有词数语义的路求和): 基座索引词数 + 卡片语料词数;
        # vocab 为构成标注, 精确统计见 extra_logs
        n_tokens = base_st["n_tokens"] + self._card_tokens
        n_ent = self.entity_index.meta["n_entities"]
        return {"n_chunks": len(self.chunks), "n_tokens": n_tokens,
                "avg_tokens": round(n_tokens / 2, 1),
                "vocab": f"graph({self.base_name}: ent{n_ent}+card+{self.hop}hop)",
                "n_arms": 3}

    @property
    def extra_logs(self) -> dict:
        degraded = ([("card", self._cards_path)] if not self.card_ok else []) + \
                   (["expansion"] if not self.graph_ok else [])
        return {f"graph_config_{self.name}.json": {
            "base": self.base_name,
            "params": {"entity_top": self.entity_top, "card_top": self.card_top,
                       "neighbor_top": self.neighbor_top, "hop": self.hop,
                       "seed_top": self.seed_top, "rrf_k": self.rrf_k},
            "entity_gate": {"rule": "多实体非空交集<=entity_top 才发全权重票(T9.3)",
                            "n_multi_entity_queries": self.n_entity_multi,
                            "n_fired": self.n_entity_fired},
            "routes": {
                "entity": {"n_entities": self.entity_index.meta["n_entities"],
                           "n_choices": self.entity_index.meta["n_choices"]},
                "card": {"ok": self.card_ok, "n_cards": len(self._card_rows),
                         "n_tokens": self._card_tokens},
                "expansion": {"ok": self.graph_ok,
                              "n_next": sum(len(v) for v in self._succ.values()),
                              "n_participant_edges": sum(len(v) for v in self._cards_of.values()),
                              "n_labels": len(self._by_label)},
            },
            "degraded": degraded,
        }}


def main() -> None:
    ap = argparse.ArgumentParser(description="图扩展检索 demo(三路明细 + 融合)")
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])
    graph_entry = next((e for e in cfg["retrievers"] if e["type"] == "graph"), None)
    if graph_entry is None:
        sys.exit("config 的 retrievers 列表无 type: graph 条目")

    # 与 run_all 同一构建路径, 只建 graph 条目引用的基座臂(demo 不触发其余臂)
    built: dict = {}
    base_entry = next((e for e in cfg["retrievers"]
                       if e["name"] == graph_entry.get("base", "bm25_bigram")), None)
    if base_entry is not None:
        built[base_entry["name"]] = build_retriever(
            base_entry, chunks, paths, index_cfg=cfg["index"], built=built)
    g = build_retriever(graph_entry, chunks, paths, index_cfg=cfg["index"], built=built)

    st = g.stats()
    print(f"[{g.name}] base={g.base_name} chunks={st['n_chunks']} vocab={st['vocab']}")
    routes = g.search_routes(args.query, args.k)
    for rname, hits in routes.items():
        print(f"  -- {rname}({len(hits)}) --")
        for h in hits[:3]:
            text = g.chunks[h["idx"]]["text"][:40].replace("\n", " ")
            print(f"     #{h['rank']} {h['chunk_id']} score={h['score']} | {text}")
    print(f"  == RRF 融合 top-{args.k} ==")
    for r in g.search(args.query, args.k):
        text = g.chunks[r["idx"]]["text"][:60].replace("\n", " ")
        print(f"    #{r['rank']} {r['chunk_id']} score={r['score']} "
              f"rfid[{r['rfid_start']},{r['rfid_end']}] | {text}")


if __name__ == "__main__":
    main()
