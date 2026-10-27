# -*- coding: utf-8 -*-
"""事件图谱组装(阶段四 T10.3/10.4)——确定性边表 + networkx 组装落盘。

输入(全部已有产物,零 LLM):graph_cards.jsonl(T10.2 抽取卡)+ graph_entities.json
(T9 实体表/选择肢节点)+ scenes.jsonl(时序真值)→ 输出 data/graph_events.json
(节点表 + 边表 + meta),一条命令确定性完成。

三类边全部确定性推导(已定决策②,不让 LLM 标跨 chunk 关系):

- next(时序):chunk 文件序(= rfid 单调序,已验证)相邻连边;**branch_ids 互斥的
  相邻对不连**——Terra 实测仅 2 对(c169[0]/c170[1] 即 seq20 的 bad_end 分叉、
  c173[0,1]/c174[2] 三级分支),互斥判定与 choices.json 自带的 branches 分解
  互相印证;rfid 顺序 ≠ 游玩时序,框架 §8.1 已注记;
- has_participant(二部):卡片非 suspect 参与者 → 实体名。**实体节点不入本文件**
  (存于 graph_entities.json,边 dst 按名引用,单一数据源);suspect 参与者
  (母神会/派罗/魔物 等真域外提及)不建边——它们是词表增长候选,不是图节点;
  needs_review 卡的 participants=speakers 真值,照常建边;
- choice_at / branch_to:47 个 selbtn_start 节点 → 挂载 chunk / has_branch 选项
  去向(branch_to 的 weight=选项 idx;branches[].chunk_ids 为空的纯控制流分支
  不建边)。5 个 selbtn 续接条目不入节点表(与 selbtn_start 同区域子菜单,
  分支信息冗余,seq 10-13 已覆盖)。

参与者边不做建图期基数截断:计划 §4 的 same_participant_top 移作 T11 检索期
参数(检索时对高基数实体邻接取 top-k)——图数据保持完整,消融轴才可调;
瑚太朗 ~315 度星型对 JSON 毫无压力。

networkx 仅用于组装校验与统计(弱连通组件/度分布),序列化为自有 JSON 格式
(非 pickle),跨版本可读(复现红线)。确定性:同输入重建逐字节一致(--check)。

用法:
    python src/graph_assemble.py            # 组装 + 落盘 + 统计
    python src/graph_assemble.py --check    # 重建与盘上文件逐字节比对
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.graph_build import EntityIndex
from src.index_bm25 import load_chunks

SCHEME_VERSION = "events_v1"
DEFAULT_OUT = PROJECT_ROOT / "data" / "graph_events.json"


def load_cards(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def derive_temporal_edges(chunks: list[dict], gap: int = 1) -> tuple[list[dict], int]:
    """文件序相邻连 next 边;branch_ids 互斥(集合不相交)的相邻对跳过。
    gap>1 时连 i→i+1..i+gap(缺省 1;留作消融,当前只用相邻)。"""
    edges: list[dict] = []
    n_skipped = 0
    for i in range(len(chunks) - 1):
        a = chunks[i]
        for j in range(i + 1, min(i + 1 + gap, len(chunks))):
            b = chunks[j]
            sa, sb = a.get("branch_ids"), b.get("branch_ids")
            if sa and sb and not set(sa) & set(sb):
                if j == i + 1:
                    n_skipped += 1      # 仅相邻对计一次(消融 gap>1 时中间跳过不重复计)
                continue
            edges.append({"src": a["chunk_id"], "dst": b["chunk_id"],
                          "type": "next", "weight": 1})
    return edges, n_skipped


def derive_participant_edges(cards: list[dict], ent_idx: EntityIndex) -> tuple[list[dict], int]:
    """卡片 → 非 suspect 参与者的二部边。参与者必须可在实体表按名解析
    (归一层保证;万一未命中按未解析计数跳过,不崩组装)。"""
    edges: list[dict] = []
    n_unresolved = 0
    for c in cards:
        suspects = set(c["extraction"]["suspect_participants"])
        for p in c["participants"]:
            if p in suspects:
                continue
            if p not in ent_idx.entities:
                n_unresolved += 1
                continue
            edges.append({"src": c["chunk_id"], "dst": p,
                          "type": "has_participant", "weight": 1})
    return edges, n_unresolved


def derive_choice_edges(choices: list[dict]) -> list[dict]:
    """selbtn_start 节点 → 挂载 chunk(choice_at)/ has_branch 选项去向(branch_to,
    weight=选项 idx;空 chunk_ids 的纯控制流分支不建边)。"""
    edges: list[dict] = []
    for c in choices:
        cid = f"choice@seq{c['seq']}"
        edges.append({"src": cid, "dst": c["chunk_id"], "type": "choice_at", "weight": 1})
        for b in c.get("branches", []):
            for target in b.get("chunk_ids", []):
                edges.append({"src": cid, "dst": target, "type": "branch_to",
                              "weight": b["idx"]})
    return edges


def build_nodes(cards: list[dict], choices: list[dict]) -> list[dict]:
    nodes = [{"id": c["chunk_id"], "type": "card", "rfid": c["rfid"],
              "n_events": len(c["events"]),
              "needs_review": c["extraction"]["needs_review"]} for c in cards]
    nodes += [{"id": f"choice@seq{c['seq']}", "type": "choice", "rfid": c["rfid"],
               "seq": c["seq"], "mount": c["chunk_id"],
               "has_branch": c.get("has_branch", False)}
              for c in sorted(choices, key=lambda c: c["seq"])]
    return nodes


def _extraction_lineage(cards: list[dict], log_path: Path) -> dict:
    """meta 里的抽取溯源:cards 可推导的优先,费用/调用量取抽取日志(存在时)。
    日志缺失时置 None——组装仍可完成(费用数字属溯源信息,不阻塞建图)。"""
    lineage = {"template": cards[0]["extraction"]["template"],
               "model": cards[0]["extraction"].get("model"),
               "needs_review": sum(1 for c in cards if c["extraction"]["needs_review"]),
               "n_suspect_cards": sum(1 for c in cards
                                      if c["extraction"]["suspect_participants"])}
    if log_path.exists():
        s = json.loads(log_path.read_text(encoding="utf-8"))["summary"]
        lineage["cost_usd_offpeak"] = s["tokens"]["cost_usd_offpeak"]
        lineage["n_llm_calls"] = s["latency"]["n_calls"]
    else:
        lineage["cost_usd_offpeak"] = None
        lineage["n_llm_calls"] = None
    return lineage


def graph_stats(nodes: list[dict], edges: list[dict]) -> dict:
    """networkx 组装校验与统计(弱连通/度分布;实体节点按边 dst 引用补入,
    不进序列化节点表)。确定性:全部为图的结构量。"""
    import networkx as nx

    g = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(n["id"], type=n["type"])
    for e in edges:
        if e["dst"] not in g:              # 实体名/分支目标(应已在节点表;防御补入)
            g.add_node(e["dst"], type="entity_ref")
        g.add_edge(e["src"], e["dst"], type=e["type"])
    card_ids = {n["id"] for n in nodes if n["type"] == "card"}
    deg = Counter(e["dst"] for e in edges if e["type"] == "has_participant")
    return {
        "n_nodes_serialized": len(nodes),
        "n_entity_refs": g.number_of_nodes() - len(nodes),
        "n_weakly_connected": nx.number_weakly_connected_components(g),
        "largest_wcc_nodes": max(len(c) for c in nx.weakly_connected_components(g)),
        "cards_isolated_in_wcc": sum(1 for c in nx.weakly_connected_components(g)
                                     if len(c) == 1 and card_ids & set(c)),
        "top_participants": deg.most_common(8),
        "participant_degrees": {"p50": sorted(deg.values())[len(deg) // 2],
                                "max": max(deg.values()), "n_entities": len(deg)},
    }


def assemble(cfg: dict, chunks: list[dict], cards: list[dict], ent_idx: EntityIndex,
             cards_log_path: Path) -> dict:
    edges_cfg = cfg.get("graph", {}).get("edges", {})
    gap = edges_cfg.get("temporal_gap", 1)
    temporal, n_skipped = derive_temporal_edges(chunks, gap)
    participant, n_unresolved = derive_participant_edges(cards, ent_idx)
    choice = derive_choice_edges(ent_idx.choices)
    edges = sorted(temporal + participant + choice,
                   key=lambda e: (e["type"], e["src"], e["dst"], e["weight"]))
    nodes = build_nodes(cards, ent_idx.choices)
    by_type = Counter(e["type"] for e in edges)
    doc = {
        "meta": {
            "scheme": SCHEME_VERSION,
            "n_card_nodes": sum(1 for n in nodes if n["type"] == "card"),
            "n_choice_nodes": sum(1 for n in nodes if n["type"] == "choice"),
            "n_edges": len(edges),
            "edges_by_type": dict(sorted(by_type.items())),
            "temporal_skipped_exclusive": n_skipped,
            "participant_unresolved": n_unresolved,
            "extraction": _extraction_lineage(cards, cards_log_path),
            "rules": [
                "next: 文件序(rfid 单调)相邻连边; branch_ids 互斥对不连(实测 2 对, 与 choices branches 互证)",
                "has_participant: 卡片非 suspect 参与者->实体名; 实体节点存 graph_entities.json, 不重复入本文件",
                "choice_at/branch_to: 47 selbtn_start 节点; branch_to weight=选项 idx, 空 chunk_ids 控制流分支不建边",
                "same_participant_top 不在建图期截断(移作 T11 检索期参数, 图数据完整)",
            ],
            "sources": ["graph_cards.jsonl", "graph_entities.json", "scenes.jsonl"],
        },
        "nodes": nodes,
        "edges": edges,
    }
    doc["meta"]["stats"] = graph_stats(nodes, edges)
    return doc


def dump_json(doc: dict) -> str:
    """固定序列化,无时间戳——同输入重建逐字节一致。"""
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="T10.3/10.4 事件图谱组装(零 LLM,确定性)")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", default=None, help="缺省 config paths.events(无则 data/graph_events.json)")
    ap.add_argument("--check", action="store_true", help="重建并与盘上文件逐字节比对")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    for key in ("cards", "entities", "scenes"):
        if not paths.get(key) or not Path(paths[key]).exists():
            sys.exit(f"缺少输入 {key}: {paths.get(key)}(先完成 T9/T10.2)")
    out_path = Path(args.out) if args.out else paths.get("events", DEFAULT_OUT)
    cards_log = Path(paths["cards"]).with_name(
        Path(paths["cards"]).stem + "_extract_log.json")

    chunks = load_chunks(paths["scenes"])
    cards = load_cards(paths["cards"])
    ent_idx = EntityIndex.from_file(paths["entities"])
    if len(cards) != len(chunks):
        sys.exit(f"卡片 {len(cards)} 与 chunk {len(chunks)} 数不一致——先跑全量抽取")
    doc = assemble(cfg, chunks, cards, ent_idx, cards_log)
    blob = dump_json(doc)

    if args.check:
        if not out_path.exists():
            sys.exit(f"[check 失败] 盘上不存在 {out_path},先运行组装")
        old = out_path.read_text(encoding="utf-8")
        if old == blob:
            print(f"[check 通过] 重建与盘上文件逐字节一致({len(blob):,} bytes): {out_path}")
        else:
            sys.exit(f"[check 失败] 重建与 {out_path} 不一致 "
                     f"(盘上 {len(old):,} vs 重建 {len(blob):,} bytes)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(blob, encoding="utf-8", newline="\n")
    m = doc["meta"]
    print(f"== T10.4 事件图谱组装 ==\n  节点 {m['n_card_nodes']} 卡 + "
          f"{m['n_choice_nodes']} 选择肢  边 {m['n_edges']}(分布 {m['edges_by_type']})"
          f"\n  时序互斥跳过 {m['temporal_skipped_exclusive']} 对  参与者未解析 "
          f"{m['participant_unresolved']} 条\n  弱连通组件 {m['stats']['n_weakly_connected']}"
          f"(最大 {m['stats']['largest_wcc_nodes']} 节点)  实体参与度 top5: "
          + ", ".join(f"{e}={d}" for e, d in m["stats"]["top_participants"][:5]))
    print(f"  抽取溯源: 模板 {m['extraction']['template']} 费用 ${m['extraction']['cost_usd_offpeak']}"
          f"(谷) 调用 {m['extraction']['n_llm_calls']} 次")
    print(f"产物: {out_path}")


if __name__ == "__main__":
    main()
