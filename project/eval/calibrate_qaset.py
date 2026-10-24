# -*- coding: utf-8 -*-
"""BM25 标定回灌:用 bm25_jieba 参照臂实测每题检索难度,产出层标签建议/干扰源/重叠率报告。

层标签的操作定义 = 「稀疏词典分词检索的不可达性」,参照臂 = bm25_jieba(config 超参)。
这是声明出来的约定(报告可直接引用);bigram 实测与 jieba 有分歧,只作对照列不作参照。

判定规则(随报告落盘 rules 字段):
  零实体锚(题面无任何实体词)           → floor 候选
  gold_chunks ≥ 3                       → structural
  gold_chunks ≤ 2 且 recall@5 == 1      → literal
  gold_chunks ≤ 2 且 脱靶或首命中 > 10  → semantic_gap
  其余(gold_chunks ≤ 2 部分命中)        → 维持现标签;无现标签则 structural 候选(人工裁决)

--apply 写回边界(保守):
  - anchor_layer 仅当 gold_chunks==1 且 建议∈{literal,semantic_gap} 且与现标签不同才改写
    (跨层改判一律留给人工);
  - distractor_note 只在字段缺失时写入(top-3 非金标 chunk,标定回填草稿);
  - unans_subtype / accept_criteria / gold_* 一律不碰。

用法:
  python eval/calibrate_qaset.py eval/qaset_v0_draft.json
  python eval/calibrate_qaset.py eval/qaset_v0_draft.json --apply
  python eval/calibrate_qaset.py eval/qaset_v0_draft.json --arms bigram
输出: eval/qaset_calibration.json(确定性报告,无时间戳,同输入重跑逐字节一致)
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.retrievers import build_retriever
from src.evaluate import RfidLocator, gold_intervals
from src.qaset_analysis import CorpusStats, load_entities, overlap_stats
from src.tokenization import get_tokenizer

SUGGEST_RULES = {
    "floor": "题面零实体锚 → floor 候选",
    "structural_ge3": "gold_chunks ≥ 3 → structural",
    "literal_r5full": "gold_chunks ≤ 2 且 recall@5 == 1 → literal",
    "semantic_gap_miss": "gold_chunks ≤ 2 且脱靶或首命中>10 → semantic_gap",
    "partial_keep": "gold_chunks ≤ 2 部分命中 → 维持现标签/人工裁决",
}


def suggest_layer(n_entities: int, gold_n: int, recall5: float, first_hit: int | None) -> tuple[str, str]:
    """返回 (建议层, 命中的规则名)。"""
    if n_entities == 0:
        return "floor", "floor"
    if gold_n >= 3:
        return "structural", "structural_ge3"
    if recall5 >= 1.0:
        return "literal", "literal_r5full"
    if first_hit is None or first_hit > 10:
        return "semantic_gap", "semantic_gap_miss"
    return "keep", "partial_keep"


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 标定回灌(参照臂 bm25_jieba)")
    ap.add_argument("qaset", default="eval/qaset_v0_draft.json", nargs="?")
    ap.add_argument("--apply", action="store_true", help="按保守边界写回 anchor_layer/distractor_note")
    ap.add_argument("--arms", nargs="*", default=[], help="附加对照臂(仅 bigram 可用)")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / "configs/main.yaml")
    paths = resolve_paths(cfg)
    chunks = [json.loads(l) for l in Path(paths["scenes"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = [json.loads(l) for l in (ROOT.parent / "Rewrite_terra/data/lines.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    qaset_path = ROOT / args.qaset
    qaset = json.loads(qaset_path.read_text(encoding="utf-8"))
    locator = RfidLocator(chunks)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    entities = load_entities(paths.get("user_dict"), lines)
    tok_jieba = get_tokenizer("jieba", paths.get("user_dict"))
    stats = CorpusStats(chunks, entities, tok_jieba)

    ref_entry = {"name": "bm25_jieba", "type": "bm25", "tokenizer": "jieba"}
    ref_index = build_retriever(ref_entry, chunks, paths, index_cfg=cfg["index"])
    extra_indexes = {}
    for arm in args.arms:
        extra_indexes[arm] = build_retriever(
            {"name": f"bm25_{arm}", "type": "bm25", "tokenizer": arm}, chunks, paths,
            index_cfg=cfg["index"])

    per_q_report = []
    flips = []
    for q in qaset["questions"]:
        if q["type"] == "unanswerable":
            continue
        qid = q["qid"]
        ivs = gold_intervals(q)
        gold = locator.gold_chunks(ivs)
        gold_n = len(gold)
        ranked = ref_index.search(q["question"], 10)
        ids = [r["chunk_id"] for r in ranked]
        first_hit = next((i + 1 for i, cid in enumerate(ids) if cid in gold), None)
        recall5 = len(gold & set(ids[:5])) / gold_n

        n_entities = sum(1 for t in tok_jieba(q["question"]) if t in entities)
        sug, rule = suggest_layer(n_entities, gold_n, recall5, first_hit)

        gold_text = "\n".join(chunk_by_id[cid]["text"] for cid in sorted(gold))
        ov = overlap_stats(q["question"], gold_text, stats, tok_jieba)

        distractors = [{"chunk_id": r["chunk_id"], "score": round(r["score"], 4),
                        "snippet": chunk_by_id[r["chunk_id"]]["text"][:24]}
                       for r in ranked if r["chunk_id"] not in gold][:3]

        # 金标命中详情(与 distractors 互补):rank/score 为 None = 该金标 chunk 未进 top-10
        by_id = {r["chunk_id"]: r for r in ranked}
        gold_hits = [{"chunk_id": cid, "rank": by_id[cid]["rank"] if cid in by_id else None,
                      "score": round(by_id[cid]["score"], 4) if cid in by_id else None}
                     for cid in sorted(gold)]

        per_arm = {}
        for arm, idx in extra_indexes.items():
            rk = idx.search(q["question"], 10)
            fh = next((i + 1 for i, r in enumerate(rk) if r["chunk_id"] in gold), None)
            per_arm[arm] = {"first_hit": fh,
                            "recall@5": round(len(gold & {r["chunk_id"] for r in rk[:5]}) / gold_n, 4)}

        cur = q.get("anchor_layer", "")
        per_q_report.append({
            "qid": qid, "type": q["type"], "anchor_layer": cur or None,
            "gold_chunks": gold_n, "first_hit": first_hit,
            "recall@5": round(recall5, 4),
            "overlap": ov["overlap"], "overlap_all": ov["overlap_all"],
            "novel_high_df": ov["novel_high_df"],
            "suggestion": sug, "rule": rule,
            "gold_hits": gold_hits, "distractors": distractors,
            "extra_arms": per_arm,
        })

        mark = ""
        if args.apply and cur and cur != sug and gold_n == 1 and sug in ("literal", "semantic_gap"):
            q["anchor_layer"] = sug
            flips.append(f"{qid}: {cur} -> {sug} ({rule})")
            mark = "  <- 已改写"
        if args.apply and "distractor_note" not in q and distractors:
            q["distractor_note"] = "标定回填: " + "; ".join(
                f"{d['chunk_id']}(score={d['score']})「{d['snippet']}…」" for d in distractors)
            mark = mark or "  <- 干扰源已回填"

        extra = "  ".join(f"{a}:fh={v['first_hit']},r5={v['recall@5']}"
                          for a, v in per_arm.items())
        print(f"{qid} [{q['type']:<11}] 层={cur or '∅':<13} 建议={sug:<13} "
              f"fh={str(first_hit):>4} R@5={recall5:.2f} chunks={gold_n} "
              f"重叠={ov['overlap'] or '∅'} novel高df={ov['novel_high_df'] or '∅'} {extra}{mark}")

    if args.apply:
        qaset_path.write_text(json.dumps(qaset, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        print(f"\n已写回 {qaset_path.name}: anchor_layer 改写 {len(flips)} 处"
              + (f"({'、'.join(flips)})" if flips else ""))

    report = {
        "version": "calibration-1",
        "qaset": qaset_path.name,
        "qaset_version": qaset.get("version"),
        "reference_arm": "bm25_jieba",
        "extra_arms": list(extra_indexes.keys()),
        "rules": SUGGEST_RULES,
        "apply_boundary": "anchor_layer 仅 gold_chunks==1 且 literal↔semantic_gap 自动改写;"
                          "distractor_note 仅缺失时回填",
        "per_question": per_q_report,
    }
    out = ROOT / "eval/qaset_calibration.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告已落盘 {out}(参照臂 bm25_jieba,规则见 rules 字段)")


if __name__ == "__main__":
    main()
