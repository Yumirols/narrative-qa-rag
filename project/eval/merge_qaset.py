# -*- coding: utf-8 -*-
"""候选题并入正式草稿:人工核对后的选题 → qid 重编号 → 写回草稿 → 打印 targets 缺口。

设计约定:
  - targets 是「冻结门槛」而非计数器,本脚本从不自动改写 targets——并题后运行
    validate_qaset.py 的分布对照即自动反映进度;只有目标矩阵本身据实修订时才手改
    (先例:aggregate/semantic_gap 3→0、structural 10→13)。
  - 并入题保留 src_cand 字段回指候选 cand_id(可追溯、天然防重复并入);
    候选侧的 cand_id / recommend / 池用 qid 在并入时丢弃,notes 原样保留
    (草稿期 notes = 核对依据,冻结时按既有约定统一清空)。
  - 硬门槛(任一不过则整批中止,不做部分并入):rfid 查表存在且落 chunk、
    schema v2 字段完整性、与草稿及本批次的问句查重、未并入过。
  - 版本号自动进位:version 尾部的 revN → revN+1。

用法:
  python eval/merge_qaset.py                            # 只读:列池子现状与缺口
  python eval/merge_qaset.py --recommend high           # 并入全部未并入的 high
  python eval/merge_qaset.py --cand-ids S06,S08         # 按编号并入
  python eval/merge_qaset.py --all                      # 并入全部未并入候选
  python eval/merge_qaset.py --recommend high --dry-run # 预演,不写盘
  python eval/merge_qaset.py --remove Q31               # 人工复核否决后移除
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import RfidLocator  # noqa: E402

DATA = ROOT.parent / "Rewrite_terra" / "data"
DRAFT = ROOT / "eval/qaset_v0_draft.json"
POOL = ROOT / "eval/qaset_v1_candidates_aggregate.json"
VALID_TYPES = {"fact", "yesno", "temporal", "aggregate", "unanswerable"}
VALID_LAYERS = {"literal", "semantic_gap", "structural", "floor"}
VALID_UNANS = {"entity_absent", "fact_absent", "mentioned_unknown"}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def norm_question(q: str) -> str:
    return re.sub(r"[\s　、「」『』？?！!。．，,——\-]", "", q)


def check_candidate(q: dict, lines: dict, locator: RfidLocator,
                    seen_questions: set[str], problems: list[str]) -> None:
    """单个候选的硬门槛;问题写进 problems。"""
    cid = q.get("cand_id", q.get("qid", "?"))
    if q.get("type") not in VALID_TYPES:
        problems.append(f"{cid}: 非法题型 {q.get('type')!r}")
    if not q.get("question") or not q.get("gold_answer"):
        problems.append(f"{cid}: 缺 question/gold_answer")
    if norm_question(q.get("question", "")) in seen_questions:
        problems.append(f"{cid}: 问句与已有题重复")
    typ = q.get("type")
    if typ in ("yesno", "unanswerable") and q.get("gold_verdict") not in ("yes", "no", "unknown"):
        problems.append(f"{cid}: gold_verdict 缺失/非法")
    if typ in ("fact", "temporal", "aggregate") and "gold_verdict" in q:
        problems.append(f"{cid}: 可答题不应有 gold_verdict")
    if typ != "unanswerable":
        if q.get("anchor_layer") not in VALID_LAYERS:
            problems.append(f"{cid}: anchor_layer 缺失/非法 {q.get('anchor_layer')!r}")
        if "unans_subtype" in q:
            problems.append(f"{cid}: 可答题不应有 unans_subtype")
    else:
        if q.get("unans_subtype") not in VALID_UNANS:
            problems.append(f"{cid}: unans_subtype 缺失/非法 {q.get('unans_subtype')!r}")
        if q.get("gold_evidence_rfids"):
            problems.append(f"{cid}: 不可回答题金标应为空")
        return
    if typ == "aggregate" and not q.get("accept_criteria"):
        problems.append(f"{cid}: aggregate 题缺 accept_criteria")
    # rfid 存在性 + chunk 覆盖(与 validator 同口径:端点必须存在于 lines.jsonl)
    for item in q.get("gold_evidence_rfids") or []:
        spans = [(item, item)] if isinstance(item, int) else [tuple(item)]
        for s, e in spans:
            if e < s:
                problems.append(f"{cid}: 区间起>止 {spans}")
            for r in {s, e}:
                if r not in lines:
                    problems.append(f"{cid}: rfid {r} 不存在于 lines.jsonl")
            if s in lines and e in lines:
                try:
                    locator.gold_chunks([(s, e)])
                except KeyError as exc:
                    problems.append(f"{cid}: 金标落在 chunk 覆盖空隙 {exc}")


def merge_question(q: dict, new_qid: str) -> dict:
    """候选 → 草稿题:换 qid、丢弃池侧字段、保留 src_cand 溯源。"""
    out = {"qid": new_qid, "type": q["type"]}
    if "gold_verdict" in q:
        out["gold_verdict"] = q["gold_verdict"]
    if q.get("anchor_layer"):
        out["anchor_layer"] = q["anchor_layer"]
    if q.get("unans_subtype"):
        out["unans_subtype"] = q["unans_subtype"]
    out["question"] = q["question"]
    out["gold_answer"] = q["gold_answer"]
    if q.get("accept_criteria"):
        out["accept_criteria"] = q["accept_criteria"]
    out["gold_evidence_rfids"] = q.get("gold_evidence_rfids", [])
    if q.get("distractor_note"):
        out["distractor_note"] = q["distractor_note"]
    if q.get("notes"):
        out["notes"] = q["notes"]
    out["src_cand"] = q["cand_id"]
    return out


def gap_summary(draft: dict) -> None:
    """并题后的实际 vs targets 缺口(validator 分布对照的紧凑版)。"""
    targets = draft.get("targets") or {}
    if not targets:
        return
    layer_counts: dict = {}
    unans_counts: dict = {}
    verdict = {"yes": 0, "no": 0}
    for q in draft["questions"]:
        if q["type"] == "unanswerable":
            unans_counts[q.get("unans_subtype")] = unans_counts.get(q.get("unans_subtype"), 0) + 1
            continue
        layer_counts[(q["type"], q.get("anchor_layer"))] = \
            layer_counts.get((q["type"], q.get("anchor_layer")), 0) + 1
        if q["type"] == "yesno":
            verdict[q.get("gold_verdict")] = verdict.get(q.get("gold_verdict"), 0) + 1
    print("\n--- targets 缺口(负数=还差,正数=超配;冻结要求全部为 0) ---")
    for typ, want in targets.get("by_layer", {}).items():
        for layer, n in want.items():
            got = layer_counts.get((typ, layer), 0)
            print(f"  {typ:<12} {layer:<13} 目标{n:>3} 实际{got:>3}  差 {got - n:+d}")
    for sub, n in targets.get("unans_subtype", {}).items():
        got = unans_counts.get(sub, 0)
        print(f"  {sub:<20} 目标{n:>3} 实际{got:>3}  差 {got - n:+d}")
    yv = targets.get("yesno_verdict", {})
    print(f"  yesno verdict            目标 yes={yv.get('yes')} no={yv.get('no')}"
          f"  实际 yes={verdict['yes']} no={verdict['no']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="候选题并入正式草稿")
    ap.add_argument("--cand-ids", default="", help="逗号分隔的 cand_id 列表")
    ap.add_argument("--recommend", choices=["high", "backup", "medium"], default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", default=None, help="从草稿移除指定 qid(如 Q31)")
    args = ap.parse_args()

    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    lines = {d["rfid"]: d for d in load_jsonl(DATA / "lines.jsonl")}
    locator = RfidLocator(load_jsonl(DATA / "scenes.jsonl"))

    if args.remove:
        qid = args.remove
        before = len(draft["questions"])
        draft["questions"] = [q for q in draft["questions"] if q["qid"] != qid]
        if len(draft["questions"]) == before:
            sys.exit(f"草稿中不存在 {qid}")
        m = re.search(r"rev(\d+)$", draft.get("version", ""))
        if m:
            draft["version"] = draft["version"][:m.start()] + f"rev{int(m.group(1)) + 1}"
        if not args.dry_run:
            DRAFT.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"已移除 {qid}(dry-run={args.dry_run}),version → {draft['version']}")
        gap_summary(draft)
        return

    merged_srcs = {q.get("src_cand") for q in draft["questions"]}
    seen_q = {norm_question(q["question"]) for q in draft["questions"]}
    ids = {s.strip() for s in args.cand_ids.split(",") if s.strip()}
    selected = []
    for q in pool["questions"]:
        cid = q.get("cand_id", q.get("qid"))
        if cid in merged_srcs:
            continue
        if ids and cid in ids:
            selected.append(q)
        elif not ids and (args.all or (args.recommend and q.get("recommend") == args.recommend)):
            selected.append(q)

    if not ids and not args.all and not args.recommend and not args.dry_run:
        merged = sorted(s for s in merged_srcs if s)
        print(f"池子共 {len(pool['questions'])} 题,已并入 {len(merged)} 题:"
              f"{merged or '∅'}")
        for lvl in ("high", "medium", "backup"):
            rest = [q["cand_id"] for q in pool["questions"]
                    if q["cand_id"] not in merged_srcs and q.get("recommend") == lvl]
            print(f"  未并入 {lvl}: {len(rest)} 题 {rest if lvl == 'high' else ''}")
        gap_summary(draft)
        print("\n(只读模式;要实际并入请加 --recommend high / --cand-ids ... / --all)")
        return

    if ids:  # 请求的编号必须都能落实:不存在/已并入都要明说,禁止静默丢弃
        pool_ids = {q.get("cand_id", q.get("qid")) for q in pool["questions"]}
        missing = sorted(ids - pool_ids)
        if missing:
            sys.exit(f"候选池中不存在: {'、'.join(missing)}——检查编号")
        already = sorted(s for s in ids if s in merged_srcs)
        if already:
            print(f"跳过已并入: {'、'.join(already)}")

    print(f"待并入 {len(selected)} 题" + ("(dry-run)" if args.dry_run else ""))
    problems: list[str] = []
    for q in selected:
        check_candidate(q, lines, locator, seen_q, problems)
        seen_q.add(norm_question(q.get("question", "")))
    if problems:
        print(f"\n硬门槛未过 {len(problems)} 项,整批中止:")
        for p in problems:
            print("  !", p)
        sys.exit(1)

    next_n = max((int(m.group(1)) for q in draft["questions"]
                  if (m := re.fullmatch(r"Q(\d+)", q["qid"]))), default=0) + 1
    mapping = []
    for i, q in enumerate(selected):
        new_qid = f"Q{next_n + i}"
        draft["questions"].append(merge_question(q, new_qid))
        mapping.append(f"{q['cand_id']}→{new_qid}")
    m = re.search(r"rev(\d+)$", draft.get("version", ""))
    if m and not args.dry_run:
        draft["version"] = draft["version"][:m.start()] + f"rev{int(m.group(1)) + 1}"

    print("qid 映射:", " ".join(mapping))
    print(f"version → {draft['version']}(targets 从不改写,冻结门槛不变)")
    if args.dry_run:
        print("dry-run:未写盘")
    else:
        DRAFT.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        print(f"已写回 {DRAFT.name}(现 {len(draft['questions'])} 题)")
    gap_summary(draft)
    print("\n后续: python eval/validate_qaset.py eval/qaset_v0_draft.json")


if __name__ == "__main__":
    main()
