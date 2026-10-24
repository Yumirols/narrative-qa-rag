# -*- coding: utf-8 -*-
"""qaset 冻结前校验(T2 冻结流程的机械检查部分,人工复核不可省):
  1. schema: qid 唯一/必填字段/题型枚举/verdict 字段;
  2. 金标 rfid 存在性: 每个点/区间端点必须在 lines.jsonl,且能映射到 chunk(落空即报);
  3. Recall@k 可达性: |gold_chunks| > k 的题单列(检索满分上限 < 1,报告需知);
  4. 时序题证据是否跨 chunk(真正跨场景时序题的覆盖检查);
  5. yesno verdict 配比;名称用字一致性(瑚太朗/瑚太郎);
  6. notes 约定: 草稿期 notes 记核对依据;冻结稿(无 status 字段)必须已清空 notes——
     无 notes = 人工已核对标记, 冻结稿残留 notes 即报错;
  7. schema v2 必填: 可答题 anchor_layer(literal/semantic_gap/structural/floor)、
     unanswerable 题 unans_subtype(entity_absent/fact_absent/mentioned_unknown)、
     aggregate 题 accept_criteria(judge 判分口径);
  8. 分布对照顶层 targets 块 + literal/semantic_gap 题的可区分重叠率机检——
     两者在草稿模式只出警告,冻结模式升级为报错(层标签由 calibrate_qaset.py 标定,
     本脚本只做复核,判定标准见 src/qaset_analysis.py)。
用法: python eval/validate_qaset.py eval/qaset_v0_draft.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import RfidLocator, gold_intervals
from src.qaset_analysis import CorpusStats, load_entities, overlap_stats
from src.tokenization import get_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Rewrite_terra" / "data"
USER_DICT = ROOT / "data/user_dict.txt"
VALID_TYPES = {"fact", "yesno", "temporal", "aggregate", "unanswerable"}
VALID_LAYERS = {"literal", "semantic_gap", "structural", "floor"}
VALID_UNANS = {"entity_absent", "fact_absent", "mentioned_unknown"}
KS = [3, 5, 10]

SPOT_CHECK = ["Q01", "Q02", "Q05", "Q08", "Q09", "Q11", "Q12", "Q13", "Q22", "Q25", "Q30"]


def check_distribution(qaset: dict, layer_counts: dict, unans_counts: dict,
                       verdict: dict, problems: list, warnings: list,
                       frozen_mode: bool) -> None:
    """扩展A: (type, anchor_layer) 二维分布 + unans 子型 + verdict 配比对照顶层 targets。
    草稿模式出警告(=扩容缺口清单),冻结模式进 problems。"""
    targets = qaset.get("targets")
    if not targets:
        warnings.append("无顶层 targets 块,跳过分布对照(schema v2 起冻结稿必备)")
        return
    bucket = problems if frozen_mode else warnings
    print("\n--- 层分布对照(目标 targets vs 实际) ---")
    for typ, want in targets.get("by_layer", {}).items():
        for layer, n in want.items():
            got = layer_counts.get((typ, layer), 0)
            mark = "OK" if got == n else (f"缺 {n - got}" if got < n else f"多 {got - n}")
            print(f"  {typ:<12} {layer:<13} 目标{n:>3} 实际{got:>3}  {mark}")
            if got != n:
                bucket.append(f"分布缺口 {typ}/{layer}: 目标 {n},实际 {got}")
    print("  --- unanswerable 子型 ---")
    for sub, n in targets.get("unans_subtype", {}).items():
        got = unans_counts.get(sub, 0)
        mark = "OK" if got == n else (f"缺 {n - got}" if got < n else f"多 {got - n}")
        print(f"  {sub:<20} 目标{n:>3} 实际{got:>3}  {mark}")
        if got != n:
            bucket.append(f"分布缺口 unanswerable/{sub}: 目标 {n},实际 {got}")
    yv = targets.get("yesno_verdict")
    if yv:
        print(f"  yesno verdict        目标 yes={yv.get('yes')} no={yv.get('no')}"
              f"  实际 yes={verdict.get('yes', 0)} no={verdict.get('no', 0)}")
        for v in ("yes", "no"):
            if verdict.get(v, 0) != yv.get(v):
                bucket.append(f"分布缺口 yesno/{v}: 目标 {yv.get(v)},实际 {verdict.get(v, 0)}")
    floor_max = targets.get("floor_max")
    n_floor = sum(n for (t, l), n in layer_counts.items() if l == "floor")
    if floor_max is not None and n_floor > floor_max:
        bucket.append(f"floor 题 {n_floor} 道超过上限 {floor_max}(floor 只进错误分析不进主表)")


def check_overlaps(qaset: dict, locator: RfidLocator, chunk_by_id: dict,
                   stats: CorpusStats, tok, problems: list, warnings: list,
                   frozen_mode: bool) -> None:
    """扩展B: literal/semantic_gap 题的可区分重叠率机检(判据: semantic_gap ≤1 < literal),
    附新词高 df 假落差检查。启发式辅助,冻结裁决仍归人工。"""
    print("\n--- 可区分重叠率机检(literal/semantic_gap 题;重叠词仅统计 df≤DF_MAX 的可区分词) ---")
    for q in qaset["questions"]:
        layer = q.get("anchor_layer")
        if layer not in ("literal", "semantic_gap"):
            continue
        try:
            gold = locator.gold_chunks(gold_intervals(q))
        except KeyError:
            continue  # 区间越界已在上面的存在性检查报过
        gold_text = "\n".join(chunk_by_id[c]["text"] for c in sorted(gold))
        ov = overlap_stats(q["question"], gold_text, stats, tok)
        flags = []
        if layer == "semantic_gap" and len(ov["overlap"]) > 1:
            flags.append(f"可区分重叠 {len(ov['overlap'])} 词 {ov['overlap']} -> semantic_gap 标签存疑")
        elif layer == "literal" and len(ov["overlap"]) < 2:
            flags.append(f"可区分重叠仅 {ov['overlap'] or '∅'} -> literal 标签可能过严")
        if ov["novel_high_df"]:
            flags.append(f"新词高df {ov['novel_high_df']} -> 疑假落差/歧义(换的说法指向别处剧情?)")
        if flags:
            (problems if frozen_mode else warnings).append(f"{q['qid']}: " + "; ".join(flags))
        print(f"  {q['qid']} [{layer:<13}] 重叠={ov['overlap'] or '∅'}"
              f" 表层={ov['overlap_all'] or '∅'} novel高df={ov['novel_high_df'] or '∅'}"
              f"{'  ! ' + '; '.join(flags) if flags else '  OK'}")


def main() -> None:
    qaset_path = ROOT / sys.argv[1] if len(sys.argv) > 1 else ROOT / "eval/qaset_v0_draft.json"
    qaset = json.loads(qaset_path.read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in (DATA / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = {}
    for l in (DATA / "lines.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            lines[d["rfid"]] = d
    locator = RfidLocator(chunks)
    tok = get_tokenizer("jieba", USER_DICT if USER_DICT.exists() else None)
    entities = load_entities(USER_DICT, list(lines.values()))
    stats = CorpusStats(chunks, entities, tok)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    frozen_mode = "status" not in qaset  # 约定: 冻结时清空 status/notes, 无 notes = 已核对标记

    problems: list[str] = []
    warnings: list[str] = []
    print(f"=== {qaset_path.name}  version={qaset.get('version')}  status={qaset.get('status')}"
          f"  模式={'冻结稿' if frozen_mode else '草稿'}")
    print(f"chunks={len(chunks)}  lines={len(lines)}  questions={len(qaset['questions'])}")

    from collections import Counter
    tc = Counter(q["type"] for q in qaset["questions"])
    print(f"题型分布: {dict(tc)}")
    qid_counts = Counter(q["qid"] for q in qaset["questions"])
    dup_qids = {k: v for k, v in qid_counts.items() if v > 1}
    for qid, n in sorted(dup_qids.items()):
        problems.append(f"qid 重复 {qid} ×{n}(编号必须唯一,同号题的警告/指标会互相串号)")

    gold_chunk_counts: dict[str, int] = {}
    layer_counts: dict = {}
    unans_counts: dict = {}
    verdict_counts: dict = {}
    for q in qaset["questions"]:
        qid, typ = q["qid"], q["type"]
        for field in ("qid", "type", "question", "gold_answer"):
            if not q.get(field):
                problems.append(f"{qid}: 缺字段 {field}")
        if typ not in VALID_TYPES:
            problems.append(f"{qid}: 未知题型 {typ}")
        if typ in ("yesno", "unanswerable") and q.get("gold_verdict") not in ("yes", "no", "unknown"):
            problems.append(f"{qid}: {typ} 题 gold_verdict 缺失/非法")
        if typ in ("fact", "temporal", "aggregate") and "gold_verdict" in q:
            problems.append(f"{qid}: {typ} 题不应有 gold_verdict")
        if typ == "unanswerable" and q.get("gold_evidence_rfids"):
            problems.append(f"{qid}: 不可回答题金标应为空")
        if frozen_mode and "notes" in q:
            problems.append(f"{qid}: 冻结稿未清空 notes(无 notes = 人工已核对标记)")
        # --- schema v2 必填(硬检查,与冻结模式无关) ---
        layer = q.get("anchor_layer")
        if typ != "unanswerable":
            if layer not in VALID_LAYERS:
                problems.append(f"{qid}: anchor_layer 缺失/非法 {layer!r}(枚举 {sorted(VALID_LAYERS)})")
            if "unans_subtype" in q:
                problems.append(f"{qid}: 可答题不应有 unans_subtype")
            layer_counts[(typ, layer)] = layer_counts.get((typ, layer), 0) + 1
        else:
            sub = q.get("unans_subtype")
            if sub not in VALID_UNANS:
                problems.append(f"{qid}: unans_subtype 缺失/非法 {sub!r}(枚举 {sorted(VALID_UNANS)})")
            if layer is not None:
                problems.append(f"{qid}: unanswerable 题不应有 anchor_layer")
            if sub in VALID_UNANS:
                unans_counts[sub] = unans_counts.get(sub, 0) + 1
        if typ == "aggregate" and not q.get("accept_criteria"):
            problems.append(f"{qid}: aggregate 题缺 accept_criteria(LLM-as-judge 的判分口径)")

        ivs = gold_intervals(q)
        rfids = sorted({r for s, e in ivs for r in (s, e)})
        missing = [r for r in rfids if r not in lines]
        if missing:
            problems.append(f"{qid}: rfid 不存在于 lines.jsonl: {missing}")
        if any(e < s for s, e in ivs):
            problems.append(f"{qid}: 区间起>止 {ivs}")
        try:
            g = locator.gold_chunks(ivs)
        except KeyError as exc:
            problems.append(f"{qid}: 金标落在 chunk 覆盖空隙 {exc}")
            continue
        gold_chunk_counts[qid] = len(g)
        if typ != "unanswerable":
            unreachable = [k for k in KS if len(g) > k]
            tag = f"  <- recall@{unreachable} 上限<1" if unreachable else ""
            cross = "跨chunk" if len(g) > 1 else "同chunk"
            print(f"  {qid} [{typ:<12}] gold_lines={len(rfids):>2} gold_chunks={len(g)} {cross}{tag}")

    ys = [q["gold_verdict"] for q in qaset["questions"] if q["type"] == "yesno"]
    verdict_counts = {"yes": ys.count("yes"), "no": ys.count("no")}
    print(f"yesno verdict 配比: yes={verdict_counts['yes']} no={verdict_counts['no']}")
    for q in qaset["questions"]:
        hits = sum(q.get(k, "").count("瑚太郎") for k in ("question", "gold_answer"))
        if hits:
            print(f"  名称用字: {q['qid']} 出现『瑚太郎』{hits} 次(语料人名为『瑚太朗』)")

    check_distribution(qaset, layer_counts, unans_counts, verdict_counts,
                       problems, warnings, frozen_mode)
    check_overlaps(qaset, locator, chunk_by_id, stats, tok,
                   problems, warnings, frozen_mode)

    print(f"\n--- 警告清单({len(warnings)}){'(冻结模式下以上各项升级为报错)' if frozen_mode else '(草稿模式,冻结前需清零)'} ---")
    for w in warnings:
        print("  ~", w)

    print(f"\n--- 问题清单({len(problems)}) ---")
    for p in problems:
        print("  !", p)

    print("\n--- 抽查引文原文 ---")
    by_q = {q["qid"]: q for q in qaset["questions"]}
    for qid in SPOT_CHECK:
        if qid not in by_q:
            continue
        q = by_q[qid]
        print(f"\n[{qid}] {q['question']}")
        print(f"  gold({q.get('gold_verdict', '-')}): {q['gold_answer'][:90]}")
        for item in q.get("gold_evidence_rfids") or []:
            if isinstance(item, int):
                spans = [(item, item)]
            else:
                spans = [item]
            for s, e in spans:
                for r in sorted(x for x in lines if s <= x <= e)[:6]:
                    d = lines[r]
                    print(f"    {r:>6} [{d['speaker_norm']}] {d['text'][:60]}")
                if e - s > 6:
                    print(f"    ... 区间 [{s},{e}] 共 {e - s + 1} 行")


if __name__ == "__main__":
    main()
