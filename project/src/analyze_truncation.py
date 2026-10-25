# -*- coding: utf-8 -*-
"""512 截尾影响评估(T5 主体, 计划坑 5 的量化收口; 切分不迁就模型, 只评估)。

回答三个层面的问题(三级口径, 防止单一数字被误读):
  块级暴露: 多少 chunk 的 token 数 > bge max_seq_len=512(编码时尾部被截);
  题级暴露: 多少可答题的 gold chunk 里至少有一个被截;
  证据丢失: 多少题的 gold 证据行**完整落在被截掉的尾部**(答案信息从向量里消失)。

方法(全部确定性, 零 LLM 调用):
  lines.jsonl 逐行按 chunk 正文规则重建(台词行加【说话人】前缀, 旁白不加),
  与 scenes.jsonl 的 text 逐块比对(应 483/483 一致); 逐行 token 数之和+2 应等于
  整块编码 token 数([CLS]/[SEP] 各 1, 可加性验证); 由此得到每行在 chunk 内的
  token 偏移, gold rfid 行的偏移与 510(=512-2 个特殊符) 比较即知证据是否被截。
  已知近似: 截尾上限按 base 臂 tokenizer 计(small 臂仅 1 块边界差异, 见 T4 报告)。

分组对照(截尾×检索命中, 观察性结论——截尾与"长文池化稀释"完全共线, 机制不可分):
  A 组: gold chunk 全部未截;  B 组: gold 被截但证据行存活;  C 组: 证据行完整落入截尾。
  逐臂输出各组 mean Recall@k / MRR(evaluate.py 占比均值口径, 数字取自该 run 的
  retrieval_results.json, 本脚本不重算检索)。

用法:
  python src/analyze_truncation.py --offline                # 最新 run
  python src/analyze_truncation.py --run 20261025-130054 --offline
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.evaluate import RfidLocator, gold_intervals, load_qaset
from src.index_vector import apply_hf_cache_redirect
from src.index_bm25 import load_chunks

KEEP = 510          # 512 - [CLS] - [SEP]: 内容 token 的实际保留上限
DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"   # 与 run 日志 n_truncated 同源(bge 臂)


def render_line(l: dict) -> str:
    """lines.jsonl 行 → chunk 正文行: 台词加【说话人】, 旁白/无说话人不加。"""
    sp = l.get("speaker_norm")
    if sp and sp != "旁白":
        return f"【{sp}】{l['text']}"
    return l["text"]


def latest_run_dir(runs_root: Path) -> Path:
    dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    if not dirs:
        raise SystemExit(f"runs/ 下没有运行目录: {runs_root}")
    return dirs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="512 截尾影响评估(T5)")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--run", default=None, help="runs/ 下的目录名; 缺省取最新")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="截尾判定用的 tokenizer")
    ap.add_argument("--offline", action="store_true", help="HF_HUB_OFFLINE=1(import 前设)")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    run_root = PROJECT_ROOT / cfg["paths"]["runs_dir"]
    run_dir = run_root / args.run if args.run else latest_run_dir(run_root)
    res_path = run_dir / "retrieval_results.json"
    if not res_path.exists():
        raise SystemExit(f"找不到 {res_path}")

    # HF 环境先就位, 再允许 import transformers(坑 6 同源时机约束)
    if args.offline:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    apply_hf_cache_redirect(paths["models"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    chunks = load_chunks(paths["scenes"])
    lines = [json.loads(l) for l in
             (PROJECT_ROOT.parent / "Rewrite_terra" / "data" / "lines.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    by_rfid = {l["rfid"]: l for l in lines}

    # ---- 1. 逐行 token 剖面 + 双重验证(重建一致 / 分词可加) ----
    n_rebuilt = n_additive = 0
    for c in chunks:
        rf = [r for r in range(c["rfid_start"], c["rfid_end"] + 1) if r in by_rfid]
        c["_rfids"] = rf
        toks = [len(tok(render_line(by_rfid[r]), add_special_tokens=False)["input_ids"])
                for r in rf]
        c["_toks"] = toks
        c["_total"] = sum(toks)
        c["_whole"] = len(tok(c["text"], add_special_tokens=True)["input_ids"])
        c["_trunc"] = c["_whole"] > 512
        cum, s = [], 0
        for t in toks:
            cum.append(s)
            s += t
        c["_cum"] = cum
        n_rebuilt += ("\n".join(render_line(by_rfid[r]) for r in rf) == c["text"])
        n_additive += (c["_total"] + 2 == c["_whole"])

    n_trunc = sum(c["_trunc"] for c in chunks)
    over = [c["_whole"] - 512 for c in chunks if c["_trunc"]]
    print(f"== 验证 == 逐行重建一致 {n_rebuilt}/{len(chunks)}  分词可加 {n_additive}/{len(chunks)}")
    print(f"== 块级 == 截尾 {n_trunc}/{len(chunks)} ({n_trunc/len(chunks):.1%})"
          f"  超出 token: mean={statistics.mean(over):.0f} max={max(over)}"
          if over else "无截尾")

    # ---- 2. 题级: gold 证据行的 token 位置 ----
    locator = RfidLocator(chunks)
    starts = [c["rfid_start"] for c in chunks]
    cid2 = {c["chunk_id"]: c for c in chunks}

    def overlap_chunks(s: int, e: int) -> list[str]:
        i = max(bisect_right(starts, s) - 1, 0)
        out = []
        while i < len(chunks) and chunks[i]["rfid_start"] <= e:
            if chunks[i]["rfid_end"] >= s:
                out.append(chunks[i]["chunk_id"])
            i += 1
        return out

    qaset = load_qaset(paths["qaset"])
    qinfo = {}
    for q in qaset["questions"]:
        ivs = gold_intervals(q)
        if not ivs:
            continue
        gold, trunc_any, lost = set(), False, False
        det = []
        for s, e in ivs:
            for cid in overlap_chunks(s, e):
                c = cid2[cid]
                gold.add(cid)
                idx = [k for k, r in enumerate(c["_rfids"]) if s <= r <= e]
                if not idx:
                    continue
                st = c["_cum"][idx[0]]
                en = c["_cum"][idx[-1]] + c["_toks"][idx[-1]]
                if c["_trunc"]:
                    trunc_any = True
                if st >= KEEP:
                    lost = True
                det.append({"chunk_id": cid, "chunk_truncated": c["_trunc"],
                            "evidence_token_span": [st, en], "chunk_total_tokens": c["_total"],
                            "evidence_lost_frac": round(max(0.0, min(1.0, (en - KEEP) / max(1, en - st))), 4)})
        qinfo[q["qid"]] = {"layer": q.get("anchor_layer", "?"), "type": q["type"],
                           "trunc_any": trunc_any, "evidence_lost": lost,
                           "n_gold_chunks": len(gold), "gold_chunks": sorted(gold),
                           "detail": det}

    ans = list(qinfo.values())
    n_eval = len(ans)
    lost_qs = [qid for qid, v in qinfo.items() if v["evidence_lost"]]
    # 去重后的 (题, gold chunk) 实例: detail 行是"区间×chunk"明细, 同一 chunk 可重复出现
    pairs = {(qid, cid) for qid, v in qinfo.items() for cid in v["gold_chunks"]}
    gc = len(pairs)
    gc_tr = sum(1 for _, cid in pairs if cid2[cid]["_trunc"])
    print(f"\n== 题级 == 可答 {n_eval} 题: gold 被截 {sum(1 for v in ans if v['trunc_any'])}"
          f" ({sum(1 for v in ans if v['trunc_any'])/n_eval:.1%})"
          f"  证据完整落入截尾 {len(lost_qs)} ({len(lost_qs)/n_eval:.1%})")
    print(f"   gold chunk 实例 {gc} 个, 其中被截 {gc_tr} ({gc_tr/gc:.1%})")

    # ---- 3. 分组 × 逐臂命中对照(数字取自本 run 的 retrieval_results) ----
    res = json.loads(res_path.read_text(encoding="utf-8"))["per_question"]
    arms = list(res.keys())
    grp = {qid: ("C" if v["evidence_lost"] else ("B" if v["trunc_any"] else "A"))
           for qid, v in qinfo.items()}
    from collections import Counter, defaultdict
    comp = Counter(grp.values())
    group_metrics = {}
    for a in arms:
        group_metrics[a] = {}
        for g in "ABC":
            rows = [e for e in res[a] if grp.get(e["qid"]) == g and "skipped" not in e]
            if not rows:
                continue
            group_metrics[a][g] = {
                "n": len(rows),
                "recall@3": round(statistics.mean(e["recall@3"] for e in rows), 4),
                "recall@5": round(statistics.mean(e["recall@5"] for e in rows), 4),
                "recall@10": round(statistics.mean(e["recall@10"] for e in rows), 4),
                "mrr": round(statistics.mean(e["rr"] for e in rows), 4)}
    print(f"\n== 分组对照 (A=gold未截 n={comp['A']}, B=gold被截证据存活 n={comp['B']},"
          f" C=证据全落截尾 n={comp['C']}; mean R@3/MRR, evaluate.py 口径) ==")
    print(f"  {'arm':<12}{'A: R@3/MRR':>18}{'B: R@3/MRR':>18}{'C: R@3/MRR':>18}")
    for a in arms:
        cells = []
        for g in "ABC":
            m = group_metrics[a].get(g)
            cells.append(f"{m['recall@3']:.3f}/{m['mrr']:.3f}" if m else "-")
        print(f"  {a:<12}{cells[0]:>18}{cells[1]:>18}{cells[2]:>18}")

    # 层内对照(排除层构成混杂; A/B 组逐层 R@3, 臂取 bm25_jieba/bge_small/bge_base)
    print("\n== 层内 A/B 对照 (R@3, 排除层构成混杂) ==")
    by_arm = {"jieba": "bm25_jieba", "small": "bge_small", "base": "bge_base"}
    for layer in ["literal", "semantic_gap", "structural"]:
        for g in ["A", "B"]:
            qids = [q for q in qinfo if qinfo[q]["layer"] == layer and grp[q] == g]
            if not qids:
                continue
            cells = []
            for lbl, a in by_arm.items():
                vals = [next(e["recall@3"] for e in res[a] if e["qid"] == q) for q in qids]
                cells.append(f"{lbl}={statistics.mean(vals):.3f}")
            print(f"  {layer:<13} {g} n={len(qids):2d}  " + "  ".join(cells))

    if lost_qs:
        print(f"\n== 证据全落截尾的题 ({'/'.join(lost_qs)}) ==")
        for a in arms:
            ranks = {qid: next((e["first_hit_rank"] for e in res[a] if e["qid"] == qid), None)
                     for qid in lost_qs}
            print(f"  {a:<12} first_hit_rank: " + "  ".join(f"{q}={r}" for q, r in ranks.items()))

    # ---- 4. 落盘 ----
    out = {"model_tokenizer": args.model, "keep_content_tokens": KEEP,
           "verification": {"rebuilt_lines_match": f"{n_rebuilt}/{len(chunks)}",
                            "token_additivity": f"{n_additive}/{len(chunks)}"},
           "chunk_level": {"n_chunks": len(chunks), "n_truncated": n_trunc,
                           "truncated_ratio": round(n_trunc / len(chunks), 6),
                           "overflow_tokens_mean": round(statistics.mean(over), 1) if over else 0,
                           "overflow_tokens_max": max(over) if over else 0},
           "question_level": {
               "n_evaluated": n_eval,
               "n_gold_truncated": sum(1 for v in ans if v["trunc_any"]),
               "n_evidence_lost": len(lost_qs),
               "gold_chunk_instances": gc, "gold_chunk_truncated": gc_tr,
               "evidence_lost_qids": lost_qs},
           "groups": {"A": comp["A"], "B": comp["B"], "C": comp["C"]},
           "group_metrics": group_metrics,
           "per_question": qinfo,
           "source_run": run_dir.name,
           "note": "截尾与长文池化稀释完全共线, A/B/C 组间差是两机制的联合效应;"
                   "观察性结论, 非因果分解"}
    out_path = run_dir / "truncation_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已落盘: {out_path}")


if __name__ == "__main__":
    main()
