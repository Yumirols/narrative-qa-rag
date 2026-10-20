# -*- coding: utf-8 -*-
"""评测脚本骨架(知识框架 §8,任务书评分项④)。

检索层(本阶段出真数字,诊断"错在检索还是生成"):
    Recall@k = |gold_chunks ∩ top_k| / |gold_chunks|
    MRR      = mean(1 / 首个命中 gold 的排名)
    gold rfid → chunk: chunk 的 [rfid_start, rfid_end] 已验证严格递增不重叠,
    二分查表确定映射;rfid 落在区间外 → 金标标注错误,立刻报错(防静默错评)。
    gold_evidence_rfids 为空(不可回答题)不计入检索指标,留给答案层拒答评测。

答案层(骨架,Day 4 接入 generate 后喂 eval/answers.jsonl):
    每行 {"qid", "verdict": "yes|no|unknown", "answer", "evidence_rfids"};
    标签准确率 = verdict 与 gold_verdict 一致的题占比(是否发生过/不可回答题);
    拒答正确率 = 不可答题 predicted unknown + 可答题 predicted 非 unknown。

用法: 由 run_all.py 调用;也可独立跑
    python src/evaluate.py --config configs/main.yaml --arm jieba
"""
from __future__ import annotations

import argparse
import json
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

TYPE_LABELS = {
    "fact": "局部事实", "yesno": "是否发生过", "temporal": "时序/过程",
    "aggregate": "跨场景聚合", "unanswerable": "不可回答",
}


# ---------- 评测集 ----------

def load_qaset(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gold_intervals(q: dict) -> list[tuple[int, int]]:
    """gold_evidence_rfids 归一化为 (start, end) 区间: int→(r,r), [s,e]→(s,e)。"""
    out = []
    for item in q.get("gold_evidence_rfids") or []:
        if isinstance(item, int):
            out.append((item, item))
        else:
            out.append((item[0], item[1]))
    return out


# ---------- rfid → chunk 查表 ----------

class RfidLocator:
    def __init__(self, chunks: list[dict]):
        self.starts = [c["rfid_start"] for c in chunks]
        self.ranges = [(c["rfid_start"], c["rfid_end"], c["chunk_id"]) for c in chunks]

    def chunk_of(self, rfid: int) -> str:
        i = bisect_right(self.starts, rfid) - 1
        if i >= 0:
            s, e, cid = self.ranges[i]
            if s <= rfid <= e:
                return cid
        raise KeyError(f"rfid {rfid} 不在任何 chunk 的 rfid 区间内——检查金标标注")

    def chunks_overlapping(self, start: int, end: int) -> set[str]:
        i = max(bisect_right(self.starts, start) - 1, 0)
        out = set()
        while i < len(self.ranges) and self.ranges[i][0] <= end:
            s, e, cid = self.ranges[i]
            if e >= start:
                out.add(cid)
            i += 1
        if not out:
            raise KeyError(f"rfid 区间 [{start},{end}] 不与任何 chunk 相交——检查金标标注")
        return out

    def gold_chunks(self, intervals: list[tuple[int, int]]) -> set[str]:
        out: set[str] = set()
        for s, e in intervals:
            out |= self.chunks_overlapping(s, e)
        return out


# ---------- 检索层指标 ----------

def eval_retrieval(index, locator: RfidLocator, qaset: dict, k_values: list[int],
                   mrr_depth: int, arm: str = "") -> tuple[dict, list[dict]]:
    per_q: list[dict] = []
    ks = sorted(k_values)

    for q in qaset["questions"]:
        entry = {"qid": q["qid"], "type": q["type"], "question": q["question"]}
        ivs = gold_intervals(q)
        if not ivs:
            entry["skipped"] = "gold_evidence_rfids 为空(不可回答题),不计检索指标"
            per_q.append(entry)
            continue

        gold = locator.gold_chunks(ivs)
        ranked = index.search(q["question"], mrr_depth)
        ids = [r["chunk_id"] for r in ranked]
        first_hit = next((i + 1 for i, cid in enumerate(ids) if cid in gold), None)
        entry.update({
            "gold_chunks": sorted(gold),
            "top": [{"rank": r["rank"], "chunk_id": r["chunk_id"], "score": r["score"],
                     "hit": r["chunk_id"] in gold} for r in ranked],
            "first_hit_rank": first_hit,
            **{f"recall@{k}": len(gold & set(ids[:k])) / len(gold) for k in ks},
            "rr": (1.0 / first_hit) if first_hit else 0.0,
        })
        per_q.append(entry)

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        return {"n": n,
                **{f"recall@{k}": round(sum(r[f"recall@{k}"] for r in rows) / n, 4) for k in ks},
                "mrr": round(sum(r["rr"] for r in rows) / n, 4)}

    evaluated = [e for e in per_q if "skipped" not in e]
    by_type: dict[str, dict] = {}
    for t in dict.fromkeys(e["type"] for e in evaluated):
        by_type[t] = agg([e for e in evaluated if e["type"] == t])

    metrics = {
        "arm": arm or index.tokenize_fn.__name__ if hasattr(index, "tokenize_fn") else "",
        "n_questions": len(qaset["questions"]),
        "n_evaluated": len(evaluated),
        "n_skipped_unanswerable": len(per_q) - len(evaluated),
        "overall": agg(evaluated),
        "by_type": by_type,
    }
    return metrics, per_q


# ---------- 答案层指标(骨架,Day 4 接入) ----------

def eval_label_accuracy(qaset: dict, answers_path: Path) -> dict | None:
    """verdict 标签准确率 + 拒答正确率。answers.jsonl 不存在时返回 None。"""
    if not answers_path.exists():
        return None
    gold = {q["qid"]: q for q in qaset["questions"] if q.get("gold_verdict")}
    pred = {}
    with open(answers_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a = json.loads(line)
                pred[a["qid"]] = a["verdict"]

    rows = [(gold[qid], pred[qid]) for qid in gold if qid in pred]
    if not rows:
        return None
    n = len(rows)
    correct = sum(1 for q, v in rows if q["gold_verdict"] == v)
    unans = [(q, v) for q, v in rows if q["type"] == "unanswerable"]
    ans = [(q, v) for q, v in rows if q["type"] != "unanswerable"]
    return {
        "n": n,
        "label_acc": round(correct / n, 4),
        "refusal_correct_unanswerable": round(
            sum(1 for _, v in unans if v == "unknown") / len(unans), 4) if unans else None,
        "no_overrefusal_answerable": round(
            sum(1 for _, v in ans if v != "unknown") / len(ans), 4) if ans else None,
        "n_missing_answers": len(gold) - n,
    }


# ---------- 独立运行 ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="检索层评测")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--arm", default="jieba", choices=["jieba", "raw", "bigram"])
    args = ap.parse_args()

    from src.index_bm25 import BM25Index, load_chunks
    from src.tokenization import get_tokenizer

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])
    index = BM25Index(chunks, get_tokenizer(args.arm, paths.get("user_dict")),
                      k1=cfg["index"]["k1"], b=cfg["index"]["b"],
                      index_speakers=cfg["index"]["index_speakers"])
    qaset = load_qaset(paths["qaset"])
    metrics, per_q = eval_retrieval(index, RfidLocator(chunks), qaset,
                                    cfg["eval"]["k_values"], cfg["eval"]["mrr_depth"], args.arm)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
