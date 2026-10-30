# -*- coding: utf-8 -*-
"""生成层——检索 top-k 证据块 → Prompt 组装 → DeepSeek → 结构化答案落盘。

管线: 问题 → hybrid 检索 top-k → 证据块组装(chunk 全文原文)
→ 系统提示 + few-shot + 用户消息 → src/llm.py 客户端 → JSON 解析(失败重试一次)
→ 引用规则校验 → answers.jsonl + generation_log.json。

约束与设计:
  - 检索/展示分离: 检索用分词/向量, 展示永远 chunk 全文原文(全角保留);
  - few-shot 证据取自语料真实 chunk 且**排除当前评测集全部 gold chunk**,
    提问为元数据锚定(说话人出场/喜好类), 绝不引入评测集题目本体(防泄漏——
    unanswerable 里真有"生日"题, 拒答示例话题已避开);
  - 不可回答题照走完整流程: 检索照样返回字面相似块, 拒答只靠 prompt 约束
    ——这正是要测的东西;
  - evidence_rfids 归一解析(int/[起,止] 兼容)后做两条确定性校验
    (RfidLocator 对孤儿 rfid 抛 KeyError 必须 catch, 不崩评测):
      (a) 区间可落到具体 chunk(孤儿/越界 → invalid);
      (b) 引用 chunk ∈ 本次送入 prompt 的证据集合(防"引用了没给它的块")。
  - 两次解析都失败不丢弃该题: 落一行 parse_failed 答案(verdict=unknown),
    计入失败率;
  - answers.jsonl 行内不含时间戳/延迟/缓存标记等非确定字段——缓存命中重跑
    逐字节一致;usage/延迟/原始响应全在 generation_log.json。

用法:
    python src/generate.py --config configs/main.yaml                 # 90 题正式集
    python src/generate.py --qaset eval/qaset_seed.json --out eval/answers_smoke.jsonl
    python src/generate.py --limit 3 ...                              # 快速试跑
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
from src.evaluate import RfidLocator, gold_intervals, load_qaset
from src.index_bm25 import load_chunks
from src.llm import LLMClient, cost_usd
from src.retrievers import build_retriever

SYSTEM_PROMPT = """你是叙事剧本问答助手。回答只能依据用户消息中给出的【证据】,严禁使用证据之外的任何知识,严禁推测编造。

输出协议(强制):只输出一个合法的 JSON 对象,不输出任何其他文字、解释或代码块标记。JSON 格式示例:
{"answer": "……", "verdict": "yes", "evidence_rfids": [[44, 89]]}

字段规则:
- answer:用中文依据证据作答;证据不足时说明缺少什么信息。
- verdict:问题若是能用"是/否"回答的判断题,填 "yes" 或 "no";证据不足以判断、问题与证据无关、或问题不是是非题时,一律填 "unknown"。
- evidence_rfids:数组,元素为 [起, 止] 形式的 rfid 区间(单点可写 [起, 起]);只允许引用本次证据块头部方括号内标注的 rfid 区间,列出全部支持回答的证据。

拒答规则:证据不足以回答、或证据与问题无关时,verdict 必须填 "unknown",answer 说明缺少什么,evidence_rfids 填 []。宁可拒答,不可猜测。"""

RETRY_HINT = ("你上一次的输出不是合法的 JSON 对象,不符合输出协议。请重新作答:"
              "只输出一个符合协议的 JSON 对象(含 answer/verdict/evidence_rfids "
              "三个字段),不要包含任何其他文字或代码块标记。")

VERDICTS = {"yes", "no", "unknown"}


# ---------- 检索臂(只建 generation.retriever 引用的臂, 不触发 w2v 训练) ----------

def build_generation_retriever(cfg: dict, chunks: list[dict], paths: dict):
    entries = {e["name"]: e for e in cfg["retrievers"]}
    name = cfg["generation"]["retriever"]
    if name not in entries:
        raise ValueError(f"generation.retriever={name!r} 不在 retrievers 列表中")
    built: dict = {}
    for arm in entries[name].get("arms", []):   # hybrid: 先建被引用臂
        built[arm] = build_retriever(entries[arm], chunks, paths,
                                     index_cfg=cfg["index"], built=built)
    return build_retriever(entries[name], chunks, paths,
                           index_cfg=cfg["index"], built=built)


# ---------- Prompt 组装 ----------

def format_evidence(rank: int, chunk: dict) -> str:
    """证据块 = 头部(chunk_id / rfid 区间 / speakers)+ chunk 全文原文(展示分离)。"""
    speakers = "、".join(chunk["speakers"]) if chunk["speakers"] else "无"
    return (f"[证据{rank} | {chunk['chunk_id']} | rfid {chunk['rfid_start']}-"
            f"{chunk['rfid_end']} | 出场: {speakers}]\n{chunk['text']}")


def build_user_message(question: str, hits: list[dict], chunk_by_id: dict) -> str:
    blocks = "\n\n".join(format_evidence(h["rank"], chunk_by_id[h["chunk_id"]])
                         for h in hits)
    return (f"【证据】(按检索融合排名排序)\n{blocks}\n\n【问题】{question}\n\n"
            f"请依据以上证据,只输出符合系统提示 JSON 协议的一个 JSON 对象。")


def build_fewshot(chunks: list[dict], qaset: dict, locator: RfidLocator) -> list[dict]:
    """3 例 few-shot: 正常作答(非是非题→verdict unknown)/ 是非题 yes / 拒答。

    证据块取语料真实 chunk, 但排除当前评测集的**全部 gold chunk**——示例若与
    某题 gold 证据同块, 其问答内容可能泄漏该题答案;题目本体
    则全部另行编写(元数据锚定), 与评测集题目无重叠。选择按语料序确定性取前 3。
    """
    gold_ids: set[str] = set()
    for q in qaset["questions"]:
        if gold_intervals(q):
            try:
                gold_ids |= locator.gold_chunks(gold_intervals(q))
            except KeyError:
                pass
    band = lambda lo, hi: [c for c in chunks if c["chunk_id"] not in gold_ids
                           and lo <= c["n_chars"] <= hi and len(c["speakers"]) >= 2]
    cands = band(200, 600) or band(100, 900)
    if len(cands) < 3:
        raise RuntimeError(f"few-shot 候选 chunk 不足(需 3, 得 {len(cands)})——"
                           f"检查语料/评测集 gold 覆盖")
    a, b, c = cands[0], cands[1], cands[2]

    def qa(chunk: dict, question: str, answer_obj: dict) -> tuple[dict, dict]:
        user = (f"【证据】(按检索融合排名排序)\n{format_evidence(1, chunk)}\n\n"
                f"【问题】{question}\n\n"
                f"请依据以上证据,只输出符合系统提示 JSON 协议的一个 JSON 对象。")
        asst = json.dumps(answer_obj, ensure_ascii=False)
        return {"role": "user", "content": user}, {"role": "assistant", "content": asst}

    # 例1 正常作答: 元数据锚定(说话人), 非是非题 → verdict=unknown(协议约定)
    names = "、".join(a["speakers"])
    m1 = qa(a, "这段证据中出现了哪些说话人?",
            {"answer": f"这段证据中出现的说话人有:{names}。",
             "verdict": "unknown",
             "evidence_rfids": [[a["rfid_start"], a["rfid_end"]]]})

    # 例2 是非题: 说话人确实发言 → yes(教 verdict 判断机制)
    sp = next((s for s in b["speakers"] if f"【{s}】" in b["text"]), b["speakers"][0])
    m2 = qa(b, f"证据中「{sp}」有发言出场吗?",
            {"answer": f"有。证据中出现了「{sp}」标记的发言。",
             "verdict": "yes",
             "evidence_rfids": [[b["rfid_start"], b["rfid_end"]]]})

    # 例3 拒答: 喜好类信息证据必无 → unknown + 说明缺口 + 空引用
    # (话题避开评测集不可回答题——其中有"生日"题)
    sp_c = c["speakers"][0]
    m3 = qa(c, f"「{sp_c}」最喜欢的食物是什么?",
            {"answer": f"证据中没有关于「{sp_c}」喜欢什么食物的任何信息,无法回答。",
             "verdict": "unknown",
             "evidence_rfids": []})

    out = [ {"role": "system", "content": SYSTEM_PROMPT} ]
    for u, a_ in (m1, m2, m3):
        out += [u, a_]
    return out


# ---------- 解析归一与引用校验(确定性) ----------

def schema_ok(parsed) -> bool:
    """协议最低要求: 是对象, answer 为字符串, evidence_rfids 为数组(归一入口)。"""
    return (isinstance(parsed, dict) and isinstance(parsed.get("answer"), str)
            and isinstance(parsed.get("evidence_rfids"), list))


def normalize_rfids(raw) -> tuple[list[tuple[int, int]], int]:
    """evidence_rfids → [(s,e), ...];复用 gold_intervals 的 int/区间兼容口径。

    不可归一的条目(非整数/非二元组/起>止)计 malformed 不抛异常——模型输出
    漂移不弄崩评测, 由引用校验标 invalid。
    """
    intervals: list[tuple[int, int]] = []
    malformed = 0
    if not isinstance(raw, list):
        return [], 1
    for item in raw:
        if isinstance(item, bool):   # bool 是 int 子类, 显式排除
            malformed += 1
        elif isinstance(item, int):
            intervals.append((item, item))
        elif (isinstance(item, (list, tuple)) and len(item) == 2
              and all(isinstance(x, int) and not isinstance(x, bool) for x in item)
              and item[0] <= item[1]):
            intervals.append((item[0], item[1]))
        else:
            malformed += 1
    return intervals, malformed


def validate_citations(intervals: list[tuple[int, int]], locator: RfidLocator,
                       evidence_ids: set[str], corpus_max: int) -> list[tuple[tuple[int, int], str | None]]:
    """两条规则校验, 返回 [(区间, 失败原因|None)];原因 None = 该条引用有效。

    (a) 区间能落到具体 chunk: 越界/孤儿 rfid(3281/14785/14898 等)由
        RfidLocator 抛 KeyError, 此处 catch 标 invalid(不崩评测);
    (b) 区间覆盖的 chunk 全部 ∈ 本次送入 prompt 的证据集合。
    """
    out = []
    for s, e in intervals:
        reason = None
        if e > corpus_max:
            reason = "out_of_corpus"
        else:
            try:
                if not locator.chunks_overlapping(s, e) <= evidence_ids:
                    reason = "not_in_prompt"
            except KeyError:
                reason = "orphan_unlocatable"
        out.append(((s, e), reason))
    return out


# ---------- 单题生成 ----------

def _attempt_log(r: dict) -> dict:
    return {"cache_hit": r["cache_hit"], "parse_path": r["parse_path"],
            "finish_reason": r.get("finish_reason"), "model": r.get("model"),
            "latency_seconds": r.get("latency_seconds"),
            "usage": r.get("usage"), "content": r.get("content"),
            "request_hash": r["request_hash"]}


def generate_one(client: LLMClient, q: dict, messages: list[dict], hits: list[dict],
                 locator: RfidLocator, corpus_max: int) -> tuple[dict, dict]:
    evidence_ids = {h["chunk_id"] for h in hits}
    attempts: list[dict] = []

    r1 = client.chat_json(messages)
    attempts.append(_attempt_log(r1))
    parsed = r1.get("parsed") if schema_ok(r1.get("parsed")) else None

    if parsed is None:   # 解析失败 → 重试一次, 附协议纠错提示(7.2)
        retry_messages = messages + [
            {"role": "assistant", "content": r1.get("content") or ""},
            {"role": "user", "content": RETRY_HINT}]
        r2 = client.chat_json(retry_messages)
        attempts.append(_attempt_log(r2))
        if schema_ok(r2.get("parsed")):
            parsed = r2["parsed"]

    if parsed is None:   # 两次全败: 不丢题, 落 parse_failed 行(计入失败率)
        return ({"qid": q["qid"], "verdict": "unknown",
                 "answer": "(生成失败:两次输出均未解析出合法 JSON 对象)",
                 "evidence_rfids": [], "parse_failed": True},
                {"qid": q["qid"], "type": q["type"], "attempts": attempts,
                 "prompt_sha": attempts[0]["request_hash"], "parse_failed": True,
                 "citations": []})

    verdict_raw = str(parsed.get("verdict", "unknown")).strip().lower()
    verdict = verdict_raw if verdict_raw in VERDICTS else "unknown"
    intervals, n_malformed = normalize_rfids(parsed.get("evidence_rfids"))
    checked = validate_citations(intervals, locator, evidence_ids, corpus_max)
    n_invalid = sum(1 for _, r in checked if r is not None)
    row = {"qid": q["qid"], "verdict": verdict,
           "answer": parsed.get("answer", "").strip(),
           "evidence_rfids": [[s, e] for (s, e), _ in checked]}
    if verdict != verdict_raw:
        row["verdict_raw"] = verdict_raw
    if n_malformed:
        row["n_malformed_rfids"] = n_malformed
    # n_invalid_citations 供 evaluate.eval_citations 从 answers.jsonl 直接算条级有效率
    # 向后兼容字段, 旧 answers 行缺省按 0 无效
    if n_invalid:
        row["n_invalid_citations"] = n_invalid
    row["all_citations_valid"] = n_invalid == 0 and n_malformed == 0
    log = {"qid": q["qid"], "type": q["type"], "attempts": attempts,
           "prompt_sha": attempts[0]["request_hash"],
           "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
           "citations": [{"rfid": [s, e], "reason": reason} for (s, e), reason in checked]}
    return row, log


# ---------- 汇总 ----------

def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return None
    i = min(int(len(sorted_vals) * p + 0.999999) - 1, len(sorted_vals) - 1)
    return round(sorted_vals[max(i, 0)], 3)


def summarize(rows: list[dict], logs: list[dict]) -> dict:
    n = len(rows)
    usages = [a["usage"] for lg in logs for a in lg["attempts"] if a.get("usage")]
    costs = [cost_usd(u) for u in usages]
    net_lat = sorted(a["latency_seconds"] for lg in logs for a in lg["attempts"]
                     if not a["cache_hit"] and a.get("latency_seconds") is not None)
    n_cache = sum(1 for lg in logs for a in lg["attempts"] if a["cache_hit"])
    models = sorted({a["model"] for lg in logs for a in lg["attempts"] if a.get("model")})
    reasoning = [a["usage"].get("reasoning_tokens") for lg in logs
                 for a in lg["attempts"] if a.get("usage") and
                 a["usage"].get("reasoning_tokens") is not None]
    cits = [c for lg in logs for c in lg["citations"]]
    reasons = Counter(c["reason"] for c in cits if c["reason"])
    parse_fail_first = sum(1 for lg in logs if len(lg["attempts"]) > 1)
    parse_fail_final = sum(1 for r in rows if r.get("parse_failed"))
    degraded = Counter(a["parse_path"] for lg in logs for a in lg["attempts"])
    total_tok = {"prompt": sum(u.get("prompt_tokens") or 0 for u in usages),
                 "completion": sum(u.get("completion_tokens") or 0 for u in usages),
                 "prompt_cache_hit": sum(u.get("prompt_cache_hit_tokens") or 0 for u in usages)}
    return {
        "n_questions": n,
        "parse": {"fail_first": parse_fail_first,
                  "fail_first_rate": round(parse_fail_first / n, 4),
                  "fail_final": parse_fail_final,
                  "fail_final_rate": round(parse_fail_final / n, 4),
                  "parse_path_counts": dict(degraded)},
        "citations": {
            "intervals_total": len(cits),
            "intervals_valid": sum(1 for c in cits if not c["reason"]),
            "entry_valid_rate": round(sum(1 for c in cits if not c["reason"])
                                      / len(cits), 4) if cits else None,
            "q_all_valid": sum(1 for r in rows if r.get("all_citations_valid")
                               and r["evidence_rfids"]),
            "q_all_valid_rate": round(sum(1 for r in rows if r.get("all_citations_valid")
                                          and r["evidence_rfids"]) / n, 4),
            "q_zero_citation": sum(1 for r in rows
                                   if not r["evidence_rfids"] and not r.get("parse_failed")),
            "invalid_reasons": dict(reasons)},
        "tokens": {**total_tok,
                   "cost_usd_offpeak": round(sum(c["usd_offpeak"] for c in costs), 4),
                   "cost_usd_peak": round(sum(c["usd_peak"] for c in costs), 4)},
        "latency": {"n_calls": sum(len(lg["attempts"]) for lg in logs),
                    "n_cache_hits": n_cache,
                    "n_network": len(net_lat),
                    "p50_s": _pct(net_lat, 0.50), "p95_s": _pct(net_lat, 0.95),
                    "mean_s": round(sum(net_lat) / len(net_lat), 3) if net_lat else None},
        "models_observed": models,
        "max_reasoning_tokens": max(reasoning) if reasoning else None,
    }


# ---------- 主流程(CLI 与 run_all 共用) ----------

def run_generation(cfg: dict, chunks: list[dict], locator: RfidLocator, retriever,
                   qaset: dict, out_path: Path, log_path: Path,
                   llm_client=None, verbose: bool = True, qaset_label: str = "") -> dict:
    """全量生成并落盘 answers/generation_log, 返回 summary(指标同键名进 run 日志)。

    run_all 直接传入其已建好的检索器实例(零重复加载)与 run 目录输出路径;
    CLI(main)传 config 构建的检索器与 --out/--log。llm_client 注入点供离线桩测试。
    """
    gen_cfg = cfg["generation"]
    top_k = gen_cfg.get("top_k", 5)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    corpus_max = max(c["rfid_end"] for c in chunks)
    questions = qaset["questions"]

    prefix = build_fewshot(chunks, qaset, locator) \
        if gen_cfg.get("prompt", {}).get("few_shot", True) \
        else [{"role": "system", "content": SYSTEM_PROMPT}]
    client = llm_client or LLMClient(gen_cfg["llm"])

    rows, logs = [], []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as out_f:   # 逐题落盘: 崩溃不丢已完成题
        for i, q in enumerate(questions, 1):
            hits = retriever.search(q["question"], top_k)
            messages = prefix + [{"role": "user",
                                  "content": build_user_message(q["question"], hits, chunk_by_id)}]
            row, log = generate_one(client, q, messages, hits, locator, corpus_max)
            rows.append(row)
            logs.append(log)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            if verbose:
                flag = "" if row.get("all_citations_valid", True) else "  [引用无效]"
                if row.get("parse_failed"):
                    flag = "  [解析失败]"
                print(f"  [{i:>3}/{len(questions)}] {q['qid']} verdict={row['verdict']:<7}"
                      f" 引用{len(row['evidence_rfids'])}条{flag}")

    summary = summarize(rows, logs)
    log_path.write_text(json.dumps(
        {"config": {"retriever": gen_cfg["retriever"], "top_k": top_k,
                    "llm": gen_cfg["llm"], "few_shot": len(prefix) > 1,
                    "qaset": qaset_label},
         "summary": summary, "questions": logs},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="生成层: 证据块 → DeepSeek → answers.jsonl")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--qaset", default=None, help="缺省用 config paths.qaset")
    ap.add_argument("--out", default=None, help="缺省用 config paths.answers")
    ap.add_argument("--log", default=None, help="缺省 <out 同名>_generation_log.json")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 n 题(0=全量)")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    qaset_path = Path(args.qaset) if args.qaset else paths["qaset"]
    out_path = Path(args.out) if args.out else paths["answers"]
    log_path = Path(args.log) if args.log else \
        out_path.with_name(out_path.stem + "_generation_log.json")

    chunks = load_chunks(paths["scenes"])
    locator = RfidLocator(chunks)
    qaset = load_qaset(qaset_path)
    if args.limit:
        qaset = {**qaset, "questions": qaset["questions"][:args.limit]}

    gen_cfg = cfg["generation"]
    retriever = build_generation_retriever(cfg, chunks, paths)
    print(f"== 生成层 ==\n  检索臂={gen_cfg['retriever']}  top_k={gen_cfg.get('top_k', 5)}  "
          f"题集={qaset_path.name}({len(qaset['questions'])} 题)  模型={gen_cfg['llm']['model']}")

    try:
        client = LLMClient(gen_cfg["llm"])
    except RuntimeError as e:
        sys.exit(f"\n[跳过生成] {e}")

    summary = run_generation(cfg, chunks, locator, retriever, qaset,
                             out_path, log_path, llm_client=client,
                             qaset_label=str(qaset_path))

    print(f"\n== 汇总 ==\n  解析失败率: 首次 {summary['parse']['fail_first_rate']} / "
          f"重试后 {summary['parse']['fail_final_rate']}  "
          f"(解析路径 {summary['parse']['parse_path_counts']})")
    c = summary["citations"]
    print(f"  引用有效率: 条级 {c['entry_valid_rate']}(有效 {c['intervals_valid']}/"
          f"{c['intervals_total']})  题级全部有效 {c['q_all_valid_rate']}"
          f"(零引用 {c['q_zero_citation']} 题, 无效原因 {c['invalid_reasons']})")
    t = summary["tokens"]
    print(f"  token: 输入 {t['prompt']:,}(缓存命中 {t['prompt_cache_hit']:,}) "
          f"输出 {t['completion']:,}  费用≈${t['cost_usd_offpeak']}(谷)/"
          f"${t['cost_usd_peak']}(峰)")
    lat = summary["latency"]
    fmt = lambda v: f"{v}s" if v is not None else "-"
    print(f"  延迟(网络调用 {lat['n_network']} 次, 缓存命中 {lat['n_cache_hits']} 次): "
          f"P50={fmt(lat['p50_s'])} P95={fmt(lat['p95_s'])} 均值={fmt(lat['mean_s'])}")
    print(f"  响应模型字段: {summary['models_observed']}  "
          f"max_reasoning_tokens={summary['max_reasoning_tokens']}(思考关=字段缺席)")
    print(f"\n产物: {out_path}\n      {log_path}")


if __name__ == "__main__":
    main()
