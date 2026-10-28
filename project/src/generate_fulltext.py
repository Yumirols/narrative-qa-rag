# -*- coding: utf-8 -*-
"""T17 长上下文对照(全文塞入 vs RAG) + 位置消融 —— 阶段计划 §2 T17(17.1~17.4)。

三个子命令, 产物全部落 runs/fulltext_T17/(runs 目录只追加, 不动既有 run):
    --smoke     17.1 冒烟: 2 题(yesno 1 + unanswerable 1)× 2 半 = 4 调用,
                先验三件事: ①端点窗口装得下 ②前缀缓存跨题命中(第 2 题同半
                读 usage.prompt_cache_hit_tokens) ③单题实际延迟;
    (默认)      17.2 全量: 36 题(yesno 18 + unanswerable 18)× 2 半 = 72 调用
                → fulltext_answers.jsonl + fulltext_metrics.json(含三方法对照表);
    --position  17.3 位置消融: 关臂(222529)检索 gold 命中 top-5 的 yesno 题
                qid 序前 10 × gold 块置首/第 3/末位 = 30 调用
                → position_ablation.json。

与 generate.py 生成协议的同构关系(权威实现见 src/generate.py):
  - 全文臂: JSON 输出协议(verdict/answer/evidence_rfids)与拒答规则**逐字保留**,
    系统提示只把"证据"改写为"剧本全文的前半/后半"; 解析重试(RETRY_HINT)与
    引用两规则校验直接复用 generate.generate_one 原函数(零复制);
  - 全文臂**不带 few-shot**, 两条理由: (a) few-shot 的证据用法演示(引用 3 块
    中 1 块)与"证据=整个半本剧本"的语义不符; (b) few-shot 插在正文前会推迟
    共享前缀的起点、白付 ~2.5K token——与"36 题共用前缀吃缓存"的布局冲突;
  - 位置消融则相反: **逐字复用** generate.py 的 SYSTEM_PROMPT + build_fewshot +
    build_user_message(含 few-shot 与"按检索融合排名排序"头), 仅重排证据顺序
    ——测的就是 RAG 自己的 prompt 在 222529 基线下的位置敏感性, 协议必须与其
    完全同构。generate.py 模块级无重依赖(retrievers 工厂延迟导入), 直接 import。
  - 超时核实: src/llm.py 的 LLMClient **不经 llm_cfg 暴露 timeout**(OpenAI() 用
    SDK 默认 600s/次), 全文臂单次调用预期 10~60s, 默认值足够, 不改 llm.py。

前缀缓存布局(费用关键, DeepSeek 前缀缓存要求跨调用逐字节一致):
    消息 = [系统提示(半别稳定)] + [user: 该半全文(带 rfid 标注行) + 问题]
    ——全文在前、问题在后, 同半 36 题共享 ~65K token 前缀, 第 2 题起命中。

红线: 本文件是 T17 唯一新增源码; 不改任何已有文件; 全部 LLM 调用走 LLMClient
(sha256 缓存 + 费用核算 cost_usd); Key 不打印不落盘; 评测集冻结不改题;
API 连续失败即停并如实报告; 冒烟无前缀命中 → 按 17.4 题量减半至 18 题。

用法:
    python src/generate_fulltext.py --smoke
    python src/generate_fulltext.py
    python src/generate_fulltext.py --position
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
from src.evaluate import RfidLocator, eval_citations, eval_parse_failures, \
    gold_intervals, load_qaset
from src.generate import build_fewshot, build_user_message, generate_one
from src.index_bm25 import load_chunks
from src.llm import LLMClient, PRICING_PER_MTOK, cost_usd

# T17 产物目录(runs 只追加惯例)
T17_DIR = PROJECT_ROOT / "runs" / "fulltext_T17"
SMOKE_LOG = T17_DIR / "fulltext_smoke_log.json"
ANSWERS_PATH = T17_DIR / "fulltext_answers.jsonl"
METRICS_PATH = T17_DIR / "fulltext_metrics.json"
POSITION_PATH = T17_DIR / "position_ablation.json"

# 17.2 题集 = verdict 载题(yesno + unanswerable); 生成类 54 题不进全文臂
# (judge 输入装不下全文, 质量面由 verdict/拒答承载)
FULLTEXT_TYPES = ("yesno", "unanswerable")

# 三方法对照的 RAG 侧对照组(已存在, 只读不跑)
RAG_RUNS = {"rag_off_hybrid_rrf": "runs/20261026-222529",
            "rag_on_hybrid3_rrf": "runs/20261026-223110"}

# 冒烟前缀命中判定门线(17.4): 同半第 2 题命中 token ≥1024(=16 个 64-token
# 缓存块)才算"显著>0"; 两个半都不达标 → 题量减半至 18 题
PREFIX_HIT_MIN_TOKENS = 1024


# ---------- 全文臂系统提示(17.1) ----------

def fulltext_system_prompt(half_label: str) -> str:
    """从 generate.py 的 SYSTEM_PROMPT 改写: 首句与 evidence_rfids 引用对象
    指向"剧本全文的前半/后半"与其方括号标注行; JSON 输出协议示例、字段规则、
    拒答规则**逐字保留**(与 RAG 臂同构, 差异点仅在证据形态描述)。"""
    return f"""你是叙事剧本问答助手。回答只能依据用户消息中给出的【证据】——即剧本全文的{half_label}(按剧情顺序逐块给出,每块前一行方括号标注其 chunk_id 与 rfid 区间),严禁使用证据之外的任何知识,严禁推测编造。

输出协议(强制):只输出一个合法的 JSON 对象,不输出任何其他文字、解释或代码块标记。JSON 格式示例:
{{"answer": "……", "verdict": "yes", "evidence_rfids": [[44, 89]]}}

字段规则:
- answer:用中文依据证据作答;证据不足时说明缺少什么信息。
- verdict:问题若是能用"是/否"回答的判断题,填 "yes" 或 "no";证据不足以判断、问题与证据无关、或问题不是是非题时,一律填 "unknown"。
- evidence_rfids:数组,元素为 [起, 止] 形式的 rfid 区间(单点可写 [起, 起]);只允许引用证据中各方括号标注行(如 [seen08000_c001 | rfid 1-43])内标注的 rfid 区间,列出全部支持回答的证据。

拒答规则:证据不足以回答、或证据与问题无关时,verdict 必须填 "unknown",answer 说明缺少什么,evidence_rfids 填 []。宁可拒答,不可猜测。"""


def build_half_text(half_chunks: list[dict]) -> str:
    """该半全文 = chunk 按 rfid 序(语料文件序已验证严格递增)拼接, 每块前带
    标注行 [chunk_id | rfid 起-止], 使模型可引用 rfid、引用校验可确定性回查。"""
    return "\n\n".join(
        f"[{c['chunk_id']} | rfid {c['rfid_start']}-{c['rfid_end']}]\n{c['text']}"
        for c in half_chunks)


def build_fulltext_user(half_text: str, half_label: str, question: str) -> str:
    """user 消息: **全文在前、问题在后**——同半 36 题共用同一字节前缀, DeepSeek
    前缀缓存跨题命中; 尾句与 generate.build_user_message 逐字相同(同构)。"""
    return (f"【证据】(剧本全文·{half_label},按剧情顺序)\n{half_text}\n\n"
            f"【问题】{question}\n\n"
            f"请依据以上证据,只输出符合系统提示 JSON 协议的一个 JSON 对象。")


def split_halves(chunks: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """按 chunk 边界切两半: 累计 n_chars 最接近 total/2 的边界; 返回(前半, 后半,
    切分记录)。切分点与两侧字数记录进产物(复现口径)。"""
    total = sum(c["n_chars"] for c in chunks)
    best_i, best_gap = 0, abs(chunks[0]["n_chars"] - total / 2)
    acc = 0
    for i, c in enumerate(chunks):
        acc += c["n_chars"]
        if abs(acc - total / 2) < best_gap:
            best_i, best_gap = i, abs(acc - total / 2)
    left, right = chunks[:best_i + 1], chunks[best_i + 1:]
    info = {
        "n_chunks_total": len(chunks),
        "split_index": best_i,                       # 前半含 chunks[0..best_i]
        "split_chunk_id_first_of_right": right[0]["chunk_id"] if right else None,
        "left": {"n_chunks": len(left),
                 "n_chars_sum": sum(c["n_chars"] for c in left),
                 "chunk_first": left[0]["chunk_id"], "chunk_last": left[-1]["chunk_id"]},
        "right": {"n_chunks": len(right),
                  "n_chars_sum": sum(c["n_chars"] for c in right),
                  "chunk_first": right[0]["chunk_id"], "chunk_last": right[-1]["chunk_id"]},
    }
    return left, right, info


# ---------- 逐题调用的合并记录(usage/延迟/费用 → 答案行内) ----------

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens",
              "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")


def merge_attempts(qlog: dict) -> dict:
    """generate_one 返回的 per-question log → {usage 合并, 延迟 ms, 费用, 哈希}。

    attempts 里的 latency/usage 均来自**首次网络调用**的原始值(sha256 缓存原样
    落盘), 本地缓存命中行读到的也是当年真网络延迟——延迟口径不受重跑影响。
    """
    us = [a["usage"] for a in qlog["attempts"] if a.get("usage")]
    usage = {k: sum(x.get(k) or 0 for x in us) for k in USAGE_KEYS
             if any(x.get(k) is not None for x in us)}
    costs = [cost_usd(x) for x in us]
    lat_ms = round(sum(a["latency_seconds"] for a in qlog["attempts"]
                       if a.get("latency_seconds") is not None) * 1000)
    return {"usage": usage,
            "latency_ms": lat_ms,
            "n_attempts": len(qlog["attempts"]),
            "local_cache_hit": any(a["cache_hit"] for a in qlog["attempts"]),
            "cost_usd_offpeak": round(sum(c["usd_offpeak"] for c in costs), 8),
            "cost_usd_peak": round(sum(c["usd_peak"] for c in costs), 8),
            "request_hash": qlog["attempts"][0]["request_hash"]}


# ---------- 指标(口径 = src/evaluate.py) ----------

def _label_metrics(rows: list[dict], qaset: dict) -> dict:
    """label_acc / 拒答正确率 / 过度拒答——公式与 evaluate.eval_label_accuracy
    逐句同式; 差异: 该函数按 qid 唯一读文件, 全文臂每题两行(半 A/半 B), 这里
    **逐行实例化**(每行一个测量实例: 单半 = 36 实例, 合并 = 72 实例);
    引用/解析指标直接复用 evaluate.eval_citations / eval_parse_failures。"""
    gold = {q["qid"]: q for q in qaset["questions"] if q.get("gold_verdict")}
    pairs = [(gold[r["qid"]], r["verdict"]) for r in rows if r["qid"] in gold]
    n = len(pairs)
    correct = sum(1 for q, v in pairs if q["gold_verdict"] == v)
    unans = [(q, v) for q, v in pairs if q["type"] == "unanswerable"]
    ans = [(q, v) for q, v in pairs if q["type"] != "unanswerable"]
    return {
        "n": n,
        "label_acc": round(correct / n, 4) if n else None,
        "refusal_correct_unanswerable": round(
            sum(1 for _, v in unans if v == "unknown") / len(unans), 4) if unans else None,
        "no_overrefusal_answerable": round(
            sum(1 for _, v in ans if v != "unknown") / len(ans), 4) if ans else None,
        "citations": eval_citations(rows),
        "parse_failure": eval_parse_failures(rows),
    }


def _pct(sorted_vals: list[float], p: float):
    """百分位(与 generate.summarize 同式, 直接内联避免引私有函数)。"""
    if not sorted_vals:
        return None
    i = min(int(len(sorted_vals) * p + 0.999999) - 1, len(sorted_vals) - 1)
    return round(sorted_vals[max(i, 0)], 3)


def arm_summary(rows: list[dict], logs: list[dict], qaset: dict,
                locator: RfidLocator, half_chunk_ids: dict[str, set[str]],
                subset_label: str) -> dict:
    """一个统计口径(某半/合并)的全部指标: 质量面 + 编造 rfid + token/费用/延迟。

    编造 rfid 口径: 引用条目里 reason=orphan_unlocatable / out_of_corpus
    (rfid 落不到任何 chunk——含历史遗留孤儿 3281/14785/14898, 模型输出它们
    同样是编造) + n_malformed_rfids(格式不归一)。not_in_prompt 单列(多为引用
    了另一半的块, 语义上不全是"编造")。
    附加诊断: yesno 按"gold 是否落入该半"分层 label_acc——把"证据不在本半"
    (结构性错) 与 "证据在但没找到/lost in the middle" 分开, 供报告如实归因。
    """
    out = _label_metrics(rows, qaset)
    reasons = Counter(c["reason"] for lg in logs for c in lg["citations"] if c["reason"])
    out["fabricated_rfids"] = {
        "orphan_or_out_of_corpus": reasons.get("orphan_unlocatable", 0)
                                   + reasons.get("out_of_corpus", 0),
        "malformed": sum(r.get("n_malformed_rfids", 0) for r in rows),
        "not_in_prompt": reasons.get("not_in_prompt", 0),
    }
    # 无效引用逐条留痕(溯源: 哪题编造了哪个 rfid; 空 = 零无效)
    out["invalid_citation_examples"] = [
        {"qid": lg["qid"], "half": lg.get("half"), "rfid": c["rfid"], "reason": c["reason"]}
        for lg in logs for c in lg["citations"] if c["reason"]][:50]
    usages = [r["usage"] for r in rows if r.get("usage")]
    lats = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    out["tokens"] = {
        "n_calls": len(rows),
        "prompt": sum(u.get("prompt_tokens") or 0 for u in usages),
        "prompt_cache_hit": sum(u.get("prompt_cache_hit_tokens") or 0 for u in usages),
        "prompt_cache_miss": sum(u.get("prompt_cache_miss_tokens") or 0 for u in usages),
        "completion": sum(u.get("completion_tokens") or 0 for u in usages),
        "prompt_per_q_mean": round(sum(u.get("prompt_tokens") or 0 for u in usages)
                                   / len(usages), 1) if usages else None,
    }
    out["cost"] = {"usd_offpeak": round(sum(r["cost_usd_offpeak"] for r in rows), 6),
                   "usd_peak": round(sum(r["cost_usd_peak"] for r in rows), 6)}
    # 冷成本估计(假设零前缀命中): 实测费用被 DeepSeek 前缀缓存大幅折价(命中
    # ~99%), "为什么要 RAG"的成本对照必须给无缓存口径才是量级真话
    n_calls = len(rows)
    cold = (out["tokens"]["prompt"] * PRICING_PER_MTOK["input_miss"]
            + out["tokens"]["completion"] * PRICING_PER_MTOK["output"]) / 1e6
    out["cost"]["usd_offpeak_cold_estimate"] = round(cold, 6)
    out["cost"]["per_call_usd_offpeak_mean"] = round(
        out["cost"]["usd_offpeak"] / n_calls, 8)
    out["cost"]["per_call_usd_offpeak_cold_mean"] = round(cold / n_calls, 8)
    out["latency"] = {"p50_ms": _pct(lats, 0.50), "p95_ms": _pct(lats, 0.95),
                      "mean_ms": round(sum(lats) / len(lats), 1) if lats else None}

    # yesno gold-in-half 分层诊断(仅单半口径有意义; 合并口径恒在半, 不分层)
    gold_map = {q["qid"]: q for q in qaset["questions"]}
    if subset_label in ("A", "B"):
        ids = half_chunk_ids[subset_label]
        in_half, out_half = [], []
        for r in rows:
            q = gold_map.get(r["qid"])
            if not q or q["type"] != "yesno":
                continue
            try:
                gc = locator.gold_chunks(gold_intervals(q))
            except KeyError:
                continue
            (in_half if gc & ids else out_half).append(r)
        acc = lambda rs: (round(sum(1 for r in rs
                                    if r["verdict"] == gold_map[r["qid"]]["gold_verdict"])
                                / len(rs), 4) if rs else None)
        out["yesno_gold_in_half_breakdown"] = {
            "gold_in_half": {"n": len(in_half), "label_acc": acc(in_half)},
            "gold_not_in_half": {"n": len(out_half), "label_acc": acc(out_half),
                                 "unknown_rate": round(
                                     sum(1 for r in out_half if r["verdict"] == "unknown")
                                     / len(out_half), 4) if out_half else None},
        }
    return out


# ---------- 17.1 冒烟 ----------

def run_smoke(client: LLMClient, halves: dict, qaset: dict) -> dict:
    """2 题(yesno 取 qid 最小 + unanswerable 取 qid 最小)× 2 半 = 4 调用。

    执行顺序刻意同半相邻(Q09-A→Q26-A→Q09-B→Q26-B): 第 2 题同半的前缀缓存
    命中才是 17.4 门线的判定依据。窗口超限(HTTP 400 context/length 类错误)
    不崩脚本: 记录后如实报告, 降级路径(四分块/停下)由人按报告决定。
    """
    yn = min(q["qid"] for q in qaset["questions"] if q["type"] == "yesno")
    ua = min(q["qid"] for q in qaset["questions"] if q["type"] == "unanswerable")
    qmap = {q["qid"]: q for q in qaset["questions"]}
    order = [(yn, "A"), (ua, "A"), (yn, "B"), (ua, "B")]

    records = []
    for qid, half in order:
        h = halves[half]
        messages = [{"role": "system", "content": fulltext_system_prompt(h["label"])},
                    {"role": "user", "content": build_fulltext_user(
                        h["text"], h["label"], qmap[qid]["question"])}]
        rec = {"qid": qid, "type": qmap[qid]["type"], "half": half}
        try:
            r = client.chat_json(messages)
            u = r.get("usage") or {}
            rec.update({
                "ok": True, "local_cache_hit": r["cache_hit"],
                "prompt_tokens": u.get("prompt_tokens"),
                "prompt_cache_hit_tokens": u.get("prompt_cache_hit_tokens"),
                "prompt_cache_miss_tokens": u.get("prompt_cache_miss_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "latency_seconds": r.get("latency_seconds"),
                "parse_path": r.get("parse_path"),
                "verdict": (r.get("parsed") or {}).get("verdict")
                           if isinstance(r.get("parsed"), dict) else None,
                "cost_usd_offpeak": round(cost_usd(u)["usd_offpeak"], 8),
                "request_hash": r["request_hash"],
            })
        except Exception as e:   # 窗口超限/网络失败: 记录后继续看下一半, 不崩
            rec.update({"ok": False, "error_type": type(e).__name__,
                        "error": str(e)[:500]})
        records.append(rec)
        hit = rec.get("prompt_cache_hit_tokens")
        print(f"  {qid}-{half}: {'ok' if rec['ok'] else 'FAIL'}"
              f"  prompt={rec.get('prompt_tokens')}  server_prefix_hit={hit}"
              f"  latency={rec.get('latency_seconds')}s"
              + ("" if rec["ok"] else f"  error={rec.get('error', '')[:200]}"))

    ok_all = all(r["ok"] for r in records)
    # 前缀命中门线: 每半的第 2 次调用(索引 1 与 3)
    second_calls = {"A": records[1], "B": records[3]}
    prefix_ok = {h: (r.get("prompt_cache_hit_tokens") or 0) >= PREFIX_HIT_MIN_TOKENS
                 for h, r in second_calls.items()}
    smoke = {
        "date": "2026-10-27",
        "questions": {"yesno": yn, "unanswerable": ua},
        "calls": records,
        "checks": {
            "window_ok": ok_all,
            "window_note": "4 调用全部成功 = 两半均装入端点上下文窗口"
                           if ok_all else "存在失败调用, 见 calls[].error",
            "prefix_cache_hit": prefix_ok,
            "prefix_cache_hit_tokens_2nd_call": {
                h: r.get("prompt_cache_hit_tokens") for h, r in second_calls.items()},
            "prefix_threshold_tokens": PREFIX_HIT_MIN_TOKENS,
            "latency_seconds_all": [r.get("latency_seconds") for r in records],
        },
    }
    T17_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_LOG.write_text(json.dumps(smoke, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n== 冒烟三查 ==\n  ①窗口: {'通过' if ok_all else '失败(需降级决策)'}")
    print(f"  ②前缀缓存命中(第2题同半): A={second_calls['A'].get('prompt_cache_hit_tokens')}"
          f" tok / B={second_calls['B'].get('prompt_cache_hit_tokens')} tok"
          f" (门线 {PREFIX_HIT_MIN_TOKENS}) → {'命中' if all(prefix_ok.values()) else '未命中→按17.4题量减半'}")
    print(f"  ③单题延迟(s): {[r.get('latency_seconds') for r in records]}")
    print(f"  冒烟费用(谷): ${round(sum(r.get('cost_usd_offpeak', 0) for r in records), 6)}")
    print(f"  产物: {SMOKE_LOG}")
    return smoke


# ---------- 17.2 全量 36 题 × 2 半 ----------

def run_full(client: LLMClient, halves: dict, qaset: dict, locator: RfidLocator,
             chunks: list[dict], reduce_questions: bool) -> None:
    """36(或冒烟无命中时减半 18)题 × 2 半; 同半内连续执行(前缀缓存吃到饱)。

    答案行 schema 与 eval/answers.jsonl 同构(qid/verdict/answer/evidence_rfids/
    all_citations_valid), 另加 half/type/usage/latency_ms/费用等实验字段
    (任务书 17.2 明示); 解析/重试/引用校验全部走 generate.generate_one 原函数。
    """
    questions = [q for q in qaset["questions"] if q["type"] in FULLTEXT_TYPES]
    note = None
    if reduce_questions:
        # 17.4 门线触发: qid 序各取前 9(yesno 9 + unanswerable 9)
        yn = sorted(q["qid"] for q in questions if q["type"] == "yesno")[:9]
        ua = sorted(q["qid"] for q in questions if q["type"] == "unanswerable")[:9]
        keep = set(yn) | set(ua)
        questions = [q for q in questions if q["qid"] in keep]
        note = (f"冒烟未观测到前缀缓存命中, 按 17.4 题量减半: yesno 9(qid 序前 9)"
                f" + unanswerable 9(qid 序前 9) = 18 题×2 半 = 36 调用")
        print(f"  [17.4 减半] {note}")
    questions.sort(key=lambda q: q["qid"])
    corpus_max = max(c["rfid_end"] for c in chunks)

    rows, logs = [], []
    ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANSWERS_PATH, "w", encoding="utf-8", newline="\n") as f:
        for half in ("A", "B"):
            h = halves[half]
            pseudo_hits = [{"chunk_id": c["chunk_id"]} for c in h["chunks"]]
            print(f"  -- 半 {half}({h['label']}, {len(h['chunks'])} 块) --")
            for i, q in enumerate(questions, 1):
                messages = [
                    {"role": "system", "content": fulltext_system_prompt(h["label"])},
                    {"role": "user", "content": build_fulltext_user(
                        h["text"], h["label"], q["question"])}]
                row, qlog = generate_one(client, q, messages, pseudo_hits,
                                         locator, corpus_max)
                m = merge_attempts(qlog)
                full_row = {**row, "half": half, "type": q["type"], **m}
                rows.append(full_row)
                logs.append({**qlog, "half": half})
                f.write(json.dumps(full_row, ensure_ascii=False) + "\n")
                f.flush()
                flag = "" if row.get("all_citations_valid", True) else "  [引用无效]"
                if row.get("parse_failed"):
                    flag = "  [解析失败]"
                print(f"  [{i:>2}/{len(questions)}] {q['qid']}-{half} "
                      f"verdict={row['verdict']:<7} 引用{len(row['evidence_rfids'])}条"
                      f" prompt={m['usage'].get('prompt_tokens')} "
                      f"server_hit={m['usage'].get('prompt_cache_hit_tokens')}"
                      f" {m['latency_ms']}ms{flag}")

    half_ids = {hk: {c["chunk_id"] for c in halves[hk]["chunks"]}
                for hk in ("A", "B")}
    metrics = {
        "task": "T17.2 全文臂(剧本全文两半塞入) vs RAG",
        "date": "2026-10-27",
        "protocol": {
            "system_prompt": "由 generate.py SYSTEM_PROMPT 改写(证据=剧本全文的前半/后半);"
                             "JSON 协议与拒答规则逐字保留",
            "few_shot": False,
            "few_shot_rationale": "few-shot 的 3 块引用演示与'证据=整个半本剧本'语义不符,"
                                  "且会推迟共享前缀起点白付 token(见模块 docstring)",
            "message_layout": "system(半别) + user(全文+问题)——全文在前问题在后,"
                              "同半共享前缀吃 DeepSeek 前缀缓存",
            "citation_validation": "generate.generate_one 原函数(RfidLocator 两规则,"
                                   "孤儿 rfid 3281/14785/14898 命中即 invalid)",
            "llm": "deepseek-flash, temperature=0, thinking disabled, max_tokens=1024",
        },
        "corpus_split": halves["split_info"],
        "n_questions": len(questions),
        "n_calls": len(rows),
        "question_set_note": note,
        "arm": {
            "half_A": arm_summary([r for r in rows if r["half"] == "A"],
                                  [l for l in logs if l["half"] == "A"],
                                  qaset, locator, half_ids, "A"),
            "half_B": arm_summary([r for r in rows if r["half"] == "B"],
                                  [l for l in logs if l["half"] == "B"],
                                  qaset, locator, half_ids, "B"),
            "pooled": arm_summary(rows, logs, qaset, locator, half_ids, "pooled"),
        },
    }
    metrics["three_method_table"] = build_three_method_table(metrics, questions)
    metrics["budget"] = fulltext_budget(metrics)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n== 产物: {ANSWERS_PATH}\n          {METRICS_PATH}")
    a = metrics["arm"]
    for k in ("half_A", "half_B", "pooled"):
        s = a[k]
        print(f"  {k}: label_acc={s['label_acc']}({s['n']}) "
              f"拒答正确={s['refusal_correct_unanswerable']} "
              f"引用条级有效率={s['citations']['entry_valid_rate']} "
              f"编造rfid={s['fabricated_rfids']} 费用=${s['cost']['usd_offpeak']}")


# ---------- 三方法对照表(全文 vs RAG 关 vs RAG 开) ----------

def build_three_method_table(full_metrics: dict, questions: list[dict]) -> dict:
    """质量面: RAG 侧 label_acc/拒答直接读各自 metrics.json 的 label_accuracy
    (口径 = evaluate.eval_label_accuracy, 与全文臂内联公式同式);
    成本/延迟: RAG 侧从 generation_log.json 汇总——注意 attempts 的
    latency/usage 均为**首次网络调用原始值**(sha256 缓存原样保存), 不受该 run
    曾被缓存重跑影响; 每题均值对齐到与全文臂相同的 36 qid 子集(另附全 90 题口径)。
    """
    qids = {q["qid"] for q in questions}
    table = {}
    for name, rel in RAG_RUNS.items():
        run_dir = PROJECT_ROOT / rel
        label = json.loads((run_dir / "metrics.json").read_text(
            encoding="utf-8"))["label_accuracy"]
        lg = json.loads((run_dir / "generation_log.json").read_text(encoding="utf-8"))
        per_q = {}
        all_lats = []
        for q in lg["questions"]:
            us = [a["usage"] for a in q["attempts"] if a.get("usage")]
            lats = [a["latency_seconds"] for a in q["attempts"]
                    if a.get("latency_seconds") is not None]
            all_lats += lats
            per_q[q["qid"]] = {
                "prompt": sum(u.get("prompt_tokens") or 0 for u in us),
                "comp": sum(u.get("completion_tokens") or 0 for u in us),
                "cost": sum(cost_usd(u)["usd_offpeak"] for u in us),
                "lat_s": sum(lats)}
        sub = [per_q[q] for q in qids if q in per_q]
        allq = list(per_q.values())
        mean = lambda vals: round(sum(vals) / len(vals), 2) if vals else None
        cold_mean = round(sum((x["prompt"] * PRICING_PER_MTOK["input_miss"]
                               + x["comp"] * PRICING_PER_MTOK["output"]) / 1e6
                              for x in sub) / len(sub), 8)
        table[name] = {
            "run": rel,
            "label_acc": label["label_acc"],
            "refusal_correct_unanswerable": label["refusal_correct_unanswerable"],
            "no_overrefusal_answerable": label["no_overrefusal_answerable"],
            "citation_entry_valid_rate": label["citations"]["entry_valid_rate"],
            "per_q_input_tokens_mean_same36": mean([x["prompt"] for x in sub]),
            # 费用均值单独算: mean() 的 2 位小数舍入会把 ~5e-4 的单题费用抹成 0
            "per_q_cost_usd_offpeak_mean_same36": round(
                sum(x["cost"] for x in sub) / len(sub), 8),
            "per_q_cost_usd_offpeak_cold_mean_same36": cold_mean,
            "per_q_input_tokens_mean_all90": mean([x["prompt"] for x in allq]),
            "latency_p50_s": _pct(sorted(all_lats), 0.50),
            "latency_p95_s": _pct(sorted(all_lats), 0.95),
            "latency_mean_s": mean(all_lats),
        }
    # 全文臂三口径(A/B/合并), 键名与 RAG 侧对齐便于并表
    for k, lab in (("half_A", "fulltext_half_A"), ("half_B", "fulltext_half_B"),
                   ("pooled", "fulltext_pooled")):
        s = full_metrics["arm"][k]
        table[lab] = {
            "run": "runs/fulltext_T17",
            "label_acc": s["label_acc"],
            "refusal_correct_unanswerable": s["refusal_correct_unanswerable"],
            "no_overrefusal_answerable": s["no_overrefusal_answerable"],
            "citation_entry_valid_rate": s["citations"]["entry_valid_rate"],
            "per_q_input_tokens_mean_same36": s["tokens"]["prompt_per_q_mean"],
            "per_q_cost_usd_offpeak_mean_same36": round(
                s["cost"]["usd_offpeak"] / s["tokens"]["n_calls"], 8),
            "per_q_cost_usd_offpeak_cold_mean_same36":
                s["cost"]["per_call_usd_offpeak_cold_mean"],
            "per_q_input_tokens_mean_all90": s["tokens"]["prompt_per_q_mean"],
            "latency_p50_s": (s["latency"]["p50_ms"] / 1000
                              if s["latency"]["p50_ms"] is not None else None),
            "latency_p95_s": (s["latency"]["p95_ms"] / 1000
                              if s["latency"]["p95_ms"] is not None else None),
            "latency_mean_s": (s["latency"]["mean_ms"] / 1000
                               if s["latency"]["mean_ms"] is not None else None),
        }
        if k in ("half_A", "half_B"):
            table[lab]["yesno_gold_in_half_label_acc"] = \
                s["yesno_gold_in_half_breakdown"]["gold_in_half"]["label_acc"]
            table[lab]["yesno_gold_in_half_n"] = \
                s["yesno_gold_in_half_breakdown"]["gold_in_half"]["n"]
    return table


def fulltext_budget(full_metrics: dict) -> dict:
    """T17 预算账(谷/峰): 17.1+17.2 的真实网络支出。冒烟 4 调用的 request_hash
    与全量中对应行完全一致(同消息→同 sha256), 全量费用已含冒烟, 不重复计;
    位置消融(17.3)单独记账在 position_ablation.json, 总账由 summary 汇总。
    门线 $0.40 谷。"""
    full_cost = full_metrics["arm"]["pooled"]["cost"]["usd_offpeak"]
    full_peak = full_metrics["arm"]["pooled"]["cost"]["usd_peak"]
    return {"cap_usd_offpeak": 0.40,
            "full_17_2_usd_offpeak": full_cost,
            "full_17_2_usd_peak": full_peak,
            "spend_17_1_17_2_usd_offpeak": full_cost,
            "note": "冒烟调用哈希与全量重复, 全量费用已含冒烟, 直接取全量值"}


# ---------- 17.3 位置消融 ----------

POSITIONS = {"first": 1, "middle": 3, "last": 5}   # gold 块目标位(1-indexed)


def run_position(client: LLMClient, qaset: dict, locator: RfidLocator,
                 chunks: list[dict]) -> dict:
    """关臂(222529)gold 命中 top-5 的 yesno 题, qid 序前 10 × 3 位置 = 30 调用。

    选择规则(预注册, 写死在产物): yesno 且 gold_chunks ∩ top5 ≠ ∅ 的全部候选
    (实测 11 题)按 qid 升序取前 10; gold 块 = 检索排名最靠前的命中块(全部候选
    均只 1 块 gold 入 top-5, 规则兜底多块情形)。
    重排: gold 块置于目标位, 其余 4 块保持原相对顺序; rank 重编号 1~5(证据N
    显示序 = 列表序, 与基线一致)。
    prompt 组装**逐字复用** generate.py: SYSTEM_PROMPT + build_fewshot(同 222529
    的 few_shot: true, 候选确定性: 语料序前 3 个非 gold 块) + build_user_message
    (含"按检索融合排名排序"头)——唯一自变量 = gold 块位置。
    """
    base = PROJECT_ROOT / RAG_RUNS["rag_off_hybrid_rrf"]
    lg = json.loads((base / "generation_log.json").read_text(encoding="utf-8"))
    retrieved = {q["qid"]: q["retrieved_chunk_ids"] for q in lg["questions"]}
    base_answers = {}
    for line in (base / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            base_answers[r["qid"]] = r

    # 候选与选择
    cands = []
    for q in sorted(qaset["questions"], key=lambda x: x["qid"]):
        if q["type"] != "yesno":
            continue
        try:
            gold = locator.gold_chunks(gold_intervals(q))
        except KeyError:
            continue
        top5 = retrieved[q["qid"]]
        hits = [c for c in top5 if c in gold]     # 已按检索排名序
        if hits:
            cands.append({"qid": q["qid"], "question": q["question"],
                          "gold_verdict": q["gold_verdict"],
                          "gold_block": hits[0],
                          "n_gold_in_top5": len(hits),
                          "baseline_rank": top5.index(hits[0]) + 1,
                          "top5": top5})
    selected = cands[:10]

    # 逐字复用的组装件(与 222529 完全同构)
    prefix = build_fewshot(chunks, qaset, locator)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    corpus_max = max(c["rfid_end"] for c in chunks)

    out_questions = []
    tot_cost = tot_cost_peak = 0.0
    for cand in selected:
        q = next(x for x in qaset["questions"] if x["qid"] == cand["qid"])
        others = [c for c in cand["top5"] if c != cand["gold_block"]]
        entry = {"qid": cand["qid"], "question": cand["question"],
                 "gold_verdict": cand["gold_verdict"],
                 "gold_block": cand["gold_block"],
                 "gold_rank_in_baseline_top5": cand["baseline_rank"],
                 "n_gold_in_top5": cand["n_gold_in_top5"],
                 "baseline_verdict": base_answers[cand["qid"]]["verdict"],
                 "baseline_answer_head": base_answers[cand["qid"]]["answer"][:60],
                 "positions": {}}
        verdicts = []
        for pos_name, pos in POSITIONS.items():
            ordered = others[:pos - 1] + [cand["gold_block"]] + others[pos - 1:]
            hits = [{"rank": i + 1, "chunk_id": cid} for i, cid in enumerate(ordered)]
            messages = prefix + [{"role": "user", "content": build_user_message(
                q["question"], hits, chunk_by_id)}]
            row, qlog = generate_one(client, q, messages, hits, locator, corpus_max)
            m = merge_attempts(qlog)
            tot_cost += m["cost_usd_offpeak"]
            tot_cost_peak += m["cost_usd_peak"]
            verdicts.append(row["verdict"])
            entry["positions"][pos_name] = {
                "gold_position": pos,
                "verdict": row["verdict"],
                "answer_head": row["answer"][:80],
                "evidence_rfids": row["evidence_rfids"],
                "all_citations_valid": row.get("all_citations_valid", False),
                "parse_failed": bool(row.get("parse_failed")),
                "usage": m["usage"], "latency_ms": m["latency_ms"],
                "cost_usd_offpeak": m["cost_usd_offpeak"],
            }
            print(f"  {cand['qid']} {pos_name:<6}(gold@{pos}) "
                  f"verdict={row['verdict']:<7} 基线={entry['baseline_verdict']:<7}"
                  f" 引用{len(row['evidence_rfids'])}条 {m['latency_ms']}ms")
        entry["verdicts"] = {"first": verdicts[0], "middle": verdicts[1],
                             "last": verdicts[2]}
        entry["stable_across_positions"] = len(set(verdicts)) == 1
        entry["matches_baseline_all"] = all(
            v == entry["baseline_verdict"] for v in verdicts)
        entry["label_correct_by_position"] = {
            pos_name: (entry["positions"][pos_name]["verdict"] == cand["gold_verdict"])
            for pos_name in POSITIONS}
        out_questions.append(entry)

    n = len(out_questions)
    stability = {
        "n_questions": n,
        "stable_across_positions": sum(1 for e in out_questions if e["stable_across_positions"]),
        "flip_within_positions": sum(1 for e in out_questions if not e["stable_across_positions"]),
        "all_match_baseline": sum(1 for e in out_questions if e["matches_baseline_all"]),
        "flip_vs_baseline_any": sum(1 for e in out_questions if not e["matches_baseline_all"]),
        "flipped_qids": [e["qid"] for e in out_questions if not e["matches_baseline_all"]],
        "label_acc_by_position": {
            pos: round(sum(1 for e in out_questions
                           if e["positions"][pos]["verdict"] == e["gold_verdict"]) / n, 4)
            for pos in POSITIONS},
    }
    out = {
        "task": "T17.3 位置消融(gold 块 首/中/末), lost in the middle 直接证据",
        "date": "2026-10-27",
        "selection_rule": ("关臂(runs/20261026-222529, hybrid_rrf k=5)检索 gold 命中 "
                          "top-5 的 yesno 题按 qid 升序取前 10; 候选共 "
                          f"{len(cands)} 题(落选 qid: "
                          f"{[c['qid'] for c in cands[10:]] or '无'})"),
        "baseline_run": RAG_RUNS["rag_off_hybrid_rrf"],
        "protocol_reuse": ("generate.py SYSTEM_PROMPT + build_fewshot + build_user_message "
                           "逐字复用(含 few-shot 与证据头), 唯一自变量 = gold 块在"
                           "证据列表中的位置(1/3/5), 其余 4 块保持原相对顺序, rank 重编号"),
        "questions": out_questions,
        "stability": stability,
        "cost": {"usd_offpeak": round(tot_cost, 6), "usd_peak": round(tot_cost_peak, 6)},
    }
    POSITION_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\n== 稳定性: 3 位置 verdict 全一致 {stability['stable_across_positions']}/{n}"
          f"  与基线全一致 {stability['all_match_baseline']}/{n}"
          f"  翻转题 {stability['flipped_qids']}")
    print(f"   分位置 label_acc: {stability['label_acc_by_position']}")
    print(f"   费用(谷) ${out['cost']['usd_offpeak']}  产物: {POSITION_PATH}")
    return out


# ---------- 主入口 ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="T17 长上下文对照(全文两半 vs RAG) + 位置消融")
    ap.add_argument("--smoke", action="store_true", help="17.1 冒烟 4 调用")
    ap.add_argument("--position", action="store_true", help="17.3 位置消融 30 调用")
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])
    locator = RfidLocator(chunks)
    qaset = load_qaset(paths["qaset"])
    left, right, split_info = split_halves(chunks)
    halves = {"A": {"label": "前半", "chunks": left,
                    "text": build_half_text(left)},
              "B": {"label": "后半", "chunks": right,
                    "text": build_half_text(right)},
              "split_info": split_info}
    print(f"== T17 == 语料 {len(chunks)} chunk; 切分点 {split_info['split_index']}"
          f"({split_info['left']['chunk_last']} | {split_info['right']['chunk_first']})"
          f"  左 {split_info['left']['n_chars_sum']} 字 / 右 {split_info['right']['n_chars_sum']} 字")

    try:
        client = LLMClient(cfg["generation"]["llm"])
    except RuntimeError as e:
        sys.exit(f"\n[跳过] {e}")

    if args.smoke:
        run_smoke(client, halves, qaset)
        return
    if args.position:
        run_position(client, qaset, locator, chunks)
        return
    # 17.2 全量(冒烟先行惯例: 未冒烟直接全量时给出提示但不阻断——缓存层兜底)
    smoke = json.loads(SMOKE_LOG.read_text(encoding="utf-8")) if SMOKE_LOG.exists() else None
    reduce_q = bool(smoke and not all(smoke["checks"]["prefix_cache_hit"].values()))
    if smoke is None:
        print("  [提示] 未发现冒烟产物, 建议先 --smoke; 继续执行")
    elif reduce_q:
        print("  [17.4] 冒烟判定无前缀命中 → 题量减半(18 题×2 半)")
    run_full(client, halves, qaset, locator, chunks, reduce_q)


if __name__ == "__main__":
    main()
