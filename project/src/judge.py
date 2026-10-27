# -*- coding: utf-8 -*-
"""LLM-as-judge 评审层(T14)——独立评审员对生成类答案打分 + 引用语义支持性复核 + 人工抽检模板。

管线(阶段五计划 §2 T14 / §3 rubric):
    被评 run 的 answers.jsonl + retrieval_results.json(取生成臂, top-5 证据确定性重建)
    + qaset_v0(gold_answer/accept_criteria) + scenes.jsonl(chunk 原文)
    → 系统提示(独立评审员) + few-shot 2 例(自编防泄漏) + 实际案例(仅末条 user)
    → src/llm.py 客户端(chat_json / sha256 缓存 / temperature=0 / thinking 关)
    → 归一化 → judge_details.jsonl / judge_metrics.json / judge_run_log.json
    →(可选 --sample) judge_agreement.json(人工抽检模板, 人工列留 null 待填)

设计决策(计划 §3/§6 与已定决策①②):
  - 评审对象 = 54 道生成类可评题(fact/temporal/aggregate)——label_acc 只覆盖 36 道
    verdict 题, 其余 60% 可评面由本层补齐(计划 §1.1);
  - 证据与被评 run 生成时严格一致: retrieval_results 该臂 top 按 rank 取前 5, 与
    generation_log.retrieved_chunk_ids 逐题核对(不一致即告警, 不静默);
  - 消息布局 [system, user例1, asst例1, user例2, asst例2, user实际]——系统提示+
    few-shot 前缀逐字节稳定, 实际案例只在末条 user(DeepSeek 前缀缓存降费);
  - 拒答不扣忠实性: verdict=unknown 且证据确实不足 → faithfulness=5(计划 §3 第 5 条),
    防止把"诚实的不知道"误判为缺陷; v2 起完整性按金标交付量计(拒答=零交付→1),
    总分"诚实拒答=3"——与人工口径卡同参照系(v1 曾按证据内口径给 4~5,校准分歧来源);
  - citations 逐条判**语义支持性**(被引用原文是否支持答案断言), 不是判存在性——
    规则层 entry_valid_rate=1.0 只验 rfid 落在 prompt 内, 语义面在此复核(计划 §1.2);
  - judge_metrics.json 零非确定字段(无时间戳; 全部由 details 的评分与 usage 推导),
    缓存命中重跑逐字节一致(验收底线 1); 时间/环境/CLI 参数全进 judge_run_log.json
    ——details/metrics 与 log 分离纪律沿 generate.py;
  - 自评偏差(judge 与被评同模型)以评审员角色隔离 + 20% 人工抽检一致率(≥0.85)校准,
    不引入第二家 API(已定决策①); 主推臂裁定走计划 §6 预注册规则, 不在本脚本内。

附加 --audit 模式(T15.4 缺证边界 mini-audit): 交叉表纯脚本计算(不调 LLM); 对
"gold_verdict=no 且检索 miss 且答对"题逐例做一次审计 LLM 调用(合理否定推断 vs
无证据猜测), 产物 judge_missing_evidence_audit.json 写入关臂 run 目录(两臂共用)。

用法:
    python src/judge.py --run runs/20261026-222529                        # 关臂全量 54 题
    python src/judge.py --run runs/20261026-223110 --limit 5 --sample 0   # 冒烟(前 5 题)
    python src/judge.py --run runs/20261026-222529 --sample 11            # 抽检模板(缓存命中 $0)
    python src/judge.py --run runs/20261026-222529 \
        --compare-run runs/20261026-223110 --audit                        # T15.4 缺证审计
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.evaluate import load_qaset
from src.index_bm25 import load_chunks
from src.llm import LLMClient, cost_usd

JUDGE_TEMPLATE_VERSION = "v2"     # v2(2026-10-27): 完整性改金标交付口径+新增拒答 few-shot——
                                  # v1 人工校准 0.8182<0.85, 归因=完整性参照系分歧(v1 证据内口径
                                  # "证据确实不足给4~5" vs 注册口径卡金标覆盖度), 对齐口径卡(预注册分支)
AUDIT_TEMPLATE_VERSION = "v1"     # --audit 模式的审计提示版本
GENERATIVE_TYPES = ("fact", "temporal", "aggregate")   # 54 道生成类可评题
JUDGE_MAX_TOKENS = 512            # 评审 JSON 输出短, 512 足够(沿 generate 的 1024 收紧)
SAMPLE_SEED = 0                   # 人工抽检抽样种子(计划 §4 配置草案, 固定保证复现)
TOP_K = 5                         # 生成层实际送 prompt 的证据数(generation.top_k)

# ---------- 评审协议(计划 §3 rubric 基线) ----------

JUDGE_SYSTEM_PROMPT = """你是叙事问答系统的独立评审员,对被评系统生成的答案进行质量评分。你只能依据用户消息中给出的材料(问题/参考要点/证据块原文/被评答案)评分,严禁使用材料之外的任何知识,严禁臆测材料外的情节。你是独立评审员而不是答题者:不要自己重新作答,只评被评答案。

输出协议(强制):只输出一个合法的 JSON 对象,不输出任何其他文字、解释或代码块标记。JSON 字段与取值:
{"score": 1~5 的整数,           // 总分
 "faithfulness": 1~5 的整数,    // 忠实性:答案断言是否仅由证据支持,5=全部可溯源
 "completeness": 1~5 的整数,    // 完整性:答案实际交付了参考要点中的多少关键信息(拒答=零交付)
 "conciseness": 1~5 的整数,     // 简洁性:直接作答,不堆砌无关内容、不大段照抄证据
 "hallucination": true 或 false, // 是否存在证据无法支持的断言
 "citations": [{"rfid": [起, 止], "supports": true 或 false}], // 逐条引用判语义支持性
 "issues": ["一句话问题描述", ...]} // 无问题给空数组

总分锚定:
- 5:可直接采用——正确、完整、有据、简洁;
- 4:基本可采用,仅有轻微瑕疵(如个别表述含糊或稍显冗长);
- 3:部分正确,但有明确遗漏或局部不准确;
- 2:正确内容占比小,或含明显无证据支持的断言;
- 1:基本错误、答非所问、或主要靠猜测。

citations 判定口径:对被评答案 evidence_rfids 的每条引用区间,核对【证据】中被引用块原文是否在语义上支持答案的关键断言——判的是"引用内容是否支持答案",不是"引用是否存在"。被引用原文支持答案断言 → supports=true;被引用原文与答案断言无关或矛盾 → supports=false;被评答案未给引用区间时 citations 给空数组。

拒答口径:被评答案为拒答(说明证据不足/无法回答)时,拒答本身不扣忠实性——若证据确实不足以回答,faithfulness=5、hallucination=false。但完整性按交付量计:拒答=零交付,completeness=1。总分:证据确实不足时的诚实拒答给 3(行为正确但未交付答案);证据其实充分却拒答给 2。"""

JUDGE_RETRY_HINT = ("你上一次的输出不是符合评审协议的合法 JSON 对象(或缺少 score/faithfulness/"
                    "completeness/conciseness 字段、分值越界)。请重新评审:只输出一个符合协议"
                    "的 JSON 对象,不要包含任何其他文字或代码块标记。")

# ---------- few-shot 2 例(全部自编; 题目/证据/答案均非评测集内容, 防泄漏) ----------

# 例1: 好答案 → score=5(教"满分长什么样": 三维全 5、引用语义支持、无问题)
_FS1_EVIDENCE = {
    "chunk_id": "FS-demo-001", "rfid_start": 101, "rfid_end": 118,
    "speakers": ["小林", "小周"],
    "text": ("【小林】「书架最上层有一本很旧的日记。」\n"
             "我踩着凳子把日记取了下来,扉页上写着祖父的名字。\n"
             "【小周】「夹页里好像还塞着东西。」\n"
             "翻开一看,是一把用红绳系着的小铜钥匙。"),
}
_FS1_CASE = {
    "question": "小林在书房里找到了哪两样东西?",
    "gold_answer": "一本旧日记(扉页写着祖父的名字)和一把系红绳的小铜钥匙。",
    "accept_criteria": "两样东西都要提到;铜钥匙的「系红绳」细节可有可无。",
    "verdict": "unknown", "answer": "小林找到了一本旧日记,扉页上写着祖父的名字,"
                "以及夹在日记里的一把小铜钥匙。",
    "evidence_rfids": [[101, 118]],
}
_FS1_JUDGE = {"score": 5, "faithfulness": 5, "completeness": 5, "conciseness": 5,
              "hallucination": False,
              "citations": [{"rfid": [101, 118], "supports": True}],
              "issues": []}

# 例2: 含幻觉答案 → score=2(教"低分长什么样": 编造原因、引用存在但不支持答案)
_FS2_EVIDENCE = {
    "chunk_id": "FS-demo-002", "rfid_start": 201, "rfid_end": 214,
    "speakers": ["老陈"],
    "text": ("【老陈】「雷声太响了,我根本睡不着。」\n"
             "他在客厅的沙发上坐到半夜,后来干脆在地板上打了地铺。"),
}
_FS2_CASE = {
    "question": "老陈那天晚上为什么没有回自己的房间?",
    "gold_answer": "因为雷声太响睡不着,他在客厅坐到半夜,之后在客厅打了地铺。",
    "accept_criteria": "需包含「雷声/打雷」这一原因。",
    "verdict": "unknown", "answer": "老陈因为害怕地下室传来的哭声,整夜不敢回房间,"
                "一直守在客厅到天亮。",
    "evidence_rfids": [[201, 214]],
}
_FS2_JUDGE = {"score": 2, "faithfulness": 1, "completeness": 2, "conciseness": 4,
              "hallucination": True,
              "citations": [{"rfid": [201, 214], "supports": False}],
              "issues": ["「害怕地下室哭声」「守到天亮」均为证据中不存在的编造,"
                         "证据所述原因是雷声太响无法入睡"]}

# 例3: 诚实拒答 → score=3(v2 新增: 锚定"完整性按交付量计,拒答=零交付"的金标口径)
_FS3_EVIDENCE = {
    "chunk_id": "FS-demo-003", "rfid_start": 301, "rfid_end": 309,
    "speakers": ["阿彻"],
    "text": ("【阿彻】「天线又歪了,风一大就这样。」\n"
             "他搬来梯子爬上天台,拧了半小时螺丝,顺手给避雷针补了层漆。"),
}
_FS3_CASE = {
    "question": "阿彻的哥哥当年为什么离开小镇?",
    "gold_answer": "因为他收到了海外大学的录取,又与父亲大吵一架,决心出走求学。",
    "accept_criteria": "需包含「与父亲争吵」与「离镇求学」两点。",
    "verdict": "unknown",
    "answer": "证据只提到阿彻在天台修理天线,没有其哥哥离开小镇原因的信息,无法回答。",
    "evidence_rfids": [],
}
_FS3_JUDGE = {"score": 3, "faithfulness": 5, "completeness": 1, "conciseness": 5,
              "hallucination": False, "citations": [],
              "issues": ["证据确实不含答案,拒答行为正确;但参考要点显示问题有明确答案,"
                         "零交付 → completeness=1,总分=3"]}

FEWSHOT = [(_FS1_EVIDENCE, _FS1_CASE, _FS1_JUDGE),
           (_FS2_EVIDENCE, _FS2_CASE, _FS2_JUDGE),
           (_FS3_EVIDENCE, _FS3_CASE, _FS3_JUDGE)]

# ---------- 审计协议(T15.4, 独立于评审 rubric) ----------

AUDIT_SYSTEM_PROMPT = """你是叙事问答系统的独立审计员。被评系统对一个否定型判断题(正确答案为「否/没有」)给出了否定回答,且检索证据并未命中出题方标注的金标证据块。你的任务是审计:这个否定回答是从检索到的邻近证据合理推断出来的,还是在没有证据依据的情况下猜测的。只依据用户消息给出的材料判断,严禁使用材料之外的任何知识。

输出协议(强制):只输出一个合法的 JSON 对象,不输出任何其他文字、解释或代码块标记:
{"classification": "reasonable_negation_inference" 或 "unsupported_guess",
 "reasoning": "一句话理由"}

判定标准:
- 检索到的证据块中存在与问题直接相关的信息(同一话题/同一场景的事实),且否定结论可以从这些信息合理推出或与之自洽 → reasonable_negation_inference;
- 检索到的证据块与问题实质无关,否定结论没有任何证据依据(连排除法也不成立) → unsupported_guess。"""

AUDIT_RETRY_HINT = ("你上一次的输出不是符合审计协议的合法 JSON 对象。请重新审计:只输出一个"
                    "JSON 对象,含 classification(二选一)与 reasoning(一句话)两个字段。")


# ---------- 通用小工具 ----------

def format_evidence(rank: int, chunk: dict) -> str:
    """证据块头部标注 + chunk 全文原文(展示分离红线)。

    与 generate.format_evidence 同款式;此处就地复制而非 import——generate 模块
    顶层会连带引入检索器/向量索引等重依赖, 评审层不需要。
    """
    speakers = "、".join(chunk["speakers"]) if chunk["speakers"] else "无"
    return (f"[证据{rank} | {chunk['chunk_id']} | rfid {chunk['rfid_start']}-"
            f"{chunk['rfid_end']} | 出场: {speakers}]\n{chunk['text']}")


def build_case_user(question: str, gold_answer: str, accept_criteria: str,
                    ev_blocks: list[str], verdict: str, answer: str,
                    rfids) -> str:
    """评审实际案例 / few-shot 案例共用的 user 消息模板(保证格式一致)。"""
    ref = f"参考答案:{gold_answer}"
    if accept_criteria:
        ref += f"\n判分要点:{accept_criteria}"
    return (f"【问题】{question}\n\n【参考要点】\n{ref}\n\n"
            f"【证据】(被评系统检索并实际使用的证据块,按融合排名排序)\n"
            + "\n\n".join(ev_blocks)
            + f"\n\n【被评答案】\nverdict: {verdict}\nanswer: {answer}\n"
              f"evidence_rfids: {json.dumps(rfids, ensure_ascii=False)}\n\n"
              f"请依据以上材料对被评答案评分,只输出符合评审协议的一个 JSON 对象。")


def build_prefix() -> list[dict]:
    """[system, 例1问, 例1答, 例2问, 例2答] —— 全 54 题逐字节相同的前缀(前缀缓存)。"""
    msgs = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    for ev, case, judge_obj in FEWSHOT:
        user = build_case_user(case["question"], case["gold_answer"],
                               case["accept_criteria"],
                               [format_evidence(1, ev)],
                               case["verdict"], case["answer"],
                               case["evidence_rfids"])
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant",
                     "content": json.dumps(judge_obj, ensure_ascii=False)})
    return msgs


def _coerce_score(v) -> int | None:
    """评分值归一: 数值 → 1~5 整数;bool/越界/非数值 → None(视为协议不符)。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        i = int(round(v))
        return i if 1 <= i <= 5 else None
    return None


def _coerce_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def judge_schema_ok(parsed) -> bool:
    """评审协议最低要求: 四个 1~5 分整数字段齐备(缺任一 → 重试路径)。"""
    return (isinstance(parsed, dict)
            and all(_coerce_score(parsed.get(k)) is not None
                    for k in ("score", "faithfulness", "completeness", "conciseness")))


def normalize_citations(raw) -> list[dict]:
    """citations → [{"rfid":[s,e], "supports":bool}];无法归一的条目丢弃不计数。"""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for it in raw:
        if not isinstance(it, dict):
            continue
        rf = it.get("rfid")
        if isinstance(rf, int) and not isinstance(rf, bool):
            interval = [rf, rf]
        elif (isinstance(rf, list) and len(rf) == 2
              and all(isinstance(x, int) and not isinstance(x, bool) for x in rf)):
            interval = [rf[0], rf[1]]
        else:
            continue
        sup = _coerce_bool(it.get("supports"))
        out.append({"rfid": interval, "supports": bool(sup)})
    return out


def merge_usages(usages: list[dict]) -> dict:
    """多次尝试的 usage 求和(费用口径: 尝试过的调用都花钱, 不因重试漏记)。"""
    out: dict = {}
    for k in ("prompt_tokens", "completion_tokens",
              "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if any(k in u for u in usages):
            out[k] = sum(u.get(k) or 0 for u in usages)
    return out


# ---------- 数据装载 ----------

def resolve_run_dir(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def default_arm(run_dir: Path) -> str:
    """缺省臂 = 该 run 生成时实际用的臂(config_copy.yaml 的 generation.retriever)。"""
    cfg = load_config(run_dir / "config_copy.yaml")
    return cfg["generation"]["retriever"]


def top5_of(entry: dict) -> list[dict]:
    """retrieval_results 单题条目 → 按 rank 排序的前 5 个 top 项(生成送 prompt 的证据)。"""
    return sorted(entry["top"], key=lambda t: t["rank"])[:TOP_K]


# ---------- 单题评审 ----------

def judge_one(client: LLMClient, q: dict, a: dict, ev_blocks: list[str]) -> dict:
    messages = build_prefix() + [{"role": "user", "content": build_case_user(
        q["question"], q.get("gold_answer", ""), q.get("accept_criteria", ""),
        ev_blocks, a["verdict"], a["answer"], a.get("evidence_rfids") or [])}]

    attempts = [client.chat_json(messages)]
    parsed = attempts[0].get("parsed")
    if not judge_schema_ok(parsed):
        parsed = None
        retry_msgs = messages + [
            {"role": "assistant", "content": attempts[0].get("content") or ""},
            {"role": "user", "content": JUDGE_RETRY_HINT}]
        attempts.append(client.chat_json(retry_msgs))
        if judge_schema_ok(attempts[1].get("parsed")):
            parsed = attempts[1]["parsed"]

    usages = [a_["usage"] for a_ in attempts if a_.get("usage")]
    row = {"qid": q["qid"], "type": q["type"], "n_attempts": len(attempts),
           "scores": None, "hallucination": None, "citations": [], "issues": [],
           "usage": merge_usages(usages) if usages else {},
           "latency_ms": round(sum(a_.get("latency_seconds") or 0
                                   for a_ in attempts) * 1000),
           "cache_hit": attempts[-1]["cache_hit"],
           "template_version": JUDGE_TEMPLATE_VERSION,
           "parse_failed": parsed is None}
    if parsed is not None:
        row["scores"] = {k: _coerce_score(parsed[k]) for k in
                         ("score", "faithfulness", "completeness", "conciseness")}
        halluc = _coerce_bool(parsed.get("hallucination"))
        row["hallucination"] = halluc if halluc is not None else False
        row["citations"] = normalize_citations(parsed.get("citations"))
        issues = parsed.get("issues")
        row["issues"] = [str(x) for x in issues][:5] if isinstance(issues, list) else []
    return row


# ---------- 汇总(全部由 details 推导, 零非确定字段) ----------

def compute_metrics(details: list[dict], arm: str, run_name: str) -> dict:
    ok = [d for d in details if not d["parse_failed"]]

    def mean(rows: list[dict], key: str):
        vals = [d["scores"][key] for d in rows]
        return round(sum(vals) / len(vals), 4) if vals else None

    entries = [c for d in ok for c in d["citations"]]
    costs = [cost_usd(d["usage"]) for d in details if d.get("usage")]
    by_type = {}
    for t in GENERATIVE_TYPES:
        rows = [d for d in ok if d["type"] == t]
        by_type[t] = {"n": len(rows), "score_mean": mean(rows, "score"),
                      "faithfulness_mean": mean(rows, "faithfulness"),
                      "completeness_mean": mean(rows, "completeness"),
                      "conciseness_mean": mean(rows, "conciseness")}
    return {
        "n": len(details),
        "n_evaluated": len(ok),
        "parse_failures": len(details) - len(ok),
        "score_mean": mean(ok, "score"),
        "faithfulness_mean": mean(ok, "faithfulness"),
        "completeness_mean": mean(ok, "completeness"),
        "conciseness_mean": mean(ok, "conciseness"),
        "hallucination_rate": (round(sum(1 for d in ok if d["hallucination"]) / len(ok), 4)
                               if ok else None),
        "citation_entries": len(entries),
        "citation_support_rate": (round(sum(1 for c in entries if c["supports"]) / len(entries), 4)
                                  if entries else None),
        "score_distribution": {str(i): sum(1 for d in ok if d["scores"]["score"] == i)
                               for i in range(1, 6)},
        "by_type": by_type,
        "cost_usd": {"valley": round(sum(c["usd_offpeak"] for c in costs), 6),
                     "peak": round(sum(c["usd_peak"] for c in costs), 6),
                     "n_calls": sum(d["n_attempts"] for d in details)},
        "template_version": JUDGE_TEMPLATE_VERSION,
        "arm": arm,
        "run_dir": run_name,
    }


def write_agreement(run_dir: Path, details: list[dict], arm: str,
                    answers: dict[str, dict], qaset: dict, n_sample: int) -> Path | None:
    """人工抽检模板: 固定 seed 抽 qid, 人工列留 null;±1 一致率由人工回填后再算。"""
    qmap = {q["qid"]: q for q in qaset["questions"]}
    qids = sorted(d["qid"] for d in details if not d["parse_failed"])
    if not qids:
        return None
    picked = random.Random(SAMPLE_SEED).sample(qids, min(n_sample, len(qids)))
    by_qid = {d["qid"]: d for d in details}
    entries = []
    for qid in picked:
        d = by_qid[qid]
        entries.append({
            "qid": qid, "question": qmap[qid]["question"],
            "answer": answers[qid]["answer"][:200],
            "judge_score": d["scores"]["score"],
            "judge_faithfulness": d["scores"]["faithfulness"],
            "judge_completeness": d["scores"]["completeness"],
            "judge_conciseness": d["scores"]["conciseness"],
            "human_score": None, "human_faithfulness": None,
            "human_completeness": None, "human_conciseness": None,
            "human_notes": ""})
    out = {"arm": arm, "sample_seed": SAMPLE_SEED,
           "formula": "±1 一致率 = |human−judge|≤1 的题数/题数;另记三维子分",
           "template_version": JUDGE_TEMPLATE_VERSION,
           "rubric_anchor": "5=可直接采用;3=部分正确有遗漏;1=错误或主要靠猜测;"
                            "拒答不扣忠实性(证据确实不足时 faithfulness=5)",
           "entries": entries}
    path = run_dir / "judge_agreement.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------- 主评审流程 ----------

def run_judge(args, cfg: dict, paths: dict) -> None:
    run_dir = resolve_run_dir(args.run)
    arm = args.arm or default_arm(run_dir)
    qaset = load_qaset(paths["qaset"])
    answers = {json.loads(l)["qid"]: json.loads(l)
               for l in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()}
    retrieval = json.loads((run_dir / "retrieval_results.json")
                           .read_text(encoding="utf-8"))["per_question"][arm]
    rmap = {e["qid"]: e for e in retrieval}
    chunk_by_id = {c["chunk_id"]: c for c in load_chunks(paths["scenes"])}

    # 证据一致性 sanity: 本脚本取的 top-5 应与该 run 生成时实际送 prompt 的块一致
    gen_log_p = run_dir / "generation_log.json"
    gen_retrieved = {}
    if gen_log_p.exists():
        gen_retrieved = {g["qid"]: g["retrieved_chunk_ids"]
                         for g in json.loads(gen_log_p.read_text(encoding="utf-8"))["questions"]}
    questions = [q for q in qaset["questions"] if q["type"] in GENERATIVE_TYPES]
    if args.limit:
        questions = questions[:args.limit]

    llm_cfg = {**cfg["generation"]["llm"], "max_tokens": JUDGE_MAX_TOKENS}
    client = LLMClient(llm_cfg)
    print(f"== LLM-as-judge ==\n  被评 run={run_dir.name}  臂={arm}  生成类题={len(questions)}"
          f"  模型={llm_cfg['model']}(max_tokens={JUDGE_MAX_TOKENS})  "
          f"模板={JUDGE_TEMPLATE_VERSION}")

    details: list[dict] = []
    details_path = run_dir / "judge_details.jsonl"
    n_mismatch = 0
    with open(details_path, "w", encoding="utf-8", newline="\n") as f:   # 逐题落盘
        for i, q in enumerate(questions, 1):
            top5 = top5_of(rmap[q["qid"]])
            ev_blocks = [format_evidence(t["rank"], chunk_by_id[t["chunk_id"]])
                         for t in top5]
            if q["qid"] in gen_retrieved:
                if [t["chunk_id"] for t in top5] != gen_retrieved[q["qid"]]:
                    n_mismatch += 1
            row = judge_one(client, q, answers[q["qid"]], ev_blocks)
            details.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if row["parse_failed"]:
                print(f"  [{i:>3}/{len(questions)}] {q['qid']} [解析失败]")
            else:
                s = row["scores"]
                n_sup = sum(1 for c in row["citations"] if c["supports"])
                cache = "缓存" if row["cache_hit"] else "网"
                print(f"  [{i:>3}/{len(questions)}] {q['qid']} 总分={s['score']} "
                      f"忠={s['faithfulness']} 完={s['completeness']} 简={s['conciseness']} "
                      f"幻觉={'是' if row['hallucination'] else '否'} "
                      f"引用{len(row['citations'])}条(支持{n_sup}) [{cache}]")
    if gen_retrieved:
        print(f"  证据一致性: top-5 与 generation_log.retrieved_chunk_ids 核对 "
              f"{len(questions) - n_mismatch}/{len(questions)} 一致"
              + ("  [告警:存在不一致!]" if n_mismatch else ""))

    metrics = compute_metrics(details, arm, run_dir.name)
    (run_dir / "judge_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 非确定信息全部收在 run_log(metrics 保持重跑逐字节一致)
    run_log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cli": {"run": str(run_dir), "arm": arm, "limit": args.limit,
                "sample": args.sample},
        "llm": {k: llm_cfg[k] for k in ("model", "base_url", "thinking",
                                        "temperature", "max_tokens")},
        "template_version": JUDGE_TEMPLATE_VERSION,
        "note": "judge_metrics.json 由 judge_details 的评分与 usage 推导, 零非确定字段;"
                "本文件收纳时间/环境/参数等非确定信息(details/metrics 与 log 分离纪律)。",
    }
    (run_dir / "judge_run_log.json").write_text(
        json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sample > 0:
        p = write_agreement(run_dir, details, arm, answers, qaset, args.sample)
        print(f"  抽检模板: {p.name}(seed={SAMPLE_SEED}, {args.sample} 题, 人工列待填)")

    c = metrics["cost_usd"]
    print(f"\n== 汇总 ==\n  总分均值 {metrics['score_mean']}(忠 {metrics['faithfulness_mean']}"
          f" / 完 {metrics['completeness_mean']} / 简 {metrics['conciseness_mean']})  "
          f"幻觉率 {metrics['hallucination_rate']}  引用支持率 "
          f"{metrics['citation_support_rate']}({metrics['citation_entries']} 条)  "
          f"分数分布 {metrics['score_distribution']}")
    print(f"  费用 ${c['valley']}(谷) / ${c['peak']}(峰)  调用 {c['n_calls']} 次"
          f"  解析失败 {metrics['parse_failures']} 题")
    lat = sorted(d["latency_ms"] for d in details)
    if lat:
        print(f"  延迟 P50={lat[len(lat)//2]}ms P95={lat[min(int(len(lat)*0.95+0.999999)-1, len(lat)-1)]}ms"
              f"(含缓存回放)")
    print(f"\n产物: {details_path}\n      {run_dir / 'judge_metrics.json'}\n"
          f"      {run_dir / 'judge_run_log.json'}")


# ---------- T15.4 缺证边界审计 ----------

def audit_one(client: LLMClient, q: dict, a: dict, ev_blocks: list[str]) -> dict:
    """一次审计调用: 合理否定推断 vs 无证据猜测。"""
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"【问题】{q['question']}(正确答案:否)\n\n"
            f"【证据】(检索 top-{TOP_K},均未命中出题方标注的金标证据块)\n"
            + "\n\n".join(ev_blocks)
            + f"\n\n【被评答案】\nverdict: {a['verdict']}\nanswer: {a['answer']}\n"
              f"evidence_rfids: {json.dumps(a.get('evidence_rfids') or [], ensure_ascii=False)}\n\n"
              f"请审计该否定回答的依据,只输出符合协议的一个 JSON 对象。")}]
    attempts = [client.chat_json(messages)]
    parsed = attempts[0].get("parsed")

    def ok(p) -> bool:
        return (isinstance(p, dict)
                and p.get("classification") in ("reasonable_negation_inference",
                                                "unsupported_guess"))

    if not ok(parsed):
        retry = messages + [
            {"role": "assistant", "content": attempts[0].get("content") or ""},
            {"role": "user", "content": AUDIT_RETRY_HINT}]
        attempts.append(client.chat_json(retry))
        if ok(attempts[1].get("parsed")):
            parsed = attempts[1]["parsed"]

    usages = [x["usage"] for x in attempts if x.get("usage")]
    row = {"classification": None, "reasoning": None,
           "parse_failed": parsed is None,
           "usage": merge_usages(usages) if usages else {},
           "latency_ms": round(sum(x.get("latency_seconds") or 0 for x in attempts) * 1000),
           "cache_hit": attempts[-1]["cache_hit"],
           "template_version": AUDIT_TEMPLATE_VERSION}
    if parsed is not None:
        row["classification"] = parsed["classification"]
        row["reasoning"] = str(parsed.get("reasoning", ""))[:200]
    return row


def run_audit(args, cfg: dict, paths: dict) -> None:
    """交叉表(纯脚本) + miss-但-答对题逐例审计 LLM 调用; 产物写关臂 run 目录。"""
    if not args.compare_run:
        sys.exit("--audit 需要 --compare-run 指定对照臂 run 目录")
    run_dir = resolve_run_dir(args.run)                  # 关臂(裁定主语境)
    cmp_dir = resolve_run_dir(args.compare_run)          # 开臂(对照)
    arm = args.arm or default_arm(run_dir)
    cmp_arm = default_arm(cmp_dir)

    qaset = load_qaset(paths["qaset"])
    chunk_by_id = {c["chunk_id"]: c for c in load_chunks(paths["scenes"])}

    def load_side(rd: Path, arm_name: str):
        res = {e["qid"]: e for e in json.loads(
            (rd / "retrieval_results.json").read_text(encoding="utf-8")
        )["per_question"][arm_name]}
        ans = {json.loads(l)["qid"]: json.loads(l)
               for l in (rd / "answers.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()}
        return res, ans

    res, ans = load_side(run_dir, arm)
    res_o, ans_o = load_side(cmp_dir, cmp_arm)

    def hit5(rmap, qid) -> bool:
        return any(t["hit"] for t in top5_of(rmap[qid]))

    def cross(rmap, amap) -> dict:
        cells = {k: [] for k in ("gold_hit|correct", "gold_hit|wrong",
                                 "gold_miss|correct", "gold_miss|wrong")}
        targets: list[str] = []
        for q in qaset["questions"]:
            if q["type"] != "yesno":
                continue
            qid = q["qid"]
            hit, pred = hit5(rmap, qid), amap[qid]["verdict"]
            correct = pred == q["gold_verdict"]          # evaluate.py 口径
            key = ("gold_hit" if hit else "gold_miss") + ("|correct" if correct else "|wrong")
            cells[key].append(f"{qid}({q['gold_verdict']}->{pred})")
            if q["gold_verdict"] == "no" and not hit and correct:
                targets.append(qid)
        return {"cells": {k: {"n": len(v), "qids": v} for k, v in cells.items()},
                "audit_targets": targets}

    closed_cross = cross(res, ans)
    open_cross = cross(res_o, ans_o)
    una = [q for q in qaset["questions"] if q["type"] == "unanswerable"]
    una_bad = [{"qid": q["qid"], "verdict": ans[q["qid"]]["verdict"]}
               for q in una if ans[q["qid"]]["verdict"] != "unknown"]

    audits = []
    targets = closed_cross["audit_targets"]
    if targets:
        llm_cfg = {**cfg["generation"]["llm"], "max_tokens": JUDGE_MAX_TOKENS}
        client = LLMClient(llm_cfg)
        print(f"== 缺证审计 ==  审计目标 {len(targets)} 题: {targets}")
        for qid in targets:
            q = next(x for x in qaset["questions"] if x["qid"] == qid)
            a = ans[qid]
            ev_blocks = [format_evidence(t["rank"], chunk_by_id[t["chunk_id"]])
                         for t in top5_of(res[qid])]
            r = audit_one(client, q, a, ev_blocks)
            open_pred = ans_o[qid]["verdict"]
            audits.append({
                "qid": qid, "question": q["question"], "answer": a["answer"],
                "closed_arm": {"arm": arm, "run_dir": run_dir.name,
                               "verdict": a["verdict"], "correct": True,
                               "gold_hit_top5": False},
                "open_arm": {"arm": cmp_arm, "run_dir": cmp_dir.name,
                             "verdict": open_pred,
                             "correct": open_pred == q["gold_verdict"],
                             "gold_hit_top5": hit5(res_o, qid),
                             "same_verdict_as_closed": open_pred == a["verdict"]},
                "evidence_top5_chunk_ids": [t["chunk_id"] for t in top5_of(res[qid])],
                **r})
            tag = r["classification"] or "[解析失败]"
            print(f"  {qid}: {tag}  (开臂 verdict={open_pred})")
    else:
        print("== 缺证审计 ==  无审计目标(gold=no 且 miss 且答对 的题为空)")

    out = {
        "arm": arm, "run_dir": run_dir.name,
        "compare_arm": cmp_arm, "compare_run_dir": cmp_dir.name,
        "template_version": AUDIT_TEMPLATE_VERSION,
        "criteria": {
            "gold_hit": f"该臂 top-{TOP_K} 内任一 gold chunk 命中(retrieval_results hit 标志)",
            "verdict_correct": "pred verdict == gold_verdict(evaluate.py 口径)",
            "audit_target": "type=yesno 且 gold_verdict=no 且检索 miss 且 verdict 正确",
            "audit_judge": "classification: reasonable_negation_inference(邻近证据合理"
                           "推出否定) vs unsupported_guess(无证据猜测)"},
        "yesno_cross_table": closed_cross,
        "compare_arm_yesno_cross_table": open_cross,
        "unanswerable_row": {"n": len(una), "expected": "全部应为 unknown",
                             "n_unknown": len(una) - len(una_bad),
                             "non_unknown": una_bad},
        "audit_targets": targets,
        "per_target_audits": audits,
        "note": "两臂共用一份, 放关臂 run 目录;交叉表零 LLM 调用, 仅 audit_targets 逐例调 LLM。",
    }
    out_path = run_dir / "judge_missing_evidence_audit.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物: {out_path}")


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge 评审层(T14/T15.4)")
    ap.add_argument("--run", required=True,
                    help="被评 run 目录(runs/ 下名字或绝对路径; audit 模式=关臂)")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--arm", default=None,
                    help="检索臂名;缺省读该 run config_copy.yaml 的 generation.retriever")
    ap.add_argument("--limit", type=int, default=0, help="只评前 n 道生成类题(0=全量)")
    ap.add_argument("--sample", type=int, default=11,
                    help="人工抽检抽样题数(0=不生成 judge_agreement.json)")
    ap.add_argument("--audit", action="store_true",
                    help="T15.4 缺证审计模式(交叉表+逐例审计, 不做常规评审)")
    ap.add_argument("--compare-run", default=None, help="audit 模式的对照臂 run 目录")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    if args.audit:
        run_audit(args, cfg, paths)
    else:
        run_judge(args, cfg, paths)


if __name__ == "__main__":
    main()
