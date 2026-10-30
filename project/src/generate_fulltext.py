# -*- coding: utf-8 -*-
"""长上下文对照(全文塞入 vs RAG) + 位置消融。

三个模式互斥, 产物全部落独立实验目录(缺省 <runs_dir>/fulltext/<YYYYMMDD-HHMMSS>/,
可用 --out-dir 显式指定, 便于冒烟→全量→位置消融三步复用同一目录):
    --smoke     冒烟: 2 题(yesno 1 + unanswerable 1)× 2 半 = 4 调用,
                先验三件事: ①端点窗口装得下 ②前缀缓存跨题命中(第 2 题同半
                读 usage.prompt_cache_hit_tokens) ③单题实际延迟;
    --fulltext  全量: 36 题(yesno 18 + unanswerable 18)× 2 半 = 72 调用
                (本目录冒烟判定无前缀命中时减半至 18 题)
                → fulltext_answers.jsonl + fulltext_metrics.json(含三方法对照表);
    --position  位置消融: 关臂检索 gold 命中 top-k 的 yesno 题 qid 序前 10 ×
                3 位置(首/中/末) = 30 调用 → position_ablation.json。
    不给模式开关等价于 --fulltext。

输入(全部显式, 代码内不含任何历史 run 名称):
    --rag-off-run  关臂基线 run 目录(--fulltext 与 --position 必填)
    --rag-on-run   开臂基线 run 目录(仅 --fulltext 必填)
    基线目录需含 generation_log.json + config_copy.yaml; 全文对照另需
    metrics.json(--label_accuracy), 位置消融另需 answers.jsonl。
    语料/评测集/模型配置来自 --config(缺省 configs/main.yaml)。

预检先于任何 API 调用与产物写入: 缺文件、字段不全、两臂检索器相同、
top-k 不一致、模型参数与当前 --config 不符、语料 chunk 数或评测集哈希对不上,
都一次性聚合报错退出(退出码非 0), 不会产生半成品。

与 generate.py 生成协议的同构关系(权威实现见 src/generate.py):
  - 全文臂: JSON 输出协议(verdict/answer/evidence_rfids)与拒答规则**逐字保留**,
    系统提示只把"证据"改写为"剧本全文的前半/后半"; 解析重试(RETRY_HINT)与
    引用两规则校验直接复用 generate.generate_one 原函数(零复制);
  - 全文臂**不带 few-shot**, 两条理由: (a) few-shot 的证据用法演示(引用 3 块
    中 1 块)与"证据=整个半本剧本"的语义不符; (b) few-shot 插在正文前会推迟
    共享前缀的起点、白付 ~2.5K token——与"36 题共用前缀吃缓存"的布局冲突;
  - 位置消融则相反: **逐字复用** generate.py 的 SYSTEM_PROMPT + build_fewshot +
    build_user_message(含 few-shot 与"按检索融合排名排序"头), 仅重排证据顺序
    ——测的就是 RAG 自己的 prompt 在指定关臂基线下的位置敏感性, 协议必须与其
    完全同构(few-shot 开关按基线 generation_log.config.few_shot 取, 不写死)。
    generate.py 模块级无重依赖(retrievers 工厂延迟导入), 直接 import。
  - 超时核实: src/llm.py 的 LLMClient **不经 llm_cfg 暴露 timeout**(OpenAI() 用
    SDK 默认 600s/次), 全文臂单次调用预期 10~60s, 默认值足够, 不改 llm.py。

前缀缓存布局(费用关键, DeepSeek 前缀缓存要求跨调用逐字节一致):
    消息 = [系统提示(半别稳定)] + [user: 该半全文(带 rfid 标注行) + 问题]
    ——全文在前、问题在后, 同半 36 题共享 ~65K token 前缀, 第 2 题起命中。

约束: 本文件只做长上下文对照与位置消融, 不改动既有实验脚本; 全部 LLM 调用走
LLMClient(sha256 缓存 + 费用核算 cost_usd); Key 不打印不落盘; 评测集冻结不改题;
API 连续失败即停并如实报告; 冒烟无前缀命中 → 题量减半至 18 题。

用法(示例基线用占位符; 冒烟→全量→位置消融共用一个实验目录):
    python src/generate_fulltext.py --smoke --out-dir runs/fulltext/<ts>
    python src/generate_fulltext.py --fulltext \
        --rag-off-run runs/<off-run> --rag-on-run runs/<on-run> \
        --out-dir runs/fulltext/<ts>
    python src/generate_fulltext.py --position --rag-off-run runs/<off-run> \
        --out-dir runs/fulltext/<ts>
    python src/generate_fulltext.py --fulltext --rag-off-run ... --rag-on-run ... --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")   # 预检错误走 stderr, 同样不能乱码
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.evaluate import RfidLocator, eval_citations, eval_parse_failures, \
    gold_intervals, load_qaset
from src.generate import SYSTEM_PROMPT, build_fewshot, build_user_message, generate_one
from src.index_bm25 import load_chunks
from src.llm import LLMClient, PRICING_PER_MTOK, cost_usd

# 产物名(目录由 --out-dir / 时间戳决定; runs 目录只追加惯例)
ARTIFACT_NAMES = {
    "smoke": "fulltext_smoke_log.json",
    "answers": "fulltext_answers.jsonl",
    "metrics": "fulltext_metrics.json",
    "position": "position_ablation.json",
    "manifest": "experiment_manifest.json",
    "config": "config_copy.yaml",
}
# 缺省输出: <runs_dir>/fulltext/<时间戳>/ —— 与主流程 runs/<时间戳>/ 同构, 不覆盖
DEFAULT_OUTPUT_SUBDIR = "fulltext"

# 题集 = verdict 载题(yesno + unanswerable); 生成类 54 题不进全文臂
# (judge 输入装不下全文, 质量面由 verdict/拒答承载)
FULLTEXT_TYPES = ("yesno", "unanswerable")

# 冒烟前缀命中判定门线: 同半第 2 题命中 token ≥1024(=16 个 64-token
# 缓存块)才算"显著>0"; 两个半都不达标 → 题量减半至 18 题
PREFIX_HIT_MIN_TOKENS = 1024

# 全文臂预算门线(谷价, 美元); 超线只告警, 费用照实记账
BUDGET_CAP_USD_OFFPEAK = 0.40

# 两基线之间必须一致的模型参数(比对键; provider/api_key_env 不参与对照)
LLM_COMPARE_KEYS = ("model", "thinking", "temperature", "max_tokens", "base_url")


# ---------- 路径与基线 run 校验 ----------

def rel_to_project(p: Path) -> str:
    """产物里的溯源字符串: 能写成 project 相对路径就写相对(正斜杠), 否则绝对。"""
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256_of(path: Path) -> str | None:
    """文件内容哈希(不存在/不可读返回 None, 由调用方决定是否算错)。"""
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_run_path(raw: str, runs_root: Path) -> Path:
    """基线 run 目录解析: 绝对路径原样; 相对路径先按 project 根, 再按 runs 根。

    兼容 "runs/20261030-101010" 与裸目录名 "20261030-101010" 两种写法。
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    for cand in (PROJECT_ROOT / p, runs_root / p):
        if cand.is_dir():
            return cand
    return PROJECT_ROOT / p    # 不存在时返回首选候选, 由预检报错


def resolve_out_dir(raw: str | None, runs_root: Path) -> Path:
    """实验目录: 显式 --out-dir 按 project 根解析; 缺省新建时间戳目录。"""
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return runs_root / DEFAULT_OUTPUT_SUBDIR / datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class Baseline:
    """一个 RAG 基线 run 的只读快照(预检通过后构造, 之后不再碰磁盘)。"""
    role: str                 # "rag_off" / "rag_on"
    run_dir: Path
    retriever: str
    top_k: int
    llm: dict
    few_shot: bool
    qaset_label: str
    genlog: dict
    retrieved: dict[str, list[str]]
    answers: dict[str, dict] | None = None          # position 模式
    label_accuracy: dict | None = None             # fulltext 三方法表

    def brief(self) -> dict:
        return {"run": rel_to_project(self.run_dir), "retriever": self.retriever,
                "top_k": self.top_k, "few_shot": self.few_shot,
                "qaset": self.qaset_label}


class PreflightLog:
    """预检问题收集器: 全部查完再一次性报错, 避免"修一个报一个"。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def raise_if_errors(self) -> None:
        if not self.errors:
            return
        lines = [f"\n[预检失败] 共 {len(self.errors)} 项问题:"]
        lines += [f"  - {e}" for e in self.errors]
        lines.append("→ 未调用任何 API, 未写入任何产物。")
        raise SystemExit("\n".join(lines))


def load_baseline(role: str, run_dir: Path, *, need_metrics: bool,
                  need_answers: bool, log: PreflightLog) -> Baseline | None:
    """读取并校验一个基线 run; 缺关键文件时记录问题并返回 None。"""
    if not run_dir.is_dir():
        log.err(f"{role}: 目录不存在 {run_dir}"
                f"(用 --{role.replace('_', '-')} 指定, 支持 runs/<ts> 或绝对路径)")
        return None

    genlog_p = run_dir / "generation_log.json"
    cfg_p = run_dir / "config_copy.yaml"
    if not genlog_p.is_file():
        log.err(f"{role}: 缺少 {rel_to_project(genlog_p)}"
                f"(该 run 未执行生成阶段, 或产物已删除; 先跑 run_all.py 产出该 run)")
        return None
    if not cfg_p.is_file():
        log.err(f"{role}: 缺少 {rel_to_project(cfg_p)}(配置快照, 无法核对检索臂与 top-k)")
        return None
    try:
        genlog = json.loads(genlog_p.read_text(encoding="utf-8"))
        cfg_copy = load_config(cfg_p)
    except (json.JSONDecodeError, OSError) as e:
        log.err(f"{role}: 读取失败 {rel_to_project(run_dir)}: {type(e).__name__}: {e}")
        return None

    gc, gcfg = genlog.get("config") or {}, (cfg_copy.get("generation") or {})
    missing = [k for k in ("retriever", "top_k", "llm", "few_shot") if k not in gc]
    if missing:
        log.err(f"{role}: generation_log.json 的 config 缺字段 {missing}")
        return None
    if "retriever" not in gcfg or "top_k" not in gcfg:
        log.err(f"{role}: config_copy.yaml 缺 generation.retriever / generation.top_k")
        return None

    if gcfg["retriever"] != gc["retriever"]:
        log.err(f"{role}: config_copy.yaml 的 generation.retriever={gcfg['retriever']!r}"
                f" 与 generation_log 的 {gc['retriever']!r} 不一致(目录内容被改过?)")
    if int(gcfg["top_k"]) != int(gc["top_k"]):
        log.err(f"{role}: config_copy.yaml 的 generation.top_k={gcfg['top_k']}"
                f" 与 generation_log 的 {gc['top_k']} 不一致(目录内容被改过?)")
    fs_copy = ((cfg_copy.get("generation") or {}).get("prompt") or {}).get("few_shot", True)
    if bool(fs_copy) != bool(gc["few_shot"]):
        log.err(f"{role}: config_copy.yaml 的 generation.prompt.few_shot={fs_copy}"
                f" 与 generation_log 的 few_shot={gc['few_shot']} 不一致")

    qs = genlog.get("questions") or []
    if not qs:
        log.err(f"{role}: generation_log.json 的 questions 为空(该 run 未跑完生成阶段)")
        return None
    missing_ret = [q.get("qid") for q in qs if not q.get("retrieved_chunk_ids")]
    if missing_ret:
        log.err(f"{role}: {len(missing_ret)} 题的 retrieved_chunk_ids 缺失"
                f"(前几题: {missing_ret[:5]})——位置/对照实验需要检索上下文")
        return None
    retrieved = {q["qid"]: list(q["retrieved_chunk_ids"]) for q in qs}

    base = Baseline(role=role, run_dir=run_dir, retriever=gc["retriever"],
                    top_k=int(gc["top_k"]), llm=gc["llm"], few_shot=bool(gc["few_shot"]),
                    qaset_label=str(gc.get("qaset") or ""),
                    genlog=genlog, retrieved=retrieved)

    if need_metrics:
        metrics_p = run_dir / "metrics.json"
        if not metrics_p.is_file():
            log.err(f"{role}: 缺少 {rel_to_project(metrics_p)}(三方法表要读 label_accuracy)")
        else:
            try:
                la = json.loads(metrics_p.read_text(encoding="utf-8")).get("label_accuracy")
            except (json.JSONDecodeError, OSError) as e:
                log.err(f"{role}: metrics.json 读取失败: {type(e).__name__}: {e}")
                la = None
            if la is None:
                log.err(f"{role}: metrics.json 无 label_accuracy(该 run 的答案层未评估)")
            else:
                base.label_accuracy = la

    if need_answers:
        ans_p = run_dir / "answers.jsonl"
        if not ans_p.is_file():
            log.err(f"{role}: 缺少 {rel_to_project(ans_p)}(位置消融要读基线答案作对比)")
        else:
            answers = {}
            for i, line in enumerate(ans_p.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    log.err(f"{role}: answers.jsonl 第 {i} 行不是合法 JSON")
                    answers = {}
                    break
                answers[r["qid"]] = r
            if answers:
                base.answers = answers
            elif not any("answers.jsonl 第" in e for e in log.errors):
                log.err(f"{role}: answers.jsonl 为空")
    return base


def check_corpus_and_qaset(role: str, base: Baseline, *, chunks: list[dict],
                           qaset: dict, qaset_path: Path, scenes_path: Path,
                           log: PreflightLog) -> None:
    """语料/评测集一致性: 按文件名 + 内容哈希 + chunk id 归属性核对。

    基线日志里记的是开发机绝对路径, 这里只取 basename 与哈希, 不依赖上级仓库布局。
    """
    env_p = base.run_dir / "env_info.json"
    if env_p.is_file():
        try:
            env = json.loads(env_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            env = {}
        paths = (env.get("config") or {}).get("paths") or {}
        scenes_label = paths.get("scenes")
        if scenes_label and Path(str(scenes_label)).name != Path(scenes_path).name:
            log.err(f"{role}: 基线语料 {scenes_label!r} 与当前 {Path(scenes_path).name!r} 不同名")
        n_chunks_base = (((env.get("graph") or {}).get("entities") or {}).get("n_chunks"))
        if isinstance(n_chunks_base, int) and n_chunks_base != len(chunks):
            log.err(f"{role}: 基线语料 {n_chunks_base} chunk, 当前 {len(chunks)} chunk"
                    f" —— 语料版本不一致, 全文切分与检索上下文都不可比")
    else:
        log.warn(f"{role}: 无 env_info.json, 跳过语料版本核对")

    # 检索块必须全部落在当前语料里(直接抓语料被替换/重建的情况)
    ids = {c["chunk_id"] for c in chunks}
    stray = {cid for v in base.retrieved.values() for cid in v if cid not in ids}
    if stray:
        log.err(f"{role}: 检索到的 {len(stray)} 个 chunk_id 不在当前语料中"
                f"(示例: {sorted(stray)[:3]})——基线与当前语料不是同一版本")

    # 评测集: 文件名 + 内容哈希
    base_q = base.qaset_label
    if base_q:
        if Path(base_q).name != qaset_path.name:
            log.err(f"{role}: 基线评测集 {Path(base_q).name!r} 与当前 "
                    f"{qaset_path.name!r} 不同名")
        h_base = sha256_of(Path(base_q)) if Path(base_q).is_file() else None
        h_cur = sha256_of(qaset_path)
        if h_base and h_cur and h_base != h_cur:
            log.err(f"{role}: 评测集内容哈希不一致"
                    f"(基线 {h_base[:12]}… vs 当前 {h_cur[:12]}…)")
    else:
        log.warn(f"{role}: generation_log 未记录评测集路径, 只按题目覆盖核对")

    # 题目覆盖面: 基线日志必须覆盖本次要用的题
    need = {q["qid"] for q in qaset["questions"] if q["type"] in FULLTEXT_TYPES}
    lack = sorted(need - set(base.retrieved))
    if lack:
        log.err(f"{role}: 基线日志缺 {len(lack)} 道本次题集的题"
                f"(示例: {lack[:5]})——评测集版本与基线不一致")


def cross_check_baselines(off: Baseline, on: Baseline, cfg: dict,
                          log: PreflightLog) -> None:
    """两基线之间、基线与当前 --config 之间的对照条件核对。"""
    if off.run_dir.resolve() == on.run_dir.resolve():
        log.err("rag_off 与 rag_on 指向同一 run 目录——两臂对照必须来自不同 run")
    if off.retriever == on.retriever:
        log.err(f"rag_off 与 rag_on 的检索臂相同({off.retriever})——"
                f"开/关对照要求两臂不同")
    if off.top_k != on.top_k:
        log.err(f"rag_off top_k={off.top_k} 与 rag_on top_k={on.top_k} 不一致"
                f"——三方法对照必须在同一证据数下比较")

    diff_llm = [k for k in LLM_COMPARE_KEYS if off.llm.get(k) != on.llm.get(k)]
    if diff_llm:
        log.err(f"两基线的模型参数不一致: {[(k, off.llm.get(k), on.llm.get(k)) for k in diff_llm]}")

    cur_llm = (cfg.get("generation") or {}).get("llm") or {}
    diff_cur = [(k, off.llm.get(k), cur_llm.get(k)) for k in LLM_COMPARE_KEYS
                if k in cur_llm and off.llm.get(k) != cur_llm.get(k)]
    if diff_cur:
        log.err(f"基线模型参数与当前 --config 的 generation.llm 不一致: {diff_cur}"
                f"——全文臂与 RAG 臂必须同模型同采样, 否则质量差异不可归因")

    cur_ret = (cfg.get("generation") or {}).get("retriever")
    if cur_ret and cur_ret != off.retriever:
        log.warn(f"当前 --config 的 generation.retriever={cur_ret!r} 与关臂 "
                 f"{off.retriever!r} 不同——确认这是有意为之")
    cur_k = (cfg.get("generation") or {}).get("top_k")
    if cur_k is not None and int(cur_k) != off.top_k:
        log.warn(f"当前 --config 的 generation.top_k={cur_k} 与基线 top_k={off.top_k} 不同"
                 f"(三方法表两侧各自用自身 run 的配置, 不影响本实验, 但报告里需注明)")


def existing_conflicts(mode: str, out_dir: Path, force: bool,
                       log: PreflightLog) -> None:
    """同一实验目录内的重复执行保护(冒烟→全量→位置消融共用目录是允许的)。"""
    checks = {"smoke": ("smoke",), "fulltext": ("answers", "metrics"),
              "position": ("position",)}[mode]
    hits = [ARTIFACT_NAMES[k] for k in checks if (out_dir / ARTIFACT_NAMES[k]).is_file()]
    if hits and not force:
        log.err(f"{mode} 模式的产物已存在: {out_dir} → {hits}"
                f"(加 --force 覆盖, 或换一个 --out-dir 保留历史)")


# ---------- 全文臂系统提示 ----------

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
    (结构性错) 与 "证据在但没找到/lost in the middle" 分开, 便于如实归因。
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
    # 无缓存成本估计(假设零前缀命中): 实测费用被 DeepSeek 前缀缓存大幅折价(命中
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


# ---------- 调用计划(预检期算定, 供 --dry-run 与 manifest 使用) ----------

@dataclass
class CallPlan:
    mode: str
    n_questions: int
    n_calls: int
    description: str
    reduced: bool = False
    reduce_reason: str | None = None
    smoke_reused: bool = False

    def brief(self) -> dict:
        return {"mode": self.mode, "n_questions": self.n_questions,
                "n_calls": self.n_calls, "description": self.description,
                "reduced": self.reduced, "reduce_reason": self.reduce_reason,
                "smoke_reused_from_out_dir": self.smoke_reused}


def select_fulltext_questions(qaset: dict,
                              reduce_questions: bool) -> tuple[list[dict], str | None]:
    """全文臂题集: verdict 载题; 冒烟门线未命中时按 qid 序各取前 9 减半。"""
    questions = [q for q in qaset["questions"] if q["type"] in FULLTEXT_TYPES]
    note = None
    if reduce_questions:
        yn = sorted(q["qid"] for q in questions if q["type"] == "yesno")[:9]
        ua = sorted(q["qid"] for q in questions if q["type"] == "unanswerable")[:9]
        keep = set(yn) | set(ua)
        questions = [q for q in questions if q["qid"] in keep]
        note = ("冒烟未观测到前缀缓存命中, 题量减半: yesno 9(qid 序前 9)"
                " + unanswerable 9(qid 序前 9) = 18 题×2 半 = 36 调用")
    questions.sort(key=lambda q: q["qid"])
    return questions, note


def select_position_candidates(qaset: dict, locator: RfidLocator, retrieved: dict,
                               top_k: int) -> list[dict]:
    """位置消融候选: yesno 且 gold 命中基线 top-k 的题, 按 qid 升序。

    gold 块 = 检索排名最靠前的命中块(规则兜底多块情形)。
    """
    cands = []
    for q in sorted(qaset["questions"], key=lambda x: x["qid"]):
        if q["type"] != "yesno":
            continue
        try:
            gold = locator.gold_chunks(gold_intervals(q))
        except KeyError:
            continue
        if q["qid"] not in retrieved:
            continue
        topk = retrieved[q["qid"]][:top_k]
        hits = [c for c in topk if c in gold]     # 已按检索排名序
        if hits:
            cands.append({"qid": q["qid"], "question": q["question"],
                          "gold_verdict": q["gold_verdict"],
                          "gold_block": hits[0],
                          "n_gold_in_topk": len(hits),
                          "baseline_rank": topk.index(hits[0]) + 1,
                          "topk": topk})
    return cands


def positions_for(top_k: int) -> dict[str, int]:
    """gold 块目标位(1-indexed): 首位 / 中间位 / 末位; 中间位 = (top_k+1)//2。"""
    return {"first": 1, "middle": (top_k + 1) // 2, "last": top_k}


# ---------- 冒烟 ----------

def run_smoke(client: LLMClient, halves: dict, qaset: dict, out_dir: Path) -> dict:
    """2 题(yesno 取 qid 最小 + unanswerable 取 qid 最小)× 2 半 = 4 调用。

    执行顺序刻意同半相邻(Q09-A→Q26-A→Q09-B→Q26-B): 第 2 题同半的前缀缓存
    命中才是题量减半门线的判定依据。窗口超限(HTTP 400 context/length 类错误)
    不崩脚本: 记录后如实报告, 降级路径(四分块/停下)由人按结果决定。
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
        "created": datetime.now().isoformat(timespec="seconds"),
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
    smoke_path = out_dir / ARTIFACT_NAMES["smoke"]
    smoke_path.write_text(json.dumps(smoke, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n== 冒烟三查 ==\n  ①窗口: {'通过' if ok_all else '失败(需降级决策)'}")
    print(f"  ②前缀缓存命中(第2题同半): A={second_calls['A'].get('prompt_cache_hit_tokens')}"
          f" tok / B={second_calls['B'].get('prompt_cache_hit_tokens')} tok"
          f" (门线 {PREFIX_HIT_MIN_TOKENS}) → {'命中' if all(prefix_ok.values()) else '未命中→题量减半'}")
    print(f"  ③单题延迟(s): {[r.get('latency_seconds') for r in records]}")
    print(f"  冒烟费用(谷): ${round(sum(r.get('cost_usd_offpeak', 0) for r in records), 6)}")
    print(f"  产物: {smoke_path}")
    return smoke


# ---------- 全量: verdict 题 × 2 半 ----------

def run_full(client: LLMClient, halves: dict, qaset: dict, locator: RfidLocator,
             chunks: list[dict], questions: list[dict], note: str | None,
             baselines: dict[str, Baseline], out_dir: Path) -> dict:
    """verdict 题 × 2 半; 同半内连续执行(前缀缓存吃到饱)。

    答案行 schema 与 eval/answers.jsonl 同构(qid/verdict/answer/evidence_rfids/
    all_citations_valid), 另加 half/type/usage/latency_ms/费用等实验字段;
    解析/重试/引用校验全部走 generate.generate_one 原函数。
    """
    corpus_max = max(c["rfid_end"] for c in chunks)
    answers_path = out_dir / ARTIFACT_NAMES["answers"]

    rows, logs = [], []
    with open(answers_path, "w", encoding="utf-8", newline="\n") as f:
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
        "task": "全文臂(剧本全文两半塞入) vs RAG",
        "out_dir": rel_to_project(out_dir),
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
        "baselines": {role: b.brief() for role, b in baselines.items()},
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
    metrics["three_method_table"] = build_three_method_table(metrics, questions,
                                                            baselines, out_dir)
    metrics["budget"] = fulltext_budget(metrics)

    metrics_path = out_dir / ARTIFACT_NAMES["metrics"]
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n== 产物: {answers_path}\n          {metrics_path}")
    a = metrics["arm"]
    for k in ("half_A", "half_B", "pooled"):
        s = a[k]
        print(f"  {k}: label_acc={s['label_acc']}({s['n']}) "
              f"拒答正确={s['refusal_correct_unanswerable']} "
              f"引用条级有效率={s['citations']['entry_valid_rate']} "
              f"编造rfid={s['fabricated_rfids']} 费用=${s['cost']['usd_offpeak']}")
    return metrics


# ---------- 三方法对照表(全文 vs RAG 关 vs RAG 开) ----------

def build_three_method_table(full_metrics: dict, questions: list[dict],
                             baselines: dict[str, Baseline], out_dir: Path) -> dict:
    """质量面: RAG 侧 label_acc/拒答直接读各自 metrics.json 的 label_accuracy
    (口径 = evaluate.eval_label_accuracy, 与全文臂内联公式同式);
    成本/延迟: RAG 侧从 generation_log.json 汇总——注意 attempts 的
    latency/usage 均为**首次网络调用原始值**(sha256 缓存原样保存), 不受该 run
    曾被缓存重跑影响; 每题均值对齐到与全文臂相同的 qid 子集(另附全 90 题口径)。
    """
    qids = {q["qid"] for q in questions}
    table = {}
    for name, b in baselines.items():
        label = b.label_accuracy
        per_q = {}
        all_lats = []
        for q in b.genlog["questions"]:
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
                              for x in sub) / len(sub), 8) if sub else None
        table[name] = {
            "run": rel_to_project(b.run_dir),
            "retriever": b.retriever,
            "top_k": b.top_k,
            "label_acc": label["label_acc"],
            "refusal_correct_unanswerable": label["refusal_correct_unanswerable"],
            "no_overrefusal_answerable": label["no_overrefusal_answerable"],
            "citation_entry_valid_rate": label["citations"]["entry_valid_rate"],
            "per_q_input_tokens_mean_same_qids": mean([x["prompt"] for x in sub]),
            # 费用均值单独算: mean() 的 2 位小数舍入会把 ~5e-4 的单题费用抹成 0
            "per_q_cost_usd_offpeak_mean_same_qids": (
                round(sum(x["cost"] for x in sub) / len(sub), 8) if sub else None),
            "per_q_cost_usd_offpeak_cold_mean_same_qids": cold_mean,
            "per_q_input_tokens_mean_all": mean([x["prompt"] for x in allq]),
            "latency_p50_s": _pct(sorted(all_lats), 0.50),
            "latency_p95_s": _pct(sorted(all_lats), 0.95),
            "latency_mean_s": mean(all_lats),
        }
    # 全文臂三口径(A/B/合并), 键名与 RAG 侧对齐便于并表
    out_rel = rel_to_project(out_dir)
    for k, lab in (("half_A", "fulltext_half_A"), ("half_B", "fulltext_half_B"),
                   ("pooled", "fulltext_pooled")):
        s = full_metrics["arm"][k]
        table[lab] = {
            "run": out_rel,
            "retriever": "fulltext(剧本全文两半塞入)",
            "top_k": None,
            "label_acc": s["label_acc"],
            "refusal_correct_unanswerable": s["refusal_correct_unanswerable"],
            "no_overrefusal_answerable": s["no_overrefusal_answerable"],
            "citation_entry_valid_rate": s["citations"]["entry_valid_rate"],
            "per_q_input_tokens_mean_same_qids": s["tokens"]["prompt_per_q_mean"],
            "per_q_cost_usd_offpeak_mean_same_qids": round(
                s["cost"]["usd_offpeak"] / s["tokens"]["n_calls"], 8),
            "per_q_cost_usd_offpeak_cold_mean_same_qids":
                s["cost"]["per_call_usd_offpeak_cold_mean"],
            "per_q_input_tokens_mean_all": s["tokens"]["prompt_per_q_mean"],
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
    """全文臂预算账(谷/峰): 冒烟+全量的真实网络支出。冒烟 4 调用的 request_hash
    与全量中对应行完全一致(同消息→同 sha256), 全量费用已含冒烟, 不重复计;
    位置消融单独记账在 position_ablation.json, 总账由 summary 汇总。"""
    full_cost = full_metrics["arm"]["pooled"]["cost"]["usd_offpeak"]
    full_peak = full_metrics["arm"]["pooled"]["cost"]["usd_peak"]
    over = round(full_cost - BUDGET_CAP_USD_OFFPEAK, 6)
    return {"cap_usd_offpeak": BUDGET_CAP_USD_OFFPEAK,
            "fulltext_usd_offpeak": full_cost,
            "fulltext_usd_peak": full_peak,
            "over_cap_usd_offpeak": over if over > 0 else 0.0,
            "note": "冒烟调用哈希与全量重复, 全量费用已含冒烟, 直接取全量值"}


# ---------- 位置消融 ----------

def run_position(client: LLMClient, qaset: dict, locator: RfidLocator,
                 chunks: list[dict], cands: list[dict], base: Baseline,
                 out_dir: Path, n_select: int = 10) -> dict:
    """关臂 gold 命中 top-k 的 yesno 题, qid 序前 n_select × 3 位置。

    选择规则(事先写死, 并记录在产物中): yesno 且 gold_chunks ∩ top-k ≠ ∅ 的全部
    候选按 qid 升序取前 n_select; gold 块 = 检索排名最靠前的命中块(规则兜底
    多块情形)。
    重排: gold 块置于目标位, 其余块保持原相对顺序; rank 重编号 1..k(证据N
    显示序 = 列表序, 与基线一致)。
    prompt 组装**逐字复用** generate.py: SYSTEM_PROMPT + build_fewshot(按基线
    generation_log.config.few_shot 取开关: 候选确定性 = 语料序前 3 个非 gold 块)
    + build_user_message(含"按检索融合排名排序"头)——唯一自变量 = gold 块位置。
    """
    top_k = base.top_k
    positions = positions_for(top_k)
    selected = cands[:n_select]

    # 逐字复用的组装件(与关臂基线完全同构; few-shot 开关随基线)
    prefix = build_fewshot(chunks, qaset, locator) if base.few_shot else \
        [{"role": "system", "content": SYSTEM_PROMPT}]
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    corpus_max = max(c["rfid_end"] for c in chunks)

    out_questions = []
    tot_cost = tot_cost_peak = 0.0
    for cand in selected:
        q = next(x for x in qaset["questions"] if x["qid"] == cand["qid"])
        others = [c for c in cand["topk"] if c != cand["gold_block"]]
        entry = {"qid": cand["qid"], "question": cand["question"],
                 "gold_verdict": cand["gold_verdict"],
                 "gold_block": cand["gold_block"],
                 "gold_rank_in_baseline_topk": cand["baseline_rank"],
                 "n_gold_in_topk": cand["n_gold_in_topk"],
                 "baseline_verdict": base.answers[cand["qid"]]["verdict"],
                 "baseline_answer_head": base.answers[cand["qid"]]["answer"][:60],
                 "positions": {}}
        verdicts = []
        for pos_name, pos in positions.items():
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
        entry["verdicts"] = {pos_name: verdicts[i]
                             for i, pos_name in enumerate(positions)}
        entry["stable_across_positions"] = len(set(verdicts)) == 1
        entry["matches_baseline_all"] = all(
            v == entry["baseline_verdict"] for v in verdicts)
        entry["label_correct_by_position"] = {
            pos_name: (entry["positions"][pos_name]["verdict"] == cand["gold_verdict"])
            for pos_name in positions}
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
            for pos in positions},
    }
    out = {
        "task": "位置消融(gold 块 首/中/末), lost in the middle 直接证据",
        "out_dir": rel_to_project(out_dir),
        "selection_rule": (f"关臂({rel_to_project(base.run_dir)}, {base.retriever} "
                          f"k={top_k})检索 gold 命中 top-{top_k} 的 yesno 题按 qid 升序"
                          f"取前 {len(selected)}; 候选共 {len(cands)} 题(落选 qid: "
                          f"{[c['qid'] for c in cands[n_select:]] or '无'})"),
        "baseline_run": rel_to_project(base.run_dir),
        "baseline_retriever": base.retriever,
        "baseline_top_k": top_k,
        "baseline_few_shot": base.few_shot,
        "protocol_reuse": ("generate.py SYSTEM_PROMPT + build_fewshot("
                           f"few_shot={base.few_shot}, 随基线) + build_user_message "
                           "逐字复用(含证据头), 唯一自变量 = gold 块在证据列表中的"
                           f"位置({'/'.join(str(p) for p in positions.values())}), "
                           "其余块保持原相对顺序, rank 重编号"),
        "positions": positions,
        "questions": out_questions,
        "stability": stability,
        "cost": {"usd_offpeak": round(tot_cost, 6), "usd_peak": round(tot_cost_peak, 6)},
    }
    position_path = out_dir / ARTIFACT_NAMES["position"]
    position_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\n== 稳定性: 3 位置 verdict 全一致 {stability['stable_across_positions']}/{n}"
          f"  与基线全一致 {stability['all_match_baseline']}/{n}"
          f"  翻转题 {stability['flipped_qids']}")
    print(f"   分位置 label_acc: {stability['label_acc_by_position']}")
    print(f"   费用(谷) ${out['cost']['usd_offpeak']}  产物: {position_path}")
    return out


# ---------- 实验目录与 manifest ----------

def prepare_out_dir(out_dir: Path, cfg_path: Path) -> None:
    """建目录 + 存配置快照(与 run_all.py 的 runs/<ts>/config_copy.yaml 同构)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out_dir / ARTIFACT_NAMES["config"])


def write_manifest(out_dir: Path, entry: dict, *, cfg_path: Path, chunks: list[dict],
                   qaset: dict, qaset_path: Path, scenes_path: Path) -> None:
    """实验清单: 顶层记配置/语料/评测集指纹, entries 追加本次模式(冒烟→全量→
    位置消融共用一个目录时逐条累积, 每份产物都能追溯到实际基线与输入版本)。"""
    path = out_dir / ARTIFACT_NAMES["manifest"]
    doc = {"created": datetime.now().isoformat(timespec="seconds"),
           "out_dir": rel_to_project(out_dir),
           "config": {"path": rel_to_project(cfg_path),
                      "sha256": sha256_of(cfg_path)},
           "corpus": {"path": rel_to_project(scenes_path),
                      "sha256": sha256_of(scenes_path),
                      "n_chunks": len(chunks),
                      "n_chars": sum(c["n_chars"] for c in chunks)},
           "qaset": {"path": rel_to_project(qaset_path),
                     "sha256": sha256_of(qaset_path),
                     "n_questions": len(qaset["questions"]),
                     "types": dict(Counter(q["type"] for q in qaset["questions"]))},
           "entries": []}
    if path.is_file():   # 同目录已有冒烟记录: 保留历史条目
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            doc["created"] = old.get("created", doc["created"])
            doc["entries"] = list(old.get("entries") or [])
        except (json.JSONDecodeError, OSError):
            pass
    doc["entries"].append(entry)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 主入口 ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="长上下文对照(全文两半 vs RAG) + 位置消融",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="冒烟→全量→位置消融三步共用一个 --out-dir 时, 冒烟判定才会被全量采纳;\n"
               "其它目录里的旧冒烟文件不参与本次判定。")
    ap.add_argument("--config", default="configs/main.yaml")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="冒烟 4 调用(不需基线 run)")
    mode.add_argument("--fulltext", action="store_true",
                      help="全文对照全量(缺省模式; 需 --rag-off-run 与 --rag-on-run)")
    mode.add_argument("--position", action="store_true",
                      help="位置消融 30 调用(需 --rag-off-run)")
    ap.add_argument("--out-dir", default=None,
                    help="实验目录; 缺省 <runs_dir>/fulltext/<YYYYMMDD-HHMMSS>")
    ap.add_argument("--rag-off-run", default=None,
                    help="图谱关臂基线 run 目录(全量/位置消融必填; runs/<ts> 或绝对路径)")
    ap.add_argument("--rag-on-run", default=None,
                    help="图谱开臂基线 run 目录(仅全量必填)")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖同目录同模式的既有产物 / 越过冒烟窗口失败门线")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做预检并打印调用计划(不建 client、不写任何产物)")
    args = ap.parse_args()

    mode_name = "smoke" if args.smoke else "position" if args.position else "fulltext"

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg)
    runs_root = Path(paths.get("runs_dir") or PROJECT_ROOT / "runs")

    chunks = load_chunks(paths["scenes"])
    locator = RfidLocator(chunks)
    qaset = load_qaset(paths["qaset"])
    left, right, split_info = split_halves(chunks)
    halves = {"A": {"label": "前半", "chunks": left,
                    "text": build_half_text(left)},
              "B": {"label": "后半", "chunks": right,
                    "text": build_half_text(right)},
              "split_info": split_info}

    out_dir = resolve_out_dir(args.out_dir, runs_root)
    print(f"== 长上下文对照 == 模式 {mode_name}; 语料 {len(chunks)} chunk"
          f"  切分点 {split_info['split_index']}"
          f"({split_info['left']['chunk_last']} | {split_info['right']['chunk_first']})"
          f"  左 {split_info['left']['n_chars_sum']} 字 / 右 {split_info['right']['n_chars_sum']} 字")
    print(f"  实验目录: {out_dir}"
          + ("" if args.out_dir else "(自动时间戳)"))

    # ---- 预检: 全部问题聚合后一次报错退出, 先于任何 API 调用与产物写入 ----
    log = PreflightLog()
    existing_conflicts(mode_name, out_dir, args.force, log)

    baselines: dict[str, Baseline] = {}
    if mode_name in ("fulltext", "position") and not args.rag_off_run:
        log.err("缺少 --rag-off-run: 全文对照与位置消融都必须显式指定图谱关臂基线 run")
    if mode_name == "fulltext" and not args.rag_on_run:
        log.err("缺少 --rag-on-run: 全文对照的三方法表需要图谱开臂基线 run")
    if mode_name == "position" and args.rag_on_run:
        log.warn("position 模式不使用 --rag-on-run, 已忽略")

    if args.rag_off_run:
        b = load_baseline("rag_off", resolve_run_path(args.rag_off_run, runs_root),
                          need_metrics=(mode_name == "fulltext"),
                          need_answers=(mode_name == "position"), log=log)
        if b:
            baselines["rag_off"] = b
            check_corpus_and_qaset("rag_off", b, chunks=chunks, qaset=qaset,
                                   qaset_path=Path(paths["qaset"]),
                                   scenes_path=Path(paths["scenes"]), log=log)
    if args.rag_on_run and mode_name == "fulltext":
        b = load_baseline("rag_on", resolve_run_path(args.rag_on_run, runs_root),
                          need_metrics=True, need_answers=False, log=log)
        if b:
            baselines["rag_on"] = b
            check_corpus_and_qaset("rag_on", b, chunks=chunks, qaset=qaset,
                                   qaset_path=Path(paths["qaset"]),
                                   scenes_path=Path(paths["scenes"]), log=log)
    if "rag_off" in baselines and "rag_on" in baselines:
        cross_check_baselines(baselines["rag_off"], baselines["rag_on"], cfg, log)
    log.raise_if_errors()
    for w in log.warnings:
        print(f"  [核对提示] {w}")

    # ---- 调用计划(题量/缩减原因/冒烟来源)也在预检期定死 ----
    if mode_name == "smoke":
        plan = CallPlan("smoke", 2, 4, "2 题(yesno 1 + unanswerable 1)× 2 半 = 4 调用")
        questions: list[dict] = []
        note = None
        cands: list[dict] = []
    elif mode_name == "fulltext":
        smoke_path = out_dir / ARTIFACT_NAMES["smoke"]
        smoke = json.loads(smoke_path.read_text(encoding="utf-8")) \
            if smoke_path.is_file() else None
        smoke_reused = smoke is not None
        reduce_q, note = False, None
        if smoke is None:
            print("  [提示] 本实验目录内无冒烟产物, 按全量题量执行"
                  "(其它目录的旧冒烟文件不参与本次判定)")
        else:
            checks = smoke.get("checks") or {}
            if not checks.get("window_ok", False) and not args.force:
                raise SystemExit(
                    f"\n[预检失败] 本目录冒烟显示两半未能全部装入上下文窗口"
                    f"({rel_to_project(smoke_path)})——需先决定降级路径(四分块/停下);"
                    f"\n  确认仍要继续请加 --force。\n→ 未调用任何 API, 未写入任何产物。")
            reduce_q = not all((checks.get("prefix_cache_hit") or {}).values())
            if reduce_q:
                print("  [门线] 冒烟判定无前缀命中 → 题量减半(18 题×2 半)")
        questions, note = select_fulltext_questions(qaset, reduce_q)
        plan = CallPlan("fulltext", len(questions), len(questions) * 2,
                        f"{len(questions)} 题 × 2 半 = {len(questions) * 2} 调用",
                        reduced=reduce_q, reduce_reason=note, smoke_reused=smoke_reused)
        cands = []
    else:
        base_off = baselines["rag_off"]
        if base_off.top_k < 3:
            raise SystemExit(
                f"\n[预检失败] 关臂基线 top_k={base_off.top_k}: 位置消融至少要 3 个证据位"
                f"(首/中/末), 否则位置会重叠。\n→ 未调用任何 API, 未写入任何产物。")
        cands = select_position_candidates(qaset, locator, base_off.retrieved,
                                           base_off.top_k)
        selected = cands[:10]
        missing_ans = [c["qid"] for c in selected if c["qid"] not in (base_off.answers or {})]
        if missing_ans:
            raise SystemExit(
                f"\n[预检失败] 关臂基线 answers.jsonl 缺 {len(missing_ans)} 道入选题的答案"
                f"({missing_ans[:5]})——位置消融要用基线答案作对比。"
                f"\n→ 未调用任何 API, 未写入任何产物。")
        if not selected:
            raise SystemExit(
                f"\n[预检失败] 关臂基线 top-{base_off.top_k} 中没有 gold 命中的 yesno 题,"
                f"位置消融无候选(检查基线检索臂与评测集)。"
                f"\n→ 未调用任何 API, 未写入任何产物。")
        n_pos = len(positions_for(base_off.top_k))
        plan = CallPlan("position", len(selected), len(selected) * n_pos,
                        f"{len(selected)} 题 × {n_pos} 位置 = {len(selected) * n_pos} 调用"
                        f"(候选 {len(cands)} 题)")
        note = None

    print(f"  计划: {plan.description}")
    for role, b in baselines.items():
        print(f"  基线 {role}: {rel_to_project(b.run_dir)}  "
              f"retriever={b.retriever} top_k={b.top_k} few_shot={b.few_shot}")

    if args.dry_run:
        print("\n== 预检通过(--dry-run: 未调用 API, 未创建/写入任何产物) ==")
        print(f"  输出目录: {out_dir}")
        print(f"  计划: {plan.description}")
        return

    # ---- 落盘前先过 Key 探测(LLMClient 只校验 Key, 不联网): 缺 Key 时不留下
    #      只含 config/manifest 的空实验目录 ----
    try:
        client = LLMClient(cfg["generation"]["llm"])
    except RuntimeError as e:
        sys.exit(f"\n[跳过] {e}")

    prepare_out_dir(out_dir, cfg_path)
    write_manifest(out_dir, {
        "mode": plan.mode,
        "cli": sys.argv[1:],
        "llm": (cfg.get("generation") or {}).get("llm"),
        "baselines": {role: b.brief() for role, b in baselines.items()},
        "scope": plan.brief(),
    }, cfg_path=cfg_path, chunks=chunks, qaset=qaset,
        qaset_path=Path(paths["qaset"]), scenes_path=Path(paths["scenes"]))

    if mode_name == "smoke":
        run_smoke(client, halves, qaset, out_dir)
        return
    if mode_name == "position":
        run_position(client, qaset, locator, chunks, cands, baselines["rag_off"], out_dir)
        return
    run_full(client, halves, qaset, locator, chunks, questions, note, baselines, out_dir)


if __name__ == "__main__":
    main()
