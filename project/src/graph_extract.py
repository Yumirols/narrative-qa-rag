# -*- coding: utf-8 -*-
"""事件卡片抽取(阶段四 T10.1/10.2,框架 §6.3b)——每 chunk 一次 LLM 调用。

管线:chunk 全文原文 + 环境上下文(speakers/bgm/bg/选择肢选项文本,框架 §6.3e)
→ 抽取 prompt(系统协议 + 用户消息)→ LLMClient(DeepSeek flash,缓存优先)
→ JSON 三级解析(失败重试一次,仍败 needs_review)→ participants 白名单
校验/归一(别名→规范名;越界名保留但标 suspect)→ 事件条数截断
→ data/graph_cards.jsonl(确定性)+ 抽取日志(usage/延迟/原始响应)。

关键决策(阶段计划 §2 T10/§5):

- **event 必须规范化转述**(prompt 显式约束"标准语汇概括,不照抄原文")——
  卡片文本将建 BM25 索引(T11 卡片路),是弥合换说法 lexical gap 的机制核心;
- **needs_review 降级不崩建图**:两次解析全败的 chunk,卡片 participants 回退
  speakers 真值、events/summary 置空——T10.3 参与者边对这类块仍然可用;
- **participants 白名单 = 实体词典全集**(EntityIndex.canonical):命中别名
  (神户小鸟/Ｋｏｋｏ１０１０)自动归并规范名;None → suspect(统计单列,
  模型幻觉参与者不静默入图);
- **确定性红线**:cards 文件行内只有确定性字段(模型名/模板版本/计数),
  usage/延迟/缓存标记/原始响应全在日志文件——缓存命中重跑 cards 逐字节一致
  (与 generate 的 answers/generation_log 同一分离模式);
- 模板版本常量 EXTRACT_TEMPLATE_VERSION:改模板必升版本(缓存键随 messages
  自动换,版本号用于日志与卡片溯源);
- 冒烟 --limit n = 等距抽样 n 块(覆盖语料首尾;若无选择肢块则用首个
  has_choice 块替换末位,保证 choices 环境分支被测到),产物写 *_smoke.jsonl,
  不污染正式 cards 文件。

用法:
    python src/graph_extract.py --limit 10     # 冒烟(≈$0.005,人工全检 10 卡)
    python src/graph_extract.py                # 全量 483(缓存命中部分零费用)
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
from src.llm import LLMClient, cost_usd

EXTRACT_TEMPLATE_VERSION = "v2"   # v1 冒烟发现: 代词/敬称入 participants、转述实体名漂移 → 加两条约束
EVENT_TYPES = ("对话", "冲突", "行动", "转折", "背景说明")
# 代词屏蔽(v2 冒烟后处理补充,prompt 约束压不住残余): 代词不是实体主张,
# 入图只会产生噪声节点;从 participants 剔除并计数,不进 suspect(suspect=名字主张)。
PRONOUNS = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
            "它们", "咱", "俺", "本人", "自己"}
# 泛指/角色引用剔除(全量 483 后处理迭代,规则全部为词形模式、不触碰实体表):
#   含"的"/"家伙" = 描述短语;以 者/师/员/们 等结尾 = 角色称谓或复数统称;
#   以亲属词结尾 = 家庭关系统称;命中即剔除计数(与代词同一性质,非实体主张)。
import re

GENERIC_ENDSWITH = ("者", "师", "员", "们", "夫妇", "夫妻", "父母", "母亲",
                    "父亲", "家人", "孩子", "母女", "父子", "兄弟", "姐妹", "凡人")
GENERIC_EQUALS = {"主人公", "主角", "某人", "对方", "旁白", "众人", "人们", "男子", "女子"}
# 数量词组开头(两名年幼少女/五位少女 = 群体统称,非个体实体主张)
GENERIC_QUANTIFIER = re.compile(r"^[一两二三四五六七八九十数几][名位个]")


def _is_generic_reference(name: str) -> bool:
    """泛指/角色引用判定(仅作用于白名单未命中的 suspect 候选)。
    判定先于包含归一——"带羽毛形发饰的少女"是描述短语,不该归到 少女 实体。"""
    return ("的" in name or "家伙" in name or name.endswith(GENERIC_ENDSWITH)
            or name in GENERIC_EQUALS or GENERIC_QUANTIFIER.match(name))


def _resolve_fullname_or_containment(name: str, ent_idx: EntityIndex) -> str | None:
    """suspect 名的确定性实体归一(全量 483 新增,顺序敏感):

    1. 全名拼接: X = 姓实体 + 名实体(后缀为实体且剩余部分也是实体,剩余 len>=2)
       → 归为后缀名实体(加岛樱 = 加岛 + 樱 → 樱;先于包含规则,防误归姓);
    2. 唯一包含: X 恰好包含一个 len>=2 的实体名 → 归该实体
       (大西队→大西;守护者组织实际行动队→守护者;加岛派魔物使→加岛)。
    其余(母神会/派罗/天王寺 等真域外提及)保持 suspect——语料真实但词典未收,
    不在本层造实体(词表增长走 user_dict 流程,且需评估对检索臂的影响)。
    """
    names = list(ent_idx.entities)
    for e in sorted(names, key=len, reverse=True):        # 规则 1: 长后缀优先
        if name.endswith(e) and name != e:
            rem = name[:-len(e)]
            if len(rem) >= 2 and rem in ent_idx.entities:
                return e
    hits = {e for e in names if len(e) >= 2 and e != name and e in name}
    return hits.pop() if len(hits) == 1 else None

SYSTEM_PROMPT = """你是叙事剧本的事件抽取器。只依据用户消息中给出的【环境信息】与【剧本文本】工作,严禁使用文本之外的任何知识,严禁推测编造。

输出协议(强制):只输出一个合法的 JSON 对象,不输出任何其他文字、解释或代码块标记。JSON 格式示例:
{"events": [{"participants": ["瑚太朗", "路易斯"], "event": "巡逻车队遭遇自杀式袭击,未造成伤亡", "event_type": "冲突", "leads_to_hint": "瑚太朗开始警惕袭击者的手段"}], "chunk_summary": "夜间巡逻中车队遇袭,瑚太朗与路易斯协同应对。"}

字段规则:
- events:该文本块中发生的事件,最多 3 个,按重要性排序;每个事件包含:
  - participants:参与该事件的实体名(人名/组织名)。只能取【环境信息】"出场说话人"列表内的名字,或剧本文本中明确写出的实体名;不得使用代词(我/你/他/她/它),第一人称叙述者须写作其本名;
  - event:用规范化的标准语汇一句话转述该事件——概括而非照抄原文,优先使用通用表达(如原文口语描述可概括为"自杀式袭击"),因为该转述将用于检索匹配;转述中出现的实体名必须与 participants 中的写法一致,不要另用译名或别名;
  - event_type:从 对话 / 冲突 / 行动 / 转折 / 背景说明 中选一个最贴切的;
  - leads_to_hint:该事件引出或铺垫了什么的简短提示,没有则填空串。
- chunk_summary:用 1~2 句话概括该文本块的剧情。
- 纯景物、氛围描写或系统文本的块,events 填空数组,chunk_summary 照常概括。
宁可少抽,不可编造。"""

RETRY_HINT = ("你上一次的输出不是合法的 JSON 对象,不符合输出协议。请重新作答:"
              "只输出一个符合协议的 JSON 对象(顶层含 events 与 chunk_summary 两个"
              "字段,events 内每项含 participants/event/event_type/leads_to_hint),"
              "不要包含任何其他文字或代码块标记。")


# ---------- Prompt 组装 ----------

def build_user_message(chunk: dict, seq_options: dict[int, list[str]],
                       context_meta: bool = True) -> str:
    """环境上下文 + chunk 全文原文(展示分离红线:原文直入 prompt)。"""
    lines = [f"【剧本文本】(块 {chunk['chunk_id']})"]
    if context_meta:
        speakers = "、".join(chunk["speakers"]) if chunk["speakers"] else "无"
        env = [f"【环境信息】", f"出场说话人:{speakers}",
               f"所在段落 label:{chunk['label']} | BGM:{chunk.get('bgm') or '无'}"
               f" | 背景:{chunk.get('bg') or '无'}"]
        if chunk.get("choices"):
            opts = [f"「{o}」" for seq in chunk["choices"]
                    for o in seq_options.get(seq, [])]
            env.append("选择肢:" + ("、".join(opts) if opts else "有(选项文本略)"))
        else:
            env.append("选择肢:无")
        lines = env + lines
    body = "\n".join(lines)
    return (f"{body}\n{chunk['text']}\n\n"
            f"请依据以上内容,只输出符合系统提示 JSON 协议的一个 JSON 对象。")


# ---------- 归一与校验 ----------

def canonicalize_participants(raw, ent_idx: EntityIndex) -> tuple[list[str], list[str], int, int]:
    """事件级 participants 原始值 → (规范名列表, 越界名列表, 代词剔除数, 泛指剔除数)。
    归一链(顺序敏感): 精确/别名 canonical → 敬称剥离 → 全名拼接/唯一包含 →
    代词/泛指引用剔除 → 其余即 suspect(保留原名,含真域外提及与模型幻觉)。"""
    canon: list[str] = []
    suspects: list[str] = []
    n_pronoun = n_generic = 0
    if not isinstance(raw, list):
        return canon, suspects, n_pronoun, n_generic
    for p in raw:
        if not isinstance(p, str) or not p.strip():
            continue
        name = p.strip()
        if name in PRONOUNS:
            n_pronoun += 1
            continue
        c = ent_idx.canonical(name)
        if c is None:                       # 敬称变体: 剥后缀重试, 命中才收(不硬拆)
            for suf in ("先生", "小姐", "女士", "桑", "様"):
                if name.endswith(suf) and len(name) > len(suf):
                    stripped = ent_idx.canonical(name[:-len(suf)])
                    if stripped is not None:
                        c = stripped
                    break
        if c is None and _is_generic_reference(name):
            n_generic += 1                  # 泛指/角色引用: 同代词剔除(先于包含归一)
            continue
        if c is None:                       # 全名拼接/唯一包含(全量迭代, 见函数注)
            c = _resolve_fullname_or_containment(name, ent_idx)
        if c is not None:
            if c not in canon:
                canon.append(c)
        elif name not in suspects:
            suspects.append(name)
    return canon, suspects, n_pronoun, n_generic


def normalize_card(parsed, chunk: dict, ent_idx: EntityIndex,
                   max_events: int) -> tuple[dict, dict]:
    """LLM 解析出的 dict → 规范卡片。宽松校验:畸形事件条目丢弃计数,
    不让单条漂移弄崩整卡(与 generate 的 malformed 口径一致)。"""
    raw_events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    events, participants, suspects = [], [], []
    dropped = n_pronoun = n_generic = 0
    for ev in raw_events:
        if not isinstance(ev, dict) or not isinstance(ev.get("event"), str) \
                or not ev["event"].strip():
            dropped += 1
            continue
        canon, susp, np_, ng_ = canonicalize_participants(ev.get("participants"), ent_idx)
        n_pronoun += np_
        n_generic += ng_
        for c in canon:
            if c not in participants:
                participants.append(c)
        for s in susp:
            if s not in suspects:
                suspects.append(s)
        et = ev.get("event_type")
        hint = ev.get("leads_to_hint")
        events.append({
            "event": ev["event"].strip(),
            "event_type": et.strip() if isinstance(et, str) and et.strip() else "背景说明",
            "leads_to_hint": hint.strip() if isinstance(hint, str) else "",
        })
    truncated = max(0, len(events) - max_events)
    summary = parsed.get("chunk_summary")
    flags = {"n_events_dropped": dropped, "n_events_truncated": truncated,
             "n_suspects": len(suspects), "n_pronoun_dropped": n_pronoun,
             "n_generic_dropped": n_generic}
    card = {
        "chunk_id": chunk["chunk_id"],
        "rfid": [chunk["rfid_start"], chunk["rfid_end"]],
        "participants": participants + suspects,   # suspects 保留(计划:越界名保留但标 suspect)
        "events": events[:max_events],
        "chunk_summary": summary.strip() if isinstance(summary, str) else "",
        "extraction": {
            "template": EXTRACT_TEMPLATE_VERSION,
            "needs_review": False,
            "suspect_participants": suspects,
            "n_events_dropped": dropped,
            "n_events_truncated": truncated,
        },
    }
    return card, flags


def fallback_card(chunk: dict) -> dict:
    """两次解析全败的降级卡:participants 回退 speakers 真值、事件/概要置空。
    T10.3 参与者边对 needs_review 块仍可用(说话人接地是引擎真值)。"""
    return {
        "chunk_id": chunk["chunk_id"],
        "rfid": [chunk["rfid_start"], chunk["rfid_end"]],
        "participants": list(chunk["speakers"]),
        "events": [],
        "chunk_summary": "",
        "extraction": {
            "template": EXTRACT_TEMPLATE_VERSION,
            "needs_review": True,
            "suspect_participants": [],
            "n_events_dropped": 0,
            "n_events_truncated": 0,
        },
    }


# ---------- 单卡抽取 ----------

def _attempt_log(r: dict) -> dict:
    return {"cache_hit": r["cache_hit"], "parse_path": r["parse_path"],
            "finish_reason": r.get("finish_reason"), "model": r.get("model"),
            "latency_seconds": r.get("latency_seconds"), "usage": r.get("usage"),
            "content": r.get("content"), "request_hash": r["request_hash"]}


def extract_one(client: LLMClient, chunk: dict, ent_idx: EntityIndex,
                seq_options: dict, ext_cfg: dict) -> tuple[dict, dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": build_user_message(chunk, seq_options,
                                               ext_cfg.get("context_meta", True))}]
    r1 = client.chat_json(messages)
    attempts = [_attempt_log(r1)]
    parsed = r1.get("parsed")

    if not isinstance(parsed, dict):   # 重试一次,附协议纠错提示(同 generate 模式)
        r2 = client.chat_json(messages + [
            {"role": "assistant", "content": r1.get("content") or ""},
            {"role": "user", "content": RETRY_HINT}])
        attempts.append(_attempt_log(r2))
        parsed = r2.get("parsed")

    if isinstance(parsed, dict):
        card, flags = normalize_card(parsed, chunk, ent_idx,
                                     ext_cfg.get("max_events_per_chunk", 3))
        card["extraction"]["model"] = attempts[-1].get("model")
    else:
        card, flags = fallback_card(chunk), {"n_events_dropped": 0,
                                             "n_events_truncated": 0,
                                             "n_suspects": 0,
                                             "n_pronoun_dropped": 0,
                                             "n_generic_dropped": 0}
        card["extraction"]["model"] = attempts[-1].get("model")
    log = {"chunk_id": chunk["chunk_id"], "attempts": attempts,
           "prompt_sha": attempts[0]["request_hash"], "flags": flags}
    return card, log


# ---------- 汇总 ----------

def _pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    i = min(int(len(sorted_vals) * p + 0.999999) - 1, len(sorted_vals) - 1)
    return round(sorted_vals[max(i, 0)], 3)


def summarize(cards: list[dict], logs: list[dict]) -> dict:
    usages = [a["usage"] for lg in logs for a in lg["attempts"] if a.get("usage")]
    costs = [cost_usd(u) for u in usages]
    net_lat = sorted(a["latency_seconds"] for lg in logs for a in lg["attempts"]
                     if not a["cache_hit"] and a.get("latency_seconds") is not None)
    n_cache = sum(1 for lg in logs for a in lg["attempts"] if a["cache_hit"])
    degraded = Counter(a["parse_path"] for lg in logs for a in lg["attempts"])
    types = Counter(t for c in cards for e in c["events"] for t in [e["event_type"]])
    n_events = sum(len(c["events"]) for c in cards)
    return {
        "n_cards": len(cards),
        "needs_review": sum(1 for c in cards if c["extraction"]["needs_review"]),
        "n_suspect_cards": sum(1 for c in cards
                               if c["extraction"]["suspect_participants"]),
        "suspect_names": sorted({s for c in cards
                                 for s in c["extraction"]["suspect_participants"]}),
        "events": {"total": n_events, "avg_per_card": round(n_events / len(cards), 2)
                   if cards else 0,
                   "empty_event_cards": sum(1 for c in cards if not c["events"]),
                   "type_counts": dict(types),
                   "dropped": sum(lg["flags"]["n_events_dropped"] for lg in logs),
                   "truncated": sum(lg["flags"]["n_events_truncated"] for lg in logs),
                   "pronoun_dropped": sum(lg["flags"].get("n_pronoun_dropped", 0)
                                          for lg in logs),
                   "generic_dropped": sum(lg["flags"].get("n_generic_dropped", 0)
                                          for lg in logs)},
        "parse": {"fail_first": sum(1 for lg in logs if len(lg["attempts"]) > 1),
                  "parse_path_counts": dict(degraded)},
        "tokens": {"prompt": sum(u.get("prompt_tokens") or 0 for u in usages),
                   "completion": sum(u.get("completion_tokens") or 0 for u in usages),
                   "prompt_cache_hit": sum(u.get("prompt_cache_hit_tokens") or 0
                                           for u in usages),
                   "cost_usd_offpeak": round(sum(c["usd_offpeak"] for c in costs), 6),
                   "cost_usd_peak": round(sum(c["usd_peak"] for c in costs), 6)},
        "latency": {"n_calls": sum(len(lg["attempts"]) for lg in logs),
                    "n_cache_hits": n_cache, "n_network": len(net_lat),
                    "p50_s": _pct(net_lat, 0.50), "p95_s": _pct(net_lat, 0.95)},
    }


# ---------- 抽样与主流程 ----------

def sample_indices(chunks: list[dict], n: int) -> list[int]:
    """等距抽样覆盖语料首尾;若无选择肢块则以首个 has_choice 块替换末位
    (保证 choices 环境分支在冒烟中被测到)。确定性,无随机性。"""
    if n >= len(chunks):
        return list(range(len(chunks)))
    idx = sorted({round(i * (len(chunks) - 1) / (n - 1)) for i in range(n)})
    if not any(chunks[i].get("has_choice") for i in idx):
        first_choice = next(i for i, c in enumerate(chunks) if c.get("has_choice"))
        idx = sorted(set(idx[:-1] + [first_choice]))
    return idx


def run_extraction(cfg: dict, chunks: list[dict], ent_idx: EntityIndex,
                   indices: list[int], out_path: Path, log_path: Path,
                   client: LLMClient, seq_options: dict[int, list[str]],
                   verbose: bool = True) -> dict:
    ext_cfg = cfg.get("graph", {}).get("extract", {})
    cards, logs = [], []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:   # 逐卡落盘,崩溃不丢
        for i, idx in enumerate(indices, 1):
            card, log = extract_one(client, chunks[idx], ent_idx, seq_options, ext_cfg)
            cards.append(card)
            logs.append(log)
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
            f.flush()
            if verbose:
                flags = ""
                if card["extraction"]["needs_review"]:
                    flags = "  [needs_review]"
                elif card["extraction"]["suspect_participants"]:
                    flags = f"  [suspect: {card['extraction']['suspect_participants']}]"
                print(f"  [{i:>3}/{len(indices)}] {card['chunk_id']} "
                      f"events={len(card['events'])} "
                      f"participants={len(card['participants'])}{flags}")
    summary = summarize(cards, logs)
    log_path.write_text(json.dumps(
        {"config": {"template": EXTRACT_TEMPLATE_VERSION, "extract": ext_cfg,
                    "llm": cfg["generation"]["llm"],
                    "n_chunks_selected": len(indices)},
         "summary": summary, "chunks": logs},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="T10.2 事件卡片抽取(每 chunk 一次 LLM 调用)")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--limit", type=int, default=0,
                    help="冒烟:等距抽样 n 块(0=全量 483);产物写 *_smoke 文件")
    ap.add_argument("--out", default=None, help="缺省 config paths.cards(--limit 时加 _smoke 后缀)")
    ap.add_argument("--log", default=None, help="缺省 <out 同名>_extract_log.json")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    if not paths.get("entities") or not Path(paths["entities"]).exists():
        sys.exit("缺少实体索引:先运行 python src/graph_build.py(阶段四 T9 产物)")
    if not paths.get("cards"):
        sys.exit("config 缺 paths.cards(阶段计划 §4)")

    out_path = Path(args.out) if args.out else Path(paths["cards"])
    if args.limit and not args.out:
        out_path = out_path.with_name(out_path.stem + "_smoke.jsonl")
    log_path = Path(args.log) if args.log else \
        out_path.with_name(out_path.stem + "_extract_log.json")

    chunks = load_chunks(paths["scenes"])
    ent_idx = EntityIndex.from_file(paths["entities"])
    # 选择肢选项文本映射(全 52 条,含 5 个 selbtn 续接;chunk.choices 存序号)
    seq_options = {c["seq"]: c["options"] for c in
                   json.loads(Path(paths["choices"]).read_text(encoding="utf-8"))["choices"]}
    indices = sample_indices(chunks, args.limit) if args.limit else list(range(len(chunks)))

    print(f"== T10 事件卡片抽取 ==\n  模板={EXTRACT_TEMPLATE_VERSION}  "
          f"选块 {len(indices)}/{len(chunks)}  模型={cfg['generation']['llm']['model']}  "
          f"产物={out_path.name}")
    try:
        client = LLMClient(cfg["generation"]["llm"])
    except RuntimeError as e:
        sys.exit(f"\n[跳过抽取] {e}")

    summary = run_extraction(cfg, chunks, ent_idx, indices, out_path, log_path,
                             client, seq_options)

    ev, t, lat = summary["events"], summary["tokens"], summary["latency"]
    print(f"\n== 汇总 ==\n  needs_review {summary['needs_review']}/{summary['n_cards']}  "
          f"suspect 卡 {summary['n_suspect_cards']}(名字: {summary['suspect_names'][:8]})")
    print(f"  事件 {ev['total']} 条(均 {ev['avg_per_card']}/卡, 空事件卡 "
          f"{ev['empty_event_cards']}, 丢弃 {ev['dropped']}, 截断 {ev['truncated']})"
          f"  类型分布 {ev['type_counts']}")
    print(f"  解析: 首次失败 {summary['parse']['fail_first']}  路径 {summary['parse']['parse_path_counts']}")
    print(f"  token: 输入 {t['prompt']:,}(前缀命中 {t['prompt_cache_hit']:,}) 输出 "
          f"{t['completion']:,}  费用≈${t['cost_usd_offpeak']}(谷)/${t['cost_usd_peak']}(峰)"
          f"  延迟 P50={lat['p50_s']}s P95={lat['p95_s']}s(网络 {lat['n_network']} 次,"
          f"缓存命中 {lat['n_cache_hits']} 次)")
    print(f"\n产物: {out_path}\n      {log_path}")


if __name__ == "__main__":
    main()
