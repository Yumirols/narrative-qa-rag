# -*- coding: utf-8 -*-
"""一条命令跑通检索层 + 生成层(任务书红线①:跑不通=不及格)。

用法:
    python run_all.py --config configs/main.yaml        # 有 DEEPSEEK_API_KEY: 全链
    python run_all.py --config configs/main.yaml        # 无 Key: 检索层全流程, 生成段跳过

流程: 加载真值 chunk → 分词对照样例(自定义词典前/后) → 图数据段(T12.3: 实体索引
      必建/事件图谱统计, 缺失降级提示) → 按 config 的 retrievers 列表循环建检索器
      (工厂见 src/retrievers.py) → 评测集检索评测(Recall@k / MRR,
      总体+分题型+分层) → 生成阶段(T8: 复用已建检索臂, 产出 answers.jsonl 并回填
      eval/answers.jsonl;Key 缺失整段跳过) → 答案层指标(标签准确率/拒答正确率/
      引用有效率/解析失败率) → demo 查询 → 日志落盘
      runs/<timestamp>/{config_copy.yaml, env_info.json, segmentation_demo.txt,
      graph_build_log.json, retrieval_results.json, metrics.json, answers.jsonl,
      generation_log.json}

检索层零随机性:同配置重跑 metrics 逐字节一致;生成层复现性 = LLM 响应缓存
(同 prompt 命中即逐字节一致,Key 与缓存均在 project/cache/llm_*.json,不进 git)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.evaluate import TYPE_LABELS, RfidLocator, eval_label_accuracy, eval_retrieval, load_qaset
from src.index_bm25 import load_chunks
from src.retrievers import build_retriever
from src.tokenization import jieba_tokenize

# 覆盖自定义词典关键词条 + 全角归一化的样例句(对照素材,报告 §实现细节用)
SEGMENTATION_SAMPLES = [
    "瑚太朗是民间警备队的成员。",
    "路易斯是出身巴西的25岁男子。",
    "键之少女叫做篝，盖亚与守护者敌对。",
    "风祭市是盖亚的圣地，魔物在那里出现过。",
    "ＯＫ，２５岁，全角归一化检查。",
]


def _pkg_ver(dist: str) -> str:
    """importlib.metadata 只读发行版版本号(阶段三); 绝不 import 包本身——
    import torch 会连带拉起整个权重加载栈, 审计只需要版本号。缺失填 "(missing)" 不崩。"""
    try:
        return pkg_version(dist)
    except Exception:  # PackageNotFoundError 等, 包不存在时兜底
        return "(missing)"


def segmentation_demo(userdict: Path, run_dir: Path) -> None:
    """词典加载前/后的切分对照。jieba 词典只能增量加载,顺序不可颠倒。"""
    before = [" | ".join(jieba_tokenize(s)) for s in SEGMENTATION_SAMPLES]
    jieba_tokenize(SEGMENTATION_SAMPLES[0], userdict)  # 触发加载,此后全部带词典
    after = [" | ".join(jieba_tokenize(s)) for s in SEGMENTATION_SAMPLES]

    lines = ["# 分词对照:jieba 默认词典 vs +data/user_dict.txt", ""]
    for s, b, a in zip(SEGMENTATION_SAMPLES, before, after):
        lines += [f"原句: {s}", f"  前: {b}", f"  后: {a}", ""]
    (run_dir / "segmentation_demo.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n== 分词对照(默认词典 → +自定义词典) ==")
    for s, b, a in zip(SEGMENTATION_SAMPLES, before, after):
        flag = "  (有变化)" if b != a else ""
        print(f"  {s}\n    前: {b}\n    后: {a}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="检索层全流程(Day 2)")
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    run_dir = PROJECT_ROOT / cfg["paths"]["runs_dir"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(PROJECT_ROOT / args.config, run_dir / "config_copy.yaml")

    # ---- 1. 语料 ----
    chunks = load_chunks(paths["scenes"])
    n_chars = sum(c["n_chars"] for c in chunks)
    print(f"== 语料 ==\n  {paths['scenes']}\n  chunks={len(chunks)}  总字数={n_chars:,}")

    # ---- 2. 分词对照(必须先于任何带词典的 jieba 调用) ----
    segmentation_demo(paths["user_dict"], run_dir)

    # ---- 2.5 图数据段(T12.3, 仅 config 含 graph 型臂时): 实体索引必建(零 LLM
    #      确定性, 盘上缺失或不一致即覆写); 事件图谱存在则统计入日志, 缺失则记
    #      降级状态(图臂卡片路/扩展路关闭, 实体路+基座 seed 仍可用——无 Key 环境
    #      一条命令照跑的降级口径) ----
    graph_env: dict = {}
    if any(e.get("type") == "graph" for e in cfg["retrievers"]):
        from src.graph_build import build_index_doc, dump_json as dump_entities_json
        ent_path, events_path = paths.get("entities"), paths.get("events")
        if ent_path and paths.get("choices"):
            ent_doc = build_index_doc(chunks, paths["user_dict"], paths["choices"])
            ent_blob = dump_entities_json(ent_doc)
            action = "loaded(与盘上逐字节一致)"
            if not ent_path.exists() or ent_path.read_text(encoding="utf-8") != ent_blob:
                ent_path.parent.mkdir(parents=True, exist_ok=True)
                ent_path.write_text(ent_blob, encoding="utf-8", newline="\n")
                action = "rebuilt(盘上缺失或与确定性重建不一致, 已覆写)"
            graph_env["entities"] = {**ent_doc["meta"], "action": action}
            em = ent_doc["meta"]
            print(f"\n== 图数据段 ==\n  实体索引: {em['n_entities']} 实体(说话人 "
                  f"{em['n_speaker_entities']} + 文本接地 {em['n_text_entities']}, 别名 "
                  f"{em['n_aliases']} 形) 选择肢 {em['n_choices']} —— {action}")
        if events_path and events_path.exists():
            ev_meta = json.loads(events_path.read_text(encoding="utf-8"))["meta"]
            graph_env["events"] = {k: ev_meta[k] for k in (
                "scheme", "n_card_nodes", "n_choice_nodes", "n_edges", "edges_by_type",
                "temporal_skipped_exclusive", "participant_unresolved", "extraction", "stats")}
            x = ev_meta["extraction"]
            print(f"  事件图谱: {ev_meta['n_card_nodes']} 卡 + {ev_meta['n_choice_nodes']} "
                  f"选择肢节点, {ev_meta['n_edges']} 边 {ev_meta['edges_by_type']}"
                  f"\n  抽取溯源: {x['model']} 模板 {x['template']}, {x['n_llm_calls']} 次调用, "
                  f"${x['cost_usd_offpeak']}(谷)")
        else:
            graph_env["events"] = {"present": False}
            print("  事件图谱: 不存在——graph 臂卡片路/扩展路将降级(实体路 + 基座 seed 仍可用);"
                  "\n  重建链路: python src/graph_extract.py(需 DEEPSEEK_API_KEY)"
                  " → python src/graph_assemble.py")

    # ---- 3. 按 retrievers 列表建检索器 + 检索评测 ----
    qaset = load_qaset(paths["qaset"])
    locator = RfidLocator(chunks)
    all_metrics, all_results, indexes = {}, {}, {}
    for entry in cfg["retrievers"]:
        name = entry["name"]
        # built=已建实例表: hybrid 臂按名引用同级条目实例(T6), 复用不重建
        retriever = build_retriever(entry, chunks, paths, index_cfg=cfg["index"], built=indexes)
        indexes[name] = retriever
        st = retriever.stats()
        print(f"\n== 建检索器 [{name}] ({entry['type']}) ==\n  chunks={st['n_chunks']}  "
              f"tokens={st['n_tokens']:,}  avg={st['avg_tokens']}  vocab={st['vocab']}")
        metrics, per_q = eval_retrieval(retriever, locator, qaset,
                                        cfg["eval"]["k_values"], cfg["eval"]["mrr_depth"], name)
        all_metrics[name], all_results[name] = metrics, per_q

    # 图臂降级状态集中提示(T12.3): cards/events 任一缺失时对应路在建索引时已静默
    # 降级(index_graph), 此处把降级面打印 + 落 env_info/graph_build_log
    graph_degraded = [
        {"arm": name, "degraded": content["degraded"]}
        for name, r in indexes.items()
        for fname, content in getattr(r, "extra_logs", {}).items()
        if fname.startswith("graph_config_") and content.get("degraded")]
    if graph_degraded:
        print(f"\n  [图臂降级] {graph_degraded}——降级路不参与融合, 实体路 + 基座 seed 照常;"
              "\n  重建链路见上方图数据段提示")

    # ---- 4. 生成阶段(T8.1: 检索评测之后;DEEPSEEK_API_KEY 缺失整段跳过并提示,
    #      与答案层"answers 不存在则跳过"同一模式——无 Key 环境仍一条命令跑通检索层) ----
    gen_cfg = cfg.get("generation")
    gen_executed, gen_summary = False, None
    if gen_cfg:
        from src.llm import get_api_key
        try:
            get_api_key(gen_cfg["llm"])   # 仅探测 Key 是否就位(不持有不打印)
        except RuntimeError as e:
            print(f"\n== 生成阶段 ==\n  [跳过] {e}")
        else:
            from src.generate import run_generation
            print(f"\n== 生成阶段 ==\n  检索臂={gen_cfg['retriever']}(复用已建实例)  "
                  f"top_k={gen_cfg.get('top_k', 5)}  模型={gen_cfg['llm']['model']}  "
                  f"题集={Path(paths['qaset']).name}({len(qaset['questions'])} 题)")
            gen_summary = run_generation(cfg, chunks, locator, indexes[gen_cfg["retriever"]],
                                         qaset, run_dir / "answers.jsonl",
                                         run_dir / "generation_log.json",
                                         qaset_label=str(paths["qaset"]))
            shutil.copy(run_dir / "answers.jsonl", paths["answers"])   # 回填 eval/answers.jsonl
            gen_executed = True
            t, lat, c = gen_summary["tokens"], gen_summary["latency"], gen_summary["citations"]
            fmt = lambda v: f"{v}s" if v is not None else "-"   # 全缓存命中时无网络延迟
            print(f"  解析失败率(重试后)={gen_summary['parse']['fail_final_rate']}  "
                  f"引用条级有效率={c['entry_valid_rate']}  "
                  f"费用≈${t['cost_usd_offpeak']}(谷)/${t['cost_usd_peak']}(峰)  "
                  f"P50={fmt(lat['p50_s'])} P95={fmt(lat['p95_s'])}  "
                  f"缓存命中 {lat['n_cache_hits']}/{lat['n_calls']} 次")
            print(f"  已回填 {paths['answers']}")

    # ---- 5. 答案层(生成已回填 eval/answers.jsonl 则自动生效;跳过生成时沿用旧文件) ----
    answers = eval_label_accuracy(qaset, paths["answers"])
    if answers:
        all_metrics["label_accuracy"] = answers
        print(f"\n== 答案层 ==\n  {answers}")
    else:
        print(f"\n== 答案层 ==\n  {paths['answers']} 不存在,跳过(Day 4 生成模块接入后自动生效)")

    # ---- 6. demo 查询(录屏素材) ----
    demo_name = cfg["retrievers"][0]["name"]
    print(f"\n== demo 查询({demo_name} top-3) ==")
    demo_dir = []
    for q in cfg["demo"]["sample_queries"]:
        hits = indexes[demo_name].search(q, 3)
        demo_dir.append({"query": q, "hits": hits})
        print(f"  Q: {q}")
        for r in hits:
            print(f"    #{r['rank']} {r['chunk_id']} score={r['score']}")

    # ---- 7. 落盘 ----
    # env_info 挪到检索器建完之后写盘(阶段三): bge 臂的 hf_info 要等检索器建好才拿得到;
    # 无 bge 臂时不加 torch 系字段, 结构与 bm25-only 阶段保持一致
    env_info = {
        "timestamp": run_dir.name,
        "python": sys.version.split()[0],
        "jieba": pkg_version("jieba"),
        "rank_bm25": pkg_version("rank_bm25"),
        "numpy": pkg_version("numpy"),
        "config": cfg,
    }
    bge_info = {}  # 按检索器名分组: {name: hf_info}; 只有 BGEIndex 有该属性, 其余臂跳过
    for name, r in indexes.items():
        hf = getattr(r, "hf_info", None)
        if hf is not None:
            bge_info[name] = hf
    if bge_info:
        env_info["torch"] = _pkg_ver("torch")
        env_info["sentence_transformers"] = _pkg_ver("sentence-transformers")
        env_info["transformers"] = _pkg_ver("transformers")
        env_info["bge"] = bge_info
    # T8.1: 生成阶段环境记录(openai 版本 / 模型与采样配置 / 执行状态)——
    # Key 从不落盘, 只有"是否就位"与配置本身
    if gen_cfg:
        env_info["generation"] = {
            "retriever": gen_cfg["retriever"], "top_k": gen_cfg.get("top_k", 5),
            "llm": gen_cfg["llm"], "openai": _pkg_ver("openai"),
            "executed": gen_executed,
            **({"summary": {k: gen_summary[k] for k in
                            ("parse", "citations", "tokens", "latency", "models_observed")}}
               if gen_executed else {"skip_reason": "DEEPSEEK_API_KEY 未设置"}),
        }
    # T12.3: 图数据环境记录(实体索引重建动作 / 事件图谱统计与抽取溯源 / 图臂降级面),
    # 同内容落 runs/<ts>/graph_build_log.json(报告数字溯源用)
    if graph_env:
        graph_env["arm_degraded"] = graph_degraded
        env_info["graph"] = graph_env
        (run_dir / "graph_build_log.json").write_text(
            json.dumps(graph_env, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "env_info.json").write_text(
        json.dumps(env_info, ensure_ascii=False, indent=2), encoding="utf-8")

    (run_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "retrieval_results.json").write_text(
        json.dumps({"demo": demo_dir, "per_question": all_results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    # 检索器附加日志(如 w2v 训练 loss 曲线/最近邻 sanity/OOV 统计, 见 index_w2v.extra_logs)
    for r in indexes.values():
        for fname, content in getattr(r, "extra_logs", {}).items():
            (run_dir / fname).write_text(
                json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 8. 摘要表 ----
    ks = cfg["eval"]["k_values"]
    m0 = all_metrics[cfg["retrievers"][0]["name"]]
    qaset_label = Path(paths["qaset"]).stem
    print(f"\n== 检索评测({qaset_label}: 评 {m0['n_evaluated']} 题, "
          f"不可回答跳过 {m0['n_skipped_unanswerable']} 题) ==")
    head = "retriever".ljust(12) + "".join(f"Recall@{k}".ljust(11) for k in ks) + "MRR"
    print("  " + head)
    for name, m in all_metrics.items():
        if "overall" not in m:   # label_accuracy 等答案层块无 overall, 不进检索表
            continue
        row = name.ljust(12) + "".join(f"{m['overall'][f'recall@{k}']:<11}" for k in ks)
        print("  " + row + f"{m['overall']['mrr']}")
    for name, m in all_metrics.items():
        if "overall" not in m:
            continue
        print(f"  -- 分题型 ({name}) --")
        for t, mm in m["by_type"].items():
            r5 = mm.get(f"recall@{ks[1]}", float("nan"))
            print(f"    {TYPE_LABELS.get(t, t).ljust(6)} n={mm['n']}  Recall@{ks[1]}={r5}  MRR={mm['mrr']}")
    if "label_accuracy" in all_metrics:
        la = all_metrics["label_accuracy"]
        print(f"\n== 答案层({la['n']} 题带 gold_verdict) ==\n"
              f"  label_acc={la['label_acc']}  拒答正确率(不可回答)={la['refusal_correct_unanswerable']}  "
              f"过度拒答(可回答)={la['no_overrefusal_answerable']}\n"
              f"  引用条级有效率={la['citations']['entry_valid_rate']}"
              f"({la['citations']['intervals_valid']}/{la['citations']['intervals_total']})  "
              f"题级全有效={la['citations']['q_all_valid_rate']}  "
              f"解析失败率={la['parse_failure']['fail_rate']}")
    print(f"\n日志已落盘: {run_dir}")


if __name__ == "__main__":
    main()
