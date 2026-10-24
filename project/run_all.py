# -*- coding: utf-8 -*-
"""一条命令跑通检索层(任务书红线①:跑不通=不及格)。

用法:
    python run_all.py --config configs/main.yaml

流程: 加载真值 chunk → 分词对照样例(自定义词典前/后) → 按 config 的 retrievers
      列表循环建检索器(工厂见 src/retrievers.py) → 评测集检索评测(Recall@k / MRR,
      总体+分题型) → demo 查询 → 日志落盘
      runs/<timestamp>/{config_copy.yaml, env_info.json, segmentation_demo.txt,
      retrieval_results.json, metrics.json}

本阶段零 LLM 调用、零随机性:同配置重跑,metrics 逐字节一致。
答案层评测(eval/answers.jsonl)Day 4 接入后自动生效,现在缺失则跳过并提示。
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

    env_info = {
        "timestamp": run_dir.name,
        "python": sys.version.split()[0],
        "jieba": pkg_version("jieba"),
        "rank_bm25": pkg_version("rank_bm25"),
        "numpy": pkg_version("numpy"),
        "config": cfg,
    }
    (run_dir / "env_info.json").write_text(
        json.dumps(env_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 1. 语料 ----
    chunks = load_chunks(paths["scenes"])
    n_chars = sum(c["n_chars"] for c in chunks)
    print(f"== 语料 ==\n  {paths['scenes']}\n  chunks={len(chunks)}  总字数={n_chars:,}")

    # ---- 2. 分词对照(必须先于任何带词典的 jieba 调用) ----
    segmentation_demo(paths["user_dict"], run_dir)

    # ---- 3. 按 retrievers 列表建检索器 + 检索评测 ----
    qaset = load_qaset(paths["qaset"])
    locator = RfidLocator(chunks)
    all_metrics, all_results, indexes = {}, {}, {}
    for entry in cfg["retrievers"]:
        name = entry["name"]
        retriever = build_retriever(entry, chunks, paths, index_cfg=cfg["index"])
        indexes[name] = retriever
        st = retriever.stats()
        print(f"\n== 建检索器 [{name}] ({entry['type']}) ==\n  chunks={st['n_chunks']}  "
              f"tokens={st['n_tokens']:,}  avg={st['avg_tokens']}  vocab={st['vocab']:,}")
        metrics, per_q = eval_retrieval(retriever, locator, qaset,
                                        cfg["eval"]["k_values"], cfg["eval"]["mrr_depth"], name)
        all_metrics[name], all_results[name] = metrics, per_q

    # ---- 4. 答案层(Day 4 接入后自动生效) ----
    answers = eval_label_accuracy(qaset, paths["answers"])
    if answers:
        all_metrics["label_accuracy"] = answers
        print(f"\n== 答案层 ==\n  {answers}")
    else:
        print(f"\n== 答案层 ==\n  {paths['answers']} 不存在,跳过(Day 4 生成模块接入后自动生效)")

    # ---- 5. demo 查询(录屏素材) ----
    demo_name = cfg["retrievers"][0]["name"]
    print(f"\n== demo 查询({demo_name} top-3) ==")
    demo_dir = []
    for q in cfg["demo"]["sample_queries"]:
        hits = indexes[demo_name].search(q, 3)
        demo_dir.append({"query": q, "hits": hits})
        print(f"  Q: {q}")
        for r in hits:
            print(f"    #{r['rank']} {r['chunk_id']} score={r['score']}")

    # ---- 6. 落盘 ----
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

    # ---- 7. 摘要表 ----
    ks = cfg["eval"]["k_values"]
    m0 = all_metrics[cfg["retrievers"][0]["name"]]
    qaset_label = Path(paths["qaset"]).stem
    print(f"\n== 检索评测({qaset_label}: 评 {m0['n_evaluated']} 题, "
          f"不可回答跳过 {m0['n_skipped_unanswerable']} 题) ==")
    head = "retriever".ljust(12) + "".join(f"Recall@{k}".ljust(11) for k in ks) + "MRR"
    print("  " + head)
    for name, m in all_metrics.items():
        row = name.ljust(12) + "".join(f"{m['overall'][f'recall@{k}']:<11}" for k in ks)
        print("  " + row + f"{m['overall']['mrr']}")
    for name in all_metrics:
        print(f"  -- 分题型 ({name}) --")
        for t, m in all_metrics[name]["by_type"].items():
            r5 = m.get(f"recall@{ks[1]}", float("nan"))
            print(f"    {TYPE_LABELS.get(t, t).ljust(6)} n={m['n']}  Recall@{ks[1]}={r5}  MRR={m['mrr']}")
    print(f"\n日志已落盘: {run_dir}")


if __name__ == "__main__":
    main()
