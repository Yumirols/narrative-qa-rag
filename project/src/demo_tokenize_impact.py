# -*- coding: utf-8 -*-
"""直观对照:分词方式如何影响 BM25 检索。

三个索引臂:
  raw    = jieba 默认词典(不加自定义词典)
  dict   = jieba + data/user_dict.txt
  bigram = 字符 bigram(不依赖任何词典)

注意:jieba 的自定义词典只能增量加载、进程内不可卸载,所以必须先建 raw 索引,
再触发加载,本脚本要在全新进程中单独运行:
    python src/demo_tokenize_impact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.evaluate import RfidLocator, gold_intervals, load_qaset
from src.index_bm25 import BM25Index, load_chunks
from src.tokenization import bigram_tokenize, jieba_tokenize

SAMPLE = "民间警备队的瑚太朗在风祭市遇到了盖亚的魔物，江坂和西九条也在。"
DEPTH = 10  # 检索深度;超过记为 >10


def show_tokens(label: str, toks: list[str]) -> None:
    print(f"  {label.ljust(8)}({len(toks):>2} tok): {' | '.join(toks)}")


def gold_rank(index, locator, q: dict) -> str:
    ids = [r["chunk_id"] for r in index.search(q["question"], DEPTH)]
    gold = locator.gold_chunks(gold_intervals(q))
    for i, cid in enumerate(ids, 1):
        if cid in gold:
            return str(i)
    return f">{DEPTH}"


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "main.yaml")
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])
    qaset = load_qaset(paths["qaset"])
    locator = RfidLocator(chunks)

    # ---- 1. token 级:同一句话三种切法 ----
    print(f"== 1) 同一句话的三种切分 ==\n  原句: {SAMPLE}")
    show_tokens("raw", jieba_tokenize(SAMPLE))                          # 默认词典(未加载)
    jieba_tokenize(SAMPLE, paths["user_dict"])                          # 此处触发加载
    show_tokens("dict", jieba_tokenize(SAMPLE, paths["user_dict"]))
    show_tokens("bigram", bigram_tokenize(SAMPLE))

    # ---- 2. 检索级:同一问题,三种索引各自把 gold chunk 排到第几 ----
    arms: dict[str, BM25Index] = {}
    arms["raw"] = BM25Index(chunks, lambda t: jieba_tokenize(t))        # 仍默认词典
    arms["dict"] = BM25Index(chunks, lambda t: jieba_tokenize(t, paths["user_dict"]))
    arms["bigram"] = BM25Index(chunks, bigram_tokenize)

    qs = [q for q in qaset["questions"] if gold_intervals(q)]
    print(f"\n== 2) gold chunk 排名对照(数字=首个命中证据块的排名, 共{len(qs)}题) ==")
    head = "qid ".ljust(5) + "问题".ljust(24) + "raw".ljust(6) + "dict".ljust(6) + "bigram"
    print("  " + head)
    for q in qs:
        ranks = {a: gold_rank(idx, locator, q) for a, idx in arms.items()}
        print("  " + q["qid"].ljust(5) + q["question"][:20].ljust(24)
              + ranks["raw"].ljust(6) + ranks["dict"].ljust(6) + ranks["bigram"])

    print("\n== 3) 三臂汇总(种子集) ==")
    ks = cfg["eval"]["k_values"]
    for name, idx in arms.items():
        rrs, r5 = [], []
        for q in qs:
            ids = [r["chunk_id"] for r in idx.search(q["question"], DEPTH)]
            gold = locator.gold_chunks(gold_intervals(q))
            first = next((i + 1 for i, c in enumerate(ids) if c in gold), None)
            rrs.append(1.0 / first if first else 0.0)
            r5.append(len(gold & set(ids[:5])) / len(gold))
        print(f"  {name.ljust(8)} Recall@5={sum(r5)/len(r5):.4f}  MRR@{DEPTH}={sum(rrs)/len(rrs):.4f}")


if __name__ == "__main__":
    main()
