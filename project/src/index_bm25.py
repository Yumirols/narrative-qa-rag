# -*- coding: utf-8 -*-
"""BM25 检索层(字面匹配基线)。

rank_bm25.BM25Okapi 的输入是分好词的 token 列表,分词质量直接影响它——
jieba(自定义词典)与字符 bigram 两臂经 get_tokenizer() 注入同一接口。

规模决策: 480 chunks 重建毫秒级,索引不持久化(工具越少越可复现);
chunk 元数据里 rfid 区间全局唯一且已验证互不重叠,随检索结果一并返回,
作为答案输出协议与引用校验的统一锚点。

用法:
  python src/index_bm25.py "路易斯是什么国籍？" [--arm jieba] [-k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.tokenization import get_tokenizer


class BM25Index:
    def __init__(self, chunks: list[dict], tokenize_fn, k1: float = 1.5, b: float = 0.75,
                 index_speakers: bool = False):
        self.chunks = chunks
        self.tokenize_fn = tokenize_fn
        # index_speakers=True 时把说话人元数据并入索引文本(消融开关):
        # 缓解"角色只在【说话人】标签出现、正文不点名"的召回漏(如 男＿ルイス→男人)。
        self.texts = [c["text"] + ((" " + " ".join(c["speakers"])) if index_speakers else "")
                      for c in chunks]
        self.corpus = [tokenize_fn(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.corpus, k1=k1, b=b)

    def search(self, query: str, k: int = 5) -> list[dict]:
        tokens = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        out = []
        for rank, i in enumerate(order, 1):
            c = self.chunks[i]
            out.append({"rank": rank, "idx": i, "chunk_id": c["chunk_id"],
                        "score": round(scores[i], 4),
                        "rfid_start": c["rfid_start"], "rfid_end": c["rfid_end"]})
        return out

    def stats(self) -> dict:
        n_tok = sum(len(t) for t in self.corpus)
        return {"n_chunks": len(self.corpus), "n_tokens": n_tok,
                "avg_tokens": round(n_tok / len(self.corpus), 1),
                "vocab": len({t for toks in self.corpus for t in toks})}


def load_chunks(scenes_path: Path) -> list[dict]:
    p = Path(scenes_path)
    if not p.is_file():
        raise SystemExit(f"[语料] 找不到 {p}\n"
                         f"  按 README §1.2 放好语料, 或改配置里的 paths.scenes 指向实际位置。")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 检索 demo")
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--arm", default="jieba", choices=["jieba", "raw", "bigram"])
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])
    index = BM25Index(chunks, get_tokenizer(args.arm, paths.get("user_dict")),
                      k1=cfg["index"]["k1"], b=cfg["index"]["b"],
                      index_speakers=cfg["index"]["index_speakers"])
    print(f"[{args.arm}] {args.query}")
    for r in index.search(args.query, args.k):
        text = chunks[r["idx"]]["text"][:60].replace("\n", " ")
        print(f"  #{r['rank']} {r['chunk_id']} score={r['score']} "
              f"rfid[{r['rfid_start']},{r['rfid_end']}] | {text}")


if __name__ == "__main__":
    main()
