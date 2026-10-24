# -*- coding: utf-8 -*-
"""实验：查询「瑚太朗的父亲叫什么名字」在各检索臂下的命中情况（报告素材取证）。"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", "E:/dev/NLP/models")
os.environ.setdefault("HF_HUB_CACHE", "E:/dev/NLP/models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, "E:/dev/NLP/project")

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.index_bm25 import BM25Index, load_chunks
from src.tokenization import get_tokenizer

cfg = load_config(PROJECT_ROOT / "configs/main.yaml")
paths = resolve_paths(cfg)
chunks = load_chunks(paths["scenes"])
GOLD = {"seen08000_c032", "seen08000_c047"}
QUERIES = ["瑚太朗的父亲叫什么名字？", "数夫是谁？"]


def show(arm, q, ranked, secs):
    tops = " ".join(f"c{r['chunk_id'][11:]}" + ("*" if r["chunk_id"] in GOLD else "")
                    for r in ranked)
    first = next((r["rank"] for r in ranked if r["chunk_id"] in GOLD), None)
    print(f"{arm:20} | first_gold=c032@{first if first else '>-10'} | {secs:5.2f}s | {tops}")


# ---- BM25 三配置 ----
for name, tok, spk in [("bm25_jieba", "jieba", False),
                       ("bm25_bigram", "bigram", False),
                       ("bm25_jieba+spk", "jieba", True)]:
    index = BM25Index(chunks, get_tokenizer(tok, paths.get("user_dict")),
                      k1=1.5, b=0.75, index_speakers=spk)
    for q in QUERIES:
        t0 = time.time()
        ranked = index.search(q, 10)
        show(f"{name}(spk={spk})", q, ranked, time.time() - t0)

# ---- bge-small（独立调用，T4 接入前的预演） ----
print("\n-- bge-small-zh-v1.5 (独立调用; 查询加官方前缀, 归一化内积) --")
import numpy as np
from sentence_transformers import SentenceTransformer

t0 = time.time()
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
doc_emb = model.encode([c["text"] for c in chunks], batch_size=32,
                       normalize_embeddings=True, show_progress_bar=False)
print(f"encode 483 chunks: {time.time()-t0:.1f}s, dim={doc_emb.shape}")

PREFIX = "为这个句子生成表示以用于检索相关文章："
for q in QUERIES:
    t0 = time.time()
    qe = model.encode([PREFIX + q], normalize_embeddings=True)[0]
    scores = doc_emb @ qe
    top = np.argsort(-scores)[:10]
    tops = " ".join(f"c{chunks[i]['chunk_id'][11:]}" + ('*' if chunks[i]['chunk_id'] in GOLD else '')
                    for i in top)
    first = next((int(np.where(top == i)[0][0]) + 1 for i in top
                  if chunks[i]["chunk_id"] in GOLD), None)
    print(f"{'bge_small':22} | {q} | first_gold=c032@{first if first else '>-10'} | {time.time()-t0:5.2f}s | {tops}")

# ---- 报告用四层取证数字 ----
import json
from pathlib import Path
d = Path("E:/dev/NLP/Rewrite_terra/data")
lines = [json.loads(l) for l in open(d / "lines.jsonl", encoding="utf-8")]
print("\n-- 四层取证 --")
for kw in ["数夫", "幸枝", "父亲"]:
    lt = [ln["rfid"] for ln in lines if kw in ln["text"]]
    st = [c["chunk_id"][11:] for c in chunks if kw in c["text"]]
    sn = sorted(int(ln["rfid"]) for ln in lines if ln["speaker_norm"] == kw)
    sp = [c["chunk_id"][11:] for c in chunks if kw in (c.get("speakers") or [])]
    print(f"{kw}: lines.text={len(lt)}行 | scenes.text={len(st)}chunk {st} | speaker_norm={len(sn)}行 {sn[:8]} | speakers字段={len(sp)}chunk {sp}")
mark_lines = sum(1 for ln in lines if "【" in ln["text"])
mark_chunks = sum(1 for c in chunks if "【" in c["text"])
print(f"含【说话人】标注: lines.text {mark_lines} 行 | scenes.text {mark_chunks}/{len(chunks)} chunk")
