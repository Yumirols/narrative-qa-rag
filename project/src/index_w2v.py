# -*- coding: utf-8 -*-
"""词向量平均检索基线(知识框架 §4.2,任务书基线②)。

课程 `02_word_representation/word2vec_skipgram.py` 的 skip-gram + 负采样实现,
面向本项目改造(课程文件本身不动,数学更新式与课程逐行对应):
  1. 训练输入从 sentences(str 列表 .split())改为**已分词 token 流**——与
     bm25_jieba 臂同一 get_tokenizer("jieba", user_dict),保证方法间可比;
  2. 负采样把课程逐次 rng.choice(p=) 换成预算 cumsum + searchsorted 逆变换
     (同分布;前者每次调用重建 O(V) 累积分布,是规划坑 2 预先点名的慢点);
  3. 词频统计从课程 O(V·N) 的 tokens.count 改为 Counter(语料 10 万 token 必须);
  4. 超参 config 可调,逐 epoch 平均 loss 返回 → runs/<ts>/w2v_train_log.json。

句向量 = chunk 内词向量**朴素平均**(§4.2 的定义,刻意不做 IDF/SIF 加权,
缺陷本身是报告素材),L2 归一化后检索 = 余弦 top-k;接口与 BM25Index 完全
一致,评测代码零改动。预期指标低于 bm25_jieba——负结果即 §4.2 要的发现:
平均丢词序(伤时序题)/静态词义/OOV(查询侧未登录词跳过并计数落日志)。

用法:
  python src/index_w2v.py "路易斯是什么国籍？" [-k 5] [--dim 100] [--epochs 15]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.index_bm25 import load_chunks
from src.tokenization import get_tokenizer

DEFAULT_SANITY_WORDS = ["瑚太朗", "盖亚", "守护者", "魔物", "路易斯"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_skipgram_stream(token_streams, dim: int = 100, window: int = 2, neg_k: int = 4,
                          epochs: int = 15, lr: float = 0.05, seed: int = 0, min_count: int = 1):
    """在 token 流上训练 skip-gram(课程 train_skipgram 的流式改造版)。

    token_streams: 每条 = 一个训练单元的 token 列表(本项目用 chunk 分词流,
    与检索单元同粒度)。返回 (W_in, vocab, vid, loss_curve)。
    固定 seed + 无 shuffle → 同环境重跑结果一致(确定性,规划验收标准 2)。
    """
    rng = np.random.default_rng(seed)

    cnt = Counter(t for stream in token_streams for t in stream)
    vocab = sorted(w for w, c in cnt.items() if c >= min_count)
    vid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    counts = np.array([cnt[w] for w in vocab], dtype=float)
    noise = counts ** 0.75            # 负采样分布: unigram^0.75(与课程一致)
    noise /= noise.sum()
    cum = np.cumsum(noise)            # 逆变换采样的累积分布(预算一次)

    id_streams = [[vid[t] for t in stream if t in vid] for stream in token_streams]
    pair_c: list[int] = []
    pair_o: list[int] = []
    for ids in id_streams:
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pair_c.append(c)
                    pair_o.append(ids[j])
    pair_c = np.asarray(pair_c, dtype=np.int64)
    pair_o = np.asarray(pair_o, dtype=np.int64)

    W_in = (rng.random((V, dim)) - 0.5) / dim
    W_out = np.zeros((V, dim))

    def sample_negs(pos_id: int) -> np.ndarray:
        """逆变换采 K 个负样本;极少数撞上正样本时按课程语义过滤。"""
        for _ in range(10):
            negs = np.minimum(np.searchsorted(cum, rng.random(neg_k)), V - 1)
            if pos_id not in negs:
                return negs
        return negs[negs != pos_id]

    loss_curve = []
    n_pairs = len(pair_c)
    for ep in range(epochs):
        cur_lr = max(lr * (1 - ep / epochs), lr * 0.01)   # 线性衰减,与课程一致
        total_loss = 0.0
        for c, o in zip(pair_c.tolist(), pair_o.tolist()):
            v = W_in[c].copy()                  # 必须拷贝: 下方 W_in[c] 原地更新,视图会拿到新值
            negs = sample_negs(int(o))          # (与 uo 同理,梯度一律用旧值;视图版实测发散 |W|~1e28)
            uo = W_out[o].copy()                # 先拷贝再更新,梯度用旧值
            s_pos = _sigmoid(v @ uo)
            s_neg = _sigmoid(W_out[negs] @ v)
            total_loss += -np.log(s_pos + 1e-9) - np.log(1 - s_neg + 1e-9).sum()

            g_pos = s_pos - 1.0                 # ∂L/∂u_o
            g_neg = s_neg                       # ∂L/∂u_k
            W_in[c] -= cur_lr * (g_pos * uo + W_out[negs].T @ g_neg)
            W_out[o] -= cur_lr * g_pos * v
            np.add.at(W_out, negs, -cur_lr * np.outer(g_neg, v))  # 处理重复下标
        loss_curve.append(float(round(total_loss / n_pairs, 6)))
    return W_in, vocab, vid, loss_curve


class W2VAvgIndex:
    """词向量平均检索器: 接口与 BM25Index 一致(search/stats),工厂直接注入。"""

    def __init__(self, chunks: list[dict], tokenize_fn, dim: int = 100, window: int = 2,
                 neg_k: int = 4, epochs: int = 15, lr: float = 0.05, seed: int = 0,
                 min_count: int = 1, name: str = "w2v_avg",
                 sanity_words: list[str] | None = None):
        self.chunks = chunks
        self.tokenize_fn = tokenize_fn
        self.name = name
        self.params = {"dim": dim, "window": window, "neg_k": neg_k, "epochs": epochs,
                       "lr": lr, "seed": seed, "min_count": min_count}

        t0 = time.time()
        self.streams = [tokenize_fn(c["text"]) for c in chunks]
        self.W_in, self.vocab, self.vid, self.loss_curve = train_skipgram_stream(
            self.streams, dim, window, neg_k, epochs, lr, seed, min_count)
        self.train_seconds = round(time.time() - t0, 2)
        self._n_train_tokens = sum(len(s) for s in self.streams)
        self._oov_tokens = sum(len(s) - sum(1 for t in s if t in self.vid)
                               for s in self.streams)

        # chunk 向量: 朴素平均 + L2 归一化(归一化后内积即余弦)
        self.normed = np.zeros((len(chunks), dim), dtype=float)
        for i, stream in enumerate(self.streams):
            ids = [self.vid[t] for t in stream if t in self.vid]
            if ids:
                v = self.W_in[ids].mean(axis=0)
                n = np.linalg.norm(v)
                if n > 0:
                    self.normed[i] = v / n

        # 训练后 sanity check: 专名最近邻(落训练日志,报告/调试两用)
        self.sanity = self._nearest_bulk(sanity_words or DEFAULT_SANITY_WORDS, 5)

        self._qoov_tokens = 0
        self._n_queries = 0

    # ---- 接口: 与 BM25Index 同构 ----

    def search(self, query: str, k: int = 5) -> list[dict]:
        tokens = self.tokenize_fn(query)
        ids = [self.vid[t] for t in tokens if t in self.vid]
        self._n_queries += 1
        self._qoov_tokens += len(tokens) - len(ids)

        if ids:
            q = self.W_in[ids].mean(axis=0)
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
        else:
            q = None   # 全 OOV 查询: 相似度全 0,按 chunk 顺序返回(接口契约要求 k 条)

        sims = self.normed @ q if q is not None else np.zeros(len(self.chunks))
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))[:k]
        out = []
        for rank, i in enumerate(order, 1):
            c = self.chunks[i]
            out.append({"rank": rank, "idx": i, "chunk_id": c["chunk_id"],
                        "score": round(float(sims[i]), 4),
                        "rfid_start": c["rfid_start"], "rfid_end": c["rfid_end"]})
        return out

    def stats(self) -> dict:
        return {"n_chunks": len(self.chunks), "n_tokens": self._n_train_tokens,
                "avg_tokens": round(self._n_train_tokens / len(self.chunks), 1),
                "vocab": len(self.vocab), "dim": self.params["dim"]}

    # ---- 训练日志: run_all 经 extra_logs 落盘 runs/<ts>/w2v_train_log.json ----

    def _nearest_bulk(self, words: list[str], top: int) -> dict:
        """专名最近邻 sanity check(课程 nearest() 的批量版)。"""
        unit = self.W_in / (np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9)
        out: dict[str, list] = {}
        for w in words:
            i = self.vid.get(w)
            if i is None:
                out[w] = "(OOV)"
                continue
            sims = unit @ unit[i]
            sims[i] = -np.inf
            top_ids = np.argsort(-sims)[:top]
            out[w] = [[self.vocab[j], round(float(sims[j]), 3)] for j in top_ids]
        return out

    @property
    def extra_logs(self) -> dict:
        return {"w2v_train_log.json": {
            "retriever": self.name,
            "params": self.params,
            "n_train_streams": len(self.streams),
            "n_train_tokens": self._n_train_tokens,
            "vocab": len(self.vocab),
            "train_seconds": self.train_seconds,
            "loss_curve": self.loss_curve,
            "sanity_nearest": self.sanity,
            "doc_oov": {"n_oov": self._oov_tokens, "n_tokens": self._n_train_tokens,
                        "rate": round(self._oov_tokens / max(self._n_train_tokens, 1), 6),
                        "note": "min_count=1 时文档侧恒为 0(词表来自训练语料本身)"},
            "query_oov": {"n_oov": self._qoov_tokens, "n_queries": self._n_queries,
                          "rate": round(self._qoov_tokens / max(self._n_queries, 1), 6),
                          "note": "评测期累计;§4.2 缺陷③(OOV)的量化证据"},
        }}


def main() -> None:
    ap = argparse.ArgumentParser(description="词向量平均检索 demo(基线②)")
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    chunks = load_chunks(paths["scenes"])
    index = W2VAvgIndex(chunks, get_tokenizer("jieba", paths.get("user_dict")),
                        dim=args.dim, epochs=args.epochs, seed=args.seed)
    st = index.stats()
    print(f"[w2v_avg] {args.query}\n  dim={st['dim']} vocab={st['vocab']:,} "
          f"训练={index.train_seconds}s loss {index.loss_curve[0]:.3f}→{index.loss_curve[-1]:.3f}")
    for r in index.search(args.query, args.k):
        text = chunks[r["idx"]]["text"][:60].replace("\n", " ")
        print(f"  #{r['rank']} {r['chunk_id']} score={r['score']} "
              f"rfid[{r['rfid_start']},{r['rfid_end']}] | {text}")
    print("  sanity 最近邻:")
    for w, nb in index.sanity.items():
        print(f"    {w} -> {nb}")


if __name__ == "__main__":
    main()
