# -*- coding: utf-8 -*-
"""bge 稠密检索——BGEIndex 检索器 + 查询前缀常量 + HF 缓存重定向 + 前缀方向冒烟。

本模块固化三件影响全局正确性的事:

1. QUERY_PREFIX —— BGE 官方查询侧指令前缀。**只加查询侧, 文档侧不加**:
   该指令在预训练时绑定"把本句当作待检索的查询"的角色; 误放文档侧会静默劣化。
   smoke_prefix_direction() 验证该约束, 干两件事:
   (a) 配线实证: 模型自 models/ 离线加载成功(= 重定向生效)+ 前缀非空操作
       检验(cos(q, P+q) 显著小于 1, 证明前缀没被分词器吞掉、确实改变了表示);
   (b) 方向敏感性记录: 官方(查询+前缀) / 反向(文档+前缀) / 无前缀 三配方在
       评测集全量上的 R@k / MRR 对比。**方向决策不依据该对比**——评测集就是
       测试集, 据它调方向即测试集泄漏; 方向按 BGE 官方用法常量写死,
       敏感性差异只作结论记录(指令为 web 检索调优, 本叙事语料存在域偏移,
       实测收益中性偏负)。

2. apply_hf_cache_redirect —— HF_HOME/HF_HUB_CACHE setdefault 到 paths.models
   (仓库根 models/, 不占 C 盘、不进压缩包)。**必须在 import sentence_transformers /
   huggingface_hub 之前调用**: 缓存路径常量在 import 时固化, 事后改环境变量不生效,
   会静默回落 C 盘。因此本模块对 HF 系一律延迟导入, 且 import 前只做
   os.environ 操作。首次下载配 HF_ENDPOINT=https://hf-mirror.com +
   HF_HUB_DISABLE_XET=1; 模型就位后设 HF_HUB_OFFLINE=1 可全程离线(离线加载成功
   本身就是重定向生效的实证——C 盘缓存不存在可加载的模型)。

3. BGEIndex —— 稠密检索臂: 全库文档向量一次性预编码(normalize_embeddings=True,
   归一化后内积即余弦; **文档侧不加 QUERY_PREFIX**, 官方用法, 方向已由
   smoke_prefix_direction 冒烟验证), search()/stats() 与 BM25Index / W2VAvgIndex
   完全同构, 评测代码零改动。文档向量落 cache/ 的 emb_<model_slug>_<hash8>.npy +
   同名 .meta.json: 命中条件 = 文本哈希/模型名/形状一致, 否则重编码覆写;
   截尾只统计不处置, 只落日志不改切分; revision 读本地 refs/main, 绝不联网。

用法(两模式共用 --model/--device/--batch-size/--offline):
  # 模式一: 前缀方向冒烟(退出码 0=前缀生效且实验完整; 模型已预取时 --offline 全离线)
  python src/index_vector.py --offline                       # small + base 都跑
  python src/index_vector.py --offline --model BAAI/bge-small-zh-v1.5
  python src/index_vector.py --offline --max-questions 20    # 快速冒烟(默认全量 72 题)
  # 模式二: BGEIndex 检索 demo(打印仿 index_w2v.main; 首跑全库编码后写缓存)
  python src/index_vector.py "路易斯是什么国籍？" --offline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.evaluate import RfidLocator, gold_intervals, load_qaset
from src.index_bm25 import load_chunks

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："   # BGE 官方查询侧指令, 文档侧不加

DEFAULT_MODELS = ["BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5"]

DEFAULT_DEMO_MODEL = "BAAI/bge-small-zh-v1.5"   # demo 模式缺省模型(开发迭代只用 small)

CACHE_DIR = PROJECT_ROOT / "cache"   # 文档向量编码缓存目录(project/cache/)

# 配对观察用三元组(句子取自语料/既有 demo 查询, 只作现象记录, 不参与判定):
# gold = 含查询答案的文档, rand = 与查询无关的文档
SMOKE_TRIPLES = [
    {"query": "路易斯是什么国籍？",
     "gold": "路易斯是出身巴西的25岁男子。",
     "rand": "风祭市是盖亚的圣地，魔物在那里出现过。"},
    {"query": "盖亚和守护者是什么关系？",
     "gold": "键之少女叫做篝，盖亚与守护者敌对。",
     "rand": "路易斯是出身巴西的25岁男子。"},
]


def apply_hf_cache_redirect(models_dir: str | Path) -> None:
    """HF 缓存重定向到 models_dir(setdefault: 外部已设则尊重外部)。

    调用时机是正确性的全部: 必须先于任何 sentence_transformers /
    huggingface_hub 的 import(路径常量在 import 时固化)。
    """
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "HF_HUB_CACHE"):
        os.environ.setdefault(var, str(models_dir))


def _load_model(model_name: str, device: str):
    """延迟导入 SentenceTransformer——前置条件: apply_hf_cache_redirect 已调用。"""
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    return model, round(time.time() - t0, 1)


def _rank_metrics(q_emb: np.ndarray, d_emb: np.ndarray, gold_sets: list[set[str]],
                  chunk_ids: list[str], k_values: list[int], mrr_depth: int) -> dict:
    """余弦全库排序 → R@k / MRR——**命中率口径**(仅冒烟/敏感性实验用)。

    口径声明(两套 R@k 不可互比):
      本函数  : 命中率口径——每题任一 gold chunk 进 top-k 记 1;
      evaluate.py(正式): 逐题 gold 占比均值口径——每题 |gold∩topk|/|gold| 再对题求均值,
      多 gold 题(如 structural 层均值 3.68 个 gold)命中一部分只给部分分。
      同一 run 同一臂两口径数字不同(如 bge_small R@3 命中率 0.5417 vs 占比均值 0.3389),
      引用冒烟数字必须注明口径; 正式指标一律用 evaluate.py 口径(与历史数字可比)。
    """
    scores = q_emb @ d_emb.T
    hit_ranks: list[int | None] = []
    for i, row in enumerate(scores):
        order = sorted(range(len(chunk_ids)), key=lambda j: (-row[j], j))
        rank = next((r for r, j in enumerate(order, 1)
                     if chunk_ids[j] in gold_sets[i]), None)
        hit_ranks.append(rank)
    out = {f"recall@{k}": round(sum(r is not None and r <= k for r in hit_ranks)
                                / len(hit_ranks), 4) for k in k_values}
    out["mrr"] = round(sum(1.0 / r for r in hit_ranks if r is not None and r <= mrr_depth)
                       / len(hit_ranks), 4)
    return out


def smoke_prefix_direction(model_name: str, questions: list[dict], locator,
                           chunk_texts: list[str], chunk_ids: list[str],
                           device: str, batch_size: int,
                           k_values: list[int], mrr_depth: int) -> dict:
    """前缀方向冒烟: 前缀放查询侧(官方) vs 放文档侧(反向) vs 不加(对照)。

    三配方在真实评测集 × 全库排序上对比(敏感性记录, 不作方向决策依据):
      A 官方  query=P+q, corpus=d      B 反向  query=q, corpus=P+d
      C 对照  query=q,   corpus=d
    配对观察(描述性)附在前面作现象记录: 两组分数分属不同表示空间, 不可跨空间比大小。
    另算前缀非空操作检验: mean cos(q, P+q) 应显著小于 1。
    """
    model, load_seconds = _load_model(model_name, device)

    t0 = time.time()
    q_pre = model.encode([QUERY_PREFIX + q["question"] for q in questions],
                         batch_size=batch_size, normalize_embeddings=True)
    q_raw = model.encode([q["question"] for q in questions],
                         batch_size=batch_size, normalize_embeddings=True)
    d_raw = model.encode(chunk_texts, batch_size=batch_size,
                         normalize_embeddings=True)                     # 文档侧不加前缀
    d_pre = model.encode([QUERY_PREFIX + t for t in chunk_texts],
                         batch_size=batch_size, normalize_embeddings=True)
    enc_seconds = round(time.time() - t0, 1)
    q_pre, q_raw, d_raw, d_pre = (np.asarray(x, dtype=float)
                                  for x in (q_pre, q_raw, d_raw, d_pre))

    gold_sets = [locator.gold_chunks(gold_intervals(q)) for q in questions]

    recipes = {"A_查询加前缀(官方)": (q_pre, d_raw),
               "B_文档加前缀(反向)": (q_raw, d_pre),
               "C_无前缀(对照)": (q_raw, d_raw)}
    metrics = {name: _rank_metrics(qm, dm, gold_sets, chunk_ids, k_values, mrr_depth)
               for name, (qm, dm) in recipes.items()}

    # 前缀非空操作检验(有约束力的配线判据): 前缀若被吞掉, 此值≈1
    noop_sim = round(float(np.mean(np.sum(q_pre * q_raw, axis=1))), 4)

    # 配对观察: 现象记录, 不参与判定(跨表示空间的绝对分数不可比)
    pairs = []
    for i, tri in enumerate(SMOKE_TRIPLES):
        eg = model.encode([QUERY_PREFIX + tri["query"], tri["query"],
                           tri["gold"], tri["rand"],
                           QUERY_PREFIX + tri["gold"], QUERY_PREFIX + tri["rand"]],
                          batch_size=batch_size, normalize_embeddings=True)
        eg = np.asarray(eg, dtype=float)
        pairs.append({"query": tri["query"],
                      "A_gold": round(float(eg[0] @ eg[2]), 4),
                      "A_rand": round(float(eg[0] @ eg[3]), 4),
                      "B_gold": round(float(eg[1] @ eg[4]), 4),
                      "B_rand": round(float(eg[1] @ eg[5]), 4)})

    return {"model": model_name, "load_seconds": load_seconds,
            "encode_seconds": enc_seconds, "n_questions": len(questions),
            "metrics": metrics, "pairs": pairs, "q_prefix_cos": noop_sim}


# ---- 文档向量编码缓存(small 全库编码约 20~40s, 缓存让开发迭代只付一次) ----

def _sha256_texts(texts: list[str]) -> str:
    """全部 chunk 文本按序 + 显式分隔符("\\0", 正文不可能出现)拼接后 sha256。

    分隔符防止 ["ab","c"] 与 ["a","bc"] 拼出同一字节串; 前 8 位 hex 作缓存文件名,
    完整 64 位落 meta 的 text_hash。
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _read_revision(model_name: str) -> str:
    """从本地 HF 缓存读 <HF_HUB_CACHE>/models--<org>--<name>/refs/main 的 commit sha。

    只碰本地文件、绝不为此联网; 读不到或内容不像 40 位 hex
    一律填 "(unknown)"。必须在 apply_hf_cache_redirect 之后调用(HF_HUB_CACHE 才就位)。
    """
    root = os.environ.get("HF_HUB_CACHE")
    if not root:
        return "(unknown)"
    repo_dir = "models--" + "--".join(model_name.split("/"))
    try:
        rev = (Path(root) / repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return "(unknown)"
    return rev if len(rev) == 40 else "(unknown)"


class BGEIndex:
    """BGE 稠密检索器: 文档向量全库预编码 + 查询侧官方前缀。

    接口与 BM25Index / W2VAvgIndex 完全同构(search/stats), 另有:
      extra_logs -> runs/<ts>/bge_encode_log.json(编码耗时/缓存命中/截尾统计)
      hf_info    -> run_all 接线用(model/revision/device/dim/缓存状态)
    """

    def __init__(self, chunks: list[dict], model_name: str, models_dir: str | Path,
                 device: str = "cpu", batch_size: int = 32, use_cache: bool = True,
                 name: str = "bge"):
        # 顺序: 重定向必须先于一切 HF 系 import(缓存路径常量 import 时固化)
        apply_hf_cache_redirect(models_dir)
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model_name = model_name
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.texts = [c["text"] for c in chunks]

        self.text_hash = _sha256_texts(self.texts)
        self.revision = _read_revision(model_name)

        self.model = SentenceTransformer(model_name, device=device)
        # ST 6.1.0: get_sentence_embedding_dimension 已更名 get_embedding_dimension(旧名 FutureWarning)
        self.dim = int(self.model.get_embedding_dimension())

        self.encode_seconds = 0.0
        self.cache_hit = False
        self.emb = self._load_or_encode(use_cache)

        # 截尾只统计不处置——超长 chunk 由 encode 内部截断, 损失如实落日志
        # (不截断地数 token; ST 6.1.0 属性名是 max_seq_length, 即 encode 实际生效的截尾上限)
        tok_lens = [len(ids) for ids in
                    self.model.tokenizer(self.texts, add_special_tokens=True)["input_ids"]]
        self.n_truncated = sum(1 for n in tok_lens if n > self.model.max_seq_length)

    def _load_or_encode(self, use_cache: bool) -> np.ndarray:
        """全库文档向量: 缓存命中读 npy(毫秒级), 否则编码并覆写缓存。

        文档侧绝不加 QUERY_PREFIX(BGE 官方用法); normalize 后内积即余弦,
        与查询侧同一套分数口径。use_cache=False 时跳过缓存读写。
        """
        # 文件名 emb_<model_slug>_<hash8>: model_slug 把 "/" 换成 "--"(路径里不能有 "/")
        slug = self.model_name.replace("/", "--")
        base = CACHE_DIR / f"emb_{slug}_{self.text_hash[:8]}"
        npy_path, meta_path = Path(f"{base}.npy"), Path(f"{base}.meta.json")

        if use_cache and npy_path.exists() and meta_path.exists():
            t0 = time.time()
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cand = np.load(npy_path)
            except (OSError, ValueError):
                meta, cand = None, None
            load_seconds = round(time.time() - t0, 4)
            # 命中四条件: 双文件在(上) + 文本哈希/模型名一致 + 形状吻合; 任一不满足即重编码覆写
            if (meta is not None and cand is not None
                    and meta.get("text_hash") == self.text_hash
                    and meta.get("model") == self.model_name
                    and cand.shape == (len(self.chunks), self.dim)):
                self.encode_seconds, self.cache_hit = load_seconds, True
                return np.asarray(cand)

        t0 = time.time()
        emb = np.asarray(self.model.encode(self.texts, batch_size=self.batch_size,
                                           normalize_embeddings=True), dtype=np.float32)
        self.encode_seconds = round(time.time() - t0, 2)
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, emb)
            meta = {"model": self.model_name, "revision": self.revision,
                    "n_chunks": len(self.chunks), "text_hash": self.text_hash,
                    "dim": self.dim, "device": self.device,
                    "encode_seconds": self.encode_seconds,
                    "created_at": datetime.now().isoformat(timespec="seconds")}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            # 缓存卫生: 文件名含 hash8, 语料演进后同模型旧哈希的 npy+meta 成为
            # 永不命中的孤儿——覆写缓存时顺手清掉(只清同 slug, 不同模型互不影响)
            for old in CACHE_DIR.glob(f"emb_{slug}_*"):
                if old in (npy_path, meta_path):
                    continue
                try:
                    old.unlink()
                except OSError:
                    pass
        return emb

    # ---- 接口: 与 BM25Index / W2VAvgIndex 同构 ----

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = np.asarray(self.model.encode(
            [QUERY_PREFIX + query], batch_size=self.batch_size,   # 只加查询侧
            normalize_embeddings=True))[0]
        sims = self.emb @ q
        # 同分按 idx 升序稳定排序(与 BM25/W2V 同一契约)
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))[:k]
        out = []
        for rank, i in enumerate(order, 1):
            c = self.chunks[i]
            out.append({"rank": rank, "idx": i, "chunk_id": c["chunk_id"],
                        "score": round(float(sims[i]), 4),
                        "rfid_start": c["rfid_start"], "rfid_end": c["rfid_end"]})
        return out

    def stats(self) -> dict:
        # bge 无词表/分词概念: 为满足 run_all.py:106-107 的打印键(chunks/tokens/
        # avg/vocab)并保持与 W2VAvgIndex.stats 同含 dim 键的风格——n_tokens/
        # avg_tokens 退化为字符数, vocab 填 "<dim>d" 字符串(无词表可数, 整数是编造)。
        # 注意: vocab 是字符串, run_all.py:107 的统计打印不能走
        # 现模板的 "{:,}" 数字格式(会 ValueError), 接入时同步改打印
        n_chars = sum(len(t) for t in self.texts)
        return {"n_chunks": len(self.chunks), "n_tokens": n_chars,
                "avg_tokens": round(n_chars / len(self.chunks), 1),
                "vocab": f"{self.dim}d", "dim": self.dim}

    # ---- 日志接口: run_all 经 extra_logs 逐检索器落盘 ----
    # 文件名必须含检索器名: 五臂循环逐个写盘时, 不含名的
    # "bge_encode_log.json" 会被后建的 bge_base 覆盖掉 bge_small 条目,
    # 两模型各自耗时对比随之丢失。
    @property
    def extra_logs(self) -> dict:
        return {f"bge_encode_log_{self.name}.json": {
            "model": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "batch_size": self.batch_size,
            "n_chunks": len(self.chunks),
            "dim": self.dim,
            "encode_seconds": self.encode_seconds,
            "cache_hit": self.cache_hit,
            "n_truncated": self.n_truncated,
            "truncated_ratio": round(self.n_truncated / max(len(self.chunks), 1), 6),
        }}

    @property
    def hf_info(self) -> dict:
        """模型侧信息(run_all 会 getattr 读它)。"""
        return {"model": self.model_name, "revision": self.revision,
                "device": self.device, "dim": self.dim,
                "encode_seconds": self.encode_seconds, "cache_hit": self.cache_hit}


def _run_demo(args, paths: dict, chunks: list[dict]) -> None:
    """BGEIndex 检索 demo——打印仿 index_w2v.main(stats 行 + top-k 行), 正常完成退出码 0。"""
    model_name = (args.model or [DEFAULT_DEMO_MODEL])[0]
    index = BGEIndex(chunks, model_name, paths["models"], device=args.device,
                     batch_size=args.batch_size)
    st, hf = index.stats(), index.hf_info
    rev = hf["revision"]
    print(f"[{index.name}] {args.query}")
    print(f"  {model_name} rev={rev[:8] if len(rev) == 40 else rev} dim={hf['dim']} "
          f"vocab={st['vocab']} chunks={st['n_chunks']} 编码={hf['encode_seconds']}s "
          f"缓存={'命中' if hf['cache_hit'] else '重编码'} "
          f"截尾={index.n_truncated}/{st['n_chunks']}(仅落日志)")
    for r in index.search(args.query, args.k):
        text = chunks[r["idx"]]["text"][:60].replace("\n", " ")
        print(f"  #{r['rank']} {r['chunk_id']} score={r['score']} "
              f"rfid[{r['rfid_start']},{r['rfid_end']}] | {text}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="bge 稠密检索 demo + 查询前缀方向冒烟")
    ap.add_argument("query", nargs="?", default=None,
                    help="给出则走 BGEIndex 检索 demo; 缺省走前缀方向冒烟")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("-k", type=int, default=5, help="demo 模式的 top-k(冒烟模式忽略)")
    ap.add_argument("--model", action="append",
                    help="冒烟: 可重复, 缺省 small+base 都跑; demo: 取第一个, 缺省 small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-questions", type=int, default=0,
                    help="只取前 n 道可回答题(快速冒烟); 0=全量")
    ap.add_argument("--offline", action="store_true",
                    help="设 HF_HUB_OFFLINE=1; 与缓存重定向一样必须 import 前设")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)

    # ---- HF 环境变量全部就位后, 才允许第一次 import HF 系(顺序要求) ----
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    apply_hf_cache_redirect(paths["models"])

    import torch
    import sentence_transformers
    import huggingface_hub
    from huggingface_hub import constants as hf_constants

    print(f"torch={torch.__version__} sentence-transformers={sentence_transformers.__version__} "
          f"huggingface_hub={huggingface_hub.__version__}")
    print(f"HF_HUB_CACHE(import 后固化值) = {getattr(hf_constants, 'HF_HUB_CACHE', '(unknown)')}")
    print(f"HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE', '(未设)')}\n")

    chunks = load_chunks(paths["scenes"])

    if args.query:   # demo 模式: 打印仿 index_w2v.main, 正常完成退出码 0
        _run_demo(args, paths, chunks)
        return

    # 语料 + 评测集(金标口径与 run_all 完全一致: RfidLocator 查表, 不可回答题跳过)
    locator = RfidLocator(chunks)
    questions = [q for q in load_qaset(paths["qaset"])["questions"]
                 if q["type"] != "unanswerable"]
    if args.max_questions > 0:
        questions = questions[:args.max_questions]
    print(f"检索实证语料: {len(chunks)} chunks × {len(questions)} 道可回答题 "
          f"(金标 RfidLocator 口径)\n")

    k_values, mrr_depth = cfg["eval"]["k_values"], cfg["eval"]["mrr_depth"]
    results = []
    for name in (args.model or DEFAULT_MODELS):
        print(f"== {name} ==")
        r = smoke_prefix_direction(name, questions, locator,
                                   [c["text"] for c in chunks],
                                   [c["chunk_id"] for c in chunks],
                                   device=args.device, batch_size=args.batch_size,
                                   k_values=k_values, mrr_depth=mrr_depth)
        results.append(r)
        print(f"  加载 {r['load_seconds']}s  编码 {r['encode_seconds']}s "
              f"(4 组: 查询×2前缀态 + 全库×2前缀态)")
        print("  [配对观察·描述性] 绝对分数分属不同表示空间, 不可跨空间比大小:")
        for p in r["pairs"]:
            print(f"    {p['query']:<16} A(查询+前缀) gold={p['A_gold']:.4f} rand={p['A_rand']:.4f} "
                  f"| B(文档+前缀) gold={p['B_gold']:.4f} rand={p['B_rand']:.4f}")
        print("  [检索实证·敏感性记录] 三配方全库排序(不作方向决策依据, 防测试集泄漏):")
        print("    " + "recipe".ljust(20) + "".join(f"R@{k}".ljust(10) for k in k_values) + "MRR")
        for rn, m in r["metrics"].items():
            print("    " + rn.ljust(20) + "".join(f"{m[f'recall@{k}']:<10}" for k in k_values)
                  + f"{m['mrr']}")
        noop_ok = r["q_prefix_cos"] < 0.995
        best = max(r["metrics"], key=lambda n: r["metrics"][n]["mrr"])
        delta = r["metrics"]["A_查询加前缀(官方)"]["mrr"] - r["metrics"]["C_无前缀(对照)"]["mrr"]
        print(f"  --> 前缀非空操作: mean cos(q, P+q) = {r['q_prefix_cos']:.4f} "
              f"{'(前缀确实生效)' if noop_ok else '(≈1, 前缀疑似未生效!)'}")
        print(f"  --> 方向敏感性: 最优配方 = {best}; 官方 vs 无前缀 MRR 差 = {delta:+.4f}"
              f"(仅作记录, 不据此调方向)")
        print("  --> 决策: 维持 BGE 官方用法(只加查询侧, 常量写死)——方向不据评测集调整\n")

    final = all(r["q_prefix_cos"] < 0.995 for r in results)
    print(f"冒烟结论: {'PASS' if final else 'FAIL'}"
          f"  (配线判据=前缀非空操作; 方向按官方用法固化, 敏感性数字见上)")
    sys.exit(0 if final else 1)


if __name__ == "__main__":
    main()
