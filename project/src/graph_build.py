# -*- coding: utf-8 -*-
"""实体-场景索引与图数据结构——零 LLM,纯确定性。

产物 data/graph_entities.json,三块内容:

1. 实体表(entities):实体 → chunk 倒排,把 483 个孤立 chunk 组织出"实体维"。
   词汇表 = scenes.jsonl speakers 全集(speaker_norm,源头 CD_NAME 真值)
           ∪ user_dict.txt 专名(词性 nr/nt/ns/nz;n 类普通名词如"魔物"不入)。
   接地(grounding)规则——每条都是确定性规则,无人工逐条标注:
   - 说话人实体:chunk 的 speakers 元数据即真值接地;自身名是否再做文本匹配
     **以 user_dict 为界**:具名(在词典,如 瑚太朗/江坂/路易斯)做全角归一+
     小写子串匹配,叙述提及块并入接地(江坂+34/路易斯+16/小鸟+27,抽样核实
     为真提及);**不在词典的泛指角色名(男人/学生/少女/司机…约 25 个)不做
     文本匹配**——这类词在叙述里到处出现,匹配必噪声;
   - 别名表面形(user_dict nr 全名/全角变体/语义别名):同上文本匹配回接;
   - 词典专名中非说话人(盖亚/守护者/风祭市/加岛…):仅文本匹配接地;
   - 单字名(樱/篝/键/源/？)不做自身名文本匹配(篝火/樱花类子串歧义),
     但 ≥2 字的别名形(键之少女)照常匹配。
   别名来源(全部可溯源):user_dict nr 全名含说话人名(神户小鸟→小鸟 等 3 例,
   实测无假阳性)+ 人工核定例外 篝←键之少女(游戏脚本说话人表 NAMAE 中
   鍵の少女＿篝 为篝的称号形态,键之少女 speaker 实体并入篝)+ 全角显示名与
   词典 ASCII 名归一化合并(Ｋｏｋｏ１０１０ ↔ Koko1010)。

2. 选择肢节点(choices):choices.json 的 47 个 selbtn_start 真值交互节点
   (选项文本/rfid/label/src_line/挂载 chunk_id;has_branch 标真分支),零 LLM。

3. EntityIndex:查询侧实体识别器(最长匹配,全角归一+小写)+ 候选 chunk 集
   (多实体取交集,空则并集;单实体返回其全集)——图扩展臂的实体路直接消费。
   主角基数缓解体现在数据侧:每实体带 n_chunks,瑚太朗 313 chunk
   类高基数实体由调用方降权为过滤/加权,不作独立排序。

确定性:无时间戳/无随机性/固定序列化,同输入重建逐字节一致(--check 验收)。

用法:
  python src/graph_build.py                       # 构建 + 落盘 + 统计摘要
  python src/graph_build.py --check               # 重建并与盘上文件逐字节比对
  python src/graph_build.py --query "瑚太朗和路易斯约定了什么？"   # 识别器演示
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

from src.config import PROJECT_ROOT, load_config, resolve_paths
from src.index_bm25 import load_chunks
from src.tokenization import normalize

SCHEME_VERSION = "entities_v1"
DEFAULT_OUT = PROJECT_ROOT / "data" / "graph_entities.json"

# 人工核定的语义别名(唯一一处人工输入,依据游戏脚本说话人表 NAMAE 的
# 语义映射 鍵の少女＿篝→键之少女:键之少女即篝的称号形态)。
# 键之少女 作为 speaker 的 chunk 并入篝,键之少女 不再是独立实体。
ALIAS_OVERRIDES: dict[str, list[str]] = {"篝": ["键之少女"]}

# user_dict 中入实体表的词性(nr 人名/nt 机构/ns 地名/nz 其他专名);
# n(魔物)等普通名词不入——实体路要的是可具名的"实体"。
ENTITY_POS = {"nr", "nt", "ns", "nz"}

TEXT_MIN_LEN = 2   # 参与文本匹配的 CJK 表面形最短长度(单字名子串歧义)


def _norm_form(name: str) -> str:
    """表面形 → 匹配键:全角归一 + 小写(Ｋｏｋｏ１０１０ → koko1010)。"""
    return normalize(name).lower()


def _recognizable(form: str) -> bool:
    """识别词汇表准入:含 CJK,或 ≥2 字符的纯字母数字(koko1010)。
    纯标点形被拒——'？'归一成 '?' 后会命中一切中文问句的句末问号。"""
    return (any("\u4e00" <= ch <= "\u9fff" for ch in form)
            or (len(form) >= 2 and form.isalnum()))


def parse_userdict(path: Path) -> list[tuple[str, str]]:
    """user_dict.txt(词 频次 词性)→ [(词, 词性)] 的专名子集,保持文件序。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] in ENTITY_POS:
            out.append((parts[0], parts[2]))
    return out


def build_entities(chunks: list[dict], userdict: Path) -> list[dict]:
    """实体倒排表构建(纯确定性)。返回按名排序的实体列表。"""
    # ---- 1. speakers 真值接地 ----
    speaker_chunks: dict[str, list[str]] = {}
    for c in chunks:                      # scenes.jsonl 即语料序
        for s in c["speakers"]:
            speaker_chunks.setdefault(s, []).append(c["chunk_id"])

    # ---- 2. speaker 层别名归并(override: 键之少女 → 篝) ----
    merged_into: dict[str, str] = {}      # 被并实体名 → 规范名
    for canon, aliases in ALIAS_OVERRIDES.items():
        for a in aliases:
            if a in speaker_chunks and canon in speaker_chunks:
                merged_into[a] = canon
    canonical_speakers: dict[str, list[str]] = {}
    for s, ids in speaker_chunks.items():
        canon = merged_into.get(s, s)
        canonical_speakers.setdefault(canon, []).extend(ids)
    for ids in canonical_speakers.values():
        ids.sort(key=lambda cid: cid)     # chunk_id 含序号,字典序=语料序

    # ---- 3. 词典专名归类 ----
    dict_entries = parse_userdict(userdict)
    by_norm = {_norm_form(s): s for s in canonical_speakers}   # 归一化合并面
    alias_of: dict[str, str] = {}         # 词典词 → 说话人规范名(其表面形作别名)
    standalone: list[tuple[str, str]] = []
    for w, pos in dict_entries:
        canon = by_norm.get(_norm_form(w))                     # 全角/大小写合并
        if canon is None:
            canon = merged_into.get(w)                         # override 表面形(键之少女→篝)
        if canon is None:
            # 子串别名: nr 全名含说话人名(神户小鸟→小鸟;实测仅 3 例无假阳性)
            if pos == "nr":
                hits = [s for s in canonical_speakers
                        if s != w and len(s) >= 2 and s in w]
                if len(hits) == 1:
                    canon = hits[0]
        if canon is not None:
            alias_of[w] = canon
        else:
            standalone.append((w, pos))

    # ---- 4. 文本接地(别名形与独立专名) ----
    #    语料文本按块预归一(【说话人】标签已烧入正文,说话块天然命中)。
    norm_texts = [_norm_form(c["text"]) for c in chunks]
    text_hit: dict[str, set[str]] = {}    # 规范名/专名 → 文本命中的 chunk_id
    surfaces: list[tuple[str, str]] = []  # (表面形, 归属名)
    for w, canon in alias_of.items():
        surfaces.append((w, canon))
    for w, _pos in standalone:
        surfaces.append((w, w))
    for form, owner in surfaces:
        f = _norm_form(form)
        if len(f) < TEXT_MIN_LEN:         # 单字 CJK 形不参与(歧义),ASCII 形≥2
            continue
        ids = text_hit.setdefault(owner, set())
        for c, t in zip(chunks, norm_texts):
            if f in t:
                ids.add(c["chunk_id"])

    # ---- 5. 组装实体条目 ----
    speaker_pos = {w: pos for w, pos in dict_entries}          # 词典词性优先
    entities = []
    for name in sorted(canonical_speakers):
        s_ids = canonical_speakers[name]
        t_ids = text_hit.get(name, set()) - set(s_ids)
        aliases = sorted(w for w, canon in alias_of.items() if canon == name)
        entities.append({
            "name": name, "pos": speaker_pos.get(name, "nr"), "source": "speakers",
            "aliases": aliases,
            "chunk_ids": s_ids + sorted(t_ids, key=lambda cid: cid),
            "n_chunks": len(s_ids) + len(t_ids),
            "grounding": {"speakers": len(s_ids), "text": len(t_ids)},
        })
    for w, pos in sorted(standalone):
        t_ids = sorted(text_hit.get(w, set()), key=lambda cid: cid)
        if not t_ids:
            continue                      # 语料零提及的专名不建实体(可复现筛除)
        entities.append({
            "name": w, "pos": pos, "source": "text", "aliases": [],
            "chunk_ids": t_ids, "n_chunks": len(t_ids),
            "grounding": {"speakers": 0, "text": len(t_ids)},
        })
    return entities


def build_choices_nodes(choices_path: Path) -> list[dict]:
    """choices.json → selbtn_start 节点(字段原样保留,含 branches/gate_values
    时一并落盘供事件图谱的分支边用),按 rfid 排序(=脚本推进序)。"""
    data = json.loads(choices_path.read_text(encoding="utf-8"))
    nodes = [c for c in data["choices"] if c["kind"] == "selbtn_start"]
    return sorted(nodes, key=lambda c: c["rfid"])


def build_index_doc(chunks: list[dict], userdict: Path, choices_path: Path) -> dict:
    entities = build_entities(chunks, userdict)
    choices = build_choices_nodes(choices_path)
    n_speaker = sum(1 for e in entities if e["source"] == "speakers")
    n_text = len(entities) - n_speaker
    n_alias = sum(len(e["aliases"]) for e in entities)
    return {
        "meta": {
            "scheme": SCHEME_VERSION,
            "scene": chunks[0]["scene"],
            "n_chunks": len(chunks),
            "n_entities": len(entities),
            "n_speaker_entities": n_speaker,
            "n_text_entities": n_text,
            "n_aliases": n_alias,
            "n_choices": len(choices),
            "n_choices_branch": sum(1 for c in choices if c.get("has_branch")),
            "rules": [
                "说话人名文本匹配以 user_dict 为界: 具名做叙述提及匹配回接, 泛指角色名(男人/学生/少女 等不在词典者)不做",
                "文本匹配 = 全角归一+小写后子串匹配, 仅 len>=2 的 CJK 形或 ASCII 形",
                "别名 = user_dict nr 全名含说话人名(3 例) + 语义例外 篝<-键之少女 + 全角显示名归一合并",
                "user_dict 词性 nr/nt/ns/nz 入实体, 普通名词 n 不入; 语料零提及专名不建实体",
            ],
            "sources": ["scenes.jsonl", "user_dict.txt", "choices.json"],
        },
        "entities": entities,
        "choices": choices,
    }


def dump_json(doc: dict) -> str:
    """固定序列化(无时间戳,键序由构造决定,列表显式排序)——逐字节可复现。"""
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


class EntityIndex:
    """查询侧实体识别器 + 候选 chunk 集(图扩展臂的实体路直接消费)。

    recognize(query): 最长匹配扫描归一化查询,返回去重后的命中实体
        [{entity, forms, n_chunks, source}](forms = 该实体被命中的表面形)。
        词汇表剔除纯标点形(？→'?')——否则每个中文问句的句末问号都会命中
        匿名说话人"？"实体并污染交集;数据表仍保留该实体(说话真值)。
        泛指角色名(学生/少女…)在词汇表内,查询误触属已知噪声——
        候选只作过滤/加权,排序交给图扩展臂的基座臂重排。
    candidate_chunks(hits): ≥2 实体取 chunk 集交集(空则并集);
        单实体返回其全集。顺序 = 命中中基数最小实体的语料序(最具体者定序)。
    """

    def __init__(self, doc: dict):
        self.meta = doc["meta"]
        self.choices = doc["choices"]
        self.entities = {e["name"]: e for e in doc["entities"]}
        pairs = {(_norm_form(e["name"]), e["name"]) for e in doc["entities"]}
        pairs |= {(_norm_form(a), e["name"])
                  for e in doc["entities"] for a in e["aliases"]}
        pairs = {p for p in pairs if _recognizable(p[0])}   # 剔除 ？→'?' 类纯标点形
        self._forms = sorted(pairs, key=lambda x: (-len(x[0]), x[0], x[1]))
        self._by_form = dict(pairs)   # 归一形 → 规范名(同形同主,无冲突)

    @classmethod
    def from_file(cls, path: Path) -> "EntityIndex":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def canonical(self, name: str) -> str | None:
        """表面形 → 规范实体名(别名/全角形自动归并);未收录返回 None。
        事件卡 participants 白名单校验的判据:suspect = canonical() 为 None。
        精确名回退兜底:识别词汇表剔除的纯标点形(？)在 participants 归一时
        仍应命中(它不在 _by_form 里,但确为真实说话实体)。"""
        c = self._by_form.get(_norm_form(name))
        if c is not None:
            return c
        return name if name in self.entities else None

    def recognize(self, query: str) -> list[dict]:
        q = _norm_form(query)
        hits: dict[str, list[str]] = {}
        i = 0
        while i < len(q):
            matched = next((f for f, _name in self._forms if q.startswith(f, i)), None)
            if matched:
                name = self._by_form[matched]
                hits.setdefault(name, [])
                if matched not in hits[name]:
                    hits[name].append(matched)
                i += len(matched)
            else:
                i += 1
        out = []
        for name, forms in hits.items():
            e = self.entities[name]
            out.append({"entity": name, "forms": sorted(forms),
                        "n_chunks": e["n_chunks"], "source": e["source"]})
        return out

    def candidate_chunks(self, hits: list[dict]) -> list[str]:
        if not hits:
            return []
        by_ent = {h["entity"]: set(self.entities[h["entity"]]["chunk_ids"]) for h in hits}
        if len(by_ent) == 1:
            sets = list(by_ent.values())
        else:
            inter = set.intersection(*by_ent.values())
            sets = [inter] if inter else list(by_ent.values())
        anchor = min(by_ent.items(), key=lambda kv: len(kv[1]))[0]   # 基数最小者定序
        order = {cid: i for i, cid in enumerate(self.entities[anchor]["chunk_ids"])}
        return sorted(set.union(*sets), key=lambda cid: order.get(cid, len(order)))


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="实体-场景索引构建(零 LLM,确定性)")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", default=None, help="缺省 config paths.entities(无则 data/graph_entities.json)")
    ap.add_argument("--check", action="store_true", help="重建并与盘上文件逐字节比对(确定性验收)")
    ap.add_argument("--query", default=None, help="实体识别演示: 识别 + 候选 chunk 统计")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    paths = resolve_paths(cfg)
    out_path = Path(args.out) if args.out else paths.get("entities", DEFAULT_OUT)
    chunks = load_chunks(paths["scenes"])
    doc = build_index_doc(chunks, paths["user_dict"], paths.get("choices"))
    blob = dump_json(doc)

    if args.query:
        idx = EntityIndex(doc)
        hits = idx.recognize(args.query)
        cands = idx.candidate_chunks(hits)
        print(f"查询: {args.query}")
        for h in hits:
            e = idx.entities[h["entity"]]
            print(f"  实体 {h['entity']}(形={h['forms']} 接地={e['grounding']} "
                  f"n_chunks={h['n_chunks']})")
        print(f"  候选 chunk: {len(cands)} 个"
              + (f"  交集/并集样例: {cands[:5]}" if cands else ""))
        return

    m = doc["meta"]
    if args.check:
        if not out_path.exists():
            sys.exit(f"[check 失败] 盘上不存在 {out_path},先运行构建")
        old = out_path.read_text(encoding="utf-8")
        if old == blob:
            print(f"[check 通过] 重建与盘上文件逐字节一致({len(blob):,} bytes): {out_path}")
        else:
            sys.exit(f"[check 失败] 重建结果与 {out_path} 不一致 "
                     f"(盘上 {len(old):,} bytes vs 重建 {len(blob):,} bytes)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(blob, encoding="utf-8", newline="\n")
    top = sorted(doc["entities"], key=lambda e: -e["n_chunks"])[:5]
    print(f"== 实体-场景索引 ==\n  语料 {m['n_chunks']} chunk → 实体 {m['n_entities']} "
          f"(说话人 {m['n_speaker_entities']} + 文本接地 {m['n_text_entities']}, "
          f"别名 {m['n_aliases']} 形)\n  选择肢节点 {m['n_choices']}(真分支 "
          f"{m['n_choices_branch']})\n  基数 top5: "
          + ", ".join(f"{e['name']}={e['n_chunks']}" for e in top)
          + f"\n产物: {out_path}")


if __name__ == "__main__":
    main()
