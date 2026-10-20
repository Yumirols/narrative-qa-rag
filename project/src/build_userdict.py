# -*- coding: utf-8 -*-
"""构建 jieba 自定义词典的候选发现脚本(知识框架 §3.3)。

词典 data/user_dict.txt 的选词依据(三源,最终人工审核定稿):
  1. 真值种子: lines.jsonl 的 speaker_norm(73 个去重说话人)——引擎 CD_NAME
     结构直接给出的角色名,零成本、免猜;
  2. 人工清单: 组织/地名/领域术语(盖亚、守护者、民间警备队、风祭市、魔物…),
     依据本脚本输出的语料频次挑选;
  3. OOV 发现: 用默认词典观察切分错误(瑚|太朗、键|之|少女),对照修复。

刻意排除的 speaker_norm: 旁白/？/男人/少年/队长/司机/讲师/博士 等通名角色
(jieba 本就能切,且加入词典无信息量); 大西/樱/源/高砂/数夫 为常见姓氏字,
加入反而可能干扰普通文本切分。

词条格式: "词 词频 词性"(jieba load_userdict 格式)。
词频 = 语料频次、下限 100——给足词频防止 HMM 把专名再切碎。
本脚本只输出统计建议;定稿词典 data/user_dict.txt 随仓库维护(进 zip)。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SCENES = Path(r"E:/dev/NLP/Rewrite_terra/data/scenes.jsonl")
DEFAULT_LINES = Path(r"E:/dev/NLP/Rewrite_terra/data/lines.jsonl")

CANDIDATE_TERMS = [
    # 组织 / 地名 / 领域术语候选,输出频次供人工挑选择收录
    "盖亚", "守护者", "民间警备队", "警备队", "圣女会", "维和部队",
    "风祭市", "风祭", "魔物", "魔女", "键之少女", "先头车辆",
    "神户小鸟", "千里朱音", "中津静流", "加岛",
]


def speaker_freq(lines_path: Path) -> Counter:
    with open(lines_path, encoding="utf-8") as f:
        return Counter(json.loads(l)["speaker_norm"] for l in f)


def term_freq(scenes_path: Path) -> Counter:
    with open(scenes_path, encoding="utf-8") as f:
        joined = "\n".join(json.loads(l)["text"] for l in f)
    return Counter({w: joined.count(w) for w in CANDIDATE_TERMS})


def check_userdict(userdict: Path, sentences: list[str]) -> None:
    """加载定稿词典,对样例句输出切分结果(人工抽查用)。"""
    import jieba

    from src.tokenization import normalize

    jieba.load_userdict(str(userdict))
    for s in sentences:
        print("  " + " | ".join(jieba.lcut(normalize(s))))


def main() -> None:
    ap = argparse.ArgumentParser(description="自定义词典候选发现")
    ap.add_argument("--check", action="store_true", help="检查现有 user_dict.txt 的切分效果")
    ap.add_argument("--lines", type=Path, default=DEFAULT_LINES)
    ap.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    args = ap.parse_args()

    if args.check:
        check_userdict(PROJECT_ROOT / "data" / "user_dict.txt", [
            "瑚太朗是民间警备队的成员。",
            "路易斯是出身巴西的25岁男子。",
            "键之少女叫做篝，盖亚与守护者敌对。",
            "风祭市是盖亚的圣地，魔物在那里出现过。",
            "ＯＫ，２５岁，全角归一化检查。",
        ])
        return

    spk = speaker_freq(args.lines)
    tf = term_freq(args.scenes)

    print("== speaker_norm 频次(候选: 人名, 排除通名角色后收录) ==")
    for name, n in spk.most_common():
        print(f"  {name}\t{n}")
    print("\n== 术语候选频次 ==")
    for w, n in tf.most_common():
        print(f"  {w}\t{n}")
    print("\n按上述统计人工审核后,定稿写入 data/user_dict.txt;用 --check 抽查切分。")


if __name__ == "__main__":
    main()
