# galgame 剧本叙事问答 · 检索层（课程项目方向 A：领域 RAG 问答）

Rewrite Terra 线（约 18.5 万字、1.1 万行台词）叙事问答的检索层与非大模型基线①：
**jieba 自定义词典 + BM25**，配评测脚本骨架（Recall@k / MRR + 答案层标签准确率接口）。
技术路线与知识地图见 `../项目知识框架_剧本问答.md`。

当前状态：**Day 2 完成**（12 天路径 §14）。检索层跑通、双臂对照可复现；Day 3 起
接入 word2vec 平均基线（基线②）与正式评测集 v0。

## 快速开始

```bash
pip install -r requirements.txt
python run_all.py --config configs/main.yaml
```

预期输出（节选，数字为 2026-10-19 种子集实测）：

```
== 分词对照(默认词典 → +自定义词典) ==
  瑚太朗是民间警备队的成员。
    前: 瑚 | 太朗 | 是 | 民间 | 警备队 | 的 | 成员
    后: 瑚太朗 | 是 | 民间警备队 | 的 | 成员  (有变化)
== 建索引 [jieba] ==
  chunks=480  tokens=99,949  avg=208.2  vocab=11,599
== 检索评测(种子集: 评 9 题, 不可回答跳过 1 题) ==
  arm     Recall@3   Recall@5   Recall@10  MRR
  jieba   0.7778     0.7778     0.8889     0.642
  bigram  0.5556     0.6667     0.8889     0.6012
日志已落盘: runs/<timestamp>
```

单模块入口：

```bash
python src/index_bm25.py "路易斯是什么国籍？" --arm jieba -k 5   # 检索 demo
python src/demo_tokenize_impact.py    # 三臂分词对照(原理点①演示,须全新进程单独跑)
python src/build_userdict.py          # 词典候选统计; --check 抽查切分
python src/evaluate.py --arm bigram   # 独立评测单臂
```

## 复现性

- 全流程**零 LLM 调用、零随机性**；同配置重跑 metrics/retrieval_results 逐字节一致。
- 版本锁定见 `requirements.txt`（jieba 0.42.1 / rank_bm25 0.2.2 / PyYAML 6.0.3）。
- 每次运行落盘 `runs/<timestamp>/`：`config_copy.yaml`、`env_info.json`（Python 与
  依赖版本）、`segmentation_demo.txt`（分词前后对照）、`retrieval_results.json`
  （每题 top-k 与命中标记，错误分析直接取材）、`metrics.json`（报告图表唯一数据源）。
- 评测口径：gold 用 **rfid**（全局唯一已读标记）；chunk 的 rfid 区间严格递增不重叠，
  gold→chunk 为确定性二分查表；rfid 落在区间外立即报错（防金标静默错评）。
  gold_evidence_rfids 为空的不可回答题不计入检索指标，留给答案层拒答评测。

## 数据（不进 zip）

`configs/main.yaml` 指向 `../Rewrite_terra/data/`（Day 1 产物）：

| 文件 | 内容 |
|---|---|
| `lines.jsonl` | 11,033 行台词，行级真值结构（rfid/说话人/src_line/koe/bgm） |
| `scenes.jsonl` | 480 个 chunk（三级边界信号真值切分），rfid 区间 1→15084 |
| `choices.json` | 47 个选择肢节点 |

重建方式（README 声明，替代打包剧本）：
1. `siglus-ssu 0.4.0 -x --disam --angou <游戏目录>` 反汇编 `seen08000.dat` → `structure/`；
2. `python ../Rewrite_terra/src/parse_disasm.py` 解析出 `data/`。

来源声明：Rewrite © Key/VisualArt's；汉化文本来自 A·T Field 补丁。仅课程使用，
不随 zip 分发；反汇编 `.txt` 含剧本全文，同为派生数据不进包。

## 目录

```
project/
├── run_all.py               # 一条命令全流程
├── configs/main.yaml        # 路径 / BM25 k1,b / 分词臂 / 评测 k 值
├── src/
│   ├── tokenization.py      # 全角→半角归一化 + jieba/字符bigram 双臂
│   ├── build_userdict.py    # 词典候选发现（speaker 真值 + 语料频次 + OOV）
│   ├── index_bm25.py        # BM25Okapi 建索引/检索（含 CLI demo）
│   └── evaluate.py          # Recall@k / MRR + 标签准确率接口 + rfid 查表
├── data/user_dict.txt       # jieba 自定义词典（36 词条，进 zip）
├── eval/qaset_seed.json     # 种子评测集 10 题（冒烟用，数字不进报告）
├── eval/answers.jsonl       # 答案层输入（Day 4 生成模块产出，schema 见 evaluate.py）
└── runs/                    # 实验日志（不进 zip）
```

## 设计要点（报告素材）

- **自定义词典三源**：73 个 `speaker_norm` 真值种子 + 人工术语清单（盖亚/守护者/
  民间警备队/风祭市/魔物…）+ 默认词典 OOV 观察（`瑚|太朗`、`键|之|少女`、`风|祭市`）；
  词频=语料频次、下限 100 防止 HMM 再切碎。`--check` 抽查切分。
- **检索/展示分离**：索引字段全角→半角（`２５岁→25岁`），证据展示永远用原文。
- **双臂同接口**（原理点①）：jieba+自定义词典 vs 字符 bigram；种子集上 jieba 臂 Recall@3/5 占优，bigram 靠 @10 追平——正式评测集上继续验证。
- **种子集上已现形的 BM25 失败模式**（错误分析素材）：换说法失效（问"国籍"而原文写"出身巴西"，S01 证据块排第 3）；高频主角名 IDF 低（S02 排第 9）；聚合题 top-10 全脱靶（S09——单 chunk 无完整答案，图谱扩展的动机证据）。
- **头号失败案例（报告⑥节素材）**：查询「江坂和瑚太朗是什么关系」——真正的定义句在 c072（rfid 2196–2202，「当你是朋友呢」）却排第 25。三重叠加：
  ①词边界吞噬——jieba 把「利害关系」当整词，c072 内不存在独立 token「关系」 （全库仅 32/480 块含独立「关系」）；
  ②干扰段误导——相邻块 c073（「我们之间的关系到此为止吗」，质疑关系是否破裂的场景而非定义场景）字面含「关系」反而登顶；
  ③江坂/瑚太朗高频低 IDF 无区分度。→ 稠密检索（§4.3）与重排（§4.4）的动机。
  复现：`python src/index_bm25.py "江坂和瑚太朗是什么关系" -k 25 --arm jieba`（`--arm raw` 对照默认词典）。

## 已知数据备注

- 3 个 rfid（3281 / 14785 / 14898）不在任何 chunk 区间内（Day 1 切分时随选择肢边界丢弃的碎片）；金标若命中会被 evaluate 报错指出。
- `千早`/`露西娅` 仅作为结尾场景的说话人【标签】出现，正文叙述从不点名——
  纯文本检索对这类"只在标签里的名字"有盲区；`index.index_speakers: true`
  可把 speakers 元数据并入索引文本（消融开关）。

## 红线自查（提交前）

不进 zip：`runs/`、`../Rewrite_terra/data/`、剧本/反汇编文本、虚拟环境、API Key、
模型权重。`data/user_dict.txt` 与 `eval/qaset_seed.json` 是代码性产物，进包。

## 下一步（Day 3）

word2vec skip-gram 平均基线（基线②，复用 `../02_word_representation/word2vec_skipgram.py`）
→ 正式评测集 v0（30 题，LLM 候选 + 人工全量核对）→ 评测脚本答案层随 Day 4 生成模块接入。
