# 剧本叙事问答：检索增强生成与图谱扩展

面向《Rewrite》Terra 线汉化剧本的叙事问答系统（课程项目方向 A：领域 RAG 问答）。系统把剧本切分为
场景 chunk，由八条检索臂召回证据，交给大模型在证据约束下生成带 rfid 引用的结构化答案，并按检索层
与答案层两组指标评测；另有一条图谱管线（实体索引 + 事件卡 + 图扩展召回）用于比较结构化召回与
纯文本召回的差异。

- 语料：11,033 行剧本记录（台词 4,358 条、旁白 6,675 条），正文（不含【说话人标签】）合计
  184,785 字符，切分为 483 个 chunk（rfid 覆盖 1–15,084）。
- 检索：BM25（jieba / 字符 bigram）、w2v 平均、BGE 稠密（small / base）、RRF 混合、图扩展、
  三路混合，共八臂。
- 生成：DeepSeek `deepseek-flash`，`temperature=0`、思考模式显式关闭、JSON 结构化输出，引用按
  rfid 区间做确定性校验。
- 评测：90 题冻结评测集（可评 72 题、不可回答 18 题）；检索层 Recall@k / MRR；答案层标签准确率、
  拒答正确率、引用有效率、解析失败率；生成类题另有 LLM-as-judge 评分。
- 默认配置 `configs/main.yaml`（由模板 `configs/main.yaml.example` 复制生成）：`hybrid_rrf` + `top_k=5`。

主要结果（详见 §8）：

| 层面 | 结果 |
| --- | --- |
| 检索最优单臂 | `graph_expand` MRR 0.5628（八臂最高）；`hybrid_rrf` R@5 0.5194，`hybrid3_rrf` R@5 0.5465 |
| 答案层（默认配置） | 标签准确率 0.8056；不可回答题拒答 18/18；引用条级有效率 1.0；解析失败 0/90 |
| k × 图谱开关消融 | 6 组中 `hybrid_rrf` + `k=5` 的标签准确率最高（0.8056） |
| 长上下文对照 | 全文塞入（70,548 token/题）标签准确率 0.7361，低于 RAG（3,280 token/题）的 0.8056 |

## 1. 快速开始

### 1.0 配置模板

仓库与提交包内的配置**只提供模板** `configs/main.yaml.example`（消融变体在
`configs/ablation_topk/*.yaml.example`），运行前先复制成实际配置：

```bash
cp configs/main.yaml.example configs/main.yaml
# 消融变体同理（只在需要复现 §8.2 的 k × 图谱开关时才用到）
cp configs/ablation_topk/k3_hybrid_rrf.yaml.example configs/ablation_topk/k3_hybrid_rrf.yaml
```

模板里的 `paths` 全部是**项目内相对路径**（`data/corpus/`、`models`），不含任何本机或上级目录
路径，复制后按 §1.2 放好语料与模型即可直接运行。实际配置 `configs/main.yaml` 与
`configs/ablation_topk/*.yaml` 已在 `.gitignore` 中（开发机上那份可指向项目外目录，不进提交包）。

路径解析规则见 `src/config.py:resolve_paths`：`paths` 中的相对路径一律以**项目根**为基准，与
当前工作目录无关；也可直接写绝对路径。语料或模型放在其他位置时，只需改 `paths.scenes`、
`paths.choices`、`paths.models` 三行，不必改动源码。若漏了复制配置或没放语料，脚本会在读文件前
直接提示缺哪个文件、该改哪一行——不会抛原始的 `FileNotFoundError`。

### 1.1 依赖

提交源码以本目录为项目根目录，解压后直接进入包含 `run_all.py`、`requirements.txt` 和 `src/` 的目录。
下文所有命令均从该目录执行，不依赖开发时的上级 `NLP/` 仓库，也不需要再执行 `cd project`。
命令中的 `export` 使用 Bash / Git Bash 语法。建议使用 Python 3.11 的独立虚拟环境。

```bash
# 在新虚拟环境中先安装 CPU torch，再安装其余依赖
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

最后一条命令应显示 CPU 构建版本，且 `torch.version.cuda` 为 `None`。如果已有环境安装了非 CPU
构建，相同版本约束可能不会触发替换；建议改用新虚拟环境执行上述步骤。

`requirements.txt` 锁定的版本：torch 2.14.1、sentence-transformers 6.1.0、transformers 5.20.2、
jieba 0.42.1、rank_bm25 0.2.2、openai 3.37.0、networkx 3.6.1、matplotlib 3.11.2、
huggingface_hub 1.33.0、PyYAML 6.0.3（numpy 要求 ≥1.24）。实测环境为 Python 3.11.9 + numpy 2.4.6
+ torch 2.14.1+cpu，全部版本号在每次运行的 `env_info.json` 中留档。

### 1.2 数据与模型

1. **语料**：可从[网盘链接](https://pan.baidu.com/s/1pq8LwU0pUFFkAakMf9vXIw?pwd=8mvt)
   （提取码：`8mvt`）的“数据集”文件夹直接获取 `lines.jsonl`、`scenes.jsonl`、`choices.json`
   与 `speaker_map.tsv`，仅提供复现实验。直接运行实验无需重新反汇编。
   将前三份文件放入项目根目录的 `data/corpus/` 即可——`configs/main.yaml.example` 的
   `paths.scenes` / `paths.choices` 已经指向该位置，无需再改（该目录内的 `README.md` 列出了
   应放的文件与 sha256 校验命令）。若改放到别处，只需同步修改这两行
   （以及 `paths.models`，见下一条）；使用 §8.2 的其他配置时同理。`speaker_map.tsv` 仅在 §3.4
   重建语料时使用，文件校验值见 §3.6。
2. **模型**：`BAAI/bge-small-zh-v1.5`（512 维）与 `BAAI/bge-base-zh-v1.5`（768 维）。缓存根由
   `paths.models` 指定，模板默认 `models`（即项目根内的 `models/`，含两个模型的完整 HF 缓存约
   900MB，不随提交包分发）。首次运行自动下载到该位置；若沿用开发机已有的缓存，把该行改成缓存
   实际所在目录即可。首次运行需联网下载一次：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HUB_DISABLE_XET=1
   ```

   之后可设 `HF_HUB_OFFLINE=1` 全程离线。缓存重定向由 `src/index_vector.apply_hf_cache_redirect`
   在 import `sentence_transformers` 之前完成（缓存路径常量在 import 时固化，重定向必须前置）。
3. **API Key**（仅生成层需要）：`export DEEPSEEK_API_KEY=sk-…`，或复制 `.env.example` 为
   `.env` 并填写。Key 只从环境变量 / `.env` 读取，不写入代码、配置或任何运行产物。

### 1.3 一条命令

```bash
cp configs/main.yaml.example configs/main.yaml   # 仅首次；已存在则跳过
python run_all.py --config configs/main.yaml
```

流程：加载 chunk → 分词对照（默认词典 vs 自定义词典）→ 图数据段（实体索引确定性重建或校验，
事件图谱统计）→ 按 `retrievers` 列表建八臂并做检索评测 → 生成（90 题）→ 答案层指标 → demo 查询 →
落盘 `runs/<timestamp>/`。

不设 Key 时生成段整段跳过（打印 `[跳过]`），检索层与图数据段不受影响，答案层沿用已有的
`eval/answers.jsonl`。`data/graph_cards.jsonl` 或 `graph_events.json` 缺失时，图臂对应路
自动降级（卡片路关闭、扩展路仅保留同 label 邻居、实体路与基座排名不变），控制台与
`env_info.json` 会打印降级面——此时 `graph_expand` / `hybrid3_rrf` 两行指标会不同于 §8.1，
其余六臂不受影响。按 §5.1 重建图数据后即可复现完整八臂。

### 1.4 预期输出

以下为节选，列对齐经整理，数值原样保留（路径、耗时与费用随机器和时段变化；检索与答案层指标为
定值，口径见 §7）：

```text
== 语料 ==
  data/corpus/scenes.jsonl
  chunks=483  总字数=184,785

== 分词对照(默认词典 → +自定义词典) ==
  瑚太朗是民间警备队的成员。
    前: 瑚 | 太朗 | 是 | 民间 | 警备队 | 的 | 成员
    后: 瑚太朗 | 是 | 民间警备队 | 的 | 成员
  键之少女叫做篝，盖亚与守护者敌对。
    前: 键 | 之 | 少女 | 叫做 | 篝 | 盖亚 | 与 | 守护者 | 敌对
    后: 键之少女 | 叫做 | 篝 | 盖亚 | 与 | 守护者 | 敌对

== 图数据段 ==
  实体索引: 80 实体 ... 选择肢 47 —— loaded(与盘上逐字节一致)
  事件图谱: 483 卡 + 47 选择肢节点, 1634 边 ...

== 检索评测(qaset_v0: 评 72 题, 不可回答跳过 18 题) ==
  retriever   Recall@3   Recall@5   Recall@10  MRR
  bm25_jieba  0.3377     0.4108     0.4963     0.4942
  bm25_bigram 0.4171     0.4720     0.5579     0.5381
  w2v_avg     0.1368     0.1693     0.2806     0.1986
  bge_small   0.3389     0.3833     0.5435     0.4490
  bge_base    0.3704     0.4476     0.5809     0.5125
  hybrid_rrf  0.4391     0.5194     0.6553     0.5461
  graph_expand 0.4204    0.5127     0.5966     0.5628
  hybrid3_rrf 0.4490     0.5465     0.6236     0.5496

== 答案层(36 题带 gold_verdict) ==
  label_acc=0.8056  拒答正确率(不可回答)=1.0  过度拒答(可回答)=0.6111
  引用条级有效率=1.0(83/83)  题级全有效=0.5556  解析失败率=0.0

日志已落盘: runs/<timestamp>
```

设置有 Key 时，检索评测与答案层之间还会多出 `== 生成阶段 ==` 一节，打印解析失败率、引用条级
有效率、费用（谷/峰）、P50/P95 延迟与缓存命中数；无 Key 时该节替换为一行 `[跳过] 缺少
DEEPSEEK_API_KEY…`，答案层改用随包的 `eval/answers.jsonl` 复评（数值与有 Key 时一致，因为生成层
的输出由响应缓存锁定）。若该文件也不存在，答案层一并跳过，此时复现的是检索层八臂指标（§8.1）。

本机 CPU 实测：缓存就绪时全流程约 4–5 分钟（w2v 每轮重训约 215–250 s，为主要耗时）。首次运行另需
下载模型并做 bge 全库编码（CPU 上 small 约 20–40 s、base 约 4–5 分钟）。`w2v_avg` 每次运行都会
重新训练，这是刻意保留的设计——loss 曲线作为训练可复现的指纹；单独跑其他模块（§12）不会触发重训。

### 1.5 运行产物

| 文件 | 内容 |
| --- | --- |
| `runs/<timestamp>/metrics.json` | **表格数字唯一来源**：八臂 Recall@k / MRR、`by_type` 分题型、`by_layer` 分层，以及答案层 `label_accuracy` 块 |
| `runs/<timestamp>/retrieval_results.json` | 每题各臂 top-k 结果与命中标记、demo 查询 |
| `runs/<timestamp>/answers.jsonl` | 生成层 90 行结构化答案（不含时间戳等非确定字段） |
| `runs/<timestamp>/generation_log.json` | 每题 prompt 哈希、usage、延迟、原始响应、引用校验明细 |
| `runs/<timestamp>/env_info.json` | Python 与全部依赖版本、模型 revision、完整配置、生成与图数据段环境 |
| `runs/<timestamp>/config_copy.yaml` | 本次运行配置快照 |
| `runs/<timestamp>/segmentation_demo.txt` | 分词对照（词典前/后） |
| `runs/<timestamp>/graph_build_log.json` | 实体索引重建动作、事件图谱统计、图臂降级面 |
| `runs/<timestamp>/w2v_train_log.json` | w2v 逐 epoch loss、词表、OOV 统计 |
| `runs/<timestamp>/bge_encode_log_<臂名>.json` | 各 bge 臂编码耗时、缓存命中、截尾统计 |
| `runs/<timestamp>/hybrid_config_<臂名>.json`、`graph_config_graph_expand.json` | 融合/图臂参数与门槛计数 |

## 2. 系统架构

```text
data/corpus/{lines,scenes,choices}.json[l]   剧本行级真值（§3；实际位置由 configs 的 paths.* 指定）
          │
          ├─ 检索层
          │    tokenization.py ─┬─ index_bm25.py    BM25：jieba / 字符 bigram
          │                     ├─ index_w2v.py     skip-gram 词向量平均
          │                     └─ index_vector.py  BGE 稠密（small / base）
          │    index_hybrid.py   RRF 融合 → hybrid_rrf（bigram ∥ bge_base）
          │
          ├─ 图谱管线（§5）
          │    graph_build.py（零 LLM）→ graph_extract.py（LLM）→ graph_assemble.py（确定性）
          │    index_graph.py    graph_expand（实体 / 卡片 / 扩展三路 RRF）
          │                      hybrid3_rrf（bigram ∥ bge ∥ graph）
          │
          ├─ evaluate.py         检索层 Recall@k / MRR（八臂同构评测）
          │
          └─ 生成层（§6）
               generate.py       top-k 证据 → prompt → DeepSeek → JSON → 引用校验 → answers.jsonl
                    │
                    ├─ evaluate.py  答案层：标签准确率 / 拒答 / 引用 / 解析失败
                    └─ judge.py     LLM-as-judge：54 道生成类题的三维评分
```

所有检索臂统一实现 `search(query, k)` / `stats()` 接口（工厂见 `src/retrievers.py`），因此评测与
生成代码对各臂完全同构，新增检索器只需在 `configs/main.yaml` 的 `retrievers` 列表加一条。

## 3. 数据

### 3.1 语料文件

首次运行时，按 §1.2 从网盘取得下列文件，放入项目内的 `data/corpus/` 即可（模板已指向此位置，
无需改配置；目录内的 `README.md` 有同样的清单与校验命令）：

| 文件 | 内容 |
| --- | --- |
| `lines.jsonl` | 11,033 行记录（台词 4,358 / 旁白 6,675），字段含 `rfid`、`text`、`speaker_raw`、`speaker_norm`、`is_dialogue`、`src_line`、`bgm`、`bg`、`branch` |
| `scenes.jsonl` | 483 个 chunk（三级边界信号切分），rfid 区间 1→15,084，含 `speakers`、`bgm`、`bg`、`branch_ids`、`has_choice` |
| `choices.json` | 52 个选择肢节点（47 个 `selbtn_start` + 5 个 `selbtn`，含真伪标注 `has_branch`）；图谱采用其中 47 个 `selbtn_start` 节点 |

### 3.2 长度口径

`scenes.jsonl` 的 `n_chars` 是**正文长度**（已扣掉所有【说话人标签】与换行），不是喂给模型的编码
长度。编码长度按 `text` 全文计：实测最长 837 字、中位 519 字，而 `n_chars` 最长 526。按 512 token
截断时，483 个 chunk 中有 187 个（38.7%）被截，详见 §11 第 1 条。

### 3.3 图谱派生数据

`data/` 下（进 git，不进提交包；重建链路见 §5）：

| 文件 | 内容 |
| --- | --- |
| `graph_entities.json` | 80 个实体（72 个说话人实体 + 8 个文本接地实体）、28 种别名、47 个选择肢节点；零 LLM 生成 |
| `graph_cards.jsonl` | 483 张事件卡（LLM 抽取，共 1,182 个事件） |
| `graph_cards_extract_log.json` | 抽取日志：模板版本、逐块状态、usage 与费用 |
| `graph_events.json` | 530 个序列化节点 + 1,634 条边（`next` 480、`has_participant` 985、`branch_to` 122、`choice_at` 47），meta 含抽取溯源 |
| `graph_cards_smoke.jsonl` | 冒烟产物（隔离存放，不参与评测） |

### 3.4 重建链路

复现实验可直接使用 §1.2 网盘“数据集”文件夹中的三份语料产物，无需执行本节。
若需自行重建，反汇编工具采用开源项目 [SiglusSceneScriptUtility](https://github.com/Jirehlov/SiglusSceneScriptUtility.git)
（`siglus-ssu`）；游戏资源获取与该工具的使用方式不在本文说明范围内。

1. 通过上述开源项目得到 `seen08000.dat.txt`。
2. 从网盘“数据集”文件夹取得 `speaker_map.tsv`（说话人归一化映射），将其与
   `seen08000.dat.txt` 一起放入重建工作目录的 `structure/` 子目录。
3. 将 `src/parse_disasm.py` 头部的 `BASE` 改为该工作目录，并确保工作目录已存在。解析器从 `BASE/structure/` 读取上述
   两个文件，向 `BASE/data/` 输出三份语料：

   ```bash
   python src/parse_disasm.py
   ```

4. 按 §1.2 将输出文件放入 `data/corpus/`，或将配置中的 `paths.scenes` / `paths.choices`
   指向实际输出位置，并按 §3.6 核验。切分参数 `MAX_CHARS=500`、`MIN_CHARS=40` 对应
   §3.1 的 483 chunk 口径。

### 3.5 来源与许可

Rewrite © Key/VisualArt's，汉化文本来自 A·T Field 补丁。网盘“数据集”中的
`lines.jsonl`、`scenes.jsonl`、`choices.json` 与 `speaker_map.tsv` 仅提供给课程教师复现实验。
反汇编工具来源见 §3.4。

### 3.6 校验值

重建完成后可用 sha256 确认输入未被改动（本文全部数字基于以下版本）：

```text
# 语料文件（§3.1，位置由 paths.scenes / paths.choices 指定；本文按 data/corpus/ 记录）
ea1f298b8e8c85c0e1d1e2106ebaf9a948501246fa31a2515d0a92700a4e49fb  lines.jsonl
60255f8625a6df2e7ddcdfeb557b5aab3dcc01f87756bb9be5f76c83278f85ab  scenes.jsonl
c46aec2bbf8ce978f977f78ac931d03dcbc5472105755ac4bd121ab5ef2abcf5  choices.json
# eval/（评测集，§7.1）
f5cb670afdcaee311c4b238e31c1d5e148008589f206632e8c6dcb3ec0a1e34f  qaset_v0.json
```

## 4. 检索层

### 4.1 分词与索引字段

- `normalize()` 把全角字母数字符号转半角（`２５岁 → 25岁`）、全角空格转半角，CJK 标点不动；
  **只用于检索字段**，证据展示始终用 `text` 原文。
- `jieba_tokenize()` 加载 `data/user_dict.txt`（37 条：说话人名真值、专名、组织名与术语，
  词频取语料频次并设下限 100，避免 HMM 把专名再次切碎）；`bigram_tokenize()` 取相邻汉字 bigram、
  ASCII 词整存，绕过词典。
- 两臂通过 `get_tokenizer()` 以同一接口注入 BM25（`rank_bm25.BM25Okapi`，`k1=1.5`、`b=0.75`）。

### 4.2 八条检索臂

| 臂 | 类型 | 说明 |
| --- | --- | --- |
| `bm25_jieba` | 稀疏 | jieba + 自定义词典分词后 BM25 |
| `bm25_bigram` | 稀疏 | 字符 bigram 分词后 BM25 |
| `w2v_avg` | 稠密（静态） | 课程 skip-gram 词向量（dim 100、window 2、neg 4、15 epoch、seed 0），chunk 内词向量平均 |
| `bge_small` | 稠密（上下文） | BAAI/bge-small-zh-v1.5，512 维，查询侧加官方指令前缀 |
| `bge_base` | 稠密（上下文） | BAAI/bge-base-zh-v1.5，768 维，同上 |
| `hybrid_rrf` | 融合 | `RRF(bm25_bigram ∥ bge_base)`，每臂取 top-20，`score = Σ 1/(60+rank)` |
| `graph_expand` | 图扩展 | 实体路 ∥ 卡片路 ∥ 扩展路，臂内 RRF（§5.3） |
| `hybrid3_rrf` | 融合 | `RRF(bm25_bigram ∥ bge_base ∥ graph_expand)` |

BGE 查询侧固定加官方指令前缀「为这个句子生成表示以用于检索相关文章：」，文档侧不加；前缀与配线
方向由 `python src/index_vector.py --offline` 的冒烟检查（`cos(q, P+q)` 显著小于 1 证明前缀未被
分词器吞掉）。文档向量按「模型 + 文本哈希」缓存到 `cache/emb_*.npy`，语料或模型变更即自动
重编码。

## 5. 图谱管线

### 5.1 构建步骤

```text
scenes.jsonl + choices.json ──graph_build.py（零 LLM）──▶ data/graph_entities.json
scenes.jsonl ──graph_extract.py（DeepSeek）──▶ data/graph_cards.jsonl（483 卡）
cards + entities ──graph_assemble.py（确定性）──▶ data/graph_events.json（530 节点 / 1,634 边）
entities + cards + events ──index_graph.py──▶ graph_expand / hybrid3_rrf
```

```bash
python src/graph_build.py                 # 实体索引；--check 重建并与盘上文件逐字节比对
python src/graph_extract.py               # 事件卡（需 DEEPSEEK_API_KEY；--limit n 冒烟）
python src/graph_assemble.py              # 边表组装；--check 逐字节比对
```

实体索引与边表组装两步完全确定性，`--check` 可逐字节验证；事件卡是 LLM 抽取产物，但响应
缓存在 `cache/llm_*.json`，**从缓存重抽零费用且逐字节一致**。抽取模板版本号写入卡片与日志，
改模板自动换缓存键。

### 5.2 图统计

- 实体：80 个（72 说话人 + 8 文本接地），别名 28 种，选择肢 47 个。
- 事件：483 张卡、1,182 个事件（对话 334 / 行动 280 / 背景说明 226 / 转折 207 / 冲突 135），
  7 张空卡，34 张含可疑参与者（`suspect` 标记，不参与 `has_participant` 边）。
- 图：530 个序列化节点、1,634 条边，单一弱连通分量，无孤立卡。
- 抽取成本：483 次调用，输入 381,055 token / 输出 96,188 token，缓存命中 123,520 token，
  谷时记账 $0.0967。

### 5.3 图扩展臂

`graph_expand` 由三路臂内 RRF 组成：

1. **实体路**：识别问题中的实体，命中的 chunk 按（多实体非空交集 ≤ `entity_top` 才投票，主角
   星型/宽集不投票）加权；
2. **卡片路**：对事件卡的规范化转述做小 BM25，弥合换说法；
3. **扩展路**：基座检索 seed top-20 + 图上 1 跳邻居，按边类型给固定尾部小分值
   （时序前后件 > 共参与者 > 同 label）。

跨 chunk 的边全部由确定性规则推导（文件序相邻、选择肢控制流、卡片参与者），不采信 LLM 标注的
跨卡引用。默认参数 `entity_top=10`、`card_top=10`、`neighbor_top=5`、`hop=1`、`seed_top=20`。

## 6. 生成层

### 6.1 模型与参数

| 项 | 值 |
| --- | --- |
| 模型 | `deepseek-flash`（V4.1-Flash-1017） |
| 端点 | `https://api.deepseek.com`（OpenAI 兼容，`openai==3.37.0`） |
| 采样 | `temperature=0`；`seed` 端点不支持、不发送；思考模式显式 `{"thinking":{"type":"disabled"}}` |
| 输出 | `response_format={"type":"json_object"}`，`max_tokens=1024` |
| 证据 | 默认检索臂 top-5，chunk 全文原文（含 rfid 区间标注） |

思考模式必须显式关闭：该端点默认开启，不关则 `temperature` 无效，且 reasoning token 会抬高延迟
与费用。关闭后 usage 不含 `reasoning_tokens` 字段。

### 6.2 回答协议与引用校验

模型只输出一个 JSON 对象：`{"answer": "...", "verdict": "yes|no|unknown", "evidence_rfids": [[起, 止], ...]}`。
生成后做两条确定性校验：

1. `evidence_rfids` 区间必须能落到具体 chunk（孤儿或越界记为 invalid）；
2. 引用的 chunk 必须 ∈ 本次送入 prompt 的证据集合（防止引用未提供的块）。

两条校验的明细逐题写入 `generation_log.json`；解析失败重试一次，仍失败则落一行
`verdict=unknown` 的答案并计入失败率，不丢弃题目。

### 6.3 提示与防泄漏

- 系统提示固定，前置 3 个 few-shot 示例（正常作答 / 是非题 / 拒答），使系统提示 + few-shot
  成为跨题稳定前缀，命中 DeepSeek 前缀缓存。
- few-shot 的证据取自语料真实 chunk，但**排除评测集全部 gold chunk**，示例问题围绕元数据
  （说话人出场、喜好类），不引入评测题本体。
- 不可回答题照走完整流程：检索照样返回字面相似块，是否拒答完全由提示约束与证据决定——这正是
  拒答评测要测的行为。

## 7. 评测协议

### 7.1 评测集

`eval/qaset_v0.json`（版本 `v0-20261023`，已冻结，90 题）：

| 题型 | 题数 | 说明 |
| --- | --- | --- |
| fact | 20 | 事实型问答 |
| yesno | 18 | 是非判断（gold verdict 9 yes / 9 no） |
| temporal | 18 | 时序比较 |
| aggregate | 16 | 聚合/列举 |
| unanswerable | 18 | 不可回答（entity_absent / fact_absent / mentioned_unknown 各 6） |

可回答的 72 题按 gold 与问题的词汇重叠分为三层：`literal` 18、`semantic_gap` 22、`structural` 31，
另有 `floor` 1 题（判据为「零实体锚 + 位置指代」，设计上只用于错误分析）。总体指标（n=72）包含
这一题，§8.1 的分层表只列前三层。层标签的操作定义与标定脚本见 `eval/README.md`；金标以
`gold_evidence_rfids`（int 或 `[起, 止]`）给出，gold 为空的题不计入检索指标，留给答案层拒答评测。

评测集构建、校验与层标定工具链在 `eval/`：

```bash
python eval/validate_qaset.py eval/qaset_v0.json    # 冻结门槛八项机检
python eval/calibrate_qaset.py eval/qaset_v0.json   # BM25 标定与难度档案
```

对冻结稿 `qaset_v0.json` 运行 validator 会在「问题清单」列出 7 条启发式提示（Q01/Q03/Q09/Q11/Q18/
Q23/Q25），全部来自层标签判据的已知盲区（df 过滤、专名切分、单锚阈值、别名词面锚、全角句柄），
经逐题对照原文裁决为误报，评测集未作改动；分层表的标签以人工裁定为准。盲区说明见
`eval/README.md` 第六节。

### 7.2 检索指标：两套 R@k 口径不可互比

| 口径 | 定义 | 实现 | 用途 |
| --- | --- | --- | --- |
| **逐题 gold 占比均值** | 每题先算命中 gold 数占该题 gold 总数的比例，再对题求均值（多 gold 题命中一部分给部分分） | `src/evaluate.py` → `metrics.json` | **本文所有表格与报告数字均为此口径** |
| 命中率 | 每题任一 gold chunk 进 top-k 记 1 | `src/index_vector.py` 的冒烟脚本 | 仅冒烟/敏感性实验，引用必须注明口径 |

例：`bge_small` R@3 命中率口径 0.5417，占比均值口径 0.3389。差异全部来自多 gold 题——structural
层 31 题全部为多 gold，平均每题 3.68 个 gold chunk。混用两套口径会得到系统性偏差的结论。

### 7.3 答案层指标

- **标签准确率**（`label_acc`）：36 道带 `gold_verdict` 的题（yesno + unanswerable）中
  `verdict` 与金标一致的比例；
- **拒答正确率**：18 道不可回答题判 `unknown` 的比例；
- **过度拒答**：18 道可回答的 verdict 题中未判 `unknown` 的比例（越高越好）；
- **引用条级有效率**：所有引用的 rfid 区间中通过 §6.2 两条校验的比例；
- **解析失败率**：90 题中两次解析均失败的比例。

### 7.4 LLM-as-judge

`src/judge.py` 对 54 道生成类题（fact 20 / temporal 18 / aggregate 16）按 rubric v2 打三维分——
忠实性、完整性、简洁性，并逐条判断引用是否在语义上支持答案断言（区别于规则层的存在性校验）。
评审证据与被评 run 生成时的 top-k 逐题核对一致；`judge_metrics.json` 不含时间戳等非确定字段，
缓存命中重跑逐字节一致。人工盲评抽检 22 题（两臂各 11 题，抽样 seed=0），与 judge 评分的
±1 一致率为 22/22（精确一致 15/22）。

默认配置（`hybrid_rrf`，k=5）的 judge 结果为：总分均值 3.3704（忠实性 4.7407、完整性 2.5556、
简洁性 4.4444），引用条级语义支持率 0.9844（64 条），hallucination 率 0。分题型总分 fact 3.90 /
temporal 2.61 / aggregate 3.56——完整性是主要失分维度，temporal 题尤甚。

## 8. 实验结果

### 8.1 八臂检索（`eval/qaset_v0.json`，评 72 题）

| retriever | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- |
| `bm25_jieba` | 0.3377 | 0.4108 | 0.4963 | 0.4942 |
| `bm25_bigram` | 0.4171 | 0.4720 | 0.5579 | 0.5381 |
| `w2v_avg` | 0.1368 | 0.1693 | 0.2806 | 0.1986 |
| `bge_small` | 0.3389 | 0.3833 | 0.5435 | 0.4490 |
| `bge_base` | 0.3704 | 0.4476 | 0.5809 | 0.5125 |
| `hybrid_rrf` | 0.4391 | 0.5194 | 0.6553 | 0.5461 |
| `graph_expand` | 0.4204 | 0.5127 | 0.5966 | **0.5628** |
| `hybrid3_rrf` | **0.4490** | **0.5465** | 0.6236 | 0.5496 |

总体表必须配分层表解读——`bge_small` 与 `bm25_jieba` 总体持平是两股相反力量对消的结果：

| 层（n） | `bm25_jieba` | `bge_small` | `bge_base` | 结论 |
| --- | --- | --- | --- | --- |
| literal（18） | **0.8611** | 0.5000 | 0.5833 | 字面专名题上 BM25 + 词典明显占优 |
| semantic_gap（22） | 0.0682 | 0.3864 | **0.4318** | 换说法题上 bge 反超约 6 倍 |
| structural（31） | 0.2359 | 0.2064 | 0.2152 | 跨 chunk 结构题上两种范式均弱 |

（表中为 R@3，`src/evaluate.py` 口径。）

结论：稀疏与稠密检索互补而非替代。RRF 把两层的强项同时保留——`hybrid_rrf` 的 literal 层 R@3
0.7500、semantic_gap 0.3864、structural 层 R@5 由 `bge_base` 的 0.3137 升至 0.4322，总体 R@5
0.5194 双超两个最佳单臂（`bm25_bigram` 0.4720、`bge_base` 0.4476）。加入图扩展第三路后
（`hybrid3_rrf`），分层 R@5 为 literal 0.8611（反超 `hybrid_rrf` 的 0.8333）、semantic_gap 0.4545、
structural 0.4467；分题型 R@5 上 temporal 由 0.5324 升至 0.6019、aggregate 由 0.4426 升至 0.4863，
但 yesno 由 0.4759 降至 0.4296。`graph_expand` 单臂 MRR 0.5628 为八臂最高，R@10 却低于两个融合臂。

分层表只覆盖三条基线臂，图臂的分层明细如下（R@5）：literal 0.9167、semantic_gap 0.4091、
structural 0.3682。`graph_expand` 在字面层仅次于 `bm25_jieba`（0.9722），换说法层低于 `bge_base`
（0.4773），结构层低于两个融合臂——它的价值体现在 MRR 上：总体 MRR 0.5628 与 structural 层 MRR
0.6180 均为八臂最高，即把正确块推到更靠前的位置，而非扩大召回面。

### 8.2 答案层：k × 图谱开关

六组实验覆盖 `top_k ∈ {3, 5, 10}` × `{hybrid_rrf（图谱关）, hybrid3_rrf（图谱开）}`；变体模板在
`configs/ablation_topk/*.yaml.example`（与 `main.yaml.example` 仅 `generation.retriever` 与
`generation.top_k` 两处不同），运行前复制为同名 `.yaml`（§1.0）。
`label_acc` 覆盖 36 道 verdict 题，judge 一律 rubric v2，证据数 = 各 run 生成时的 top_k。

| 指标 | 关 k3 | 关 k5 | 关 k10 | 开 k3 | 开 k5 | 开 k10 |
| --- | --- | --- | --- | --- | --- | --- |
| 标签准确率 | 0.7500 | **0.8056** | 0.7778 | 0.7778 | 0.6944 | 0.7778 |
| 可答题答出率（过度拒答反面） | 0.5000 | **0.6111** | 0.5556 | 0.5556 | 0.3889 | 0.5556 |
| 不可回答题拒答正确率 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 引用条级有效率 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 解析失败数（90 题） | 0 | 0 | 0 | 0 | 0 | 0 |
| judge 总分（54 题，v2） | 3.3148 | 3.3704 | 3.4074 | 3.3148 | 3.4074 | **3.4630** |
| 生成费用，谷时（90 题） | $0.0073 | $0.0297 | $0.0304 | $0.0073 | $0.0240 | $0.0309 |

解读：

- **标签准确率对 k 的形态两臂不同**：关臂在 k=5 见顶（0.8056），k=10 回落 0.0278；开臂呈 V 形，
  k=5 为谷底（0.6944），k=10 回到 0.7778。
- **judge 总分随 k 单调上升**（关臂 +0.0926、开臂 +0.1482），驱动力是完整性（关臂 2.43→2.94、
  开臂 2.35→2.83）；但 verdict 正确性没有同步收益，增量证据同时带入了干扰面。
- **引用纪律与格式不受 k 影响**：六组引用条级有效率全为 1.0、解析失败全为 0；引用条数随 k 增长
  （关臂 74→83→101）。
- **图谱开臂检索层净增益、答案层净回退**：开臂 R@5 高于关臂，但 k=5 时 `label_acc` 由 0.8056
  降至 0.6944。verdict 题全部落在 yesno / unanswerable，而图扩展的增益集中在 fact / temporal /
  aggregate 题型，不经过 verdict 面。
- **verdict 翻转分两类来源**（同臂内不同 k 之间的逐题对照，全 90 题）：一类是「gold 名次进出
  top-k」——Q09（gold 名次 4）、Q52（名次 5）、Q15（名次 5–8）、Q46（名次 9）、Q35（名次 3/9）
  等题在 k 扩大后 gold 首次进入证据窗口，verdict 随之修复或改变；另一类是「干扰块驱动」——
  Q44（gold 名次 1）、Q50（名次 3）的 gold 始终在场，但 k 增大后新入的块把答案推向 `unknown`，
  Q51 的 gold 始终缺席、靠邻近证据推断否定，同样被更多证据推回 `unknown`。这说明 verdict 对
  top-k 边界相当敏感：单个块的换入换出即可改变判断。

据此，默认配置取图谱关臂 `hybrid_rrf`、`top_k=5`：该组标签准确率六组最高，且开臂在任何 k 下
都没有同时超过它的标签准确率与 judge 均值。图谱开臂作为检索层增益臂保留在 `retrievers` 列表中，
可随时切换（`generation.retriever: hybrid3_rrf`）。

### 8.3 长上下文对照

把 Terra 线全文按 chunk 边界切成前后两半（92,207 / 92,578 字），对 36 道 verdict 题各跑两半
（共 72 次调用），协议与 RAG 生成层同构（同一 JSON 输出与引用校验），与 RAG 默认配置对照
（复现命令见 §12：`--smoke` → `--fulltext`，基线用 `--rag-off-run` / `--rag-on-run` 显式指定，
产物落 `runs/fulltext/<timestamp>/`）：

| 方法 | label_acc | 输入 token/题 | 费用/题（实付 / 冷估） | P50 延迟 |
| --- | --- | --- | --- | --- |
| 全文臂（72 实例） | 0.7361 | 70,548 | $0.00057 / $0.01062 | 1.36 s |
| RAG 关臂（36 题） | **0.8056** | 3,280 | $0.00031 / $0.00053 | 1.48 s |
| RAG 开臂（36 题） | 0.6944 | 3,342 | $0.00023 / $0.00053 | 1.57 s |

（冷估 = 假设零前缀命中，全文臂实付被 DeepSeek 前缀缓存折价约 18 倍。两行题数口径不同：全文臂是
36 题 × 2 半 = 72 个实例，RAG 是 36 题各一次，全文臂因强制切半而天然处于劣势。）

机制拆开后结论更清楚：**全文臂的失分主要来自证据定位，而非推理能力**。gold 落在所给半本内的
19 个实例中 17 例答对（0.8947，半 B 单独为 0.9333），gold 不在该半的 17 个实例 0 例答对且 15/17
判 `unknown`；18 道不可回答题在两半中 36/36 全部正确拒答。也就是说模型能读、也能诚实承认缺证，
瓶颈是「在 18.5 万字的剧本里定位到正确段落」——这正是检索层要解决的问题。冷启动口径下全文臂
单题输入是 RAG 的 21.5 倍、成本约 20 倍；延迟并未拉开（前缀缓存 + flash 模型）。

### 8.4 位置消融

对 RAG 关臂检索命中 gold 的 10 道 yesno 题，把 gold 块分别置于证据列表首位 / 第 3 位 / 末位
（其余 4 块保持原相对顺序，prompt 逐字复用生成层实现），共 30 次调用。位置由基线 top-k 推出
（首位 / 中间位 / 末位），few-shot 开关随基线 `generation_log.config.few_shot`，唯一自变量是
gold 块位置（复现命令见 §12 的 `--position --rag-off-run ...`）：

- 9/10 题在三个位置下 verdict 完全一致，且与基线一致；
- 唯一翻转 Q52：gold 置首时判 `unknown`，置第 3 / 末位时判 `yes`（与基线一致）；
- 分位置准确率 0.7 / 0.8 / 0.8，未出现经典 lost-in-the-middle 的 U 形。

样本仅 10 题，这一结果只作佐证：本设置下未见系统性位置惩罚。

## 9. 复现性与随机种子

- **检索层零随机性**：w2v `seed=0`，其余检索臂与评测流程均为纯确定性计算。
- **生成层复现性来自响应缓存**：键 = `sha256(model + thinking + temperature + max_tokens +
  response_format + messages)`，缓存在 `cache/llm_<hash16>.json`；同 prompt 命中即零二次
  请求。LLM 端点本身不提供 `seed`，复现性由缓存兜底。
- **逐字节验证**：同配置重复运行结果逐字节一致——2026-10-25 三连跑（`runs/20261025-184349` /
  `184856` / `185352`，当时为六臂版本）的检索块 sha256 均为 `50f0c7a2…`；加入图臂并回填默认臂后
  `runs/20261026-222529` 与 `runs/20261029-222857` 的 `metrics.json` 均为 `f9bb4479…`，
  `answers.jsonl` 均为 `9e3f83f6…`（与随包的 `eval/answers.jsonl` 相同）。`judge_metrics.json` 在
  缓存命中重跑下同样逐字节一致。
- **"重跑一致"限定同机器、同依赖版本**：浮点实现与库版本相关，全部版本号落在 `env_info.json`
  （Python、jieba、numpy、torch、sentence-transformers、transformers、openai、模型 revision、
  生成配置）。换机器或换版本只承诺方向与量级一致。
- **缓存**：bge 文档向量缓存命中时纯矩阵加载（毫秒级）；w2v 每 run 重训（约 247 s，loss 曲线
  落 `w2v_train_log.json`，作为训练可复现的指纹）。

## 10. 成本与延迟

### 10.1 单价与总费用

`deepseek-flash` 谷时定价（USD / 1M token）：输入未命中 $0.15、命中 $0.003、输出 $0.60；
峰时（工作日 09–12 / 14–18 点）×2。各环节费用按 usage 记账，缓存命中重放不产生新费用：

| 环节 | 调用数 | 谷时费用 |
| --- | --- | --- |
| 图谱事件卡抽取（483 chunk） | 483 | $0.0967 |
| 生成：默认配置（90 题） | 90 | $0.0297 |
| 生成：图谱开臂 k=5（90 题） | 90 | $0.0240 |
| 生成：k 消融四组（k ∈ {3,10} × 图谱开/关） | 360 | $0.0759 |
| LLM-as-judge：rubric v1 两臂（54 题 × 2）+ 缺证审计 2 次 | 110 | $0.0394 |
| LLM-as-judge：rubric v2 两臂复评（54 题 × 2） | 108 | $0.0377 |
| LLM-as-judge：k 消融四组（54 题 × 4） | 216 | $0.0527 |
| 长上下文对照（72）+ 位置消融（30） | 102 | $0.0499 |
| 评测集冒烟（10 题） | 10 | $0.0035 |
| **合计** | **1,569** | **≈ $0.41**（峰时口径约 $0.82） |

缓存命中重放不产生新费用，同配置反复重跑（用于复现性验证）不增加支出；因此表中每个不同配置只
计一次，金额是该配置按 usage 折算的等效费用，而不是逐次账单累加。rubric 从 v1 升到 v2 后的全量
复评按新口径重跑，两次费用都计入。

### 10.2 延迟

生成层单题网络延迟（`temperature=0`、思考关闭、无 reasoning token）：k 消融六组中五组实测
P50 0.97–1.57 s、P95 1.40–2.14 s；k=3/k=10 三档之间 P50 基本持平（0.97–1.10 s），说明 k 不是
延迟瓶颈。基线 run（`runs/20261026-222529`）为缓存重放执行，其日志中的延迟字段为空，费用仍按
usage 记账。全文臂因前缀缓存命中，P50 1.36 s 与 RAG 持平（见 §8.3）。

### 10.3 前缀缓存

系统提示 + few-shot 构成跨题稳定前缀，默认配置 90 题共 299,383 输入 token，其中 129,536
（43%）命中前缀缓存；叠加本地响应缓存后，重跑全量新增费用为 $0。

## 11. 已知局限

1. **512 token 截断（三级口径，勿只引 38.7%）**：块级 187/483 = 38.7% 被截；题级 gold 被截
   56/72 = 77.8%；gold 证据**完整落入截尾**仅 5/72 = 6.9%。分组对照（A = gold 未截、
   B = 被截但证据存活、C = 证据全丢）：`bge_small` R@3 为 0.573 / 0.276 / 0.233，`bm25_jieba`
   为 0.385 / 0.330 / 0.267——gold 被截的题上 bge 显著失分，BM25 几乎不动（BM25 全文入索引，
   无 512 限制）。截断与长文池化稀释完全共线，组间差是联合效应；切分不迁就模型（真值结构决定），
   分析产物见 `runs/<timestamp>/truncation_analysis.json`。
2. **w2v 平均在全部题型垫底**（R@3 0.1368）：静态词义、丢词序、OOV（评测期累计 120 个查询
   token OOV）、长 chunk 平均稀释，四个缺陷共同作用。这一负结果本身说明上下文相关表示的必要性。
3. **专名判别与字面匹配的分工**：literal 层上 `bm25_jieba` R@3 0.8611，而 `bge_small` 仅 0.5000；
   Q06 / Q09 这类题面专名题上 BM25 可 rank-1 命中，bge 的 top-5 被主题相近的更长 chunk 占据。
   稠密检索的强项在 semantic_gap 层（0.3864 vs 0.0682），两者不可互相替代。
4. **结构性题两范式同败**：structural 层 R@3 三条基线臂都在 0.21–0.24，是图扩展与 RRF 的动机，
   但图扩展带来的答案层收益并未兑现（§8.2）。
5. **过度拒答仍在**：默认配置下 7/18 道可回答 verdict 题被判 `unknown`，其中 5 例 gold 不在
   top-5（检索失败）、2 例 gold 在证据内但模型仍拒答（部分证据不足以支撑判断）。
6. **缺证否定推断的边界**：gold=no 且检索未命中时，模型可能从相邻证据推出正确否定（Q45、Q51
   两例审计判为合理否定推断）。「缺证即 `unknown`」的协议在「邻近证据显式蕴含否定」时会牺牲
   正确性，目前表述为「缺证即 `unknown`，除非证据显式蕴含否定」，但该边界未被完全约束。
7. **judge 自评偏差**：judge 与被评系统同为 `deepseek-flash`，以评审员角色隔离 + 22 题人工盲评
   （±1 一致率 22/22）缓解，不是独立第三方评测；`hallucination` 布尔标志偏保守，条级引用支持率
   是更灵敏的指标。
8. **位置消融样本小**：10 题、3 个位置，只能说明本设置下未见系统性位置惩罚，不足以推广。
9. **图谱的 LLM 抽取噪声**：34/483 张卡含可疑参与者（`suspect` 标记已排除出参与者边），
   事件卡由逐块单次抽取得到，未做多轮投票或人工校订。
10. **金标与切分口径**：3 个 rfid（3281 / 14785 / 14898）不在任何 chunk 区间内（切分时随选择肢
    边界丢弃的碎片），金标若命中这些位置，`evaluate.py` 会直接报错而非静默错评。
11. **只在标签里出现的名字**：`千早`、`露西娅` 仅作为场景收尾的说话人【标签】出现，正文叙述从不
    点名，纯文本检索对这类名字有盲区；`index.index_speakers: true` 可把 `speakers` 元数据并入
    索引文本（默认关闭的消融开关）。

## 12. 单模块命令

```bash
# 检索 demo
python src/index_bm25.py "路易斯是什么国籍？" --arm jieba -k 5    # BM25（jieba）
python src/index_bm25.py "江坂和瑚太朗是什么关系" --arm bigram -k 5 # BM25（bigram）
python src/index_w2v.py "路易斯是什么国籍？"                      # w2v 平均（含训练，约 4 分钟）
python src/index_vector.py "路易斯是什么国籍？" --offline         # bge demo
python src/index_vector.py --offline                              # 前缀方向冒烟
python src/index_hybrid.py "盖亚和守护者是什么关系？" -k 3         # RRF 融合 demo
python src/index_graph.py "盖亚和守护者是什么关系？" -k 5          # 图扩展臂三路明细

# 图数据
python src/graph_build.py --query "瑚太朗和路易斯约定了什么？"      # 实体识别 + 候选 chunk
python src/graph_build.py --check                                 # 实体索引确定性验收
python src/graph_assemble.py --check                              # 边表确定性验收

# 评测与生成
python src/evaluate.py --arm bigram                               # 单臂检索评测
python src/generate.py                                            # 生成层（90 题 → eval/answers.jsonl）
python src/generate.py --qaset eval/qaset_seed.json --out eval/answers_smoke.jsonl  # 冒烟
python src/judge.py --run runs/<timestamp>                        # LLM-as-judge（54 题）
python src/generate_fulltext.py --smoke                            # 长上下文对照：冒烟（4 次调用）
python src/generate_fulltext.py --fulltext \
    --rag-off-run runs/<off-run> --rag-on-run runs/<on-run>        # 全文对照（72 次调用）
python src/generate_fulltext.py --position --rag-off-run runs/<off-run>  # 位置消融（30 次调用）

# 分析工具
python src/analyze_truncation.py --offline                         # 512 截尾影响评估
python src/demo_tokenize_impact.py                                 # 三臂分词对照（须全新进程）
python src/build_userdict.py                                       # 词典候选统计（默认读 data/corpus/）
python src/build_userdict.py --lines <lines.jsonl> --scenes <scenes.jsonl>  # 指定语料位置
python src/build_userdict.py --check                               # 抽查定稿词典的切分效果
python src/plot_w2v_loss.py                                        # 渲染 loss 曲线（读 runs/ 日志）
```

`src/build_userdict.py` 只做候选发现与统计，不写入任何文件：不加开关时输出 `speaker_norm` 频次与
术语候选频次供人工审核，`--check` 则对样例句打印定稿 `data/user_dict.txt` 的切分结果。它默认读取
项目根下的 `data/corpus/lines.jsonl` 与 `data/corpus/scenes.jsonl`（§1.2 的落位建议），**不含任何
硬编码的绝对路径**；语料放在别处时用 `--lines` / `--scenes` 传入实际路径即可（默认文件缺失时脚本会
直接提示改用这两个参数）。该脚本不是主流程的一环，`run_all.py` 不调用它。

`python src/demo_tokenize_impact.py` 需要在全新进程中运行：jieba 自定义词典只能增量加载、无法
卸载，同进程内先加载词典会污染「默认词典」对照。

`src/judge.py` 需要先有完整的 `runs/<timestamp>/`，并通过 `--run` 指定。

`src/generate_fulltext.py` 的三个模式（`--smoke` / `--fulltext` / `--position`，不给开关等价于
`--fulltext`）互相排斥，基线一律显式指定：`--rag-off-run` 是图谱关臂（对照与位置消融都必填），
`--rag-on-run` 是图谱开臂（仅全文对照必填）。基线目录须含 `generation_log.json` 与
`config_copy.yaml`，全文对照另需 `metrics.json`，位置消融另需 `answers.jsonl`。预检在**任何 API
调用与产物写入之前**完成：缺文件、两臂检索器相同、top-k 或模型参数不一致、语料 chunk 数或评测集
哈希对不上都会一次性报错退出；`--dry-run` 可只跑预检并打印调用计划。

产物写入独立实验目录，缺省 `runs/fulltext/<timestamp>/`（可用 `--out-dir` 指定，便于把冒烟、全量、
位置消融三步放在同一目录）。目录内含 `experiment_manifest.json`（配置/语料/评测集指纹 + 每步的
基线目录、检索臂、top-k、实际题量）与 `config_copy.yaml`，同名产物已存在时需 `--force` 才覆盖。
冒烟判定只认本实验目录内的 `fulltext_smoke_log.json`，其它目录的旧冒烟文件不参与本次判定。
两类脚本都需要 `DEEPSEEK_API_KEY`；`judge.py` 的产物写回被评 run 目录，全文/位置消融的产物写入
各自的实验目录。复现流程与历史问题记录见

## 13. 目录结构

```text
./                            # 项目根（提交包解压后的目录，含 run_all.py）
├── run_all.py               # 一条命令全流程（八臂检索 × qaset_v0 + 图数据段 + 生成 + 答案层）
├── requirements.txt         # 依赖锁定（torch 需另装 CPU 轮子）
├── .env.example             # DEEPSEEK_API_KEY 模板（复制为 .env，不进仓库）
├── configs/
│   ├── main.yaml.example    # 配置模板（项目内相对路径；复制为 main.yaml 使用，见 §1.0）
│   ├── main.yaml            # 本地实际配置（.gitignore；不进提交包）
│   └── ablation_topk/       # k ∈ {3,10} × 图谱开/关 的消融变体（k=5 用 main.yaml）
│       └── *.yaml.example   # 变体模板；同名 .yaml 为本地实际配置（不进提交包）
├── src/
│   ├── config.py            # 配置加载与路径解析（相对项目根）
│   ├── tokenization.py      # 全角→半角归一化 + jieba / 字符 bigram 分词
│   ├── build_userdict.py    # 词典候选发现（说话人真值 + 语料频次 + OOV）
│   ├── index_bm25.py        # BM25Okapi 建索引 / 检索（含 CLI demo）
│   ├── index_w2v.py         # skip-gram 词向量平均检索臂
│   ├── index_vector.py      # BGE 稠密检索 + 查询前缀 + HF 缓存重定向 + 冒烟
│   ├── index_hybrid.py      # RRF 融合（按名复用已建臂）
│   ├── graph_build.py       # 实体-场景索引（零 LLM）+ EntityIndex 识别器
│   ├── graph_extract.py     # 事件卡抽取（DeepSeek，模板 v2）
│   ├── graph_assemble.py    # 确定性边表 + networkx 组装校验
│   ├── index_graph.py       # 图扩展臂：实体 / 卡片 / 扩展三路臂内 RRF
│   ├── retrievers.py        # 检索器工厂：config 条目 → 统一 search 接口
│   ├── llm.py               # DeepSeek 客户端：Key / 缓存 / 三级 JSON 解析 / 重试 / 计价
│   ├── generate.py          # 生成层：prompt 组装 / few-shot / 引用校验
│   ├── evaluate.py          # Recall@k / MRR + 答案层 + rfid 查表
│   ├── judge.py             # LLM-as-judge 评审 + 人工抽检模板 + 缺证审计
│   ├── generate_fulltext.py # 长上下文对照与位置消融
│   ├── analyze_truncation.py# 512 截尾影响评估（三级口径 + 分组对照）
│   ├── demo_tokenize_impact.py # 三臂分词对照演示
│   ├── plot_w2v_loss.py     # w2v loss 曲线渲染
│   ├── qaset_analysis.py    # 评测集共享分析层（重叠率 / 实体集 / df）
│   └── parse_disasm.py      # 反汇编解析（语料重建链路第 2 步；BASE 常量按本机修改）
├── data/
│   ├── user_dict.txt        # jieba 自定义词典
│   ├── corpus/              # 语料落位目录；内含 README.md 说明放什么与校验方式（语料本身不进提交包）
│   └── graph_*.json[l]      # 图谱派生数据（可重建，见 §5）
├── eval/
│   ├── qaset_v0.json        # 正式评测集 90 题（冻结；表格数字唯一来源）
│   ├── qaset_seed.json      # 10 题冒烟集（数字不进报告）
│   ├── qaset_calibration.json # 难度档案（每题 R@k / 首命中 / 层归属）
│   ├── validate_qaset.py / calibrate_qaset.py / merge_qaset.py
│   ├── qaset_v0_draft.json / qaset_v1_candidates_*.json  # 构建期中间产物
│   └── README.md            # 评测集 schema、生命周期命令流与已知盲区
├── cache/                   # 可再生缓存（不进 git）：emb_*.npy + llm_*.json
└── runs/                    # 运行日志（主流程每次新建 runs/<timestamp>/；
                             #   全文/位置消融写 runs/fulltext/<timestamp>/）
```

## 14. 提交包范围

包含：`README.md`（由本文作为提交版）、`run_all.py`、`src/`（全部源码）、`configs/` 下的
**模板文件**（`main.yaml.example` 与 `ablation_topk/*.yaml.example`，均为项目内相对路径）、
`requirements.txt`、`.env.example`、`全文与位置消融复现流程待办说明.md`、
`data/user_dict.txt`、`data/corpus/README.md`（语料落位与校验说明）、`eval/` 的评测集
（`qaset_v0.json`、`qaset_seed.json`、
`qaset_calibration.json`）、评测工具链（`validate_qaset.py`、`calibrate_qaset.py`、
`merge_qaset.py`、`README.md`），以及 `eval/answers.jsonl`（默认配置 90 题生成结果，27KB）。

提交包内不含任何实际配置文件（`configs/main.yaml`、`configs/ablation_topk/*.yaml`）：它们是本地
运行配置，已在 `.gitignore` 中，避免把开发机的上级目录路径带进提交物。收到源码后按 §1.0 从模板
复制即可，模板的路径默认值即 §1.2 的落位建议。

`eval/answers.jsonl` 是参考生成产物，随包附带的目的有二：无 Key 时 `run_all.py` 仍能复算出
§8.2 的答案层指标（标签准确率 0.8056 等），且可与自己重跑的结果逐字节比对。若把它删掉，`run_all.py`
会跳过答案层，检索层指标不受影响，答案层与 judge 指标则需自备 Key 重跑。

语料与映射通过 §1.2 的网盘“数据集”文件夹提供：`lines.jsonl`、`scenes.jsonl`、`choices.json`、
`speaker_map.tsv`，仅供课程复现实验。提交源码根目录即本项目目录，不包含上级 `NLP/` 仓库；
独立运行的路径设置见 §1.2。

源码包不含以下可下载或可再生内容：

| 内容 | 原因 | 复现方式 |
| --- | --- | --- |
| `models/`（bge 权重，约 900MB） | 体积超限，可从公共仓库获取 | §1.2，首次运行自动下载 |
| `cache/`（向量与 LLM 响应） | 可再生 | 运行 `run_all.py` 或对应模块时自动重建 |
| `runs/`（运行日志） | 主流程日志可再生，体积随实验次数增长 | `run_all.py` 生成新 run；全文/位置消融用 `--rag-off-run`/`--rag-on-run` 指定基线后重新生成 |
| `data/graph_*`（图谱派生数据） | 可重建 | §5.1，实体索引与边表组装不调用 LLM；事件卡抽取需 Key，完整响应缓存命中时无新增 API 费用，冷启动需付费 |
| `eval/answers_smoke*.jsonl`、`eval/answers_generation_log.json`、`eval/qaset_v0_draft.json`、`eval/qaset_v1_candidates_*.json` | 冒烟产物与构建期中间文件 | 运行 `generate.py` 重新生成 |

`.env` 与 `DEEPSEEK_API_KEY` 不进入任何提交物。
