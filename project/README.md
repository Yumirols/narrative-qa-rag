# galgame 剧本叙事问答 · RAG + 图谱增强（课程项目方向 A：领域 RAG 问答）

Rewrite Terra 线（约 19.5 万字、1.1 万行台词）叙事问答的检索与生成全链：**八臂检索**
（bm25_jieba / bm25_bigram / w2v_avg / bge_small / bge_base / hybrid_rrf / **graph_expand** /
**hybrid3_rrf**）× 正式评测集 `eval/qaset_v0.json`（90 题，评 72、不可回答跳过 18）+ **生成层**
（检索 top-5 证据块 → DeepSeek 结构化 JSON 回答 → 引用规则校验）+ 答案层指标（标签准确率/
拒答正确率/引用有效率/解析失败率）+ **图谱管线**（T9 实体-场景索引零 LLM → T10 事件卡
LLM 抽取 + 确定性边表 → T11 图扩展召回臂）。技术路线与知识地图见
`../项目知识框架_剧本问答.md`，阶段执行记录见 `../Rewrite_terra/report/`。

当前状态：**阶段四（图谱增强）收口**（T9~T12 完成，报告见
`../Rewrite_terra/report/阶段报告_T9~T12_图谱增强.md`）。图谱开（hybrid3_rrf）检索总体
R@5=0.5465 > 图谱关 0.5194（temporal +0.070 / semantic_gap +0.045 / aggregate +0.044 /
literal 反超），但答案层消融净回退（label_acc 0.8056→0.6944、过度拒答 0.611→0.389，
0 翻正 / 4 翻错全为新过度拒答）——检索增益与 verdict 题型错位的机制归因见报告 §4；
建图全程 ¥0.68、阶段四 LLM 合计 ≈$0.125。一条命令全链跑通、无 Key 自动降级
（实体索引重建 + 图臂降级提示 + 生成段跳过）、缓存重跑逐字节一致。

## 快速开始

```bash
pip install -r requirements.txt        # torch 必须 CPU 轮子, 见下"模型与环境"
export DEEPSEEK_API_KEY=sk-…           # 生成层用; 不设则 run_all 自动跳过生成段
python run_all.py --config configs/main.yaml
```

八臂总体表（evaluate.py 口径，qaset_v0，2026-10-26 三 run 检索指标逐字节一致）：

```
retriever     Recall@3   Recall@5   Recall@10  MRR
bm25_jieba    0.3377     0.4108     0.4963     0.4942
bm25_bigram   0.4171     0.4720     0.5579     0.5381
w2v_avg       0.1368     0.1693     0.2806     0.1986
bge_small     0.3389     0.3833     0.5435     0.4490
bge_base      0.3704     0.4476     0.5809     0.5125
hybrid_rrf    0.4391     0.5194     0.6553     0.5461   # RRF(bigram∥bge_base) T6
graph_expand  0.4204     0.5127     0.5966     0.5628   # 图扩展臂 T11: MRR 全场最高
hybrid3_rrf   0.4490     0.5465     0.6236     0.5496   # RRF(bigram∥bge∥graph) T11
```

**总体表必须配分层表解读**（bge_small 与 bm25_jieba 总体持平是两股力量对消的假象）：

| 层 | n | bm25_jieba | bge_small | bge_base | 结论 |
| --- | --- | --- | --- | --- | --- |
| literal | 18 | **0.861** | 0.500 | 0.583 | 字面专名题 BM25+词典碾压稠密 |
| semantic_gap | 22 | 0.068 | **0.386** | **0.432** | 换说法题 bge 反超 ≈6 倍 |
| structural | 31 | 0.236 | 0.206 | 0.215 | 跨 chunk 结构两范式同败（图谱动机） |

（R@3，evaluate.py 口径）检索范式结论：稀疏与稠密**互补而非替代**——RRF 融合
（`src/index_hybrid.py`，各臂 top-20，`score=Σ1/(60+rank)`）把两层强项同时保住：
hybrid literal R@3=0.75 / semantic_gap 0.3864 / **structural R@5 0.3137→0.4322**，
总体 R@5=0.5194 双超最佳单臂（bigram 0.4720 / bge_base 0.4476）。融合参数敏感性
（rrf_k/depth/臂组合）见阶段报告 §2.3——总体增益对参数稳健，默认值为先验选择
（防测试集泄漏，不据评测集调参）。阶段四加入图扩展第三路后（T11，
`src/index_graph.py`）：hybrid3_rrf 分层 R@5 = literal 0.8611（反超关臂 0.8333）/
semantic_gap 0.4545 / structural 0.4467，temporal 分型 0.5324→**0.6019**、aggregate
0.4426→0.4863；机制迭代消融归因与实体路 T9.3 门槛设计见阶段四报告 §3。

答案层指标（阶段四图谱开/关消融对，`runs/20261026-222529`[关] vs
`runs/20261026-223110`[开]，generation top_k=5）：

| 指标 | 图谱关 hybrid_rrf | 图谱开 hybrid3_rrf |
| --- | --- | --- |
| verdict 标签准确率 | **0.8056**（29/36） | 0.6944（25/36） |
| 不可回答题拒答正确率 | **1.0**（18/18） | **1.0**（18/18） |
| 可答题过度拒答 | **0.6111**（答出 11/18） | 0.3889（答出 7/18） |
| 引用有效率（条级，规则校验） | **1.0**（83/83） | **1.0**（87/87） |
| JSON 解析失败率（重试后） | **0** | **0** |
| token / 成本 | 90 题全缓存命中 $0（首轮 $0.0297） | 输入 303,976（前缀命中 57%）/ 输出 6,299，**$0.024** |

消融结论（报告 §4）：图谱开检索层净增益，答案层净回退——verdict 题全落在
yesno/unanswerable，而图谱增益落在 fact/temporal/aggregate（不载 verdict）、yesno
恰回退 0.046；叠加卡片路尾部干扰块引发 4 题新过度拒答（0 翻正/4 翻错，Q44/Q50
干扰注入、Q52 融合挤出 gold、Q51 侥幸推断打断）。原 7 错题全部维持（检索层未翻正）。

单模块入口：

```bash
python src/index_bm25.py "路易斯是什么国籍？" --arm jieba -k 5   # BM25 检索 demo
python src/index_w2v.py "路易斯是什么国籍？"                      # w2v demo(含训练,约4min)
python src/index_vector.py "路易斯是什么国籍？" --offline         # bge demo(首跑编码,毫秒级缓存)
python src/index_vector.py --offline                              # 前缀方向冒烟(坑8)
python src/index_hybrid.py "盖亚和守护者是什么关系？" -k 3         # hybrid RRF demo(只建引用臂)
python src/graph_build.py --query "瑚太朗和路易斯约定了什么？"      # T9 实体识别+候选集演示
python src/index_graph.py "盖亚和守护者是什么关系？" -k 5          # T11 图扩展臂三路明细+融合
python src/analyze_truncation.py --offline                        # 512 截尾影响评估
python src/demo_tokenize_impact.py    # 三臂分词对照(原理点①演示,须全新进程单独跑)
python src/build_userdict.py          # 词典候选统计; --check 抽查切分
python src/evaluate.py --arm bigram   # 独立评测单臂
python src/generate.py                # 生成层单独跑(90 题,→ eval/answers.jsonl)
python src/generate.py --qaset eval/qaset_seed.json --out eval/answers_smoke.jsonl  # 冒烟
```

## 两套 R@k 口径声明（引用数字前必读）

同一 run 同一臂，两口径数字**不可互比**：

| 口径 | 定义 | 出处 | 用途 |
| --- | --- | --- | --- |
| **逐题 gold 占比均值** | 每题 `|gold∩topk|/|gold|` 再对题求均值（多 gold 题命中一部分给部分分） | `src/evaluate.py`（eval_retrieval）→ `metrics.json` | **正式报告一律用这套**（与全部历史数字可比） |
| 命中率 | 每题任一 gold chunk 进 top-k 记 1 | `src/index_vector.py` `_rank_metrics`（冒烟脚本自带） | 仅冒烟/敏感性实验；引用时必须注明口径 |

例：bge_small R@3 命中率口径 0.5417 vs 占比均值口径 0.3389——差异全部来自
多 gold 题（structural 层 31 题全部多 gold，均值 3.68 个 gold chunk）。

## 复现性

- 检索层全流程**零随机性**（w2v seed=0、评测纯确定性）；生成层复现性 = **LLM 响应缓存**
  （键 = sha256(model+thinking+temperature+max_tokens+response_format+messages)，
  `project/cache/llm_<hash16>.json`，同 prompt 命中即零二次请求、逐字节一致）。
- **"重跑一致"限定同机器同依赖版本**（浮点实现相关）：同 config 重跑
  `metrics.json` **与 `answers.jsonl`** 逐字节一致（2026-10-25 两跑验证，sha256
  `87ebf117…7914`）。全部版本号落 `env_info.json`（Python / jieba / numpy / torch /
  sentence-transformers / transformers / openai / 模型 revision / 生成配置）。换机器/换
  版本只承诺方向与量级。
- bge 文档向量缓存命中时纯矩阵加载（首跑 small ≈10s / base ≈140s CPU，命中
  毫秒级）；w2v 每 run 重训（设计取舍，约 214~226s，loss 曲线为复现指纹）。
- 每次运行落盘 `runs/<timestamp>/`：`config_copy.yaml`、`env_info.json`（含 T12.3 graph
  块：实体索引动作/图谱统计/抽取溯源/图臂降级面）、`segmentation_demo.txt`（分词对照）、
  `graph_build_log.json`（T12.3 图数据段日志）、`metrics.json`（报告图表唯一数据源，
  含八臂检索 + label_accuracy 块）、`retrieval_results.json`（每题 top-k 与命中标记）、
  **`answers.jsonl` + `generation_log.json`**（T8 生成产物，含每题 prompt 哈希/usage/
  延迟/原始响应/引用校验明细；answers 同时回填 `eval/answers.jsonl`）、
  `w2v_train_log.json`、`bge_encode_log_<name>.json`（按检索器名隔离）、
  `hybrid_config_<name>.json`、`graph_config_graph_expand.json`（图臂参数/实体门槛计数/
  降级状态）；`truncation_analysis.json` 由 `src/analyze_truncation.py` 补挂到指定 run 目录。
- 评测口径：gold 用 **rfid**（全局唯一已读标记）；chunk 的 rfid 区间严格递增不
  重叠，gold→chunk 为确定性二分查表；rfid 落在区间外立即报错（防金标静默错评）。
  gold_evidence_rfids 为空的不可回答题不计入检索指标，留给答案层拒答评测。

## 模型与环境

- torch 必须装 **CPU 轮子**：`pip install torch==2.14.1 --index-url
  https://download.pytorch.org/whl/cpu`（默认 PyPI 拉 CUDA 版 2.5GB+）。
  版本锁定：torch 2.14.1+cpu / sentence-transformers 6.1.0 / transformers 5.20.2。
- 模型：BAAI/bge-small-zh-v1.5（512 维）+ BAAI/bge-base-zh-v1.5（768 维），
  本地缓存在仓库根 `models/`（873MB，`.gitignore` 已覆盖，不进 zip）。
- **首次运行需联网下载一次**：`HF_ENDPOINT=https://hf-mirror.com` +
  `HF_HUB_DISABLE_XET=1`；此后 `HF_HUB_OFFLINE=1` 可全程离线。
- `HF_HOME`/`HF_HUB_CACHE` 由 `src/index_vector.apply_hf_cache_redirect` 在
  import 前 setdefault 到 `models/`（缓存路径常量 import 时固化，**重定向必须在
  import sentence_transformers 之前**——自定义脚本请复用该函数，坑 6）。
- bge 查询侧指令前缀（`为这个句子生成表示以用于检索相关文章：`）为常量写死，
  只加查询侧、文档侧不加（BGE 官方用法，坑 8；方向冒烟见 `index_vector.py --offline`）。

## 生成 LLM（DeepSeek,阶段三）

- **模型 `deepseek-flash`**（V4.1-Flash-1017），OpenAI 兼容端点
  `https://api.deepseek.com`，`openai==3.37.0` SDK；`temperature=0` +
  **思考模式显式关闭**（`extra_body={"thinking":{"type":"disabled"}}`——端点默认
  开启，不关则 temperature 无效且 reasoning token 抬高延迟费用）；`seed` 端点
  不支持、不发送（复现性由响应缓存兜底）；`response_format=json_object`。
- **Key 只走环境变量 `DEEPSEEK_API_KEY`**（或 `project/.env`，手写解析器加载，
  不引 python-dotenv；`.env` 已进 `.gitignore`，模板 `.env.example`）——
  **绝不进代码/config/仓库/zip**（任务书红线③）。Key 缺失时 `run_all.py` 生成段
  整段跳过，检索层不受影响。
- 实测口径修正（2026-10-25）：思考关闭时 usage **不含** `reasoning_tokens` 字段
  （非返回 0）；"思考确已关"守门判据 = 字段缺席 + 短输出（均值 70 token/题）+
  ~1.5s 延迟。定价（USD/1M）：输入未命中 $0.15 / 命中 $0.003 / 输出 $0.60，
  峰时（工作日 09-12/14-18 点）×2。全量 90 题实测 $0.0297（谷口径），
  其中前缀缓存命中 129,536/299,383 输入 token。

## 图谱管线（阶段四 T9~T11）

```
scenes.jsonl ──graph_build.py(T9, 零 LLM)──▶ data/graph_entities.json   80 实体/47 选择肢
scenes.jsonl ──graph_extract.py(T10, LLM ¥0.68)──▶ data/graph_cards.jsonl   483 事件卡
cards+entities ──graph_assemble.py(T10.3, 确定性)──▶ data/graph_events.json 530 节点/1634 边
entities+cards+events ──index_graph.py(T11)──▶ graph_expand / hybrid3_rrf 检索臂
```

- **建图三步全部一条命令**，前两步产物 `--check` 重建逐字节一致；事件卡为 LLM 抽取产物，
  **从 `cache/llm_*.json` 响应缓存重抽零费用、逐字节一致**（模板版本常量 v2 入卡片与日志，
  改模板自动换缓存键）。
- **graph_expand 臂内三路 RRF**（`src/index_graph.py`）：实体路（T9.3 门槛——多实体非空
  交集 ≤entity_top 才投票，主角星型/宽集不投票）∥ 卡片路（规范化转述小 BM25，弥合换说法）
  ∥ 扩展路（基座 seed top-20 + 图上 1 跳邻居固定尾部小分值：时序前后件>共参与者>同 label）。
- **降级（无 Key 环境红线）**：entities 缺失 run_all 自动重建（零 LLM）；cards/events 缺失
  图臂对应路静默降级（实体路+seed 仍可用）并打印重建链路提示；生成段整段跳过——一条命令
  照跑（`runs/20261026-222237` 实证）。
- 成本：建图 493 次 ¥0.68（单卡 ≈$0.0002）；生成图谱开 $0.024 / 图谱关 $0（缓存命中）；
  阶段四 LLM 合计 ≈$0.125，低于 $0.6 预算上限。

## 数据（不进 zip）

`configs/main.yaml` 指向 `../Rewrite_terra/data/`（Day 1 产物）：

| 文件 | 内容 |
| --- | --- |
| `lines.jsonl` | 11,033 行台词，行级真值结构（rfid/说话人/src_line/bgm） |
| `scenes.jsonl` | 483 个 chunk（三级边界信号真值切分），rfid 区间 1→15084 |
| `choices.json` | 47 个选择肢节点 |

图谱派生数据在 `project/data/`（**进 git、不进 zip**，重建链路见上节：graph_build →
graph_extract → graph_assemble，前两步可从缓存零费用复现）：`graph_entities.json`（T9 零
LLM）/ `graph_cards.jsonl` + `graph_cards_extract_log.json`（T10 LLM 抽取卡与 usage 日志）/
`graph_events.json`（T10.4 确定性组装，meta 含抽取溯源与费用）。`graph_cards_smoke.jsonl`
为冒烟产物（隔离存放）。

**长度口径警告**：`scenes.jsonl` 的 `n_chars` 字段 = **正文长**（len(text) −
所有【说话人标签】 − 换行），不是喂给模型的编码长度。编码长度以 `text` 全文计
（实测 max 837 字、中位 519 字；`n_chars` max 526——曾据此误估"512 截尾约 1 块"，
实测 187/483=38.7%，见下"已知局限"）。

重建方式（README 声明，替代打包剧本）：
1. `siglus-ssu 0.4.0 -x --disam --angou <游戏目录>` 反汇编 `seen08000.dat` → `structure/`；
2. `python ../Rewrite_terra/src/parse_disasm.py` 解析出 `data/`。

来源声明：Rewrite © Key/VisualArt's；汉化文本来自 A·T Field 补丁。仅课程使用，
不随 zip 分发；反汇编 `.txt` 含剧本全文，同为派生数据不进包。

## 目录

```
project/
├── run_all.py               # 一条命令全流程(八臂检索 × qaset_v0 + 图数据段 + 生成 + 答案层)
├── configs/main.yaml        # 路径 / retrievers 八臂 / graph 参数 / generation 节 / 评测 k 值
├── .env.example             # DEEPSEEK_API_KEY 模板(复制为 .env, 不进 git)
├── src/
│   ├── tokenization.py      # 全角→半角归一化 + jieba/字符bigram 分词
│   ├── build_userdict.py    # 词典候选发现（speaker 真值 + 语料频次 + OOV）
│   ├── index_bm25.py        # BM25Okapi 建索引/检索（含 CLI demo）
│   ├── index_w2v.py         # w2v 平均基线：课程 skip-gram 改造 + W2VAvgIndex
│   ├── index_vector.py      # bge 稠密检索：BGEIndex + 前缀常量 + HF 重定向 + 方向冒烟
│   ├── index_hybrid.py      # hybrid RRF 融合臂(T6): 按名复用已建臂 + rrf_fuse 融合函数
│   ├── graph_build.py       # T9 实体-场景索引(零 LLM) + EntityIndex 识别器
│   ├── graph_extract.py     # T10 事件卡抽取(DeepSeek, 模板 v2, 冒烟/全量 CLI)
│   ├── graph_assemble.py    # T10.3/10.4 确定性边表 + networkx 组装校验落盘
│   ├── index_graph.py       # T11 图扩展臂: 实体/卡片/扩展三路臂内 RRF
│   ├── retrievers.py        # 检索器工厂: config retrievers 条目 → 统一 search 接口
│   ├── llm.py               # DeepSeek 客户端(T7): env Key/.env/缓存/三级解析/退避重试
│   ├── generate.py          # 生成层(T7): prompt 组装/few-shot 防泄漏/重试/引用校验
│   ├── evaluate.py          # Recall@k / MRR + 答案层(标签/拒答/引用/解析) + rfid 查表
│   ├── qaset_analysis.py / eval/validate_qaset.py / eval/calibrate_qaset.py
│   │                        # 评测集构建/校验/层标定工具链
│   └── analyze_truncation.py# 512 截尾影响评估（三级口径 + A/B/C 分组对照）
├── data/user_dict.txt       # jieba 自定义词典（进 zip）
├── data/graph_*.json[l]     # 图谱派生数据(进 git 不进 zip; 重建链路见"图谱管线"节)
├── cache/                   # 可再生缓存(不进 git): emb_*.npy 向量 + llm_*.json 响应
├── eval/qaset_v0.json       # 正式评测集 90 题定稿（冻结；数字唯一来源）
├── eval/qaset_seed.json     # 种子评测集 10 题（冒烟用，数字不进报告）
├── eval/answers.jsonl       # 生成回填(run_all 每次覆写; 实验产物不进 zip)
└── runs/                    # 实验日志（不进 zip）
```

## 设计要点（报告素材）

- **自定义词典三源**：73 个 `speaker_norm` 真值种子 + 人工术语清单（盖亚/守护者/
  民间警备队/风祭市/魔物…）+ 默认词典 OOV 观察（`瑚|太朗`、`键|之|少女`、`风|祭市`）；
  词频=语料频次、下限 100 防止 HMM 再切碎。`--check` 抽查切分。
- **检索/展示分离**：索引字段全角→半角（`２５岁→25岁`），证据展示永远用原文。
- **w2v_avg 负结果即发现**（框架 §4.2）：全题型垫底（R@3 0.137）——丢词序伤时序
  题、静态词义、OOV、长 chunk 平均稀释，四缺陷对应"为什么要上下文相关表示"。
- **bge 工程要素**：查询侧指令前缀（官方用法常量写死，方向冒烟实证前缀非空操作
  cos(q,P+q)=0.96/0.93）；文档向量缓存按模型+文本哈希隔离（改语料/换模型自动重建）；
  截尾统计落日志不改切分。
- **前缀方向敏感性**（域偏移下指令收益中性偏负，官方用法仍固化）：small ΔMRR
  -0.013（vs 无前缀）、base 反被反向配方超 0.04——指令为 web 检索调优；若接线
  接反**指标不会自我报警**，守门员是配线检验而非指标。
- **头号 BM25 失败案例**：查询「江坂和瑚太朗是什么关系」——定义句 c072 排第 25：
  ①词边界吞噬（「利害关系」当整词，c072 无独立「关系」token）；②干扰段登顶
  （c073 质疑关系破裂的字面块）；③主角名高频低 IDF。
  复现：`python src/index_bm25.py "江坂和瑚太朗是什么关系" -k 25 --arm jieba`。
- **bge literal 失守代表题**（Q06/Q09/Q32）：题面专名 BM25+词典 rank-1 锚定，
  bge 的 top-3 被主题相近的更长截尾块占据——专名判别力被池化稀释 + 截尾联合效应。
- **RRF 融合只吃排名**：BM25 分数与 bge 余弦不可比的问题被 `Σ1/(60+rank)` 绕开，
  无量纲无缩放超参；depth 压到 10 会伤 semantic_gap（0.3864→0.2955，深列表让语义臂
  晚名次 gold 仍能浮出）——敏感性记录而非调参依据（防测试集泄漏）。
- **拒答 prompt 约束在对抗设定下全部成立**：18 道不可回答题检索全部返回了字面相似
  非空证据，模型 18/18 正确拒答——"检索必返回相似块→误答"的计划预期风险没有发生；
  真正的失分反过来在可答题过度拒答（7/18，5 检索失败 + 2 部分证据）。
- **引用规则 (b) 的泄漏面守卫**（桩测实证）：引用 chunk 必须 ∈ 本题检索块（不含
  few-shot 示例块）——模型若引用示例块回答本题，会被确定性判 not_in_prompt，
  "答案来自示例而非本题证据"的泄漏在指标层可见。
- **否定裁决的缺证边界**（Q45/Q51 诚实注记）：gold=no 且检索 miss 的两题，模型从
  相邻证据推出正确否定（答文甚至是缺证口吻但 verdict 填 no）——"缺证即 no"与
  "缺证即 unknown"的边界协议未完全约束，留给阶段五 judge 口径讨论。
- **成本-延迟杠杆**（任务书拉分点实测）：系统提示+few-shot 前缀稳定 → DeepSeek
  前缀缓存命中 43%（129,536/299,383 输入 token），叠加本地响应缓存，全量 $0.03、
  重跑零费用；P50=1.48s/P95=2.18s（temperature=0 + 思考显式关，无 reasoning token）。

## 已知局限与数据备注

- **512 截尾（三级口径，勿只引 38.7%）**：块级 187/483=38.7%；题级 gold 暴露
  56/72=77.8%；gold 证据**完整落入截尾**仅 5/72=6.9%。分组对照（A=gold 未截 /
  B=被截证据存活 / C=证据全丢）：bge_small R@3 0.573/0.276/0.233，bm25_jieba
  0.385/0.330/0.267——gold 被截的题上 bge 显著失分而 BM25 几乎不动（BM25 全文
  入索引无 512 限制）。**截尾与长文池化稀释完全共线**，组间差是联合效应；
  切分不迁就模型（真值结构决定），详见 `runs/<ts>/truncation_analysis.json`。
- 3 个 rfid（3281 / 14785 / 14898）不在任何 chunk 区间内（Day 1 切分时随选择肢
  边界丢弃的碎片）；金标若命中会被 evaluate 报错指出。
- `千早`/`露西娅` 仅作为结尾场景的说话人【标签】出现，正文叙述从不点名——
  纯文本检索对这类"只在标签里的名字"有盲区；`index.index_speakers: true`
  可把 speakers 元数据并入索引文本（消融开关）。
- `qaset_v0.json` 已冻结（v0-20261023）：7 条冻结模式机检报错已终裁全部为启发式
  误报（判据记录见 `../Rewrite_terra/report/阶段报告_T5_收口.md`），集子不变更。

## 红线自查（提交前）

不进 zip：`runs/`、`../Rewrite_terra/data/`、`project/data/graph_*.json[l]`（图谱派生数据，
进 git、重建链路见"图谱管线"节）、剧本/反汇编文本、虚拟环境、**API Key
（`DEEPSEEK_API_KEY` 环境变量 / `project/.env`，绝不进仓库）**、模型权重（`models/`）、
向量与 LLM 响应缓存（`project/cache/`，可再生）、`eval/answers*.jsonl`（生成实验产物，
重跑可再生）。`data/user_dict.txt`、`.env.example` 与 `eval/qaset_*.json` 评测集是
代码性产物，进包。生成阶段三条红线落地：检索/展示分离（展示永远 chunk 全文原文）、
few-shot 证据排除评测集 gold chunk 且题目全部自编（防泄漏）、引用规则校验确定性
（孤儿 rfid catch 不崩评测）。阶段四新增红线落地：跨 chunk 边全部确定性推导
（不采信 LLM 标注跨卡引用，leads_to_hint 不入召回）、图臂无 Key 降级路径明确
（实体索引零 LLM 独立可用，runs/20261026-222237 实证）。

## 下一步（阶段五/六）

阶段五（LLM-as-judge 答文质量/支持性引用复核）：以阶段四图谱开/关两组 `answers.jsonl`
（runs/20261026-223110 与 222529）为评估对象；judge 口径需裁决"缺证即 no vs unknown"边界
（阶段三 Q45/Q51 诚实注记，阶段四 Q51 翻错强化该议题）。阶段六（k∈{3,5,10} 消融 ×
图谱开/关 × rerank 开关、终报）：图谱开的答案层回退对证据组合高度敏感（4 翻错中 2 题
仅第 5 证据位一个块之差），k 消融可量化该敏感度；图谱开/关默认取舍以 judge 数据定夺。
图谱后续机制方向（future work）：稠密卡片向量对准 str/agg 未达线与 Q10/Q46/Q49 三型
词法失配；RAPTOR-lite 按计划默认不做。
