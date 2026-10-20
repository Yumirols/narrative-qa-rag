# NLP 算法最小实现(第 1–15 讲)+ 课程项目工作区

对应课程:《自然语言处理》2026—2027 秋季学期
课程主页:<https://6ccfkuh9.qwenwork.host/>

本仓库包含两部分内容:

1. **第 2–15 讲算法最小实现**:每个文件 = 一个可独立运行的脚本,依赖只有 numpy,
   注释里保留了公式推导,配合课件食用(见下文「目录结构与算法清单」)。
2. **课程项目语料工作区 `Rewrite_terra/`**:galgame 剧本叙事问答(任务书方向 A:领域 RAG 问答)
   的数据提取链路与行级真值语料(见「课程项目语料工作区」一节)。

> 第 1 讲是导论(任务全景 / 发展史),没有需要实现的算法,直接读课件即可。
> 第 9 讲(序列标注)、第 13 讲(微调与对齐)、第 16 讲(Agent)暂未单独实现。

## 仓库总览

```
NLP/
├── README.md
├── requirements.txt                 # 教学代码依赖:仅 numpy
├── 01_text_processing/ … 09_rag/    # 第 2–15 讲最小实现,共 13 个脚本(见下表)
├── 课程项目任务书_要点整理.md          # 任务书规则速查:选题、评分细则、交付物、合规红线、自查清单
├── 项目知识框架_剧本问答.md            # 项目知识地图:任务定义、总体架构、五类问题、评测设计与原理点实验
└── Rewrite_terra/                   # 课程项目语料工作区(详见下文专节)
    ├── seen08000.dat                # 游戏剧本二进制(版权文本,禁止打包进提交 zip)
    ├── seen08000.dat.csv / .txt     # 反汇编中间产物(.txt 与 structure/ 下同名文件相同,工作副本)
    ├── Terra_汉化文本.txt             # 带说话人前缀的整部纯文本(~19.5 万汉字)
    ├── _make_txt.py                 # dat.csv → Terra_汉化文本.txt
    ├── src/parse_disasm.py          # 反汇编文本 → data/ 行级真值结构(核心提取脚本)
    ├── structure/                   # 反汇编输入、说话人核对表与辅助脚本
    └── data/                        # 提取产物:项目真正的输入语料(lines / scenes / choices)
```

## 目录结构与算法清单

| 讲次 | 模块 | 主题 | 核心算法 | 代码 |
|------|------|------|----------|------|
| 2 | 一、文本处理基础 | 文本清洗与正则表达式 | 正则抽取、清洗、脱敏 | `01_text_processing/regex_cleaning.py` |
| 3 | 一、文本处理基础 | 编辑距离与拼写纠错 | 最小编辑距离(DP + 回溯)、纠错 | `01_text_processing/edit_distance.py` |
| 4 | 一、文本处理基础 | 分词与子词表示 | 正向最大匹配、BPE | `01_text_processing/tokenization_bpe.py` |
| 5 | 二、词表示与语言模型 | 词向量与语义表示 | 共现矩阵、PPMI、Word2Vec(Skip-gram + 负采样) | `02_word_representation/word2vec_skipgram.py` |
| 6 | 二、词表示与语言模型 | 神经网络基础 | 感知机、MLP、反向传播、梯度检查 | `03_neural_networks/mlp_backprop.py` |
| 7 | 二、词表示与语言模型 | 统计语言模型 | N-gram、Add-k 平滑、困惑度、句生成 | `04_language_models/ngram_lm.py` |
| 8 | 二、词表示与语言模型 | 神经语言模型 | RNN / LSTM(手写前向 + BPTT)、字符级 LM | `04_language_models/rnn_lstm_lm.py` |
| 10 | 五、注意力与生成 | Seq2Seq 与注意力机制 | 注意力三步(打分→softmax→加权)、四种打分函数、软对齐、信息瓶颈 | `05_attention_seq2seq/attention_seq2seq.py` |
| 10 | 五、注意力与生成 | 解码与评测 | greedy / beam search(长度归一化)、BLEU(截断计数 + BP) | `05_attention_seq2seq/decoding_bleu.py` |
| 11 | 六、Transformer | Transformer 原理精讲 | 缩放点积注意力(√d_k 实验)、多头、causal mask、位置编码、LN/RMSNorm、置换等变 | `06_transformer/transformer_minimal.py` |
| 12 | 七、预训练语言模型 | GPT 与 BERT 的预训练目标 | AR vs MLM(15% + 80/10/10)、因果/双向掩码、手推反向传播 + 梯度检查、四种任务头 | `07_pretrained_lm/pretrain_objectives.py` |
| 14 | 八、提示工程与评估 | 解码策略与 LLM 评测 | temperature / top-k / top-p、六大要素提示、自洽性投票、评委位置偏差校正 | `08_prompting_eval/decoding_prompting.py` |
| 15 | 九、RAG 与推理优化 | RAG 全链路 | 切块(固定/递归)、BM25+向量混合检索、RRF 融合、重排、三条约束、四类评估指标 | `09_rag/rag_minimal.py` |
| 15 | 九、RAG 与推理优化 | 推理优化 | KV Cache 显存账、INT8/INT4 量化、投机解码、IVF 近似检索 | `09_rag/inference_cost.py` |

## 快速开始

```bash
pip install -r requirements.txt

python "01_text_processing/regex_cleaning.py"     # 正则抽取 / 清洗 / 脱敏
python "01_text_processing/edit_distance.py"      # 编辑距离 + 拼写纠错
python "01_text_processing/tokenization_bpe.py"   # 中文分词 + BPE 子词
python "02_word_representation/word2vec_skipgram.py"  # 训练词向量 ~10 秒
python "03_neural_networks/mlp_backprop.py"       # 感知机 vs MLP 解 XOR
python "04_language_models/ngram_lm.py"           # bigram + 平滑 + 困惑度
python "04_language_models/rnn_lstm_lm.py"        # 字符级 RNN/LSTM LM ~10 秒
python "05_attention_seq2seq/attention_seq2seq.py"    # 注意力三步 + 手算复现
python "05_attention_seq2seq/decoding_bleu.py"    # greedy/beam + BLEU 例题
python "06_transformer/transformer_minimal.py"    # 自注意力手算 + 三组数值实验
python "07_pretrained_lm/pretrain_objectives.py"  # AR vs MLM 对比训练 ~几秒
python "08_prompting_eval/decoding_prompting.py"  # 解码策略 + 评测模拟(离线)
python "09_rag/rag_minimal.py"                    # mini-RAG 全链路(离线可跑)
python "09_rag/inference_cost.py"                 # KV Cache / 量化 / IVF
```

每个脚本自带演示数据,跑起来会打印分步结果;所有随机过程都设了随机种子,结果可复现。

## 学习路径建议

1. **第 2–4 讲(预处理)**:直接跑通,然后换自己的文本观察行为。
   重点:第 3 讲务必手画一次 dp 表;第 4 讲注意 FMM 切错「研究生命起源」的原因。
2. **第 5 讲(词向量)**:先看计数派(PPMI),再看预测派(Skip-gram),
   对比两者对「猫/狗」「学生/老师」相似度的刻画。
3. **第 6 讲(神经网络)**:先跑梯度检查确认手推公式正确,再照注释手推一遍反向传播;
   观察感知机为什么解不了 XOR、两层 MLP 为什么可以。
4. **第 7 讲(N-gram)**:对比 k=0(会爆零)/ k=1 / k=0.01 的困惑度,理解平滑的动机。
5. **第 8 讲(RNN/LSTM)**:先看两个 Cell 的梯度检查,再对照注释读 BPTT 循环;
   对比 RNN 和 LSTM 训练同一语料的损失下降速度,体会「门控解决梯度消失」。
6. **第 10 讲(注意力与解码)**:先复现课件的手算例题(s=[1.0,0.5] 那两组),
   再用两分支概率表对比 greedy 与 beam;把 BLEU 例题(today→the)重算一遍看截断计数。
7. **第 11 讲(Transformer)**:对照课件第 19–21 页手算「猫追鱼」,再跑 √d_k、mask、
   置换等变三组实验;观察去掉缩放后 softmax 权重逼近 one-hot、梯度趋近 0。
8. **第 12 讲(预训练)**:看同一小模型在 AR 与 MLM 两种目标下的信号密度与收敛速度,
   体会「目标函数决定能力边界」:AR 会续写,MLM 只会完形填空。
9. **第 14–15 讲(提示 / 评估 / RAG)**:先看三版提示的差异与三条生成约束,
   再读 mini-RAG 的六步链路;用四类指标定位「瓶颈在检索端还是生成端」。

## 课程项目语料工作区(Rewrite_terra/)

课程项目选题为**方向 A:领域 RAG 问答**,语料取自 galgame《Rewrite》Terra 线汉化剧本
(约 19.5 万汉字)。该目录负责把游戏剧本二进制还原成**行级真值结构**,供后续 RAG 链路使用。

### 提取链路

```
seen08000.dat(剧本二进制)
  │  siglus-ssu -x --disam(外部工具,产物已存为 structure/seen08000.dat.txt)
  ▼
structure/seen08000.dat.txt(反汇编文本,~13 MB)
  │  src/parse_disasm.py + structure/speaker_map.tsv(说话人归一化映射)
  ▼
data/lines.jsonl    行级真值:11,033 行(4,358 条台词 + 6,675 条旁白/其他),
                    每行含说话人原文/归一化名、是否台词、源文件行号、语音/BGM 等字段
data/scenes.jsonl   480 个场景 chunk:由边界信号切分,含起止行号、说话人列表、BGM/BG 等
data/choices.json   选择肢节点
```

`structure/` 内另有提取过程的辅助脚本与核对材料:
`_search_engine_strings.py`(在引擎/配置文件中定位说话人字符串)、
`_make_speaker_stats.py` 与 `_make_speaker_xlsx.py`(统计 #NAMAE 说话人频次并生成人工核对表
`speaker_labels_filled.xlsx`)、`namae_map.tsv`(全部说话人 188 项)、
`speaker_map.tsv`(Terra 线归一化映射 83 项)、`gameexe_orig.ini / gameexe_chs.ini`(原版/汉化配置对照)。
`data/seen08000_jp.ss` 为日文原版场景脚本,可作对照。

### 注意事项

- **版权红线**:剧本为版权文本(Key/VisualArt's),按任务书要求**不打包进提交 zip**,
  只提交提取脚本 + 来源声明,报告写明仅课程使用。
- 提取脚本内使用的是绝对路径(如 `E:/dev/NLP/Rewrite_terra/...`),换机器运行需先改路径;
  `_make_txt.py` 引用的 `E:/dev/NLP/seen08000.dat.csv` 现已移入本目录,路径待同步。
- `structure/` 与根目录下的 `seen08000.dat.txt` 是同一份反汇编产物的副本,改脚本时以
  `structure/seen08000.dat.txt`(parse_disasm.py 的输入)为准。

## 与课程项目(RAG 剧本问答)的衔接

项目规则速查见 `课程项目任务书_要点整理.md`(评分细则、交付物、红线),
技术路线与知识梳理见 `项目知识框架_剧本问答.md`。涉及的课程章节与对应代码:

| 课程内容 | 对应代码 | 在项目中的用途 |
|---|---|---|
| 第 2 讲 正则清洗 | `01_text_processing/regex_cleaning.py` | 清洗脚本基础(说话人归一化、全角转半角) |
| 第 4 讲 分词与 BPE | `01_text_processing/tokenization_bpe.py` | 原理点①:子词/字符级检索对照 |
| 第 5 讲 词向量 | `02_word_representation/word2vec_skipgram.py` | 基线②:词向量平均检索直接复用训练 |
| 第 7 讲 N-gram | `04_language_models/ngram_lm.py` | 困惑度概念(备用原理点素材) |
| 第 10 讲 注意力 | `05_attention_seq2seq/attention_seq2seq.py` | Q/K/V 软检索 = 图扩展召回的思想源头 |
| 第 10 讲 解码与 BLEU | `05_attention_seq2seq/decoding_bleu.py` | 评测协议参照:n-gram 指标的局限 → 为什么用 LLM-as-judge |
| 第 11 讲 Transformer | `06_transformer/transformer_minimal.py` | bge / LLM 的骨架;位置编码与 lost-in-the-middle 位置消融的原理基础 |
| 第 12 讲 预训练目标 | `07_pretrained_lm/pretrain_objectives.py` | 「上下文相关表示优于静态词向量」的机理(原理点②) |
| 第 14 讲 提示与评估 | `08_prompting_eval/decoding_prompting.py` | 输出协议(JSON+拒答)与 LLM-as-judge 偏差校正的最小原型 |
| 第 15 讲 RAG | `09_rag/rag_minimal.py` | 项目主链路同构:切块→混合检索→RRF→重排→三条约束→四类指标 |
| 第 15 讲 推理优化 | `09_rag/inference_cost.py` | 成本 / 延迟测量(任务书拉分点)与 KV Cache、量化、ANN 的量级直觉 |
