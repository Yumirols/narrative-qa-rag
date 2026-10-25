# 阶段报告:RAG 主流程与提示设计——T6~T8 收口

> 对应《阶段计划_阶段三_RAG主流程.md》T6(混合召回+RRF)/T7(客户端+生成)/T8(接线+收口)。
> 数字数据源:T6 = `project/runs/20261025-170131/`;T8 官方全链 = **`runs/20261025-184856/`**
> (两跑 `metrics.json`+`answers.jsonl` sha256 逐字节一致,`87ebf117f05188fabf67d7bc2a60003a67f16e89b6468fca51cf458aad177914`);
> 首网延迟/成本口径 = `eval/answers_generation_log.json`(T7 CLI 首次真实调用);
> 无 Key 降级 = `runs/20261025-185352/`;冒烟 = `eval/answers_smoke*.jsonl/json`。

---

## 0. 一句话结论

**T6~T8 收口,阶段计划验收四条全绿**:hybrid RRF 第六列总体 R@5=**0.5194**/MRR=**0.5461**,
同时压过两个参照单臂(bge_base 0.4476/0.5125、bm25_bigram 0.4720/0.5381)——分层互补转化为
总体增益;90 题生成**解析失败率 0%、引用条级有效率 100%**;答案层 label_acc=**0.8056**、
**不可回答题拒答 18/18**(计划押注的最大预期风险反成满分);全量成本实测 **$0.0297**
(预算 $0.06~0.12 的一半以下,DeepSeek 前缀缓存命中 129,536 token 是主因);
`python run_all.py` 一条命令覆盖"六臂检索评测 → 90 题生成 → 答案层指标",无 Key 环境自动
降级为检索层全流程;7 道 verdict 错题全部完成 §5.3 六模式归因。

---

## 1. 验收标准对照(计划 §6 四条底线)

| # | 验收 | 状态 | 证据 |
| --- | --- | --- | --- |
| 1 | 一条命令跑通全链;无 Key 仍可完成检索层 | ✅ | 有 Key 全链 `runs/20261025-184856/`(六臂+生成+答案层);无 Key `runs/20261025-185352/`(生成段跳过并提示,检索层+答案层完整) |
| 2 | 缓存命中重跑 metrics/answers 逐字节一致 | ✅ | `runs/20261025-184349/` vs `184856/` 两跑 sha256 一致(`87ebf117…7914`);生成 90/90 缓存命中零网络 |
| 3 | 数字可溯源;成本延迟有实测汇总 | ✅ | §6 溯源表;成本 $0.0297 谷/$0.0593 峰,P50=1.48s/P95=2.18s(首网口径) |
| 4 | ≥3 失败案例六模式归因 | ✅ | §5:7 题全归因(5 召回失败 + 2 部分证据过度拒答),3 案例展开 |

指标对计划 §6 预期表:hybrid 总体 R@5 ≥0.50 ✅(0.5194);literal 层 R@3 ≥0.75 ✅(恰 0.75);
semantic_gap ≥0.43 ✗(0.3864,弱臂字面干扰所致,敏感性记录见 §2.3);verdict 标签准确率
0.70~0.90 ✅(0.8056);拒答正确率 0.6~0.8 **✅ 超预期**(1.0);引用有效率 ≥0.9 ✅(1.0);
解析失败率 <5% ✅(0)。

---

## 2. T6:混合召回 + RRF 融合(`src/index_hybrid.py`)

### 2.1 设计

- 臂构成按名引用 config 同级条目(单一配置源,臂超参不内联复制);run_all 传已建实例表
  `built`,hybrid 复用臂实例零重建(避免 w2v 重训练 3.7min/bge 重载模型);被引用臂须排在
  hybrid 前,违反显式报错。
- `score(d)=Σ_arms 1/(rrf_k+rank)`,默认 arms=[bm25_bigram, bge_base]、depth=20、rrf_k=60;
  只用排名不用原始分数——BM25 分数与 bge 余弦天然不可比的问题被绕开。同分 tie-break 按
  idx 升序(与单臂契约一致);`eval_retrieval` 零改动,自动成为主表第六列。

### 2.2 结果(`runs/20261025-170131/metrics.json`,官方 run `184856` 复核一致)

| 臂 | R@3 | R@5 | R@10 | MRR |
| --- | --- | --- | --- | --- |
| bm25_bigram(最佳单臂) | 0.4171 | 0.4720 | 0.5579 | 0.5381 |
| bge_base(计划参照臂) | 0.3704 | 0.4476 | 0.5809 | 0.5125 |
| **hybrid_rrf** | **0.4391** | **0.5194** | **0.6553** | **0.5461** |

分层:literal R@3=0.75(R@5 0.8333)、semantic_gap R@3=0.3864、structural R@5
0.3137→**0.4322**(bigram 强项继承);分题型四类三类最优(局部事实 0.6083/是否 0.4759/
跨场景 0.4426,时序 0.5324 略低于 bigram 0.5648)。**核心问题"互补性是否转化为总体增益"
的答案:是**——R@5 较最佳单臂 +0.047,R@10 +0.097。

### 2.3 敏感性记录(防测试集泄漏:先验默认不据评测集调整,仿 T4 前缀冒烟口径)

| 变体 | R@5 | MRR | semantic_gap R@3 |
| --- | --- | --- | --- |
| bigram+bge k=60 d=20(**默认,已过验收**) | 0.5194 | 0.5461 | 0.3864 |
| rrf_k=30 / k=10 | 0.5194 / 0.5430 | 0.5491 / 0.5593 | 0.3864 / 0.3864 |
| depth=10 | 0.5449 | 0.5450 | **0.2955**(深列表让语义臂晚名次 gold 仍能浮出) |
| jieba+bge k=60 | 0.5251 | 0.5361 | 0.3182 |
| 3 臂 jieba+bigram+bge k=60 | 0.5390 | 0.5635 | 0.3409 |

总体增益对融合参数稳健(全部变体 R@5 ≥0.52 且优于最佳单臂);RRF 对 k 经典不敏感;
3 臂无总体优势。**维持先验默认**。确定性:两次独立构建(含 bge 重载)metrics 逐字节一致。

---

## 3. T7:LLM 客户端 + 生成层(`src/llm.py` / `src/generate.py`)

### 3.1 设计要点

- **客户端**:Key 只走 `DEEPSEEK_API_KEY`(环境变量或 project/.env 手写解析器,.env 已进
  .gitignore,仓库只有 .env.example);`temperature=0` + `extra_body={"thinking":{"type":"disabled"}}`
  显式关思考;seed 端点不支持、不发送;**响应缓存**键 = sha256(model+thinking+temperature+
  max_tokens+response_format+messages),落 `cache/llm_<hash16>.json`(可再生不进 zip);
  json_object + 三级解析降级(直接 loads→剥围栏→括号配平,含字符串内转义花括号);网络瞬断
  退避重试 ×3。
- **生成层**:证据块 `[证据i | chunk_id | rfid 起止 | speakers]` 头部 + chunk 全文原文
  (检索/展示分离红线);系统提示按框架 §5.2(强制 JSON 协议/拒答规则/非是非题 verdict=unknown);
  **few-shot 3 例**(正常作答/是非题 yes/拒答)运行时构建,证据块取真实 chunk 且排除当前
  题集全部 gold chunk(v0 141 块、seed 4 块,两题集实测零交集),拒答话题避开 Q29"生日"题
  (示例若泄漏 gold verdict 等于教模型答对不可回答题);解析失败重试一次(附协议纠错提示);
  rfid 归一(int/[起,止] 兼容)+ 引用两规则校验(孤儿 rfid catch KeyError 标 invalid 不崩评测;
  引用必须 ∈ 本次证据集合——该规则天然守住"答案引用 few-shot 示例块"的泄漏面,桩测实证);
  answers 逐题落盘、行内零非确定字段。
- **成本/延迟**:usage 私有字段(hit/miss/reasoning)防御读取;三分计价(命中 $0.003/未命中
  $0.15/输出 $0.60 每 1M,谷/峰双口径)。

### 3.2 §3 规格核对(真实 API,含一处实测修正)

| 项 | 结果 |
| --- | --- |
| 响应 model 字段 | `deepseek-flash` ✅ |
| json_object | 10/10 与 90/90 全 direct 路径解析,零降级零重试 ✅ |
| usage 细项 | prompt_cache_hit/miss_tokens 齐全;总输入 33.8K 中前缀命中 13.1K(冒烟)——"系统提示+few-shot 稳定则复跑输入近零成本"实证 ✅ |
| **reasoning_tokens** | **实测修正 §3:思考关闭时端点不返回该字段(整体缺席,非返回 0)**;守门判据改为"字段缺席+短输出(49~200 token)+~1.4s 延迟"三证 ✅ |
| temperature/thinking | thinking disabled 显式发送;输出 6,315 token/90 题(均值 70)与 1.5s 延迟侧面印证无推理 token ✅ |

### 3.3 冒烟与正式集

| | 冒烟 10 题(seed) | 正式 90 题(v0) |
| --- | --- | --- |
| 解析失败率(重试后) | 0%(10/10 direct) | 0%(90/90 direct) |
| 引用条级有效率 | 1.0(9/9) | 1.0(83/83) |
| 零引用题(拒答行) | 3 | 40 |
| token(输入/前缀命中/输出) | 33,754 / 13,056 / 551 | 299,383 / 129,536 / 6,315 |
| 成本 | $0.0035(谷) | **$0.0297(谷)/$0.0593(峰)** |
| 延迟(首网) | P50 1.39s / P95 2.49s | P50 **1.48s** / P95 **2.18s** / 均值 1.544s |
| 缓存重跑 | 10/10 命中,answers 逐字节一致 | 90/90 命中,answers+metrics 逐字节一致 |

### 3.4 所用模型、调用量、单价与总花费估算(框架 §6.4:调用量与费用必须写明)

**所用模型**:`deepseek-flash`(服务版本 V4.1-Flash-1017,响应 `model` 字段核对);
采样配置:temperature=0、思考模式 disabled、max_tokens=1024、response_format=json_object。

**调用量与费用**(LLM 网络调用;本地缓存命中零网络零计费,不列入):

| 批次 | 调用次数 | 输入 token(前缀缓存命中) | 输出 token | 费用估算(标准价) |
| --- | --- | --- | --- | --- |
| 冒烟 qaset_seed(10 题) | 10 | 33,754(13,056) | 551 | $0.0035 |
| 正式 qaset_v0(90 题) | 90 | 299,383(129,536) | 6,315 | $0.0297 |
| **合计** | **100** | **333,137(142,592,占 42.7%)** | **6,866** | **$0.0331** |

解析失败重试 0 次(90/90 direct,协议纠错重试机制备而未用);冒烟二次运行、正式集
二次运行、T8 官方 run 的重跑全部命中本地响应缓存,零计费调用。

**单价**(flash,USD/1M token,2026-10-25 官方定价表):输入未命中 **$0.15** /
输入缓存命中 **$0.003**(≈未命中的 1/50)/ 输出 **$0.60**;峰时(工作日北京时间
09–12 / 14–18 点)×2,其余时段标准价——本次两批均为周日(2026-10-25)标准价口径,
峰时参照值 $0.0662。

**平台账单核对**(DeepSeek 开放平台用量面板,2026-10-25 用户截图):**Cost ¥0.22 CNY /
API requests 100 / Tokens 340,003**。三项与本地 `generation_log.json` 完全对账:
requests 100 = 冒烟 10 + 正式 90(重试 0、缓存重跑 0);tokens 340,003 = 333,137 输入 +
6,866 输出**逐 token 一致**;实付 ¥0.22 与标准价估算 $0.0331(按平台 CNY 计费牌价
折算)吻合。单题均值 $0.00033(正式集,约 ¥0.0024/题);本阶段开发全程总花费
**¥0.22**,远低于计划预算"单次全量 $0.06~0.12(¥1 以内)"。

---

## 4. T8.1/8.2:run_all 接线与指标扩展

- `run_all.py` 生成阶段插在检索评测之后:复用已建检索臂实例(零重复加载),产出
  `runs/<ts>/answers.jsonl` + `generation_log.json` 并回填 `eval/answers.jsonl`;Key 缺失
  整段跳过并提示(与"answers 不存在则跳过"同一模式);`env_info.json` 增记 openai 版本、
  模型/采样配置、执行状态与生成 summary(Key 从不落盘)。
- `evaluate.py`:`eval_label_accuracy` 内聚 `eval_citations`(条级/题级引用有效率,从
  answers 行字段直接算,向后兼容)与 `eval_parse_failures`(解析失败率);answers 行 schema
  不变,新增 `n_invalid_citations` 等字段向后兼容。
- 生成主流程抽为 `generate.run_generation()`(CLI 与 run_all 共用,llm_client 注入点供离线桩测)。

---

## 5. T8.3:首轮失败归因(框架 §5.3 六模式)

### 5.1 Recall@k 与答对与否的相关性(原理点③首批数据,yesno 18 题)

| | verdict 正确 | verdict 错误 |
| --- | --- | --- |
| gold 命中 top-5(11 题) | 9 | 2 |
| gold 完全 miss(7 题) | 2 | 5 |

**P(答对|gold 命中)=9/11=0.818,P(答对|gold 全 miss)=2/7=0.286**——检索是否把 gold
送进证据集合是生成侧裁决正确性的主导因子(相对风险 ≈2.9 倍)。另:7 道错题 verdict 全部
是 unknown(零例 yes/no 裁错)——**本批失败模式清一色"该答而拒答",无一例"答错方向"**。

### 5.2 七道错题六模式归因(全量,>3 案例)

| qid | 层/题型 | gold 入 top-5 | §5.3 模式 | 说明 |
| --- | --- | --- | --- | --- |
| Q47 | semantic_gap/yesno | 0/1 | **召回失败** | 「加岛樱顺着魔物线索查明送信人」——动作链换说法,两臂都未命中 gold |
| Q46 | semantic_gap/yesno | 0/1 | **召回失败** | 「江坂丢下不管的铺子盼的是赚钱吗」——隐喻表述,字面与语义双脱靶 |
| Q10 | semantic_gap/yesno | 0/2 | **召回失败** | 「瑚太朗最终除掉键之少女了吗」——结局性事件,检索返回的全是前期相关块(模型诚实拒答,答文甚至复述了证据 5"键之少女已能对话自称篝") |
| Q49 | semantic_gap/yesno | 0/2 | **召回失败** | 地下城市基础设施——场景细节换说法 |
| Q55 | structural/yesno | 0/3 | **召回失败(跨 chunk 结构)** | 「结尾五名少女谁曾同属一支小队」——跨场景成员关系,五臂同败层的再现,图谱动机 |
| Q13 | structural/yesno | **2/5** | **top-k 截断/分散证据聚合失败→拒答** | 洲崎线 5 个 gold 只召回 2 个,部分弧线不足以裁决,模型拒答(gold=no) |
| Q48 | structural/yesno | **1/3** | **top-k 截断/分散证据聚合失败→拒答** | 地铁救援 3 个 gold 只召回 1 个,单块只有袭击无救援,拒答(gold=yes) |

- **干扰段误导:0 例**(90 题中无一例因干扰块把 yes/no 裁反);**位置偏差:未测**(需
  证据顺序消融,原理点④备用,阶段五后);**拒答失败:0 例**(不可回答 18/18,且检索全部
  返回了字面相似的非空证据——计划预期的主要风险"检索必返回相似块→误答"没有发生,
  拒答 prompt 约束在对抗设定下全部成立)。
- 诚实注记:gold 全 miss 却答对的 2 例(Q45/Q51)gold 均为 no——模型从相邻证据推出正确
  否定(Q51 答文是"证据中没有…今宫只是语塞"的缺证推理但 verdict 填了 no);否定裁决的
  "缺证即 no"与"缺证即 unknown"边界在协议上未完全约束,是 0.818 而非更高的部分来源,
  留给阶段五 judge 口径讨论。

### 5.3 对阶段四的直接动机

4/7 错题在 semantic_gap 层(换说法召回失败)、2/7 在 structural 层(跨 chunk 聚合)——
与 T5 分层互补表、T6 structural 层六臂同败(≈0.2~0.43 R@5)相互印证:**检索侧的语义鸿沟
与结构缺失是答案层失分的上游主因**,图谱扩展召回(实体-关系-事件边)恰好对准这两层。

---

## 6. 数字溯源表

| 数字 | 日志位置 |
| --- | --- |
| 六臂检索指标(含 hybrid 第六列) | `runs/20261025-184856/metrics.json`(与 170131 一致;两跑 sha256 `87ebf117…7914`) |
| hybrid 敏感性表(§2.3) | 本报告表;过程记录见《阶段计划_阶段三》6.3 注记 |
| label_acc / 拒答正确率 / 过度拒答 | `runs/20261025-184856/metrics.json` → label_accuracy |
| 引用有效率 / 解析失败率 | 同上(label_accuracy.citations / .parse_failure);逐题明细 `generation_log.json` |
| token / 成本 / 延迟 | 首网口径 `eval/answers_generation_log.json`(P50/P95);官方 run 为 90/90 缓存命中(`runs/20261025-184856/generation_log.json`) |
| 调用量 / 单价 / 实付(§3.4) | 平台用量面板截图(用户,2026-10-25:¥0.22 / 100 次 / 340,003 token)+ `eval/answers_{smoke_,}generation_log.json` 逐 token 对账 |
| 失败案例 top-k 与答文 | `runs/20261025-184856/{retrieval_results.json, generation_log.json, answers.jsonl}` 按 qid 索引 |
| §3 规格核对 / reasoning_tokens 修正 | `eval/answers_smoke_generation_log.json`(usage 原始字段) |
| 无 Key 降级 | `runs/20261025-185352/`(生成段跳过,检索层+答案层完整) |

## 7. 复现命令(Git Bash,cwd=`E:/dev/NLP/project`)

```bash
pip install -r requirements.txt            # openai==3.37.0 已锁
export DEEPSEEK_API_KEY=sk-…               # 或 project/.env(不进 git)
python run_all.py --config configs/main.yaml
# 产物: runs/<ts>/{metrics.json, answers.jsonl, generation_log.json,
#         retrieval_results.json, env_info.json, hybrid_config_hybrid_rrf.json, …}
# 首跑全量生成 ≈$0.03;缓存命中重跑零费用且 metrics/answers 逐字节一致
# 无 Key: 同命令自动跳过生成段, 检索层全流程照常(runs/20261025-185352 实证)
python src/generate.py --qaset eval/qaset_seed.json --out eval/answers_smoke.jsonl   # 单独冒烟
python src/index_hybrid.py "盖亚和守护者是什么关系？" -k 3                            # hybrid demo
```

## 8. 遗留与衔接

- **本阶段无遗留悬置项**。rerank 仅 config 开关未实现(计划范围外);k∈{3,5,10} 消融轴、
  证据位置消融留给阶段六;LLM-as-judge 与长上下文对照留给阶段五。
- **阶段四(图谱)**:动机证据链齐备——structural 层六臂同败 + Q55/Q13/Q48 三案例 +
  semantic_gap 4 案例召回失败;图谱扩展召回目标层明确。
- **阶段五(评测/judge)**:answer 文本质量与 gold_answer 的语义比对、支持性引用复核
  (LLM judge)、原理点③扩展(全 90 题 Recall@k 相关性)从本轮 answers/generation_log 起算。
- 风险提示:DeepSeek flash 定价与思考模式默认值均以 2026-10-25 官方文档为准,复跑前如
  隔久建议复核(本报告 §3.2 的实测修正即文档与端点的偏差实例)。
