# 阶段三执行计划:RAG 主流程与提示设计(T6~T8)

> 依据:`项目知识框架_剧本问答.md` §5(RAG 主流程)/§7.3(结构化输出)/§13(复现性);阶段一/二收口状态见 `project/README.md` 与 `Rewrite_terra/report/阶段报告_T5_收口.md`。
> **已定决策(2026-10-25)**:生成 LLM 用 **DeepSeek**(OpenAI 兼容端点),模型 **`deepseek-flash`**;**API Key 走环境变量 `DEEPSEEK_API_KEY`**,绝不进代码/仓库/zip(任务书红线③)。调用格式已对照当日官方文档(api-docs.deepseek.com)核对,详见 §3。
> 计划日期 2026-10-25,预计工期 **1.5~2 天**(10-25 晚 ~ 10-27),与 10-30 截止前的阶段四(图谱)、五(评测/judge)、六(消融与报告)不冲突。

---

## 0. 现状与前置条件(全部已就位,开工前无需补作业)

| 前置 | 产物 | 阶段三的衔接点 |
| --- | --- | --- |
| 真值切分 | `Rewrite_terra/data/scenes.jsonl`:483 chunk,rfid 区间 1→15084,元数据(speakers/label/bgm/bg/choices) | 证据块引用锚点 + 展示原文 |
| 检索器抽象 | `src/retrievers.py` 工厂,统一 `search(query, k)` 接口,五臂(bm25_jieba / bm25_bigram / w2v_avg / bge_small / bge_base) | 混合召回两路直接复用;新增类型 = 加一个工厂分支 |
| 评测集 | `eval/qaset_v0.json` 90 题冻结(fact 20 / yesno 18 / temporal 18 / aggregate 16 / unanswerable 18;yesno+unanswerable 共 36 题带 `gold_verdict`;多数题带 `accept_criteria`) | 生成层输入 + 标签准确率金标 |
| 答案层接口 | `evaluate.py: eval_label_accuracy()` 读 `eval/answers.jsonl`(每行 `{qid, verdict, answer, evidence_rfids}`);`run_all.py` 已预留"存在即自动生效"钩子 | 阶段三只需产出该文件 |
| 混合召回动机 | 分层互补表:literal 层 bm25_jieba R@3=0.861 碾压 bge(0.500/0.583);semantic_gap 层 bge 反超 jieba 约 6 倍(0.432/0.386 vs 0.068);structural 层五臂同败(≈0.2) | RRF 融合的实证依据;structural 失败留给阶段四 |
| 关键口径 | structural 层 gold 均值 3.68 chunk → 生成 top-k 下界为 5;生成侧喂 chunk **全文**(不受 bge 512 截尾影响,README 已声明) | top-k 默认值与证据块组装规则 |

**尚不存在(= 本阶段全部新建)**:`src/llm.py`(客户端)、`src/generate.py`(生成)、hybrid 检索分支、config 的 generation 节、`.env` 加载与 `.env.example`;`requirements.txt` 无 `openai`。

---

## 1. 主要目标

1. **打通端到端管线**:问题 → 多路召回(BM25 ∥ bge)→ RRF 融合 → top-k 证据块 → Prompt 组装 → DeepSeek → 结构化 JSON 回答 → 落盘 `eval/answers.jsonl`。`python run_all.py --config configs/main.yaml` 一条命令覆盖"检索评测 → 生成 → 答案层评测"全链(红线①:跑不通 = 不及格)。
2. **Prompt 与输出协议工程**(框架 §5.2):强制 JSON 协议、拒答规则、few-shot 2~3 例、temperature=0;解析失败重试一次并记录失败率(§7.3)。
3. **引用校验**(框架 §5.2/§8.2):规则层确定性校验——rfid 查表存在 + 引用块确在本次送入 prompt 的证据集合内;支持性复核(LLM judge)留给阶段五。
4. **第一批答案层数字与失败归因**:verdict 标签准确率(36 题)、拒答正确率(18 题不可回答)、引用有效率、JSON 解析失败率、token 成本与 P50/P95 延迟;按 §5.3 失败模式表逐题归因,喂给阶段四动机与报告第⑥节。顺带产出原理点③素材(每题 Recall@k 与答对与否的相关性)。

**范围外(明确不做,防止蔓延)**:图谱扩展召回(阶段四)、LLM-as-judge(阶段五)、长上下文对照(阶段五后)、rerank(仅预留 config 开关,不默认实现)、评测集任何变更(已冻结;发现题目问题走 T5 终裁流程记录,不改题)。

---

## 2. 执行过程

### T6:混合召回 + RRF 融合(纯确定性、零 LLM,约半天)✅ 2026-10-25 完成

- [x] 6.1 `src/retrievers.py` 注册 `{type: hybrid}` 分支:按条目引用两臂(默认 `bm25_bigram + bge_base`,各取 top-`depth`=20),RRF 融合 `score(d) = Σ_arms 1/(rrf_k + rank)`,`rrf_k=60`,输出融合 top-k。**复用统一 search 接口,`eval_retrieval` 零改动,自动成为主表第六列**。
  - 实现注记:hybrid 条目需拿到两臂配置——条目内联完整臂配置,或由 run_all 传同级 retrievers 条目表,二选一,避免 config 双源漂移。
  - **落地**:实现 `src/index_hybrid.py:HybridRRFIndex`;arm 配置取"同级条目表"方案——config 里 `arms: [bm25_bigram, bge_base]` 按名引用,run_all 传已建实例表 `built`,复用实例不重建(避免 w2v 重训练/bge 重载模型);被引用臂须排在 hybrid 前,违反显式报错。同分 tie-break 按 idx 升序(与各单臂契约一致)。
- [x] 6.2 `configs/main.yaml` 追加 hybrid 条目(见 §4 配置草案)。**落地**:条目 `hybrid_rrf`(arms/depth/rrf_k 全量进 config);另按 6.4 加 `generation:` 节(retriever=hybrid_rrf, top_k=5;llm/prompt 子节 T7 落)。
- [x] 6.3 跑 `run_all.py` 出混合臂 Recall@k / MRR(总体+分题型+分层),**核心问题:互补性是否转化为总体增益**。验收:hybrid 总体不低于最佳单臂(bge_base R@5=0.4476 / MRR=0.5125);若不达标,检查臂选择(试 jieba 替换 bigram——literal 层 jieba 0.861 更强)与融合参数,记录调参过程。
  - **结果(runs/20261025-170131/metrics.json)**:hybrid_rrf 总体 R@3=0.4391 / **R@5=0.5194** / R@10=0.6553 / **MRR=0.5461**——R@5/MRR 同时压过两个参照臂(bge_base 0.4476/0.5125;实际最佳单臂 bm25_bigram 0.4720/0.5381),R@10 增益最大(+0.097 vs bigram)。分题型四类中三类最优(局部事实 0.6083 / 是否 0.4759 / 跨场景 0.4426,时序 0.5324 略低于 bigram 0.5648)。分层:literal R@3=0.75、structural R@5=0.3137→0.4322(bigram 强项继承),semantic_gap R@3=0.3864(低于预期假设 0.43,弱臂字面干扰所致,见敏感性)。**验收通过,且达成 §6 预期"总体 R@5 ≥ 0.50"**。
  - **敏感性记录(防测试集泄漏:先验默认不据评测集调整,仿 T4 前缀冒烟口径)**:rrf_k∈{10,30,60} R@5 仅 0.5194~0.5430 波动(k=10 略优但幅度小,RRF 对 k 经典不敏感);depth=10 伤 semantic_gap(0.3864→0.2955,深列表让语义臂晚名次 gold 仍能浮出);jieba 替换 bigram:literal 更强(0.8611)但 semantic_gap/总体更弱;3 臂无总体优势。**结论:默认配置已过验收,维持 bigram+bge_base / k=60 / d=20**。
  - 确定性:两次独立构建(含 bge 模型重载)hybrid metrics 逐字节一致;正式 run 数字与冒烟逐位一致。
- [x] 6.4 确定**生成用 top-k 默认值 = 5**(structural 层 gold 均值 3.68 决定下界);k∈{3,5,10} 留作阶段六消融轴,config 可改。**落地**:`configs/main.yaml` 新增 `generation:` 节(`retriever: hybrid_rrf`, `top_k: 5`),T7 接入 run_all 时消费。

### T7:LLM 客户端 + 生成模块(核心,约 1 天)✅ 2026-10-25 完成(含 90 题正式集)

- [x] 7.1 **`src/llm.py` 客户端层**(OpenAI 兼容 SDK,`pip install openai`):
  - Key 只读 `os.environ["DEEPSEEK_API_KEY"]`;支持可选 `.env` 文件(存在则用十余行手写解析器加载,**不引入 python-dotenv**,保持轻依赖);`.gitignore` 追加 `.env`,另提交 `.env.example`(仅含空键名)。
  - `temperature=0` + **思考模式显式关闭**(`extra_body={"thinking": {"type": "disabled"}}`——端点默认 enabled,不关则 temperature 无效、reasoning token 抬高延迟与费用);`seed` 端点不支持、不发送,日志记录该事实;**复现性真正兜底 = 响应缓存**:键 `sha256(model + thinking + temperature + max_tokens + messages)`,缓存 JSON 落 `project/cache/llm_<hash16>.json`(含原始响应、usage、延迟),相同 prompt 零二次请求——复现 + 省钱双红线,缓存可再生不进 zip(与 emb_*.npy 同一目录约定)。
  - `response_format={"type":"json_object"}` 优先(要求 prompt 含 "json" 字样并给格式示例——协议模板天然满足;官方注明偶发空 content,计入解析失败、由重试覆盖);**正则提取 JSON 块降级**(§7.3),降级事件落日志。
  - **落地(2026-10-25)**:`src/llm.py` 全部实现;缓存键 = sha256(model+thinking+temperature+max_tokens+response_format+messages 规范序列化,§3 超集);JSON 三级解析:直接 loads → 剥 ```json 围栏 → 括号配平扫描(含字符串内 `}` 转义处理),路径逐调用落日志;usage 私有字段(hit/miss/reasoning)走 getattr+model_extra 防御读取;另有:cost_usd 三分计价(命中 $0.003/未命中 $0.15/输出 $0.60 每 1M,谷/峰双口径)、网络瞬断退避重试 ×3(90 题连跑稳健性,超出计划的最小增件)。`.gitignore` 已加 `.env`;`.env.example` 已提交;`requirements.txt` 已锁 `openai==3.37.0`(T8.4 提前完成该项)。**离线验证**:三级解析 8 用例、缓存写→命中→model/thinking/messages 换键隔离(坑 10)、费用精度,全 PASS。
- [x] 7.2 **`src/generate.py` 生成层**:
  - 证据块格式:`[证据i | chunk_id | rfid 起止 | speakers]` 头部 + **chunk 全文原文**(全角保留;检索/展示分离红线——展示永远原文);
  - 系统提示按 §5.2 基线(见 §5),few-shot 2~3 例(一个正常作答、一个拒答);few-shot 的证据块取自语料真实 chunk,**不得引入评测集题目本体**(防泄漏);
  - 90 题全量生成(**不可回答题也走完整流程**——检索照样返回字面相似块,拒答只靠 prompt 约束,这正是要测的东西);
  - 解析:复用 `gold_intervals()` 的 int/区间兼容逻辑归一 `evidence_rfids`;失败重试一次(附"上一次输出不是合法 JSON"提示),两次解析的失败率分别落日志;
  - 每题记录:prompt 哈希、输入/输出 token、墙钟延迟 → 汇总单题费用、P50/P95(任务书成本-延迟拉分点)。
  - **落地**:few-shot 定 3 例(正常作答[非是非题→verdict unknown 演示协议约定]/是非题 yes/拒答),运行时构建:证据块取语料真实 chunk 且**排除当前题集全部 gold chunk**(v0 gold 141 chunk、seed 4 chunk,两题集实测零交集),提问元数据锚定自编,**拒答话题避开 Q29"生日"题**(坑 7 的实质面:示例若泄漏 gold verdict 等于教模型答对不可回答题);答案行零非确定字段(时间戳/延迟/缓存标记全在 generation_log),缓存命中重跑逐字节一致;answers 逐题落盘(崩溃不丢已完成题,缓存让重跑近零成本)。协议纠错重试一次已实现,首次/最终失败率分口径落日志。CLI:`python src/generate.py [--qaset …] [--out …] [--limit n]`,缺省 config 的 qaset/answers 路径。
- [x] 7.3 **引用规则校验**(generate 内,确定性):
  - (a) rfid 区间落在语料 [1,15084] 且能落到具体 chunk——`RfidLocator` 对 3 个孤儿 rfid(3281/14785/14898)抛 KeyError,**生成侧必须 catch 并标 invalid,不得让模型输出弄崩评测**;
  - (b) 引用 chunk ∈ 本次送入 prompt 的证据集合(防"引用了没给它的块");
  - 指标:引用有效率(条级)+ "全部引用有效"题占比(题级),两口径都落 metrics,报告引用时注明。
  - **落地**:`validate_citations` 两规则三失败原因(out_of_corpus / orphan_unlocatable[catch KeyError,坑 2] / not_in_prompt)+ malformed 归一计数(坑 3,不丢题);条级/题级两口径+零引用题数+原因分布落 generation_log.summary(T8.2 接线进 metrics.json)。**实测附带发现**:规则 (b) 的证据集合=本题检索块(不含 few-shot 示例块)——桩测试中引用 few-shot 块被正确判 not_in_prompt,即该规则天然守住"答案来自示例而非本题证据"的泄漏面(报告讨论点)。离线桩测覆盖:孤儿 3281/越界 15085/未提供区间/整块有效四类,全 PASS。
- [x] 7.4 冒烟:先 `eval/qaset_seed.json` 10 题跑通全链(解析、校验、落盘、缓存命中二次运行逐字节一致),再上 90 题正式集。冒烟同时核对 §3 规格:响应 `model` 字段为 flash 版本、`reasoning_tokens≈0`(思考确已关)、usage 含缓存命中细项、json_object 生效——**文档与端点不符时以实测修正 §3**。
  - **状态(2026-10-25)**:代码全链已用脚本化桩客户端离线验证(3 题走完"正常作答/坏 JSON 重试恢复/三类无效引用"全路径,二次运行缓存 4/4 命中、answers 逐字节一致、指标不变;few-shot 防泄漏与 prompt 组装实数据核验)。
  - **冒烟已完成(2026-10-25,真实 API,10 题 seed)**:解析 10/10 全 direct(json_object 生效,零重试);引用 9/9 条有效(3 题拒答零引用,无效原因空);费用 $0.0035(谷)≈¥0.025;延迟 P50=1.39s / P95=2.49s / 均值 1.53s;二次运行缓存 10/10 命中、零网络、answers 逐字节一致。§3 规格核对:model 字段=`deepseek-flash` ✓;usage 含 prompt_cache_hit/miss_tokens ✓(总输入 33.8K 中前缀命中 13.1K——"系统提示+few-shot 稳定→复跑输入近零成本"实证);**实测修正 §3**:思考关闭时端点**不返回** completion_tokens_details(reasoning_tokens 字段整体缺席,非返回 0)——守门判据改为"字段缺席 + 输出 49~200 token + 延迟 ~1.4s"三证。冒烟现象记录(报告素材):S01 gold 未入 hybrid top-5(c189/c211/c190/c208/c205,gold rfid 14)→ 模型诚实拒答——"召回失败→拒答"正是 T8.3 归因靶型;S10 不可回答题正确拒答。
  - **待办**:90 题正式集(`python src/generate.py`,预算 $0.06~0.12)——等用户看过冒烟质量后放行。
  - **90 题正式集已完成(2026-10-25,真实 API)**:解析 **90/90 全 direct**(失败率 0%,预期 <5%);引用条级 **83/83=1.0**(预期 ≥0.9)、题级全有效 0.5556(零引用 40 题=拒答行,详见 T8.2 接线);token 输入 299,383(**前缀缓存命中 129,536**)/输出 6,315;费用实测 **$0.0297(谷)/$0.0593(峰)**——低于 $0.06~0.12 预算一半以下,周日跑更接近半价口径;延迟 P50=1.48s / P95=2.18s / 均值 1.544s(90 次网络调用)。
  - **答案层首批数字(eval_label_accuracy,36 题)**:label_acc=**0.8056**(29/36,预期 0.70~0.90 内);**不可回答题拒答正确率 1.0(18/18,远超 0.6~0.8 预期——计划押注的最大风险点反成强项,拒答 prompt 约束在"检索必返回字面相似块"的对抗设定下全部成立)**;no_overrefusal_answerable=0.6111(11/18,7 题 yesno 过度拒答)。
  - **7 题错因归因(T8.3 前瞻)**:5 题检索失败(gold 0/N 入 top-5:Q10/Q46/Q47/Q49 均为 semantic_gap 层、Q55 structural)——与 T6 已知"弱臂字面干扰 semantic_gap"一致,图谱扩展召回(阶段四)的直接动机;2 题生成侧过保守拒答(Q13 gold 2/5 在证据内、Q48 1/3 在,仍拒答)——部分证据不足以让模型下 yes/no 判断,或与 top_k=5 截断有关,留给 T8.3 正式归因与消融。

### T8:答案层评测接线 + 阶段收口(约半天)✅ 2026-10-25 完成

- [x] 8.1 `run_all.py` 插入生成阶段(检索评测之后、落盘之前):产出 `runs/<ts>/answers.jsonl` 并回填 `eval/answers.jsonl`;`env_info.json` 增记 openai 包版本、模型名、temperature/thinking 配置;**`DEEPSEEK_API_KEY` 缺失时整段跳过并提示**——保持无 Key 环境一条命令仍可跑检索层(与现有"answers 不存在则跳过"同一模式)。
  - **落地**:生成主流程抽为 `generate.run_generation()`(CLI/run_all 共用,llm_client 注入点供桩测);run_all 复用已建检索臂实例(零重复加载);无 Key 降级实证 `runs/20261025-185352/`(生成段跳过并提示,检索层+答案层完整);摘要表过滤非检索臂块并新增答案层打印(首版把 label_accuracy 当检索臂遍历致 KeyError,崩溃在打印阶段、落盘无损,已修)。
- [x] 8.2 `evaluate.py` 扩展:在 `eval_label_accuracy`(已有:label_acc / refusal_correct_unanswerable / no_overrefusal_answerable)之上,新增引用有效率与解析失败率入 metrics;answers 行 schema 不变,新增字段向后兼容。
  - **落地**:`eval_citations`(条级/题级,从 answers 行字段直接算)+ `eval_parse_failures` 内聚进 `eval_label_accuracy`;answers 行新增 `n_invalid_citations`(缺省按 0,向后兼容);官方 run `runs/20261025-184856/metrics.json` → label_accuracy 块齐备。
- [x] 8.3 **首轮失败归因**(框架 §5.3 六模式:召回失败 / top-k 截断 / 分散证据聚合失败 / 干扰段误导 / 位置偏差 / 拒答失败):对答错题逐题查 `retrieval_results.json` 该题 top-k 是否含 gold,判定错在检索还是生成;产出 ≥3 个可归因案例(报告第⑥节素材 + 阶段四图谱动机);统计 Recall@k 与答对与否的相关性(原理点③首批数据)。
  - **落地**:7 道错题全部归因——**5 召回失败**(Q10/Q46/Q47/Q49 均为 semantic_gap 换说法、Q55 structural 跨 chunk)+ **2 top-k 截断/分散聚合→过度拒答**(Q13 gold 2/5、Q48 1/3 部分在仍拒答);干扰段误导 0 例(90 题无一是/非裁反)、位置偏差未测(原理点④备消融)、拒答失败 0 例(计划预期主风险反成满分)。相关性(原理点③):**P(答对|gold 命中)=9/11=0.818 vs P(答对|gold 全 miss)=2/7=0.286**;错题 verdict 全为 unknown(零例方向性错误)。诚实注记:gold miss 仍答对的 Q45/Q51 均为 gold=no 的缺证否定推理。详见阶段报告 §5。
- [x] 8.4 收口:`Rewrite_terra/report/阶段报告_T6~T8_RAG主流程.md`(含数字溯源表);`project/README.md` 更新(答案层指标表、DeepSeek/环境变量说明、目录树);`requirements.txt` + `openai`(锁版本)。
  - **落地**:阶段报告已写(验收四条对照 + T6/T7/T8 正文 + 六模式归因 + 溯源表 + 复现命令 + 阶段四五衔接);README 升级为 RAG 主流程收口口径(六臂表 + 答案层指标表 + DeepSeek 节 + 目录树 + 红线自查 + 设计要点 6 条新增);requirements 已于 T7 锁 `openai==3.37.0`。**阶段验收四条全绿**:①一条命令全链(`runs/20261025-184856`)/无 Key 降级(`runs/20261025-185352`)②两跑 metrics+answers 逐字节一致(sha256 `87ebf117…7914`)③全部数字可溯源、成本延迟实测④7 案例(>3)归因入报告。

---

## 3. DeepSeek 接入规格(已定,对照 2026-10-25 官方文档核对)

| 项 | 值 | 依据 |
| --- | --- | --- |
| 端点 | `base_url = https://api.deepseek.com`(OpenAI 兼容;另有 Anthropic 兼容端点,不用) | Quick Start |
| 模型 | **`deepseek-flash`**(当前服务版本 V4.1-Flash-1017;旧名 `deepseek-v4-flash` 仍可用、同价映射)。现行定价表仅 `deepseek-flash` 与 `deepseek-v4-pro` 两款,`deepseek-chat`/`deepseek-reasoner` 不在列——**代码与报告统一用 `deepseek-flash`**。上下文 1M、最大输出 384K | Models & Pricing |
| 认证 | 环境变量 `DEEPSEEK_API_KEY`(`Authorization: Bearer`);可选 `.env` 文件加载;**绝不进代码/config 提交物/zip** | Quick Start |
| 思考模式 | **显式关闭**:`extra_body={"thinking": {"type": "disabled"}}`(thinking 非 OpenAI SDK 标准参数,必须走 extra_body)。**端点默认 enabled**——不关则 `temperature` 无效,且 reasoning token 抬高延迟与费用。本项目是"给定证据的抽取/聚合/裁决"任务,非思考模式足够;thinking 开/关(`reasoning_effort: none/low/high/max`)留作后期可选消融,不进阶段三默认 | API Reference |
| 采样 | `temperature=0`(0~2,默认 1,**仅在非思考模式生效**);`max_tokens=1024`(非思考模式端点默认 8K;协议输出短,answer 限长在 prompt 内约束) | API Reference |
| 结构化 | `response_format={"type":"json_object"}`(要求 prompt 含 "json" 并给出格式示例——协议模板已满足);官方注明**偶发空 content**,计入解析失败、由重试覆盖;正则提取 JSON 降级常备 | JSON Output 指南 |
| 不支持/不使用 | **`seed` 不在端点参数表——不发送**(日志记录该事实,复现性由缓存兜底);`frequency_penalty`/`presence_penalty` 已废弃,不使用 | API Reference |
| usage 细项 | `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` / `completion_tokens_details.reasoning_tokens`——成本核算用:命中缓存的输入单价仅约未命中的 1/50,系统提示+few-shot 稳定则复跑输入近零成本;**冒烟实测(10-25):思考关闭时 reasoning_tokens 字段整体缺席而非返回 0,守门判据改为"字段缺席+短输出+~1.4s 延迟"三证;前缀缓存命中实证(33.8K 输入中 13.1K 命中)** | API Reference + 冒烟实测 |
| 成本预算 | 90 题 × 约 3K 输入 + 0.4K 输出 ≈ 27 万输入 + 3.6 万输出/全量一次。flash 定价(USD/1M token):输入未命中 $0.15、命中 $0.003、输出 $0.60;峰时(工作日北京时间 09–12 / 14–18 点)翻倍,其余半价时段。估算:**单次全量 ≈ $0.06(谷)~ $0.12(峰),即 ¥1 以内**,开发迭代数次仍个位数人民币——**以运行时 usage 实测为准**,数字进阶段报告(框架 §6.4:调用量与费用必须写明) | Models & Pricing |

---

## 4. 配置草案(`configs/main.yaml` 增量,示意)

```yaml
retrievers:                      # 五臂保持不变,追加:
  - {name: hybrid_rrf, type: hybrid, arms: [bm25_bigram, bge_base],
     depth: 20, rrf_k: 60}       # T6;评测自动成主表第六列

generation:                      # T7;DEEPSEEK_API_KEY 缺失时整段跳过
  retriever: hybrid_rrf          # 生成用检索臂(可切单臂作对照)
  top_k: 5                       # structural gold 均值 3.68 → 下界;消融轴 {3,5,10}
  llm:
    provider: deepseek
    base_url: https://api.deepseek.com
    model: deepseek-flash          # V4.1-Flash-1017;报告/代码统一此名
    thinking: disabled             # 端点默认 enabled,必须显式关;不关 temperature 无效
    temperature: 0                 # 仅非思考模式生效
    max_tokens: 1024
    api_key_env: DEEPSEEK_API_KEY  # seed 端点不支持,不发送
  prompt:
    few_shot: true               # 2~3 例:正常作答 + 拒答
    rerank: false                # 消融开关预留,阶段三不实现
```

---

## 5. Prompt 与输出协议基线(§5.2 落地)

```
系统提示要点:
1. 角色:叙事文本问答助手,只依据【证据】回答,不得使用剧本外的知识
2. 输出协议(强制 JSON,且仅输出 JSON):
   {"answer": "...",                        // 引用证据作答;证据不足时说明
    "verdict": "yes|no|unknown",            // 是否类问题给裁决;其余题型照填(不参与评分)
    "evidence_rfids": [[起,止], ...]}       // 只能引用证据块头部的 rfid 区间
3. 拒答规则:证据不足或与问题无关 → verdict=unknown,answer 说明缺少什么,不得猜测
4. few-shot 2~3 例:一个基于真实 chunk 的正常作答 + 一个拒答示例
```

- 证据块头部携带 chunk_id / rfid 区间 / speakers,正文为 chunk 全文原文(全角保留)。
- 证据顺序 = RRF 融合排名顺序;后续阶段可做 lost-in-the-middle 位置消融(原理点④备用)。
- temperature=0(仅非思考模式生效,thinking 已显式 disabled);端点不支持 seed、不发送;复现兜底 = 缓存(同 prompt 命中即逐字节一致)。

---

## 6. 预期结果

**产物清单**:hybrid 检索分支与第六列指标;`src/llm.py`、`src/generate.py`;config generation 节;`.env.example` + `.gitignore` 增项;`requirements.txt` +openai;每 run 新增 `answers.jsonl`、`generation_log.json`(prompt 哈希/usage/延迟/原始响应)、metrics 中 `label_accuracy` 与引用指标块;阶段报告 + README 更新。

**指标预期**(诚实口径:待验证假设,不是承诺;全部数字以 `runs/<ts>/` 实测为准):

| 指标 | 预期 | 依据 |
| --- | --- | --- |
| hybrid 总体 R@5 | ≥ 0.50(超最佳单臂 bge_base 0.4476) | 分层互补:literal 靠 BM25、semantic_gap 靠 bge,RRF 应同时保住两层 |
| hybrid literal / semantic_gap 层 R@3 | 分别 ≥ 0.75 / ≥ 0.43(不低于两范式各自强项的约 85%) | 两臂强项应被继承 |
| structural 层 | 基本不动(≈0.2) | 五臂同败是结构问题——**本阶段要坐实"图谱必要性"这一结论本身** |
| verdict 标签准确率(36 题) | 0.70~0.90 | 协议化输出 + few-shot 的常规区间 |
| 不可回答题拒答正确率(18 题) | 0.6~0.8(最可能不达预期) | 检索必返回字面相似块,拒答只靠 prompt——预期主要失败模式 |
| 引用有效率(规则校验) | ≥ 0.9 | JSON 协议 + "只引用给定块"约束 |
| JSON 解析失败率 | < 5%(重试后) | §7.3 工程指标 |

**验收底线(全满足才算 T6~T8 收口)**:

1. `python run_all.py --config configs/main.yaml` 一条命令跑通:五臂+hybrid 检索评测、90 题生成、答案层指标全部落盘;无 Key 环境该命令仍可完成检索层全流程。
2. 缓存命中时重跑,`metrics.json` 与 `answers.jsonl` 逐字节一致(同机器同依赖口径,沿 T5 惯例)。
3. 每个数字可指回 `runs/<ts>/` 日志;成本与延迟有实测汇总。
4. ≥3 个失败案例完成 §5.3 六模式归因,写入阶段报告。

---

## 7. 风险与已知坑(开工前通读)

| # | 坑 | 处置 |
| --- | --- | --- |
| 1 | Key 进仓库(红线③) | 只走环境变量;`.gitignore` 加 `.env`;提交前红线自查 |
| 2 | `RfidLocator` 对孤儿 rfid 抛 KeyError(3281/14785/14898 及模型乱引的区间) | 生成侧校验 catch 并标 invalid,不崩评测 |
| 3 | 模型输出 `evidence_rfids` 格式漂移(整数/区间/越界) | 归一解析(int/区间兼容)+ 无效标记,不丢弃该题 |
| 4 | **thinking 端点默认开启**:temperature 失效、延迟与 reasoning token 费用大增 | 显式 `extra_body={"thinking":{"type":"disabled"}}`;`usage.reasoning_tokens≈0` 作守门核对 |
| 5 | DeepSeek temperature=0 非逐字节保证 | 复现性靠响应缓存;重跑命中缓存 |
| 6 | json_object 模式要求 prompt 含 "json",且官方注明偶发空 content | 协议模板含字样与格式示例;空 content 计入解析失败、重试一次覆盖;降级正则常备 |
| 7 | few-shot 泄漏评测集题目 | few-shot 证据取自语料真实 chunk,题目另行编写 |
| 8 | prompt 预算:k=5 × 全文 chunk(max 837 字)≈ 3K 字 + 系统 + few-shot | flash 上下文 1M,余量极大,无需截断;若后续加大 k 再评估 |
| 9 | 评测集冻结纪律 | 生成结果差不改题;题目异议走 T5 终裁流程留痕 |
| 10 | 缓存键漏参数导致换模型/开关思考串缓存 | 键含 model+thinking+temperature+max_tokens+messages 全量 |
| 11 | Windows 控制台编码 | 沿用 `sys.stdout.reconfigure(encoding="utf-8")` 既有模式 |

---

## 8. 数字溯源与复现命令(收口时填 run 目录)

```bash
# cwd = E:/dev/NLP/project
export DEEPSEEK_API_KEY=...                # 或放 .env(不进 git)
python run_all.py --config configs/main.yaml   # 全链:五臂+hybrid 检索评测 → 90 题生成 → 答案层指标
# 产物:runs/<ts>/{metrics.json, answers.jsonl, generation_log.json, retrieval_results.json, ...}
```

| 数字 | 日志位置(收口时回填) |
| --- | --- |
| hybrid 六列检索指标 | `runs/20261025-170131/metrics.json`(T6 已出;T7/T8 收口以最终 run 为准) |
| 标签准确率 / 拒答正确率 | T7 已出首批:0.8056 / 1.0 / 0.6111,源 `eval/answers.jsonl`;T8 接线后以 `runs/<ts>/metrics.json` 为准 |
| 引用有效率 / 解析失败率 | T7 已出:条级 1.0(83/83)、题级 0.5556、解析失败率 0,源 `eval/answers_generation_log.json`;T8 接线后进 metrics |
| token 成本 / P50/P95 延迟 | T7 已出:输入 299,383(前缀命中 129,536)/输出 6,315,$0.0297 谷;P50=1.48s P95=2.18s,源 `eval/answers_generation_log.json` |
| 失败案例 top-k 与 prompt | `retrieval_results.json`(按 qid)+ `eval/answers_generation_log.json`(T8 复制进 run 目录) |
