# eval/ 评测集工具链说明

> 适用于 qaset schema v2（`anchor_layer` / `unans_subtype` / `accept_criteria` / `targets`）。
> 所有命令默认在 `project/` 根目录执行；涉及语料的脚本读
> `../Rewrite_terra/data/{lines.jsonl, scenes.jsonl}`（只读，不修改语料）。

## 一、目录内容

| 文件 | 角色 |
|---|---|
| `qaset_v0_draft.json` | **正式评测集草稿**（v0 已收 30 题，向 targets=90 扩容中）。冻结稿由此产出 |
| `qaset_v1_candidates_<type>.json` | 按题型分文件的候选池（如 `..._temporal.json`；status=`candidate-pool-do-not-eval`，数字绝不直接进报告）。旧全量池 `qaset_v1_candidates.json` 已删除，避免对后续定向出题产生锚定 |
| `qaset_seed.json` | 10 题冒烟集，**旧 schema**，仅供管线冒烟，不适用新校验，不进报告 |
| `qaset_calibration.json` | 标定报告（派生文件，可随时重算，重跑逐字节一致） |
| `validate_qaset.py` | 冻结门槛校验（八项检查） |
| `calibrate_qaset.py` | BM25 标定回灌（层标签的测量与裁决依据） |
| `merge_qaset.py` | 候选并入正式草稿（qid 重编号 + 硬门槛 + 缺口表） |
| 共享依赖 | `../src/qaset_analysis.py`（重叠率/实体集/df 统计的唯一实现）、`../src/evaluate.py`（指标 + by_layer） |

## 二、schema v2 每题字段

| 字段 | 适用 | 说明 |
|---|---|---|
| `qid` / `type` | 全部 | Q01 起编号；fact / yesno / temporal / aggregate / unanswerable |
| `question` / `gold_answer` | 全部 | gold_answer 只依据语料，不得用其他线路知识补全 |
| `gold_evidence_rfids` | 可答题 | int 或 `[起,止]` 区间；端点必须存在于 lines.jsonl；不可回答题一律空数组 |
| `gold_verdict` | yesno / unanswerable | yes / no / unknown |
| `anchor_layer` | 可答题 | literal / semantic_gap / structural / floor（判定见下表） |
| `unans_subtype` | unanswerable | entity_absent（实体零命中）/ fact_absent（实体在、事实不在）/ mentioned_unknown（提及但不交代） |
| `accept_criteria` | aggregate 必填，fact/temporal 建议 | LLM-as-judge 判分口径：数值/别名等价、部分给分规则、何种表述不给分 |
| `distractor_note` | 可选 | 易混 chunk；「标定回填:」前缀为机器草稿，冻结前人工修剪 |
| `notes` | 草稿期 | 核对依据 + 标签裁决理由；**冻结时清空**（无 notes = 已核对标记） |
| `src_cand` | 并入题 | merge 时自动写入的溯源（候选 cand_id），冻结后保留 |
| `recommend` | 仅候选池 | high / backup；并入选题的依据 |

**anchor_layer 操作定义**（与 `calibrate_qaset.py` 的 rules 字段同口径）：

| 层 | 判据 | 报告用途 |
|---|---|---|
| literal | 可区分重叠 ≥2（原词/引语/专名） | 原理点①地板 + sanity |
| semantic_gap | 重叠 ≤1，保留 ≥1 实体锚，gold≤2 chunk 且参照臂脱靶/首命中>10 | 原理点②（w2v vs bge 主战场） |
| structural | gold≥3 chunk；或 gold≤2 但分散为作答必要条件（部分命中） | 原理点③ + 图谱动机 |
| floor | 零实体锚/位置指代，全卷 ≤2（`targets.floor_max`） | 只进错误分析，不进主表 |

顶层 `targets` 块 = **冻结门槛**（90 题目标矩阵），不是计数器：并入题目不会也不应改写它，
validator 的分布对照显示全部「差 0」时才具备冻结条件。只有目标本身据实修订时才手改
（先例：aggregate semantic_gap 3→0、structural 10→13，因聚合题金标天然跨 chunk）。

## 三、全生命周期命令流

```bash
# 0. 出题：子代理/人工按题型写候选池 eval/qaset_v1_candidates_<type>.json
#    （schema 见上，候选池顶层 status 必须是 candidate-pool-do-not-eval）
#    自检两连（候选池侧，<type> 代入题型）：
python eval/validate_qaset.py eval/qaset_v1_candidates_<type>.json     # 问题清单须为 0
python eval/calibrate_qaset.py eval/qaset_v1_candidates_<type>.json    # 建议与标签应一致

# 1. 人工逐题核对候选（对照原文，gold/verdict/层标签）；核对依据记入 notes

# 2. 并入草稿（qid 自动接续、src_cand 溯源、硬门槛拦截、version 进位）：
python eval/merge_qaset.py                                      # 只读：池子现状 + 缺口表
python eval/merge_qaset.py --recommend high --dry-run           # 预演
python eval/merge_qaset.py --recommend high                     # 实际并入
python eval/merge_qaset.py --remove Q31                         # 复核否决后移除

# 3. 收口校验（问题清单须 0；分布缺口须全部为 0 才可冻结）：
python eval/validate_qaset.py eval/qaset_v0_draft.json

# 4. 五臂评测（by_layer 分层表自动进 metrics.json）：
python run_all.py --config configs/main.yaml
```

## 四、各脚本详解

### `validate_qaset.py` — 冻结门槛

```bash
python eval/validate_qaset.py eval/qaset_v0_draft.json
```

八项检查：① schema/qid 唯一/题型/verdict 枚举；② 金标 rfid 存在且落 chunk（越界即报）；
③ Recall@k 可达性（gold_chunks>k 单列提示）；④ 时序题跨 chunk 覆盖；⑤ verdict 配比；
⑥ notes 约定（冻结稿残留 notes 即报错）；⑦ schema v2 必填字段（硬检查，与模式无关）；
⑧ targets 分布对照 + literal/semantic_gap 可区分重叠率机检。
**双模式**：文件含 `status` = 草稿（⑧只出警告）；无 `status` = 冻结稿（⑧升级为报错，
且 notes 必须已清空）。末尾「问题清单(N)」为 0 才算通过；警告清单需逐条人工裁决。

### `calibrate_qaset.py` — BM25 标定回灌

```bash
python eval/calibrate_qaset.py <qaset文件>              # 报告（确定性，无时间戳）
python eval/calibrate_qaset.py <qaset文件> --apply      # 保守写回
python eval/calibrate_qaset.py <qaset文件> --arms bigram  # 附加对照臂列
```

参照臂 = bm25_jieba（config 超参）；层标签的操作定义 =「稀疏词典分词检索的不可达性」。
判定规则随报告落盘（rules 字段）。`--apply` 写回边界：anchor_layer 仅在 gold_chunks==1
且 literal↔semantic_gap 之间自动改写，跨层改判留给人工；distractor_note 只在字段缺失时
回填 top-3 干扰 chunk 草稿；其余字段一律不碰。产物 `eval/qaset_calibration.json`
可作报告附录的「难度档案」（每题 R@k/首命中/金标命中详情/重叠率/层归属）。

### `merge_qaset.py` — 候选并入正式草稿

```bash
python eval/merge_qaset.py                            # 只读列表 + 缺口表
python eval/merge_qaset.py --recommend high           # 按推荐级并入
python eval/merge_qaset.py --cand-ids S06,S08         # 按编号并入
python eval/merge_qaset.py --recommend high --dry-run # 预演不写盘
python eval/merge_qaset.py --remove Q31               # 移除草稿中的题
```

并入时自动：qid 接续现有最大号重编号（Q31, Q32…）；丢弃池侧字段（cand_id/recommend/
池用 qid），写入 `src_cand` 溯源（防重复并入）；version 尾部 revN 自动 +1。
硬门槛（任一不过**整批中止**，绝不部分并入）：rfid 存在且落 chunk、schema v2 完整、
与草稿及本批次问句查重、编号不存在或已并入时显式报错。**从不改写 targets**——
缺口表只读展示进度，冻结门槛的语义不被自动更新稀释。

### `../src/qaset_analysis.py` — 共享分析层（无 CLI）

`validate` 与 `calibrate` 共用的唯一实现，防两套口径漂移：内容词提取（jieba+自定义词典，
自带全角→半角）、实体集（user_dict ∪ 说话人名）、语料 df 表、重叠率。改判定标准只改这里：
`DF_MAX=20`（可区分词的 chunk 级 df 上限，高频词表层重叠不算检索可达性）、
`NOVEL_DF_MAX=30`（新词高 df → 疑假落差/歧义警告）。

### `../src/evaluate.py` — 指标计算（run_all.py 调用）

```bash
python src/evaluate.py --config configs/main.yaml --arm jieba   # 单臂调试
```

检索层 Recall@{3,5,10}/MRR（不可回答题跳过，留给答案层拒答评测）；metrics.json 含
`by_type` 与 `by_layer` 两套分解——报告的分层对比表（语义落差层上五臂对比）直接取自后者。

## 五、冻结流程 checklist

1. 候选池逐题人工核对（对照原文；gold/verdict/层标签；recommend 辅助择优）；
2. `merge_qaset.py` 并入（建议按批 `--recommend high` 起步，缺口表引导后续选题）；
3. `validate_qaset.py` 确认问题 0 且分布缺口全部为 0（不足的格回步骤 0 定向补题）；
4. `calibrate_qaset.py --apply` 复核并题后的层标签一致性；
5. 人工修剪全部「标定回填:」distractor_note 草稿；机检分歧（如重叠率提示）逐条终裁；
6. 清空顶层 `status` 与各题 `notes`（= 已核对标记），定稿 version，改名 `qaset_v1.json`。

## 六、已知盲区（机检辅助、人工终裁）

- **实体探测器**：复合专名不在 user_dict 会被误建议 floor（例：Q03「拜恩带剑绿地骑士团」）；
  单字人名可能被 jieba 与邻字合并切分而漏检（例：Q14「篝杀」）。对应 remedy：把确认的
  复合专名补进 `data/user_dict.txt`（本身也是免费的检索增强）。
- **重叠率检查**：实体锚题型的可区分重叠恒低（实体词被排除出内容词），「literal 可能过严」
  多为误报；判据阈值 ≤1/<2 是约定值，分词边界差异会引入噪声。
- **机检与人工分歧**（例：Q11「铃木/凡人」）：保留双方证据于 notes，冻结时人工终裁。
- `qaset_seed.json` 为旧 schema，跑新 validator 会整批判缺字段——属预期，勿用它对标校验。
