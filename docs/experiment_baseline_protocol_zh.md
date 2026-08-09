# 工业 KG 基准与持续实验协议

## 固定基准

当前对照组为 `industrial-v1-llama31-8b-20260809`，模型固定为
`llama3.1:8b@46e0c10c039e`。基准真实 LLM 实验只运行一次，之后的代码优化读取
已保存的逐场景结果进行配对比较，不再次调用原始基准模型。

基准标签为 `baseline-industrial-v1-2026-08-09`。标签对应两层提交：第一层是被
测试的代码快照，第二层加入完整实验结果、校验值和不可覆盖标记。

## 三个代码模块

| 固定模块名 | 中文 | 主要目录 | 主要指标 |
| --- | --- | --- | --- |
| `data_cleaning` | 数据清理 | `backend/cleaning/` | 文件与记录覆盖、错误、重复、时间戳、provenance |
| `text_preprocessing` | 文本预处理 | `backend/preprocessing/` 与 IndEgo adapter | evidence 支持、字段覆盖、遗漏和下游 KG 变化 |
| `kg_extraction` | 知识图谱抽取 | extraction、pipeline、graph、evaluation | Node/Edge F1、幻觉、schema、稳定性和运行时间 |

模块版本记录在 `config/module_versions.json`。每次修改对应模块时必须更新该模块
版本，并生成一个不可重复的 Change ID，例如 `EXP-20260815-001`。

## 基准输入条件

基准一次性运行四个条件：

1. `raw + one_shot`；
2. `raw + layered`；
3. `auto_unified + layered`；
4. `reviewed_unified + layered`。

`reviewed_unified` 来自人工审核 Gold，只是理想预处理上限。它不能写成自动数据
清理模块的成果。`auto_unified` 来自现有 IndEgo 清理与预处理流程，并被限制为只
保留当前 Gold source excerpt 中能找到的 evidence。

## 每次优化必须记录

- 改动模块、文件、函数或类；
- 改动原因与可验证假设；
- 使用的方法及论文或代码来源；
- 预期改变的指标；
- 运行条件、场景和模型身份；
- 基准值、新值、绝对差值和相对差值；
- 错误、超时及所有失败运行；
- 最终决定：`improved`、`neutral`、`regressed` 或 `incomplete`。

默认使用固定 quick regression scenes。里程碑实验再运行 development 或 all，
但仍然只运行新代码，不重跑 V1。评价器改变时，应从 baseline 的
`raw_runs.jsonl.gz` 重算，不重新请求 LLM。

## 运行示例

完整基准：

```bash
UV_CACHE_DIR=/private/tmp/aut-kg-uv-cache uv run python \
  scripts/run_versioned_baseline.py \
  --code-commit <tested-code-commit>
```

后续改动：

```bash
UV_CACHE_DIR=/private/tmp/aut-kg-uv-cache uv run python \
  scripts/run_change_experiment.py \
  --change-id EXP-20260815-001 \
  --module kg_extraction \
  --reason "Relation recall is low" \
  --hypothesis "Expanded candidates increase USES_TOOL recall" \
  --method "Context-window relation candidate generation" \
  --method-source "project relation-candidate design" \
  --expected-effect "Higher Edge F1 without more hallucinations"
```

任何含 `COMPLETED` 的实验目录都不得覆盖。如果模型摘要、Gold 或指标定义发生实质
变化，应建立新的 baseline version。
