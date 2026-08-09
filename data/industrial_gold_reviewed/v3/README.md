# Industrial Gold V3 全量优化版（20条）

本批次从完整 IndEgo 原始标准输入重新选取并人工整理，明确排除：

- 桌子/书桌拆卸场景；
- 电脑、PC、工作站或嵌入式计算单元组装场景；
- 上传参考 JSONL 中的全部 6 个 scene_id。

## 文件

- `industrial_gold_v3_reviewed.jsonl`：20条最终 reviewed Gold，逐行 JSON。
- `industrial_gold_v3_reviewed.json`：相同内容的 JSON 数组版。
- `examples/`：按 01–20 编号的单条 Gold JSON。
- `中文审核说明/`：按 01–20 编号的逐条中文审核说明。
- `20个Gold样本中文审核说明_汇总.md`：所有中文说明的整体文件。
- `Gold_V3全量优化报告.md`：动作、工具和关系的优化前后对照及逐样本统计。
- `manifest.json`：编号、scene_id、类别、任务与文件路径对照表。
- `validation_report.json`：数量、排除项、证据、关系端点等自动验证结果。

## 审核原则

1. 上传的 V2 文件只参考结构和审核写法，不复制或混入样本。
2. 完整转录中明确出现的检查、清理、纠错、失败尝试、回装和收尾动作不得省略。
3. 句子中明确实际使用的工具必须创建 Tool 与 `USES_TOOL`；仅出现但未使用的物品不误标工具关系。
4. 只创建证据支持的相邻 `BEFORE`；警告模板来自多个 run，不创建跨 run 的 `BEFORE`。
5. 含糊代词、部件身份、工具映射和结果缺失均写入中文说明，并从正式 Gold 事实中排除。

自动验证状态：**passed**；Action **224** 个，Tool **34** 个，`USES_TOOL` **72** 条，待确认点 **35** 个。
