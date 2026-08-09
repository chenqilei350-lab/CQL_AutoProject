# S15 称重并包装货物

**场景编号：** `indego_user_17_user_15_451_0402_1_s3`  
**类别：** `logistics_organisation`  
**英文任务：** weigh and pack goods  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 复核完整视频转录与带时间戳 action annotation；正式 Gold 继续以精确动作注释为主。
- scale 已作为 weigh goods 的工具保留。
- double-sided tape 只被提到、未明确执行使用，因此没有错误添加 USES_TOOL。

## S15 称重并包装货物 仍有 3 处需要确认

1. **待确认：drop pallet 是放下还是意外掉落？**

   当前处理：保留注释原词，中文采用中性“放下”。

2. **待确认：double-sided tape 是否实际用于 pack goods？**

   当前处理：转录只讨论现场已有胶带，没有明确使用动作，不建立工具关系。

3. **待确认：两次 pick goods 是否不同货物？**

   当前处理：保留两个动作，共用 generic goods。

## 当前保留的 Gold 动作顺序

1. `drop pallet` — 放下托盘
2. `pick goods` — 拿取货物（第一次）
3. `open scale` — 打开秤
4. `take out goods` — 取出货物
5. `weigh goods` — 称重货物
6. `put back goods` — 放回货物
7. `pack goods` — 包装货物
8. `pick goods` — 拿取货物（第二次）

## 已确认的工具与使用动作

- **scale**：称重货物（`weigh goods`）

## 已确认内容

- 八个 action annotation 带明确时间戳。
- scale 明确对应 weigh goods。

## 暂不纳入 Gold

- 排除仅被提及但未明确使用的 double-sided tape，以及货物种类推测。

## 文件内统计

- Action：8
- Object：2
- Tool：1
- Relation：15
- Quality result / warning：0
- 待确认点：3

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
