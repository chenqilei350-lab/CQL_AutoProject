# S19 钻孔错误检测流程模板

**场景编号：** `indego_warning_A_Task_08_s1`  
**类别：** `mistake_detection`  
**英文任务：** drilling mistake-detection procedure template  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 继续按 run_id 隔离11条 warning，避免不同失败运行形成错误时序。
- 标准步骤保留 prescribed_step 属性；缺失动作不会被当作已执行动作。
- 根据 keystep 与 warning 补入 clamp 和 drill 两个流程工具，但不把 warning 误当作完成动作。

## S19 钻孔错误检测流程模板 仍有 2 处需要确认

1. **待确认：11条 warning 是否来自同一次失败执行？**

   当前处理：不是；按各自 run_id 保留，禁止合并为一条实际时间线。

2. **待确认：plug in 的具体对象是什么？**

   当前处理：来源未说明是钻机、电源线还是延长线，不创建 ACTS_ON。

## 当前保留的 Gold 动作顺序

1. `mark hole` — 标记孔位；属性：`{"kind": "prescribed_step"}`
2. `clamp block` — 夹紧木块；属性：`{"kind": "prescribed_step"}`
3. `plug in` — 接通电源；属性：`{"kind": "prescribed_step"}`
4. `drill hole` — 钻孔；属性：`{"kind": "prescribed_step"}`

## 已确认的工具与使用动作

- **clamp**：夹紧木块（`clamp block`）
- **drill**：钻孔（`drill hole`）

## 已确认内容

- 四个步骤是 prescribed_step，而不是每个 run 都已完成。
- clamp 用于夹紧 block，drill 用于 drill hole。
- 每条 warning 的 step、description 和 run_id 均可追溯。

## 暂不纳入 Gold

- 排除跨 run BEFORE、plug in 对象推测，以及把 didn't... 转成已执行动作。

## 文件内统计

- Action：4
- Object：2
- Tool：2
- Relation：5
- Quality result / warning：11
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
