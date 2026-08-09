# S20 包装错误检测流程模板

**场景编号：** `indego_warning_A_Task_19_s1`  
**类别：** `mistake_detection`  
**英文任务：** packaging mistake-detection procedure template  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 继续按 run_id 隔离12条 warning，避免不同失败运行形成错误时序。
- 未从错误描述中反推未明确的工具或正确数量。

## S20 包装错误检测流程模板 仍有 4 处需要确认

1. **待确认：12条 warning 是否描述一条连续失败流程？**

   当前处理：不是；它们属于多个 run_id，分别保留，不能串联。

2. **待确认：“not centered”相对于哪个设备或中心位置？**

   当前处理：缺少参照物，只作为 warning 文本保留。

3. **待确认：“only one belt”要求的正确捆扎带数量是多少？**

   当前处理：来源只说明一条不足，不推断标准数量。

4. **待确认：胶带、缠绕膜、捆扎带和标签是否需要创建 Tool/Object 实体？**

   当前处理：短模板未给出稳定角色与具体材料实例，只保留 package 作为操作对象。

## 当前保留的 Gold 动作顺序

1. `put on gloves` — 戴手套；属性：`{"kind": "prescribed_step"}`
2. `place package` — 放置包裹；属性：`{"kind": "prescribed_step"}`
3. `tape package` — 给包裹贴胶带；属性：`{"kind": "prescribed_step"}`
4. `wrap package` — 包裹缠膜；属性：`{"kind": "prescribed_step"}`
5. `belt package` — 给包裹打带；属性：`{"kind": "prescribed_step"}`
6. `label` — 贴标签；属性：`{"kind": "prescribed_step"}`

## 已确认的工具与使用动作

- 本 scene 没有能够由来源确认实际使用的具名工具；未根据常识补写工具。

## 已确认内容

- 六个包装标准步骤和12条带 run_id 的错误注释明确。

## 暂不纳入 Gold

- 未建立跨 run 时序、未推断中心参照或正确捆扎带数量。

## 文件内统计

- Action：6
- Object：2
- Tool：0
- Relation：6
- Quality result / warning：12
- 待确认点：4

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
