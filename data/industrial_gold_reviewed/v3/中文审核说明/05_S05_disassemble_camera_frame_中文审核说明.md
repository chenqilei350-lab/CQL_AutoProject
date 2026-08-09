# S05 拆卸相机框架

**场景编号：** `indego_232cf338-98a7-4cdc-8683-417c0e2adf7a_s1`  
**类别：** `disassembly`  
**英文任务：** disassemble camera frame  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 把原先四个合并动作细化为10个有证据的拆卸动作。
- 补入螺钉与连接件的拆卸，不再直接跳到杆件和底座。

## S05 拆卸相机框架 仍有 2 处需要确认

1. **待确认：road、rope、roll 是否都是 rod 的 ASR 变体？**

   当前处理：实体名称统一为 rod，但 evidence_text 保留原始词形。

2. **待确认：各螺钉使用什么工具？**

   当前处理：完整转录未命名工具，不添加 USES_TOOL。

## 当前保留的 Gold 动作顺序

1. `remove main camera screw` — 拆下相机主螺钉
2. `remove two small camera screws` — 拆下两颗相机小螺钉
3. `remove camera` — 拆下相机
4. `remove base connector` — 拆下底座连接件
5. `remove long rod from base` — 从底座取下长杆
6. `remove rod connector` — 拆下杆件连接件
7. `separate short rod from long rod` — 把短杆与长杆分离
8. `loosen base screw` — 松开底座螺钉
9. `remove final short rod` — 取下最后一根短杆
10. `separate base` — 分离底座

## 已确认的工具与使用动作

- 本 scene 没有能够由来源确认实际使用的具名工具；未根据常识补写工具。

## 已确认内容

- 相机螺钉、连接件、长杆、短杆和底座的先后顺序明确。

## 暂不纳入 Gold

- 排除未命名工具；不因常识补写螺丝刀或 Allen key。

## 文件内统计

- Action：10
- Object：9
- Tool：0
- Relation：21
- Quality result / warning：0
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
