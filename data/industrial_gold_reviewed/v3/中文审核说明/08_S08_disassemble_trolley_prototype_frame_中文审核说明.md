# S08 拆卸推车原型框架

**场景编号：** `indego_user_14_user_16_407_1712_1_s1`  
**类别：** `disassembly`  
**英文任务：** disassemble trolley prototype frame  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 把泛化的 clean up 改为原文明确的 clean plates。
- 补入最后收集 screws，避免拆解后直接结束。

## S08 拆卸推车原型框架 仍有 2 处需要确认

1. **待确认：开头“Maybe we'll be using this”指什么工具？**

   当前处理：工具未命名且未确认实际使用，排除 Tool 和 USES_TOOL。

2. **待确认：blades 是否为 plates 的 ASR 错词？**

   当前处理：结合紧邻上下文归一为 plates，证据保留 blades。

## 当前保留的 Gold 动作顺序

1. `put on gloves` — 戴手套
2. `remove vertical bar` — 拆下竖杆
3. `remove plates` — 拆下板件
4. `remove horizontal bar` — 拆下横杆
5. `disassemble remaining frame` — 拆解剩余框架
6. `clean plates` — 清洁板件
7. `gather screws` — 收集螺钉

## 已确认的工具与使用动作

- 本 scene 没有能够由来源确认实际使用的具名工具；未根据常识补写工具。

## 已确认内容

- 竖杆、板件、横杆、框架、清洁和收集螺钉的顺序明确。

## 暂不纳入 Gold

- 排除身份不明、是否使用也不确定的开场工具。

## 文件内统计

- Action：7
- Object：6
- Tool：0
- Relation：13
- Quality result / warning：0
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
