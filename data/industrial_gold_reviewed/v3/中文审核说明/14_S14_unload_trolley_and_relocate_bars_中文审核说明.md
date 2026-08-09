# S14 卸下推车中的杆件并搬运归位

**场景编号：** `indego_user_14_412_02024_2_s1`  
**类别：** `logistics_organisation`  
**英文任务：** unload trolley and relocate bars  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入开灯、两次开门和最终脱手套。
- 把两个 unpack 细化为取杆/解捆与目的地放置。
- trolley 在本段是被卸载和归还的容器对象，不误标为实际使用工具。

## S14 卸下推车中的杆件并搬运归位 仍有 1 处需要确认

1. **待确认：put this back here 中 this 指捆绑带还是其他物体？**

   当前处理：该代词动作仍排除，不影响已确认的归还 trolley 与搬运 bars。

## 当前保留的 Gold 动作顺序

1. `turn on lights` — 开灯
2. `put on working gloves` — 戴工作手套
3. `open door for unloading` — 打开卸货门
4. `take bars out of trolley` — 从推车取出杆件
5. `place bars on floor` — 把杆件放到地面
6. `unbind belt` — 解开捆绑带
7. `return trolley` — 归还推车
8. `open destination door` — 打开目的房间的门
9. `transport bars to another room` — 把杆件搬到另一房间
10. `place bars at destination` — 在目的地放置杆件
11. `remove working gloves` — 脱下工作手套

## 已确认的工具与使用动作

- 本 scene 没有能够由来源确认实际使用的具名工具；未根据常识补写工具。

## 已确认内容

- 卸货、解捆、归还推车、搬运和放置杆件的顺序明确。

## 暂不纳入 Gold

- 排除指向不明的 put this back here 和未命名地点实体。

## 文件内统计

- Action：11
- Object：6
- Tool：0
- Relation：22
- Quality result / warning：0
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
