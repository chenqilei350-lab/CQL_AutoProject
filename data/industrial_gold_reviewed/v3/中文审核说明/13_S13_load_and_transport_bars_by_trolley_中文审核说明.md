# S13 用推车装载并运输杆件

**场景编号：** `indego_user_14_412_02024_1_s1`  
**类别：** `logistics_organisation`  
**英文任务：** load and transport bars by trolley  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入携带捆绑带、开灯、开门、关闭推车等中间动作。
- binding belt 和 trolley 同时按实际用途建立 USES_TOOL。

## S13 用推车装载并运输杆件 仍有 1 处需要确认

1. **待确认：最终 deliver 具体卸货动作是什么？**

   当前处理：完整转录在运输后结束，只按带时间戳 keystep 保留 deliver，不细化卸货。

## 当前保留的 Gold 动作顺序

1. `put on working gloves` — 戴工作手套
2. `retrieve trolley` — 取推车
3. `take binding belt` — 携带捆绑带
4. `move empty trolley to another room` — 把空推车移到另一房间
5. `turn on lights` — 开灯
6. `retrieve bars` — 取杆件
7. `bind bars` — 用捆绑带捆扎杆件
8. `open door` — 开门
9. `load bars into trolley` — 把杆件装入推车
10. `close trolley` — 关闭推车
11. `transport loaded trolley` — 运输已装载推车
12. `deliver shipment` — 交付货物

## 已确认的工具与使用动作

- **binding belt**：用捆绑带捆扎杆件（`bind bars`）
- **trolley**：把杆件装入推车（`load bars into trolley`）；运输已装载推车（`transport loaded trolley`）

## 已确认内容

- 手套、推车、捆绑带、杆件、门和灯的操作顺序明确。

## 暂不纳入 Gold

- 排除没有转录细节的交付子动作和房间身份。

## 文件内统计

- Action：12
- Object：7
- Tool：2
- Relation：27
- Quality result / warning：0
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
