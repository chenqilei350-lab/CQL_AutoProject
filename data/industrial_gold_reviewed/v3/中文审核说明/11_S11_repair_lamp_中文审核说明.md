# S11 维修台灯

**场景编号：** `indego_user_14_user_15_411_1010_1_s1`  
**类别：** `inspection_repair`  
**英文任务：** repair lamp  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入维修前的故障检查和无法移动测试。
- 补入拆件、入槽、正确方向回装、复测和锁紧。
- 将 clamp 与 screwdriver 分别绑定到支撑和紧固动作。

## S11 维修台灯 仍有 1 处需要确认

1. **待确认：round part、top、that part 的正式部件名称是什么？**

   当前处理：保留原文描述，不映射到未证实部件编号。

## 当前保留的 Gold 动作顺序

1. `inspect incorrect lamp assembly` — 检查台灯错误装配
2. `test blocked lamp movement` — 测试并发现台灯无法下移
3. `unscrew adjustment screw` — 松开调节螺钉
4. `remove misassembled lamp parts` — 拆下装错的台灯部件
5. `position connector in groove` — 把连接件放入槽内
6. `mount lamp assembly on clamp` — 把台灯组件装回夹座
7. `remove round part and top` — 拆下并识别圆形件和顶部件
8. `reposition round part` — 重新定位圆形件
9. `reinstall top in correct orientation` — 按正确方向装回顶部
10. `insert lamp screws` — 把螺钉插入孔中
11. `tighten lamp screws` — 用螺丝刀顺时针紧固
12. `reinstall adjustment part` — 装回调节部件
13. `test lamp adjustability` — 测试台灯可调性
14. `tighten adjustment joint` — 重新锁紧调节接头

## 已确认的工具与使用动作

- **clamp**：把台灯组件装回夹座（`mount lamp assembly on clamp`）
- **screwdriver**：用螺丝刀顺时针紧固（`tighten lamp screws`）

## 已确认内容

- 维修前灯臂无法移动，维修后可再次调整。
- 螺丝刀明确用于顺时针紧固螺钉。
- clamp 明确用于承托组件。

## 暂不纳入 Gold

- 排除未命名部件的标准名称推测。

## 文件内统计

- Action：14
- Object：9
- Tool：2
- Relation：31
- Quality result / warning：1
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
