# S16 在木块上钻孔并打磨清理

**场景编号：** `indego_user_14_411_2409_1_s1`  
**类别：** `woodworking`  
**英文任务：** drill and finish a hole in wooden block  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入延长线、接电、启动钻机、钻孔附件和钻后检查。
- 把 clamp 同时按夹持工具使用，补齐 USES_TOOL。
- 所有 measuring tape、pencil、drill、attachment、sandpaper、brush 都绑定到具体动作。

## S16 在木块上钻孔并打磨清理 仍有 2 处需要确认

1. **待确认：drill knot 的标准名称是什么？**

   当前处理：作为 unnamed drilling attachment 保留，不直接改写 drill bit。

2. **待确认：约4厘米从哪条边测量？**

   当前处理：只记录近似距离，不建立参考边关系。

## 当前保留的 Gold 动作顺序

1. `put on working gloves` — 戴工作手套
2. `attach clamp to table` — 把夹具固定到桌面
3. `secure wooden block in clamp` — 把木块夹紧
4. `measure and mark drill position` — 测量并标记约4厘米处；属性：`{"approximate_distance": "about 4 cm"}`
5. `retrieve drill machine` — 取钻机
6. `select unnamed drilling attachment` — 选择名称不确定的钻孔附件
7. `retrieve extension cord` — 取延长线
8. `plug in extension cord` — 接通延长线
9. `plug in drill machine` — 把钻机接入电源
10. `turn on drill machine` — 启动钻机
11. `set up drill machine` — 设置钻机
12. `drill four-millimeter hole` — 钻4毫米孔；属性：`{"diameter": "4 mm"}`
13. `inspect drilled hole` — 检查钻孔
14. `smooth wooden block` — 用砂纸打磨木块
15. `remove wooden block` — 取下木块检查
16. `remove clamp` — 拆下夹具
17. `clean wood chips` — 用刷子清理木屑
18. `discard wood chips` — 把木屑丢入垃圾桶

## 已确认的工具与使用动作

- **clamp**：把木块夹紧（`secure wooden block in clamp`）
- **measuring tape**：测量并标记约4厘米处（`measure and mark drill position`）
- **pencil**：测量并标记约4厘米处（`measure and mark drill position`）
- **drill machine**：设置钻机（`set up drill machine`）；钻4毫米孔（`drill four-millimeter hole`）
- **unnamed drilling attachment**：设置钻机（`set up drill machine`）；钻4毫米孔（`drill four-millimeter hole`）
- **extension cord**：把钻机接入电源（`plug in drill machine`）
- **sandpaper**：用砂纸打磨木块（`smooth wooden block`）
- **brush**：用刷子清理木屑（`clean wood chips`）

## 已确认内容

- 4毫米孔、约4厘米标记、全部工具与清理步骤均有直接证据。

## 暂不纳入 Gold

- 排除钻孔附件的未经确认标准名称和测量参考边。

## 文件内统计

- Action：18
- Object：8
- Tool：8
- Relation：46
- Quality result / warning：1
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
