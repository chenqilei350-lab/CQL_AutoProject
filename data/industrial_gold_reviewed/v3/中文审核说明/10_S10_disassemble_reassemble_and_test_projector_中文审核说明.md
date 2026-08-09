# S10 拆装并测试投影仪

**场景编号：** `indego_user_15_412_0110_1_s1`  
**类别：** `inspection_repair`  
**英文任务：** disassemble, reassemble, and test projector  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 恢复25个实际动作，包括失败尝试、回装、检查和现场收尾。
- 把 screwdriver 绑定到所有明确的拆装螺钉步骤。
- 明确区分“讲解如何使用”与“实际通电测试”，不再把未通电讲解写成 test passed。

## S10 拆装并测试投影仪 仍有 2 处需要确认

1. **待确认：battery 是否确为电池而不是 ASR 错词？**

   当前处理：完整转录多次一致使用 battery，保留 battery compartment，但不补充电池型号。

2. **待确认：两把螺丝刀的具体尺寸分别是多少？**

   当前处理：来源只说 two / different one，合并为通用 screwdriver 工具，不猜尺寸。

## 当前保留的 Gold 动作顺序

1. `turn on light` — 开灯
2. `move projector to workstation` — 把投影仪搬到工作台
3. `retrieve two screwdrivers` — 取两把螺丝刀
4. `select screwdriver` — 选择更合适的螺丝刀
5. `unscrew battery-cover screw` — 逆时针松开电池仓盖螺钉
6. `pry off battery cover` — 用手指撬下电池仓盖
7. `inspect battery compartment` — 检查电池仓
8. `attempt to loosen battery-compartment screws` — 尝试松开电池仓内部螺钉
9. `retighten battery-compartment screws` — 把内部螺钉重新拧回
10. `open filter cover` — 打开滤网盖
11. `remove filter` — 小心取出滤网
12. `unscrew lower-cover screws` — 松开下盖三颗螺钉
13. `loosen side screws` — 松开两颗侧面螺钉
14. `attempt to remove large cover` — 尝试拆下大盖板
15. `inspect inside through handle opening` — 通过把手开口检查内部
16. `reinstall long screws` — 装回长螺钉
17. `reinstall side screws` — 装回侧面螺钉
18. `reinstall battery cover` — 装回电池仓盖
19. `reinstall filter` — 装回滤网
20. `reinstall filter grate` — 装回滤网格栅
21. `attempt to locate power cable` — 尝试寻找电源线
22. `demonstrate projector controls without power test` — 在未通电情况下讲解投影仪控制
23. `replace lens cap` — 装回镜头盖
24. `return screwdrivers` — 归还两把螺丝刀
25. `turn off light` — 关灯

## 已确认的工具与使用动作

- **screwdriver**：逆时针松开电池仓盖螺钉（`unscrew battery-cover screw`）；尝试松开电池仓内部螺钉（`attempt to loosen battery-compartment screws`）；把内部螺钉重新拧回（`retighten battery-compartment screws`）；松开下盖三颗螺钉（`unscrew lower-cover screws`）；松开两颗侧面螺钉（`loosen side screws`）；装回长螺钉（`reinstall long screws`）；装回侧面螺钉（`reinstall side screws`）；装回电池仓盖（`reinstall battery cover`）

## 已确认内容

- 实际没有找到电源线，因此未执行通电测试。
- 大盖板因担心损坏而未成功拆下。
- 拆装螺钉明确使用 screwdriver。

## 暂不纳入 Gold

- 排除螺丝刀尺寸推测和不存在的成功测试结果。

## 文件内统计

- Action：25
- Object：15
- Tool：1
- Relation：54
- Quality result / warning：2
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
