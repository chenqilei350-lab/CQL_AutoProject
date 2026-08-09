# S09 检查并维修投影仪

**场景编号：** `indego_user_14_407_0110_1_s1`  
**类别：** `inspection_repair`  
**英文任务：** inspect and repair projector  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 用18个转录级动作替换两轮 coarse 拆装标签。
- 补入内部检查、孔位对齐、接电和失败测试。
- 把 screwdriver 绑定到所有明确的拆/装螺钉动作。

## S09 检查并维修投影仪 仍有 1 处需要确认

1. **待确认：small bulb or a torch 的标准部件名称是什么？**

   当前处理：使用中性名称 internal light component，不判定为灯泡或手电筒。

## 当前保留的 Gold 动作顺序

1. `place projector on table` — 把投影仪放到桌面
2. `retrieve screwdriver` — 取螺丝刀
3. `remove small cover` — 拆下小盖板
4. `unscrew light-component screws` — 松开内部光源部件螺钉
5. `remove internal light component` — 取出内部光源部件
6. `reinstall internal light component` — 装回内部光源部件
7. `tighten light-component screws` — 紧固光源部件螺钉
8. `reinstall small cover` — 装回小盖板
9. `remove three upper-part screws` — 拆下上部三颗螺钉
10. `remove side screws` — 拆下侧面螺钉
11. `remove large cover` — 拆下大盖板
12. `inspect projector internals` — 检查投影仪内部
13. `install large cover` — 安装大盖板
14. `align cover holes` — 对齐盖板孔位
15. `tighten large-cover screws` — 紧固大盖板螺钉
16. `find power socket` — 寻找电源插座
17. `connect projector to power` — 接通投影仪电源
18. `test projector` — 测试投影仪

## 已确认的工具与使用动作

- **screwdriver**：松开内部光源部件螺钉（`unscrew light-component screws`）；紧固光源部件螺钉（`tighten light-component screws`）；拆下上部三颗螺钉（`remove three upper-part screws`）；拆下侧面螺钉（`remove side screws`）；紧固大盖板螺钉（`tighten large-cover screws`）

## 已确认内容

- 检查内部部件没有明显问题。
- 最终接电测试结果为 not working。
- screwdriver 的各个使用阶段明确。

## 暂不纳入 Gold

- 排除内部光源部件的未经确认标准型号。

## 文件内统计

- Action：18
- Object：10
- Tool：1
- Relation：41
- Quality result / warning：2
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
