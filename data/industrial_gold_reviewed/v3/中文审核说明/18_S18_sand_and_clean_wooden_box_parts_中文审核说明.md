# S18 打磨并清洁木箱零件

**场景编号：** `indego_user_15_414_0702_1_s11`  
**类别：** `woodworking`  
**英文任务：** sand and clean wooden-box parts  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 利用完整转录把 unknown tool 识别为 sandpaper。
- 把 inspect 细化为检查尖锐边缘，并补入 cleaning agent 与 rag 的拿取。
- 只把实际已用于 sand 的 sandpaper 建为 USES_TOOL；清洁动作发生在本片段之后，不跨片段添加。

## S18 打磨并清洁木箱零件 仍有 1 处需要确认

1. **待确认：cleaning agent 与 rag 是否在 s11 内已经开始清洁？**

   当前处理：s11 动作注释只到拿取/放下，实际 clean 动作不在本段 Gold 中。

## 当前保留的 Gold 动作顺序

1. `remove parts` — 拆下零件
2. `inspect sharp edges` — 检查并发现边缘较尖锐
3. `pick up sandpaper` — 拿起砂纸
4. `sand sharp edges` — 打磨尖锐边缘
5. `put down sandpaper` — 放下砂纸
6. `pick up cleaning agent` — 拿起清洁剂
7. `pick up rag` — 拿起抹布
8. `put down cleaning agent` — 放下清洁剂

## 已确认的工具与使用动作

- **sandpaper**：打磨尖锐边缘（`sand sharp edges`）

## 已确认内容

- sandpaper 明确用于打磨尖锐边缘。
- cleaning agent 和 rag 在本段末尾被拿起。

## 暂不纳入 Gold

- 排除 s11 之后的实际清洁、上油和干燥动作。

## 文件内统计

- Action：8
- Object：5
- Tool：1
- Relation：16
- Quality result / warning：0
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。
