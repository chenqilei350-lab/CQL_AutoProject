# Gold V3 全量优化报告

本轮针对三类问题逐条复核：遗漏的中间动作、无法确认却强加的动作顺序、藏在转录句子中的工具。完整视频转录只在与当前 scene 的动作/keystep 对齐时使用。

## 优化前后

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| Action | 117 | 224 |
| Tool | 13 | 34 |
| USES_TOOL | 16 | 72 |
| 待确认点 | 75 | 35 |

## 逐样本统计

| 编号 | 场景 | Action | Tool | USES_TOOL | 待确认 |
| --- | --- | ---: | ---: | ---: | ---: |
| S01 | `indego_7020e12d-d5a2-4f01-beab-94f38d887eca_s2` | 2 | 0 | 0 | 1 |
| S02 | `indego_user_15_411_2309_2_1_s1` | 8 | 2 | 7 | 2 |
| S03 | `indego_user_14_user_1_1410_2_s1` | 12 | 1 | 1 | 2 |
| S04 | `indego_user_14_user_15_2410_1_s1` | 9 | 1 | 1 | 2 |
| S05 | `indego_232cf338-98a7-4cdc-8683-417c0e2adf7a_s1` | 10 | 0 | 0 | 2 |
| S06 | `indego_user_15_411_2309_2_s1` | 9 | 3 | 7 | 1 |
| S07 | `indego_user_15_412_2509_1_s1` | 15 | 4 | 12 | 1 |
| S08 | `indego_user_14_user_16_407_1712_1_s1` | 7 | 0 | 0 | 2 |
| S09 | `indego_user_14_407_0110_1_s1` | 18 | 1 | 5 | 1 |
| S10 | `indego_user_15_412_0110_1_s1` | 25 | 1 | 8 | 2 |
| S11 | `indego_user_14_user_15_411_1010_1_s1` | 14 | 2 | 2 | 1 |
| S12 | `indego_user_2_407_2910_1_s1` | 15 | 3 | 10 | 2 |
| S13 | `indego_user_14_412_02024_1_s1` | 12 | 2 | 3 | 1 |
| S14 | `indego_user_14_412_02024_2_s1` | 11 | 0 | 0 | 1 |
| S15 | `indego_user_17_user_15_451_0402_1_s3` | 8 | 1 | 1 | 3 |
| S16 | `indego_user_14_411_2409_1_s1` | 18 | 8 | 10 | 2 |
| S17 | `indego_user_14_414_1712_woodworking_2_s1` | 13 | 2 | 2 | 2 |
| S18 | `indego_user_15_414_0702_1_s11` | 8 | 1 | 1 | 1 |
| S19 | `indego_warning_A_Task_08_s1` | 4 | 2 | 2 | 2 |
| S20 | `indego_warning_A_Task_19_s1` | 6 | 0 | 0 | 4 |

## 强制验证

- 每个 Action 的证据必须能在 source_text 中找到。
- 每个 Tool 必须至少被一条 USES_TOOL 使用；只出现但未实际使用的物品不伪装成工具关系。
- ACTS_ON 只能指向 Object，USES_TOOL 只能指向 Tool，BEFORE 只能连接 Action。
- 有明确叙述/时间顺序的样本只保留相邻 BEFORE；多 run warning 样本不建立跨 run BEFORE。
- 继续排除桌子/书桌拆卸、电脑组装和参考 JSONL 的全部 scene_id。

最终自动验证：**passed**。
