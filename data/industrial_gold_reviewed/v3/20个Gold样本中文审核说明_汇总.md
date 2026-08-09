# 20个 Gold 样本中文审核说明·全量优化版（排除桌子拆卸与电脑组装）

本文件按 S01–S20 汇总。上传的 `industrial_gold_v2_reviewed.jsonl` 只用于参考字段结构和人工审核风格，其中任何样本都没有复制到本批次。本轮已逐条补齐明确的中间动作、检查、清理、纠错和实际使用工具；所有待确认内容均未被当作确定事实写入 Gold。

# S01 完成相机框架底座连接并清理

**场景编号：** `indego_7020e12d-d5a2-4f01-beab-94f38d887eca_s2`  
**类别：** `assembly`  
**英文任务：** finish camera-frame assembly and clean up  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 重新核对完整视频转录与 s2 时间范围；视频中的相机安装、稳定性检查位于本片段之前，没有错误混入 s2。
- 保留 s2 内明确的连接底座和清理两步，并保留相邻时序。

## S01 完成相机框架底座连接并清理 仍有 1 处需要确认

1. **待确认：clean up 的具体清理对象和工具是什么？**

   当前处理：s2 注释和片段文本均未命名，因此只保留清理动作，不补猜工具或对象。

## 当前保留的 Gold 动作顺序

1. `connect base` — 连接底座
2. `clean up` — 清理工作区

## 已确认的工具与使用动作

- 本 scene 没有能够由来源确认实际使用的具名工具；未根据常识补写工具。

## 已确认内容

- s2 的时间范围仅覆盖 connect base 和 clean up。
- 完整视频中更早的相机安装和检查不属于 s2，已排除。

## 暂不纳入 Gold

- 排除完整视频中位于 s2 之前的装配、纠错和稳定性检查，不跨片段复制动作。

## 文件内统计

- Action：2
- Object：1
- Tool：0
- Relation：2
- Quality result / warning：0
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

# S02 组装相机支架

**场景编号：** `indego_user_15_411_2309_2_1_s1`  
**类别：** `assembly`  
**英文任务：** assemble camera mount  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入中途松开、改变顺序并重新安装的纠错过程。
- 补入最后安装相机时实际使用的 Allen key；不再只记录前三步的 screwdriver。
- 补入完成后的整体检查。

## S02 组装相机支架 仍有 2 处需要确认

1. **待确认：big piece、small piece 分别对应哪一个标准杆件？**

   当前处理：保留为 support sections，不强行映射 vertical/upper bar。

2. **待确认：多次 bracket 是否为同一个零件实例？**

   当前处理：不建立实例同一性，只保留框架安装动作。

## 当前保留的 Gold 动作顺序

1. `connect base with screw` — 用螺钉连接底座
2. `attach first support section` — 安装第一段支撑结构
3. `loosen misaligned connection` — 松开未对齐的连接
4. `reattach support in corrected order` — 按修正顺序重新安装支撑结构
5. `attach remaining frame sections` — 安装其余框架段
6. `attach camera with small screws` — 用小螺钉安装相机
7. `tighten camera screws` — 用内六角扳手紧固相机螺钉
8. `inspect completed camera mount` — 检查完成的相机支架

## 已确认的工具与使用动作

- **screwdriver**：用螺钉连接底座（`connect base with screw`）；安装第一段支撑结构（`attach first support section`）；松开未对齐的连接（`loosen misaligned connection`）；按修正顺序重新安装支撑结构（`reattach support in corrected order`）；安装其余框架段（`attach remaining frame sections`）
- **Allen key**：用小螺钉安装相机（`attach camera with small screws`）；用内六角扳手紧固相机螺钉（`tighten camera screws`）

## 已确认内容

- screwdriver 用于前部框架连接和纠错。
- small screws 与 Allen key 明确用于安装并紧固相机。
- 纠错前后顺序由完整转录直接支持。

## 暂不纳入 Gold

- 排除无法确认的大小部件标准型号与 bracket 实例同一性。

## 文件内统计

- Action：8
- Object：6
- Tool：2
- Relation：24
- Quality result / warning：1
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

# S03 组装相机支架子组件

**场景编号：** `indego_user_14_user_1_1410_2_s1`  
**类别：** `assembly`  
**英文任务：** assemble camera-mount subassembly  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入开头结构检查，避免直接从安装动作开始。
- 补入松开、校直、复检稳定性与最后使用 trolley 转运。
- 不再把两次 attach vertical bars 简单重复而不解释中间纠错。

## S03 组装相机支架子组件 仍有 2 处需要确认

1. **待确认：bigger arm 对应哪个正式部件编号？**

   当前处理：保留原文名称 bigger arm。

2. **待确认：多人协作时每一步由谁完成？**

   当前处理：不创建 PERFORMED_BY。

## 当前保留的 Gold 动作顺序

1. `inspect structural integrity` — 检查结构完整性
2. `reposition loose component` — 重新定位松动部件
3. `align holes and guide bolt` — 对齐孔位并导入螺栓
4. `position bigger arm` — 定位较大的支臂
5. `correct misalignment` — 松开并纠正未对齐位置
6. `install side plates and bars` — 安装侧板和杆件
7. `fix upper part` — 固定上部组件
8. `insert guided pin` — 插入导向销
9. `tighten upper bolt` — 紧固上部螺栓
10. `replace cap` — 装回端帽
11. `inspect final stability` — 检查最终稳定性
12. `transfer assembly on trolley` — 用推车转运组件

## 已确认的工具与使用动作

- **trolley**：用推车转运组件（`transfer assembly on trolley`）

## 已确认内容

- 检查—调整—安装—复检—转运的顺序由完整转录支持。
- trolley 明确用于转运 assembly。

## 暂不纳入 Gold

- 排除操作者归属与 bigger arm 的未经证实标准化名称。

## 文件内统计

- Action：12
- Object：9
- Tool：1
- Relation：25
- Quality result / warning：1
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

# S04 重新组装台灯铰链

**场景编号：** `indego_user_14_user_15_2410_1_s1`  
**类别：** `assembly`  
**英文任务：** reassemble lamp hinges  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 用完整转录替换重复且含义不清的 lower/middle/upper hinge 粗标签。
- 补入螺钉、螺栓、方孔板、另一侧板和弹簧的实际安装顺序。
- 把句子中的 hand twister 明确记录为紧固工具。

## S04 重新组装台灯铰链 仍有 2 处需要确认

1. **待确认：hand twister 的标准工具名称是什么？**

   当前处理：保留原文名称，不改写成旋钮或扳手。

2. **待确认：末尾 it works / it should 是否构成正式测试通过？**

   当前处理：没有明确测试动作，结果不进入正式 Gold。

## 当前保留的 Gold 动作顺序

1. `attach lamp mount` — 安装台灯底座连接件
2. `insert shorter part` — 插入较短部件
3. `insert mounting screw` — 插入安装螺钉
4. `tighten bolt with hand twister` — 用手拧工具紧固螺栓
5. `install first square-hole plate` — 安装第一块方孔板
6. `insert connecting screw` — 插入连接螺钉
7. `install opposite plates` — 安装另一侧板件
8. `tighten lamp joint` — 紧固台灯连接处
9. `install springs` — 安装弹簧

## 已确认的工具与使用动作

- **hand twister**：用手拧工具紧固螺栓（`tighten bolt with hand twister`）

## 已确认内容

- hand twister 明确用于紧固。
- mount、螺钉、方孔板、另一侧板与 springs 的操作顺序明确。

## 暂不纳入 Gold

- 排除 hand twister 的猜测性标准化名称和未经测试的成功结果。

## 文件内统计

- Action：9
- Object：8
- Tool：1
- Relation：19
- Quality result / warning：0
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

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


---

# S06 拆卸相机支架

**场景编号：** `indego_user_15_411_2309_2_s1`  
**类别：** `disassembly`  
**英文任务：** disassemble camera mount  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 把小金属件、顶部件、角形件与框架段拆分补回动作序列。
- 将 small Allen wrench、Allen key、screwdriver 分别绑定到实际步骤。
- 补入最后确认全部拆开的检查动作。

## S06 拆卸相机支架 仍有 1 处需要确认

1. **待确认：small metal part 和 angle type thing 的标准部件名称是什么？**

   当前处理：保留原文描述，不猜测标准部件名。

## 当前保留的 Gold 动作顺序

1. `unscrew camera screws` — 松开相机小螺钉
2. `remove camera` — 取下相机
3. `detach small metal part` — 拆下小金属件
4. `remove top part` — 拆下顶部部件
5. `remove angled part` — 拆下角形部件
6. `separate first frame sections` — 分离第一组框架段
7. `separate next frame section` — 分离下一段框架
8. `disassemble base section` — 拆解底座段
9. `inspect disassembled parts` — 检查全部部件已拆开

## 已确认的工具与使用动作

- **small Allen wrench**：松开相机小螺钉（`unscrew camera screws`）
- **Allen key**：拆下小金属件（`detach small metal part`）；拆下顶部部件（`remove top part`）；拆下角形部件（`remove angled part`）；分离第一组框架段（`separate first frame sections`）；分离下一段框架（`separate next frame section`）
- **screwdriver**：拆解底座段（`disassemble base section`）

## 已确认内容

- 三类工具与对应拆卸阶段在完整转录中明确。
- 所有动作的先后由叙述顺序直接支持。

## 暂不纳入 Gold

- 排除小金属件和角形件的未经确认标准化名称。

## 文件内统计

- Action：9
- Object：7
- Tool：3
- Relation：26
- Quality result / warning：0
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

# S07 拆卸相机支架子组件

**场景编号：** `indego_user_15_412_2509_1_s1`  
**类别：** `disassembly`  
**英文任务：** disassemble camera-mount subassembly  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 补入安全鞋、清理工作台、wagon 转运与端帽拆卸。
- 补入 screwdriver、5号红色 Allen key、6号 Allen key 的逐步工具关系。
- 保留拆卸失败后把螺钉装回的纠错动作。

## S07 拆卸相机支架子组件 仍有 1 处需要确认

1. **待确认：转录中短暂提到的 purple key 是否实际使用？**

   当前处理：说话人立即改为 size-6 key，purple key 不进入正式 Gold。

## 当前保留的 Gold 动作顺序

1. `put on steel-cap shoes` — 穿钢头安全鞋
2. `clear workstation` — 移走显示器和鼠标以清理工作台
3. `load assembly onto wagon` — 把组件装上运输车
4. `transport assembly to workstation` — 把组件运到工作台
5. `pry off caps` — 用螺丝刀撬下端帽
6. `loosen big-plate screws` — 用5号红色内六角扳手松开大板螺钉
7. `remove big plate` — 拆下大板
8. `loosen metal-bracket screws` — 松开金属支架螺钉
9. `remove metal brackets` — 拆下金属支架
10. `detach short metal rod` — 从长杆拆下短金属杆
11. `loosen larger connector` — 用6号内六角扳手松开较大连接件
12. `attempt to detach small metal bar` — 尝试拆下小金属杆
13. `restore unremovable screws` — 把无法拆下位置的螺钉装回
14. `detach first large section` — 分离第一侧大组件
15. `detach matching section` — 分离另一侧对应组件

## 已确认的工具与使用动作

- **transport wagon**：把组件装上运输车（`load assembly onto wagon`）；把组件运到工作台（`transport assembly to workstation`）
- **screwdriver**：用螺丝刀撬下端帽（`pry off caps`）
- **size-5 red Allen key**：用5号红色内六角扳手松开大板螺钉（`loosen big-plate screws`）；松开金属支架螺钉（`loosen metal-bracket screws`）；拆下金属支架（`remove metal brackets`）；从长杆拆下短金属杆（`detach short metal rod`）
- **size-6 Allen key**：用6号内六角扳手松开较大连接件（`loosen larger connector`）；尝试拆下小金属杆（`attempt to detach small metal bar`）；把无法拆下位置的螺钉装回（`restore unremovable screws`）；分离第一侧大组件（`detach first large section`）；分离另一侧对应组件（`detach matching section`）

## 已确认内容

- wagon、screwdriver、size-5 与 size-6 Allen key 的实际用途明确。
- 无法松开小杆螺钉及随后恢复的顺序明确。

## 暂不纳入 Gold

- 排除没有确认实际使用的 purple key。

## 文件内统计

- Action：15
- Object：13
- Tool：4
- Relation：42
- Quality result / warning：1
- 待确认点：1

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

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


---

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


---

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


---

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


---

# S12 检查并维修演示装置子组件

**场景编号：** `indego_user_2_407_2910_1_s1`  
**类别：** `inspection_repair`  
**英文任务：** inspect and repair demonstrator subassembly  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 用15个详细动作替换 inspect/repair 两个粗标签。
- 补入翻转安装板的纠错、短杆和旋钮重定位、盖板安装。
- 补齐 Allen key、bigger Allen key、screwdriver 的逐步工具关系。

## S12 检查并维修演示装置子组件 仍有 2 处需要确认

1. **待确认：side pieces 与 final bar 的正式组件编号是什么？**

   当前处理：使用转录级描述，不猜测编号。

2. **待确认：转录在最后安装尚未说完时截断，是否最终完成？**

   当前处理：保留已开始且有明确证据的 attach final bar and knob，不记录完成 outcome。

## 当前保留的 Gold 动作顺序

1. `put on working gloves` — 戴工作手套
2. `detach plate` — 拆下安装板
3. `reposition rectangular frame` — 重新定位矩形框架
4. `reattach plate with four screws` — 用四颗螺钉装回安装板
5. `correct plate orientation` — 翻转安装板以纠正方向
6. `tighten four plate bolts` — 紧固四颗安装板螺栓
7. `detach incorrectly positioned short bars` — 拆下位置错误的短杆
8. `reposition knob` — 重新定位旋钮
9. `attach side pieces with two bolts` — 用两颗螺栓安装侧部件
10. `align side piece` — 对齐侧部件
11. `reposition top bar` — 重新定位顶部杆
12. `tighten top-bar bolts` — 紧固顶部杆螺栓
13. `install cover` — 安装盖板
14. `fasten cover` — 用螺丝刀固定盖板
15. `attach final bar and knob` — 安装最后一根杆和旋钮

## 已确认的工具与使用动作

- **Allen key**：拆下安装板（`detach plate`）；用四颗螺钉装回安装板（`reattach plate with four screws`）；紧固四颗安装板螺栓（`tighten four plate bolts`）；拆下位置错误的短杆（`detach incorrectly positioned short bars`）
- **bigger Allen key**：重新定位旋钮（`reposition knob`）；用两颗螺栓安装侧部件（`attach side pieces with two bolts`）；重新定位顶部杆（`reposition top bar`）；紧固顶部杆螺栓（`tighten top-bar bolts`）；安装最后一根杆和旋钮（`attach final bar and knob`）
- **screwdriver**：用螺丝刀固定盖板（`fasten cover`）

## 已确认内容

- 安装板方向纠错和四螺栓紧固明确。
- 三类工具的使用阶段明确。

## 暂不纳入 Gold

- 排除最终完成状态和未提供的部件编号。

## 文件内统计

- Action：15
- Object：12
- Tool：3
- Relation：42
- Quality result / warning：0
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

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


---

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


---

# S15 称重并包装货物

**场景编号：** `indego_user_17_user_15_451_0402_1_s3`  
**类别：** `logistics_organisation`  
**英文任务：** weigh and pack goods  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 复核完整视频转录与带时间戳 action annotation；正式 Gold 继续以精确动作注释为主。
- scale 已作为 weigh goods 的工具保留。
- double-sided tape 只被提到、未明确执行使用，因此没有错误添加 USES_TOOL。

## S15 称重并包装货物 仍有 3 处需要确认

1. **待确认：drop pallet 是放下还是意外掉落？**

   当前处理：保留注释原词，中文采用中性“放下”。

2. **待确认：double-sided tape 是否实际用于 pack goods？**

   当前处理：转录只讨论现场已有胶带，没有明确使用动作，不建立工具关系。

3. **待确认：两次 pick goods 是否不同货物？**

   当前处理：保留两个动作，共用 generic goods。

## 当前保留的 Gold 动作顺序

1. `drop pallet` — 放下托盘
2. `pick goods` — 拿取货物（第一次）
3. `open scale` — 打开秤
4. `take out goods` — 取出货物
5. `weigh goods` — 称重货物
6. `put back goods` — 放回货物
7. `pack goods` — 包装货物
8. `pick goods` — 拿取货物（第二次）

## 已确认的工具与使用动作

- **scale**：称重货物（`weigh goods`）

## 已确认内容

- 八个 action annotation 带明确时间戳。
- scale 明确对应 weigh goods。

## 暂不纳入 Gold

- 排除仅被提及但未明确使用的 double-sided tape，以及货物种类推测。

## 文件内统计

- Action：8
- Object：2
- Tool：1
- Relation：15
- Quality result / warning：0
- 待确认点：3

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

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


---

# S17 涂装并重新组装木箱

**场景编号：** `indego_user_14_414_1712_woodworking_2_s1`  
**类别：** `woodworking`  
**英文任务：** paint and reassemble wooden box  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 把句子中明确出现的 rag 和 band 都加入涂装工具关系。
- 补入箱盖开合检查、搭扣紧固、最终检查。
- 恢复官方 clean up keystep，不再在组装完成处提前结束。

## S17 涂装并重新组装木箱 仍有 2 处需要确认

1. **待确认：band 的标准工具类型是什么？**

   当前处理：作为 unidentified band tool 保留原文，不猜成刷子或绑带。

2. **待确认：clean up 的具体对象是什么？**

   当前处理：注释明确有动作，但未给对象，所以不建立 ACTS_ON。

## 当前保留的 Gold 动作顺序

1. `put on gloves` — 戴手套
2. `put on mask` — 戴口罩
3. `paint inside of wooden box` — 用抹布和未识别 band 涂装木箱内部
4. `attach box cover` — 安装箱盖
5. `align hinge holes` — 对齐铰链孔
6. `fasten hinge screws` — 紧固铰链螺钉
7. `inspect cover movement` — 检查箱盖开合
8. `install clasp` — 安装搭扣
9. `fasten clasp` — 紧固搭扣
10. `install unnamed upper parts` — 安装未命名的上部零件
11. `align remaining holes` — 对齐剩余孔位
12. `inspect completed box` — 检查完成的木箱
13. `clean up` — 清理工作区

## 已确认的工具与使用动作

- **rag**：用抹布和未识别 band 涂装木箱内部（`paint inside of wooden box`）
- **unidentified band tool**：用抹布和未识别 band 涂装木箱内部（`paint inside of wooden box`）

## 已确认内容

- rag 与 band 明确用于内侧涂装。
- 箱盖开合正常，箱体可使用。
- 组装之后存在独立 clean up keystep。

## 暂不纳入 Gold

- 排除 band 的猜测性标准名称和清理对象推测。

## 文件内统计

- Action：13
- Object：7
- Tool：2
- Relation：26
- Quality result / warning：2
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

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


---

# S19 钻孔错误检测流程模板

**场景编号：** `indego_warning_A_Task_08_s1`  
**类别：** `mistake_detection`  
**英文任务：** drilling mistake-detection procedure template  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 继续按 run_id 隔离11条 warning，避免不同失败运行形成错误时序。
- 标准步骤保留 prescribed_step 属性；缺失动作不会被当作已执行动作。
- 根据 keystep 与 warning 补入 clamp 和 drill 两个流程工具，但不把 warning 误当作完成动作。

## S19 钻孔错误检测流程模板 仍有 2 处需要确认

1. **待确认：11条 warning 是否来自同一次失败执行？**

   当前处理：不是；按各自 run_id 保留，禁止合并为一条实际时间线。

2. **待确认：plug in 的具体对象是什么？**

   当前处理：来源未说明是钻机、电源线还是延长线，不创建 ACTS_ON。

## 当前保留的 Gold 动作顺序

1. `mark hole` — 标记孔位；属性：`{"kind": "prescribed_step"}`
2. `clamp block` — 夹紧木块；属性：`{"kind": "prescribed_step"}`
3. `plug in` — 接通电源；属性：`{"kind": "prescribed_step"}`
4. `drill hole` — 钻孔；属性：`{"kind": "prescribed_step"}`

## 已确认的工具与使用动作

- **clamp**：夹紧木块（`clamp block`）
- **drill**：钻孔（`drill hole`）

## 已确认内容

- 四个步骤是 prescribed_step，而不是每个 run 都已完成。
- clamp 用于夹紧 block，drill 用于 drill hole。
- 每条 warning 的 step、description 和 run_id 均可追溯。

## 暂不纳入 Gold

- 排除跨 run BEFORE、plug in 对象推测，以及把 didn't... 转成已执行动作。

## 文件内统计

- Action：4
- Object：2
- Tool：2
- Relation：5
- Quality result / warning：11
- 待确认点：2

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---

# S20 包装错误检测流程模板

**场景编号：** `indego_warning_A_Task_19_s1`  
**类别：** `mistake_detection`  
**英文任务：** packaging mistake-detection procedure template  
**审核状态：** `accepted`（不确定内容已从正式 Gold 中排除或降级说明）

## 本轮已经补齐和优化

- 继续按 run_id 隔离12条 warning，避免不同失败运行形成错误时序。
- 未从错误描述中反推未明确的工具或正确数量。

## S20 包装错误检测流程模板 仍有 4 处需要确认

1. **待确认：12条 warning 是否描述一条连续失败流程？**

   当前处理：不是；它们属于多个 run_id，分别保留，不能串联。

2. **待确认：“not centered”相对于哪个设备或中心位置？**

   当前处理：缺少参照物，只作为 warning 文本保留。

3. **待确认：“only one belt”要求的正确捆扎带数量是多少？**

   当前处理：来源只说明一条不足，不推断标准数量。

4. **待确认：胶带、缠绕膜、捆扎带和标签是否需要创建 Tool/Object 实体？**

   当前处理：短模板未给出稳定角色与具体材料实例，只保留 package 作为操作对象。

## 当前保留的 Gold 动作顺序

1. `put on gloves` — 戴手套；属性：`{"kind": "prescribed_step"}`
2. `place package` — 放置包裹；属性：`{"kind": "prescribed_step"}`
3. `tape package` — 给包裹贴胶带；属性：`{"kind": "prescribed_step"}`
4. `wrap package` — 包裹缠膜；属性：`{"kind": "prescribed_step"}`
5. `belt package` — 给包裹打带；属性：`{"kind": "prescribed_step"}`
6. `label` — 贴标签；属性：`{"kind": "prescribed_step"}`

## 已确认的工具与使用动作

- 本 scene 没有能够由来源确认实际使用的具名工具；未根据常识补写工具。

## 已确认内容

- 六个包装标准步骤和12条带 run_id 的错误注释明确。

## 暂不纳入 Gold

- 未建立跨 run 时序、未推断中心参照或正确捆扎带数量。

## 文件内统计

- Action：6
- Object：2
- Tool：0
- Relation：6
- Quality result / warning：12
- 待确认点：4

> 审核原则：只把来源明确支持的事实写入 Gold；含糊代词、部件身份、工具映射、动作边界和结果一律在本说明中标出，不自行猜测。


---
