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
