# Industrial Gold v2（人工审核开发集）

该目录是 S01–S16 人工审核结果的统一评估入口。当前纳入 12 个场景片段，来自
11 个独立视频；S08、S11、S13、S14 因内容高度重复而明确排除。

## 文件

- `benchmark.jsonl`：可直接交给现有 `BenchmarkDataset`、Raw/Unified 实验与图评估流程。
- `manifest.json`：数据量、类别、节点、关系、视频分组、排除记录及评估约束。
- `../industrial_gold_candidates/industrial_gold_v2_reviewed.jsonl`：人工审核的完整审计源，
  包含节点属性、质量检查结果、结果状态、纠错和审核备注。
- `../industrial_gold_candidates/industrial_gold_v2_exclusions.jsonl`：重复样本排除依据。

## 加载

```python
from backend.datasets.benchmark import BenchmarkDataset
from backend.datasets.industrial_gold import load_industrial_gold_dataset

# 完整人工审核合同，适合审计和进一步编辑。
reviewed = load_industrial_gold_dataset()

# 当前实验运行器直接使用的视图。
benchmark = BenchmarkDataset.from_jsonl(
    "data/industrial_gold_v2/benchmark.jsonl",
    name="industrial_gold_v2_reviewed",
)
```

重新验证并生成评估包：

```bash
uv run python scripts/build_industrial_gold_v2_package.py
```

## 使用边界

- 数据集划分必须以 `video_id` 为单位，不能以 `scene_id` 随机划分。S01 与 S07
  是同一视频的不同时间段，必须进入同一个 split。
- 被排除的场景不能进入训练、few-shot prompt 或评估。
- `benchmark.jsonl` 当前对齐项目已有的图评分合同，评估 Action、Object、Tool 节点，
  以及 BEFORE、ACTS_ON、USES_TOOL 关系。
- 质量结果、装配错误、条件动作和不确定性以人工审核源为准；它们保留在审计数据和
  Unified 输入中，但尚未全部进入当前图评分器。
- 这 12 个场景适合作为开发集和流程回归测试，尚不足以单独构成有统计代表性的正式
  模型评估集。
