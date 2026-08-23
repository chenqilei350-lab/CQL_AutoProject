# Industrial Gold V2 Human-Reviewed Development Set

This directory is the unified evaluation entry point for reviewed records
S01 through S16. It contains 12 scene segments from 11 independent videos.
S08, S11, S13, and S14 are explicitly excluded because of substantial overlap.

## Files

- `benchmark.jsonl`: directly loadable by `BenchmarkDataset`, raw/unified
  experiments, and graph evaluation.
- `manifest.json`: dataset size, categories, nodes, relations, video grouping,
  exclusions, and evaluation constraints.
- `../industrial_gold_candidates/industrial_gold_v2_reviewed.jsonl`: complete
  human-reviewed audit source with node attributes, quality checks, outcomes,
  corrections, and reviewer notes.
- `../industrial_gold_candidates/industrial_gold_v2_exclusions.jsonl`: evidence
  for excluding duplicate samples.

## Loading

```python
from backend.datasets.benchmark import BenchmarkDataset
from backend.datasets.industrial_gold import load_industrial_gold_dataset

# Complete reviewed contract for auditing and further editing.
reviewed = load_industrial_gold_dataset()

# View consumed directly by the current experiment runner.
benchmark = BenchmarkDataset.from_jsonl(
    "data/industrial_gold_v2/benchmark.jsonl",
    name="industrial_gold_v2_reviewed",
)
```

Revalidate and rebuild the evaluation package:

```bash
uv run python scripts/build_industrial_gold_v2_package.py
```

## Usage Boundaries

- Split by `video_id`, never randomly by `scene_id`. S01 and S07 are different
  time ranges from the same video and must remain in the same split.
- Excluded scenes must not enter training, few-shot prompts, or evaluation.
- `benchmark.jsonl` follows the current graph-scoring contract for Action,
  Object, and Tool nodes plus BEFORE, ACTS_ON, and USES_TOOL relations.
- Quality outcomes, assembly errors, conditional actions, and uncertainty follow
  the reviewed audit source. They remain in audit data and unified input but are
  not yet all represented by the current graph scorer.
- These 12 scenes are suitable for development and pipeline regression, but are
  not independently large enough for statistically representative evaluation.
