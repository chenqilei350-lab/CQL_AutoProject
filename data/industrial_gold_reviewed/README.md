# Reviewed Industrial Gold Datasets

This directory keeps reviewed datasets separate from candidate annotation templates.

## Canonical files

- `../industrial_gold_candidates/industrial_gold_v2_reviewed.jsonl`: original reviewed V2, 12 scenes.
- `v3/industrial_gold_v3_reviewed.jsonl`: teammate-provided reviewed V3, 20 scenes.
- `industrial_gold_combined_v2_v3_reviewed.jsonl`: reproducible 32-scene combination used by current WP7 experiments.

The combined file is generated with:

```bash
uv run python scripts/build_combined_industrial_gold.py
```

The builder validates each source with the project adapter, rejects non-accepted records and duplicate `scene_id` values, and never modifies the V2 or V3 sources.

## Review boundary

All formal Gold entities and relations must be supported by the corresponding `source_text`. Review notes and unclear points remain audit metadata and do not become graph facts.
