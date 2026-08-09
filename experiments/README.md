# Versioned Experiments

This directory stores frozen baseline evidence and later change experiments.

- `baselines/industrial-v1-llama31-8b-20260809/` is the one-time V1 control.
- `changes/<change_id>/` stores only the changed variant and its paired comparison.
- A directory containing `COMPLETED` is immutable. Never overwrite or cherry-pick runs.
- Full LLM responses are compressed as `raw_runs.jsonl.gz`; CSV summaries remain readable.
- Raw LLM relation metrics and final post-processed metrics are reported separately.

The raw IndEgo/EASG files remain outside Git. Baseline manifests retain their
SHA256 fingerprints so a missing or changed source is visible.
