# Public Industrial KG Baseline V1 Results

This directory is the license-safe public export of `industrial-v1-llama31-8b-20260809`.
It contains aggregate and per-scene numeric results for the completed 256-call
local LLM baseline. It deliberately excludes licensed IndEgo-derived text,
cleaned records, prompts, evidence excerpts, and raw model responses.

The canonical local artifact remains under `experiments/baselines/` and is
required when authorized researchers need raw-output rescoring. The source
identity in `public_manifest.json` is an aggregate hash, so authorized copies
can be checked without publishing local paths or source contents.

`reviewed_unified` is a human-derived upper-bound condition. It must not be
reported as the performance of automatic text preprocessing.
