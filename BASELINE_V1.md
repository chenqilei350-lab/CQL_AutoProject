# Industrial KG Baseline V1

This branch is the reproducible code control for the industrial text-to-KG
pipeline. It is intended for team download, local reproduction, and future
paired change experiments.

## Frozen Identity

| Item | Value |
| --- | --- |
| Baseline ID | `industrial-v1-llama31-8b-20260809` |
| Branch | `codex/industrial-baseline-v1` |
| Result tag | `baseline-industrial-v1-2026-08-09` |
| Tested code commit | `28899d59b7b383a3e0d79a8624d313924b6e7bce` |
| Model | `llama3.1:8b` |
| Ollama digest | `46e0c10c039e` |
| Reference machine | Apple M4 MacBook Air, 16 GB |
| Baseline implementation suite | `137 passed` |
| Publication suite | `139 passed` |

The one-time 256-call experiment is complete. All required artifacts exist and
14 recorded checksums pass. The annotated result tag points to the commit that
adds these immutable results. A partial checkpoint was never published as a
final baseline result.

## Final Baseline Results

| Condition | Errors / 64 | Mean Node F1 | Mean raw Edge F1 | Mean final Edge F1 | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw + one_shot` | 6 | 0.2374 | 0.0260 | 0.0799 | 91.10 s |
| `raw + layered` | 5 | 0.2804 | 0.0947 | 0.1009 | 101.87 s |
| `auto_unified + layered` | 3 | 0.2505 | 0.0418 | 0.0704 | 85.68 s |
| `reviewed_unified + layered` | 12 | 0.7321 | 0.4423 | 0.6015 | 114.39 s |

`reviewed_unified` is a human-derived upper bound, not an automatic
preprocessing result. The automatic condition is faster and has fewer errors,
but it does not improve Edge F1 over raw layered extraction. The detailed
interpretation is in `docs/results/industrial_baseline_v1_report_en.md`.

## Public and Private Artifacts

The complete immutable baseline is retained locally at
`experiments/baselines/industrial-v1-llama31-8b-20260809/`. It includes
licensed cleaned records and raw model responses needed for authorized
rescoring, so the directory is ignored by Git.

The public repository contains a sanitized export at
`experiments/public_baselines/industrial-v1-llama31-8b-20260809/`. It contains
aggregate and per-scene numeric metrics, a path-free source identity hash, and
its own checksums. It excludes source text, prompts, evidence excerpts, raw
responses, and local source paths. Recreate or verify it with:

```bash
uv run python scripts/export_public_baseline.py
uv run python scripts/export_public_baseline.py --verify-only
```

## Install

Prerequisites: macOS or Linux, Python 3.12, `uv`, and Ollama.

```bash
git clone https://github.com/chenqilei350-lab/CQL_AutoProject.git
cd CQL_AutoProject
git switch codex/industrial-baseline-v1
uv sync
ollama pull llama3.1:8b
ollama serve
```

Use the project environment created by `uv`; do not install the project into a
shared Conda base environment.

## Convert One Industrial Text

Prepare a UTF-8 `.txt` file containing an industrial procedure or scene
description, then run:

```bash
uv run python scripts/run_custom_industrial_text_to_kg.py \
  --input /absolute/path/to/input.txt \
  --output results/custom_industrial_kg_output.json \
  --condition unified \
  --strategy layered \
  --model llama3.1:8b
```

The JSON output contains the structured extraction, graph nodes and edges,
relation-origin metadata, raw model response, and Cypher text.

For a directory of `.txt` files:

```bash
uv run python scripts/run_batch_industrial_texts_to_kg.py \
  --input-dir /absolute/path/to/input_texts \
  --output-dir /absolute/path/to/output_kg \
  --condition unified \
  --strategy layered \
  --model llama3.1:8b
```

## Validate the Checkout

```bash
uv run ruff check backend scripts tests
uv run python -m pytest -q
```

## Three Versioned Modules

| Module | Responsibility | Main paths |
| --- | --- | --- |
| `data_cleaning` | Parse source files and preserve provenance | `backend/cleaning/` |
| `text_preprocessing` | Build evidence-grounded unified text | `backend/preprocessing/`, IndEgo adapter |
| `kg_extraction` | Extract, construct, validate, and evaluate KGs | extraction, pipeline, graph, evaluation |

Module versions are stored in `config/module_versions.json`. Any optimization
must receive a unique Change ID and be compared with saved V1 artifacts instead
of rerunning the baseline model.

## Data Boundary

The repository does not publish licensed raw IndEgo or Ego4D/EASG source data.
The reviewed project Gold included in Git contains only the approved benchmark
records required for reproducible scoring. Every Gold fact must be supported by
source evidence; inferred or AI-added facts are excluded.

## Experiment Files

- Baseline runner: `scripts/run_versioned_baseline.py`
- Change runner: `scripts/run_change_experiment.py`
- Offline rescoring: `scripts/rescore_versioned_baseline.py`
- Chinese protocol: `docs/experiment_baseline_protocol_zh.md`
- English plan: `docs/industrial_baseline_and_improvement_plan_en.md`
- Artifact contract: `experiments/README.md`

Do not edit an experiment directory containing `COMPLETED`. Failed calls,
timeouts, and parse failures remain part of the aggregate result.
