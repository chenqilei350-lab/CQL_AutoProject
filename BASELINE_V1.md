# Industrial KG Baseline V1

This branch is the reproducible code control for the industrial text-to-KG
pipeline. It is intended for team download, local reproduction, and future
paired change experiments.

## Frozen Identity

| Item | Value |
| --- | --- |
| Baseline ID | `industrial-v1-llama31-8b-20260809` |
| Branch | `codex/industrial-baseline-v1` |
| Result tag | `baseline-industrial-v1-2026-08-09` after completion |
| Tested code commit | `28899d59b7b383a3e0d79a8624d313924b6e7bce` |
| Model | `llama3.1:8b` |
| Ollama digest | `46e0c10c039e` |
| Reference machine | Apple M4 MacBook Air, 16 GB |
| Current regression suite | `137 passed` |

The annotated result tag is deliberately created only after the one-time
256-call experiment has finished, all required artifacts exist, and checksums
pass. A partial checkpoint is not a final baseline result.

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
