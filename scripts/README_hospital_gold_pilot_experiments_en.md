# Hospital Gold-Case Experiment Reproduction Guide

This directory contains two runnable scripts:

- `scripts/run_hospital_gold_pilot_experiments_zh.py`: full implementation with Chinese comments.
- `scripts/run_hospital_gold_pilot_experiments_en.py`: English-commented entry point that reuses the same implementation, so both entry points produce identical results.

## 1. Preparation

Go to the project root:

```bash
cd /Users/qileichen/Desktop/aut-kg-extraction-pipeline
```

Check that the gold-case data directory exists:

```bash
ls gold_case_admission_29366372
```

Start Ollama:

```bash
ollama serve
```

If Ollama is already running in the background, this command may report that the port is already in use. That is fine.

Pull the local models:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:latest
```

For the main experiments, at least this model is required:

```bash
ollama pull llama3.1:8b
```

## 2. Recommended Quick Run

Run only the P1/P3/P4 stress experiments:

```bash
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite stress
```

This is the best option for a quick group demonstration because it clearly shows:

- the effect of schema constraints;
- the effect of chunking on feasibility;
- the effect of unified text structure.

## 3. Run All Experiments

```bash
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite all
```

The full suite includes:

| Suite | Content |
| --- | --- |
| P1 | Minimal strict extraction sanity check |
| P1 Stress | Loose prompt vs strict schema prompt |
| P2 | One-shot full graph vs layered extraction |
| P3 | LabEvent chunk size |
| P3 Stress | Chunk size under dense narrative input |
| P4 | Raw vs unified LabEvent input |
| P4 Stress | Raw noisy text vs unified canonical structure |
| P5 | Model comparison |
| P6 | Repeated-run stability |

Note: `--suite all` can be slow because P2 one-shot and P3 stress large-chunk calls may run until timeout.

## 4. Run One Experiment

```bash
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite p1
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite p2
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite p3
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite p4
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite p5
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite p6
```

## 5. Use the Chinese-Commented Entry Point

The Chinese version uses the same arguments:

```bash
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite stress
```

Both entry points call the same implementation, so the results are identical.

## 6. Output Location

By default, outputs are written to:

```text
results/hospital_gold_repro_<timestamp>/
```

The output directory contains:

| File | Purpose |
| --- | --- |
| `all_results.csv` | Combined table for all experiments |
| `summary.md` | Markdown summary |
| `p1_*.csv` | P1 result tables |
| `p2_*.csv` | P2 result tables |
| `p3_*.csv` | P3 result tables |
| `p4_*.csv` | P4 result tables |
| `p5_*.csv` | P5 model comparison |
| `p6_*.csv` | P6 stability results |
| `*.svg` | Core plots, generated if matplotlib is installed |

Specify a custom output directory:

```bash
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite stress --output-dir results/my_stress_run
```

## 7. Common Parameters

```bash
python3 scripts/run_hospital_gold_pilot_experiments_en.py \
  --suite stress \
  --model llama3.1:8b \
  --timeout 120 \
  --output-dir results/stress_demo
```

| Parameter | Meaning |
| --- | --- |
| `--suite` | Select all/base/stress/p1/p2/p3/p4/p5/p6 |
| `--model` | Main model used by P1/P2/P3/P4 |
| `--models` | Comma-separated model list for P5/P6 |
| `--timeout` | Timeout in seconds for one Ollama call |
| `--repetitions` | Number of repeated runs for P6 |
| `--gold-dir` | Gold-case data directory |
| `--output-dir` | Output directory |

## 8. Interpretation Notes

The experiments have two roles:

| Type | Interpretation |
| --- | --- |
| Base experiments | Pilot checks for the pipeline |
| Stress experiments | Controlled stress tests for variable effects and failure modes |

Avoid claiming:

> Unified text structure always improves KG generation.

A safer statement is:

> In stress-test conditions with noisy input and implicit IDs, unified text structure improved schema-conformant graph extraction by making canonical IDs, labels, and relation directions explicit.

