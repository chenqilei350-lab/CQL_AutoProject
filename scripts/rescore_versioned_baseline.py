#!/usr/bin/env python3
"""Rescore frozen baseline raw outputs without making any LLM calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from backend.datasets.industrial_reviewed_gold import load_industrial_reviewed_gold_dataset
from backend.evaluation.versioned_experiment import (
    comparison_rows,
    compare_metric_rows,
    ensure_artifact_directory,
    read_gzip_jsonl,
    write_csv,
    write_gzip_jsonl,
    write_json,
)
from backend.schemas.egocentric_video import EgocentricVideoExtraction
from scripts.run_change_experiment import (
    DEFAULT_BASELINE_DIR,
    comparison_markdown,
    load_complete_baseline,
    read_csv,
    write_checksums,
)
from scripts.run_versioned_baseline import build_run_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate stored baseline outputs with current metric code."
    )
    parser.add_argument("--change-id", required=True)
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--gold", default=str(
        Path("data/industrial_gold_reviewed/industrial_gold_combined_v2_v3_reviewed.jsonl")
    ))
    parser.add_argument("--output-root", default="experiments/changes")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    baseline = load_complete_baseline(baseline_dir)
    output_dir = ensure_artifact_directory(Path(args.output_root) / args.change_id)
    dataset = load_industrial_reviewed_gold_dataset(args.gold)
    scene_by_id = {scene.scene_id: scene for scene in dataset.scenes}
    stored = read_gzip_jsonl(baseline_dir / "raw_runs.jsonl.gz")
    rescored = []
    for record in stored:
        scene = scene_by_id[str(record["scene_id"])]
        extraction = EgocentricVideoExtraction.model_validate(record["extraction"])
        rescored.append(
            build_run_record(
                scene=scene,
                condition=str(record["condition"]),
                strategy=str(record["strategy"]),
                run_number=int(record["run_number"]),
                model=baseline.environment.model,
                extraction=extraction,
                raw_response=record.get("raw_response", {}),
                execution_error=str(record["metrics"].get("execution_error", "")),
                runtime_seconds=float(record["metrics"].get("runtime_seconds", 0.0)),
            )
        )
    metrics = [record["metrics"] for record in rescored]
    original_metrics = read_csv(baseline_dir / "per_scene_metrics.csv")
    comparison = compare_metric_rows(
        baseline.baseline_id,
        args.change_id,
        original_metrics,
        metrics,
    )
    write_gzip_jsonl(output_dir / "all_runs.jsonl.gz", rescored)
    write_csv(output_dir / "metrics.csv", metrics)
    write_json(output_dir / "comparison_to_baseline.json", comparison)
    write_csv(output_dir / "comparison_to_baseline.csv", comparison_rows(comparison))
    (output_dir / "comparison_to_baseline.md").write_text(
        comparison_markdown(comparison), encoding="utf-8"
    )
    write_json(
        output_dir / "rescore_manifest.json",
        {
            "change_id": args.change_id,
            "baseline_id": baseline.baseline_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "llm_calls": 0,
            "purpose": "Recompute baseline metrics from frozen raw outputs.",
        },
    )
    write_checksums(output_dir)
    (output_dir / "COMPLETED").write_text(
        f"{args.change_id}\n{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    print(comparison_markdown(comparison), flush=True)
    print(f"Rescoring complete without LLM calls: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
