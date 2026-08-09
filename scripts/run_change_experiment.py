#!/usr/bin/env python3
"""Run only a changed variant and compare it with the frozen V1 baseline."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.datasets.indego_adapter import load_indego_standard_dataset
from backend.datasets.industrial_reviewed_gold import load_industrial_reviewed_gold_dataset
from backend.evaluation.versioned_experiment import (
    BaselineManifest,
    ChangeExperimentManifest,
    ModuleVersions,
    comparison_rows,
    compare_metric_rows,
    ensure_artifact_directory,
    sha256_file,
    write_csv,
    write_gzip_jsonl,
    write_json,
)
from scripts.run_versioned_baseline import (
    CONDITION_SPECS,
    DEFAULT_GOLD_PATH,
    DEFAULT_INDEGO_ROOT,
    DEFAULT_MODULE_VERSIONS,
    align_auto_unified_to_gold,
    run_real_llm_matrix,
)


DEFAULT_BASELINE_DIR = Path(
    "experiments/baselines/industrial-v1-llama31-8b-20260809"
)
EXPERIMENT_CODE_PREFIXES = ("backend/", "scripts/", "config/", "tests/")
EXPERIMENT_ROOT_FILES = {"pyproject.toml", "uv.lock", "README.md"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate changed code without rerunning the frozen baseline."
    )
    parser.add_argument("--change-id", required=True)
    parser.add_argument(
        "--module",
        required=True,
        choices=["data_cleaning", "text_preprocessing", "kg_extraction"],
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--method-source", required=True)
    parser.add_argument("--expected-effect", required=True)
    parser.add_argument("--changed-symbol", action="append", default=[])
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--output-root", default="experiments/changes")
    parser.add_argument(
        "--scope",
        choices=["quick", "development", "all"],
        default="quick",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=[
            "raw_one_shot",
            "raw_layered",
            "auto_unified_layered",
            "reviewed_unified_layered",
        ],
        default=None,
    )
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--indego-root", default=str(DEFAULT_INDEGO_ROOT))
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    baseline = load_complete_baseline(baseline_dir)
    output_dir = ensure_artifact_directory(Path(args.output_root) / args.change_id)
    conditions = args.condition or default_conditions_for_module(args.module)
    condition_specs = tuple(parse_condition(value) for value in conditions)
    scene_ids = scenes_for_scope(baseline, args.scope)
    dataset = load_industrial_reviewed_gold_dataset(args.gold)
    scenes = [scene for scene in dataset.scenes if scene.scene_id in set(scene_ids)]
    if len(scenes) != len(scene_ids):
        missing = sorted(set(scene_ids) - {scene.scene_id for scene in scenes})
        raise ValueError(f"Change experiment scenes are missing from Gold: {missing}")
    auto_dataset = load_indego_standard_dataset(args.indego_root)
    auto_by_id = {scene.scene_id: scene for scene in auto_dataset.scenes}
    aligned_auto = {
        scene.scene_id: align_auto_unified_to_gold(scene, auto_by_id[scene.scene_id])
        for scene in scenes
    }
    module_versions = ModuleVersions.model_validate_json(
        DEFAULT_MODULE_VERSIONS.read_text(encoding="utf-8")
    )
    manifest = ChangeExperimentManifest(
        change_id=args.change_id,
        baseline_id=baseline.baseline_id,
        baseline_tag=baseline.baseline_tag,
        module=args.module,
        module_versions=module_versions,
        created_at=datetime.now(timezone.utc).isoformat(),
        code_commit=git_output(["rev-parse", "HEAD"]),
        changed_files=changed_files_since(baseline.code_commit),
        changed_symbols=args.changed_symbol,
        reason=args.reason,
        hypothesis=args.hypothesis,
        method=args.method,
        method_source=args.method_source,
        expected_effect=args.expected_effect,
        conditions=conditions,
        scene_ids=scene_ids,
    )
    write_json(output_dir / "change_manifest.json", manifest)

    records = run_real_llm_matrix(
        artifact_dir=output_dir,
        scenes=scenes,
        auto_unified=aligned_auto,
        model=baseline.environment.model,
        repetitions=baseline.parameters.repetitions,
        timeout=baseline.parameters.timeout_seconds,
        max_tokens=baseline.parameters.max_tokens,
        condition_specs=condition_specs,
    )
    candidate_metrics = [record["metrics"] for record in records]
    write_gzip_jsonl(output_dir / "all_runs.jsonl.gz", records)
    write_csv(output_dir / "metrics.csv", candidate_metrics)

    baseline_metrics = read_csv(baseline_dir / "per_scene_metrics.csv")
    candidate_keys = {
        (
            str(row["scene_id"]),
            str(row["condition"]),
            str(row["strategy"]),
            int(row["run_number"]),
        )
        for row in candidate_metrics
    }
    selected_baseline = [
        row
        for row in baseline_metrics
        if (
            str(row["scene_id"]),
            str(row["condition"]),
            str(row["strategy"]),
            int(row["run_number"]),
        )
        in candidate_keys
    ]
    comparison = compare_metric_rows(
        baseline.baseline_id,
        args.change_id,
        selected_baseline,
        candidate_metrics,
    )
    write_json(output_dir / "comparison_to_baseline.json", comparison)
    write_csv(output_dir / "comparison_to_baseline.csv", comparison_rows(comparison))
    (output_dir / "comparison_to_baseline.md").write_text(
        comparison_markdown(comparison), encoding="utf-8"
    )
    manifest.decision = comparison.decision
    manifest.decision_note = comparison.decision_note
    write_json(output_dir / "change_manifest.json", manifest, overwrite=True)
    write_checksums(output_dir)
    (output_dir / "COMPLETED").write_text(
        f"{args.change_id}\n{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    print(comparison_markdown(comparison), flush=True)
    print(f"Change experiment complete: {output_dir.resolve()}", flush=True)


def load_complete_baseline(path: Path) -> BaselineManifest:
    if not (path / "COMPLETED").exists():
        raise FileNotFoundError(f"Frozen baseline is not complete: {path}")
    manifest = BaselineManifest.model_validate_json(
        (path / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.status != "complete":
        raise ValueError(f"Baseline status must be complete, got {manifest.status}")
    return manifest


def default_conditions_for_module(module: str) -> list[str]:
    if module == "text_preprocessing":
        return ["auto_unified_layered"]
    if module == "data_cleaning":
        return ["auto_unified_layered"]
    return [f"{condition}_{strategy}" for condition, strategy in CONDITION_SPECS]


def parse_condition(value: str) -> tuple[str, str]:
    if value.endswith("_one_shot"):
        return value.removesuffix("_one_shot"), "one_shot"
    if value.endswith("_layered"):
        return value.removesuffix("_layered"), "layered"
    raise ValueError(f"Unsupported condition: {value}")


def scenes_for_scope(manifest: BaselineManifest, scope: str) -> list[str]:
    if scope == "quick":
        return manifest.split.quick_regression_scene_ids
    if scope == "development":
        return manifest.split.development_scene_ids
    return sorted(manifest.split.video_id_by_scene)


def changed_files_since(code_commit: str) -> list[str]:
    committed = set(git_output(["diff", "--name-only", f"{code_commit}..HEAD"]).splitlines())
    working = set(git_output(["diff", "--name-only"]).splitlines())
    untracked = set(git_output(["ls-files", "--others", "--exclude-standard"]).splitlines())
    return sorted(
        item
        for item in committed | working | untracked
        if item
        and (
            item in EXPERIMENT_ROOT_FILES
            or item.startswith(EXPERIMENT_CODE_PREFIXES)
        )
    )


def git_output(arguments: list[str]) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def comparison_markdown(report: Any) -> str:
    lines = [
        f"# Change Experiment {report.change_id}",
        "",
        f"- Frozen baseline: `{report.baseline_id}`",
        f"- Paired runs: `{report.paired_run_count}`",
        f"- Decision: `{report.decision}`",
        f"- Reason: {report.decision_note}",
        f"- Baseline errors: `{report.baseline_error_count}`",
        f"- Candidate errors: `{report.candidate_error_count}`",
        "",
        "| Metric | Baseline | Candidate | Absolute delta | Relative delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report.metrics:
        relative = "n/a" if item.relative_delta is None else f"{item.relative_delta:.6f}"
        lines.append(
            f"| {item.metric} | {item.baseline_mean:.6f} | "
            f"{item.candidate_mean:.6f} | {item.absolute_delta:.6f} | {relative} |"
        )
    if report.missing_baseline_keys or report.missing_candidate_keys:
        lines.extend(
            [
                "",
                f"- Missing baseline keys: `{len(report.missing_baseline_keys)}`",
                f"- Missing candidate keys: `{len(report.missing_candidate_keys)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_checksums(directory: Path) -> None:
    lines = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name in {"SHA256SUMS", "COMPLETED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
