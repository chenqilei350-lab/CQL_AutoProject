"""Versioned baseline and change-experiment contracts.

The module keeps experiment identity, model/data fingerprints, immutable
artifacts, and paired baseline comparisons separate from the extraction
implementation. Existing product entry points do not depend on this module.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import multiprocessing
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Literal, Sequence

from pydantic import BaseModel, Field, model_validator


ModuleName = Literal["data_cleaning", "text_preprocessing", "kg_extraction"]
BaselineStatus = Literal["draft", "running", "complete"]
ChangeDecision = Literal["improved", "neutral", "regressed", "incomplete"]


class SubprocessTimeoutError(TimeoutError):
    """Raised when an isolated experiment call exceeds its wall-clock limit."""


class ModuleVersions(BaseModel):
    """Independent semantic versions for the three project modules."""

    data_cleaning: str
    text_preprocessing: str
    kg_extraction: str


class FileFingerprint(BaseModel):
    """Stable identity for one code, data, prompt, or configuration file."""

    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class RuntimeEnvironment(BaseModel):
    """Environment facts required to interpret a local-model experiment."""

    model: str
    model_digest: str
    machine: str
    chip: str
    memory_gb: int
    operating_system: str
    python_version: str
    uv_version: str


class BaselineParameters(BaseModel):
    """Frozen real-LLM settings for the industrial V1 baseline."""

    temperature: float = 0.0
    repetitions: int = Field(default=2, ge=1)
    timeout_seconds: float = Field(default=180.0, gt=0)
    max_tokens: int = Field(default=3000, ge=1)
    conditions: tuple[str, ...] = (
        "raw_one_shot",
        "raw_layered",
        "auto_unified_layered",
        "reviewed_unified_layered",
    )


class DatasetSplitManifest(BaseModel):
    """Video-grouped dataset partitions used by future comparisons."""

    development_scene_ids: list[str]
    quick_regression_scene_ids: list[str]
    holdout_scene_ids: list[str]
    video_id_by_scene: dict[str, str]

    @model_validator(mode="after")
    def validate_partitions(self) -> "DatasetSplitManifest":
        development = set(self.development_scene_ids)
        holdout = set(self.holdout_scene_ids)
        quick = set(self.quick_regression_scene_ids)
        if development & holdout:
            raise ValueError("Development and holdout scenes must be disjoint.")
        if not quick <= development:
            raise ValueError("Quick regression scenes must be part of development.")
        development_videos = {
            self.video_id_by_scene[scene_id] for scene_id in development
        }
        holdout_videos = {
            self.video_id_by_scene[scene_id] for scene_id in holdout
        }
        if development_videos & holdout_videos:
            raise ValueError("Dataset partitions must be grouped by video_id.")
        return self


class BaselineManifest(BaseModel):
    """Complete identity of a frozen model, pipeline, data, and result set."""

    baseline_id: str
    baseline_tag: str
    code_commit: str
    created_at: str
    status: BaselineStatus
    module_versions: ModuleVersions
    environment: RuntimeEnvironment
    parameters: BaselineParameters
    dataset_name: str
    scene_count: int
    video_count: int
    gold_entity_count: int
    gold_relation_count: int
    split: DatasetSplitManifest
    code_and_config_files: list[FileFingerprint] = Field(default_factory=list)
    gold_files: list[FileFingerprint] = Field(default_factory=list)
    source_files: list[FileFingerprint] = Field(default_factory=list)
    result_files: list[FileFingerprint] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ChangeExperimentManifest(BaseModel):
    """Audit record for one optimization evaluated against a frozen baseline."""

    change_id: str
    baseline_id: str
    baseline_tag: str
    module: ModuleName
    module_versions: ModuleVersions
    created_at: str
    code_commit: str
    changed_files: list[str]
    changed_symbols: list[str] = Field(default_factory=list)
    reason: str
    hypothesis: str
    method: str
    method_source: str
    expected_effect: str
    conditions: list[str]
    scene_ids: list[str]
    decision: ChangeDecision = "incomplete"
    decision_note: str = ""


class MetricDelta(BaseModel):
    """One paired metric comparison between baseline and changed code."""

    metric: str
    baseline_mean: float
    candidate_mean: float
    absolute_delta: float
    relative_delta: float | None = None


class ComparisonReport(BaseModel):
    """Paired comparison that never reruns or rewrites the baseline."""

    baseline_id: str
    change_id: str
    paired_run_count: int
    missing_baseline_keys: list[str] = Field(default_factory=list)
    missing_candidate_keys: list[str] = Field(default_factory=list)
    metrics: list[MetricDelta] = Field(default_factory=list)
    baseline_error_count: int = 0
    candidate_error_count: int = 0
    decision: ChangeDecision = "incomplete"
    decision_note: str = ""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path, *, root: str | Path | None = None) -> FileFingerprint:
    """Create a portable fingerprint for one file."""

    file_path = Path(path)
    display_path = file_path
    if root is not None:
        try:
            display_path = file_path.relative_to(Path(root))
        except ValueError:
            display_path = file_path
    return FileFingerprint(
        path=display_path.as_posix(),
        sha256=sha256_file(file_path),
        size_bytes=file_path.stat().st_size,
    )


def fingerprint_tree(
    root: str | Path,
    *,
    exclude_names: frozenset[str] = frozenset({".DS_Store"}),
) -> list[FileFingerprint]:
    """Fingerprint every regular file under a directory in stable order."""

    root_path = Path(root)
    return [
        fingerprint_file(path, root=root_path)
        for path in sorted(root_path.rglob("*"))
        if path.is_file() and path.name not in exclude_names
    ]


def verify_fingerprints(
    fingerprints: Iterable[FileFingerprint],
    *,
    root: str | Path,
) -> list[str]:
    """Return mismatches instead of silently comparing different artifacts."""

    root_path = Path(root)
    issues: list[str] = []
    for item in fingerprints:
        path = root_path / item.path
        if not path.exists():
            issues.append(f"missing:{item.path}")
            continue
        actual = sha256_file(path)
        if actual != item.sha256:
            issues.append(f"sha256_mismatch:{item.path}")
    return issues


def ensure_artifact_directory(path: str | Path) -> Path:
    """Create a new artifact directory, or resume only an incomplete one."""

    directory = Path(path)
    if (directory / "COMPLETED").exists():
        raise FileExistsError(f"Completed experiment is immutable: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_process_with_timeout(
    worker: Callable[..., None],
    *,
    args: tuple[Any, ...] = (),
    timeout_seconds: float,
) -> Any:
    """Run a pipe-writing worker in an isolated spawn process.

    The worker receives the child connection as its first argument and must
    send one serializable payload. A process boundary makes the timeout a real
    wall-clock limit even when an HTTP client or local model call is stuck.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(child_connection, *args))
    process.start()
    child_connection.close()
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if parent_connection.poll(min(0.1, remaining)):
                payload = parent_connection.recv()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                return payload
            if not process.is_alive():
                if parent_connection.poll():
                    return parent_connection.recv()
                raise RuntimeError(
                    "Isolated experiment worker exited without returning a result "
                    f"(exit code {process.exitcode})."
                )

        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=2.0)
        raise SubprocessTimeoutError(
            f"Isolated experiment call exceeded {timeout_seconds:.1f} seconds."
        )
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        raise
    finally:
        parent_connection.close()


def write_json(path: str | Path, value: BaseModel | dict[str, Any], *, overwrite: bool = False) -> Path:
    """Write JSON while protecting completed experiment artifacts."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite experiment artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], *, overwrite: bool = False) -> Path:
    """Write a stable CSV without silently replacing an existing result."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite experiment artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_gzip_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Store inspectable full runs compactly."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite experiment artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def read_gzip_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load compressed full-run records for rescoring or comparison."""

    with gzip.open(Path(path), "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """Return the exact key required for paired comparison."""

    return (
        str(row["scene_id"]),
        str(row["condition"]),
        str(row["strategy"]),
        int(row["run_number"]),
    )


def format_run_key(key: tuple[str, str, str, int]) -> str:
    return "|".join([key[0], key[1], key[2], str(key[3])])


def compare_metric_rows(
    baseline_id: str,
    change_id: str,
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    metric_names: tuple[str, ...] = (
        "raw_node_f1",
        "raw_edge_f1",
        "final_node_f1",
        "final_edge_f1",
        "relation_hallucination_rate",
        "ontology_conformance",
        "runtime_seconds",
    ),
) -> ComparisonReport:
    """Compare only matching run keys and retain missing-key diagnostics."""

    baseline = {run_key(row): row for row in baseline_rows}
    candidate = {run_key(row): row for row in candidate_rows}
    shared = sorted(baseline.keys() & candidate.keys())
    missing_baseline = sorted(candidate.keys() - baseline.keys())
    missing_candidate = sorted(baseline.keys() - candidate.keys())
    deltas: list[MetricDelta] = []
    for metric in metric_names:
        baseline_values = [float(baseline[key].get(metric, 0.0) or 0.0) for key in shared]
        candidate_values = [float(candidate[key].get(metric, 0.0) or 0.0) for key in shared]
        baseline_mean = mean(baseline_values) if baseline_values else 0.0
        candidate_mean = mean(candidate_values) if candidate_values else 0.0
        absolute = candidate_mean - baseline_mean
        relative = absolute / baseline_mean if baseline_mean else None
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline_mean=round(baseline_mean, 6),
                candidate_mean=round(candidate_mean, 6),
                absolute_delta=round(absolute, 6),
                relative_delta=round(relative, 6) if relative is not None else None,
            )
        )

    baseline_errors = sum(bool(baseline[key].get("execution_error")) for key in shared)
    candidate_errors = sum(bool(candidate[key].get("execution_error")) for key in shared)
    report = ComparisonReport(
        baseline_id=baseline_id,
        change_id=change_id,
        paired_run_count=len(shared),
        missing_baseline_keys=[format_run_key(key) for key in missing_baseline],
        missing_candidate_keys=[format_run_key(key) for key in missing_candidate],
        metrics=deltas,
        baseline_error_count=baseline_errors,
        candidate_error_count=candidate_errors,
    )
    report.decision, report.decision_note = classify_comparison(report)
    return report


def classify_comparison(report: ComparisonReport) -> tuple[ChangeDecision, str]:
    """Apply transparent primary-metric and guardrail rules."""

    if report.missing_baseline_keys or report.missing_candidate_keys or not report.paired_run_count:
        return "incomplete", "Run keys are incomplete; no keep/revert conclusion is allowed."
    by_name = {item.metric: item for item in report.metrics}
    edge_delta = by_name.get("final_edge_f1", MetricDelta(metric="final_edge_f1", baseline_mean=0, candidate_mean=0, absolute_delta=0)).absolute_delta
    node_delta = by_name.get("final_node_f1", MetricDelta(metric="final_node_f1", baseline_mean=0, candidate_mean=0, absolute_delta=0)).absolute_delta
    hallucination_delta = by_name.get(
        "relation_hallucination_rate",
        MetricDelta(metric="relation_hallucination_rate", baseline_mean=0, candidate_mean=0, absolute_delta=0),
    ).absolute_delta
    if report.candidate_error_count > report.baseline_error_count:
        return "regressed", "Error count increased relative to the frozen baseline."
    if node_delta < -0.02 or hallucination_delta > 0.02:
        return "regressed", "A correctness or hallucination guardrail regressed by more than 0.02."
    if edge_delta >= 0.01:
        return "improved", "Final Edge F1 improved by at least 0.01 without guardrail regression."
    if edge_delta <= -0.01:
        return "regressed", "Final Edge F1 decreased by at least 0.01."
    return "neutral", "The primary metric changed by less than 0.01 and guardrails held."


def comparison_rows(report: ComparisonReport) -> list[dict[str, Any]]:
    """Flatten a comparison report for CSV output."""

    return [
        {
            "baseline_id": report.baseline_id,
            "change_id": report.change_id,
            "paired_run_count": report.paired_run_count,
            "metric": item.metric,
            "baseline_mean": item.baseline_mean,
            "candidate_mean": item.candidate_mean,
            "absolute_delta": item.absolute_delta,
            "relative_delta": item.relative_delta,
            "decision": report.decision,
            "decision_note": report.decision_note,
        }
        for item in report.metrics
    ]
