#!/usr/bin/env python3
"""Freeze and run the one-time industrial V1 model/pipeline baseline.

The runner is intentionally resumable. Each completed LLM call is appended to
an inspectable checkpoint and skipped on restart. Once COMPLETED exists, the
artifact directory is immutable and the runner only permits verification.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from backend.cleaning.factory_data import FactoryCleaningReport, clean_factory_data_source
from backend.datasets.benchmark import BenchmarkScene
from backend.datasets.indego_adapter import IndEgoStandardScene, load_indego_standard_dataset
from backend.datasets.industrial_reviewed_gold import (
    SCORING_NODE_LABELS,
    SCORING_RELATION_TYPES,
    load_industrial_reviewed_gold_dataset,
)
from backend.evaluation.ontology_validation import validate_egocentric_extraction
from backend.evaluation.versioned_experiment import (
    BaselineManifest,
    BaselineParameters,
    DatasetSplitManifest,
    ModuleVersions,
    RuntimeEnvironment,
    ensure_artifact_directory,
    fingerprint_file,
    fingerprint_tree,
    sha256_file,
    run_process_with_timeout,
    write_csv,
    write_gzip_jsonl,
    write_json,
)
from backend.graph.property_graph import PropertyGraph, build_property_graph
from backend.pipeline.industrial_text_to_kg import summarize_relation_origins
from backend.pipeline.relation_candidate_pipeline import (
    DeterministicRelationCandidateScorer,
    EntityMention,
    ProceduralRelationCandidateGenerator,
)
from backend.preprocessing.unified_text import GroundedEntry, UnifiedTextRecord
from backend.schemas.egocentric_video import EgocentricVideoExtraction
from scripts.run_easg_real_llm_lightweight_experiments import (
    LLMJsonParseError,
    extract_scene,
    graph_sets,
    jaccard,
    prf1_from_sets,
)
from scripts.run_industrial_reviewed_gold_diagnostics import gold_edge_set


BASELINE_ID = "industrial-v1-llama31-8b-20260809"
BASELINE_TAG = "baseline-industrial-v1-2026-08-09"
DEFAULT_ARTIFACT_DIR = Path("experiments/baselines") / BASELINE_ID
DEFAULT_GOLD_PATH = Path(
    "data/industrial_gold_reviewed/industrial_gold_combined_v2_v3_reviewed.jsonl"
)
DEFAULT_INDEGO_ROOT = Path("raw/indego_text_layers")
DEFAULT_MODULE_VERSIONS = Path("config/module_versions.json")
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_MODEL_DIGEST = "46e0c10c039e"
CONDITION_SPECS = (
    ("raw", "one_shot"),
    ("raw", "layered"),
    ("auto_unified", "layered"),
    ("reviewed_unified", "layered"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the immutable industrial V1 baseline.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--indego-root", default=str(DEFAULT_INDEGO_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-digest", default=DEFAULT_MODEL_DIGEST)
    parser.add_argument("--code-commit")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-only scene limit.")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    if args.verify_only:
        verify_completed_baseline(artifact_dir)
        return

    if not args.code_commit:
        parser.error("--code-commit is required unless --verify-only is used")

    ensure_artifact_directory(artifact_dir)
    run_baseline(
        artifact_dir=artifact_dir,
        gold_path=Path(args.gold),
        indego_root=Path(args.indego_root),
        model=args.model,
        model_digest=args.model_digest,
        code_commit=args.code_commit,
        repetitions=args.repetitions,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        limit=args.limit,
    )


def run_baseline(
    *,
    artifact_dir: Path,
    gold_path: Path,
    indego_root: Path,
    model: str,
    model_digest: str,
    code_commit: str,
    repetitions: int,
    timeout: float,
    max_tokens: int,
    limit: int | None = None,
) -> BaselineManifest:
    """Run deterministic modules, relation diagnostics, and real LLM calls."""

    if model != DEFAULT_MODEL or model_digest != DEFAULT_MODEL_DIGEST:
        raise ValueError(
            f"V1 baseline requires {DEFAULT_MODEL}@{DEFAULT_MODEL_DIGEST}; "
            f"received {model}@{model_digest}."
        )
    module_versions = ModuleVersions.model_validate_json(
        DEFAULT_MODULE_VERSIONS.read_text(encoding="utf-8")
    )
    gold_records = load_gold_records(gold_path)
    dataset = load_industrial_reviewed_gold_dataset(gold_path)
    if limit is not None:
        dataset = dataset.model_copy(update={"scenes": dataset.scenes[:limit]})
        selected = {scene.scene_id for scene in dataset.scenes}
        gold_records = [row for row in gold_records if row["scene_id"] in selected]
    if limit is None:
        validate_canonical_gold(gold_records)

    print("[1/5] Data cleaning baseline", flush=True)
    cleaning_report = clean_factory_data_source(indego_root)
    cleaning_metrics = cleaning_metric_payload(cleaning_report, indego_root)
    write_json(
        artifact_dir / "cleaning_metrics.json",
        cleaning_metrics,
        overwrite=(artifact_dir / "cleaning_metrics.json").exists(),
    )
    write_cleaning_archive(
        artifact_dir / "cleaned_records.jsonl.gz",
        cleaning_report,
        overwrite=(artifact_dir / "cleaned_records.jsonl.gz").exists(),
    )

    print("[2/5] Automatic and reviewed preprocessing baseline", flush=True)
    auto_dataset = load_indego_standard_dataset(indego_root)
    auto_by_id = {scene.scene_id: scene for scene in auto_dataset.scenes}
    aligned_auto = {
        scene.scene_id: align_auto_unified_to_gold(scene, auto_by_id[scene.scene_id])
        for scene in dataset.scenes
    }
    preprocessing_rows = build_preprocessing_rows(dataset.scenes, auto_by_id, aligned_auto)
    write_csv(
        artifact_dir / "preprocessing_metrics.csv",
        preprocessing_rows,
        overwrite=(artifact_dir / "preprocessing_metrics.csv").exists(),
    )

    split = build_split_manifest(gold_records)
    write_json(
        artifact_dir / "split_manifest.json",
        split,
        overwrite=(artifact_dir / "split_manifest.json").exists(),
    )

    print("[3/5] Gold-entity relation diagnostic", flush=True)
    diagnostic = run_relation_diagnostic(dataset.scenes)
    write_csv(
        artifact_dir / "relation_diagnostic.csv",
        diagnostic["relation_rows"],
        overwrite=(artifact_dir / "relation_diagnostic.csv").exists(),
    )
    write_json(
        artifact_dir / "relation_diagnostic_summary.json",
        diagnostic["summary"],
        overwrite=(artifact_dir / "relation_diagnostic_summary.json").exists(),
    )

    parameters = BaselineParameters(
        repetitions=repetitions,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
    )
    environment = capture_runtime_environment(model, model_digest)
    manifest = build_manifest(
        artifact_dir=artifact_dir,
        gold_path=gold_path,
        indego_root=indego_root,
        code_commit=code_commit,
        module_versions=module_versions,
        environment=environment,
        parameters=parameters,
        split=split,
        gold_records=gold_records,
        status="running",
    )
    write_json(
        artifact_dir / "baseline_manifest.json",
        manifest,
        overwrite=(artifact_dir / "baseline_manifest.json").exists(),
    )

    print("[4/5] Real LLM baseline with checkpoint/resume", flush=True)
    run_records = run_real_llm_matrix(
        artifact_dir=artifact_dir,
        scenes=dataset.scenes,
        auto_unified=aligned_auto,
        model=model,
        repetitions=repetitions,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    materialize_run_artifacts(artifact_dir, run_records)

    print("[5/5] Final checksums and immutable marker", flush=True)
    write_baseline_readme(artifact_dir, manifest, len(run_records), limit=limit)
    manifest.status = "complete"
    manifest.result_files = result_fingerprints(artifact_dir)
    write_json(artifact_dir / "baseline_manifest.json", manifest, overwrite=True)
    write_checksums(artifact_dir)
    (artifact_dir / "COMPLETED").write_text(
        f"{BASELINE_ID}\n{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    verify_completed_baseline(artifact_dir)
    print(f"Baseline complete: {artifact_dir.resolve()}", flush=True)
    return manifest


def load_gold_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_canonical_gold(records: list[dict[str, Any]]) -> None:
    entities = sum(len(row.get("gold_entities", [])) for row in records)
    relations = sum(len(row.get("gold_relations", [])) for row in records)
    videos = {str(row["video_id"]) for row in records}
    if (len(records), len(videos), entities, relations) != (32, 30, 674, 828):
        raise ValueError(
            "Canonical Gold mismatch: expected 32 scenes, 30 videos, "
            f"674 entities, 828 relations; got {len(records)}, {len(videos)}, "
            f"{entities}, {relations}."
        )


def cleaning_metric_payload(report: FactoryCleaningReport, root: Path) -> dict[str, Any]:
    all_issues = [*report.issues]
    for record in report.records:
        all_issues.extend(record.issues)
    severity = Counter(issue.severity for issue in all_issues)
    source_files = fingerprint_tree(root)
    return {
        "source_name": report.source_name,
        "root_path": str(root),
        "files_seen": report.files_seen,
        "source_file_count": len(source_files),
        "source_total_bytes": sum(item.size_bytes for item in source_files),
        "record_count": len(report.records),
        "records_by_type": report.records_by_type(),
        "issues_by_code": report.issues_by_code(),
        "issues_by_severity": dict(sorted(severity.items())),
        "empty_normalized_text_count": sum(not row.normalized_text for row in report.records),
        "records_with_provenance": sum(bool(row.source_path) for row in report.records),
        "records_with_timestamp": sum(row.timestamp is not None for row in report.records),
    }


def write_cleaning_archive(path: Path, report: FactoryCleaningReport, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    with gzip.open(path, "wt", encoding="utf-8") as file:
        for record in report.records:
            file.write(record.model_dump_json(exclude_none=True) + "\n")


def align_auto_unified_to_gold(
    gold_scene: BenchmarkScene,
    automatic_scene: IndEgoStandardScene,
) -> UnifiedTextRecord:
    """Restrict automatic fields to evidence inside the reviewed source excerpt."""

    source = normalize_text(gold_scene.raw_text)

    def supported(entries: Iterable[GroundedEntry]) -> list[GroundedEntry]:
        return [entry for entry in entries if normalize_text(entry.evidence) in source]

    record = automatic_scene.unified_record
    return UnifiedTextRecord(
        scene_id=gold_scene.scene_id,
        segment_id=record.segment_id,
        timestamp=record.timestamp,
        scene_segment=record.scene_segment,
        actors=supported(record.actors),
        action_sequence=supported(record.action_sequence),
        tools_objects=supported(record.tools_objects),
        process_parameters=supported(record.process_parameters),
        quality_results=supported(record.quality_results),
        outcomes_parameters=supported(record.outcomes_parameters),
        evidence_uncertainty=[
            "Automatically generated from IndEgo text layers and restricted to "
            "evidence in the reviewed evaluation excerpt."
        ],
        source_text=gold_scene.raw_text,
    )


def build_preprocessing_rows(
    scenes: list[BenchmarkScene],
    automatic: dict[str, IndEgoStandardScene],
    aligned: dict[str, UnifiedTextRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        automatic_record = automatic[scene.scene_id].unified_record
        aligned_record = aligned[scene.scene_id]
        reviewed_record = scene.unified_record
        auto_entries = all_grounded_entries(automatic_record)
        aligned_entries = all_grounded_entries(aligned_record)
        reviewed_entries = all_grounded_entries(reviewed_record)
        source = normalize_text(scene.raw_text)
        rows.append(
            {
                "scene_id": scene.scene_id,
                "raw_characters": len(scene.raw_text),
                "auto_unified_characters": len(aligned_record.to_prompt_text()),
                "reviewed_unified_characters": len(reviewed_record.to_prompt_text()),
                "auto_original_entry_count": len(auto_entries),
                "auto_supported_entry_count": len(aligned_entries),
                "auto_outside_excerpt_count": sum(
                    normalize_text(entry.evidence) not in source for entry in auto_entries
                ),
                "auto_evidence_support_rate": evidence_support_rate(aligned_entries, source),
                "reviewed_entry_count": len(reviewed_entries),
                "reviewed_evidence_support_rate": evidence_support_rate(reviewed_entries, source),
                "auto_action_count": len(aligned_record.action_sequence),
                "auto_tool_object_count": len(aligned_record.tools_objects),
                "reviewed_action_count": len(reviewed_record.action_sequence),
                "reviewed_tool_object_count": len(reviewed_record.tools_objects),
                "raw_source_exactly_matches_auto_source": (
                    scene.raw_text == automatic_record.source_text
                ),
            }
        )
    return rows


def all_grounded_entries(record: UnifiedTextRecord) -> list[GroundedEntry]:
    return [
        *record.actors,
        *record.action_sequence,
        *record.tools_objects,
        *record.process_parameters,
        *record.quality_results,
        *record.outcomes_parameters,
    ]


def evidence_support_rate(entries: list[GroundedEntry], source: str) -> float:
    if not entries:
        return 1.0
    supported = sum(normalize_text(entry.evidence) in source for entry in entries)
    return round(supported / len(entries), 6)


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def build_split_manifest(records: list[dict[str, Any]]) -> DatasetSplitManifest:
    """Create deterministic category-aware, video-grouped partitions."""

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_video[str(row["video_id"])].append(row)
    categories = sorted({str(row.get("category") or "unknown") for row in records})
    holdout_videos: set[str] = set()
    for category in categories:
        candidates = sorted(
            {
                video_id
                for video_id, rows in by_video.items()
                if any(str(row.get("category") or "unknown") == category for row in rows)
            },
            key=lambda item: stable_rank(f"holdout|{category}|{item}"),
        )
        for candidate in candidates:
            if candidate not in holdout_videos:
                holdout_videos.add(candidate)
                break
    all_videos = set(by_video)
    development_videos = all_videos - holdout_videos
    development_records = [
        row for row in records if str(row["video_id"]) in development_videos
    ]
    quick = select_quick_regression_scenes(development_records, target_count=8)
    video_by_scene = {str(row["scene_id"]): str(row["video_id"]) for row in records}
    return DatasetSplitManifest(
        development_scene_ids=sorted(
            scene_id
            for scene_id, video_id in video_by_scene.items()
            if video_id in development_videos
        ),
        quick_regression_scene_ids=quick,
        holdout_scene_ids=sorted(
            scene_id
            for scene_id, video_id in video_by_scene.items()
            if video_id in holdout_videos
        ),
        video_id_by_scene=video_by_scene,
    )


def select_quick_regression_scenes(
    records: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[str]:
    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    categories = sorted({str(row.get("category") or "unknown") for row in records})

    def score(row: dict[str, Any]) -> tuple[int, int, str]:
        relations = row.get("gold_relations", [])
        uses_tool = sum(item.get("relation") == "USES_TOOL" for item in relations)
        return (-uses_tool, -len(relations), stable_rank(str(row["scene_id"])))

    for category in categories:
        options = sorted(
            [row for row in records if str(row.get("category") or "unknown") == category],
            key=score,
        )
        for row in options:
            if str(row["video_id"]) not in used_videos:
                selected.append(row)
                used_videos.add(str(row["video_id"]))
                break
    for row in sorted(records, key=score):
        if len(selected) >= target_count:
            break
        if str(row["video_id"]) in used_videos:
            continue
        selected.append(row)
        used_videos.add(str(row["video_id"]))
    return [str(row["scene_id"]) for row in selected[:target_count]]


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{BASELINE_ID}|{value}".encode("utf-8")).hexdigest()


def run_relation_diagnostic(scenes: list[BenchmarkScene]) -> dict[str, Any]:
    generator = ProceduralRelationCandidateGenerator()
    scorer = DeterministicRelationCandidateScorer()
    counters: dict[str, Counter[str]] = {
        key: Counter()
        for key in ["gold", "candidate", "matched", "accepted", "accepted_matched"]
    }
    for scene in scenes:
        extraction = scene.gold_extraction
        mentions = gold_mentions(extraction)
        candidates = generator.generate(
            actions=mentions["actions"],
            tools=mentions["tools"],
            objects=mentions["objects"],
            source_text=scene.raw_text,
        )
        report = scorer.score(candidates, source_text=scene.raw_text)
        gold = gold_edge_set(extraction)
        candidate = {
            (item.subject_id, item.relation_type, item.object_id)
            for item in candidates
            if item.relation_type in SCORING_RELATION_TYPES
        }
        accepted = {
            (item.subject_id, item.relation_type, item.object_id)
            for item in report.accepted_candidates()
            if item.relation_type in SCORING_RELATION_TYPES
        }
        for relation_type in SCORING_RELATION_TYPES:
            gold_typed = {item for item in gold if item[1] == relation_type}
            candidate_typed = {item for item in candidate if item[1] == relation_type}
            accepted_typed = {item for item in accepted if item[1] == relation_type}
            counters["gold"][relation_type] += len(gold_typed)
            counters["candidate"][relation_type] += len(candidate_typed)
            counters["matched"][relation_type] += len(gold_typed & candidate_typed)
            counters["accepted"][relation_type] += len(accepted_typed)
            counters["accepted_matched"][relation_type] += len(gold_typed & accepted_typed)
    rows = []
    for relation_type in sorted(SCORING_RELATION_TYPES):
        gold_count = counters["gold"][relation_type]
        accepted_count = counters["accepted"][relation_type]
        matched = counters["accepted_matched"][relation_type]
        rows.append(
            {
                "relation_type": relation_type,
                "gold_edges": gold_count,
                "candidate_edges": counters["candidate"][relation_type],
                "matched_gold_edges": counters["matched"][relation_type],
                "candidate_coverage_recall": safe_divide(counters["matched"][relation_type], gold_count),
                "accepted_precision": safe_divide(matched, accepted_count),
                "accepted_recall": safe_divide(matched, gold_count),
                "accepted_f1": f1_from_counts(matched, accepted_count - matched, gold_count - matched),
            }
        )
    gold_total = sum(counters["gold"].values())
    accepted_total = sum(counters["accepted"].values())
    matched_total = sum(counters["accepted_matched"].values())
    summary = {
        "diagnostic_mode": "gold_entity_oracle_relation_candidate",
        "scene_count": len(scenes),
        "gold_edge_count": gold_total,
        "deterministic_accepted_edge_count": accepted_total,
        "deterministic_precision": safe_divide(matched_total, accepted_total),
        "deterministic_recall": safe_divide(matched_total, gold_total),
        "deterministic_f1": f1_from_counts(
            matched_total,
            accepted_total - matched_total,
            gold_total - matched_total,
        ),
    }
    return {"summary": summary, "relation_rows": rows}


def gold_mentions(extraction: EgocentricVideoExtraction) -> dict[str, list[EntityMention]]:
    return {
        "actions": [
            EntityMention(
                entity_id=item.entity_id or f"action_{index}",
                label="Action",
                name=item.name,
                evidence_text=item.evidence_text or "",
                sequence_index=item.sequence_index,
            )
            for index, item in enumerate(extraction.actions, 1)
        ],
        "tools": [
            EntityMention(
                entity_id=item.entity_id or f"tool_{index}",
                label="Tool",
                name=item.name,
                evidence_text=item.evidence_text or "",
            )
            for index, item in enumerate(extraction.tools, 1)
        ],
        "objects": [
            EntityMention(
                entity_id=item.entity_id or f"object_{index}",
                label="Object",
                name=item.name,
                evidence_text=item.evidence_text or "",
            )
            for index, item in enumerate(extraction.objects, 1)
        ],
    }


def run_real_llm_matrix(
    *,
    artifact_dir: Path,
    scenes: list[BenchmarkScene],
    auto_unified: dict[str, UnifiedTextRecord],
    model: str,
    repetitions: int,
    timeout: float,
    max_tokens: int,
    condition_specs: tuple[tuple[str, str], ...] = CONDITION_SPECS,
) -> list[dict[str, Any]]:
    work_dir = artifact_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = work_dir / "checkpoint.jsonl"
    records = load_checkpoint(checkpoint)
    completed = {record_key(item) for item in records}
    total = len(scenes) * len(condition_specs) * repetitions
    done = len(completed)
    if done:
        print(f"Resuming from {done}/{total} completed calls", flush=True)

    for condition, strategy in condition_specs:
        for scene in scenes:
            eval_scene = scene_for_condition(scene, condition, auto_unified)
            model_condition = "raw" if condition == "raw" else "unified"
            for run_number in range(1, repetitions + 1):
                key = (scene.scene_id, condition, strategy, run_number)
                if key in completed:
                    continue
                started = time.perf_counter()
                error = ""
                raw_response: dict[str, Any] = {}
                try:
                    extraction, raw_response = extract_scene_with_hard_timeout(
                        scene=eval_scene,
                        condition=model_condition,
                        strategy=strategy,
                        model=model,
                        timeout=timeout,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    if isinstance(exc, LLMJsonParseError):
                        raw_response = {"parse_error_raw_content": exc.raw_content}
                    extraction = EgocentricVideoExtraction(source_text=eval_scene.raw_text)
                runtime = round(time.perf_counter() - started, 4)
                record = build_run_record(
                    scene=scene,
                    condition=condition,
                    strategy=strategy,
                    run_number=run_number,
                    model=model,
                    extraction=extraction,
                    raw_response=raw_response,
                    execution_error=error,
                    runtime_seconds=runtime,
                )
                append_checkpoint(checkpoint, record)
                records.append(record)
                completed.add(key)
                done += 1
                metrics = record["metrics"]
                print(
                    f"[{done}/{total}] {condition}/{strategy} | {scene.scene_id} | "
                    f"run={run_number} node={metrics['final_node_f1']:.4f} "
                    f"edge={metrics['final_edge_f1']:.4f} time={runtime:.1f}s "
                    f"error={error or '-'}",
                    flush=True,
                )
    return records


def extract_scene_with_hard_timeout(
    *,
    scene: BenchmarkScene,
    condition: str,
    strategy: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> tuple[EgocentricVideoExtraction, dict[str, Any]]:
    """Extract one scene with a process-level wall-clock timeout."""

    payload = run_process_with_timeout(
        _llm_extraction_worker,
        args=(
            scene.model_dump(mode="json"),
            condition,
            strategy,
            model,
            timeout,
            max_tokens,
        ),
        timeout_seconds=timeout,
    )
    if not payload.get("ok"):
        error_type = payload.get("error_type", "RuntimeError")
        error_message = payload.get("error_message", "Unknown extraction failure")
        if error_type == "LLMJsonParseError":
            raise LLMJsonParseError(
                error_message,
                raw_content=str(payload.get("raw_content", "")),
            )
        raise RuntimeError(f"{error_type}: {error_message}")
    return (
        EgocentricVideoExtraction.model_validate(payload["extraction"]),
        dict(payload.get("raw_response", {})),
    )


def _llm_extraction_worker(
    connection: Any,
    scene_payload: dict[str, Any],
    condition: str,
    strategy: str,
    model: str,
    timeout: float,
    max_tokens: int,
) -> None:
    """Execute one model call and return only serializable data over a pipe."""

    try:
        from backend.llm.client import get_openai_client

        scene = BenchmarkScene.model_validate(scene_payload)
        client = get_openai_client(timeout=timeout)
        extraction, raw_response = extract_scene(
            client=client,
            scene=scene,
            condition=condition,
            strategy=strategy,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        connection.send(
            {
                "ok": True,
                "extraction": extraction.model_dump(mode="json"),
                "raw_response": raw_response,
            }
        )
    except Exception as exc:
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc).splitlines()[0],
                "raw_content": (
                    exc.raw_content if isinstance(exc, LLMJsonParseError) else ""
                ),
            }
        )
    finally:
        connection.close()


def scene_for_condition(
    scene: BenchmarkScene,
    condition: str,
    auto_unified: dict[str, UnifiedTextRecord],
) -> BenchmarkScene:
    if condition != "auto_unified":
        return scene
    return scene.model_copy(update={"unified_record": auto_unified[scene.scene_id]})


def build_run_record(
    *,
    scene: BenchmarkScene,
    condition: str,
    strategy: str,
    run_number: int,
    model: str,
    extraction: EgocentricVideoExtraction,
    raw_response: dict[str, Any],
    execution_error: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    gold_graph = build_property_graph(scene.gold_extraction)
    final_graph = build_property_graph(extraction)
    raw_graph = llm_only_graph(final_graph)
    gold_nodes, gold_edges = graph_sets(
        gold_graph,
        node_labels=SCORING_NODE_LABELS,
        edge_types=SCORING_RELATION_TYPES,
    )
    raw_nodes, raw_edges = graph_sets(
        raw_graph,
        node_labels=SCORING_NODE_LABELS,
        edge_types=SCORING_RELATION_TYPES,
    )
    final_nodes, final_edges = graph_sets(
        final_graph,
        node_labels=SCORING_NODE_LABELS,
        edge_types=SCORING_RELATION_TYPES,
    )
    raw_node_metric = prf1_from_sets("raw_nodes", gold_nodes, raw_nodes)
    raw_edge_metric = prf1_from_sets("raw_edges", gold_edges, raw_edges)
    final_node_metric = prf1_from_sets("final_nodes", gold_nodes, final_nodes)
    final_edge_metric = prf1_from_sets("final_edges", gold_edges, final_edges)
    validation = validate_egocentric_extraction(extraction, source_text=scene.raw_text)
    origins = summarize_relation_origins(final_graph.edges)
    metrics = {
        "scene_id": scene.scene_id,
        "condition": condition,
        "strategy": strategy,
        "run_number": run_number,
        "model": model,
        "raw_node_precision": raw_node_metric.precision,
        "raw_node_recall": raw_node_metric.recall,
        "raw_node_f1": raw_node_metric.f1,
        "raw_edge_precision": raw_edge_metric.precision,
        "raw_edge_recall": raw_edge_metric.recall,
        "raw_edge_f1": raw_edge_metric.f1,
        "final_node_precision": final_node_metric.precision,
        "final_node_recall": final_node_metric.recall,
        "final_node_f1": final_node_metric.f1,
        "final_edge_precision": final_edge_metric.precision,
        "final_edge_recall": final_edge_metric.recall,
        "final_edge_f1": final_edge_metric.f1,
        "gold_node_count": len(gold_nodes),
        "gold_edge_count": len(gold_edges),
        "raw_node_count": len(raw_nodes),
        "raw_edge_count": len(raw_edges),
        "final_node_count": len(final_nodes),
        "final_edge_count": len(final_edges),
        "ontology_conformance": validation.ontology_conformance,
        "relation_hallucination_rate": validation.relation_hallucination_rate,
        "subject_hallucinations": validation.subject_hallucinations,
        "object_hallucinations": validation.object_hallucinations,
        "evidence_hallucinations": validation.evidence_hallucinations,
        "schema_invalid_relations": validation.invalid_relations,
        "runtime_seconds": runtime_seconds,
        "execution_error": execution_error,
        "llm_extracted_edges": origins.by_origin.get("llm_extracted", 0),
        "fallback_edges": sum(
            count for origin, count in origins.by_origin.items() if origin != "llm_extracted"
        ),
    }
    return {
        "scene_id": scene.scene_id,
        "condition": condition,
        "strategy": strategy,
        "run_number": run_number,
        "metrics": metrics,
        "gold_nodes": sorted(gold_nodes),
        "gold_edges": sorted(gold_edges),
        "raw_nodes": sorted(raw_nodes),
        "raw_edges": sorted(raw_edges),
        "final_nodes": sorted(final_nodes),
        "final_edges": sorted(final_edges),
        "raw_response": raw_response,
        "extraction": extraction.model_dump(mode="json"),
        "final_graph": final_graph.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "relation_origin_summary": {
            "total_edges": origins.total_edges,
            "by_origin": origins.by_origin,
            "by_relation": origins.by_relation,
            "by_relation_origin": origins.by_relation_origin,
        },
    }


def llm_only_graph(graph: PropertyGraph) -> PropertyGraph:
    return PropertyGraph(
        nodes=graph.nodes,
        edges=[
            edge
            for edge in graph.edges
            if edge.properties.get("relation_origin") == "llm_extracted"
        ],
        stable_id_index=graph.stable_id_index,
    )


def record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["scene_id"]),
        str(record["condition"]),
        str(record["strategy"]),
        int(record["run_number"]),
    )


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def materialize_run_artifacts(artifact_dir: Path, records: list[dict[str, Any]]) -> None:
    metrics = [record["metrics"] for record in records]
    write_gzip_jsonl(
        artifact_dir / "raw_runs.jsonl.gz",
        records,
        overwrite=(artifact_dir / "raw_runs.jsonl.gz").exists(),
    )
    write_csv(
        artifact_dir / "per_scene_metrics.csv",
        metrics,
        overwrite=(artifact_dir / "per_scene_metrics.csv").exists(),
    )
    write_csv(
        artifact_dir / "end_to_end_summary.csv",
        aggregate_summary(metrics),
        overwrite=(artifact_dir / "end_to_end_summary.csv").exists(),
    )
    write_csv(
        artifact_dir / "stability.csv",
        stability_rows(records),
        overwrite=(artifact_dir / "stability.csv").exists(),
    )
    write_csv(
        artifact_dir / "relation_type_metrics.csv",
        relation_type_rows(records),
        overwrite=(artifact_dir / "relation_type_metrics.csv").exists(),
    )
    write_csv(
        artifact_dir / "hallucination_report.csv",
        hallucination_rows(records),
        overwrite=(artifact_dir / "hallucination_report.csv").exists(),
    )


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["strategy"]))].append(row)
    output = []
    for (condition, strategy), selected in sorted(grouped.items()):
        output.append(
            {
                "condition": condition,
                "strategy": strategy,
                "run_count": len(selected),
                "error_count": sum(bool(row["execution_error"]) for row in selected),
                **{
                    f"mean_{key}": rounded_mean(selected, key)
                    for key in [
                        "raw_node_f1",
                        "raw_edge_f1",
                        "final_node_f1",
                        "final_edge_f1",
                        "ontology_conformance",
                        "relation_hallucination_rate",
                        "runtime_seconds",
                    ]
                },
                "std_final_node_f1": rounded_std(selected, "final_node_f1"),
                "std_final_edge_f1": rounded_std(selected, "final_edge_f1"),
            }
        )
    return output


def stability_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scene_id"], record["condition"], record["strategy"])].append(record)
    rows = []
    for (scene_id, condition, strategy), selected in sorted(grouped.items()):
        node_scores = []
        edge_scores = []
        for first, second in combinations(selected, 2):
            node_scores.append(jaccard(set(first["final_nodes"]), set(second["final_nodes"])))
            edge_scores.append(jaccard(set(first["final_edges"]), set(second["final_edges"])))
        rows.append(
            {
                "scene_id": scene_id,
                "condition": condition,
                "strategy": strategy,
                "run_count": len(selected),
                "pair_count": len(node_scores),
                "mean_node_overlap": round(mean(node_scores), 6) if node_scores else 1.0,
                "mean_edge_agreement": round(mean(edge_scores), 6) if edge_scores else 1.0,
            }
        )
    return rows


def relation_type_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        for stage in ["raw", "final"]:
            predicted = set(record[f"{stage}_edges"])
            gold = set(record["gold_edges"])
            for relation_type in sorted(SCORING_RELATION_TYPES):
                gold_typed = {item for item in gold if edge_type(item) == relation_type}
                predicted_typed = {item for item in predicted if edge_type(item) == relation_type}
                counter = counters[
                    (record["condition"], record["strategy"], stage, relation_type)
                ]
                counter["tp"] += len(gold_typed & predicted_typed)
                counter["fp"] += len(predicted_typed - gold_typed)
                counter["fn"] += len(gold_typed - predicted_typed)
    rows = []
    for (condition, strategy, stage, relation_type), values in sorted(counters.items()):
        rows.append(
            {
                "condition": condition,
                "strategy": strategy,
                "stage": stage,
                "relation_type": relation_type,
                "true_positives": values["tp"],
                "false_positives": values["fp"],
                "false_negatives": values["fn"],
                "precision": safe_divide(values["tp"], values["tp"] + values["fp"]),
                "recall": safe_divide(values["tp"], values["tp"] + values["fn"]),
                "f1": f1_from_counts(values["tp"], values["fp"], values["fn"]),
            }
        )
    return rows


def hallucination_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": record["scene_id"],
            "condition": record["condition"],
            "strategy": record["strategy"],
            "run_number": record["run_number"],
            "execution_error": record["metrics"]["execution_error"],
            "ontology_conformance": record["metrics"]["ontology_conformance"],
            "relation_hallucination_rate": record["metrics"]["relation_hallucination_rate"],
            "subject_hallucinations": record["metrics"]["subject_hallucinations"],
            "object_hallucinations": record["metrics"]["object_hallucinations"],
            "evidence_hallucinations": record["metrics"]["evidence_hallucinations"],
            "schema_invalid_relations": record["metrics"]["schema_invalid_relations"],
        }
        for record in records
    ]


def edge_type(value: str) -> str:
    parts = value.split("|")
    return parts[2] if len(parts) >= 3 else ""


def rounded_mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(row.get(key, 0) or 0) for row in rows), 6) if rows else 0.0


def rounded_std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0) or 0) for row in rows]
    return round(pstdev(values), 6) if values else 0.0


def safe_divide(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0


def capture_runtime_environment(model: str, model_digest: str) -> RuntimeEnvironment:
    return RuntimeEnvironment(
        model=model,
        model_digest=model_digest,
        machine="MacBook Air Mac16,12",
        chip="Apple M4 (10 cores)",
        memory_gb=16,
        operating_system=platform.platform(),
        python_version=platform.python_version(),
        uv_version=command_output(["uv", "--version"]),
    )


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_manifest(
    *,
    artifact_dir: Path,
    gold_path: Path,
    indego_root: Path,
    code_commit: str,
    module_versions: ModuleVersions,
    environment: RuntimeEnvironment,
    parameters: BaselineParameters,
    split: DatasetSplitManifest,
    gold_records: list[dict[str, Any]],
    status: str,
) -> BaselineManifest:
    code_paths = [
        Path("pyproject.toml"),
        Path("uv.lock"),
        DEFAULT_MODULE_VERSIONS,
        Path("config/industrial_tool_keywords.txt"),
        Path("backend/llm/client.py"),
        Path("backend/extraction/extractor.py"),
        Path("backend/preprocessing/unified_text.py"),
        Path("backend/cleaning/factory_data.py"),
        Path("backend/pipeline/industrial_text_to_kg.py"),
        Path("backend/pipeline/relation_candidate_pipeline.py"),
        Path("backend/graph/property_graph.py"),
        Path("scripts/run_easg_real_llm_lightweight_experiments.py"),
        Path("scripts/run_versioned_baseline.py"),
    ]
    return BaselineManifest(
        baseline_id=BASELINE_ID,
        baseline_tag=BASELINE_TAG,
        code_commit=code_commit,
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,  # type: ignore[arg-type]
        module_versions=module_versions,
        environment=environment,
        parameters=parameters,
        dataset_name="industrial_reviewed_gold_combined_v2_v3",
        scene_count=len(gold_records),
        video_count=len({str(row["video_id"]) for row in gold_records}),
        gold_entity_count=sum(len(row.get("gold_entities", [])) for row in gold_records),
        gold_relation_count=sum(len(row.get("gold_relations", [])) for row in gold_records),
        split=split,
        code_and_config_files=[fingerprint_file(path) for path in code_paths],
        gold_files=[fingerprint_file(gold_path)],
        source_files=fingerprint_tree(indego_root),
        notes=[
            "The baseline LLM is run once; future experiments load these stored rows.",
            "reviewed_unified is annotation-derived and is an upper-bound condition.",
            "Raw and post-processed relation metrics are reported separately.",
            "Historical five-scene metrics are not this baseline.",
        ],
    )


def result_fingerprints(artifact_dir: Path) -> list[Any]:
    excluded = {"baseline_manifest.json", "SHA256SUMS", "COMPLETED"}
    return [
        fingerprint_file(path, root=artifact_dir)
        for path in sorted(artifact_dir.iterdir())
        if path.is_file() and path.name not in excluded
    ]


def write_baseline_readme(
    artifact_dir: Path,
    manifest: BaselineManifest,
    run_count: int,
    *,
    limit: int | None,
) -> None:
    label = "SMOKE" if limit is not None else "CANONICAL"
    text = f"""# Industrial V1 Frozen Baseline

- Baseline ID: `{manifest.baseline_id}`
- Git tag: `{manifest.baseline_tag}`
- Code commit: `{manifest.code_commit}`
- Model: `{manifest.environment.model}@{manifest.environment.model_digest}`
- Dataset: `{manifest.dataset_name}`
- Mode: `{label}`
- Scenes: `{manifest.scene_count}`
- Real LLM run records: `{run_count}`

This baseline is executed once. Future optimization experiments must load its
stored per-run metrics and raw outputs instead of rerunning the baseline model.

`reviewed_unified` is derived from human-reviewed annotations and represents a
controlled upper bound. It is not evidence that automatic preprocessing has
reached the same quality.

Failed or timed-out calls remain in the aggregate. Raw LLM relation metrics and
final post-processed metrics are stored separately.
"""
    path = artifact_dir / "README.md"
    if path.exists():
        path.write_text(text, encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")


def write_checksums(artifact_dir: Path) -> None:
    lines = []
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file() or path.name in {"SHA256SUMS", "COMPLETED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.name}")
    (artifact_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_completed_baseline(artifact_dir: Path) -> None:
    required = {
        "baseline_manifest.json",
        "cleaning_metrics.json",
        "preprocessing_metrics.csv",
        "end_to_end_summary.csv",
        "per_scene_metrics.csv",
        "relation_type_metrics.csv",
        "stability.csv",
        "hallucination_report.csv",
        "raw_runs.jsonl.gz",
        "README.md",
        "SHA256SUMS",
        "COMPLETED",
    }
    missing = sorted(name for name in required if not (artifact_dir / name).exists())
    if missing:
        raise FileNotFoundError(f"Baseline is incomplete; missing: {missing}")
    manifest = BaselineManifest.model_validate_json(
        (artifact_dir / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.status != "complete":
        raise ValueError(f"Baseline manifest is not complete: {manifest.status}")
    checksum_rows = [
        line.split("  ", 1)
        for line in (artifact_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    issues = []
    for expected, name in checksum_rows:
        path = artifact_dir / name
        if not path.exists() or sha256_file(path) != expected:
            issues.append(name)
    if issues:
        raise ValueError(f"Baseline checksum mismatch: {issues}")
    print(
        f"Verified {manifest.baseline_id}: {len(checksum_rows)} checksums, "
        f"status={manifest.status}",
        flush=True,
    )


if __name__ == "__main__":
    main()
