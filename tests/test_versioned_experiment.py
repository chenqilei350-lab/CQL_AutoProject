import json
from pathlib import Path

import pytest

from backend.evaluation.versioned_experiment import (
    BaselineManifest,
    BaselineParameters,
    DatasetSplitManifest,
    ModuleVersions,
    RuntimeEnvironment,
    compare_metric_rows,
    ensure_artifact_directory,
    fingerprint_file,
    verify_fingerprints,
    write_json,
)
from scripts.run_versioned_baseline import (
    append_checkpoint,
    build_split_manifest,
    load_checkpoint,
    load_gold_records,
)


GOLD_PATH = Path(
    "data/industrial_gold_reviewed/industrial_gold_combined_v2_v3_reviewed.jsonl"
)


def test_baseline_manifest_round_trip() -> None:
    split = DatasetSplitManifest(
        development_scene_ids=["s1"],
        quick_regression_scene_ids=["s1"],
        holdout_scene_ids=["s2"],
        video_id_by_scene={"s1": "v1", "s2": "v2"},
    )
    manifest = BaselineManifest(
        baseline_id="baseline",
        baseline_tag="tag",
        code_commit="abc123",
        created_at="2026-08-09T00:00:00+00:00",
        status="complete",
        module_versions=ModuleVersions(
            data_cleaning="1.0.0",
            text_preprocessing="1.0.0",
            kg_extraction="1.0.0",
        ),
        environment=RuntimeEnvironment(
            model="llama3.1:8b",
            model_digest="46e0c10c039e",
            machine="MacBook Air",
            chip="Apple M4",
            memory_gb=16,
            operating_system="macOS",
            python_version="3.12.13",
            uv_version="0.11.13",
        ),
        parameters=BaselineParameters(),
        dataset_name="industrial",
        scene_count=2,
        video_count=2,
        gold_entity_count=4,
        gold_relation_count=2,
        split=split,
    )
    restored = BaselineManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest


def test_fingerprint_detects_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("first", encoding="utf-8")
    fingerprint = fingerprint_file(path, root=tmp_path)
    assert verify_fingerprints([fingerprint], root=tmp_path) == []
    path.write_text("second", encoding="utf-8")
    assert verify_fingerprints([fingerprint], root=tmp_path) == [
        "sha256_mismatch:input.txt"
    ]


def test_completed_artifact_directory_is_immutable(tmp_path: Path) -> None:
    directory = tmp_path / "baseline"
    directory.mkdir()
    (directory / "COMPLETED").write_text("done\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        ensure_artifact_directory(directory)


def test_json_writer_refuses_silent_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json(path, {"version": 1})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_json(path, {"version": 2})
    write_json(path, {"version": 2}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2


def test_checkpoint_round_trip_supports_resume(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    first = {
        "scene_id": "s1",
        "condition": "raw",
        "strategy": "layered",
        "run_number": 1,
    }
    second = {**first, "run_number": 2}
    append_checkpoint(path, first)
    append_checkpoint(path, second)
    assert load_checkpoint(path) == [first, second]


def test_paired_comparison_does_not_require_baseline_rerun() -> None:
    baseline = [metric_row("s1", 0.20, 0.50)]
    candidate = [metric_row("s1", 0.25, 0.51)]
    report = compare_metric_rows("baseline", "change", baseline, candidate)
    assert report.paired_run_count == 1
    assert report.decision == "improved"
    edge = next(item for item in report.metrics if item.metric == "final_edge_f1")
    assert edge.absolute_delta == pytest.approx(0.05)


def test_missing_run_key_blocks_keep_revert_conclusion() -> None:
    report = compare_metric_rows(
        "baseline",
        "change",
        [metric_row("s1", 0.20, 0.50)],
        [metric_row("s2", 0.30, 0.60)],
    )
    assert report.decision == "incomplete"
    assert report.missing_baseline_keys
    assert report.missing_candidate_keys


def test_canonical_split_is_video_grouped_and_has_fixed_quick_set() -> None:
    split = build_split_manifest(load_gold_records(GOLD_PATH))
    assert len(split.video_id_by_scene) == 32
    assert len(split.quick_regression_scene_ids) == 8
    development_videos = {
        split.video_id_by_scene[scene_id] for scene_id in split.development_scene_ids
    }
    holdout_videos = {
        split.video_id_by_scene[scene_id] for scene_id in split.holdout_scene_ids
    }
    assert development_videos.isdisjoint(holdout_videos)


def metric_row(scene_id: str, edge_f1: float, node_f1: float) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "condition": "auto_unified",
        "strategy": "layered",
        "run_number": 1,
        "raw_node_f1": node_f1,
        "raw_edge_f1": edge_f1,
        "final_node_f1": node_f1,
        "final_edge_f1": edge_f1,
        "relation_hallucination_rate": 0.0,
        "ontology_conformance": 1.0,
        "runtime_seconds": 1.0,
        "execution_error": "",
    }
