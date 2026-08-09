import csv
import json
from pathlib import Path

import pytest

from backend.evaluation.versioned_experiment import (
    BaselineManifest,
    BaselineParameters,
    DatasetSplitManifest,
    ModuleVersions,
    RuntimeEnvironment,
)
from scripts.export_public_baseline import (
    PRIVATE_ARTIFACTS,
    PUBLIC_RESULT_FILES,
    export_public_baseline,
    verify_public_export,
)


def _write_fake_baseline(root: Path) -> None:
    split = DatasetSplitManifest(
        development_scene_ids=["scene-1"],
        quick_regression_scene_ids=["scene-1"],
        holdout_scene_ids=["scene-2"],
        video_id_by_scene={"scene-1": "video-1", "scene-2": "video-2"},
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
            model_digest="digest",
            machine="MacBook Air",
            chip="Apple M4",
            memory_gb=16,
            operating_system="macOS",
            python_version="3.12",
            uv_version="0.11",
        ),
        parameters=BaselineParameters(),
        dataset_name="industrial",
        scene_count=2,
        video_count=2,
        gold_entity_count=4,
        gold_relation_count=3,
        split=split,
        source_files=[
            {
                "path": "/private/licensed/transcript.json",
                "sha256": "a" * 64,
                "size_bytes": 123,
            }
        ],
    )
    root.mkdir()
    (root / "baseline_manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    for name in PUBLIC_RESULT_FILES:
        if name == "cleaning_metrics.json":
            (root / name).write_text(
                json.dumps(
                    {
                        "root_path": "/private/licensed",
                        "record_count": 2,
                        "source_file_count": 1,
                    }
                ),
                encoding="utf-8",
            )
        elif name.endswith(".json"):
            (root / name).write_text("{}\n", encoding="utf-8")
        else:
            with (root / name).open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["scene_id", "metric"])
                writer.writerow(["scene-1", "0.5"])
    (root / "cleaned_records.jsonl.gz").write_bytes(b"licensed text")
    (root / "raw_runs.jsonl.gz").write_bytes(b"raw response")


def test_public_export_excludes_private_artifacts_and_local_paths(tmp_path: Path) -> None:
    source = tmp_path / "private"
    output = tmp_path / "public"
    _write_fake_baseline(source)

    export_public_baseline(source, output, verify_source=False)
    verify_public_export(output)

    names = {path.name for path in output.iterdir()}
    assert not names.intersection(PRIVATE_ARTIFACTS)
    manifest_text = (output / "public_manifest.json").read_text(encoding="utf-8")
    assert "/private/licensed" not in manifest_text
    cleaning_text = (output / "cleaning_metrics.json").read_text(encoding="utf-8")
    assert "/private/licensed" not in cleaning_text
    assert json.loads(manifest_text)["source_identity"]["file_count"] == 1


def test_completed_public_export_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "private"
    output = tmp_path / "public"
    _write_fake_baseline(source)
    export_public_baseline(source, output, verify_source=False)

    with pytest.raises(FileExistsError, match="immutable"):
        export_public_baseline(source, output, verify_source=False)
