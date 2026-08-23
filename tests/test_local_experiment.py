"""File-output tests for the local experiment entry point without Ollama."""

from backend.pipeline.local_experiment import run_local_experiment
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    WELDING_SCENE_EXPECTED,
)
from backend.schemas.egocentric_video import EgocentricVideoExtraction
from backend.schemas.expanded_egocentric_examples import (
    ASSEMBLY_SCENE_EXPECTED,
    MAINTENANCE_SCENE_EXPECTED,
    TEMPERATURE_SCENE_EXPECTED,
)


class ExpandedGoldExtractor:
    """Return human-authored Gold by scene to test experiment persistence."""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        examples = {
            "weld_demo_01": WELDING_SCENE_EXPECTED,
            "inspect_demo_01": INSPECTION_SCENE_EXPECTED,
            "assembly_demo_01": ASSEMBLY_SCENE_EXPECTED,
            "maintenance_demo_01": MAINTENANCE_SCENE_EXPECTED,
            "thermal_demo_01": TEMPERATURE_SCENE_EXPECTED,
        }
        for scene_id, expected in examples.items():
            if scene_id in text:
                return expected.model_copy(deep=True)
        raise ValueError("No known scene ID was found in the test input.")


class FailsInspectionRawExtractor(ExpandedGoldExtractor):
    """Simulate one real model output that fails schema validation."""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        if "inspect_demo_01" in text and "[NORMALIZED SEGMENT]" not in text:
            raise ValueError("Output fields do not satisfy the schema")
        return super().extract(text, response_model, system_prompt)


def test_local_experiment_saves_full_result_bundle(tmp_path) -> None:
    """A five-scene run saves raw records, stability, and both report formats."""

    report = run_local_experiment(
        repetitions=1,
        output_dir=tmp_path,
        extractor=ExpandedGoldExtractor(),
    )

    assert report.row_for("raw").node_metrics.f1 == 1.0
    assert report.row_for("unified").relation_metrics.f1 == 1.0
    assert (tmp_path / "extraction_runs.jsonl").exists()
    assert (tmp_path / "extraction_runs.partial.jsonl").exists()
    assert (tmp_path / "stability_report.json").exists()
    assert (tmp_path / "comparison_report.md").exists()
    assert (tmp_path / "comparison_summary.csv").exists()
    assert len((tmp_path / "extraction_runs.jsonl").read_text().splitlines()) == 10


def test_local_experiment_can_limit_scene_and_condition_for_diagnosis(tmp_path) -> None:
    """Diagnostics can rerun one scene under one input condition."""

    report = run_local_experiment(
        output_dir=tmp_path,
        extractor=ExpandedGoldExtractor(),
        scene_ids=["inspect_demo_01"],
        conditions=("raw",),
    )

    assert report.row_for("raw").relation_metrics.f1 == 1.0
    assert len((tmp_path / "extraction_runs.jsonl").read_text().splitlines()) == 1


def test_local_experiment_counts_failed_run_in_error_report(tmp_path) -> None:
    """A failed extraction is recorded while the full experiment continues."""

    report = run_local_experiment(
        output_dir=tmp_path,
        extractor=FailsInspectionRawExtractor(),
    )

    markdown = report.to_markdown()
    assert "Execution failure" in markdown
    assert "inspect_demo_01" in markdown
    assert len((tmp_path / "extraction_runs.jsonl").read_text().splitlines()) == 10
