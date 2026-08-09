"""Staged optimization workflow tests."""

from pathlib import Path

from backend.pipeline.staged_optimization import (
    report_to_markdown,
    run_staged_optimization,
)


def test_staged_smoke_runs_hospital_summary_and_egocentric_cv(tmp_path) -> None:
    """Smoke mode should verify the full staged flow without real LLM calls."""

    report = run_staged_optimization(
        mode="smoke",
        stage="all",
        output_dir=tmp_path,
    )

    assert report.hospital_reference is not None
    assert report.hospital_reference.selected_signals
    assert report.egocentric_cv is not None
    assert report.egocentric_cv.condition_summaries
    assert (tmp_path / "staged_optimization_report.json").exists()
    assert (tmp_path / "egocentric_cv" / "cross_validation_report.json").exists()
    assert (tmp_path / "egocentric_cv" / "cross_validation_summary.csv").exists()
    assert (tmp_path / "egocentric_cv" / "algorithm_runs.jsonl").exists()
    assert "debug_trace" in (
        tmp_path / "egocentric_cv" / "algorithm_runs.jsonl"
    ).read_text(encoding="utf-8")


def test_staged_cv_smoke_compares_raw_and_unified_conditions(tmp_path) -> None:
    """CV stage should include input representation and algorithm condition names."""

    report = run_staged_optimization(
        mode="smoke",
        stage="cv",
        output_dir=tmp_path,
        cv_conditions=("baseline_one_stage",),
        input_conditions=("raw", "unified"),
    )

    conditions = {
        summary.condition
        for summary in report.egocentric_cv.condition_summaries
    }

    assert conditions == {"raw:baseline_one_stage", "unified:baseline_one_stage"}
    assert all(
        summary.fold_count == 5
        for summary in report.egocentric_cv.condition_summaries
    )


def test_staged_report_markdown_contains_key_sections(tmp_path) -> None:
    """Terminal summary should mention the three intended stages."""

    report = run_staged_optimization(
        mode="smoke",
        stage="all",
        output_dir=tmp_path,
    )
    markdown = report_to_markdown(report)

    assert "Hospital Reference" in markdown
    assert "Egocentric Cross-validation" in markdown
    assert "Real Data" in markdown
    assert str(Path(tmp_path)) in markdown
