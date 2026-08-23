"""Tests for experiment result tables and error analysis.

These tests use controlled extraction results instead of a real Ollama service.
They verify that correct output receives full credit and that extra relations
appear in both summary tables and error analysis.
"""

import csv
import io

from backend.datasets.benchmark import MVP_BENCHMARK
from backend.evaluation.reporting import build_comparison_report
from backend.pipeline.experiment_runner import (
    ExperimentRunConfig,
    RawUnifiedExperimentRunner,
)
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    WELDING_SCENE_EXPECTED,
)
from backend.schemas.egocentric_video import (
    Action,
    EgocentricVideoExtraction,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


class GoldForRawAndNoisyForUnifiedExtractor:
    """Return Gold for raw input and add one unsupported unified relation."""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        if "weld_demo_01" in text:
            result = WELDING_SCENE_EXPECTED.model_copy(deep=True)
        else:
            result = INSPECTION_SCENE_EXPECTED.model_copy(deep=True)
        if "[NORMALIZED SEGMENT]" in text:
            result.uses_tool.append(
                UsesTool(
                    action=Action(name="measure gap"),
                    tool=Tool(name="laser scanner"),
                )
            )
        return result


def _build_report():
    """Create a controlled report containing two repeated runs."""

    batch = RawUnifiedExperimentRunner(
        config=ExperimentRunConfig(repetitions=2),
        extractor=GoldForRawAndNoisyForUnifiedExtractor(),
    ).run(MVP_BENCHMARK)
    return build_comparison_report(batch, MVP_BENCHMARK)


def test_report_compares_raw_and_unified_against_gold() -> None:
    """Gold raw output is perfect; extra unified facts reduce precision."""

    report = _build_report()
    raw = report.row_for("raw")
    unified = report.row_for("unified")

    assert raw.node_metrics.f1 == 1.0
    assert raw.relation_metrics.f1 == 1.0
    assert unified.node_metrics.precision < 1.0
    assert unified.relation_metrics.precision < 1.0
    assert unified.mean_filtered_relation_count > raw.mean_filtered_relation_count


def test_report_includes_grounding_and_graph_errors() -> None:
    """Extra tools and relations must appear in readable error analysis."""

    report = _build_report()
    details = "\n".join(item.detail for item in report.error_observations)

    assert "laser scanner" in details
    assert any(
        item.error_type == "Extra relation"
        for item in report.error_observations
    )
    assert any(
        item.error_type.startswith("Validation issue:")
        for item in report.error_observations
    )


def test_report_exports_markdown_and_csv(tmp_path) -> None:
    """Aggregate results can be saved as Markdown and chart-ready CSV."""

    report = _build_report()
    markdown_path = report.save_markdown(tmp_path / "comparison.md")
    csv_path = report.save_csv(tmp_path / "comparison.csv")

    markdown = markdown_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))

    assert "Raw vs Unified Experiment Results" in markdown
    assert "Relation F1" in markdown
    assert "laser scanner" in markdown
    assert [row["condition"] for row in rows] == ["raw", "unified"]
    assert "mean_graph_overlap" in rows[0]
