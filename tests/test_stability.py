"""Tests for graph stability metrics.

Controlled fake extractors create identical and variable repeated outputs so
the metrics can be checked against graph and grounding variation.
"""

from backend.datasets.benchmark import MVP_BENCHMARK
from backend.evaluation.stability import compare_graphs, evaluate_stability
from backend.graph.property_graph import build_property_graph
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


class StableExtractor:
    """Return the same Gold graph on every call."""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        if "weld_demo_01" in text:
            return WELDING_SCENE_EXPECTED.model_copy(deep=True)
        return INSPECTION_SCENE_EXPECTED.model_copy(deep=True)


class VariableExtractor:
    """Add one unsupported relation to the second unified inspection run."""

    def __init__(self) -> None:
        self.unified_inspection_calls = 0

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        if "weld_demo_01" in text:
            return WELDING_SCENE_EXPECTED.model_copy(deep=True)

        result = INSPECTION_SCENE_EXPECTED.model_copy(deep=True)
        if "[NORMALIZED SEGMENT]" in text:
            self.unified_inspection_calls += 1
            if self.unified_inspection_calls == 2:
                result.uses_tool.append(
                    UsesTool(
                        action=Action(name="measure gap"),
                        tool=Tool(name="laser scanner"),
                    )
                )
        return result


def test_identical_repeated_runs_receive_perfect_overlap() -> None:
    """Identical repeated output receives perfect overlap scores."""

    batch = RawUnifiedExperimentRunner(
        config=ExperimentRunConfig(repetitions=2),
        extractor=StableExtractor(),
    ).run(MVP_BENCHMARK)

    report = evaluate_stability(batch)
    summary = report.summary_for("weld_demo_01", "raw")

    assert summary.run_count == 2
    assert summary.pair_count == 1
    assert summary.mean_node_overlap == 1.0
    assert summary.mean_relation_agreement == 1.0
    assert summary.mean_graph_overlap == 1.0
    assert summary.variations["relation_count"].variance == 0.0


def test_graph_comparison_detects_changed_relation() -> None:
    """Adding one relation reduces both node and relation overlap."""

    first = INSPECTION_SCENE_EXPECTED.model_copy(deep=True)
    second = INSPECTION_SCENE_EXPECTED.model_copy(deep=True)
    second.uses_tool.append(
        UsesTool(
            action=Action(name="measure gap"),
            tool=Tool(name="laser scanner"),
        )
    )

    agreement = compare_graphs(
        build_property_graph(first),
        build_property_graph(second),
        first_run_number=1,
        second_run_number=2,
    )

    assert agreement.node_overlap < 1.0
    assert agreement.relation_agreement < 1.0
    assert agreement.graph_overlap < 1.0


def test_variable_output_produces_stability_variation() -> None:
    """One unsupported extraction reduces unified stability."""

    batch = RawUnifiedExperimentRunner(
        config=ExperimentRunConfig(repetitions=2),
        extractor=VariableExtractor(),
    ).run(MVP_BENCHMARK)

    report = evaluate_stability(batch)
    raw = report.summary_for("inspect_demo_01", "raw")
    unified = report.summary_for("inspect_demo_01", "unified")

    assert raw.mean_graph_overlap == 1.0
    assert unified.mean_graph_overlap < 1.0
    assert unified.mean_relation_agreement < 1.0
    assert unified.variations["relation_count"].variance > 0.0
    assert unified.variations["filtered_relation_count"].variance > 0.0


def test_one_run_is_marked_as_without_pairwise_comparison() -> None:
    """A single run is reportable but has no comparison pair."""

    batch = RawUnifiedExperimentRunner(extractor=StableExtractor()).run(
        MVP_BENCHMARK
    )

    summary = evaluate_stability(batch).summary_for("weld_demo_01", "raw")

    assert summary.run_count == 1
    assert summary.pair_count == 0
    assert summary.mean_graph_overlap == 1.0
