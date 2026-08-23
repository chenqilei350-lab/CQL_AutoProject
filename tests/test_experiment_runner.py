"""Tests for the repeated extraction experiment runner.

These tests use fake extractors instead of Ollama so experiment orchestration
can be checked without depending on a running local model service.
"""

import json

import pytest

from backend.datasets.benchmark import MVP_BENCHMARK
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


class FakeExtractor:
    """Return Gold by scene ID and record the actual input text."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, type[EgocentricVideoExtraction]]] = []

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        self.calls.append((text, response_model))
        if "weld_demo_01" in text:
            return WELDING_SCENE_EXPECTED.model_copy(deep=True)
        return INSPECTION_SCENE_EXPECTED.model_copy(deep=True)


class HallucinatingUnifiedExtractor(FakeExtractor):
    """Add one source-unsupported tool relation only for unified input."""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        extraction = super().extract(text, response_model, system_prompt)
        if "[NORMALIZED SEGMENT]" in text and "inspect_demo_01" in text:
            extraction.uses_tool.append(
                UsesTool(
                    action=Action(name="measure gap"),
                    tool=Tool(name="laser scanner"),
                    source_text="measure gap with laser scanner",
                )
            )
        return extraction


def test_runner_executes_each_scene_condition_and_repeat() -> None:
    """Two scenes, two inputs, and two repeats should produce eight runs."""

    extractor = FakeExtractor()
    runner = RawUnifiedExperimentRunner(
        config=ExperimentRunConfig(repetitions=2),
        extractor=extractor,
    )

    result = runner.run(MVP_BENCHMARK)

    assert len(result.runs) == 8
    assert len(extractor.calls) == 8
    assert all(call[1] is EgocentricVideoExtraction for call in extractor.calls)
    assert len(result.runs_for("weld_demo_01", "raw")) == 2
    assert len(result.runs_for("weld_demo_01", "unified")) == 2


def test_runner_only_switches_input_representation() -> None:
    """Raw and unified use the same schema but different text forms."""

    result = RawUnifiedExperimentRunner(extractor=FakeExtractor()).run(
        MVP_BENCHMARK
    )

    raw_run = result.runs_for("inspect_demo_01", "raw")[0]
    unified_run = result.runs_for("inspect_demo_01", "unified")[0]

    assert raw_run.input_text == MVP_BENCHMARK.get_scene(
        "inspect_demo_01"
    ).raw_text
    assert "[ACTION SEQUENCE]" in unified_run.input_text
    assert raw_run.input_text != unified_run.input_text
    assert raw_run.model == unified_run.model == "llama3.1:8b"


def test_runner_attaches_graph_and_validation_report() -> None:
    """Each run includes its graph and ontology/grounding validation."""

    result = RawUnifiedExperimentRunner(extractor=FakeExtractor()).run(
        MVP_BENCHMARK
    )
    run = result.runs_for("weld_demo_01", "raw")[0]

    assert run.graph.nodes
    assert run.graph.edges
    assert run.validation.invalid_relations == 0
    assert run.validation.ontology_conformance == 1.0


def test_unified_validation_still_uses_original_source_text() -> None:
    """Original evidence checks still detect unsupported unified facts."""

    result = RawUnifiedExperimentRunner(
        extractor=HallucinatingUnifiedExtractor()
    ).run(MVP_BENCHMARK)
    unified_run = result.runs_for("inspect_demo_01", "unified")[0]

    assert unified_run.validation.object_hallucinations >= 1
    assert unified_run.validation.filtered_relation_count >= 1


def test_batch_result_can_be_saved_as_jsonl(tmp_path) -> None:
    """Experiment output can be saved for metrics and result tables."""

    result = RawUnifiedExperimentRunner(extractor=FakeExtractor()).run(
        MVP_BENCHMARK
    )
    output_path = result.save_jsonl(tmp_path / "runs.jsonl")
    records = [
        json.loads(line) for line in output_path.read_text().splitlines()
    ]

    assert len(records) == 4
    assert records[0]["scene_id"] == "weld_demo_01"
    assert "extraction" in records[0]
    assert "graph" in records[0]
    assert "validation" in records[0]


def test_repetition_count_must_be_positive() -> None:
    """The repetition count must be positive for comparisons."""

    with pytest.raises(ValueError):
        ExperimentRunConfig(repetitions=0)


def test_runner_can_record_failed_extraction_and_continue() -> None:
    """Batch experiments record one failure without dropping other scenes."""

    class FailingExtractor(FakeExtractor):
        def extract(
            self,
            text: str,
            response_model: type[EgocentricVideoExtraction],
            system_prompt: str | None = None,
        ) -> EgocentricVideoExtraction:
            if "inspect_demo_01" in text and "[NORMALIZED SEGMENT]" not in text:
                raise ValueError("Model output did not satisfy the schema")
            return super().extract(text, response_model, system_prompt)

    result = RawUnifiedExperimentRunner(
        config=ExperimentRunConfig(continue_on_error=True),
        extractor=FailingExtractor(),
    ).run(MVP_BENCHMARK)
    failed = result.runs_for("inspect_demo_01", "raw")[0]

    assert len(result.runs) == 4
    assert failed.execution_error is not None
    assert failed.validation.ontology_conformance == 0.0
    assert not failed.graph.nodes
