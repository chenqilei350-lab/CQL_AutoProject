"""重复抽取实验运行器的测试。

这些测试使用假抽取器代替 Ollama，因此可以快速检查实验组织逻辑，
而不会依赖本地模型服务是否正在运行。
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
    """按输入中的场景编号返回 gold 结果，并记录实际调用文本。"""

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
    """仅在统一格式条件下加入一个原文不支持的工具关系。"""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        extraction = super().extract(text, response_model, system_prompt)
        if "[场景 / 片段]" in text and "inspect_demo_01" in text:
            extraction.uses_tool.append(
                UsesTool(
                    action=Action(name="measure gap"),
                    tool=Tool(name="laser scanner"),
                    source_text="measure gap with laser scanner",
                )
            )
        return extraction


def test_runner_executes_each_scene_condition_and_repeat() -> None:
    """两个场景、两种输入、两次重复应形成八条运行记录。"""

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
    """同一场景的 Raw 与 Unified 运行使用相同 schema，但输入文本不同。"""

    result = RawUnifiedExperimentRunner(extractor=FakeExtractor()).run(
        MVP_BENCHMARK
    )

    raw_run = result.runs_for("inspect_demo_01", "raw")[0]
    unified_run = result.runs_for("inspect_demo_01", "unified")[0]

    assert raw_run.input_text == MVP_BENCHMARK.get_scene(
        "inspect_demo_01"
    ).raw_text
    assert "[动作顺序]" in unified_run.input_text
    assert raw_run.input_text != unified_run.input_text
    assert raw_run.model == unified_run.model == "llama3.1:8b"


def test_runner_attaches_graph_and_validation_report() -> None:
    """每条抽取记录都应包含图结果与 ontology/grounding 校验报告。"""

    result = RawUnifiedExperimentRunner(extractor=FakeExtractor()).run(
        MVP_BENCHMARK
    )
    run = result.runs_for("weld_demo_01", "raw")[0]

    assert run.graph.nodes
    assert run.graph.edges
    assert run.validation.invalid_relations == 0
    assert run.validation.ontology_conformance == 1.0


def test_unified_validation_still_uses_original_source_text() -> None:
    """统一格式输入引入的新工具关系仍会被原始证据检查发现。"""

    result = RawUnifiedExperimentRunner(
        extractor=HallucinatingUnifiedExtractor()
    ).run(MVP_BENCHMARK)
    unified_run = result.runs_for("inspect_demo_01", "unified")[0]

    assert unified_run.validation.object_hallucinations >= 1
    assert unified_run.validation.filtered_relation_count >= 1


def test_batch_result_can_be_saved_as_jsonl(tmp_path) -> None:
    """实验输出可以保存，供下一阶段统计指标和制作结果表。"""

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
    """没有运行次数就无法比较结果，因此次数不得为零。"""

    with pytest.raises(ValueError):
        ExperimentRunConfig(repetitions=0)


def test_runner_can_record_failed_extraction_and_continue() -> None:
    """真实批量实验可把单条失败记入结果，而不丢弃其他场景。"""

    class FailingExtractor(FakeExtractor):
        def extract(
            self,
            text: str,
            response_model: type[EgocentricVideoExtraction],
            system_prompt: str | None = None,
        ) -> EgocentricVideoExtraction:
            if "inspect_demo_01" in text and "[场景 / 片段]" not in text:
                raise ValueError("模型输出没有通过 schema")
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
