"""Raw 与 Unified 对比实验的重复抽取运行器。

本模块位于实验流程的中间位置：

    小型数据集 -> 本运行器 -> 稳定性指标与结果表

它负责用同一套抽取设置反复处理两种输入文本，并把每一次运行的
结构化抽取、property graph 与 grounding 校验结果一起保存下来。
这样下一阶段才能公平比较两种文本表示是否影响结果质量与稳定性。

注意：
- 本模块不负责自动生成统一格式文本，该工作由 preprocessing 模块完成。
- 本模块不负责计算多次运行的重合度或方差，该工作留给稳定性评价模块。
- 对统一格式输入的证据检查仍回到原始文本，避免格式化文本掩盖无依据事实。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Protocol

from pydantic import BaseModel, Field

from backend.datasets.benchmark import (
    InputCondition,
    MVP_BENCHMARK,
    BenchmarkDataset,
    BenchmarkScene,
)
from backend.evaluation.ontology_validation import (
    ValidationIssue,
    ValidationReport,
    validate_egocentric_extraction,
)
from backend.extraction.extractor import Extractor
from backend.graph.property_graph import PropertyGraph, build_property_graph
from backend.llm.client import DEFAULT_MODEL
from backend.schemas.egocentric_video import EgocentricVideoExtraction


class ExtractionBackend(Protocol):
    """抽取后端的最小接口，便于真实 Ollama 与测试替身共用运行器。"""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        """根据给定 schema 从文本中抽取结构化结果。"""


class ExperimentRunConfig(BaseModel):
    """一次 Raw/Unified 对比实验中保持固定的运行设置。"""

    experiment_name: str = "raw_vs_unified_mvp"
    model: str = DEFAULT_MODEL
    language: str = "English"
    temperature: float = 0.0
    max_retries: int = 3
    repetitions: int = Field(default=1, ge=1)
    conditions: tuple[InputCondition, ...] = ("raw", "unified")
    continue_on_error: bool = False


class ExtractionRunRecord(BaseModel):
    """保存一个场景在一种输入条件下的一次完整运行结果。"""

    experiment_name: str
    scene_id: str
    condition: InputCondition
    run_number: int
    input_text: str
    model: str
    extraction: EgocentricVideoExtraction
    graph: PropertyGraph
    validation: ValidationReport
    execution_error: str | None = None


class ExperimentBatchResult(BaseModel):
    """保存整个批量实验结果，并提供导出和筛选功能。"""

    config: ExperimentRunConfig
    dataset_name: str
    runs: list[ExtractionRunRecord] = Field(default_factory=list)

    def runs_for(
        self, scene_id: str, condition: InputCondition
    ) -> list[ExtractionRunRecord]:
        """取得某个场景在指定输入条件下的全部重复运行记录。"""

        return [
            run
            for run in self.runs
            if run.scene_id == scene_id and run.condition == condition
        ]

    def to_jsonl(self) -> str:
        """将每次运行保存为一行 JSON，便于后续统计或人工检查。"""

        return "\n".join(
            json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
            for run in self.runs
        )

    def save_jsonl(self, output_path: str | Path) -> Path:
        """把批量结果写入 JSONL 文件，并返回实际输出路径。"""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl() + "\n", encoding="utf-8")
        return path


class RawUnifiedExperimentRunner:
    """使用同一抽取设置执行 Raw 与 Unified 输入对比实验。"""

    def __init__(
        self,
        config: ExperimentRunConfig | None = None,
        extractor: ExtractionBackend | None = None,
    ) -> None:
        """创建运行器；测试时可传入不调用真实 LLM 的替代抽取器。"""

        self.config = config or ExperimentRunConfig()
        self.extractor = extractor or Extractor(
            model=self.config.model,
            language=self.config.language,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )

    def run(
        self, dataset: BenchmarkDataset = MVP_BENCHMARK
    ) -> ExperimentBatchResult:
        """按场景、输入形式和重复次数执行完整抽取实验。"""

        records = list(self.iter_runs(dataset))
        return ExperimentBatchResult(
            config=self.config,
            dataset_name=dataset.name,
            runs=records,
        )

    def iter_runs(
        self, dataset: BenchmarkDataset = MVP_BENCHMARK
    ) -> Iterator[ExtractionRunRecord]:
        """逐条产生运行结果，供长时间真实实验即时保存 checkpoint。"""

        for scene in dataset.scenes:
            for condition in self.config.conditions:
                for run_number in range(1, self.config.repetitions + 1):
                    try:
                        yield self._run_once(
                            scene=scene,
                            condition=condition,
                            run_number=run_number,
                        )
                    except Exception as error:
                        if not self.config.continue_on_error:
                            raise
                        yield self._failed_run_record(
                            scene=scene,
                            condition=condition,
                            run_number=run_number,
                            error=error,
                        )

    def _run_once(
        self,
        scene: BenchmarkScene,
        condition: InputCondition,
        run_number: int,
    ) -> ExtractionRunRecord:
        """完成单次抽取、构图和基于原文的证据校验。"""

        input_text = scene.input_text(condition)
        extraction = self.extractor.extract(
            text=input_text,
            response_model=EgocentricVideoExtraction,
        )
        graph = build_property_graph(extraction)

        # 即使模型阅读的是统一格式文本，生成事实也必须由原始文本支持。
        validation = validate_egocentric_extraction(
            extraction,
            source_text=scene.raw_text,
        )

        return ExtractionRunRecord(
            experiment_name=self.config.experiment_name,
            scene_id=scene.scene_id,
            condition=condition,
            run_number=run_number,
            input_text=input_text,
            model=self.config.model,
            extraction=extraction,
            graph=graph,
            validation=validation,
        )

    def _failed_run_record(
        self,
        scene: BenchmarkScene,
        condition: InputCondition,
        run_number: int,
        error: Exception,
    ) -> ExtractionRunRecord:
        """把一次 schema/LLM 失败保存为可评价记录，而不是中断完整实验。"""

        error_message = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        return ExtractionRunRecord(
            experiment_name=self.config.experiment_name,
            scene_id=scene.scene_id,
            condition=condition,
            run_number=run_number,
            input_text=scene.input_text(condition),
            model=self.config.model,
            extraction=EgocentricVideoExtraction(source_text=scene.raw_text),
            graph=PropertyGraph(),
            validation=ValidationReport(
                invalid_relations=1,
                ontology_conformance=0.0,
                relation_hallucination_rate=1.0,
                filtered_relation_count=1,
                issues=[
                    ValidationIssue(
                        severity="error",
                        kind="extraction_failed",
                        message=error_message,
                    )
                ],
            ),
            execution_error=error_message,
        )
