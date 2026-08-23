"""Repeated-extraction runner for raw versus unified experiments.

The runner applies identical extraction settings to both representations and
stores structured output, property graphs, and grounding validation for every
run. Preprocessing and stability aggregation remain separate concerns.
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
    """Minimal extraction interface shared by Ollama and test doubles."""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        """Extract structured output from text under the supplied schema."""


class ExperimentRunConfig(BaseModel):
    """Settings held constant during one raw/unified comparison."""

    experiment_name: str = "raw_vs_unified_mvp"
    model: str = DEFAULT_MODEL
    language: str = "English"
    temperature: float = 0.0
    max_retries: int = 3
    repetitions: int = Field(default=1, ge=1)
    conditions: tuple[InputCondition, ...] = ("raw", "unified")
    continue_on_error: bool = False


class ExtractionRunRecord(BaseModel):
    """Complete run record for one scene and input condition."""

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
    """Batch experiment results with export and filtering helpers."""

    config: ExperimentRunConfig
    dataset_name: str
    runs: list[ExtractionRunRecord] = Field(default_factory=list)

    def runs_for(
        self, scene_id: str, condition: InputCondition
    ) -> list[ExtractionRunRecord]:
        """Return all repeated runs for one scene and condition."""

        return [
            run
            for run in self.runs
            if run.scene_id == scene_id and run.condition == condition
        ]

    def to_jsonl(self) -> str:
        """Serialize one run per JSON Lines row."""

        return "\n".join(
            json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
            for run in self.runs
        )

    def save_jsonl(self, output_path: str | Path) -> Path:
        """Write batch results to JSONL and return the output path."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl() + "\n", encoding="utf-8")
        return path


class RawUnifiedExperimentRunner:
    """Run raw and unified inputs under identical extraction settings."""

    def __init__(
        self,
        config: ExperimentRunConfig | None = None,
        extractor: ExtractionBackend | None = None,
    ) -> None:
        """Create a runner; tests may inject a backend that does not call an LLM."""

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
        """Run extraction by scene, input condition, and repetition."""

        records = list(self.iter_runs(dataset))
        return ExperimentBatchResult(
            config=self.config,
            dataset_name=dataset.name,
            runs=records,
        )

    def iter_runs(
        self, dataset: BenchmarkDataset = MVP_BENCHMARK
    ) -> Iterator[ExtractionRunRecord]:
        """Yield runs incrementally so long experiments can checkpoint promptly."""

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
        """Perform one extraction, graph build, and source-grounding validation."""

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
        """Store a schema/LLM failure as an evaluable record instead of aborting."""

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
