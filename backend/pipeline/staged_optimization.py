"""Staged experiment orchestration for algorithm optimization.

The intended workflow is:

1. Use the hospital reference track to screen whether strategy changes are
   useful on a fast, well-understood stress case.
2. Run egocentric cross-validation with the chosen extraction strategy and
   input representation variants.
3. Run the same optimized path on real external data once it is available.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from backend.datasets.benchmark import BenchmarkDataset, InputCondition
from backend.datasets.expanded_benchmark import EXPANDED_BENCHMARK
from backend.evaluation.cross_validation import (
    CrossValidationConfig,
    CrossValidationReport,
    FoldConditionResult,
    make_folds,
    summarize_fold_results,
)
from backend.pipeline.algorithm_experiment import (
    AlgorithmCondition,
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
    EntityExtractionResult,
    MinimalEntity,
    MinimalEntityExtractionResult,
    MinimalRelation,
    MinimalRelationExtractionResult,
    RelationExtractionResult,
)
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


StageMode = Literal["smoke", "real"]
StageName = Literal["hospital", "cv", "real_data", "all"]


class HospitalReferenceSummary(BaseModel):
    """Summary of the hospital reference strategy-screening stage."""

    source: str
    output_dir: str | None = None
    rows: list[dict[str, str]] = Field(default_factory=list)
    selected_signals: list[str] = Field(default_factory=list)
    execution_error: str | None = None


class StagedOptimizationReport(BaseModel):
    """Top-level report for staged optimization."""

    mode: StageMode
    output_dir: str
    hospital_reference: HospitalReferenceSummary | None = None
    egocentric_cv: CrossValidationReport | None = None
    real_data_message: str | None = None

    def save(self, output_dir: str | Path) -> Path:
        """Save the report as JSON."""

        path = Path(output_dir) / "staged_optimization_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


def run_staged_optimization(
    mode: StageMode = "smoke",
    stage: StageName = "all",
    output_dir: str | Path = "results/staged_optimization_smoke",
    model: str = "llama3.1:8b",
    hospital_timeout: int = 120,
    cv_conditions: tuple[AlgorithmCondition, ...] = (
        "minimal_entity_relation_with_validation",
    ),
    input_conditions: tuple[InputCondition, ...] = ("raw", "unified"),
    dataset: BenchmarkDataset = EXPANDED_BENCHMARK,
) -> StagedOptimizationReport:
    """Run the staged experiment workflow."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = StagedOptimizationReport(mode=mode, output_dir=str(output_path))

    if stage in {"hospital", "all"}:
        report.hospital_reference = run_hospital_reference_stage(
            mode=mode,
            output_dir=output_path / "hospital_reference",
            model=model,
            timeout=hospital_timeout,
        )

    if stage in {"cv", "all"}:
        report.egocentric_cv = run_egocentric_cv_stage(
            mode=mode,
            output_dir=output_path / "egocentric_cv",
            dataset=dataset,
            model=model,
            timeout=float(hospital_timeout),
            algorithm_conditions=cv_conditions,
            input_conditions=input_conditions,
        )

    if stage in {"real_data", "all"}:
        report.real_data_message = (
            "Real-data stage is prepared but not executed in smoke mode. "
            "Run with --mode real and provide a real-data adapter once the "
            "external group delivers standardized text."
        )

    report.save(output_path)
    return report


def run_hospital_reference_stage(
    mode: StageMode,
    output_dir: Path,
    model: str,
    timeout: int,
) -> HospitalReferenceSummary:
    """Run or summarize the hospital reference strategy-screening stage."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        return _load_existing_hospital_reference_summary()

    command = [
        sys.executable,
        "scripts/run_hospital_gold_pilot_experiments_zh.py",
        "--suite",
        "stress",
        "--model",
        model,
        "--timeout",
        str(timeout),
        "--output-dir",
        str(output_dir),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        return HospitalReferenceSummary(
            source="live_hospital_reference",
            output_dir=str(output_dir),
            execution_error=f"Hospital reference stage failed: {error}",
        )
    return _summarize_hospital_output(output_dir)


def run_egocentric_cv_stage(
    mode: StageMode,
    output_dir: Path,
    dataset: BenchmarkDataset,
    algorithm_conditions: tuple[AlgorithmCondition, ...],
    input_conditions: tuple[InputCondition, ...],
    model: str = "llama3.1:8b",
    timeout: float = 120.0,
) -> CrossValidationReport:
    """Run egocentric cross-validation with raw/unified input variants."""

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = GoldOracleExtractor() if mode == "smoke" else None
    folds = make_folds(
        dataset,
        CrossValidationConfig(
            track="egocentric_main",
            split_strategy="leave_one_scene_out",
            conditions=tuple(
                f"{input_condition}:{condition}"
                for input_condition in input_conditions
                for condition in algorithm_conditions
            ),
        ),
    )
    fold_results: list[FoldConditionResult] = []
    run_records: list[str] = []
    for fold in folds:
        for input_condition in input_conditions:
            runner = AlgorithmExperimentRunner(
                config=AlgorithmExperimentConfig(
                    track="egocentric_main",
                    model=model,
                    timeout=timeout,
                    max_retries=1,
                    input_condition=input_condition,
                    conditions=algorithm_conditions,
                    continue_on_error=True,
                ),
                extractor=extractor,
            )
            condition_results, algorithm_result = runner.run_fold(
                dataset,
                fold,
                CrossValidationConfig(
                    track="egocentric_main",
                    conditions=algorithm_conditions,
                ),
            )
            for run in algorithm_result.runs:
                run_records.append(
                    run.model_dump_json(exclude_none=True)
                )
            for result in condition_results:
                result.condition = f"{input_condition}:{result.condition}"
                fold_results.append(result)

    config = CrossValidationConfig(
        track="egocentric_main",
        split_strategy="leave_one_scene_out",
        conditions=tuple(
            f"{input_condition}:{condition}"
            for input_condition in input_conditions
            for condition in algorithm_conditions
        ),
    )
    report = CrossValidationReport(
        dataset_name=dataset.name,
        config=config,
        folds=folds,
        fold_results=fold_results,
        condition_summaries=summarize_fold_results(fold_results),
    )
    (output_dir / "cross_validation_report.json").write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "algorithm_runs.jsonl").write_text(
        "\n".join(run_records) + ("\n" if run_records else ""),
        encoding="utf-8",
    )
    _write_cv_summary_csv(output_dir / "cross_validation_summary.csv", report)
    return report


class GoldOracleExtractor:
    """Test extractor that returns gold records without calling a real LLM."""

    def extract(
        self,
        text: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        gold = _gold_for_text(text)
        if response_model is EntityExtractionResult:
            return EntityExtractionResult(
                video_id=gold.video_id,
                scenes=gold.scenes,
                procedures=gold.procedures,
                actors=gold.actors,
                actions=gold.actions,
                tools=gold.tools,
                objects=gold.objects,
                parameters=gold.parameters,
                source_text=gold.source_text,
            )
        if response_model is RelationExtractionResult:
            return RelationExtractionResult(
                uses_tool=gold.uses_tool,
                acts_on_object=gold.acts_on_object,
                action_order=gold.action_order,
                action_causes=gold.action_causes,
                part_of_procedure=gold.part_of_procedure,
                observed_in_scene=gold.observed_in_scene,
            )
        if response_model is MinimalEntityExtractionResult:
            return _minimal_entities_for_gold(gold)
        if response_model is MinimalRelationExtractionResult:
            return _minimal_relations_for_gold(gold)
        return gold


def _gold_for_text(text: str) -> EgocentricVideoExtraction:
    mapping = {
        "weld_demo_01": WELDING_SCENE_EXPECTED,
        "inspect_demo_01": INSPECTION_SCENE_EXPECTED,
        "assembly_demo_01": ASSEMBLY_SCENE_EXPECTED,
        "maintenance_demo_01": MAINTENANCE_SCENE_EXPECTED,
        "thermal_demo_01": TEMPERATURE_SCENE_EXPECTED,
    }
    for scene_id, gold in mapping.items():
        if scene_id in text:
            return gold.model_copy(deep=True)
    return WELDING_SCENE_EXPECTED.model_copy(deep=True)


def _minimal_entities_for_gold(
    gold: EgocentricVideoExtraction,
) -> MinimalEntityExtractionResult:
    entities: list[MinimalEntity] = []
    for index, action in enumerate(gold.actions, start=1):
        entities.append(
            MinimalEntity(
                entity_id=f"action_{index}",
                entity_type="Action",
                name=action.name,
                evidence_text=action.evidence_text
                or action.source_text
                or action.name,
                confidence=action.confidence,
            )
        )
    for index, tool in enumerate(gold.tools, start=1):
        entities.append(
            MinimalEntity(
                entity_id=f"tool_{index}",
                entity_type="Tool",
                name=tool.name,
                evidence_text=tool.evidence_text or tool.source_text or tool.name,
            )
        )
    for index, obj in enumerate(gold.objects, start=1):
        entities.append(
            MinimalEntity(
                entity_id=f"object_{index}",
                entity_type="Object",
                name=obj.name,
                evidence_text=obj.evidence_text or obj.source_text or obj.name,
                confidence=obj.confidence,
            )
        )
    return MinimalEntityExtractionResult(entities=entities)


def _minimal_relations_for_gold(
    gold: EgocentricVideoExtraction,
) -> MinimalRelationExtractionResult:
    action_ids = {action.name: f"action_{index}" for index, action in enumerate(gold.actions, start=1)}
    tool_ids = {tool.name: f"tool_{index}" for index, tool in enumerate(gold.tools, start=1)}
    object_ids = {obj.name: f"object_{index}" for index, obj in enumerate(gold.objects, start=1)}
    relations: list[MinimalRelation] = []
    for relation in gold.uses_tool:
        subject_id = action_ids.get(relation.action.name)
        object_id = tool_ids.get(relation.tool.name)
        if subject_id and object_id:
            relations.append(
                MinimalRelation(
                    relation_type="USES_TOOL",
                    subject_id=subject_id,
                    object_id=object_id,
                    evidence_text=relation.evidence_text
                    or relation.source_text
                    or relation.action.name,
                    confidence=relation.confidence,
                )
            )
    for relation in gold.acts_on_object:
        subject_id = action_ids.get(relation.action.name)
        object_id = object_ids.get(relation.object.name)
        if subject_id and object_id:
            relations.append(
                MinimalRelation(
                    relation_type="ACTS_ON",
                    subject_id=subject_id,
                    object_id=object_id,
                    evidence_text=relation.evidence_text
                    or relation.source_text
                    or relation.action.name,
                    confidence=relation.confidence,
                )
            )
    for relation in gold.action_order:
        subject_id = action_ids.get(relation.before.name)
        object_id = action_ids.get(relation.after.name)
        if subject_id and object_id:
            relations.append(
                MinimalRelation(
                    relation_type="BEFORE",
                    subject_id=subject_id,
                    object_id=object_id,
                    evidence_text=relation.evidence_text
                    or relation.source_text
                    or relation.before.name,
                    confidence=relation.confidence,
                )
            )
    for relation in gold.action_causes:
        subject_id = action_ids.get(relation.cause.name)
        object_id = action_ids.get(relation.effect.name)
        if subject_id and object_id:
            relations.append(
                MinimalRelation(
                    relation_type="CAUSES",
                    subject_id=subject_id,
                    object_id=object_id,
                    evidence_text=relation.evidence_text
                    or relation.source_text
                    or relation.cause.name,
                    confidence=relation.confidence,
                )
            )
    return MinimalRelationExtractionResult(relations=relations)


def _load_existing_hospital_reference_summary() -> HospitalReferenceSummary:
    path = Path("docs/results/hospital_gold_pilot_p134_stress/p134_stress_summary.csv")
    rows = _read_csv(path) if path.exists() else []
    return HospitalReferenceSummary(
        source=str(path),
        rows=rows,
        selected_signals=[
            "strict schema prompt beats loose prompt for relation correctness",
            "dense narrative chunking improves feasibility but needs node normalization",
            "unified canonical structure helps when raw text lacks IDs/types/directions",
        ],
    )


def _summarize_hospital_output(output_dir: Path) -> HospitalReferenceSummary:
    rows: list[dict[str, str]] = []
    for filename in [
        "p1_stress_prompt_constraint.csv",
        "p3_stress_dense_lab_narrative.csv",
        "p4_stress_raw_noisy_vs_unified.csv",
    ]:
        rows.extend(_read_csv(output_dir / filename))
    return HospitalReferenceSummary(
        source="live_hospital_reference",
        output_dir=str(output_dir),
        rows=rows,
        selected_signals=_select_hospital_signals(rows),
    )


def _select_hospital_signals(rows: list[dict[str, str]]) -> list[str]:
    signals: list[str] = []
    for row in rows:
        experiment = row.get("experiment", "")
        condition = row.get("condition", "")
        edge_f1 = row.get("edge_f1", "")
        if experiment == "P1_stress_prompt_constraint" and condition == "strict_schema_prompt":
            signals.append(f"strict_schema_prompt edge_f1={edge_f1}")
        if experiment == "P4_stress_raw_noisy_vs_unified" and condition == "unified_text_structure":
            signals.append(f"unified_text_structure edge_f1={edge_f1}")
    return signals


def _write_cv_summary_csv(path: Path, report: CrossValidationReport) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "fold_count",
                "node_f1_mean",
                "node_f1_std",
                "relation_f1_mean",
                "relation_f1_std",
                "ontology_conformance_mean",
                "grounding_rate_mean",
                "hallucination_rate_mean",
                "schema_success_rate_mean",
                "runtime_seconds_mean",
            ]
        )
        for summary in report.condition_summaries:
            writer.writerow(
                [
                    summary.condition,
                    summary.fold_count,
                    summary.node_f1_mean,
                    summary.node_f1_std,
                    summary.relation_f1_mean,
                    summary.relation_f1_std,
                    summary.ontology_conformance_mean,
                    summary.grounding_rate_mean,
                    summary.hallucination_rate_mean,
                    summary.schema_success_rate_mean,
                    summary.runtime_seconds_mean,
                ]
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def report_to_markdown(report: StagedOptimizationReport) -> str:
    """Render a short Markdown summary for terminal output."""

    lines = [
        "# Staged Optimization Report",
        "",
        f"- Mode: `{report.mode}`",
        f"- Output: `{report.output_dir}`",
    ]
    if report.hospital_reference:
        lines.extend(
            [
                "",
                "## Hospital Reference",
                f"- Source: `{report.hospital_reference.source}`",
            ]
        )
        for signal in report.hospital_reference.selected_signals:
            lines.append(f"- {signal}")
    if report.egocentric_cv:
        lines.extend(["", "## Egocentric Cross-validation"])
        for summary in report.egocentric_cv.condition_summaries:
            lines.append(
                "- "
                f"`{summary.condition}`: "
                f"Node F1={summary.node_f1_mean:.4f}, "
                f"Relation F1={summary.relation_f1_mean:.4f}, "
                f"Schema success={summary.schema_success_rate_mean:.4f}"
            )
    if report.real_data_message:
        lines.extend(["", "## Real Data", f"- {report.real_data_message}"])
    return "\n".join(lines)
