"""Result tables and error analysis for raw versus unified experiments.

The module compares generated graphs with human-authored Gold graphs and reports
node and relation P/R/F1, schema and grounding validation, repeated-run stability,
and human-readable error observations.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from pydantic import BaseModel, Field

from backend.datasets.benchmark import (
    InputCondition,
    MVP_BENCHMARK,
    BenchmarkDataset,
)
from backend.evaluation.metrics import PRF1, aggregate_prf1, compute_prf1
from backend.evaluation.stability import StabilityReport, evaluate_stability
from backend.graph.property_graph import PropertyGraph, build_property_graph, normalize_name
from backend.pipeline.experiment_runner import (
    ExperimentBatchResult,
    ExtractionRunRecord,
)


class RunQualityResult(BaseModel):
    """Quality of one model run relative to the human-authored Gold graph."""

    scene_id: str
    condition: InputCondition
    run_number: int
    node_metrics: PRF1
    relation_metrics: PRF1
    ontology_conformance: float
    object_grounding_rate: float
    filtered_relation_count: int


class ErrorObservation(BaseModel):
    """One human-readable observation for later error analysis."""

    scene_id: str
    condition: InputCondition
    run_number: int
    error_type: str
    detail: str


class ConditionComparisonRow(BaseModel):
    """Aggregate result row for a raw or unified condition."""

    condition: InputCondition
    run_count: int
    node_metrics: PRF1
    relation_metrics: PRF1
    mean_ontology_conformance: float
    mean_object_grounding_rate: float
    mean_filtered_relation_count: float
    mean_node_overlap: float
    mean_relation_agreement: float
    mean_graph_overlap: float


class ExperimentComparisonReport(BaseModel):
    """Exportable raw-versus-unified comparison report."""

    dataset_name: str
    run_results: list[RunQualityResult] = Field(default_factory=list)
    condition_rows: list[ConditionComparisonRow] = Field(default_factory=list)
    error_observations: list[ErrorObservation] = Field(default_factory=list)

    def row_for(self, condition: InputCondition) -> ConditionComparisonRow:
        """Return the summary row for one input condition."""

        for row in self.condition_rows:
            if row.condition == condition:
                return row
        raise KeyError(f"No result row found for input condition {condition!r}")

    def to_markdown(self) -> str:
        """Generate Markdown suitable for project documents or slides."""

        lines = [
            "# Raw vs Unified Experiment Results",
            "",
            "## Aggregate Comparison",
            "",
            "| Input | Runs | Node F1 | Relation F1 | Schema Conformance | Object Grounding | Mean Filtered Relations | Node Overlap | Relation Agreement | Graph Overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in self.condition_rows:
            lines.append(
                "| "
                f"{row.condition} | {row.run_count} | {row.node_metrics.f1:.4f} | "
                f"{row.relation_metrics.f1:.4f} | {row.mean_ontology_conformance:.4f} | "
                f"{row.mean_object_grounding_rate:.4f} | "
                f"{row.mean_filtered_relation_count:.4f} | {row.mean_node_overlap:.4f} | "
                f"{row.mean_relation_agreement:.4f} | {row.mean_graph_overlap:.4f} |"
            )

        lines.extend(["", "## Error Analysis", ""])
        if not self.error_observations:
            lines.append("- No missing, extra, or ungrounded graph facts were found.")
        else:
            for observation in self.error_observations:
                lines.append(
                    "- "
                    f"`{observation.condition}` / `{observation.scene_id}` / "
                    f"run {observation.run_number}: "
                    f"{observation.error_type} - {observation.detail}"
                )
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Generate aggregate-only CSV for charts and spreadsheet tools."""

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "condition",
                "run_count",
                "node_precision",
                "node_recall",
                "node_f1",
                "relation_precision",
                "relation_recall",
                "relation_f1",
                "mean_ontology_conformance",
                "mean_object_grounding_rate",
                "mean_filtered_relation_count",
                "mean_node_overlap",
                "mean_relation_agreement",
                "mean_graph_overlap",
            ]
        )
        for row in self.condition_rows:
            writer.writerow(
                [
                    row.condition,
                    row.run_count,
                    row.node_metrics.precision,
                    row.node_metrics.recall,
                    row.node_metrics.f1,
                    row.relation_metrics.precision,
                    row.relation_metrics.recall,
                    row.relation_metrics.f1,
                    row.mean_ontology_conformance,
                    row.mean_object_grounding_rate,
                    row.mean_filtered_relation_count,
                    row.mean_node_overlap,
                    row.mean_relation_agreement,
                    row.mean_graph_overlap,
                ]
            )
        return buffer.getvalue()

    def save_markdown(self, output_path: str | Path) -> Path:
        """Save the human-readable report as Markdown."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown() + "\n", encoding="utf-8")
        return path

    def save_csv(self, output_path: str | Path) -> Path:
        """Save the aggregate table as CSV."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_csv(), encoding="utf-8")
        return path


def build_comparison_report(
    batch_result: ExperimentBatchResult,
    dataset: BenchmarkDataset = MVP_BENCHMARK,
    stability_report: StabilityReport | None = None,
) -> ExperimentComparisonReport:
    """Evaluate every run against Gold and build a raw/unified report."""

    stability_report = stability_report or evaluate_stability(batch_result)
    run_results: list[RunQualityResult] = []
    observations: list[ErrorObservation] = []

    for run in batch_result.runs:
        scene = dataset.get_scene(run.scene_id)
        gold_graph = build_property_graph(scene.gold_extraction)
        run_quality, run_observations = _evaluate_run_against_gold(run, gold_graph)
        run_results.append(run_quality)
        observations.extend(run_observations)

    conditions = sorted({run.condition for run in batch_result.runs})
    rows = [
        _build_condition_row(condition, run_results, stability_report)
        for condition in conditions
    ]
    return ExperimentComparisonReport(
        dataset_name=batch_result.dataset_name,
        run_results=run_results,
        condition_rows=rows,
        error_observations=observations,
    )


def _evaluate_run_against_gold(
    run: ExtractionRunRecord,
    gold_graph: PropertyGraph,
) -> tuple[RunQualityResult, list[ErrorObservation]]:
    """Compare one run with Gold and collect error explanations."""

    gold_nodes = _node_facts(gold_graph)
    extracted_nodes = _node_facts(run.graph)
    gold_relations = _relation_facts(gold_graph)
    extracted_relations = _relation_facts(run.graph)

    quality = RunQualityResult(
        scene_id=run.scene_id,
        condition=run.condition,
        run_number=run.run_number,
        node_metrics=_facts_to_prf1("nodes", gold_nodes, extracted_nodes),
        relation_metrics=_facts_to_prf1(
            "relations", gold_relations, extracted_relations
        ),
        ontology_conformance=run.validation.ontology_conformance,
        object_grounding_rate=run.validation.object_grounding_rate,
        filtered_relation_count=run.validation.filtered_relation_count,
    )
    observations: list[ErrorObservation] = []

    if run.execution_error:
        observations.append(_observation(run, "Execution failure", run.execution_error))
    for fact in sorted(gold_nodes - extracted_nodes):
        observations.append(_observation(run, "Missing node", fact))
    for fact in sorted(extracted_nodes - gold_nodes):
        observations.append(_observation(run, "Extra node", fact))
    for fact in sorted(gold_relations - extracted_relations):
        observations.append(_observation(run, "Missing relation", fact))
    for fact in sorted(extracted_relations - gold_relations):
        observations.append(_observation(run, "Extra relation", fact))
    for issue in run.validation.issues:
        observations.append(
            _observation(run, f"Validation issue:{issue.kind}", issue.message)
        )

    return quality, observations


def _build_condition_row(
    condition: InputCondition,
    run_results: list[RunQualityResult],
    stability_report: StabilityReport,
) -> ConditionComparisonRow:
    """Aggregate multi-scene repeated-run metrics into one condition row."""

    selected = [result for result in run_results if result.condition == condition]
    stability_selected = [
        summary
        for summary in stability_report.summaries
        if summary.condition == condition
    ]
    return ConditionComparisonRow(
        condition=condition,
        run_count=len(selected),
        node_metrics=aggregate_prf1(
            [result.node_metrics for result in selected],
            category=f"{condition}_nodes",
        ),
        relation_metrics=aggregate_prf1(
            [result.relation_metrics for result in selected],
            category=f"{condition}_relations",
        ),
        mean_ontology_conformance=_mean(
            [result.ontology_conformance for result in selected]
        ),
        mean_object_grounding_rate=_mean(
            [result.object_grounding_rate for result in selected]
        ),
        mean_filtered_relation_count=_mean(
            [float(result.filtered_relation_count) for result in selected]
        ),
        mean_node_overlap=_mean(
            [summary.mean_node_overlap for summary in stability_selected]
        ),
        mean_relation_agreement=_mean(
            [summary.mean_relation_agreement for summary in stability_selected]
        ),
        mean_graph_overlap=_mean(
            [summary.mean_graph_overlap for summary in stability_selected]
        ),
    )


def _facts_to_prf1(
    category: str, expected: set[str], extracted: set[str]
) -> PRF1:
    """Convert Gold/prediction set differences into P/R/F1."""

    return compute_prf1(
        category=category,
        true_positives=len(expected & extracted),
        false_positives=len(extracted - expected),
        false_negatives=len(expected - extracted),
    )


def _node_facts(graph: PropertyGraph) -> set[str]:
    """Represent a node fact by label and normalized name."""

    return {
        f"{node.label}: {normalize_name(node.name)}"
        for node in graph.nodes.values()
    }


def _relation_facts(graph: PropertyGraph) -> set[str]:
    """Represent a relation fact as a readable triple."""

    facts: set[str] = set()
    for edge in graph.edges:
        source = graph.node(edge.source)
        target = graph.node(edge.target)
        facts.add(
            f"{normalize_name(source.name)} --{edge.type}--> "
            f"{normalize_name(target.name)}"
        )
    return facts


def _observation(
    run: ExtractionRunRecord, error_type: str, detail: str
) -> ErrorObservation:
    """Attach traceable scene, condition, and run information to a difference."""

    return ErrorObservation(
        scene_id=run.scene_id,
        condition=run.condition,
        run_number=run.run_number,
        error_type=error_type,
        detail=detail,
    )


def _mean(values: list[float]) -> float:
    """Return a report mean, or zero when no values are available."""

    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
