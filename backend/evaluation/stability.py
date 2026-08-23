"""Stability metrics for repeated extraction results.

The module reads batch results without calling an LLM and reports node overlap,
relation agreement, graph overlap, and repeated-run metric variation.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt

from pydantic import BaseModel, Field

from backend.datasets.benchmark import InputCondition
from backend.graph.property_graph import PropertyGraph, normalize_name
from backend.pipeline.experiment_runner import (
    ExperimentBatchResult,
    ExtractionRunRecord,
)


class PairwiseGraphAgreement(BaseModel):
    """Pairwise graph agreement for one scene and condition."""

    first_run_number: int
    second_run_number: int
    node_overlap: float
    relation_agreement: float
    graph_overlap: float


class MetricVariation(BaseModel):
    """Mean, variance, and standard deviation of a repeated metric."""

    metric_name: str
    values: list[float]
    mean: float
    variance: float
    standard_deviation: float


class StabilitySummary(BaseModel):
    """Stability summary for one scene and input condition."""

    scene_id: str
    condition: InputCondition
    run_count: int
    pair_count: int
    mean_node_overlap: float
    mean_relation_agreement: float
    mean_graph_overlap: float
    pairwise_scores: list[PairwiseGraphAgreement] = Field(default_factory=list)
    variations: dict[str, MetricVariation] = Field(default_factory=dict)


class StabilityReport(BaseModel):
    """Stability report for an entire benchmark."""

    dataset_name: str
    summaries: list[StabilitySummary] = Field(default_factory=list)

    def summary_for(
        self, scene_id: str, condition: InputCondition
    ) -> StabilitySummary:
        """Return one stability summary by scene ID and input condition."""

        for summary in self.summaries:
            if summary.scene_id == scene_id and summary.condition == condition:
                return summary
        raise KeyError(
            f"No stability result found for scene {scene_id!r} and condition {condition!r}"
        )


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Compute set overlap; two empty sets are treated as identical."""

    union = left | right
    if not union:
        return 1.0
    return round(len(left & right) / len(union), 4)


def compare_graphs(
    first: PropertyGraph,
    second: PropertyGraph,
    first_run_number: int,
    second_run_number: int,
) -> PairwiseGraphAgreement:
    """Compare graph nodes and relation triples from two runs."""

    first_nodes = _node_signatures(first)
    second_nodes = _node_signatures(second)
    first_relations = _relation_signatures(first)
    second_relations = _relation_signatures(second)

    # 整体图重合度将节点与边都视为可比较的知识事实。
    first_graph_facts = {f"node::{item}" for item in first_nodes} | {
        f"relation::{item}" for item in first_relations
    }
    second_graph_facts = {f"node::{item}" for item in second_nodes} | {
        f"relation::{item}" for item in second_relations
    }

    return PairwiseGraphAgreement(
        first_run_number=first_run_number,
        second_run_number=second_run_number,
        node_overlap=jaccard_similarity(first_nodes, second_nodes),
        relation_agreement=jaccard_similarity(first_relations, second_relations),
        graph_overlap=jaccard_similarity(first_graph_facts, second_graph_facts),
    )


def evaluate_stability(batch_result: ExperimentBatchResult) -> StabilityReport:
    """Aggregate repeated-run stability by scene and input condition."""

    summaries: list[StabilitySummary] = []
    scene_conditions = {
        (run.scene_id, run.condition) for run in batch_result.runs
    }

    for scene_id, condition in sorted(scene_conditions):
        runs = sorted(
            batch_result.runs_for(scene_id, condition),
            key=lambda run: run.run_number,
        )
        summaries.append(_summarize_runs(scene_id, condition, runs))

    return StabilityReport(
        dataset_name=batch_result.dataset_name,
        summaries=summaries,
    )


def _summarize_runs(
    scene_id: str,
    condition: InputCondition,
    runs: list[ExtractionRunRecord],
) -> StabilitySummary:
    """Summarize repeated runs for one scene and input representation."""

    pairwise_scores = [
        compare_graphs(
            first.graph,
            second.graph,
            first_run_number=first.run_number,
            second_run_number=second.run_number,
        )
        for first, second in combinations(runs, 2)
    ]

    # 只有一次运行时没有可比较的差异，重合度设为 1，同时 pair_count 明确为 0。
    mean_node_overlap = _mean_or_perfect(
        [score.node_overlap for score in pairwise_scores]
    )
    mean_relation_agreement = _mean_or_perfect(
        [score.relation_agreement for score in pairwise_scores]
    )
    mean_graph_overlap = _mean_or_perfect(
        [score.graph_overlap for score in pairwise_scores]
    )

    variations = {
        "node_count": _variation(
            "node_count", [float(len(run.graph.nodes)) for run in runs]
        ),
        "relation_count": _variation(
            "relation_count", [float(len(run.graph.edges)) for run in runs]
        ),
        "ontology_conformance": _variation(
            "ontology_conformance",
            [run.validation.ontology_conformance for run in runs],
        ),
        "object_grounding_rate": _variation(
            "object_grounding_rate",
            [run.validation.object_grounding_rate for run in runs],
        ),
        "filtered_relation_count": _variation(
            "filtered_relation_count",
            [float(run.validation.filtered_relation_count) for run in runs],
        ),
    }

    return StabilitySummary(
        scene_id=scene_id,
        condition=condition,
        run_count=len(runs),
        pair_count=len(pairwise_scores),
        mean_node_overlap=mean_node_overlap,
        mean_relation_agreement=mean_relation_agreement,
        mean_graph_overlap=mean_graph_overlap,
        pairwise_scores=pairwise_scores,
        variations=variations,
    )


def _node_signatures(graph: PropertyGraph) -> set[str]:
    """Convert nodes to comparable signatures independent of internal IDs."""

    return {
        f"{node.label}|{normalize_name(node.name)}"
        for node in graph.nodes.values()
    }


def _relation_signatures(graph: PropertyGraph) -> set[str]:
    """Convert edges into source-type-target signatures."""

    signatures: set[str] = set()
    for edge in graph.edges:
        source = graph.node(edge.source)
        target = graph.node(edge.target)
        signatures.add(
            "|".join(
                [
                    source.label,
                    normalize_name(source.name),
                    edge.type,
                    target.label,
                    normalize_name(target.name),
                ]
            )
        )
    return signatures


def _variation(metric_name: str, values: list[float]) -> MetricVariation:
    """Compute population variance; lower values indicate greater stability."""

    if not values:
        return MetricVariation(
            metric_name=metric_name,
            values=[],
            mean=0.0,
            variance=0.0,
            standard_deviation=0.0,
        )
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return MetricVariation(
        metric_name=metric_name,
        values=values,
        mean=round(mean, 4),
        variance=round(variance, 4),
        standard_deviation=round(sqrt(variance), 4),
    )


def _mean_or_perfect(values: list[float]) -> float:
    """Return mean agreement, or the 1.0 baseline when no pairs exist."""

    if not values:
        return 1.0
    return round(sum(values) / len(values), 4)
