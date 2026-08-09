"""重复抽取结果的稳定性评价指标。

本模块回答实验中的稳定性问题：

    同一个场景、同一种输入表示重复送给模型时，
    模型生成的节点和关系是否保持一致？

它读取 experiment_runner 产生的批量运行结果，不再调用 LLM。
指标设计保持简单、可解释，便于在项目报告中说明：

- Node overlap：两次运行抽到的节点集合重合程度。
- Relation agreement：两次运行抽到的关系三元组重合程度。
- Graph overlap：节点与关系合并后的整体图重合程度。
- Repeated-run variation：多次运行中图规模和校验指标的均值与方差。
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
    """同一场景同一条件下，两次运行之间的图一致性比较结果。"""

    first_run_number: int
    second_run_number: int
    node_overlap: float
    relation_agreement: float
    graph_overlap: float


class MetricVariation(BaseModel):
    """某个数值指标在重复运行中的均值、方差和标准差。"""

    metric_name: str
    values: list[float]
    mean: float
    variance: float
    standard_deviation: float


class StabilitySummary(BaseModel):
    """一个场景在一种输入条件下的稳定性汇总。"""

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
    """整个 benchmark 的稳定性报告。"""

    dataset_name: str
    summaries: list[StabilitySummary] = Field(default_factory=list)

    def summary_for(
        self, scene_id: str, condition: InputCondition
    ) -> StabilitySummary:
        """按场景编号与输入类型取回一条稳定性汇总。"""

        for summary in self.summaries:
            if summary.scene_id == scene_id and summary.condition == condition:
                return summary
        raise KeyError(f"未找到场景 {scene_id!r} 的 {condition!r} 稳定性结果")


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """计算两个事实集合的重合度；两者都为空时视为完全一致。"""

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
    """比较两次抽取生成的 graph nodes 与 relation triples。"""

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
    """按场景与输入条件汇总批量重复实验的稳定性指标。"""

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
    """生成单个场景和单种输入形式的重复运行汇总。"""

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
    """把节点转成与内部 id 无关的可比较事实表示。"""

    return {
        f"{node.label}|{normalize_name(node.name)}"
        for node in graph.nodes.values()
    }


def _relation_signatures(graph: PropertyGraph) -> set[str]:
    """把边转成 source-type-target 三元组，比较关系是否重复出现。"""

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
    """计算总体方差；实验报告中数值越低表示重复运行越稳定。"""

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
    """汇总成平均一致性；没有比较对时返回基线值 1.0。"""

    if not values:
        return 1.0
    return round(sum(values) / len(values), 4)
