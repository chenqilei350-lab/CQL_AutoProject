"""Raw 与 Unified 对比实验的结果表和错误分析输出模块。

本模块位于实验链的最后汇总阶段：

    数据集 -> 重复抽取 -> 稳定性评价 -> 结果表与错误分析

它把模型生成的 graph 与人工 gold graph 进行对比，输出：

- 节点检测 Precision / Recall / F1；
- 关系检测 Precision / Recall / F1；
- schema 与原文证据校验指标；
- 重复运行的 graph stability 指标；
- 便于人工阅读的典型错误记录。

这里采用图事实比较，而不是只评价一种实体类型，因为本项目同时关注
Action、Tool、Object 等多类节点，以及 USES_TOOL、CAUSES 等多种关系。
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
    """一次模型运行相对于人工 gold graph 的质量结果。"""

    scene_id: str
    condition: InputCondition
    run_number: int
    node_metrics: PRF1
    relation_metrics: PRF1
    ontology_conformance: float
    object_grounding_rate: float
    filtered_relation_count: int


class ErrorObservation(BaseModel):
    """报告中可读的一条错误观察，用于后续 error analysis。"""

    scene_id: str
    condition: InputCondition
    run_number: int
    error_type: str
    detail: str


class ConditionComparisonRow(BaseModel):
    """Raw 或 Unified 条件的总体结果行。"""

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
    """可导出的 Raw vs Unified 对比报告。"""

    dataset_name: str
    run_results: list[RunQualityResult] = Field(default_factory=list)
    condition_rows: list[ConditionComparisonRow] = Field(default_factory=list)
    error_observations: list[ErrorObservation] = Field(default_factory=list)

    def row_for(self, condition: InputCondition) -> ConditionComparisonRow:
        """返回指定输入条件的汇总结果行。"""

        for row in self.condition_rows:
            if row.condition == condition:
                return row
        raise KeyError(f"没有找到输入条件 {condition!r} 的结果表行")

    def to_markdown(self) -> str:
        """生成可直接放入项目文档或展示材料的 Markdown 表格。"""

        lines = [
            "# Raw vs Unified 实验结果",
            "",
            "## 汇总对比表",
            "",
            "| 输入 | 运行数 | Node F1 | Relation F1 | Schema 合规率 | 对象证据率 | 需过滤关系均值 | Node overlap | Relation agreement | Graph overlap |",
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

        lines.extend(["", "## 错误分析", ""])
        if not self.error_observations:
            lines.append("- 当前运行未发现缺失、额外或无原文依据的图事实。")
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
        """生成仅包含总体指标的 CSV，便于制图或表格软件处理。"""

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
        """将可读报告保存为 Markdown 文件。"""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown() + "\n", encoding="utf-8")
        return path

    def save_csv(self, output_path: str | Path) -> Path:
        """将汇总表保存为 CSV 文件。"""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_csv(), encoding="utf-8")
        return path


def build_comparison_report(
    batch_result: ExperimentBatchResult,
    dataset: BenchmarkDataset = MVP_BENCHMARK,
    stability_report: StabilityReport | None = None,
) -> ExperimentComparisonReport:
    """对全部运行结果做 gold 评价，并生成 Raw/Unified 汇总报告。"""

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
    """计算单次运行与标准图之间的差异，并收集错误解释。"""

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
        observations.append(_observation(run, "运行失败", run.execution_error))
    for fact in sorted(gold_nodes - extracted_nodes):
        observations.append(_observation(run, "缺失节点", fact))
    for fact in sorted(extracted_nodes - gold_nodes):
        observations.append(_observation(run, "额外节点", fact))
    for fact in sorted(gold_relations - extracted_relations):
        observations.append(_observation(run, "缺失关系", fact))
    for fact in sorted(extracted_relations - gold_relations):
        observations.append(_observation(run, "额外关系", fact))
    for issue in run.validation.issues:
        observations.append(
            _observation(run, f"校验问题:{issue.kind}", issue.message)
        )

    return quality, observations


def _build_condition_row(
    condition: InputCondition,
    run_results: list[RunQualityResult],
    stability_report: StabilityReport,
) -> ConditionComparisonRow:
    """将一种输入形式的多场景、多次运行指标汇总成表格一行。"""

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
    """把 gold 与抽取事实集合的交集/差集转换为 P/R/F1。"""

    return compute_prf1(
        category=category,
        true_positives=len(expected & extracted),
        false_positives=len(extracted - expected),
        false_negatives=len(expected - extracted),
    )


def _node_facts(graph: PropertyGraph) -> set[str]:
    """用节点类型和标准化名称表示一个图节点事实。"""

    return {
        f"{node.label}: {normalize_name(node.name)}"
        for node in graph.nodes.values()
    }


def _relation_facts(graph: PropertyGraph) -> set[str]:
    """用可读三元组表示一个图关系事实。"""

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
    """为一条差异添加可追踪的场景、条件和运行次数信息。"""

    return ErrorObservation(
        scene_id=run.scene_id,
        condition=run.condition,
        run_number=run.run_number,
        error_type=error_type,
        detail=detail,
    )


def _mean(values: list[float]) -> float:
    """计算报告中的平均值，没有输入时返回 0。"""

    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
