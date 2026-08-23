"""Relation-type evaluation with optional evidence-safe node-name alignment.

Strict metrics remain unchanged. This module adds aligned diagnostics
to separate surface-name mismatch from genuine relation errors.
"""

from __future__ import annotations

import re

from backend.evaluation.metrics import PRF1, compute_prf1
from backend.graph.property_graph import GraphNode, PropertyGraph, normalize_name


def evaluate_relation_types(
    expected: PropertyGraph,
    extracted: PropertyGraph,
    *,
    relation_types: set[str] | None = None,
    align_nodes: bool = False,
    alignment_threshold: float = 0.6,
) -> dict[str, PRF1]:
    """Return one strict or node-aligned P/R/F1 result per relation type."""

    selected_types = relation_types or {
        edge.type for edge in [*expected.edges, *extracted.edges]
    }
    mapping = (
        align_extracted_nodes(
            expected,
            extracted,
            threshold=alignment_threshold,
        )
        if align_nodes
        else {}
    )
    results: dict[str, PRF1] = {}
    for relation_type in sorted(selected_types):
        if align_nodes:
            expected_facts = {
                (edge.source, edge.target)
                for edge in expected.edges
                if edge.type == relation_type
            }
            extracted_facts = {
                (
                    mapping.get(edge.source, f"extracted::{edge.source}"),
                    mapping.get(edge.target, f"extracted::{edge.target}"),
                )
                for edge in extracted.edges
                if edge.type == relation_type
            }
            # 中文：对齐诊断不能把严格名称已经命中的关系降级。重复同名节点可能
            # 让一对一映射选择另一个实例，因此严格命中数作为 TP 下限。
            # English: Alignment must never downgrade an already strict name hit;
            # duplicate names can otherwise select a different one-to-one instance.
            strict_true_positives = len(
                _named_relation_facts(expected, relation_type)
                & _named_relation_facts(extracted, relation_type)
            )
            true_positives = max(
                len(expected_facts & extracted_facts),
                strict_true_positives,
            )
            false_positives = max(len(extracted_facts) - true_positives, 0)
            false_negatives = max(len(expected_facts) - true_positives, 0)
        else:
            # EN: Gold and prediction use independent internal ID namespaces.
            # Strict comparison therefore uses normalized endpoint names, just
            # like the main relation metric, rather than incomparable IDs.
            # ZH: Gold 与预测的内部 ID 命名空间彼此独立；严格比较应使用标准化
            # 端点名称（与主关系指标一致），不能直接比较天然不同的 ID。
            expected_facts = _named_relation_facts(expected, relation_type)
            extracted_facts = _named_relation_facts(extracted, relation_type)
            true_positives = len(expected_facts & extracted_facts)
            false_positives = len(extracted_facts - expected_facts)
            false_negatives = len(expected_facts - extracted_facts)
        results[relation_type] = compute_prf1(
            category=f"relation:{relation_type}",
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )
    return results


def _named_relation_facts(
    graph: PropertyGraph,
    relation_type: str,
) -> set[tuple[str, str]]:
    """Return strict normalized-name endpoint pairs for one relation type."""

    return {
        (
            normalize_name(graph.node(edge.source).name),
            normalize_name(graph.node(edge.target).name),
        )
        for edge in graph.edges
        if edge.type == relation_type
    }


def align_extracted_nodes(
    expected: PropertyGraph,
    extracted: PropertyGraph,
    *,
    threshold: float = 0.6,
) -> dict[str, str]:
    """Align nodes with exact-name priority and maximum-weight matching.

    Exact names receive priority, duplicate exact names use relation and
    evidence context, and remaining nodes use global maximum-weight assignment.
    """

    mapping: dict[str, str] = {}
    labels = sorted(
        {node.label for node in extracted.nodes.values()}
        & {node.label for node in expected.nodes.values()}
    )
    for label in labels:
        extracted_nodes = sorted(
            (node for node in extracted.nodes.values() if node.label == label),
            key=lambda node: node.id,
        )
        expected_nodes = sorted(
            (node for node in expected.nodes.values() if node.label == label),
            key=lambda node: node.id,
        )
        weights: list[list[float]] = []
        for extracted_node in extracted_nodes:
            row: list[float] = []
            for expected_node in expected_nodes:
                name_score = node_name_alignment_score(
                    extracted_node,
                    expected_node,
                )
                if name_score < threshold:
                    row.append(0.0)
                    continue
                exact_bonus = (
                    10_000.0
                    if normalize_name(extracted_node.name)
                    == normalize_name(expected_node.name)
                    else 0.0
                )
                row.append(
                    exact_bonus
                    + name_score * 1_000.0
                    + _relation_neighbour_score(
                        extracted,
                        extracted_node,
                        expected,
                        expected_node,
                    )
                    * 100.0
                    + _node_evidence_score(extracted_node, expected_node) * 10.0
                )
            weights.append(row)

        for extracted_index, expected_index in _maximum_weight_assignment(weights):
            if weights[extracted_index][expected_index] <= 0:
                continue
            mapping[extracted_nodes[extracted_index].id] = expected_nodes[
                expected_index
            ].id
    return mapping


def _maximum_weight_assignment(
    weights: list[list[float]],
) -> list[tuple[int, int]]:
    """Return a deterministic rectangular Hungarian maximum-weight assignment."""

    if not weights or not weights[0]:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    size = max(row_count, column_count)
    maximum = max(max(row) for row in weights)
    cost = [
        [
            maximum
            - (
                weights[row][column]
                if row < row_count and column < column_count
                else 0.0
            )
            for column in range(size)
        ]
        for row in range(size)
    ]

    # Classic O(n^3) Hungarian algorithm for a square minimum-cost matrix.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment: list[tuple[int, int]] = []
    for column in range(1, size + 1):
        row = p[column]
        if 1 <= row <= row_count and column <= column_count:
            assignment.append((row - 1, column - 1))
    return sorted(assignment)


def _relation_neighbour_score(
    extracted: PropertyGraph,
    extracted_node: GraphNode,
    expected: PropertyGraph,
    expected_node: GraphNode,
) -> float:
    left = _incident_relation_signatures(extracted, extracted_node.id)
    right = _incident_relation_signatures(expected, expected_node.id)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _incident_relation_signatures(
    graph: PropertyGraph,
    node_id: str,
) -> set[tuple[str, str, str]]:
    signatures: set[tuple[str, str, str]] = set()
    for edge in graph.edges:
        if edge.source == node_id:
            signatures.add(
                ("out", edge.type, normalize_name(graph.node(edge.target).name))
            )
        if edge.target == node_id:
            signatures.add(
                ("in", edge.type, normalize_name(graph.node(edge.source).name))
            )
    return signatures


def _node_evidence_score(left: GraphNode, right: GraphNode) -> float:
    left_evidence = normalize_name(
        str(left.properties.get("evidence_text") or left.properties.get("source_text") or "")
    )
    right_evidence = normalize_name(
        str(right.properties.get("evidence_text") or right.properties.get("source_text") or "")
    )
    if not left_evidence or not right_evidence:
        return 0.0
    if left_evidence == right_evidence:
        return 1.0
    left_tokens = set(_name_tokens(left_evidence))
    right_tokens = set(_name_tokens(right_evidence))
    return (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )


def node_name_alignment_score(left: GraphNode, right: GraphNode) -> float:
    """Score two same-type names without changing either graph."""

    left_name = normalize_name(left.name)
    right_name = normalize_name(right.name)
    if not left_name or not right_name:
        return 0.0
    if left_name == right_name:
        return 1.0

    left_tokens = _name_tokens(left_name)
    right_tokens = _name_tokens(right_name)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / min(
        len(left_tokens),
        len(right_tokens),
    )
    if left_name in right_name or right_name in left_name:
        overlap = max(overlap, 0.9)
    return round(overlap, 4)


def _name_tokens(value: str) -> set[str]:
    stopwords = {"a", "an", "and", "in", "of", "on", "the", "to", "with"}
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in stopwords
    }


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
