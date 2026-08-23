"""Tests for strict per-type and node-aligned relation diagnostics."""

from backend.evaluation.relation_alignment import (
    align_extracted_nodes,
    evaluate_relation_types,
)
from backend.graph.property_graph import PropertyGraph


def _graph(action_name: str, object_name: str, relation_type: str) -> PropertyGraph:
    graph = PropertyGraph()
    action = graph.add_node("Action", action_name)
    target = graph.add_node("Object", object_name)
    graph.add_edge(relation_type, action, target)
    return graph


def test_aligned_relation_metric_matches_longer_grounded_names() -> None:
    gold = _graph("align steel plate", "steel plate", "ACTS_ON")
    predicted = _graph(
        "align the steel plate on table",
        "steel plate",
        "ACTS_ON",
    )

    strict = evaluate_relation_types(gold, predicted, align_nodes=False)
    aligned = evaluate_relation_types(gold, predicted, align_nodes=True)

    assert strict["ACTS_ON"].f1 == 0.0
    assert aligned["ACTS_ON"].f1 == 1.0


def test_alignment_never_changes_relation_type() -> None:
    gold = _graph("align steel plate", "steel plate", "ACTS_ON")
    predicted = _graph("align steel plate", "steel plate", "USES_TOOL")

    metrics = evaluate_relation_types(
        gold,
        predicted,
        relation_types={"ACTS_ON", "USES_TOOL"},
        align_nodes=True,
    )

    assert metrics["ACTS_ON"].false_negatives == 1
    assert metrics["USES_TOOL"].false_positives == 1


def test_strict_relation_metric_does_not_compare_internal_ids() -> None:
    gold = PropertyGraph()
    gold_action = gold.add_node(
        "Action",
        "tighten screw",
        {"entity_id": "gold_action"},
    )
    gold_object = gold.add_node(
        "Object",
        "screw",
        {"entity_id": "gold_object"},
    )
    gold.add_edge("ACTS_ON", gold_action, gold_object)

    predicted = PropertyGraph()
    predicted_action = predicted.add_node(
        "Action",
        "tighten screw",
        {"entity_id": "predicted_action"},
    )
    predicted_object = predicted.add_node(
        "Object",
        "screw",
        {"entity_id": "predicted_object"},
    )
    predicted.add_edge("ACTS_ON", predicted_action, predicted_object)

    strict = evaluate_relation_types(gold, predicted, align_nodes=False)

    assert gold_action.id != predicted_action.id
    assert strict["ACTS_ON"].f1 == 1.0


def test_node_alignment_is_one_to_one_within_same_label() -> None:
    gold = PropertyGraph()
    gold.add_node("Action", "tighten first screw")
    gold.add_node("Action", "tighten second screw")
    predicted = PropertyGraph()
    predicted.add_node("Action", "tighten screw")

    mapping = align_extracted_nodes(gold, predicted)

    assert len(mapping) == 1


def test_duplicate_exact_names_do_not_downgrade_strict_relation_hit() -> None:
    gold = PropertyGraph()
    gold_action = gold.add_node(
        "Action",
        "remove plates",
        {"entity_id": "gold_action", "evidence_text": "remove these plates"},
    )
    gold_object = gold.add_node("Object", "plates", {"entity_id": "gold_object"})
    gold.add_edge("ACTS_ON", gold_action, gold_object)

    predicted = PropertyGraph()
    correct_action = predicted.add_node(
        "Action",
        "remove plates",
        {"entity_id": "A2", "evidence_text": "remove these plates"},
    )
    predicted.add_node(
        "Action",
        "remove plates",
        {"entity_id": "A3", "evidence_text": "remove this one"},
    )
    predicted_object = predicted.add_node(
        "Object",
        "plates",
        {"entity_id": "O2"},
    )
    predicted.add_edge("ACTS_ON", correct_action, predicted_object)

    strict = evaluate_relation_types(gold, predicted, align_nodes=False)
    aligned = evaluate_relation_types(gold, predicted, align_nodes=True)

    assert strict["ACTS_ON"].f1 == 1.0
    assert aligned["ACTS_ON"].f1 == 1.0
