from pathlib import Path
import sys
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hospital_kg_postprocessing import (  # noqa: E402
    load_hospital_postprocess_context,
    normalize_node,
    normalize_relation,
    postprocess_hospital_kg,
)


def locate_gold_case_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "gold_case_admission_29366372",
        PROJECT_ROOT / "data" / "hospital_gold" / "gold_case_admission_29366372",
        PROJECT_ROOT.parent / "data" / "hospital_gold" / "gold_case_admission_29366372",
        PROJECT_ROOT.parent / "gold_case_admission_29366372",
    ]
    for candidate in candidates:
        if (candidate / "gold_case_schema.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate gold_case_admission_29366372. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


try:
    GOLD_DIR = locate_gold_case_dir()
except FileNotFoundError as error:
    pytest.skip(str(error), allow_module_level=True)


def test_normalizes_bare_ids_and_icd_order() -> None:
    context = load_hospital_postprocess_context(GOLD_DIR)

    assert normalize_node("Admission", "29366372", context) == (
        "Admission",
        "admission:29366372",
    )
    assert normalize_node("LabEvent", "221530", context) == (
        "LabEvent",
        "labevent:221530",
    )
    assert normalize_node("LabItem", "50808", context) == (
        "LabItem",
        "labitem:50808",
    )
    assert normalize_node("Diagnosis", "I25110:10", context) == (
        "Diagnosis",
        "diagnosis:10:I25110",
    )
    assert normalize_node("Diagnosis", "10:I25110", context) == (
        "Diagnosis",
        "diagnosis:10:I25110",
    )


def test_normalizes_relation_aliases() -> None:
    context = load_hospital_postprocess_context(GOLD_DIR)

    assert normalize_relation("HAS_DIAGNOSIS", context) == "has_diagnosis"
    assert normalize_relation("has_medication", context) == "prescribed_medication"
    assert normalize_relation("HAS_LAB_EVENT", context) == "has_lab_event"
    assert normalize_relation("treats", context) is None


def test_postprocess_corrects_reversed_edges_and_merges_duplicates() -> None:
    context = load_hospital_postprocess_context(GOLD_DIR)
    nodes = {
        ("Admission", "29366372"),
        ("Diagnosis", "I25110:10"),
    }
    edges = {
        ("diagnosis:10:I25110", "HAS_DIAGNOSIS", "29366372"),
        ("diagnosis:10:I25110", "HAS_DIAGNOSIS", "29366372"),
        ("I25110:10", "has_diagnosis", "admission:29366372"),
    }

    result = postprocess_hospital_kg(nodes, edges, context)

    assert ("Admission", "admission:29366372") in result.nodes
    assert ("Diagnosis", "diagnosis:10:I25110") in result.nodes
    assert result.edges == {
        ("admission:29366372", "has_diagnosis", "diagnosis:10:I25110")
    }
    assert result.stats.auto_corrected_edges >= 1
    assert result.stats.wrong_direction_count >= 1
    assert result.stats.merged_duplicate_edges >= 1
    assert result.stats.hallucinated_edges == 0


def test_postprocess_detects_hallucinated_node_and_edge() -> None:
    context = load_hospital_postprocess_context(GOLD_DIR)
    nodes = {
        ("Admission", "29366372"),
        ("Diagnosis", "10:FAKE"),
    }
    edges = {
        ("admission:29366372", "has_diagnosis", "diagnosis:10:FAKE"),
        ("admission:29366372", "treats", "diagnosis:10:I25110"),
    }

    result = postprocess_hospital_kg(nodes, edges, context)

    assert ("Diagnosis", "diagnosis:10:FAKE") in result.nodes
    assert result.stats.hallucinated_nodes >= 1
    assert result.stats.hallucinated_edges >= 1
    assert result.stats.invalid_relation_count == 1
    assert any("unsupported_edge" in issue for issue in result.stats.issues)


def test_experiment_row_can_include_raw_and_normalized_metrics() -> None:
    import run_hospital_gold_pilot_experiments_zh as runner

    context = load_hospital_postprocess_context(GOLD_DIR)
    runner.POSTPROCESS_CONTEXT = context
    prediction = runner.GraphPrediction(
        nodes={
            ("Admission", "29366372"),
            ("Diagnosis", "I25110:10"),
        },
        edges={
            ("diagnosis:10:I25110", "HAS_DIAGNOSIS", "29366372"),
        },
        raw_json={},
    )
    gold_nodes = {
        ("Admission", "admission:29366372"),
        ("Diagnosis", "diagnosis:10:I25110"),
    }
    gold_edges = {
        ("admission:29366372", "has_diagnosis", "diagnosis:10:I25110")
    }
    raw_node_metric = runner.compute_prf(prediction.nodes, gold_nodes)
    raw_edge_metric = runner.compute_prf(prediction.edges, gold_edges)
    row = runner.metric_row_from_metrics(
        experiment="test",
        condition="raw_then_normalized",
        model="fake",
        seconds=0.0,
        error="",
        node_metric=raw_node_metric,
        edge_metric=raw_edge_metric,
    )

    runner.add_postprocess_metrics(row, prediction, gold_nodes, gold_edges)

    assert row["node_f1"] == 0.0
    assert row["edge_f1"] == 0.0
    assert row["raw_node_f1"] == 0.0
    assert row["raw_edge_f1"] == 0.0
    assert row["normalized_node_f1"] == 1.0
    assert row["normalized_edge_f1"] == 1.0
    assert row["auto_corrected_edges"] == 1
