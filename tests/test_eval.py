"""
Quick smoke test for all evaluation functions.
Run: python -m pytest tests/test_eval.py -v
Or just: python tests/test_eval.py
"""

from backend.schemas.process_knowledge.entities import Tool, Step
from backend.schemas.process_knowledge.relations import StepOrder
from backend.evaluation.matching import compare_values, MatchStrategy
from backend.evaluation.metrics import PRF1, compute_prf1, format_prf1
from backend.evaluation.ontology_validation import validate_egocentric_extraction
from backend.evaluation.experiment import ExperimentConfig
from backend.evaluation.evaluators import (
    evaluate_entity_detection,
    evaluate_attributes,
    evaluate_relation_detection,
    evaluate_extraction_level,
)
from backend.schemas.egocentric_examples import WELDING_SCENE_EXPECTED, WELDING_SCENE_TEXT
from backend.schemas.egocentric_video import (
    Action,
    EgocentricVideoExtraction,
    UsesTool,
)
from backend.schemas.ontology import build_egocentric_ontology, compact_schema_guide


def test_matching_strategies():
    """Test all matching strategies return valid scores."""
    a = "Fronius TPS 400i"
    b = "MIG-Schweißgerät Fronius TPS 400i"

    for strategy in MatchStrategy:
        if strategy == MatchStrategy.EMBEDDING:
            continue  # requires sentence-transformers
        result = compare_values(a, b, strategy=strategy)
        assert 0.0 <= result["score"] <= 1.0, f"{strategy}: score {result['score']} out of range"
        print(f"  {strategy.value:20s} → {result['score']:.3f}")

    # Exact match
    result = compare_values("hello", "hello", strategy=MatchStrategy.EXACT)
    assert result["score"] == 1.0, "Exact match failed"

    print("✅ All matching strategies work.\n")


def test_metrics():
    """Test PRF1 computation and formatting."""
    metrics = compute_prf1(category="TestEntity", true_positives=2, false_positives=1, false_negatives=1)
    assert isinstance(metrics, PRF1)
    assert abs(metrics.precision - 2 / 3) < 0.01
    assert abs(metrics.recall - 2 / 3) < 0.01
    assert metrics.f1 > 0

    # Edge case: all zeros
    metrics_zero = compute_prf1(category="Empty", true_positives=0, false_positives=0, false_negatives=0)
    assert metrics_zero.precision == 0.0
    assert metrics_zero.recall == 0.0

    output = format_prf1(metrics)
    assert "TestEntity" in output

    print(f"  {output}")
    print("✅ Metrics computation works.\n")


def test_experiment_config():
    """Test experiment config creation."""
    config = ExperimentConfig(
        experiment_name="test_run_1",
        model="llama3.1:8b",
        prompt_strategy="default",
        schema_level=1,
        domain="process_knowledge",
        temperature=0.0,
    )
    d = config.model_dump()
    assert d["model"] == "llama3.1:8b"
    assert d["experiment_name"] == "test_run_1"

    print(f"  Config: {d}")
    print("✅ ExperimentConfig works.\n")


def test_evaluate_entity_detection():
    """Test entity detection evaluation."""
    expected = [
        Tool(name="Fronius TPS 400i", tool_type="Schweißgerät"),
        Tool(name="ER70S-6 Schweißdraht", tool_type="Schweißdraht"),
        Tool(name="Schutzgas Argon/CO2", tool_type="Schutzgas"),
    ]
    extracted = [
        Tool(name="MIG-Schweißgerät Fronius TPS 400i", tool_type="Schweißgerät"),
        Tool(name="ER70S-6 Schweißdraht", tool_type="Schweißdraht"),
    ]

    metrics, log = evaluate_entity_detection(
        expected, extracted,
        match_strategy=MatchStrategy.TOKEN_OVERLAP,
        match_threshold=0.6,
    )

    assert metrics.true_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 0
    assert len(log) == 3  # 2 TP + 1 FN

    print(f"  {format_prf1(metrics)}")
    print(f"  Log entries: {len(log)}")
    print("✅ Entity detection evaluation works.\n")


def test_evaluate_attributes():
    """Test attribute-level evaluation."""
    expected = Tool(name="Fronius TPS 400i", tool_type="Schweißgerät", manufacturer="Fronius")
    extracted = Tool(name="MIG-Schweißgerät Fronius TPS 400i", tool_type="Schweißgerät", manufacturer="Fronius")

    metrics, log = evaluate_attributes(
        expected, extracted,
        match_strategy=MatchStrategy.TOKEN_OVERLAP,
        match_threshold=0.5,
    )

    assert isinstance(metrics, PRF1)
    assert len(log) > 0

    print(f"  {format_prf1(metrics)}")
    for entry in log:
        print(f"    {entry['field']}: expected={entry['expected']} → extracted={entry['extracted']} ({entry['status']})")
    print("✅ Attribute evaluation works.\n")


def test_evaluate_relation_detection():
    """Test relation detection evaluation."""
    step_a = Step(name="Vorbereitung")
    step_b = Step(name="Heften")
    step_c = Step(name="Wurzellage")

    expected = [
        StepOrder(before=step_a, after=step_b),
        StepOrder(before=step_b, after=step_c),
    ]
    extracted = [
        StepOrder(before=step_a, after=step_b),
    ]

    metrics, log = evaluate_relation_detection(
        expected, extracted,
        role_fields=["before", "after"],
        match_strategy=MatchStrategy.TOKEN_OVERLAP,
        match_threshold=0.6,
    )

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 0

    print(f"  {format_prf1(metrics)}")
    print("✅ Relation detection evaluation works.\n")


def test_evaluate_extraction_level():
    """Test full level evaluation."""
    expected_entities = [
        Tool(name="Fronius TPS 400i", tool_type="Schweißgerät"),
    ]
    extracted_entities = [
        Tool(name="Fronius TPS 400i", tool_type="Schweißgerät"),
    ]

    results = evaluate_extraction_level(
        level=1,
        expected_entities=expected_entities,
        extracted_entities=extracted_entities,
    )

    assert results["level"] == 1
    assert "entity_detection" in results
    assert "attribute_evaluations" in results

    print(f"  Level: {results['level']}")
    print(f"  Entity detection: {format_prf1(results['entity_detection'])}")
    print(f"  Attribute evaluations: {len(results['attribute_evaluations'])} pairs")
    print("✅ Full level evaluation works.\n")


def test_egocentric_ontology_introspection():
    """Test Pydantic schemas can be summarized as an executable ontology."""
    ontology = build_egocentric_ontology()

    assert "Action" in ontology.entities
    assert "Tool" in ontology.entities
    assert ontology.relation_by_label("USES_TOOL").domain == "Action"
    assert ontology.relation_by_label("USES_TOOL").range == "Tool"
    assert ontology.entities["Tool"].identity_fields == ["manufacturer", "model", "name"]


def test_schema_prompt_guidance_is_compact_and_informative():
    """Test compact field guidance can be generated from Pydantic schema."""
    guide = compact_schema_guide(EgocentricVideoExtraction)

    assert "actions" in guide
    assert "uses_tool" in guide
    assert len(guide) < 3500


def test_ontology_validation_reports_grounded_extraction():
    """Test ontology conformance and grounding on a gold scene."""
    report = validate_egocentric_extraction(
        WELDING_SCENE_EXPECTED,
        source_text=WELDING_SCENE_TEXT,
    )

    assert report.total_relations > 0
    assert report.invalid_relations == 0
    assert report.ontology_conformance == 1.0
    assert report.subject_grounding_rate > 0.5
    assert report.object_grounding_rate > 0.5


def test_ontology_validation_detects_hallucinated_object():
    """Test ungrounded relation endpoints are counted as hallucinations."""
    extraction = EgocentricVideoExtraction(
        source_text="Hans measures the gap with a caliper.",
        uses_tool=[
            UsesTool(
                action=Action(name="measure gap"),
                tool=Tool(name="laser scanner"),
            )
        ],
    )

    report = validate_egocentric_extraction(extraction)

    assert report.total_relations == 1
    assert report.invalid_relations == 0
    assert report.ontology_conformance == 1.0
    assert report.object_hallucinations == 1
    assert report.object_grounding_rate == 0.0
    assert report.filtered_relation_count >= 1


if __name__ == "__main__":
    print("\n🧪 Testing Evaluation Pipeline\n" + "=" * 40 + "\n")

    test_matching_strategies()
    test_metrics()
    test_experiment_config()
    test_evaluate_entity_detection()
    test_evaluate_attributes()
    test_evaluate_relation_detection()
    test_evaluate_extraction_level()
    test_egocentric_ontology_introspection()
    test_schema_prompt_guidance_is_compact_and_informative()
    test_ontology_validation_reports_grounded_extraction()
    test_ontology_validation_detects_hallucinated_object()

    print("=" * 40)
    print("🎉 All tests passed!")
