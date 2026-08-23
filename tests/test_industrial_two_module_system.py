"""Tests for the industrial data-cleaning and text-to-KG module boundary."""

from backend.pipeline.industrial_text_to_kg import (
    IndustrialEdge,
    IndustrialNode,
    ModuleExperimentLogRecord,
    classify_industrial_object,
    postprocess_industrial_kg,
    summarize_relation_origins,
    write_module_experiment_log,
)
from backend.preprocessing.unified_text import (
    EvidenceNotFoundError,
    GroundedEntry,
    build_industrial_unified_text,
)
from backend.graph.property_graph import build_property_graph
from backend.schemas.egocentric_video import Action, EgocentricVideoExtraction, SceneObject
from backend.schemas.process_knowledge.entities import Tool
from scripts.select_indego_gold_candidates import build_annotation_template
from scripts.run_easg_real_llm_lightweight_experiments import (
    flatten_relation_payload,
    normalize_payload_shape,
    normalize_relation_type,
)


RAW_TEXT = (
    "Operator Lena positions the cover plate on the housing, tightens the bolt "
    "with the torque wrench, and checks the final fit. The torque is 12 Nm."
)


def test_industrial_unified_text_keeps_source_supported_fields() -> None:
    """Industrial cleaning output should retain evidence for every new field."""

    record = build_industrial_unified_text(
        raw_text=RAW_TEXT,
        scene_id="assembly_demo_01",
        segment_id="s1",
        timestamp="00:00-00:20",
        scene="cover plate assembly",
        actors=[GroundedEntry(text="Operator Lena", evidence="Operator Lena")],
        action_sequence=[
            GroundedEntry(
                text="position cover plate",
                evidence="positions the cover plate",
            ),
            GroundedEntry(text="tighten bolt", evidence="tightens the bolt"),
        ],
        tools_objects=[
            GroundedEntry(text="torque wrench", evidence="torque wrench"),
            GroundedEntry(text="cover plate", evidence="cover plate"),
        ],
        process_parameters=[
            GroundedEntry(text="torque: 12 Nm", evidence="torque is 12 Nm"),
        ],
        quality_results=[
            GroundedEntry(text="final fit checked", evidence="checks the final fit"),
        ],
        uncertainty=["All facts are directly supported by the source text."],
    )

    prompt_text = record.to_prompt_text()

    assert "[PROCESS PARAMETERS]" in prompt_text
    assert "torque: 12 Nm" in prompt_text
    assert "[QUALITY / RESULTS]" in prompt_text
    assert "final fit checked" in prompt_text
    assert record.source_text == RAW_TEXT


def test_industrial_unified_text_rejects_unsupported_new_facts() -> None:
    """Gold/unified records must not add facts that do not appear in the source."""

    try:
        build_industrial_unified_text(
            raw_text=RAW_TEXT,
            scene_id="assembly_demo_01",
            segment_id="s1",
            timestamp=None,
            scene="cover plate assembly",
            action_sequence=[
                GroundedEntry(text="tighten bolt", evidence="tightens the bolt"),
            ],
            tools_objects=[],
            process_parameters=[
                GroundedEntry(text="temperature: 42 C", evidence="42 C"),
            ],
        )
    except EvidenceNotFoundError as error:
        assert "42 C" in str(error)
    else:
        raise AssertionError("Unsupported industrial facts must be rejected.")


def test_industrial_postprocessing_normalizes_merges_and_flags_hallucination() -> None:
    """Text-to-KG post-processing should keep useful pilot behaviour without hospital IDs."""

    nodes = [
        IndustrialNode("step", "Tighten Bolt", "tightens the bolt"),
        IndustrialNode("tool", "Torque Wrench", "torque wrench"),
        IndustrialNode("object", "Cover Plate", "cover plate"),
        IndustrialNode("parameter", "Laser Power", "laser power"),
    ]
    edges = [
        IndustrialEdge("Torque Wrench", "uses_tool", "Tighten Bolt", "tightens the bolt with the torque wrench"),
        IndustrialEdge("tighten bolt", "USES_TOOL", "torque wrench", "tightens the bolt with the torque wrench"),
        IndustrialEdge("Tighten Bolt", "acts_on", "Cover Plate", "positions the cover plate"),
        IndustrialEdge("Tighten Bolt", "related_to", "Cover Plate", "tightens the bolt"),
    ]

    result = postprocess_industrial_kg(nodes, edges, RAW_TEXT)

    assert ("Action", "tighten bolt") in result.nodes
    assert ("Tool", "torque wrench") in result.nodes
    assert ("tighten bolt", "USES_TOOL", "torque wrench") in result.edges
    assert result.stats.auto_corrected_edges == 1
    assert result.stats.merged_duplicate_edges == 1
    assert result.stats.invalid_relation_count == 1
    assert result.stats.hallucinated_nodes == 1


def test_tool_classifier_handles_nylon_brush_without_turning_workpieces_into_tools() -> None:
    """The product classifier should catch common tools while staying conservative."""

    assert (
        classify_industrial_object(
            entity_id="object_1",
            name="nylon brush",
            object_type="object",
            evidence_text="The operator cleans the surface using the nylon brush.",
        )
        == "Tool"
    )
    assert (
        classify_industrial_object(
            entity_id="object_2",
            name="steel bracket",
            object_type="object",
            evidence_text="The operator positions the steel bracket on the base plate.",
        )
        == "Object"
    )


def test_relation_origin_metadata_is_preserved_in_graph_summary() -> None:
    """Final product graphs should reveal which edges came from LLM vs fallback."""

    action_1 = Action(name="clean surface")
    action_2 = Action(name="inspect surface")
    brush = Tool(name="nylon brush")
    surface = SceneObject(name="surface")
    extraction = EgocentricVideoExtraction(
        actions=[action_1, action_2],
        tools=[brush],
        objects=[surface],
        uses_tool=[
            {
                "action": action_1,
                "tool": brush,
                "evidence_text": "using the nylon brush",
                "relation_origin": "evidence_fallback",
            }
        ],
        acts_on_object=[
            {
                "action": action_1,
                "object": surface,
                "evidence_text": "cleans the surface",
                "relation_origin": "llm_extracted",
            }
        ],
        action_order=[
            {
                "before": action_1,
                "after": action_2,
                "evidence_text": "action sequence",
                "relation_origin": "sequence_fallback",
            }
        ],
    )

    graph = build_property_graph(extraction)
    summary = summarize_relation_origins(graph.edges)

    assert summary.by_origin["llm_extracted"] == 1
    assert summary.by_origin["evidence_fallback"] == 1
    assert summary.by_origin["sequence_fallback"] == 1
    assert summary.by_relation_origin["USES_TOOL:evidence_fallback"] == 1


def test_relation_dict_payload_is_flattened_without_losing_llm_relations() -> None:
    """Small LLMs often return relations grouped by type instead of as one list."""

    flattened = flatten_relation_payload(
        {
            "ACTS_ON": [{"subject_id": "action_1", "object_id": "object_1"}],
            "USES_TOOL": [{"action_id": "action_1", "target_id": "object_2"}],
        }
    )

    assert flattened == [
        {"subject_id": "action_1", "object_id": "object_1", "relation_type": "ACTS_ON"},
        {
            "action_id": "action_1",
            "target_id": "object_2",
            "relation_type": "USES_TOOL",
            "subject_id": "action_1",
            "object_id": "object_2",
        },
    ]

    normalized = normalize_payload_shape(
        {
            "ACTS_ON": [{"subject_id": "action_1", "object_id": "object_1"}],
            "USES_TOOL": [{"subject_id": "action_1", "object_id": "object_2"}],
        }
    )
    assert [item["relation_type"] for item in normalized["relations"]] == ["ACTS_ON", "USES_TOOL"]
    assert normalize_relation_type("ACTS_ON") == "ACTS_ON"
    assert normalize_relation_type("acts on") == "ACTS_ON"


def test_indego_gold_template_is_candidate_not_final_gold() -> None:
    """Automatic IndEgo templates must not be mislabeled as reviewed gold."""

    record = {
        "scene_id": "indego_demo_s1",
        "video_id": "indego_demo",
        "category": "assembly",
        "source_paths": ["demo.json"],
        "raw_text": "Step 1: clean surface with nylon brush. Step 2: position steel bracket.",
        "unified_record": {
            "action_sequence": [
                {"text": "clean surface", "evidence": "Step 1: clean surface with nylon brush."},
                {"text": "position steel bracket", "evidence": "Step 2: position steel bracket."},
            ],
            "tools_objects": [
                {"text": "nylon brush", "evidence": "Step 1: clean surface with nylon brush."},
                {"text": "steel bracket", "evidence": "Step 2: position steel bracket."},
            ],
        },
    }

    template = build_annotation_template(record, rank=1)

    assert template["status"] == "candidate_gold_not_reviewed"
    assert any(node["label"] == "Tool" and node["name"] == "nylon brush" for node in template["candidate_gold_nodes"])
    assert any(edge["relation"] == "BEFORE" for edge in template["candidate_gold_edges"])


def test_module_experiment_log_records_raw_and_normalized_metrics(tmp_path) -> None:
    """Module screening logs should preserve raw and post-processed scores."""

    path = write_module_experiment_log(
        [
            ModuleExperimentLogRecord(
                input_type="unified",
                extraction_strategy="layered",
                model="llama3.1:8b",
                raw_node_f1=0.625,
                raw_edge_f1=0.6,
                normalized_node_f1=1.0,
                normalized_edge_f1=1.0,
                hallucination_count=0,
                module_decision="retain",
                notes="Unified structure improved noisy input extraction.",
            )
        ],
        tmp_path / "module_log.md",
    )

    text = path.read_text(encoding="utf-8")

    assert "Raw Node F1" in text
    assert "Normalized Edge F1" in text
    assert "retain" in text
