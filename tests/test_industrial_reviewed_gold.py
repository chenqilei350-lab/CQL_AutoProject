from pathlib import Path

from backend.datasets.industrial_reviewed_gold import (
    DEFAULT_REVIEWED_GOLD_PATH,
    SCORING_NODE_LABELS,
    SCORING_RELATION_TYPES,
    V3_REVIEWED_GOLD_PATH,
    load_industrial_reviewed_gold_dataset,
)
from backend.graph.property_graph import build_property_graph
from scripts.run_easg_real_llm_lightweight_experiments import load_experiment_scenes


REVIEWED_PATH = Path(
    "data/industrial_gold_candidates/industrial_gold_v2_reviewed.jsonl"
)


def test_loads_all_accepted_reviewed_gold_scenes() -> None:
    dataset = load_industrial_reviewed_gold_dataset(REVIEWED_PATH)

    assert dataset.name == "industrial_reviewed_gold_v2"
    assert len(dataset.scenes) == 12
    assert all(scene.gold_extraction.source_text == scene.raw_text for scene in dataset.scenes)
    assert all(scene.unified_record.source_text == scene.raw_text for scene in dataset.scenes)


def test_loads_v3_reviewed_gold_scenes() -> None:
    dataset = load_industrial_reviewed_gold_dataset(V3_REVIEWED_GOLD_PATH)

    assert dataset.name == "industrial_reviewed_gold_v3"
    assert len(dataset.scenes) == 20
    assert sum(len(scene.gold_extraction.actions) for scene in dataset.scenes) == 224
    assert sum(len(scene.gold_extraction.objects) for scene in dataset.scenes) == 144
    assert sum(len(scene.gold_extraction.tools) for scene in dataset.scenes) == 34


def test_combined_reviewed_gold_has_no_duplicate_scenes() -> None:
    dataset = load_industrial_reviewed_gold_dataset(DEFAULT_REVIEWED_GOLD_PATH)
    scene_ids = [scene.scene_id for scene in dataset.scenes]

    assert dataset.name == "industrial_reviewed_gold_combined_v2_v3"
    assert len(scene_ids) == 32
    assert len(set(scene_ids)) == 32
    assert sum(len(scene.gold_extraction.actions) for scene in dataset.scenes) == 369
    assert sum(len(scene.gold_extraction.objects) for scene in dataset.scenes) == 257
    assert sum(len(scene.gold_extraction.tools) for scene in dataset.scenes) == 48
    assert sum(len(scene.gold_extraction.action_order) for scene in dataset.scenes) == 329
    assert sum(len(scene.gold_extraction.acts_on_object) for scene in dataset.scenes) == 405
    assert sum(len(scene.gold_extraction.uses_tool) for scene in dataset.scenes) == 94


def test_reviewed_gold_counts_and_relation_signatures() -> None:
    dataset = load_industrial_reviewed_gold_dataset(REVIEWED_PATH)
    node_counts = {label: 0 for label in SCORING_NODE_LABELS}
    relation_counts = {relation: 0 for relation in SCORING_RELATION_TYPES}

    for scene in dataset.scenes:
        graph = build_property_graph(scene.gold_extraction)
        for node in graph.nodes.values():
            node_counts[node.label] += 1
        for edge in graph.edges:
            relation_counts[edge.type] += 1
            source = graph.node(edge.source)
            target = graph.node(edge.target)
            if edge.type == "BEFORE":
                assert (source.label, target.label) == ("Action", "Action")
            elif edge.type == "ACTS_ON":
                assert (source.label, target.label) == ("Action", "Object")
            elif edge.type == "USES_TOOL":
                assert (source.label, target.label) == ("Action", "Tool")

    assert node_counts == {"Action": 145, "Object": 113, "Tool": 14}
    assert relation_counts == {"BEFORE": 133, "ACTS_ON": 170, "USES_TOOL": 22}


def test_unified_condition_is_explicitly_annotation_derived() -> None:
    dataset = load_industrial_reviewed_gold_dataset(REVIEWED_PATH, limit=1)
    scene = dataset.scenes[0]

    assert "human-reviewed annotations" in scene.input_text("unified")
    assert scene.raw_text in scene.input_text("unified")


def test_experiment_loader_selects_requested_reviewed_scenes() -> None:
    requested = [
        "indego_user_16_414_2101_1_s2",
        "indego_user_15_414_0702_1_s1",
    ]

    dataset_name, scenes = load_experiment_scenes(
        "industrial_reviewed_gold",
        REVIEWED_PATH,
        limit=10,
        scene_ids=requested,
    )

    assert dataset_name == "industrial_reviewed_gold_v2"
    assert [scene.scene_id for scene in scenes] == requested
