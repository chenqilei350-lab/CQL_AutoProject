"""扩展五场景 benchmark 的测试。"""

import json

from backend.datasets.expanded_benchmark import EXPANDED_BENCHMARK
from backend.evaluation.ontology_validation import validate_egocentric_extraction
from backend.graph.property_graph import build_property_graph


def test_expanded_benchmark_contains_five_grounded_scenes() -> None:
    """扩展数据集应保留基础样例并增加三个不同任务场景。"""

    assert [scene.scene_id for scene in EXPANDED_BENCHMARK.scenes] == [
        "weld_demo_01",
        "inspect_demo_01",
        "assembly_demo_01",
        "maintenance_demo_01",
        "thermal_demo_01",
    ]
    assert all(
        scene.unified_record.source_text == scene.raw_text
        for scene in EXPANDED_BENCHMARK.scenes
    )


def test_additional_gold_graphs_cover_tool_order_causality_and_parameter() -> None:
    """新增人工标准图应覆盖本项目需要比较的重要知识类型。"""

    assembly_graph = build_property_graph(
        EXPANDED_BENCHMARK.get_scene("assembly_demo_01").gold_extraction
    )
    maintenance_graph = build_property_graph(
        EXPANDED_BENCHMARK.get_scene("maintenance_demo_01").gold_extraction
    )
    thermal_extraction = EXPANDED_BENCHMARK.get_scene(
        "thermal_demo_01"
    ).gold_extraction
    thermal_graph = build_property_graph(thermal_extraction)

    assert [node.name for node in assembly_graph.tools_for_action("tighten bolt")] == [
        "torque wrench"
    ]
    assert [
        node.name for node in maintenance_graph.effects_caused_by("switch off machine")
    ] == ["wipe sensor lens"]
    assert thermal_extraction.parameters[0].name == "temperature"
    assert thermal_extraction.parameters[0].nominal_value == 42.0
    parameter = thermal_graph.find_node("ProcessParameter", "temperature")
    assert parameter is not None
    assert parameter.properties["nominal_value"] == 42.0


def test_additional_gold_relations_follow_ontology_contract() -> None:
    """新增 gold relations 的 domain/range 必须符合既有 ontology schema。"""

    for scene_id in [
        "assembly_demo_01",
        "maintenance_demo_01",
        "thermal_demo_01",
    ]:
        scene = EXPANDED_BENCHMARK.get_scene(scene_id)
        report = validate_egocentric_extraction(
            scene.gold_extraction,
            source_text=scene.raw_text,
        )
        assert report.invalid_relations == 0
        assert report.ontology_conformance == 1.0


def test_expanded_dataset_exports_all_raw_unified_gold_records() -> None:
    """扩展数据集导出后可作为真实模型实验的输入清单。"""

    records = [
        json.loads(line) for line in EXPANDED_BENCHMARK.to_jsonl().splitlines()
    ]

    assert len(records) == 5
    thermal = next(
        record for record in records if record["scene_id"] == "thermal_demo_01"
    )
    assert "42 degrees" in thermal["raw_text"]
    assert "[结果 / 参数]" in thermal["unified_text"]
    assert thermal["gold_extraction"]["parameters"][0]["nominal_value"] == 42.0
