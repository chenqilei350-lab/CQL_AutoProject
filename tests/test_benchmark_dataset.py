"""小型实验数据集模块的功能测试。"""

import json

import pytest

from backend.datasets.benchmark import (
    BenchmarkScene,
    MVP_BENCHMARK,
)
from backend.graph.property_graph import build_property_graph
from backend.preprocessing.unified_text import GroundedEntry, build_unified_text
from backend.schemas.egocentric_examples import WELDING_SCENE_EXPECTED, WELDING_SCENE_TEXT


def test_mvp_benchmark_contains_two_seed_scenes():
    """确认当前数据集包含焊接与质量检查两个起始场景。"""

    assert MVP_BENCHMARK.name == "egocentric_raw_unified_mvp"
    assert [scene.scene_id for scene in MVP_BENCHMARK.scenes] == [
        "weld_demo_01",
        "inspect_demo_01",
    ]


def test_each_scene_provides_raw_unified_and_gold_inputs():
    """确认每个场景都具备后续公平比较所需要的三类内容。"""

    for scene in MVP_BENCHMARK.scenes:
        assert scene.input_text("raw") == scene.raw_text
        assert "[动作顺序]" in scene.input_text("unified")
        assert scene.input_text("unified") != scene.input_text("raw")
        assert scene.gold_extraction.source_text == scene.raw_text


def test_gold_extractions_can_be_turned_into_queryable_graphs():
    """确认数据集中人工标准答案能够继续接入现有图构建模块。"""

    welding_scene = MVP_BENCHMARK.get_scene("weld_demo_01")
    welding_graph = build_property_graph(welding_scene.gold_extraction)

    assert [node.name for node in welding_graph.tools_for_action("start root weld")] == [
        "Fronius TPS 400i torch"
    ]
    assert [node.name for node in welding_graph.effects_caused_by("align plate")] == [
        "start root weld"
    ]


def test_dataset_exports_json_lines_with_both_input_forms():
    """确认数据集可导出为便于保存和检查的逐行 JSON 文本。"""

    lines = MVP_BENCHMARK.to_jsonl().splitlines()
    records = [json.loads(line) for line in lines]

    assert len(records) == 2
    assert records[0]["scene_id"] == "weld_demo_01"
    assert records[0]["raw_text"] == WELDING_SCENE_TEXT
    assert "[场景 / 片段]" in records[0]["unified_text"]
    assert records[0]["gold_extraction"]["video_id"] == "weld_demo_01"


def test_scene_rejects_unaligned_raw_and_unified_source_text():
    """确认一个场景不能错误绑定来自不同原文的统一输入。"""

    unrelated_unified = build_unified_text(
        raw_text="Maria measures a gap.",
        scene_id="incorrect",
        segment_id="s1",
        timestamp=None,
        scene_segment="inspection",
        actors=[GroundedEntry(text="Maria", evidence="Maria")],
        action_sequence=[],
        tools_objects=[],
        outcomes_parameters=[],
    )

    with pytest.raises(ValueError, match="raw_text"):
        BenchmarkScene(
            scene_id="weld_demo_01",
            description="错误绑定测试",
            raw_text=WELDING_SCENE_TEXT,
            unified_record=unrelated_unified,
            gold_extraction=WELDING_SCENE_EXPECTED,
        )


def test_unknown_input_condition_is_rejected():
    """确认实验运行器以后不能意外使用未定义的输入条件。"""

    scene = MVP_BENCHMARK.get_scene("inspect_demo_01")

    with pytest.raises(ValueError, match="不支持的输入条件"):
        scene.input_text("noisy")  # type: ignore[arg-type]
