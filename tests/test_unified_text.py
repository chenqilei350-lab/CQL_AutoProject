"""统一格式文本生成模块的功能测试。"""

import pytest

from backend.preprocessing.unified_text import (
    EvidenceNotFoundError,
    GroundedEntry,
    UnifiedTextRecord,
    build_unified_text,
)


RAW_INSPECTION_TEXT = (
    "The quality inspector Maria places the caliper on the bracket, "
    "measures the gap, and records the result."
)


def build_inspection_record() -> UnifiedTextRecord:
    """构造一个可复用的质量检查场景，用于验证正常生成流程。"""

    return build_unified_text(
        raw_text=RAW_INSPECTION_TEXT,
        scene_id="inspect_demo_01",
        segment_id="s2",
        timestamp="00:20-00:45",
        scene_segment="quality inspection",
        actors=[
            GroundedEntry(
                text="quality inspector Maria",
                evidence="quality inspector Maria",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="place caliper on bracket",
                evidence="places the caliper on the bracket",
            ),
            GroundedEntry(text="measure gap", evidence="measures the gap"),
            GroundedEntry(text="record result", evidence="records the result"),
        ],
        tools_objects=[
            GroundedEntry(text="caliper", evidence="caliper"),
            GroundedEntry(text="bracket", evidence="bracket"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="measurement result recorded",
                evidence="records the result",
            )
        ],
        evidence_uncertainty=["动作顺序由句中先后顺序提供支持。"],
    )


def test_build_unified_text_preserves_scene_and_source_information():
    """确认统一记录保存场景来源信息与完整原始文本。"""

    record = build_inspection_record()

    assert record.scene_id == "inspect_demo_01"
    assert record.segment_id == "s2"
    assert record.timestamp == "00:20-00:45"
    assert record.source_text == RAW_INSPECTION_TEXT
    assert record.action_sequence[1].text == "measure gap"


def test_unified_text_can_be_saved_as_structured_json():
    """确认记录能够作为后续 benchmark 的 JSON 数据保存。"""

    record = build_inspection_record()
    data = record.model_dump(mode="json")

    assert data["scene_segment"] == "quality inspection"
    assert data["actors"][0]["evidence"] == "quality inspector Maria"
    assert data["source_text"] == RAW_INSPECTION_TEXT


def test_to_prompt_text_renders_fixed_sections_and_action_order():
    """确认模型输入文本具有固定栏目，并保留动作顺序。"""

    prompt_text = build_inspection_record().to_prompt_text()

    assert "[场景 / 片段]" in prompt_text
    assert "[执行人员]" in prompt_text
    assert "[动作顺序]" in prompt_text
    assert "1. place caliper on bracket" in prompt_text
    assert "2. measure gap" in prompt_text
    assert "3. record result" in prompt_text
    assert "[原始文本]" in prompt_text
    assert RAW_INSPECTION_TEXT in prompt_text


def test_optional_sections_can_be_empty_and_still_render():
    """确认没有结果字段或不确定性记录时仍能生成统一文本。"""

    record = build_unified_text(
        raw_text="Hans picks up the welding torch.",
        scene_id="weld_demo_01",
        segment_id="s1",
        timestamp=None,
        scene_segment="welding preparation",
        actors=[GroundedEntry(text="Hans", evidence="Hans")],
        action_sequence=[
            GroundedEntry(text="pick up welding torch", evidence="picks up the welding torch")
        ],
        tools_objects=[],
        outcomes_parameters=[],
    )
    prompt_text = record.to_prompt_text()

    assert "[工具 / 对象]\n无" in prompt_text
    assert "[结果 / 参数]\n无" in prompt_text
    assert "[证据 / 不确定性]\n无额外不确定性记录" in prompt_text


def test_evidence_not_found_in_source_text_is_rejected():
    """确认不能把原文没有提到的工具加入统一格式输入。"""

    with pytest.raises(EvidenceNotFoundError, match="laser scanner"):
        build_unified_text(
            raw_text="Maria measures the gap with a caliper.",
            scene_id="inspect_demo_02",
            segment_id="s1",
            timestamp=None,
            scene_segment="quality inspection",
            actors=[],
            action_sequence=[
                GroundedEntry(text="measure gap", evidence="measures the gap")
            ],
            tools_objects=[
                GroundedEntry(text="laser scanner", evidence="laser scanner")
            ],
            outcomes_parameters=[],
        )
