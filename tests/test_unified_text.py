"""Tests for unified-format text generation."""

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
    """Build a reusable quality-inspection record."""

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
        evidence_uncertainty=["The sentence order supports the action sequence."],
    )


def test_build_unified_text_preserves_scene_and_source_information():
    """A unified record preserves provenance and complete source text."""

    record = build_inspection_record()

    assert record.scene_id == "inspect_demo_01"
    assert record.segment_id == "s2"
    assert record.timestamp == "00:20-00:45"
    assert record.source_text == RAW_INSPECTION_TEXT
    assert record.action_sequence[1].text == "measure gap"


def test_unified_text_can_be_saved_as_structured_json():
    """The record can be serialized as benchmark JSON."""

    record = build_inspection_record()
    data = record.model_dump(mode="json")

    assert data["scene_segment"] == "quality inspection"
    assert data["actors"][0]["evidence"] == "quality inspector Maria"
    assert data["source_text"] == RAW_INSPECTION_TEXT


def test_to_prompt_text_renders_fixed_sections_and_action_order():
    """Prompt text has fixed sections and preserves action order."""

    prompt_text = build_inspection_record().to_prompt_text()

    assert "[SCENE / SEGMENT]" in prompt_text
    assert "[ACTORS]" in prompt_text
    assert "[ACTION SEQUENCE]" in prompt_text
    assert "1. place caliper on bracket" in prompt_text
    assert "2. measure gap" in prompt_text
    assert "3. record result" in prompt_text
    assert "[SOURCE TEXT]" in prompt_text
    assert RAW_INSPECTION_TEXT in prompt_text


def test_optional_sections_can_be_empty_and_still_render():
    """Optional sections render safely when empty."""

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

    assert "[TOOLS / OBJECTS]\nNone" in prompt_text
    assert "[OUTCOMES / PARAMETERS]\nNone" in prompt_text
    assert "[EVIDENCE / UNCERTAINTY]\nNo additional uncertainty recorded" in prompt_text


def test_evidence_not_found_in_source_text_is_rejected():
    """A tool absent from the source cannot enter the unified input."""

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
