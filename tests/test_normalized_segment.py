"""Tests for the six-layer normalized-segment extraction boundary."""

from __future__ import annotations

import pytest

from backend.pipeline.algorithm_experiment import (
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
    ControlledEntityDecision,
    ControlledEntityDecisionResult,
    MinimalEntity,
    MinimalEntityExtractionResult,
    MinimalRelation,
    MinimalRelationExtractionResult,
)
from backend.preprocessing.normalized_segment import (
    NormalizedAction,
    NormalizedMention,
    NormalizedSegment,
)
from backend.preprocessing.unified_text import (
    GroundedEntry,
    build_industrial_unified_text,
)


SOURCE = "Operator Lena tightens the mounting screw with a torque wrench."


def build_segment() -> NormalizedSegment:
    record = build_industrial_unified_text(
        raw_text=SOURCE,
        scene_id="assembly_01",
        segment_id="s1",
        timestamp="00:01-00:03",
        scene="assembly",
        actors=[
            GroundedEntry(
                text="Operator Lena",
                evidence="Operator Lena",
                entry_type="role",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="tighten mounting screw with torque wrench",
                evidence="tightens the mounting screw with a torque wrench",
                entry_type="action",
                verb="tighten",
                direct_object="mounting screw",
                tool="torque wrench",
                role="Operator Lena",
            )
        ],
        tools_objects=[
            GroundedEntry(
                text="mounting screw",
                evidence="mounting screw",
                entry_type="object",
            ),
            GroundedEntry(
                text="torque wrench",
                evidence="torque wrench",
                entry_type="tool",
            ),
        ],
        uncertainty=["No uncertainty in the annotation."],
        source_adapter="test_adapter",
        annotation_text="tighten mounting screw with torque wrench",
        transcript_text=SOURCE,
    )
    assert record.normalized_segment is not None
    return record.normalized_segment


def test_normalized_segment_splits_action_object_tool_and_role() -> None:
    segment = build_segment()

    assert segment.source_adapter == "test_adapter"
    assert [item.mention_id for item in segment.roles] == ["R1"]
    assert [item.mention_id for item in segment.objects] == ["O1"]
    assert [item.mention_id for item in segment.tools] == ["T1"]
    assert segment.actions[0].direct_object_id == "O1"
    assert segment.actions[0].tool_id == "T1"
    assert segment.actions[0].role_id == "R1"


def test_controlled_renderer_uses_slots_and_preserves_source_evidence() -> None:
    controlled = build_segment().to_controlled_text()

    assert "[NORMALIZED SEGMENT]" in controlled
    assert "O1 | OBJECT | name=mounting screw" in controlled
    assert "T1 | TOOL | name=torque wrench" in controlled
    assert (
        "sentence=tighten mounting screw with torque wrench."
        in controlled
    )
    assert "direct_object_id=O1" in controlled
    assert "direct_object_name=mounting screw" in controlled
    assert "tool_id=T1" in controlled
    assert "T1:torque wrench" not in controlled
    assert "[SOURCE EVIDENCE]" in controlled
    assert SOURCE in controlled


def test_legacy_mixed_column_uses_only_explicit_grammatical_tool_cue() -> None:
    record = build_industrial_unified_text(
        raw_text="The worker tightens a screw with a wrench.",
        scene_id="legacy",
        segment_id="s1",
        timestamp=None,
        scene="legacy record",
        action_sequence=[
            GroundedEntry(
                text="tighten screw with wrench",
                evidence="tightens a screw with a wrench",
            )
        ],
        tools_objects=[
            GroundedEntry(text="screw", evidence="screw"),
            GroundedEntry(text="wrench", evidence="wrench"),
        ],
    )

    segment = record.normalized_segment

    assert segment is not None
    assert [item.text for item in segment.objects] == ["screw"]
    assert [item.text for item in segment.tools] == ["wrench"]


def test_normalized_segment_rejects_missing_action_reference() -> None:
    with pytest.raises(ValueError, match="Unknown tool_id"):
        NormalizedSegment(
            source_adapter="test",
            scene_id="s",
            segment_id="1",
            actions=[
                NormalizedAction(
                    action_id="A1",
                    text="tighten screw",
                    verb="tighten",
                    evidence="tighten screw",
                    tool_id="T404",
                )
            ],
            objects=[
                NormalizedMention(
                    mention_id="O1",
                    kind="object",
                    text="screw",
                    evidence="screw",
                )
            ],
            source_text="tighten screw",
        )


def test_normalized_segment_runs_entity_relation_and_hard_validation_layers() -> None:
    class SixLayerFakeExtractor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object, str | None]] = []

        def extract(self, text, response_model, system_prompt=None):
            self.calls.append((text, response_model, system_prompt))
            if response_model is ControlledEntityDecisionResult:
                return ControlledEntityDecisionResult(
                    decisions=[
                        ControlledEntityDecision(
                            entity_id="R1",
                            keep=True,
                            entity_type="Worker",
                            name="Operator Lena",
                            confidence=0.99,
                        ),
                        ControlledEntityDecision(
                            entity_id="O1",
                            keep=True,
                            entity_type="Object",
                            name="mounting screw",
                            confidence=0.99,
                        ),
                        ControlledEntityDecision(
                            entity_id="T1",
                            keep=True,
                            entity_type="Tool",
                            name="torque wrench",
                            confidence=0.99,
                        ),
                        ControlledEntityDecision(
                            entity_id="A1",
                            keep=True,
                            entity_type="Action",
                            name="tighten mounting screw with torque wrench",
                            confidence=0.99,
                        ),
                    ]
                )
            if response_model is MinimalEntityExtractionResult:
                return MinimalEntityExtractionResult(
                    entities=[
                        MinimalEntity(
                            entity_id="action_1",
                            entity_type="Action",
                            name="tighten mounting screw",
                            evidence_text=(
                                "tightens the mounting screw with a torque wrench"
                            ),
                        ),
                        MinimalEntity(
                            entity_id="object_1",
                            entity_type="Object",
                            name="mounting screw",
                            evidence_text="mounting screw",
                        ),
                        MinimalEntity(
                            entity_id="tool_1",
                            entity_type="Tool",
                            name="torque wrench",
                            evidence_text="torque wrench",
                        ),
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                return MinimalRelationExtractionResult(
                    relations=[
                        MinimalRelation(
                            relation_type="USES_TOOL",
                            subject_id="A1",
                            object_id="T1",
                            evidence_text=SOURCE,
                        ),
                        MinimalRelation(
                            relation_type="ACTS_ON",
                            subject_id="A1",
                            object_id="O1",
                            evidence_text=SOURCE,
                        ),
                    ]
                )
            raise AssertionError(response_model)

    extractor = SixLayerFakeExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(conditions=("llm_relation_proposal",)),
        extractor=extractor,
    )

    extraction = runner.extract_normalized_segment(build_segment())

    assert len(extraction.uses_tool) == 1
    assert len(extraction.acts_on_object) == 1
    assert len(extractor.calls) == 2
    # 中文：两个 LLM 阶段接收同一 Controlled Text 的最小受控视图，关系
    # 阶段仍必须保留完整 SOURCE EVIDENCE 供 Hard Validation 回查。
    # English: Each LLM stage receives its compact controlled view; relation
    # reasoning still retains complete source evidence for hard validation.
    assert "[CONTROLLED ENTITY DECISION VIEW]" in extractor.calls[0][0]
    assert "candidate_evidence=Operator Lena" in extractor.calls[0][0]
    assert "candidate_evidence=tightens the mounting screw" in extractor.calls[0][0]
    assert "[CONTROLLED RELATION VIEW]" in extractor.calls[1][0]
    assert "[SOURCE EVIDENCE]" in extractor.calls[1][0]
    assert SOURCE in extractor.calls[1][0]
    assert "Return exactly one decision for every ID" in (
        extractor.calls[0][2] or ""
    )
    assert runner.last_debug_trace is not None
    assert not runner.last_debug_trace.rejected_relations
    assert not runner.last_debug_trace.missing_entity_decision_ids


def test_long_segment_splits_layers_four_to_six_with_global_ids() -> None:
    segment = NormalizedSegment(
        source_adapter="test",
        scene_id="long",
        segment_id="s1",
        actions=[
            NormalizedAction(
                action_id=f"A{index}",
                text=f"check item {index}",
                verb="check",
                evidence=f"Check item {index}.",
            )
            for index in range(1, 14)
        ],
        source_text=" ".join(f"Check item {index}." for index in range(1, 14)),
    )

    parts = segment.split_by_actions(max_actions=6, overlap_actions=1)

    assert [len(part.actions) for part in parts] == [6, 6, 3]
    assert [action.action_id for action in parts[0].actions] == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    ]
    assert parts[1].actions[0].action_id == "A6"
    assert parts[2].actions[0].action_id == "A11"
    assert "Check item 13." in parts[2].source_text


def test_partition_timeout_keeps_successful_sibling_entities() -> None:
    class SecondPartitionTimeoutExtractor:
        def __init__(self) -> None:
            self.entity_calls = 0

        def extract(self, _text, response_model, system_prompt=None):
            del system_prompt
            if response_model is ControlledEntityDecisionResult:
                self.entity_calls += 1
                if self.entity_calls == 2:
                    raise TimeoutError("partition timed out")
                if self.entity_calls == 3:
                    return ControlledEntityDecisionResult(
                        decisions=[
                            ControlledEntityDecision(
                                entity_id=f"A{index}",
                                keep=True,
                                entity_type="Action",
                                name=f"check item {index}",
                            )
                            for index in range(7, 9)
                        ]
                    )
                return ControlledEntityDecisionResult(
                    decisions=[
                        ControlledEntityDecision(
                            entity_id=f"A{index}",
                            keep=True,
                            entity_type="Action",
                            name=f"check item {index}",
                        )
                        for index in range(1, 7)
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                return MinimalRelationExtractionResult()
            raise AssertionError(response_model)

    segment = NormalizedSegment(
        source_adapter="test",
        scene_id="long",
        segment_id="s1",
        actions=[
            NormalizedAction(
                action_id=f"A{index}",
                text=f"check item {index}",
                verb="check",
                evidence=f"Check item {index}.",
            )
            for index in range(1, 9)
        ],
        source_text=" ".join(f"Check item {index}." for index in range(1, 9)),
    )
    runner = AlgorithmExperimentRunner(extractor=SecondPartitionTimeoutExtractor())

    extraction = runner.extract_normalized_segment(segment)

    assert [action.entity_id for action in extraction.actions] == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
    ]
    assert runner.last_debug_trace is not None
    assert runner.last_debug_trace.segment_count == 2
    assert runner.last_debug_trace.successful_segment_count == 1
    assert "partition 2/2 failed" in runner.last_debug_trace.stage_warnings[0]
    assert not runner.last_debug_trace.missing_entity_decision_ids
    assert any(
        "Entity-only partition recovery batch 1 reviewed renderer IDs: A7, A8"
        in warning
        for warning in runner.last_debug_trace.stage_warnings
    )


def test_partition_merge_keeps_entity_decision_and_kg_consistent() -> None:
    class ConflictingOverlapExtractor:
        def extract(self, text, response_model, system_prompt=None):
            del system_prompt
            if response_model is ControlledEntityDecisionResult:
                ids = [
                    token.split(" |", 1)[0]
                    for token in text.splitlines()
                    if token.startswith("A")
                ]
                return ControlledEntityDecisionResult(
                    decisions=[
                        ControlledEntityDecision(
                            entity_id=entity_id,
                            # A6 is rejected in its first partition and kept in
                            # the overlapping second partition.
                            keep=not (entity_id == "A6" and "A1 |" in text),
                            entity_type="Action",
                            name=f"check item {entity_id[1:]}",
                        )
                        for entity_id in ids
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                return MinimalRelationExtractionResult()
            raise AssertionError(response_model)

    segment = NormalizedSegment(
        source_adapter="test",
        scene_id="long",
        segment_id="s1",
        actions=[
            NormalizedAction(
                action_id=f"A{index}",
                text=f"check item {index}",
                verb="check",
                evidence=f"Check item {index}.",
            )
            for index in range(1, 9)
        ],
        source_text=" ".join(f"Check item {index}." for index in range(1, 9)),
    )
    runner = AlgorithmExperimentRunner(extractor=ConflictingOverlapExtractor())

    extraction = runner.extract_normalized_segment(segment)

    assert "A6" not in {action.entity_id for action in extraction.actions}
    assert runner.last_debug_trace is not None
    decision = next(
        item
        for item in runner.last_debug_trace.entity_decisions
        if item.entity_id == "A6"
    )
    assert not decision.keep
    assert any(
        "Conflicting LLM entity decisions" in warning
        for warning in runner.last_debug_trace.stage_warnings
    )


def test_many_undecided_ids_are_recovered_in_bounded_batches() -> None:
    class FailedPartitionsThenBatchedRecoveryExtractor:
        def __init__(self) -> None:
            self.entity_calls = 0
            self.recovery_batch_sizes: list[int] = []

        def extract(self, text, response_model, system_prompt=None):
            del system_prompt
            if response_model is ControlledEntityDecisionResult:
                self.entity_calls += 1
                if self.entity_calls <= 3:
                    raise TimeoutError("partition entity stage timed out")
                ids = [
                    line.split(" |", 1)[0]
                    for line in text.splitlines()
                    if line.startswith("A")
                ]
                self.recovery_batch_sizes.append(len(ids))
                return ControlledEntityDecisionResult(
                    decisions=[
                        ControlledEntityDecision(
                            entity_id=entity_id,
                            keep=True,
                            entity_type="Action",
                            name=f"check item {entity_id[1:]}",
                        )
                        for entity_id in ids
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                return MinimalRelationExtractionResult()
            raise AssertionError(response_model)

    segment = NormalizedSegment(
        source_adapter="test",
        scene_id="many_missing",
        segment_id="s1",
        actions=[
            NormalizedAction(
                action_id=f"A{index}",
                text=f"check item {index}",
                verb="check",
                evidence=f"Check item {index}.",
            )
            for index in range(1, 14)
        ],
        source_text=" ".join(f"Check item {index}." for index in range(1, 14)),
    )
    extractor = FailedPartitionsThenBatchedRecoveryExtractor()
    runner = AlgorithmExperimentRunner(extractor=extractor)

    extraction = runner.extract_normalized_segment(segment)

    assert extractor.recovery_batch_sizes == [6, 6, 1]
    assert len(extraction.actions) == 13
    assert runner.last_debug_trace is not None
    assert runner.last_debug_trace.successful_segment_count == 0
    assert not runner.last_debug_trace.missing_entity_decision_ids
