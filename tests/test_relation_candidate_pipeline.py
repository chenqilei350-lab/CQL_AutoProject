"""Tests for the generic relation-candidate pipeline."""

from backend.pipeline.relation_candidate_pipeline import (
    DeterministicRelationCandidateScorer,
    EntityMention,
    LLMBinaryJudgeScorer,
    OptionalGLiRELScorer,
    ProceduralRelationCandidateGenerator,
    RelationCandidateDecision,
    build_binary_judge_prompt,
)


SOURCE_TEXT = (
    "Step 1: The operator cleans the pump housing with a nylon brush. "
    "Step 2: The operator positions the steel bracket. "
    "Warning 1: step 'clean pump housing' has issue 'brush not fully applied'."
)


def test_procedural_candidate_generator_emits_core_relations() -> None:
    generator = ProceduralRelationCandidateGenerator()
    actions = [
        EntityMention(
            entity_id="a1",
            label="Action",
            name="clean pump housing",
            evidence_text="The operator cleans the pump housing with a nylon brush.",
            start_seconds=1,
            end_seconds=5,
            sequence_index=1,
        ),
        EntityMention(
            entity_id="a2",
            label="Action",
            name="position steel bracket",
            evidence_text="The operator positions the steel bracket.",
            start_seconds=6,
            end_seconds=9,
            sequence_index=2,
        ),
    ]
    tools = [
        EntityMention(
            entity_id="t1",
            label="Tool",
            name="nylon brush",
            evidence_text="The operator cleans the pump housing with a nylon brush.",
        )
    ]
    objects = [
        EntityMention(
            entity_id="o1",
            label="Object",
            name="pump housing",
            evidence_text="The operator cleans the pump housing with a nylon brush.",
        ),
        EntityMention(
            entity_id="o2",
            label="Object",
            name="steel bracket",
            evidence_text="The operator positions the steel bracket.",
        ),
    ]
    scenes = [EntityMention(entity_id="s1", label="Scene", name="pump assembly scene")]
    keysteps = [
        EntityMention(
            entity_id="k1",
            label="Keystep",
            name="cleaning step",
            start_seconds=0,
            end_seconds=5,
        )
    ]
    warnings = [
        EntityMention(
            entity_id="w1",
            label="Warning",
            name="brush not fully applied",
            evidence_text="Warning 1: step 'clean pump housing' has issue 'brush not fully applied'.",
            metadata={"step": "clean pump housing"},
        )
    ]

    candidates = generator.generate(
        actions=actions,
        tools=tools,
        objects=objects,
        scenes=scenes,
        keysteps=keysteps,
        warnings=warnings,
        source_text=SOURCE_TEXT,
    )
    triples = {
        (candidate.relation_type, candidate.subject_id, candidate.object_id)
        for candidate in candidates
    }

    assert ("USES_TOOL", "a1", "t1") in triples
    assert ("ACTS_ON", "a1", "o1") in triples
    assert ("ACTS_ON", "a2", "o2") in triples
    assert ("BEFORE", "a1", "a2") in triples
    assert ("PART_OF", "a1", "k1") in triples
    assert ("OBSERVED_IN", "a1", "s1") in triples
    assert ("WARNING_FOR", "w1", "a1") in triples
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)


def test_deterministic_scorer_returns_validation_report() -> None:
    generator = ProceduralRelationCandidateGenerator()
    candidates = generator.generate(
        actions=[
            EntityMention(
                entity_id="a1",
                label="Action",
                name="clean pump housing",
                evidence_text="The operator cleans the pump housing with a nylon brush.",
            )
        ],
        tools=[EntityMention(entity_id="t1", label="Tool", name="nylon brush")],
        source_text=SOURCE_TEXT,
    )

    report = DeterministicRelationCandidateScorer().score(candidates, source_text=SOURCE_TEXT)

    assert report.scorer_name == "deterministic_rules"
    assert report.accepted_candidates()
    assert report.relation_counts()["USES_TOOL"] == 1
    assert report.decisions[0].is_supported is True


def test_candidate_generator_does_not_cross_link_unrelated_tools_or_objects() -> None:
    """Fallback candidates must stay local to the action evidence."""

    generator = ProceduralRelationCandidateGenerator()
    candidates = generator.generate(
        actions=[
            EntityMention(
                entity_id="a1",
                label="Action",
                name="clean",
                evidence_text="The operator cleans the pump housing with a nylon brush.",
                sequence_index=1,
            ),
            EntityMention(
                entity_id="a2",
                label="Action",
                name="tighten",
                evidence_text="The operator tightens the bolt with a torque wrench.",
                sequence_index=2,
            ),
        ],
        tools=[
            EntityMention(entity_id="t1", label="Tool", name="nylon brush"),
            EntityMention(entity_id="t2", label="Tool", name="torque wrench"),
        ],
        objects=[
            EntityMention(entity_id="o1", label="Object", name="pump housing"),
            EntityMention(entity_id="o2", label="Object", name="bolt"),
        ],
        source_text=(
            "The operator cleans the pump housing with a nylon brush.\n"
            "The operator tightens the bolt with a torque wrench."
        ),
    )
    triples = {
        (candidate.relation_type, candidate.subject_id, candidate.object_id)
        for candidate in candidates
    }

    assert ("USES_TOOL", "a1", "t1") in triples
    assert ("USES_TOOL", "a2", "t2") in triples
    assert ("ACTS_ON", "a1", "o1") in triples
    assert ("ACTS_ON", "a2", "o2") in triples
    assert ("USES_TOOL", "a1", "t2") not in triples
    assert ("USES_TOOL", "a2", "t1") not in triples
    assert ("ACTS_ON", "a1", "o2") not in triples
    assert ("ACTS_ON", "a2", "o1") not in triples


def test_llm_binary_judge_can_only_return_candidate_ids() -> None:
    class FakeBackend:
        def extract(self, text, response_model, system_prompt=None):
            assert "Do not invent" in system_prompt
            assert "c_001" in text
            return response_model(
                decisions=[
                    RelationCandidateDecision(
                        candidate_id="c_001",
                        is_supported=True,
                        confidence=0.82,
                        evidence_text="The operator cleans the pump housing with a nylon brush.",
                        reason="The sentence names the action and tool.",
                    ),
                    RelationCandidateDecision(
                        candidate_id="invented",
                        is_supported=True,
                        confidence=1.0,
                        evidence_text="not allowed",
                        reason="This must be ignored.",
                    ),
                ]
            )

    candidates = ProceduralRelationCandidateGenerator().generate(
        actions=[
            EntityMention(
                entity_id="a1",
                label="Action",
                name="clean pump housing",
                evidence_text="The operator cleans the pump housing with a nylon brush.",
            )
        ],
        tools=[EntityMention(entity_id="t1", label="Tool", name="nylon brush")],
        source_text=SOURCE_TEXT,
    )
    report = LLMBinaryJudgeScorer(FakeBackend()).score(candidates, source_text=SOURCE_TEXT)

    assert [decision.candidate_id for decision in report.decisions] == ["c_001"]
    assert report.decisions[0].is_supported is True
    assert report.decisions[0].scorer_name == "llm_binary_judge"


def test_binary_judge_prompt_uses_candidate_json_contract() -> None:
    candidates = ProceduralRelationCandidateGenerator().generate(
        actions=[EntityMention(entity_id="a1", label="Action", name="clean pump housing")],
        scenes=[EntityMention(entity_id="s1", label="Scene", name="pump scene")],
    )

    prompt = build_binary_judge_prompt(candidates, SOURCE_TEXT)

    assert '"candidate_id": "c_001"' in prompt
    assert '"is_supported": true' in prompt
    assert "source_text" in prompt


def test_optional_glirel_scorer_is_injectable_without_hard_dependency() -> None:
    class FakeGLiREL:
        def score(self, candidate, source_text):
            return 0.91 if candidate.relation_type == "OBSERVED_IN" else 0.2

    candidates = ProceduralRelationCandidateGenerator().generate(
        actions=[EntityMention(entity_id="a1", label="Action", name="clean pump housing")],
        scenes=[EntityMention(entity_id="s1", label="Scene", name="pump scene")],
    )
    report = OptionalGLiRELScorer(FakeGLiREL(), threshold=0.5).score(candidates, source_text=SOURCE_TEXT)

    assert report.scorer_name == "optional_glirel"
    assert report.accepted_candidates()[0].relation_type == "OBSERVED_IN"
