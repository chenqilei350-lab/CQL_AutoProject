"""Algorithm-selection experiment tests."""

from pydantic import BaseModel

from backend.datasets.benchmark import MVP_BENCHMARK
from backend.evaluation.cross_validation import CrossValidationConfig, make_folds
from backend.graph.property_graph import build_property_graph
from backend.pipeline.algorithm_experiment import (
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
    ControlledEntityDecision,
    ControlledEntityDecisionResult,
    EntityExtractionResult,
    MinimalEntity,
    MinimalEntityExtractionResult,
    MinimalRelation,
    MinimalRelationCandidate,
    MinimalRelationExtractionResult,
    RelationExtractionResult,
    available_relation_contracts,
    enrich_minimal_action_names,
    filter_minimal_relations_by_evidence,
    filter_minimal_relations_by_candidates,
    generate_relation_candidates,
    llm_empty_relation_review_stage_prompt,
    llm_relation_proposal_stage_prompt,
    minimal_candidate_relation_stage_prompt,
    minimal_entity_inventory_text,
    minimal_to_egocentric_extraction,
    normalize_minimal_relation_ids,
    reconcile_minimal_entities_with_controlled_text,
    controlled_entity_extraction_input,
    controlled_relation_proposal_input,
    relation_focus_instruction,
    relation_subject_focus_groups,
    validate_llm_relation_proposals,
)
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    WELDING_SCENE_EXPECTED,
)
from backend.schemas.egocentric_video import (
    Action,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


class AlgorithmFakeExtractor:
    """Return gold entities/relations while recording requested schemas."""

    def __init__(self) -> None:
        self.response_models: list[type[BaseModel]] = []
        self.system_prompts: list[str | None] = []

    def extract(
        self,
        text: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        self.response_models.append(response_model)
        self.system_prompts.append(system_prompt)
        gold = (
            WELDING_SCENE_EXPECTED.model_copy(deep=True)
            if "weld_demo_01" in text
            else INSPECTION_SCENE_EXPECTED.model_copy(deep=True)
        )
        if response_model is EntityExtractionResult:
            return EntityExtractionResult(
                video_id=gold.video_id,
                scenes=gold.scenes,
                procedures=gold.procedures,
                actors=gold.actors,
                actions=gold.actions,
                tools=gold.tools,
                objects=gold.objects,
                parameters=gold.parameters,
                source_text=gold.source_text,
            )
        if response_model is RelationExtractionResult:
            return RelationExtractionResult(
                uses_tool=gold.uses_tool,
                acts_on_object=gold.acts_on_object,
                action_order=gold.action_order,
                action_causes=gold.action_causes,
                part_of_procedure=gold.part_of_procedure,
                observed_in_scene=gold.observed_in_scene,
            )
        if response_model is MinimalEntityExtractionResult:
            return _minimal_entities_for_gold(gold)
        if response_model is MinimalRelationExtractionResult:
            return _minimal_relations_for_text(text)
        return gold


class HallucinatingRelationExtractor(AlgorithmFakeExtractor):
    """Inject an ungrounded relation during the relation stage."""

    def extract(
        self,
        text: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        result = super().extract(text, response_model, system_prompt)
        if response_model is RelationExtractionResult:
            result.uses_tool.append(
                UsesTool(
                    action=Action(name="measure gap"),
                    tool=Tool(name="laser scanner"),
                )
            )
        return result


class EmptyThenGroundedRelationExtractor:
    """Return one empty relation pass followed by one evidence-grounded pass."""

    def __init__(self) -> None:
        self.relation_calls = 0

    def extract(
        self,
        _text: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        del system_prompt
        if response_model is ControlledEntityDecisionResult:
            return ControlledEntityDecisionResult(
                decisions=[
                    ControlledEntityDecision(
                        entity_id="O1",
                        keep=True,
                        entity_type="Object",
                        name="wood",
                        confidence=0.99,
                    ),
                    ControlledEntityDecision(
                        entity_id="A1",
                        keep=True,
                        entity_type="Action",
                        name="place",
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
                        name="place",
                        evidence_text="place wood.",
                    ),
                    MinimalEntity(
                        entity_id="object_1",
                        entity_type="Object",
                        name="wood",
                        evidence_text="wood",
                    ),
                    MinimalEntity(
                        entity_id="tool_1",
                        entity_type="Tool",
                        name="NONE",
                        evidence_text="place wood.",
                    ),
                ]
            )
        if response_model is MinimalRelationExtractionResult:
            self.relation_calls += 1
            if self.relation_calls == 1:
                return MinimalRelationExtractionResult()
            return MinimalRelationExtractionResult(
                relations=[
                    MinimalRelation(
                        relation_type="ACTS_ON",
                        subject_id="A1",
                        object_id="O1",
                        evidence_text=(
                            "Annotation-derived text: action 'place' involves wood"
                        ),
                    )
                ]
            )
        raise AssertionError(f"Unexpected response model: {response_model}")


def test_two_stage_algorithm_uses_entity_and_relation_schemas() -> None:
    """Entity-first/relation-second should call both stage schemas."""

    extractor = AlgorithmFakeExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("entity_first_relation_second",),
        ),
        extractor=extractor,
    )

    result = runner.run(MVP_BENCHMARK)

    assert len(result.runs) == len(MVP_BENCHMARK.scenes)
    assert EntityExtractionResult in extractor.response_models
    assert RelationExtractionResult in extractor.response_models
    assert result.runs[0].node_metrics.f1 == 1.0


def test_minimal_algorithm_uses_flat_id_based_schemas() -> None:
    """Minimal mode should avoid nested full-relation schemas."""

    extractor = AlgorithmFakeExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("minimal_entity_relation",),
        ),
        extractor=extractor,
    )

    result = runner.run(MVP_BENCHMARK)

    assert MinimalEntityExtractionResult in extractor.response_models
    assert MinimalRelationExtractionResult in extractor.response_models
    assert result.runs[0].graph.edges
    assert result.runs[0].relation_metrics.f1 > 0.0


def test_llm_relation_proposal_condition_has_no_rule_candidate_list() -> None:
    """LLM proposal mode should discover relations over IDs without rule candidates."""

    extractor = AlgorithmFakeExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("llm_relation_proposal",),
        ),
        extractor=extractor,
    )

    result = runner.run(MVP_BENCHMARK)

    assert result.runs[0].execution_error is None
    assert result.runs[0].debug_trace is not None
    assert not result.runs[0].debug_trace.relation_candidates
    assert any(
        prompt and "Discover relations directly" in prompt
        for prompt in extractor.system_prompts
    )
    assert not any(
        prompt and "Candidate relations:" in prompt
        for prompt in extractor.system_prompts
    )


def test_llm_relation_proposal_rechecks_empty_easg_style_result() -> None:
    source = "Annotation-derived text: action 'place' involves wood, right hand."
    controlled = """[NORMALIZED SEGMENT]
source_adapter=easg_adapter
scene_id=easg_1
segment_id=s1

[NORMALIZED ENTITIES]
O1 | OBJECT | name=wood | canonical_name=NONE | semantic_role=target | source_evidence=wood | uncertainty=NONE

[ACTION SEQUENCE]
A1 | ACTION | verb=place | direct_object=O1:wood | tool=NONE | role=NONE | sentence=place wood. | source_evidence=place | uncertainty=NONE

[SOURCE EVIDENCE]
Annotation-derived text: action 'place' involves wood, right hand.
"""
    extractor = EmptyThenGroundedRelationExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("llm_relation_proposal",),
        ),
        extractor=extractor,
    )

    extraction = runner.extract_text(
        controlled,
        source_text=source,
        condition="llm_relation_proposal",
    )

    assert extractor.relation_calls == 2
    assert len(extraction.acts_on_object) == 1
    assert extraction.acts_on_object[0].action.entity_id == "A1"
    assert extraction.acts_on_object[0].object.entity_id == "O1"
    assert runner.last_debug_trace is not None
    assert runner.last_debug_trace.repair_relations


def test_runner_extracts_free_text_without_gold_scene() -> None:
    """Real standardized inputs should be extractable without benchmark gold."""

    scene = MVP_BENCHMARK.get_scene("weld_demo_01")
    extractor = AlgorithmFakeExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("minimal_candidate_relation_with_validation",),
        ),
        extractor=extractor,
    )

    extraction = runner.extract_text(
        scene.input_text("unified"),
        source_text=scene.raw_text,
        condition="minimal_candidate_relation_with_validation",
    )

    assert extraction.source_text == scene.raw_text
    assert extraction.actions
    assert runner.last_debug_trace is not None
    assert runner.last_debug_trace.relation_candidates


def test_minimal_conversion_drops_unknown_relation_ids() -> None:
    """Relations may only reference known entity IDs."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="measure gap",
                evidence_text="measures the gap",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="caliper",
                evidence_text="caliper",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="measures the gap",
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_404",
                object_id="tool_1",
                evidence_text="unsupported",
            ),
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        relations,
        "Maria measures the gap with the caliper.",
    )

    assert len(extraction.uses_tool) == 1
    assert extraction.uses_tool[0].tool.name == "caliper"


def test_minimal_action_name_enrichment_uses_grounded_relation_targets() -> None:
    """Underspecified actions should borrow context only from grounded relations."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="picks up",
                evidence_text="Hans picks up the torch.",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="aligns",
                evidence_text="Hans aligns the steel plate.",
            ),
            MinimalEntity(
                entity_id="action_3",
                entity_type="Action",
                name="measure gap",
                evidence_text="Maria measures the gap.",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="Fronius TPS 400i torch",
                evidence_text="Fronius TPS 400i torch",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="steel plate",
                evidence_text="steel plate",
            ),
            MinimalEntity(
                entity_id="tool_2",
                entity_type="Tool",
                name="caliper",
                evidence_text="caliper",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="Hans picks up the Fronius TPS 400i torch.",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_1",
                object_id="object_1",
                evidence_text=(
                    "Hans picks up the Fronius TPS 400i torch, "
                    "then aligns the steel plate."
                ),
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_1",
                evidence_text="Hans aligns the steel plate on the table.",
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_3",
                object_id="tool_2",
                evidence_text="Maria measures the gap with the caliper.",
            ),
        ]
    )

    enriched = enrich_minimal_action_names(entities, relations)
    names = {entity.entity_id: entity.name for entity in enriched.entities}

    assert names["action_1"] == "pick up welding torch"
    assert names["action_2"] == "align steel plate"
    assert names["action_3"] == "measure gap"


def test_minimal_action_name_enrichment_uses_grounded_action_phrase() -> None:
    """Short start/begin actions may borrow a grounded action phrase target."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="starts",
                evidence_text="Hans starts the root weld.",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="root weld",
                evidence_text="Hans starts the root weld.",
            ),
        ]
    )

    enriched = enrich_minimal_action_names(
        entities,
        MinimalRelationExtractionResult(),
    )
    names = {entity.entity_id: entity.name for entity in enriched.entities}

    assert names["action_1"] == "start root weld"
    assert names["action_2"] == "root weld"


def test_minimal_relation_evidence_filter_drops_unsupported_edges() -> None:
    """Relation evidence must be grounded and mention Action -> target."""

    source_text = (
        "Hans picks up the Fronius TPS 400i torch, "
        "aligns the steel plate on the table, and then starts the root weld."
    )
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="picks up",
                evidence_text=source_text,
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="aligns",
                evidence_text=source_text,
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="Fronius TPS 400i torch",
                evidence_text="Fronius TPS 400i torch",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="steel plate",
                evidence_text="steel plate",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="picks up the Fronius TPS 400i torch",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_1",
                object_id="object_1",
                evidence_text="picks up the steel plate on the table",
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_2",
                object_id="tool_1",
                evidence_text="aligns the steel plate on the table",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_1",
                evidence_text="aligns the steel plate on the table",
            ),
        ]
    )

    filtered = filter_minimal_relations_by_evidence(
        relations,
        entities,
        source_text,
    )
    triples = {
        (relation.relation_type, relation.subject_id, relation.object_id)
        for relation in filtered.relations
    }

    assert triples == {
        ("USES_TOOL", "action_1", "tool_1"),
        ("ACTS_ON", "action_2", "object_1"),
    }


def test_llm_relation_proposals_are_hard_validated_with_rejection_reasons() -> None:
    """Autonomous LLM proposals must pass ID, type, evidence, and dedup checks."""

    source_text = (
        "The worker picks up the screwdriver and then tightens the screw "
        "with the screwdriver. The worker inspects the screw."
    )
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="pick up screwdriver",
                evidence_text="The worker picks up the screwdriver",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="tighten screw",
                evidence_text="tightens the screw with the screwdriver",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="screwdriver",
                evidence_text="screwdriver",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="screw",
                evidence_text="screw",
            ),
        ]
    )
    valid_evidence = (
        "The worker picks up the screwdriver and then tightens the screw "
        "with the screwdriver."
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_2",
                object_id="tool_1",
                evidence_text=valid_evidence,
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_1",
                evidence_text=valid_evidence,
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_404",
                object_id="tool_1",
                evidence_text=valid_evidence,
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="tool_1",
                object_id="action_2",
                evidence_text=valid_evidence,
            ),
            MinimalRelation(
                relation_type="BEFORE",
                subject_id="action_1",
                object_id="action_1",
                evidence_text=valid_evidence,
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_2",
                object_id="tool_1",
                evidence_text="The worker uses a hammer.",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_1",
                evidence_text="The worker inspects the screw.",
            ),
            MinimalRelation(
                relation_type="CAUSES",
                subject_id="action_1",
                object_id="action_2",
                evidence_text=valid_evidence,
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_2",
                object_id="tool_1",
                evidence_text=valid_evidence,
            ),
        ]
    )

    validation = validate_llm_relation_proposals(
        relations,
        entities,
        source_text,
    )

    accepted_triples = {
        (relation.relation_type, relation.subject_id, relation.object_id)
        for relation in validation.accepted.relations
    }
    rejection_codes = {rejection.code for rejection in validation.rejected}

    assert accepted_triples == {
        ("USES_TOOL", "action_2", "tool_1"),
        ("ACTS_ON", "action_2", "object_1"),
    }
    assert rejection_codes == {
        "unknown_entity_id",
        "invalid_domain_range",
        "self_relation",
        "ungrounded_evidence",
        "endpoints_not_in_evidence",
        "missing_causal_cue",
        "duplicate_relation",
    }


def test_hard_validation_rejects_ambiguous_duplicate_entity_ids() -> None:
    """Duplicate entity IDs cannot silently resolve to the last LLM record."""

    source_text = "The worker tightens the screw with a screwdriver."
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="tighten screw",
                evidence_text="tightens the screw",
            ),
            MinimalEntity(
                entity_id="action_1",
                entity_type="Worker",
                name="worker",
                evidence_text="worker",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="screwdriver",
                evidence_text="screwdriver",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text=source_text,
            )
        ]
    )

    validation = validate_llm_relation_proposals(relations, entities, source_text)

    assert not validation.accepted.relations
    assert [item.code for item in validation.rejected] == ["ambiguous_entity_id"]


def test_llm_relation_proposal_prompt_requires_abstention_and_exact_evidence() -> None:
    """The autonomous prompt must constrain IDs, evidence, and uncertain facts."""

    prompt = llm_relation_proposal_stage_prompt(
        "- action_1 | Action | tighten screw\n- tool_1 | Tool | screwdriver"
    )

    assert "No candidate relation list is provided" in prompt
    assert "exact, contiguous" in prompt
    assert "complete clause or sentence" in prompt
    assert "actual use" in prompt
    assert "Common-sense knowledge is not evidence" in prompt
    assert "only to interpret the source wording" in prompt
    assert "omit the relation" in prompt
    assert "Candidate relations:" not in prompt


def test_temporal_relation_accepts_grounded_cue_with_containing_sentence() -> None:
    """A short 'then' quote may use its bounded source sentence to verify endpoints."""

    source_text = (
        "Hans aligns the steel plate on the table,\n"
        "and then starts the root weld."
    )
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="aligns steel plate",
                evidence_text="aligns the steel plate on the table",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="starts root weld",
                evidence_text="starts the root weld",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="BEFORE",
                subject_id="action_1",
                object_id="action_2",
                evidence_text="and then starts the root weld",
            )
        ]
    )

    validation = validate_llm_relation_proposals(relations, entities, source_text)

    assert len(validation.accepted.relations) == 1
    assert not validation.rejected


def test_single_action_sentence_can_complete_short_acts_on_evidence() -> None:
    """A relation-only quote may use an unambiguous one-action source sentence."""

    source_text = "Annotation: action 'place' involves wood, right hand."
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="place",
                evidence_text="action 'place'",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="wood",
                evidence_text="wood",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_1",
                object_id="object_1",
                evidence_text="involves wood",
            )
        ]
    )

    validation = validate_llm_relation_proposals(relations, entities, source_text)

    assert len(validation.accepted.relations) == 1
    assert not validation.rejected


def test_causal_relation_matches_grounded_nominalized_action() -> None:
    """Lexical grounding should match aligns with alignment without using gold."""

    source_text = "The alignment step prepares the plate for the root weld."
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="aligns steel plate",
                evidence_text="alignment step",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="starts root weld",
                evidence_text="root weld",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="CAUSES",
                subject_id="action_1",
                object_id="action_2",
                evidence_text=source_text,
            )
        ]
    )

    validation = validate_llm_relation_proposals(relations, entities, source_text)

    assert len(validation.accepted.relations) == 1
    assert not validation.rejected


def test_llm_relation_proposal_conversion_disables_rule_fallback_edges() -> None:
    """Pure proposal mode must not add BEFORE or tool edges after LLM validation."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="pick up screwdriver",
                evidence_text="The worker picks up the screwdriver.",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="tighten screw",
                evidence_text="The worker tightens the screw.",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="screwdriver",
                evidence_text="screwdriver",
            ),
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        MinimalRelationExtractionResult(),
        "The worker picks up the screwdriver. The worker tightens the screw.",
        add_deterministic_relation_fallbacks=False,
    )

    assert not extraction.uses_tool
    assert not extraction.action_order


def test_proposal_conversion_does_not_auto_attach_single_scene() -> None:
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A1",
                entity_type="Action",
                name="place",
                evidence_text="place",
            ),
            MinimalEntity(
                entity_id="S1",
                entity_type="Scene",
                name="scene one",
                evidence_text="place",
            ),
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        MinimalRelationExtractionResult(),
        "action place",
        add_deterministic_relation_fallbacks=False,
        enrich_action_context=False,
        infer_source_scene=False,
        attach_single_context=False,
    )

    assert extraction.actions[0].scene is None
    assert not extraction.observed_in_scene


def test_llm_relation_proposal_supports_part_of_and_observed_in() -> None:
    """Extended autonomous relations should validate and convert with known IDs."""

    source_text = (
        "During root welding preparation, Hans aligns the steel plate in the scene."
    )
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="align steel plate",
                evidence_text="aligns the steel plate",
            ),
            MinimalEntity(
                entity_id="procedure_1",
                entity_type="Procedure",
                name="root welding preparation",
                evidence_text="root welding preparation",
            ),
            MinimalEntity(
                entity_id="scene_1",
                entity_type="Scene",
                name="welding scene",
                evidence_text="in the scene",
            ),
        ]
    )
    proposals = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="PART_OF",
                subject_id="action_1",
                object_id="procedure_1",
                evidence_text=source_text,
            ),
            MinimalRelation(
                relation_type="OBSERVED_IN",
                subject_id="action_1",
                object_id="scene_1",
                evidence_text="Hans aligns the steel plate in the scene.",
            ),
        ]
    )

    validation = validate_llm_relation_proposals(
        proposals,
        entities,
        source_text,
    )
    extraction = minimal_to_egocentric_extraction(
        entities,
        validation.accepted,
        source_text,
        add_deterministic_relation_fallbacks=False,
    )

    assert not validation.rejected
    assert len(extraction.part_of_procedure) == 1
    assert len(extraction.observed_in_scene) == 1


def test_llm_relation_proposal_repair_can_correct_invalid_endpoint_role() -> None:
    """One constrained repair pass may replace a Worker endpoint with an Action ID."""

    class RepairingExtractor:
        def __init__(self) -> None:
            self.relation_calls = 0

        def extract(self, text, response_model, system_prompt=None):
            if response_model is MinimalEntityExtractionResult:
                return MinimalEntityExtractionResult(
                    entities=[
                        MinimalEntity(
                            entity_id="action_1",
                            entity_type="Action",
                            name="tighten screw",
                            evidence_text="tightens the screw",
                        ),
                        MinimalEntity(
                            entity_id="tool_1",
                            entity_type="Tool",
                            name="screwdriver",
                            evidence_text="screwdriver",
                        ),
                        MinimalEntity(
                            entity_id="worker_1",
                            entity_type="Worker",
                            name="worker",
                            evidence_text="worker",
                        ),
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                self.relation_calls += 1
                subject_id = "worker_1" if self.relation_calls == 1 else "action_1"
                return MinimalRelationExtractionResult(
                    relations=[
                        MinimalRelation(
                            relation_type="USES_TOOL",
                            subject_id=subject_id,
                            object_id="tool_1",
                            evidence_text=(
                                "The worker tightens the screw with a screwdriver."
                            ),
                        )
                    ]
                )
            raise AssertionError(response_model)

    extractor = RepairingExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("llm_relation_proposal_with_repair",),
        ),
        extractor=extractor,
    )
    extraction = runner.extract_text(
        "The worker tightens the screw with a screwdriver.",
        condition="llm_relation_proposal_with_repair",
    )

    assert extractor.relation_calls == 2
    assert len(extraction.uses_tool) == 1
    assert runner.last_debug_trace is not None
    assert runner.last_debug_trace.repair_relations
    assert runner.last_debug_trace.rejected_relations[0].code == "invalid_domain_range"


def test_minimal_conversion_enriches_action_names_in_graph() -> None:
    """Converted graph nodes should keep enriched names and original entity IDs."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="pick up",
                evidence_text="Hans picks up the torch.",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="torch",
                evidence_text="torch",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="Hans picks up the torch.",
            )
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        relations,
        "Hans picks up the torch.",
    )
    graph = build_property_graph(extraction)
    action_nodes = [
        node for node in graph.nodes.values() if node.label == "Action"
    ]

    assert extraction.actions[0].name == "pick up torch"
    assert action_nodes[0].name == "pick up torch"
    assert action_nodes[0].properties["entity_id"] == "action_1"


def test_minimal_conversion_adds_recall_relations_without_fragments() -> None:
    """Deterministic recall augmentation should add high-confidence graph facts."""

    source_text = (
        "Video weld_demo_01, segment s1, 00:00-00:20. "
        "Hans picks up the Fronius TPS 400i torch, aligns the steel plate "
        "on the table, and then starts the root weld. "
        "The alignment step prepares the plate for the root weld."
    )
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="worker_1",
                entity_type="Worker",
                name="Hans",
                evidence_text="Hans picks up the Fronius TPS 400i torch",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="Fronius TPS 400i torch",
                evidence_text="Fronius TPS 400i torch",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="steel plate",
                evidence_text="steel plate",
            ),
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="picks up",
                evidence_text=source_text,
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="aligns",
                evidence_text=source_text,
            ),
            MinimalEntity(
                entity_id="action_3",
                entity_type="Action",
                name="starts",
                evidence_text=source_text,
            ),
            MinimalEntity(
                entity_id="weld_1",
                entity_type="Action",
                name="root weld",
                evidence_text=source_text,
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="picks up the Fronius TPS 400i torch",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_1",
                evidence_text="aligns the steel plate on the table",
            ),
            MinimalRelation(
                relation_type="BEFORE",
                subject_id="action_1",
                object_id="action_2",
                evidence_text="adjacent actions in extracted order",
            ),
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        relations,
        source_text,
    )
    graph = build_property_graph(extraction)
    action_names = [action.name for action in extraction.actions]
    edge_facts = {
        (
            edge.type,
            graph.node(edge.source).name,
            graph.node(edge.target).name,
        )
        for edge in graph.edges
    }

    assert action_names == [
        "pick up welding torch",
        "align steel plate",
        "start root weld",
    ]
    assert all(action.actor and action.actor.name == "Hans" for action in extraction.actions)
    assert all(
        action.scene and action.scene.name == "weld_demo_01 segment s1"
        for action in extraction.actions
    )
    assert [
        (relation.before.name, relation.after.name)
        for relation in extraction.action_order
    ] == [
        ("pick up welding torch", "align steel plate"),
        ("align steel plate", "start root weld"),
    ]
    assert extraction.action_order[0].evidence_text == source_text
    assert (
        "USES_TOOL",
        "start root weld",
        "Fronius TPS 400i torch",
    ) in edge_facts
    assert (
        "PERFORMED_BY",
        "align steel plate",
        "Hans",
    ) in edge_facts
    assert (
        "OBSERVED_IN",
        "start root weld",
        "weld_demo_01 segment s1",
    ) in edge_facts
    assert (
        "CAUSES",
        "align steel plate",
        "start root weld",
    ) in edge_facts
    assert all("root weld" != fact[1] for fact in edge_facts)


def test_tool_carryover_only_adds_semantic_matches() -> None:
    """Held tools should not be copied to unrelated later actions."""

    source_text = (
        "Video weld_demo_01, segment s1, 00:00-00:20. "
        "Hans picks up the Fronius TPS 400i torch and aligns the steel plate."
    )
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="Fronius TPS 400i torch",
                evidence_text="Fronius TPS 400i torch",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="steel plate",
                evidence_text="steel plate",
            ),
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="picks up",
                evidence_text=source_text,
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="aligns",
                evidence_text=source_text,
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="picks up the Fronius TPS 400i torch",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_1",
                evidence_text="aligns the steel plate",
            ),
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        relations,
        source_text,
    )

    assert [
        relation.action.name for relation in extraction.uses_tool
    ] == ["pick up welding torch"]


def test_minimal_entity_inventory_is_compact() -> None:
    """Relation prompts should receive compact ontology snippets."""

    inventory = minimal_entity_inventory_text(
        MinimalEntityExtractionResult(
            entities=[
                MinimalEntity(
                    entity_id="action_1",
                    entity_type="Action",
                    name="pick up torch",
                    evidence_text="picks up the torch",
                )
            ]
        )
    )

    assert inventory == (
        "- action_1 | Action | pick up torch | evidence=picks up the torch"
    )


def test_controlled_entity_reconciliation_reuses_ids_and_drops_none() -> None:
    source = "Annotation-derived text: action 'place' involves wood, right hand."
    controlled = """[NORMALIZED SEGMENT]
source_adapter=easg_adapter
scene_id=easg_1
segment_id=s1

[NORMALIZED ENTITIES]
O1 | OBJECT | name=wood | canonical_name=NONE | semantic_role=target | source_evidence=wood | uncertainty=NONE

[ACTION SEQUENCE]
A1 | ACTION | verb=place | direct_object=O1:wood | tool=NONE | role=NONE | sentence=place wood. | source_evidence=place | uncertainty=NONE

[SOURCE EVIDENCE]
Annotation-derived text: action 'place' involves wood, right hand.
"""
    extracted = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="place",
                evidence_text="place wood.",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="wood",
                evidence_text="wood",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="NONE",
                evidence_text="place wood.",
            ),
        ]
    )

    reconciled, aliases = reconcile_minimal_entities_with_controlled_text(
        extracted,
        controlled,
        source,
    )

    assert [entity.entity_id for entity in reconciled.entities] == ["A1", "O1"]
    assert reconciled.entities[0].evidence_text == "place"
    assert aliases["action_1"] == "A1"
    assert aliases["object_1"] == "O1"
    assert "tool_1" not in aliases


def test_controlled_action_name_is_not_expanded_from_relation_target() -> None:
    """Renderer-owned action names must remain stable in strict relation metrics."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A1",
                entity_type="Action",
                name="place",
                evidence_text="place",
            ),
            MinimalEntity(
                entity_id="O1",
                entity_type="Object",
                name="wood",
                evidence_text="wood",
            ),
        ]
    )
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="A1",
                object_id="O1",
                evidence_text="action 'place' involves wood",
            )
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        relations,
        "action 'place' involves wood",
        add_deterministic_relation_fallbacks=False,
        enrich_action_context=False,
    )

    assert extraction.actions[0].name == "place"
    assert extraction.acts_on_object[0].action.name == "place"


def test_controlled_action_name_drops_only_stray_single_letter_prefix() -> None:
    """A format typo must not hide an otherwise grounded action endpoint."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A1",
                entity_type="Action",
                name="t tightens torque wrench",
                evidence_text="tightens the bolt with the torque wrench",
            )
        ]
    )

    extraction = minimal_to_egocentric_extraction(
        entities,
        MinimalRelationExtractionResult(),
        "The operator tightens the bolt with the torque wrench.",
        add_deterministic_relation_fallbacks=False,
        enrich_action_context=False,
        infer_source_scene=False,
        attach_single_context=False,
    )

    assert [action.name for action in extraction.actions] == [
        "tighten torque wrench"
    ]


def test_controlled_stage_views_remove_only_duplicated_evidence() -> None:
    controlled = """[NORMALIZED SEGMENT]
source_adapter=test
scene_id=s1
segment_id=g1

[NORMALIZED ENTITIES]
O1 | OBJECT | name=wood | canonical_name=NONE | semantic_role=target | source_evidence=wood | uncertainty=NONE

[ACTION SEQUENCE]
A1 | ACTION | verb=place | direct_object=O1:wood | tool=NONE | role=NONE | sentence=place wood. | source_evidence=place | uncertainty=NONE

[SOURCE EVIDENCE]
Annotation-derived text: action 'place' involves wood.

[UNCERTAINTY]
NONE
"""

    entity_view = controlled_entity_extraction_input(controlled)
    relation_view = controlled_relation_proposal_input(controlled)

    assert entity_view == (
        "[CONTROLLED ENTITY EXTRACTION VIEW]\n"
        "O1 | Object | name=wood\n"
        "A1 | Action | name=place"
    )
    assert "direct_object_id=O1" in relation_view
    assert "direct_object_name=wood" in relation_view
    assert "O1:wood" not in relation_view
    assert "sentence=" not in relation_view
    assert "source_evidence=" not in relation_view
    assert "Annotation-derived text: action 'place' involves wood." in relation_view
    assert "[UNCERTAINTY]" not in relation_view


def test_id_name_normalization_requires_known_id_and_writes_log() -> None:
    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="A1:hold",
                object_id="O4:horizontal bar",
                evidence_text="hold the horizontal bar",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="A1",
                object_id="O9:invented",
                evidence_text="hold the horizontal bar",
            ),
        ]
    )

    normalized, logs = normalize_minimal_relation_ids(
        relations,
        {"A1": "A1", "O4": "O4"},
        known_entity_ids={"A1", "O4"},
    )

    assert normalized.relations[0].subject_id == "A1"
    assert normalized.relations[0].object_id == "O4"
    assert normalized.relations[1].object_id == "O9:invented"
    assert [(item.original_id, item.normalized_id) for item in logs] == [
        ("A1:hold", "A1"),
        ("O4:horizontal bar", "O4"),
    ]


def test_controlled_entity_stage_reviews_every_missing_renderer_id() -> None:
    class MissingThenCompleteExtractor:
        def __init__(self) -> None:
            self.decision_calls = 0

        def extract(self, _text, response_model, system_prompt=None):
            del system_prompt
            if response_model is ControlledEntityDecisionResult:
                self.decision_calls += 1
                if self.decision_calls == 1:
                    return ControlledEntityDecisionResult(
                        decisions=[
                            ControlledEntityDecision(
                                entity_id="A1",
                                keep=True,
                                entity_type="Action",
                                name="hold horizontal bar",
                            )
                        ]
                    )
                return ControlledEntityDecisionResult(
                    decisions=[
                        ControlledEntityDecision(
                            entity_id="O1",
                            keep=True,
                            entity_type="Object",
                            name="horizontal bar",
                        )
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                return MinimalRelationExtractionResult(
                    relations=[
                        MinimalRelation(
                            relation_type="ACTS_ON",
                            subject_id="A1",
                            object_id="O1",
                            evidence_text="Hold the horizontal bar.",
                        )
                    ]
                )
            raise AssertionError(response_model)

    controlled = """[NORMALIZED SEGMENT]
source_adapter=test
scene_id=s1
segment_id=g1

[NORMALIZED ENTITIES]
O1 | OBJECT | name=horizontal bar | canonical_name=NONE | semantic_role=target | source_evidence=horizontal bar | uncertainty=NONE

[ACTION SEQUENCE]
A1 | ACTION | verb=hold | direct_object_id=O1 | direct_object_name=horizontal bar | tool_id=NONE | tool_name=NONE | role_id=NONE | role_name=NONE | sentence=hold horizontal bar. | source_evidence=Hold the horizontal bar. | uncertainty=NONE

[SOURCE EVIDENCE]
Hold the horizontal bar.
"""
    extractor = MissingThenCompleteExtractor()
    runner = AlgorithmExperimentRunner(extractor=extractor)

    extraction = runner.extract_text(
        controlled,
        source_text="Hold the horizontal bar.",
        condition="llm_relation_proposal",
    )

    assert extractor.decision_calls == 2
    assert len(extraction.acts_on_object) == 1
    assert runner.last_debug_trace is not None
    assert not runner.last_debug_trace.missing_entity_decision_ids
    assert {item.entity_id for item in runner.last_debug_trace.entity_decisions} == {
        "A1",
        "O1",
    }


def test_relation_contracts_open_only_when_endpoint_types_exist() -> None:
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A1",
                entity_type="Action",
                name="hold bar",
                evidence_text="hold bar",
            ),
            MinimalEntity(
                entity_id="O1",
                entity_type="Object",
                name="bar",
                evidence_text="bar",
            ),
        ]
    )

    contracts = available_relation_contracts(entities)
    prompt = llm_relation_proposal_stage_prompt(
        minimal_entity_inventory_text(entities),
        contracts,
    )

    assert contracts == ("ACTS_ON",)
    assert "ACTS_ON Action->Object" in prompt
    assert "PART_OF Action->Procedure" not in prompt
    assert "OBSERVED_IN Action->Scene" not in prompt


def test_local_pronoun_resolution_requires_adjacent_antecedent() -> None:
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A4",
                entity_type="Action",
                name="hold horizontal bar",
                evidence_text="Can you hold it?",
            ),
            MinimalEntity(
                entity_id="O4",
                entity_type="Object",
                name="horizontal bar",
                evidence_text="this horizontal bar",
            ),
        ]
    )
    relation = MinimalRelation(
        relation_type="ACTS_ON",
        subject_id="A4",
        object_id="O4",
        relation_evidence="Can you hold it?",
        antecedent_evidence="This horizontal bar.",
        coreference="it -> O4",
    )

    adjacent = validate_llm_relation_proposals(
        MinimalRelationExtractionResult(relations=[relation]),
        entities,
        "This horizontal bar. Can you hold it?",
    )
    distant = validate_llm_relation_proposals(
        MinimalRelationExtractionResult(relations=[relation]),
        entities,
        "This horizontal bar. Put the screws away. Can you hold it?",
    )

    assert len(adjacent.accepted.relations) == 1
    assert not distant.accepted.relations
    assert distant.rejected[0].code == "invalid_reference_resolution"


def test_coreference_chain_requires_both_quotes_from_source_and_known_target() -> None:
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A4",
                entity_type="Action",
                name="hold horizontal bar",
                evidence_text="Can you hold it?",
            ),
            MinimalEntity(
                entity_id="O4",
                entity_type="Object",
                name="horizontal bar",
                evidence_text="This is the horizontal bar.",
            ),
        ]
    )
    source = "This is the horizontal bar. Can you hold it?"

    ungrounded_antecedent = MinimalRelation(
        relation_type="ACTS_ON",
        subject_id="A4",
        object_id="O4",
        relation_evidence="Can you hold it?",
        antecedent_evidence="That is the horizontal bar.",
        coreference="it -> O4",
    )
    unknown_target = ungrounded_antecedent.model_copy(
        update={
            "object_id": "O9",
            "antecedent_evidence": "This is the horizontal bar.",
            "coreference": "it -> O9",
        }
    )

    ungrounded_result = validate_llm_relation_proposals(
        MinimalRelationExtractionResult(relations=[ungrounded_antecedent]),
        entities,
        source,
    )
    unknown_result = validate_llm_relation_proposals(
        MinimalRelationExtractionResult(relations=[unknown_target]),
        entities,
        source,
    )

    assert ungrounded_result.rejected[0].code == "invalid_reference_resolution"
    assert unknown_result.rejected[0].code == "unknown_entity_id"


def test_ambiguous_coreference_requires_high_explicit_confidence() -> None:
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="A4",
                entity_type="Action",
                name="hold horizontal bar",
                evidence_text="Can you hold it?",
            ),
            MinimalEntity(
                entity_id="O4",
                entity_type="Object",
                name="horizontal bar",
                evidence_text="horizontal bar",
            ),
            MinimalEntity(
                entity_id="O5",
                entity_type="Object",
                name="vertical bar",
                evidence_text="vertical bar",
            ),
        ]
    )
    source = (
        "This is the horizontal bar beside the vertical bar. Can you hold it?"
    )
    ambiguous = MinimalRelation(
        relation_type="ACTS_ON",
        subject_id="A4",
        object_id="O4",
        relation_evidence="Can you hold it?",
        antecedent_evidence=(
            "This is the horizontal bar beside the vertical bar."
        ),
        coreference="it -> O4",
        coreference_confidence=0.79,
    )

    rejected = validate_llm_relation_proposals(
        MinimalRelationExtractionResult(relations=[ambiguous]),
        entities,
        source,
    )
    accepted = validate_llm_relation_proposals(
        MinimalRelationExtractionResult(
            relations=[ambiguous.model_copy(update={"coreference_confidence": 0.9})]
        ),
        entities,
        source,
    )

    assert rejected.rejected[0].code == "ambiguous_reference_resolution"
    assert len(accepted.accepted.relations) == 1


def test_minimal_relation_serializes_new_coreference_contract() -> None:
    relation = MinimalRelation(
        relation_type="ACTS_ON",
        subject_id="A4",
        object_id="O4",
        relation_evidence="Can you hold it?",
        antecedent_evidence="This is the horizontal bar.",
        coreference="it -> O4",
    )

    payload = relation.model_dump(mode="json")

    assert payload["relation_evidence"] == "Can you hold it?"
    assert payload["coreference"] == "it -> O4"
    assert "evidence_text" not in payload
    assert "resolved_reference" not in payload


def test_renderer_action_name_uses_source_evidence_granularity() -> None:
    controlled = """[NORMALIZED SEGMENT]
source_adapter=test
scene_id=s1
segment_id=g1

[NORMALIZED ENTITIES]
O1 | OBJECT | name=bolt | canonical_name=NONE | semantic_role=target | source_evidence=bolt | uncertainty=NONE
T1 | TOOL | name=screwdriver | canonical_name=NONE | semantic_role=instrument | source_evidence=screwdriver | uncertainty=NONE

[ACTION SEQUENCE]
A1 | ACTION | verb=unscrew | direct_object=O1:bolt | tool=T1:screwdriver | role=NONE | sentence=unscrew bolt with screwdriver. | source_evidence=The worker unscrews the bolt with a screwdriver. | uncertainty=NONE

[SOURCE EVIDENCE]
The worker unscrews the bolt with a screwdriver.
"""

    entity_view = controlled_entity_extraction_input(controlled)

    assert "A1 | Action | name=unscrew bolt with screwdriver" in entity_view


def test_long_relation_inventory_is_grouped_by_subject_not_candidate_edge() -> None:
    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id=f"A{index}",
                entity_type="Action",
                name=f"action {index}",
                evidence_text=f"action {index}",
            )
            for index in range(1, 13)
        ]
    )

    groups = relation_subject_focus_groups(entities)
    instruction = relation_focus_instruction(
        groups[0],
        group_index=1,
        group_count=len(groups),
    )

    assert [len(group) for group in groups] == [4, 4, 4]
    assert "A1, A2, A3, A4" in instruction
    assert "object_id may be any compatible known entity ID" in instruction


def test_single_relation_group_still_gets_per_action_checklist() -> None:
    instruction = relation_focus_instruction(
        ("A1", "A2"),
        group_index=1,
        group_count=1,
    )

    assert "[SUBJECT FOCUS 1/1]" in instruction
    assert "Review every focused action independently" in instruction
    assert "relation_evidence" in instruction
    assert "antecedent_evidence" in instruction
    assert "coreference" in instruction
    assert "it -> O4" in instruction


def test_failed_relation_repair_keeps_first_pass_accepted_relation() -> None:
    class RepairTimeoutExtractor:
        def __init__(self) -> None:
            self.relation_calls = 0

        def extract(self, _text, response_model, system_prompt=None):
            del system_prompt
            if response_model is MinimalEntityExtractionResult:
                return MinimalEntityExtractionResult(
                    entities=[
                        MinimalEntity(
                            entity_id="action_1",
                            entity_type="Action",
                            name="tighten screw",
                            evidence_text="tightens the screw",
                        ),
                        MinimalEntity(
                            entity_id="tool_1",
                            entity_type="Tool",
                            name="screwdriver",
                            evidence_text="screwdriver",
                        ),
                        MinimalEntity(
                            entity_id="worker_1",
                            entity_type="Worker",
                            name="worker",
                            evidence_text="worker",
                        ),
                    ]
                )
            if response_model is MinimalRelationExtractionResult:
                self.relation_calls += 1
                if self.relation_calls == 2:
                    raise TimeoutError("repair timed out")
                return MinimalRelationExtractionResult(
                    relations=[
                        MinimalRelation(
                            relation_type="USES_TOOL",
                            subject_id="action_1",
                            object_id="tool_1",
                            evidence_text=(
                                "The worker tightens the screw with a screwdriver."
                            ),
                        ),
                        MinimalRelation(
                            relation_type="USES_TOOL",
                            subject_id="worker_1",
                            object_id="tool_1",
                            evidence_text=(
                                "The worker tightens the screw with a screwdriver."
                            ),
                        ),
                    ]
                )
            raise AssertionError(response_model)

    extractor = RepairTimeoutExtractor()
    runner = AlgorithmExperimentRunner(extractor=extractor)

    extraction = runner.extract_text(
        "The worker tightens the screw with a screwdriver.",
        condition="llm_relation_proposal_with_repair",
    )

    assert len(extraction.uses_tool) == 1
    assert runner.last_debug_trace is not None
    assert "Relation repair failed" in runner.last_debug_trace.stage_warnings[0]


def test_empty_relation_review_prompt_keeps_llm_judgment_bounded() -> None:
    prompt = llm_empty_relation_review_stage_prompt(
        "- A1 | Action | place | evidence=place\n"
        "- O1 | Object | wood | evidence=wood"
    )

    assert "first relation proposal was empty" in prompt
    assert "direct_object_id=O" in prompt
    assert "Do not create or rename IDs" in prompt


def test_relation_candidate_generator_uses_legal_domain_ranges() -> None:
    """Candidate generation should only emit allowed endpoint types."""

    entities = MinimalEntityExtractionResult(
        entities=[
            MinimalEntity(
                entity_id="action_1",
                entity_type="Action",
                name="pick up torch",
                evidence_text="picks up the torch",
            ),
            MinimalEntity(
                entity_id="action_2",
                entity_type="Action",
                name="start weld",
                evidence_text="starts the root weld",
            ),
            MinimalEntity(
                entity_id="tool_1",
                entity_type="Tool",
                name="torch",
                evidence_text="torch",
            ),
            MinimalEntity(
                entity_id="object_1",
                entity_type="Object",
                name="steel plate",
                evidence_text="steel plate",
            ),
        ]
    )

    candidates = generate_relation_candidates(
        entities,
        "Picking up the torch prepares the worker to start the weld.",
    )
    triples = {
        (candidate.relation_type, candidate.subject_id, candidate.object_id)
        for candidate in candidates
    }

    assert ("USES_TOOL", "action_1", "tool_1") in triples
    assert ("ACTS_ON", "action_1", "object_1") in triples
    assert ("BEFORE", "action_1", "action_2") in triples
    assert ("CAUSES", "action_1", "action_2") in triples
    assert all(not triple[1].startswith("tool_") for triple in triples)


def test_candidate_prompt_includes_allowed_relations_and_candidates() -> None:
    """Candidate prompt should constrain the relation stage."""

    candidate_text = "- c1 | USES_TOOL | action_1 -> tool_1 | action uses tool"
    prompt = minimal_candidate_relation_stage_prompt(
        "- action_1 | Action | pick up torch\n- tool_1 | Tool | torch",
        candidate_text,
    )

    assert "Allowed relation_type values and endpoints" in prompt
    assert "USES_TOOL: Action -> Tool" in prompt
    assert "Few-shot example" in prompt
    assert candidate_text in prompt


def test_candidate_filter_drops_relations_outside_candidate_list() -> None:
    """LLM output outside deterministic candidates should be ignored."""

    relations = MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="uses torch",
            ),
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_2",
                object_id="tool_1",
                evidence_text="unsupported candidate",
            ),
        ]
    )
    filtered = filter_minimal_relations_by_candidates(
        relations,
        [
            MinimalRelationCandidate(
                candidate_id="c1",
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
            )
        ],
    )

    assert len(filtered.relations) == 1
    assert filtered.relations[0].subject_id == "action_1"


def test_minimal_candidate_algorithm_records_trace_and_entity_ids() -> None:
    """Candidate mode should preserve relation debug data and node entity IDs."""

    extractor = AlgorithmFakeExtractor()
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("minimal_candidate_relation_with_validation",),
        ),
        extractor=extractor,
    )

    result = runner.run(MVP_BENCHMARK)
    first_run = result.runs[0]
    action_nodes = [
        node for node in first_run.graph.nodes.values() if node.label == "Action"
    ]

    assert first_run.debug_trace is not None
    assert first_run.debug_trace.relation_candidates
    assert first_run.debug_trace.minimal_relations
    assert first_run.debug_trace.filtered_relations
    assert any(node.properties.get("entity_id") for node in action_nodes)
    assert any(
        prompt and "Candidate relations" in prompt
        for prompt in extractor.system_prompts
    )


def test_validation_condition_filters_ungrounded_relation() -> None:
    """Validation-enabled algorithm variants should remove unsupported edges."""

    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            conditions=("entity_first_relation_second_with_validation",),
        ),
        extractor=HallucinatingRelationExtractor(),
    )

    result = runner.run(MVP_BENCHMARK)
    details = "\n".join(
        f"{edge.type}:{result_run.graph.node(edge.target).name}"
        for result_run in result.runs
        for edge in result_run.graph.edges
    )

    assert "laser scanner" not in details
    assert all(run.validation.filtered_relation_count == 0 for run in result.runs)


def test_algorithm_runner_can_evaluate_cross_validation_fold() -> None:
    """Algorithm runner should plug into the generic cross-validation module."""

    config = CrossValidationConfig(
        conditions=("baseline_one_stage",),
        split_strategy="leave_one_scene_out",
    )
    fold = make_folds(MVP_BENCHMARK, config)[0]
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(conditions=("baseline_one_stage",)),
        extractor=AlgorithmFakeExtractor(),
    )

    fold_results = runner.evaluate_fold(MVP_BENCHMARK, fold, config)

    assert len(fold_results) == 1
    assert fold_results[0].test_scene_ids == fold.test_scene_ids
    assert fold_results[0].condition == "baseline_one_stage"
    assert fold_results[0].schema_success_rate == 1.0


def _minimal_entities_for_gold(gold) -> MinimalEntityExtractionResult:
    entities: list[MinimalEntity] = []
    for index, action in enumerate(gold.actions, start=1):
        entities.append(
            MinimalEntity(
                entity_id=f"action_{index}",
                entity_type="Action",
                name=action.name,
                evidence_text=action.evidence_text
                or action.source_text
                or action.name,
            )
        )
    for index, tool in enumerate(gold.tools, start=1):
        entities.append(
            MinimalEntity(
                entity_id=f"tool_{index}",
                entity_type="Tool",
                name=tool.name,
                evidence_text=tool.evidence_text or tool.source_text or tool.name,
            )
        )
    for index, obj in enumerate(gold.objects, start=1):
        entities.append(
            MinimalEntity(
                entity_id=f"object_{index}",
                entity_type="Object",
                name=obj.name,
                evidence_text=obj.evidence_text or obj.source_text or obj.name,
            )
        )
    return MinimalEntityExtractionResult(entities=entities)


def _minimal_relations_for_text(text: str) -> MinimalRelationExtractionResult:
    if "weld_demo_01" in text:
        return MinimalRelationExtractionResult(
            relations=[
                MinimalRelation(
                    relation_type="USES_TOOL",
                    subject_id="action_1",
                    object_id="tool_1",
                    evidence_text="Hans picks up the Fronius TPS 400i torch",
                ),
                MinimalRelation(
                    relation_type="ACTS_ON",
                    subject_id="action_2",
                    object_id="object_1",
                    evidence_text="aligns the steel plate",
                ),
                MinimalRelation(
                    relation_type="BEFORE",
                    subject_id="action_1",
                    object_id="action_2",
                    evidence_text="picks up ... aligns",
                ),
            ]
        )
    return MinimalRelationExtractionResult(
        relations=[
            MinimalRelation(
                relation_type="USES_TOOL",
                subject_id="action_1",
                object_id="tool_1",
                evidence_text="places the caliper on the bracket",
            ),
            MinimalRelation(
                relation_type="ACTS_ON",
                subject_id="action_2",
                object_id="object_2",
                evidence_text="measures the gap",
            ),
        ]
    )
