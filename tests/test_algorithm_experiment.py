"""Algorithm-selection experiment tests."""

from pydantic import BaseModel

from backend.datasets.benchmark import MVP_BENCHMARK
from backend.evaluation.cross_validation import CrossValidationConfig, make_folds
from backend.graph.property_graph import build_property_graph
from backend.pipeline.algorithm_experiment import (
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
    EntityExtractionResult,
    MinimalEntity,
    MinimalEntityExtractionResult,
    MinimalRelation,
    MinimalRelationCandidate,
    MinimalRelationExtractionResult,
    RelationExtractionResult,
    enrich_minimal_action_names,
    filter_minimal_relations_by_evidence,
    filter_minimal_relations_by_candidates,
    generate_relation_candidates,
    minimal_candidate_relation_stage_prompt,
    minimal_entity_inventory_text,
    minimal_to_egocentric_extraction,
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

    assert inventory == "- action_1 | Action | pick up torch"


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
