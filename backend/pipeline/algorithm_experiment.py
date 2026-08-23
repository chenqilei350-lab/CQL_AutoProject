"""Algorithm-selection experiments for text-to-KG extraction.

This module is intentionally separate from data cleaning and unified-text
conversion.  It compares extraction strategies first, then later preprocessing
modules can be plugged in as input conditions.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import AliasChoices, BaseModel, Field

from backend.datasets.benchmark import BenchmarkDataset, BenchmarkScene, InputCondition
from backend.evaluation.cross_validation import (
    CrossValidationConfig,
    DatasetFold,
    FoldConditionResult,
    subset_dataset,
)
from backend.evaluation.metrics import PRF1, aggregate_prf1, compute_prf1
from backend.evaluation.ontology_validation import (
    ValidationReport,
    validate_egocentric_extraction,
)
from backend.graph.property_graph import PropertyGraph, build_property_graph, normalize_name
from backend.llm.client import DEFAULT_MODEL, DEFAULT_REQUEST_TIMEOUT_SECONDS
from backend.pipeline.experiment_runner import ExtractionBackend
from backend.preprocessing.normalized_segment import NormalizedSegment
from backend.schemas.egocentric_video import (
    Action,
    ActionCauses,
    ActionObservedInScene,
    ActionOrder,
    ActionPartOfProcedure,
    ActsOnObject,
    EgocentricVideoExtraction,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.ontology import build_egocentric_ontology
from backend.schemas.process_knowledge.entities import (
    ProcessParameter,
    Procedure,
    Tool,
    Worker,
)


AlgorithmCondition = Literal[
    "baseline_one_stage",
    "strict_schema_prompt",
    "entity_first_relation_second",
    "entity_first_relation_second_with_validation",
    "entity_first_relation_second_with_fewshot",
    "minimal_entity_relation",
    "minimal_entity_relation_with_validation",
    "llm_relation_proposal",
    "llm_relation_proposal_with_repair",
    "minimal_candidate_relation",
    "minimal_candidate_relation_with_validation",
    "validation_driven_refinement",
]

MinimalEntityType = Literal[
    "Action",
    "Tool",
    "Object",
    "Worker",
    "Parameter",
    "Scene",
    "Procedure",
]

MinimalRelationType = Literal[
    "USES_TOOL",
    "ACTS_ON",
    "BEFORE",
    "CAUSES",
    "PART_OF",
    "OBSERVED_IN",
]

CAUSAL_CUES = (
    "cause",
    "causes",
    "caused",
    "enable",
    "enables",
    "enabled",
    "allow",
    "allows",
    "allowed",
    "prepare",
    "prepares",
    "prepared",
)
SCENE_HEADER_RE = re.compile(
    r"Video\s+(?P<video_id>[\w.-]+),\s*segment\s+"
    r"(?P<segment_id>[\w.-]+),\s*"
    r"(?P<start>\d{1,2}:\d{2})-(?P<end>\d{1,2}:\d{2})",
    re.IGNORECASE,
)

ACTION_CONTEXT_RELATIONS: tuple[MinimalRelationType, ...] = ("ACTS_ON", "USES_TOOL")
COREFERENCE_CONFIDENCE_THRESHOLD = 0.8
SUPPORTED_COREFERENCE_EXPRESSIONS = {
    "it",
    "this",
    "that",
    "this one",
    "that one",
    "them",
    "these",
    "those",
}
TOOL_CONTEXT_ACTION_TOKENS = {
    "grab",
    "grasp",
    "hold",
    "pick",
    "take",
    "use",
}
OBJECT_CONTEXT_ACTION_TOKENS = {
    "align",
    "attach",
    "clean",
    "inspect",
    "measure",
    "place",
    "point",
    "position",
    "read",
    "record",
    "switch",
    "tighten",
    "wipe",
}
ACTION_PHRASE_TARGET_ACTION_TOKENS = {
    "begin",
    "start",
}
ACTION_INITIAL_TOKEN_NORMALIZATION = {
    "aligns": "align",
    "attaches": "attach",
    "checks": "check",
    "cleans": "clean",
    "grabs": "grab",
    "grasps": "grasp",
    "holds": "hold",
    "inspects": "inspect",
    "measures": "measure",
    "picks": "pick",
    "places": "place",
    "points": "point",
    "positions": "position",
    "reads": "read",
    "records": "record",
    "starts": "start",
    "switches": "switch",
    "takes": "take",
    "tightens": "tighten",
    "uses": "use",
    "wipes": "wipe",
}
ACTION_EVENT_INITIAL_TOKENS = (
    set(ACTION_INITIAL_TOKEN_NORMALIZATION.values())
    | TOOL_CONTEXT_ACTION_TOKENS
    | OBJECT_CONTEXT_ACTION_TOKENS
    | ACTION_PHRASE_TARGET_ACTION_TOKENS
)
TOOL_ACTION_COMPATIBILITY: tuple[tuple[set[str], set[str]], ...] = (
    ({"torch"}, {"weld"}),
    ({"wrench"}, {"tighten", "bolt"}),
    ({"thermometer"}, {"point", "read", "temperature"}),
    ({"caliper"}, {"measure", "gap"}),
    ({"cloth"}, {"wipe", "clean"}),
)
TOOL_ACTION_TARGET_CANONICAL_TYPES: tuple[tuple[set[str], str, set[str]], ...] = (
    ({"torch"}, "welding torch", {"welding", "fronius", "tps", "400i"}),
    ({"torque", "wrench"}, "torque wrench", {"torque", "wrench"}),
    (
        {"infrared", "thermometer"},
        "infrared thermometer",
        {"infrared", "thermometer"},
    ),
    ({"caliper"}, "caliper", {"caliper"}),
    ({"cleaning", "cloth"}, "cleaning cloth", {"cleaning", "cloth"}),
)
ACTION_NAME_PARTICLES = {
    "a",
    "an",
    "and",
    "at",
    "down",
    "in",
    "into",
    "of",
    "off",
    "on",
    "onto",
    "out",
    "the",
    "then",
    "to",
    "up",
    "with",
}
MIN_TARGET_TOKEN_LENGTH = 3
ACTION_TARGET_EVIDENCE_WINDOW = 8
ENTITY_RECOVERY_BATCH_SIZE = 6


class AlgorithmExperimentConfig(BaseModel):
    """Settings for algorithm-selection runs."""

    experiment_name: str = "egocentric_algorithm_optimization"
    track: Literal["egocentric_main", "hospital_reference"] = "egocentric_main"
    model: str = DEFAULT_MODEL
    max_retries: int = 1
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    input_condition: InputCondition = "raw"
    conditions: tuple[AlgorithmCondition, ...] = (
        "baseline_one_stage",
        "entity_first_relation_second_with_validation",
    )
    repetitions: int = Field(default=1, ge=1)
    continue_on_error: bool = True


class EntityExtractionResult(BaseModel):
    """Entity-only result for the first extraction stage."""

    video_id: str | None = None
    scenes: list[Scene] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    actors: list[Worker] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    parameters: list[ProcessParameter] = Field(default_factory=list)
    source_text: str | None = None


class RelationExtractionResult(BaseModel):
    """Relation-only result for the second extraction stage."""

    uses_tool: list[UsesTool] = Field(default_factory=list)
    acts_on_object: list[ActsOnObject] = Field(default_factory=list)
    action_order: list[ActionOrder] = Field(default_factory=list)
    action_causes: list[ActionCauses] = Field(default_factory=list)
    part_of_procedure: list[ActionPartOfProcedure] = Field(default_factory=list)
    observed_in_scene: list[ActionObservedInScene] = Field(default_factory=list)


class MinimalEntity(BaseModel):
    """Flat LLM-friendly entity record inspired by schema optimization work."""

    entity_id: str
    entity_type: MinimalEntityType
    name: str
    evidence_text: str
    confidence: float | None = None


class MinimalEntityExtractionResult(BaseModel):
    """Minimal entity-only response without nested Pydantic objects."""

    entities: list[MinimalEntity] = Field(default_factory=list)


class ControlledEntityDecision(BaseModel):
    """One LLM decision for one renderer-owned candidate ID.

    The LLM decides keep/type/name for every renderer ID;
    an omitted row is detected instead of silently becoming a missing entity.
    """

    entity_id: str
    keep: bool
    entity_type: MinimalEntityType
    name: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = ""


class ControlledEntityDecisionResult(BaseModel):
    """Structured per-candidate decisions returned by the entity LLM."""

    decisions: list[ControlledEntityDecision] = Field(default_factory=list)


class _ControlledEntitySpec(BaseModel):
    """Renderer-owned ID and evidence used to reconcile LLM entity output."""

    entity_id: str
    entity_type: MinimalEntityType
    names: list[str]
    evidence_text: str


class MinimalRelation(BaseModel):
    """Flat relation proposal that references known entity IDs only.

    This is an LLM-proposed relation, not a prewritten candidate edge.
    """

    relation_type: MinimalRelationType
    subject_id: str
    object_id: str
    # 中文：LLM 输出使用 relation_evidence；validation_alias 只用于读取旧实验
    # 检查点和旧测试，不会让新 JSON schema 继续暴露 evidence_text。
    # English: New LLM output uses relation_evidence. The validation alias only
    # keeps old checkpoints/tests readable; it is not exposed in the new schema.
    relation_evidence: str = Field(
        validation_alias=AliasChoices("relation_evidence", "evidence_text")
    )
    antecedent_evidence: str | None = None
    coreference: str | None = Field(
        default=None,
        validation_alias=AliasChoices("coreference", "resolved_reference"),
    )
    coreference_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # 中文：reason 只用于审计 LLM 的判断，不参与图三元组匹配。
    # English: reason audits the LLM decision and is not part of graph matching.
    reason: str = ""

    @property
    def evidence_text(self) -> str:
        """Backward-compatible internal accessor for existing graph conversion."""

        return self.relation_evidence

    @property
    def resolved_reference(self) -> str | None:
        """Backward-compatible accessor for pre-coreference field consumers."""

        return self.coreference


class MinimalRelationExtractionResult(BaseModel):
    """Minimal relation-only response constrained by an entity inventory."""

    relations: list[MinimalRelation] = Field(default_factory=list)


RelationProposalRejectionCode = Literal[
    "unknown_entity_id",
    "ambiguous_entity_id",
    "invalid_domain_range",
    "self_relation",
    "ungrounded_evidence",
    "endpoints_not_in_evidence",
    "invalid_reference_resolution",
    "ambiguous_reference_resolution",
    "missing_causal_cue",
    "duplicate_relation",
]


class RelationIdNormalizationLog(BaseModel):
    """Auditable repair of one relation endpoint token before validation."""

    relation_type: MinimalRelationType
    endpoint: Literal["subject_id", "object_id"]
    original_id: str
    normalized_id: str
    reason: str = "stripped renderer name suffix from a known entity ID"


class RelationProposalRejection(BaseModel):
    """Rejected LLM relation with a machine-readable reason.

    Preserve rejected relations and reasons so filtering stays auditable.
    """

    relation: MinimalRelation
    code: RelationProposalRejectionCode
    message: str


class RelationProposalValidationResult(BaseModel):
    """Accepted and rejected results from hard relation-proposal validation."""

    accepted: MinimalRelationExtractionResult = Field(
        default_factory=MinimalRelationExtractionResult
    )
    rejected: list[RelationProposalRejection] = Field(default_factory=list)


class MinimalRelationCandidate(BaseModel):
    """Deterministic candidate edge that the LLM may select."""

    candidate_id: str
    relation_type: MinimalRelationType
    subject_id: str
    object_id: str
    evidence_hint: str | None = None


class MinimalRelationDebugTrace(BaseModel):
    """Debug payload for relation extraction failure analysis."""

    minimal_entities: list[MinimalEntity] = Field(default_factory=list)
    entity_decisions: list[ControlledEntityDecision] = Field(default_factory=list)
    missing_entity_decision_ids: list[str] = Field(default_factory=list)
    entity_inventory: str = ""
    relation_candidates: list[MinimalRelationCandidate] = Field(default_factory=list)
    minimal_relations: list[MinimalRelation] = Field(default_factory=list)
    filtered_relations: list[MinimalRelation] = Field(default_factory=list)
    rejected_relations: list[RelationProposalRejection] = Field(default_factory=list)
    repair_relations: list[MinimalRelation] = Field(default_factory=list)
    id_normalization_logs: list[RelationIdNormalizationLog] = Field(
        default_factory=list
    )
    segment_count: int = 1
    successful_segment_count: int = 1
    stage_warnings: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)


class RefinementBackend(Protocol):
    """Optional extra interface for validation-driven refinement tests."""

    def extract(
        self,
        text: str,
        response_model: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Return a Pydantic extraction object."""


class AlgorithmRunRecord(BaseModel):
    """One algorithm condition run on one scene."""

    experiment_name: str
    track: Literal["egocentric_main", "hospital_reference"]
    scene_id: str
    condition: AlgorithmCondition
    input_condition: InputCondition
    run_number: int
    input_text: str
    extraction: EgocentricVideoExtraction
    graph: PropertyGraph
    validation: ValidationReport
    node_metrics: PRF1
    relation_metrics: PRF1
    runtime_seconds: float
    execution_error: str | None = None
    debug_trace: MinimalRelationDebugTrace | None = None


class AlgorithmExperimentResult(BaseModel):
    """All runs for one algorithm-selection experiment."""

    dataset_name: str
    config: AlgorithmExperimentConfig
    runs: list[AlgorithmRunRecord] = Field(default_factory=list)

    def runs_for(self, condition: AlgorithmCondition) -> list[AlgorithmRunRecord]:
        """Return all runs for an algorithm condition."""

        return [run for run in self.runs if run.condition == condition]


class AlgorithmExperimentRunner:
    """Compare extraction algorithms on a benchmark dataset."""

    def __init__(
        self,
        config: AlgorithmExperimentConfig | None = None,
        extractor: ExtractionBackend | None = None,
    ) -> None:
        self.config = config or AlgorithmExperimentConfig()
        if extractor is None:
            from backend.extraction.extractor import Extractor

            extractor = Extractor(
                model=self.config.model,
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
            )
        self.extractor = extractor
        self._current_debug_trace: MinimalRelationDebugTrace | None = None

    @property
    def last_debug_trace(self) -> MinimalRelationDebugTrace | None:
        """Return the debug trace from the latest free-text or benchmark run."""

        return self._current_debug_trace

    def extract_text(
        self,
        input_text: str,
        *,
        source_text: str | None = None,
        condition: AlgorithmCondition | None = None,
    ) -> EgocentricVideoExtraction:
        """
        Extract a KG contract from arbitrary standardized text without gold labels.

        This is the entry point for real IndEgo/external-group inputs, where we
        have raw/unified text but do not yet have a BenchmarkScene with gold
        extraction for evaluation.
        """

        source = source_text or input_text
        selected_condition = condition or self.config.conditions[0]
        self._current_debug_trace = None
        extraction = self._extract_free_text_by_condition(
            input_text,
            source,
            selected_condition,
        )
        if selected_condition in {
            "entity_first_relation_second_with_validation",
            "entity_first_relation_second_with_fewshot",
            "minimal_entity_relation_with_validation",
            "llm_relation_proposal",
            "llm_relation_proposal_with_repair",
            "minimal_candidate_relation_with_validation",
            "validation_driven_refinement",
        }:
            extraction = filter_extraction_to_supported_relations(extraction, source)
        return extraction

    def extract_normalized_segment(
        self,
        segment: NormalizedSegment,
        *,
        condition: Literal[
            "llm_relation_proposal",
            "llm_relation_proposal_with_repair",
        ] = "llm_relation_proposal",
    ) -> EgocentricVideoExtraction:
        """Run layers 3-6 from a source-independent Normalized Segment.

        Entity extraction consumes controlled text, relation proposal uses
        fixed entity IDs, and hard validation returns to the original source text.
        """

        extraction = self._extract_partitioned_normalized_segment(
            segment,
            repair_rejected=condition == "llm_relation_proposal_with_repair",
        )
        return filter_extraction_to_supported_relations(
            extraction,
            segment.source_text,
        )

    def run(self, dataset: BenchmarkDataset) -> AlgorithmExperimentResult:
        """Run configured algorithm conditions on every scene."""

        runs: list[AlgorithmRunRecord] = []
        for scene in dataset.scenes:
            for condition in self.config.conditions:
                for run_number in range(1, self.config.repetitions + 1):
                    runs.append(self._run_once(scene, condition, run_number))
        return AlgorithmExperimentResult(
            dataset_name=dataset.name,
            config=self.config,
            runs=runs,
        )

    def evaluate_fold(
        self,
        dataset: BenchmarkDataset,
        fold: DatasetFold,
        cv_config: CrossValidationConfig,
    ) -> list[FoldConditionResult]:
        """Run this algorithm runner on one cross-validation test fold."""

        fold_results, _ = self.run_fold(dataset, fold, cv_config)
        return fold_results

    def run_fold(
        self,
        dataset: BenchmarkDataset,
        fold: DatasetFold,
        cv_config: CrossValidationConfig,
    ) -> tuple[list[FoldConditionResult], AlgorithmExperimentResult]:
        """Run one fold and return both summaries and traceable run records."""

        test_dataset = subset_dataset(dataset, fold.test_scene_ids)
        result = self.run(test_dataset)
        fold_results: list[FoldConditionResult] = []
        for condition in cv_config.conditions:
            condition_runs = [
                run for run in result.runs if run.condition == condition
            ]
            if not condition_runs:
                continue
            fold_results.append(_fold_result_from_runs(fold, cv_config, condition_runs))
        return fold_results, result

    def _run_once(
        self,
        scene: BenchmarkScene,
        condition: AlgorithmCondition,
        run_number: int,
    ) -> AlgorithmRunRecord:
        """Run one scene/condition pair and evaluate against the gold graph."""

        input_text = scene.input_text(self.config.input_condition)
        started_at = time.perf_counter()
        execution_error: str | None = None
        self._current_debug_trace = None
        try:
            extraction = self._extract_by_condition(scene, input_text, condition)
        except Exception as error:
            if not self.config.continue_on_error:
                raise
            execution_error = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            extraction = EgocentricVideoExtraction(source_text=scene.raw_text)

        if condition in {
            "entity_first_relation_second_with_validation",
            "entity_first_relation_second_with_fewshot",
            "minimal_entity_relation_with_validation",
            "llm_relation_proposal",
            "llm_relation_proposal_with_repair",
            "minimal_candidate_relation_with_validation",
            "validation_driven_refinement",
        }:
            extraction = filter_extraction_to_supported_relations(extraction, scene.raw_text)

        graph = build_property_graph(extraction)
        validation = validate_egocentric_extraction(extraction, source_text=scene.raw_text)
        if self._current_debug_trace is not None:
            self._current_debug_trace.validation_issues = [
                f"{issue.kind}: {issue.message}" for issue in validation.issues
            ]
        gold_graph = build_property_graph(scene.gold_extraction)
        node_metrics = _facts_to_prf1("nodes", _node_facts(gold_graph), _node_facts(graph))
        relation_metrics = _facts_to_prf1(
            "relations",
            _relation_facts(gold_graph),
            _relation_facts(graph),
        )

        return AlgorithmRunRecord(
            experiment_name=self.config.experiment_name,
            track=self.config.track,
            scene_id=scene.scene_id,
            condition=condition,
            input_condition=self.config.input_condition,
            run_number=run_number,
            input_text=input_text,
            extraction=extraction,
            graph=graph,
            validation=validation,
            node_metrics=node_metrics,
            relation_metrics=relation_metrics,
            runtime_seconds=round(time.perf_counter() - started_at, 4),
            execution_error=execution_error,
            debug_trace=self._current_debug_trace,
        )

    def _extract_by_condition(
        self,
        scene: BenchmarkScene,
        input_text: str,
        condition: AlgorithmCondition,
    ) -> EgocentricVideoExtraction:
        """Dispatch algorithm variants."""

        if condition == "baseline_one_stage":
            return self.extractor.extract(input_text, EgocentricVideoExtraction)

        if condition == "strict_schema_prompt":
            return self.extractor.extract(
                input_text,
                EgocentricVideoExtraction,
                system_prompt=strict_schema_prompt(),
            )

        if condition in {
            "entity_first_relation_second",
            "entity_first_relation_second_with_validation",
            "entity_first_relation_second_with_fewshot",
            "validation_driven_refinement",
        }:
            fewshot = scene.scene_id if condition == "entity_first_relation_second_with_fewshot" else None
            extraction = self._two_stage_extract(input_text, scene.raw_text, fewshot)
            if condition == "validation_driven_refinement":
                extraction = self._refine_once(input_text, extraction, scene.raw_text)
            return extraction

        if condition in {
            "minimal_entity_relation",
            "minimal_entity_relation_with_validation",
        }:
            return self._minimal_two_stage_extract(input_text, scene.raw_text)

        if condition in {
            "llm_relation_proposal",
            "llm_relation_proposal_with_repair",
        }:
            if (
                self.config.input_condition == "unified"
                and scene.unified_record is not None
                and scene.unified_record.normalized_segment is not None
            ):
                return self._extract_partitioned_normalized_segment(
                    scene.unified_record.normalized_segment,
                    repair_rejected=(
                        condition == "llm_relation_proposal_with_repair"
                    ),
                )
            return self._llm_relation_proposal_extract(
                input_text,
                scene.raw_text,
                repair_rejected=condition == "llm_relation_proposal_with_repair",
            )

        if condition in {
            "minimal_candidate_relation",
            "minimal_candidate_relation_with_validation",
        }:
            return self._minimal_candidate_extract(input_text, scene.raw_text)

        raise ValueError(f"Unsupported algorithm condition: {condition!r}")

    def _extract_free_text_by_condition(
        self,
        input_text: str,
        source_text: str,
        condition: AlgorithmCondition,
    ) -> EgocentricVideoExtraction:
        """Dispatch algorithm variants for input that has no benchmark scene."""

        if condition == "baseline_one_stage":
            return self.extractor.extract(input_text, EgocentricVideoExtraction)

        if condition == "strict_schema_prompt":
            return self.extractor.extract(
                input_text,
                EgocentricVideoExtraction,
                system_prompt=strict_schema_prompt(),
            )

        if condition in {
            "entity_first_relation_second",
            "entity_first_relation_second_with_validation",
            "entity_first_relation_second_with_fewshot",
            "validation_driven_refinement",
        }:
            extraction = self._two_stage_extract(input_text, source_text)
            if condition == "validation_driven_refinement":
                extraction = self._refine_once(input_text, extraction, source_text)
            return extraction

        if condition in {
            "minimal_entity_relation",
            "minimal_entity_relation_with_validation",
        }:
            return self._minimal_two_stage_extract(input_text, source_text)

        if condition in {
            "llm_relation_proposal",
            "llm_relation_proposal_with_repair",
        }:
            return self._llm_relation_proposal_extract(
                input_text,
                source_text,
                repair_rejected=condition == "llm_relation_proposal_with_repair",
            )

        if condition in {
            "minimal_candidate_relation",
            "minimal_candidate_relation_with_validation",
        }:
            return self._minimal_candidate_extract(input_text, source_text)

        raise ValueError(f"Unsupported algorithm condition: {condition!r}")

    def _two_stage_extract(
        self,
        input_text: str,
        source_text: str,
        fewshot_scene_id: str | None = None,
    ) -> EgocentricVideoExtraction:
        """Extract entities first, then relations constrained to known entities."""

        entities = self.extractor.extract(
            input_text,
            EntityExtractionResult,
            system_prompt=entity_stage_prompt(),
        )
        entity_text = _entity_inventory_text(entities)
        relations = self.extractor.extract(
            f"{input_text}\n\nKnown entities:\n{entity_text}",
            RelationExtractionResult,
            system_prompt=relation_stage_prompt(entity_text, fewshot_scene_id),
        )
        return _compose_extraction(entities, relations, source_text)

    def _minimal_two_stage_extract(
        self,
        input_text: str,
        source_text: str,
    ) -> EgocentricVideoExtraction:
        """
        Use flat ID-based schemas to avoid the full nested egocentric contract.

        This follows the PARSE/ODKE+/iText2KG-style idea: optimize the schema for
        the LLM, then convert back into the project's richer graph contract.
        """

        entities = self.extractor.extract(
            input_text,
            MinimalEntityExtractionResult,
            system_prompt=minimal_entity_stage_prompt(),
        )
        inventory = minimal_entity_inventory_text(entities)
        relations = self.extractor.extract(
            f"{input_text}\n\nKnown entity inventory:\n{inventory}",
            MinimalRelationExtractionResult,
            system_prompt=minimal_relation_stage_prompt(inventory),
        )
        endpoint_filtered = filter_minimal_relations_by_entities(relations, entities)
        filtered = filter_minimal_relations_by_evidence(
            endpoint_filtered,
            entities,
            source_text,
        )
        self._current_debug_trace = MinimalRelationDebugTrace(
            minimal_entities=entities.entities,
            entity_inventory=inventory,
            minimal_relations=relations.relations,
            filtered_relations=filtered.relations,
        )
        return minimal_to_egocentric_extraction(entities, filtered, source_text)

    def _minimal_candidate_extract(
        self,
        input_text: str,
        source_text: str,
    ) -> EgocentricVideoExtraction:
        """Extract minimal entities, generate relation candidates, then classify candidates."""

        entities = self.extractor.extract(
            input_text,
            MinimalEntityExtractionResult,
            system_prompt=minimal_entity_stage_prompt(),
        )
        inventory = minimal_entity_inventory_text(entities)
        candidates = generate_relation_candidates(entities, source_text)
        candidate_text = relation_candidate_inventory_text(candidates)
        relations = self.extractor.extract(
            (
                f"{input_text}\n\nKnown entity inventory:\n{inventory}\n\n"
                f"Candidate relations:\n{candidate_text}"
            ),
            MinimalRelationExtractionResult,
            system_prompt=minimal_candidate_relation_stage_prompt(
                inventory,
                candidate_text,
            ),
        )
        candidate_filtered = filter_minimal_relations_by_candidates(
            relations,
            candidates,
        )
        filtered = filter_minimal_relations_by_evidence(
            candidate_filtered,
            entities,
            source_text,
        )
        self._current_debug_trace = MinimalRelationDebugTrace(
            minimal_entities=entities.entities,
            entity_inventory=inventory,
            relation_candidates=candidates,
            minimal_relations=relations.relations,
            filtered_relations=filtered.relations,
        )
        return minimal_to_egocentric_extraction(entities, filtered, source_text)

    # [CB-03 中文] 关系 LLM 在已确认实体 ID 之间自主提出关系。
    # [CB-03 EN] The relation LLM proposes edges among confirmed entity IDs.
    def _llm_relation_proposal_extract(
        self,
        input_text: str,
        source_text: str,
        *,
        repair_rejected: bool = False,
    ) -> EgocentricVideoExtraction:
        """Let the LLM propose edges over fixed entity IDs, then hard-validate them.

        No rule-generated candidate list is supplied. The LLM discovers
        relations from the text but may only use known IDs and allowed types.
        """

        specs = _controlled_entity_specs(input_text)
        entity_decisions: list[ControlledEntityDecision] = []
        missing_entity_decision_ids: list[str] = []
        entity_stage_warnings: list[str] = []
        if specs:
            (
                entities,
                entity_decisions,
                missing_entity_decision_ids,
                entity_stage_warnings,
            ) = self._extract_controlled_entity_decisions(specs)
            entity_id_aliases = {
                entity.entity_id: entity.entity_id for entity in entities.entities
            }
        else:
            entity_input = controlled_entity_extraction_input(input_text)
            raw_entities = self.extractor.extract(
                entity_input,
                MinimalEntityExtractionResult,
                system_prompt=minimal_entity_stage_prompt(),
            )
            (
                entities,
                entity_id_aliases,
            ) = reconcile_minimal_entities_with_controlled_text(
                raw_entities,
                input_text,
                source_text,
            )
        # [CB-01 中文] 把实体阶段结果整理成关系 LLM 可读取的固定实体清单。
        # [CB-01 EN] Render entity-stage results as the fixed relation inventory.
        inventory = minimal_entity_inventory_text(entities)
        # [CB-02 中文] 仅开放当前实体类型真正支持的关系合同。
        # [CB-02 EN] Expose only contracts supported by the current entity types.
        relation_contracts = available_relation_contracts(entities)
        relation_input = controlled_relation_proposal_input(input_text)
        proposal_parts: list[MinimalRelationExtractionResult] = []
        empty_review_relations: list[MinimalRelation] = []
        id_normalization_logs: list[RelationIdNormalizationLog] = []
        stage_warnings: list[str] = list(entity_stage_warnings)
        focus_groups = relation_subject_focus_groups(entities)
        # [CB-03 中文] 按动作小组调用 LLM；程序不预先生成候选关系答案。
        # [CB-03 EN] Call the LLM by action group without rule-made edge answers.
        for group_index, subject_ids in enumerate(focus_groups, start=1):
            focus_instruction = relation_focus_instruction(
                subject_ids,
                group_index=group_index,
                group_count=len(focus_groups),
            )
            try:
                part = self.extractor.extract(
                    f"{relation_input}{focus_instruction}",
                    MinimalRelationExtractionResult,
                    system_prompt=llm_relation_proposal_stage_prompt(
                        inventory,
                        relation_contracts,
                    ),
                )
                part, normalization_logs = normalize_minimal_relation_ids(
                    part,
                    entity_id_aliases,
                    known_entity_ids={
                        entity.entity_id for entity in entities.entities
                    },
                )
                id_normalization_logs.extend(normalization_logs)
            except Exception as error:
                # 中文：一个动作组失败不能抹掉其他组已完成的 LLM 结果。
                # English: One failed action group must not erase proposals from
                # the groups that completed successfully.
                stage_warnings.append(
                    "Relation proposal group "
                    f"{group_index}/{len(focus_groups)} failed: "
                    f"{type(error).__name__}: {str(error).splitlines()[0]}"
                )
                continue

            if not part.relations and _has_legal_relation_pair(entities):
                # 中文：首轮空结果不等于“确实无关系”。再让 LLM 做一次受限复核，
                # 但仍不提供程序写死的候选答案，也不允许新增实体。
                # English: An empty first pass is not proof that no relation exists.
                # Run one constrained LLM review without rule-made answers.
                try:
                    reviewed = self.extractor.extract(
                        (
                            f"{relation_input}{focus_instruction}\n\n"
                            "The first relation pass returned an empty list. Re-check "
                            "every type-compatible pair against SOURCE EVIDENCE."
                        ),
                        MinimalRelationExtractionResult,
                        system_prompt=llm_empty_relation_review_stage_prompt(
                            inventory,
                            relation_contracts,
                        ),
                    )
                    reviewed, normalization_logs = normalize_minimal_relation_ids(
                        reviewed,
                        entity_id_aliases,
                        known_entity_ids={
                            entity.entity_id for entity in entities.entities
                        },
                    )
                    id_normalization_logs.extend(normalization_logs)
                    empty_review_relations.extend(reviewed.relations)
                    part = reviewed
                except Exception as error:
                    # 中文：空结果复核是可选增强；失败时保留“该组无提议”。
                    # English: Empty-result review is optional; on failure the
                    # group's original empty result remains valid.
                    stage_warnings.append(
                        "Empty relation review group "
                        f"{group_index}/{len(focus_groups)} failed: "
                        f"{type(error).__name__}: {str(error).splitlines()[0]}"
                    )
            proposal_parts.append(part)

        proposals = merge_minimal_relation_results(proposal_parts)
        # [CB-04 中文] LLM 的全部提议必须先通过硬验证，之后才能进入 KG。
        # [CB-04 EN] Every LLM proposal must pass hard validation before the KG.
        validation = validate_llm_relation_proposals(
            proposals,
            entities,
            source_text,
        )
        repair_relations: list[MinimalRelation] = list(empty_review_relations)
        if repair_rejected and validation.rejected:
            try:
                repaired = self.extractor.extract(
                    llm_relation_repair_input(
                        source_text,
                        inventory,
                        validation.rejected,
                    ),
                    MinimalRelationExtractionResult,
                    system_prompt=llm_relation_repair_stage_prompt(
                        inventory,
                        relation_contracts,
                    ),
                )
                repaired, normalization_logs = normalize_minimal_relation_ids(
                    repaired,
                    entity_id_aliases,
                    known_entity_ids={
                        entity.entity_id for entity in entities.entities
                    },
                )
                id_normalization_logs.extend(normalization_logs)
                repair_relations = repaired.relations
                repair_validation = validate_llm_relation_proposals(
                    repaired,
                    entities,
                    source_text,
                )
                validation = merge_relation_proposal_validations(
                    validation,
                    repair_validation,
                )
            except Exception as error:
                # 中文：修复轮失败时保留首轮已通过 Hard Validation 的关系。
                # English: If repair fails, retain every first-pass relation that
                # already passed hard validation instead of failing the scene.
                stage_warnings.append(
                    "Relation repair failed; kept first-pass accepted relations: "
                    f"{type(error).__name__}: {str(error).splitlines()[0]}"
                )
        self._current_debug_trace = MinimalRelationDebugTrace(
            minimal_entities=entities.entities,
            entity_decisions=entity_decisions,
            missing_entity_decision_ids=missing_entity_decision_ids,
            entity_inventory=inventory,
            minimal_relations=proposals.relations,
            filtered_relations=validation.accepted.relations,
            rejected_relations=validation.rejected,
            repair_relations=repair_relations,
            id_normalization_logs=id_normalization_logs,
            stage_warnings=stage_warnings,
        )
        return minimal_to_egocentric_extraction(
            entities,
            validation.accepted,
            source_text,
            add_deterministic_relation_fallbacks=False,
            # 中文：Controlled Renderer 已经给出了标准动作名；这里不能再把
            # 关系目标拼回动作名，否则 place 会漂移成 place wood。
            # English: Controlled Renderer owns the canonical action surface;
            # do not append a relation target and drift `place` into `place wood`.
            enrich_action_context=not bool(_controlled_entity_specs(input_text)),
            infer_source_scene=False,
            attach_single_context=False,
        )

    def _extract_controlled_entity_decisions(
        self,
        specs: list[_ControlledEntitySpec],
    ) -> tuple[
        MinimalEntityExtractionResult,
        list[ControlledEntityDecision],
        list[str],
        list[str],
    ]:
        """Require one auditable LLM decision for every renderer candidate."""

        first = self.extractor.extract(
            controlled_entity_decision_input(specs),
            ControlledEntityDecisionResult,
            system_prompt=controlled_entity_decision_prompt(specs),
        )
        decisions, warnings = reconcile_controlled_entity_decisions(first, specs)
        decided_ids = {decision.entity_id for decision in decisions}
        missing_specs = [spec for spec in specs if spec.entity_id not in decided_ids]
        if missing_specs:
            try:
                follow_up = self.extractor.extract(
                    controlled_entity_decision_input(
                        missing_specs,
                        missing_only=True,
                    ),
                    ControlledEntityDecisionResult,
                    system_prompt=controlled_entity_decision_prompt(missing_specs),
                )
                follow_up_decisions, follow_up_warnings = (
                    reconcile_controlled_entity_decisions(follow_up, missing_specs)
                )
                decisions.extend(follow_up_decisions)
                warnings.extend(follow_up_warnings)
            except Exception as error:
                warnings.append(
                    "Missing entity decision review failed: "
                    f"{type(error).__name__}: {str(error).splitlines()[0]}"
                )

        decisions = _dedupe_entity_decisions(decisions)
        decided_ids = {decision.entity_id for decision in decisions}
        missing_ids = [
            spec.entity_id for spec in specs if spec.entity_id not in decided_ids
        ]
        if missing_ids:
            warnings.append(
                "Entity decisions remain incomplete for renderer IDs: "
                + ", ".join(missing_ids)
            )
        entities = controlled_decisions_to_minimal_entities(decisions, specs)
        return entities, decisions, missing_ids, warnings

    def _extract_partitioned_normalized_segment(
        self,
        segment: NormalizedSegment,
        *,
        repair_rejected: bool,
        max_actions: int = 6,
    ) -> EgocentricVideoExtraction:
        """Run entity/relation/validation per action partition and merge by ID.

        A timed-out partition becomes a warning while successful sibling
        partitions are still merged into the final KG by renderer-owned IDs.
        """

        parts = segment.split_by_actions(max_actions=max_actions, overlap_actions=1)
        traces: list[MinimalRelationDebugTrace] = []
        partition_warnings: list[str] = []
        for index, part in enumerate(parts, start=1):
            self._current_debug_trace = None
            try:
                self._llm_relation_proposal_extract(
                    part.to_controlled_text(),
                    part.source_text,
                    repair_rejected=repair_rejected,
                )
            except Exception as error:
                partition_warnings.append(
                    "Layers 4-6 partition "
                    f"{index}/{len(parts)} failed and was omitted: "
                    f"{type(error).__name__}: {str(error).splitlines()[0]}"
                )
                continue
            if self._current_debug_trace is not None:
                traces.append(self._current_debug_trace.model_copy(deep=True))

        merged_decisions = _dedupe_entity_decisions(
            decision
            for trace in traces
            for decision in trace.entity_decisions
        )
        global_specs = _controlled_entity_specs(segment.to_controlled_text())
        recovery_warnings: list[str] = []
        decided_ids = {decision.entity_id for decision in merged_decisions}
        recovery_specs = [
            spec for spec in global_specs if spec.entity_id not in decided_ids
        ]
        if recovery_specs:
            # 中文：整个关系子段失败时，只对未判断的 Renderer ID 做一次小型
            # 实体恢复；不重跑已成功子段，也不由程序补任何关系。
            # English: If a whole relation partition fails, run one small entity-
            # only recovery for undecided renderer IDs. Never rerun successful
            # siblings or synthesize relation answers in code.
            # 中文：极端场景可能一次留下几十个未判断 ID；恢复也必须小批次，
            # 否则恢复调用本身会重复原来的超时问题。
            # English: Extreme scenes may leave dozens of undecided IDs. Recovery
            # is batched too, preventing the recovery call from recreating the
            # original oversized-output timeout.
            for batch_index, start in enumerate(
                range(0, len(recovery_specs), ENTITY_RECOVERY_BATCH_SIZE),
                start=1,
            ):
                batch = recovery_specs[start : start + ENTITY_RECOVERY_BATCH_SIZE]
                try:
                    (
                        _recovered_entities,
                        recovered_decisions,
                        recovery_missing_ids,
                        entity_recovery_warnings,
                    ) = self._extract_controlled_entity_decisions(batch)
                    merged_decisions = _dedupe_entity_decisions(
                        [*merged_decisions, *recovered_decisions]
                    )
                    recovery_warnings.extend(entity_recovery_warnings)
                    recovered_ids = {
                        decision.entity_id for decision in recovered_decisions
                    }
                    recovery_warnings.append(
                        "Entity-only partition recovery batch "
                        f"{batch_index} reviewed renderer IDs: "
                        + ", ".join(
                            spec.entity_id
                            for spec in batch
                            if spec.entity_id in recovered_ids
                        )
                    )
                    if recovery_missing_ids:
                        recovery_warnings.append(
                            "Entity-only partition recovery batch "
                            f"{batch_index} still missed IDs: "
                            + ", ".join(recovery_missing_ids)
                        )
                except Exception as error:
                    recovery_warnings.append(
                        "Entity-only partition recovery batch "
                        f"{batch_index} failed: {type(error).__name__}: "
                        f"{str(error).splitlines()[0]}"
                    )
        # 中文：重叠子段可能对同一 ID 给出相反判断。全局实体必须严格由同一份
        # 去重后的 LLM 判断重建，不能出现 trace 说 keep=false、KG 却保留实体。
        # English: Overlapping parts may disagree on one ID. Rebuild the global
        # inventory from the same deduplicated LLM decisions so trace and KG agree.
        entities = controlled_decisions_to_minimal_entities(
            merged_decisions,
            global_specs,
        )
        merged_relation_candidates = merge_minimal_relation_results(
            [
                MinimalRelationExtractionResult(
                    relations=trace.filtered_relations,
                )
                for trace in traces
            ]
        )
        # 中文：局部已通过的边在全局实体决策合并后再次验证，避免端点被全局
        # 丢弃或改型后仍进入 KG。English: Revalidate locally accepted edges after
        # global entity reconciliation so removed/retyped endpoints cannot leak in.
        global_validation = validate_llm_relation_proposals(
            merged_relation_candidates,
            entities,
            segment.source_text,
        )
        accepted = global_validation.accepted
        decided_ids = {decision.entity_id for decision in merged_decisions}
        global_missing_ids = [
            spec.entity_id for spec in global_specs if spec.entity_id not in decided_ids
        ]
        merged_trace = MinimalRelationDebugTrace(
            minimal_entities=entities.entities,
            entity_decisions=merged_decisions,
            missing_entity_decision_ids=global_missing_ids,
            entity_inventory=minimal_entity_inventory_text(entities),
            minimal_relations=merge_minimal_relation_results(
                [
                    MinimalRelationExtractionResult(
                        relations=trace.minimal_relations,
                    )
                    for trace in traces
                ]
            ).relations,
            filtered_relations=accepted.relations,
            rejected_relations=[
                item for trace in traces for item in trace.rejected_relations
            ]
            + global_validation.rejected,
            repair_relations=merge_minimal_relation_results(
                [
                    MinimalRelationExtractionResult(
                        relations=trace.repair_relations,
                    )
                    for trace in traces
                ]
            ).relations,
            id_normalization_logs=[
                item for trace in traces for item in trace.id_normalization_logs
            ],
            stage_warnings=[
                *partition_warnings,
                *recovery_warnings,
                *_entity_decision_conflict_warnings(traces),
                *(warning for trace in traces for warning in trace.stage_warnings),
            ],
            segment_count=len(parts),
            successful_segment_count=len(traces),
        )
        self._current_debug_trace = merged_trace
        return minimal_to_egocentric_extraction(
            entities,
            accepted,
            segment.source_text,
            add_deterministic_relation_fallbacks=False,
            enrich_action_context=False,
            infer_source_scene=False,
            attach_single_context=False,
        )

    def _refine_once(
        self,
        input_text: str,
        extraction: EgocentricVideoExtraction,
        source_text: str,
    ) -> EgocentricVideoExtraction:
        """Ask for one repair pass only when validation found issues."""

        report = validate_egocentric_extraction(extraction, source_text=source_text)
        if not report.issues:
            return extraction
        repaired = self.extractor.extract(
            (
                f"{input_text}\n\nCurrent extraction:\n"
                f"{extraction.model_dump_json()}\n\nValidation issues:\n"
                f"{report.model_dump_json()}"
            ),
            EgocentricVideoExtraction,
            system_prompt=refinement_prompt(),
        )
        return repaired


def strict_schema_prompt() -> str:
    """Prompt fragment for one-stage ontology-constrained extraction."""

    ontology = build_egocentric_ontology()
    relations = "\n".join(
        f"- {relation.label}: {relation.endpoints}"
        for relation in ontology.relations
    )
    return (
        "Extract an egocentric procedural knowledge graph. Return JSON only. "
        "Use only the schema fields. Every relation must use one of these "
        f"ontology contracts:\n{relations}\n"
        "Do not invent entities or relations without evidence in the input."
    )


def entity_stage_prompt() -> str:
    """Prompt fragment for entity-only extraction."""

    return (
        "Extract only entities from the scene text. Return JSON only. "
        "Do not output relations. Keep names short and evidence-grounded."
    )


def relation_stage_prompt(entity_inventory: str, fewshot_scene_id: str | None = None) -> str:
    """Prompt fragment for relation extraction among known entities."""

    fewshot_note = (
        f"\nUse train-fold style examples only; do not copy gold labels for {fewshot_scene_id}."
        if fewshot_scene_id
        else ""
    )
    return (
        "Extract only relations between the known entities. Return JSON only. "
        "Do not create new entities. Allowed relation contracts are: "
        "USES_TOOL Action->Tool; ACTS_ON Action->Object; BEFORE Action->Action; "
        "CAUSES Action->Action; PART_OF Action->Procedure; OBSERVED_IN Action->Scene. "
        "Each relation must have text evidence in the input."
        f"{fewshot_note}\nKnown entities:\n{entity_inventory}"
    )


def minimal_entity_stage_prompt() -> str:
    """Prompt for flat coarse-grained entity extraction."""

    return (
        "Extract a minimal entity inventory from the controlled scene text. Return "
        "JSON only. When [NORMALIZED ENTITIES] and [ACTION SEQUENCE] are present, create "
        "entities only from their name, verb, direct_object, tool, role, and sentence "
        "fields. Do not create additional entities from [SOURCE EVIDENCE]. Use "
        "SOURCE EVIDENCE only to copy an exact evidence_text. Map ROLE records to "
        "Worker entities. "
        "Use flat records with fields: entity_id, entity_type, name, evidence_text, "
        "confidence. Allowed entity_type values: Action, Tool, Object, Worker, "
        "Parameter, Scene, Procedure. Copy renderer IDs exactly: an ACTION line "
        "starting with A1 must use entity_id A1; OBJECT O1, TOOL T1, and ROLE R1 "
        "must keep O1, T1, and R1. Never generate a second ID namespace such as "
        "action_1 or object_1 when renderer IDs are present. Never create an entity "
        "from a value equal to NONE, null, N/A, or not specified. Every Action name "
        "must contain an explicit verb. Include "
        "its direct target when stated, for example 'tighten mounting screw', but "
        "do not omit an explicit action only because its target is implicit. Copy "
        "the shortest exact evidence span that supports the action. Split distinct "
        "coordinated or repeated verbs, sides, and steps instead of merging them. "
        "Annotation-derived text such as action 'place' involves wood explicitly "
        "states Action place. Worker is only a person or worker role; body parts "
        "are Objects. Do not output relations."
    )


def minimal_relation_stage_prompt(entity_inventory: str) -> str:
    """Prompt for ID-based relation extraction over a small ontology snippet."""

    return (
        "Extract only relations between the known entity IDs. Return JSON only. "
        "Do not create new entities. Allowed relation_type values and endpoints: "
        "USES_TOOL Action->Tool; ACTS_ON Action->Object; BEFORE Action->Action; "
        "CAUSES Action->Action; PART_OF Action->Procedure; OBSERVED_IN "
        "Action->Scene. Use subject_id and object_id exactly from the "
        "inventory, and include relation_evidence from the input. Known entities:\n"
        f"{entity_inventory}"
    )


# [CB-03 Prompt 中文] 定义 LLM 自主提议关系时使用的证据与输出约束。
# [CB-03 Prompt EN] Define evidence and output constraints for LLM proposals.
def llm_relation_proposal_stage_prompt(
    entity_inventory: str,
    relation_contracts: tuple[MinimalRelationType, ...] | None = None,
) -> str:
    """Prompt the LLM to discover relations without a rule-generated candidate list.

    The LLM may choose entity pairs, but cannot invent entities, types,
    or unsupported facts.
    """

    contracts = relation_contracts if relation_contracts is not None else (
        "USES_TOOL",
        "ACTS_ON",
        "BEFORE",
        "CAUSES",
        "PART_OF",
        "OBSERVED_IN",
    )
    return (
        "Discover relations directly from the controlled action frames and their "
        "SOURCE EVIDENCE, then return JSON only. "
        # 中文：常识只能帮助理解表达和提出候选，不能替代数据证据。
        # English: Common knowledge may interpret wording and suggest candidates,
        # but it can never serve as evidence for an edge.
        "You may use domain or common-sense knowledge only to interpret the source "
        "wording and propose a candidate pair. Common-sense knowledge is not "
        "evidence: omit any relation that lacks explicit SOURCE EVIDENCE. "
        "No candidate relation list is provided: decide which supplied entity IDs "
        "are related. Never create or rename entity IDs. Use only these currently "
        f"available contracts: {_relation_contract_text(contracts)}. For every "
        "relation, copy one exact, contiguous relation_evidence "
        "span from [SOURCE EVIDENCE] and provide confidence and a short reason. "
        "Never use a generated sentence= field as evidence unless the same exact "
        "text also occurs in SOURCE EVIDENCE. The "
        "span should mention both endpoints; for BEFORE and CAUSES, quote the "
        "shortest complete clause or sentence containing both actions and the "
        "temporal or causal cue. "
        "When relation_evidence uses a local pronoun such as it or this one for "
        "the object, do not replace the pronoun with a guessed name. Return the "
        "complete auditable chain in the same relation record: relation_evidence, "
        "antecedent_evidence, and coreference such as 'it -> O4'. The antecedent "
        "must be in the same or immediately preceding source sentence and must "
        "explicitly name O4. If that antecedent names more than one compatible "
        "target, also return coreference_confidence from 0 to 1. Omit the relation "
        "when you cannot resolve the reference confidently. "
        "USES_TOOL requires actual use, not merely a visible or available tool. "
        "A body part or item whose supplied type is Object cannot be a USES_TOOL "
        "target; consider ACTS_ON only when the evidence states its direct "
        "participation. "
        "ACTS_ON requires the action to directly affect the object. BEFORE requires "
        "explicit temporal/order support; list order alone is insufficient. CAUSES "
        "requires explicit causal wording; temporal order alone is insufficient. "
        "Do not use general knowledge. When evidence is uncertain, omit the relation. "
        "An empty relations list is valid.\n\nKnown entities:\n"
        f"{entity_inventory}"
    )


def llm_empty_relation_review_stage_prompt(
    entity_inventory: str,
    relation_contracts: tuple[MinimalRelationType, ...] | None = None,
) -> str:
    """Re-check an empty proposal without injecting dataset-specific answers."""

    contracts = relation_contracts if relation_contracts is not None else (
        "USES_TOOL",
        "ACTS_ON",
        "BEFORE",
        "CAUSES",
        "PART_OF",
        "OBSERVED_IN",
    )
    return (
        "The first relation proposal was empty. Independently re-check every known "
        "Action against every entity allowed by these available contracts: "
        f"{_relation_contract_text(contracts)}. In a controlled action frame, "
        "direct_object_id=O... is a clue for ACTS_ON and tool_id=T... is a clue "
        "for USES_TOOL, but accept it only when "
        "SOURCE EVIDENCE explicitly supports the fact. Annotation wording such as "
        "action 'place' involves wood may support ACTS_ON between the supplied "
        "place Action ID and wood Object ID. Do not infer BEFORE from list order. "
        "Do not create or rename IDs. relation_evidence must be one exact contiguous "
        "quote from SOURCE EVIDENCE that mentions both endpoints. When its target "
        "is a pronoun, instead provide the complete relation_evidence, "
        "antecedent_evidence, coreference chain, and coreference_confidence when "
        "the antecedent contains multiple compatible targets. Return an empty list "
        "only after checking all type-compatible pairs.\n\nKnown entities:\n"
        f"{entity_inventory}"
    )


def llm_relation_repair_stage_prompt(
    entity_inventory: str,
    relation_contracts: tuple[MinimalRelationType, ...] | None = None,
) -> str:
    """Repair rejected proposals without widening the entity or relation space.

    Repair may correct IDs, direction, or evidence but cannot add facts.
    """

    contracts = relation_contracts if relation_contracts is not None else (
        "USES_TOOL",
        "ACTS_ON",
        "BEFORE",
        "CAUSES",
        "PART_OF",
        "OBSERVED_IN",
    )
    return (
        "Repair only the supplied rejected relation proposals. Return JSON only. "
        "Use only known entity IDs and these currently available contracts: "
        f"{_relation_contract_text(contracts)}. "
        "Correct an endpoint, direction, or evidence span only when the source "
        "explicitly supports it. relation_evidence must be an exact contiguous "
        "source quote. A pronoun-mediated repair must return relation_evidence, "
        "antecedent_evidence, and coreference together; add coreference_confidence "
        "when multiple compatible antecedents exist. "
        "Do not create additional unrelated relations. Omit proposals that cannot be "
        "repaired with explicit evidence.\n\nKnown entities:\n"
        f"{entity_inventory}"
    )


# [CB-02 中文] 根据实体端点类型动态生成当前可用关系合同。
# [CB-02 EN] Build available relation contracts from current endpoint types.
def available_relation_contracts(
    entities: MinimalEntityExtractionResult,
) -> tuple[MinimalRelationType, ...]:
    """Expose only relation types whose endpoint types exist in this inventory."""

    types = [entity.entity_type for entity in entities.entities]
    contracts: list[MinimalRelationType] = []
    if "Action" in types and "Tool" in types:
        contracts.append("USES_TOOL")
    if "Action" in types and "Object" in types:
        contracts.append("ACTS_ON")
    if types.count("Action") >= 2:
        contracts.extend(("BEFORE", "CAUSES"))
    if "Action" in types and "Procedure" in types:
        contracts.append("PART_OF")
    if "Action" in types and "Scene" in types:
        contracts.append("OBSERVED_IN")
    return tuple(contracts)


def _relation_contract_text(
    contracts: tuple[MinimalRelationType, ...],
) -> str:
    endpoint_contracts = {
        "USES_TOOL": "USES_TOOL Action->Tool",
        "ACTS_ON": "ACTS_ON Action->Object",
        "BEFORE": "BEFORE Action->Action",
        "CAUSES": "CAUSES Action->Action",
        "PART_OF": "PART_OF Action->Procedure",
        "OBSERVED_IN": "OBSERVED_IN Action->Scene",
    }
    return "; ".join(endpoint_contracts[item] for item in contracts) or "none"


def llm_relation_repair_input(
    source_text: str,
    entity_inventory: str,
    rejected: list[RelationProposalRejection],
) -> str:
    """Build an auditable repair request from rejected relations and reason codes."""

    rejected_payload = [item.model_dump(mode="json") for item in rejected]
    return (
        f"Source text:\n{source_text}\n\nKnown entity inventory:\n{entity_inventory}"
        "\n\nRejected proposals and validation reasons:\n"
        f"{json.dumps(rejected_payload, ensure_ascii=False, indent=2)}"
    )


def minimal_candidate_relation_stage_prompt(
    entity_inventory: str,
    candidate_inventory: str,
) -> str:
    """Prompt for candidate-constrained relation extraction."""

    return (
        "Extract only relations that are supported by text evidence. Return JSON only. "
        "You must choose relations only from the candidate list. Do not create new "
        "entity IDs, relation types, or candidate endpoints. If no candidate has "
        "clear evidence, return an empty relations list.\n\n"
        "Allowed relation_type values and endpoints:\n"
        "- USES_TOOL: Action -> Tool\n"
        "- ACTS_ON: Action -> Object\n"
        "- BEFORE: Action -> Action\n"
        "- CAUSES: Action -> Action, only with explicit causal words such as "
        "causes, enables, allows, prepares\n"
        "- PART_OF: Action -> Procedure\n"
        "- OBSERVED_IN: Action -> Scene\n\n"
        "Few-shot example:\n"
        "Text: The worker picks up the screwdriver and tightens the screw.\n"
        "Known entities: action_1 Action pick up screwdriver; action_2 Action "
        "tighten screw; tool_1 Tool screwdriver; object_1 Object screw.\n"
        "Candidates: c1 USES_TOOL action_2 tool_1; c2 ACTS_ON action_2 object_1; "
        "c3 BEFORE action_1 action_2.\n"
        "Output: {\"relations\": [{\"relation_type\": \"USES_TOOL\", "
        "\"subject_id\": \"action_2\", \"object_id\": \"tool_1\", "
        "\"relation_evidence\": \"tightens the screw\"}, {\"relation_type\": "
        "\"ACTS_ON\", \"subject_id\": \"action_2\", \"object_id\": "
        "\"object_1\", \"relation_evidence\": \"tightens the screw\"}, "
        "{\"relation_type\": \"BEFORE\", \"subject_id\": \"action_1\", "
        "\"object_id\": \"action_2\", \"relation_evidence\": \"picks up ... and "
        "tightens\"}]}\n\n"
        f"Known entities:\n{entity_inventory}\n\n"
        f"Candidate relations:\n{candidate_inventory}"
    )


def refinement_prompt() -> str:
    """Prompt fragment for one validation-driven repair pass."""

    return (
        "Repair the extraction using only the validation issues. Return a full "
        "EgocentricVideoExtraction JSON object. Remove unsupported or invalid "
        "relations instead of inventing new facts."
    )


def filter_extraction_to_supported_relations(
    extraction: EgocentricVideoExtraction,
    source_text: str,
) -> EgocentricVideoExtraction:
    """Drop relations that fail ontology or grounding checks."""

    cleaned = extraction.model_copy(deep=True)
    valid_report = validate_egocentric_extraction(cleaned, source_text=source_text)
    if valid_report.filtered_relation_count == 0:
        return cleaned

    cleaned.uses_tool = [
        relation for relation in cleaned.uses_tool
        if _relation_is_supported("USES_TOOL", relation, source_text)
    ]
    cleaned.acts_on_object = [
        relation for relation in cleaned.acts_on_object
        if _relation_is_supported("ACTS_ON", relation, source_text)
    ]
    cleaned.action_order = [
        relation for relation in cleaned.action_order
        if _relation_is_supported("BEFORE", relation, source_text)
    ]
    cleaned.action_causes = [
        relation for relation in cleaned.action_causes
        if _relation_is_supported("CAUSES", relation, source_text)
    ]
    cleaned.part_of_procedure = [
        relation for relation in cleaned.part_of_procedure
        if _relation_is_supported("PART_OF", relation, source_text)
    ]
    cleaned.observed_in_scene = [
        relation for relation in cleaned.observed_in_scene
        if _relation_is_supported("OBSERVED_IN", relation, source_text)
    ]
    return cleaned


def generate_relation_candidates(
    entities: MinimalEntityExtractionResult,
    source_text: str,
) -> list[MinimalRelationCandidate]:
    """Generate legal relation candidates from minimal entities."""

    actions = [entity for entity in entities.entities if entity.entity_type == "Action"]
    tools = [entity for entity in entities.entities if entity.entity_type == "Tool"]
    objects = [entity for entity in entities.entities if entity.entity_type == "Object"]
    procedures = [
        entity for entity in entities.entities if entity.entity_type == "Procedure"
    ]
    scenes = [entity for entity in entities.entities if entity.entity_type == "Scene"]
    candidates: list[MinimalRelationCandidate] = []

    def add(
        relation_type: MinimalRelationType,
        subject: MinimalEntity,
        target: MinimalEntity,
        evidence_hint: str | None = None,
    ) -> None:
        candidates.append(
            MinimalRelationCandidate(
                candidate_id=f"c{len(candidates) + 1}",
                relation_type=relation_type,
                subject_id=subject.entity_id,
                object_id=target.entity_id,
                evidence_hint=evidence_hint
                or f"{subject.name} -> {target.name}",
            )
        )

    for action in actions:
        for tool in tools:
            add("USES_TOOL", action, tool)
        for obj in objects:
            add("ACTS_ON", action, obj)
        for procedure in procedures:
            add("PART_OF", action, procedure)
        for scene in scenes:
            add("OBSERVED_IN", action, scene)

    for before, after in zip(actions, actions[1:]):
        add("BEFORE", before, after, "adjacent actions in extracted order")

    if _has_causal_cue(source_text):
        for before, after in zip(actions, actions[1:]):
            add("CAUSES", before, after, "source text contains causal cue")

    return candidates


def relation_candidate_inventory_text(
    candidates: list[MinimalRelationCandidate],
) -> str:
    """Render candidates for compact prompting."""

    if not candidates:
        return "- none"
    return "\n".join(
        "- "
        f"{candidate.candidate_id} | {candidate.relation_type} | "
        f"{candidate.subject_id} -> {candidate.object_id} | "
        f"{candidate.evidence_hint or ''}"
        for candidate in candidates
    )


def filter_minimal_relations_by_entities(
    relations: MinimalRelationExtractionResult,
    entities: MinimalEntityExtractionResult,
) -> MinimalRelationExtractionResult:
    """Drop relations with unknown IDs or illegal domain/range types."""

    entity_types = {
        entity.entity_id: entity.entity_type for entity in entities.entities
    }
    return MinimalRelationExtractionResult(
        relations=[
            relation
            for relation in relations.relations
            if _minimal_relation_has_valid_endpoints(relation, entity_types)
        ]
    )


# [CB-04 中文] 验证 ID、端点类型、原文证据、指代链与重复关系。
# [CB-04 EN] Validate IDs, endpoint types, evidence, coreference, and duplicates.
def validate_llm_relation_proposals(
    relations: MinimalRelationExtractionResult,
    entities: MinimalEntityExtractionResult,
    source_text: str,
) -> RelationProposalValidationResult:
    """Hard-validate LLM-proposed relations against IDs, ontology, and evidence.

    The LLM discovers relations; code enforces non-negotiable safeguards.
    """

    entity_by_id = {entity.entity_id: entity for entity in entities.entities}
    entity_id_counts: dict[str, int] = {}
    for entity in entities.entities:
        entity_id_counts[entity.entity_id] = (
            entity_id_counts.get(entity.entity_id, 0) + 1
        )
    entity_types = {
        entity.entity_id: entity.entity_type for entity in entities.entities
    }
    accepted: list[MinimalRelation] = []
    rejected: list[RelationProposalRejection] = []
    seen: set[tuple[MinimalRelationType, str, str]] = set()

    def reject(
        relation: MinimalRelation,
        code: RelationProposalRejectionCode,
        message: str,
    ) -> None:
        rejected.append(
            RelationProposalRejection(
                relation=relation,
                code=code,
                message=message,
            )
        )

    for relation in relations.relations:
        # Gate 1 / 门 1：候选两端必须已经由实体阶段抽取。
        subject = entity_by_id.get(relation.subject_id)
        target = entity_by_id.get(relation.object_id)
        if subject is None or target is None:
            reject(
                relation,
                "unknown_entity_id",
                "Subject and object must both reference supplied entity IDs.",
            )
            continue
        if (
            entity_id_counts[relation.subject_id] != 1
            or entity_id_counts[relation.object_id] != 1
        ):
            reject(
                relation,
                "ambiguous_entity_id",
                "Relation endpoints must reference unique supplied entity IDs.",
            )
            continue
        if relation.subject_id == relation.object_id:
            reject(
                relation,
                "self_relation",
                "Self-relations are not allowed in the current procedural ontology.",
            )
            continue
        # Gate 2 / 门 2：例如 USES_TOOL 必须严格为 Action -> Tool。
        if not _minimal_relation_has_valid_endpoints(relation, entity_types):
            reject(
                relation,
                "invalid_domain_range",
                "The entity types do not match the relation domain and range.",
            )
            continue
        # Gate 3 / 门 3：LLM 必须引用原文中的连续证据，常识不能通过此门。
        if not _evidence_text_is_grounded(relation.evidence_text, source_text):
            reject(
                relation,
                "ungrounded_evidence",
                "Evidence must be copied from the original source text.",
            )
            continue
        # Gate 4 / 门 4：证据片段必须同时支持关系两端；仅仅知道某工具
        # 通常可用于某动作，不足以生成 USES_TOOL。
        endpoints_supported = (
            _evidence_mentions_entity(subject, relation.evidence_text)
            if relation.relation_type == "OBSERVED_IN"
            else _evidence_mentions_both_relation_endpoints(
                subject,
                target,
                relation.evidence_text,
            )
        )
        # 中文：小模型有时只引用 "and then ..."。对于时序/因果关系，可在
        # 原文中扩展到包含该引文的完整句子，但不能跨到任意上下文或 Gold。
        # English: Small models sometimes quote only "and then ...". For temporal
        # and causal edges, inspect the containing source sentence, never gold data.
        if not endpoints_supported:
            support_sentence = _source_sentence_containing_evidence(
                relation.evidence_text,
                source_text,
            )
            sentence_supports_endpoints = bool(support_sentence) and (
                _evidence_mentions_both_relation_endpoints(
                    subject,
                    target,
                    support_sentence,
                )
            )
            if relation.relation_type in {"BEFORE", "CAUSES"}:
                endpoints_supported = sentence_supports_endpoints
            elif (
                sentence_supports_endpoints
                and relation.relation_type in {"USES_TOOL", "ACTS_ON"}
            ):
                # 中文：短引文可借用所在句，但该句只能支持一个候选动作；
                # 否则 "pick X, align Y" 可能把 Y 错连到 pick。
                # English: A short quote may use its sentence only when that
                # sentence supports one action, preventing cross-action edges.
                endpoints_supported = (
                    _supported_action_count(
                        entities,
                        support_sentence,
                    )
                    == 1
                )
        has_reference_payload = bool(
            relation.antecedent_evidence
            or relation.coreference
            or relation.coreference_confidence is not None
        )
        if has_reference_payload:
            # [CB-05 中文] 只核验 LLM 给出的局部指代链，程序不自行猜代词。
            # [CB-05 EN] Verify the LLM's local coreference chain without guessing.
            reference_error = _local_reference_resolution_error(
                relation,
                subject,
                target,
                entities,
                source_text,
            )
            if reference_error is not None:
                rejection_code, rejection_message = reference_error
                reject(
                    relation,
                    rejection_code,
                    rejection_message,
                )
                continue
            # 中文：两端不在同一句时，只有完整指代链通过四项硬验证才能补足
            # 目标端证据；程序本身不猜代词指向。
            # English: When both endpoints are not in one quote, only a complete
            # chain passing all four hard checks can supply target-side support.
            endpoints_supported = True
        if not endpoints_supported:
            reject(
                relation,
                "endpoints_not_in_evidence",
                "Evidence must mention both relation endpoints.",
            )
            continue
        # Additional semantic guard / 补充语义门：因果边还必须有因果词。
        if relation.relation_type == "CAUSES" and not _has_causal_cue(
            relation.evidence_text
        ):
            reject(
                relation,
                "missing_causal_cue",
                "CAUSES requires an explicit causal cue in its own evidence span.",
            )
            continue

        triple = (
            relation.relation_type,
            relation.subject_id,
            relation.object_id,
        )
        if triple in seen:
            reject(
                relation,
                "duplicate_relation",
                "An equivalent source-type-target proposal was already accepted.",
            )
            continue
        # Gate 5 / 门 5：只有通过全部检查且未重复的候选才能进入 KG。
        seen.add(triple)
        accepted.append(relation)

    return RelationProposalValidationResult(
        accepted=MinimalRelationExtractionResult(relations=accepted),
        rejected=rejected,
    )


def merge_relation_proposal_validations(
    initial: RelationProposalValidationResult,
    repair: RelationProposalValidationResult,
) -> RelationProposalValidationResult:
    """Merge one repair pass while preserving initial and repair audit records."""

    accepted = list(initial.accepted.relations)
    rejected = [*initial.rejected, *repair.rejected]
    seen = {
        (item.relation_type, item.subject_id, item.object_id)
        for item in accepted
    }
    for relation in repair.accepted.relations:
        triple = (
            relation.relation_type,
            relation.subject_id,
            relation.object_id,
        )
        if triple in seen:
            rejected.append(
                RelationProposalRejection(
                    relation=relation,
                    code="duplicate_relation",
                    message="Repair duplicated an already accepted relation.",
                )
            )
            continue
        seen.add(triple)
        accepted.append(relation)
    return RelationProposalValidationResult(
        accepted=MinimalRelationExtractionResult(relations=accepted),
        rejected=rejected,
    )


def filter_minimal_relations_by_candidates(
    relations: MinimalRelationExtractionResult,
    candidates: list[MinimalRelationCandidate],
) -> MinimalRelationExtractionResult:
    """Keep only relations that exactly match deterministic candidates."""

    allowed = {
        (candidate.relation_type, candidate.subject_id, candidate.object_id)
        for candidate in candidates
    }
    return MinimalRelationExtractionResult(
        relations=[
            relation
            for relation in relations.relations
            if (
                relation.relation_type,
                relation.subject_id,
                relation.object_id,
            )
            in allowed
        ]
    )


def filter_minimal_relations_by_evidence(
    relations: MinimalRelationExtractionResult,
    entities: MinimalEntityExtractionResult,
    source_text: str,
) -> MinimalRelationExtractionResult:
    """Keep only minimal relations whose evidence supports their endpoints."""

    entity_by_id = {entity.entity_id: entity for entity in entities.entities}
    filtered_relations: list[MinimalRelation] = []

    for relation in relations.relations:
        subject = entity_by_id.get(relation.subject_id)
        target = entity_by_id.get(relation.object_id)
        if subject is None or target is None:
            continue
        if relation.relation_type == "BEFORE":
            filtered_relations.append(relation)
            continue
        if not _evidence_text_is_grounded(relation.evidence_text, source_text):
            continue
        if relation.relation_type in {"USES_TOOL", "ACTS_ON"}:
            if not _evidence_mentions_action_and_target(
                subject,
                target,
                relation.evidence_text,
            ):
                continue
        elif relation.relation_type == "CAUSES" and not _has_causal_cue(source_text):
            continue
        elif relation.relation_type == "PART_OF" and not _evidence_mentions_both_relation_endpoints(
            subject,
            target,
            relation.evidence_text,
        ):
            continue
        elif relation.relation_type == "OBSERVED_IN" and not _evidence_mentions_entity(
            subject,
            relation.evidence_text,
        ):
            continue
        filtered_relations.append(relation)

    return MinimalRelationExtractionResult(relations=filtered_relations)


def enrich_minimal_action_names(
    entities: MinimalEntityExtractionResult,
    relations: MinimalRelationExtractionResult,
) -> MinimalEntityExtractionResult:
    """Add grounded target context to underspecified action names.

    Small local models often emit verb-only actions such as "picks up" while
    relation evidence contains the full phrase "picks up the torch".  Enriching
    only when the relation evidence mentions both sides keeps this deterministic
    and avoids inventing new graph facts.
    """

    entity_by_id = {entity.entity_id: entity for entity in entities.entities}
    entity_types = {
        entity.entity_id: entity.entity_type for entity in entities.entities
    }
    enriched_entities: list[MinimalEntity] = []

    for entity in entities.entities:
        if entity.entity_type != "Action":
            enriched_entities.append(entity)
            continue

        updated_name = entity.name
        if _action_name_needs_context(entity.name):
            target = _best_grounded_action_target(
                action=entity,
                relations=relations.relations,
                entity_by_id=entity_by_id,
                entity_types=entity_types,
            )
            if target is None:
                target = _best_grounded_action_phrase_target(entity, entity_by_id)
            if target is not None and not _target_already_in_action_name(
                entity.name,
                target.name,
            ):
                target_name = _canonical_action_target_name(entity, target)
                updated_name = f"{entity.name} {target_name}"

        updated_name = _normalize_action_surface_name(updated_name)
        if updated_name == entity.name:
            enriched_entities.append(entity)
            continue

        enriched_entities.append(
            entity.model_copy(update={"name": updated_name})
        )

    return MinimalEntityExtractionResult(entities=enriched_entities)


def minimal_to_egocentric_extraction(
    entities: MinimalEntityExtractionResult,
    relations: MinimalRelationExtractionResult,
    source_text: str,
    *,
    add_deterministic_relation_fallbacks: bool = True,
    enrich_action_context: bool = True,
    infer_source_scene: bool = True,
    attach_single_context: bool = True,
) -> EgocentricVideoExtraction:
    """Convert flat minimal records back to the project's rich extraction schema.

    The LLM-proposal condition disables rule fallbacks so its relations
    can be evaluated independently.
    """

    if enrich_action_context:
        entities = enrich_minimal_action_names(entities, relations)
    event_action_ids = _event_action_ids(entities)
    entity_map = {entity.entity_id: entity for entity in entities.entities}
    actions: dict[str, Action] = {}
    tools: dict[str, Tool] = {}
    objects: dict[str, SceneObject] = {}
    actors: dict[str, Worker] = {}
    parameters: dict[str, ProcessParameter] = {}
    scenes: dict[str, Scene] = {}
    procedures: dict[str, Procedure] = {}

    for entity in entities.entities:
        if entity.entity_type == "Action":
            if entity.entity_id not in event_action_ids:
                continue
            # 中文：即使关闭“按关系补全动作名”，仍执行不改变语义的表面格式
            # 规范化，避免孤立字符或屈折形式破坏端点匹配。
            # English: Even when relation-based name enrichment is disabled,
            # retain semantics while normalizing harmless surface-form noise.
            action_name = _normalize_action_surface_name(entity.name)
            actions[entity.entity_id] = Action(
                name=action_name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
                confidence=entity.confidence,
            )
        elif entity.entity_type == "Tool":
            tools[entity.entity_id] = Tool(
                name=entity.name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
            )
        elif entity.entity_type == "Object":
            objects[entity.entity_id] = SceneObject(
                name=entity.name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
                confidence=entity.confidence,
            )
        elif entity.entity_type == "Worker":
            actors[entity.entity_id] = Worker(
                name=entity.name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
            )
        elif entity.entity_type == "Parameter":
            parameters[entity.entity_id] = ProcessParameter(
                name=entity.name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
            )
        elif entity.entity_type == "Scene":
            scenes[entity.entity_id] = Scene(
                name=entity.name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
                confidence=entity.confidence,
            )
        elif entity.entity_type == "Procedure":
            procedures[entity.entity_id] = Procedure(
                name=entity.name,
                entity_id=entity.entity_id,
                evidence_text=entity.evidence_text,
                source_text=entity.evidence_text,
            )

    if infer_source_scene and not scenes:
        parsed_scene = _parse_scene_from_source_text(source_text)
        if parsed_scene:
            scenes["source_scene"] = parsed_scene
    if attach_single_context:
        _attach_single_actor_and_scene(actions, actors, scenes)

    extraction = EgocentricVideoExtraction(
        actions=list(actions.values()),
        tools=list(tools.values()),
        objects=list(objects.values()),
        actors=list(actors.values()),
        parameters=list(parameters.values()),
        scenes=list(scenes.values()),
        procedures=list(procedures.values()),
        source_text=source_text,
    )

    for relation in relations.relations:
        subject = entity_map.get(relation.subject_id)
        target = entity_map.get(relation.object_id)
        if subject is None or target is None:
            continue
        if relation.relation_type == "USES_TOOL":
            action = actions.get(relation.subject_id)
            tool = tools.get(relation.object_id)
            if action and tool:
                extraction.uses_tool.append(
                    UsesTool(
                        action=action,
                        tool=tool,
                        evidence_text=relation.evidence_text,
                        source_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
        elif relation.relation_type == "ACTS_ON":
            action = actions.get(relation.subject_id)
            obj = objects.get(relation.object_id)
            if action and obj:
                extraction.acts_on_object.append(
                    ActsOnObject(
                        action=action,
                        object=obj,
                        evidence_text=relation.evidence_text,
                        source_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
        elif relation.relation_type == "BEFORE":
            before = actions.get(relation.subject_id)
            after = actions.get(relation.object_id)
            if before and after:
                extraction.action_order.append(
                    ActionOrder(
                        before=before,
                        after=after,
                        evidence_text=relation.evidence_text,
                        source_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
        elif relation.relation_type == "CAUSES":
            cause = actions.get(relation.subject_id)
            effect = actions.get(relation.object_id)
            if cause and effect:
                extraction.action_causes.append(
                    ActionCauses(
                        cause=cause,
                        effect=effect,
                        evidence_text=relation.evidence_text,
                        source_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
        elif relation.relation_type == "PART_OF":
            action = actions.get(relation.subject_id)
            procedure = procedures.get(relation.object_id)
            if action and procedure:
                extraction.part_of_procedure.append(
                    ActionPartOfProcedure(
                        action=action,
                        procedure=procedure,
                        evidence_text=relation.evidence_text,
                        source_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
        elif relation.relation_type == "OBSERVED_IN":
            action = actions.get(relation.subject_id)
            scene = scenes.get(relation.object_id)
            if action and scene:
                extraction.observed_in_scene.append(
                    ActionObservedInScene(
                        action=action,
                        scene=scene,
                        evidence_text=relation.evidence_text,
                        source_text=relation.evidence_text,
                        confidence=relation.confidence,
                    )
                )
    if add_deterministic_relation_fallbacks:
        # 中文：旧实验条件保留这些补边，避免改变已有基线行为。
        # English: Preserve fallback augmentation for existing baseline conditions.
        _append_deterministic_action_order(extraction, source_text)
        _append_deterministic_causal_relations(extraction, source_text)
        _append_tool_carryover_relations(extraction, source_text)
    return extraction


# [CB-01 中文] 输出 ID、类型、名称和证据，供关系阶段只连接已知实体。
# [CB-01 EN] Render ID, type, name, and evidence for known-entity linking only.
def minimal_entity_inventory_text(entities: MinimalEntityExtractionResult) -> str:
    """Render compact ID/type/name/evidence lines for relation extraction."""

    if not entities.entities:
        return "- none"
    return "\n".join(
        f"- {entity.entity_id} | {entity.entity_type} | {entity.name} | "
        f"evidence={entity.evidence_text}"
        for entity in entities.entities
    )


def controlled_entity_extraction_input(input_text: str) -> str:
    """Build a compact entity-only view from renderer-owned records.

    Entity extraction selects among rendered records without receiving
    duplicated long evidence. Full controlled text still restores IDs/evidence.
    """

    specs = _controlled_entity_specs(input_text)
    if not specs:
        return input_text
    lines = ["[CONTROLLED ENTITY EXTRACTION VIEW]"]
    lines.extend(
        f"{spec.entity_id} | {spec.entity_type} | name={spec.names[0]}"
        for spec in specs
    )
    return "\n".join(lines)


def controlled_entity_decision_input(
    specs: list[_ControlledEntitySpec],
    *,
    missing_only: bool = False,
) -> str:
    """Render every candidate as an explicit keep/type/name decision row."""

    title = (
        "[MISSING CONTROLLED ENTITY DECISIONS]"
        if missing_only
        else "[CONTROLLED ENTITY DECISION VIEW]"
    )
    return "\n".join(
        [
            title,
            *(
                f"{spec.entity_id} | candidate_type={spec.entity_type} | "
                f"candidate_name={spec.names[0]} | "
                f"candidate_evidence={spec.evidence_text}"
                for spec in specs
            ),
        ]
    )


def controlled_entity_decision_prompt(
    specs: list[_ControlledEntitySpec],
) -> str:
    """Require exactly one semantic decision for every supplied candidate."""

    ids = ", ".join(spec.entity_id for spec in specs)
    return (
        "Judge every supplied renderer candidate independently and return JSON only. "
        "For each candidate return entity_id, keep, entity_type, name, confidence, "
        "and a short reason. Return exactly one decision for every ID and no other "
        f"IDs. Required IDs: {ids}. The LLM owns the semantic keep/type/name "
        "decision, but it must never rename entity_id. Mark keep=false only when "
        "the row is a placeholder or not a real entity/action. Allowed entity_type "
        "values: Action, Tool, Object, Worker, Parameter, Scene, Procedure. "
        "Keep candidate_name unchanged unless a semantic type/name correction is "
        "needed. Every kept name must be supported by candidate_evidence. Never "
        "add a name token that is unsupported by candidate_name or "
        "candidate_evidence."
    )


def controlled_relation_proposal_input(input_text: str) -> str:
    """Keep controlled links plus one source copy for relation reasoning.

    Long per-row source evidence duplicates the final source block.
    Relation reasoning receives link slots plus one source copy; hard validation
    still checks the complete original source.
    """

    if "[NORMALIZED ENTITIES]" not in input_text or "[ACTION SEQUENCE]" not in input_text:
        return input_text
    source_marker = "[SOURCE EVIDENCE]"
    source_start = input_text.find(source_marker)
    if source_start < 0:
        return input_text
    uncertainty_start = input_text.find("[UNCERTAINTY]", source_start)
    source_block = (
        input_text[source_start:uncertainty_start]
        if uncertainty_start >= 0
        else input_text[source_start:]
    ).strip()

    compact_lines = ["[CONTROLLED RELATION VIEW]"]
    for line in input_text[:source_start].splitlines():
        record = _parse_controlled_record(line)
        if record is None:
            continue
        entity_id, kind, fields = record
        if kind in {"OBJECT", "TOOL", "ROLE", "SCENE", "PROCEDURE"}:
            compact_lines.append(
                " | ".join(
                    [
                        entity_id,
                        kind,
                        f"name={fields.get('name', 'NONE')}",
                        f"canonical_name={fields.get('canonical_name', 'NONE')}",
                        f"semantic_role={fields.get('semantic_role', 'NONE')}",
                    ]
                )
            )
        elif kind == "ACTION":
            direct_id, direct_name = _controlled_reference_fields(
                fields,
                "direct_object",
            )
            tool_id, tool_name = _controlled_reference_fields(fields, "tool")
            role_id, role_name = _controlled_reference_fields(fields, "role")
            compact_lines.append(
                " | ".join(
                    [
                        entity_id,
                        kind,
                        f"verb={fields.get('verb', '')}",
                        f"direct_object_id={direct_id}",
                        f"direct_object_name={direct_name}",
                        f"tool_id={tool_id}",
                        f"tool_name={tool_name}",
                        f"role_id={role_id}",
                        f"role_name={role_name}",
                    ]
                )
            )
    compact_lines.extend(("", source_block))
    return "\n".join(compact_lines)


def reconcile_minimal_entities_with_controlled_text(
    entities: MinimalEntityExtractionResult,
    input_text: str,
    source_text: str,
) -> tuple[MinimalEntityExtractionResult, dict[str, str]]:
    """Keep LLM-selected entities but restore renderer-owned IDs.

    The LLM still selects entities. Code removes placeholders, maps the
    selected records back to renderer-owned A/O/T/R IDs, and drops extra entities
    outside the controlled fields.
    """

    specs = _controlled_entity_specs(input_text)
    aliases: dict[str, str] = {}
    placeholders = {"", "none", "null", "n/a", "not specified"}
    filtered = [
        entity
        for entity in entities.entities
        if normalize_name(entity.name) not in placeholders
    ]
    if not specs:
        return (
            MinimalEntityExtractionResult(entities=filtered),
            {entity.entity_id: entity.entity_id for entity in filtered},
        )

    controlled_types: set[MinimalEntityType] = {
        "Action",
        "Tool",
        "Object",
        "Worker",
    }
    used_spec_ids: set[str] = set()
    reconciled: list[MinimalEntity] = []
    for entity in filtered:
        if entity.entity_type not in controlled_types:
            aliases[entity.entity_id] = entity.entity_id
            reconciled.append(entity)
            continue

        candidates = [
            spec
            for spec in specs
            if spec.entity_type == entity.entity_type
            and spec.entity_id not in used_spec_ids
        ]
        if not candidates:
            continue
        scored = [
            (_controlled_entity_match_score(entity, spec), spec)
            for spec in candidates
        ]
        score, selected = max(scored, key=lambda item: item[0])
        if score < 0.5:
            continue

        aliases[entity.entity_id] = selected.entity_id
        aliases[selected.entity_id] = selected.entity_id
        used_spec_ids.add(selected.entity_id)
        evidence_text = entity.evidence_text
        if not _evidence_text_is_grounded(evidence_text, source_text) and (
            _evidence_text_is_grounded(selected.evidence_text, source_text)
        ):
            evidence_text = selected.evidence_text
        reconciled.append(
            entity.model_copy(
                update={
                    "entity_id": selected.entity_id,
                    # 中文：名称由 Controlled Renderer 的证据粒度决定，避免
                    # LLM 在不同调用中把同一动作缩成动词或扩写成整句。
                    # English: Renderer evidence determines the stable surface
                    # name, preventing stochastic verb-only/full-sentence drift.
                    "name": selected.names[0],
                    "evidence_text": evidence_text,
                }
            )
        )

    return MinimalEntityExtractionResult(entities=reconciled), aliases


def remap_minimal_relation_ids(
    relations: MinimalRelationExtractionResult,
    aliases: dict[str, str],
) -> MinimalRelationExtractionResult:
    """Map relation endpoints from an LLM alias back to renderer-owned IDs."""

    return MinimalRelationExtractionResult(
        relations=[
            relation.model_copy(
                update={
                    "subject_id": aliases.get(
                        relation.subject_id,
                        relation.subject_id,
                    ),
                    "object_id": aliases.get(
                        relation.object_id,
                        relation.object_id,
                    ),
                }
            )
            for relation in relations.relations
        ]
    )


def normalize_minimal_relation_ids(
    relations: MinimalRelationExtractionResult,
    aliases: dict[str, str],
    *,
    known_entity_ids: set[str],
) -> tuple[MinimalRelationExtractionResult, list[RelationIdNormalizationLog]]:
    """Normalize aliases and `ID:name` only when the base ID is known.

    This repairs endpoint representation, not relation semantics, and
    records every accepted normalization in the debug trace.
    """

    logs: list[RelationIdNormalizationLog] = []
    normalized_relations: list[MinimalRelation] = []
    for relation in relations.relations:
        updates: dict[str, str] = {}
        for endpoint in ("subject_id", "object_id"):
            original = getattr(relation, endpoint)
            mapped = aliases.get(original, original)
            if mapped not in known_entity_ids and ":" in mapped:
                base_id = mapped.split(":", 1)[0].strip()
                base_id = aliases.get(base_id, base_id)
                if base_id in known_entity_ids:
                    logs.append(
                        RelationIdNormalizationLog(
                            relation_type=relation.relation_type,
                            endpoint=endpoint,
                            original_id=original,
                            normalized_id=base_id,
                        )
                    )
                    mapped = base_id
            updates[endpoint] = mapped
        normalized_relations.append(relation.model_copy(update=updates))
    return (
        MinimalRelationExtractionResult(relations=normalized_relations),
        logs,
    )


def reconcile_controlled_entity_decisions(
    result: ControlledEntityDecisionResult,
    specs: list[_ControlledEntitySpec],
) -> tuple[list[ControlledEntityDecision], list[str]]:
    """Drop unknown/duplicate decision rows and report every contract violation."""

    allowed_ids = {spec.entity_id for spec in specs}
    decisions: list[ControlledEntityDecision] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for decision in result.decisions:
        if decision.entity_id not in allowed_ids:
            warnings.append(
                f"Ignored entity decision for unknown renderer ID {decision.entity_id}."
            )
            continue
        if decision.entity_id in seen:
            warnings.append(
                f"Ignored duplicate entity decision for {decision.entity_id}."
            )
            continue
        seen.add(decision.entity_id)
        decisions.append(decision)
    return decisions, warnings


def controlled_decisions_to_minimal_entities(
    decisions: list[ControlledEntityDecision],
    specs: list[_ControlledEntitySpec],
) -> MinimalEntityExtractionResult:
    """Restore renderer evidence while preserving LLM semantic decisions."""

    spec_by_id = {spec.entity_id: spec for spec in specs}
    placeholders = {"", "none", "null", "n/a", "not specified"}
    entities: list[MinimalEntity] = []
    for decision in decisions:
        spec = spec_by_id.get(decision.entity_id)
        if spec is None or not decision.keep:
            continue
        name = decision.name.strip()
        if normalize_name(name) in placeholders:
            continue
        entities.append(
            MinimalEntity(
                entity_id=decision.entity_id,
                entity_type=decision.entity_type,
                name=name,
                evidence_text=spec.evidence_text,
                confidence=decision.confidence,
            )
        )
    return MinimalEntityExtractionResult(entities=entities)


def _dedupe_entity_decisions(
    decisions: Iterable[ControlledEntityDecision],
) -> list[ControlledEntityDecision]:
    """Keep the first explicit decision for each stable renderer ID."""

    deduped: list[ControlledEntityDecision] = []
    seen: set[str] = set()
    for decision in decisions:
        if decision.entity_id in seen:
            continue
        seen.add(decision.entity_id)
        deduped.append(decision)
    return deduped


def _entity_decision_conflict_warnings(
    traces: list[MinimalRelationDebugTrace],
) -> list[str]:
    """Audit semantic disagreements for IDs repeated by overlapping partitions."""

    first_by_id: dict[str, ControlledEntityDecision] = {}
    conflicting_ids: list[str] = []
    for trace in traces:
        for decision in trace.entity_decisions:
            first = first_by_id.setdefault(decision.entity_id, decision)
            signature = (decision.keep, decision.entity_type, normalize_name(decision.name))
            first_signature = (first.keep, first.entity_type, normalize_name(first.name))
            if signature != first_signature and decision.entity_id not in conflicting_ids:
                conflicting_ids.append(decision.entity_id)
    if not conflicting_ids:
        return []
    # 中文：程序只报告冲突并采用稳定的首次判断，不投票生成新的语义标签。
    # English: Code reports the conflict and uses the stable first decision; it
    # does not vote a new semantic label into existence.
    return [
        "Conflicting LLM entity decisions across overlapping partitions; "
        "kept the first explicit decision for IDs: " + ", ".join(conflicting_ids)
    ]


def merge_minimal_entity_results(
    results: list[MinimalEntityExtractionResult],
) -> MinimalEntityExtractionResult:
    """Merge partition entities by global ID without inventing missing records."""

    merged: list[MinimalEntity] = []
    seen: set[str] = set()
    for result in results:
        for entity in result.entities:
            if entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            merged.append(entity)
    return MinimalEntityExtractionResult(entities=merged)


def _controlled_entity_specs(input_text: str) -> list[_ControlledEntitySpec]:
    """Parse renderer records without reading source-specific dataset fields."""

    if "[NORMALIZED ENTITIES]" not in input_text or "[ACTION SEQUENCE]" not in input_text:
        return []

    specs: list[_ControlledEntitySpec] = []
    mention_types: dict[str, MinimalEntityType] = {
        "OBJECT": "Object",
        "TOOL": "Tool",
        "ROLE": "Worker",
        "SCENE": "Scene",
        "PROCEDURE": "Procedure",
    }
    for line in input_text.splitlines():
        record = _parse_controlled_record(line)
        if record is None:
            continue
        entity_id, kind, fields = record
        if kind in mention_types:
            names = [fields.get("name", "").strip()]
            canonical = fields.get("canonical_name", "").strip()
            if normalize_name(canonical) not in {"", "none"}:
                names.append(canonical)
            specs.append(
                _ControlledEntitySpec(
                    entity_id=entity_id,
                    entity_type=mention_types[kind],
                    names=names,
                    evidence_text=fields.get("source_evidence", "").strip(),
                )
            )
            continue

        if kind == "ACTION" and re.fullmatch(r"A\d+", entity_id):
            verb = fields.get("verb", "").strip()
            sentence = fields.get("sentence", "").strip().rstrip(".")
            evidence = fields.get("source_evidence", "").strip()
            # 中文：若原始证据本身只是 annotation 动作标签，就保留短动词；
            # transcript 中的完整动作证据则采用“动词+对象+工具”标准名称。
            # English: Preserve a verb-only annotation label, while transcript
            # evidence receives the controlled verb+object+tool surface.
            preferred_name = (
                verb
                if normalize_name(evidence) == normalize_name(verb)
                else sentence or verb
            )
            specs.append(
                _ControlledEntitySpec(
                    entity_id=entity_id,
                    entity_type="Action",
                    names=list(dict.fromkeys([preferred_name, verb, sentence])),
                    evidence_text=evidence,
                )
            )
    return specs


def _parse_controlled_record(
    line: str,
) -> tuple[str, str, dict[str, str]] | None:
    """Parse one renderer line from either legacy or pure-ID controlled text."""

    parts = [part.strip() for part in line.split(" | ")]
    if len(parts) < 2 or not re.fullmatch(r"[A-Z]+\d+", parts[0]):
        return None
    fields = {
        key.strip(): value.strip()
        for part in parts[2:]
        if "=" in part
        for key, value in [part.split("=", 1)]
    }
    return parts[0], parts[1].upper(), fields


def _controlled_reference_fields(
    fields: dict[str, str],
    prefix: str,
) -> tuple[str, str]:
    """Read new `*_id/*_name` fields and safely migrate legacy `ID:name`."""

    explicit_id = fields.get(f"{prefix}_id")
    if explicit_id is not None:
        return explicit_id or "NONE", fields.get(f"{prefix}_name", "NONE") or "NONE"
    legacy = fields.get(prefix, "NONE")
    if ":" in legacy:
        entity_id, name = legacy.split(":", 1)
        return entity_id.strip(), name.strip()
    return legacy or "NONE", "NONE"


def _controlled_entity_match_score(
    entity: MinimalEntity,
    spec: _ControlledEntitySpec,
) -> float:
    """Score one same-type LLM record against one renderer record."""

    if entity.entity_id == spec.entity_id:
        return 1.0
    entity_name = normalize_name(entity.name)
    names = [normalize_name(name) for name in spec.names if name.strip()]
    if entity_name in names:
        return 0.95
    if any(
        entity_name in name or name in entity_name
        for name in names
        if entity_name and name
    ):
        return 0.8
    entity_tokens = set(_tokens(entity_name))
    best_overlap = max(
        (
            len(entity_tokens & set(_tokens(name)))
            / max(len(entity_tokens | set(_tokens(name))), 1)
            for name in names
        ),
        default=0.0,
    )
    evidence_bonus = (
        0.2
        if normalize_name(entity.evidence_text)
        == normalize_name(spec.evidence_text)
        else 0.0
    )
    return min(best_overlap + evidence_bonus, 0.9)


def relation_subject_focus_groups(
    entities: MinimalEntityExtractionResult,
    max_actions: int = 4,
) -> list[tuple[str, ...]]:
    """Partition only relation subjects, never preselect relation answers.

    Grouping caps one response size. Each focused action may still relate
    to any known entity; no candidate edge is supplied by code.
    """

    if max_actions < 1:
        raise ValueError("max_actions must be at least 1")
    action_ids = [
        entity.entity_id
        for entity in entities.entities
        if entity.entity_type == "Action"
    ]
    return [
        tuple(action_ids[start : start + max_actions])
        for start in range(0, len(action_ids), max_actions)
    ]


def relation_focus_instruction(
    subject_ids: tuple[str, ...],
    *,
    group_index: int,
    group_count: int,
) -> str:
    """Tell the LLM which subjects require an independent relation review."""

    joined_ids = ", ".join(subject_ids)
    return (
        f"\n\n[SUBJECT FOCUS {group_index}/{group_count}]\n"
        "Propose only relations whose subject_id is one of: "
        f"{joined_ids}. The object_id may be any compatible known entity ID. "
        # 中文：这是逐动作检查清单，不是程序生成的候选答案；每一项都可判断为无。
        # English: This is a per-action review checklist, not program-generated
        # candidate answers; every check may legitimately result in no relation.
        "Review every focused action independently for each currently available "
        "contract: ACTS_ON, USES_TOOL, BEFORE, CAUSES, PART_OF, and OBSERVED_IN. "
        "Output only relations supported by SOURCE EVIDENCE. If an action uses a "
        "local pronoun such as it or this one for its target, do not silently skip "
        "the action: either omit it as unresolved or provide relation_evidence, "
        "antecedent_evidence, and coreference such as 'it -> O4' together. If the "
        "antecedent names multiple compatible entities, also provide "
        "coreference_confidence from 0 to 1."
    )


def merge_minimal_relation_results(
    results: list[MinimalRelationExtractionResult],
) -> MinimalRelationExtractionResult:
    """Merge grouped LLM responses while preserving first-seen proposals."""

    merged: list[MinimalRelation] = []
    seen: set[tuple[str, str, str, str]] = set()
    for result in results:
        for relation in result.relations:
            key = (
                relation.relation_type,
                relation.subject_id,
                relation.object_id,
                normalize_name(relation.evidence_text),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(relation)
    return MinimalRelationExtractionResult(relations=merged)


def _has_legal_relation_pair(entities: MinimalEntityExtractionResult) -> bool:
    """Return whether the inventory contains any ontology-compatible pair."""

    types = [entity.entity_type for entity in entities.entities]
    return (
        ("Action" in types and "Tool" in types)
        or ("Action" in types and "Object" in types)
        or types.count("Action") >= 2
        or ("Action" in types and "Procedure" in types)
        or ("Action" in types and "Scene" in types)
    )


def _event_action_ids(entities: MinimalEntityExtractionResult) -> set[str]:
    """Return action IDs that should behave as procedural events."""

    actions = [
        entity for entity in entities.entities
        if entity.entity_type == "Action"
    ]
    event_ids: set[str] = set()
    for action in actions:
        if _is_action_fragment(action, actions):
            continue
        event_ids.add(action.entity_id)
    return event_ids


def _is_action_fragment(
    action: MinimalEntity,
    all_actions: list[MinimalEntity],
) -> bool:
    action_tokens = _tokens(action.name)
    if not action_tokens or action_tokens[0] in ACTION_EVENT_INITIAL_TOKENS:
        return False
    action_token_set = set(action_tokens)
    return any(
        other.entity_id != action.entity_id
        and action_token_set < set(_tokens(other.name))
        for other in all_actions
    )


def _parse_scene_from_source_text(source_text: str) -> Scene | None:
    match = SCENE_HEADER_RE.search(source_text)
    if not match:
        return None
    video_id = match.group("video_id")
    segment_id = match.group("segment_id")
    return Scene(
        name=f"{video_id} segment {segment_id}",
        video_id=video_id,
        segment_id=segment_id,
        timestamp_start_seconds=_timestamp_to_seconds(match.group("start")),
        timestamp_end_seconds=_timestamp_to_seconds(match.group("end")),
        source_text=match.group(0),
        confidence=1.0,
    )


def _timestamp_to_seconds(value: str) -> float:
    minutes, seconds = value.split(":")
    return float(int(minutes) * 60 + int(seconds))


def _attach_single_actor_and_scene(
    actions: dict[str, Action],
    actors: dict[str, Worker],
    scenes: dict[str, Scene],
) -> None:
    actor = next(iter(actors.values())) if len(actors) == 1 else None
    scene = next(iter(scenes.values())) if len(scenes) == 1 else None
    for action in actions.values():
        if actor and action.actor is None:
            action.actor = actor
        if scene and action.scene is None:
            action.scene = scene


def _append_deterministic_action_order(
    extraction: EgocentricVideoExtraction,
    source_text: str,
) -> None:
    for before, after in zip(extraction.actions, extraction.actions[1:]):
        existing = _find_action_order(extraction, before, after)
        if existing:
            if not _evidence_text_is_grounded(
                existing.evidence_text or existing.source_text or "",
                source_text,
            ):
                existing.evidence_text = source_text
                existing.source_text = source_text
                existing.confidence = existing.confidence or 0.8
            continue
        extraction.action_order.append(
            ActionOrder(
                before=before,
                after=after,
                evidence_text=source_text,
                source_text=source_text,
                confidence=0.8,
            )
        )


def _append_deterministic_causal_relations(
    extraction: EgocentricVideoExtraction,
    source_text: str,
) -> None:
    causal_sentences = _causal_evidence_sentences(source_text)
    if not causal_sentences:
        return

    for before, after in zip(extraction.actions, extraction.actions[1:]):
        if _has_action_causes(extraction, before, after):
            continue
        evidence = _causal_evidence_for_action_pair(before, after, causal_sentences)
        if not evidence:
            continue
        extraction.action_causes.append(
            ActionCauses(
                cause=before,
                effect=after,
                evidence_text=evidence,
                source_text=evidence,
                confidence=0.75,
            )
        )


def _append_tool_carryover_relations(
    extraction: EgocentricVideoExtraction,
    source_text: str,
) -> None:
    action_positions = {
        action.entity_id: index
        for index, action in enumerate(extraction.actions)
        if action.entity_id
    }
    held_tools: list[tuple[int, Tool]] = []
    for relation in extraction.uses_tool:
        action_id = relation.action.entity_id
        if not action_id or action_id not in action_positions:
            continue
        if _is_tool_pickup_action(relation.action):
            held_tools.append((action_positions[action_id], relation.tool))

    for tool_position, tool in held_tools:
        for action in extraction.actions[tool_position + 1:]:
            if not _tool_semantically_matches_action(tool, action):
                continue
            if _has_uses_tool(extraction, action, tool):
                continue
            extraction.uses_tool.append(
                UsesTool(
                    action=action,
                    tool=tool,
                    evidence_text=source_text,
                    source_text=source_text,
                    confidence=0.75,
                )
        )


def _find_action_order(
    extraction: EgocentricVideoExtraction,
    before: Action,
    after: Action,
) -> ActionOrder | None:
    for relation in extraction.action_order:
        if (
            normalize_name(relation.before.name) == normalize_name(before.name)
            and normalize_name(relation.after.name) == normalize_name(after.name)
        ):
            return relation
    return None


def _has_action_causes(
    extraction: EgocentricVideoExtraction,
    cause: Action,
    effect: Action,
) -> bool:
    return any(
        normalize_name(relation.cause.name) == normalize_name(cause.name)
        and normalize_name(relation.effect.name) == normalize_name(effect.name)
        for relation in extraction.action_causes
    )


def _has_uses_tool(
    extraction: EgocentricVideoExtraction,
    action: Action,
    tool: Tool,
) -> bool:
    return any(
        normalize_name(relation.action.name) == normalize_name(action.name)
        and normalize_name(relation.tool.name) == normalize_name(tool.name)
        for relation in extraction.uses_tool
    )


def _is_tool_pickup_action(action: Action) -> bool:
    return _has_token_match(
        _meaningful_action_tokens(action.name),
        TOOL_CONTEXT_ACTION_TOKENS,
    )


def _tool_semantically_matches_action(tool: Tool, action: Action) -> bool:
    tool_tokens = set(_tokens(tool.name))
    action_tokens = set(_tokens(action.name))
    return any(
        tool_tokens & tool_markers and action_tokens & action_markers
        for tool_markers, action_markers in TOOL_ACTION_COMPATIBILITY
    )


def _causal_evidence_sentences(source_text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source_text.strip())
        if sentence.strip() and _has_causal_cue(sentence)
    ]


def _causal_evidence_for_action_pair(
    cause: Action,
    effect: Action,
    causal_sentences: list[str],
) -> str | None:
    for sentence in causal_sentences:
        if _sentence_supports_action(cause, sentence) and _sentence_supports_action(
            effect,
            sentence,
        ):
            return sentence
    return None


def _sentence_supports_action(action: Action, sentence: str) -> bool:
    sentence_tokens = set(_tokens(sentence))
    action_tokens = _meaningful_action_tokens(action.name)
    if action_tokens and action_tokens[0] in ACTION_PHRASE_TARGET_ACTION_TOKENS:
        action_tokens = action_tokens[1:] or action_tokens
    return _has_token_match(action_tokens, sentence_tokens)


def _minimal_relation_has_valid_endpoints(
    relation: MinimalRelation,
    entity_types: dict[str, MinimalEntityType],
) -> bool:
    subject_type = entity_types.get(relation.subject_id)
    object_type = entity_types.get(relation.object_id)
    expected = {
        "USES_TOOL": ("Action", "Tool"),
        "ACTS_ON": ("Action", "Object"),
        "BEFORE": ("Action", "Action"),
        "CAUSES": ("Action", "Action"),
        "PART_OF": ("Action", "Procedure"),
        "OBSERVED_IN": ("Action", "Scene"),
    }[relation.relation_type]
    return (subject_type, object_type) == expected


def _best_grounded_action_target(
    action: MinimalEntity,
    relations: list[MinimalRelation],
    entity_by_id: dict[str, MinimalEntity],
    entity_types: dict[str, MinimalEntityType],
) -> MinimalEntity | None:
    context_relations = [
        relation
        for relation in relations
        if relation.subject_id == action.entity_id
        and relation.relation_type in ACTION_CONTEXT_RELATIONS
        and _minimal_relation_has_valid_endpoints(relation, entity_types)
    ]
    for relation_type in _action_context_relation_order(action.name):
        for relation in context_relations:
            if relation.relation_type != relation_type:
                continue
            target = entity_by_id.get(relation.object_id)
            if target and _evidence_mentions_action_and_target(
                action,
                target,
                relation.evidence_text,
            ):
                return target
    return None


def _best_grounded_action_phrase_target(
    action: MinimalEntity,
    entity_by_id: dict[str, MinimalEntity],
) -> MinimalEntity | None:
    action_tokens = _meaningful_action_tokens(action.name)
    if not _has_token_match(action_tokens, ACTION_PHRASE_TARGET_ACTION_TOKENS):
        return None

    for target in entity_by_id.values():
        if target.entity_type != "Action" or target.entity_id == action.entity_id:
            continue
        if _target_already_in_action_name(action.name, target.name):
            continue
        if _evidence_mentions_action_and_target(
            action,
            target,
            action.evidence_text,
        ):
            return target
    return None


def _action_context_relation_order(action_name: str) -> tuple[MinimalRelationType, ...]:
    action_tokens = _meaningful_action_tokens(action_name)
    if _has_token_match(action_tokens, TOOL_CONTEXT_ACTION_TOKENS):
        return ("USES_TOOL", "ACTS_ON")
    if _has_token_match(action_tokens, OBJECT_CONTEXT_ACTION_TOKENS):
        return ("ACTS_ON", "USES_TOOL")
    return ACTION_CONTEXT_RELATIONS


def _canonical_action_target_name(
    action: MinimalEntity,
    target: MinimalEntity,
) -> str:
    if target.entity_type != "Tool":
        return target.name
    if not _has_token_match(
        _meaningful_action_tokens(action.name),
        TOOL_CONTEXT_ACTION_TOKENS,
    ):
        return target.name

    target_tokens = set(_tokens(target.name))
    for required_tokens, canonical_name, context_tokens in TOOL_ACTION_TARGET_CANONICAL_TYPES:
        if required_tokens <= target_tokens and context_tokens & target_tokens:
            return canonical_name
    return target.name


def _action_name_needs_context(name: str) -> bool:
    return len(_meaningful_action_tokens(name)) <= 1


def _normalize_action_surface_name(name: str) -> str:
    parts = name.split()
    if not parts:
        return name
    # 中文：小模型偶尔会输出 `t tightens ...` 这类孤立打字前缀。仅当下一词
    # 是已知动作词时删除单字符前缀；这属于格式修复，不替 LLM 判断语义。
    # English: Small models sometimes emit a stray prefix such as `t tightens`.
    # Drop it only before a known action word; this is format repair, not a
    # program-generated semantic decision.
    if (
        len(parts) >= 2
        and len(parts[0].strip(".,:;!?")) == 1
        and (
            parts[1].lower() in ACTION_INITIAL_TOKEN_NORMALIZATION
            or parts[1].lower() in ACTION_EVENT_INITIAL_TOKENS
        )
    ):
        parts = parts[1:]
    normalized_initial = ACTION_INITIAL_TOKEN_NORMALIZATION.get(parts[0].lower())
    if not normalized_initial:
        return name
    return " ".join([normalized_initial, *parts[1:]])


def _meaningful_action_tokens(value: str) -> list[str]:
    return [
        token for token in _tokens(value)
        if token not in ACTION_NAME_PARTICLES
    ]


def _target_tokens(value: str) -> list[str]:
    return [
        token for token in _tokens(value)
        if token not in ACTION_NAME_PARTICLES
        and len(token) >= MIN_TARGET_TOKEN_LENGTH
    ]


def _evidence_mentions_action_and_target(
    action: MinimalEntity,
    target: MinimalEntity,
    evidence_text: str,
) -> bool:
    evidence_tokens = _tokens(evidence_text)
    action_tokens = _meaningful_action_tokens(action.name)
    target_tokens = _target_tokens(target.name)
    return _has_ordered_token_window(action_tokens, target_tokens, evidence_tokens)


def _evidence_mentions_both_relation_endpoints(
    subject: MinimalEntity,
    target: MinimalEntity,
    evidence_text: str,
) -> bool:
    """Require explicit lexical support for both endpoints in one evidence span.

    Check both endpoints in the quoted span without consulting gold labels.
    """

    evidence_tokens = set(_tokens(evidence_text))
    # 中文：Action 端点优先检查动作词，避免只匹配到共同对象名就误判为有证据。
    # English: Match an Action by its leading action token so a shared object noun
    # does not falsely ground the action endpoint.
    return (
        _relation_endpoint_is_supported(subject, evidence_tokens)
        and _relation_endpoint_is_supported(target, evidence_tokens)
    )


def _evidence_mentions_entity(entity: MinimalEntity, evidence_text: str) -> bool:
    """Check one endpoint for relations whose context endpoint is metadata-backed."""

    evidence_tokens = set(_tokens(evidence_text))
    return _relation_endpoint_is_supported(entity, evidence_tokens)


def _relation_endpoint_is_supported(
    entity: MinimalEntity,
    evidence_tokens: set[str],
) -> bool:
    """Match an endpoint conservatively, including common nominalized verbs.

    Prefer the action verb. If it is nominalized away, require at least
    two informative name tokens so a shared object noun cannot ground an edge.
    """

    entity_tokens = (
        _meaningful_action_tokens(entity.name)
        if entity.entity_type == "Action"
        else _target_tokens(entity.name)
    )
    if not entity_tokens:
        return False
    if entity.entity_type != "Action":
        return _has_endpoint_token_match(entity_tokens, evidence_tokens)
    if _has_endpoint_token_match(entity_tokens[:1], evidence_tokens):
        return True
    matched = {
        token
        for token in entity_tokens
        if _has_endpoint_token_match([token], evidence_tokens)
    }
    return len(matched) >= 2


def _target_already_in_action_name(action_name: str, target_name: str) -> bool:
    return _has_token_match(_target_tokens(target_name), set(_tokens(action_name)))


def _evidence_text_is_grounded(evidence_text: str, source_text: str) -> bool:
    evidence = _normalized_text_for_substring(evidence_text)
    source = _normalized_text_for_substring(source_text)
    return bool(evidence) and evidence in source


def _source_sentence_containing_evidence(
    evidence_text: str,
    source_text: str,
) -> str | None:
    """Return the bounded source sentence containing an already-grounded quote.

    This widens only the endpoint-check window; it never proposes an edge.
    """

    evidence = _normalized_text_for_substring(evidence_text)
    if not evidence:
        return None
    for sentence in re.findall(r"[^.!?]+(?:[.!?]|$)", source_text):
        if evidence in _normalized_text_for_substring(sentence):
            return sentence.strip()
    return None


# [CB-05 中文] 对 LLM 输出的代词、先行词、句距和目标 ID 做硬验证。
# [CB-05 EN] Hard-validate pronoun, antecedent, sentence distance, and target ID.
def _local_reference_resolution_error(
    relation: MinimalRelation,
    subject: MinimalEntity,
    target: MinimalEntity,
    entities: MinimalEntityExtractionResult,
    source_text: str,
) -> tuple[RelationProposalRejectionCode, str] | None:
    """Return a hard-validation error for one LLM-proposed coreference chain.

    Code does not resolve the pronoun itself. It verifies the LLM's
    relation quote, antecedent quote, `it -> O4` chain, bounded distance,
    candidate ambiguity, and known target ID.
    """

    if relation.relation_type not in {"USES_TOOL", "ACTS_ON"}:
        return (
            "invalid_reference_resolution",
            "Coreference is supported only for ACTS_ON and USES_TOOL.",
        )
    antecedent = (relation.antecedent_evidence or "").strip()
    resolution = (relation.coreference or "").strip()
    if not antecedent or not resolution or "->" not in resolution:
        return (
            "invalid_reference_resolution",
            "Pronoun relations must provide relation_evidence, "
            "antecedent_evidence, and coreference together.",
        )
    reference_text, target_id = [part.strip() for part in resolution.split("->", 1)]
    if target_id != relation.object_id:
        return (
            "invalid_reference_resolution",
            "The coreference target ID must equal the relation object_id.",
        )
    if normalize_name(reference_text) not in SUPPORTED_COREFERENCE_EXPRESSIONS:
        return (
            "invalid_reference_resolution",
            "The left side of coreference must be a supported local pronoun.",
        )
    # Check 1 / 检查 1：关系证据与先行词证据都必须逐字来自原文。
    if not _evidence_text_is_grounded(relation.relation_evidence, source_text):
        return (
            "invalid_reference_resolution",
            "relation_evidence must be copied from the original source text.",
        )
    if not _evidence_text_is_grounded(antecedent, source_text):
        return (
            "invalid_reference_resolution",
            "antecedent_evidence must be copied from the original source text.",
        )
    if not _evidence_mentions_entity(subject, relation.evidence_text):
        return (
            "invalid_reference_resolution",
            "relation_evidence must explicitly support the subject Action.",
        )
    if not _evidence_mentions_entity(target, antecedent):
        return (
            "invalid_reference_resolution",
            "antecedent_evidence must explicitly name the relation target.",
        )
    if not _evidence_text_is_grounded(reference_text, relation.evidence_text):
        return (
            "invalid_reference_resolution",
            "The coreference pronoun must occur in relation_evidence.",
        )

    sentences = [
        sentence.strip()
        for sentence in re.findall(r"[^.!?]+(?:[.!?]|$)", source_text)
        if sentence.strip()
    ]
    relation_index = _sentence_index_for_evidence(
        relation.evidence_text,
        sentences,
    )
    antecedent_index = _sentence_index_for_evidence(antecedent, sentences)
    # Check 2 / 检查 2：先行词只能在同句或紧邻的前一句，不能引用未来句或
    # 任意远处上下文。English: The antecedent must be in the same sentence or
    # immediately preceding sentence, never a later or arbitrarily distant one.
    if (
        relation_index is None
        or antecedent_index is None
        or not 0 <= relation_index - antecedent_index <= 1
    ):
        return (
            "invalid_reference_resolution",
            "The antecedent must be in the same or immediately preceding sentence.",
        )

    # Check 3 / 检查 3：只在同类型、且被先行词证据明确提到的实体间计算候选。
    # 唯一候选可直接通过；多候选必须有达到阈值的显式指代置信度。
    # English: Candidate ambiguity is computed only among type-compatible known
    # entities named by the antecedent. Multiple candidates require high explicit
    # coreference confidence.
    candidate_ids = {
        entity.entity_id
        for entity in entities.entities
        if entity.entity_type == target.entity_type
        and _evidence_mentions_entity(entity, antecedent)
    }
    if target.entity_id not in candidate_ids:
        return (
            "invalid_reference_resolution",
            "The supplied target is not a grounded antecedent candidate.",
        )
    if len(candidate_ids) > 1 and (
        relation.coreference_confidence is None
        or relation.coreference_confidence < COREFERENCE_CONFIDENCE_THRESHOLD
    ):
        return (
            "ambiguous_reference_resolution",
            "Multiple compatible antecedent candidates require "
            f"coreference_confidence >= {COREFERENCE_CONFIDENCE_THRESHOLD:.2f}.",
        )

    # Check 4 / 检查 4：object_id 已由调用方的实体清单门验证；这里再次确保
    # 它仍在候选集合中。English: The caller already checked object_id existence;
    # membership here ties that known ID to this exact antecedent quote.
    return None


def _sentence_index_for_evidence(
    evidence_text: str,
    sentences: list[str],
) -> int | None:
    evidence = _normalized_text_for_substring(evidence_text)
    for index, sentence in enumerate(sentences):
        if evidence and evidence in _normalized_text_for_substring(sentence):
            return index
    return None


def _supported_action_count(
    entities: MinimalEntityExtractionResult,
    evidence_text: str,
) -> int:
    """Count distinct supplied actions lexically supported by a source sentence."""

    evidence_tokens = set(_tokens(evidence_text))
    return sum(
        entity.entity_type == "Action"
        and _relation_endpoint_is_supported(entity, evidence_tokens)
        for entity in entities.entities
    )


def _normalized_text_for_substring(value: str) -> str:
    return " ".join(_tokens(value))


def _has_token_match(query_tokens: list[str], candidate_tokens: set[str]) -> bool:
    for token in query_tokens:
        if any(_token_matches(token, candidate) for candidate in candidate_tokens):
            return True
    return False


def _has_endpoint_token_match(
    query_tokens: list[str],
    candidate_tokens: set[str],
) -> bool:
    """Use light stemming only for relation endpoint evidence checks."""

    for query in query_tokens:
        query_stem = _simple_token_stem(query)
        for candidate in candidate_tokens:
            candidate_stem = _simple_token_stem(candidate)
            if _token_matches(query, candidate) or (
                len(query_stem) >= 4
                and (
                    candidate_stem.startswith(query_stem)
                    or query_stem.startswith(candidate_stem)
                )
            ):
                return True
    return False


def _has_ordered_token_window(
    action_tokens: list[str],
    target_tokens: list[str],
    evidence_tokens: list[str],
) -> bool:
    if not action_tokens or not target_tokens or not evidence_tokens:
        return False

    action_positions = _matching_positions(action_tokens, evidence_tokens)
    target_positions = _matching_positions(target_tokens, evidence_tokens)
    return any(
        0 <= target_position - action_position <= ACTION_TARGET_EVIDENCE_WINDOW
        for action_position in action_positions
        for target_position in target_positions
    )


def _matching_positions(
    query_tokens: list[str],
    candidate_tokens: list[str],
) -> list[int]:
    return [
        index
        for index, candidate in enumerate(candidate_tokens)
        if any(_token_matches(query, candidate) for query in query_tokens)
    ]


def _token_matches(query: str, candidate: str) -> bool:
    return (
        query == candidate
        or f"{query}s" == candidate
        or (query.endswith("s") and query[:-1] == candidate)
        or (len(query) >= 4 and candidate.startswith(query))
    )


def _simple_token_stem(token: str) -> str:
    """Normalize common English verb endings for evidence-only matching."""

    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", value.lower())


def _has_causal_cue(source_text: str) -> bool:
    lowered = source_text.lower()
    return any(cue in lowered for cue in CAUSAL_CUES)


def _relation_is_supported(label: str, relation: BaseModel, source_text: str) -> bool:
    candidate = _compose_extraction(
        EntityExtractionResult(source_text=source_text),
        RelationExtractionResult(**_relation_payload(label, relation)),
        source_text,
    )
    report = validate_egocentric_extraction(candidate, source_text=source_text)
    return report.filtered_relation_count == 0


def _relation_payload(label: str, relation: BaseModel) -> dict[str, list[BaseModel]]:
    mapping = {
        "USES_TOOL": "uses_tool",
        "ACTS_ON": "acts_on_object",
        "BEFORE": "action_order",
        "CAUSES": "action_causes",
        "PART_OF": "part_of_procedure",
        "OBSERVED_IN": "observed_in_scene",
    }
    return {mapping[label]: [relation]}


def _compose_extraction(
    entities: EntityExtractionResult,
    relations: RelationExtractionResult,
    source_text: str,
) -> EgocentricVideoExtraction:
    return EgocentricVideoExtraction(
        video_id=entities.video_id,
        scenes=entities.scenes,
        procedures=entities.procedures,
        actors=entities.actors,
        actions=entities.actions,
        tools=entities.tools,
        objects=entities.objects,
        parameters=entities.parameters,
        uses_tool=relations.uses_tool,
        acts_on_object=relations.acts_on_object,
        action_order=relations.action_order,
        action_causes=relations.action_causes,
        part_of_procedure=relations.part_of_procedure,
        observed_in_scene=relations.observed_in_scene,
        source_text=source_text,
    )


def _entity_inventory_text(entities: EntityExtractionResult) -> str:
    rows: list[str] = []
    for label, values in [
        ("Scene", entities.scenes),
        ("Procedure", entities.procedures),
        ("Actor", entities.actors),
        ("Action", entities.actions),
        ("Tool", entities.tools),
        ("Object", entities.objects),
        ("ProcessParameter", entities.parameters),
    ]:
        for value in values:
            rows.append(f"- {label}: {value.name}")
    return "\n".join(rows) if rows else "- none"


def _fold_result_from_runs(
    fold: DatasetFold,
    cv_config: CrossValidationConfig,
    runs: list[AlgorithmRunRecord],
) -> FoldConditionResult:
    node_metrics = aggregate_prf1([run.node_metrics for run in runs], "fold_nodes")
    relation_metrics = aggregate_prf1(
        [run.relation_metrics for run in runs],
        "fold_relations",
    )
    successes = [1.0 if not run.execution_error else 0.0 for run in runs]
    return FoldConditionResult(
        fold_id=fold.fold_id,
        track=cv_config.track,
        condition=runs[0].condition,
        test_scene_ids=fold.test_scene_ids,
        node_f1=node_metrics.f1,
        relation_f1=relation_metrics.f1,
        ontology_conformance=_mean([run.validation.ontology_conformance for run in runs]),
        grounding_rate=_mean([run.validation.evidence_grounding_rate for run in runs]),
        hallucination_rate=_mean([run.validation.relation_hallucination_rate for run in runs]),
        schema_success_rate=_mean(successes),
        runtime_seconds=_mean([run.runtime_seconds for run in runs]),
    )


def _facts_to_prf1(category: str, expected: set[str], extracted: set[str]) -> PRF1:
    return compute_prf1(
        category=category,
        true_positives=len(expected & extracted),
        false_positives=len(extracted - expected),
        false_negatives=len(expected - extracted),
    )


def _node_facts(graph: PropertyGraph) -> set[str]:
    return {f"{node.label}: {normalize_name(node.name)}" for node in graph.nodes.values()}


def _relation_facts(graph: PropertyGraph) -> set[str]:
    facts: set[str] = set()
    for edge in graph.edges:
        source = graph.node(edge.source)
        target = graph.node(edge.target)
        facts.add(
            f"{normalize_name(source.name)} --{edge.type}--> {normalize_name(target.name)}"
        )
    return facts


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
