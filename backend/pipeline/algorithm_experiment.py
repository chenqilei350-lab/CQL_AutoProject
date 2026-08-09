"""Algorithm-selection experiments for text-to-KG extraction.

This module is intentionally separate from data cleaning and unified-text
conversion.  It compares extraction strategies first, then later preprocessing
modules can be plugged in as input conditions.
"""

from __future__ import annotations

import re
import time
from typing import Literal, Protocol

from pydantic import BaseModel, Field

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

MinimalRelationType = Literal["USES_TOOL", "ACTS_ON", "BEFORE", "CAUSES"]

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


class MinimalRelation(BaseModel):
    """Flat relation record that references known entity IDs only."""

    relation_type: MinimalRelationType
    subject_id: str
    object_id: str
    evidence_text: str
    confidence: float | None = None


class MinimalRelationExtractionResult(BaseModel):
    """Minimal relation-only response constrained by an entity inventory."""

    relations: list[MinimalRelation] = Field(default_factory=list)


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
    entity_inventory: str = ""
    relation_candidates: list[MinimalRelationCandidate] = Field(default_factory=list)
    minimal_relations: list[MinimalRelation] = Field(default_factory=list)
    filtered_relations: list[MinimalRelation] = Field(default_factory=list)
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
            "minimal_candidate_relation_with_validation",
            "validation_driven_refinement",
        }:
            extraction = filter_extraction_to_supported_relations(extraction, source)
        return extraction

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
        "Extract a minimal entity inventory from the scene text. Return JSON only. "
        "Use flat records with fields: entity_id, entity_type, name, evidence_text, "
        "confidence. Allowed entity_type values: Action, Tool, Object, Worker, "
        "Parameter, Scene, Procedure. Use stable lowercase IDs like action_1, "
        "tool_1, object_1. Do not output relations."
    )


def minimal_relation_stage_prompt(entity_inventory: str) -> str:
    """Prompt for ID-based relation extraction over a small ontology snippet."""

    return (
        "Extract only relations between the known entity IDs. Return JSON only. "
        "Do not create new entities. Allowed relation_type values and endpoints: "
        "USES_TOOL Action->Tool; ACTS_ON Action->Object; BEFORE Action->Action; "
        "CAUSES Action->Action. Use subject_id and object_id exactly from the "
        "inventory, and include evidence_text from the input. Known entities:\n"
        f"{entity_inventory}"
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
        "causes, enables, allows, prepares\n\n"
        "Few-shot example:\n"
        "Text: The worker picks up the screwdriver and tightens the screw.\n"
        "Known entities: action_1 Action pick up screwdriver; action_2 Action "
        "tighten screw; tool_1 Tool screwdriver; object_1 Object screw.\n"
        "Candidates: c1 USES_TOOL action_2 tool_1; c2 ACTS_ON action_2 object_1; "
        "c3 BEFORE action_1 action_2.\n"
        "Output: {\"relations\": [{\"relation_type\": \"USES_TOOL\", "
        "\"subject_id\": \"action_2\", \"object_id\": \"tool_1\", "
        "\"evidence_text\": \"tightens the screw\"}, {\"relation_type\": "
        "\"ACTS_ON\", \"subject_id\": \"action_2\", \"object_id\": "
        "\"object_1\", \"evidence_text\": \"tightens the screw\"}, "
        "{\"relation_type\": \"BEFORE\", \"subject_id\": \"action_1\", "
        "\"object_id\": \"action_2\", \"evidence_text\": \"picks up ... and "
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
) -> EgocentricVideoExtraction:
    """Convert flat minimal records back to the project's rich extraction schema."""

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
            actions[entity.entity_id] = Action(
                name=entity.name,
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

    if not scenes:
        parsed_scene = _parse_scene_from_source_text(source_text)
        if parsed_scene:
            scenes["source_scene"] = parsed_scene
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
    _append_deterministic_action_order(extraction, source_text)
    _append_deterministic_causal_relations(extraction, source_text)
    _append_tool_carryover_relations(extraction, source_text)
    return extraction


def minimal_entity_inventory_text(entities: MinimalEntityExtractionResult) -> str:
    """Render compact ID/type/name lines for relation extraction."""

    if not entities.entities:
        return "- none"
    return "\n".join(
        f"- {entity.entity_id} | {entity.entity_type} | {entity.name}"
        for entity in entities.entities
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


def _target_already_in_action_name(action_name: str, target_name: str) -> bool:
    return _has_token_match(_target_tokens(target_name), set(_tokens(action_name)))


def _evidence_text_is_grounded(evidence_text: str, source_text: str) -> bool:
    evidence = _normalized_text_for_substring(evidence_text)
    source = _normalized_text_for_substring(source_text)
    return bool(evidence) and evidence in source


def _normalized_text_for_substring(value: str) -> str:
    return " ".join(_tokens(value))


def _has_token_match(query_tokens: list[str], candidate_tokens: set[str]) -> bool:
    for token in query_tokens:
        if any(_token_matches(token, candidate) for candidate in candidate_tokens):
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
