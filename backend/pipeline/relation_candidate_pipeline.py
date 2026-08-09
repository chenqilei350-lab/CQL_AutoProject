"""Generic relation-candidate pipeline for procedural KG extraction.

The module separates three concerns:

1. deterministic candidate generation;
2. candidate scoring / binary validation;
3. reporting which candidate edges are accepted or rejected.

LLMs used through this module are not allowed to invent new edges. They only
judge existing candidate IDs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from backend.graph.property_graph import normalize_name


RelationCandidateType = Literal[
    "USES_TOOL",
    "ACTS_ON",
    "BEFORE",
    "PART_OF",
    "OBSERVED_IN",
    "WARNING_FOR",
]


class EntityMention(BaseModel):
    """Small typed entity/action/annotation record used by candidate generation."""

    entity_id: str
    label: str
    name: str
    evidence_text: str = ""
    start_seconds: float | None = None
    end_seconds: float | None = None
    sequence_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationCandidate(BaseModel):
    """A legal candidate relation that may be accepted by a scorer."""

    candidate_id: str
    relation_type: RelationCandidateType
    subject_id: str
    object_id: str
    subject_label: str
    object_label: str
    subject_name: str
    object_name: str
    evidence_hint: str = ""
    generation_rule: str
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationCandidateDecision(BaseModel):
    """Binary decision for one candidate relation."""

    candidate_id: str
    is_supported: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_text: str = ""
    reason: str = ""
    scorer_name: str = "unknown"


class RelationValidationReport(BaseModel):
    """Complete candidate validation result."""

    candidates: list[RelationCandidate] = Field(default_factory=list)
    decisions: list[RelationCandidateDecision] = Field(default_factory=list)
    scorer_name: str = "unknown"
    source_text: str = ""
    issues: list[str] = Field(default_factory=list)

    def decision_by_id(self) -> dict[str, RelationCandidateDecision]:
        return {decision.candidate_id: decision for decision in self.decisions}

    def accepted_candidates(self) -> list[RelationCandidate]:
        decisions = self.decision_by_id()
        return [
            candidate
            for candidate in self.candidates
            if decisions.get(candidate.candidate_id)
            and decisions[candidate.candidate_id].is_supported
        ]

    def rejected_candidates(self) -> list[RelationCandidate]:
        decisions = self.decision_by_id()
        return [
            candidate
            for candidate in self.candidates
            if decisions.get(candidate.candidate_id)
            and not decisions[candidate.candidate_id].is_supported
        ]

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.accepted_candidates():
            counts[candidate.relation_type] = counts.get(candidate.relation_type, 0) + 1
        return counts


class RelationCandidateGenerator(Protocol):
    """Generates legal relation candidates from known entities and annotations."""

    def generate(
        self,
        *,
        actions: list[EntityMention],
        tools: list[EntityMention] | None = None,
        objects: list[EntityMention] | None = None,
        scenes: list[EntityMention] | None = None,
        procedures: list[EntityMention] | None = None,
        keysteps: list[EntityMention] | None = None,
        warnings: list[EntityMention] | None = None,
        source_text: str = "",
    ) -> list[RelationCandidate]:
        """Return candidate relations. Implementations must not call an LLM."""


class RelationCandidateScorer(Protocol):
    """Scores or validates generated candidates."""

    scorer_name: str

    def score(
        self,
        candidates: list[RelationCandidate],
        *,
        source_text: str,
    ) -> RelationValidationReport:
        """Return one binary decision per candidate."""


@dataclass
class ProceduralRelationCandidateGenerator:
    """Rule-based relation-candidate generator for industrial procedural text."""

    min_overlap_ratio: float = 0.01

    def generate(
        self,
        *,
        actions: list[EntityMention],
        tools: list[EntityMention] | None = None,
        objects: list[EntityMention] | None = None,
        scenes: list[EntityMention] | None = None,
        procedures: list[EntityMention] | None = None,
        keysteps: list[EntityMention] | None = None,
        warnings: list[EntityMention] | None = None,
        source_text: str = "",
    ) -> list[RelationCandidate]:
        tools = tools or []
        objects = objects or []
        scenes = scenes or []
        procedures = procedures or []
        keysteps = keysteps or []
        warnings = warnings or []
        candidates: list[RelationCandidate] = []

        def add(
            relation_type: RelationCandidateType,
            subject: EntityMention,
            target: EntityMention,
            rule: str,
            evidence: str,
            confidence: float,
        ) -> None:
            candidates.append(
                RelationCandidate(
                    candidate_id=f"c_{len(candidates) + 1:03d}",
                    relation_type=relation_type,
                    subject_id=subject.entity_id,
                    object_id=target.entity_id,
                    subject_label=subject.label,
                    object_label=target.label,
                    subject_name=subject.name,
                    object_name=target.name,
                    evidence_hint=evidence,
                    generation_rule=rule,
                    confidence=confidence,
                )
            )

        for action in actions:
            for tool in tools:
                evidence = _best_evidence(action, tool)
                if _same_temporal_region(action, tool) or _same_action_evidence(action, tool) or _tool_match(action, tool):
                    add("USES_TOOL", action, tool, "tool_match_or_same_context", evidence, 0.78)

            for obj in objects:
                evidence = _best_evidence(action, obj)
                if _action_object_cooccur(action, obj, source_text):
                    add("ACTS_ON", action, obj, "action_object_cooccurrence", evidence, 0.74)

            for scene in scenes:
                add("OBSERVED_IN", action, scene, "all_actions_observed_in_scene", action.evidence_text, 1.0)

        for before, after in _ordered_pairs(actions):
            add("BEFORE", before, after, "action_sequence_or_annotation_order", _best_evidence(before, after), 0.95)

        containers = [*procedures, *keysteps]
        for action in actions:
            for container in containers:
                if _same_temporal_region(action, container) or _same_sentence(source_text, action.name, container.name):
                    add("PART_OF", action, container, "time_overlap_or_keystep_context", _best_evidence(action, container), 0.82)

        warning_targets = [*actions, *keysteps]
        for warning in warnings:
            for target in warning_targets:
                if _same_temporal_region(warning, target) or _warning_step_match(warning, target):
                    add("WARNING_FOR", warning, target, "warning_step_or_time_match", _best_evidence(warning, target), 0.86)

        return _deduplicate_candidates(candidates)


class DeterministicRelationCandidateScorer:
    """Accept candidates generated by high-precision deterministic rules."""

    scorer_name = "deterministic_rules"

    def score(
        self,
        candidates: list[RelationCandidate],
        *,
        source_text: str,
    ) -> RelationValidationReport:
        decisions = [
            RelationCandidateDecision(
                candidate_id=candidate.candidate_id,
                is_supported=True,
                confidence=candidate.confidence or 0.75,
                evidence_text=candidate.evidence_hint,
                reason=f"Accepted by deterministic rule: {candidate.generation_rule}",
                scorer_name=self.scorer_name,
            )
            for candidate in candidates
        ]
        return RelationValidationReport(
            candidates=candidates,
            decisions=decisions,
            scorer_name=self.scorer_name,
            source_text=source_text,
        )


class LLMBinaryJudgeScorer:
    """Binary LLM judge for existing candidate IDs.

    The injected backend must expose ``extract(text, response_model,
    system_prompt=None)`` and return a ``RelationJudgeBatch``.
    """

    scorer_name = "llm_binary_judge"

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def score(
        self,
        candidates: list[RelationCandidate],
        *,
        source_text: str,
    ) -> RelationValidationReport:
        prompt = build_binary_judge_prompt(candidates, source_text)
        try:
            result = self.backend.extract(
                prompt,
                RelationJudgeBatch,
                system_prompt=binary_judge_system_prompt(),
            )
        except TypeError:
            result = self.backend.extract(prompt, RelationJudgeBatch)
        decisions = _sanitize_judge_decisions(candidates, result.decisions, self.scorer_name)
        return RelationValidationReport(
            candidates=candidates,
            decisions=decisions,
            scorer_name=self.scorer_name,
            source_text=source_text,
        )


class OptionalGLiRELScorer:
    """Optional adapter slot for GLiREL-style tuple scoring.

    This class deliberately has no hard dependency. A concrete GLiREL model can
    be injected later and must provide a ``score(candidate, source_text)`` style
    callable returning a confidence between 0 and 1.
    """

    scorer_name = "optional_glirel"

    def __init__(self, scorer: Any, threshold: float = 0.5) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def score(
        self,
        candidates: list[RelationCandidate],
        *,
        source_text: str,
    ) -> RelationValidationReport:
        decisions: list[RelationCandidateDecision] = []
        for candidate in candidates:
            confidence = float(self.scorer.score(candidate, source_text))
            decisions.append(
                RelationCandidateDecision(
                    candidate_id=candidate.candidate_id,
                    is_supported=confidence >= self.threshold,
                    confidence=max(0.0, min(1.0, confidence)),
                    evidence_text=candidate.evidence_hint,
                    reason="GLiREL-style tuple scorer result.",
                    scorer_name=self.scorer_name,
                )
            )
        return RelationValidationReport(
            candidates=candidates,
            decisions=decisions,
            scorer_name=self.scorer_name,
            source_text=source_text,
        )


class RelationJudgeBatch(BaseModel):
    """Strict binary judge output. It must not contain new relation endpoints."""

    decisions: list[RelationCandidateDecision] = Field(default_factory=list)


def binary_judge_system_prompt() -> str:
    return (
        "You are a strict binary relation validator. Do not invent relation "
        "types, endpoints, or candidate IDs. For every candidate, return only "
        "whether the candidate is supported by the source text."
    )


def build_binary_judge_prompt(candidates: list[RelationCandidate], source_text: str) -> str:
    payload = {
        "task": "Judge whether each candidate relation is supported by the source text.",
        "required_output": {
            "decisions": [
                {
                    "candidate_id": "c_001",
                    "is_supported": True,
                    "confidence": 0.82,
                    "evidence_text": "source evidence",
                    "reason": "why the relation is supported or rejected",
                }
            ]
        },
        "source_text": source_text,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _sanitize_judge_decisions(
    candidates: list[RelationCandidate],
    decisions: list[RelationCandidateDecision],
    scorer_name: str,
) -> list[RelationCandidateDecision]:
    allowed = {candidate.candidate_id for candidate in candidates}
    by_id = {
        decision.candidate_id: decision
        for decision in decisions
        if decision.candidate_id in allowed
    }
    sanitized: list[RelationCandidateDecision] = []
    for candidate in candidates:
        decision = by_id.get(candidate.candidate_id)
        if decision is None:
            sanitized.append(
                RelationCandidateDecision(
                    candidate_id=candidate.candidate_id,
                    is_supported=False,
                    confidence=0.0,
                    evidence_text="",
                    reason="LLM judge did not return this candidate ID.",
                    scorer_name=scorer_name,
                )
            )
            continue
        decision.scorer_name = scorer_name
        sanitized.append(decision)
    return sanitized


def _ordered_pairs(actions: list[EntityMention]) -> list[tuple[EntityMention, EntityMention]]:
    ordered = sorted(
        actions,
        key=lambda entity: (
            entity.sequence_index if entity.sequence_index is not None else 10**9,
            entity.start_seconds if entity.start_seconds is not None else 10**9,
            entity.entity_id,
        ),
    )
    return list(zip(ordered, ordered[1:]))


def _deduplicate_candidates(candidates: list[RelationCandidate]) -> list[RelationCandidate]:
    seen: set[tuple[str, str, str]] = set()
    result: list[RelationCandidate] = []
    for candidate in candidates:
        key = (candidate.relation_type, candidate.subject_id, candidate.object_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate.model_copy(update={"candidate_id": f"c_{len(result) + 1:03d}"}))
    return result


def _same_temporal_region(left: EntityMention, right: EntityMention) -> bool:
    if left.start_seconds is None or left.end_seconds is None:
        return False
    if right.start_seconds is None or right.end_seconds is None:
        return False
    start = max(left.start_seconds, right.start_seconds)
    end = min(left.end_seconds, right.end_seconds)
    return end >= start


def _same_sentence(source_text: str, left_name: str, right_name: str) -> bool:
    left = normalize_name(left_name)
    right = normalize_name(right_name)
    if not left or not right:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
        normalized = normalize_name(sentence)
        if left in normalized and right in normalized:
            return True
    return False


def _tool_match(action: EntityMention, tool: EntityMention) -> bool:
    haystack = normalize_name(f"{action.name} {action.evidence_text}")
    tool_name = normalize_name(tool.name)
    if tool_name and tool_name in haystack:
        return True
    tool_tokens = _tokens(tool.name)
    return bool(tool_tokens and tool_tokens <= _tokens(haystack))


def _same_action_evidence(action: EntityMention, target: EntityMention) -> bool:
    evidence = normalize_name(action.evidence_text)
    target_name = normalize_name(target.name)
    if not evidence or not target_name:
        return False
    return target_name in evidence or _tokens(target.name) <= _tokens(evidence)


def _action_object_cooccur(action: EntityMention, obj: EntityMention, source_text: str) -> bool:
    if _same_sentence(source_text, action.name, obj.name):
        return True
    action_tokens = _tokens(action.name)
    object_tokens = _tokens(obj.name)
    evidence_tokens = _tokens(f"{action.name} {action.evidence_text}")
    return bool(action_tokens and object_tokens and object_tokens <= evidence_tokens)


def _warning_step_match(warning: EntityMention, target: EntityMention) -> bool:
    warning_text = normalize_name(f"{warning.name} {warning.evidence_text} {warning.metadata.get('step', '')}")
    target_text = normalize_name(f"{target.name} {target.evidence_text}")
    return bool(warning_text and target_text and (_tokens(target_text) & _tokens(warning_text)))


def _best_evidence(left: EntityMention, right: EntityMention) -> str:
    if left.evidence_text and right.evidence_text and left.evidence_text == right.evidence_text:
        return left.evidence_text
    if left.evidence_text and normalize_name(right.name) in normalize_name(left.evidence_text):
        return left.evidence_text
    if right.evidence_text and normalize_name(left.name) in normalize_name(right.evidence_text):
        return right.evidence_text
    return left.evidence_text or right.evidence_text or f"{left.name} -> {right.name}"


def _tokens(value: str) -> set[str]:
    stopwords = {"a", "an", "and", "in", "into", "of", "on", "the", "then", "to", "with"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_name(value))
        if len(token) > 1 and token not in stopwords
    }
