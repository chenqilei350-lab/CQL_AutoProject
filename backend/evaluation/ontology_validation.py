"""
Ontology conformance and grounding validation for extracted KG facts.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from backend.schemas.base import KGEntity, KGRelation
from backend.schemas.egocentric_video import EgocentricVideoExtraction, Provenance
from backend.schemas.ontology import OntologySpec, build_egocentric_ontology


class ValidationIssue(BaseModel):
    """One ontology or grounding validation issue."""

    severity: str
    kind: str
    relation_type: str | None = None
    field: str | None = None
    message: str


class ValidationReport(BaseModel):
    """Aggregated validation metrics for one extraction result."""

    total_relations: int = 0
    valid_relations: int = 0
    invalid_relations: int = 0
    ontology_conformance: float = 1.0
    relation_hallucination_rate: float = 0.0
    subject_grounding_rate: float = 1.0
    object_grounding_rate: float = 1.0
    evidence_grounding_rate: float = 1.0
    subject_hallucinations: int = 0
    object_hallucinations: int = 0
    evidence_hallucinations: int = 0
    filtered_relation_count: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)


def validate_egocentric_extraction(
    extraction: EgocentricVideoExtraction,
    source_text: str | None = None,
    ontology: OntologySpec | None = None,
    grounding_threshold: float = 0.5,
) -> ValidationReport:
    """
    Validate relation conformance and text grounding for egocentric extraction.

    The report is intentionally metric-oriented so it can be reused in tests and
    experiments without depending on graph construction.
    """

    ontology = ontology or build_egocentric_ontology()
    source_text = source_text or extraction.source_text or ""
    relation_entries = _egocentric_relations(extraction)
    report = ValidationReport(total_relations=len(relation_entries))
    subject_checks = 0
    object_checks = 0
    evidence_checks = 0

    for relation_label, relation in relation_entries:
        spec = ontology.relation_by_label(relation_label)
        if spec is None:
            report.invalid_relations += 1
            report.issues.append(
                ValidationIssue(
                    severity="error",
                    kind="invalid_relation_type",
                    relation_type=relation_label,
                    message=f"Relation type {relation_label} is not allowed by the ontology.",
                )
            )
            continue

        endpoint_issues = _validate_relation_endpoints(relation, spec.endpoints)
        if endpoint_issues:
            report.invalid_relations += 1
            report.issues.extend(endpoint_issues)
        else:
            report.valid_relations += 1

        endpoint_values = [
            getattr(relation, field_name, None)
            for field_name in spec.endpoints
        ]
        if endpoint_values:
            subject_checks += 1
            if not _is_grounded(endpoint_values[0], source_text, grounding_threshold):
                report.subject_hallucinations += 1
                report.issues.append(
                    ValidationIssue(
                        severity="warning",
                        kind="ungrounded_subject",
                        relation_type=relation_label,
                        field=next(iter(spec.endpoints.keys()), None),
                        message=f"Subject {getattr(endpoint_values[0], 'name', endpoint_values[0])!r} is not grounded in source text.",
                    )
                )
        if len(endpoint_values) > 1:
            object_checks += 1
            if not _is_grounded(endpoint_values[1], source_text, grounding_threshold):
                report.object_hallucinations += 1
                report.issues.append(
                    ValidationIssue(
                        severity="warning",
                        kind="ungrounded_object",
                        relation_type=relation_label,
                        field=list(spec.endpoints.keys())[1],
                        message=f"Object {getattr(endpoint_values[1], 'name', endpoint_values[1])!r} is not grounded in source text.",
                    )
                )

        evidence = _evidence_text(relation)
        if evidence:
            evidence_checks += 1
            if not _text_supported(evidence, source_text, grounding_threshold):
                report.evidence_hallucinations += 1
                report.issues.append(
                    ValidationIssue(
                        severity="warning",
                        kind="ungrounded_evidence",
                        relation_type=relation_label,
                        message=f"Evidence text {evidence!r} is not supported by source text.",
                    )
                )

    if report.total_relations:
        report.ontology_conformance = round(report.valid_relations / report.total_relations, 4)
        report.relation_hallucination_rate = round(
            report.invalid_relations / report.total_relations,
            4,
        )
    if subject_checks:
        report.subject_grounding_rate = round(
            (subject_checks - report.subject_hallucinations) / subject_checks,
            4,
        )
    if object_checks:
        report.object_grounding_rate = round(
            (object_checks - report.object_hallucinations) / object_checks,
            4,
        )
    if evidence_checks:
        report.evidence_grounding_rate = round(
            (evidence_checks - report.evidence_hallucinations) / evidence_checks,
            4,
        )
    report.filtered_relation_count = (
        report.invalid_relations
        + report.subject_hallucinations
        + report.object_hallucinations
        + report.evidence_hallucinations
    )
    return report


def _egocentric_relations(
    extraction: EgocentricVideoExtraction,
) -> list[tuple[str, KGRelation]]:
    relation_lists: list[tuple[str, list[KGRelation]]] = [
        ("USES_TOOL", extraction.uses_tool),
        ("ACTS_ON", extraction.acts_on_object),
        ("BEFORE", extraction.action_order),
        ("CAUSES", extraction.action_causes),
        ("PART_OF", extraction.part_of_procedure),
        ("OBSERVED_IN", extraction.observed_in_scene),
    ]
    return [
        (label, relation)
        for label, relations in relation_lists
        for relation in relations
    ]


def _validate_relation_endpoints(
    relation: KGRelation,
    endpoints: dict[str, str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field_name, expected_type in endpoints.items():
        value = getattr(relation, field_name, None)
        actual_type = type(value).__name__ if value is not None else None
        if actual_type != expected_type:
            issues.append(
                ValidationIssue(
                    severity="error",
                    kind="domain_range_violation",
                    relation_type=type(relation).__name__,
                    field=field_name,
                    message=f"Expected {field_name} to be {expected_type}, got {actual_type}.",
                )
            )
    return issues


def _is_grounded(value: Any, source_text: str, threshold: float) -> bool:
    if isinstance(value, KGEntity):
        candidates = [
            value.name,
            value.source_text,
            value.evidence_text,
        ]
        provenance = getattr(value, "provenance", None)
        if isinstance(provenance, Provenance):
            candidates.append(provenance.source_text)
    else:
        candidates = [str(value)]
    return any(_text_supported(candidate, source_text, threshold) for candidate in candidates)


def _evidence_text(value: BaseModel) -> str | None:
    evidence = getattr(value, "evidence_text", None) or getattr(value, "source_text", None)
    provenance = getattr(value, "provenance", None)
    if evidence:
        return str(evidence)
    if isinstance(provenance, Provenance) and provenance.source_text:
        return provenance.source_text
    return None


def _text_supported(candidate: str | None, source_text: str, threshold: float) -> bool:
    if not candidate:
        return False
    normalized_candidate = _normalize(candidate)
    normalized_source = _normalize(source_text)
    if not normalized_candidate or not normalized_source:
        return False
    if normalized_candidate in normalized_source:
        return True
    candidate_tokens = _tokens(normalized_candidate)
    source_tokens = _tokens(normalized_source)
    if not candidate_tokens:
        return False
    return len(candidate_tokens & source_tokens) / len(candidate_tokens) >= threshold


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _tokens(value: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9_]+", value.lower())
    normalized = set()
    for token in raw_tokens:
        if len(token) > 4 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 3 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        normalized.add(token)
    return normalized
