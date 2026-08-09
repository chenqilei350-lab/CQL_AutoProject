#!/usr/bin/env python3
"""
Post-processing utilities for the hospital gold-case KG experiments.

This module is deliberately scoped to ``gold_case_admission_29366372`` style
data.  It normalizes model-produced IDs/relations, merges duplicate edges, and
checks whether nodes/edges are supported by the source CSV rows.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


NodeFact = tuple[str, str]
EdgeFact = tuple[str, str, str]


@dataclass
class HospitalPostprocessStats:
    """Counters and messages produced during KG post-processing."""

    invalid_id_count: int = 0
    invalid_relation_count: int = 0
    wrong_direction_count: int = 0
    merged_duplicate_edges: int = 0
    auto_corrected_edges: int = 0
    hallucinated_nodes: int = 0
    hallucinated_edges: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class HospitalPostprocessResult:
    """Normalized graph facts plus post-processing diagnostics."""

    nodes: set[NodeFact]
    edges: set[EdgeFact]
    stats: HospitalPostprocessStats


@dataclass
class HospitalPostprocessContext:
    """Indexes derived from source CSV and the gold-case schema."""

    canonical_nodes: set[NodeFact]
    supported_edges: set[EdgeFact]
    id_aliases: dict[tuple[str, str], str]
    id_to_label: dict[str, str]
    edge_schema: dict[str, tuple[str, str]]
    relation_aliases: dict[str, str]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_hospital_postprocess_context(gold_dir: str | Path) -> HospitalPostprocessContext:
    """Build canonical ID and supported-edge indexes from a gold-case directory."""

    base = Path(gold_dir)
    schema = json.loads((base / "gold_case_schema.json").read_text(encoding="utf-8"))
    patient_rows = read_csv_rows(base / "source_patient.csv")
    admission_rows = read_csv_rows(base / "source_admission.csv")
    diagnosis_rows = read_csv_rows(base / "source_diagnoses.csv")
    procedure_rows = read_csv_rows(base / "source_procedures.csv")
    medication_rows = read_csv_rows(base / "source_medications.csv")
    lab_event_rows = read_csv_rows(base / "source_lab_events.csv")
    lab_item_rows = read_csv_rows(base / "source_lab_items.csv")

    canonical_nodes: set[NodeFact] = set()
    supported_edges: set[EdgeFact] = set()
    id_aliases: dict[tuple[str, str], str] = {}
    id_to_label: dict[str, str] = {}

    def add_node(label: str, canonical_id: str, *aliases: str | None) -> None:
        canonical_nodes.add((label, canonical_id))
        id_to_label[canonical_id] = label
        for alias in (canonical_id, *aliases):
            if alias:
                id_aliases[(label, _key(alias))] = canonical_id

    for row in patient_rows:
        add_node("Patient", row["patient_id"], row.get("subject_id"))

    for row in admission_rows:
        add_node("Admission", row["admission_id"], row.get("hadm_id"))
        supported_edges.add((row["patient_id"], "has_admission", row["admission_id"]))

    for row in diagnosis_rows:
        version = row["icd_version"]
        code = row["icd_code"]
        add_node(
            "Diagnosis",
            row["diagnosis_id"],
            code,
            f"{version}:{code}",
            f"{code}:{version}",
        )
        supported_edges.add((row["admission_id"], "has_diagnosis", row["diagnosis_id"]))

    for row in procedure_rows:
        version = row["icd_version"]
        code = row["icd_code"]
        add_node(
            "Procedure",
            row["procedure_id"],
            code,
            f"{version}:{code}",
            f"{code}:{version}",
        )
        supported_edges.add((row["admission_id"], "has_procedure", row["procedure_id"]))

    for row in medication_rows:
        add_node(
            "Medication",
            row["medication_id"],
            row.get("drug"),
            _medication_slug(row.get("drug", "")),
        )
        supported_edges.add((row["admission_id"], "prescribed_medication", row["medication_id"]))

    for row in lab_event_rows:
        add_node("LabEvent", row["lab_event_id"], row.get("labevent_id"))
        supported_edges.add((row["admission_id"], "has_lab_event", row["lab_event_id"]))
        supported_edges.add((row["lab_event_id"], "is_test_of", row["lab_item_id"]))

    for row in lab_item_rows:
        add_node("LabItem", row["lab_item_id"], row.get("itemid"))

    edge_schema = {
        relation: _parse_domain_range(domain_range)
        for relation, domain_range in schema.get("edge_schema", {}).items()
    }

    relation_aliases = _default_relation_aliases(edge_schema)
    return HospitalPostprocessContext(
        canonical_nodes=canonical_nodes,
        supported_edges=supported_edges,
        id_aliases=id_aliases,
        id_to_label=id_to_label,
        edge_schema=edge_schema,
        relation_aliases=relation_aliases,
    )


def postprocess_hospital_kg(
    nodes: set[NodeFact],
    edges: set[EdgeFact],
    context: HospitalPostprocessContext,
) -> HospitalPostprocessResult:
    """Normalize nodes/edges, merge duplicate edges, and flag hallucinations."""

    stats = HospitalPostprocessStats()
    normalized_nodes: set[NodeFact] = set()
    node_id_map: dict[str, str] = {}

    for label, node_id in nodes:
        normalized = normalize_node(label, node_id, context)
        if normalized is None:
            stats.invalid_id_count += 1
            stats.hallucinated_nodes += 1
            stats.issues.append(f"invalid_node_id:{label}:{node_id}")
            continue
        normalized_nodes.add(normalized)
        node_id_map[node_id] = normalized[1]
        if normalized not in context.canonical_nodes:
            stats.hallucinated_nodes += 1
            stats.issues.append(f"unsupported_node:{normalized[0]}:{normalized[1]}")

    normalized_edges: set[EdgeFact] = set()
    for source_id, relation, target_id in edges:
        normalized_relation = normalize_relation(relation, context)
        if not normalized_relation:
            stats.invalid_relation_count += 1
            stats.issues.append(f"invalid_relation:{relation}")
            continue

        normalized_edge = normalize_edge(
            source_id=source_id,
            relation=normalized_relation,
            target_id=target_id,
            context=context,
            node_id_map=node_id_map,
        )
        if normalized_edge is None:
            stats.invalid_id_count += 1
            stats.issues.append(f"invalid_edge_endpoint:{source_id}:{normalized_relation}:{target_id}")
            continue

        corrected_edge, was_reversed = maybe_correct_reverse_edge(normalized_edge, context)
        if was_reversed:
            stats.wrong_direction_count += 1
            stats.auto_corrected_edges += 1
            normalized_edge = corrected_edge

        if not edge_has_valid_domain_range(normalized_edge, context):
            stats.wrong_direction_count += 1
            stats.issues.append(
                f"domain_range_violation:{normalized_edge[0]}:{normalized_edge[1]}:{normalized_edge[2]}"
            )

        if normalized_edge in normalized_edges:
            stats.merged_duplicate_edges += 1
            continue
        normalized_edges.add(normalized_edge)

    for edge in normalized_edges:
        if edge not in context.supported_edges:
            stats.hallucinated_edges += 1
            stats.issues.append(f"unsupported_edge:{edge[0]}:{edge[1]}:{edge[2]}")

    return HospitalPostprocessResult(nodes=normalized_nodes, edges=normalized_edges, stats=stats)


def normalize_node(
    label: str,
    node_id: str,
    context: HospitalPostprocessContext,
) -> NodeFact | None:
    """Normalize a node label and ID to the canonical source-supported form."""

    normalized_label = normalize_label(label)
    if not normalized_label:
        return None

    raw_id = str(node_id).strip()
    direct = context.id_aliases.get((normalized_label, _key(raw_id)))
    if direct:
        return (normalized_label, direct)

    prefixed = _prefix_guess(normalized_label, raw_id)
    if prefixed:
        aliased = context.id_aliases.get((normalized_label, _key(prefixed)))
        if aliased:
            return (normalized_label, aliased)
        return (normalized_label, prefixed)

    return None


def normalize_relation(relation: str, context: HospitalPostprocessContext) -> str | None:
    """Normalize relation labels and known aliases to schema labels."""

    return context.relation_aliases.get(_relation_key(relation))


def normalize_edge(
    source_id: str,
    relation: str,
    target_id: str,
    context: HospitalPostprocessContext,
    node_id_map: dict[str, str],
) -> EdgeFact | None:
    """Normalize both endpoints of an edge using schema domain/range labels."""

    endpoints = context.edge_schema.get(relation)
    if not endpoints:
        return None
    domain, range_ = endpoints

    source = _normalize_endpoint_id(source_id, domain, context, node_id_map)
    target = _normalize_endpoint_id(target_id, range_, context, node_id_map)
    if source and target:
        return (source, relation, target)

    # Try reversed endpoints.  The caller will record the reversal.
    reversed_source = _normalize_endpoint_id(source_id, range_, context, node_id_map)
    reversed_target = _normalize_endpoint_id(target_id, domain, context, node_id_map)
    if reversed_source and reversed_target:
        return (reversed_source, relation, reversed_target)
    return None


def maybe_correct_reverse_edge(
    edge: EdgeFact,
    context: HospitalPostprocessContext,
) -> tuple[EdgeFact, bool]:
    """Flip an edge when the schema uniquely proves that it is reversed."""

    source_id, relation, target_id = edge
    expected = context.edge_schema.get(relation)
    if not expected:
        return edge, False
    domain, range_ = expected
    source_label = context.id_to_label.get(source_id)
    target_label = context.id_to_label.get(target_id)

    if source_label == domain and target_label == range_:
        return edge, False
    if source_label == range_ and target_label == domain:
        return (target_id, relation, source_id), True
    return edge, False


def edge_has_valid_domain_range(edge: EdgeFact, context: HospitalPostprocessContext) -> bool:
    """Return whether an edge respects the schema domain/range labels."""

    source_id, relation, target_id = edge
    expected = context.edge_schema.get(relation)
    if not expected:
        return False
    domain, range_ = expected
    return context.id_to_label.get(source_id) == domain and context.id_to_label.get(target_id) == range_


def _normalize_endpoint_id(
    value: str,
    expected_label: str,
    context: HospitalPostprocessContext,
    node_id_map: dict[str, str],
) -> str | None:
    if value in node_id_map:
        mapped = node_id_map[value]
        return mapped if context.id_to_label.get(mapped) == expected_label else None

    known_label = _known_label_for_alias(value, context)
    if known_label and known_label != expected_label:
        return None

    normalized = normalize_node(expected_label, value, context)
    return normalized[1] if normalized else None


def _known_label_for_alias(value: str, context: HospitalPostprocessContext) -> str | None:
    """Return the known source label for an ID/alias, regardless of expected label."""

    key = _key(value)
    for (label, alias), canonical_id in context.id_aliases.items():
        if alias == key and context.id_to_label.get(canonical_id) == label:
            return label
    return None


def normalize_label(label: str) -> str | None:
    """Normalize common label spelling variants."""

    key = re.sub(r"[^a-z0-9]+", "", str(label).lower())
    return {
        "patient": "Patient",
        "admission": "Admission",
        "diagnosis": "Diagnosis",
        "procedure": "Procedure",
        "medication": "Medication",
        "drug": "Medication",
        "labevent": "LabEvent",
        "labitem": "LabItem",
        "labtest": "LabItem",
    }.get(key)


def _prefix_guess(label: str, value: str) -> str | None:
    """Construct a canonical-looking ID when the label makes it unambiguous."""

    clean = str(value).strip()
    if not clean:
        return None
    if ":" in clean and clean.split(":", 1)[0].lower() in {
        "patient",
        "admission",
        "diagnosis",
        "procedure",
        "medication",
        "labevent",
        "labitem",
    }:
        prefix, rest = clean.split(":", 1)
        canonical_prefix = {
            "patient": "patient",
            "admission": "admission",
            "diagnosis": "diagnosis",
            "procedure": "procedure",
            "medication": "medication",
            "labevent": "labevent",
            "labitem": "labitem",
        }[prefix.lower()]
        return f"{canonical_prefix}:{rest}"

    if label == "Patient" and clean.isdigit():
        return f"patient:{clean}"
    if label == "Admission" and clean.isdigit():
        return f"admission:{clean}"
    if label == "LabEvent" and clean.isdigit():
        return f"labevent:{clean}"
    if label == "LabItem" and clean.isdigit():
        return f"labitem:{clean}"
    if label in {"Diagnosis", "Procedure"}:
        parts = clean.split(":")
        if len(parts) == 2:
            left, right = parts
            if left.isdigit():
                return f"{label.lower()}:{left}:{right}"
            if right.isdigit():
                return f"{label.lower()}:{right}:{left}"
        return f"{label.lower()}:{clean}"
    if label == "Medication":
        return f"medication:{_medication_slug(clean)}"
    return None


def _default_relation_aliases(edge_schema: dict[str, tuple[str, str]]) -> dict[str, str]:
    aliases = {_relation_key(relation): relation for relation in edge_schema}
    aliases.update(
        {
            _relation_key("HAS_DIAGNOSIS"): "has_diagnosis",
            _relation_key("diagnosis"): "has_diagnosis",
            _relation_key("has_procedure"): "has_procedure",
            _relation_key("HAS_PROCEDURE"): "has_procedure",
            _relation_key("has_medication"): "prescribed_medication",
            _relation_key("medication"): "prescribed_medication",
            _relation_key("prescribed"): "prescribed_medication",
            _relation_key("HAS_LAB_EVENT"): "has_lab_event",
            _relation_key("lab_event"): "has_lab_event",
            _relation_key("is_test_of"): "is_test_of",
            _relation_key("test_of"): "is_test_of",
            _relation_key("HAS_ADMISSION"): "has_admission",
        }
    )
    return aliases


def _parse_domain_range(value: str) -> tuple[str, str]:
    left, right = value.split("->", 1)
    return left.strip(), right.strip()


def _relation_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _key(value: str) -> str:
    return str(value).strip().lower()


def _medication_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug
