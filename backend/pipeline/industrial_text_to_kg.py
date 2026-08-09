"""Industrial text-to-KG module contracts and post-processing helpers.

This module is the domain-neutral counterpart of the hospital pilot
post-processing code.  It keeps the useful ideas from the pilot experiment
without carrying over hospital-specific ID rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


IndustrialNodeType = Literal[
    "Scene",
    "Action",
    "Tool",
    "Object",
    "Parameter",
    "Procedure",
    "QualityCheck",
]
IndustrialRelationType = Literal[
    "USES_TOOL",
    "ACTS_ON",
    "HAS_PARAMETER",
    "BEFORE",
    "CAUSES",
    "PART_OF",
    "HAS_RESULT",
    "OBSERVED_IN",
]
InputType = Literal["raw", "unified"]
ExtractionStrategy = Literal["one_shot", "layered"]
RelationOrigin = Literal["llm_extracted", "evidence_fallback", "sequence_fallback", "postprocessed"]


DEFAULT_TOOL_TERMS: tuple[str, ...] = (
    "allen key",
    "brush",
    "caliper",
    "cleaning brush",
    "cleaning cloth",
    "cloth",
    "drill",
    "gauge",
    "hammer",
    "hex key",
    "measuring tape",
    "meter",
    "micrometer",
    "nylon brush",
    "paint brush",
    "pliers",
    "probe",
    "rag",
    "ruler",
    "scanner",
    "screwdriver",
    "sensor",
    "spanner",
    "torch",
    "torque wrench",
    "wire brush",
    "wrench",
)

OBJECT_CONTEXT_TERMS: tuple[str, ...] = (
    "assemble",
    "attach",
    "install",
    "mount",
    "move",
    "place",
    "position",
    "remove",
    "tighten",
)


@dataclass(frozen=True)
class IndustrialNode:
    """A source-supported industrial KG node candidate."""

    label: str
    name: str
    evidence_text: str | None = None
    segment_id: str | None = None


@dataclass(frozen=True)
class IndustrialEdge:
    """A source-supported industrial KG edge candidate."""

    source: str
    relation: str
    target: str
    evidence_text: str | None = None
    segment_id: str | None = None
    relation_origin: RelationOrigin | None = None


@dataclass
class IndustrialPostprocessStats:
    """Diagnostics for industrial KG post-processing."""

    hallucinated_nodes: int = 0
    hallucinated_edges: int = 0
    invalid_relation_count: int = 0
    wrong_direction_count: int = 0
    merged_duplicate_edges: int = 0
    auto_corrected_edges: int = 0
    relation_origin_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass
class IndustrialPostprocessResult:
    """Normalized industrial graph facts plus diagnostics."""

    nodes: set[tuple[str, str]]
    edges: set[tuple[str, str, str]]
    stats: IndustrialPostprocessStats


@dataclass(frozen=True)
class RelationOriginSummary:
    """Counts of final graph edges by relation type and extraction origin."""

    total_edges: int
    by_origin: dict[str, int]
    by_relation: dict[str, int]
    by_relation_origin: dict[str, int]


@dataclass(frozen=True)
class ModuleExperimentLogRecord:
    """A compact log record for deciding which modules should be retained."""

    input_type: InputType
    extraction_strategy: ExtractionStrategy
    model: str
    raw_node_f1: float
    raw_edge_f1: float
    normalized_node_f1: float
    normalized_edge_f1: float
    hallucination_count: int
    timeout_or_error: str = ""
    module_decision: str = ""
    notes: str = ""


RELATION_SCHEMA: dict[str, tuple[str, str]] = {
    "USES_TOOL": ("Action", "Tool"),
    "ACTS_ON": ("Action", "Object"),
    "HAS_PARAMETER": ("Action", "Parameter"),
    "BEFORE": ("Action", "Action"),
    "CAUSES": ("Action", "Action"),
    "PART_OF": ("Action", "Procedure"),
    "HAS_RESULT": ("Action", "QualityCheck"),
    "OBSERVED_IN": ("Action", "Scene"),
}

RELATION_ALIASES: dict[str, str] = {
    "uses_tool": "USES_TOOL",
    "use_tool": "USES_TOOL",
    "tool_used": "USES_TOOL",
    "acts_on": "ACTS_ON",
    "acted_on": "ACTS_ON",
    "has_parameter": "HAS_PARAMETER",
    "parameter": "HAS_PARAMETER",
    "before": "BEFORE",
    "precedes": "BEFORE",
    "then": "BEFORE",
    "causes": "CAUSES",
    "enables": "CAUSES",
    "part_of": "PART_OF",
    "belongs_to": "PART_OF",
    "has_result": "HAS_RESULT",
    "quality_result": "HAS_RESULT",
    "observed_in": "OBSERVED_IN",
}

LABEL_ALIASES: dict[str, str] = {
    "scene": "Scene",
    "segment": "Scene",
    "action": "Action",
    "step": "Action",
    "tool": "Tool",
    "object": "Object",
    "sceneobject": "Object",
    "workpiece": "Object",
    "parameter": "Parameter",
    "processparameter": "Parameter",
    "procedure": "Procedure",
    "qualitycheck": "QualityCheck",
    "result": "QualityCheck",
}


def postprocess_industrial_kg(
    nodes: list[IndustrialNode],
    edges: list[IndustrialEdge],
    source_text: str,
) -> IndustrialPostprocessResult:
    """Normalize industrial KG facts and flag unsupported facts."""

    stats = IndustrialPostprocessStats()
    normalized_nodes: set[tuple[str, str]] = set()
    node_label_by_name: dict[str, str] = {}

    for node in nodes:
        label = normalize_label(node.label)
        name = normalize_name(node.name)
        if label is None or not name:
            stats.hallucinated_nodes += 1
            stats.issues.append(f"invalid_node:{node.label}:{node.name}")
            continue
        normalized_nodes.add((label, name))
        node_label_by_name[name] = label
        if not _is_supported(node.evidence_text or node.name, source_text):
            stats.hallucinated_nodes += 1
            stats.issues.append(f"unsupported_node:{label}:{name}")

    normalized_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        relation = normalize_relation(edge.relation)
        if relation is None:
            stats.invalid_relation_count += 1
            stats.issues.append(f"invalid_relation:{edge.relation}")
            continue
        origin = edge.relation_origin or "llm_extracted"
        stats.relation_origin_counts[origin] = stats.relation_origin_counts.get(origin, 0) + 1
        source = normalize_name(edge.source)
        target = normalize_name(edge.target)
        candidate = (source, relation, target)
        corrected, was_reversed = correct_reversed_edge(candidate, node_label_by_name)
        if was_reversed:
            stats.wrong_direction_count += 1
            stats.auto_corrected_edges += 1
            candidate = corrected

        if not edge_has_valid_direction(candidate, node_label_by_name):
            stats.wrong_direction_count += 1
            stats.hallucinated_edges += 1
            stats.issues.append(f"domain_range_violation:{candidate[0]}:{candidate[1]}:{candidate[2]}")

        if not _is_supported(edge.evidence_text or f"{edge.source} {edge.target}", source_text):
            stats.hallucinated_edges += 1
            stats.issues.append(f"unsupported_edge:{candidate[0]}:{candidate[1]}:{candidate[2]}")

        if candidate in normalized_edges:
            stats.merged_duplicate_edges += 1
            continue
        normalized_edges.add(candidate)

    return IndustrialPostprocessResult(
        nodes=normalized_nodes,
        edges=normalized_edges,
        stats=stats,
    )


def normalize_label(label: str) -> str | None:
    """Normalize industrial node type aliases."""

    key = re.sub(r"[^a-z0-9]+", "", str(label).lower())
    return LABEL_ALIASES.get(key)


def normalize_relation(relation: str) -> str | None:
    """Normalize industrial relation aliases."""

    key = re.sub(r"[^a-z0-9]+", "_", str(relation).lower()).strip("_")
    if relation in RELATION_SCHEMA:
        return relation
    return RELATION_ALIASES.get(key)


def normalize_name(value: str) -> str:
    """Normalize an industrial entity name for matching."""

    return " ".join(str(value).casefold().strip().split())


def load_tool_terms(path: str | Path | None = None) -> set[str]:
    """Load maintainable industrial tool terms from a newline-delimited file."""

    terms = set(DEFAULT_TOOL_TERMS)
    if path is None:
        default_path = Path(__file__).resolve().parents[2] / "config" / "industrial_tool_keywords.txt"
        path = default_path if default_path.exists() else None
    if path is None:
        return terms
    keyword_path = Path(path)
    if not keyword_path.exists():
        return terms
    for line in keyword_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        terms.add(normalize_name(value))
    return terms


def classify_industrial_object(
    *,
    entity_id: str,
    name: str,
    object_type: str,
    evidence_text: str = "",
    tool_reference_ids: set[str] | None = None,
    tool_terms: set[str] | None = None,
) -> str:
    """Classify an extracted physical entity as ``Tool`` or ``Object`` conservatively."""

    if looks_like_industrial_tool(
        entity_id=entity_id,
        name=name,
        object_type=object_type,
        evidence_text=evidence_text,
        tool_reference_ids=tool_reference_ids or set(),
        tool_terms=tool_terms,
    ):
        return "Tool"
    return "Object"


def looks_like_industrial_tool(
    *,
    entity_id: str,
    name: str,
    object_type: str,
    evidence_text: str = "",
    tool_reference_ids: set[str] | None = None,
    tool_terms: set[str] | None = None,
) -> bool:
    """Return whether a physical entity is likely a tool under source-supported rules."""

    references = tool_reference_ids or set()
    if object_type.casefold() == "tool" or entity_id in references:
        return True
    normalized_name = normalize_name(name)
    normalized_evidence = normalize_name(evidence_text)
    terms = tool_terms or load_tool_terms()
    if any(term and term in normalized_name for term in terms):
        return True

    tool_context_patterns = (
        f"use {normalized_name}",
        f"uses {normalized_name}",
        f"using {normalized_name}",
        f"use a {normalized_name}",
        f"use an {normalized_name}",
        f"use the {normalized_name}",
        f"uses a {normalized_name}",
        f"uses an {normalized_name}",
        f"uses the {normalized_name}",
        f"using a {normalized_name}",
        f"using an {normalized_name}",
        f"using the {normalized_name}",
        f"with a {normalized_name}",
        f"with an {normalized_name}",
        f"with the {normalized_name}",
        f"by using {normalized_name}",
    )
    if normalized_name and any(pattern in normalized_evidence for pattern in tool_context_patterns):
        return True

    object_context_patterns = tuple(
        f"{verb} {normalized_name}" for verb in OBJECT_CONTEXT_TERMS if normalized_name
    )
    if any(pattern in normalized_evidence for pattern in object_context_patterns):
        return False
    return False


def summarize_relation_origins(graph_edges: list[object]) -> RelationOriginSummary:
    """Summarize final graph edges by relation type and origin for reports."""

    by_origin: dict[str, int] = {}
    by_relation: dict[str, int] = {}
    by_relation_origin: dict[str, int] = {}
    for edge in graph_edges:
        relation = getattr(edge, "type", "")
        properties = getattr(edge, "properties", {}) or {}
        origin = str(properties.get("relation_origin") or "unspecified")
        by_origin[origin] = by_origin.get(origin, 0) + 1
        by_relation[relation] = by_relation.get(relation, 0) + 1
        key = f"{relation}:{origin}"
        by_relation_origin[key] = by_relation_origin.get(key, 0) + 1
    return RelationOriginSummary(
        total_edges=len(graph_edges),
        by_origin=by_origin,
        by_relation=by_relation,
        by_relation_origin=by_relation_origin,
    )


def correct_reversed_edge(
    edge: tuple[str, str, str],
    node_label_by_name: dict[str, str],
) -> tuple[tuple[str, str, str], bool]:
    """Flip an edge when schema domain/range makes reversal unambiguous."""

    source, relation, target = edge
    expected = RELATION_SCHEMA.get(relation)
    if expected is None:
        return edge, False
    domain, range_ = expected
    source_label = node_label_by_name.get(source)
    target_label = node_label_by_name.get(target)
    if source_label == domain and target_label == range_:
        return edge, False
    if source_label == range_ and target_label == domain:
        return (target, relation, source), True
    return edge, False


def edge_has_valid_direction(
    edge: tuple[str, str, str],
    node_label_by_name: dict[str, str],
) -> bool:
    """Return whether an edge satisfies the industrial relation schema."""

    source, relation, target = edge
    expected = RELATION_SCHEMA.get(relation)
    if expected is None:
        return False
    domain, range_ = expected
    return node_label_by_name.get(source) == domain and node_label_by_name.get(target) == range_


def write_module_experiment_log(
    records: list[ModuleExperimentLogRecord],
    output_path: str | Path,
    title: str = "Module Screening Experiment Log",
) -> Path:
    """Write a short Markdown log for module-retention decisions."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "This log records which modules are useful before migrating to industrial data.",
        "",
        "| Input | Strategy | Model | Raw Node F1 | Raw Edge F1 | Normalized Node F1 | Normalized Edge F1 | Hallucinations | Error | Decision |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.input_type,
                    record.extraction_strategy,
                    record.model,
                    f"{record.raw_node_f1:.4f}",
                    f"{record.raw_edge_f1:.4f}",
                    f"{record.normalized_node_f1:.4f}",
                    f"{record.normalized_edge_f1:.4f}",
                    str(record.hallucination_count),
                    record.timeout_or_error or "",
                    record.module_decision or "",
                ]
            )
            + " |"
        )
        if record.notes:
            lines.append(f"\nNote: {record.notes}\n")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _is_supported(candidate: str | None, source_text: str) -> bool:
    if not candidate:
        return False
    return normalize_name(candidate) in normalize_name(source_text)
