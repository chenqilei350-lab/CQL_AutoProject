"""
Ontology helpers derived from the project's Pydantic schemas.

The MVP treats Pydantic classes as an executable ontology: entity classes
describe graph node types, relation classes describe allowed edge/relation
types, and typed relation endpoint fields define domain/range constraints.
"""

from __future__ import annotations

from hashlib import blake2b
import json
import re
from types import UnionType
from typing import Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from backend.schemas.base import KGEntity, KGRelation
from backend.schemas.egocentric_video import (
    ActionCauses,
    ActionObservedInScene,
    ActionOrder,
    ActionPartOfProcedure,
    ActsOnObject,
    UsesTool,
)


RELATION_LABELS: dict[type[KGRelation], str] = {
    UsesTool: "USES_TOOL",
    ActsOnObject: "ACTS_ON",
    ActionOrder: "BEFORE",
    ActionCauses: "CAUSES",
    ActionPartOfProcedure: "PART_OF",
    ActionObservedInScene: "OBSERVED_IN",
}

IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "Scene": ("video_id", "segment_id", "name"),
    "Action": ("video_id", "segment_id", "sequence_index", "name"),
    "Tool": ("manufacturer", "model", "name"),
    "Worker": ("name",),
    "Actor": ("name",),
    "SceneObject": ("name", "object_type"),
    "Object": ("name", "object_type"),
    "Procedure": ("procedure_id", "version", "name"),
    "ProcessParameter": ("name", "parameter_type", "unit"),
}


class EntitySpec(BaseModel):
    """A graph entity type discovered from Pydantic schemas."""

    name: str
    identity_fields: list[str]
    is_component: bool = False


class RelationSpec(BaseModel):
    """A relation type and its typed endpoint constraints."""

    name: str
    label: str
    endpoints: dict[str, str]

    @property
    def domain(self) -> str | None:
        return next(iter(self.endpoints.values()), None)

    @property
    def range(self) -> str | None:
        values = list(self.endpoints.values())
        return values[1] if len(values) > 1 else None


class OntologySpec(BaseModel):
    """Executable ontology summary used by prompts, validation, and graph export."""

    entities: dict[str, EntitySpec]
    relations: dict[str, RelationSpec]

    def relation_by_label(self, label: str) -> RelationSpec | None:
        for spec in self.relations.values():
            if spec.label == label:
                return spec
        return None


def normalize_identity_value(value: Any) -> str:
    """Normalize an identity value before hashing or comparison."""

    return " ".join(str(value).lower().strip().split())


def stable_node_id(
    label: str,
    name: str,
    properties: dict[str, Any] | None = None,
    *,
    context_fields: tuple[str, ...] = (
        "entity_id",
        "video_id",
        "segment_id",
        "scene_id",
    ),
) -> str:
    """
    Create a deterministic node ID from type, normalized name, and optional context.

    Context is included only when present so global tools/workers stay reusable,
    while scene-bound nodes can still be disambiguated.
    """

    properties = properties or {}
    identity = {
        "label": label,
        "name": normalize_identity_value(name),
    }
    for field in context_fields:
        if properties.get(field) not in (None, ""):
            identity[field] = normalize_identity_value(properties[field])
    digest = blake2b(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_") or "Node"
    return f"{safe_label}:{digest}"


def build_egocentric_ontology() -> OntologySpec:
    """Build the ontology spec for the egocentric extraction contract."""

    from backend.schemas.egocentric_video import (
        Action,
        EgocentricVideoExtraction,
        Scene,
        SceneObject,
    )
    from backend.schemas.process_knowledge.entities import (
        ProcessParameter,
        Procedure,
        Tool,
        Worker,
    )

    entity_classes: list[type[KGEntity]] = [
        Scene,
        Procedure,
        Worker,
        Action,
        Tool,
        SceneObject,
        ProcessParameter,
    ]
    relation_classes = [
        UsesTool,
        ActsOnObject,
        ActionOrder,
        ActionCauses,
        ActionPartOfProcedure,
        ActionObservedInScene,
    ]

    entities = {
        cls.__name__: EntitySpec(
            name=cls.__name__,
            identity_fields=list(IDENTITY_FIELDS.get(cls.__name__, ("name",))),
            is_component=False,
        )
        for cls in entity_classes
    }
    relations = {
        cls.__name__: RelationSpec(
            name=cls.__name__,
            label=RELATION_LABELS.get(cls, camel_to_upper_snake(cls.__name__)),
            endpoints=_relation_endpoints(cls),
        )
        for cls in relation_classes
    }

    # Keep the root class referenced by importers and future prompt builders.
    _ = EgocentricVideoExtraction
    return OntologySpec(entities=entities, relations=relations)


def compact_schema_guide(model: type[BaseModel], max_chars: int = 3500) -> str:
    """Generate compact field guidance from Pydantic descriptions and enums."""

    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    lines: list[str] = []
    total = 0

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            resolved = defs.get(ref.split("/")[-1])
            if isinstance(resolved, dict):
                return resolved
        return node

    def walk(node: dict[str, Any], prefix: str = "", depth: int = 0) -> None:
        nonlocal total
        if depth > 2 or total >= max_chars:
            return
        node = resolve(node)
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            return
        required = set(node.get("required", []))
        for field_name, raw in properties.items():
            if not isinstance(raw, dict):
                continue
            prop = resolve(raw)
            path = f"{prefix}.{field_name}" if prefix else field_name
            bits = [f"- {path}"]
            if field_name in required:
                bits.append("(required)")
            description = str(prop.get("description") or "").strip()
            if description:
                bits.append(description[:180])
            enum_values = prop.get("enum")
            if isinstance(enum_values, list):
                bits.append("Allowed: " + ", ".join(map(str, enum_values[:12])))
            line = " ".join(bits)
            if total + len(line) + 1 > max_chars:
                return
            lines.append(line)
            total += len(line) + 1
            if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
                walk(prop["items"], f"{path}[]", depth + 1)
            elif prop.get("type") == "object" or isinstance(prop.get("properties"), dict):
                walk(prop, path, depth + 1)

    walk(schema)
    return "\n".join(lines) if lines else "Extract fields that match the schema."


def camel_to_upper_snake(value: str) -> str:
    """Convert CamelCase names to ALL_CAPS relation labels."""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).upper()


def _relation_endpoints(cls: type[KGRelation]) -> dict[str, str]:
    endpoints: dict[str, str] = {}
    for field_name, field_info in cls.model_fields.items():
        target = _unwrap_model(field_info.annotation)
        if target is None:
            continue
        if issubclass(target, KGEntity):
            endpoints[field_name] = target.__name__
    return endpoints


def _unwrap_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin in (Optional, Union, UnionType, list, tuple):
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            unwrapped = _unwrap_model(arg)
            if unwrapped is not None:
                return unwrapped
    return None
