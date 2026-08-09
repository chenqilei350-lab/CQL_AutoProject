"""Annotation-only adapter for Ego4D-EASG style graph annotations.

The adapter deliberately reads only graph/text metadata.  It does not download,
open, or decode any Ego4D video file.  Its output mirrors the existing project
shape: raw text, unified text, and a schema-compatible gold extraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.datasets.benchmark import BenchmarkScene, InputCondition
from backend.preprocessing.unified_text import GroundedEntry, UnifiedTextRecord, build_industrial_unified_text
from backend.schemas.egocentric_video import (
    Action,
    ActionObservedInScene,
    ActionOrder,
    ActsOnObject,
    EgocentricVideoExtraction,
    Provenance,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


DEFAULT_EASG_ANNOTATION_ROOT = Path("raw/easg_annotations")
DEFAULT_EASG_STANDARD_OUTPUT = Path("kg_ready_data/easg_standard_inputs.jsonl")

TOOL_OBJECT_HINTS = {
    "knife",
    "spoon",
    "fork",
    "scissors",
    "screwdriver",
    "wrench",
    "drill",
    "hammer",
    "brush",
    "cloth",
    "pliers",
    "saw",
    "pan",
    "pot",
    "bottle",
    "cup",
}
EASG_OBJECT_RELATIONS = {
    "dobj",
    "with",
    "in",
    "from",
    "on",
    "into",
    "to",
    "towards",
    "through",
    "under",
    "inside",
    "around",
    "against",
    "onto",
    "between",
}
UNSUPPORTED_RELATIONS = {"unsupported", "unknown", "near", "left_of", "right_of"}


class EASGSkippedRelation(BaseModel):
    """An EASG relation that was not mapped to the project schema."""

    source: str | None = None
    relation: str
    target: str | None = None
    reason: str


class EASGStandardScene(BaseModel):
    """A KG-ready EASG annotation record without video payloads."""

    scene_id: str
    video_id: str
    description: str
    source_path: str
    raw_text: str
    unified_record: UnifiedTextRecord
    gold_extraction: EgocentricVideoExtraction
    skipped_relations: list[EASGSkippedRelation] = Field(default_factory=list)
    annotation_kind: str = "annotation-derived text"

    def input_text(self, condition: InputCondition) -> str:
        """Return raw or unified text for downstream extraction."""

        if condition == "raw":
            return self.raw_text
        if condition == "unified":
            return self.unified_record.to_prompt_text()
        raise ValueError(f"不支持的输入条件: {condition!r}")

    def to_benchmark_scene(self) -> BenchmarkScene:
        """Convert to the existing benchmark scene container."""

        return BenchmarkScene(
            scene_id=self.scene_id,
            description=self.description,
            raw_text=self.raw_text,
            unified_record=self.unified_record,
            gold_extraction=self.gold_extraction,
        )


class EASGStandardDataset(BaseModel):
    """Collection of annotation-only EASG standard scenes."""

    name: str = "easg_annotation_only_inputs"
    scenes: list[EASGStandardScene] = Field(default_factory=list)

    def get_scene(self, scene_id: str) -> EASGStandardScene:
        """Return a scene by ID."""

        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"EASG 标准输入中不存在场景: {scene_id!r}")

    def inputs_for(self, condition: InputCondition) -> list[tuple[str, str]]:
        """Return `(scene_id, input_text)` pairs for one input condition."""

        return [(scene.scene_id, scene.input_text(condition)) for scene in self.scenes]

    def to_jsonl(self) -> str:
        """Serialize the dataset as JSON Lines."""

        return "\n".join(
            json.dumps(scene.model_dump(mode="json"), ensure_ascii=False)
            for scene in self.scenes
        )

    def save_jsonl(self, output_path: str | Path = DEFAULT_EASG_STANDARD_OUTPUT) -> Path:
        """Write the dataset to a JSONL file."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_jsonl()
        path.write_text(f"{text}\n" if text else "", encoding="utf-8")
        return path

    @classmethod
    def from_jsonl(
        cls,
        input_path: str | Path,
        name: str = "easg_annotation_only_inputs",
    ) -> "EASGStandardDataset":
        """Load standard EASG scenes exported by `save_jsonl`."""

        path = Path(input_path)
        scenes = [
            EASGStandardScene.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(name=name, scenes=scenes)

    def to_benchmark_scenes(self) -> list[BenchmarkScene]:
        """Return existing benchmark-scene objects for experiment runners."""

        return [scene.to_benchmark_scene() for scene in self.scenes]


def load_easg_annotation_dataset(
    source_dir: str | Path = DEFAULT_EASG_ANNOTATION_ROOT,
    *,
    limit: int | None = None,
) -> EASGStandardDataset:
    """Load EASG annotation files from a local annotation-only folder."""

    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"EASG annotation folder not found: {root}")

    scenes: list[EASGStandardScene] = []
    for path in _annotation_json_paths(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in _iter_annotation_records(payload):
            scenes.append(_record_to_scene(record, path, root))
            if limit is not None and len(scenes) >= limit:
                return EASGStandardDataset(scenes=scenes)
    return EASGStandardDataset(scenes=scenes)


def _annotation_json_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    master_files = sorted(root.glob("*master*.json"))
    if master_files:
        return master_files
    return sorted(
        path
        for path in root.rglob("*.json")
        if "-sequence-" not in path.name and path.name != ".DS_Store"
    )


def _iter_annotation_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("annotations", "clips", "samples", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if _looks_like_easg_master(payload):
        return _flatten_easg_master(payload)
    return [payload]


def _record_to_scene(record: dict[str, Any], path: Path, root: Path) -> EASGStandardScene:
    video_id = str(_first(record, "video_id", "clip_uid", "video_uid", default=path.stem))
    action_id = str(_first(record, "action_id", "annotation_id", "id", "graph_uid", default="a1"))
    scene_id = _safe_id(f"easg_{video_id}_{action_id}")
    action_text = _action_text(record)
    object_roles = _object_roles(record)
    object_names = _dedupe_preserve_order(item["name"] for item in object_roles)
    raw_text = _raw_text(record, action_text, object_names)
    timestamp = _timestamp(record)
    source_path = str(path.relative_to(root))
    provenance = _provenance(record, source_path, video_id)

    action = Action(
        name=action_text,
        evidence_text=action_text,
        source_text=action_text,
        sequence_index=_int_or_none(_first(record, "sequence_index", "order", "action_index")),
        provenance=provenance,
    )
    scene = Scene(
        name=scene_id,
        video_id=video_id,
        segment_id=action_id,
        timestamp_start_seconds=_float_or_none(_first(record, "start_seconds", "start_time", "start")),
        timestamp_end_seconds=_float_or_none(_first(record, "end_seconds", "end_time", "end")),
        source_text=raw_text,
    )
    objects = [
        SceneObject(
            name=name,
            object_type="tool" if _is_tool_name(name) else "object",
            evidence_text=name,
            source_text=name,
            provenance=provenance,
        )
        for name in object_names
    ]
    tools = [
        Tool(
            name=obj.name,
            tool_type="tool",
            evidence_text=obj.evidence_text,
            source_text=obj.source_text,
        )
        for obj in objects
        if obj.object_type == "tool"
    ]

    acts_on_object = [
        ActsOnObject(
            action=Action(name=action.name, evidence_text=action.evidence_text),
            object=SceneObject(name=item["name"], evidence_text=item["name"]),
            role=_schema_object_role(item["role"]),
            evidence_text=f"{action.name} {item['role']} {item['name']}",
            provenance=provenance,
        )
        for item in object_roles
    ]
    uses_tool = [
        UsesTool(
            action=Action(name=action.name, evidence_text=action.evidence_text),
            tool=Tool(name=tool.name, evidence_text=tool.evidence_text),
            evidence_text=f"{action.name} {tool.name}",
            provenance=provenance,
        )
        for tool in tools
    ]
    action_order = _action_order(record, action, provenance)
    skipped_relations = _skipped_relations(record)

    gold = EgocentricVideoExtraction(
        video_id=video_id,
        scenes=[scene],
        actions=[action],
        tools=tools,
        objects=objects,
        uses_tool=uses_tool,
        acts_on_object=acts_on_object,
        action_order=action_order,
        observed_in_scene=[
            ActionObservedInScene(
                action=Action(name=action.name, evidence_text=action.evidence_text),
                scene=Scene(name=scene.name, video_id=video_id, segment_id=action_id),
                evidence_text=action.name,
                provenance=provenance,
            )
        ],
        source_text=raw_text,
    )
    unified = build_industrial_unified_text(
        raw_text=raw_text,
        scene_id=scene_id,
        segment_id=action_id,
        timestamp=timestamp,
        scene="EASG annotation-derived action scene graph",
        action_sequence=[GroundedEntry(text=action_text, evidence=action_text)],
        tools_objects=[
            GroundedEntry(text=name, evidence=name)
            for name in object_names
            if _contains_evidence(raw_text, name)
        ],
        uncertainty=[
            "This input is generated from EASG graph annotations/metadata only; no video frames are used.",
            *[
                f"Skipped unsupported relation {item.relation}: {item.reason}"
                for item in skipped_relations
            ],
        ],
    )
    return EASGStandardScene(
        scene_id=scene_id,
        video_id=video_id,
        description=f"EASG annotation-only scene for {video_id}/{action_id}",
        source_path=source_path,
        raw_text=raw_text,
        unified_record=unified,
        gold_extraction=gold,
        skipped_relations=skipped_relations,
    )


def _action_text(record: dict[str, Any]) -> str:
    narration = _first(record, "narration", "narration_text", "text", "description")
    if isinstance(narration, str) and narration.strip():
        return _clean_text(narration)
    action = _first(record, "action", "verb", "verb_label", "action_label", default="unknown action")
    if isinstance(action, dict):
        action = _first(action, "label", "name", "verb", default="unknown action")
    if action == "unknown action":
        triplet_action = _triplet_action(record)
        if triplet_action:
            action = triplet_action
    if action == "unknown action":
        graph_action = _graph_node_label(record, {"verb", "action"})
        if graph_action:
            action = graph_action
    return _clean_text(str(action))


def _object_names(record: dict[str, Any]) -> list[str]:
    return _dedupe_preserve_order(item["name"] for item in _object_roles(record))


def _object_roles(record: dict[str, Any]) -> list[dict[str, str]]:
    roles: list[dict[str, str]] = [
        {"name": name, "role": "target"}
        for name in _object_names_without_triplets(record)
    ]
    for _, relation, target in _triplets(record):
        relation_key = _relation_key(relation)
        if relation_key == "verb" or relation_key not in EASG_OBJECT_RELATIONS:
            continue
        if not target or _relation_key(target) == "cw":
            continue
        roles.append({"name": _clean_text(target), "role": relation_key})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in roles:
        key = (item["name"].casefold(), item["role"].casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _object_names_without_triplets(record: dict[str, Any]) -> list[str]:
    raw_objects = _first(record, "objects", "object_nodes", "nouns", "interacted_objects", default=[])
    names: list[str] = []
    if isinstance(raw_objects, list):
        for item in raw_objects:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                value = _first(item, "name", "label", "noun", "object", "category")
                if value:
                    names.append(str(value))
    elif isinstance(raw_objects, dict):
        for value in raw_objects.values():
            if isinstance(value, str):
                names.append(value)
            elif isinstance(value, dict):
                name = _first(value, "name", "label", "noun", "object", "category")
                if name:
                    names.append(str(name))
    noun = _first(record, "noun", "noun_label", "direct_object")
    if noun:
        names.append(str(noun))
    names.extend(_graph_object_labels(record))
    return _dedupe_preserve_order(_clean_text(name) for name in names if str(name).strip())


def _raw_text(record: dict[str, Any], action_text: str, object_names: list[str]) -> str:
    text = _first(record, "text", "narration", "narration_text", "description")
    if isinstance(text, str) and text.strip():
        return "Annotation-derived text: " + _clean_text(text)
    object_phrase = ", ".join(object_names) if object_names else "no explicit object"
    return f"Annotation-derived text: action '{action_text}' involves {object_phrase}."


def _action_order(
    record: dict[str, Any],
    action: Action,
    provenance: Provenance,
) -> list[ActionOrder]:
    relations: list[ActionOrder] = []
    previous = _first(record, "previous_action", "before", "predecessor")
    following = _first(record, "next_action", "after", "successor")
    if previous:
        previous_text = _clean_text(str(previous))
        if previous_text.casefold() != action.name.casefold():
            relations.append(
                ActionOrder(
                    before=Action(name=previous_text, evidence_text=previous_text),
                    after=Action(name=action.name, evidence_text=action.evidence_text),
                    evidence_text=f"{previous_text} before {action.name}",
                    provenance=provenance,
                )
            )
    if following:
        following_text = _clean_text(str(following))
        if following_text.casefold() != action.name.casefold():
            relations.append(
                ActionOrder(
                    before=Action(name=action.name, evidence_text=action.evidence_text),
                    after=Action(name=following_text, evidence_text=following_text),
                    evidence_text=f"{action.name} before {following_text}",
                    provenance=provenance,
                )
            )
    return relations


def _skipped_relations(record: dict[str, Any]) -> list[EASGSkippedRelation]:
    skipped: list[EASGSkippedRelation] = []
    raw_relations = _first(record, "relations", "edges", "graph_edges", default=[])
    graph = record.get("graph")
    if isinstance(graph, dict) and not raw_relations:
        raw_relations = graph.get("edges") or graph.get("relations") or []
    if not isinstance(raw_relations, list):
        return skipped
    for relation in raw_relations:
        if not isinstance(relation, dict):
            continue
        label = str(_first(relation, "relation", "predicate", "label", "type", default="unknown"))
        if _relation_key(label) in UNSUPPORTED_RELATIONS:
            skipped.append(
                EASGSkippedRelation(
                    source=_string_or_none(_first(relation, "source", "subject", "from")),
                    relation=label,
                    target=_string_or_none(_first(relation, "target", "object", "to")),
                    reason="Relation is spatial/unsupported in current procedural KG schema.",
                )
            )
    for source, relation, target in _triplets(record):
        relation_key = _relation_key(relation)
        if relation_key in {"verb", *EASG_OBJECT_RELATIONS}:
            continue
        skipped.append(
            EASGSkippedRelation(
                source=source,
                relation=relation,
                target=target,
                reason="Triplet relation is unsupported in current procedural KG schema.",
            )
        )
    return skipped


def _provenance(record: dict[str, Any], source_path: str, video_id: str) -> Provenance:
    return Provenance(
        video_id=video_id,
        segment_id=_string_or_none(_first(record, "action_id", "annotation_id", "id", "graph_uid")),
        frame_start=_int_or_none(_first(record, "pre_frame", "pre", "frame_start")),
        frame_end=_int_or_none(_first(record, "post_frame", "post", "frame_end")),
        timestamp_start_seconds=_float_or_none(_first(record, "start_seconds", "start_time", "start")),
        timestamp_end_seconds=_float_or_none(_first(record, "end_seconds", "end_time", "end")),
        source_text=source_path,
        confidence=1.0,
    )


def _timestamp(record: dict[str, Any]) -> str | None:
    start = _float_or_none(_first(record, "start_seconds", "start_time", "start"))
    end = _float_or_none(_first(record, "end_seconds", "end_time", "end"))
    if start is None or end is None:
        return None
    return f"{start:.3f}-{end:.3f}s"


def _looks_like_easg_master(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    sample_values = list(payload.values())[:5]
    return all(isinstance(value, dict) and isinstance(value.get("graphs"), list) for value in sample_values)


def _flatten_easg_master(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for action_uid, item in payload.items():
        if not isinstance(item, dict):
            continue
        graphs = item.get("graphs")
        if not isinstance(graphs, list):
            continue
        graph_actions = [
            _action_from_graph(graph) if isinstance(graph, dict) else None
            for graph in graphs
        ]
        for index, graph in enumerate(graphs):
            if not isinstance(graph, dict):
                continue
            record = {
                "action_uid": action_uid,
                "action_id": graph.get("graph_uid") or f"{action_uid}_{index}",
                "graph_uid": graph.get("graph_uid"),
                "video_uid": item.get("video_uid") or action_uid,
                "split": item.get("split"),
                "width": item.get("W"),
                "height": item.get("H"),
                "pre": graph.get("pre"),
                "pnr": graph.get("pnr"),
                "post": graph.get("post"),
                "triplets": graph.get("triplets") or [],
                "groundings": graph.get("groundings") or {},
                "sequence_index": index,
            }
            if index > 0 and graph_actions[index - 1]:
                record["previous_action"] = graph_actions[index - 1]
            if index + 1 < len(graph_actions) and graph_actions[index + 1]:
                record["next_action"] = graph_actions[index + 1]
            records.append(record)
    return records


def _action_from_graph(graph: dict[str, Any]) -> str | None:
    return _triplet_action(graph)


def _triplet_action(record: dict[str, Any]) -> str | None:
    for _, relation, target in _triplets(record):
        if _relation_key(relation) == "verb" and target:
            return _clean_text(target)
    return None


def _triplets(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    raw_triplets = record.get("triplets") or []
    result: list[tuple[str, str, str]] = []
    if not isinstance(raw_triplets, list):
        return result
    for item in raw_triplets:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            result.append((_clean_text(str(item[0])), _clean_text(str(item[1])), _clean_text(str(item[2]))))
        elif isinstance(item, dict):
            source = _first(item, "source", "subject", "from", "head")
            relation = _first(item, "relation", "predicate", "label", "type")
            target = _first(item, "target", "object", "to", "tail")
            if source and relation and target:
                result.append((_clean_text(str(source)), _clean_text(str(relation)), _clean_text(str(target))))
    return result


def _first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return default


def _graph_node_label(record: dict[str, Any], node_types: set[str]) -> str | None:
    for node in _graph_nodes(record):
        kind = str(_first(node, "type", "node_type", "kind", default="")).casefold()
        if kind in node_types:
            label = _first(node, "label", "name", "value")
            if label:
                return str(label)
    return None


def _graph_object_labels(record: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for node in _graph_nodes(record):
        kind = str(_first(node, "type", "node_type", "kind", default="")).casefold()
        if kind in {"object", "active_object", "direct_object", "other_object", "hand"}:
            label = _first(node, "label", "name", "value")
            if label:
                labels.append(str(label))
    return labels


def _graph_nodes(record: dict[str, Any]) -> list[dict[str, Any]]:
    graph = record.get("graph")
    if isinstance(graph, dict):
        nodes = graph.get("nodes") or graph.get("vertices")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, dict)]
        if isinstance(nodes, dict):
            return [
                {"id": key, **value} if isinstance(value, dict) else {"id": key, "label": value}
                for key, value in nodes.items()
            ]
    nodes = record.get("nodes")
    if isinstance(nodes, list):
        return [node for node in nodes if isinstance(node, dict)]
    return []


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "easg_scene"


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split())


def _contains_evidence(source: str, evidence: str) -> bool:
    return " ".join(evidence.casefold().split()) in " ".join(source.casefold().split())


def _dedupe_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _is_tool_name(value: str) -> bool:
    normalized = value.casefold()
    return any(hint in normalized for hint in TOOL_OBJECT_HINTS)


def _schema_object_role(value: str) -> str:
    relation = _relation_key(value)
    if relation == "dobj":
        return "target"
    if relation in {"from"}:
        return "input"
    if relation in {"into", "to", "onto"}:
        return "output"
    if relation in {"with", "in", "on", "towards", "through", "under", "inside", "around", "against", "between"}:
        return "support"
    return "unknown"


def _relation_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _string_or_none(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
