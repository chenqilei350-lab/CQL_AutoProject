"""Load human-reviewed industrial Gold datasets.

The source JSONL is the result of manual review.  This adapter converts its
flat node/edge representation into the project's existing benchmark and
Pydantic extraction contracts without changing the reviewed facts.

The unified condition is annotation-derived and therefore represents a
controlled, human-reviewed preprocessing condition.  It must not be described
as an automatic data-cleaning result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.datasets.benchmark import BenchmarkDataset, BenchmarkScene
from backend.preprocessing.unified_text import GroundedEntry, UnifiedTextRecord
from backend.schemas.egocentric_video import (
    Action,
    ActionOrder,
    ActsOnObject,
    EgocentricVideoExtraction,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


V2_REVIEWED_GOLD_PATH = Path(
    "data/industrial_gold_candidates/industrial_gold_v2_reviewed.jsonl"
)
V3_REVIEWED_GOLD_PATH = Path(
    "data/industrial_gold_reviewed/v3/industrial_gold_v3_reviewed.jsonl"
)
DEFAULT_REVIEWED_GOLD_PATH = Path(
    "data/industrial_gold_reviewed/industrial_gold_combined_v2_v3_reviewed.jsonl"
)
SCORING_NODE_LABELS = frozenset({"Action", "Object", "Tool"})
SCORING_RELATION_TYPES = frozenset({"BEFORE", "ACTS_ON", "USES_TOOL"})

_ELLIPSIS_RE = re.compile(r"\s*(?:\.{3,}|…)\s*")
_SENTENCE_RE = re.compile(r"[^.!?;]+(?:[.!?;]+|$)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_industrial_reviewed_gold_dataset(
    path: str | Path = DEFAULT_REVIEWED_GOLD_PATH,
    *,
    limit: int | None = None,
) -> BenchmarkDataset:
    """Load accepted human-reviewed records as a benchmark dataset."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Reviewed industrial Gold file not found: {input_path}")

    scenes: list[BenchmarkScene] = []
    seen_scene_ids: set[str] = set()
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid reviewed Gold JSONL at {input_path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"Reviewed Gold line must be an object: {input_path}:{line_number}"
            )
        if record.get("review_status") != "accepted":
            continue

        scene_id = str(record.get("scene_id") or "").strip()
        if not scene_id:
            raise ValueError(f"Missing scene_id at {input_path}:{line_number}")
        if scene_id in seen_scene_ids:
            raise ValueError(f"Duplicate reviewed scene_id: {scene_id}")
        seen_scene_ids.add(scene_id)

        scenes.append(_record_to_benchmark_scene(record, input_path, line_number))
        if limit is not None and len(scenes) >= limit:
            break

    if not scenes:
        raise ValueError(f"No accepted reviewed Gold scenes found in {input_path}")
    return BenchmarkDataset(name=_dataset_name(input_path), scenes=scenes)


def _dataset_name(input_path: Path) -> str:
    name = input_path.stem.lower()
    if "combined" in name:
        return "industrial_reviewed_gold_combined_v2_v3"
    if "v3" in name:
        return "industrial_reviewed_gold_v3"
    if "v2" in name:
        return "industrial_reviewed_gold_v2"
    return f"industrial_reviewed_gold_{input_path.stem}"


def _record_to_benchmark_scene(
    record: dict[str, Any], source_path: Path, line_number: int
) -> BenchmarkScene:
    scene_id = str(record["scene_id"])
    source_text = str(record.get("source_text") or "").strip()
    if not source_text:
        raise ValueError(f"Missing source_text at {source_path}:{line_number}")

    entities = record.get("gold_entities") or []
    relations = record.get("gold_relations") or []
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError(
            f"gold_entities and gold_relations must be lists at {source_path}:{line_number}"
        )

    actions: dict[str, Action] = {}
    objects: dict[str, SceneObject] = {}
    tools: dict[str, Tool] = {}
    for item in entities:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid entity in scene {scene_id}")
        entity_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or "").strip()
        name = str(item.get("name") or "").strip()
        evidence = str(item.get("evidence_text") or "").strip()
        if not entity_id or not name or not evidence:
            raise ValueError(
                f"Reviewed entity requires id, name, and evidence in scene {scene_id}"
            )
        common = {
            "name": name,
            "entity_id": entity_id,
            "source_text": source_text,
            "evidence_text": evidence,
        }
        if label == "Action":
            actions[entity_id] = Action(
                **common,
                sequence_index=item.get("sequence_index"),
            )
        elif label == "Object":
            objects[entity_id] = SceneObject(**common, object_type="object")
        elif label == "Tool":
            tools[entity_id] = Tool(**common, tool_type="tool")
        else:
            raise ValueError(
                f"Unsupported reviewed entity label {label!r} in scene {scene_id}"
            )

    all_ids = set(actions) | set(objects) | set(tools)
    if len(all_ids) != len(actions) + len(objects) + len(tools):
        raise ValueError(f"Duplicate reviewed entity IDs in scene {scene_id}")

    action_order: list[ActionOrder] = []
    acts_on_object: list[ActsOnObject] = []
    uses_tool: list[UsesTool] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for item in relations:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid relation in scene {scene_id}")
        source_id = str(item.get("source") or "").strip()
        relation = str(item.get("relation") or "").strip().upper()
        target_id = str(item.get("target") or "").strip()
        key = (source_id, relation, target_id)
        if source_id not in all_ids or target_id not in all_ids:
            raise ValueError(f"Invalid reviewed relation endpoint {key} in scene {scene_id}")
        if key in seen_relations:
            raise ValueError(f"Duplicate reviewed relation {key} in scene {scene_id}")
        seen_relations.add(key)
        relation_evidence = _relation_evidence(
            source_text, source_id, target_id, actions, objects, tools
        )
        if relation == "BEFORE" and source_id in actions and target_id in actions:
            action_order.append(
                ActionOrder(
                    before=actions[source_id],
                    after=actions[target_id],
                    source_text=source_text,
                    evidence_text=relation_evidence,
                )
            )
        elif relation == "ACTS_ON" and source_id in actions and target_id in objects:
            acts_on_object.append(
                ActsOnObject(
                    action=actions[source_id],
                    object=objects[target_id],
                    role="target",
                    source_text=source_text,
                    evidence_text=relation_evidence,
                )
            )
        elif relation == "USES_TOOL" and source_id in actions and target_id in tools:
            uses_tool.append(
                UsesTool(
                    action=actions[source_id],
                    tool=tools[target_id],
                    source_text=source_text,
                    evidence_text=relation_evidence,
                )
            )
        else:
            raise ValueError(
                f"Invalid reviewed relation signature {key} in scene {scene_id}"
            )

    gold = EgocentricVideoExtraction(
        video_id=str(record.get("video_id") or scene_id),
        actions=list(actions.values()),
        objects=list(objects.values()),
        tools=list(tools.values()),
        action_order=action_order,
        acts_on_object=acts_on_object,
        uses_tool=uses_tool,
        source_text=source_text,
    )
    unified = _build_annotation_derived_unified(record, gold)
    return BenchmarkScene(
        scene_id=scene_id,
        description=str(record.get("task") or record.get("category") or scene_id),
        raw_text=source_text,
        unified_record=unified,
        gold_extraction=gold,
    )


def _build_annotation_derived_unified(
    record: dict[str, Any], gold: EgocentricVideoExtraction
) -> UnifiedTextRecord:
    source_text = gold.source_text or ""

    def entry(name: str, evidence: str | None) -> GroundedEntry:
        return GroundedEntry(
            text=name,
            evidence=_locate_evidence(source_text, evidence or name, name),
        )

    quality_results = [
        entry(str(item.get("name") or "quality result"), item.get("evidence_text"))
        for item in record.get("quality_results", [])
        if isinstance(item, dict) and item.get("name")
    ]
    outcomes = [
        entry(str(item.get("name") or "outcome"), item.get("evidence_text"))
        for item in record.get("outcomes", [])
        if isinstance(item, dict) and item.get("name")
    ]
    return UnifiedTextRecord(
        scene_id=str(record["scene_id"]),
        segment_id=str(record["scene_id"]),
        timestamp=record.get("timestamp"),
        scene_segment=str(record.get("task") or record.get("category") or "industrial procedure"),
        action_sequence=[entry(action.name, action.evidence_text) for action in gold.actions],
        tools_objects=[
            *(entry(tool.name, tool.evidence_text) for tool in gold.tools),
            *(entry(obj.name, obj.evidence_text) for obj in gold.objects),
        ],
        quality_results=quality_results,
        outcomes_parameters=outcomes,
        evidence_uncertainty=[
            "This unified condition is derived from human-reviewed annotations; "
            "it is a controlled preprocessing condition, not automatic cleaning."
        ],
        source_text=source_text,
    )


def _relation_evidence(
    source_text: str,
    source_id: str,
    target_id: str,
    actions: dict[str, Action],
    objects: dict[str, SceneObject],
    tools: dict[str, Tool],
) -> str:
    entities: dict[str, Action | SceneObject | Tool] = {**actions, **objects, **tools}
    source = entities[source_id]
    target = entities[target_id]
    return " | ".join(
        [
            _locate_evidence(source_text, source.evidence_text or source.name, source.name),
            _locate_evidence(source_text, target.evidence_text or target.name, target.name),
        ]
    )


def _locate_evidence(source_text: str, evidence: str, name: str) -> str:
    """Return an exact source span supporting a reviewed item."""

    for candidate in [evidence, *_ELLIPSIS_RE.split(evidence), name]:
        candidate = candidate.strip()
        if not candidate:
            continue
        index = source_text.lower().find(candidate.lower())
        if index >= 0:
            return source_text[index : index + len(candidate)]

    query_tokens = _tokens(f"{name} {evidence}")
    sentences = [item.strip() for item in _SENTENCE_RE.findall(source_text) if item.strip()]
    if query_tokens and sentences:
        best = max(
            sentences,
            key=lambda sentence: len(query_tokens & _tokens(sentence))
            / max(len(query_tokens | _tokens(sentence)), 1),
        )
        if query_tokens & _tokens(best):
            return best

    # The whole reviewed excerpt is still a valid source span.  This fallback
    # keeps the benchmark loadable while making weak localization visible.
    return source_text


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))
