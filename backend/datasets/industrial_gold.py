"""Load human-reviewed industrial gold records as the project benchmark contract.

The review JSONL intentionally keeps richer audit metadata than
``EgocentricVideoExtraction`` (quality results, exclusions, conditional rules,
and reviewer decisions).  This module validates that audit format and exposes a
lossless review dataset plus an evaluation-compatible ``BenchmarkDataset``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


DEFAULT_REVIEWED_GOLD_PATH = Path(
    "data/industrial_gold_candidates/industrial_gold_v2_reviewed.jsonl"
)
DEFAULT_EXCLUSIONS_PATH = Path(
    "data/industrial_gold_candidates/industrial_gold_v2_exclusions.jsonl"
)
DEFAULT_CANDIDATES_PATH = Path(
    "data/industrial_gold_candidates/industrial_gold_candidate_templates.jsonl"
)

EntityLabel = Literal["Action", "Object", "Tool"]
RelationType = Literal["BEFORE", "ACTS_ON", "USES_TOOL"]


class IndustrialGoldEntity(BaseModel):
    """One reviewed graph node, identified by a stable scene-local ID."""

    id: str
    label: EntityLabel
    name: str
    evidence_text: str
    sequence_index: int | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class IndustrialGoldRelation(BaseModel):
    """One reviewed graph edge between scene-local entity IDs."""

    source: str
    relation: RelationType
    target: str
    evidence_text: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class IndustrialGoldRecord(BaseModel):
    """A complete human-reviewed scene with audit metadata."""

    model_config = ConfigDict(extra="allow")

    scene_id: str
    video_id: str
    category: str
    task: str
    timestamp: str | None = None
    source_text: str
    gold_entities: list[IndustrialGoldEntity]
    gold_relations: list[IndustrialGoldRelation]
    quality_results: list[dict[str, Any]] = Field(default_factory=list)
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    review_status: Literal["accepted"]
    reviewer_note: str

    @model_validator(mode="after")
    def validate_graph_contract(self) -> "IndustrialGoldRecord":
        entity_by_id = {entity.id: entity for entity in self.gold_entities}
        if len(entity_by_id) != len(self.gold_entities):
            raise ValueError(f"{self.scene_id}: entity IDs must be unique")

        actions = [entity for entity in self.gold_entities if entity.label == "Action"]
        sequence = [entity.sequence_index for entity in actions]
        if sequence != list(range(1, len(actions) + 1)):
            raise ValueError(
                f"{self.scene_id}: action sequence_index values must be contiguous"
            )

        relation_keys: set[tuple[str, str, str]] = set()
        for relation in self.gold_relations:
            if relation.source not in entity_by_id or relation.target not in entity_by_id:
                raise ValueError(
                    f"{self.scene_id}: relation endpoint does not exist: {relation}"
                )
            key = (relation.source, relation.relation, relation.target)
            if key in relation_keys:
                raise ValueError(f"{self.scene_id}: duplicate relation: {key}")
            relation_keys.add(key)

            source_label = entity_by_id[relation.source].label
            target_label = entity_by_id[relation.target].label
            expected = {
                "BEFORE": ("Action", "Action"),
                "ACTS_ON": ("Action", "Object"),
                "USES_TOOL": ("Action", "Tool"),
            }[relation.relation]
            if (source_label, target_label) != expected:
                raise ValueError(
                    f"{self.scene_id}: {relation.relation} requires {expected}, "
                    f"got {(source_label, target_label)}"
                )

        before_edges = {
            (relation.source, relation.target)
            for relation in self.gold_relations
            if relation.relation == "BEFORE"
        }
        expected_before = {
            (actions[index].id, actions[index + 1].id)
            for index in range(len(actions) - 1)
        }
        if before_edges != expected_before:
            raise ValueError(
                f"{self.scene_id}: BEFORE edges must exactly follow action order"
            )
        return self

    def to_extraction(self, raw_text: str | None = None) -> EgocentricVideoExtraction:
        """Convert the reviewed graph subset to the extraction evaluation schema."""

        source_text = raw_text or self.source_text
        action_by_id = {
            entity.id: Action(
                entity_id=entity.id,
                name=entity.name,
                evidence_text=entity.evidence_text,
                source_text=source_text,
                sequence_index=entity.sequence_index,
            )
            for entity in self.gold_entities
            if entity.label == "Action"
        }
        object_by_id = {
            entity.id: SceneObject(
                entity_id=entity.id,
                name=entity.name,
                evidence_text=entity.evidence_text,
                source_text=source_text,
                object_type="object",
            )
            for entity in self.gold_entities
            if entity.label == "Object"
        }
        tool_by_id = {
            entity.id: Tool(
                entity_id=entity.id,
                name=entity.name,
                evidence_text=entity.evidence_text,
                source_text=source_text,
                tool_type="tool",
            )
            for entity in self.gold_entities
            if entity.label == "Tool"
        }

        action_order: list[ActionOrder] = []
        acts_on_object: list[ActsOnObject] = []
        uses_tool: list[UsesTool] = []
        for relation in self.gold_relations:
            evidence = relation.evidence_text or action_by_id[relation.source].evidence_text
            common = {
                "evidence_text": evidence,
                "relation_origin": "deterministic_candidate",
            }
            if relation.relation == "BEFORE":
                action_order.append(
                    ActionOrder(
                        before=action_by_id[relation.source],
                        after=action_by_id[relation.target],
                        **common,
                    )
                )
            elif relation.relation == "ACTS_ON":
                acts_on_object.append(
                    ActsOnObject(
                        action=action_by_id[relation.source],
                        object=object_by_id[relation.target],
                        role="target",
                        **common,
                    )
                )
            else:
                uses_tool.append(
                    UsesTool(
                        action=action_by_id[relation.source],
                        tool=tool_by_id[relation.target],
                        **common,
                    )
                )

        return EgocentricVideoExtraction(
            video_id=self.video_id,
            actions=list(action_by_id.values()),
            tools=list(tool_by_id.values()),
            objects=list(object_by_id.values()),
            uses_tool=uses_tool,
            acts_on_object=acts_on_object,
            action_order=action_order,
            source_text=source_text,
        )

    def to_unified_record(self, raw_text: str | None = None) -> UnifiedTextRecord:
        """Build the fixed-section input while retaining reviewed uncertainty."""

        source_text = raw_text or self.source_text
        actions = sorted(
            (entity for entity in self.gold_entities if entity.label == "Action"),
            key=lambda entity: entity.sequence_index or 0,
        )
        tools_objects = [
            entity for entity in self.gold_entities if entity.label in {"Tool", "Object"}
        ]
        return UnifiedTextRecord(
            scene_id=self.scene_id,
            segment_id=_segment_id(self.scene_id),
            timestamp=self.timestamp,
            scene_segment=self.task,
            action_sequence=[
                GroundedEntry(
                    text=entity.name,
                    evidence=entity.evidence_text,
                    uncertainty=_entity_uncertainty(entity),
                )
                for entity in actions
            ],
            tools_objects=[
                GroundedEntry(
                    text=entity.name,
                    evidence=entity.evidence_text,
                    uncertainty=_entity_uncertainty(entity),
                )
                for entity in tools_objects
            ],
            quality_results=[
                GroundedEntry(
                    text=str(item.get("name") or "quality result"),
                    evidence=str(item.get("evidence_text") or "human-reviewed result"),
                    uncertainty=(
                        "status: " + str(item["status"]) if item.get("status") else None
                    ),
                )
                for item in self.quality_results
            ],
            outcomes_parameters=[
                GroundedEntry(
                    text=str(item.get("name") or "outcome"),
                    evidence=str(item.get("evidence_text") or "human-reviewed outcome"),
                )
                for item in self.outcomes
            ],
            # Reviewer notes and split policy belong to dataset metadata.  They must
            # not leak into the text presented to an extraction model.
            evidence_uncertainty=[],
            source_text=source_text,
        )


class IndustrialGoldExclusion(BaseModel):
    """An explicit human decision to omit a duplicate candidate scene."""

    scene_id: str
    video_id: str
    category: str
    review_decision: Literal["excluded"]
    reason: str
    duplicate_of: str
    reviewer_note: str


class IndustrialGoldDataset(BaseModel):
    """Reviewed industrial gold plus exclusions and original candidate sources."""

    name: str = "industrial_gold_v2_reviewed"
    records: list[IndustrialGoldRecord]
    exclusions: list[IndustrialGoldExclusion] = Field(default_factory=list)
    raw_text_by_scene: dict[str, str] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> "IndustrialGoldDataset":
        reviewed_ids = [record.scene_id for record in self.records]
        if len(reviewed_ids) != len(set(reviewed_ids)):
            raise ValueError("reviewed scene IDs must be unique")
        excluded_ids = [record.scene_id for record in self.exclusions]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("excluded scene IDs must be unique")
        overlap = set(reviewed_ids) & set(excluded_ids)
        if overlap:
            raise ValueError(f"scenes cannot be both reviewed and excluded: {sorted(overlap)}")
        missing_sources = set(reviewed_ids) - set(self.raw_text_by_scene)
        if missing_sources:
            raise ValueError(f"missing original source text: {sorted(missing_sources)}")
        reviewed_by_id = {record.scene_id: record for record in self.records}
        for exclusion in self.exclusions:
            if exclusion.duplicate_of not in reviewed_by_id:
                raise ValueError(
                    f"excluded scene {exclusion.scene_id} references missing reviewed scene "
                    f"{exclusion.duplicate_of}"
                )
            if reviewed_by_id[exclusion.duplicate_of].video_id != exclusion.video_id:
                raise ValueError(
                    f"excluded scene {exclusion.scene_id} and duplicate_of must share video_id"
                )
        return self

    @classmethod
    def from_files(
        cls,
        reviewed_path: str | Path = DEFAULT_REVIEWED_GOLD_PATH,
        exclusions_path: str | Path = DEFAULT_EXCLUSIONS_PATH,
        candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    ) -> "IndustrialGoldDataset":
        """Load reviewed records, explicit exclusions, and original source text."""

        reviewed = [
            IndustrialGoldRecord.model_validate(row)
            for row in _read_jsonl(Path(reviewed_path))
        ]
        exclusions = [
            IndustrialGoldExclusion.model_validate(row)
            for row in _read_jsonl(Path(exclusions_path))
        ]
        candidate_rows = _read_jsonl(Path(candidates_path))
        raw_text_by_scene = {
            str(row["scene_id"]): str(row["raw_text"])
            for row in candidate_rows
            if row.get("scene_id") and row.get("raw_text")
        }
        return cls(
            records=reviewed,
            exclusions=exclusions,
            raw_text_by_scene=raw_text_by_scene,
        )

    def to_benchmark_dataset(self) -> BenchmarkDataset:
        """Return the current experiment runner's native benchmark container."""

        return BenchmarkDataset(
            name=self.name,
            scenes=[
                BenchmarkScene(
                    scene_id=record.scene_id,
                    description=record.task,
                    raw_text=self.raw_text_by_scene[record.scene_id],
                    unified_record=record.to_unified_record(
                        self.raw_text_by_scene[record.scene_id]
                    ),
                    gold_extraction=record.to_extraction(
                        self.raw_text_by_scene[record.scene_id]
                    ),
                )
                for record in self.records
            ],
        )

    def manifest(self) -> dict[str, Any]:
        """Return deterministic package statistics and leakage-safe grouping data."""

        entity_counts = Counter(
            entity.label for record in self.records for entity in record.gold_entities
        )
        relation_counts = Counter(
            relation.relation
            for record in self.records
            for relation in record.gold_relations
        )
        category_counts = Counter(record.category for record in self.records)
        scenes_by_video: dict[str, list[str]] = defaultdict(list)
        for record in self.records:
            scenes_by_video[record.video_id].append(record.scene_id)
        return {
            "dataset_name": self.name,
            "format_version": "2.0",
            "reviewed_scene_count": len(self.records),
            "independent_video_count": len(scenes_by_video),
            "excluded_scene_count": len(self.exclusions),
            "category_counts": dict(sorted(category_counts.items())),
            "entity_counts": dict(sorted(entity_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "quality_result_count": sum(
                len(record.quality_results) for record in self.records
            ),
            "outcome_count": sum(len(record.outcomes) for record in self.records),
            "video_groups": {
                video_id: sorted(scene_ids)
                for video_id, scene_ids in sorted(scenes_by_video.items())
            },
            "excluded_scenes": [
                exclusion.model_dump(mode="json") for exclusion in self.exclusions
            ],
            "evaluation_constraints": [
                "Split by video_id, never by scene_id.",
                "Excluded scenes must not enter training, prompting, or evaluation.",
                "Quality results and outcomes remain audit metadata and unified-input fields; "
                "the current property-graph scorer evaluates Action/Object/Tool nodes and "
                "BEFORE/ACTS_ON/USES_TOOL edges.",
                "This dataset is a development benchmark, not yet a statistically sufficient "
                "formal evaluation set.",
            ],
        }


def load_industrial_gold_dataset(
    reviewed_path: str | Path = DEFAULT_REVIEWED_GOLD_PATH,
    exclusions_path: str | Path = DEFAULT_EXCLUSIONS_PATH,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
) -> IndustrialGoldDataset:
    """Convenience loader used by scripts and tests."""

    return IndustrialGoldDataset.from_files(
        reviewed_path=reviewed_path,
        exclusions_path=exclusions_path,
        candidates_path=candidates_path,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _segment_id(scene_id: str) -> str:
    suffix = scene_id.rsplit("_", 1)[-1]
    return suffix if suffix.startswith("s") and suffix[1:].isdigit() else "s1"


def _entity_uncertainty(entity: IndustrialGoldEntity) -> str | None:
    uncertainty_keys = [
        key for key in entity.properties if "uncertainty" in key or "correction" in key
    ]
    if not uncertainty_keys:
        return None
    return "; ".join(f"{key}: {entity.properties[key]}" for key in uncertainty_keys)
