"""Adapter from downloaded IndEgo text layers to project standard inputs.

The adapter is intentionally deterministic: it reads local transcript,
temporal-segment annotation, keystep, and mistake/warning files and turns them
into the same raw/unified input shape used by the rest of the project. It does
not call an LLM and it does not require the raw video files.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from backend.cleaning.factory_data import (
    CleanFactoryRecord,
    FactoryCleaningReport,
    clean_indego_text_layers,
)
from backend.datasets.benchmark import InputCondition
from backend.preprocessing.unified_text import (
    GroundedEntry,
    UnifiedTextRecord,
    build_industrial_unified_text,
)


IndEgoSegmentLayer = Literal["action", "keystep"]

DEFAULT_INDEGO_TEXT_ROOT = Path("raw/indego_text_layers")
DEFAULT_INDEGO_STANDARD_OUTPUT = Path("kg_ready_data/indego_standard_inputs.jsonl")
DEFAULT_MAX_SEGMENTS_PER_SCENE = 8
DEFAULT_MAX_TRANSCRIPT_CHARS = 1600

_TEMPORAL_LAYER_BY_MANIFEST_LAYER = {
    "fine_grained_annotations": "action",
    "annotated_keysteps": "keystep",
}
_VIDEO_SUFFIX_RE = re.compile(r"(_480)?(\.mp4)?$", re.IGNORECASE)
_LEADING_STEP_RE = re.compile(r"^\s*\d+\s*[_:\-.]?\s*")
_USER_RE = re.compile(r"User[_\s-]*(\d+)", re.IGNORECASE)

_TOOL_OBJECT_TERMS = (
    "allen wrench",
    "torque wrench",
    "screwdriver",
    "drill machine",
    "drilling machine",
    "box cutter",
    "cutter knife",
    "spirit level",
    "multimeter",
    "protractor",
    "caliper",
    "hammer",
    "hole puncher",
    "sandpaper",
    "clamp",
    "cloth",
    "gloves",
    "safety shoes",
    "tripod",
    "camera",
    "battery",
    "sd card",
    "monitor",
    "power cable",
    "cable",
    "toolbox",
    "trolley",
    "tabletop",
    "table",
    "drawer",
    "shelf",
    "cardboard box",
    "box",
    "wooden block",
    "wood",
    "metal plate",
    "cover plate",
    "plate",
    "housing",
    "bolt",
    "screw",
    "leg",
    "feet",
    "rod",
    "connector",
)

# 中文：该集合仅用于 IndEgo Adapter 将已解析名词标成 Tool/Object；它不参与
# 原始文本清洗，也不是跨数据集工具本体。
# English: This set types already parsed IndEgo mentions. It is neither a source
# cleaner nor a cross-dataset tool ontology.
_INDEGO_TOOL_TERMS = {
    "allen wrench",
    "torque wrench",
    "screwdriver",
    "drill machine",
    "drilling machine",
    "box cutter",
    "cutter knife",
    "spirit level",
    "multimeter",
    "protractor",
    "caliper",
    "hammer",
    "hole puncher",
    "sandpaper",
    "clamp",
    "cloth",
    "gloves",
    "safety shoes",
}


class IndEgoTemporalSegment(BaseModel):
    """One temporal action or keystep annotation from a VIA JSON file."""

    label: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    video_name: str | None = None
    source_path: str
    layer: IndEgoSegmentLayer

    @property
    def timestamp(self) -> str | None:
        """Return a compact timestamp range for text rendering."""

        if self.start_seconds is None or self.end_seconds is None:
            return None
        return f"{_format_seconds(self.start_seconds)}-{_format_seconds(self.end_seconds)}"


class IndEgoWarning(BaseModel):
    """A mistake/warning item derived from IndEgo mistake annotations."""

    step: str
    description: str
    source_path: str
    run_id: str | None = None


class IndEgoStandardScene(BaseModel):
    """A KG-ready IndEgo input record with raw and unified text forms."""

    scene_id: str
    video_id: str
    category: str
    description: str
    source_paths: list[str] = Field(default_factory=list)
    raw_text: str
    unified_record: UnifiedTextRecord
    action_segments: list[IndEgoTemporalSegment] = Field(default_factory=list)
    keystep_segments: list[IndEgoTemporalSegment] = Field(default_factory=list)
    warnings: list[IndEgoWarning] = Field(default_factory=list)

    def input_text(self, condition: InputCondition) -> str:
        """Return raw or unified text for downstream KG extraction."""

        if condition == "raw":
            return self.raw_text
        if condition == "unified":
            return self.unified_record.to_extraction_text()
        raise ValueError(f"Unsupported input condition: {condition!r}")


class IndEgoStandardDataset(BaseModel):
    """Collection of standard IndEgo scene inputs."""

    name: str = "indego_standard_inputs"
    scenes: list[IndEgoStandardScene] = Field(default_factory=list)

    def get_scene(self, scene_id: str) -> IndEgoStandardScene:
        """Return a scene by ID."""

        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"Scene does not exist in the IndEgo standard input: {scene_id!r}")

    def inputs_for(self, condition: InputCondition) -> list[tuple[str, str]]:
        """Return `(scene_id, input_text)` pairs for one input condition."""

        return [(scene.scene_id, scene.input_text(condition)) for scene in self.scenes]

    def to_jsonl(self) -> str:
        """Serialize the dataset as JSON Lines."""

        return "\n".join(
            json.dumps(scene.model_dump(mode="json"), ensure_ascii=False)
            for scene in self.scenes
        )

    def save_jsonl(self, output_path: str | Path) -> Path:
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
        name: str = "indego_standard_inputs",
    ) -> "IndEgoStandardDataset":
        """Load standard scenes exported by `save_jsonl`."""

        path = Path(input_path)
        scenes = [
            IndEgoStandardScene.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(name=name, scenes=scenes)


def load_indego_standard_dataset(
    source_dir: str | Path = DEFAULT_INDEGO_TEXT_ROOT,
    *,
    max_segments_per_scene: int = DEFAULT_MAX_SEGMENTS_PER_SCENE,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
    include_transcript_only: bool = True,
    include_warning_scenes: bool = True,
    limit: int | None = None,
) -> IndEgoStandardDataset:
    """Build standard KG inputs from a downloaded IndEgo text-layer folder."""

    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"IndEgo text-layer folder not found: {root}")

    cleaning_report = clean_indego_text_layers(root)
    return build_indego_standard_dataset_from_cleaning_report(
        cleaning_report,
        max_segments_per_scene=max_segments_per_scene,
        max_transcript_chars=max_transcript_chars,
        include_transcript_only=include_transcript_only,
        include_warning_scenes=include_warning_scenes,
        limit=limit,
    )


def build_indego_standard_dataset_from_cleaning_report(
    cleaning_report: FactoryCleaningReport,
    *,
    max_segments_per_scene: int = DEFAULT_MAX_SEGMENTS_PER_SCENE,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
    include_transcript_only: bool = True,
    include_warning_scenes: bool = True,
    limit: int | None = None,
) -> IndEgoStandardDataset:
    """Build standard raw/unified KG inputs from already-cleaned IndEgo records."""

    if cleaning_report.source_name != "indego":
        raise ValueError(
            f"IndEgo adapter received unsupported cleaned source: {cleaning_report.source_name!r}"
        )

    transcript_by_video = _transcripts_from_clean_records(cleaning_report.records)
    temporal_groups = _temporal_groups_from_clean_records(cleaning_report.records)
    scenes: list[IndEgoStandardScene] = []

    for video_key in sorted(temporal_groups):
        group = temporal_groups[video_key]
        transcript = transcript_by_video.get(video_key)
        action_segments = sorted(group["action"], key=_segment_sort_key)
        keystep_segments = sorted(group["keystep"], key=_segment_sort_key)
        category = _first_non_empty(
            transcript.get("category") if transcript else None,
            group["category"],
            "indego",
        )
        scenes.extend(
            _build_chunked_scenes(
                video_key=video_key,
                category=category,
                transcript=transcript,
                action_segments=action_segments,
                keystep_segments=keystep_segments,
                source_paths=sorted(group["source_paths"]),
                max_segments_per_scene=max_segments_per_scene,
                max_transcript_chars=max_transcript_chars,
            )
        )

    if include_transcript_only:
        annotated_videos = set(temporal_groups)
        for video_key in sorted(set(transcript_by_video) - annotated_videos):
            transcript = transcript_by_video[video_key]
            scenes.append(
                _build_transcript_only_scene(
                    video_key=video_key,
                    transcript=transcript,
                    max_transcript_chars=max_transcript_chars,
                )
            )

    if include_warning_scenes:
        scenes.extend(_warning_scenes_from_clean_records(cleaning_report.records))

    if limit is not None:
        scenes = scenes[:limit]
    return IndEgoStandardDataset(scenes=scenes)


def parse_via_temporal_segments(
    path: str | Path,
    *,
    source_root: str | Path | None = None,
    layer: IndEgoSegmentLayer | None = None,
) -> list[IndEgoTemporalSegment]:
    """Parse temporal segments from an IndEgo VIA annotation JSON file."""

    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "metadata" not in data:
        return []

    inferred_layer = layer or _infer_temporal_layer(file_path)
    source_path = _relative_source_path(file_path, source_root)
    file_index = data.get("file") if isinstance(data.get("file"), dict) else {}
    attributes = data.get("attribute") if isinstance(data.get("attribute"), dict) else {}
    segments: list[IndEgoTemporalSegment] = []

    for metadata in data.get("metadata", {}).values():
        if not isinstance(metadata, dict):
            continue
        label = _label_from_metadata(metadata, attributes)
        if not label:
            continue
        z_value = metadata.get("z") if isinstance(metadata.get("z"), list) else []
        start = _safe_float(z_value[0]) if len(z_value) >= 1 else None
        end = _safe_float(z_value[1]) if len(z_value) >= 2 else None
        video_id = str(metadata.get("vid", ""))
        video_info = file_index.get(video_id, {}) if isinstance(file_index, dict) else {}
        video_name = video_info.get("fname") if isinstance(video_info, dict) else None
        segments.append(
            IndEgoTemporalSegment(
                label=_normalize_label(label),
                start_seconds=start,
                end_seconds=end,
                video_name=video_name,
                source_path=source_path,
                layer=inferred_layer,
            )
        )
    return sorted(segments, key=_segment_sort_key)


def export_indego_standard_inputs(
    source_dir: str | Path = DEFAULT_INDEGO_TEXT_ROOT,
    output_path: str | Path = DEFAULT_INDEGO_STANDARD_OUTPUT,
    *,
    max_segments_per_scene: int = DEFAULT_MAX_SEGMENTS_PER_SCENE,
    max_transcript_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
    include_transcript_only: bool = True,
    include_warning_scenes: bool = True,
    limit: int | None = None,
) -> IndEgoStandardDataset:
    """Build and write standard IndEgo JSONL inputs."""

    dataset = load_indego_standard_dataset(
        source_dir,
        max_segments_per_scene=max_segments_per_scene,
        max_transcript_chars=max_transcript_chars,
        include_transcript_only=include_transcript_only,
        include_warning_scenes=include_warning_scenes,
        limit=limit,
    )
    dataset.save_jsonl(output_path)
    return dataset


def _transcripts_from_clean_records(
    records: list[CleanFactoryRecord],
) -> dict[str, dict[str, str]]:
    transcript_by_video: dict[str, dict[str, str]] = {}
    for record in records:
        if record.record_type != "transcript" or not record.video_id:
            continue
        transcript_by_video[record.video_id] = {
            "category": record.category or "indego",
            "name": str(record.metadata.get("raw_name") or record.video_id),
            "transcript": record.text,
            "source_path": record.source_path,
        }
    return transcript_by_video


def _temporal_groups_from_clean_records(
    records: list[CleanFactoryRecord],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "action": [],
            "keystep": [],
            "source_paths": set(),
            "category": None,
        }
    )
    for record in records:
        layer = _segment_layer_from_record_type(record.record_type)
        if layer is None:
            continue
        video_key = record.video_id or _video_key(str(record.metadata.get("raw_video_name") or record.source_path))
        segment = IndEgoTemporalSegment(
            label=record.text,
            start_seconds=record.start_seconds,
            end_seconds=record.end_seconds,
            video_name=str(record.metadata.get("raw_video_name") or record.video_id or ""),
            source_path=record.source_path,
            layer=layer,
        )
        groups[video_key][layer].append(segment)
        groups[video_key]["source_paths"].add(record.source_path)
        if not groups[video_key]["category"]:
            groups[video_key]["category"] = record.category or "indego"
    return groups


def _segment_layer_from_record_type(record_type: str) -> IndEgoSegmentLayer | None:
    if record_type == "action_segment":
        return "action"
    if record_type == "keystep_segment":
        return "keystep"
    return None


def _warning_scenes_from_clean_records(
    records: list[CleanFactoryRecord],
) -> list[IndEgoStandardScene]:
    grouped: dict[str, dict[str, list[CleanFactoryRecord]]] = defaultdict(
        lambda: {"steps": [], "warnings": []}
    )
    for record in records:
        if record.record_type == "mistake_step":
            grouped[record.source_path]["steps"].append(record)
        elif record.record_type == "mistake_warning":
            grouped[record.source_path]["warnings"].append(record)

    scenes: list[IndEgoStandardScene] = []
    for source_path in sorted(grouped):
        group = grouped[source_path]
        steps = sorted(
            group["steps"],
            key=lambda record: int(record.metadata.get("step_index") or 0),
        )
        warnings = sorted(
            group["warnings"],
            key=lambda record: (
                str(record.metadata.get("run_id") or ""),
                int(record.metadata.get("step_index") or 0),
                record.text,
            ),
        )
        if not steps and not warnings:
            continue
        video_key = f"warning_{_safe_identifier(Path(source_path).stem)}"
        action_segments = [
            IndEgoTemporalSegment(
                label=record.text,
                source_path=source_path,
                layer="keystep",
            )
            for record in steps
        ]
        warning_items = [
            IndEgoWarning(
                step=str(record.metadata.get("step") or record.text.split(":", 1)[0]),
                description=str(record.metadata.get("warning") or record.text),
                source_path=source_path,
                run_id=str(record.metadata.get("run_id") or ""),
            )
            for record in warnings[:12]
        ]
        scenes.append(
            _build_standard_scene(
                video_key=video_key,
                segment_index=1,
                category="mistake_detection",
                transcript=None,
                action_segments=[],
                keystep_segments=action_segments,
                warnings=warning_items,
                source_paths=[source_path],
                max_transcript_chars=0,
            )
        )
    return scenes


def _load_transcripts(root: Path) -> dict[str, dict[str, str]]:
    transcript_by_video: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("*/narration_transcript_*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            name = str(record.get("name") or "").strip()
            transcript = str(record.get("transcript") or "").strip()
            if not name or not transcript:
                continue
            transcript_by_video[_video_key(name)] = {
                "category": _normalize_category(
                    str(record.get("category") or _category_from_path(path))
                ),
                "name": name,
                "transcript": _collapse_ws(transcript),
                "source_path": _relative_source_path(path, root),
            }
    return transcript_by_video


def _load_temporal_groups(root: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "action": [],
            "keystep": [],
            "source_paths": set(),
            "category": None,
        }
    )
    for path, layer in _temporal_annotation_files(root):
        segments = parse_via_temporal_segments(path, source_root=root, layer=layer)
        if not segments:
            continue
        category = _category_from_path(path)
        source_path = _relative_source_path(path, root)
        for video_key, video_segments in _segments_by_video_key(segments).items():
            groups[video_key][layer].extend(video_segments)
            groups[video_key]["source_paths"].add(source_path)
            if not groups[video_key]["category"]:
                groups[video_key]["category"] = category
    return groups


def _temporal_annotation_files(root: Path) -> list[tuple[Path, IndEgoSegmentLayer]]:
    manifest_path = root / "MANIFEST.json"
    candidates: list[tuple[Path, IndEgoSegmentLayer]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("files", []):
            layer = _TEMPORAL_LAYER_BY_MANIFEST_LAYER.get(row.get("layer"))
            if not layer:
                continue
            path = root / row["path"]
            if path.exists():
                candidates.append((path, layer))
        return sorted(candidates, key=lambda item: str(item[0]))

    for path in root.glob("**/*.json"):
        parts = {part.lower() for part in path.parts}
        if ".cache" in parts or "mistake_detection" in parts:
            continue
        if "annotated" in path.parent.name.lower() and "keystep" in path.parent.name.lower():
            candidates.append((path, "keystep"))
        elif path.parent.name.lower() == "annotations":
            candidates.append((path, "action"))
    return sorted(candidates, key=lambda item: str(item[0]))


def _build_chunked_scenes(
    *,
    video_key: str,
    category: str,
    transcript: dict[str, str] | None,
    action_segments: list[IndEgoTemporalSegment],
    keystep_segments: list[IndEgoTemporalSegment],
    source_paths: list[str],
    max_segments_per_scene: int,
    max_transcript_chars: int,
) -> list[IndEgoStandardScene]:
    driver_segments = action_segments or keystep_segments
    if not driver_segments:
        return []
    chunks = list(_chunks(driver_segments, max_segments_per_scene))
    scenes: list[IndEgoStandardScene] = []
    for index, chunk in enumerate(chunks, start=1):
        start, end = _range_for_segments(chunk)
        chunk_keysteps = _overlapping_segments(keystep_segments, start, end)
        chunk_actions = chunk if action_segments else []
        if not chunk_actions and action_segments:
            chunk_actions = _overlapping_segments(action_segments, start, end)
        scene = _build_standard_scene(
            video_key=video_key,
            segment_index=index,
            category=category,
            transcript=transcript,
            action_segments=chunk_actions,
            keystep_segments=chunk_keysteps if action_segments else chunk,
            warnings=[],
            source_paths=source_paths
            + ([transcript["source_path"]] if transcript else []),
            max_transcript_chars=max_transcript_chars,
        )
        scenes.append(scene)
    return scenes


def _build_transcript_only_scene(
    *,
    video_key: str,
    transcript: dict[str, str],
    max_transcript_chars: int,
) -> IndEgoStandardScene:
    return _build_standard_scene(
        video_key=video_key,
        segment_index=1,
        category=transcript["category"],
        transcript=transcript,
        action_segments=[],
        keystep_segments=[],
        warnings=[],
        source_paths=[transcript["source_path"]],
        max_transcript_chars=max_transcript_chars,
    )


def _build_standard_scene(
    *,
    video_key: str,
    segment_index: int,
    category: str,
    transcript: dict[str, str] | None,
    action_segments: list[IndEgoTemporalSegment],
    keystep_segments: list[IndEgoTemporalSegment],
    warnings: list[IndEgoWarning],
    source_paths: list[str],
    max_transcript_chars: int,
) -> IndEgoStandardScene:
    video_id = _safe_identifier(video_key)
    segment_id = f"s{segment_index}"
    scene_id = f"indego_{video_id}_{segment_id}"
    start, end = _range_for_segments([*action_segments, *keystep_segments])
    timestamp = _format_time_range(start, end)
    transcript_text = _truncate_text(
        transcript["transcript"] if transcript else "",
        max_transcript_chars,
    )

    raw_lines = [
        f"Video {video_id}, segment {segment_id}, {timestamp or 'time unknown'}.",
        f"Category: {category}.",
    ]
    if transcript and transcript.get("name"):
        raw_lines.append(f"Source video file: {transcript['name']}.")
    if transcript_text:
        raw_lines.append(f"Transcript excerpt: {transcript_text}")

    action_evidence = _render_segment_lines("Action", action_segments)
    keystep_evidence = _render_segment_lines("Keystep", keystep_segments)
    warning_evidence = _render_warning_lines(warnings)
    if action_evidence:
        raw_lines.extend(["Annotated actions:", *action_evidence])
    if keystep_evidence:
        raw_lines.extend(["Annotated keysteps:", *keystep_evidence])
    if warning_evidence:
        raw_lines.extend(["Warnings:", *warning_evidence])

    if not action_evidence and not keystep_evidence:
        raw_lines.append("No temporal action annotations were available for this video.")

    raw_text = "\n".join(raw_lines)
    action_entries = [
        GroundedEntry(
            text=segment.label,
            evidence=line,
            entry_type="action",
        )
        for segment, line in zip(action_segments, action_evidence)
    ]
    if not action_entries:
        action_entries = [
            GroundedEntry(
                text=segment.label,
                evidence=line,
                entry_type="action",
            )
            for segment, line in zip(keystep_segments, keystep_evidence)
        ]
    tool_object_entries = _infer_tool_object_entries(
        [*action_segments, *keystep_segments],
        [*action_evidence, *keystep_evidence],
    )
    actor_entries = _actor_entries(video_id, raw_text)
    warning_entries = [
        GroundedEntry(
            text=f"warning: {warning.description}",
            evidence=line,
            entry_type="quality",
        )
        for warning, line in zip(warnings, warning_evidence)
    ]
    uncertainty = [
        "Generated deterministically from IndEgo text/annotation layers; no LLM was used.",
    ]
    if transcript and len(transcript["transcript"]) > max_transcript_chars:
        uncertainty.append("Transcript was truncated for compact KG extraction input.")
    if warnings:
        uncertainty.append("Warning records come from IndEgo mistake annotations.")

    unified = build_industrial_unified_text(
        raw_text=raw_text,
        scene_id=video_id,
        segment_id=segment_id,
        timestamp=timestamp,
        scene=f"{category} procedural scene",
        actors=actor_entries,
        action_sequence=action_entries,
        tools_objects=tool_object_entries,
        quality_results=warning_entries,
        uncertainty=uncertainty,
        source_adapter="indego_adapter",
        annotation_text="\n".join(
            [*action_evidence, *keystep_evidence, *warning_evidence]
        )
        or None,
        transcript_text=transcript_text or None,
    )
    description = f"{category} scene from {video_id}"
    return IndEgoStandardScene(
        scene_id=scene_id,
        video_id=video_id,
        category=category,
        description=description,
        source_paths=sorted(set(source_paths)),
        raw_text=raw_text,
        unified_record=unified,
        action_segments=action_segments,
        keystep_segments=keystep_segments,
        warnings=warnings,
    )


def _load_warning_scenes(root: Path) -> list[IndEgoStandardScene]:
    warning_dir = root / "Mistake_Detection" / "renamed_annotation_mistakes"
    if not warning_dir.exists():
        return []
    scenes: list[IndEgoStandardScene] = []
    for path in sorted(warning_dir.glob("A_Task_*.json")):
        warnings, steps = _parse_warning_file(path, root)
        if not warnings and not steps:
            continue
        video_key = _safe_identifier(path.stem)
        action_segments = [
            IndEgoTemporalSegment(
                label=step,
                source_path=_relative_source_path(path, root),
                layer="keystep",
            )
            for step in steps
        ]
        scenes.append(
            _build_standard_scene(
                video_key=f"warning_{video_key}",
                segment_index=1,
                category="mistake_detection",
                transcript=None,
                action_segments=[],
                keystep_segments=action_segments,
                warnings=warnings[:12],
                source_paths=[_relative_source_path(path, root)],
                max_transcript_chars=0,
            )
        )
    return scenes


def _parse_warning_file(path: Path, root: Path) -> tuple[list[IndEgoWarning], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return [], []
    template = data.get("template") if isinstance(data.get("template"), dict) else {}
    steps = [
        _normalize_label(str(step))
        for step in template.get("description", [])
        if step
    ]
    warnings: list[IndEgoWarning] = []
    seen: set[tuple[str, str]] = set()
    for run_id, record in data.items():
        if run_id == "template" or not isinstance(record, dict):
            continue
        mistakes = record.get("mistakes", [])
        descriptions = record.get("description", [])
        if not isinstance(mistakes, list) or not isinstance(descriptions, list):
            continue
        for index, flag in enumerate(mistakes):
            if not flag or index >= len(descriptions) or not descriptions[index]:
                continue
            step = steps[index] if index < len(steps) else f"step {index + 1}"
            description = _normalize_label(str(descriptions[index]))
            key = (step, description)
            if key in seen:
                continue
            seen.add(key)
            warnings.append(
                IndEgoWarning(
                    step=step,
                    description=description,
                    source_path=_relative_source_path(path, root),
                    run_id=str(run_id),
                )
            )
    return warnings, steps


def _label_from_metadata(metadata: dict[str, Any], attributes: dict[str, Any]) -> str | None:
    av = metadata.get("av")
    if not isinstance(av, dict):
        return None
    for attr_id, raw_value in av.items():
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        attr = attributes.get(str(attr_id), {}) if isinstance(attributes, dict) else {}
        options = attr.get("options", {}) if isinstance(attr, dict) else {}
        if value in options:
            label = str(options[value]).strip()
        else:
            label = value
        if label and label.lower() != "default":
            return label
    return None


def _normalize_label(value: str) -> str:
    label = value.replace("__", "_").replace("_", " ")
    label = _LEADING_STEP_RE.sub("", label)
    return _collapse_ws(label).strip(" .")


def _segments_by_video_key(
    segments: Iterable[IndEgoTemporalSegment],
) -> dict[str, list[IndEgoTemporalSegment]]:
    grouped: dict[str, list[IndEgoTemporalSegment]] = defaultdict(list)
    for segment in segments:
        key = _video_key(segment.video_name or Path(segment.source_path).stem)
        grouped[key].append(segment)
    return grouped


def _render_segment_lines(
    prefix: str,
    segments: list[IndEgoTemporalSegment],
) -> list[str]:
    lines = []
    for index, segment in enumerate(segments, start=1):
        timestamp = f" ({segment.timestamp})" if segment.timestamp else ""
        lines.append(f"{prefix} {index}: {segment.label}{timestamp}.")
    return lines


def _render_warning_lines(warnings: list[IndEgoWarning]) -> list[str]:
    return [
        f"Warning {index}: step '{warning.step}' has issue '{warning.description}'."
        for index, warning in enumerate(warnings, start=1)
    ]


def _infer_tool_object_entries(
    segments: list[IndEgoTemporalSegment],
    evidence_lines: list[str],
) -> list[GroundedEntry]:
    entries: list[GroundedEntry] = []
    seen: set[str] = set()
    for segment, evidence in zip(segments, evidence_lines):
        label = segment.label.casefold()
        for term in _TOOL_OBJECT_TERMS:
            if term in label and term not in seen:
                entries.append(
                    GroundedEntry(
                        text=term,
                        evidence=evidence,
                        entry_type=(
                            "tool" if term in _INDEGO_TOOL_TERMS else "object"
                        ),
                    )
                )
                seen.add(term)
    return entries


def _actor_entries(video_id: str, raw_text: str) -> list[GroundedEntry]:
    entries: list[GroundedEntry] = []
    for match in _USER_RE.finditer(video_id):
        evidence = match.group(0)
        if evidence in raw_text:
            entries.append(
                GroundedEntry(
                    text=f"User {match.group(1)}",
                    evidence=evidence,
                    entry_type="role",
                )
            )
    return entries


def _chunks(
    items: list[IndEgoTemporalSegment],
    chunk_size: int,
) -> Iterable[list[IndEgoTemporalSegment]]:
    size = max(1, chunk_size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _overlapping_segments(
    segments: list[IndEgoTemporalSegment],
    start: float | None,
    end: float | None,
) -> list[IndEgoTemporalSegment]:
    if start is None or end is None:
        return segments
    overlapping = [
        segment
        for segment in segments
        if segment.start_seconds is not None
        and segment.end_seconds is not None
        and segment.end_seconds >= start
        and segment.start_seconds <= end
    ]
    return overlapping


def _range_for_segments(
    segments: list[IndEgoTemporalSegment],
) -> tuple[float | None, float | None]:
    starts = [segment.start_seconds for segment in segments if segment.start_seconds is not None]
    ends = [segment.end_seconds for segment in segments if segment.end_seconds is not None]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _segment_sort_key(segment: IndEgoTemporalSegment) -> tuple[float, str]:
    return (
        segment.start_seconds if segment.start_seconds is not None else float("inf"),
        segment.label,
    )


def _format_time_range(start: float | None, end: float | None) -> str | None:
    if start is None or end is None:
        return None
    return f"{_format_seconds(start)}-{_format_seconds(end)}"


def _format_seconds(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate_text(text: str, max_chars: int) -> str:
    collapsed = _collapse_ws(text)
    if max_chars <= 0 or len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rsplit(" ", 1)[0] + " ..."


def _collapse_ws(text: str) -> str:
    return " ".join(str(text).split())


def _video_key(name: str) -> str:
    stem = Path(str(name)).stem
    return _VIDEO_SUFFIX_RE.sub("", stem).casefold()


def _safe_identifier(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("_") or "indego_scene"


def _category_from_path(path: Path) -> str:
    parts = path.parts
    for part in parts:
        if re.match(r"^\d+_", part):
            return _normalize_category(part.split("_", 1)[1])
        if part == "Mistake_Detection":
            return "mistake_detection"
    return "indego"


def _normalize_category(value: str) -> str:
    return _normalize_label(value).casefold().replace(" ", "_")


def _infer_temporal_layer(path: Path) -> IndEgoSegmentLayer:
    if "keystep" in path.parent.name.casefold():
        return "keystep"
    return "action"


def _relative_source_path(path: Path, root: str | Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(Path(root)))
    except ValueError:
        return str(path)


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value:
            return value
    return "indego"
