"""Factory data-source cleaning before text preprocessing or KG extraction.

The first supported source is the downloaded text portion of Fraunhofer IPK's
IndEgo dataset.  This module deliberately stops at deterministic cleaning:
source inspection, label normalization, timestamp validation, provenance, and
quality diagnostics.  It does not create unified prompt text and does not call
an LLM.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


FactorySource = Literal["indego"]
RecordType = Literal[
    "transcript",
    "action_segment",
    "keystep_segment",
    "scenario_keystep",
    "mistake_step",
    "mistake_warning",
    "vqa_item",
    "file_notice",
]
IssueSeverity = Literal["info", "warning", "error"]


_INDEGO_TEMPORAL_LAYERS = {
    "fine_grained_annotations": "action_segment",
    "annotated_keysteps": "keystep_segment",
}
_TEXT_KEYSTEP_LAYERS = {"scenario_keysteps", "category_keysteps"}
_VIDEO_SUFFIX_RE = re.compile(r"(_480)?(\.mp4)?$", re.IGNORECASE)
_LEADING_STEP_RE = re.compile(r"^\s*\d+\s*[_:\-.]?\s*")


class CleaningIssue(BaseModel):
    """A quality issue found while cleaning a source file or record."""

    severity: IssueSeverity
    code: str
    message: str
    source_path: str | None = None
    record_id: str | None = None


class CleanFactoryRecord(BaseModel):
    """One deterministic, provenance-bearing record from a factory source."""

    source_name: FactorySource
    source_layer: str
    source_path: str
    record_type: RecordType
    record_id: str
    text: str
    normalized_text: str
    category: str | None = None
    video_id: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    issues: list[CleaningIssue] = Field(default_factory=list)

    @property
    def timestamp(self) -> str | None:
        """Return a compact timestamp range when temporal bounds are present."""

        if self.start_seconds is None or self.end_seconds is None:
            return None
        return f"{_format_seconds(self.start_seconds)}-{_format_seconds(self.end_seconds)}"


class FactoryCleaningReport(BaseModel):
    """Clean records plus run-level diagnostics for one factory source."""

    source_name: FactorySource
    root_path: str
    files_seen: int
    records: list[CleanFactoryRecord] = Field(default_factory=list)
    issues: list[CleaningIssue] = Field(default_factory=list)

    def records_by_type(self) -> dict[str, int]:
        """Return a stable count of cleaned records per record type."""

        return dict(sorted(Counter(record.record_type for record in self.records).items()))

    def issues_by_code(self) -> dict[str, int]:
        """Return a stable count of issues per issue code."""

        all_issues = [*self.issues]
        for record in self.records:
            all_issues.extend(record.issues)
        return dict(sorted(Counter(issue.code for issue in all_issues).items()))

    def to_jsonl(self) -> str:
        """Serialize cleaned records as JSON Lines."""

        return "\n".join(
            record.model_dump_json(exclude_none=True)
            for record in self.records
        )

    def save_jsonl(self, output_path: str | Path) -> Path:
        """Write cleaned records to a JSONL file for downstream modules."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_jsonl()
        path.write_text(f"{text}\n" if text else "", encoding="utf-8")
        return path


def clean_factory_data_source(
    source_dir: str | Path,
    *,
    source_name: FactorySource = "indego",
) -> FactoryCleaningReport:
    """Clean a supported factory data source."""

    if source_name == "indego":
        return clean_indego_text_layers(source_dir)
    raise ValueError(f"Unsupported factory data source: {source_name!r}")


def clean_indego_text_layers(source_dir: str | Path) -> FactoryCleaningReport:
    """Clean the downloaded IndEgo text-layer subset into standard records."""

    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"IndEgo text-layer folder not found: {root}")

    files = _indego_files(root)
    records: list[CleanFactoryRecord] = []
    issues: list[CleaningIssue] = []

    for row in files:
        path = root / row["path"]
        layer = row["layer"]
        if not path.exists():
            issues.append(
                _issue(
                    "error",
                    "missing_manifest_file",
                    "Manifest entry points to a missing local file.",
                    row["path"],
                )
            )
            continue
        try:
            parsed = _clean_indego_file(root, path, layer)
        except json.JSONDecodeError as error:
            issues.append(
                _issue(
                    "error",
                    "invalid_json",
                    f"JSON parsing failed: {error.msg}",
                    _relative_path(path, root),
                )
            )
            continue
        records.extend(parsed.records)
        issues.extend(parsed.issues)

    records, duplicate_issues = _flag_duplicate_record_ids(records)
    issues.extend(duplicate_issues)
    return FactoryCleaningReport(
        source_name="indego",
        root_path=str(root),
        files_seen=len(files),
        records=records,
        issues=issues,
    )


class _ParsedFile(BaseModel):
    records: list[CleanFactoryRecord] = Field(default_factory=list)
    issues: list[CleaningIssue] = Field(default_factory=list)


def _clean_indego_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    if layer == "narration_transcript":
        return _clean_transcript_file(root, path, layer)
    if layer in _INDEGO_TEMPORAL_LAYERS:
        return _clean_temporal_annotation_file(root, path, layer)
    if layer in _TEXT_KEYSTEP_LAYERS:
        return _clean_keystep_text_file(root, path, layer)
    if layer == "mistake_warning_annotations":
        return _clean_mistake_warning_file(root, path, layer)
    if layer == "vqa_text_benchmark":
        return _clean_vqa_file(root, path, layer)
    if layer == "tool_object_metadata":
        return _file_notice(
            root,
            path,
            layer,
            "tool_object_metadata_not_parsed",
            "Tool/object metadata is present but not parsed in the first cleaning pass.",
        )
    if layer in {"dataset_documentation", "singular_action_labels", "mistake_task_annotations"}:
        return _file_notice(
            root,
            path,
            layer,
            "reference_layer_not_expanded",
            "Reference layer kept as provenance but not expanded into atomic cleaning records.",
        )
    return _file_notice(
        root,
        path,
        layer,
        "unknown_layer",
        "Unknown IndEgo layer was not expanded into atomic cleaning records.",
    )


def _clean_transcript_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    source_path = _relative_path(path, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return _ParsedFile(
            issues=[
                _issue("error", "unexpected_transcript_shape", "Expected a list of transcript records.", source_path)
            ]
        )

    records = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            continue
        raw_name = _collapse_ws(str(row.get("name") or ""))
        text = _collapse_ws(str(row.get("transcript") or ""))
        video_id = _video_id(raw_name) if raw_name else None
        record_id = f"{source_path}#transcript:{video_id or index}"
        record_issues = []
        if not raw_name:
            record_issues.append(
                _issue("warning", "missing_video_name", "Transcript record has no video file name.", source_path, record_id)
            )
        if not text:
            record_issues.append(
                _issue("error", "empty_text", "Transcript record has no transcript text.", source_path, record_id)
            )
        if not text:
            continue
        records.append(
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type="transcript",
                record_id=record_id,
                text=text,
                normalized_text=_normalize_text(text),
                category=_normalize_category(str(row.get("category") or _category_from_path(path))),
                video_id=video_id,
                metadata={"raw_name": raw_name},
                issues=record_issues,
            )
        )
    return _ParsedFile(records=records)


def _clean_temporal_annotation_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    source_path = _relative_path(path, root)
    record_type = _INDEGO_TEMPORAL_LAYERS[layer]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "metadata" not in data:
        return _ParsedFile(
            issues=[
                _issue("error", "unexpected_via_shape", "Expected a VIA JSON object with metadata.", source_path)
            ]
        )

    attributes = data.get("attribute") if isinstance(data.get("attribute"), dict) else {}
    files = data.get("file") if isinstance(data.get("file"), dict) else {}
    records = []
    issues = []
    for metadata_id, metadata in data.get("metadata", {}).items():
        if not isinstance(metadata, dict):
            continue
        text = _normalize_label(_label_from_metadata(metadata, attributes) or "")
        if not text:
            continue

        z_value = metadata.get("z") if isinstance(metadata.get("z"), list) else []
        start = _safe_float(z_value[0]) if len(z_value) >= 1 else None
        end = _safe_float(z_value[1]) if len(z_value) >= 2 else None
        video_ref = str(metadata.get("vid") or "")
        file_record = files.get(video_ref, {}) if isinstance(files, dict) else {}
        raw_video_name = str(file_record.get("fname") or "") if isinstance(file_record, dict) else ""
        video_id = _video_id(raw_video_name or path.stem)
        record_id = f"{source_path}#{metadata_id}"
        record_issues = _temporal_issues(source_path, record_id, start, end)
        if not raw_video_name:
            record_issues.append(
                _issue("warning", "missing_video_name", "Temporal record has no file name mapping.", source_path, record_id)
            )
        records.append(
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type=record_type,  # type: ignore[arg-type]
                record_id=record_id,
                text=text,
                normalized_text=_normalize_text(text),
                category=_category_from_path(path),
                video_id=video_id,
                start_seconds=start,
                end_seconds=end,
                metadata={"metadata_id": metadata_id, "raw_video_name": raw_video_name},
                issues=record_issues,
            )
        )
    if not records:
        issues.append(_issue("warning", "empty_temporal_file", "No temporal labels were found.", source_path))
    return _ParsedFile(records=records, issues=issues)


def _clean_keystep_text_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    source_path = _relative_path(path, root)
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = _normalize_label(line)
        if not text:
            continue
        record_id = f"{source_path}#line:{index}"
        records.append(
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type="scenario_keystep",
                record_id=record_id,
                text=text,
                normalized_text=_normalize_text(text),
                category=_category_from_path(path),
                metadata={"line_number": index},
            )
        )
    return _ParsedFile(records=records)


def _clean_mistake_warning_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    if path.suffix == ".txt":
        return _clean_mistake_warning_text_file(root, path, layer)

    source_path = _relative_path(path, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _ParsedFile(
            issues=[_issue("error", "unexpected_mistake_shape", "Expected a mistake annotation object.", source_path)]
        )

    template = data.get("template") if isinstance(data.get("template"), dict) else {}
    steps = [
        _normalize_label(str(step))
        for step in template.get("description", [])
        if step
    ]
    records = []
    for index, step in enumerate(steps, start=1):
        record_id = f"{source_path}#template_step:{index}"
        records.append(
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type="mistake_step",
                record_id=record_id,
                text=step,
                normalized_text=_normalize_text(step),
                category="mistake_detection",
                metadata={"step_index": index},
            )
        )

    seen_warnings: set[tuple[str, str, str]] = set()
    for run_id, row in data.items():
        if run_id == "template" or not isinstance(row, dict):
            continue
        mistakes = row.get("mistakes", [])
        descriptions = row.get("description", [])
        if not isinstance(mistakes, list) or not isinstance(descriptions, list):
            continue
        for index, flag in enumerate(mistakes):
            if not flag:
                continue
            description = descriptions[index] if index < len(descriptions) else None
            warning = _normalize_label(str(description or ""))
            if not warning:
                warning = "unspecified mistake"
            step = steps[index] if index < len(steps) else f"step {index + 1}"
            key = (str(run_id), step, warning)
            if key in seen_warnings:
                continue
            seen_warnings.add(key)
            record_id = f"{source_path}#{run_id}:warning:{index + 1}"
            records.append(
                CleanFactoryRecord(
                    source_name="indego",
                    source_layer=layer,
                    source_path=source_path,
                    record_type="mistake_warning",
                    record_id=record_id,
                    text=f"{step}: {warning}",
                    normalized_text=_normalize_text(f"{step}: {warning}"),
                    category="mistake_detection",
                    metadata={
                        "run_id": str(run_id),
                        "step": step,
                        "warning": warning,
                        "step_index": index + 1,
                    },
                )
            )
    return _ParsedFile(records=records)


def _clean_mistake_warning_text_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    source_path = _relative_path(path, root)
    lines = [
        _collapse_ws(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if _collapse_ws(line)
    ]
    if not lines:
        return _ParsedFile(
            issues=[_issue("warning", "empty_mistake_text_file", "Mistake annotation text file is empty.", source_path)]
        )

    steps = _parse_bracket_list(lines[0])
    records = []
    for index, step in enumerate(steps, start=1):
        record_id = f"{source_path}#template_step:{index}"
        records.append(
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type="mistake_step",
                record_id=record_id,
                text=step,
                normalized_text=_normalize_text(step),
                category="mistake_detection",
                metadata={"step_index": index},
            )
        )

    for line_index, line in enumerate(lines[1:], start=2):
        if ":" not in line:
            continue
        run_id, rest = line.split(":", 1)
        flags = _parse_flag_list(rest)
        descriptions = _parse_warning_descriptions(rest)
        warning_index = 0
        for step_index, flag in enumerate(flags, start=1):
            if flag != "1":
                continue
            step = steps[step_index - 1] if step_index <= len(steps) else f"step {step_index}"
            warning = descriptions[warning_index] if warning_index < len(descriptions) else "unspecified mistake"
            warning_index += 1
            record_id = f"{source_path}#{run_id.strip()}:warning:{step_index}"
            records.append(
                CleanFactoryRecord(
                    source_name="indego",
                    source_layer=layer,
                    source_path=source_path,
                    record_type="mistake_warning",
                    record_id=record_id,
                    text=f"{step}: {warning}",
                    normalized_text=_normalize_text(f"{step}: {warning}"),
                    category="mistake_detection",
                    metadata={
                        "run_id": run_id.strip(),
                        "step": step,
                        "warning": warning,
                        "step_index": step_index,
                        "line_number": line_index,
                    },
                )
            )
    return _ParsedFile(records=records)


def _clean_vqa_file(root: Path, path: Path, layer: str) -> _ParsedFile:
    source_path = _relative_path(path, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
    records = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        question = _collapse_ws(str(row.get("question") or row.get("Q") or ""))
        answer = _collapse_ws(str(row.get("answer") or row.get("A") or ""))
        text = _collapse_ws(" ".join(part for part in [question, answer] if part))
        if not text:
            continue
        record_id = f"{source_path}#vqa:{index}"
        records.append(
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type="vqa_item",
                record_id=record_id,
                text=text,
                normalized_text=_normalize_text(text),
                category=_category_from_path(path),
                metadata={"row_index": index},
            )
        )
    return _ParsedFile(records=records)


def _file_notice(root: Path, path: Path, layer: str, code: str, message: str) -> _ParsedFile:
    source_path = _relative_path(path, root)
    record_id = f"{source_path}#notice"
    issue = _issue("info", code, message, source_path, record_id)
    return _ParsedFile(
        records=[
            CleanFactoryRecord(
                source_name="indego",
                source_layer=layer,
                source_path=source_path,
                record_type="file_notice",
                record_id=record_id,
                text=message,
                normalized_text=_normalize_text(message),
                category=_category_from_path(path),
                issues=[issue],
            )
        ]
    )


def _indego_files(root: Path) -> list[dict[str, str]]:
    manifest_path = root / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", []) if isinstance(manifest, dict) else []
        rows = [
            {"path": str(row["path"]), "layer": str(row.get("layer") or "unknown")}
            for row in files
            if isinstance(row, dict) and row.get("path")
        ]
        return sorted(rows, key=lambda row: row["path"])

    rows = []
    for path in sorted(root.glob("**/*")):
        if path.is_file() and path.name != "MANIFEST.json":
            rows.append({"path": _relative_path(path, root), "layer": _infer_layer_from_path(path)})
    return rows


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
        label = str(options.get(value, value)).strip()
        if label and label.casefold() != "default":
            return label
    return None


def _parse_bracket_list(value: str) -> list[str]:
    match = re.search(r"\[([^\]]*)\]", value)
    if not match:
        return []
    return [
        _normalize_label(item)
        for item in match.group(1).split(",")
        if _normalize_label(item)
    ]


def _parse_flag_list(value: str) -> list[str]:
    match = re.search(r"\[([^\]]*)\]", value)
    if not match:
        return []
    return [
        item.strip()
        for item in match.group(1).split(",")
        if item.strip()
    ]


def _parse_warning_descriptions(value: str) -> list[str]:
    without_flags = re.sub(r"\[[^\]]*\]", "", value, count=1)
    return [
        _normalize_label(item)
        for item in without_flags.split(",")
        if _normalize_label(item)
    ]


def _temporal_issues(
    source_path: str,
    record_id: str,
    start: float | None,
    end: float | None,
) -> list[CleaningIssue]:
    issues = []
    if start is None or end is None:
        issues.append(
            _issue("warning", "missing_time_bounds", "Temporal segment has incomplete time bounds.", source_path, record_id)
        )
    elif end < start:
        issues.append(
            _issue("error", "invalid_time_range", "Temporal segment end is before start.", source_path, record_id)
        )
    return issues


def _flag_duplicate_record_ids(
    records: list[CleanFactoryRecord],
) -> tuple[list[CleanFactoryRecord], list[CleaningIssue]]:
    counts = Counter(record.record_id for record in records)
    duplicate_ids = {record_id for record_id, count in counts.items() if count > 1}
    if not duplicate_ids:
        return records, []
    issues = [
        _issue("warning", "duplicate_record_id", "Duplicate cleaned record id detected.", record.source_path, record.record_id)
        for record in records
        if record.record_id in duplicate_ids
    ]
    return records, issues


def _issue(
    severity: IssueSeverity,
    code: str,
    message: str,
    source_path: str | None = None,
    record_id: str | None = None,
) -> CleaningIssue:
    return CleaningIssue(
        severity=severity,
        code=code,
        message=message,
        source_path=source_path,
        record_id=record_id,
    )


def _normalize_label(value: str) -> str:
    label = value.replace("__", "_").replace("_", " ")
    label = _LEADING_STEP_RE.sub("", label)
    return _collapse_ws(label).strip(" .")


def _normalize_text(value: str) -> str:
    return _collapse_ws(value).casefold()


def _collapse_ws(value: str) -> str:
    return " ".join(str(value).split())


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _video_id(name: str) -> str:
    stem = Path(str(name)).stem
    clean = _VIDEO_SUFFIX_RE.sub("", stem)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean).strip("_").casefold()


def _category_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^\d+_", part):
            return _normalize_category(part.split("_", 1)[1])
        if part == "Mistake_Detection":
            return "mistake_detection"
        if part == "VQA":
            return "vqa"
    return "indego"


def _normalize_category(value: str) -> str:
    return _normalize_label(value).casefold().replace(" ", "_")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _infer_layer_from_path(path: Path) -> str:
    lower = str(path).casefold()
    if "narration_transcript" in path.name:
        return "narration_transcript"
    if "annotated" in lower and "keystep" in lower and path.suffix == ".json":
        return "annotated_keysteps"
    if path.parent.name.casefold() == "annotations" and path.suffix == ".json":
        return "fine_grained_annotations"
    if path.name.startswith("keysteps") and path.suffix == ".txt":
        return "scenario_keysteps"
    if "mistake_detection" in lower and path.suffix in {".json", ".txt"}:
        return "mistake_warning_annotations"
    if path.parent.name == "VQA" and path.suffix == ".json":
        return "vqa_text_benchmark"
    if path.suffix == ".xlsx":
        return "tool_object_metadata"
    return "unknown"
