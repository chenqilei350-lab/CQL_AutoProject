"""Unified-format text generation for grounded industrial extraction.

This module prepares experiment input through the following path:
raw scene description -> unified record -> unified text for the same extractor.

It does not ask an LLM to polish text or add facts. It places reviewed facts into
fixed sections and requires every fact to retain source evidence, enabling a fair
comparison between raw and unified inputs for KG quality and stability.

Treat this file as the preprocessing coordinator. If category lists are already
supplied, it validates and renders them directly. If no lists are supplied, it
calls ``automatic.py``. Start with ``build_industrial_unified_text()`` and skip
underscore-prefixed helpers on the first read.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from backend.preprocessing.normalized_segment import (
    NormalizedAction,
    NormalizedMention,
    NormalizedSegment,
)


GroundedEntryType = Literal[
    "action",
    "object",
    "tool",
    "role",
    "parameter",
    "quality",
]


class EvidenceNotFoundError(ValueError):
    """Raised when evidence for an organized fact is absent from the source."""


class AutomaticPreprocessingError(RuntimeError):
    """Raised when automatic preprocessing fails."""


# [Block 01] 一张保存“整理结果 + 原文证据”的事实卡片。
# [Block 01] One fact card that stores an organized fact plus source evidence.
class GroundedEntry(BaseModel):
    """One organized fact grounded in the original source text.

    Args:
        text: A concise fact rendered in the unified text.
        evidence: An exact source span supporting the fact.
        uncertainty: An optional note about source uncertainty.
        aliases: Alternative names retained during canonicalization.
        additional_evidence: Other source spans retained after alias merging.
    """

    text: str
    evidence: str
    uncertainty: str | None = None
    # 中文：标准名只作为附加属性保存，不能覆盖原文中有证据的名称。
    # English: A canonical name is metadata and never replaces the grounded mention.
    canonical_name: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    aliases: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    additional_evidence: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    # 中文：这些可选字段让 Adapter 明确提供类型和动作槽位，避免 Renderer
    # 根据某个数据集的词表猜测 Tool/Object/Role。
    # English: Optional type and action slots let adapters provide semantics
    # explicitly instead of making the renderer guess from dataset vocabularies.
    entry_type: GroundedEntryType | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    verb: str | None = Field(default=None, exclude_if=lambda value: value is None)
    direct_object: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    tool: str | None = Field(default=None, exclude_if=lambda value: value is None)
    role: str | None = Field(default=None, exclude_if=lambda value: value is None)


# [Block 02] 一个场景的统一记录，以及生成固定栏目文本的方法。
# [Block 02] One unified scene record and its fixed-section text renderer.
class UnifiedTextRecord(BaseModel):
    """A unified-format text record for one scene.

    The object can be saved as experiment JSON or rendered through
    ``to_prompt_text()`` as fixed-section input for the downstream extractor.
    """

    scene_id: str
    segment_id: str
    timestamp: str | None = None
    scene_segment: str
    actors: list[GroundedEntry] = Field(default_factory=list)
    action_sequence: list[GroundedEntry] = Field(default_factory=list)
    tools_objects: list[GroundedEntry] = Field(default_factory=list)
    process_parameters: list[GroundedEntry] = Field(default_factory=list)
    quality_results: list[GroundedEntry] = Field(default_factory=list)
    outcomes_parameters: list[GroundedEntry] = Field(default_factory=list)
    evidence_uncertainty: list[str] = Field(default_factory=list)
    source_text: str
    normalized_segment: NormalizedSegment | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    def to_prompt_text(self) -> str:
        """Render the record as fixed-section text for the extraction model.

        Fixed sections isolate the experimental difference to input representation.
        The original source remains at the end for model and validator grounding.
        """

        scene_lines = [
            f"Scene ID: {self.scene_id}",
            f"Segment ID: {self.segment_id}",
            f"Scene Description: {self.scene_segment}",
        ]
        if self.timestamp:
            scene_lines.append(f"Time Range: {self.timestamp}")

        sections = [
            ("[SCENE / SEGMENT]", scene_lines),
            ("[ACTORS]", _render_entries(self.actors)),
            ("[ACTION SEQUENCE]", _render_entries(self.action_sequence, numbered=True)),
            ("[TOOLS / OBJECTS]", _render_entries(self.tools_objects)),
            ("[PROCESS PARAMETERS]", _render_entries(self.process_parameters)),
            ("[QUALITY / RESULTS]", _render_entries(self.quality_results)),
            ("[OUTCOMES / PARAMETERS]", _render_entries(self.outcomes_parameters)),
            (
                "[EVIDENCE / UNCERTAINTY]",
                self.evidence_uncertainty or ["No additional uncertainty recorded"],
            ),
            ("[SOURCE TEXT]", [self.source_text]),
        ]

        return "\n\n".join(
            "\n".join([title, *content])
            for title, content in sections
        )

    def to_extraction_text(self) -> str:
        """Render the six-layer pipeline input consumed by entity extraction.

        The legacy renderer remains for historical comparisons. Production
        extraction uses this controlled representation.
        """

        segment = self.normalized_segment or _normalized_segment_from_record(self)
        return segment.to_controlled_text()


# [Block 03] 人工/规则模式：严格检查证据后组装统一记录。
# [Block 03] Manual/rule mode: strictly validate evidence, then build the record.
def build_unified_text(
    raw_text: str,
    scene_id: str,
    segment_id: str,
    timestamp: str | None,
    scene_segment: str,
    actors: list[GroundedEntry],
    action_sequence: list[GroundedEntry],
    tools_objects: list[GroundedEntry],
    outcomes_parameters: list[GroundedEntry],
    evidence_uncertainty: list[str] | None = None,
    *,
    source_adapter: str = "generic",
    annotation_text: str | None = None,
    transcript_text: str | None = None,
) -> UnifiedTextRecord:
    """Build a unified record from manually organized scene fields.

    Args:
        raw_text: The original scene description and evidence source.
        scene_id / segment_id / timestamp: Scene provenance information.
        Entry lists: Human-organized facts and their source evidence.

    Returns:
        A ``UnifiedTextRecord`` that can be serialized or rendered for extraction.

    Notes:
        This function never calls an LLM or adds facts. Missing evidence rejects
        the record.
    """

    if not raw_text.strip():
        raise ValueError("raw_text must not be empty.")

    # 所有栏目都必须引用原文证据，避免加入无依据事实。
    # Every section must cite source evidence so unsupported facts are not added.
    entries = _all_entries(actors, action_sequence, tools_objects, outcomes_parameters)
    _validate_evidence(entries, raw_text)

    # 保留完整原文与片段信息，供后续 grounding/hallucination 校验使用。
    # Keep source text and segment metadata for downstream grounding checks.
    record = UnifiedTextRecord(
        scene_id=scene_id,
        segment_id=segment_id,
        timestamp=timestamp,
        scene_segment=scene_segment,
        actors=actors,
        action_sequence=action_sequence,
        tools_objects=tools_objects,
        outcomes_parameters=outcomes_parameters,
        evidence_uncertainty=evidence_uncertainty or [],
        source_text=raw_text,
    )
    return record.model_copy(
        update={
            "normalized_segment": _normalized_segment_from_record(
                record,
                source_adapter=source_adapter,
                annotation_text=annotation_text,
                transcript_text=transcript_text,
            )
        }
    )


# [Block 04] 工业文本主入口：决定使用已有分类还是调用 LLM 自动分类。
# [Block 04] Main industrial entry: use supplied categories or invoke the LLM.
def build_industrial_unified_text(
    raw_text: str,
    scene_id: str,
    segment_id: str,
    timestamp: str | None,
    scene: str,
    action_sequence: list[GroundedEntry] | None = None,
    tools_objects: list[GroundedEntry] | None = None,
    process_parameters: list[GroundedEntry] | None = None,
    quality_results: list[GroundedEntry] | None = None,
    actors: list[GroundedEntry] | None = None,
    uncertainty: list[str] | None = None,
    *,
    source_adapter: str = "industrial_text",
    annotation_text: str | None = None,
    transcript_text: str | None = None,
    model: str = "llama3.1:8b",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    alias_candidate_threshold: float = 0.65,
    alias_top_k: int = 3,
    preprocessing_chunk_chars: int = 500,
) -> UnifiedTextRecord:
    """Build a unified record for industrial text preprocessing.

    Explicit category lists keep the deterministic manual/rule-assisted path.
    When every category is ``None``, the local LLM classifies the source, invalid
    evidence is filtered, and tool/object mentions are canonicalized.
    """

    provided_groups = (
        actors,
        action_sequence,
        tools_objects,
        process_parameters,
        quality_results,
    )
    # 第一步：只要调用方提供了任意分类列表，就继续使用原来的确定性流程。
    # Step 1: Any supplied category list keeps the original deterministic path.
    automatic_mode = all(group is None for group in provided_groups)
    preprocessing_notes: list[str] = []
    if automatic_mode:
        # 第二步：所有分类列表都没提供时，交给 automatic.py 自动整理。
        # Step 2: When every category is absent, delegate to automatic.py.
        draft, preprocessing_notes = _automatically_preprocess_industrial_text(
            raw_text=raw_text,
            model=model,
            embedding_model=embedding_model,
            alias_candidate_threshold=alias_candidate_threshold,
            alias_top_k=alias_top_k,
            chunk_max_chars=preprocessing_chunk_chars,
        )
        actors = draft.actors
        action_sequence = draft.action_sequence
        tools_objects = draft.tools_objects
        process_parameters = draft.process_parameters
        quality_results = draft.quality_results

    # 第三步：无论数据来自人工、规则还是 LLM，最后都转成相同的栏目列表。
    # Step 3: Manual, rule-based, and LLM results all use the same category lists.
    actors = actors or []
    action_sequence = action_sequence or []
    tools_objects = tools_objects or []
    process_parameters = process_parameters or []
    quality_results = quality_results or []
    _validate_evidence([*process_parameters, *quality_results], raw_text)
    # 第四步：生成最终 UnifiedTextRecord，供后面的抽取模型读取。
    # Step 4: Build the final UnifiedTextRecord for the downstream extractor.
    return build_unified_text(
        raw_text=raw_text,
        scene_id=scene_id,
        segment_id=segment_id,
        timestamp=timestamp,
        scene_segment=scene,
        actors=actors,
        action_sequence=action_sequence,
        tools_objects=tools_objects,
        outcomes_parameters=[],
        evidence_uncertainty=[*(uncertainty or []), *preprocessing_notes],
        source_adapter=source_adapter,
        annotation_text=annotation_text,
        transcript_text=transcript_text,
    ).model_copy(
        update={
            "process_parameters": process_parameters,
            "quality_results": quality_results,
        }
    )


def _all_entries(*entry_groups: list[GroundedEntry]) -> list[GroundedEntry]:
    """Flatten category groups for shared validation."""

    return [entry for group in entry_groups for entry in group]


def _validate_evidence(entries: Iterable[GroundedEntry], raw_text: str) -> None:
    """Verify that every evidence span occurs in the source text."""

    normalized_source = _normalize_text(raw_text)
    for entry in entries:
        evidence_values = [entry.evidence, *entry.additional_evidence]
        for evidence in evidence_values:
            normalized_evidence = _normalize_text(evidence)
            if not normalized_evidence or normalized_evidence not in normalized_source:
                raise EvidenceNotFoundError(
                    f"Evidence for fact {entry.text!r} was not found in source text: "
                    f"{evidence!r}"
                )


def _render_entries(entries: list[GroundedEntry], numbered: bool = False) -> list[str]:
    """Render category entries and optional uncertainty notes."""

    if not entries:
        return ["None"]

    lines = []
    for index, entry in enumerate(entries, start=1):
        prefix = f"{index}. " if numbered else "- "
        line = f"{prefix}{entry.text}"
        if entry.canonical_name:
            line += f" | Canonical Name: {entry.canonical_name}"
        if entry.aliases:
            line += f" | Aliases: {', '.join(entry.aliases)}"
        evidence = "; ".join([entry.evidence, *entry.additional_evidence])
        line += f" | Evidence: {evidence}"
        if entry.uncertainty:
            line += f" | Uncertainty: {entry.uncertainty}"
        lines.append(line)
    return lines


def _normalized_segment_from_record(
    record: UnifiedTextRecord,
    *,
    source_adapter: str = "legacy_unified_record",
    annotation_text: str | None = None,
    transcript_text: str | None = None,
) -> NormalizedSegment:
    """Convert grounded columns into the common Normalized Segment contract.

    Link existing grounded entries only. Explicit adapter metadata wins
    over source-independent lexical matching.
    """

    roles = [
        _normalized_mention(entry, "role", index)
        for index, entry in enumerate(record.actors, start=1)
    ]
    tool_entries = [
        entry
        for entry in record.tools_objects
        if _entry_is_explicit_or_legacy_tool(entry, record.action_sequence)
    ]
    object_entries = [
        entry for entry in record.tools_objects if entry not in tool_entries
    ]
    tools = [
        _normalized_mention(entry, "tool", index)
        for index, entry in enumerate(tool_entries, start=1)
    ]
    objects = [
        _normalized_mention(entry, "object", index)
        for index, entry in enumerate(object_entries, start=1)
    ]
    # 中文：Scene/Procedure 只有在输入来源提供可追溯证据时才进入合同；
    # 否则关系层不会开放 OBSERVED_IN/PART_OF。
    # English: Scene/Procedure enter the contract only with traceable source
    # evidence; otherwise their relation types remain unavailable downstream.
    scene_evidence = _first_exact_context_evidence(
        record.source_text,
        (record.scene_id, record.segment_id),
    )
    procedure_evidence = _first_exact_context_evidence(
        record.source_text,
        (record.scene_segment,),
    )
    scenes = (
        [
            NormalizedMention(
                mention_id="S1",
                kind="scene",
                text=record.scene_id,
                evidence=scene_evidence,
            )
        ]
        if scene_evidence
        else []
    )
    procedures = (
        [
            NormalizedMention(
                mention_id="P1",
                kind="procedure",
                text=record.scene_segment,
                evidence=procedure_evidence,
            )
        ]
        if procedure_evidence
        else []
    )

    actions: list[NormalizedAction] = []
    for index, entry in enumerate(record.action_sequence, start=1):
        direct_object = _select_action_mention(
            entry,
            [*objects, *tools],
            explicit_name=entry.direct_object,
            preferred_roles={"target", "direct_object", "dobj"},
            allow_single=len(record.action_sequence) == 1,
        )
        tool = _select_action_mention(
            entry,
            tools,
            explicit_name=entry.tool,
            allow_single=len(record.action_sequence) == 1,
        )
        role = _select_action_mention(
            entry,
            roles,
            explicit_name=entry.role,
            allow_single=len(roles) == 1,
        )
        if direct_object and tool and direct_object.mention_id == tool.mention_id:
            # 中文：同一 Tool 不能同时渲染成直接对象和 "with tool"，否则会生成
            # "pick up drill with drill"。显式槽位优先，推断槽位清空。
            # English: Never render one Tool twice as both object and instrument.
            if entry.tool and not entry.direct_object:
                direct_object = None
            else:
                tool = None
        actions.append(
            NormalizedAction(
                action_id=f"A{index}",
                text=entry.text,
                verb=entry.verb
                or _infer_verb_phrase(
                    entry.text,
                    direct_object.text if direct_object else None,
                    tool.text if tool else None,
                    role.text if role else None,
                ),
                evidence=entry.evidence,
                direct_object_id=(
                    direct_object.mention_id if direct_object else None
                ),
                tool_id=tool.mention_id if tool else None,
                role_id=role.mention_id if role else None,
                uncertainty=entry.uncertainty,
            )
        )

    return NormalizedSegment(
        source_adapter=source_adapter,
        scene_id=record.scene_id,
        segment_id=record.segment_id,
        timestamp=record.timestamp,
        annotation_text=annotation_text,
        transcript_text=transcript_text,
        actions=actions,
        objects=objects,
        tools=tools,
        roles=roles,
        scenes=scenes,
        procedures=procedures,
        uncertainty=record.evidence_uncertainty,
        source_text=record.source_text,
    )


def _normalized_mention(
    entry: GroundedEntry,
    kind: Literal["object", "tool", "role", "scene", "procedure"],
    index: int,
) -> NormalizedMention:
    prefix = {
        "object": "O",
        "tool": "T",
        "role": "R",
        "scene": "S",
        "procedure": "P",
    }[kind]
    return NormalizedMention(
        mention_id=f"{prefix}{index}",
        kind=kind,
        text=entry.text,
        evidence=entry.evidence,
        canonical_name=entry.canonical_name,
        semantic_role=entry.role if kind == "object" else None,
        uncertainty=entry.uncertainty,
    )


def _first_exact_context_evidence(
    source_text: str,
    candidates: tuple[str, ...],
) -> str | None:
    """Return an exact source-backed context label without semantic guessing."""

    normalized_source = source_text.casefold()
    for candidate in candidates:
        value = candidate.strip()
        if value and value.casefold() in normalized_source:
            return value
    return None


def _entry_is_explicit_or_legacy_tool(
    entry: GroundedEntry,
    actions: list[GroundedEntry],
) -> bool:
    """Migrate legacy mixed columns using generic grammatical tool cues only."""

    if entry.entry_type == "tool":
        return True
    if entry.entry_type == "object":
        return False
    escaped = re.escape(entry.text.strip())
    tool_pattern = re.compile(
        rf"\b(?:with|using|use|uses|used)\s+(?:a\s+|an\s+|the\s+)?{escaped}\b",
        re.IGNORECASE,
    )
    return any(
        tool_pattern.search(f"{action.text} {action.evidence}")
        for action in actions
    )


def _select_action_mention(
    action: GroundedEntry,
    mentions: list[NormalizedMention],
    *,
    explicit_name: str | None,
    preferred_roles: set[str] | None = None,
    allow_single: bool = False,
) -> NormalizedMention | None:
    if explicit_name:
        explicit = _normalize_text(explicit_name)
        for mention in mentions:
            if explicit in {
                _normalize_text(mention.text),
                _normalize_text(mention.canonical_name or ""),
            }:
                return mention

    searchable = _normalize_text(f"{action.text} {action.evidence}")
    lexical_matches = [
        mention
        for mention in mentions
        if _normalize_text(mention.text) in searchable
        or (
            mention.canonical_name
            and _normalize_text(mention.canonical_name) in searchable
        )
    ]
    if preferred_roles:
        preferred = [
            mention
            for mention in lexical_matches
            if (mention.semantic_role or "").casefold() in preferred_roles
        ]
        if preferred:
            return preferred[0]
    if lexical_matches:
        return lexical_matches[0]
    if allow_single and len(mentions) == 1:
        return mentions[0]
    if allow_single and preferred_roles:
        preferred = [
            mention
            for mention in mentions
            if (mention.semantic_role or "").casefold() in preferred_roles
        ]
        if len(preferred) == 1:
            return preferred[0]
    return None


def _infer_verb_phrase(
    action_text: str,
    direct_object: str | None,
    tool: str | None,
    role: str | None,
) -> str:
    """Infer a source-grounded verb phrase without a dataset vocabulary."""

    phrase = action_text.strip().rstrip(".")
    if role and _normalize_text(phrase).startswith(_normalize_text(role)):
        phrase = phrase[len(role) :].lstrip(" ,:-")
    boundaries = []
    lowered = phrase.casefold()
    for value in (direct_object, tool):
        if not value:
            continue
        position = lowered.find(value.casefold())
        if position > 0:
            boundaries.append(position)
    if boundaries:
        phrase = phrase[: min(boundaries)].strip()
    phrase = re.sub(r"\b(with|using|on|into|to|the|a|an)\s*$", "", phrase, flags=re.I)
    return phrase.strip() or action_text.strip()


def _automatically_preprocess_industrial_text(
    *,
    raw_text: str,
    model: str,
    embedding_model: str,
    alias_candidate_threshold: float,
    alias_top_k: int,
    chunk_max_chars: int,
):
    """Lazily import LLM and embedding dependencies."""

    from backend.preprocessing.automatic import automatically_preprocess_industrial_text

    return automatically_preprocess_industrial_text(
        raw_text=raw_text,
        model=model,
        embedding_model=embedding_model,
        alias_candidate_threshold=alias_candidate_threshold,
        alias_top_k=alias_top_k,
        chunk_max_chars=chunk_max_chars,
    )


def _normalize_text(value: str) -> str:
    """Normalize case and whitespace for evidence checks."""

    return " ".join(value.casefold().split())
