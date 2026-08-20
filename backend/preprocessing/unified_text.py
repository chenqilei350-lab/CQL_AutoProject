"""
统一格式文本生成模块。
Unified-format text generation module.

本模块位于实验流程的输入准备阶段：
原始场景描述 -> 统一格式记录 -> 供同一个抽取模型读取的统一文本。
This module prepares experiment input:
raw scene description -> unified record -> unified text for the same extractor.

它的目的不是让语言模型润色或补充事实，而是把人工确认过的信息放入固定栏目，
并强制每条整理后的事实保留原文证据。这样，后续可以公平比较：
“原始文本输入”与“统一格式文本输入”对知识图谱抽取质量和稳定性的影响。
It does not ask an LLM to polish text or add facts. It places reviewed facts into
fixed sections and requires every fact to retain source evidence, enabling a fair
comparison between raw and unified inputs for KG quality and stability.

初学者阅读提示：可以把本文件理解为“文本预处理管理员”。
它先判断调用方是否已经分好类别：如果已经分好，就直接检查并生成统一文本；
如果没有分好，就调用 automatic.py 自动分类。阅读时先看
``build_industrial_unified_text()``，其他以下划线开头的函数暂时可以跳过。

Beginner guide: treat this file as the preprocessing coordinator. If category
lists are already supplied, it validates and renders them directly. If no lists
are supplied, it calls automatic.py. Start with ``build_industrial_unified_text()``
and skip underscore-prefixed helpers on the first read.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field


class EvidenceNotFoundError(ValueError):
    """整理事实的证据不在原文中时抛出。 / Raised when evidence is absent."""


class AutomaticPreprocessingError(RuntimeError):
    """自动预处理失败时抛出。 / Raised when automatic preprocessing fails."""


# [Block 01] 一张保存“整理结果 + 原文证据”的事实卡片。
# [Block 01] One fact card that stores an organized fact plus source evidence.
class GroundedEntry(BaseModel):
    """
    一条有原文支撑的整理信息。
    One organized fact grounded in the original source text.

    参数：
        text: 放入统一格式文本中的简洁事实，例如“measure gap”。
        evidence: 原始场景描述中支持该事实的原文片段。
        uncertainty: 人工发现的不确定之处，例如“工具型号不可见”。
        aliases: 名称统一前保留下来的其他名称。
        additional_evidence: 别名条目合并后保留的其他原文证据。

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
    aliases: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    additional_evidence: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


# [Block 02] 一个场景的统一记录，以及生成固定栏目文本的方法。
# [Block 02] One unified scene record and its fixed-section text renderer.
class UnifiedTextRecord(BaseModel):
    """
    一个场景的统一格式文本记录。
    A unified-format text record for one scene.

    该对象既能保存为 JSON 作为实验数据，也能通过 ``to_prompt_text()``
    生成固定栏目文本，作为后续抽取模型的 unified 输入。
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

    def to_prompt_text(self) -> str:
        """
        把记录渲染成给抽取模型读取的固定栏目文本。
        Render the record as fixed-section text for the extraction model.

        固定栏目让 raw 与 unified 实验的唯一区别保持在输入表示方式上；
        原始文本始终保留在末尾，便于模型和后续验证器追踪证据。
        Fixed sections isolate the experimental difference to input representation.
        The original source remains at the end for model and validator grounding.
        """

        scene_lines = [
            f"场景编号: {self.scene_id}",
            f"片段编号: {self.segment_id}",
            f"场景说明: {self.scene_segment}",
        ]
        if self.timestamp:
            scene_lines.append(f"时间范围: {self.timestamp}")

        sections = [
            ("[场景 / 片段]", scene_lines),
            ("[执行人员]", _render_entries(self.actors)),
            ("[动作顺序]", _render_entries(self.action_sequence, numbered=True)),
            ("[工具 / 对象]", _render_entries(self.tools_objects)),
            ("[工艺参数]", _render_entries(self.process_parameters)),
            ("[质量 / 结果]", _render_entries(self.quality_results)),
            ("[结果 / 参数]", _render_entries(self.outcomes_parameters)),
            ("[证据 / 不确定性]", self.evidence_uncertainty or ["无额外不确定性记录"]),
            ("[原始文本]", [self.source_text]),
        ]

        return "\n\n".join(
            "\n".join([title, *content])
            for title, content in sections
        )


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
) -> UnifiedTextRecord:
    """
    从人工整理的场景字段构建统一格式文本记录。
    Build a unified record from manually organized scene fields.

    输入：
        raw_text: 原始视频场景描述，是所有整理信息的证据来源。
        scene_id / segment_id / timestamp: 场景来源信息。
        各条目列表: 人工从原文中整理出的事实及证据。

    输出：
        可以保存为 JSON、也可以渲染成模型输入文本的 ``UnifiedTextRecord``。

    重要约束：
        本函数不会调用 LLM，也不会自动新增事实；
        如果某条事实提供的证据不存在于原始文本中，函数会拒绝生成记录。

    Inputs:
        raw_text: The original scene description and evidence source.
        scene_id / segment_id / timestamp: Scene provenance information.
        Entry lists: Human-organized facts and their source evidence.

    Output:
        A ``UnifiedTextRecord`` that can be serialized or rendered for extraction.

    Constraint:
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
    return UnifiedTextRecord(
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
    model: str = "llama3.1:8b",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    alias_candidate_threshold: float = 0.65,
    alias_top_k: int = 3,
) -> UnifiedTextRecord:
    """
    构建工业数据清洗模块的统一文本记录。
    Build a unified record for industrial text preprocessing.

    该入口面向最终工业数据，而不是医院 pilot 数据。显式提供任意分类列表时，
    它继续使用确定性的人工/规则辅助流程。所有分类列表都为 ``None`` 时，它会
    调用本地 LLM 自动分类，再过滤无原文证据的条目并统一工具/对象名称。
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
    ).model_copy(
        update={
            "process_parameters": process_parameters,
            "quality_results": quality_results,
        }
    )


def _all_entries(*entry_groups: list[GroundedEntry]) -> list[GroundedEntry]:
    """合并栏目供统一校验。 / Flatten category groups for shared validation."""

    return [entry for group in entry_groups for entry in group]


def _validate_evidence(entries: Iterable[GroundedEntry], raw_text: str) -> None:
    """检查证据是否在原文中。 / Verify that every evidence span is in source."""

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
    """渲染栏目条目。 / Render category entries and optional uncertainty notes."""

    if not entries:
        return ["无"]

    lines = []
    for index, entry in enumerate(entries, start=1):
        prefix = f"{index}. " if numbered else "- "
        line = f"{prefix}{entry.text}"
        if entry.aliases:
            line += f" | 别名: {', '.join(entry.aliases)}"
        evidence = "; ".join([entry.evidence, *entry.additional_evidence])
        line += f" | 证据: {evidence}"
        if entry.uncertainty:
            line += f" | 不确定性: {entry.uncertainty}"
        lines.append(line)
    return lines


def _automatically_preprocess_industrial_text(
    *,
    raw_text: str,
    model: str,
    embedding_model: str,
    alias_candidate_threshold: float,
    alias_top_k: int,
):
    """延迟导入自动逻辑。 / Lazily import LLM and embedding dependencies."""

    from backend.preprocessing.automatic import automatically_preprocess_industrial_text

    return automatically_preprocess_industrial_text(
        raw_text=raw_text,
        model=model,
        embedding_model=embedding_model,
        alias_candidate_threshold=alias_candidate_threshold,
        alias_top_k=alias_top_k,
    )


def _normalize_text(value: str) -> str:
    """统一大小写和空白。 / Normalize case and whitespace for evidence checks."""

    return " ".join(value.casefold().split())
