"""
统一格式文本生成模块。

本模块位于实验流程的输入准备阶段：
原始场景描述 -> 统一格式记录 -> 供同一个抽取模型读取的统一文本。

它的目的不是让语言模型润色或补充事实，而是把人工确认过的信息放入固定栏目，
并强制每条整理后的事实保留原文证据。这样，后续可以公平比较：
“原始文本输入”与“统一格式文本输入”对知识图谱抽取质量和稳定性的影响。
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field


class EvidenceNotFoundError(ValueError):
    """整理后的事实所引用的证据无法在原始文本中找到时抛出的错误。"""


class GroundedEntry(BaseModel):
    """
    一条有原文支撑的整理信息。

    参数：
        text: 放入统一格式文本中的简洁事实，例如“measure gap”。
        evidence: 原始场景描述中支持该事实的原文片段。
        uncertainty: 人工发现的不确定之处，例如“工具型号不可见”。
    """

    text: str
    evidence: str
    uncertainty: str | None = None


class UnifiedTextRecord(BaseModel):
    """
    一个场景的统一格式文本记录。

    该对象既能保存为 JSON 作为实验数据，也能通过 ``to_prompt_text()``
    生成固定栏目文本，作为后续抽取模型的 unified 输入。
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

        固定栏目让 raw 与 unified 实验的唯一区别保持在输入表示方式上；
        原始文本始终保留在末尾，便于模型和后续验证器追踪证据。
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

    输入：
        raw_text: 原始视频场景描述，是所有整理信息的证据来源。
        scene_id / segment_id / timestamp: 场景来源信息。
        各条目列表: 人工从原文中整理出的事实及证据。

    输出：
        可以保存为 JSON、也可以渲染成模型输入文本的 ``UnifiedTextRecord``。

    重要约束：
        本函数不会调用 LLM，也不会自动新增事实；
        如果某条事实提供的证据不存在于原始文本中，函数会拒绝生成记录。
    """

    if not raw_text.strip():
        raise ValueError("原始文本不能为空。")

    # 所有输入栏目都必须引用原文证据，确保 unified 输入没有无依据的新事实。
    entries = _all_entries(actors, action_sequence, tools_objects, outcomes_parameters)
    _validate_evidence(entries, raw_text)

    # 保留完整原始文本与片段信息，后续 grounding/hallucination 校验会继续使用它们。
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


def build_industrial_unified_text(
    raw_text: str,
    scene_id: str,
    segment_id: str,
    timestamp: str | None,
    scene: str,
    action_sequence: list[GroundedEntry],
    tools_objects: list[GroundedEntry],
    process_parameters: list[GroundedEntry] | None = None,
    quality_results: list[GroundedEntry] | None = None,
    actors: list[GroundedEntry] | None = None,
    uncertainty: list[str] | None = None,
) -> UnifiedTextRecord:
    """
    构建工业数据清洗模块的统一文本记录。

    该入口面向最终工业数据，而不是医院 pilot 数据。它只接受人工或规则辅助
    整理出的原文支持事实，并复用 ``build_unified_text`` 的 evidence 校验。
    """

    process_parameters = process_parameters or []
    quality_results = quality_results or []
    _validate_evidence([*process_parameters, *quality_results], raw_text)
    return build_unified_text(
        raw_text=raw_text,
        scene_id=scene_id,
        segment_id=segment_id,
        timestamp=timestamp,
        scene_segment=scene,
        actors=actors or [],
        action_sequence=action_sequence,
        tools_objects=tools_objects,
        outcomes_parameters=[],
        evidence_uncertainty=uncertainty or [],
    ).model_copy(
        update={
            "process_parameters": process_parameters,
            "quality_results": quality_results,
        }
    )


def _all_entries(*entry_groups: list[GroundedEntry]) -> list[GroundedEntry]:
    """把多个栏目的信息合并为一组，便于统一执行证据检查。"""

    return [entry for group in entry_groups for entry in group]


def _validate_evidence(entries: Iterable[GroundedEntry], raw_text: str) -> None:
    """检查每条整理信息引用的 evidence 是否确实位于原始文本中。"""

    normalized_source = _normalize_text(raw_text)
    for entry in entries:
        normalized_evidence = _normalize_text(entry.evidence)
        if not normalized_evidence or normalized_evidence not in normalized_source:
            raise EvidenceNotFoundError(
                f"事实 {entry.text!r} 的证据未在原始文本中找到: {entry.evidence!r}"
            )


def _render_entries(entries: list[GroundedEntry], numbered: bool = False) -> list[str]:
    """把一个栏目里的条目渲染成可读文本，并在需要时展示不确定说明。"""

    if not entries:
        return ["无"]

    lines = []
    for index, entry in enumerate(entries, start=1):
        prefix = f"{index}. " if numbered else "- "
        line = f"{prefix}{entry.text} | 证据: {entry.evidence}"
        if entry.uncertainty:
            line += f" | 不确定性: {entry.uncertainty}"
        lines.append(line)
    return lines


def _normalize_text(value: str) -> str:
    """统一大小写与空白，允许证据检查忽略换行和额外空格差异。"""

    return " ".join(value.casefold().split())
