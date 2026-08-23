"""Common source-independent segment contract and controlled text renderer.

Source adapters map heterogeneous fields into this contract. This module
does not read datasets, clean source data, or call an LLM.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


NormalizedMentionKind = Literal[
    "object",
    "tool",
    "role",
    "scene",
    "procedure",
]


class NormalizedMention(BaseModel):
    """One typed, evidence-grounded mention in a normalized segment."""

    mention_id: str
    kind: NormalizedMentionKind
    text: str
    evidence: str
    canonical_name: str | None = None
    semantic_role: str | None = None
    uncertainty: str | None = None


class NormalizedAction(BaseModel):
    """One source-ordered action frame used by the controlled renderer."""

    action_id: str
    text: str
    verb: str
    evidence: str
    direct_object_id: str | None = None
    tool_id: str | None = None
    role_id: str | None = None
    uncertainty: str | None = None


class NormalizedSegment(BaseModel):
    """Shared boundary between source adapters and KG extraction.

    Fields come from adapters or grounded preprocessing; no gold labels
    are stored in this extraction input contract.
    """

    source_adapter: str
    scene_id: str
    segment_id: str
    timestamp: str | None = None
    annotation_text: str | None = None
    transcript_text: str | None = None
    actions: list[NormalizedAction] = Field(default_factory=list)
    objects: list[NormalizedMention] = Field(default_factory=list)
    tools: list[NormalizedMention] = Field(default_factory=list)
    roles: list[NormalizedMention] = Field(default_factory=list)
    scenes: list[NormalizedMention] = Field(default_factory=list)
    procedures: list[NormalizedMention] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    source_text: str

    @model_validator(mode="after")
    def validate_references(self) -> "NormalizedSegment":
        """Reject duplicate IDs and action links to absent normalized mentions."""

        mentions = [
            *self.objects,
            *self.tools,
            *self.roles,
            *self.scenes,
            *self.procedures,
        ]
        all_ids = [
            *(action.action_id for action in self.actions),
            *(mention.mention_id for mention in mentions),
        ]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Normalized segment IDs must be unique.")

        object_ids = {
            item.mention_id for item in [*self.objects, *self.tools]
        }
        tool_ids = {item.mention_id for item in self.tools}
        role_ids = {item.mention_id for item in self.roles}
        for action in self.actions:
            if action.direct_object_id and action.direct_object_id not in object_ids:
                raise ValueError(
                    f"Unknown direct_object_id {action.direct_object_id!r}."
                )
            if action.tool_id and action.tool_id not in tool_ids:
                raise ValueError(f"Unknown tool_id {action.tool_id!r}.")
            if action.role_id and action.role_id not in role_ids:
                raise ValueError(f"Unknown role_id {action.role_id!r}.")
        return self

    def to_controlled_text(self) -> str:
        """Render a deterministic extraction input from normalized fields."""

        return render_controlled_segment(self)

    def split_by_actions(
        self,
        *,
        max_actions: int = 6,
        overlap_actions: int = 1,
    ) -> list["NormalizedSegment"]:
        """Split a long scene without changing renderer-owned global IDs.

        Layers 4-6 run independently on these parts. One overlapping
        action retains boundary context and makes a single part failure local.
        """

        if max_actions < 1:
            raise ValueError("max_actions must be at least 1")
        if overlap_actions < 0 or overlap_actions >= max_actions:
            raise ValueError("overlap_actions must be between 0 and max_actions - 1")
        if len(self.actions) <= max_actions:
            return [self]

        mention_by_id = {
            item.mention_id: item
            for item in [*self.objects, *self.tools, *self.roles]
        }
        parts: list[NormalizedSegment] = []
        start = 0
        step = max_actions - overlap_actions
        while start < len(self.actions):
            actions = self.actions[start : start + max_actions]
            referenced_ids = {
                reference
                for action in actions
                for reference in (
                    action.direct_object_id,
                    action.tool_id,
                    action.role_id,
                )
                if reference
            }
            # 中文：显式动作链接必须保留；另外保留证据与本地动作窗口相交的实体，
            # 让 LLM 仍能自主发现未预先写入槽位的关系。
            # English: Keep explicit links plus mentions grounded in the local
            # action window, so the LLM may still discover non-slotted relations.
            action_evidence = " ".join(action.evidence for action in actions)
            local_mentions = [
                mention
                for mention in mention_by_id.values()
                if mention.mention_id in referenced_ids
                or _text_contains(action_evidence, mention.evidence)
                or _text_contains(action_evidence, mention.text)
            ]
            local_source = _bounded_source_evidence(
                self.source_text,
                [action.evidence for action in actions],
                [mention.evidence for mention in local_mentions],
            )
            part_number = len(parts) + 1
            parts.append(
                self.model_copy(
                    update={
                        "segment_id": f"{self.segment_id}__part_{part_number}",
                        "actions": actions,
                        "objects": [
                            item for item in local_mentions if item.kind == "object"
                        ],
                        "tools": [
                            item for item in local_mentions if item.kind == "tool"
                        ],
                        "roles": [
                            item for item in local_mentions if item.kind == "role"
                        ],
                        # Scene/Procedure are small, source-level context entities.
                        # 场景与流程实体数量很小，作为每个子段的公共上下文保留。
                        "scenes": self.scenes,
                        "procedures": self.procedures,
                        "source_text": local_source,
                        "uncertainty": [
                            *self.uncertainty,
                            (
                                "Layers 4-6 partition "
                                f"{part_number}/{_part_count(len(self.actions), max_actions, overlap_actions)} "
                                f"from parent segment {self.segment_id}."
                            ),
                        ],
                    },
                    deep=True,
                )
            )
            if start + max_actions >= len(self.actions):
                break
            start += step
        return parts


def render_controlled_segment(segment: NormalizedSegment) -> str:
    """Render verb + direct object + tool frames with original evidence.

    Controlled sentences only reorder normalized fields. Original source
    evidence remains available for relation proposals and hard validation.
    """

    mention_by_id = {
        item.mention_id: item
        for item in [
            *segment.objects,
            *segment.tools,
            *segment.roles,
            *segment.scenes,
            *segment.procedures,
        ]
    }
    segment_lines = [
        f"source_adapter={segment.source_adapter}",
        f"scene_id={segment.scene_id}",
        f"segment_id={segment.segment_id}",
    ]
    if segment.timestamp:
        segment_lines.append(f"timestamp={segment.timestamp}")

    entity_lines = [
        _render_mention(item)
        for item in [
            *segment.roles,
            *segment.objects,
            *segment.tools,
            *segment.scenes,
            *segment.procedures,
        ]
    ] or ["NONE"]
    action_lines: list[str] = []
    for action in segment.actions:
        direct_object = mention_by_id.get(action.direct_object_id or "")
        tool = mention_by_id.get(action.tool_id or "")
        role = mention_by_id.get(action.role_id or "")
        sentence = _controlled_action_sentence(action, direct_object, tool, role)
        action_lines.append(
            " | ".join(
                [
                    action.action_id,
                    "ACTION",
                    f"verb={action.verb}",
                    f"direct_object_id={_mention_id(direct_object)}",
                    f"direct_object_name={_mention_name(direct_object)}",
                    f"tool_id={_mention_id(tool)}",
                    f"tool_name={_mention_name(tool)}",
                    f"role_id={_mention_id(role)}",
                    f"role_name={_mention_name(role)}",
                    f"sentence={sentence}",
                    f"source_evidence={action.evidence}",
                    f"uncertainty={action.uncertainty or 'NONE'}",
                ]
            )
        )
    if not action_lines:
        action_lines = ["NONE"]

    evidence_lines = [segment.source_text]
    uncertainty_lines = segment.uncertainty or ["NONE"]
    return "\n\n".join(
        [
            "[NORMALIZED SEGMENT]\n" + "\n".join(segment_lines),
            "[NORMALIZED ENTITIES]\n" + "\n".join(entity_lines),
            "[ACTION SEQUENCE]\n" + "\n".join(action_lines),
            "[SOURCE EVIDENCE]\n" + "\n".join(evidence_lines),
            "[UNCERTAINTY]\n" + "\n".join(uncertainty_lines),
        ]
    )


def _render_mention(item: NormalizedMention) -> str:
    values = [
        item.mention_id,
        item.kind.upper(),
        f"name={item.text}",
        f"canonical_name={item.canonical_name or 'NONE'}",
        f"semantic_role={item.semantic_role or 'NONE'}",
        f"source_evidence={item.evidence}",
        f"uncertainty={item.uncertainty or 'NONE'}",
    ]
    return " | ".join(values)


def _mention_id(item: NormalizedMention | None) -> str:
    """Return only the stable ID; names are rendered in a separate field."""

    return item.mention_id if item else "NONE"


def _mention_name(item: NormalizedMention | None) -> str:
    """Return a readable name without contaminating the ID token."""

    return item.text if item else "NONE"


def _controlled_action_sentence(
    action: NormalizedAction,
    direct_object: NormalizedMention | None,
    tool: NormalizedMention | None,
    role: NormalizedMention | None,
) -> str:
    if not direct_object and not tool:
        action_phrase = action.text.strip().rstrip(".")
        return action_phrase + "."

    parts = []
    # 中文：Role 单独保存在 role=R... 槽位；标准动作句严格保持
    # “动词 + 直接对象 + 工具”，避免为不同语言做主谓变位。
    # English: Role stays in its own slot. The sentence is strictly
    # verb + direct object + tool and needs no language-specific conjugation.
    parts.append(action.verb)
    if direct_object:
        parts.append(direct_object.text)
    if tool:
        # 中文：当工具本身是被拿取/放置的直接对象时，不强行生成 "with"。
        # English: If a tool is the only affected item, do not force a "with" phrase.
        parts.extend((["with", tool.text] if direct_object else [tool.text]))
    sentence = " ".join(part.strip() for part in parts if part.strip()).strip()
    if not sentence:
        sentence = action.text.strip()
    return sentence.rstrip(".") + "."


def _part_count(action_count: int, max_actions: int, overlap_actions: int) -> int:
    if action_count <= max_actions:
        return 1
    step = max_actions - overlap_actions
    return 1 + (action_count - max_actions + step - 1) // step


def _bounded_source_evidence(
    source_text: str,
    action_evidence: list[str],
    mention_evidence: list[str],
) -> str:
    """Keep matched sentences plus one neighbour for bounded coreference.

    Retain one neighbouring sentence for local pronouns without copying
    the full transcript into every partition.
    """

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", source_text)
        if item.strip()
    ]
    needles = [
        item.strip()
        for item in [*action_evidence, *mention_evidence]
        if item and item.strip()
    ]
    selected: set[int] = set()
    for index, sentence in enumerate(sentences):
        if any(
            _text_contains(sentence, needle)
            or _text_contains(needle, sentence)
            for needle in needles
        ):
            selected.update(
                candidate
                for candidate in (index - 1, index, index + 1)
                if 0 <= candidate < len(sentences)
            )
    if selected:
        return " ".join(sentences[index] for index in sorted(selected))

    # 中文：找不到句界时仍只保留原始证据片段；这些片段必须来自已有合同。
    # English: If sentence lookup fails, retain only contract-provided evidence.
    fallback = list(dict.fromkeys(needles))
    return " ".join(fallback) if fallback else source_text


def _text_contains(container: str, value: str) -> bool:
    normalized_container = " ".join(container.casefold().split())
    normalized_value = " ".join(value.casefold().split())
    return bool(normalized_value) and normalized_value in normalized_container
