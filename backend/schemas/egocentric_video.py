"""
Schemas for egocentric video scene descriptions.

These models are the MVP extraction contract for the project route:
scene descriptions -> typed procedural knowledge -> property graph.

中文说明：
这个文件定义“第一人称专家视频场景描述”要抽取成什么结构。
LLM 的输出会被 Instructor 校验成这些 Pydantic 类，后续 graph 层再把这些类转换成节点和边。
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from backend.schemas.base import KGEntity, KGRelation
from backend.schemas.process_knowledge.entities import (
    ProcessParameter,
    Procedure,
    Tool,
    Worker,
)


class Provenance(BaseModel):
    """Source metadata for an extracted entity or relation."""

    # provenance 用来追踪“这个信息来自哪一段视频/文本”，方便后续评估和溯源。
    video_id: Optional[str] = None
    segment_id: Optional[str] = None
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    timestamp_start_seconds: Optional[float] = None
    timestamp_end_seconds: Optional[float] = None
    source_text: Optional[str] = None
    confidence: Optional[float] = None


class Scene(KGEntity):
    """A temporally bounded egocentric video scene or segment."""

    # Scene 表示视频中的一个时间片段，例如某个 00:00-00:20 的操作场景。
    video_id: Optional[str] = None
    segment_id: Optional[str] = None
    timestamp_start_seconds: Optional[float] = None
    timestamp_end_seconds: Optional[float] = None
    location: Optional[str] = None
    confidence: Optional[float] = None


class SceneObject(KGEntity):
    """A physical object observed in the scene."""

    # SceneObject 表示画面里被操作或被观察到的物体，例如 steel plate、bracket。
    object_type: Optional[str] = None
    state: Optional[str] = None
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class Action(KGEntity):
    """An action or procedural step observed in an egocentric scene."""

    # Action 是核心实体：表示视频中发生的一个动作/步骤，例如 align steel plate。
    action_type: Optional[Literal[
        "preparation",
        "manipulation",
        "inspection",
        "measurement",
        "assembly",
        "disassembly",
        "cleaning",
        "safety_check",
        "documentation",
        "other",
    ]] = None
    actor: Optional[Worker] = None
    scene: Optional[Scene] = None
    sequence_index: Optional[int] = None
    timestamp_start_seconds: Optional[float] = None
    timestamp_end_seconds: Optional[float] = None
    parameters: list[ProcessParameter] = Field(default_factory=list)
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class UsesTool(KGRelation):
    """An action uses a tool."""

    # 对应图里的边：Action --USES_TOOL--> Tool。
    action: Action
    tool: Tool
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class ActsOnObject(KGRelation):
    """An action manipulates, inspects, or otherwise acts on an object."""

    # 对应图里的边：Action --ACTS_ON--> Object，用来表达动作作用在哪个物体上。
    action: Action
    object: SceneObject
    role: Optional[Literal["input", "output", "target", "support", "unknown"]] = None
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class ActionOrder(KGRelation):
    """Temporal ordering between two actions."""

    # 对应图里的边：before Action --BEFORE--> after Action，用来表达步骤顺序。
    before: Action
    after: Action
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class ActionCauses(KGRelation):
    """A causal relation between two actions or observed states."""

    # 对应图里的边：cause Action --CAUSES--> effect Action，用来表达因果关系。
    cause: Action
    effect: Action
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class ActionPartOfProcedure(KGRelation):
    """An observed action belongs to a procedure."""

    # 对应图里的边：Action --PART_OF--> Procedure，用来把动作归入某个工艺流程。
    action: Action
    procedure: Procedure
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class ActionObservedInScene(KGRelation):
    """An action is observed in a scene segment."""

    # 对应图里的边：Action --OBSERVED_IN--> Scene，用来保留动作发生的视频片段。
    action: Action
    scene: Scene
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None


class EgocentricVideoExtraction(BaseModel):
    """Full extraction result for one egocentric video or scene-description batch."""

    # 这是一次完整抽取的总容器：实体列表和关系列表都会放在这里。
    video_id: Optional[str] = None
    scenes: list[Scene] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    actors: list[Worker] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    parameters: list[ProcessParameter] = Field(default_factory=list)
    uses_tool: list[UsesTool] = Field(default_factory=list)
    acts_on_object: list[ActsOnObject] = Field(default_factory=list)
    action_order: list[ActionOrder] = Field(default_factory=list)
    action_causes: list[ActionCauses] = Field(default_factory=list)
    part_of_procedure: list[ActionPartOfProcedure] = Field(default_factory=list)
    observed_in_scene: list[ActionObservedInScene] = Field(default_factory=list)
    source_text: Optional[str] = None
