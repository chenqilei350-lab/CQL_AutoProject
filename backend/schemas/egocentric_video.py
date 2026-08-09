"""
Schemas for egocentric video scene descriptions.

These models are the MVP extraction contract for the project route:
scene descriptions -> typed procedural knowledge -> property graph.

中文说明：
这个文件定义“第一人称专家视频场景描述”要抽取成什么结构。
LLM 的输出会被 Instructor 校验成这些 Pydantic 类，后续 graph 层再把这些类转换成节点和边。
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def normalize_scene_fields(cls, data: Any) -> Any:
        """
        把模型常用的 ``scene_id`` 对应到现有视频片段字段。

        Scene 的节点名称采用片段编号构成稳定名称，避免仅使用泛化的
        description 导致不同视频片段在图中被错误合并。
        """

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        scene_id = normalized.get("scene_id")
        segment_id = normalized.get("segment_id") or normalized.get("id")
        if segment_id and not normalized.get("segment_id"):
            normalized["segment_id"] = segment_id
        if scene_id and not normalized.get("video_id"):
            normalized["video_id"] = scene_id
        if not normalized.get("name"):
            if scene_id:
                normalized["name"] = (
                    f"{scene_id} segment {segment_id}" if segment_id else str(scene_id)
                )
            elif segment_id:
                normalized["name"] = f"segment {segment_id}"
        return normalized


class SceneObject(KGEntity):
    """A physical object observed in the scene."""

    # SceneObject 表示画面里被操作或被观察到的物体，例如 steel plate、bracket。
    object_type: Optional[str] = None
    state: Optional[str] = None
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_object_fields(cls, data: Any) -> Any:
        """当模型把对象名称写在 ``type`` 中时，保留其显式对象名称。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("name") and normalized.get("type"):
            normalized["name"] = normalized["type"]
        if not normalized.get("object_type") and normalized.get("type"):
            normalized["object_type"] = normalized["type"]
        return normalized


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

    @model_validator(mode="before")
    @classmethod
    def normalize_action_fields(cls, data: Any) -> Any:
        """将模型输出的动作别名对应为 graph 使用的名称和顺序字段。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("sequence_index") is None and normalized.get("order") is not None:
            normalized["sequence_index"] = normalized["order"]
        if not normalized.get("name") and normalized.get("action"):
            normalized["name"] = normalized["action"]
        action_verb = normalized.get("type")
        action_object = normalized.get("object")
        if not normalized.get("name") and action_verb and action_object:
            normalized["name"] = f"{action_verb} {action_object}"
        action_type_aliases = {
            "place": "manipulation",
            "measure": "measurement",
            "record": "documentation",
            "cause": "other",
        }
        if not normalized.get("action_type") and action_verb in action_type_aliases:
            normalized["action_type"] = action_type_aliases[action_verb]
        return normalized


class UsesTool(KGRelation):
    """An action uses a tool."""

    # 对应图里的边：Action --USES_TOOL--> Tool。
    action: Action
    tool: Tool
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_string_endpoints(cls, data: Any) -> Any:
        """把模型明确输出的动作名与工具名包装成 relation endpoint 实体。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["action"] = _as_named_entity(normalized.get("action"))
        normalized["tool"] = _as_named_entity(normalized.get("tool"))
        return normalized


class ActsOnObject(KGRelation):
    """An action manipulates, inspects, or otherwise acts on an object."""

    # 对应图里的边：Action --ACTS_ON--> Object，用来表达动作作用在哪个物体上。
    action: Action
    object: SceneObject
    role: Optional[Literal["input", "output", "target", "support", "unknown"]] = None
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_string_endpoints(cls, data: Any) -> Any:
        """把字符串动作与对象端点转为现有 schema 所需的实体对象。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["action"] = _as_named_entity(normalized.get("action"))
        normalized["object"] = _as_named_entity(normalized.get("object"))
        return normalized


class ActionOrder(KGRelation):
    """Temporal ordering between two actions."""

    # 对应图里的边：before Action --BEFORE--> after Action，用来表达步骤顺序。
    before: Action
    after: Action
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_order_endpoints(cls, data: Any) -> Any:
        """
        兼容模型输出的 ``action`` / ``following_action`` 顺序表示。

        若模型只给出 preceding 与当前 action，也可直接还原为 BEFORE 边；
        这里仅改写已经被模型显式表达的先后关系。
        """

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("before") or not normalized.get("after"):
            action = normalized.get("action")
            preceding = normalized.get("preceding_action")
            following = normalized.get("following_action")
            if action and following:
                normalized["before"], normalized["after"] = action, following
            elif preceding and action:
                normalized["before"], normalized["after"] = preceding, action
        normalized["before"] = _as_named_entity(normalized.get("before"))
        normalized["after"] = _as_named_entity(normalized.get("after"))
        return normalized


class ActionCauses(KGRelation):
    """A causal relation between two actions or observed states."""

    # 对应图里的边：cause Action --CAUSES--> effect Action，用来表达因果关系。
    cause: Action
    effect: Action
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_causal_endpoints(cls, data: Any) -> Any:
        """兼容 ``causing_action`` 导致 ``action`` 的因果表示。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("cause") and normalized.get("causing_action"):
            normalized["cause"] = normalized["causing_action"]
        if not normalized.get("effect") and normalized.get("action"):
            normalized["effect"] = normalized["action"]
        normalized["cause"] = _as_named_entity(normalized.get("cause"))
        normalized["effect"] = _as_named_entity(normalized.get("effect"))
        return normalized


class ActionPartOfProcedure(KGRelation):
    """An observed action belongs to a procedure."""

    # 对应图里的边：Action --PART_OF--> Procedure，用来把动作归入某个工艺流程。
    action: Action
    procedure: Procedure
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_string_endpoints(cls, data: Any) -> Any:
        """把明确给出的动作名和流程名转换为嵌套实体端点。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["action"] = _as_named_entity(normalized.get("action"))
        normalized["procedure"] = _as_named_entity(normalized.get("procedure"))
        return normalized


class ActionObservedInScene(KGRelation):
    """An action is observed in a scene segment."""

    # 对应图里的边：Action --OBSERVED_IN--> Scene，用来保留动作发生的视频片段。
    action: Action
    scene: Scene
    confidence: Optional[float] = None
    provenance: Optional[Provenance] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_string_endpoints(cls, data: Any) -> Any:
        """把明确给出的动作和场景名称转换为嵌套实体端点。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["action"] = _as_named_entity(normalized.get("action"))
        normalized["scene"] = _as_named_entity(normalized.get("scene"))
        return normalized


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

    @model_validator(mode="before")
    @classmethod
    def add_scene_context_from_root(cls, data: Any) -> Any:
        """将顶层视频编号传给只返回 segment `id` 的场景对象。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        video_id = normalized.get("video_id")
        scenes = []
        for scene in normalized.get("scenes", []):
            if not isinstance(scene, dict):
                scenes.append(scene)
                continue
            scene_data = dict(scene)
            segment_id = scene_data.get("segment_id") or scene_data.get("id")
            if segment_id and not scene_data.get("segment_id"):
                scene_data["segment_id"] = segment_id
            if video_id and not scene_data.get("video_id"):
                scene_data["video_id"] = video_id
            if video_id and segment_id and not scene_data.get("name"):
                scene_data["name"] = f"{video_id} segment {segment_id}"
            scenes.append(scene_data)
        normalized["scenes"] = scenes
        observed_relations = []
        for relation in normalized.get("observed_in_scene", []):
            if not isinstance(relation, dict):
                observed_relations.append(relation)
                continue
            relation_data = dict(relation)
            scene_endpoint = relation_data.get("scene")
            if video_id and isinstance(scene_endpoint, str):
                relation_data["scene"] = {
                    "name": f"{video_id} segment {scene_endpoint}",
                    "video_id": video_id,
                    "segment_id": scene_endpoint,
                }
            observed_relations.append(relation_data)
        normalized["observed_in_scene"] = observed_relations
        return normalized


def _as_named_entity(value: Any) -> Any:
    """将模型已经给出的端点名称字符串包装成实体对象，不创建新事实。"""

    if isinstance(value, str):
        return {"name": value}
    return value
