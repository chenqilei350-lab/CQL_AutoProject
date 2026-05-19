"""
In-memory property graph for MVP graph construction and querying.

The project can later swap this layer for LadybugDB or Neo4j/Cypher export.

中文说明：
这个文件实现一个轻量的“内存版知识图谱”。
它不依赖 Neo4j/LadybugDB，主要用于 MVP 阶段验证：
1. Pydantic 抽取结果能不能变成 graph nodes/edges。
2. 重复节点能不能合并。
3. 能不能查询动作顺序、工具使用和因果关系。
"""

from typing import Any, Iterable, Optional
from pydantic import BaseModel, Field

from backend.evaluation.matching import MatchStrategy, compare_values
from backend.schemas.egocentric_video import (
    Action,
    ActionCauses,
    ActionObservedInScene,
    ActionOrder,
    ActionPartOfProcedure,
    ActsOnObject,
    EgocentricVideoExtraction,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Procedure, Tool, Worker


class GraphNode(BaseModel):
    """A property graph node."""

    # 图节点，例如 Action、Tool、Object、Scene、Procedure。
    id: str
    label: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A property graph edge."""

    # 图边，例如 USES_TOOL、ACTS_ON、BEFORE、CAUSES。
    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)


def normalize_name(value: str) -> str:
    """Normalize node names for deterministic IDs and exact merge checks."""
    # 标准化名称，避免大小写和多余空格导致同一个节点被重复创建。
    return " ".join(value.lower().strip().split())


def model_properties(model: BaseModel, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Convert a Pydantic model to JSON-friendly graph properties."""
    # 把 Pydantic 对象转成可以挂在 graph node/edge 上的属性字典。
    excluded = {"name", *exclude}
    return model.model_dump(mode="json", exclude=excluded, exclude_none=True)


class PropertyGraph(BaseModel):
    """A small in-memory property graph with simple deduplication and query helpers."""

    # nodes 用 dict 存，方便通过 node id 快速找到节点；edges 用 list 保存所有关系。
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)

    def add_node(
        self,
        label: str,
        name: str,
        properties: Optional[dict[str, Any]] = None,
        merge_strategy: MatchStrategy = MatchStrategy.NORMALIZED,
        match_threshold: float = 0.85,
    ) -> GraphNode:
        """Add a node or merge it with an existing node of the same label."""
        # 先查找是否已有同类同名节点；如果有，就更新属性而不是新建重复节点。
        properties = properties or {}
        existing = self.find_node(label, name, merge_strategy, match_threshold)
        if existing:
            existing.properties.update({k: v for k, v in properties.items() if v is not None})
            return existing

        node_id = self._stable_node_id(label, name)
        node = GraphNode(id=node_id, label=label, name=name, properties=properties)
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        edge_type: str,
        source: GraphNode,
        target: GraphNode,
        properties: Optional[dict[str, Any]] = None,
    ) -> GraphEdge:
        """Add an edge if an equivalent source/type/target edge does not already exist."""
        # 同一个 source/type/target 的边只保留一条，重复添加时合并属性。
        properties = properties or {}
        for edge in self.edges:
            if edge.type == edge_type and edge.source == source.id and edge.target == target.id:
                edge.properties.update({k: v for k, v in properties.items() if v is not None})
                return edge

        edge_id = f"e{len(self.edges) + 1}"
        edge = GraphEdge(
            id=edge_id,
            type=edge_type,
            source=source.id,
            target=target.id,
            properties=properties,
        )
        self.edges.append(edge)
        return edge

    def find_node(
        self,
        label: str,
        name: str,
        strategy: MatchStrategy = MatchStrategy.NORMALIZED,
        threshold: float = 0.85,
    ) -> Optional[GraphNode]:
        """Find the best matching node for a label/name pair."""
        # 根据 label 限定节点类型，再用匹配策略比较名称，返回分数最高的节点。
        best: Optional[GraphNode] = None
        best_score = 0.0
        for node in self.nodes.values():
            if node.label != label:
                continue
            comparison = compare_values(name, node.name, strategy=strategy, threshold=threshold)
            if comparison["score"] > best_score:
                best = node
                best_score = comparison["score"]
        return best if best_score >= threshold else None

    def outgoing(self, node: GraphNode, edge_type: Optional[str] = None) -> list[GraphEdge]:
        """Return outgoing edges for a node, optionally filtered by type."""
        # 查询某个节点发出的边，例如 Action -> Tool。
        return [
            edge for edge in self.edges
            if edge.source == node.id and (edge_type is None or edge.type == edge_type)
        ]

    def incoming(self, node: GraphNode, edge_type: Optional[str] = None) -> list[GraphEdge]:
        """Return incoming edges for a node, optionally filtered by type."""
        # 查询某个节点收到的边，例如 Tool <- Action。
        return [
            edge for edge in self.edges
            if edge.target == node.id and (edge_type is None or edge.type == edge_type)
        ]

    def node(self, node_id: str) -> GraphNode:
        """Return a node by ID."""
        # 通过节点 id 取回完整节点对象。
        return self.nodes[node_id]

    def tools_for_action(self, action_name: str) -> list[GraphNode]:
        """Return tools used by the named action."""
        # 用来回答问题：“某个动作使用了哪些工具？”
        action = self.find_node("Action", action_name, MatchStrategy.TOKEN_OVERLAP, 0.6)
        if not action:
            return []
        return [self.node(edge.target) for edge in self.outgoing(action, "USES_TOOL")]

    def actions_using_tool(self, tool_name: str) -> list[GraphNode]:
        """Return actions that use the named tool."""
        # 用来回答问题：“某个工具被哪些动作使用？”
        tool = self.find_node("Tool", tool_name, MatchStrategy.TOKEN_OVERLAP, 0.6)
        if not tool:
            return []
        return [self.node(edge.source) for edge in self.incoming(tool, "USES_TOOL")]

    def ordered_actions(self) -> list[GraphNode]:
        """Return actions sorted by sequence index when available."""
        # 用 sequence_index 排序，得到工艺步骤/动作顺序。
        actions = [node for node in self.nodes.values() if node.label == "Action"]
        return sorted(actions, key=lambda n: n.properties.get("sequence_index", 10**9))

    def effects_caused_by(self, action_name: str) -> list[GraphNode]:
        """Return effect actions caused by the named action."""
        # 用来回答问题：“某个动作导致了哪些后续动作/结果？”
        action = self.find_node("Action", action_name, MatchStrategy.TOKEN_OVERLAP, 0.6)
        if not action:
            return []
        return [self.node(edge.target) for edge in self.outgoing(action, "CAUSES")]

    def _stable_node_id(self, label: str, name: str) -> str:
        # 用 label 和标准化名称生成稳定 id；如果冲突，就加数字后缀。
        base = f"{label}:{normalize_name(name)}"
        candidate = base
        suffix = 2
        while candidate in self.nodes:
            candidate = f"{base}:{suffix}"
            suffix += 1
        return candidate


def build_property_graph(extraction: EgocentricVideoExtraction) -> PropertyGraph:
    """Build a property graph from an egocentric video extraction result."""
    # 这是 graph construction 的主入口：先加所有实体节点，再加所有关系边。
    graph = PropertyGraph()

    for scene in extraction.scenes:
        _add_scene(graph, scene)
    for procedure in extraction.procedures:
        _add_procedure(graph, procedure)
    for actor in extraction.actors:
        _add_actor(graph, actor)
    for action in extraction.actions:
        _add_action(graph, action)
    for tool in extraction.tools:
        _add_tool(graph, tool)
    for obj in extraction.objects:
        _add_object(graph, obj)

    for relation in extraction.uses_tool:
        _add_uses_tool(graph, relation)
    for relation in extraction.acts_on_object:
        _add_acts_on_object(graph, relation)
    for relation in extraction.action_order:
        _add_action_order(graph, relation)
    for relation in extraction.action_causes:
        _add_action_causes(graph, relation)
    for relation in extraction.part_of_procedure:
        _add_part_of_procedure(graph, relation)
    for relation in extraction.observed_in_scene:
        _add_observed_in_scene(graph, relation)

    return graph


def _add_scene(graph: PropertyGraph, scene: Scene) -> GraphNode:
    # 把 Scene schema 对象转成 Scene 节点。
    return graph.add_node("Scene", scene.name, model_properties(scene))


def _add_procedure(graph: PropertyGraph, procedure: Procedure) -> GraphNode:
    # 把 Procedure schema 对象转成 Procedure 节点。
    return graph.add_node("Procedure", procedure.name, model_properties(procedure))


def _add_actor(graph: PropertyGraph, actor: Worker) -> GraphNode:
    # Worker 在视频场景中作为 Actor 节点保存。
    return graph.add_node("Actor", actor.name, model_properties(actor))


def _add_action(graph: PropertyGraph, action: Action) -> GraphNode:
    # 把 Action schema 对象转成 Action 节点，并自动连接 actor 和 scene。
    properties = model_properties(action, exclude={"actor", "scene", "parameters", "provenance"})
    if action.provenance:
        properties["provenance"] = action.provenance.model_dump(mode="json", exclude_none=True)
    node = graph.add_node("Action", action.name, properties)
    if action.actor:
        graph.add_edge("PERFORMED_BY", node, _add_actor(graph, action.actor))
    if action.scene:
        graph.add_edge("OBSERVED_IN", node, _add_scene(graph, action.scene))
    return node


def _add_tool(graph: PropertyGraph, tool: Tool) -> GraphNode:
    # 把 Tool schema 对象转成 Tool 节点。
    return graph.add_node("Tool", tool.name, model_properties(tool))


def _add_object(graph: PropertyGraph, obj: SceneObject) -> GraphNode:
    # 把 SceneObject schema 对象转成 Object 节点。
    properties = model_properties(obj, exclude={"provenance"})
    if obj.provenance:
        properties["provenance"] = obj.provenance.model_dump(mode="json", exclude_none=True)
    return graph.add_node("Object", obj.name, properties)


def _relation_properties(relation: BaseModel, exclude: Iterable[str]) -> dict[str, Any]:
    # 把关系对象上的非端点字段保留下来，例如 confidence、rationale、role。
    return model_properties(relation, exclude=exclude)


def _add_uses_tool(graph: PropertyGraph, relation: UsesTool) -> None:
    # 创建 Action --USES_TOOL--> Tool。
    graph.add_edge(
        "USES_TOOL",
        _add_action(graph, relation.action),
        _add_tool(graph, relation.tool),
        _relation_properties(relation, {"action", "tool", "provenance"}),
    )


def _add_acts_on_object(graph: PropertyGraph, relation: ActsOnObject) -> None:
    # 创建 Action --ACTS_ON--> Object。
    graph.add_edge(
        "ACTS_ON",
        _add_action(graph, relation.action),
        _add_object(graph, relation.object),
        _relation_properties(relation, {"action", "object", "provenance"}),
    )


def _add_action_order(graph: PropertyGraph, relation: ActionOrder) -> None:
    # 创建 before Action --BEFORE--> after Action。
    graph.add_edge(
        "BEFORE",
        _add_action(graph, relation.before),
        _add_action(graph, relation.after),
        _relation_properties(relation, {"before", "after", "provenance"}),
    )


def _add_action_causes(graph: PropertyGraph, relation: ActionCauses) -> None:
    # 创建 cause Action --CAUSES--> effect Action。
    graph.add_edge(
        "CAUSES",
        _add_action(graph, relation.cause),
        _add_action(graph, relation.effect),
        _relation_properties(relation, {"cause", "effect", "provenance"}),
    )


def _add_part_of_procedure(graph: PropertyGraph, relation: ActionPartOfProcedure) -> None:
    # 创建 Action --PART_OF--> Procedure。
    graph.add_edge(
        "PART_OF",
        _add_action(graph, relation.action),
        _add_procedure(graph, relation.procedure),
        _relation_properties(relation, {"action", "procedure", "provenance"}),
    )


def _add_observed_in_scene(graph: PropertyGraph, relation: ActionObservedInScene) -> None:
    # 创建 Action --OBSERVED_IN--> Scene。
    graph.add_edge(
        "OBSERVED_IN",
        _add_action(graph, relation.action),
        _add_scene(graph, relation.scene),
        _relation_properties(relation, {"action", "scene", "provenance"}),
    )
