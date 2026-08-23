"""
In-memory property graph for MVP graph construction and querying.

The project can later swap this layer for LadybugDB or Neo4j/Cypher export.
"""

from typing import Any, Iterable, Optional
from pydantic import BaseModel, Field

from backend.evaluation.matching import MatchStrategy, compare_values
from backend.schemas.ontology import stable_node_id
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
from backend.schemas.process_knowledge.entities import (
    ProcessParameter,
    Procedure,
    Tool,
    Worker,
)


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
    stable_id_index: dict[str, str] = Field(default_factory=dict)

    def add_node(
        self,
        label: str,
        name: str,
        properties: Optional[dict[str, Any]] = None,
        merge_strategy: MatchStrategy = MatchStrategy.NORMALIZED,
        match_threshold: float = 0.85,
    ) -> GraphNode:
        """Add a node or merge it with an existing node of the same label."""
        # 先用 stable id 做 O(1) 查重，再回退到名称匹配。
        properties = properties or {}
        stable_id = stable_node_id(label, name, properties)
        indexed_id = self.stable_id_index.get(stable_id)
        if indexed_id and indexed_id in self.nodes:
            existing = self.nodes[indexed_id]
            existing.properties.update({k: v for k, v in properties.items() if v is not None})
            return existing

        # Human-reviewed and source adapters may provide stable entity IDs for
        # repeated actions with the same label at different times.  Preserve
        # those occurrences instead of collapsing them by normalized name.
        if not properties.get("entity_id"):
            existing = self.find_node(label, name, merge_strategy, match_threshold)
            if existing:
                existing.properties.update(
                    {k: v for k, v in properties.items() if v is not None}
                )
                self.stable_id_index[stable_id] = existing.id
                return existing

        node_id = self._unique_node_id(stable_id)
        node = GraphNode(id=node_id, label=label, name=name, properties=properties)
        self.nodes[node_id] = node
        self.stable_id_index[stable_id] = node_id
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

    def to_cypher(self) -> str:
        """Export the in-memory graph as a Cypher script for Neo4j/LadybugDB-style review."""
        lines = [
            "// Cypher script generated from AUT KG Extraction Pipeline",
            "// Nodes",
        ]
        for node in self.nodes.values():
            props = {"id": node.id, "name": node.name, **node.properties}
            lines.append(
                f"MERGE (n:{_cypher_identifier(node.label)} {{id: {_cypher_value(node.id)}}}) "
                f"SET {_cypher_props('n', props)}"
            )

        lines.append("")
        lines.append("// Relationships")
        for edge in self.edges:
            props = {"id": edge.id, **edge.properties}
            lines.extend(
                [
                    (
                        f"MATCH (source {{id: {_cypher_value(edge.source)}}}), "
                        f"(target {{id: {_cypher_value(edge.target)}}})"
                    ),
                    (
                        f"MERGE (source)-[r:{_cypher_identifier(edge.type)} "
                        f"{{id: {_cypher_value(edge.id)}}}]->(target) "
                        f"SET {_cypher_props('r', props)}"
                    ),
                ]
            )
        return "\n".join(lines)

    def _unique_node_id(self, base: str) -> str:
        # stable id 是首选；如果上下文真的冲突，就加数字后缀。
        candidate = base
        suffix = 2
        while candidate in self.nodes:
            candidate = f"{base}:{suffix}"
            suffix += 1
        return candidate


def _cypher_identifier(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value) or "Node"


def _cypher_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _cypher_props(var_name: str, props: dict[str, Any]) -> str:
    assignments = [
        f"{var_name}.{_cypher_identifier(key)} = {_cypher_value(value)}"
        for key, value in props.items()
        if value is not None and not isinstance(value, list | dict)
    ]
    return ", ".join(assignments) if assignments else f"{var_name}.id = {var_name}.id"


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
    for parameter in extraction.parameters:
        _add_parameter(graph, parameter)

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


def _add_parameter(graph: PropertyGraph, parameter: ProcessParameter) -> GraphNode:
    # 把独立测量参数转成节点，例如 temperature = 42 degrees Celsius。
    # 当前 schema 尚无 Action 与 Parameter 的专用关系，因此先保留参数节点和属性。
    return graph.add_node("ProcessParameter", parameter.name, model_properties(parameter))


def _relation_properties(relation: BaseModel, exclude: Iterable[str]) -> dict[str, Any]:
    # 把关系对象上的非端点字段保留下来，例如 confidence、rationale、role。
    properties = model_properties(relation, exclude=exclude)
    provenance = getattr(relation, "provenance", None)
    if provenance:
        properties["provenance"] = provenance.model_dump(mode="json", exclude_none=True)
    return properties


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
