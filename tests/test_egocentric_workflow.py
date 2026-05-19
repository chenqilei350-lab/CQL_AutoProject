from backend.graph.property_graph import PropertyGraph, build_property_graph
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    WELDING_SCENE_EXPECTED,
)
from backend.schemas.egocentric_video import (
    Action,
    EgocentricVideoExtraction,
    Provenance,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


# 测试 1：确认 egocentric video schema 可以正常创建，并且 provenance 等字段能保存。
def test_egocentric_schema_validation():
    extraction = EgocentricVideoExtraction(
        video_id="demo_01",
        actions=[
            Action(
                name="tighten screw",
                action_type="assembly",
                sequence_index=1,
                provenance=Provenance(video_id="demo_01", segment_id="s1", confidence=0.8),
            )
        ],
        tools=[Tool(name="torque wrench", tool_type="wrench")],
        objects=[SceneObject(name="screw", object_type="fastener")],
        uses_tool=[
            UsesTool(action=Action(name="tighten screw"), tool=Tool(name="torque wrench"))
        ],
    )

    assert extraction.video_id == "demo_01"
    assert extraction.actions[0].provenance.video_id == "demo_01"
    assert extraction.uses_tool[0].tool.name == "torque wrench"


# 测试 2：确认 gold example 可以转换成 graph，并且基础查询能返回预期结果。
def test_property_graph_builds_nodes_edges_and_queries():
    graph = build_property_graph(WELDING_SCENE_EXPECTED)

    assert isinstance(graph, PropertyGraph)
    assert graph.find_node("Action", "align steel plate") is not None
    assert graph.find_node("Tool", "Fronius TPS 400i torch") is not None
    assert any(edge.type == "USES_TOOL" for edge in graph.edges)
    assert any(edge.type == "BEFORE" for edge in graph.edges)
    assert any(edge.type == "CAUSES" for edge in graph.edges)

    tools = graph.tools_for_action("start root weld")
    assert [tool.name for tool in tools] == ["Fronius TPS 400i torch"]

    actions = graph.actions_using_tool("TPS 400i")
    assert {action.name for action in actions} == {"pick up welding torch", "start root weld"}

    ordered = [action.name for action in graph.ordered_actions()]
    assert ordered[:3] == ["pick up welding torch", "align steel plate", "start root weld"]

    effects = graph.effects_caused_by("align plate")
    assert [effect.name for effect in effects] == ["start root weld"]


# 测试 3：确认重复工具节点会被合并，而不是生成两个几乎一样的 Tool 节点。
def test_property_graph_merges_repeated_nodes():
    extraction = WELDING_SCENE_EXPECTED.model_copy(deep=True)
    extraction.tools.append(Tool(name="fronius   tps 400i torch", tool_type="welding torch"))

    graph = build_property_graph(extraction)
    tool_nodes = [node for node in graph.nodes.values() if node.label == "Tool"]

    assert len(tool_nodes) == 1
    assert tool_nodes[0].properties["tool_type"] == "welding torch"


# 测试 4：确认 provenance 和关系属性没有在 graph construction 过程中丢失。
def test_graph_preserves_provenance_and_relation_properties():
    graph = build_property_graph(WELDING_SCENE_EXPECTED)
    action = graph.find_node("Action", "pick up welding torch")
    assert action is not None
    assert action.properties["provenance"]["video_id"] == "weld_demo_01"
    assert action.properties["provenance"]["segment_id"] == "s1"

    cause_edges = [edge for edge in graph.edges if edge.type == "CAUSES"]
    assert cause_edges
    assert cause_edges[0].properties["rationale"] == "The alignment step prepares the plate for the root weld."


# 测试 5：确认第二个质量检查 example 也能覆盖 measurement/documentation workflow。
def test_second_gold_example_covers_measurement_workflow():
    graph = build_property_graph(INSPECTION_SCENE_EXPECTED)

    assert graph.tools_for_action("measure gap")[0].name == "caliper"
    assert [effect.name for effect in graph.effects_caused_by("measure gap")] == ["record result"]
    assert [action.name for action in graph.ordered_actions()] == [
        "place caliper on bracket",
        "measure gap",
        "record result",
    ]
