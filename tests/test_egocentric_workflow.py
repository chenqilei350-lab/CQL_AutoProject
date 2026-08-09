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


def test_schema_normalizes_common_local_model_field_aliases():
    """确认本地小模型的常见等价字段能够进入严格 schema。"""

    extraction = EgocentricVideoExtraction.model_validate(
        {
            "scenes": [
                {
                    "scene_id": "demo_01",
                    "segment_id": "s1",
                    "description": "assembly scene",
                }
            ],
            "actions": [
                {
                    "description": "tighten bolt",
                    "order": 2,
                    "evidence": "tightens the bolt",
                }
            ],
        }
    )

    assert extraction.scenes[0].name == "demo_01 segment s1"
    assert extraction.scenes[0].video_id == "demo_01"
    assert extraction.actions[0].name == "tighten bolt"
    assert extraction.actions[0].sequence_index == 2
    assert extraction.actions[0].evidence_text == "tightens the bolt"


def test_schema_normalizes_inspection_style_local_model_output():
    """确认模型以 id、type 和空格角色输出时仍可保留显式事实。"""

    extraction = EgocentricVideoExtraction.model_validate(
        {
            "video_id": "inspect_demo_01",
            "scenes": [{"id": "s2"}],
            "actors": [{"name": "Maria", "role": "quality inspector"}],
            "actions": [
                {
                    "id": "action1",
                    "type": "measure",
                    "object": "gap",
                    "source_text": "measures the gap",
                }
            ],
            "objects": [{"id": "object1", "type": "gap"}],
        }
    )

    assert extraction.scenes[0].name == "inspect_demo_01 segment s2"
    assert extraction.actors[0].role == "quality_inspector"
    assert extraction.actions[0].name == "measure gap"
    assert extraction.actions[0].action_type == "measurement"
    assert extraction.objects[0].name == "gap"


def test_schema_normalizes_string_relation_endpoints_from_local_model():
    """确认小模型以字符串表示显式关系端点时仍能构建类型化关系。"""

    extraction = EgocentricVideoExtraction.model_validate(
        {
            "video_id": "weld_demo_01",
            "actions": [{"action": "align steel plate"}],
            "uses_tool": [{"action": "start root weld", "tool": "torch"}],
            "acts_on_object": [{"action": "align steel plate", "object": "plate"}],
            "action_order": [
                {"action": "align steel plate", "following_action": "start root weld"}
            ],
            "action_causes": [
                {"action": "start root weld", "causing_action": "align steel plate"}
            ],
            "part_of_procedure": [{"action": "start root weld", "procedure": "welding"}],
            "observed_in_scene": [{"action": "start root weld", "scene": "s1"}],
        }
    )

    assert extraction.actions[0].name == "align steel plate"
    assert extraction.uses_tool[0].tool.name == "torch"
    assert extraction.acts_on_object[0].object.name == "plate"
    assert extraction.action_order[0].before.name == "align steel plate"
    assert extraction.action_causes[0].cause.name == "align steel plate"
    assert extraction.part_of_procedure[0].procedure.name == "welding"
    assert extraction.observed_in_scene[0].scene.name == "weld_demo_01 segment s1"


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


def test_graph_preserves_relation_provenance():
    extraction = EgocentricVideoExtraction(
        source_text="Hans uses the caliper.",
        actions=[Action(name="measure gap")],
        tools=[Tool(name="caliper")],
        uses_tool=[
            UsesTool(
                action=Action(name="measure gap"),
                tool=Tool(name="caliper"),
                provenance=Provenance(source_text="Hans uses the caliper.", confidence=0.8),
            )
        ],
    )

    graph = build_property_graph(extraction)
    edge = next(edge for edge in graph.edges if edge.type == "USES_TOOL")

    assert edge.properties["provenance"]["source_text"] == "Hans uses the caliper."
    assert edge.properties["provenance"]["confidence"] == 0.8


def test_property_graph_uses_stable_ids_for_fast_deduplication():
    graph = PropertyGraph()
    first = graph.add_node("Tool", "Fronius TPS 400i torch", {"model": "TPS 400i"})
    second = graph.add_node("Tool", "fronius   tps 400i torch", {"tool_type": "welding torch"})

    assert first.id == second.id
    assert len(graph.nodes) == 1
    assert len(graph.stable_id_index) == 1
    assert graph.nodes[first.id].properties["tool_type"] == "welding torch"


def test_property_graph_exports_cypher():
    graph = build_property_graph(WELDING_SCENE_EXPECTED)
    cypher = graph.to_cypher()

    assert "MERGE (n:Action" in cypher
    assert "MERGE (source)-[r:USES_TOOL" in cypher
    assert "Fronius TPS 400i torch" in cypher


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
