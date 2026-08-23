"""
Gold-standard examples for egocentric video procedural knowledge extraction.

These hand-authored expected results do not call an LLM. They verify schemas,
graph construction, and downstream evaluation.
"""

from backend.schemas.egocentric_video import (
    Action,
    ActionCauses,
    ActionObservedInScene,
    ActionOrder,
    ActionPartOfProcedure,
    ActsOnObject,
    EgocentricVideoExtraction,
    Provenance,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Procedure, Tool, Worker


# 示例 1：焊接场景。原始输入先用文字描述视频片段，MVP 阶段暂不直接读取视频帧。
WELDING_SCENE_TEXT = """
Video weld_demo_01, segment s1, 00:00-00:20. From the worker's perspective,
Hans picks up the Fronius TPS 400i torch, aligns the steel plate on the table,
and then starts the root weld. The alignment step prepares the plate for the
root weld.
"""

# 这是示例 1 的标准结构化结果：包含 scene、actor、actions、tools、objects 和关系。
WELDING_SCENE_EXPECTED = EgocentricVideoExtraction(
    video_id="weld_demo_01",
    source_text=WELDING_SCENE_TEXT,
    scenes=[
        Scene(
            name="weld_demo_01 segment s1",
            video_id="weld_demo_01",
            segment_id="s1",
            timestamp_start_seconds=0.0,
            timestamp_end_seconds=20.0,
            location="welding table",
            source_text="Video weld_demo_01, segment s1, 00:00-00:20",
            confidence=1.0,
        )
    ],
    procedures=[
        Procedure(name="Root welding preparation", domain="welding")
    ],
    actors=[
        Worker(name="Hans", role="operator")
    ],
    actions=[
        Action(
            name="pick up welding torch",
            action_type="preparation",
            actor=Worker(name="Hans", role="operator"),
            sequence_index=1,
            timestamp_start_seconds=0.0,
            timestamp_end_seconds=5.0,
            source_text="Hans picks up the Fronius TPS 400i torch",
            provenance=Provenance(video_id="weld_demo_01", segment_id="s1", source_text="Hans picks up the Fronius TPS 400i torch", confidence=0.95),
            confidence=0.95,
        ),
        Action(
            name="align steel plate",
            action_type="manipulation",
            actor=Worker(name="Hans", role="operator"),
            sequence_index=2,
            timestamp_start_seconds=5.0,
            timestamp_end_seconds=12.0,
            source_text="aligns the steel plate on the table",
            provenance=Provenance(video_id="weld_demo_01", segment_id="s1", source_text="aligns the steel plate on the table", confidence=0.9),
            confidence=0.9,
        ),
        Action(
            name="start root weld",
            action_type="manipulation",
            actor=Worker(name="Hans", role="operator"),
            sequence_index=3,
            timestamp_start_seconds=12.0,
            timestamp_end_seconds=20.0,
            source_text="starts the root weld",
            provenance=Provenance(video_id="weld_demo_01", segment_id="s1", source_text="starts the root weld", confidence=0.9),
            confidence=0.9,
        ),
    ],
    tools=[
        Tool(name="Fronius TPS 400i torch", tool_type="welding torch", manufacturer="Fronius", model="TPS 400i")
    ],
    objects=[
        SceneObject(name="steel plate", object_type="workpiece", state="unaligned")
    ],
    uses_tool=[
        UsesTool(
            action=Action(name="pick up welding torch"),
            tool=Tool(name="Fronius TPS 400i torch"),
            confidence=0.95,
        ),
        UsesTool(
            action=Action(name="start root weld"),
            tool=Tool(name="Fronius TPS 400i torch"),
            confidence=0.9,
        ),
    ],
    acts_on_object=[
        ActsOnObject(
            action=Action(name="align steel plate"),
            object=SceneObject(name="steel plate"),
            role="target",
            confidence=0.9,
        )
    ],
    action_order=[
        ActionOrder(before=Action(name="pick up welding torch"), after=Action(name="align steel plate"), confidence=0.9),
        ActionOrder(before=Action(name="align steel plate"), after=Action(name="start root weld"), confidence=0.9),
    ],
    action_causes=[
        ActionCauses(
            cause=Action(name="align steel plate"),
            effect=Action(name="start root weld"),
            rationale="The alignment step prepares the plate for the root weld.",
            confidence=0.8,
        )
    ],
    part_of_procedure=[
        ActionPartOfProcedure(action=Action(name="align steel plate"), procedure=Procedure(name="Root welding preparation")),
        ActionPartOfProcedure(action=Action(name="start root weld"), procedure=Procedure(name="Root welding preparation")),
    ],
    observed_in_scene=[
        ActionObservedInScene(action=Action(name="pick up welding torch"), scene=Scene(name="weld_demo_01 segment s1")),
        ActionObservedInScene(action=Action(name="align steel plate"), scene=Scene(name="weld_demo_01 segment s1")),
        ActionObservedInScene(action=Action(name="start root weld"), scene=Scene(name="weld_demo_01 segment s1")),
    ],
)


# 示例 2：质量检查场景，用来覆盖 measurement/documentation 这类动作。
INSPECTION_SCENE_TEXT = """
Video inspect_demo_01, segment s2, 00:20-00:45. The quality inspector Maria
places the caliper on the bracket, measures the gap, and records the result.
The measurement causes the documentation step.
"""

# 这是示例 2 的标准结构化结果，后续测试会把它转换成 property graph。
INSPECTION_SCENE_EXPECTED = EgocentricVideoExtraction(
    video_id="inspect_demo_01",
    source_text=INSPECTION_SCENE_TEXT,
    scenes=[
        Scene(name="inspect_demo_01 segment s2", video_id="inspect_demo_01", segment_id="s2", timestamp_start_seconds=20.0, timestamp_end_seconds=45.0)
    ],
    actors=[
        Worker(name="Maria", role="quality_inspector")
    ],
    actions=[
        Action(name="place caliper on bracket", action_type="measurement", actor=Worker(name="Maria", role="quality_inspector"), sequence_index=1),
        Action(name="measure gap", action_type="measurement", actor=Worker(name="Maria", role="quality_inspector"), sequence_index=2),
        Action(name="record result", action_type="documentation", actor=Worker(name="Maria", role="quality_inspector"), sequence_index=3),
    ],
    tools=[
        Tool(name="caliper", tool_type="measurement tool")
    ],
    objects=[
        SceneObject(name="bracket", object_type="workpiece"),
        SceneObject(name="gap", object_type="measurement target"),
    ],
    uses_tool=[
        UsesTool(action=Action(name="measure gap"), tool=Tool(name="caliper"), confidence=0.9)
    ],
    acts_on_object=[
        ActsOnObject(action=Action(name="place caliper on bracket"), object=SceneObject(name="bracket"), role="target"),
        ActsOnObject(action=Action(name="measure gap"), object=SceneObject(name="gap"), role="target"),
    ],
    action_order=[
        ActionOrder(before=Action(name="place caliper on bracket"), after=Action(name="measure gap")),
        ActionOrder(before=Action(name="measure gap"), after=Action(name="record result")),
    ],
    action_causes=[
        ActionCauses(cause=Action(name="measure gap"), effect=Action(name="record result"), rationale="The measurement causes the documentation step.")
    ],
)


# 统一收集所有 examples，方便以后批量跑 benchmark 或 few-shot prompt。
ALL_EGOCENTRIC_EXAMPLES = [
    WELDING_SCENE_EXPECTED,
    INSPECTION_SCENE_EXPECTED,
]
