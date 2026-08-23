"""Human-authored egocentric-video examples for expanded experiments.

Three seed scenes extend welding and caliper inspection with assembly,
maintenance cleaning, and temperature inspection. These are verifiable MVP scene
descriptions rather than real video transcripts.
"""

from backend.schemas.egocentric_video import (
    Action,
    ActionCauses,
    ActionOrder,
    ActsOnObject,
    EgocentricVideoExtraction,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import ProcessParameter, Tool, Worker


# 示例 3：装配场景，覆盖工具紧固操作、对象操作和明确的步骤依赖。
ASSEMBLY_SCENE_TEXT = """
Video assembly_demo_01, segment s3, 00:00-00:30. Operator Lena picks up the
torque wrench, positions the cover plate on the housing, and tightens the bolt
with the torque wrench. Positioning the cover plate enables tightening the bolt.
"""

ASSEMBLY_SCENE_EXPECTED = EgocentricVideoExtraction(
    video_id="assembly_demo_01",
    source_text=ASSEMBLY_SCENE_TEXT,
    scenes=[
        Scene(
            name="assembly_demo_01 segment s3",
            video_id="assembly_demo_01",
            segment_id="s3",
            timestamp_start_seconds=0.0,
            timestamp_end_seconds=30.0,
        )
    ],
    actors=[Worker(name="Lena", role="operator")],
    actions=[
        Action(name="pick up torque wrench", action_type="preparation", sequence_index=1),
        Action(name="position cover plate", action_type="assembly", sequence_index=2),
        Action(name="tighten bolt", action_type="assembly", sequence_index=3),
    ],
    tools=[Tool(name="torque wrench", tool_type="assembly tool")],
    objects=[
        SceneObject(name="cover plate", object_type="component"),
        SceneObject(name="bolt", object_type="fastener"),
    ],
    uses_tool=[
        UsesTool(action=Action(name="tighten bolt"), tool=Tool(name="torque wrench"))
    ],
    acts_on_object=[
        ActsOnObject(
            action=Action(name="position cover plate"),
            object=SceneObject(name="cover plate"),
            role="target",
        ),
        ActsOnObject(
            action=Action(name="tighten bolt"),
            object=SceneObject(name="bolt"),
            role="target",
        ),
    ],
    action_order=[
        ActionOrder(
            before=Action(name="pick up torque wrench"),
            after=Action(name="position cover plate"),
        ),
        ActionOrder(
            before=Action(name="position cover plate"),
            after=Action(name="tighten bolt"),
        ),
    ],
    action_causes=[
        ActionCauses(
            cause=Action(name="position cover plate"),
            effect=Action(name="tighten bolt"),
            rationale="Positioning the cover plate enables tightening the bolt.",
        )
    ],
)


# 示例 4：维护清洁场景，覆盖停机、清洁工具以及安全先后关系。
MAINTENANCE_SCENE_TEXT = """
Video maintenance_demo_01, segment s4, 00:30-00:55. Technician Omar switches
off the machine, wipes the sensor lens with a cleaning cloth, and inspects the
clean lens. Switching off the machine allows safe cleaning.
"""

MAINTENANCE_SCENE_EXPECTED = EgocentricVideoExtraction(
    video_id="maintenance_demo_01",
    source_text=MAINTENANCE_SCENE_TEXT,
    scenes=[
        Scene(
            name="maintenance_demo_01 segment s4",
            video_id="maintenance_demo_01",
            segment_id="s4",
            timestamp_start_seconds=30.0,
            timestamp_end_seconds=55.0,
        )
    ],
    actors=[Worker(name="Omar", role="maintenance")],
    actions=[
        Action(name="switch off machine", action_type="safety_check", sequence_index=1),
        Action(name="wipe sensor lens", action_type="cleaning", sequence_index=2),
        Action(name="inspect clean lens", action_type="inspection", sequence_index=3),
    ],
    tools=[Tool(name="cleaning cloth", tool_type="cleaning tool")],
    objects=[
        SceneObject(name="machine", object_type="equipment"),
        SceneObject(name="sensor lens", object_type="component"),
    ],
    uses_tool=[
        UsesTool(
            action=Action(name="wipe sensor lens"),
            tool=Tool(name="cleaning cloth"),
        )
    ],
    acts_on_object=[
        ActsOnObject(
            action=Action(name="switch off machine"),
            object=SceneObject(name="machine"),
            role="target",
        ),
        ActsOnObject(
            action=Action(name="wipe sensor lens"),
            object=SceneObject(name="sensor lens"),
            role="target",
        ),
    ],
    action_order=[
        ActionOrder(
            before=Action(name="switch off machine"),
            after=Action(name="wipe sensor lens"),
        ),
        ActionOrder(
            before=Action(name="wipe sensor lens"),
            after=Action(name="inspect clean lens"),
        ),
    ],
    action_causes=[
        ActionCauses(
            cause=Action(name="switch off machine"),
            effect=Action(name="wipe sensor lens"),
            rationale="Switching off the machine allows safe cleaning.",
        )
    ],
)


# 示例 5：温度检查场景，覆盖测量参数与记录动作。
TEMPERATURE_SCENE_TEXT = """
Video thermal_demo_01, segment s5, 01:00-01:20. Quality inspector Noor points
the infrared thermometer at the coated panel, reads a temperature of 42 degrees
Celsius, and records the value. Reading the temperature enables recording the value.
"""

TEMPERATURE_SCENE_EXPECTED = EgocentricVideoExtraction(
    video_id="thermal_demo_01",
    source_text=TEMPERATURE_SCENE_TEXT,
    scenes=[
        Scene(
            name="thermal_demo_01 segment s5",
            video_id="thermal_demo_01",
            segment_id="s5",
            timestamp_start_seconds=60.0,
            timestamp_end_seconds=80.0,
        )
    ],
    actors=[Worker(name="Noor", role="quality_inspector")],
    actions=[
        Action(
            name="point infrared thermometer",
            action_type="inspection",
            sequence_index=1,
        ),
        Action(
            name="read temperature",
            action_type="measurement",
            sequence_index=2,
        ),
        Action(name="record value", action_type="documentation", sequence_index=3),
    ],
    tools=[Tool(name="infrared thermometer", tool_type="measurement tool")],
    objects=[SceneObject(name="coated panel", object_type="workpiece")],
    parameters=[
        ProcessParameter(
            name="temperature",
            parameter_type="temperature",
            unit="degrees Celsius",
            nominal_value=42.0,
        )
    ],
    uses_tool=[
        UsesTool(
            action=Action(name="point infrared thermometer"),
            tool=Tool(name="infrared thermometer"),
        ),
        UsesTool(
            action=Action(name="read temperature"),
            tool=Tool(name="infrared thermometer"),
        ),
    ],
    acts_on_object=[
        ActsOnObject(
            action=Action(name="point infrared thermometer"),
            object=SceneObject(name="coated panel"),
            role="target",
        )
    ],
    action_order=[
        ActionOrder(
            before=Action(name="point infrared thermometer"),
            after=Action(name="read temperature"),
        ),
        ActionOrder(
            before=Action(name="read temperature"),
            after=Action(name="record value"),
        ),
    ],
    action_causes=[
        ActionCauses(
            cause=Action(name="read temperature"),
            effect=Action(name="record value"),
            rationale="Reading the temperature enables recording the value.",
        )
    ],
)


# 单独暴露扩展部分，基础两场景继续留在原有 examples 文件中。
ADDITIONAL_EGOCENTRIC_EXAMPLES = [
    ASSEMBLY_SCENE_EXPECTED,
    MAINTENANCE_SCENE_EXPECTED,
    TEMPERATURE_SCENE_EXPECTED,
]
