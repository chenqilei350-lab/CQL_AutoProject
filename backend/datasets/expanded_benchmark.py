"""Five-scene raw/unified pilot benchmark.

The module extends the two seed scenes with assembly, maintenance cleaning, and
temperature inspection while preserving the original ``MVP_BENCHMARK``.
"""

from backend.datasets.benchmark import BenchmarkDataset, BenchmarkScene, MVP_BENCHMARK
from backend.preprocessing.unified_text import GroundedEntry, build_unified_text
from backend.schemas.expanded_egocentric_examples import (
    ASSEMBLY_SCENE_EXPECTED,
    ASSEMBLY_SCENE_TEXT,
    MAINTENANCE_SCENE_EXPECTED,
    MAINTENANCE_SCENE_TEXT,
    TEMPERATURE_SCENE_EXPECTED,
    TEMPERATURE_SCENE_TEXT,
)


def build_expanded_benchmark() -> BenchmarkDataset:
    """Build five scenes with source evidence and human-authored Gold graphs."""

    # 装配场景：检验模型是否能识别工具紧固、步骤顺序与前置关系。
    assembly_unified = build_unified_text(
        raw_text=ASSEMBLY_SCENE_TEXT,
        scene_id="assembly_demo_01",
        segment_id="s3",
        timestamp="00:00-00:30",
        scene_segment="cover plate assembly",
        actors=[
            GroundedEntry(
                text="Operator Lena",
                evidence="Operator Lena",
                entry_type="role",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="pick up torque wrench",
                evidence="picks up the\n torque wrench",
                entry_type="action",
                verb="pick up",
                direct_object="torque wrench",
            ),
            GroundedEntry(
                text="position cover plate on housing",
                evidence="positions the cover plate on the housing",
                entry_type="action",
                verb="position",
                direct_object="cover plate",
            ),
            GroundedEntry(
                text="tighten bolt with torque wrench",
                evidence="tightens the bolt\nwith the torque wrench",
                entry_type="action",
                verb="tighten",
                direct_object="bolt",
                tool="torque wrench",
            ),
        ],
        tools_objects=[
            GroundedEntry(
                text="torque wrench",
                evidence="torque wrench",
                entry_type="tool",
            ),
            GroundedEntry(
                text="cover plate",
                evidence="cover plate",
                entry_type="object",
            ),
            GroundedEntry(text="bolt", evidence="bolt", entry_type="object"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="positioning enables tightening",
                evidence="Positioning the cover plate enables tightening the bolt",
            )
        ],
        evidence_uncertainty=[
            "The scene text directly states the assembly order and prerequisite relation."
        ],
    )

    # 维护场景：检验模型是否保留停机后再清洁的安全顺序。
    maintenance_unified = build_unified_text(
        raw_text=MAINTENANCE_SCENE_TEXT,
        scene_id="maintenance_demo_01",
        segment_id="s4",
        timestamp="00:30-00:55",
        scene_segment="sensor lens maintenance",
        actors=[
            GroundedEntry(
                text="Technician Omar",
                evidence="Technician Omar",
                entry_type="role",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="switch off machine",
                evidence="switches\noff the machine",
                entry_type="action",
                verb="switch off",
                direct_object="machine",
            ),
            GroundedEntry(
                text="wipe sensor lens with cleaning cloth",
                evidence="wipes the sensor lens with a cleaning cloth",
                entry_type="action",
                verb="wipe",
                direct_object="sensor lens",
                tool="cleaning cloth",
            ),
            GroundedEntry(
                text="inspect clean lens",
                evidence="inspects the\nclean lens",
                entry_type="action",
                verb="inspect",
                direct_object="sensor lens",
            ),
        ],
        tools_objects=[
            GroundedEntry(
                text="cleaning cloth",
                evidence="cleaning cloth",
                entry_type="tool",
            ),
            GroundedEntry(text="machine", evidence="machine", entry_type="object"),
            GroundedEntry(
                text="sensor lens",
                evidence="sensor lens",
                entry_type="object",
            ),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="machine shutdown allows safe cleaning",
                evidence="Switching off the machine allows safe cleaning",
            )
        ],
        evidence_uncertainty=[
            "The source explicitly states the safety relation between shutdown and cleaning."
        ],
    )

    # 温度检查场景：检验模型是否提取测量工具、参数和值记录动作。
    temperature_unified = build_unified_text(
        raw_text=TEMPERATURE_SCENE_TEXT,
        scene_id="thermal_demo_01",
        segment_id="s5",
        timestamp="01:00-01:20",
        scene_segment="coating temperature inspection",
        actors=[
            GroundedEntry(
                text="Quality inspector Noor",
                evidence="Quality inspector Noor",
                entry_type="role",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="point infrared thermometer at coated panel",
                evidence="points\nthe infrared thermometer at the coated panel",
                entry_type="action",
                verb="point",
                direct_object="infrared thermometer",
            ),
            GroundedEntry(
                text="read temperature of 42 degrees Celsius",
                evidence="reads a temperature of 42 degrees\nCelsius",
                entry_type="action",
                verb="read",
            ),
            GroundedEntry(
                text="record value",
                evidence="records the value",
                entry_type="action",
                verb="record",
            ),
        ],
        tools_objects=[
            GroundedEntry(
                text="infrared thermometer",
                evidence="infrared thermometer",
                entry_type="tool",
            ),
            GroundedEntry(
                text="coated panel",
                evidence="coated panel",
                entry_type="object",
            ),
        ],
        outcomes_parameters=[
            GroundedEntry(text="temperature: 42 degrees Celsius", evidence="42 degrees\nCelsius"),
            GroundedEntry(
                text="reading enables recording",
                evidence="Reading the temperature enables recording the value",
            ),
        ],
        evidence_uncertainty=[
            "The source supports both the temperature value and the recording order."
        ],
    )

    return BenchmarkDataset(
        name="egocentric_raw_unified_expanded_v1",
        scenes=[
            *MVP_BENCHMARK.scenes,
            BenchmarkScene(
                scene_id="assembly_demo_01",
                description="Assembly: position the cover plate and tighten the bolt with a torque wrench.",
                raw_text=ASSEMBLY_SCENE_TEXT,
                unified_record=assembly_unified,
                gold_extraction=ASSEMBLY_SCENE_EXPECTED,
            ),
            BenchmarkScene(
                scene_id="maintenance_demo_01",
                description="Maintenance: shut down the machine and clean the sensor lens with a cloth.",
                raw_text=MAINTENANCE_SCENE_TEXT,
                unified_record=maintenance_unified,
                gold_extraction=MAINTENANCE_SCENE_EXPECTED,
            ),
            BenchmarkScene(
                scene_id="thermal_demo_01",
                description="Quality inspection: read and record panel temperature with an infrared thermometer.",
                raw_text=TEMPERATURE_SCENE_TEXT,
                unified_record=temperature_unified,
                gold_extraction=TEMPERATURE_SCENE_EXPECTED,
            ),
        ],
    )


# 扩展数据集用于真实模型的第一轮对比实验；原 MVP 数据集仍适合快速测试。
EXPANDED_BENCHMARK = build_expanded_benchmark()
