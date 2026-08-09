"""五场景 Raw/Unified 初步实验数据集。

本模块在原有两个种子场景之外，再加入装配、维护清洁和温度检查场景。
它保留原 `MVP_BENCHMARK`，避免影响已经跑通的最小示例；需要更完整的
初步实验时，应使用本文件导出的 `EXPANDED_BENCHMARK`。
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
    """创建包含五个带原文证据和人工 gold graph 的初步实验集。"""

    # 装配场景：检验模型是否能识别工具紧固、步骤顺序与前置关系。
    assembly_unified = build_unified_text(
        raw_text=ASSEMBLY_SCENE_TEXT,
        scene_id="assembly_demo_01",
        segment_id="s3",
        timestamp="00:00-00:30",
        scene_segment="cover plate assembly",
        actors=[GroundedEntry(text="Operator Lena", evidence="Operator Lena")],
        action_sequence=[
            GroundedEntry(text="pick up torque wrench", evidence="picks up the\n torque wrench"),
            GroundedEntry(
                text="position cover plate on housing",
                evidence="positions the cover plate on the housing",
            ),
            GroundedEntry(
                text="tighten bolt with torque wrench",
                evidence="tightens the bolt\nwith the torque wrench",
            ),
        ],
        tools_objects=[
            GroundedEntry(text="torque wrench", evidence="torque wrench"),
            GroundedEntry(text="cover plate", evidence="cover plate"),
            GroundedEntry(text="bolt", evidence="bolt"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="positioning enables tightening",
                evidence="Positioning the cover plate enables tightening the bolt",
            )
        ],
        evidence_uncertainty=["装配动作顺序和前置关系均由场景文字直接描述。"],
    )

    # 维护场景：检验模型是否保留停机后再清洁的安全顺序。
    maintenance_unified = build_unified_text(
        raw_text=MAINTENANCE_SCENE_TEXT,
        scene_id="maintenance_demo_01",
        segment_id="s4",
        timestamp="00:30-00:55",
        scene_segment="sensor lens maintenance",
        actors=[GroundedEntry(text="Technician Omar", evidence="Technician Omar")],
        action_sequence=[
            GroundedEntry(
                text="switch off machine",
                evidence="switches\noff the machine",
            ),
            GroundedEntry(
                text="wipe sensor lens with cleaning cloth",
                evidence="wipes the sensor lens with a cleaning cloth",
            ),
            GroundedEntry(text="inspect clean lens", evidence="inspects the\nclean lens"),
        ],
        tools_objects=[
            GroundedEntry(text="cleaning cloth", evidence="cleaning cloth"),
            GroundedEntry(text="machine", evidence="machine"),
            GroundedEntry(text="sensor lens", evidence="sensor lens"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="machine shutdown allows safe cleaning",
                evidence="Switching off the machine allows safe cleaning",
            )
        ],
        evidence_uncertainty=["停机和清洁的安全关系由原文明确给出。"],
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
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="point infrared thermometer at coated panel",
                evidence="points\nthe infrared thermometer at the coated panel",
            ),
            GroundedEntry(
                text="read temperature of 42 degrees Celsius",
                evidence="reads a temperature of 42 degrees\nCelsius",
            ),
            GroundedEntry(text="record value", evidence="records the value"),
        ],
        tools_objects=[
            GroundedEntry(text="infrared thermometer", evidence="infrared thermometer"),
            GroundedEntry(text="coated panel", evidence="coated panel"),
        ],
        outcomes_parameters=[
            GroundedEntry(text="temperature: 42 degrees Celsius", evidence="42 degrees\nCelsius"),
            GroundedEntry(
                text="reading enables recording",
                evidence="Reading the temperature enables recording the value",
            ),
        ],
        evidence_uncertainty=["温度值与记录顺序均有文字证据。"],
    )

    return BenchmarkDataset(
        name="egocentric_raw_unified_expanded_v1",
        scenes=[
            *MVP_BENCHMARK.scenes,
            BenchmarkScene(
                scene_id="assembly_demo_01",
                description="装配：放置盖板并使用扭矩扳手紧固螺栓。",
                raw_text=ASSEMBLY_SCENE_TEXT,
                unified_record=assembly_unified,
                gold_extraction=ASSEMBLY_SCENE_EXPECTED,
            ),
            BenchmarkScene(
                scene_id="maintenance_demo_01",
                description="维护：停机后使用清洁布清洁传感器镜片。",
                raw_text=MAINTENANCE_SCENE_TEXT,
                unified_record=maintenance_unified,
                gold_extraction=MAINTENANCE_SCENE_EXPECTED,
            ),
            BenchmarkScene(
                scene_id="thermal_demo_01",
                description="质量检查：使用红外温度计读取并记录涂层板温度。",
                raw_text=TEMPERATURE_SCENE_TEXT,
                unified_record=temperature_unified,
                gold_extraction=TEMPERATURE_SCENE_EXPECTED,
            ),
        ],
    )


# 扩展数据集用于真实模型的第一轮对比实验；原 MVP 数据集仍适合快速测试。
EXPANDED_BENCHMARK = build_expanded_benchmark()
