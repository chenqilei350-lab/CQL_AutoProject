"""
小型实验数据集模块。

本模块把同一个场景的三类内容绑定在一起：
1. 原始场景描述（raw input）；
2. 由原文事实整理出的统一格式文本（unified input）；
3. 人工编写的标准抽取结果（gold extraction）。

后续实验运行器可以读取本模块，并保证 raw 与 unified 条件使用同一个模型、
同一个提示词、同一个抽取格式和同一个图构建方法，仅改变输入文本表示形式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.preprocessing.unified_text import (
    GroundedEntry,
    UnifiedTextRecord,
    build_unified_text,
)
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    INSPECTION_SCENE_TEXT,
    WELDING_SCENE_EXPECTED,
    WELDING_SCENE_TEXT,
)
from backend.schemas.egocentric_video import EgocentricVideoExtraction


InputCondition = Literal["raw", "unified"]


class BenchmarkScene(BaseModel):
    """
    单个实验场景的完整输入与标准答案。

    字段作用：
        scene_id: 实验中稳定使用的场景编号。
        description: 便于阅读的场景简介。
        raw_text: 直接来自场景描述的原始输入。
        unified_record: 经过固定栏目整理、并通过证据检查的统一输入。
        gold_extraction: 人工标准答案，用于后续评价模型抽取结果。
    """

    scene_id: str
    description: str
    raw_text: str
    unified_record: UnifiedTextRecord
    gold_extraction: EgocentricVideoExtraction

    @model_validator(mode="after")
    def validate_source_alignment(self) -> "BenchmarkScene":
        """
        检查原始文本是否在统一输入和标准答案中保持一致。

        这样可以避免实验中不小心比较了两个内容不同的场景。
        """

        if self.unified_record.source_text != self.raw_text:
            raise ValueError("统一格式记录的原始文本必须与 raw_text 完全一致。")
        if self.gold_extraction.source_text != self.raw_text:
            raise ValueError("人工标准答案的原始文本必须与 raw_text 完全一致。")
        return self

    def input_text(self, condition: InputCondition) -> str:
        """
        返回指定实验条件下送入抽取模型的文本。

        ``raw`` 返回原始描述；``unified`` 返回固定栏目文本。
        后续实验运行器会调用此函数来公平切换两种输入形式。
        """

        if condition == "raw":
            return self.raw_text
        if condition == "unified":
            return self.unified_record.to_prompt_text()
        raise ValueError(f"不支持的输入条件: {condition!r}")


class BenchmarkDataset(BaseModel):
    """
    可重复使用的场景基准数据集。

    同一个容器既可承载最初的两条 MVP 示例，也可承载扩展后的人工审核数据。
    """

    name: str
    scenes: list[BenchmarkScene] = Field(default_factory=list)

    def get_scene(self, scene_id: str) -> BenchmarkScene:
        """按编号取得一个场景；编号不存在时明确报告错误。"""

        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"数据集中不存在场景: {scene_id!r}")

    def inputs_for(self, condition: InputCondition) -> list[tuple[str, str]]:
        """
        批量获得某种输入条件的文本。

        输出为 ``(scene_id, input_text)`` 列表，下一模块的实验运行器可以直接循环使用。
        """

        return [(scene.scene_id, scene.input_text(condition)) for scene in self.scenes]

    def to_jsonl(self) -> str:
        """
        将数据集导出为 JSON Lines 文本。

        每行对应一个场景，便于以后保存为文件、检查标注或交给实验脚本加载。
        """

        records = []
        for scene in self.scenes:
            record = scene.model_dump(mode="json")
            record["unified_text"] = scene.input_text("unified")
            records.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(records)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> "BenchmarkDataset":
        """从导出的 JSONL 文件恢复完整基准数据集。"""

        input_path = Path(path)
        scenes: list[BenchmarkScene] = []
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"无效 JSONL：{input_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL 每一行必须是对象：{input_path}:{line_number}"
                )
            # ``unified_text`` 是供人工检查的冗余导出字段，不属于模型合同。
            record.pop("unified_text", None)
            scenes.append(BenchmarkScene.model_validate(record))
        return cls(name=name or input_path.stem, scenes=scenes)

    def save_jsonl(self, path: str | Path) -> Path:
        """将数据集保存为可由 :meth:`from_jsonl` 直接加载的文件。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_jsonl()
        output_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return output_path


def build_mvp_benchmark() -> BenchmarkDataset:
    """
    创建当前 MVP 使用的小型基准数据集。

    这里复用已经人工确认过的 gold extraction，只补充 raw/unified 输入对。
    每条 unified 信息都通过 ``build_unified_text`` 进行原文证据检查。
    """

    # 场景一：焊接准备，覆盖工具使用、动作顺序与因果关系。
    welding_unified = build_unified_text(
        raw_text=WELDING_SCENE_TEXT,
        scene_id="weld_demo_01",
        segment_id="s1",
        timestamp="00:00-00:20",
        scene_segment="root welding preparation",
        actors=[GroundedEntry(text="Hans", evidence="Hans")],
        action_sequence=[
            GroundedEntry(
                text="pick up Fronius TPS 400i torch",
                evidence="Hans picks up the Fronius TPS 400i torch",
            ),
            GroundedEntry(
                text="align steel plate on table",
                evidence="aligns the steel plate on the table",
            ),
            GroundedEntry(
                text="start root weld",
                evidence="starts the root weld",
            ),
        ],
        tools_objects=[
            GroundedEntry(
                text="Fronius TPS 400i torch",
                evidence="Fronius TPS 400i torch",
            ),
            GroundedEntry(text="steel plate", evidence="steel plate"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="alignment prepares plate for root weld",
                evidence="The alignment step prepares the plate for the root weld",
            )
        ],
        evidence_uncertainty=["动作顺序与准备关系均由原文明确描述。"],
    )

    # 场景二：质量检查，覆盖测量工具、记录动作与因果关系。
    inspection_unified = build_unified_text(
        raw_text=INSPECTION_SCENE_TEXT,
        scene_id="inspect_demo_01",
        segment_id="s2",
        timestamp="00:20-00:45",
        scene_segment="quality inspection",
        actors=[
            GroundedEntry(
                text="quality inspector Maria",
                evidence="quality inspector Maria",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="place caliper on bracket",
                evidence="places the caliper on the bracket",
            ),
            GroundedEntry(text="measure gap", evidence="measures the gap"),
            GroundedEntry(text="record result", evidence="records the result"),
        ],
        tools_objects=[
            GroundedEntry(text="caliper", evidence="caliper"),
            GroundedEntry(text="bracket", evidence="bracket"),
            GroundedEntry(text="gap", evidence="gap"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="measurement causes documentation step",
                evidence="The measurement causes the documentation step",
            )
        ],
        evidence_uncertainty=["测量与记录之间的关系由原文明确描述。"],
    )

    return BenchmarkDataset(
        name="egocentric_raw_unified_mvp",
        scenes=[
            BenchmarkScene(
                scene_id="weld_demo_01",
                description="焊接准备：拿取焊枪、对齐钢板并开始根焊。",
                raw_text=WELDING_SCENE_TEXT,
                unified_record=welding_unified,
                gold_extraction=WELDING_SCENE_EXPECTED,
            ),
            BenchmarkScene(
                scene_id="inspect_demo_01",
                description="质量检查：放置卡尺、测量间隙并记录结果。",
                raw_text=INSPECTION_SCENE_TEXT,
                unified_record=inspection_unified,
                gold_extraction=INSPECTION_SCENE_EXPECTED,
            ),
        ],
    )


# 数据集以常量形式暴露，便于 notebook、测试和后续实验运行器直接导入。
MVP_BENCHMARK = build_mvp_benchmark()
