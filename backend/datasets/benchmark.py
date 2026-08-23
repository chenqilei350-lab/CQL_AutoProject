"""Small benchmark dataset for controlled raw/unified experiments.

Each scene binds the raw description, an evidence-grounded unified input, and a
human-authored Gold extraction. Experiment runners can therefore change only the
input representation while keeping model, prompt, schema, and graph construction
constant.
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
    """Complete inputs and Gold extraction for one benchmark scene."""

    scene_id: str
    description: str
    raw_text: str
    unified_record: UnifiedTextRecord
    gold_extraction: EgocentricVideoExtraction

    @model_validator(mode="after")
    def validate_source_alignment(self) -> "BenchmarkScene":
        """Ensure raw, unified, and Gold records refer to identical source text."""

        if self.unified_record.source_text != self.raw_text:
            raise ValueError("Unified-record source text must exactly match raw_text.")
        if self.gold_extraction.source_text != self.raw_text:
            raise ValueError("Gold-extraction source text must exactly match raw_text.")
        return self

    def input_text(self, condition: InputCondition) -> str:
        """Return model input for the requested raw or unified condition."""

        if condition == "raw":
            return self.raw_text
        if condition == "unified":
            return self.unified_record.to_extraction_text()
        raise ValueError(f"Unsupported input condition: {condition!r}")


class BenchmarkDataset(BaseModel):
    """Reusable scene benchmark for seed examples and reviewed datasets."""

    name: str
    scenes: list[BenchmarkScene] = Field(default_factory=list)

    def get_scene(self, scene_id: str) -> BenchmarkScene:
        """Return a scene by ID and report a missing ID clearly."""

        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"Scene does not exist in dataset: {scene_id!r}")

    def inputs_for(self, condition: InputCondition) -> list[tuple[str, str]]:
        """Return ``(scene_id, input_text)`` pairs for one input condition."""

        return [(scene.scene_id, scene.input_text(condition)) for scene in self.scenes]

    def to_jsonl(self) -> str:
        """Serialize one benchmark scene per JSON Lines row."""

        records = []
        for scene in self.scenes:
            # 中文：Gold/benchmark JSONL 保持历史可复现格式；运行时受控文本由
            # `scene.input_text("unified")` 动态生成，不写回 reviewed 数据包。
            # English: Preserve byte-reproducible benchmark packages. Controlled
            # extraction text is generated at runtime and is not written into gold.
            record = scene.model_dump(
                mode="json",
                exclude={"unified_record": {"normalized_segment"}},
            )
            record["unified_text"] = scene.unified_record.to_prompt_text()
            records.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(records)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> "BenchmarkDataset":
        """Restore a complete benchmark dataset from exported JSONL."""

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
                    f"Invalid JSONL at {input_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Every JSONL row must be an object: {input_path}:{line_number}"
                )
            # ``unified_text`` 是供人工检查的冗余导出字段，不属于模型合同。
            record.pop("unified_text", None)
            scenes.append(BenchmarkScene.model_validate(record))
        return cls(name=name or input_path.stem, scenes=scenes)

    def save_jsonl(self, path: str | Path) -> Path:
        """Save a dataset in a format directly loadable by :meth:`from_jsonl`."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_jsonl()
        output_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
        return output_path


def build_mvp_benchmark() -> BenchmarkDataset:
    """Build the MVP benchmark from reviewed Gold and grounded input pairs."""

    # 场景一：焊接准备，覆盖工具使用、动作顺序与因果关系。
    welding_unified = build_unified_text(
        raw_text=WELDING_SCENE_TEXT,
        scene_id="weld_demo_01",
        segment_id="s1",
        timestamp="00:00-00:20",
        scene_segment="root welding preparation",
        actors=[GroundedEntry(text="Hans", evidence="Hans", entry_type="role")],
        action_sequence=[
            GroundedEntry(
                text="pick up Fronius TPS 400i torch",
                evidence="Hans picks up the Fronius TPS 400i torch",
                entry_type="action",
                verb="pick up",
                tool="Fronius TPS 400i torch",
            ),
            GroundedEntry(
                text="align steel plate on table",
                evidence="aligns the steel plate on the table",
                entry_type="action",
                verb="align",
                direct_object="steel plate",
            ),
            GroundedEntry(
                text="start root weld",
                evidence="starts the root weld",
                entry_type="action",
                verb="start",
            ),
        ],
        tools_objects=[
            GroundedEntry(
                text="Fronius TPS 400i torch",
                evidence="Fronius TPS 400i torch",
                entry_type="tool",
            ),
            GroundedEntry(
                text="steel plate",
                evidence="steel plate",
                entry_type="object",
            ),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="alignment prepares plate for root weld",
                evidence="The alignment step prepares the plate for the root weld",
            )
        ],
        evidence_uncertainty=[
            "The source explicitly states the action order and preparation relation."
        ],
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
                entry_type="role",
            )
        ],
        action_sequence=[
            GroundedEntry(
                text="place caliper on bracket",
                evidence="places the caliper on the bracket",
                entry_type="action",
                verb="place",
                direct_object="caliper",
            ),
            GroundedEntry(
                text="measure gap",
                evidence="measures the gap",
                entry_type="action",
                verb="measure",
                direct_object="gap",
                tool="caliper",
            ),
            GroundedEntry(
                text="record result",
                evidence="records the result",
                entry_type="action",
                verb="record",
            ),
        ],
        tools_objects=[
            GroundedEntry(text="caliper", evidence="caliper", entry_type="tool"),
            GroundedEntry(
                text="bracket",
                evidence="bracket",
                entry_type="object",
            ),
            GroundedEntry(text="gap", evidence="gap", entry_type="object"),
        ],
        outcomes_parameters=[
            GroundedEntry(
                text="measurement causes documentation step",
                evidence="The measurement causes the documentation step",
            )
        ],
        evidence_uncertainty=[
            "The source explicitly states the relation between measurement and recording."
        ],
    )

    return BenchmarkDataset(
        name="egocentric_raw_unified_mvp",
        scenes=[
            BenchmarkScene(
                scene_id="weld_demo_01",
                description="Welding preparation: pick up the torch, align the plate, and start the root weld.",
                raw_text=WELDING_SCENE_TEXT,
                unified_record=welding_unified,
                gold_extraction=WELDING_SCENE_EXPECTED,
            ),
            BenchmarkScene(
                scene_id="inspect_demo_01",
                description="Quality inspection: place the caliper, measure the gap, and record the result.",
                raw_text=INSPECTION_SCENE_TEXT,
                unified_record=inspection_unified,
                gold_extraction=INSPECTION_SCENE_EXPECTED,
            ),
        ],
    )


# 数据集以常量形式暴露，便于 notebook、测试和后续实验运行器直接导入。
MVP_BENCHMARK = build_mvp_benchmark()
