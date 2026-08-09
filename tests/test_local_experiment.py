"""本地实验入口的文件输出测试，不调用真实 Ollama。"""

from backend.pipeline.local_experiment import run_local_experiment
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    WELDING_SCENE_EXPECTED,
)
from backend.schemas.egocentric_video import EgocentricVideoExtraction
from backend.schemas.expanded_egocentric_examples import (
    ASSEMBLY_SCENE_EXPECTED,
    MAINTENANCE_SCENE_EXPECTED,
    TEMPERATURE_SCENE_EXPECTED,
)


class ExpandedGoldExtractor:
    """根据输入场景返回人工 gold 结果，以便只测试实验保存流程。"""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        examples = {
            "weld_demo_01": WELDING_SCENE_EXPECTED,
            "inspect_demo_01": INSPECTION_SCENE_EXPECTED,
            "assembly_demo_01": ASSEMBLY_SCENE_EXPECTED,
            "maintenance_demo_01": MAINTENANCE_SCENE_EXPECTED,
            "thermal_demo_01": TEMPERATURE_SCENE_EXPECTED,
        }
        for scene_id, expected in examples.items():
            if scene_id in text:
                return expected.model_copy(deep=True)
        raise ValueError("测试输入中没有已知场景编号。")


class FailsInspectionRawExtractor(ExpandedGoldExtractor):
    """模拟某一次真实模型输出无法通过 schema 的情况。"""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        if "inspect_demo_01" in text and "[场景 / 片段]" not in text:
            raise ValueError("输出字段结构不合规")
        return super().extract(text, response_model, system_prompt)


def test_local_experiment_saves_full_result_bundle(tmp_path) -> None:
    """一次五场景实验应保存原始记录、稳定性和两种报告格式。"""

    report = run_local_experiment(
        repetitions=1,
        output_dir=tmp_path,
        extractor=ExpandedGoldExtractor(),
    )

    assert report.row_for("raw").node_metrics.f1 == 1.0
    assert report.row_for("unified").relation_metrics.f1 == 1.0
    assert (tmp_path / "extraction_runs.jsonl").exists()
    assert (tmp_path / "extraction_runs.partial.jsonl").exists()
    assert (tmp_path / "stability_report.json").exists()
    assert (tmp_path / "comparison_report.md").exists()
    assert (tmp_path / "comparison_summary.csv").exists()
    assert len((tmp_path / "extraction_runs.jsonl").read_text().splitlines()) == 10


def test_local_experiment_can_limit_scene_and_condition_for_diagnosis(tmp_path) -> None:
    """诊断真实模型问题时，可以只重跑一个场景的一种输入。"""

    report = run_local_experiment(
        output_dir=tmp_path,
        extractor=ExpandedGoldExtractor(),
        scene_ids=["inspect_demo_01"],
        conditions=("raw",),
    )

    assert report.row_for("raw").relation_metrics.f1 == 1.0
    assert len((tmp_path / "extraction_runs.jsonl").read_text().splitlines()) == 1


def test_local_experiment_counts_failed_run_in_error_report(tmp_path) -> None:
    """某条抽取失败时，完整试验继续，并将失败作为结果问题记录。"""

    report = run_local_experiment(
        output_dir=tmp_path,
        extractor=FailsInspectionRawExtractor(),
    )

    markdown = report.to_markdown()
    assert "运行失败" in markdown
    assert "inspect_demo_01" in markdown
    assert len((tmp_path / "extraction_runs.jsonl").read_text().splitlines()) == 10
