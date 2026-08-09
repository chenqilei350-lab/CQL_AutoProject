"""实验结果表与错误分析模块测试。

本测试不访问真实 Ollama，而是用可控抽取结果验证：
正确输出能够得到满分，带有额外关系的输出会反映在表格与错误分析中。
"""

import csv
import io

from backend.datasets.benchmark import MVP_BENCHMARK
from backend.evaluation.reporting import build_comparison_report
from backend.pipeline.experiment_runner import (
    ExperimentRunConfig,
    RawUnifiedExperimentRunner,
)
from backend.schemas.egocentric_examples import (
    INSPECTION_SCENE_EXPECTED,
    WELDING_SCENE_EXPECTED,
)
from backend.schemas.egocentric_video import (
    Action,
    EgocentricVideoExtraction,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


class GoldForRawAndNoisyForUnifiedExtractor:
    """Raw 返回标准答案，Unified 故意增加一个无依据工具关系。"""

    def extract(
        self,
        text: str,
        response_model: type[EgocentricVideoExtraction],
        system_prompt: str | None = None,
    ) -> EgocentricVideoExtraction:
        if "weld_demo_01" in text:
            result = WELDING_SCENE_EXPECTED.model_copy(deep=True)
        else:
            result = INSPECTION_SCENE_EXPECTED.model_copy(deep=True)
        if "[场景 / 片段]" in text:
            result.uses_tool.append(
                UsesTool(
                    action=Action(name="measure gap"),
                    tool=Tool(name="laser scanner"),
                )
            )
        return result


def _build_report():
    """创建含两次重复运行的可控测试报告。"""

    batch = RawUnifiedExperimentRunner(
        config=ExperimentRunConfig(repetitions=2),
        extractor=GoldForRawAndNoisyForUnifiedExtractor(),
    ).run(MVP_BENCHMARK)
    return build_comparison_report(batch, MVP_BENCHMARK)


def test_report_compares_raw_and_unified_against_gold() -> None:
    """Raw 的 gold 输出应为满分，Unified 额外事实应降低精确率。"""

    report = _build_report()
    raw = report.row_for("raw")
    unified = report.row_for("unified")

    assert raw.node_metrics.f1 == 1.0
    assert raw.relation_metrics.f1 == 1.0
    assert unified.node_metrics.precision < 1.0
    assert unified.relation_metrics.precision < 1.0
    assert unified.mean_filtered_relation_count > raw.mean_filtered_relation_count


def test_report_includes_grounding_and_graph_errors() -> None:
    """额外工具和关系必须出现在人工可读的错误分析清单中。"""

    report = _build_report()
    details = "\n".join(item.detail for item in report.error_observations)

    assert "laser scanner" in details
    assert any(
        item.error_type == "额外关系"
        for item in report.error_observations
    )
    assert any(
        item.error_type.startswith("校验问题:")
        for item in report.error_observations
    )


def test_report_exports_markdown_and_csv(tmp_path) -> None:
    """汇总结果能够保存成报告文本和可制图的 CSV。"""

    report = _build_report()
    markdown_path = report.save_markdown(tmp_path / "comparison.md")
    csv_path = report.save_csv(tmp_path / "comparison.csv")

    markdown = markdown_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))

    assert "Raw vs Unified 实验结果" in markdown
    assert "Relation F1" in markdown
    assert "laser scanner" in markdown
    assert [row["condition"] for row in rows] == ["raw", "unified"]
    assert "mean_graph_overlap" in rows[0]
