"""运行本地 Ollama Raw/Unified 对比实验的命令入口。

本模块把前面完成的多个功能串联成一次可以复现的真实实验：

    五场景 benchmark
    -> 本地 llama3.1:8b 结构化抽取
    -> graph 与 grounding/schema 校验
    -> 稳定性评价
    -> Markdown / CSV / JSONL 结果文件

第一次运行建议使用一次重复，先确认模型能够稳定返回符合 schema 的结果。
确认正常后，再将重复次数提高到五次，以获得用于报告的稳定性指标。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.datasets.benchmark import BenchmarkDataset, InputCondition
from backend.datasets.expanded_benchmark import EXPANDED_BENCHMARK
from backend.evaluation.reporting import (
    ExperimentComparisonReport,
    build_comparison_report,
)
from backend.evaluation.stability import evaluate_stability
from backend.llm.client import DEFAULT_MODEL
from backend.pipeline.experiment_runner import (
    ExperimentBatchResult,
    ExperimentRunConfig,
    ExtractionBackend,
    RawUnifiedExperimentRunner,
)


def run_local_experiment(
    repetitions: int = 1,
    model: str = DEFAULT_MODEL,
    output_dir: str | Path = "results/local_experiment",
    extractor: ExtractionBackend | None = None,
    max_retries: int = 1,
    scene_ids: list[str] | None = None,
    conditions: tuple[InputCondition, ...] = ("raw", "unified"),
) -> ExperimentComparisonReport:
    """执行真实或测试替代抽取器实验，并保存所有后续分析需要的输出。"""

    scenes = (
        EXPANDED_BENCHMARK.scenes
        if not scene_ids
        else [EXPANDED_BENCHMARK.get_scene(scene_id) for scene_id in scene_ids]
    )
    dataset = BenchmarkDataset(
        name=EXPANDED_BENCHMARK.name,
        scenes=scenes,
    )
    config = ExperimentRunConfig(
        model=model,
        repetitions=repetitions,
        max_retries=max_retries,
        conditions=conditions,
        continue_on_error=True,
    )
    runner = RawUnifiedExperimentRunner(config=config, extractor=extractor)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 同一运行器仅切换输入表示形式，保证 Raw/Unified 比较公平。
    # 实验每成功完成一次抽取就保存临时记录，避免后续单条失败丢失前序结果。
    completed_records = []
    total_runs = len(dataset.scenes) * len(config.conditions) * repetitions
    for index, record in enumerate(runner.iter_runs(dataset), start=1):
        completed_records.append(record)
        partial_result = ExperimentBatchResult(
            config=config,
            dataset_name=dataset.name,
            runs=completed_records,
        )
        partial_result.save_jsonl(output_path / "extraction_runs.partial.jsonl")
        print(
            f"[{index}/{total_runs}] "
            f"{'失败已记录' if record.execution_error else '已完成'} "
            f"{record.scene_id} / {record.condition} / run {record.run_number}",
            flush=True,
        )

    batch_result = ExperimentBatchResult(
        config=config,
        dataset_name=dataset.name,
        runs=completed_records,
    )
    stability_report = evaluate_stability(batch_result)
    comparison_report = build_comparison_report(
        batch_result,
        dataset=dataset,
        stability_report=stability_report,
    )

    # JSONL 保留最终完整输出，JSON 保留稳定性详情，MD/CSV 用于阅读与制图。
    batch_result.save_jsonl(output_path / "extraction_runs.jsonl")
    (output_path / "stability_report.json").write_text(
        stability_report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    comparison_report.save_markdown(output_path / "comparison_report.md")
    comparison_report.save_csv(output_path / "comparison_summary.csv")
    return comparison_report


def main() -> None:
    """读取终端参数，执行实验并在终端显示最终汇总表。"""

    parser = argparse.ArgumentParser(
        description="运行 AUT KG Extraction Pipeline 的 Raw/Unified 本地对比实验。"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="每个场景、每种输入形式的重复抽取次数；首次运行建议为 1。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama 模型名称，项目当前默认使用 llama3.1:8b。",
    )
    parser.add_argument(
        "--output-dir",
        default="results/local_experiment",
        help="实验结果保存目录。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="单条 schema 校验失败后的重试次数；诊断试运行建议为 1。",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        help="只运行指定场景编号；可以重复给出多个场景。默认运行全部场景。",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=["raw", "unified"],
        help="只运行指定输入形式；可以同时指定 raw 与 unified。默认运行两者。",
    )
    args = parser.parse_args()

    report = run_local_experiment(
        repetitions=args.repetitions,
        model=args.model,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        scene_ids=args.scene_id,
        conditions=tuple(args.condition) if args.condition else ("raw", "unified"),
    )
    print(report.to_markdown())
    print(f"\n结果已保存到: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
