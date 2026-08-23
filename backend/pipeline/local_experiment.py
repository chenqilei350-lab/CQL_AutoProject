"""Command entry point for local Ollama raw/unified comparison experiments.

The module connects the five-scene benchmark, local structured extraction,
graph and grounding validation, stability evaluation, and report exports.
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
    """Run a real or test extraction experiment and save analysis artifacts."""

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
            f"{'failure recorded' if record.execution_error else 'completed'} "
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
    """Parse CLI arguments, run the experiment, and print its summary."""

    parser = argparse.ArgumentParser(
        description="Run a local raw/unified AUT KG Extraction Pipeline experiment."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeated extractions per scene and condition; start with 1.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name; the project default is llama3.1:8b.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/local_experiment",
        help="Directory for experiment results.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries after schema validation failure; use 1 for diagnostic runs.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        help="Run only the specified scene ID; repeat for multiple scenes.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=["raw", "unified"],
        help="Run only the selected input condition; default runs both.",
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
    print(f"\nResults saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
