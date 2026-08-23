#!/usr/bin/env python3
"""Checkpointed full real-LLM evaluation of the six-layer KG pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.datasets.benchmark import BenchmarkDataset, BenchmarkScene
from backend.datasets.easg_adapter import EASGStandardDataset
from backend.datasets.expanded_benchmark import EXPANDED_BENCHMARK
from backend.datasets.industrial_reviewed_gold import (
    DEFAULT_REVIEWED_GOLD_PATH,
    load_industrial_reviewed_gold_dataset,
)
from backend.evaluation.metrics import PRF1, aggregate_prf1, compute_prf1
from backend.evaluation.relation_alignment import evaluate_relation_types
from backend.graph.property_graph import build_property_graph, normalize_name
from backend.pipeline.algorithm_experiment import (
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
)
from backend.preprocessing.unified_text import (
    UnifiedTextRecord,
    build_industrial_unified_text,
)


CORE_RELATIONS = {
    "USES_TOOL",
    "ACTS_ON",
    "BEFORE",
    "CAUSES",
    "PART_OF",
    "OBSERVED_IN",
}
DEFAULT_EASG_INPUT = Path("kg_ready_data/easg_standard_inputs_sample.jsonl")
DEFAULT_OUTPUT = Path("results/six_layer_full_real_2026-08-21_qwen25_r1")


class ExperimentCase(BaseModel):
    dataset: str
    scene: BenchmarkScene
    preprocessing_mode: str

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.scene.scene_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Expanded, all reviewed Industrial Gold, and EASG sample through "
            "automatic/adapter normalization plus autonomous relation proposal."
        )
    )
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--industrial-gold", type=Path, default=DEFAULT_REVIEWED_GOLD_PATH)
    parser.add_argument("--easg-input", type=Path, default=DEFAULT_EASG_INPUT)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to each dataset.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Run only this scene ID; repeat the option for a targeted set.",
    )
    parser.add_argument(
        "--reuse-preprocessing-from",
        type=Path,
        default=None,
        help=(
            "Optional preprocessing_checkpoint.jsonl from a compatible earlier "
            "run. Selected rows are copied into the new checkpoint."
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--condition",
        choices=("llm_relation_proposal", "llm_relation_proposal_with_repair"),
        default="llm_relation_proposal_with_repair",
        help=(
            "Relation mode. The default lets the LLM repair proposals rejected "
            "by hard validation once, without adding rule-generated edges."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    cases = load_cases(args.industrial_gold, args.easg_input)
    if args.scene_id:
        requested = set(args.scene_id)
        available = {case.scene.scene_id for case in cases}
        unknown = requested - available
        if unknown:
            raise ValueError(
                "Unknown --scene-id values: " + ", ".join(sorted(unknown))
            )
        cases = [case for case in cases if case.scene.scene_id in requested]
    if args.limit_per_dataset is not None:
        if args.limit_per_dataset < 1:
            raise ValueError("limit_per_dataset must be at least 1")
        cases = limit_cases_per_dataset(cases, args.limit_per_dataset)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = output_dir / "preprocessing_checkpoint.jsonl"
    run_path = output_dir / "run_checkpoint.jsonl"
    preprocessing_rows = load_jsonl_by_key(preprocessing_path, "case_key")
    reusable_preprocessing_rows = (
        load_jsonl_by_key(args.reuse_preprocessing_from, "case_key")
        if args.reuse_preprocessing_from is not None
        else {}
    )
    run_rows = load_jsonl_by_key(run_path, "run_key")
    expected_run_keys = {
        f"{case.key}:r{run_number}"
        for case in cases
        for run_number in range(1, args.repetitions + 1)
    }
    write_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "condition": args.condition,
            "input": "six-layer controlled text",
            "repetitions": args.repetitions,
            "scene_counts": dict(Counter(case.dataset for case in cases)),
            "total_scenes": len(cases),
            "total_expected_runs": len(expected_run_keys),
            "industrial_gold": str(args.industrial_gold),
            "easg_input": str(args.easg_input),
            "limit_per_dataset": args.limit_per_dataset,
            "scene_ids": list(args.scene_id),
            "reuse_preprocessing_from": (
                str(args.reuse_preprocessing_from)
                if args.reuse_preprocessing_from is not None
                else None
            ),
            "resume": True,
        },
    )

    print(
        f"Six-layer full real experiment: {len(cases)} scenes, "
        f"{len(expected_run_keys)} runs, model={args.model}",
        flush=True,
    )
    for case_index, case in enumerate(cases, start=1):
        preprocessed = preprocessing_rows.get(case.key)
        if (
            preprocessed is None
            and case.preprocessing_mode != "source_adapter"
            and case.key in reusable_preprocessing_rows
        ):
            preprocessed = reusable_preprocessing_rows[case.key]
            append_jsonl(preprocessing_path, preprocessed)
            preprocessing_rows[case.key] = preprocessed
            print(f"PREPROCESS reuse-cache {case.key}", flush=True)
        if preprocessed is None:
            preprocessed = preprocess_case(case, args.model)
            append_jsonl(preprocessing_path, preprocessed)
            preprocessing_rows[case.key] = preprocessed
        else:
            print(f"PREPROCESS resume {case.key}", flush=True)

        for run_number in range(1, args.repetitions + 1):
            run_key = f"{case.key}:r{run_number}"
            if run_key in run_rows:
                print(f"RUN resume {run_key}", flush=True)
                continue
            print(
                f"RUN start {case_index}/{len(cases)} {run_key}",
                flush=True,
            )
            row = run_case(
                case,
                preprocessed,
                run_number=run_number,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                condition=args.condition,
            )
            append_jsonl(run_path, row)
            run_rows[run_key] = row
            write_summaries(output_dir, list(run_rows.values()), expected_run_keys)
            print(
                "RUN done "
                f"{run_key} error={row['execution_error']} "
                f"node_f1={row['node_metrics']['f1']:.4f} "
                f"relation_f1={row['relation_metrics']['f1']:.4f} "
                f"aligned_f1={row['core_aligned']['f1']:.4f} "
                f"seconds={row['runtime_seconds']:.2f}",
                flush=True,
            )

    write_summaries(output_dir, list(run_rows.values()), expected_run_keys)
    if expected_run_keys <= set(run_rows):
        write_json(
            output_dir / "COMPLETED.json",
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "completed_runs": len(run_rows),
                "expected_runs": len(expected_run_keys),
            },
        )
        print(f"COMPLETED {output_dir}", flush=True)


def load_cases(
    industrial_gold: Path,
    easg_input: Path,
) -> list[ExperimentCase]:
    cases = [
        ExperimentCase(
            dataset="expanded",
            scene=scene,
            preprocessing_mode="automatic_from_raw",
        )
        for scene in EXPANDED_BENCHMARK.scenes
    ]
    industrial = load_industrial_reviewed_gold_dataset(industrial_gold)
    cases.extend(
        ExperimentCase(
            dataset="industrial_reviewed",
            scene=scene,
            preprocessing_mode="automatic_from_raw",
        )
        for scene in industrial.scenes
    )
    easg = EASGStandardDataset.from_jsonl(easg_input)
    cases.extend(
        ExperimentCase(
            dataset="easg_sample",
            scene=scene.to_benchmark_scene(),
            preprocessing_mode="source_adapter",
        )
        for scene in easg.scenes
    )
    return cases


def limit_cases_per_dataset(
    cases: list[ExperimentCase],
    limit: int,
) -> list[ExperimentCase]:
    """Select the first N cases per dataset for cross-source smoke tests."""

    selected: list[ExperimentCase] = []
    counts: Counter[str] = Counter()
    for case in cases:
        if counts[case.dataset] >= limit:
            continue
        counts[case.dataset] += 1
        selected.append(case)
    return selected


def preprocess_case(case: ExperimentCase, model: str) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"PREPROCESS start {case.key} mode={case.preprocessing_mode}", flush=True)
    try:
        if case.preprocessing_mode == "automatic_from_raw":
            record = build_industrial_unified_text(
                raw_text=case.scene.raw_text,
                scene_id=case.scene.scene_id,
                segment_id=case.scene.scene_id,
                timestamp=None,
                scene=case.scene.description,
                source_adapter=f"automatic_{case.dataset}",
                transcript_text=case.scene.raw_text,
                model=model,
            )
        else:
            record = case.scene.unified_record
        segment = record.normalized_segment
        row = {
            "case_key": case.key,
            "dataset": case.dataset,
            "scene_id": case.scene.scene_id,
            "mode": case.preprocessing_mode,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "execution_error": None,
            "action_count": len(segment.actions) if segment else 0,
            "object_count": len(segment.objects) if segment else 0,
            "tool_count": len(segment.tools) if segment else 0,
            "role_count": len(segment.roles) if segment else 0,
            "record": record.model_dump(mode="json"),
        }
    except Exception as error:
        row = {
            "case_key": case.key,
            "dataset": case.dataset,
            "scene_id": case.scene.scene_id,
            "mode": case.preprocessing_mode,
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "execution_error": f"{type(error).__name__}: {error}",
            "action_count": 0,
            "object_count": 0,
            "tool_count": 0,
            "role_count": 0,
            "record": None,
        }
    print(
        f"PREPROCESS done {case.key} error={row['execution_error']} "
        f"actions={row['action_count']} seconds={row['runtime_seconds']:.2f}",
        flush=True,
    )
    return row


def run_case(
    case: ExperimentCase,
    preprocessing: dict[str, Any],
    *,
    run_number: int,
    model: str,
    timeout: float,
    max_retries: int,
    condition: str,
) -> dict[str, Any]:
    run_key = f"{case.key}:r{run_number}"
    if preprocessing["execution_error"] or not preprocessing.get("record"):
        return failed_run_row(
            case,
            run_key,
            run_number,
            "Preprocessing failed: " + str(preprocessing["execution_error"]),
            preprocessing_runtime_seconds=preprocessing["runtime_seconds"],
        )

    record = UnifiedTextRecord.model_validate(preprocessing["record"])
    scene = case.scene.model_copy(update={"unified_record": record}, deep=True)
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            experiment_name="six_layer_full_real",
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            input_condition="unified",
            # 中文：默认采用“LLM 自主提议 -> 硬验证 -> LLM 仅修复拒绝项”；
            # English: By default the LLM proposes freely, hard validation rejects,
            # and one LLM pass may repair only those rejected proposals.
            conditions=(condition,),
            repetitions=1,
            continue_on_error=True,
        )
    )
    run = runner.run(BenchmarkDataset(name=case.dataset, scenes=[scene])).runs[0]
    gold_graph = build_property_graph(scene.gold_extraction)
    exact = evaluate_relation_types(
        gold_graph,
        run.graph,
        relation_types=CORE_RELATIONS,
        align_nodes=False,
    )
    aligned = evaluate_relation_types(
        gold_graph,
        run.graph,
        relation_types=CORE_RELATIONS,
        align_nodes=True,
    )
    exact_total = aggregate_prf1(list(exact.values()), "core_exact")
    aligned_total = aggregate_prf1(list(aligned.values()), "core_aligned")
    trace = run.debug_trace
    return {
        "run_key": run_key,
        "dataset": case.dataset,
        "scene_id": scene.scene_id,
        "run_number": run_number,
        "preprocessing_mode": case.preprocessing_mode,
        "preprocessing_runtime_seconds": preprocessing["runtime_seconds"],
        "execution_error": run.execution_error,
        "runtime_seconds": run.runtime_seconds,
        "node_metrics": run.node_metrics.model_dump(mode="json"),
        "relation_metrics": run.relation_metrics.model_dump(mode="json"),
        "core_exact": exact_total.model_dump(mode="json"),
        "core_aligned": aligned_total.model_dump(mode="json"),
        "per_type_exact": {
            key: value.model_dump(mode="json") for key, value in exact.items()
        },
        "per_type_aligned": {
            key: value.model_dump(mode="json") for key, value in aligned.items()
        },
        "ontology_conformance": run.validation.ontology_conformance,
        "evidence_grounding_rate": run.validation.evidence_grounding_rate,
        "proposal_count": len(trace.minimal_relations) if trace else 0,
        "accepted_count": len(trace.filtered_relations) if trace else 0,
        "rejection_codes": [item.code for item in trace.rejected_relations]
        if trace
        else [],
        "stage_warnings": trace.stage_warnings if trace else [],
        "run": run.model_dump(mode="json"),
    }


def failed_run_row(
    case: ExperimentCase,
    run_key: str,
    run_number: int,
    error: str,
    *,
    preprocessing_runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    # 中文：失败样本是空预测，但 Gold 节点/关系必须全部计为 FN；否则
    # end_to_end 汇总会错误地把失败样本从分母中删除。
    # English: A failed run is an empty prediction whose gold facts are all FNs;
    # zero denominators would silently remove failures from end-to-end metrics.
    gold_graph = build_property_graph(case.scene.gold_extraction)
    node_facts = {
        (node.label, normalize_name(node.name))
        for node in gold_graph.nodes.values()
    }
    relation_facts = {
        (
            edge.type,
            normalize_name(gold_graph.node(edge.source).name),
            normalize_name(gold_graph.node(edge.target).name),
        )
        for edge in gold_graph.edges
    }
    core_facts = {fact for fact in relation_facts if fact[0] in CORE_RELATIONS}
    node_zero = compute_prf1(
        "nodes",
        0,
        0,
        len(node_facts),
    ).model_dump(mode="json")
    relation_zero = compute_prf1(
        "relations",
        0,
        0,
        len(relation_facts),
    ).model_dump(mode="json")
    core_zero = compute_prf1(
        "core",
        0,
        0,
        len(core_facts),
    ).model_dump(mode="json")
    per_type_zero = {
        relation_type: compute_prf1(
            f"relation:{relation_type}",
            0,
            0,
            sum(fact[0] == relation_type for fact in core_facts),
        ).model_dump(mode="json")
        for relation_type in sorted(CORE_RELATIONS)
    }
    return {
        "run_key": run_key,
        "dataset": case.dataset,
        "scene_id": case.scene.scene_id,
        "run_number": run_number,
        "preprocessing_mode": case.preprocessing_mode,
        "preprocessing_runtime_seconds": preprocessing_runtime_seconds,
        "execution_error": error,
        "runtime_seconds": 0.0,
        "node_metrics": node_zero,
        "relation_metrics": relation_zero,
        "core_exact": core_zero,
        "core_aligned": core_zero,
        "per_type_exact": per_type_zero,
        "per_type_aligned": per_type_zero,
        "ontology_conformance": 0.0,
        "evidence_grounding_rate": 0.0,
        "proposal_count": 0,
        "accepted_count": 0,
        "rejection_codes": [],
        "stage_warnings": [],
        "run": None,
    }


def write_summaries(
    output_dir: Path,
    rows: list[dict[str, Any]],
    expected_run_keys: set[str],
) -> None:
    summaries = []
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        successful = [row for row in selected if row["execution_error"] is None]
        summaries.append(summarize_rows(dataset, selected, successful))
    overall_successful = [row for row in rows if row["execution_error"] is None]
    overall = summarize_rows("overall", rows, overall_successful)
    rejections = Counter(
        code for row in rows for code in row.get("rejection_codes", [])
    )
    stage_warning_count = sum(
        len(row.get("stage_warnings", [])) for row in rows
    )
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "completed_runs": len(rows),
        "expected_runs": len(expected_run_keys),
        "remaining_runs": len(expected_run_keys - {row["run_key"] for row in rows}),
        "overall": overall,
        "datasets": summaries,
        "rejection_counts": dict(rejections.most_common()),
        "stage_warning_count": stage_warning_count,
    }
    write_json(output_dir / "summary.json", payload)
    (output_dir / "summary.md").write_text(
        render_markdown_summary(payload),
        encoding="utf-8",
    )


def summarize_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    successful: list[dict[str, Any]],
) -> dict[str, Any]:
    def metric_summary(
        selected: list[dict[str, Any]],
        scope: str,
    ) -> dict[str, Any]:
        def aggregate(field: str) -> dict[str, Any]:
            metrics = [PRF1.model_validate(row[field]) for row in selected]
            return aggregate_prf1(
                metrics,
                f"{dataset}_{scope}_{field}",
            ).model_dump(mode="json")

        return {
            "node_metrics": aggregate("node_metrics"),
            "relation_metrics": aggregate("relation_metrics"),
            "core_exact": aggregate("core_exact"),
            "core_aligned": aggregate("core_aligned"),
        }

    success_only = metric_summary(successful, "success_only")
    end_to_end = metric_summary(rows, "end_to_end")
    return {
        "dataset": dataset,
        "runs": len(rows),
        "successful_runs": len(successful),
        "failed_runs": len(rows) - len(successful),
        # Backward-compatible aliases remain success-only for old consumers.
        # 旧字段保留成功样本口径；新报告必须显式选择以下两个 scope。
        **success_only,
        "success_only": success_only,
        "end_to_end": end_to_end,
        "mean_runtime_seconds": mean(
            [float(row["runtime_seconds"]) for row in successful]
        ),
        "mean_preprocessing_seconds": mean(
            [float(row["preprocessing_runtime_seconds"]) for row in successful]
        ),
        "total_proposals": sum(int(row["proposal_count"]) for row in successful),
        "total_accepted": sum(int(row["accepted_count"]) for row in successful),
    }


def render_markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Six-layer Full Real-LLM Experiment",
        "",
        f"- Completed: `{payload['completed_runs']}/{payload['expected_runs']}`",
        f"- Remaining: `{payload['remaining_runs']}`",
        "",
        "| Dataset | Runs | Success | Success Node F1 | Success Relation F1 | End-to-end Node F1 | End-to-end Relation F1 | Mean seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [*payload["datasets"], payload["overall"]]:
        lines.append(
            f"| {row['dataset']} | {row['runs']} | {row['successful_runs']} | "
            f"{row['success_only']['node_metrics']['f1']:.4f} | "
            f"{row['success_only']['relation_metrics']['f1']:.4f} | "
            f"{row['end_to_end']['node_metrics']['f1']:.4f} | "
            f"{row['end_to_end']['relation_metrics']['f1']:.4f} | "
            f"{row['mean_runtime_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Hard-validation rejections",
            "",
            *(
                [
                    f"- `{code}`: {count}"
                    for code, count in payload["rejection_counts"].items()
                ]
                or ["- None recorded." ]
            ),
            "",
            f"- Recoverable LLM stage warnings: `{payload['stage_warning_count']}`",
            "",
            "This is a one-repetition coverage experiment, not a stability claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_jsonl_by_key(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row[key_field])] = row
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


if __name__ == "__main__":
    main()
