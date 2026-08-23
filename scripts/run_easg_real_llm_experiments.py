#!/usr/bin/env python3
"""Run real local-LLM experiments on EASG standard JSONL records.

This script compares:

- raw vs unified input
- one-shot vs layered extraction
- repeated-run stability

It uses the local Ollama/OpenAI-compatible extractor through
AlgorithmExperimentRunner and evaluates against EASG gold graph annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

from backend.datasets.benchmark import BenchmarkDataset
from backend.datasets.easg_adapter import EASGStandardDataset
from backend.graph.property_graph import PropertyGraph, normalize_name
from backend.pipeline.algorithm_experiment import (
    AlgorithmCondition,
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
    AlgorithmRunRecord,
)


InputCondition = Literal["raw", "unified"]

STRATEGY_TO_CONDITION: dict[str, AlgorithmCondition] = {
    "one_shot": "baseline_one_stage",
    "layered": "minimal_candidate_relation_with_validation",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real LLM raw/unified and one-shot/layered experiments on EASG JSONL."
    )
    parser.add_argument(
        "--input",
        default="kg_ready_data/easg_standard_inputs.jsonl",
        help="EASG standard JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/easg_real_llm_experiment_2026-06-20",
        help="Output directory.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of EASG scenes to run.")
    parser.add_argument("--repetitions", type=int, default=1, help="Repeated runs per condition.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-call timeout in seconds.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["raw_one_shot", "raw_layered", "unified_one_shot", "unified_layered"],
        default=["raw_one_shot", "raw_layered", "unified_one_shot", "unified_layered"],
        help="Subset of experiment cells to run.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    dataset = EASGStandardDataset.from_jsonl(input_path)
    scenes = dataset.to_benchmark_scenes()[: args.limit]
    benchmark = BenchmarkDataset(name="easg_real_llm_subset", scenes=scenes)

    all_runs_path = output_dir / "all_runs.jsonl"
    csv_rows: list[dict[str, object]] = []
    full_runs: list[AlgorithmRunRecord] = []

    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Scenes: {len(scenes)}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Model: {args.model}")
    print("Mode: real_llm")

    with all_runs_path.open("w", encoding="utf-8") as jsonl_file:
        for cell in args.conditions:
            input_condition, strategy = parse_cell(cell)
            algorithm_condition = STRATEGY_TO_CONDITION[strategy]
            runner = AlgorithmExperimentRunner(
                config=AlgorithmExperimentConfig(
                    experiment_name="easg_real_llm_raw_unified_layered",
                    model=args.model,
                    timeout=args.timeout,
                    max_retries=1,
                    input_condition=input_condition,
                    conditions=(algorithm_condition,),
                    repetitions=args.repetitions,
                    continue_on_error=True,
                )
            )
            print(f"\nRunning {cell}: {algorithm_condition}")
            result = runner.run(benchmark)
            for run in result.runs:
                full_runs.append(run)
                jsonl_file.write(json.dumps(run.model_dump(mode="json"), ensure_ascii=False) + "\n")
                jsonl_file.flush()
                row = run_to_csv_row(run, strategy, args.model)
                csv_rows.append(row)
                print(
                    f"{cell} | {run.scene_id} | run {run.run_number} | "
                    f"node_f1={row['node_f1']} edge_f1={row['edge_f1']} "
                    f"error={row['execution_error'] or '-'}"
                )

    stability_rows = build_stability_rows(full_runs)
    summary_rows = build_summary_rows(csv_rows, stability_rows)
    write_csv(output_dir / "all_runs.csv", csv_rows)
    write_csv(output_dir / "stability.csv", stability_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.md").write_text(
        build_summary_md(input_path, output_dir, len(scenes), args.repetitions, args.model, summary_rows),
        encoding="utf-8",
    )

    print("\nSummary")
    for row in summary_rows:
        print(
            f"{row['condition']}/{row['strategy']}: "
            f"node_f1={row['mean_node_f1']:.4f}, "
            f"edge_f1={row['mean_edge_f1']:.4f}, "
            f"graph_overlap={row['mean_graph_overlap']:.4f}, "
            f"errors={row['error_count']}"
        )


def parse_cell(cell: str) -> tuple[InputCondition, str]:
    if cell.startswith("raw_"):
        condition: InputCondition = "raw"
        strategy = cell.removeprefix("raw_")
    elif cell.startswith("unified_"):
        condition = "unified"
        strategy = cell.removeprefix("unified_")
    else:
        raise ValueError(f"Unsupported cell: {cell}")
    return condition, strategy


def run_to_csv_row(run: AlgorithmRunRecord, strategy: str, model: str) -> dict[str, object]:
    validation = run.validation
    return {
        "experiment_mode": "real_llm",
        "scene_id": run.scene_id,
        "condition": run.input_condition,
        "strategy": strategy,
        "algorithm_condition": run.condition,
        "run_number": run.run_number,
        "model": model,
        "node_precision": run.node_metrics.precision,
        "node_recall": run.node_metrics.recall,
        "node_f1": run.node_metrics.f1,
        "edge_precision": run.relation_metrics.precision,
        "edge_recall": run.relation_metrics.recall,
        "edge_f1": run.relation_metrics.f1,
        "node_tp": run.node_metrics.true_positives,
        "node_fp": run.node_metrics.false_positives,
        "node_fn": run.node_metrics.false_negatives,
        "edge_tp": run.relation_metrics.true_positives,
        "edge_fp": run.relation_metrics.false_positives,
        "edge_fn": run.relation_metrics.false_negatives,
        "ontology_conformance": validation.ontology_conformance,
        "relation_hallucination_rate": validation.relation_hallucination_rate,
        "object_grounding_rate": validation.object_grounding_rate,
        "evidence_grounding_rate": validation.evidence_grounding_rate,
        "filtered_relation_count": validation.filtered_relation_count,
        "runtime_seconds": run.runtime_seconds,
        "node_count": len(run.graph.nodes),
        "edge_count": len(run.graph.edges),
        "execution_error": run.execution_error or "",
    }


def build_stability_rows(runs: list[AlgorithmRunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str, str], list[AlgorithmRunRecord]] = {}
    for run in runs:
        strategy = "one_shot" if run.condition == "baseline_one_stage" else "layered"
        groups.setdefault((run.scene_id, run.input_condition, strategy), []).append(run)

    for (scene_id, condition, strategy), selected in groups.items():
        node_scores: list[float] = []
        edge_scores: list[float] = []
        graph_scores: list[float] = []
        for first, second in combinations(selected, 2):
            first_nodes, first_edges = graph_sets(first.graph)
            second_nodes, second_edges = graph_sets(second.graph)
            node_scores.append(jaccard(first_nodes, second_nodes))
            edge_scores.append(jaccard(first_edges, second_edges))
            graph_scores.append(
                jaccard(
                    {f"node::{item}" for item in first_nodes}
                    | {f"edge::{item}" for item in first_edges},
                    {f"node::{item}" for item in second_nodes}
                    | {f"edge::{item}" for item in second_edges},
                )
            )
        rows.append(
            {
                "scene_id": scene_id,
                "condition": condition,
                "strategy": strategy,
                "run_count": len(selected),
                "pair_count": len(node_scores),
                "mean_node_overlap": round(mean(node_scores), 4) if node_scores else 1.0,
                "mean_edge_agreement": round(mean(edge_scores), 4) if edge_scores else 1.0,
                "mean_graph_overlap": round(mean(graph_scores), 4) if graph_scores else 1.0,
            }
        )
    return rows


def graph_sets(graph: PropertyGraph) -> tuple[set[str], set[str]]:
    nodes = {f"{node.label}|{normalize_name(node.name)}" for node in graph.nodes.values()}
    edges: set[str] = set()
    for edge in graph.edges:
        source = graph.node(edge.source)
        target = graph.node(edge.target)
        edges.add(
            "|".join(
                [
                    source.label,
                    normalize_name(source.name),
                    edge.type,
                    target.label,
                    normalize_name(target.name),
                ]
            )
        )
    return nodes, edges


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def build_summary_rows(
    rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    present = sorted({(row["condition"], row["strategy"]) for row in rows})
    for condition, strategy in present:
        selected = [
            row for row in rows if row["condition"] == condition and row["strategy"] == strategy
        ]
        selected_stability = [
            row
            for row in stability_rows
            if row["condition"] == condition and row["strategy"] == strategy
        ]
        summary.append(
            {
                "condition": condition,
                "strategy": strategy,
                "run_count": len(selected),
                "error_count": sum(1 for row in selected if row["execution_error"]),
                "mean_node_precision": rounded_mean(selected, "node_precision"),
                "mean_node_recall": rounded_mean(selected, "node_recall"),
                "mean_node_f1": rounded_mean(selected, "node_f1"),
                "mean_edge_precision": rounded_mean(selected, "edge_precision"),
                "mean_edge_recall": rounded_mean(selected, "edge_recall"),
                "mean_edge_f1": rounded_mean(selected, "edge_f1"),
                "std_node_f1": rounded_std(selected, "node_f1"),
                "std_edge_f1": rounded_std(selected, "edge_f1"),
                "mean_ontology_conformance": rounded_mean(selected, "ontology_conformance"),
                "mean_filtered_relation_count": rounded_mean(selected, "filtered_relation_count"),
                "mean_runtime_seconds": rounded_mean(selected, "runtime_seconds"),
                "mean_graph_overlap": rounded_mean(selected_stability, "mean_graph_overlap"),
            }
        )
    return summary


def rounded_mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(mean(values), 4) if values else 0.0


def rounded_std(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(pstdev(values), 4) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_md(
    input_path: Path,
    output_dir: Path,
    scene_count: int,
    repetitions: int,
    model: str,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "# EASG Real LLM Experiment Summary",
        "",
        "Mode: `real_llm`; real local-LLM experiment.",
        "",
        f"- Input: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Model: `{model}`",
        f"- Scenes: `{scene_count}`",
        f"- Repetitions per condition: `{repetitions}`",
        "",
        "| Condition | Strategy | Runs | Errors | Node F1 | Edge F1 | Graph overlap | Runtime(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {strategy} | {run_count} | {error_count} | "
            "{mean_node_f1:.4f} | {mean_edge_f1:.4f} | {mean_graph_overlap:.4f} | "
            "{mean_runtime_seconds:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Node/Edge F1 are measured against EASG gold graph annotations.",
            "- Graph overlap measures repeated-run stability; with one repetition it is 1.0 by definition.",
            "- Layered means minimal entity extraction plus candidate-constrained relation extraction and validation.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
