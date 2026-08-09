#!/usr/bin/env python3
"""Run EASG raw/unified, one-shot/layered, and stability module experiments.

This runner is intentionally annotation-only and offline. It does not call an
LLM. Instead, it uses controlled perturbations of the EASG gold graph to test
whether the project metrics, logging, and stability analysis work on the public
EASG JSONL dataset before launching expensive real-LLM runs.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

from backend.datasets.easg_adapter import EASGStandardDataset
from backend.evaluation.metrics import PRF1, compute_prf1
from backend.graph.property_graph import PropertyGraph, build_property_graph, normalize_name
from backend.schemas.egocentric_video import (
    Action,
    ActsOnObject,
    EgocentricVideoExtraction,
    SceneObject,
)


InputCondition = Literal["raw", "unified"]
ExtractionStrategy = Literal["one_shot", "layered"]


@dataclass(frozen=True)
class SimulationProfile:
    """Controlled module-behaviour profile for one experiment condition."""

    node_keep_rate: float
    edge_keep_rate: float
    hallucinated_node_rate: float
    hallucinated_edge_rate: float


PROFILES: dict[tuple[InputCondition, ExtractionStrategy], SimulationProfile] = {
    ("raw", "one_shot"): SimulationProfile(0.82, 0.62, 0.10, 0.14),
    ("unified", "one_shot"): SimulationProfile(0.90, 0.73, 0.05, 0.08),
    ("raw", "layered"): SimulationProfile(0.88, 0.76, 0.06, 0.08),
    ("unified", "layered"): SimulationProfile(0.96, 0.88, 0.02, 0.03),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline EASG module experiments on standard JSONL inputs."
    )
    parser.add_argument(
        "--input",
        default="kg_ready_data/easg_standard_inputs.jsonl",
        help="EASG standard JSONL produced by build_easg_standard_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to results/easg_module_experiment_<date-like seed>.",
    )
    parser.add_argument("--limit", type=int, default=300, help="Maximum scenes to evaluate.")
    parser.add_argument("--repetitions", type=int, default=5, help="Repeated runs per condition.")
    parser.add_argument("--seed", type=int, default=20260620, help="Deterministic seed.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir or f"results/easg_module_experiment_seed_{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EASGStandardDataset.from_jsonl(input_path)
    scenes = dataset.scenes[: args.limit] if args.limit else dataset.scenes

    run_rows: list[dict[str, object]] = []
    graph_sets: dict[tuple[str, str, str], list[tuple[set[str], set[str]]]] = {}

    for scene_index, scene in enumerate(scenes):
        gold_graph = build_property_graph(scene.gold_extraction)
        gold_nodes = node_signatures(gold_graph)
        gold_edges = edge_signatures(gold_graph)
        for condition in ("raw", "unified"):
            for strategy in ("one_shot", "layered"):
                profile = PROFILES[(condition, strategy)]
                key = (scene.scene_id, condition, strategy)
                graph_sets.setdefault(key, [])
                for run_number in range(1, args.repetitions + 1):
                    rng = random.Random(f"{args.seed}:{scene_index}:{condition}:{strategy}:{run_number}")
                    prediction = perturb_extraction(scene.gold_extraction, profile, rng, run_number)
                    predicted_graph = build_property_graph(prediction)
                    predicted_nodes = node_signatures(predicted_graph)
                    predicted_edges = edge_signatures(predicted_graph)
                    graph_sets[key].append((predicted_nodes, predicted_edges))

                    node_metric = prf1_from_sets("nodes", gold_nodes, predicted_nodes)
                    edge_metric = prf1_from_sets("edges", gold_edges, predicted_edges)
                    run_rows.append(
                        {
                            "experiment_mode": "module_simulation",
                            "scene_id": scene.scene_id,
                            "condition": condition,
                            "strategy": strategy,
                            "run_number": run_number,
                            "node_precision": node_metric.precision,
                            "node_recall": node_metric.recall,
                            "node_f1": node_metric.f1,
                            "edge_precision": edge_metric.precision,
                            "edge_recall": edge_metric.recall,
                            "edge_f1": edge_metric.f1,
                            "hallucinated_nodes": node_metric.false_positives,
                            "hallucinated_edges": edge_metric.false_positives,
                            "missing_nodes": node_metric.false_negatives,
                            "missing_edges": edge_metric.false_negatives,
                            "gold_node_count": len(gold_nodes),
                            "gold_edge_count": len(gold_edges),
                            "predicted_node_count": len(predicted_nodes),
                            "predicted_edge_count": len(predicted_edges),
                        }
                    )

    stability_rows = build_stability_rows(graph_sets)
    summary_rows = build_summary_rows(run_rows, stability_rows)

    write_csv(output_dir / "all_runs.csv", run_rows)
    write_csv(output_dir / "stability.csv", stability_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.md").write_text(
        build_markdown_summary(input_path, output_dir, len(scenes), args.repetitions, summary_rows),
        encoding="utf-8",
    )
    write_svg_chart(output_dir / "summary_node_edge_f1.svg", summary_rows, "f1")
    write_svg_chart(output_dir / "summary_stability.svg", summary_rows, "stability")

    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Scenes: {len(scenes)}")
    print(f"Repetitions: {args.repetitions}")
    print("Mode: module_simulation (no LLM calls)")
    for row in summary_rows:
        print(
            f"{row['condition']}/{row['strategy']}: "
            f"node_f1={row['mean_node_f1']:.4f}, "
            f"edge_f1={row['mean_edge_f1']:.4f}, "
            f"graph_overlap={row['mean_graph_overlap']:.4f}"
        )


def perturb_extraction(
    gold: EgocentricVideoExtraction,
    profile: SimulationProfile,
    rng: random.Random,
    run_number: int,
) -> EgocentricVideoExtraction:
    """Create a deterministic, condition-specific simulated extraction."""

    prediction = gold.model_copy(deep=True)
    kept_actions = {
        normalize_name(action.name)
        for action in prediction.actions
        if rng.random() <= profile.node_keep_rate
    }
    kept_objects = {
        normalize_name(obj.name)
        for obj in prediction.objects
        if rng.random() <= profile.node_keep_rate
    }
    kept_tools = {
        normalize_name(tool.name)
        for tool in prediction.tools
        if rng.random() <= profile.node_keep_rate
    }

    prediction.actions = [
        action for action in prediction.actions if normalize_name(action.name) in kept_actions
    ]
    prediction.objects = [
        obj for obj in prediction.objects if normalize_name(obj.name) in kept_objects
    ]
    prediction.tools = [
        tool for tool in prediction.tools if normalize_name(tool.name) in kept_tools
    ]
    prediction.acts_on_object = [
        relation
        for relation in prediction.acts_on_object
        if normalize_name(relation.action.name) in kept_actions
        and normalize_name(relation.object.name) in kept_objects
        and rng.random() <= profile.edge_keep_rate
    ]
    prediction.uses_tool = [
        relation
        for relation in prediction.uses_tool
        if normalize_name(relation.action.name) in kept_actions
        and normalize_name(relation.tool.name) in kept_tools
        and rng.random() <= profile.edge_keep_rate
    ]
    prediction.action_order = [
        relation
        for relation in prediction.action_order
        if normalize_name(relation.before.name) in kept_actions
        and normalize_name(relation.after.name) in kept_actions
        and rng.random() <= profile.edge_keep_rate
    ]
    prediction.observed_in_scene = [
        relation
        for relation in prediction.observed_in_scene
        if normalize_name(relation.action.name) in kept_actions
        and rng.random() <= profile.edge_keep_rate
    ]

    if rng.random() < profile.hallucinated_node_rate:
        hallucinated_action = Action(
            name=f"unsupported extra action {run_number}",
            evidence_text=f"unsupported extra action {run_number}",
        )
        prediction.actions.append(hallucinated_action)
    if rng.random() < profile.hallucinated_node_rate:
        hallucinated_object = SceneObject(
            name=f"unsupported extra object {run_number}",
            object_type="object",
            evidence_text=f"unsupported extra object {run_number}",
        )
        prediction.objects.append(hallucinated_object)

    if rng.random() < profile.hallucinated_edge_rate and prediction.actions and prediction.objects:
        prediction.acts_on_object.append(
            ActsOnObject(
                action=Action(name=prediction.actions[-1].name, evidence_text=prediction.actions[-1].name),
                object=SceneObject(name=prediction.objects[-1].name, evidence_text=prediction.objects[-1].name),
                role="unknown",
                evidence_text="unsupported relation generated by module simulation",
            )
        )

    return prediction


def node_signatures(graph: PropertyGraph) -> set[str]:
    return {f"{node.label}|{normalize_name(node.name)}" for node in graph.nodes.values()}


def edge_signatures(graph: PropertyGraph) -> set[str]:
    signatures: set[str] = set()
    for edge in graph.edges:
        source = graph.node(edge.source)
        target = graph.node(edge.target)
        signatures.add(
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
    return signatures


def prf1_from_sets(category: str, gold: set[str], predicted: set[str]) -> PRF1:
    return compute_prf1(
        category,
        true_positives=len(gold & predicted),
        false_positives=len(predicted - gold),
        false_negatives=len(gold - predicted),
    )


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def build_stability_rows(
    graph_sets: dict[tuple[str, str, str], list[tuple[set[str], set[str]]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (scene_id, condition, strategy), runs in graph_sets.items():
        node_scores: list[float] = []
        edge_scores: list[float] = []
        graph_scores: list[float] = []
        for first, second in combinations(runs, 2):
            first_nodes, first_edges = first
            second_nodes, second_edges = second
            node_scores.append(jaccard(first_nodes, second_nodes))
            edge_scores.append(jaccard(first_edges, second_edges))
            graph_scores.append(
                jaccard(
                    {f"node::{item}" for item in first_nodes} | {f"edge::{item}" for item in first_edges},
                    {f"node::{item}" for item in second_nodes} | {f"edge::{item}" for item in second_edges},
                )
            )
        rows.append(
            {
                "scene_id": scene_id,
                "condition": condition,
                "strategy": strategy,
                "run_count": len(runs),
                "pair_count": len(node_scores),
                "mean_node_overlap": round(mean(node_scores), 4) if node_scores else 1.0,
                "mean_edge_agreement": round(mean(edge_scores), 4) if edge_scores else 1.0,
                "mean_graph_overlap": round(mean(graph_scores), 4) if graph_scores else 1.0,
            }
        )
    return rows


def build_summary_rows(
    run_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in ("raw", "unified"):
        for strategy in ("one_shot", "layered"):
            selected_runs = [
                row for row in run_rows if row["condition"] == condition and row["strategy"] == strategy
            ]
            selected_stability = [
                row
                for row in stability_rows
                if row["condition"] == condition and row["strategy"] == strategy
            ]
            rows.append(
                {
                    "condition": condition,
                    "strategy": strategy,
                    "run_count": len(selected_runs),
                    "mean_node_precision": rounded_mean(selected_runs, "node_precision"),
                    "mean_node_recall": rounded_mean(selected_runs, "node_recall"),
                    "mean_node_f1": rounded_mean(selected_runs, "node_f1"),
                    "mean_edge_precision": rounded_mean(selected_runs, "edge_precision"),
                    "mean_edge_recall": rounded_mean(selected_runs, "edge_recall"),
                    "mean_edge_f1": rounded_mean(selected_runs, "edge_f1"),
                    "mean_hallucinated_nodes": rounded_mean(selected_runs, "hallucinated_nodes"),
                    "mean_hallucinated_edges": rounded_mean(selected_runs, "hallucinated_edges"),
                    "std_node_f1": rounded_std(selected_runs, "node_f1"),
                    "std_edge_f1": rounded_std(selected_runs, "edge_f1"),
                    "mean_node_overlap": rounded_mean(selected_stability, "mean_node_overlap"),
                    "mean_edge_agreement": rounded_mean(selected_stability, "mean_edge_agreement"),
                    "mean_graph_overlap": rounded_mean(selected_stability, "mean_graph_overlap"),
                }
            )
    return rows


def rounded_mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(mean(values), 4) if values else 0.0


def rounded_std(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(pstdev(values), 4) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_summary(
    input_path: Path,
    output_dir: Path,
    scene_count: int,
    repetitions: int,
    rows: list[dict[str, object]],
) -> str:
    lines = [
        "# EASG Module Experiment Summary",
        "",
        "Mode: `module_simulation` / 模块模拟实验，不调用真实 LLM。",
        "",
        f"- Input: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Scenes: `{scene_count}`",
        f"- Repetitions per condition: `{repetitions}`",
        "- Compared conditions: `raw` vs `unified`, `one_shot` vs `layered`",
        "",
        "| Condition | Strategy | Node F1 | Edge F1 | Graph overlap | Hallucinated nodes | Hallucinated edges |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {strategy} | {mean_node_f1:.4f} | {mean_edge_f1:.4f} | "
            "{mean_graph_overlap:.4f} | {mean_hallucinated_nodes:.4f} | "
            "{mean_hallucinated_edges:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation / 解释：",
            "",
            "- Higher F1 means better correctness/completeness against EASG gold graph.",
            "- Higher graph overlap means better repeated-run stability.",
            "- Hallucinated nodes/edges are predicted facts not present in the EASG gold graph.",
            "- Because this is a module simulation, it validates the experimental pipeline and module assumptions, not final real-LLM performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_svg_chart(path: Path, rows: list[dict[str, object]], chart_type: str) -> None:
    labels = [f"{row['condition']}\\n{row['strategy']}" for row in rows]
    if chart_type == "f1":
        series = [
            ("Node F1", [float(row["mean_node_f1"]) for row in rows], "#3b82f6"),
            ("Edge F1", [float(row["mean_edge_f1"]) for row in rows], "#10b981"),
        ]
        title = "EASG Module Experiment: Node/Edge F1"
    else:
        series = [
            ("Graph overlap", [float(row["mean_graph_overlap"]) for row in rows], "#f59e0b"),
        ]
        title = "EASG Module Experiment: Stability"

    width = 900
    height = 420
    margin_left = 80
    margin_bottom = 80
    plot_width = width - margin_left - 40
    plot_height = height - 80 - margin_bottom
    group_width = plot_width / max(len(labels), 1)
    bar_width = min(42, group_width / (len(series) + 1))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-40}" y2="{height-margin_bottom}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="60" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = height - margin_bottom - value * plot_height
        svg.append(f'<line x1="{margin_left-5}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="#111827"/>')
        svg.append(
            f'<text x="{margin_left-12}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>'
        )
        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{width-40}" y2="{y}" stroke="#e5e7eb"/>')

    for label_index, label in enumerate(labels):
        group_x = margin_left + label_index * group_width + group_width / 2
        for series_index, (_, values, color) in enumerate(series):
            value = values[label_index]
            x = group_x - (len(series) * bar_width) / 2 + series_index * bar_width
            bar_height = value * plot_height
            y = height - margin_bottom - bar_height
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width-4}" height="{bar_height}" fill="{color}"/>')
            svg.append(
                f'<text x="{x+(bar_width-4)/2}" y="{y-6}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>'
            )
        first, second = label.split("\\n")
        svg.append(
            f'<text x="{group_x}" y="{height-52}" text-anchor="middle" font-family="Arial" font-size="12">{first}</text>'
        )
        svg.append(
            f'<text x="{group_x}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="12">{second}</text>'
        )

    legend_x = width - 260
    for idx, (name, _, color) in enumerate(series):
        y = 64 + idx * 22
        svg.append(f'<rect x="{legend_x}" y="{y-12}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+22}" y="{y}" font-family="Arial" font-size="13">{name}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
