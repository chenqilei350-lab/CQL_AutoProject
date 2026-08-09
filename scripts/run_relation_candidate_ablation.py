#!/usr/bin/env python3
"""Ablation for relation-candidate based edge extraction.

The experiment keeps entity extraction fixed per scene/run and swaps only the
relation stage:

1. current_minimal_candidate: current layered relation prompt + existing
   conservative fallback;
2. candidate_deterministic_rules: generated candidates accepted by deterministic
   rule validation;
3. candidate_llm_binary_judge: generated candidates judged by an LLM that may
   only answer supported/not-supported for existing candidate IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

from backend.datasets.benchmark import BenchmarkScene
from backend.datasets.easg_adapter import EASGStandardScene
from backend.datasets.industrial_reviewed_gold import (
    SCORING_NODE_LABELS,
    SCORING_RELATION_TYPES,
)
from backend.graph.property_graph import PropertyGraph, build_property_graph
from backend.llm.client import get_openai_client
from backend.pipeline.relation_candidate_pipeline import (
    DeterministicRelationCandidateScorer,
    EntityMention,
    LLMBinaryJudgeScorer,
    ProceduralRelationCandidateGenerator,
    RelationJudgeBatch,
    RelationValidationReport,
)
from backend.schemas.egocentric_video import (
    Action,
    ActionObservedInScene,
    ActionOrder,
    ActionPartOfProcedure,
    ActsOnObject,
    EgocentricVideoExtraction,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Procedure, Tool
from scripts.run_easg_real_llm_lightweight_experiments import (
    LLMJsonParseError,
    call_json_llm,
    entity_inventory_text,
    entity_prompt,
    graph_sets,
    lightweight_json_to_extraction,
    load_experiment_scenes,
    looks_like_tool,
    prf1_from_sets,
    relation_prompt,
)


InputCondition = Literal["raw", "unified"]
AblationMethod = Literal[
    "current_minimal_candidate",
    "candidate_deterministic_rules",
    "candidate_llm_binary_judge",
]


class LightweightJudgeBackend:
    """Adapter that lets LLMBinaryJudgeScorer reuse the lightweight JSON caller."""

    def __init__(self, client: Any, model: str, timeout: float, max_tokens: int) -> None:
        self.client = client
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_raw_error = ""

    def extract(
        self,
        text: str,
        response_model: type[RelationJudgeBatch],
        system_prompt: str | None = None,
    ) -> RelationJudgeBatch:
        strict_system_prompt = (
            f"{system_prompt or ''}\n"
            "Return exactly one compact JSON object with key decisions. "
            "Do not output Markdown, comments, prose, or an empty response."
        ).strip()
        try:
            payload = call_json_llm(
                self.client,
                model=self.model,
                system_prompt=strict_system_prompt,
                user_prompt=text,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
            )
        except LLMJsonParseError as exc:
            self.last_raw_error = exc.raw_content
            retry_prompt = (
                f"{text}\n\n"
                "Previous answer was not valid JSON. Retry now. "
                "Output only: {\"decisions\":[{\"candidate_id\":\"c_001\","
                "\"is_supported\":true,\"confidence\":0.82,"
                "\"evidence_text\":\"...\",\"reason\":\"...\"}]}"
            )
            payload = call_json_llm(
                self.client,
                model=self.model,
                system_prompt=strict_system_prompt,
                user_prompt=retry_prompt,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
            )
        return response_model.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run relation-candidate Edge F1 ablation.")
    parser.add_argument(
        "--dataset",
        choices=["easg", "expanded_benchmark", "industrial_reviewed_gold"],
        default="expanded_benchmark",
        help="Use expanded_benchmark for the current industrial gold experiment.",
    )
    parser.add_argument("--input", default="kg_ready_data/easg_standard_inputs.jsonl")
    parser.add_argument("--output-dir", default="results/relation_candidate_ablation_2026-07-07")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=None,
        help="Optional reviewed scene IDs to run before applying --limit.",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--input-condition", choices=["raw", "unified"], default="unified")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=550)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs/results")
    docs_dir.mkdir(parents=True, exist_ok=True)

    dataset_name, scenes = load_experiment_scenes(
        args.dataset,
        input_path,
        args.limit,
        scene_ids=args.scene_ids,
    )
    client = get_openai_client(timeout=args.timeout)
    judge_backend = LightweightJudgeBackend(
        client=client,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.judge_max_tokens,
    )

    all_rows: list[dict[str, Any]] = []
    graph_runs: dict[tuple[str, str], list[tuple[set[str], set[str]]]] = {}
    jsonl_path = output_dir / "all_runs.jsonl"

    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Input condition: {args.input_condition}", flush=True)
    print(f"Scenes: {len(scenes)}", flush=True)
    print(f"Repetitions: {args.repetitions}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for scene in scenes:
            input_text = scene.input_text(args.input_condition)
            gold_graph = build_property_graph(scene.gold_extraction)
            scoring_scope = args.dataset == "industrial_reviewed_gold"
            gold_nodes, gold_edges = graph_sets(
                gold_graph,
                node_labels=SCORING_NODE_LABELS if scoring_scope else None,
                edge_types=SCORING_RELATION_TYPES if scoring_scope else None,
            )
            for run_number in range(1, args.repetitions + 1):
                print(f"\nEntity extraction | {scene.scene_id} | run {run_number}", flush=True)
                entity_payload: dict[str, Any] = {}
                entity_error = ""
                entity_started = time.perf_counter()
                try:
                    entity_payload = call_json_llm(
                        client,
                        model=args.model,
                        system_prompt=entity_prompt(),
                        user_prompt=input_text,
                        timeout=args.timeout,
                        max_tokens=args.max_tokens,
                    )
                except Exception as exc:
                    entity_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                entity_runtime = round(time.perf_counter() - entity_started, 4)

                for method in ablation_methods():
                    started = time.perf_counter()
                    error = entity_error
                    payload: dict[str, Any] = {"entities": entity_payload}
                    report: RelationValidationReport | None = None
                    extraction = EgocentricVideoExtraction(source_text=input_text)
                    try:
                        if entity_error:
                            raise RuntimeError(entity_error)
                        if method == "current_minimal_candidate":
                            extraction, payload = run_current_minimal(
                                client=client,
                                scene=scene,
                                input_text=input_text,
                                entity_payload=entity_payload,
                                model=args.model,
                                timeout=args.timeout,
                                max_tokens=args.max_tokens,
                            )
                        elif method == "candidate_deterministic_rules":
                            extraction, report = run_candidate_method(
                                scene=scene,
                                input_text=input_text,
                                entity_payload=entity_payload,
                                scorer=DeterministicRelationCandidateScorer(),
                                relation_origin="deterministic_candidate",
                            )
                            payload = {"entities": entity_payload, "candidate_report": report.model_dump(mode="json")}
                        else:
                            extraction, report = run_candidate_method(
                                scene=scene,
                                input_text=input_text,
                                entity_payload=entity_payload,
                                scorer=LLMBinaryJudgeScorer(judge_backend),
                                relation_origin="llm_binary_judge",
                            )
                            payload = {"entities": entity_payload, "candidate_report": report.model_dump(mode="json")}
                    except Exception as exc:
                        if not error:
                            error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

                    graph = build_property_graph(extraction)
                    pred_nodes, pred_edges = graph_sets(
                        graph,
                        node_labels=SCORING_NODE_LABELS if scoring_scope else None,
                        edge_types=SCORING_RELATION_TYPES if scoring_scope else None,
                    )
                    node_metric = prf1_from_sets("nodes", gold_nodes, pred_nodes)
                    edge_metric = prf1_from_sets("edges", gold_edges, pred_edges)
                    runtime = round(time.perf_counter() - started, 4)
                    graph_runs.setdefault((scene.scene_id, method), []).append((pred_nodes, pred_edges))
                    candidate_count = len(report.candidates) if report else 0
                    accepted_count = len(report.accepted_candidates()) if report else 0
                    accepted_by_type = report.relation_counts() if report else {}
                    origin_counts = relation_origin_counts(graph)
                    row = {
                        "experiment_mode": "relation_candidate_ablation",
                        "scene_id": scene.scene_id,
                        "input_condition": args.input_condition,
                        "ablation_method": method,
                        "run_number": run_number,
                        "model": args.model,
                        "node_precision": node_metric.precision,
                        "node_recall": node_metric.recall,
                        "node_f1": node_metric.f1,
                        "edge_precision": edge_metric.precision,
                        "edge_recall": edge_metric.recall,
                        "edge_f1": edge_metric.f1,
                        "node_tp": node_metric.true_positives,
                        "node_fp": node_metric.false_positives,
                        "node_fn": node_metric.false_negatives,
                        "edge_tp": edge_metric.true_positives,
                        "edge_fp": edge_metric.false_positives,
                        "edge_fn": edge_metric.false_negatives,
                        "predicted_node_count": len(pred_nodes),
                        "predicted_edge_count": len(pred_edges),
                        "gold_node_count": len(gold_nodes),
                        "gold_edge_count": len(gold_edges),
                        "candidate_count": candidate_count,
                        "accepted_candidate_count": accepted_count,
                        "accepted_by_type": json.dumps(accepted_by_type, sort_keys=True),
                        "relation_origin_counts": json.dumps(origin_counts, sort_keys=True),
                        "entity_runtime_seconds": entity_runtime,
                        "runtime_seconds": runtime,
                        "execution_error": error,
                    }
                    all_rows.append(row)
                    jsonl_file.write(
                        json.dumps(
                            {
                                **row,
                                "payload": payload,
                                "extraction": extraction.model_dump(mode="json"),
                                "graph": graph.model_dump(mode="json"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    jsonl_file.flush()
                    print(
                        f"{method} | {scene.scene_id} | run {run_number} | "
                        f"node_f1={node_metric.f1:.4f} edge_f1={edge_metric.f1:.4f} "
                        f"candidates={candidate_count} accepted={accepted_count} "
                        f"time={runtime:.1f}s error={error or '-'}",
                        flush=True,
                    )

    stability_rows = build_stability_rows(graph_runs)
    summary_rows = build_summary_rows(all_rows, stability_rows)
    write_csv(output_dir / "all_runs.csv", all_rows)
    write_csv(output_dir / "stability.csv", stability_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    report_text = build_summary_md(
        dataset_name=dataset_name,
        input_condition=args.input_condition,
        output_dir=output_dir,
        scene_count=len(scenes),
        repetitions=args.repetitions,
        model=args.model,
        rows=summary_rows,
    )
    (output_dir / "summary.md").write_text(report_text, encoding="utf-8")
    report_path = docs_dir / f"{output_dir.name}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print("\nSummary", flush=True)
    for row in summary_rows:
        print(
            f"{row['ablation_method']}: edge_f1={row['mean_edge_f1']:.4f}, "
            f"node_f1={row['mean_node_f1']:.4f}, "
            f"edge_agreement={row['mean_edge_agreement']:.4f}, "
            f"errors={row['error_count']}",
            flush=True,
        )
    print(f"\nReport: {report_path.resolve()}", flush=True)


def ablation_methods() -> list[AblationMethod]:
    return [
        "current_minimal_candidate",
        "candidate_deterministic_rules",
        "candidate_llm_binary_judge",
    ]


def run_current_minimal(
    *,
    client: Any,
    scene: EASGStandardScene | BenchmarkScene,
    input_text: str,
    entity_payload: dict[str, Any],
    model: str,
    timeout: float,
    max_tokens: int,
) -> tuple[EgocentricVideoExtraction, dict[str, Any]]:
    inventory = entity_inventory_text(entity_payload)
    relation_payload = call_json_llm(
        client,
        model=model,
        system_prompt=relation_prompt(inventory),
        user_prompt=f"{input_text}\n\nKnown entities:\n{inventory}",
        timeout=timeout,
        max_tokens=max_tokens,
    )
    merged_payload = {
        "actions": entity_payload.get("actions", []),
        "objects": entity_payload.get("objects", []),
        "relations": relation_payload.get("relations", relation_payload),
    }
    return lightweight_json_to_extraction(merged_payload, scene, input_text), {
        "entities": entity_payload,
        "relations": relation_payload,
    }


def run_candidate_method(
    *,
    scene: EASGStandardScene | BenchmarkScene,
    input_text: str,
    entity_payload: dict[str, Any],
    scorer: Any,
    relation_origin: Literal["deterministic_candidate", "llm_binary_judge"],
) -> tuple[EgocentricVideoExtraction, RelationValidationReport]:
    entity_state = build_entity_state(scene, input_text, entity_payload)
    candidates = ProceduralRelationCandidateGenerator().generate(
        actions=entity_state["action_mentions"],
        tools=entity_state["tool_mentions"],
        objects=entity_state["object_mentions"],
        scenes=entity_state["scene_mentions"],
        procedures=entity_state["procedure_mentions"],
        source_text=input_text,
    )
    report = scorer.score(candidates, source_text=input_text)
    extraction = extraction_from_report(
        entity_state=entity_state,
        report=report,
        relation_origin=relation_origin,
        input_text=input_text,
    )
    return extraction, report


def build_entity_state(
    scene: EASGStandardScene | BenchmarkScene,
    input_text: str,
    entity_payload: dict[str, Any],
) -> dict[str, Any]:
    action_by_id: dict[str, Action] = {}
    tool_by_id: dict[str, Tool] = {}
    object_by_id: dict[str, SceneObject] = {}
    procedures: list[Procedure] = []

    for index, item in enumerate(entity_payload.get("actions", []) or [], 1):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or f"action_{index}")
        name = clean(item.get("name"))
        if not name:
            continue
        action_by_id[entity_id] = Action(
            name=name,
            evidence_text=clean(item.get("evidence_text")) or name,
            source_text=input_text,
            sequence_index=index,
        )

    tool_reference_ids: set[str] = set()
    for index, item in enumerate(entity_payload.get("objects", []) or [], 1):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or f"object_{index}")
        name = clean(item.get("name"))
        if not name:
            continue
        object_type = clean(item.get("object_type")) or "object"
        evidence_text = clean(item.get("evidence_text")) or name
        if looks_like_tool(entity_id, name, object_type, evidence_text, tool_reference_ids):
            tool_by_id[entity_id] = Tool(
                name=name,
                tool_type="tool",
                evidence_text=evidence_text,
                source_text=input_text,
            )
        else:
            object_by_id[entity_id] = SceneObject(
                name=name,
                object_type=object_type,
                evidence_text=evidence_text,
                source_text=input_text,
            )

    scene_node = scene.gold_extraction.scenes[0] if scene.gold_extraction.scenes else Scene(name=scene.scene_id)
    procedure_by_id = {f"procedure_{index}": procedure for index, procedure in enumerate(procedures, 1)}
    return {
        "scene": scene_node,
        "video_id": getattr(scene, "video_id", None) or scene.gold_extraction.video_id or scene.scene_id,
        "action_by_id": action_by_id,
        "tool_by_id": tool_by_id,
        "object_by_id": object_by_id,
        "procedure_by_id": procedure_by_id,
        "action_mentions": [
            EntityMention(
                entity_id=entity_id,
                label="Action",
                name=action.name,
                evidence_text=action.evidence_text or action.name,
                sequence_index=action.sequence_index,
            )
            for entity_id, action in action_by_id.items()
        ],
        "tool_mentions": [
            EntityMention(
                entity_id=entity_id,
                label="Tool",
                name=tool.name,
                evidence_text=tool.evidence_text or tool.name,
            )
            for entity_id, tool in tool_by_id.items()
        ],
        "object_mentions": [
            EntityMention(
                entity_id=entity_id,
                label="Object",
                name=obj.name,
                evidence_text=obj.evidence_text or obj.name,
            )
            for entity_id, obj in object_by_id.items()
        ],
        "scene_mentions": [
            EntityMention(
                entity_id="scene_1",
                label="Scene",
                name=scene_node.name,
                evidence_text=scene_node.evidence_text or scene_node.source_text or input_text,
            )
        ],
        "procedure_mentions": [
            EntityMention(
                entity_id=entity_id,
                label="Procedure",
                name=procedure.name,
                evidence_text=procedure.evidence_text or procedure.name,
            )
            for entity_id, procedure in procedure_by_id.items()
        ],
    }


def extraction_from_report(
    *,
    entity_state: dict[str, Any],
    report: RelationValidationReport,
    relation_origin: Literal["deterministic_candidate", "llm_binary_judge"],
    input_text: str,
) -> EgocentricVideoExtraction:
    action_by_id: dict[str, Action] = entity_state["action_by_id"]
    tool_by_id: dict[str, Tool] = entity_state["tool_by_id"]
    object_by_id: dict[str, SceneObject] = entity_state["object_by_id"]
    procedure_by_id: dict[str, Procedure] = entity_state["procedure_by_id"]
    scene_node: Scene = entity_state["scene"]

    uses_tool: list[UsesTool] = []
    acts_on: list[ActsOnObject] = []
    before: list[ActionOrder] = []
    part_of: list[ActionPartOfProcedure] = []
    observed: list[ActionObservedInScene] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for candidate in report.accepted_candidates():
        edge_key = (candidate.relation_type, candidate.subject_id, candidate.object_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        decision = report.decision_by_id().get(candidate.candidate_id)
        evidence = decision.evidence_text if decision and decision.evidence_text else candidate.evidence_hint
        confidence = decision.confidence if decision else candidate.confidence
        if candidate.relation_type == "USES_TOOL":
            action = action_by_id.get(candidate.subject_id)
            tool = tool_by_id.get(candidate.object_id)
            if action and tool:
                uses_tool.append(
                    UsesTool(
                        action=action,
                        tool=tool,
                        evidence_text=evidence,
                        confidence=confidence,
                        relation_origin=relation_origin,
                    )
                )
        elif candidate.relation_type == "ACTS_ON":
            action = action_by_id.get(candidate.subject_id)
            obj = object_by_id.get(candidate.object_id)
            if action and obj:
                acts_on.append(
                    ActsOnObject(
                        action=action,
                        object=obj,
                        role="target",
                        evidence_text=evidence,
                        confidence=confidence,
                        relation_origin=relation_origin,
                    )
                )
        elif candidate.relation_type == "BEFORE":
            first = action_by_id.get(candidate.subject_id)
            second = action_by_id.get(candidate.object_id)
            if first and second:
                before.append(
                    ActionOrder(
                        before=first,
                        after=second,
                        evidence_text=evidence,
                        confidence=confidence,
                        relation_origin=relation_origin,
                    )
                )
        elif candidate.relation_type == "OBSERVED_IN":
            action = action_by_id.get(candidate.subject_id)
            if action:
                observed.append(
                    ActionObservedInScene(
                        action=action,
                        scene=scene_node,
                        evidence_text=evidence,
                        confidence=confidence,
                        relation_origin=relation_origin,
                    )
                )
        elif candidate.relation_type == "PART_OF":
            action = action_by_id.get(candidate.subject_id)
            procedure = procedure_by_id.get(candidate.object_id)
            if action and procedure:
                part_of.append(
                    ActionPartOfProcedure(
                        action=action,
                        procedure=procedure,
                        evidence_text=evidence,
                        confidence=confidence,
                        relation_origin=relation_origin,
                    )
                )

    return EgocentricVideoExtraction(
        video_id=entity_state["video_id"],
        scenes=[scene_node],
        procedures=list(procedure_by_id.values()),
        actions=list(action_by_id.values()),
        tools=list(tool_by_id.values()),
        objects=list(object_by_id.values()),
        uses_tool=uses_tool,
        acts_on_object=acts_on,
        action_order=before,
        part_of_procedure=part_of,
        observed_in_scene=observed,
        source_text=input_text,
    )


def relation_origin_counts(graph: PropertyGraph) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for edge in graph.edges:
        origin = edge.properties.get("relation_origin") or "unknown"
        counts[f"{edge.type}:{origin}"] += 1
    return dict(counts)


def build_stability_rows(
    graph_runs: dict[tuple[str, str], list[tuple[set[str], set[str]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scene_id, method), runs in graph_runs.items():
        edge_scores: list[float] = []
        graph_scores: list[float] = []
        for index, left in enumerate(runs):
            for right in runs[index + 1 :]:
                left_nodes, left_edges = left
                right_nodes, right_edges = right
                edge_scores.append(jaccard(left_edges, right_edges))
                graph_scores.append(
                    jaccard(
                        {f"node::{item}" for item in left_nodes} | {f"edge::{item}" for item in left_edges},
                        {f"node::{item}" for item in right_nodes} | {f"edge::{item}" for item in right_edges},
                    )
                )
        rows.append(
            {
                "scene_id": scene_id,
                "ablation_method": method,
                "run_count": len(runs),
                "pair_count": len(edge_scores),
                "mean_edge_agreement": round(mean(edge_scores), 4) if edge_scores else 1.0,
                "mean_graph_overlap": round(mean(graph_scores), 4) if graph_scores else 1.0,
            }
        )
    return rows


def build_summary_rows(
    rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in ablation_methods():
        selected = [row for row in rows if row["ablation_method"] == method]
        selected_stability = [row for row in stability_rows if row["ablation_method"] == method]
        summary.append(
            {
                "ablation_method": method,
                "run_count": len(selected),
                "error_count": sum(1 for row in selected if row["execution_error"]),
                "mean_node_f1": rounded_mean(selected, "node_f1"),
                "mean_edge_f1": rounded_mean(selected, "edge_f1"),
                "std_node_f1": rounded_std(selected, "node_f1"),
                "std_edge_f1": rounded_std(selected, "edge_f1"),
                "mean_predicted_edges": rounded_mean(selected, "predicted_edge_count"),
                "mean_candidates": rounded_mean(selected, "candidate_count"),
                "mean_accepted_candidates": rounded_mean(selected, "accepted_candidate_count"),
                "mean_edge_agreement": rounded_mean(selected_stability, "mean_edge_agreement"),
                "mean_graph_overlap": rounded_mean(selected_stability, "mean_graph_overlap"),
                "mean_runtime_seconds": rounded_mean(selected, "runtime_seconds"),
            }
        )
    return summary


def build_summary_md(
    *,
    dataset_name: str,
    input_condition: str,
    output_dir: Path,
    scene_count: int,
    repetitions: int,
    model: str,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Relation Candidate Ablation",
        "",
        "This experiment compares relation-stage variants while keeping the same entity extraction call per scene/run.",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Input condition: `{input_condition}`",
        f"- Output directory: `{output_dir}`",
        f"- Model: `{model}`",
        f"- Scenes: `{scene_count}`",
        f"- Repetitions: `{repetitions}`",
        "",
        "| Method | Runs | Errors | Node F1 | Edge F1 | Edge agreement | Pred. edges | Candidates | Accepted | Runtime(s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {ablation_method} | {run_count} | {error_count} | {mean_node_f1:.4f} | "
            "{mean_edge_f1:.4f} | {mean_edge_agreement:.4f} | {mean_predicted_edges:.2f} | "
            "{mean_candidates:.2f} | {mean_accepted_candidates:.2f} | {mean_runtime_seconds:.2f} |".format(**row)
        )
    best = max(rows, key=lambda row: row["mean_edge_f1"]) if rows else None
    if best:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- Best Edge F1 in this run: `{best['ablation_method']}` with `{best['mean_edge_f1']:.4f}`.",
                "- `current_minimal_candidate` is the current layered relation prompt plus existing grounded fallback.",
                "- `candidate_deterministic_rules` does not ask the LLM to create edges; it accepts generated candidates by deterministic evidence rules.",
                "- `candidate_llm_binary_judge` asks the LLM only to validate candidate IDs, not to invent endpoints.",
                "- If candidate methods improve Edge F1, the new direction is useful. If they mainly increase predicted edge count without recall gains, candidate generation is over-broad or entity extraction still misses needed endpoints.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rounded_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(mean(values), 4) if values else 0.0


def rounded_std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(pstdev(values), 4) if values else 0.0


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").split())


if __name__ == "__main__":
    main()
