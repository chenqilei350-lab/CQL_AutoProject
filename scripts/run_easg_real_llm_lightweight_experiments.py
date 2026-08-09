#!/usr/bin/env python3
"""Run lightweight real-LLM EASG extraction experiments.

Unlike the Instructor/Pydantic runner, this script asks the local LLM for a
small JSON contract and then converts that output into the project graph
schema. This is much more realistic for small local models and supports:

- raw vs unified input
- one-shot vs layered extraction
- repeated-run stability
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import time
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

from backend.datasets.benchmark import BenchmarkScene
from backend.datasets.easg_adapter import EASGStandardDataset, EASGStandardScene
from backend.datasets.expanded_benchmark import EXPANDED_BENCHMARK
from backend.datasets.industrial_reviewed_gold import (
    SCORING_NODE_LABELS,
    SCORING_RELATION_TYPES,
    load_industrial_reviewed_gold_dataset,
)
from backend.evaluation.metrics import PRF1, compute_prf1
from backend.graph.property_graph import PropertyGraph, build_property_graph, normalize_name
from backend.llm.client import get_openai_client
from backend.pipeline.industrial_text_to_kg import looks_like_industrial_tool
from backend.pipeline.relation_candidate_pipeline import (
    DeterministicRelationCandidateScorer,
    EntityMention,
    ProceduralRelationCandidateGenerator,
)
from backend.schemas.egocentric_video import (
    Action,
    ActionObservedInScene,
    ActionOrder,
    ActsOnObject,
    EgocentricVideoExtraction,
    Scene,
    SceneObject,
    UsesTool,
)
from backend.schemas.process_knowledge.entities import Tool


InputCondition = Literal["raw", "unified"]
Strategy = Literal["one_shot", "layered"]


class LLMJsonParseError(ValueError):
    """Raised when a model response cannot be safely converted to JSON."""

    def __init__(self, message: str, raw_content: str) -> None:
        super().__init__(message)
        self.raw_content = raw_content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight real-LLM EASG raw/unified and one-shot/layered experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["easg", "expanded_benchmark", "industrial_reviewed_gold"],
        default="easg",
        help="Dataset source. Use expanded_benchmark for the industrial gold experiment.",
    )
    parser.add_argument("--input", default="kg_ready_data/easg_standard_inputs.jsonl")
    parser.add_argument("--output-dir", default="results/easg_real_llm_lightweight_2026-06-20")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=None,
        help="Optional reviewed scene IDs to run before applying --limit.",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["raw_one_shot", "raw_layered", "unified_one_shot", "unified_layered"],
        default=["raw_one_shot", "raw_layered", "unified_one_shot", "unified_layered"],
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name, scenes = load_experiment_scenes(
        args.dataset,
        input_path,
        args.limit,
        scene_ids=args.scene_ids,
    )
    client = get_openai_client(timeout=args.timeout)

    all_rows: list[dict[str, Any]] = []
    graph_runs: dict[tuple[str, str, str], list[tuple[set[str], set[str]]]] = {}
    jsonl_path = output_dir / "all_runs.jsonl"

    print(f"Input: {input_path.resolve()}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)
    print(f"Scenes: {len(scenes)}", flush=True)
    print(f"Repetitions: {args.repetitions}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print("Mode: real_llm_lightweight", flush=True)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for cell in args.conditions:
            condition, strategy = parse_cell(cell)
            print(f"\nRunning {cell}", flush=True)
            for scene in scenes:
                gold_graph = build_property_graph(scene.gold_extraction)
                scoring_scope = args.dataset == "industrial_reviewed_gold"
                gold_nodes, gold_edges = graph_sets(
                    gold_graph,
                    node_labels=SCORING_NODE_LABELS if scoring_scope else None,
                    edge_types=SCORING_RELATION_TYPES if scoring_scope else None,
                )
                for run_number in range(1, args.repetitions + 1):
                    started = time.perf_counter()
                    error = ""
                    raw_response: dict[str, Any] = {}
                    try:
                        extraction, raw_response = extract_scene(
                            client=client,
                            scene=scene,
                            condition=condition,
                            strategy=strategy,
                            model=args.model,
                            timeout=args.timeout,
                            max_tokens=args.max_tokens,
                        )
                    except Exception as exc:  # keep long runs inspectable
                        error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                        if isinstance(exc, LLMJsonParseError):
                            raw_response = {"parse_error_raw_content": exc.raw_content}
                        extraction = EgocentricVideoExtraction(source_text=scene.raw_text)
                    graph = build_property_graph(extraction)
                    pred_nodes, pred_edges = graph_sets(
                        graph,
                        node_labels=SCORING_NODE_LABELS if scoring_scope else None,
                        edge_types=SCORING_RELATION_TYPES if scoring_scope else None,
                    )
                    node_metric = prf1_from_sets("nodes", gold_nodes, pred_nodes)
                    edge_metric = prf1_from_sets("edges", gold_edges, pred_edges)
                    runtime = round(time.perf_counter() - started, 4)
                    graph_runs.setdefault((scene.scene_id, condition, strategy), []).append(
                        (pred_nodes, pred_edges)
                    )
                    row = {
                        "experiment_mode": "real_llm_lightweight",
                        "scene_id": scene.scene_id,
                        "condition": condition,
                        "strategy": strategy,
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
                        "runtime_seconds": runtime,
                        "execution_error": error,
                    }
                    all_rows.append(row)
                    jsonl_file.write(
                        json.dumps(
                            {
                                **row,
                                "raw_response": raw_response,
                                "extraction": extraction.model_dump(mode="json"),
                                "graph": graph.model_dump(mode="json"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    jsonl_file.flush()
                    print(
                        f"{cell} | {scene.scene_id} | run {run_number} | "
                        f"node_f1={node_metric.f1:.4f} edge_f1={edge_metric.f1:.4f} "
                        f"time={runtime:.1f}s error={error or '-'}",
                        flush=True,
                    )

    stability_rows = build_stability_rows(graph_runs)
    summary_rows = build_summary_rows(all_rows, stability_rows)
    write_csv(output_dir / "all_runs.csv", all_rows)
    write_csv(output_dir / "stability.csv", stability_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.md").write_text(
        build_summary_md(
            input_path,
            output_dir,
            dataset_name,
            len(scenes),
            args.repetitions,
            args.model,
            summary_rows,
        ),
        encoding="utf-8",
    )

    print("\nSummary", flush=True)
    for row in summary_rows:
        print(
            f"{row['condition']}/{row['strategy']}: "
            f"node_f1={row['mean_node_f1']:.4f}, edge_f1={row['mean_edge_f1']:.4f}, "
            f"graph_overlap={row['mean_graph_overlap']:.4f}, errors={row['error_count']}",
            flush=True,
        )


def load_experiment_scenes(
    dataset_source: str,
    input_path: Path,
    limit: int,
    *,
    scene_ids: list[str] | None = None,
) -> tuple[str, list[EASGStandardScene | BenchmarkScene]]:
    if dataset_source == "expanded_benchmark":
        dataset_name = EXPANDED_BENCHMARK.name
        scenes: list[EASGStandardScene | BenchmarkScene] = list(
            EXPANDED_BENCHMARK.scenes
        )
    elif dataset_source == "industrial_reviewed_gold":
        dataset = load_industrial_reviewed_gold_dataset(input_path)
        dataset_name = dataset.name
        scenes = list(dataset.scenes)
    else:
        dataset = EASGStandardDataset.from_jsonl(input_path)
        dataset_name = dataset.name
        scenes = list(dataset.scenes)

    if scene_ids:
        requested = set(scene_ids)
        scenes = [scene for scene in scenes if scene.scene_id in requested]
        missing = requested - {scene.scene_id for scene in scenes}
        if missing:
            raise ValueError(f"Requested scene IDs not found: {sorted(missing)}")
    return dataset_name, scenes[:limit]


def parse_cell(cell: str) -> tuple[InputCondition, Strategy]:
    if cell.startswith("raw_"):
        return "raw", cell.removeprefix("raw_")  # type: ignore[return-value]
    if cell.startswith("unified_"):
        return "unified", cell.removeprefix("unified_")  # type: ignore[return-value]
    raise ValueError(f"Unsupported condition cell: {cell}")


def extract_scene(
    *,
    client: Any,
    scene: EASGStandardScene | BenchmarkScene,
    condition: InputCondition,
    strategy: Strategy,
    model: str,
    timeout: float,
    max_tokens: int,
) -> tuple[EgocentricVideoExtraction, dict[str, Any]]:
    input_text = scene.input_text(condition)
    if strategy == "one_shot":
        payload = call_json_llm(
            client,
            model=model,
            system_prompt=one_shot_prompt(),
            user_prompt=input_text,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        extraction = lightweight_json_to_extraction(payload, scene, input_text)
        return extraction, payload

    entity_payload = call_json_llm(
        client,
        model=model,
        system_prompt=entity_prompt(),
        user_prompt=input_text,
        timeout=timeout,
        max_tokens=max_tokens,
    )
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
    extraction = lightweight_json_to_extraction(merged_payload, scene, input_text)
    return extraction, {"entities": entity_payload, "relations": relation_payload}


def call_json_llm(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return parse_json_object(content)
    except Exception as exc:
        raise LLMJsonParseError(str(exc), content) from exc


def one_shot_prompt() -> str:
    return (
        "Extract a procedural action graph from the input. Return exactly one valid JSON object. "
        "Do not use Markdown fences or explanatory text. "
        "with keys actions, objects, relations. actions: list of {id,name,evidence_text}. "
        "objects: list of {id,name,object_type,evidence_text}; use object_type tool only "
        "for explicit tools. relations: list of {relation_type,subject_id,object_id,evidence_text}. "
        "Allowed relation_type: ACTS_ON, USES_TOOL, BEFORE. Use only facts explicitly in text. "
        "Keep names concise and each evidence_text at no more than eight words. "
        "Example: input \"action 'place' involves wood, right hand\" means action place, "
        "objects wood and right hand, and ACTS_ON relations from place to each object."
    )


def entity_prompt() -> str:
    return (
        "Extract only entities. Return exactly one valid JSON object. "
        "Do not use Markdown fences or explanatory text. Use keys actions and objects. "
        "actions: list of {id,name,evidence_text}. objects: list of "
        "{id,name,object_type,evidence_text}. Do not output relations. Use only text evidence. "
        "Keep names concise and each evidence_text at no more than eight words. "
        "For text like \"action 'place' involves wood, right hand\", extract action place "
        "and objects wood and right hand. IDs must be strings like \"action_1\", \"object_1\"."
    )


def relation_prompt(inventory: str) -> str:
    return (
        "Extract only relations between known IDs. Return exactly one valid JSON object. "
        "Do not use Markdown fences or explanatory text. Use key relations. "
        "Each relation is {relation_type,subject_id,object_id,evidence_text}. "
        "Allowed relation_type: ACTS_ON Action->Object, USES_TOOL Action->Tool, BEFORE Action->Action. "
        "Do not create IDs outside this inventory. If a needed endpoint ID is missing, omit the relation. "
        "Keep each evidence_text at no more than eight words. "
        "Do not write comments or explanation outside JSON.\n"
        f"{inventory}"
    )


def parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()
    else:
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            candidate = repair_json_like_text(match.group(0))
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(candidate)
        else:
            list_match = re.search(r"\[.*\]", content, re.DOTALL)
            if not list_match:
                raise
            candidate = repair_json_like_text(list_match.group(0))
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(candidate)
    if isinstance(parsed, list):
        parsed = list_to_payload(parsed)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")
    return normalize_payload_shape(parsed)


def repair_json_like_text(value: str) -> str:
    repaired = value
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    repaired = re.sub(r"//.*", "", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = repaired.replace("'", '"')
    return repaired


def list_to_payload(value: list[Any]) -> dict[str, Any]:
    if all(isinstance(item, dict) and "relation_type" in item for item in value):
        return {"relations": value}
    if all(isinstance(item, dict) and "name" in item for item in value):
        if any("object_type" in item for item in value):
            return {"objects": value}
        return {"actions": value}
    raise ValueError("LLM response list shape is ambiguous")


def normalize_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    entities = normalized.get("entities")
    if isinstance(entities, dict):
        normalized.setdefault("actions", entities.get("actions", []))
        normalized.setdefault("objects", entities.get("objects", []))
    relations = normalized.get("relations")
    if isinstance(relations, dict):
        normalized["relations"] = flatten_relation_payload(relations)
    if "relations" not in normalized and isinstance(normalized.get("relation"), list):
        normalized["relations"] = normalized["relation"]
    if "relations" not in normalized and any(
        key.upper() in {"ACTS_ON", "USES_TOOL", "BEFORE", "CAUSES", "PART_OF", "HAS_RESULT"}
        for key in normalized
    ):
        normalized["relations"] = flatten_relation_payload(normalized)
    return normalized


def flatten_relation_payload(relations: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize several common small-LLM relation shapes to one list."""

    if isinstance(relations.get("relations"), list):
        return [item for item in relations["relations"] if isinstance(item, dict)]

    flattened: list[dict[str, Any]] = []
    for relation_type, items in relations.items():
        if relation_type == "relations":
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized.setdefault("relation_type", relation_type)
            if "action_id" in normalized and "subject_id" not in normalized:
                normalized["subject_id"] = normalized["action_id"]
            if "target_id" in normalized and "object_id" not in normalized:
                normalized["object_id"] = normalized["target_id"]
            flattened.append(normalized)
    return flattened


def entity_inventory_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in payload.get("actions", []):
        lines.append(f"- {item.get('id')}: Action | {item.get('name')}")
    for item in payload.get("objects", []):
        kind = "Tool" if str(item.get("object_type", "")).casefold() == "tool" else "Object"
        lines.append(f"- {item.get('id')}: {kind} | {item.get('name')}")
    return "\n".join(lines) if lines else "- none"


def lightweight_json_to_extraction(
    payload: dict[str, Any],
    scene: EASGStandardScene | BenchmarkScene,
    input_text: str,
) -> EgocentricVideoExtraction:
    action_by_id: dict[str, Action] = {}
    object_by_id: dict[str, SceneObject] = {}
    tool_by_id: dict[str, Tool] = {}
    raw_relations = payload.get("relations", []) or []
    if isinstance(raw_relations, dict):
        raw_relations = flatten_relation_payload(raw_relations)
    relation_items = [item for item in raw_relations if isinstance(item, dict)]
    tool_reference_ids = {
        str(item.get("object_id") or "")
        for item in relation_items
        if normalize_relation_type(item.get("relation_type")) == "USES_TOOL"
    }

    for index, item in enumerate(payload.get("actions", []) or [], 1):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or f"action_{index}")
        name = clean_name(item.get("name"))
        if not name:
            continue
        action_by_id[entity_id] = Action(
            name=name,
            evidence_text=clean_name(item.get("evidence_text")) or name,
            source_text=input_text,
            sequence_index=index,
        )

    for index, item in enumerate(payload.get("objects", []) or [], 1):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or f"object_{index}")
        name = clean_name(item.get("name"))
        if not name:
            continue
        object_type = clean_name(item.get("object_type")) or "object"
        if looks_like_tool(entity_id, name, object_type, clean_name(item.get("evidence_text")), tool_reference_ids):
            tool_by_id[entity_id] = Tool(
                name=name,
                tool_type="tool",
                evidence_text=clean_name(item.get("evidence_text")) or name,
                source_text=input_text,
            )
        else:
            object_by_id[entity_id] = SceneObject(
                name=name,
                object_type=object_type,
                evidence_text=clean_name(item.get("evidence_text")) or name,
                source_text=input_text,
            )

    acts_on: list[ActsOnObject] = []
    uses_tool: list[UsesTool] = []
    before: list[ActionOrder] = []
    for item in relation_items:
        relation_type = normalize_relation_type(item.get("relation_type"))
        subject_id = str(item.get("subject_id") or "")
        object_id = str(item.get("object_id") or "")
        evidence = clean_name(item.get("evidence_text"))
        if not evidence and subject_id in action_by_id:
            evidence = action_by_id[subject_id].evidence_text or action_by_id[subject_id].name
        evidence = evidence or f"{subject_id} {object_id}"
        if (
            relation_type == "ACTS_ON"
            and subject_id in action_by_id
            and object_id in object_by_id
            and relation_is_source_supported(evidence, input_text, action_by_id[subject_id].name, object_by_id[object_id].name)
        ):
            acts_on.append(
                ActsOnObject(
                    action=action_by_id[subject_id],
                    object=object_by_id[object_id],
                    role="target",
                    evidence_text=evidence,
                    relation_origin="llm_extracted",
                )
            )
        elif (
            relation_type == "USES_TOOL"
            and subject_id in action_by_id
            and object_id in tool_by_id
            and relation_is_source_supported(evidence, input_text, action_by_id[subject_id].name, tool_by_id[object_id].name)
        ):
            uses_tool.append(
                UsesTool(
                    action=action_by_id[subject_id],
                    tool=tool_by_id[object_id],
                    evidence_text=evidence,
                    relation_origin="llm_extracted",
                )
            )
        elif relation_type == "BEFORE" and subject_id in action_by_id and object_id in action_by_id:
            before.append(
                ActionOrder(
                    before=action_by_id[subject_id],
                    after=action_by_id[object_id],
                    evidence_text=evidence,
                    relation_origin="llm_extracted",
                )
            )
    supplement_grounded_relations(
        action_by_id=action_by_id,
        object_by_id=object_by_id,
        tool_by_id=tool_by_id,
        acts_on=acts_on,
        uses_tool=uses_tool,
    )
    if not before and len(action_by_id) > 1:
        ordered_actions = list(action_by_id.values())
        for first, second in zip(ordered_actions, ordered_actions[1:]):
            before.append(
                ActionOrder(
                    before=first,
                    after=second,
                    evidence_text="Adjacent action order inferred from extracted action sequence.",
                    relation_origin="sequence_fallback",
                )
            )

    scene_node = scene.gold_extraction.scenes[0] if scene.gold_extraction.scenes else Scene(name=scene.scene_id)
    observed = [
        ActionObservedInScene(
            action=action,
            scene=scene_node,
            evidence_text=action.evidence_text,
            relation_origin="postprocessed",
        )
        for action in action_by_id.values()
    ]
    video_id = (
        getattr(scene, "video_id", None)
        or scene.gold_extraction.video_id
        or scene.scene_id
    )
    return EgocentricVideoExtraction(
        video_id=video_id,
        scenes=[scene_node],
        actions=list(action_by_id.values()),
        objects=list(object_by_id.values()),
        tools=list(tool_by_id.values()),
        acts_on_object=acts_on,
        uses_tool=uses_tool,
        action_order=before,
        observed_in_scene=observed,
        source_text=input_text,
    )


def supplement_grounded_relations(
    *,
    action_by_id: dict[str, Action],
    object_by_id: dict[str, SceneObject],
    tool_by_id: dict[str, Tool],
    acts_on: list[ActsOnObject],
    uses_tool: list[UsesTool],
) -> None:
    """Add conservative candidate-generated relations when the LLM omits usable edges."""

    existing_acts_on = {
        (normalize_name(relation.action.name), normalize_name(relation.object.name))
        for relation in acts_on
    }
    existing_uses_tool = {
        (normalize_name(relation.action.name), normalize_name(relation.tool.name))
        for relation in uses_tool
    }
    action_mentions = [
        EntityMention(
            entity_id=entity_id,
            label="Action",
            name=action.name,
            evidence_text=action.evidence_text or action.source_text or action.name,
            sequence_index=action.sequence_index,
        )
        for entity_id, action in action_by_id.items()
    ]
    tool_mentions = [
        EntityMention(
            entity_id=entity_id,
            label="Tool",
            name=tool.name,
            evidence_text=tool.evidence_text or tool.source_text or tool.name,
        )
        for entity_id, tool in tool_by_id.items()
    ]
    object_mentions = [
        EntityMention(
            entity_id=entity_id,
            label="Object",
            name=obj.name,
            evidence_text=obj.evidence_text or obj.source_text or obj.name,
        )
        for entity_id, obj in object_by_id.items()
    ]
    source_text = "\n".join(
        mention.evidence_text for mention in [*action_mentions, *tool_mentions, *object_mentions]
    )
    candidates = ProceduralRelationCandidateGenerator().generate(
        actions=action_mentions,
        tools=tool_mentions,
        objects=object_mentions,
        source_text=source_text,
    )
    report = DeterministicRelationCandidateScorer().score(candidates, source_text=source_text)
    for candidate in report.accepted_candidates():
        if candidate.relation_type == "USES_TOOL":
            action = action_by_id.get(candidate.subject_id)
            tool = tool_by_id.get(candidate.object_id)
            if not action or not tool:
                continue
            key = (normalize_name(action.name), normalize_name(tool.name))
            if key in existing_uses_tool:
                continue
            uses_tool.append(
                UsesTool(
                    action=action,
                    tool=tool,
                    evidence_text=candidate.evidence_hint,
                    relation_origin="evidence_fallback",
                )
            )
            existing_uses_tool.add(key)
        elif candidate.relation_type == "ACTS_ON":
            action = action_by_id.get(candidate.subject_id)
            obj = object_by_id.get(candidate.object_id)
            if not action or not obj:
                continue
            key = (normalize_name(action.name), normalize_name(obj.name))
            if key in existing_acts_on:
                continue
            acts_on.append(
                ActsOnObject(
                    action=action,
                    object=obj,
                    role="target",
                    evidence_text=candidate.evidence_hint,
                    relation_origin="evidence_fallback",
                )
            )
            existing_acts_on.add(key)


def evidence_mentions_entity(action_evidence: str, entity_name: str, entity_evidence: str | None) -> bool:
    action_text = normalize_name(action_evidence)
    candidates = [normalize_name(entity_name)]
    if entity_evidence:
        candidates.append(normalize_name(entity_evidence))
    for candidate in candidates:
        if candidate and candidate in action_text:
            return True
    entity_tokens = [token for token in normalize_name(entity_name).split() if len(token) > 3]
    if not entity_tokens:
        return False
    return all(token in action_text for token in entity_tokens)


def relation_is_source_supported(
    evidence_text: str,
    input_text: str,
    source_name: str,
    target_name: str,
) -> bool:
    """Keep explicit LLM relations only when their endpoints are grounded."""

    evidence = evidence_text or ""
    evidence_or_input = f"{evidence} {input_text}"
    return evidence_mentions_entity(evidence_or_input, source_name, None) and evidence_mentions_entity(
        evidence_or_input,
        target_name,
        None,
    )


def looks_like_tool(
    entity_id: str,
    name: str,
    object_type: str,
    evidence_text: str,
    tool_reference_ids: set[str],
) -> bool:
    return looks_like_industrial_tool(
        entity_id=entity_id,
        name=name,
        object_type=object_type,
        evidence_text=evidence_text,
        tool_reference_ids=tool_reference_ids,
    )


def clean_name(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").split())


def normalize_relation_type(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")


def graph_sets(
    graph: PropertyGraph,
    *,
    node_labels: frozenset[str] | None = None,
    edge_types: frozenset[str] | None = None,
) -> tuple[set[str], set[str]]:
    nodes = {
        f"{node.label}|{normalize_name(node.name)}"
        for node in graph.nodes.values()
        if node_labels is None or node.label in node_labels
    }
    edges: set[str] = set()
    for edge in graph.edges:
        if edge_types is not None and edge.type not in edge_types:
            continue
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


def prf1_from_sets(category: str, gold: set[str], predicted: set[str]) -> PRF1:
    return compute_prf1(
        category,
        true_positives=len(gold & predicted),
        false_positives=len(predicted - gold),
        false_negatives=len(gold - predicted),
    )


def build_stability_rows(
    graph_runs: dict[tuple[str, str, str], list[tuple[set[str], set[str]]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scene_id, condition, strategy), runs in graph_runs.items():
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


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def build_summary_rows(rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for condition, strategy in sorted({(row["condition"], row["strategy"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition and row["strategy"] == strategy]
        selected_stability = [
            row for row in stability_rows if row["condition"] == condition and row["strategy"] == strategy
        ]
        summary.append(
            {
                "condition": condition,
                "strategy": strategy,
                "run_count": len(selected),
                "error_count": sum(1 for row in selected if row["execution_error"]),
                "mean_node_f1": rounded_mean(selected, "node_f1"),
                "mean_edge_f1": rounded_mean(selected, "edge_f1"),
                "std_node_f1": rounded_std(selected, "node_f1"),
                "std_edge_f1": rounded_std(selected, "edge_f1"),
                "mean_graph_overlap": rounded_mean(selected_stability, "mean_graph_overlap"),
                "mean_runtime_seconds": rounded_mean(selected, "runtime_seconds"),
            }
        )
    return summary


def rounded_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(mean(values), 4) if values else 0.0


def rounded_std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(pstdev(values), 4) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    dataset_name: str,
    scene_count: int,
    repetitions: int,
    model: str,
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Lightweight Real LLM Experiment Summary",
        "",
        "Mode: `real_llm_lightweight` / 真实本地 LLM + 轻量 JSON schema。",
        "",
        f"- Dataset: `{dataset_name}`",
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
            "Interpretation / 解释：",
            "",
            "- Node/Edge F1 are measured against EASG gold graph annotations.",
            "- Graph overlap measures repeated-run stability.",
            "- This is a real LLM experiment, but with a lightweight JSON contract rather than full nested Pydantic output.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
