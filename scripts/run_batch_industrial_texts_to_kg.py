#!/usr/bin/env python3
"""Batch convert industrial text files into procedural knowledge graphs.

This is the product-facing entry point for the handoff package. Put one or
more `.txt` files in an input folder and run this script once. It writes one
complete result folder per input text.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from backend.graph.property_graph import PropertyGraph, build_property_graph
from backend.pipeline.industrial_text_to_kg import summarize_relation_origins
from scripts.run_custom_industrial_text_to_kg import build_custom_scene
from scripts.run_easg_real_llm_lightweight_experiments import (
    LLMJsonParseError,
    extract_scene,
    get_openai_client,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert every .txt file in a folder into procedural KG outputs."
    )
    parser.add_argument("--input-dir", default="../input_texts", help="Folder containing .txt files.")
    parser.add_argument("--output-dir", default="../output_kg", help="Folder for generated KG outputs.")
    parser.add_argument("--condition", choices=["raw", "unified"], default="unified")
    parser.add_argument("--strategy", choices=["one_shot", "layered"], default="layered")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(path for path in input_dir.glob("*.txt") if path.is_file())
    if not text_files:
        raise SystemExit(
            f"No .txt files found in {input_dir.resolve()}. "
            "Put industrial text files there and run again."
        )

    client = get_openai_client(timeout=args.timeout)
    index_rows: list[dict[str, Any]] = []

    print(f"Input folder: {input_dir.resolve()}", flush=True)
    print(f"Output folder: {output_dir.resolve()}", flush=True)
    print(f"Files: {len(text_files)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Condition: {args.condition}", flush=True)
    print(f"Strategy: {args.strategy}", flush=True)

    for input_path in text_files:
        started = time.perf_counter()
        scene_id = safe_stem(input_path)
        result_dir = output_dir / scene_id
        result_dir.mkdir(parents=True, exist_ok=True)
        raw_text = input_path.read_text(encoding="utf-8").strip()

        print(f"\nProcessing: {input_path.name}", flush=True)
        if not raw_text:
            row = write_error_result(
                input_path=input_path,
                result_dir=result_dir,
                error="Input file is empty.",
                runtime_seconds=round(time.perf_counter() - started, 4),
            )
            index_rows.append(row)
            print("Skipped: empty file", flush=True)
            continue

        try:
            scene = build_custom_scene(raw_text, scene_id)
            extraction, raw_response = extract_scene(
                client=client,
                scene=scene,
                condition=args.condition,
                strategy=args.strategy,
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
            graph = build_property_graph(extraction)
            runtime = round(time.perf_counter() - started, 4)
            write_success_outputs(
                input_path=input_path,
                result_dir=result_dir,
                condition=args.condition,
                strategy=args.strategy,
                model=args.model,
                raw_text=raw_text,
                raw_response=raw_response,
                extraction=extraction.model_dump(mode="json"),
                graph=graph,
                runtime_seconds=runtime,
            )
            row = {
                "input_file": str(input_path),
                "status": "success",
                "output_dir": str(result_dir),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "runtime_seconds": runtime,
                "error": "",
            }
            index_rows.append(row)
            print(f"Done: nodes={len(graph.nodes)} edges={len(graph.edges)} time={runtime:.1f}s", flush=True)
        except Exception as exc:  # keep batch runs useful even if one file fails
            runtime = round(time.perf_counter() - started, 4)
            message = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            raw_content = exc.raw_content if isinstance(exc, LLMJsonParseError) else ""
            row = write_error_result(
                input_path=input_path,
                result_dir=result_dir,
                error=message,
                runtime_seconds=runtime,
                raw_content=raw_content,
            )
            index_rows.append(row)
            print(f"Failed: {message}", flush=True)

    write_index(output_dir, index_rows)
    print(f"\nFinished. Open this report: {(output_dir / 'RUN_SUMMARY.md').resolve()}", flush=True)


def safe_stem(path: Path) -> str:
    value = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem.strip())
    return value or "industrial_text"


def write_success_outputs(
    *,
    input_path: Path,
    result_dir: Path,
    condition: str,
    strategy: str,
    model: str,
    raw_text: str,
    raw_response: dict[str, Any],
    extraction: dict[str, Any],
    graph: PropertyGraph,
    runtime_seconds: float,
) -> None:
    payload = {
        "input_file": str(input_path.resolve()),
        "condition": condition,
        "strategy": strategy,
        "model": model,
        "runtime_seconds": runtime_seconds,
        "raw_text": raw_text,
        "raw_response": raw_response,
        "extraction": extraction,
        "relation_origin_summary": summarize_relation_origins(graph.edges).__dict__,
        "graph": graph.model_dump(mode="json"),
        "cypher": graph.to_cypher(),
    }
    write_json(result_dir / "kg_result.json", payload)
    write_nodes_csv(result_dir / "graph_nodes.csv", graph)
    write_edges_csv(result_dir / "graph_edges.csv", graph)
    (result_dir / "graph.cypher").write_text(graph.to_cypher() + "\n", encoding="utf-8")
    (result_dir / "summary.md").write_text(
        build_result_summary(
            input_path=input_path,
            condition=condition,
            strategy=strategy,
            model=model,
            graph=graph,
            relation_origin_summary=summarize_relation_origins(graph.edges).__dict__,
            runtime_seconds=runtime_seconds,
            error="",
        ),
        encoding="utf-8",
    )


def write_error_result(
    *,
    input_path: Path,
    result_dir: Path,
    error: str,
    runtime_seconds: float,
    raw_content: str = "",
) -> dict[str, Any]:
    payload = {
        "input_file": str(input_path.resolve()),
        "status": "error",
        "runtime_seconds": runtime_seconds,
        "error": error,
        "raw_content": raw_content,
    }
    write_json(result_dir / "kg_result.json", payload)
    (result_dir / "summary.md").write_text(
        build_result_summary(
            input_path=input_path,
            condition="-",
            strategy="-",
            model="-",
            graph=PropertyGraph(),
            runtime_seconds=runtime_seconds,
            error=error,
        ),
        encoding="utf-8",
    )
    return {
        "input_file": str(input_path),
        "status": "error",
        "output_dir": str(result_dir),
        "node_count": 0,
        "edge_count": 0,
        "runtime_seconds": runtime_seconds,
        "error": error,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_nodes_csv(path: Path, graph: PropertyGraph) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "label", "name", "properties_json"])
        writer.writeheader()
        for node in graph.nodes.values():
            writer.writerow(
                {
                    "id": node.id,
                    "label": node.label,
                    "name": node.name,
                    "properties_json": json.dumps(node.properties, ensure_ascii=False, sort_keys=True),
                }
            )


def write_edges_csv(path: Path, graph: PropertyGraph) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "type", "source", "target", "properties_json"])
        writer.writeheader()
        for edge in graph.edges:
            writer.writerow(
                {
                    "id": edge.id,
                    "type": edge.type,
                    "source": edge.source,
                    "target": edge.target,
                    "properties_json": json.dumps(edge.properties, ensure_ascii=False, sort_keys=True),
                }
            )


def build_result_summary(
    *,
    input_path: Path,
    condition: str,
    strategy: str,
    model: str,
    graph: PropertyGraph,
    relation_origin_summary: dict[str, Any] | None = None,
    runtime_seconds: float,
    error: str,
) -> str:
    lines = [
        f"# KG Result: {input_path.name}",
        "",
        f"- Status: {'error' if error else 'success'}",
        f"- Input file: `{input_path}`",
        f"- Model: `{model}`",
        f"- Condition: `{condition}`",
        f"- Strategy: `{strategy}`",
        f"- Runtime seconds: `{runtime_seconds}`",
        f"- Node count: `{len(graph.nodes)}`",
        f"- Edge count: `{len(graph.edges)}`",
        "",
    ]
    if error:
        lines.extend(["## Error", "", error, ""])
        return "\n".join(lines)

    lines.extend(["## Nodes", ""])
    for node in graph.nodes.values():
        lines.append(f"- `{node.id}` | {node.label} | {node.name}")
    lines.extend(["", "## Edges", ""])
    if graph.edges:
        for edge in graph.edges:
            lines.append(f"- `{edge.source}` -[{edge.type}]-> `{edge.target}`")
    else:
        lines.append("- No edges were extracted.")
    if relation_origin_summary:
        lines.extend(["", "## Relation Origin Summary", ""])
        lines.append(
            "This separates LLM-extracted relations from grounding-based fallback and post-processing."
        )
        lines.extend(["", "| Origin | Edge Count |", "| --- | ---: |"])
        for origin, count in sorted(relation_origin_summary.get("by_origin", {}).items()):
            lines.append(f"| `{origin}` | {count} |")
        lines.extend(["", "| Relation and Origin | Edge Count |", "| --- | ---: |"])
        for key, count in sorted(relation_origin_summary.get("by_relation_origin", {}).items()):
            lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `kg_result.json`: complete machine-readable result",
            "- `graph_nodes.csv`: node table",
            "- `graph_edges.csv`: edge table",
            "- `graph.cypher`: Cypher graph export",
        ]
    )
    return "\n".join(lines) + "\n"


def write_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "run_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "input_file",
                "status",
                "output_dir",
                "node_count",
                "edge_count",
                "runtime_seconds",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(1 for row in rows if row["status"] == "success")
    error_count = len(rows) - success_count
    lines = [
        "# Batch Industrial Text-to-KG Run Summary",
        "",
        f"- Files processed: `{len(rows)}`",
        f"- Successful files: `{success_count}`",
        f"- Failed files: `{error_count}`",
        "",
        "## Files",
        "",
        "| Input | Status | Nodes | Edges | Output | Error |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{Path(row['input_file']).name}` | "
            f"{row['status']} | "
            f"{row['node_count']} | "
            f"{row['edge_count']} | "
            f"`{row['output_dir']}` | "
            f"{row['error']} |"
        )
    lines.append("")
    lines.append("Open each output folder to inspect `summary.md`, `kg_result.json`, CSV files, and Cypher export.")
    (output_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
