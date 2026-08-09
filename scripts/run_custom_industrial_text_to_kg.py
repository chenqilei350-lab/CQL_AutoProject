#!/usr/bin/env python3
"""Convert one industrial text file into a lightweight procedural KG.

This is the handoff-friendly entry point for teammates: provide a plain text
scene description and get a structured extraction plus property graph JSON.
It uses the same lightweight JSON contract as the final experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.datasets.benchmark import BenchmarkScene
from backend.preprocessing.unified_text import GroundedEntry, build_industrial_unified_text
from backend.schemas.egocentric_video import EgocentricVideoExtraction
from backend.graph.property_graph import build_property_graph
from backend.pipeline.industrial_text_to_kg import summarize_relation_origins
from scripts.run_easg_real_llm_lightweight_experiments import (
    extract_scene,
    get_openai_client,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an industrial scene text file into a procedural KG JSON."
    )
    parser.add_argument("--input", required=True, help="Plain text input file.")
    parser.add_argument(
        "--output",
        default="results/custom_industrial_kg_output.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--condition",
        choices=["raw", "unified"],
        default="unified",
        help="Send raw text or generated unified text to the LLM.",
    )
    parser.add_argument(
        "--strategy",
        choices=["one_shot", "layered"],
        default="layered",
        help="Use one-shot extraction or layered entity-then-relation extraction.",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name.")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()

    input_path = Path(args.input)
    raw_text = input_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Input file is empty: {input_path}")

    scene = build_custom_scene(raw_text, input_path.stem)
    client = get_openai_client(timeout=args.timeout)
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

    output = {
        "input_file": str(input_path),
        "condition": args.condition,
        "strategy": args.strategy,
        "model": args.model,
        "raw_response": raw_response,
        "extraction": extraction.model_dump(mode="json"),
        "relation_origin_summary": summarize_relation_origins(graph.edges).__dict__,
        "graph": graph.model_dump(mode="json"),
        "cypher": graph.to_cypher(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")


def build_custom_scene(raw_text: str, scene_id: str) -> BenchmarkScene:
    """Build a minimal scene object for the shared lightweight runner."""

    unified = build_industrial_unified_text(
        raw_text=raw_text,
        scene_id=scene_id,
        segment_id="custom_s1",
        timestamp=None,
        scene="custom industrial scene description",
        action_sequence=[GroundedEntry(text=raw_text[:160], evidence=raw_text[:160])],
        tools_objects=[],
        uncertainty=[
            "Automatically wrapped custom input; no new facts were added by preprocessing."
        ],
    )
    gold = EgocentricVideoExtraction(
        video_id=scene_id,
        source_text=raw_text,
    )
    return BenchmarkScene(
        scene_id=scene_id,
        description="Custom industrial text input",
        raw_text=raw_text,
        unified_record=unified,
        gold_extraction=gold,
    )


if __name__ == "__main__":
    main()
