#!/usr/bin/env python3
"""Run KG extraction on IndEgo standard input JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.datasets.indego_adapter import (
    DEFAULT_INDEGO_STANDARD_OUTPUT,
    IndEgoStandardDataset,
)
from backend.evaluation.ontology_validation import validate_egocentric_extraction
from backend.graph.property_graph import build_property_graph
from backend.pipeline.algorithm_experiment import (
    AlgorithmCondition,
    AlgorithmExperimentConfig,
    AlgorithmExperimentRunner,
)

ALGORITHM_CONDITIONS: tuple[AlgorithmCondition, ...] = (
    "baseline_one_stage",
    "strict_schema_prompt",
    "entity_first_relation_second",
    "entity_first_relation_second_with_validation",
    "entity_first_relation_second_with_fewshot",
    "minimal_entity_relation",
    "minimal_entity_relation_with_validation",
    "llm_relation_proposal",
    "llm_relation_proposal_with_repair",
    "minimal_candidate_relation",
    "minimal_candidate_relation_with_validation",
    "validation_driven_refinement",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract egocentric KG records from IndEgo standard inputs."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INDEGO_STANDARD_OUTPUT),
        help="JSONL file produced by scripts/build_indego_standard_inputs.py.",
    )
    parser.add_argument(
        "--output",
        default="results/indego_kg_extractions.jsonl",
        help="JSONL output path for extraction, graph, validation, and trace records.",
    )
    parser.add_argument(
        "--input-condition",
        choices=["raw", "unified"],
        default="unified",
        help="Which standard input representation to send to the extractor.",
    )
    parser.add_argument(
        "--condition",
        choices=ALGORITHM_CONDITIONS,
        default="minimal_candidate_relation_with_validation",
        help="Extraction algorithm condition.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-call timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of scenes to extract.",
    )
    args = parser.parse_args()

    dataset = IndEgoStandardDataset.from_jsonl(args.input)
    runner = AlgorithmExperimentRunner(
        config=AlgorithmExperimentConfig(
            model=args.model,
            timeout=args.timeout,
            conditions=(args.condition,),
            input_condition=args.input_condition,
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for scene in dataset.scenes[: args.limit]:
        input_text = scene.input_text(args.input_condition)  # type: ignore[arg-type]
        extraction = runner.extract_text(
            input_text,
            source_text=scene.raw_text,
            condition=args.condition,
        )
        graph = build_property_graph(extraction)
        validation = validate_egocentric_extraction(extraction, scene.raw_text)
        records.append(
            {
                "scene_id": scene.scene_id,
                "video_id": scene.video_id,
                "input_condition": args.input_condition,
                "condition": args.condition,
                "input_text": input_text,
                "extraction": extraction.model_dump(mode="json"),
                "graph": graph.model_dump(mode="json"),
                "validation": validation.model_dump(mode="json"),
                "debug_trace": runner.last_debug_trace.model_dump(mode="json")
                if runner.last_debug_trace
                else None,
            }
        )

    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"Extracted scenes: {len(records)}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
