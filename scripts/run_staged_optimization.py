#!/usr/bin/env python3
"""Run staged KG extraction optimization experiments.

Recommended workflow:

1. smoke mode: verify hospital reference summary and egocentric CV plumbing.
2. real hospital stage: run hospital stress tests with Ollama.
3. real cv stage: run egocentric cross-validation with the selected strategy.
4. real_data stage: reserved for external standardized text once delivered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.pipeline.staged_optimization import (
    report_to_markdown,
    run_staged_optimization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run staged optimization for AUT KG extraction."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "real"],
        default="smoke",
        help="smoke avoids real LLM calls; real runs configured model stages.",
    )
    parser.add_argument(
        "--stage",
        choices=["hospital", "cv", "real_data", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/staged_optimization_smoke",
        help="Output directory for JSON/CSV reports.",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model for real-mode hospital/reference stages.",
    )
    parser.add_argument(
        "--hospital-timeout",
        type=int,
        default=120,
        help="Per-call timeout for hospital reference LLM calls.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=[
            "baseline_one_stage",
            "strict_schema_prompt",
            "entity_first_relation_second",
            "entity_first_relation_second_with_validation",
            "entity_first_relation_second_with_fewshot",
            "minimal_entity_relation",
            "minimal_entity_relation_with_validation",
            "minimal_candidate_relation",
            "minimal_candidate_relation_with_validation",
            "validation_driven_refinement",
        ],
        help=(
            "Algorithm condition for CV. Repeat to compare multiple conditions. "
            "Default is minimal_entity_relation_with_validation."
        ),
    )
    args = parser.parse_args()

    report = run_staged_optimization(
        mode=args.mode,
        stage=args.stage,
        output_dir=args.output_dir,
        model=args.model,
        hospital_timeout=args.hospital_timeout,
        cv_conditions=tuple(args.condition)
        if args.condition
        else ("minimal_entity_relation_with_validation",),
    )
    print(report_to_markdown(report))
    print(f"\nReport saved to: {(Path(args.output_dir) / 'staged_optimization_report.json').resolve()}")


if __name__ == "__main__":
    main()
