#!/usr/bin/env python3
"""
Hospital gold-case KG extraction experiments (legacy reference entry file).

This script is retained as a hospital reference/stress-test track.  It is not
the default AUT/IndEgo egocentric experiment path.  Use it to compare algorithm
variables such as schema constraints, chunking, canonical IDs, and relation
direction before transferring promising settings to the egocentric main track.

This file is intentionally kept as a thin runnable entry point. It reuses the
same implementation as the Chinese-commented script:

    scripts/run_hospital_gold_pilot_experiments_zh.py

Why reuse the same implementation?
    The experiments call local LLMs and produce quantitative results. Keeping
    two independent implementations would make it easy for the Chinese and
    English versions to diverge. This English file therefore documents the
    experiment modules in English and delegates execution to the shared,
    fully commented implementation.

Experiment modules included in the shared implementation:
    P1  Minimal strict extraction sanity check.
        Tests whether the model, JSON output, schema following, and metric
        calculation work on a small 7-node / 6-edge graph.

    P1 Stress  Prompt constraint strength.
        Compares a loose prompt with a strict ontology/schema-guided prompt.
        This exposes relation-label and relation-direction errors.

    P2  One-shot vs layered extraction.
        Compares full-graph extraction in one prompt with layer-by-layer
        extraction and graph merging.

    P3  LabEvent chunk-size experiment.
        Tests chunk sizes of 12, 6, 3, and 1 rows for a narrow LabEvent task.

    P3 Stress  Dense narrative chunk-size experiment.
        Uses more difficult narrative-style lab text and extracts both
        LabEvent/LabItem nodes and relations.

    P4  Raw vs unified LabEvent input.
        Compares raw clinical-note style text with a field-structured input.

    P4 Stress  Raw noisy text vs unified canonical structure.
        Tests whether explicit canonical IDs, labels, and relation directions
        improve graph construction.

    P5  Model comparison.
        Compares local small models under the same mini-graph extraction task.

    P6  Repeated-run stability.
        Runs repeated extractions and computes node overlap, relation
        agreement, graph overlap, and F1 variation.

Usage:
    python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite stress
    python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite all

Requirements:
    - Python 3
    - Ollama running locally at http://localhost:11434
    - A pulled local model, for example: ollama pull llama3.1:8b
    - The gold-case directory: gold_case_admission_29366372
"""

from __future__ import annotations

from pathlib import Path
import sys


# Make sure Python can import the sibling Chinese implementation file when this
# script is executed from any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# Reuse the shared implementation so the Chinese and English entry points always
# produce identical experiment behavior and identical metrics.
from run_hospital_gold_pilot_experiments_zh import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
