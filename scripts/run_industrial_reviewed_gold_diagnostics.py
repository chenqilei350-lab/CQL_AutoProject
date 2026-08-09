#!/usr/bin/env python3
"""Diagnose relation-candidate coverage on all human-reviewed industrial Gold.

This is an oracle-entity diagnostic: candidate generation receives the reviewed
entities so that relation-stage limitations can be measured independently from
entity-extraction errors.  It does not claim end-to-end LLM performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.datasets.industrial_reviewed_gold import (
    DEFAULT_REVIEWED_GOLD_PATH,
    SCORING_RELATION_TYPES,
    load_industrial_reviewed_gold_dataset,
)
from backend.evaluation.metrics import PRF1, compute_prf1
from backend.pipeline.relation_candidate_pipeline import (
    DeterministicRelationCandidateScorer,
    EntityMention,
    ProceduralRelationCandidateGenerator,
)
from backend.schemas.egocentric_video import EgocentricVideoExtraction


EdgeKey = tuple[str, str, str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reviewed-Gold relation candidate coverage diagnostics."
    )
    parser.add_argument("--input", default=str(DEFAULT_REVIEWED_GOLD_PATH))
    parser.add_argument(
        "--output-dir",
        default="results/industrial_reviewed_gold_combined_relation_diagnostic_2026-08-01",
    )
    args = parser.parse_args()

    dataset = load_industrial_reviewed_gold_dataset(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path("docs/results")
    docs_dir.mkdir(parents=True, exist_ok=True)

    generator = ProceduralRelationCandidateGenerator()
    scorer = DeterministicRelationCandidateScorer()
    scene_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    gold_by_type: Counter[str] = Counter()
    candidate_by_type: Counter[str] = Counter()
    matched_by_type: Counter[str] = Counter()
    accepted_by_type: Counter[str] = Counter()
    accepted_matched_by_type: Counter[str] = Counter()

    for scene in dataset.scenes:
        gold_edges = gold_edge_set(scene.gold_extraction)
        actions = [
            EntityMention(
                entity_id=action.entity_id or f"action_{index}",
                label="Action",
                name=action.name,
                evidence_text=action.evidence_text or "",
                sequence_index=action.sequence_index,
            )
            for index, action in enumerate(scene.gold_extraction.actions, 1)
        ]
        tools = [
            EntityMention(
                entity_id=tool.entity_id or f"tool_{index}",
                label="Tool",
                name=tool.name,
                evidence_text=tool.evidence_text or "",
            )
            for index, tool in enumerate(scene.gold_extraction.tools, 1)
        ]
        objects = [
            EntityMention(
                entity_id=obj.entity_id or f"object_{index}",
                label="Object",
                name=obj.name,
                evidence_text=obj.evidence_text or "",
            )
            for index, obj in enumerate(scene.gold_extraction.objects, 1)
        ]
        candidates = generator.generate(
            actions=actions,
            tools=tools,
            objects=objects,
            source_text=scene.raw_text,
        )
        report = scorer.score(candidates, source_text=scene.raw_text)
        candidate_edges = {
            (candidate.subject_id, candidate.relation_type, candidate.object_id)
            for candidate in candidates
            if candidate.relation_type in SCORING_RELATION_TYPES
        }
        accepted_edges = {
            (candidate.subject_id, candidate.relation_type, candidate.object_id)
            for candidate in report.accepted_candidates()
            if candidate.relation_type in SCORING_RELATION_TYPES
        }

        candidate_metric = edge_metric(gold_edges, candidate_edges)
        accepted_metric = edge_metric(gold_edges, accepted_edges)
        scene_rows.append(
            {
                "scene_id": scene.scene_id,
                "gold_edges": len(gold_edges),
                "candidate_edges": len(candidate_edges),
                "candidate_precision": candidate_metric.precision,
                "candidate_recall": candidate_metric.recall,
                "candidate_f1": candidate_metric.f1,
                "accepted_edges": len(accepted_edges),
                "accepted_precision": accepted_metric.precision,
                "accepted_recall": accepted_metric.recall,
                "accepted_f1": accepted_metric.f1,
                "missed_gold_edges": len(gold_edges - candidate_edges),
            }
        )

        for relation_type in SCORING_RELATION_TYPES:
            gold_typed = {edge for edge in gold_edges if edge[1] == relation_type}
            candidate_typed = {
                edge for edge in candidate_edges if edge[1] == relation_type
            }
            accepted_typed = {edge for edge in accepted_edges if edge[1] == relation_type}
            gold_by_type[relation_type] += len(gold_typed)
            candidate_by_type[relation_type] += len(candidate_typed)
            matched_by_type[relation_type] += len(gold_typed & candidate_typed)
            accepted_by_type[relation_type] += len(accepted_typed)
            accepted_matched_by_type[relation_type] += len(gold_typed & accepted_typed)

        names = entity_names(scene.gold_extraction)
        for source_id, relation_type, target_id in sorted(gold_edges - candidate_edges):
            missed_rows.append(
                {
                    "scene_id": scene.scene_id,
                    "source_id": source_id,
                    "source_name": names.get(source_id, ""),
                    "relation_type": relation_type,
                    "target_id": target_id,
                    "target_name": names.get(target_id, ""),
                }
            )

    gold_all = sum(gold_by_type.values())
    candidate_all = sum(candidate_by_type.values())
    matched_all = sum(matched_by_type.values())
    accepted_all = sum(accepted_by_type.values())
    accepted_matched_all = sum(accepted_matched_by_type.values())
    overall_candidate = compute_prf1(
        "candidate_edges",
        true_positives=matched_all,
        false_positives=candidate_all - matched_all,
        false_negatives=gold_all - matched_all,
    )
    overall_accepted = compute_prf1(
        "accepted_edges",
        true_positives=accepted_matched_all,
        false_positives=accepted_all - accepted_matched_all,
        false_negatives=gold_all - accepted_matched_all,
    )
    relation_rows = []
    for relation_type in sorted(SCORING_RELATION_TYPES):
        metric = compute_prf1(
            relation_type,
            true_positives=accepted_matched_by_type[relation_type],
            false_positives=(
                accepted_by_type[relation_type]
                - accepted_matched_by_type[relation_type]
            ),
            false_negatives=(
                gold_by_type[relation_type]
                - accepted_matched_by_type[relation_type]
            ),
        )
        relation_rows.append(
            {
                "relation_type": relation_type,
                "gold_edges": gold_by_type[relation_type],
                "candidate_edges": candidate_by_type[relation_type],
                "matched_gold_edges": matched_by_type[relation_type],
                "candidate_coverage_recall": (
                    matched_by_type[relation_type] / gold_by_type[relation_type]
                    if gold_by_type[relation_type]
                    else 0.0
                ),
                "accepted_precision": metric.precision,
                "accepted_recall": metric.recall,
                "accepted_f1": metric.f1,
            }
        )

    summary = {
        "dataset": dataset.name,
        "diagnostic_mode": "gold_entity_oracle_relation_candidate",
        "scene_count": len(dataset.scenes),
        "gold_entity_count": sum(
            len(scene.gold_extraction.actions)
            + len(scene.gold_extraction.objects)
            + len(scene.gold_extraction.tools)
            for scene in dataset.scenes
        ),
        "gold_edge_count": gold_all,
        "candidate_edge_count": candidate_all,
        "candidate_coverage_recall": overall_candidate.recall,
        "candidate_precision": overall_candidate.precision,
        "candidate_f1": overall_candidate.f1,
        "deterministic_accepted_edge_count": accepted_all,
        "deterministic_precision": overall_accepted.precision,
        "deterministic_recall": overall_accepted.recall,
        "deterministic_f1": overall_accepted.f1,
        "missed_gold_edge_count": gold_all - matched_all,
    }

    write_csv(output_dir / "scene_metrics.csv", scene_rows)
    write_csv(output_dir / "relation_type_metrics.csv", relation_rows)
    write_csv(output_dir / "missed_gold_edges.csv", missed_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_report(summary, relation_rows, scene_rows)
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    report_path = docs_dir / f"{output_dir.name}.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path.resolve()}")


def gold_edge_set(extraction: EgocentricVideoExtraction) -> set[EdgeKey]:
    edges: set[EdgeKey] = set()
    for relation in extraction.action_order:
        edges.add((required_id(relation.before), "BEFORE", required_id(relation.after)))
    for relation in extraction.acts_on_object:
        edges.add((required_id(relation.action), "ACTS_ON", required_id(relation.object)))
    for relation in extraction.uses_tool:
        edges.add((required_id(relation.action), "USES_TOOL", required_id(relation.tool)))
    return edges


def entity_names(extraction: EgocentricVideoExtraction) -> dict[str, str]:
    return {
        required_id(entity): entity.name
        for entity in [*extraction.actions, *extraction.objects, *extraction.tools]
    }


def required_id(entity: Any) -> str:
    if not entity.entity_id:
        raise ValueError(f"Reviewed Gold entity lacks entity_id: {entity.name}")
    return entity.entity_id


def edge_metric(gold: set[EdgeKey], predicted: set[EdgeKey]) -> PRF1:
    return compute_prf1(
        "edges",
        true_positives=len(gold & predicted),
        false_positives=len(predicted - gold),
        false_negatives=len(gold - predicted),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    summary: dict[str, Any],
    relation_rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Industrial Reviewed Gold Relation Diagnostic",
        "",
        "This diagnostic gives the relation-candidate generator the human-reviewed Gold entities. It therefore isolates relation-stage coverage and is not an end-to-end LLM score.",
        "",
        f"- Accepted scenes: `{summary['scene_count']}`",
        f"- Reviewed entities: `{summary['gold_entity_count']}`",
        f"- Reviewed relations: `{summary['gold_edge_count']}`",
        f"- Candidate coverage recall: `{summary['candidate_coverage_recall']:.4f}`",
        f"- Deterministic accepted Precision / Recall / F1: `{summary['deterministic_precision']:.4f}` / `{summary['deterministic_recall']:.4f}` / `{summary['deterministic_f1']:.4f}`",
        f"- Missed reviewed relations: `{summary['missed_gold_edge_count']}`",
        "",
        "## Relation-Type Results",
        "",
        "| Relation | Gold | Candidates | Matched | Coverage recall | Accepted precision | Accepted recall | Accepted F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in relation_rows:
        lines.append(
            "| {relation_type} | {gold_edges} | {candidate_edges} | {matched_gold_edges} | "
            "{candidate_coverage_recall:.4f} | {accepted_precision:.4f} | "
            "{accepted_recall:.4f} | {accepted_f1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Scene Results",
            "",
            "| Scene | Gold edges | Candidates | Coverage recall | Accepted precision | Accepted F1 | Missed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scene_rows:
        lines.append(
            "| {scene_id} | {gold_edges} | {candidate_edges} | {candidate_recall:.4f} | "
            "{accepted_precision:.4f} | {accepted_f1:.4f} | {missed_gold_edges} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Candidate coverage recall is the maximum recall available to any later binary judge when entity extraction is perfect.",
            "- The current deterministic scorer accepts every generated candidate, so its low precision measures over-generation rather than LLM hallucination.",
            "- Missing candidates require generator changes; false-positive candidates require a stronger scorer or binary judge.",
            "- End-to-end performance will be lower when the model omits or renames entity endpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
