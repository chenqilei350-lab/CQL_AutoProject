#!/usr/bin/env python3
"""Select IndEgo scenes and generate source-supported gold annotation templates.

The generated files are candidate/silver templates only. They are not final
gold graphs until a human reviewer confirms every node, edge, and evidence
span.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.pipeline.industrial_text_to_kg import classify_industrial_object, normalize_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate human-review templates for industrial gold graph expansion."
    )
    parser.add_argument("--input", default="kg_ready_data/indego_standard_inputs.jsonl")
    parser.add_argument("--output-dir", default="data/industrial_gold_candidates")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    selected = select_candidates(records, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = [build_annotation_template(record, rank=index + 1) for index, record in enumerate(selected)]
    write_jsonl(output_dir / "industrial_gold_candidate_templates.jsonl", templates)
    (output_dir / "README.md").write_text(build_readme(args.input, len(templates)), encoding="utf-8")
    print(f"Selected scenes: {len(templates)}")
    print(f"Templates: {(output_dir / 'industrial_gold_candidate_templates.jsonl').resolve()}")
    print(f"README: {(output_dir / 'README.md').resolve()}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def select_candidates(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pick diverse, information-rich scenes deterministically."""

    scored = sorted(
        records,
        key=lambda record: (-candidate_score(record), str(record.get("category", "")), str(record.get("scene_id", ""))),
    )
    selected: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    for record in scored:
        category = str(record.get("category") or "unknown")
        if category_counts.get(category, 0) >= max(2, limit // 5):
            continue
        if not is_usable(record):
            continue
        selected.append(record)
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(selected) >= limit:
            return selected

    for record in scored:
        if len(selected) >= limit:
            break
        if record in selected or not is_usable(record):
            continue
        selected.append(record)
    return selected


def is_usable(record: dict[str, Any]) -> bool:
    unified = record.get("unified_record") or {}
    actions = unified.get("action_sequence") or []
    objects = unified.get("tools_objects") or []
    raw_text = str(record.get("raw_text") or "")
    return len(actions) >= 2 and len(objects) >= 1 and len(raw_text) >= 120


def candidate_score(record: dict[str, Any]) -> int:
    unified = record.get("unified_record") or {}
    actions = unified.get("action_sequence") or []
    objects = unified.get("tools_objects") or []
    raw_text = str(record.get("raw_text") or "")
    return len(actions) * 4 + len(objects) * 2 + min(len(raw_text) // 300, 10)


def build_annotation_template(record: dict[str, Any], rank: int) -> dict[str, Any]:
    unified = record.get("unified_record") or {}
    raw_text = str(record.get("raw_text") or unified.get("source_text") or "")
    action_entries = [entry for entry in unified.get("action_sequence", []) if isinstance(entry, dict)]
    object_entries = [entry for entry in unified.get("tools_objects", []) if isinstance(entry, dict)]

    nodes: list[dict[str, Any]] = [
        {
            "id": f"action_{index}",
            "label": "Action",
            "name": clean(entry.get("text")),
            "evidence_text": clean(entry.get("evidence")),
            "review_status": "candidate_needs_human_review",
        }
        for index, entry in enumerate(action_entries, 1)
        if clean(entry.get("text"))
    ]
    for index, entry in enumerate(object_entries, 1):
        name = clean(entry.get("text"))
        evidence = clean(entry.get("evidence"))
        if not name:
            continue
        label = classify_industrial_object(
            entity_id=f"object_{index}",
            name=name,
            object_type="",
            evidence_text=evidence,
        )
        nodes.append(
            {
                "id": f"{label.lower()}_{index}",
                "label": label,
                "name": name,
                "evidence_text": evidence,
                "review_status": "candidate_needs_human_review",
            }
        )

    edges = build_candidate_edges(nodes)
    return {
        "status": "candidate_gold_not_reviewed",
        "rank": rank,
        "scene_id": record.get("scene_id"),
        "video_id": record.get("video_id"),
        "category": record.get("category"),
        "source_paths": record.get("source_paths", []),
        "raw_text": raw_text,
        "unified_record": unified,
        "candidate_gold_nodes": nodes,
        "candidate_gold_edges": edges,
        "review_instructions": (
            "Keep only source-supported facts. Delete unsupported candidate nodes/edges. "
            "Do not add inferred facts unless the evidence text explicitly supports them."
        ),
    }


def build_candidate_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = [node for node in nodes if node["label"] == "Action"]
    physical_nodes = [node for node in nodes if node["label"] in {"Tool", "Object"}]
    edges: list[dict[str, Any]] = []

    for before, after in zip(actions, actions[1:]):
        edges.append(
            {
                "source": before["id"],
                "relation": "BEFORE",
                "target": after["id"],
                "evidence_text": before.get("evidence_text") or after.get("evidence_text") or "",
                "review_status": "candidate_needs_human_review",
            }
        )

    for action in actions:
        action_text = normalize_name(f"{action.get('name', '')} {action.get('evidence_text', '')}")
        for node in physical_nodes:
            node_name = normalize_name(str(node.get("name", "")))
            node_evidence = normalize_name(str(node.get("evidence_text", "")))
            if not node_name:
                continue
            if node_name not in action_text and node_evidence != normalize_name(str(action.get("evidence_text", ""))):
                continue
            edges.append(
                {
                    "source": action["id"],
                    "relation": "USES_TOOL" if node["label"] == "Tool" else "ACTS_ON",
                    "target": node["id"],
                    "evidence_text": action.get("evidence_text") or node.get("evidence_text") or "",
                    "review_status": "candidate_needs_human_review",
                }
            )
    return edges


def build_readme(input_path: str, count: int) -> str:
    return "\n".join(
        [
            "# Industrial Gold Candidate Templates",
            "",
            f"- Source input: `{input_path}`",
            f"- Candidate scenes: `{count}`",
            "- Status: `candidate_gold_not_reviewed`",
            "",
            "These files are annotation templates, not final gold graphs.",
            "A final gold graph must contain only source-supported facts verified by a human reviewer.",
            "",
            "Review rules:",
            "",
            "1. Keep a node or edge only if its evidence text is present in the source text.",
            "2. Do not add facts inferred from domain knowledge alone.",
            "3. Mark confirmed facts as `reviewed_source_supported`.",
            "4. Save reviewed examples separately, for example as `industrial_gold_v2_reviewed.jsonl`.",
            "",
        ]
    )


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


if __name__ == "__main__":
    main()
