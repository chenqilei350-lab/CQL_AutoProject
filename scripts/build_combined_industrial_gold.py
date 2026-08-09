#!/usr/bin/env python3
"""Build the combined reviewed industrial Gold dataset without altering sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.datasets.industrial_reviewed_gold import (
    DEFAULT_REVIEWED_GOLD_PATH,
    V2_REVIEWED_GOLD_PATH,
    V3_REVIEWED_GOLD_PATH,
    load_industrial_reviewed_gold_dataset,
)


def read_accepted_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        if record.get("review_status") != "accepted":
            raise ValueError(f"Non-accepted record at {path}:{line_number}")
        records.append(record)
    return records


def combine_reviewed_gold(input_paths: list[Path], output_path: Path) -> dict[str, int]:
    combined: list[dict[str, Any]] = []
    seen_scene_ids: set[str] = set()
    entity_count = 0
    relation_count = 0

    for path in input_paths:
        # The project adapter validates labels, IDs, relation signatures, and evidence fields.
        dataset = load_industrial_reviewed_gold_dataset(path)
        records = read_accepted_records(path)
        if len(dataset.scenes) != len(records):
            raise ValueError(f"Adapter record mismatch for {path}")
        for record in records:
            scene_id = str(record["scene_id"])
            if scene_id in seen_scene_ids:
                raise ValueError(f"Duplicate scene_id across reviewed datasets: {scene_id}")
            seen_scene_ids.add(scene_id)
            entity_count += len(record.get("gold_entities", []))
            relation_count += len(record.get("gold_relations", []))
            combined.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in combined),
        encoding="utf-8",
    )
    # Validate the emitted artifact through the same adapter used by experiments.
    emitted = load_industrial_reviewed_gold_dataset(output_path)
    if len(emitted.scenes) != len(combined):
        raise ValueError("Combined reviewed Gold validation failed")
    return {
        "scene_count": len(combined),
        "entity_count": entity_count,
        "relation_count": relation_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(V2_REVIEWED_GOLD_PATH), str(V3_REVIEWED_GOLD_PATH)],
    )
    parser.add_argument("--output", default=str(DEFAULT_REVIEWED_GOLD_PATH))
    args = parser.parse_args()

    summary = combine_reviewed_gold(
        [Path(value) for value in args.inputs],
        Path(args.output),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Combined Gold: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
