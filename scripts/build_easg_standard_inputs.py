#!/usr/bin/env python3
"""Build annotation-only EASG standard inputs.

This script reads local EASG JSON annotations and exports project-standard
raw/unified/gold graph records.  It never downloads or opens video files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.datasets.easg_adapter import (
    DEFAULT_EASG_ANNOTATION_ROOT,
    DEFAULT_EASG_STANDARD_OUTPUT,
    load_easg_annotation_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KG-ready records from local EASG graph annotations only."
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_EASG_ANNOTATION_ROOT),
        help="Local EASG annotation folder. Videos are not read.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EASG_STANDARD_OUTPUT),
        help="JSONL output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of annotation records to export.",
    )
    args = parser.parse_args()

    dataset = load_easg_annotation_dataset(args.source_dir, limit=args.limit)
    output = dataset.save_jsonl(args.output)
    skipped = sum(len(scene.skipped_relations) for scene in dataset.scenes)

    print(f"Source: {Path(args.source_dir).resolve()}")
    print(f"Output: {output.resolve()}")
    print(f"Scenes: {len(dataset.scenes)}")
    print(f"Skipped unsupported relations: {skipped}")
    print("Video files were not downloaded, opened, or decoded.")


if __name__ == "__main__":
    main()
