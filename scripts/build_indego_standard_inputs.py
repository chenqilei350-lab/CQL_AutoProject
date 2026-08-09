#!/usr/bin/env python3
"""Build KG-ready standard inputs from local IndEgo text layers."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.datasets.indego_adapter import (
    DEFAULT_INDEGO_STANDARD_OUTPUT,
    DEFAULT_INDEGO_TEXT_ROOT,
    DEFAULT_MAX_SEGMENTS_PER_SCENE,
    DEFAULT_MAX_TRANSCRIPT_CHARS,
    export_indego_standard_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert downloaded IndEgo text layers into standard raw/unified KG inputs."
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_INDEGO_TEXT_ROOT),
        help="Local folder containing downloaded IndEgo text layers.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_INDEGO_STANDARD_OUTPUT),
        help="JSONL output path for standard input records.",
    )
    parser.add_argument(
        "--max-segments-per-scene",
        type=int,
        default=DEFAULT_MAX_SEGMENTS_PER_SCENE,
        help="Maximum action/keystep segments per generated scene chunk.",
    )
    parser.add_argument(
        "--max-transcript-chars",
        type=int,
        default=DEFAULT_MAX_TRANSCRIPT_CHARS,
        help="Maximum transcript characters copied into each standard input.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of standard scenes to export.",
    )
    parser.add_argument(
        "--no-transcript-only",
        action="store_true",
        help="Skip videos that only have narration transcripts and no temporal annotations.",
    )
    parser.add_argument(
        "--no-warning-scenes",
        action="store_true",
        help="Skip mistake/warning-derived standard scenes.",
    )
    args = parser.parse_args()

    dataset = export_indego_standard_inputs(
        source_dir=args.source_dir,
        output_path=args.output,
        max_segments_per_scene=args.max_segments_per_scene,
        max_transcript_chars=args.max_transcript_chars,
        include_transcript_only=not args.no_transcript_only,
        include_warning_scenes=not args.no_warning_scenes,
        limit=args.limit,
    )
    output = Path(args.output).resolve()
    print(f"Exported scenes: {len(dataset.scenes)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
