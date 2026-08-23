"""Build the integrated, evaluation-ready view of reviewed industrial Gold v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.datasets.industrial_gold import (
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_EXCLUSIONS_PATH,
    DEFAULT_REVIEWED_GOLD_PATH,
    load_industrial_gold_dataset,
)


DEFAULT_OUTPUT_DIR = Path("data/industrial_gold_v2")


def build_package(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    reviewed_path: str | Path = DEFAULT_REVIEWED_GOLD_PATH,
    exclusions_path: str | Path = DEFAULT_EXCLUSIONS_PATH,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
) -> dict[str, Any]:
    """Validate the reviewed records and emit deterministic benchmark artifacts."""

    dataset = load_industrial_gold_dataset(
        reviewed_path=reviewed_path,
        exclusions_path=exclusions_path,
        candidates_path=candidates_path,
    )
    benchmark = dataset.to_benchmark_dataset()

    package_dir = Path(output_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark.save_jsonl(package_dir / "benchmark.jsonl")

    manifest = dataset.manifest()
    manifest.update(
        {
            "benchmark_file": benchmark_path.name,
            "source_files": {
                "reviewed_gold": str(Path(reviewed_path)),
                "exclusions": str(Path(exclusions_path)),
                "candidate_raw_texts": str(Path(candidates_path)),
            },
        }
    )
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and build the unified Industrial Gold V2 evaluation package."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED_GOLD_PATH)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(
        args.output_dir,
        reviewed_path=args.reviewed,
        exclusions_path=args.exclusions,
        candidates_path=args.candidates,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
