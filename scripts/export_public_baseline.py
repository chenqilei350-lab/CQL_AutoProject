"""Export a license-safe, aggregate-only copy of a completed baseline.

The canonical baseline remains local because its cleaned records and raw model
responses can contain licensed source text. This exporter copies only audited
metric files, removes local paths, and writes an independent public manifest
and checksum list suitable for a public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.evaluation.versioned_experiment import BaselineManifest, sha256_file
from scripts.run_versioned_baseline import verify_completed_baseline


BASELINE_ID = "industrial-v1-llama31-8b-20260809"
DEFAULT_SOURCE = Path("experiments/baselines") / BASELINE_ID
DEFAULT_OUTPUT = Path("experiments/public_baselines") / BASELINE_ID

PUBLIC_RESULT_FILES = (
    "cleaning_metrics.json",
    "preprocessing_metrics.csv",
    "end_to_end_summary.csv",
    "per_scene_metrics.csv",
    "relation_type_metrics.csv",
    "stability.csv",
    "hallucination_report.csv",
    "relation_diagnostic.csv",
    "relation_diagnostic_summary.json",
    "split_manifest.json",
)

PRIVATE_ARTIFACTS = {
    "cleaned_records.jsonl.gz": "Contains normalized licensed source records.",
    "raw_runs.jsonl.gz": "Contains prompts, source evidence, and raw model responses.",
    "baseline_manifest.json": "Contains local source-file paths and private fingerprints.",
}


def _aggregate_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Hash private fingerprints without exposing their file paths."""

    safe_rows = sorted(
        (str(row["sha256"]), int(row["size_bytes"])) for row in rows
    )
    payload = json.dumps(safe_rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sanitized_cleaning_metrics(source: Path) -> dict[str, Any]:
    metrics = json.loads(source.read_text(encoding="utf-8"))
    metrics.pop("root_path", None)
    metrics["privacy_note"] = (
        "The local source root was removed from this public aggregate."
    )
    return metrics


def _write_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"SHA256SUMS", "COMPLETED_PUBLIC"}:
            rows.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def verify_public_export(output_dir: str | Path) -> None:
    """Verify the public export marker, allowlist, and checksums."""

    directory = Path(output_dir)
    required = {
        *PUBLIC_RESULT_FILES,
        "public_manifest.json",
        "README.md",
        "SHA256SUMS",
        "COMPLETED_PUBLIC",
    }
    names = {path.name for path in directory.iterdir() if path.is_file()}
    missing = sorted(required - names)
    if missing:
        raise FileNotFoundError(f"Public baseline is incomplete; missing: {missing}")
    forbidden = sorted(set(PRIVATE_ARTIFACTS) & names)
    if forbidden:
        raise ValueError(f"Private artifacts found in public export: {forbidden}")

    issues = []
    for row in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = row.split("  ", 1)
        path = directory / name
        if not path.exists() or sha256_file(path) != expected:
            issues.append(name)
    if issues:
        raise ValueError(f"Public baseline checksum mismatch: {issues}")

    manifest_text = (directory / "public_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    if "source_files" in manifest or "root_path" in manifest_text:
        raise ValueError("Public manifest exposes private source metadata.")


def export_public_baseline(
    source_dir: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    verify_source: bool = True,
) -> Path:
    """Create one immutable public export from a completed local baseline."""

    source = Path(source_dir)
    output = Path(output_dir)
    if (output / "COMPLETED_PUBLIC").exists():
        raise FileExistsError(f"Completed public export is immutable: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty export: {output}")
    if verify_source:
        verify_completed_baseline(source)

    manifest = BaselineManifest.model_validate_json(
        (source / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True, exist_ok=True)

    for name in PUBLIC_RESULT_FILES:
        if name == "cleaning_metrics.json":
            safe_metrics = _sanitized_cleaning_metrics(source / name)
            (output / name).write_text(
                json.dumps(safe_metrics, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source / name, output / name)

    public_manifest = {
        "baseline_id": manifest.baseline_id,
        "baseline_tag": manifest.baseline_tag,
        "tested_code_commit": manifest.code_commit,
        "baseline_status": manifest.status,
        "exported_at": datetime.now(UTC).isoformat(),
        "module_versions": manifest.module_versions.model_dump(),
        "environment": manifest.environment.model_dump(),
        "parameters": manifest.parameters.model_dump(),
        "dataset": {
            "name": manifest.dataset_name,
            "scene_count": manifest.scene_count,
            "video_count": manifest.video_count,
            "gold_entity_count": manifest.gold_entity_count,
            "gold_relation_count": manifest.gold_relation_count,
        },
        "source_identity": {
            "file_count": len(manifest.source_files),
            "aggregate_sha256": _aggregate_fingerprint(
                [row.model_dump() for row in manifest.source_files]
            ),
            "paths_published": False,
        },
        "public_result_files": list(PUBLIC_RESULT_FILES),
        "private_artifacts_excluded": PRIVATE_ARTIFACTS,
        "notes": [
            "This export contains aggregate and per-scene numeric results only.",
            "Licensed source text and raw model responses remain in the local baseline.",
            "reviewed_unified is a human-derived upper bound.",
            "Raw and post-processed relation metrics are reported separately.",
        ],
    }
    (output / "public_manifest.json").write_text(
        json.dumps(public_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    readme = f"""# Public Industrial KG Baseline V1 Results

This directory is the license-safe public export of `{manifest.baseline_id}`.
It contains aggregate and per-scene numeric results for the completed 256-call
local LLM baseline. It deliberately excludes licensed IndEgo-derived text,
cleaned records, prompts, evidence excerpts, and raw model responses.

The canonical local artifact remains under `experiments/baselines/` and is
required when authorized researchers need raw-output rescoring. The source
identity in `public_manifest.json` is an aggregate hash, so authorized copies
can be checked without publishing local paths or source contents.

`reviewed_unified` is a human-derived upper-bound condition. It must not be
reported as the performance of automatic text preprocessing.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(output)
    (output / "COMPLETED_PUBLIC").write_text(
        datetime.now(UTC).isoformat() + "\n", encoding="utf-8"
    )
    verify_public_export(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_public_export(args.output)
        print(f"Verified public baseline export: {args.output}")
        return
    output = export_public_baseline(args.source, args.output)
    print(f"Created public baseline export: {output}")


if __name__ == "__main__":
    main()
