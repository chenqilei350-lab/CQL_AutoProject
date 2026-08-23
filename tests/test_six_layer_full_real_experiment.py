"""Focused tests for full-run success-only and end-to-end reporting."""

from backend.evaluation.metrics import compute_prf1
from scripts.run_six_layer_full_real_experiment import summarize_rows


def _metric(tp: int, fp: int, fn: int) -> dict[str, object]:
    return compute_prf1("test", tp, fp, fn).model_dump(mode="json")


def _row(*, failed: bool) -> dict[str, object]:
    metric = _metric(0, 0, 1) if failed else _metric(1, 0, 0)
    return {
        "execution_error": "timeout" if failed else None,
        "runtime_seconds": 120.0 if failed else 10.0,
        "preprocessing_runtime_seconds": 5.0,
        "node_metrics": metric,
        "relation_metrics": metric,
        "core_exact": metric,
        "core_aligned": metric,
        "proposal_count": 0 if failed else 1,
        "accepted_count": 0 if failed else 1,
    }


def test_summary_reports_success_only_and_end_to_end_separately() -> None:
    rows = [_row(failed=False), _row(failed=True)]

    summary = summarize_rows("test", rows, [rows[0]])

    assert summary["success_only"]["relation_metrics"]["f1"] == 1.0
    assert summary["end_to_end"]["relation_metrics"]["f1"] == 0.6667
    assert summary["relation_metrics"] == summary["success_only"][
        "relation_metrics"
    ]
