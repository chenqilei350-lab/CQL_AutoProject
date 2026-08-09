"""Cross-validation helpers for KG extraction experiments.

The project uses cross-validation to separate algorithm selection from final
data-cleaning and text-conversion integration.  Small egocentric benchmarks use
leave-one-scene-out by default; larger reference datasets can use deterministic
K-fold splits.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from statistics import mean, pstdev
from typing import Callable, Iterable, Literal

from pydantic import BaseModel, Field, model_validator

from backend.datasets.benchmark import BenchmarkDataset, BenchmarkScene


ExperimentTrack = Literal["egocentric_main", "hospital_reference"]
SplitStrategy = Literal["kfold", "leave_one_scene_out"]


class CrossValidationConfig(BaseModel):
    """Configuration for dataset-level cross-validation."""

    track: ExperimentTrack = "egocentric_main"
    split_strategy: SplitStrategy = "leave_one_scene_out"
    k: int | None = None
    seed: int = 42
    repetitions: int = Field(default=1, ge=1)
    conditions: tuple[str, ...] = ("baseline_one_stage",)

    @model_validator(mode="after")
    def validate_k(self) -> "CrossValidationConfig":
        """K-fold requires a useful fold count."""

        if self.split_strategy == "kfold" and (self.k is None or self.k < 2):
            raise ValueError("kfold cross-validation requires k >= 2.")
        return self


class DatasetFold(BaseModel):
    """One train/test split."""

    fold_id: str
    train_scene_ids: list[str]
    test_scene_ids: list[str]

    def contains_test_scene(self, scene_id: str) -> bool:
        """Return whether a scene belongs to this fold's held-out set."""

        return scene_id in self.test_scene_ids


class FoldConditionResult(BaseModel):
    """Metrics for one condition on one fold."""

    fold_id: str
    track: ExperimentTrack
    condition: str
    test_scene_ids: list[str]
    node_f1: float = 0.0
    relation_f1: float = 0.0
    ontology_conformance: float = 0.0
    grounding_rate: float = 0.0
    hallucination_rate: float = 0.0
    schema_success_rate: float = 0.0
    runtime_seconds: float = 0.0


class ConditionSummary(BaseModel):
    """Mean/std summary across folds for one condition."""

    track: ExperimentTrack
    condition: str
    fold_count: int
    node_f1_mean: float
    node_f1_std: float
    relation_f1_mean: float
    relation_f1_std: float
    ontology_conformance_mean: float
    grounding_rate_mean: float
    hallucination_rate_mean: float
    schema_success_rate_mean: float
    runtime_seconds_mean: float


class CrossValidationReport(BaseModel):
    """Full cross-validation output."""

    dataset_name: str
    config: CrossValidationConfig
    folds: list[DatasetFold] = Field(default_factory=list)
    fold_results: list[FoldConditionResult] = Field(default_factory=list)
    condition_summaries: list[ConditionSummary] = Field(default_factory=list)

    def summary_for(self, condition: str) -> ConditionSummary:
        """Return a condition summary by name."""

        for summary in self.condition_summaries:
            if summary.condition == condition:
                return summary
        raise KeyError(f"No cross-validation summary for condition {condition!r}.")


FoldEvaluator = Callable[
    [BenchmarkDataset, DatasetFold, CrossValidationConfig],
    Iterable[FoldConditionResult],
]


def make_folds(
    dataset: BenchmarkDataset,
    config: CrossValidationConfig | None = None,
) -> list[DatasetFold]:
    """Create deterministic folds for a benchmark dataset."""

    config = config or CrossValidationConfig()
    scene_ids = [scene.scene_id for scene in dataset.scenes]
    if not scene_ids:
        return []

    if config.split_strategy == "leave_one_scene_out":
        return [
            DatasetFold(
                fold_id=f"fold_{index + 1:02d}",
                train_scene_ids=[other for other in scene_ids if other != scene_id],
                test_scene_ids=[scene_id],
            )
            for index, scene_id in enumerate(scene_ids)
        ]

    k = min(config.k or len(scene_ids), len(scene_ids))
    buckets = [[] for _ in range(k)]
    for index, scene_id in enumerate(_deterministic_shuffle(scene_ids, config.seed)):
        buckets[index % k].append(scene_id)

    folds: list[DatasetFold] = []
    all_ids = set(scene_ids)
    for index, test_ids in enumerate(buckets):
        folds.append(
            DatasetFold(
                fold_id=f"fold_{index + 1:02d}",
                train_scene_ids=sorted(all_ids - set(test_ids)),
                test_scene_ids=sorted(test_ids),
            )
        )
    return folds


def run_cross_validation(
    dataset: BenchmarkDataset,
    evaluator: FoldEvaluator,
    config: CrossValidationConfig | None = None,
) -> CrossValidationReport:
    """Run a supplied fold evaluator and summarize condition-level metrics."""

    config = config or CrossValidationConfig()
    folds = make_folds(dataset, config)
    fold_results: list[FoldConditionResult] = []
    for fold in folds:
        fold_results.extend(evaluator(dataset, fold, config))

    return CrossValidationReport(
        dataset_name=dataset.name,
        config=config,
        folds=folds,
        fold_results=fold_results,
        condition_summaries=summarize_fold_results(fold_results),
    )


def summarize_fold_results(
    fold_results: Iterable[FoldConditionResult],
) -> list[ConditionSummary]:
    """Aggregate fold-level metrics by track and condition."""

    grouped: dict[tuple[ExperimentTrack, str], list[FoldConditionResult]] = defaultdict(list)
    for result in fold_results:
        grouped[(result.track, result.condition)].append(result)

    summaries: list[ConditionSummary] = []
    for (track, condition), results in sorted(grouped.items()):
        summaries.append(
            ConditionSummary(
                track=track,
                condition=condition,
                fold_count=len(results),
                node_f1_mean=_rounded_mean(result.node_f1 for result in results),
                node_f1_std=_rounded_std(result.node_f1 for result in results),
                relation_f1_mean=_rounded_mean(result.relation_f1 for result in results),
                relation_f1_std=_rounded_std(result.relation_f1 for result in results),
                ontology_conformance_mean=_rounded_mean(
                    result.ontology_conformance for result in results
                ),
                grounding_rate_mean=_rounded_mean(result.grounding_rate for result in results),
                hallucination_rate_mean=_rounded_mean(
                    result.hallucination_rate for result in results
                ),
                schema_success_rate_mean=_rounded_mean(
                    result.schema_success_rate for result in results
                ),
                runtime_seconds_mean=_rounded_mean(result.runtime_seconds for result in results),
            )
        )
    return summaries


def subset_dataset(dataset: BenchmarkDataset, scene_ids: Iterable[str]) -> BenchmarkDataset:
    """Create a dataset view containing only selected scenes."""

    selected = set(scene_ids)
    scenes: list[BenchmarkScene] = [
        scene for scene in dataset.scenes if scene.scene_id in selected
    ]
    return BenchmarkDataset(name=f"{dataset.name}_subset", scenes=scenes)


def _deterministic_shuffle(values: list[str], seed: int) -> list[str]:
    """Shuffle without importing global random state into tests."""

    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def _rounded_mean(values: Iterable[float]) -> float:
    collected = list(values)
    return round(mean(collected), 4) if collected else 0.0


def _rounded_std(values: Iterable[float]) -> float:
    collected = list(values)
    return round(pstdev(collected), 4) if len(collected) > 1 else 0.0
