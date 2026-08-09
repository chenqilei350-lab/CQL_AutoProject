"""Cross-validation split and summary tests."""

from backend.datasets.benchmark import MVP_BENCHMARK
from backend.evaluation.cross_validation import (
    CrossValidationConfig,
    FoldConditionResult,
    make_folds,
    run_cross_validation,
    subset_dataset,
)


def test_leave_one_scene_out_folds_have_no_leakage() -> None:
    """Each scene should appear once as held-out data and never in train."""

    config = CrossValidationConfig(
        split_strategy="leave_one_scene_out",
        conditions=("baseline_one_stage",),
    )
    folds = make_folds(MVP_BENCHMARK, config)

    assert len(folds) == len(MVP_BENCHMARK.scenes)
    held_out = [fold.test_scene_ids[0] for fold in folds]
    assert sorted(held_out) == sorted(scene.scene_id for scene in MVP_BENCHMARK.scenes)
    assert all(
        not set(fold.train_scene_ids) & set(fold.test_scene_ids)
        for fold in folds
    )


def test_kfold_splits_are_deterministic_and_cover_dataset() -> None:
    """K-fold should produce stable folds that cover all scenes."""

    config = CrossValidationConfig(split_strategy="kfold", k=2, seed=7)
    folds_a = make_folds(MVP_BENCHMARK, config)
    folds_b = make_folds(MVP_BENCHMARK, config)

    assert folds_a == folds_b
    covered = sorted(scene_id for fold in folds_a for scene_id in fold.test_scene_ids)
    assert covered == sorted(scene.scene_id for scene in MVP_BENCHMARK.scenes)


def test_subset_dataset_keeps_selected_scenes_only() -> None:
    """Fold runners can receive a dataset view containing held-out scenes."""

    scene_id = MVP_BENCHMARK.scenes[0].scene_id
    subset = subset_dataset(MVP_BENCHMARK, [scene_id])

    assert subset.name.endswith("_subset")
    assert [scene.scene_id for scene in subset.scenes] == [scene_id]


def test_run_cross_validation_summarizes_fold_results() -> None:
    """Generic evaluator results should be summarized by condition."""

    config = CrossValidationConfig(
        split_strategy="leave_one_scene_out",
        conditions=("baseline_one_stage", "strict_schema_prompt"),
    )

    def evaluator(dataset, fold, cv_config):
        return [
            FoldConditionResult(
                fold_id=fold.fold_id,
                track=cv_config.track,
                condition=condition,
                test_scene_ids=fold.test_scene_ids,
                node_f1=1.0,
                relation_f1=0.5 if condition == "baseline_one_stage" else 1.0,
                ontology_conformance=1.0,
                grounding_rate=1.0,
                hallucination_rate=0.0,
                schema_success_rate=1.0,
                runtime_seconds=0.1,
            )
            for condition in cv_config.conditions
        ]

    report = run_cross_validation(MVP_BENCHMARK, evaluator, config)

    assert report.summary_for("baseline_one_stage").relation_f1_mean == 0.5
    assert report.summary_for("strict_schema_prompt").relation_f1_mean == 1.0
    assert report.summary_for("baseline_one_stage").fold_count == len(MVP_BENCHMARK.scenes)
