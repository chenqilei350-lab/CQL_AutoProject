"""人工审核 Industrial Gold v2 的合同与实验集成测试。"""

import json
from pathlib import Path

from backend.datasets.benchmark import BenchmarkDataset
from backend.datasets.industrial_gold import load_industrial_gold_dataset
from backend.graph.property_graph import build_property_graph
from scripts.build_industrial_gold_v2_package import build_package


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_gold_has_expected_reviewed_and_excluded_scenes():
    dataset = load_industrial_gold_dataset()

    assert len(dataset.records) == 12
    assert len({record.video_id for record in dataset.records}) == 11
    assert {record.scene_id for record in dataset.exclusions} == {
        "indego_user_16_414_2101_1_s8",
        "indego_user_15_414_0702_1_s10",
        "indego_user_15_451_2401_1_s1",
        "indego_user_15_451_2401_1_s3",
    }
    assert not ({record.scene_id for record in dataset.records} & {
        record.scene_id for record in dataset.exclusions
    })


def test_manifest_matches_the_human_reviewed_contract():
    manifest = load_industrial_gold_dataset().manifest()

    assert manifest["entity_counts"] == {
        "Action": 145,
        "Object": 113,
        "Tool": 14,
    }
    assert manifest["relation_counts"] == {
        "ACTS_ON": 170,
        "BEFORE": 133,
        "USES_TOOL": 22,
    }
    assert manifest["quality_result_count"] == 32
    assert manifest["outcome_count"] == 1
    assert manifest["video_groups"]["user_16_414_2101_1"] == [
        "indego_user_16_414_2101_1_s1",
        "indego_user_16_414_2101_1_s2",
    ]


def test_reviewed_gold_converts_to_existing_benchmark_and_graph_contracts():
    dataset = load_industrial_gold_dataset()
    benchmark = dataset.to_benchmark_dataset()

    assert len(benchmark.scenes) == 12
    for scene in benchmark.scenes:
        assert scene.raw_text == dataset.raw_text_by_scene[scene.scene_id]
        assert scene.gold_extraction.source_text == scene.raw_text
        assert scene.unified_record.source_text == scene.raw_text
        build_property_graph(scene.gold_extraction)

    assert sum(len(scene.gold_extraction.actions) for scene in benchmark.scenes) == 145
    assert sum(len(scene.gold_extraction.objects) for scene in benchmark.scenes) == 113
    assert sum(len(scene.gold_extraction.tools) for scene in benchmark.scenes) == 14
    assert sum(len(scene.gold_extraction.action_order) for scene in benchmark.scenes) == 133
    assert sum(len(scene.gold_extraction.acts_on_object) for scene in benchmark.scenes) == 170
    assert sum(len(scene.gold_extraction.uses_tool) for scene in benchmark.scenes) == 22


def test_reviewer_notes_and_split_policy_do_not_leak_into_model_input():
    benchmark = load_industrial_gold_dataset().to_benchmark_dataset()
    s07 = benchmark.get_scene("indego_user_16_414_2101_1_s1")
    unified_text = s07.input_text("unified")

    assert "Human-corrected" not in unified_text
    assert "Dataset splits" not in unified_text
    assert "must remain in the same evaluation split" not in unified_text


def test_generated_package_round_trips_without_excluded_scenes(tmp_path):
    output_dir = tmp_path / "industrial_gold_v2"
    manifest = build_package(output_dir)
    loaded = BenchmarkDataset.from_jsonl(
        output_dir / "benchmark.jsonl",
        name=manifest["dataset_name"],
    )
    exported_rows = [
        json.loads(line)
        for line in (output_dir / "benchmark.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert loaded.name == "industrial_gold_v2_reviewed"
    assert len(loaded.scenes) == 12
    assert all("unified_text" in row for row in exported_rows)
    assert not ({scene.scene_id for scene in loaded.scenes} & {
        exclusion["scene_id"] for exclusion in manifest["excluded_scenes"]
    })
    assert (output_dir / "manifest.json").is_file()


def test_checked_in_package_is_reproducible(tmp_path):
    output_dir = tmp_path / "rebuilt"
    build_package(output_dir)

    assert (output_dir / "benchmark.jsonl").read_text(encoding="utf-8") == (
        PROJECT_ROOT / "data/industrial_gold_v2/benchmark.jsonl"
    ).read_text(encoding="utf-8")
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == (
        json.loads(
            (PROJECT_ROOT / "data/industrial_gold_v2/manifest.json").read_text(
                encoding="utf-8"
            )
        )
    )
