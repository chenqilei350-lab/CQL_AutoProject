"""Tests for converting IndEgo text layers into standard KG inputs."""

import json

from backend.cleaning.factory_data import clean_indego_text_layers
from backend.datasets.indego_adapter import (
    IndEgoStandardDataset,
    build_indego_standard_dataset_from_cleaning_report,
    load_indego_standard_dataset,
    parse_via_temporal_segments,
)


def test_parse_via_temporal_segments_resolves_option_labels(tmp_path) -> None:
    """VIA keystep labels may be stored as option IDs and should become text."""

    annotation = {
        "attribute": {
            "1": {
                "options": {
                    "0": "00_preparation",
                    "1": "01_put_on_gloves",
                }
            }
        },
        "file": {
            "1": {
                "fname": "User_1_480.mp4",
            }
        },
        "metadata": {
            "1_a": {"vid": "1", "z": [1.0, 2.5], "av": {"1": "1"}},
            "1_b": {"vid": "1", "z": [3.0, 4.0], "av": {"1": "0"}},
        },
    }
    path = tmp_path / "AS_User_1.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")

    segments = parse_via_temporal_segments(path, source_root=tmp_path, layer="keystep")

    assert [segment.label for segment in segments] == [
        "put on gloves",
        "preparation",
    ]
    assert segments[0].timestamp == "00:01.000-00:02.500"
    assert segments[0].video_name == "User_1_480.mp4"


def test_load_indego_standard_dataset_builds_raw_and_unified_inputs(tmp_path) -> None:
    """The adapter should turn transcripts and VIA actions into standard records."""

    root = tmp_path / "indego_text_layers"
    transcript_dir = root / "1_Assembly"
    action_dir = transcript_dir / "03_Mechanical_Desk" / "annotations"
    keystep_dir = transcript_dir / "annotated_keysteps_assembly_disassembly"
    action_dir.mkdir(parents=True)
    keystep_dir.mkdir(parents=True)

    (transcript_dir / "narration_transcript_assembly.json").write_text(
        json.dumps(
            [
                {
                    "category": "assembly",
                    "name": "User_1_480.mp4",
                    "transcript": "I put on gloves and tighten the screw with an allen wrench.",
                }
            ]
        ),
        encoding="utf-8",
    )
    action_json = {
        "attribute": {"1": {"options": {"default": "Default"}}},
        "file": {"1": {"fname": "User_1_480.mp4"}},
        "metadata": {
            "1_a": {"vid": "1", "z": [1.0, 2.0], "av": {"1": "put on gloves"}},
            "1_b": {
                "vid": "1",
                "z": [3.0, 5.0],
                "av": {"1": "tighten screw with allen wrench"},
            },
            "1_c": {"vid": "1", "z": [6.0, 8.0], "av": {"1": "attach plate"}},
        },
    }
    (action_dir / "A_User_1.json").write_text(json.dumps(action_json), encoding="utf-8")
    keystep_json = {
        "attribute": {"1": {"options": {"1": "01_put_on_gloves", "2": "02_attach_plate"}}},
        "file": {"1": {"fname": "User_1_480.mp4"}},
        "metadata": {
            "1_a": {"vid": "1", "z": [1.0, 2.0], "av": {"1": "1"}},
            "1_b": {"vid": "1", "z": [6.0, 8.0], "av": {"1": "2"}},
        },
    }
    (keystep_dir / "AS_User_1.json").write_text(json.dumps(keystep_json), encoding="utf-8")

    dataset = load_indego_standard_dataset(
        root,
        max_segments_per_scene=2,
        include_transcript_only=False,
        include_warning_scenes=False,
    )

    assert len(dataset.scenes) == 2
    first = dataset.scenes[0]
    assert first.category == "assembly"
    assert "Annotated actions:" in first.raw_text
    assert "Action 2: tighten screw with allen wrench" in first.raw_text
    assert "[动作顺序]" in first.input_text("unified")
    assert first.unified_record.action_sequence[1].text == "tighten screw with allen wrench"
    assert any(entry.text == "allen wrench" for entry in first.unified_record.tools_objects)


def test_warning_annotations_become_quality_warning_inputs(tmp_path) -> None:
    """Mistake annotations should be available as warning-bearing standard scenes."""

    warning_dir = tmp_path / "indego_text_layers" / "Mistake_Detection" / "renamed_annotation_mistakes"
    warning_dir.mkdir(parents=True)
    warning_json = {
        "template": {
            "mistakes": [0, 0],
            "description": ["bring tripod", "mount camera"],
        },
        "01_u01_m_01": {
            "mistakes": [0, 1],
            "description": [None, "camera loose"],
        },
    }
    (warning_dir / "A_Task_01.json").write_text(json.dumps(warning_json), encoding="utf-8")

    dataset = load_indego_standard_dataset(
        tmp_path / "indego_text_layers",
        include_transcript_only=False,
        include_warning_scenes=True,
    )

    scene = dataset.get_scene("indego_warning_A_Task_01_s1")
    assert "Warning 1: step 'mount camera' has issue 'camera loose'." in scene.raw_text
    assert scene.unified_record.quality_results[0].text == "warning: camera loose"
    assert "Warning 1" in scene.input_text("raw")


def test_standard_dataset_jsonl_roundtrip(tmp_path) -> None:
    """Standard inputs should be exportable and loadable for later extraction."""

    warning_dir = tmp_path / "indego_text_layers" / "Mistake_Detection" / "renamed_annotation_mistakes"
    warning_dir.mkdir(parents=True)
    (warning_dir / "A_Task_01.json").write_text(
        json.dumps(
            {
                "template": {"description": ["bring tripod"]},
                "run": {"mistakes": [1], "description": ["wrong spot"]},
            }
        ),
        encoding="utf-8",
    )
    dataset = load_indego_standard_dataset(tmp_path / "indego_text_layers")
    output = dataset.save_jsonl(tmp_path / "standard.jsonl")

    loaded = IndEgoStandardDataset.from_jsonl(output)

    assert loaded.scenes[0].scene_id == dataset.scenes[0].scene_id
    assert loaded.scenes[0].input_text("unified") == dataset.scenes[0].input_text("unified")


def test_standard_dataset_can_be_built_from_cleaning_report(tmp_path) -> None:
    """Cleaned factory records should feed the unified text builder directly."""

    root = tmp_path / "indego_text_layers"
    action_dir = root / "1_Assembly" / "annotations"
    warning_dir = root / "Mistake_Detection" / "renamed_annotation_mistakes"
    action_dir.mkdir(parents=True)
    warning_dir.mkdir(parents=True)

    (root / "1_Assembly" / "narration_transcript_assembly.json").write_text(
        json.dumps(
            [
                {
                    "category": "assembly",
                    "name": "User_1_480.mp4",
                    "transcript": "I tighten the screw with an allen wrench.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (action_dir / "A_User_1.json").write_text(
        json.dumps(
            {
                "attribute": {"1": {"options": {"default": "Default"}}},
                "file": {"1": {"fname": "User_1_480.mp4"}},
                "metadata": {
                    "1_a": {
                        "vid": "1",
                        "z": [1.0, 2.5],
                        "av": {"1": "tighten screw with allen wrench"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (warning_dir / "A_Task_01.txt").write_text(
        "\n".join(
            [
                "[bring tripod,mount camera]",
                "run_m_01: [0,1] camera loose",
            ]
        ),
        encoding="utf-8",
    )

    report = clean_indego_text_layers(root)
    dataset = build_indego_standard_dataset_from_cleaning_report(report)

    action_scene = dataset.get_scene("indego_user_1_s1")
    warning_scene = dataset.get_scene("indego_warning_A_Task_01_s1")

    assert "Action 1: tighten screw with allen wrench" in action_scene.raw_text
    assert any(entry.text == "allen wrench" for entry in action_scene.unified_record.tools_objects)
    assert warning_scene.unified_record.quality_results[0].text == "warning: camera loose"
