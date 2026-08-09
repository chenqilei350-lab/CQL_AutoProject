"""Tests for deterministic factory data-source cleaning."""

import json

from backend.cleaning.factory_data import (
    clean_factory_data_source,
    clean_indego_text_layers,
)


def test_clean_indego_text_layers_keeps_provenance_and_layer_types(tmp_path) -> None:
    """IndEgo cleaning should normalize local text layers without LLM calls."""

    root = tmp_path / "indego_text_layers"
    action_dir = root / "1_Assembly" / "03_Mechanical_Desk" / "annotations"
    warning_dir = root / "Mistake_Detection" / "renamed_annotation_mistakes"
    action_dir.mkdir(parents=True)
    warning_dir.mkdir(parents=True)

    transcript_path = root / "1_Assembly" / "narration_transcript_assembly.json"
    transcript_path.write_text(
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
    action_path = action_dir / "A_User_1.json"
    action_path.write_text(
        json.dumps(
            {
                "attribute": {"1": {"options": {"2": "02_tighten_screw"}}},
                "file": {"1": {"fname": "User_1_480.mp4"}},
                "metadata": {"1_a": {"vid": "1", "z": [1.0, 2.5], "av": {"1": "2"}}},
            }
        ),
        encoding="utf-8",
    )
    keystep_path = root / "1_Assembly" / "03_Mechanical_Desk" / "keysteps_03_assembly.txt"
    keystep_path.write_text("01_prepare workstation\n02_tighten screw\n", encoding="utf-8")
    warning_path = warning_dir / "A_Task_01.json"
    warning_path.write_text(
        json.dumps(
            {
                "template": {"description": ["bring tripod", "mount camera"]},
                "run_1": {"mistakes": [0, 1], "description": [None, "camera loose"]},
            }
        ),
        encoding="utf-8",
    )
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "1_Assembly/narration_transcript_assembly.json", "layer": "narration_transcript"},
                    {"path": "1_Assembly/03_Mechanical_Desk/annotations/A_User_1.json", "layer": "fine_grained_annotations"},
                    {"path": "1_Assembly/03_Mechanical_Desk/keysteps_03_assembly.txt", "layer": "scenario_keysteps"},
                    {
                        "path": "Mistake_Detection/renamed_annotation_mistakes/A_Task_01.json",
                        "layer": "mistake_warning_annotations",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = clean_indego_text_layers(root)

    assert report.files_seen == 4
    assert report.records_by_type() == {
        "action_segment": 1,
        "mistake_step": 2,
        "mistake_warning": 1,
        "scenario_keystep": 2,
        "transcript": 1,
    }
    action = next(record for record in report.records if record.record_type == "action_segment")
    assert action.text == "tighten screw"
    assert action.video_id == "user_1"
    assert action.timestamp == "00:01.000-00:02.500"
    assert action.source_path.endswith("A_User_1.json")
    warning = next(record for record in report.records if record.record_type == "mistake_warning")
    assert warning.metadata["step"] == "mount camera"
    assert warning.metadata["warning"] == "camera loose"


def test_cleaning_flags_invalid_temporal_ranges(tmp_path) -> None:
    """Temporal annotations with impossible bounds should remain visible as issues."""

    root = tmp_path / "indego_text_layers"
    action_dir = root / "1_Assembly" / "annotations"
    action_dir.mkdir(parents=True)
    action_path = action_dir / "A_User_1.json"
    action_path.write_text(
        json.dumps(
            {
                "attribute": {"1": {"options": {"default": "Default"}}},
                "file": {"1": {"fname": "User_1_480.mp4"}},
                "metadata": {
                    "bad_time": {"vid": "1", "z": [5.0, 2.0], "av": {"1": "fastening bolt"}},
                    "missing_time": {"vid": "1", "z": [], "av": {"1": "align plate"}},
                },
            }
        ),
        encoding="utf-8",
    )

    report = clean_factory_data_source(root)

    assert report.records_by_type() == {"action_segment": 2}
    assert report.issues_by_code() == {
        "invalid_time_range": 1,
        "missing_time_bounds": 1,
    }


def test_cleaning_parses_mistake_warning_text_files(tmp_path) -> None:
    """Some IndEgo warning annotations are text files, not JSON files."""

    root = tmp_path / "indego_text_layers"
    warning_dir = root / "Mistake_Detection" / "renamed_annotation_mistakes"
    warning_dir.mkdir(parents=True)
    (warning_dir / "A_Task_01.txt").write_text(
        "\n".join(
            [
                "[bring tripod,level tripod,fasten legs]",
                "run_c_01: [0,0,0]",
                "run_m_01: [1,0,1] wrong spot, didn't fasten legs",
            ]
        ),
        encoding="utf-8",
    )
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "Mistake_Detection/renamed_annotation_mistakes/A_Task_01.txt",
                        "layer": "mistake_warning_annotations",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = clean_indego_text_layers(root)

    assert report.records_by_type() == {"mistake_step": 3, "mistake_warning": 2}
    warnings = [record for record in report.records if record.record_type == "mistake_warning"]
    assert warnings[0].metadata["step"] == "bring tripod"
    assert warnings[0].metadata["warning"] == "wrong spot"
    assert warnings[1].metadata["step"] == "fasten legs"
    assert warnings[1].metadata["warning"] == "didn't fasten legs"


def test_cleaning_jsonl_export_is_downstream_friendly(tmp_path) -> None:
    """Cleaned records can be exported as JSONL without losing provenance."""

    root = tmp_path / "indego_text_layers"
    root.mkdir()
    (root / "keysteps_logistics_organization.txt").write_text(
        "01_pick up box\n02_place box on trolley\n",
        encoding="utf-8",
    )

    report = clean_indego_text_layers(root)
    output = report.save_jsonl(tmp_path / "cleaned" / "factory_records.jsonl")
    lines = output.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert '"record_type":"scenario_keystep"' in lines[0]
    assert '"source_path":"keysteps_logistics_organization.txt"' in lines[0]
