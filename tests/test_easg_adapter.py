"""Tests for the annotation-only EASG adapter."""

import json

from backend.datasets.easg_adapter import (
    EASGStandardDataset,
    load_easg_annotation_dataset,
)


def test_easg_adapter_builds_standard_scene_without_video_files(tmp_path) -> None:
    """Graph annotations and narration should be enough to build KG inputs."""

    source = tmp_path / "easg_annotations"
    source.mkdir()
    (source / "sample.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "video_id": "ego_clip_001",
                        "action_id": "a1",
                        "narration": "Camera wearer cuts the bread with a knife.",
                        "verb": "cut",
                        "objects": [
                            {"name": "bread"},
                            {"name": "knife", "object_type": "tool"},
                        ],
                        "relations": [
                            {"source": "knife", "relation": "near", "target": "bread"}
                        ],
                        "pre_frame": 10,
                        "pnr_frame": 15,
                        "post_frame": 20,
                        "start_seconds": 1.0,
                        "end_seconds": 3.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = load_easg_annotation_dataset(source)

    assert len(dataset.scenes) == 1
    scene = dataset.scenes[0]
    assert scene.video_id == "ego_clip_001"
    assert "Annotation-derived text" in scene.raw_text
    assert "cuts the bread" in scene.raw_text
    assert scene.gold_extraction.actions[0].name == "Camera wearer cuts the bread with a knife."
    assert {obj.name for obj in scene.gold_extraction.objects} == {"bread", "knife"}
    assert scene.gold_extraction.uses_tool[0].tool.name == "knife"
    assert scene.gold_extraction.acts_on_object
    assert scene.gold_extraction.scenes[0].timestamp_start_seconds == 1.0
    assert scene.gold_extraction.actions[0].provenance.frame_start == 10
    assert scene.skipped_relations[0].relation == "near"
    assert "no video frames are used" in scene.input_text("unified")
    normalized = scene.unified_record.normalized_segment
    assert normalized is not None
    assert normalized.source_adapter == "easg_adapter"
    assert normalized.actions[0].direct_object_id == "O1"
    assert [tool.text for tool in normalized.tools] == ["knife"]
    assert [scene.mention_id for scene in normalized.scenes] == ["S1"]
    assert "sentence=cut bread with knife." in scene.input_text("unified")


def test_easg_adapter_falls_back_to_annotation_derived_text(tmp_path) -> None:
    """If no narration is available, action and object labels should form text input."""

    source = tmp_path / "easg_annotations"
    source.mkdir()
    (source / "fallback.json").write_text(
        json.dumps(
            {
                "video_uid": "ego_clip_002",
                "id": "a7",
                "verb_label": "open",
                "nouns": ["drawer"],
                "next_action": "take screwdriver",
            }
        ),
        encoding="utf-8",
    )

    dataset = load_easg_annotation_dataset(source)
    scene = dataset.scenes[0]

    assert scene.raw_text == "Annotation-derived text: action 'open' involves drawer."
    assert scene.gold_extraction.actions[0].name == "open"
    assert scene.gold_extraction.objects[0].name == "drawer"
    assert scene.gold_extraction.action_order[0].after.name == "take screwdriver"


def test_easg_adapter_reads_nested_graph_nodes(tmp_path) -> None:
    """Common graph-shaped annotations should map verb/object nodes."""

    source = tmp_path / "easg_annotations"
    source.mkdir()
    (source / "graph.json").write_text(
        json.dumps(
            {
                "video_id": "ego_clip_graph",
                "action_id": "a3",
                "graph": {
                    "nodes": [
                        {"id": "cw", "type": "camera_wearer", "label": "camera wearer"},
                        {"id": "v", "type": "verb", "label": "wash"},
                        {"id": "o1", "type": "object", "label": "plate"},
                        {"id": "o2", "type": "object", "label": "cloth"},
                    ],
                    "edges": [
                        {"source": "o2", "relation": "left_of", "target": "o1"}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = load_easg_annotation_dataset(source)
    scene = dataset.scenes[0]

    assert scene.raw_text == "Annotation-derived text: action 'wash' involves plate, cloth."
    assert scene.gold_extraction.actions[0].name == "wash"
    assert {obj.name for obj in scene.gold_extraction.objects} == {"plate", "cloth"}
    assert scene.skipped_relations[0].relation == "left_of"


def test_easg_adapter_reads_real_master_file_shape(tmp_path) -> None:
    """The public EASG master JSON maps graph triplets without video frames."""

    source = tmp_path / "EASG"
    source.mkdir()
    (source / "EASG_unict_master_final.json").write_text(
        json.dumps(
            {
                "action_uid_1": {
                    "graphs": [
                        {
                            "pnr": 62,
                            "pre": 48,
                            "post": 69,
                            "triplets": [
                                ["place", "dobj", "wood"],
                                ["place", "with", "right hand"],
                                ["CW", "verb", "place"],
                            ],
                            "groundings": {"pre": {"wood": {"left": 1}}},
                            "graph_uid": "graph_1",
                        },
                        {
                            "pnr": 80,
                            "pre": 70,
                            "post": 85,
                            "triplets": [
                                ["pick", "dobj", "drill"],
                                ["pick", "from", "table"],
                                ["CW", "verb", "pick"],
                            ],
                            "graph_uid": "graph_2",
                        },
                    ],
                    "split": "train",
                    "video_uid": "video_real_shape",
                    "W": 1920,
                    "H": 1080,
                }
            }
        ),
        encoding="utf-8",
    )

    dataset = load_easg_annotation_dataset(source)

    assert len(dataset.scenes) == 2
    first = dataset.scenes[0]
    second = dataset.scenes[1]
    assert first.gold_extraction.actions[0].name == "place"
    assert {obj.name for obj in first.gold_extraction.objects} == {"wood", "right hand"}
    assert {(edge.object.name, edge.role, edge.evidence_text) for edge in first.gold_extraction.acts_on_object} == {
        ("wood", "target", "place dobj wood"),
        ("right hand", "support", "place with right hand"),
    }
    assert first.gold_extraction.actions[0].provenance.frame_start == 48
    assert first.gold_extraction.actions[0].provenance.frame_end == 69
    assert first.raw_text == "Annotation-derived text: action 'place' involves wood, right hand."
    assert first.gold_extraction.action_order[0].after.name == "pick"
    assert second.gold_extraction.tools[0].name == "drill"
    assert second.gold_extraction.acts_on_object[1].role == "input"
    assert second.gold_extraction.acts_on_object[1].evidence_text == "pick from table"


def test_easg_dataset_jsonl_roundtrip_and_benchmark_conversion(tmp_path) -> None:
    """Exported EASG standard inputs should be reusable by experiment runners."""

    source = tmp_path / "easg_annotations"
    source.mkdir()
    (source / "sample.json").write_text(
        json.dumps(
            [
                {
                    "clip_uid": "clip_a",
                    "annotation_id": "step_1",
                    "text": "Camera wearer stirs soup with a spoon.",
                    "object_nodes": {"o1": {"label": "soup"}, "o2": {"label": "spoon"}},
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset = load_easg_annotation_dataset(source)
    output = dataset.save_jsonl(tmp_path / "easg_standard_inputs.jsonl")

    loaded = EASGStandardDataset.from_jsonl(output)
    benchmark_scene = loaded.scenes[0].to_benchmark_scene()

    assert loaded.scenes[0].scene_id == dataset.scenes[0].scene_id
    assert benchmark_scene.raw_text == loaded.scenes[0].raw_text
    assert benchmark_scene.gold_extraction.source_text == benchmark_scene.raw_text


def test_legacy_easg_jsonl_migration_does_not_need_gold_fields(tmp_path) -> None:
    """Pre-contract JSONL should receive typed slots from adapter text alone."""

    record = {
        "scene_id": "legacy_easg",
        "video_id": "video_1",
        "description": "legacy",
        "source_path": "legacy.json",
        "raw_text": "Annotation-derived text: action 'place' involves wood, right hand.",
        "unified_record": {
            "scene_id": "legacy_easg",
            "segment_id": "a1",
            "scene_segment": "legacy",
            "action_sequence": [{"text": "place", "evidence": "place"}],
            "tools_objects": [
                {"text": "wood", "evidence": "wood"},
                {"text": "right hand", "evidence": "right hand"},
            ],
            "source_text": (
                "Annotation-derived text: action 'place' involves wood, right hand."
            ),
        },
        "gold_extraction": {"source_text": "unused gold sentinel"},
    }
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    scene = EASGStandardDataset.from_jsonl(path).scenes[0]
    segment = scene.unified_record.normalized_segment

    assert segment is not None
    assert segment.source_adapter == "easg_adapter_legacy_migration"
    assert segment.actions[0].verb == "place"
    assert segment.actions[0].direct_object_id == "O1"
    assert [item.mention_id for item in segment.scenes] == ["S1"]
    assert "unused gold sentinel" not in scene.input_text("unified")


def test_easg_adapter_limit_reads_subset(tmp_path) -> None:
    """The adapter should support small annotation-only pilot subsets."""

    source = tmp_path / "easg_annotations"
    source.mkdir()
    (source / "sample.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"video_id": "v1", "id": "a1", "verb": "take", "objects": ["cup"]},
                    {"video_id": "v2", "id": "a2", "verb": "put", "objects": ["plate"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = load_easg_annotation_dataset(source, limit=1)

    assert len(dataset.scenes) == 1
