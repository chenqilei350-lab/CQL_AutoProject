import json
from pathlib import Path

from scripts.build_combined_industrial_gold import combine_reviewed_gold


def _record(scene_id: str) -> dict[str, object]:
    source_text = "The worker inspects the bracket."
    return {
        "scene_id": scene_id,
        "video_id": scene_id,
        "category": "inspection",
        "task": "inspect bracket",
        "source_text": source_text,
        "gold_entities": [
            {
                "id": "action_1",
                "label": "Action",
                "name": "inspect bracket",
                "evidence_text": "inspects the bracket",
                "sequence_index": 1,
            },
            {
                "id": "object_1",
                "label": "Object",
                "name": "bracket",
                "evidence_text": "bracket",
            },
        ],
        "gold_relations": [
            {"source": "action_1", "relation": "ACTS_ON", "target": "object_1"}
        ],
        "quality_results": [],
        "outcomes": [],
        "review_status": "accepted",
        "reviewer_note": "test fixture",
    }


def test_combines_reviewed_files_without_changing_records(tmp_path: Path) -> None:
    first = tmp_path / "industrial_gold_v2_reviewed.jsonl"
    second = tmp_path / "industrial_gold_v3_reviewed.jsonl"
    output = tmp_path / "industrial_gold_combined_v2_v3_reviewed.jsonl"
    first_record = _record("scene_1")
    second_record = _record("scene_2")
    first.write_text(json.dumps(first_record) + "\n", encoding="utf-8")
    second.write_text(json.dumps(second_record) + "\n", encoding="utf-8")

    summary = combine_reviewed_gold([first, second], output)
    emitted = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary == {"scene_count": 2, "entity_count": 4, "relation_count": 2}
    assert emitted == [first_record, second_record]
