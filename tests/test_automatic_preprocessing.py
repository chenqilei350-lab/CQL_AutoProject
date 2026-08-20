"""Tests for conditional LLM preprocessing and conservative name normalization."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocessing import automatic
from backend.preprocessing.automatic import PreprocessingDraft
from backend.preprocessing.unified_text import (
    AutomaticPreprocessingError,
    GroundedEntry,
    build_industrial_unified_text,
)


RAW_TEXT = (
    "Operator Lena measures the cover plate with a vernier caliper. "
    "She records a width of 12 mm."
)


def test_explicit_groups_do_not_invoke_automatic_preprocessing(monkeypatch) -> None:
    def fail_if_called(**_kwargs):
        raise AssertionError("automatic preprocessing must not run")

    monkeypatch.setattr(
        "backend.preprocessing.unified_text._automatically_preprocess_industrial_text",
        fail_if_called,
    )

    record = build_industrial_unified_text(
        raw_text=RAW_TEXT,
        scene_id="manual_01",
        segment_id="s1",
        timestamp=None,
        scene="manual preprocessing",
        actors=[],
        action_sequence=[],
        tools_objects=[],
    )

    assert record.action_sequence == []


def test_empty_alias_fields_do_not_change_existing_json_shape() -> None:
    data = GroundedEntry(text="caliper", evidence="caliper").model_dump(mode="json")

    assert data == {"text": "caliper", "evidence": "caliper", "uncertainty": None}


def test_missing_groups_trigger_one_llm_classification_and_filter_bad_evidence(
    monkeypatch,
) -> None:
    calls = []

    def fake_extract(**kwargs):
        calls.append(kwargs["response_model"])
        return PreprocessingDraft(
            actors=[GroundedEntry(text="Operator Lena", evidence="Operator Lena")],
            action_sequence=[
                GroundedEntry(
                    text="measure cover plate",
                    evidence="measures the cover plate",
                )
            ],
            tools_objects=[],
            process_parameters=[
                GroundedEntry(text="width = 12 mm", evidence="12 mm"),
                GroundedEntry(text="temperature = 42 C", evidence="42 C"),
            ],
        )

    monkeypatch.setattr(automatic, "_extract_structured_response", fake_extract)

    record = build_industrial_unified_text(
        raw_text=RAW_TEXT,
        scene_id="auto_01",
        segment_id="s1",
        timestamp=None,
        scene="automatic preprocessing",
    )

    assert calls == [PreprocessingDraft]
    assert [entry.text for entry in record.actors] == ["Operator Lena"]
    assert [entry.text for entry in record.process_parameters] == ["width = 12 mm"]
    assert any("temperature = 42 C" in note for note in record.evidence_uncertainty)


def test_automatic_llm_failure_has_clear_domain_error(monkeypatch) -> None:
    def fail(**_kwargs):
        raise ConnectionError("Ollama unavailable")

    monkeypatch.setattr(automatic, "_extract_structured_response", fail)

    with pytest.raises(AutomaticPreprocessingError, match="llama3.1:8b"):
        build_industrial_unified_text(
            raw_text=RAW_TEXT,
            scene_id="auto_02",
            segment_id="s1",
            timestamp=None,
            scene="automatic preprocessing",
        )


def test_exact_catalog_name_does_not_load_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        automatic,
        "_load_default_tool_object_terms",
        lambda: ("caliper", "hammer"),
    )

    def fail_if_encoded(*_args, **_kwargs):
        raise AssertionError("exact catalog names need no embedding model")

    monkeypatch.setattr(automatic, "_encode_texts", fail_if_encoded)

    entries, notes = automatic._canonicalize_tools_objects(
        [GroundedEntry(text="caliper", evidence="vernier caliper")],
        raw_text=RAW_TEXT,
        model="fake",
        embedding_model="fake",
        threshold=0.65,
        top_k=3,
    )

    assert notes == []
    assert [entry.text for entry in entries] == ["caliper"]


def test_vector_candidate_and_llm_judge_merge_catalog_aliases(monkeypatch) -> None:
    monkeypatch.setattr(
        automatic,
        "_load_default_tool_object_terms",
        lambda: ("caliper", "hammer"),
    )
    monkeypatch.setattr(
        automatic,
        "_encode_texts",
        lambda _texts, _model: np.asarray([[1.0, 0.0], [0.98, 0.02]]),
    )
    monkeypatch.setattr(
        automatic,
        "_encode_catalog",
        lambda _catalog, _model: np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    monkeypatch.setattr(
        automatic,
        "_judge_catalog_alias",
        lambda **_kwargs: automatic._CatalogAliasDecision(
            same_entity=True,
            canonical_name="caliper",
        ),
    )

    entries, notes = automatic._canonicalize_tools_objects(
        [
            GroundedEntry(text="vernier caliper", evidence="vernier caliper"),
            GroundedEntry(text="measuring gauge", evidence="measures the cover plate"),
        ],
        raw_text=RAW_TEXT,
        model="fake",
        embedding_model="fake",
        threshold=0.65,
        top_k=2,
    )

    assert notes == []
    assert len(entries) == 1
    assert entries[0].text == "caliper"
    assert entries[0].aliases == ["measuring gauge", "vernier caliper"]
    assert entries[0].evidence == "measures the cover plate"
    assert entries[0].additional_evidence == ["vernier caliper"]


def test_scene_alias_judge_can_keep_similar_objects_separate(monkeypatch) -> None:
    monkeypatch.setattr(automatic, "_load_default_tool_object_terms", lambda: ())
    monkeypatch.setattr(
        automatic,
        "_encode_texts",
        lambda _texts, _model: np.asarray([[1.0, 0.0], [0.99, 0.01]]),
    )
    monkeypatch.setattr(
        automatic,
        "_judge_scene_alias",
        lambda **_kwargs: automatic._PairAliasDecision(same_entity=False),
    )

    source = "The clamp holds the fixture. The holder supports another plate."
    entries, notes = automatic._canonicalize_tools_objects(
        [
            GroundedEntry(text="fixture", evidence="fixture"),
            GroundedEntry(text="holder", evidence="holder"),
        ],
        raw_text=source,
        model="fake",
        embedding_model="fake",
        threshold=0.65,
        top_k=1,
    )

    assert notes == []
    assert [entry.text for entry in entries] == ["fixture", "holder"]


def test_embedding_failure_keeps_original_names_and_records_note(monkeypatch) -> None:
    monkeypatch.setattr(
        automatic,
        "_load_default_tool_object_terms",
        lambda: ("caliper",),
    )

    def fail_embedding(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(automatic, "_encode_texts", fail_embedding)

    entries, notes = automatic._canonicalize_tools_objects(
        [GroundedEntry(text="vernier caliper", evidence="vernier caliper")],
        raw_text=RAW_TEXT,
        model="fake",
        embedding_model="fake",
        threshold=0.65,
        top_k=3,
    )

    assert entries[0].text == "vernier caliper"
    assert any("Vector name canonicalization was unavailable" in note for note in notes)


def test_rendered_entry_includes_aliases_and_all_grounded_evidence() -> None:
    record = build_industrial_unified_text(
        raw_text=RAW_TEXT,
        scene_id="manual_alias_01",
        segment_id="s1",
        timestamp=None,
        scene="manual alias record",
        action_sequence=[],
        tools_objects=[
            GroundedEntry(
                text="caliper",
                evidence="vernier caliper",
                aliases=["vernier caliper"],
                additional_evidence=["measures the cover plate"],
            )
        ],
    )

    prompt = record.to_prompt_text()
    assert "别名: vernier caliper" in prompt
    assert "证据: vernier caliper; measures the cover plate" in prompt
