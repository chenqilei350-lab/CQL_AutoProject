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


def test_automatic_prompt_preserves_implicit_and_annotation_actions() -> None:
    """Cross-source action wording must not be filtered by an industrial-only rule."""

    prompt = automatic._PREPROCESSING_SYSTEM_PROMPT

    assert "target is implicit" in prompt
    assert "action 'place' involves wood" in prompt
    assert "split coordinated or repeated events" in prompt


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


def test_long_source_is_split_and_partial_chunk_failure_is_preserved(
    monkeypatch,
) -> None:
    source = " ".join(
        [
            "Operator opens the maintenance cover before inspecting the long assembly area carefully and checking every visible fastener.",
            "FAIL chunk describes unrelated noisy speech that the local model cannot structure safely despite the repeated explanation.",
            "Operator closes the maintenance cover after completing the inspection carefully and documenting the final condition.",
        ]
    )
    calls: list[str] = []

    def fake_extract(**kwargs):
        chunk = kwargs["text"]
        calls.append(chunk)
        if "FAIL chunk" in chunk:
            raise TimeoutError("simulated chunk timeout")
        if "opens the maintenance cover" in chunk:
            return PreprocessingDraft(
                action_sequence=[
                    GroundedEntry(
                        text="open maintenance cover",
                        evidence="opens the maintenance cover",
                    )
                ]
            )
        return PreprocessingDraft(
            action_sequence=[
                GroundedEntry(
                    text="close maintenance cover",
                    evidence="closes the maintenance cover",
                )
            ]
        )

    monkeypatch.setattr(automatic, "_extract_structured_response", fake_extract)

    draft, notes = automatic.automatically_preprocess_industrial_text(
        raw_text=source,
        model="fake",
        chunk_max_chars=200,
    )

    assert len(calls) == 4
    assert all(len(chunk) <= 200 for chunk in calls)
    assert [entry.text for entry in draft.action_sequence] == [
        "open maintenance cover",
        "close maintenance cover",
    ]
    assert any("2/3 failed after 2 attempts" in note for note in notes)
    assert any("2/3 attempt 1/2 failed" in note for note in notes)
    assert any("split the source into 3 chunks" in note for note in notes)


def test_preprocessing_chunk_retry_recovers_without_repeating_siblings(
    monkeypatch,
) -> None:
    attempts = 0

    def transient_failure(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        return PreprocessingDraft(
            action_sequence=[
                GroundedEntry(text="align plate", evidence="aligns the plate")
            ]
        )

    monkeypatch.setattr(
        automatic,
        "_extract_structured_response",
        transient_failure,
    )

    draft, notes = automatic.automatically_preprocess_industrial_text(
        raw_text="The worker aligns the plate.",
        model="fake",
    )

    assert attempts == 2
    assert [entry.text for entry in draft.action_sequence] == ["align plate"]
    assert any("succeeded on attempt 2/2" in note for note in notes)


def test_chunk_splitter_keeps_every_chunk_grounded_in_source() -> None:
    source = (
        "First the worker aligns the plate. "
        "Then the worker tightens the mounting screws. "
        "Finally the worker checks the completed assembly."
    )

    chunks = automatic._split_preprocessing_chunks(source, 200)

    assert chunks == [source]
    assert all(chunk in source for chunk in chunks)


def test_model_added_outer_quotes_are_removed_before_grounding() -> None:
    """Formatting quotes must not erase an otherwise exact source span."""

    source = "Hans aligns the steel plate and starts the root weld."
    draft = PreprocessingDraft(
        action_sequence=[
            GroundedEntry(
                text="align steel plate",
                evidence="'aligns the steel plate'",
            )
        ]
    )

    grounded, notes = automatic._filter_and_order_grounded_entries(draft, source)

    assert notes == []
    assert grounded.action_sequence[0].evidence == "aligns the steel plate"


def test_real_source_quotes_are_preserved_when_already_grounded() -> None:
    source = "Annotation-derived text: action 'place' involves wood."
    entry = GroundedEntry(text="place", evidence="'place'")

    normalized = automatic._normalize_grounded_entry_evidence(entry, source)

    assert normalized.evidence == "'place'"


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
    assert entries[0].text == "vernier caliper"
    assert entries[0].canonical_name == "caliper"
    assert entries[0].aliases == ["measuring gauge"]
    assert entries[0].evidence == "vernier caliper"
    assert entries[0].additional_evidence == ["measures the cover plate"]


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
    assert "Aliases: vernier caliper" in prompt
    assert "Evidence: vernier caliper; measures the cover plate" in prompt
