"""LLM-assisted, evidence-grounded preprocessing for industrial source text.

Treat this file as the automatic preprocessing worker. It asks
the LLM for five categories, removes entries without source evidence, compares
tool/object names with the known vocabulary, verifies similar names, and returns
the organized result to unified_text.py.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field

from backend.preprocessing.unified_text import (
    AutomaticPreprocessingError,
    GroundedEntry,
    _normalize_text,
)


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# 中文：7B 本地模型在约 800 字符且动作密集的工业转录上会反复超时；500
# 字符保留完整句边界，同时限制一次结构化输出的动作数量。
# English: Dense ~800-character industrial transcripts repeatedly time out on
# local 7B models. A 500-character boundary keeps sentences intact while capping
# the number of structured records requested per call.
DEFAULT_PREPROCESSING_CHUNK_CHARS = 500
DEFAULT_PREPROCESSING_CHUNK_ATTEMPTS = 2


# [Block 05] 给 LLM 的固定分类表，只允许输出五个预设栏目。
# [Block 05] Fixed LLM classification form with only five allowed categories.
class PreprocessingDraft(BaseModel):
    """Five grounded categories produced by one LLM call."""

    actors: list[GroundedEntry] = Field(default_factory=list)
    action_sequence: list[GroundedEntry] = Field(default_factory=list)
    tools_objects: list[GroundedEntry] = Field(default_factory=list)
    process_parameters: list[GroundedEntry] = Field(default_factory=list)
    quality_results: list[GroundedEntry] = Field(default_factory=list)


class _CatalogAliasDecision(BaseModel):
    """Decision that maps a mention to a catalog term."""

    same_entity: bool
    canonical_name: str | None = None


class _PairAliasDecision(BaseModel):
    """Same-entity decision for two mentions in one scene."""

    same_entity: bool


_PREPROCESSING_SYSTEM_PROMPT = """You preprocess industrial manufacturing text.
Return exactly the requested schema with five categories:
- actors: explicitly mentioned people or worker roles.
- action_sequence: explicitly stated actions, kept in source order.
- tools_objects: explicitly mentioned physical tools and operated objects.
- process_parameters: explicitly stated measurements, settings, values, or conditions.
- quality_results: explicitly stated checks, warnings, defects, or outcomes.

For every entry:
- text must be a short faithful description, not a new fact.
- evidence must be the shortest exact contiguous quote that supports the entry.
- set entry_type to role, action, tool, object, parameter, or quality. In the
  tools_objects list, distinguish an actively used instrument (tool) from the
  item being manipulated (object); do not rely on a closed vocabulary.
- every action text must contain an explicit action verb. Include its direct target
  when the target is stated, but do not omit an explicit action only because its
  target is implicit.
- for action entries, also fill verb, direct_object, tool, and role when explicitly
  stated; leave an absent slot null instead of guessing.
- action evidence must include that action verb and target, not a whole paragraph.
- split coordinated or repeated events into separate actions when the source states
  distinct verbs, sides, or steps; never combine "pick up ... and then start ...".
- annotation-derived text such as "action 'place' involves wood" explicitly states
  the Action "place"; its exact evidence may be the short span "action 'place'".
- tools_objects evidence should be the shortest grounded noun phrase.
- process_parameters are only explicit values/settings/conditions, not action names.
- quality_results are only checks, warnings, defects, or outcomes, not instructions.
- actors are named people or worker roles; body parts are tools_objects.
- use uncertainty only when the source itself is unclear.
- leave aliases and additional_evidence empty; they are filled later.
Do not infer missing information. Return an empty list when a category is absent.
"""


# [Block 06] 自动模式总流程：LLM 分类、Evidence 过滤、名称统一。
# [Block 06] Automatic flow: LLM classification, evidence filtering, name canonicalization.
def automatically_preprocess_industrial_text(
    *,
    raw_text: str,
    model: str = "llama3.1:8b",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    alias_candidate_threshold: float = 0.65,
    alias_top_k: int = 3,
    chunk_max_chars: int = DEFAULT_PREPROCESSING_CHUNK_CHARS,
) -> tuple[PreprocessingDraft, list[str]]:
    """Classify source text, filter ungrounded entries, and normalize names."""

    if not raw_text.strip():
        raise ValueError("raw_text must not be empty.")
    if not 0.0 <= alias_candidate_threshold <= 1.0:
        raise ValueError("alias_candidate_threshold must be between 0.0 and 1.0.")
    if alias_top_k < 1:
        raise ValueError("alias_top_k must be at least 1.")
    if chunk_max_chars < 200:
        raise ValueError("chunk_max_chars must be at least 200.")

    # 第一步：长文本先按原文边界切块，避免一个大 JSON 输出耗尽本地模型超时。
    # Step 1: Chunk long source text at source boundaries so one large JSON
    # response cannot consume the entire local-model timeout budget.
    chunks = _split_preprocessing_chunks(raw_text, chunk_max_chars)
    notes = (
        [f"Automatic preprocessing split the source into {len(chunks)} chunks."]
        if len(chunks) > 1
        else []
    )
    grounded_chunks: list[PreprocessingDraft] = []
    last_error: Exception | None = None
    for chunk_index, chunk in enumerate(chunks, start=1):
        draft: PreprocessingDraft | None = None
        # 中文：重试只针对当前原文块；它不会重跑已成功块，也不会因一块最终
        # 失败而清空整个场景。English: Retry only the current source chunk;
        # successful siblings are never repeated or discarded with a failed part.
        for attempt in range(1, DEFAULT_PREPROCESSING_CHUNK_ATTEMPTS + 1):
            try:
                draft = _extract_structured_response(
                    text=chunk,
                    response_model=PreprocessingDraft,
                    model=model,
                    system_prompt=_PREPROCESSING_SYSTEM_PROMPT,
                )
                if attempt > 1:
                    notes.append(
                        "Automatic preprocessing chunk "
                        f"{chunk_index}/{len(chunks)} succeeded on attempt "
                        f"{attempt}/{DEFAULT_PREPROCESSING_CHUNK_ATTEMPTS}."
                    )
                break
            except Exception as error:
                last_error = error
                notes.append(
                    "Automatic preprocessing chunk "
                    f"{chunk_index}/{len(chunks)} attempt "
                    f"{attempt}/{DEFAULT_PREPROCESSING_CHUNK_ATTEMPTS} failed: "
                    f"{type(error).__name__}: {str(error).splitlines()[0]}."
                )
        if draft is None:
            notes.append(
                "Automatic preprocessing chunk "
                f"{chunk_index}/{len(chunks)} failed after "
                f"{DEFAULT_PREPROCESSING_CHUNK_ATTEMPTS} attempts and was omitted."
            )
            continue

        # 第二步：每块只保留能在该块原文中定位的 Evidence。
        # Step 2: Keep only evidence that is grounded in that source chunk.
        grounded, chunk_notes = _filter_and_order_grounded_entries(draft, chunk)
        grounded_chunks.append(_apply_entry_types(grounded))
        notes.extend(
            f"Chunk {chunk_index}/{len(chunks)}: {note}" for note in chunk_notes
        )

    if not grounded_chunks:
        raise AutomaticPreprocessingError(
            f"Automatic preprocessing with the local LLM failed (model: {model})."
        ) from last_error

    grounded_draft = _merge_preprocessing_drafts(grounded_chunks, raw_text)
    # 中文：把栏目语义写入条目；工具/对象仍由 LLM 判断，缺失时保守按 Object。
    # English: Persist column semantics. Tool/object is the LLM decision; a
    # missing type defaults conservatively to Object.
    grounded_draft = _apply_entry_types(grounded_draft)
    # 第三步：只对工具/对象做名称统一，人物、动作和参数保持原样。
    # Step 3: Canonicalize only tools/objects; keep other categories unchanged.
    normalized_tools, alias_notes = _canonicalize_tools_objects(
        grounded_draft.tools_objects,
        raw_text=raw_text,
        model=model,
        embedding_model=embedding_model,
        threshold=alias_candidate_threshold,
        top_k=alias_top_k,
    )
    grounded_draft.tools_objects = normalized_tools
    notes.extend(alias_notes)
    # 第四步：把有效分类、标准名称和处理说明交回 unified_text.py。
    # Step 4: Return grounded categories, canonical names, and notes to the caller.
    return grounded_draft, notes


def _split_preprocessing_chunks(raw_text: str, max_chars: int) -> list[str]:
    """Split source text without rewriting it or introducing overlap.

    Prefer sentence/newline boundaries and use whitespace only for an
    exceptionally long single sentence.
    """

    source = raw_text.strip()
    if len(source) <= max_chars:
        return [source]

    chunks: list[str] = []
    start = 0
    while start < len(source):
        remaining = len(source) - start
        if remaining <= max_chars:
            chunks.append(source[start:].strip())
            break

        window = source[start : start + max_chars + 1]
        sentence_boundaries = [
            match.end()
            for match in re.finditer(r"[.!?;](?=\s)|\n", window)
            if match.end() >= max_chars // 2
        ]
        if sentence_boundaries:
            cut = sentence_boundaries[-1]
        else:
            whitespace = window.rfind(" ")
            cut = whitespace if whitespace >= max_chars // 2 else max_chars
        end = start + max(cut, 1)
        chunk = source[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(source) and source[start].isspace():
            start += 1
    return chunks


def _merge_preprocessing_drafts(
    drafts: list[PreprocessingDraft],
    raw_text: str,
) -> PreprocessingDraft:
    """Merge chunk results, deduplicate them, and restore global source order."""

    categories = (
        "actors",
        "action_sequence",
        "tools_objects",
        "process_parameters",
        "quality_results",
    )
    updates: dict[str, list[GroundedEntry]] = {}
    for category in categories:
        merged: list[GroundedEntry] = []
        seen: set[tuple[str, str, str]] = set()
        for draft in drafts:
            for entry in getattr(draft, category):
                key = (
                    _normalize_text(entry.text),
                    _normalize_text(entry.evidence),
                    entry.entry_type or "",
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(entry)
        updates[category] = sorted(
            merged,
            key=lambda item: _evidence_start(item, raw_text),
        )
    return PreprocessingDraft(**updates)


def _apply_entry_types(draft: PreprocessingDraft) -> PreprocessingDraft:
    """Attach normalized mention kinds without source-specific vocabularies."""

    fixed_types = {
        "actors": "role",
        "action_sequence": "action",
        "process_parameters": "parameter",
        "quality_results": "quality",
    }
    updates: dict[str, list[GroundedEntry]] = {}
    for category, entry_type in fixed_types.items():
        updates[category] = [
            entry.model_copy(update={"entry_type": entry_type})
            for entry in getattr(draft, category)
        ]
    updates["tools_objects"] = [
        entry.model_copy(
            update={
                "entry_type": (
                    entry.entry_type
                    if entry.entry_type in {"tool", "object"}
                    else "object"
                )
            }
        )
        for entry in draft.tools_objects
    ]
    return draft.model_copy(update=updates, deep=True)


def _extract_structured_response(
    *,
    text: str,
    response_model: type[BaseModel],
    model: str,
    system_prompt: str,
) -> Any:
    """Delay the Ollama import so deterministic preprocessing stays lightweight."""

    from backend.llm.client import extract_structured

    return extract_structured(
        text=text,
        response_model=response_model,
        model=model,
        system_prompt=system_prompt,
        temperature=0.0,
        # EN: Full-corpus runs must expose a slow/invalid generation instead of
        # multiplying it into several opaque multi-minute retries.
        # ZH: 全量实测要显式暴露慢生成或非法输出，不能将其隐式放大为多轮、
        # 数分钟的重试。
        max_retries=0,
    )


# [Block 07] Evidence 门卫：过滤无原文证据的条目，并按原文顺序排列。
# [Block 07] Evidence gate: drop unsupported entries and restore source order.
def _filter_and_order_grounded_entries(
    draft: PreprocessingDraft,
    raw_text: str,
) -> tuple[PreprocessingDraft, list[str]]:
    """Drop unsupported automatic entries while preserving valid results."""

    notes: list[str] = []
    updates: dict[str, list[GroundedEntry]] = {}
    for category in (
        "actors",
        "action_sequence",
        "tools_objects",
        "process_parameters",
        "quality_results",
    ):
        valid: list[GroundedEntry] = []
        seen: set[tuple[str, str]] = set()
        for raw_entry in getattr(draft, category):
            # 中文：本地模型偶尔会在“精确引文”外再包一层引号。仅当去掉
            # 外层引号后的内容能逐字定位到原文时才接受，内部文字绝不改写。
            # English: Local models sometimes wrap an exact quote in one extra
            # pair of quotes. Strip that wrapper only when the inner text occurs
            # verbatim in the source; never rewrite the evidence itself.
            entry = _normalize_grounded_entry_evidence(raw_entry, raw_text)
            invalid_evidence = [
                evidence
                for evidence in _entry_evidence(entry)
                if not _evidence_is_grounded(evidence, raw_text)
            ]
            if not entry.text.strip() or invalid_evidence:
                evidence_label = invalid_evidence[0] if invalid_evidence else entry.evidence
                notes.append(
                    f"Filtered automatic {category} entry {entry.text!r} because "
                    f"evidence {evidence_label!r} was not found in source text."
                )
                continue
            key = (_normalize_text(entry.text), _normalize_text(entry.evidence))
            if key in seen:
                continue
            seen.add(key)
            valid.append(entry)
        updates[category] = sorted(valid, key=lambda item: _evidence_start(item, raw_text))
    return draft.model_copy(update=updates, deep=True), notes


# [Block 08] 名称统一：目录精确匹配、向量选候选、LLM 最终复核。
# [Block 08] Name canonicalization: exact catalog match, vector shortlist, LLM review.
def _canonicalize_tools_objects(
    entries: list[GroundedEntry],
    *,
    raw_text: str,
    model: str,
    embedding_model: str,
    threshold: float,
    top_k: int,
) -> tuple[list[GroundedEntry], list[str]]:
    """Map mentions to known terms, then merge remaining scene-local aliases."""

    if not entries:
        return [], []

    notes: list[str] = []
    try:
        # 关键词库是“小型已知名称表”，不是世界上所有工具的完整清单。
        # The vocabulary is a small known-name list, not an exhaustive tool catalog.
        catalog = _load_default_tool_object_terms()
    except Exception as error:
        catalog = ()
        notes.append(
            "Tool/object vocabulary could not be loaded; original names were kept: "
            f"{error}."
        )

    catalog_by_normalized = {_normalize_text(term): term for term in catalog}
    canonicalized: list[GroundedEntry] = []
    mapped_to_catalog: list[bool] = []
    unresolved_indices: list[int] = []

    for entry in entries:
        # 名称完全相同时直接采用目录名称，不需要加载向量模型或再次调用 LLM。
        # Exact names use the catalog term directly without embeddings or another LLM call.
        exact = catalog_by_normalized.get(_normalize_text(entry.text))
        if exact:
            canonicalized.append(_with_canonical_name(entry, exact))
            mapped_to_catalog.append(True)
        else:
            unresolved_indices.append(len(canonicalized))
            canonicalized.append(entry.model_copy(deep=True))
            mapped_to_catalog.append(False)

    if not unresolved_indices:
        return _merge_same_names(canonicalized, raw_text), notes

    try:
        mention_embeddings = _encode_texts(
            [entry.text for entry in canonicalized],
            embedding_model,
        )
        catalog_embeddings = (
            _encode_catalog(catalog, embedding_model) if catalog else None
        )
    except Exception as error:
        notes.append(
            "Vector name canonicalization was unavailable; original names were kept: "
            f"{error}."
        )
        return _merge_same_names(canonicalized, raw_text), notes

    if catalog_embeddings is not None:
        for index in unresolved_indices:
            # 向量只负责从关键词库中选出少量相近候选，不直接决定是否合并。
            # Embeddings shortlist similar candidates; they do not decide the merge.
            candidates = _top_candidates(
                mention_embeddings[index],
                catalog_embeddings,
                catalog,
                threshold=threshold,
                top_k=top_k,
            )
            if not candidates:
                continue
            try:
                # 最终是否为同一个物品由本地 LLM 保守复核。
                # The local LLM makes the final conservative same-entity decision.
                decision = _judge_catalog_alias(
                    entry=canonicalized[index],
                    candidates=candidates,
                    raw_text=raw_text,
                    model=model,
                )
            except Exception as error:
                notes.append(
                    f"Catalog verification for name {canonicalized[index].text!r} "
                    f"failed; the original name was kept: {error}."
                )
                continue
            candidate_map = {_normalize_text(item): item for item in candidates}
            selected = candidate_map.get(_normalize_text(decision.canonical_name or ""))
            if decision.same_entity and selected:
                canonicalized[index] = _with_canonical_name(
                    canonicalized[index],
                    selected,
                )
                mapped_to_catalog[index] = True

    unmatched = [
        index for index, mapped in enumerate(mapped_to_catalog) if not mapped
    ]
    groups = _scene_alias_groups(
        canonicalized,
        unmatched,
        mention_embeddings,
        raw_text=raw_text,
        model=model,
        threshold=threshold,
        top_k=top_k,
        notes=notes,
    )
    merged = [_merge_entry_group(group, raw_text) for group in groups]
    return _merge_same_names(merged, raw_text), notes


def _scene_alias_groups(
    entries: list[GroundedEntry],
    unmatched: list[int],
    embeddings: np.ndarray,
    *,
    raw_text: str,
    model: str,
    threshold: float,
    top_k: int,
    notes: list[str],
) -> list[list[GroundedEntry]]:
    """Use vector candidates and conservative LLM checks for local aliases."""

    parent = list(range(len(entries)))

    def find(index: int) -> int:
        """Find the root of an alias group."""

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """Merge two alias groups."""

        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    candidate_pairs: dict[tuple[int, int], float] = {}
    for left in unmatched:
        scored = [
            (right, float(np.dot(embeddings[left], embeddings[right])))
            for right in unmatched
            if right != left
        ]
        for right, score in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]:
            if score >= threshold:
                pair = tuple(sorted((left, right)))
                candidate_pairs[pair] = max(score, candidate_pairs.get(pair, 0.0))

    for (left, right), _score in sorted(
        candidate_pairs.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        try:
            decision = _judge_scene_alias(
                left=entries[left],
                right=entries[right],
                raw_text=raw_text,
                model=model,
            )
        except Exception as error:
            notes.append(
                f"Scene-local alias verification for {entries[left].text!r}/"
                f"{entries[right].text!r} failed; names were not merged: {error}."
            )
            continue
        if decision.same_entity:
            union(left, right)

    grouped: dict[int, list[GroundedEntry]] = {}
    for index, entry in enumerate(entries):
        grouped.setdefault(find(index), []).append(entry)
    return list(grouped.values())


def _judge_catalog_alias(
    *,
    entry: GroundedEntry,
    candidates: list[str],
    raw_text: str,
    model: str,
) -> _CatalogAliasDecision:
    """Verify a mention against catalog candidates."""

    prompt = {
        "source_text": raw_text,
        "mention": entry.text,
        "mention_evidence": entry.evidence,
        "catalog_candidates": candidates,
    }
    return _extract_structured_response(
        text=json.dumps(prompt, ensure_ascii=False),
        response_model=_CatalogAliasDecision,
        model=model,
        system_prompt=(
            "Decide conservatively whether the source mention denotes exactly the same "
            "physical tool/object as one catalog candidate. Related types are not enough. "
            "If yes, set same_entity=true and copy exactly one supplied candidate into "
            "canonical_name. Otherwise return false and null."
        ),
    )


def _judge_scene_alias(
    *,
    left: GroundedEntry,
    right: GroundedEntry,
    raw_text: str,
    model: str,
) -> _PairAliasDecision:
    """Verify whether two scene mentions are co-referent."""

    prompt = {
        "source_text": raw_text,
        "left": {"name": left.text, "evidence": left.evidence},
        "right": {"name": right.text, "evidence": right.evidence},
    }
    return _extract_structured_response(
        text=json.dumps(prompt, ensure_ascii=False),
        response_model=_PairAliasDecision,
        model=model,
        system_prompt=(
            "Return same_entity=true only when both mentions refer to the same physical "
            "tool/object occurrence in this scene. Similar or related objects stay separate."
        ),
    )


@lru_cache(maxsize=1)
def _load_default_tool_object_terms() -> tuple[str, ...]:
    """Load the IndEgo vocabulary lazily to avoid an import-time cycle."""

    from backend.datasets.indego_adapter import _TOOL_OBJECT_TERMS

    return tuple(dict.fromkeys(_TOOL_OBJECT_TERMS))


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    """Load and cache the embedding model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_texts(texts: Sequence[str], model_name: str) -> np.ndarray:
    """Encode and normalize text embeddings."""

    model = _load_embedding_model(model_name)
    encoded = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(encoded, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


@lru_cache(maxsize=8)
def _cached_catalog_embeddings(
    catalog: tuple[str, ...],
    model_name: str,
) -> np.ndarray:
    """Cache embeddings for catalog terms."""

    return _encode_texts(catalog, model_name)


def _encode_catalog(catalog: Sequence[str], model_name: str) -> np.ndarray:
    """Encode catalog terms as vectors."""

    return _cached_catalog_embeddings(tuple(catalog), model_name)


def _top_candidates(
    mention_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog: Sequence[str],
    *,
    threshold: float,
    top_k: int,
) -> list[str]:
    """Return the top-K candidates above the threshold."""

    scores = catalog_embeddings @ mention_embedding
    ranked = np.argsort(-scores)[:top_k]
    return [str(catalog[index]) for index in ranked if float(scores[index]) >= threshold]


def _with_canonical_name(entry: GroundedEntry, canonical_name: str) -> GroundedEntry:
    """Store a canonical name without replacing the grounded text."""

    return entry.model_copy(
        update={
            "canonical_name": canonical_name,
        },
        deep=True,
    )


def _merge_same_names(
    entries: Iterable[GroundedEntry],
    raw_text: str,
) -> list[GroundedEntry]:
    """Merge entries with the same normalized name."""

    grouped: dict[str, list[GroundedEntry]] = {}
    order: list[str] = []
    for entry in entries:
        # 中文：同一标准名可以合并证据，但最终显示仍采用最早的原文名称。
        # English: Canonical identity may group evidence while display text stays sourced.
        key = _normalize_text(entry.canonical_name or entry.text)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(entry)
    merged = [_merge_entry_group(grouped[key], raw_text) for key in order]
    return sorted(merged, key=lambda item: _evidence_start(item, raw_text))


def _merge_entry_group(
    entries: list[GroundedEntry],
    raw_text: str,
) -> GroundedEntry:
    """Merge one alias group and its evidence spans."""

    # 中文：优先选择原文中真实出现的名称，避免用 LLM 改写名替换证据名称。
    # English: Prefer a source-occurring mention over an LLM paraphrase.
    ordered = sorted(
        entries,
        key=lambda item: (
            not _evidence_is_grounded(item.text, raw_text),
            _evidence_start(item, raw_text),
        ),
    )
    primary = ordered[0]
    aliases: list[str] = []
    evidence: list[str] = []
    uncertainties: list[str] = []
    canonical_names: list[str] = []
    for entry in ordered:
        aliases.extend([entry.text, *entry.aliases])
        evidence.extend(_entry_evidence(entry))
        if entry.canonical_name:
            canonical_names.append(entry.canonical_name)
        if entry.uncertainty:
            uncertainties.append(entry.uncertainty)
    unique_evidence = _unique_texts(evidence)
    return primary.model_copy(
        update={
            "aliases": _unique_texts(aliases, exclude=primary.text),
            "canonical_name": (
                _unique_texts(canonical_names)[0] if canonical_names else None
            ),
            "evidence": unique_evidence[0],
            "additional_evidence": unique_evidence[1:],
            "uncertainty": "; ".join(_unique_texts(uncertainties)) or None,
        },
        deep=True,
    )


def _entry_evidence(entry: GroundedEntry) -> list[str]:
    """Return all evidence spans for an entry."""

    return [entry.evidence, *entry.additional_evidence]


def _normalize_grounded_entry_evidence(
    entry: GroundedEntry,
    raw_text: str,
) -> GroundedEntry:
    """Remove a model-added outer quote only when the inner span is grounded."""

    evidence = _grounded_evidence_value(entry.evidence, raw_text)
    additional = [
        _grounded_evidence_value(value, raw_text)
        for value in entry.additional_evidence
    ]
    if evidence == entry.evidence and additional == entry.additional_evidence:
        return entry
    return entry.model_copy(
        update={"evidence": evidence, "additional_evidence": additional},
        deep=True,
    )


def _grounded_evidence_value(evidence: str, raw_text: str) -> str:
    """Return the smallest safely unwrapped source-grounded evidence value."""

    cleaned = evidence.strip()
    if _evidence_is_grounded(cleaned, raw_text):
        return cleaned
    quote_pairs = (("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”"), ("`", "`"))
    for opening, closing in quote_pairs:
        if cleaned.startswith(opening) and cleaned.endswith(closing):
            inner = cleaned[len(opening) : -len(closing)].strip()
            if _evidence_is_grounded(inner, raw_text):
                return inner
    return cleaned


def _evidence_is_grounded(evidence: str, raw_text: str) -> bool:
    """Check whether evidence occurs in the source text."""

    normalized_evidence = _normalize_text(evidence)
    normalized_source = _normalize_text(raw_text)
    return bool(normalized_evidence) and normalized_evidence in normalized_source


def _evidence_start(entry: GroundedEntry, raw_text: str) -> int:
    """Locate the evidence start in the source text."""

    parts = [re.escape(part) for part in entry.evidence.split()]
    if not parts:
        return len(raw_text)
    match = re.search(r"\s+".join(parts), raw_text, flags=re.IGNORECASE)
    return match.start() if match else len(raw_text)


def _unique_texts(values: Iterable[str], *, exclude: str | None = None) -> list[str]:
    """Deduplicate normalized text while preserving source order."""

    excluded = _normalize_text(exclude or "")
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        normalized = _normalize_text(cleaned)
        if not normalized or normalized == excluded or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(cleaned)
    return unique
