"""
面向工业原文的 LLM 辅助、证据约束文本预处理。
LLM-assisted, evidence-grounded preprocessing for industrial source text.

初学者阅读提示：可以把本文件理解为“自动整理工作人员”。它依次完成：
1. 让 LLM 把原文分成人物、动作、工具/对象、参数和质量结果；
2. 删除 Evidence 不在原文中的条目；
3. 把工具/对象名称与现有关键词库比较；
4. 用向量找相近名称，再让 LLM 确认是否为同一个物品；
5. 返回整理结果给 unified_text.py。

Beginner guide: treat this file as the automatic preprocessing worker. It asks
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


# [Block 05] 给 LLM 的固定分类表，只允许输出五个预设栏目。
# [Block 05] Fixed LLM classification form with only five allowed categories.
class PreprocessingDraft(BaseModel):
    """一次 LLM 调用产生的五类结果。 / Five grounded LLM categories."""

    actors: list[GroundedEntry] = Field(default_factory=list)
    action_sequence: list[GroundedEntry] = Field(default_factory=list)
    tools_objects: list[GroundedEntry] = Field(default_factory=list)
    process_parameters: list[GroundedEntry] = Field(default_factory=list)
    quality_results: list[GroundedEntry] = Field(default_factory=list)


class _CatalogAliasDecision(BaseModel):
    """名称到目录词的映射判断。 / Decision mapping a mention to a catalog term."""

    same_entity: bool
    canonical_name: str | None = None


class _PairAliasDecision(BaseModel):
    """场景内两个名称的同一性判断。 / Same-entity decision for two mentions."""

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
- evidence must be an exact contiguous quote copied from the source text.
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
) -> tuple[PreprocessingDraft, list[str]]:
    """
    分类原文、过滤无证据条目并统一工具名称。
    Classify source text, filter ungrounded entries, and normalize tool names.
    """

    if not raw_text.strip():
        raise ValueError("raw_text must not be empty.")
    if not 0.0 <= alias_candidate_threshold <= 1.0:
        raise ValueError("alias_candidate_threshold must be between 0.0 and 1.0.")
    if alias_top_k < 1:
        raise ValueError("alias_top_k must be at least 1.")

    # 第一步：一次 LLM 调用完成五个预设类别的初步分类。
    # Step 1: One LLM call produces the five predefined categories.
    try:
        draft = _extract_structured_response(
            text=raw_text,
            response_model=PreprocessingDraft,
            model=model,
            system_prompt=_PREPROCESSING_SYSTEM_PROMPT,
        )
    except Exception as error:
        raise AutomaticPreprocessingError(
            f"Automatic preprocessing with the local LLM failed (model: {model})."
        ) from error

    # 第二步：检查每条 Evidence；找不到原文证据的条目会被过滤并记录。
    # Step 2: Filter entries whose evidence cannot be found in the source text.
    grounded_draft, notes = _filter_and_order_grounded_entries(draft, raw_text)
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


def _extract_structured_response(
    *,
    text: str,
    response_model: type[BaseModel],
    model: str,
    system_prompt: str,
) -> Any:
    """
    延迟导入 Ollama 客户端，保持确定性路径轻量。
    Delay the Ollama import so deterministic preprocessing stays lightweight.
    """

    from backend.llm.client import extract_structured

    return extract_structured(
        text=text,
        response_model=response_model,
        model=model,
        system_prompt=system_prompt,
        temperature=0.0,
        max_retries=3,
    )


# [Block 07] Evidence 门卫：过滤无原文证据的条目，并按原文顺序排列。
# [Block 07] Evidence gate: drop unsupported entries and restore source order.
def _filter_and_order_grounded_entries(
    draft: PreprocessingDraft,
    raw_text: str,
) -> tuple[PreprocessingDraft, list[str]]:
    """
    只过滤无原文支持的自动条目，并保留其余有效结果。
    Drop unsupported automatic entries while preserving valid results.
    """

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
        for entry in getattr(draft, category):
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
    """
    先映射已知目录词，再合并场景内剩余别名。
    Map mentions to known terms, then merge remaining scene-local aliases.
    """

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
    """
    使用向量候选和保守的 LLM 复核判断场景内别名。
    Use vector candidates and conservative LLM checks for scene-local aliases.
    """

    parent = list(range(len(entries)))

    def find(index: int) -> int:
        """查找合并组根节点。 / Find the root of an alias group."""

        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """合并两个别名组。 / Merge two alias groups."""

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
    """复核名称是否对应目录候选。 / Verify a mention against catalog candidates."""

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
    """复核场景内两个名称是否同指。 / Verify two scene mentions are co-referent."""

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
    """
    延迟加载 IndEgo 词表，避免导入循环。
    Load the IndEgo vocabulary lazily to avoid an import-time cycle.
    """

    from backend.datasets.indego_adapter import _TOOL_OBJECT_TERMS

    return tuple(dict.fromkeys(_TOOL_OBJECT_TERMS))


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    """加载并缓存向量模型。 / Load and cache the embedding model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode_texts(texts: Sequence[str], model_name: str) -> np.ndarray:
    """编码并归一化文本向量。 / Encode and normalize text embeddings."""

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
    """缓存目录词向量。 / Cache embeddings for catalog terms."""

    return _encode_texts(catalog, model_name)


def _encode_catalog(catalog: Sequence[str], model_name: str) -> np.ndarray:
    """把目录词编码为向量。 / Encode catalog terms as vectors."""

    return _cached_catalog_embeddings(tuple(catalog), model_name)


def _top_candidates(
    mention_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog: Sequence[str],
    *,
    threshold: float,
    top_k: int,
) -> list[str]:
    """返回高于阈值的前 K 个候选。 / Return top-K candidates above threshold."""

    scores = catalog_embeddings @ mention_embedding
    ranked = np.argsort(-scores)[:top_k]
    return [str(catalog[index]) for index in ranked if float(scores[index]) >= threshold]


def _with_canonical_name(entry: GroundedEntry, canonical_name: str) -> GroundedEntry:
    """采用标准名称并保留原别名。 / Apply a canonical name and retain aliases."""

    aliases = list(entry.aliases)
    if _normalize_text(entry.text) != _normalize_text(canonical_name):
        aliases.append(entry.text)
    return entry.model_copy(
        update={
            "text": canonical_name,
            "aliases": _unique_texts(aliases, exclude=canonical_name),
        },
        deep=True,
    )


def _merge_same_names(
    entries: Iterable[GroundedEntry],
    raw_text: str,
) -> list[GroundedEntry]:
    """合并标准化后名称相同的条目。 / Merge entries with the same normalized name."""

    grouped: dict[str, list[GroundedEntry]] = {}
    order: list[str] = []
    for entry in entries:
        key = _normalize_text(entry.text)
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
    """合并一组别名及其证据。 / Merge one alias group and its evidence spans."""

    ordered = sorted(entries, key=lambda item: _evidence_start(item, raw_text))
    primary = ordered[0]
    aliases: list[str] = []
    evidence: list[str] = []
    uncertainties: list[str] = []
    for entry in ordered:
        aliases.extend([entry.text, *entry.aliases])
        evidence.extend(_entry_evidence(entry))
        if entry.uncertainty:
            uncertainties.append(entry.uncertainty)
    unique_evidence = _unique_texts(evidence)
    return primary.model_copy(
        update={
            "aliases": _unique_texts(aliases, exclude=primary.text),
            "evidence": unique_evidence[0],
            "additional_evidence": unique_evidence[1:],
            "uncertainty": "; ".join(_unique_texts(uncertainties)) or None,
        },
        deep=True,
    )


def _entry_evidence(entry: GroundedEntry) -> list[str]:
    """返回条目的全部证据。 / Return all evidence spans for an entry."""

    return [entry.evidence, *entry.additional_evidence]


def _evidence_is_grounded(evidence: str, raw_text: str) -> bool:
    """检查证据是否来自原文。 / Check whether evidence occurs in source text."""

    normalized_evidence = _normalize_text(evidence)
    normalized_source = _normalize_text(raw_text)
    return bool(normalized_evidence) and normalized_evidence in normalized_source


def _evidence_start(entry: GroundedEntry, raw_text: str) -> int:
    """定位证据在原文中的起点。 / Locate the evidence start in source text."""

    parts = [re.escape(part) for part in entry.evidence.split()]
    if not parts:
        return len(raw_text)
    match = re.search(r"\s+".join(parts), raw_text, flags=re.IGNORECASE)
    return match.start() if match else len(raw_text)


def _unique_texts(values: Iterable[str], *, exclude: str | None = None) -> list[str]:
    """按标准化文本去重并保持顺序。 / Deduplicate normalized text in source order."""

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
