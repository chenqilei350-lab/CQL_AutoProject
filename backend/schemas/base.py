"""
Base classes for all schema definitions.

These provide common fields and patterns used across domains.
Every entity and relation in the knowledge graph inherits from these.
"""

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, model_validator


class KGEntity(BaseModel):
    """Base class for all knowledge graph entities."""

    name: str
    entity_id: Optional[str] = None
    description: Optional[str] = None
    source_text: Optional[str] = None
    evidence_text: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_common_llm_fields(cls, data: Any) -> Any:
        """
        接受小模型常见的等价字段写法，同时仍由 schema 检查最终结构。

        本地模型有时会把实体名称输出为 ``description``，把证据输出为
        ``evidence``。它们的语义可以无歧义映射到现有字段，因此在进入
        graph 和 evaluation 前统一转换，而不新增任何原文之外的事实。
        """

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("name") and normalized.get("description"):
            normalized["name"] = normalized["description"]
        if not normalized.get("evidence_text") and normalized.get("evidence"):
            normalized["evidence_text"] = normalized["evidence"]
        return normalized


class KGRelation(BaseModel):
    """
    Base class for all knowledge graph relations.
    
    In a property graph (LadybugDB/Neo4j), n-ary relations are represented
    through reification: this relation becomes an intermediate node with 
    binary edges to each participant.
    
    In TypeDB, these map directly to n-ary relations with roles.
    """
    source_text: Optional[str] = None
    evidence_text: Optional[str] = None
    relation_origin: Optional[
        Literal[
            "llm_extracted",
            "evidence_fallback",
            "sequence_fallback",
            "postprocessed",
            "deterministic_candidate",
            "llm_binary_judge",
        ]
    ] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence_field(cls, data: Any) -> Any:
        """将模型输出的 ``evidence`` 别名保存为统一证据字段。"""

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if not normalized.get("evidence_text") and normalized.get("evidence"):
            normalized["evidence_text"] = normalized["evidence"]
        return normalized


class ExtractionResult(BaseModel):
    """
    Container for all extracted information from a single text chunk.
    
    The extractor fills this with whatever it finds in the text.
    Assembly and deduplication happen downstream.
    """
    entities: List[KGEntity] = []
    relations: List[KGRelation] = []
    source_chunk: Optional[str] = None
    extraction_model: Optional[str] = None
    extraction_timestamp: Optional[datetime] = None
