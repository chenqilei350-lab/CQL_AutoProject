"""
Extractor — the core extraction interface.

Takes text + a Pydantic schema → returns structured data.
Model-agnostic: works with any Ollama model (base or fine-tuned).

Usage:
    from backend.extraction.extractor import Extractor
    from backend.schemas.process_knowledge.entities import Tool

    extractor = Extractor(model="llama3.1:8b")
    
    # Extract a single type
    tool = extractor.extract(
        text="Wir verwenden das MIG-Schweißgerät Fronius TPS 400i.",
        response_model=Tool,
    )
    
    # Extract a list of entities
    tools = extractor.extract_list(
        text="Benötigt werden: Fronius TPS 400i, Winkelschleifer und Drehmomentschlüssel.",
        item_model=Tool,
    )
"""

from typing import TypeVar, Type, List, Optional
from pydantic import BaseModel, create_model
from backend.llm.client import (
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    extract_structured,
)
from backend.schemas.egocentric_video import EgocentricVideoExtraction
from backend.schemas.ontology import build_egocentric_ontology, compact_schema_guide

T = TypeVar("T", bound=BaseModel)


class Extractor:
    """
    Schema-driven extraction from text using LLMs.
    
    Wraps the LLM client with extraction-specific prompting
    and supports both single-item and list extraction.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str = "German",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ):
        self.model = model
        self.language = language
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def extract(
        self,
        text: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Extract a single structured object from text.
        
        Args:
            text: Source text to extract from
            response_model: Pydantic model defining what to extract
            system_prompt: Optional custom system prompt
        
        Returns:
            Instance of response_model with extracted data
        """
        prompt = system_prompt or self._build_system_prompt(response_model)

        return extract_structured(
            text=text,
            response_model=response_model,
            model=self.model,
            system_prompt=prompt,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def extract_list(
        self,
        text: str,
        item_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> List[T]:
        """
        Extract a list of structured objects from text.
        
        Creates a wrapper model with a list field, extracts, then unwraps.
        
        Args:
            text: Source text to extract from
            item_model: Pydantic model for each item
            system_prompt: Optional custom system prompt
            
        Returns:
            List of item_model instances
        """
        # Dynamically create a wrapper model: class Items(BaseModel): items: List[item_model]
        wrapper_model = create_model(
            f"{item_model.__name__}List",
            items=(List[item_model], ...),
        )

        prompt = system_prompt or self._build_list_system_prompt(item_model)

        result = extract_structured(
            text=text,
            response_model=wrapper_model,
            model=self.model,
            system_prompt=prompt,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

        return getattr(result, "items")

    def _build_system_prompt(self, response_model: Type[BaseModel]) -> str:
        """Build a default system prompt for single-item extraction."""
        model_name = response_model.__name__
        docstring = response_model.__doc__ or ""
        
        return (
            f"You are an expert information extractor for industrial manufacturing knowledge.\n"
            f"Extract a {model_name} from the given text.\n"
            f"Context: {docstring.strip()}\n"
            f"The input text is in {self.language}.\n"
            f"Only extract information that is explicitly stated in the text.\n"
            f"Leave fields as null if the information is not present.\n"
            f"Preserve source_text or evidence_text when the text supports an entity or relation.\n"
            f"{self._schema_guidance(response_model)}"
        )

    def _build_list_system_prompt(self, item_model: Type[BaseModel]) -> str:
        """Build a default system prompt for list extraction."""
        model_name = item_model.__name__
        docstring = item_model.__doc__ or ""

        return (
            f"You are an expert information extractor for industrial manufacturing knowledge.\n"
            f"Extract ALL {model_name} instances from the given text.\n"
            f"Context: {docstring.strip()}\n"
            f"The input text is in {self.language}.\n"
            f"Only extract information that is explicitly stated in the text.\n"
            f"Leave fields as null if the information is not present.\n"
            f"If no {model_name} is found, return an empty list.\n"
            f"Preserve source_text or evidence_text when the text supports an entity or relation.\n"
            f"{self._schema_guidance(item_model)}"
        )

    def _schema_guidance(self, response_model: Type[BaseModel]) -> str:
        """Build compact ontology/schema guidance for extraction prompts."""
        if response_model is EgocentricVideoExtraction:
            ontology = build_egocentric_ontology()
            entity_names = ", ".join(sorted(ontology.entities))
            relation_labels = ", ".join(
                sorted(relation.label for relation in ontology.relations.values())
            )
            constraints = "; ".join(
                f"{relation.label}: {relation.domain}->{relation.range}"
                for relation in ontology.relations.values()
                if relation.domain and relation.range
            )
            # 小模型需要简短、完整的关系合同；展开所有嵌套字段会挤占上下文，
            # 反而导致 BEFORE/CAUSES 等关键端点字段被截断。
            return "\n".join(
                [
                    "\nStrict output contract:",
                    "Every entity object must contain a `name` string.",
                    "Relation endpoints must be nested objects with `name`, never plain strings.",
                    "Use exact relation forms:",
                    '- uses_tool: [{"action": {"name": "..."}, "tool": {"name": "..."}}]',
                    '- acts_on_object: [{"action": {"name": "..."}, "object": {"name": "..."}}]',
                    '- action_order: [{"before": {"name": "..."}, "after": {"name": "..."}}]',
                    '- action_causes: [{"cause": {"name": "..."}, "effect": {"name": "..."}}]',
                    '- part_of_procedure: [{"action": {"name": "..."}, "procedure": {"name": "..."}}]',
                    '- observed_in_scene: [{"action": {"name": "..."}, "scene": {"name": "..."}}]',
                    f"Allowed entity types: {entity_names}.",
                    f"Allowed relation types: {relation_labels}.",
                    f"Relation domain/range constraints: {constraints}.",
                    "Do not invent entities, relations, parameters, or evidence absent from the input text.",
                ]
            )
        lines = ["\nCompact schema guidance:"]
        lines.append(compact_schema_guide(response_model))
        return "\n".join(lines)
