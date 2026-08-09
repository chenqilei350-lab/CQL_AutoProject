from backend.datasets.industrial_reviewed_gold import (
    load_industrial_reviewed_gold_dataset,
)
from backend.pipeline.relation_candidate_pipeline import (
    EntityMention,
    ProceduralRelationCandidateGenerator,
)
from scripts.run_industrial_reviewed_gold_diagnostics import gold_edge_set


def test_oracle_candidate_diagnostic_uses_reviewed_entity_ids() -> None:
    scene = load_industrial_reviewed_gold_dataset(limit=1).scenes[0]
    extraction = scene.gold_extraction
    actions = [
        EntityMention(
            entity_id=action.entity_id or "",
            label="Action",
            name=action.name,
            evidence_text=action.evidence_text or "",
            sequence_index=action.sequence_index,
        )
        for action in extraction.actions
    ]
    tools = [
        EntityMention(
            entity_id=tool.entity_id or "",
            label="Tool",
            name=tool.name,
            evidence_text=tool.evidence_text or "",
        )
        for tool in extraction.tools
    ]
    objects = [
        EntityMention(
            entity_id=obj.entity_id or "",
            label="Object",
            name=obj.name,
            evidence_text=obj.evidence_text or "",
        )
        for obj in extraction.objects
    ]

    candidates = ProceduralRelationCandidateGenerator().generate(
        actions=actions,
        tools=tools,
        objects=objects,
        source_text=scene.raw_text,
    )
    candidate_edges = {
        (candidate.subject_id, candidate.relation_type, candidate.object_id)
        for candidate in candidates
        if candidate.relation_type in {"BEFORE", "ACTS_ON", "USES_TOOL"}
    }

    gold_edges = gold_edge_set(extraction)
    assert len(gold_edges) == 30
    assert any(edge[1] == "BEFORE" for edge in candidate_edges)
    assert candidate_edges & gold_edges
