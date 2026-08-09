# Industrial Gold Candidate Templates

- Source input: `kg_ready_data/indego_standard_inputs.jsonl`
- Candidate scenes: `20`
- Status: `candidate_gold_not_reviewed`

These files are annotation templates, not final gold graphs.
A final gold graph must contain only source-supported facts verified by a human reviewer.

Review rules:

1. Keep a node or edge only if its evidence text is present in the source text.
2. Do not add facts inferred from domain knowledge alone.
3. Mark confirmed facts as `reviewed_source_supported`.
4. Save reviewed examples separately, for example as `industrial_gold_v2_reviewed.jsonl`.
