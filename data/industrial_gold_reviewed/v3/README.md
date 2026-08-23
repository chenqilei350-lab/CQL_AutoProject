# Industrial Gold V3 Fully Reviewed Dataset (20 Records)

This batch was reselected and manually reviewed from the complete standardized
IndEgo inputs. It explicitly excludes:

- table or desk disassembly scenes;
- computer, PC, workstation, or embedded-computing-unit assembly scenes;
- all six `scene_id` values from the uploaded reference JSONL.

## Files

- `industrial_gold_v3_reviewed.jsonl`: 20 final reviewed Gold records in JSONL.
- `industrial_gold_v3_reviewed.json`: the same records as a JSON array.
- `examples/`: individual Gold JSON files numbered 01 through 20.
- A review-notes directory containing Chinese per-record notes numbered 01 through 20.
- A combined Chinese review-notes document for all 20 records.
- A full optimization report covering before/after action, tool, and relation analysis.
- `manifest.json`: mapping of number, scene ID, category, task, and file path.
- `validation_report.json`: automated counts, exclusions, evidence, and endpoint checks.

## Review Rules

1. V2 is used only as a structural and review-style reference; its records are
   not copied or mixed into V3.
2. Explicit inspections, cleaning, corrections, failed attempts, reassembly,
   and cleanup actions in the full transcript must not be omitted.
3. A tool explicitly used in a sentence requires a Tool node and a `USES_TOOL`
   edge. Merely mentioned items must not be mislabeled as used tools.
4. Create only evidence-supported adjacent `BEFORE` edges. Warning templates
   combine multiple runs, so no cross-run `BEFORE` edge is allowed.
5. Ambiguous pronouns, component identity, tool mapping, and missing outcomes
   remain review notes and are excluded from formal Gold facts.

Automated validation status: **passed**. The dataset contains **224** Actions,
**34** Tools, **72** `USES_TOOL` edges, and **35** items requiring confirmation.
