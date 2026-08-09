# Industrial Text-to-KG Baseline V1 Results

## 1. Status and Integrity

- Baseline ID: `industrial-v1-llama31-8b-20260809`
- Model: `llama3.1:8b@46e0c10c039e`
- Tested code commit: `28899d59b7b383a3e0d79a8624d313924b6e7bce`
- Dataset: 32 scenes, 30 video IDs, 674 Gold entities, 828 Gold relations
- Runs: 256 real local-LLM calls
- Parameters: temperature 0, two repetitions, 180 s hard timeout, 3000 maximum tokens
- Artifact verification: 14 SHA256 entries verified
- Baseline implementation regression tests: 137 passed
- Publication suite after adding license-safe export guards: 139 passed

All failed calls remain in the aggregate. The baseline model will not be run
again for future V1 comparisons.

The complete local baseline includes licensed cleaned records and raw model
responses for authorized rescoring. The GitHub-safe export excludes those
files and local source paths; it publishes only audited numeric metrics, an
aggregate source identity hash, and independent checksums.

## 2. Data Cleaning Baseline

The IndEgo cleaning module processed 579 recognized files and fingerprinted
1,163 source files, totaling 4,068,281 bytes. It emitted 7,883 normalized
records with provenance and no empty normalized texts.

| Record type | Count |
| --- | ---: |
| Action segment | 619 |
| Keystep segment | 2,392 |
| Mistake step | 314 |
| Mistake warning | 2,946 |
| Scenario keystep | 219 |
| Transcript | 394 |
| VQA item | 868 |
| File notice | 131 |

Recorded issues were 129 unexpanded reference-layer notices, two unparsed
tool/object metadata notices, and six missing-time-bound warnings. These are
retained as cleaning diagnostics rather than silently dropped.

## 3. Text Preprocessing Baseline

| Measure | Result |
| --- | ---: |
| Mean raw length | 2,469.7 characters |
| Mean aligned automatic unified length | 3,046.1 characters |
| Mean reviewed unified length | 4,170.1 characters |
| Original automatic grounded entries | 302 |
| Retained source-supported automatic entries | 134 |
| Automatic entries outside the Gold excerpt | 168 |
| Reviewed grounded entries | 741 |
| Exact automatic-source/Gold-source matches | 0 / 32 |
| Automatic action entries after alignment | 96 |
| Automatic tool/object entries after alignment | 12 |
| Reviewed action entries | 369 |
| Reviewed tool/object entries | 305 |

The aligned automatic entries have evidence support rate 1.0 because
unsupported entries are filtered. This does not mean preprocessing is complete:
55.6% of the original automatic entries fall outside the selected Gold source
excerpt. The main engineering issue is source and segment alignment between
IndEgo layers and reviewed scene excerpts.

## 4. End-to-End Results

The table reports the mean of per-run F1 values. `raw Edge F1` scores only
LLM-origin relations. `final Edge F1` includes explicitly marked grounded
post-processing and fallback relations.

| Condition | Runs | Errors | Node F1 | Raw Edge F1 | Final Edge F1 | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw + one-shot | 64 | 6 | 0.2374 | 0.0260 | 0.0799 | 91.10 s |
| Raw + layered | 64 | 5 | 0.2804 | 0.0947 | 0.1009 | 101.87 s |
| Auto-unified + layered | 64 | 3 | 0.2505 | 0.0418 | 0.0704 | 85.68 s |
| Reviewed-unified + layered | 64 | 12 | 0.7321 | 0.4423 | 0.6015 | 114.39 s |

There were 25 hard timeouts and one JSON parse failure across all conditions.

### Direct Comparisons

- Raw layered versus raw one-shot: Node F1 increases by 0.0429 and final Edge
  F1 by 0.0210, while mean runtime increases by 10.77 seconds.
- Automatic unified versus raw layered: the automatic condition is faster and
  has fewer failures, but final Edge F1 decreases by 0.0304.
- Reviewed unified versus raw layered: Node F1 increases by 0.4517 and final
  Edge F1 by 0.5006. This is a human-derived upper bound, not an automatic
  preprocessing result.
- Post-processing raises final Edge F1 above raw Edge F1 in every condition.
  These gains must be attributed to the final product pipeline, not to the LLM
  alone.

## 5. Relation-Type Results

Final relation-type F1 values:

| Condition | ACTS_ON | BEFORE | USES_TOOL |
| --- | ---: | ---: | ---: |
| Raw + one-shot | 0.0682 | 0.0515 | 0.0086 |
| Raw + layered | 0.0811 | 0.0499 | 0.0120 |
| Auto-unified + layered | 0.0412 | 0.0448 | 0.0113 |
| Reviewed-unified + layered | 0.6222 | 0.6578 | 0.2819 |

`USES_TOOL` is the weakest relation in all conditions. Even the reviewed upper
bound reaches only 0.2819 F1, so a dedicated candidate scorer and stronger
tool/entity linking remain necessary.

## 6. Gold-Entity Relation Diagnostic

With Gold entities supplied as an oracle, deterministic relation candidate
generation and validation produce:

| Metric | Result |
| --- | ---: |
| Gold relations | 828 |
| Accepted candidates | 773 |
| Precision | 0.8926 |
| Recall | 0.8333 |
| F1 | 0.8620 |

This diagnostic is not an end-to-end score. It shows that relation candidate
logic is much stronger when entity identity and evidence are already correct.
The end-to-end bottleneck therefore includes entity/evidence alignment and
relation linking, not only the final graph builder.

## 7. Stability and Metric Caveats

| Condition | Mean Node Jaccard | Mean Edge Jaccard |
| --- | ---: | ---: |
| Raw + one-shot | 1.0000 | 0.9984 |
| Raw + layered | 0.9509 | 0.9231 |
| Auto-unified + layered | 0.9688 | 0.9609 |
| Reviewed-unified + layered | 0.9904 | 0.9875 |

These values measure agreement, not quality. Identical empty graphs and paired
timeouts can inflate agreement. Stability must always be interpreted together
with F1 and error rate.

All conditions report ontology conformance 1.0 and relation hallucination rate
0.0 under the current validator. This means generated structures pass the
implemented schema/domain/range and basic grounding checks. It does not mean
all relations are semantically correct: relation scoring still contains many
false positives. A future hallucination validator should verify quoted evidence
for each candidate and distinguish schema validity, source grounding, and Gold
correctness.

## 8. Conclusions

1. Layered extraction is worth retaining: it improves mean Node and final Edge
   F1 over raw one-shot and slightly reduces failures.
2. Current automatic unified preprocessing is not yet a quality improvement.
   It improves speed and error count but loses substantial source-supported
   content because source excerpts and IndEgo layers are not aligned.
3. The reviewed representation demonstrates a large achievable upper bound,
   but also causes more timeouts because it is longer and produces larger
   outputs.
4. Edge prediction remains the main end-to-end bottleneck, especially
   `USES_TOOL`.
5. Gold-entity relation diagnostics show that candidate generation is useful;
   source alignment, entity extraction, and a dedicated relation scorer should
   be improved before adding a larger graph database framework.

## 9. Prioritized Changes

1. `EXP-SOURCE-ALIGNMENT-001`: align full IndEgo source layers to reviewed
   excerpts by video ID, timestamps, and evidence spans.
2. `EXP-STRUCTURED-OUTPUT-001`: use Ollama-native JSON Schema constrained
   output to reduce parse failures and separate syntax reliability from KG F1.
3. `EXP-RELATION-SCORER-001`: compare deterministic, LLM binary, GLiREL, and
   hybrid relation candidate scoring on manually reviewed relation Gold.
4. `EXP-ATOMIC-PREPROCESS-001`: test evidence-preserving atomic propositions
   without replacing raw source text.
5. `EXP-SIMILAR-FEW-SHOT-001`: retrieve development-only similar Gold examples
   and ablate context size.

Detailed methods and sources are documented in
`docs/industrial_baseline_and_improvement_plan_en.md`.
