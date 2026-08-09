# Industrial KG V1 Baseline and Improvement Plan

## 1. Objective

The project converts video-derived industrial instructional text into an
ontology-guided procedural knowledge graph. The frozen V1 baseline separates
three concerns: data cleaning, text preprocessing, and KG extraction. Future
changes are evaluated as paired experiments against stored baseline outputs.
The original baseline model is not called again.

The research scope starts after video-to-text conversion. The system does not
claim to train a video model, and text preprocessing must not introduce facts
that are absent from the source.

## 2. Frozen Baseline Identity

| Item | Frozen value |
| --- | --- |
| Baseline ID | `industrial-v1-llama31-8b-20260809` |
| Git branch | `codex/industrial-baseline-v1` |
| Annotated result tag | `baseline-industrial-v1-2026-08-09` |
| Tested code commit | `28899d59b7b383a3e0d79a8624d313924b6e7bce` |
| Model | `llama3.1:8b@46e0c10c039e` |
| Reference device | Apple M4 MacBook Air, 16 GB |
| Temperature | `0` |
| Repetitions | `2` |
| Per-call hard timeout | `180 s` |
| Maximum output tokens | `3000` |
| Baseline implementation suite | `137 passed` |
| Publication suite after license guards | `139 passed` |

The full experiment is complete and 14 artifact checksums pass. Historical
scores such as Node F1 `0.6019` and Edge F1 `0.1421` remain historical
observations and are not relabeled as V1 baseline results.

### Final V1 Result Snapshot

| Condition | Errors / 64 | Mean Node F1 | Mean raw Edge F1 | Mean final Edge F1 | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw + one_shot` | 6 | 0.2374 | 0.0260 | 0.0799 | 91.10 s |
| `raw + layered` | 5 | 0.2804 | 0.0947 | 0.1009 | 101.87 s |
| `auto_unified + layered` | 3 | 0.2505 | 0.0418 | 0.0704 | 85.68 s |
| `reviewed_unified + layered` | 12 | 0.7321 | 0.4423 | 0.6015 | 114.39 s |

The Gold-entity relation diagnostic reaches precision `0.8926`, recall
`0.8333`, and F1 `0.8620`. This is an oracle diagnostic with Gold entities and
must not be compared directly with end-to-end model F1.

The complete baseline, including licensed cleaned records and raw model
responses, remains local and Git-ignored. The public export contains only
audited numeric metrics, an aggregate source identity hash, and checksums. This
separation preserves experiment identity without redistributing restricted
source text.

## 3. Versioned Modules

| Module | Purpose | Main implementation | Evaluation |
| --- | --- | --- | --- |
| `data_cleaning` | Parse industrial source layers and preserve traceability | `backend/cleaning/` | files, records, parse errors, empty text, duplicate IDs, timestamp issues, provenance |
| `text_preprocessing` | Convert evidence into a unified textual representation | `backend/preprocessing/`, `backend/datasets/indego_adapter.py` | evidence support, field coverage, omissions, unsupported additions, downstream KG delta |
| `kg_extraction` | Extract entities and relations, build the graph, validate, and score | `backend/extraction/`, `backend/pipeline/`, `backend/graph/`, `backend/evaluation/` | Node/Edge P/R/F1, relation-type recall, hallucination, schema conformance, stability, runtime |

Independent module versions are stored in `config/module_versions.json`.

## 4. One-Time Baseline Experiment

### 4.1 Dataset

The end-to-end experiment uses the combined reviewed Industrial Gold:

- 32 scenes;
- 30 distinct video IDs;
- 674 Gold entities;
- 828 Gold relations;
- only source-supported facts;
- development, quick-regression, and holdout partitions grouped by `video_id`.

Grouping by video prevents scenes from one recording from leaking into both
development and holdout. This follows the same non-overlapping group principle
provided by scikit-learn's `GroupKFold`.

### 4.2 Input Conditions

| Condition | Interpretation |
| --- | --- |
| `raw + one_shot` | Raw text and a single full-graph generation call |
| `raw + layered` | Raw text with entity-first, relation-second extraction |
| `auto_unified + layered` | Automatically produced evidence-grounded unified text |
| `reviewed_unified + layered` | Human-reviewed Gold-derived upper-bound representation |

`reviewed_unified` is an upper bound. It must never be reported as the output
quality of the automatic preprocessing module.

### 4.3 Run Count and Failure Policy

```text
32 scenes x 4 conditions x 2 repetitions = 256 real LLM calls
```

Every call is checkpointed. The runner has a process-level wall-clock timeout.
After a timeout it terminates the isolated client process, stops the remaining
server-side Ollama generation, records the error, and continues. Failed calls
are never deleted or selectively rerun.

### 4.4 Metrics

Primary metrics:

- Edge Precision, Recall, and F1;
- Node Precision, Recall, and F1;
- recall and F1 by relation type.

Reliability and engineering metrics:

- relation hallucination rate;
- ontology/schema conformance;
- error, timeout, and JSON parse-failure rates;
- repeated-run Node and Edge Jaccard agreement;
- runtime.

Raw LLM edges and final post-processed edges are scored separately. A fallback
edge is a product reliability mechanism, not evidence that the LLM extracted
the relation by itself.

### 4.5 Immutable Artifacts

The completed baseline is stored under:

```text
experiments/baselines/industrial-v1-llama31-8b-20260809/
```

It contains the manifest, cleaning and preprocessing metrics, per-run and
aggregate KG metrics, relation-type results, stability, hallucination reports,
compressed raw runs, SHA256 checksums, and a `COMPLETED` marker. Comparison code
must not modify this directory.

## 5. Future Change Protocol

Every optimization receives a Change ID, for example `EXP-20260815-001`, and a
manifest containing:

- changed module, files, classes, and functions;
- Git diff and code commit;
- reason and falsifiable hypothesis;
- method and literature or code source;
- expected metric effect;
- actual baseline and candidate values;
- absolute and relative deltas;
- all errors and timeouts;
- final decision: `improved`, `neutral`, `regressed`, or `incomplete`.

Run only the changed implementation. Load the frozen matching V1 rows and use
the exact same scene, condition, and repetition keys for paired comparison. If
the model digest, Gold, or metric definition changes materially, create a new
baseline version instead of overwriting V1.

## 6. Recommended Improvements

### P0. Repair Source and Segment Alignment Before Changing the Model

**Current problem.** None of the 32 automatic IndEgo source texts exactly
matches the reviewed Gold scene excerpt. Of 302 original automatic grounded
entries, only 134 are supported within the selected excerpt and 168 are outside
it. The aligned automatic representation consequently contains only 96 action
entries and 12 tool/object entries, compared with 369 actions and 305
tool/object entries in the reviewed upper bound.

**Solution.** Add an explicit source-alignment layer keyed by `video_id`, source
layer, segment ID, and timestamp overlap. Resolve the Gold excerpt to its full
IndEgo transcript/action/keystep window before filtering evidence. Use exact
normalized spans first, timestamp overlap second, and auditable fuzzy matching
only as a reported fallback. Never copy reviewed Gold facts into automatic
input.

**Experiment.** Compare current `auto_unified_v1` with `aligned_auto_unified_v2`
on the fixed quick set. Report source-match coverage, retained evidence entries,
unsupported additions, Node/Edge F1, and runtime. Keep the raw baseline rows
frozen.

**Proposed Change ID.** `EXP-SOURCE-ALIGNMENT-001`, module
`text_preprocessing`.

### P0. Native JSON-Schema-Constrained Ollama Output

**Current problem.** Some model calls fail because the generated JSON is
truncated or malformed. Prompt-only JSON instructions do not enforce syntax.

**Solution.** Add an optional Ollama-native client path that sends the Pydantic
`model_json_schema()` through `format` or OpenAI-compatible `response_format`.
Keep temperature at zero and validate the returned JSON with the same Pydantic
model. Ollama documents JSON Schema structured outputs specifically for
reliable structured extraction.

**Experiment.** Compare the current client with native constrained output on
the fixed quick set. Primary endpoints are JSON parse-failure rate and schema
conformance; Node and Edge F1 are guardrails because syntactic validity alone
does not guarantee semantic correctness.

**Proposed Change ID.** `EXP-STRUCTURED-OUTPUT-001`, module `kg_extraction`.

### P0. Relation Candidates Plus a Dedicated Scorer

**Current problem.** End-to-end generation must discover entities, enumerate
entity pairs, assign relation labels, and serialize a graph at once. Edge recall
remains the main bottleneck.

**Solution.** Preserve the existing `RelationCandidateGenerator ->
RelationCandidateScorer -> RelationValidationReport` interface. Generate only
schema-valid candidates from known entities and evidence windows. Compare:

1. deterministic rules;
2. LLM binary support judgment;
3. GLiREL zero-shot relation scoring;
4. a hybrid union followed by evidence and schema validation.

The LLM judge must only classify a supplied candidate and quote evidence; it
must not invent edges. GLiREL is designed to classify relation labels for known
entity pairs in a single forward pass and is therefore a good targeted
comparison for a local Mac workflow.

**Experiment.** Use manually reviewed relation Gold. Report candidate coverage,
accepted precision/recall/F1, per-relation recall, runtime, and unsupported
edges. Tune thresholds only on development videos and evaluate once on holdout.

**Proposed Change ID.** `EXP-RELATION-SCORER-001`, module `kg_extraction`.

### P1. Evidence-Preserving Atomic Proposition Preprocessing

**Current problem.** Dense instructional passages can contain multiple actions,
objects, conditions, and temporal statements. Small extractors may miss
relations when too much information is packed into one chunk.

**Solution.** Add an optional atomization representation that separates safe
conjunctive facts into minimal propositions while retaining source spans and
the original text. Do not split disjunctions, implications, or uncertain
statements into stronger claims. Reject any proposition without exact or
approved normalized evidence.

Recent work reports that atomic propositions can improve relation recall for
weaker extractors, but may reduce entity recall; combining raw text and
propositions can recover complementary facts. Therefore this method should be
an ablation, not an immediate replacement for unified text.

**Experiment.** Compare `raw`, `auto_unified`, `atomic`, and
`raw + atomic` on the same quick scenes. Measure evidence support, omission,
unsupported additions, Node recall, Edge recall, and runtime.

**Proposed Change ID.** `EXP-ATOMIC-PREPROCESS-001`, module
`text_preprocessing`.

### P1. Development-Only Similar-Example Retrieval

**Current problem.** A fixed generic prompt gives the 8B model little guidance
for rare industrial relation labels.

**Solution.** Retrieve a small number of semantically similar, source-supported
Gold examples from development videos only. Include compact schema-valid
entity/relation demonstrations. Never retrieve from the current scene's video
or the holdout partition.

The maintenance-text study by van Cauter and Yakovets combines ontology-guided
triplet extraction with in-context learning and reports competitive results
with 20 semantically similar examples using a much larger Llama-3-70B model.
That result motivates the method but does not guarantee improvement for the
local 8B model, so `k = 0, 2, 5, 10` must be ablated for context cost and F1.

**Proposed Change ID.** `EXP-SIMILAR-FEW-SHOT-001`, module `kg_extraction`.

### P1. Strengthen the Relation Gold and Holdout

**Current problem.** Thirty-two scenes are useful for development and
regression but too small for a strong industrial-generalization claim. Rare
relations have little statistical support.

**Solution.** Manually review additional industrial scenes, prioritizing
`USES_TOOL`, `ACTS_ON`, `BEFORE`, `PART_OF`, `WARNING_FOR`, parameters, and
quality checks. Every edge must include an evidence span or a timestamp-backed
annotation reference. Preserve video-level grouping and freeze a holdout that
is not used for prompts, thresholds, or error-driven rule creation.

**Experiment.** Publish support counts and per-type confidence intervals or
bootstrap intervals in addition to micro F1. EASG may remain an auxiliary
public benchmark, but it does not replace industrial-domain Gold.

### P2. Calibrated Confidence and Abstention

**Current problem.** A binary accepted/rejected edge can hide uncertainty and
encourage unsupported low-confidence edges.

**Solution.** Calibrate relation thresholds on the development set and permit
abstention. Store the scorer, threshold, quoted evidence, and relation origin
for every candidate. Evaluate precision-recall curves and choose operating
points before holdout evaluation.

### P2. Direct Video-Derived Input as a Later Extension

Once the text-only system is stable, add an upstream adapter for processed
video outputs such as timestamped narrations, action labels, object labels, and
segment metadata. Do not download or decode large raw videos for the current
MVP. EASG demonstrates that manually labeled action scene graphs can provide
temporally grounded action-object information, but the final industrial
evaluation still requires domain-relevant source text and Gold.

## 7. Recommended Execution Order

1. Preserve and publish the completed V1 baseline and annotated tag.
2. Repair source/segment alignment and evaluate automatic evidence retention.
3. Implement native Ollama JSON Schema output and evaluate parse reliability.
4. Manually verify relation Gold and run deterministic/LLM/GLiREL/hybrid
   relation ablations.
5. Test atomic proposition preprocessing with strict evidence validation.
6. Add development-only similar-example retrieval.
7. Expand the industrial Gold and perform a frozen video-grouped holdout run.

## 8. Sources

- Mihindukulasooriya et al. (2023), [Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text](https://arxiv.org/abs/2308.02357). It defines ontology conformance, fact extraction, and hallucination-oriented evaluation.
- van Cauter and Yakovets (2024), [Ontology-guided Knowledge Graph Construction from Maintenance Short Texts](https://aclanthology.org/2024.kallm-1.8/). It motivates ontology-guided extraction with semantically similar in-context examples in an industrial maintenance domain.
- Boylan et al. (2025), [GLiREL: Generalist Model for Zero-Shot Relation Extraction](https://aclanthology.org/2025.naacl-long.418/), with [official code](https://github.com/jackboyla/GLiREL). It motivates a dedicated scorer over known entity pairs and relation labels.
- Ollama, [Structured Outputs documentation](https://docs.ollama.com/capabilities/structured-outputs). It documents JSON and JSON-Schema-constrained local generation with Pydantic schemas.
- Pommeret et al. (2026), [LLM-based Atomic Propositions Help Weak Extractors](https://arxiv.org/abs/2604.02866). It motivates evidence-preserving proposition decomposition as an ablation for weaker triplet extractors.
- scikit-learn, [GroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html). It defines non-overlapping group-based splits used here for `video_id` separation.
- Rodin et al. (CVPR 2024), [Action Scene Graphs for Long-Form Understanding of Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Rodin_Action_Scene_Graphs_for_Long-Form_Understanding_of_Egocentric_Videos_CVPR_2024_paper.pdf). It supports the use of temporally grounded action-scene graph annotations as an auxiliary benchmark.
