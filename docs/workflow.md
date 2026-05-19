# 项目完成路线 Workflow

## 目标

本项目的目标是设计并评估一个 Knowledge Graph (KG) 架构和抽取管线，用于从 egocentric expert videos 的场景描述中捕获程序性知识，并转化为结构化、可查询的图数据。

当前 MVP 不直接处理原始视频帧，而是以视频片段的文字化 scene descriptions 或 transcript-like text 作为输入。这样可以先验证 schema、LLM 抽取、图构建和评估方法，再逐步接入视觉模型或视频 scene graph generation 方法。

## Reference Architecture

```text
Egocentric video
      |
      v
Scene descriptions / transcripts
      |
      v
LLM extraction with Ollama + Instructor
      |
      v
Pydantic extraction contracts
      |
      v
Typed procedural entities and relations
      |
      v
Property graph construction
      |
      v
Deduplication, provenance, query, evaluation
```

## 设计需求

| 模块 | MVP 设计 | 后续扩展 |
| --- | --- | --- |
| 输入 | 文字化 scene descriptions | 原始视频、帧、ASR transcript、VLM caption |
| Schema | Pydantic v2 typed contracts | PKO 对齐、ontology-driven schema selection |
| 抽取 | Ollama + Instructor structured output | few-shot prompting、multi-pass extraction、fine-tuned model |
| 图构建 | 内存 property graph | LadybugDB、Neo4j、Cypher export |
| 去重 | normalized name、token overlap | embeddings、alias table、human correction |
| provenance | source text、video id、segment/frame id、confidence | frame masks、bounding boxes、temporal tracking |
| 评估 | entity / attribute / relation PRF1 | graph-level query accuracy、temporal consistency |

## Workflow

1. **Literature & Reference Architecture**
   - 阅读 KG/ontology 基础、PKO、LLM+KG、Neo4j/LadybugDB/Cypher 资料。
   - 对比类似开源项目，提炼本项目架构。
   - 输出 reference architecture 和设计需求表。

2. **Schema Definition**
   - 基于现有 `process_knowledge` schema 扩展 egocentric video 场景知识。
   - 覆盖 actions、objects、tools、actors、scenes、procedure、sequence、causality、provenance。
   - 保持 Pydantic v2 schema 作为抽取合同。

3. **Extraction Pipeline**
   - 输入采用 scene descriptions 或 transcript-like text。
   - 使用当前 `Extractor` + Ollama/Instructor 做 schema-driven extraction。
   - 输出统一为 typed extraction result，并保留 provenance。

4. **Graph Construction**
   - 把 Pydantic 对象转换为 property graph nodes/edges。
   - MVP 使用内存图，避免过早绑定 Neo4j 或 LadybugDB。
   - 实现基础节点合并和查询能力。

5. **Evaluation**
   - 复用现有 `backend/evaluation/` 的 PRF1 思路。
   - 覆盖 entity detection、attribute extraction、relation detection 和 graph-level query checks。
   - 使用 gold examples 构造小型 benchmark。

6. **Prototype & Report**
   - 用 notebook 或 CLI 展示端到端流程。
   - Demo 输出包括结构化 JSON、graph nodes/edges、查询结果和评估表。
   - 项目报告建议结构：problem、related work、schema design、pipeline architecture、implementation、evaluation、limitations、future work。

## 类似开源资料与参考

- [Neo4j LLM Graph Builder](https://github.com/neo4j-labs/llm-graph-builder)：从非结构化文档、网页和 YouTube transcript 抽取 nodes / relationships，写入 Neo4j，并支持 schema 配置、GraphRAG 和可视化。
- [Neo4j Developer Guide: LLM Knowledge Graph Builder](https://neo4j.com/developer/genai-ecosystem/llm-graph-builder/)：展示 document/chunk nodes、embedding、entity graph、schema-guided extraction 和 cleanup operations 的完整工程流程。
- [Docling Graph](https://github.com/docling-project/docling-graph)：把文档转换成 validated Pydantic objects，再构建 directed knowledge graph；支持 local runtime/Ollama、Pydantic templates 和 Cypher export。
- [OntoCast](https://github.com/growgraph/ontocast)：ontology-guided semantic triple extraction、entity disambiguation、RDF/Turtle 输出和 Neo4j/Fuseki 集成。
- [LlamaIndex PropertyGraphIndex](https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_basic/)：从文本中抽取 triples/paths 并构建 property graph index。
- [CocoIndex Knowledge Graph for Docs](https://cocoindex.io/examples/knowledge-graph-for-docs)：持续更新的 LLM extraction + Neo4j graph pipeline。
- [OpenPVSG](https://github.com/LilyDaytoy/OpenPVSG) / [Panoptic Video Scene Graph Generation](https://arxiv.org/abs/2311.17058)：视频 scene graph generation，包含 egocentric video 数据与 temporal scene graph 任务定义。
- [SAMJAM: Zero-Shot Video Scene Graph Generation for Egocentric Kitchen Videos](https://arxiv.org/abs/2504.07867)：egocentric video scene graph 的 zero-shot pipeline，用 VLM + temporal tracking 处理动态场景。

## Milestones

- **M1: Documentation & design**：维护 `docs/workflow.md` 和 `AGENTS.md`，明确 MVP 输入是 scene descriptions。
- **M2: Schema MVP**：扩展 Pydantic schema，覆盖 actions、objects、tools、sequence、causality、provenance。
- **M3: Extraction MVP**：用当前 `Extractor` 抽取 entities 和 relations，输出结构化 JSON。
- **M4: Graph MVP**：实现 Pydantic -> graph nodes/edges 转换、基础去重和查询。
- **M5: Evaluation & report**：扩展测试与 evaluation examples，输出 P/R/F1、错误分析和报告材料。

