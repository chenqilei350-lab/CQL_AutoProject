# AGENTS.md

## 项目概览

本项目是 `AUT KG Extraction Pipeline`，用于从工业制造相关文本中抽取结构化知识，并进一步服务于知识图谱构建。

项目定位：

- 研究背景：TU Berlin，Institut fuer Industrielle Automatisierungstechnik。
- 核心任务：使用本地 LLM 将文本抽取为 Pydantic schema 定义的结构化实体和关系。
- 当前重点：schema-driven extraction、process knowledge schema、egocentric video scene schema、property graph MVP、抽取结果评估。
- 当前主要领域：`process_knowledge`，面向工艺、步骤、工具、材料、人员、质量检查等制造过程知识。

注意：`README.md` 中提到的 `training/`、`pipeline/`、`organizational_knowledge/` 等目录更像是规划中的项目结构，当前仓库内尚未实现这些目录。当前已有一个轻量 `backend/graph/` MVP，但还没有接入 LadybugDB 或 Neo4j。

## 项目完成路线

详细 workflow 见：

```text
docs/workflow.md
```

当前路线：

- MVP 输入采用 egocentric expert video 的文字化 scene descriptions 或 transcript-like text。
- 先完成 `scene text -> Pydantic extraction contract -> property graph -> query/evaluation`。
- 暂不直接处理原始视频帧；后续可参考 OpenPVSG、PVSG、SAMJAM 等视频 scene graph 工作接入视觉模型。
- Graph MVP 先使用内存 property graph 验证 schema、去重、provenance 和查询，再决定是否落到 LadybugDB 或 Neo4j。

参考资料：

- Neo4j LLM Graph Builder：https://github.com/neo4j-labs/llm-graph-builder
- Neo4j Developer Guide: LLM Knowledge Graph Builder：https://neo4j.com/developer/genai-ecosystem/llm-graph-builder/
- Docling Graph：https://github.com/docling-project/docling-graph
- OntoCast：https://github.com/growgraph/ontocast
- LlamaIndex PropertyGraphIndex：https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_basic/
- CocoIndex Knowledge Graph for Docs：https://cocoindex.io/examples/knowledge-graph-for-docs
- OpenPVSG：https://github.com/LilyDaytoy/OpenPVSG
- Panoptic Video Scene Graph Generation：https://arxiv.org/abs/2311.17058
- SAMJAM：https://arxiv.org/abs/2504.07867

## 技术栈与依赖

项目使用 Python 工程结构，并通过 `uv` 管理依赖。

主要技术：

- Python `>=3.12`
- `uv`
- Pydantic v2
- Instructor
- OpenAI Python client
- Ollama，本地 OpenAI-compatible API 地址为 `http://localhost:11434/v1`
- 默认本地模型：`llama3.1:8b`
- pytest
- ruff
- Jupyter / ipykernel

依赖定义在 `pyproject.toml`，锁文件为 `uv.lock`。

## 常用命令

首次安装：

```bash
uv sync
uv pip install -e .
```

准备本地 LLM：

```bash
ollama pull llama3.1:8b
ollama serve
```

运行评估 smoke tests：

```bash
python -m pytest tests/test_eval.py -v
```

README 推荐的快速体验入口是：

```text
notebooks/01_extraction_basics.ipynb
```

## 当前代码结构

当前仓库的核心代码集中在 `backend/`。

### LLM 客户端

`backend/llm/client.py`

- 定义 Ollama 的 OpenAI-compatible base URL。
- 使用 OpenAI client 连接本地 Ollama。
- 使用 Instructor 包装 OpenAI client，实现 Pydantic 结构化输出。
- 暴露 `extract_structured(...)`，输入文本、Pydantic response model、模型名和 prompt，返回结构化对象。

### 抽取接口

`backend/extraction/extractor.py`

- 定义 `Extractor` 类，是当前最主要的抽取入口。
- 支持抽取单个 Pydantic 对象：`extract(...)`。
- 支持抽取对象列表：`extract_list(...)`。
- 默认语言为 German。
- 默认温度为 `0.0`，用于更稳定的抽取。
- 默认模型来自 `backend.llm.client.DEFAULT_MODEL`，当前为 `llama3.1:8b`。

典型用法：

```python
from backend.extraction.extractor import Extractor
from backend.schemas.process_knowledge.entities import Tool

extractor = Extractor(model="llama3.1:8b")
tools = extractor.extract_list(
    "Wir verwenden das Fronius TPS 400i.",
    Tool,
)
```

### 基础 schema

`backend/schemas/base.py`

- `KGEntity`：所有知识图谱实体的基类，包含 `name`、`description`、`source_text`。
- `KGRelation`：所有知识图谱关系的基类，包含 `source_text`。
- `ExtractionResult`：用于承载一次文本 chunk 的实体、关系、来源 chunk、抽取模型和时间戳。

### Process Knowledge schema

`backend/schemas/process_knowledge/entities.py`

定义工艺知识实体，并按抽取复杂度组织：

- Level 1：扁平实体，例如 `Tool`、`Material`、`PPE`。
- Level 2：带枚举或约束字段的实体，例如 `Worker`、`Step`、`Procedure`。
- Level 3：带更多结构化属性的实体，例如 `ProcessParameter`、`QualityRequirement`。

`backend/schemas/process_knowledge/relations.py`

定义工艺知识关系，并按抽取复杂度组织：

- Level 2：二元关系，例如 `StepOrder`、`ProcedureHasStep`。
- Level 3：带自身属性的关系，例如 `ToolRequirement`、`MaterialRequirement`、`PPERequirement`。
- Level 4：多元关系，例如 `StepExecution`、`QualityCheck`、`QualificationRecord`。
- Level 5：深层嵌套结构，例如 `StepFeedback`、`ProcedureExecution`。

`backend/schemas/process_knowledge/examples.py`

- 存放 gold-standard examples。
- 可用于 few-shot prompt、测试 fixture、训练数据种子和人工对照。
- 示例文本主要为德语制造场景，例如 MIG welding procedure、焊接参数、质量检查等。

### Egocentric Video schema

`backend/schemas/egocentric_video.py`

- 定义 egocentric video scene description 的 MVP 抽取合同。
- 核心节点包括 `Scene`、`Action`、`SceneObject`，并复用 `Tool`、`Worker`、`Procedure`、`ProcessParameter`。
- 核心关系包括 `UsesTool`、`ActsOnObject`、`ActionOrder`、`ActionCauses`、`ActionPartOfProcedure`、`ActionObservedInScene`。
- `Provenance` 用于记录 `video_id`、`segment_id`、frame/timestamp、`source_text` 和 `confidence`。

`backend/schemas/egocentric_examples.py`

- 存放 egocentric video procedural knowledge 的 gold-standard examples。
- 当前包含焊接准备和质量检查两个 scene-description examples。

## Graph MVP

`backend/graph/property_graph.py`

- 提供轻量内存 property graph。
- 将 `EgocentricVideoExtraction` 转换为 `GraphNode` 和 `GraphEdge`。
- 支持 normalized name 与 token overlap 的基础节点合并。
- 保留 action/object 的 provenance 和 relation properties。
- 提供基础查询：
  - `tools_for_action(...)`
  - `actions_using_tool(...)`
  - `ordered_actions()`
  - `effects_caused_by(...)`

该层是 LadybugDB/Neo4j 之前的 MVP 验证层；不要把它视为最终数据库实现。

## Evaluation 模块

当前评估代码位于 `backend/evaluation/`。

`backend/evaluation/matching.py`

- 定义匹配策略 `MatchStrategy`。
- 支持：
  - `EXACT`
  - `NORMALIZED`
  - `TOKEN_OVERLAP`
  - `EMBEDDING`
- `compare_values(...)` 可以处理 `None`、布尔值、数值和字符串。

`backend/evaluation/metrics.py`

- 定义 `PRF1`。
- 提供 `compute_prf1(...)`、`aggregate_prf1(...)`、`format_prf1(...)` 和 `format_prf1_table(...)`。
- 指标包括 precision、recall、F1、TP、FP、FN、total expected、total extracted。

`backend/evaluation/evaluators.py`

- `evaluate_entity_detection(...)`：实体级检测评估。
- `evaluate_attributes(...)`：属性级评估。
- `evaluate_relation_detection(...)`：关系级评估。
- `evaluate_extraction_level(...)`：按复杂度 level 组合实体、属性和关系评估。

`backend/evaluation/experiment.py`

- `ExperimentConfig`：记录一次实验的配置，例如模型、prompt 策略、schema level、领域、温度等。
- `ExperimentResult`：组合实验配置和评估指标。

## 测试

当前测试文件：

```text
tests/test_eval.py
tests/test_egocentric_workflow.py
```

测试覆盖：

- 匹配策略。
- P/R/F1 指标计算。
- 实验配置模型。
- 实体检测评估。
- 属性评估。
- 关系检测评估。
- 单个 extraction level 的组合评估。
- Egocentric video schema validation。
- Pydantic extraction result 到 property graph 的转换。
- 节点合并/去重。
- provenance 和 relation properties 保留。
- 基础 graph queries。

这些测试是 evaluation pipeline 的 smoke tests，不需要真实调用 Ollama 或 LLM。

推荐在修改 evaluation、schema 或抽取相关逻辑后运行：

```bash
python -m pytest tests/test_eval.py -v
python -m pytest tests/test_egocentric_workflow.py -v
```

## 当前入口与 Notebook

`main.py`

- 当前只是占位入口。
- 运行时只会输出 `Hello!`。
- 不应把它视为完整 pipeline 入口。

`notebooks/01_extraction_basics.ipynb`

- README 中推荐的快速体验入口。
- 用于演示基础抽取流程。

## 开发注意事项

- 修改 schema 时，需要同步检查 examples、evaluation 和 tests 是否仍然匹配。
- LLM 抽取依赖本地 Ollama 服务；如果 Ollama 没有运行，抽取相关代码会失败。
- Evaluation smoke tests 不依赖 Ollama，适合作为快速回归检查。
- 当前 schema 使用 Pydantic v2 写法，后续应保持一致。
- 不要将 README 中尚未存在的目录或模块当作已经实现的代码使用。
- 如果新增领域 schema，建议沿用 `backend/schemas/process_knowledge/` 的组织方式：entities、relations、examples 分离。
- 如果新增评估逻辑，优先复用 `matching.py` 和 `metrics.py` 中已有的匹配与指标函数。
- 如果新增 egocentric video 抽取能力，优先维护 `backend/schemas/egocentric_video.py`、`backend/schemas/egocentric_examples.py` 和 `backend/graph/property_graph.py` 的一致性。

## 给后续 Agent 的约定

- 优先使用 `uv` 管理依赖、同步环境和运行测试。
- 搜索代码优先用 `rg`。
- 对抽取逻辑、schema 或 evaluation 的改动，至少运行：

```bash
python -m pytest tests/test_eval.py -v
python -m pytest tests/test_egocentric_workflow.py -v
```

- 不要假设未来规划目录已经存在。
- 不要在没有需求的情况下引入新的框架或大型抽象。
- 对 LLM 相关行为做改动时，要区分：
  - 本地结构化抽取接口是否正确。
  - Ollama 服务是否可用。
  - 模型输出质量是否符合 schema 和评估标准。
