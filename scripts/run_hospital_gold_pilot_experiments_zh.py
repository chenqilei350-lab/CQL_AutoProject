#!/usr/bin/env python3
"""
医院黄金案例 KG 抽取实验复现脚本（legacy/reference 中文注释版）。

本脚本现在保留为 hospital reference/stress-test track，用于快速比较
schema constraint、chunking、canonical ID 和 relation direction 等算法变量。
它不是 AUT/IndEgo 第一人称工业场景的默认主实验入口。

用途：
    读取 gold_case_admission_29366372 中的 source CSV 和 gold graph，
    调用本地 Ollama 模型，复现 P1-P6 基础实验以及 P1/P3/P4 stress 实验。

运行前提：
    1. 已安装并启动 Ollama。
    2. 已拉取至少一个模型，例如：
           ollama pull llama3.1:8b
    3. 当前仓库中存在 gold_case_admission_29366372 数据目录。

最常用命令：
    python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite stress

完整实验：
    python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from hospital_kg_postprocessing import (
    HospitalPostprocessContext,
    postprocess_hospital_kg,
    load_hospital_postprocess_context,
)


# -----------------------------
# 基础数据结构
# -----------------------------


@dataclass
class Metric:
    """保存 Precision / Recall / F1 及 TP/FP/FN 计数。"""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    predicted: int
    gold: int


@dataclass
class GraphPrediction:
    """模型输出解析后的节点集合与边集合。"""

    nodes: set[tuple[str, str]]
    edges: set[tuple[str, str, str]]
    raw_json: dict[str, Any]


@dataclass
class HospitalGoldData:
    """集中保存 gold case 的 source rows，避免各实验重复读文件。"""

    patient: dict[str, str]
    admission: dict[str, str]
    diagnoses: list[dict[str, str]]
    procedures: list[dict[str, str]]
    medications: list[dict[str, str]]
    lab_events: list[dict[str, str]]
    lab_items: list[dict[str, str]]


POSTPROCESS_CONTEXT: HospitalPostprocessContext | None = None


# -----------------------------
# 通用工具函数
# -----------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件为字典列表。"""

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_gold_data(gold_dir: Path) -> HospitalGoldData:
    """从 gold case 目录读取所有 source CSV。"""

    return HospitalGoldData(
        patient=read_csv_rows(gold_dir / "source_patient.csv")[0],
        admission=read_csv_rows(gold_dir / "source_admission.csv")[0],
        diagnoses=read_csv_rows(gold_dir / "source_diagnoses.csv"),
        procedures=read_csv_rows(gold_dir / "source_procedures.csv"),
        medications=read_csv_rows(gold_dir / "source_medications.csv"),
        lab_events=read_csv_rows(gold_dir / "source_lab_events.csv"),
        lab_items=read_csv_rows(gold_dir / "source_lab_items.csv"),
    )


def compute_prf(predicted: set[Any], gold: set[Any]) -> Metric:
    """计算 Precision（正确性）、Recall（完整性）和 F1。"""

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metric(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp,
        fp=fp,
        fn=fn,
        predicted=len(predicted),
        gold=len(gold),
    )


def jaccard(left: set[Any], right: set[Any]) -> float:
    """计算两个集合的 Jaccard 重合度，用于稳定性评价。"""

    union = left | right
    return round(len(left & right) / len(union), 4) if union else 1.0


def mean(values: list[float]) -> float:
    """计算均值；空列表时返回 0。"""

    return round(sum(values) / len(values), 4) if values else 0.0


def population_std(values: list[float]) -> float:
    """计算总体标准差，用于 repeated-run metric variance。"""

    if not values:
        return 0.0
    m = sum(values) / len(values)
    return round(math.sqrt(sum((value - m) ** 2 for value in values) / len(values)), 4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """把实验结果保存为 CSV。"""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_json_loads(text: str) -> dict[str, Any]:
    """尽量把模型输出解析成 JSON；解析失败时抛出更清楚的错误。"""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"模型没有返回合法 JSON: {text[:300]!r}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"模型 JSON 顶层不是 object: {type(parsed).__name__}")
    return parsed


def call_ollama(model: str, prompt: str, timeout: int, num_ctx: int = 8192) -> tuple[dict[str, Any], float]:
    """调用本地 Ollama chat API，并要求模型返回 JSON。"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": 4096,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama 调用失败: {error}") from error
    elapsed = time.monotonic() - started
    content = response_data.get("message", {}).get("content", "")
    return safe_json_loads(content), elapsed


def parse_prediction(obj: dict[str, Any]) -> GraphPrediction:
    """把模型 JSON 输出转成可比较的节点集合和边集合。"""

    nodes: set[tuple[str, str]] = set()
    for node in obj.get("nodes", []):
        if isinstance(node, dict) and node.get("label") and node.get("node_id"):
            nodes.add((str(node["label"]), str(node["node_id"])))

    edges: set[tuple[str, str, str]] = set()
    for edge in obj.get("edges", []):
        if (
            isinstance(edge, dict)
            and edge.get("source_id")
            and edge.get("relation")
            and edge.get("target_id")
        ):
            edges.add((str(edge["source_id"]), str(edge["relation"]), str(edge["target_id"])))

    return GraphPrediction(nodes=nodes, edges=edges, raw_json=obj)


def graph_facts(prediction: GraphPrediction) -> set[str]:
    """把节点和边合并成图事实集合，用于 graph overlap。"""

    return {f"N::{label}|{node_id}" for label, node_id in prediction.nodes} | {
        f"E::{source}|{relation}|{target}" for source, relation, target in prediction.edges
    }


def add_postprocess_metrics(
    row: dict[str, Any],
    prediction: GraphPrediction | None,
    gold_nodes: set[tuple[str, str]],
    gold_edges: set[tuple[str, str, str]],
) -> GraphPrediction | None:
    """Append raw/normalized metric fields while keeping node_f1/edge_f1 as raw."""

    row["raw_node_f1"] = row.get("node_f1", 0.0)
    row["raw_edge_f1"] = row.get("edge_f1", 0.0)
    row["raw_node_precision"] = row.get("node_precision", 0.0)
    row["raw_node_recall"] = row.get("node_recall", 0.0)
    row["raw_edge_precision"] = row.get("edge_precision", 0.0)
    row["raw_edge_recall"] = row.get("edge_recall", 0.0)

    if prediction is None or POSTPROCESS_CONTEXT is None:
        row["normalized_node_precision"] = row.get("node_precision", 0.0)
        row["normalized_node_recall"] = row.get("node_recall", 0.0)
        row["normalized_node_f1"] = row.get("node_f1", 0.0)
        row["normalized_edge_precision"] = row.get("edge_precision", 0.0)
        row["normalized_edge_recall"] = row.get("edge_recall", 0.0)
        row["normalized_edge_f1"] = row.get("edge_f1", 0.0)
        row["hallucinated_nodes"] = 0
        row["hallucinated_edges"] = 0
        row["invalid_id_count"] = 0
        row["invalid_relation_count"] = 0
        row["wrong_direction_count"] = 0
        row["merged_duplicate_edges"] = 0
        row["auto_corrected_edges"] = 0
        row["postprocess_issues"] = ""
        return prediction

    processed = postprocess_hospital_kg(
        nodes=prediction.nodes,
        edges=prediction.edges,
        context=POSTPROCESS_CONTEXT,
    )
    normalized_prediction = GraphPrediction(
        nodes=processed.nodes,
        edges=processed.edges,
        raw_json=prediction.raw_json,
    )
    node_metric = compute_prf(processed.nodes, gold_nodes)
    edge_metric = compute_prf(processed.edges, gold_edges)
    row["normalized_node_precision"] = node_metric.precision
    row["normalized_node_recall"] = node_metric.recall
    row["normalized_node_f1"] = node_metric.f1
    row["normalized_edge_precision"] = edge_metric.precision
    row["normalized_edge_recall"] = edge_metric.recall
    row["normalized_edge_f1"] = edge_metric.f1
    row["hallucinated_nodes"] = processed.stats.hallucinated_nodes
    row["hallucinated_edges"] = processed.stats.hallucinated_edges
    row["invalid_id_count"] = processed.stats.invalid_id_count
    row["invalid_relation_count"] = processed.stats.invalid_relation_count
    row["wrong_direction_count"] = processed.stats.wrong_direction_count
    row["merged_duplicate_edges"] = processed.stats.merged_duplicate_edges
    row["auto_corrected_edges"] = processed.stats.auto_corrected_edges
    row["postprocess_issues"] = "; ".join(processed.stats.issues)
    return normalized_prediction


# -----------------------------
# Gold graph 构造
# -----------------------------


def mini_gold(data: HospitalGoldData) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]], str]:
    """构造 P1/P5/P6 使用的 7 节点、6 边小型 gold graph。"""

    patient = data.patient
    admission = data.admission
    diagnosis = data.diagnoses[0]
    procedure = data.procedures[0]
    medication = data.medications[0]
    lab_event = data.lab_events[0]
    lab_item = next(item for item in data.lab_items if item["lab_item_id"] == lab_event["lab_item_id"])

    nodes = {
        ("Patient", patient["patient_id"]),
        ("Admission", admission["admission_id"]),
        ("Diagnosis", diagnosis["diagnosis_id"]),
        ("Procedure", procedure["procedure_id"]),
        ("Medication", medication["medication_id"]),
        ("LabEvent", lab_event["lab_event_id"]),
        ("LabItem", lab_item["lab_item_id"]),
    }
    edges = {
        (patient["patient_id"], "has_admission", admission["admission_id"]),
        (admission["admission_id"], "has_diagnosis", diagnosis["diagnosis_id"]),
        (admission["admission_id"], "has_procedure", procedure["procedure_id"]),
        (admission["admission_id"], "prescribed_medication", medication["medication_id"]),
        (admission["admission_id"], "has_lab_event", lab_event["lab_event_id"]),
        (lab_event["lab_event_id"], "is_test_of", lab_item["lab_item_id"]),
    }
    rows = [
        ("Patient", patient),
        ("Admission", admission),
        ("Diagnosis", diagnosis),
        ("Procedure", procedure),
        ("Medication", medication),
        ("LabEvent", lab_event),
        ("LabItem", lab_item),
    ]
    input_text = "\n".join(
        f"{label}: " + ", ".join(f"{key}={value}" for key, value in row.items() if value)
        for label, row in rows
    )
    return nodes, edges, input_text


def lab_event_gold(
    data: HospitalGoldData,
    include_items: bool = False,
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    """构造 LabEvent 层 gold graph；可选加入 LabItem 节点和 is_test_of 边。"""

    nodes = {("LabEvent", row["lab_event_id"]) for row in data.lab_events}
    edges = {(row["admission_id"], "has_lab_event", row["lab_event_id"]) for row in data.lab_events}
    if include_items:
        used_item_ids = {row["lab_item_id"] for row in data.lab_events}
        nodes |= {("LabItem", row["lab_item_id"]) for row in data.lab_items if row["lab_item_id"] in used_item_ids}
        edges |= {(row["lab_event_id"], "is_test_of", row["lab_item_id"]) for row in data.lab_events}
    return nodes, edges


def mixed_gold(
    data: HospitalGoldData,
    diagnosis_count: int = 3,
    procedure_count: int = 2,
    medication_count: int = 3,
    lab_count: int = 3,
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    """构造 P4 stress 使用的多类型混合 gold graph。"""

    patient = data.patient
    admission = data.admission
    diagnoses = data.diagnoses[:diagnosis_count]
    procedures = data.procedures[:procedure_count]
    medications = data.medications[:medication_count]
    labs = data.lab_events[:lab_count]
    used_item_ids = {row["lab_item_id"] for row in labs}
    lab_items = [row for row in data.lab_items if row["lab_item_id"] in used_item_ids]

    nodes = {("Patient", patient["patient_id"]), ("Admission", admission["admission_id"])}
    nodes |= {("Diagnosis", row["diagnosis_id"]) for row in diagnoses}
    nodes |= {("Procedure", row["procedure_id"]) for row in procedures}
    nodes |= {("Medication", row["medication_id"]) for row in medications}
    nodes |= {("LabEvent", row["lab_event_id"]) for row in labs}
    nodes |= {("LabItem", row["lab_item_id"]) for row in lab_items}

    edges = {(patient["patient_id"], "has_admission", admission["admission_id"])}
    edges |= {(admission["admission_id"], "has_diagnosis", row["diagnosis_id"]) for row in diagnoses}
    edges |= {(admission["admission_id"], "has_procedure", row["procedure_id"]) for row in procedures}
    edges |= {(admission["admission_id"], "prescribed_medication", row["medication_id"]) for row in medications}
    edges |= {(admission["admission_id"], "has_lab_event", row["lab_event_id"]) for row in labs}
    edges |= {(row["lab_event_id"], "is_test_of", row["lab_item_id"]) for row in labs}
    return nodes, edges


# -----------------------------
# Prompt 构造
# -----------------------------


RELATION_RULES = """Allowed relation directions:
Patient -> Admission: has_admission
Admission -> Diagnosis: has_diagnosis
Admission -> Procedure: has_procedure
Admission -> Medication: prescribed_medication
Admission -> LabEvent: has_lab_event
LabEvent -> LabItem: is_test_of"""


def strict_mini_prompt(input_text: str) -> str:
    """P1/P5/P6 的严格 schema prompt。"""

    return f"""You extract a small hospital knowledge graph. Return JSON only.
Allowed node labels: Patient, Admission, Diagnosis, Procedure, Medication, LabEvent, LabItem.
{RELATION_RULES}
Use exact IDs from the input. Do not invent IDs.
Return exactly this JSON shape:
{{"nodes":[{{"node_id":"...","label":"...","name":"..."}}],"edges":[{{"source_id":"...","relation":"...","target_id":"..."}}]}}

Input rows:
{input_text}"""


def loose_mini_prompt(input_text: str) -> str:
    """P1 stress 的宽松 prompt，用于暴露非 schema relation 错误。"""

    return f"""Extract a hospital knowledge graph from the following data.
Return JSON only with nodes and edges.
JSON shape:
{{"nodes":[{{"node_id":"...","label":"...","name":"..."}}],"edges":[{{"source_id":"...","relation":"...","target_id":"..."}}]}}

Data:
{input_text}"""


def lab_rows_as_table(rows: list[dict[str, str]]) -> str:
    """把 LabEvent 行转成字段化文本。"""

    keep = ["lab_event_id", "admission_id", "labevent_id", "itemid", "label", "value", "valuenum", "valueuom", "flag", "charttime"]
    return "\n".join(
        "- " + ", ".join(f"{key}={row[key]}" for key in keep if row.get(key))
        for row in rows
    )


def lab_rows_as_narrative(rows: list[dict[str, str]]) -> str:
    """把 LabEvent 行转成密集自然句，用于 P3 stress。"""

    parts = []
    for row in rows:
        parts.append(
            f"At {row['charttime']}, admission {row['hadm_id']} had lab event {row['labevent_id']} "
            f"for item {row['itemid']} called {row['label']}; value {row['value']} {row['valueuom']}; "
            f"flag {row['flag'] or 'none'}. Use admission node admission:{row['hadm_id']}, "
            f"event node labevent:{row['labevent_id']}, item node labitem:{row['itemid']}."
        )
    return " ".join(parts)


def lab_event_only_prompt(input_text: str) -> str:
    """P3 基础实验：只抽 LabEvent 节点和 has_lab_event 边。"""

    return f"""Return JSON only. For every input row, output exactly:
1) one node: {{"node_id": lab_event_id, "label": "LabEvent", "name": label}}
2) one edge: {{"source_id": admission_id, "relation": "has_lab_event", "target_id": lab_event_id}}
Do not output LabItem nodes. Do not invent IDs. Do not skip rows.
JSON shape:
{{"nodes":[{{"node_id":"...","label":"LabEvent","name":"..."}}],"edges":[{{"source_id":"...","relation":"has_lab_event","target_id":"..."}}]}}

Input rows:
{input_text}"""


def lab_event_item_prompt(input_text: str) -> str:
    """P3 stress：同时抽 LabEvent、LabItem 以及两类边。"""

    return f"""Extract a hospital lab knowledge graph. Return JSON only.
Allowed node labels: LabEvent, LabItem.
Allowed relation directions:
Admission -> LabEvent: has_lab_event
LabEvent -> LabItem: is_test_of
Use exact canonical IDs mentioned in the text: admission:..., labevent:..., labitem:...
Return JSON shape:
{{"nodes":[{{"node_id":"...","label":"...","name":"..."}}],"edges":[{{"source_id":"...","relation":"...","target_id":"..."}}]}}

Text:
{input_text}"""


def p4_basic_raw_text(data: HospitalGoldData) -> str:
    """P4 基础实验：自然句形式的 LabEvent 输入。"""

    lines = ["Clinical note style input:"]
    for row in data.lab_events:
        lines.append(
            f"For hospital admission {row['hadm_id']} (admission node {row['admission_id']}), "
            f"lab record {row['labevent_id']} / {row['lab_event_id']} reports {row['label']} "
            f"in {row['fluid']} at {row['charttime']}; the observed value was {row['value']} "
            f"{row['valueuom']} and the flag was {row['flag'] or 'not flagged'}."
        )
    return "\n".join(lines)


def p4_basic_unified_text(data: HospitalGoldData) -> str:
    """P4 基础实验：统一字段形式的 LabEvent 输入。"""

    lines = ["[LAB_EVENTS]"]
    for index, row in enumerate(data.lab_events, start=1):
        lines.append(
            f"{index}. lab_event_id={row['lab_event_id']} | admission_id={row['admission_id']} "
            f"| label={row['label']} | value={row['value']} | unit={row['valueuom']} "
            f"| flag={row['flag']} | charttime={row['charttime']}"
        )
    return "\n".join(lines)


def p4_lab_event_prompt(input_text: str) -> str:
    """P4 基础实验使用的 LabEvent-only prompt。"""

    return f"""Return JSON only. Extract LabEvent nodes and has_lab_event edges.
Rules:
- For every source-supported lab event, output one LabEvent node using the exact lab_event_id.
- For every source-supported lab event, output one edge: admission_id --has_lab_event--> lab_event_id.
- Do not output LabItem nodes, diagnoses, procedures, or medications.
- Do not invent IDs.
JSON shape:
{{"nodes":[{{"node_id":"...","label":"LabEvent","name":"..."}}],"edges":[{{"source_id":"...","relation":"has_lab_event","target_id":"..."}}]}}

Input:
{input_text}"""


def p4_stress_raw_text(data: HospitalGoldData) -> str:
    """P4 stress：不给 canonical ID，让模型从自然文本自己构造 ID。"""

    patient = data.patient
    admission = data.admission
    diagnoses = data.diagnoses[:3]
    procedures = data.procedures[:2]
    medications = data.medications[:3]
    labs = data.lab_events[:3]
    lines = [
        f"Patient subject {patient['subject_id']} is a {patient['gender']} patient aged {patient['anchor_age']}.",
        f"The admission hadm id is {admission['hadm_id']}; admission type {admission['admission_type']}; "
        f"admitted at {admission['admittime']} and discharged at {admission['dischtime']}.",
        "Diagnoses mentioned: "
        + "; ".join(
            f"ICD version {row['icd_version']} code {row['icd_code']} meaning {row['long_title']}"
            for row in diagnoses
        )
        + ".",
        "Procedures mentioned: "
        + "; ".join(
            f"ICD version {row['icd_version']} procedure code {row['icd_code']} on {row['chartdate']} meaning {row['long_title']}"
            for row in procedures
        )
        + ".",
        "Medication orders: "
        + "; ".join(
            f"drug {row['drug']} with route {row['route']} and dose {row['dose_val_rx']} {row['dose_unit_rx']}"
            for row in medications
        )
        + ".",
        "Laboratory events: "
        + "; ".join(
            f"event id {row['labevent_id']} item {row['itemid']} {row['label']} value {row['value']} {row['valueuom']} flag {row['flag']}"
            for row in labs
        )
        + ".",
    ]
    return "\n".join(lines)


def p4_stress_unified_text(data: HospitalGoldData) -> str:
    """P4 stress：显式给出标准 node_id、label 和 edge。"""

    patient = data.patient
    admission = data.admission
    diagnoses = data.diagnoses[:3]
    procedures = data.procedures[:2]
    medications = data.medications[:3]
    labs = data.lab_events[:3]
    used_item_ids = {row["lab_item_id"] for row in labs}
    lab_items = [row for row in data.lab_items if row["lab_item_id"] in used_item_ids]

    lines = ["[NODES]"]

    def node(label: str, node_id: str, name: str) -> None:
        lines.append(f"- node_id={node_id} | label={label} | name={name}")

    node("Patient", patient["patient_id"], patient["patient_id"])
    node("Admission", admission["admission_id"], admission["admission_id"])
    for row in diagnoses:
        node("Diagnosis", row["diagnosis_id"], row["long_title"])
    for row in procedures:
        node("Procedure", row["procedure_id"], row["long_title"])
    for row in medications:
        node("Medication", row["medication_id"], row["drug"])
    for row in labs:
        node("LabEvent", row["lab_event_id"], row["label"])
    for row in lab_items:
        node("LabItem", row["lab_item_id"], row["label"])

    lines.append("[EDGES]")

    def edge(source: str, relation: str, target: str) -> None:
        lines.append(f"- source_id={source} | relation={relation} | target_id={target}")

    edge(patient["patient_id"], "has_admission", admission["admission_id"])
    for row in diagnoses:
        edge(admission["admission_id"], "has_diagnosis", row["diagnosis_id"])
    for row in procedures:
        edge(admission["admission_id"], "has_procedure", row["procedure_id"])
    for row in medications:
        edge(admission["admission_id"], "prescribed_medication", row["medication_id"])
    for row in labs:
        edge(admission["admission_id"], "has_lab_event", row["lab_event_id"])
        edge(row["lab_event_id"], "is_test_of", row["lab_item_id"])
    return "\n".join(lines)


def p4_stress_prompt(input_text: str) -> str:
    """P4 stress 使用的多类型 KG prompt。"""

    return f"""Extract a hospital knowledge graph. Return JSON only.
Allowed labels: Patient, Admission, Diagnosis, Procedure, Medication, LabEvent, LabItem.
{RELATION_RULES}
If canonical node IDs are not explicitly shown, construct them using these rules:
patient:{{subject_id}}, admission:{{hadm_id}}, diagnosis:{{icd_version}}:{{icd_code}},
procedure:{{icd_version}}:{{icd_code}}, medication:{{lowercase drug name}},
labevent:{{labevent_id}}, labitem:{{itemid}}.
Return JSON shape:
{{"nodes":[{{"node_id":"...","label":"...","name":"..."}}],"edges":[{{"source_id":"...","relation":"...","target_id":"..."}}]}}

Input:
{input_text}"""


# -----------------------------
# 实验执行函数
# -----------------------------


def evaluate_once(
    model: str,
    prompt: str,
    gold_nodes: set[tuple[str, str]],
    gold_edges: set[tuple[str, str, str]],
    timeout: int,
) -> tuple[dict[str, Any], GraphPrediction | None]:
    """执行一次模型调用并计算节点/边指标。"""

    started = datetime.now().isoformat(timespec="seconds")
    try:
        obj, seconds = call_ollama(model=model, prompt=prompt, timeout=timeout)
        prediction = parse_prediction(obj)
        node_metric = compute_prf(prediction.nodes, gold_nodes)
        edge_metric = compute_prf(prediction.edges, gold_edges)
        row = {
            "started_at": started,
            "model": model,
            "seconds": round(seconds, 2),
            "error": "",
            "node_precision": node_metric.precision,
            "node_recall": node_metric.recall,
            "node_f1": node_metric.f1,
            "node_tp": node_metric.tp,
            "node_fp": node_metric.fp,
            "node_fn": node_metric.fn,
            "node_predicted": node_metric.predicted,
            "node_gold": node_metric.gold,
            "edge_precision": edge_metric.precision,
            "edge_recall": edge_metric.recall,
            "edge_f1": edge_metric.f1,
            "edge_tp": edge_metric.tp,
            "edge_fp": edge_metric.fp,
            "edge_fn": edge_metric.fn,
            "edge_predicted": edge_metric.predicted,
            "edge_gold": edge_metric.gold,
        }
        add_postprocess_metrics(row, prediction, gold_nodes, gold_edges)
        return row, prediction
    except Exception as error:
        node_metric = compute_prf(set(), gold_nodes)
        edge_metric = compute_prf(set(), gold_edges)
        row = {
            "started_at": started,
            "model": model,
            "seconds": timeout,
            "error": f"{type(error).__name__}: {error}",
            "node_precision": node_metric.precision,
            "node_recall": node_metric.recall,
            "node_f1": node_metric.f1,
            "node_tp": node_metric.tp,
            "node_fp": node_metric.fp,
            "node_fn": node_metric.fn,
            "node_predicted": 0,
            "node_gold": node_metric.gold,
            "edge_precision": edge_metric.precision,
            "edge_recall": edge_metric.recall,
            "edge_f1": edge_metric.f1,
            "edge_tp": edge_metric.tp,
            "edge_fp": edge_metric.fp,
            "edge_fn": edge_metric.fn,
            "edge_predicted": 0,
            "edge_gold": edge_metric.gold,
        }
        add_postprocess_metrics(row, None, gold_nodes, gold_edges)
        return row, None


def run_p1(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P1：最小严格抽取 sanity check。"""

    gold_nodes, gold_edges, input_text = mini_gold(data)
    row, _ = evaluate_once(model, strict_mini_prompt(input_text), gold_nodes, gold_edges, timeout)
    row["experiment"] = "P1_minimal_strict_sanity"
    row["condition"] = "strict_schema_prompt"
    write_csv(output_dir / "p1_minimal_strict_sanity.csv", [row])
    return [row]


def run_p1_stress(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P1 stress：比较宽松 prompt 和严格 schema prompt。"""

    gold_nodes, gold_edges, input_text = mini_gold(data)
    rows = []
    for condition, prompt in [
        ("loose_prompt", loose_mini_prompt(input_text)),
        ("strict_schema_prompt", strict_mini_prompt(input_text)),
    ]:
        row, _ = evaluate_once(model, prompt, gold_nodes, gold_edges, timeout)
        row["experiment"] = "P1_stress_prompt_constraint"
        row["condition"] = condition
        rows.append(row)
        print_progress(row)
    write_csv(output_dir / "p1_stress_prompt_constraint.csv", rows)
    return rows


def run_p2(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P2：完整 one-shot 与分层抽取对比。"""

    full_nodes, full_edges = full_gold_from_csv(data)
    full_input = full_source_text(data)
    rows = []

    one_shot_row, _ = evaluate_once(model, strict_full_prompt(full_input), full_nodes, full_edges, timeout)
    one_shot_row["experiment"] = "P2_one_shot_vs_layered"
    one_shot_row["condition"] = "one_shot_full_graph"
    rows.append(one_shot_row)
    print_progress(one_shot_row)

    merged_prediction = GraphPrediction(nodes=set(), edges=set(), raw_json={})
    layer_errors: list[str] = []
    for layer_name, prompt, gold_node_subset, gold_edge_subset in layered_prompts(data):
        row, prediction = evaluate_once(model, prompt, gold_node_subset, gold_edge_subset, timeout)
        row["experiment"] = "P2_layer_detail"
        row["condition"] = layer_name
        rows.append(row)
        print_progress(row)
        if prediction:
            merged_prediction.nodes |= prediction.nodes
            merged_prediction.edges |= prediction.edges
        if row["error"]:
            layer_errors.append(f"{layer_name}: {row['error']}")

    node_metric = compute_prf(merged_prediction.nodes, full_nodes)
    edge_metric = compute_prf(merged_prediction.edges, full_edges)
    merged_row = metric_row_from_metrics(
        experiment="P2_one_shot_vs_layered",
        condition="layered_merged_graph",
        model=model,
        seconds=sum(float(row["seconds"]) for row in rows if row["condition"] != "one_shot_full_graph"),
        error="; ".join(layer_errors),
        node_metric=node_metric,
        edge_metric=edge_metric,
    )
    add_postprocess_metrics(merged_row, merged_prediction, full_nodes, full_edges)
    rows.append(merged_row)
    print_progress(merged_row)

    write_csv(output_dir / "p2_one_shot_vs_layered.csv", rows)
    return rows


def run_p3(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P3：基础 LabEvent 分块实验。"""

    gold_nodes, gold_edges = lab_event_gold(data, include_items=False)
    rows = []
    for chunk_size in [12, 6, 3, 1]:
        prediction, seconds, errors = run_chunks(
            model=model,
            chunks=chunk_rows(data.lab_events, chunk_size),
            prompt_builder=lambda chunk: lab_event_only_prompt(lab_rows_as_table(chunk)),
            timeout=timeout,
        )
        node_metric = compute_prf(prediction.nodes, gold_nodes)
        edge_metric = compute_prf(prediction.edges, gold_edges)
        row = metric_row_from_metrics(
            experiment="P3_chunk_size_lab_event",
            condition=f"chunk_size_{chunk_size}",
            model=model,
            seconds=seconds,
            error="; ".join(errors),
            node_metric=node_metric,
            edge_metric=edge_metric,
        )
        row["chunk_size"] = chunk_size
        row["chunks"] = len(chunk_rows(data.lab_events, chunk_size))
        add_postprocess_metrics(row, prediction, gold_nodes, gold_edges)
        rows.append(row)
        print_progress(row)
    write_csv(output_dir / "p3_chunk_size_lab_event.csv", rows)
    return rows


def run_p3_stress(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P3 stress：密集自然句 + LabEvent/LabItem 混合抽取的分块实验。"""

    gold_nodes, gold_edges = lab_event_gold(data, include_items=True)
    rows = []
    for chunk_size in [12, 6, 3, 1]:
        prediction, seconds, errors = run_chunks(
            model=model,
            chunks=chunk_rows(data.lab_events, chunk_size),
            prompt_builder=lambda chunk: lab_event_item_prompt(lab_rows_as_narrative(chunk)),
            timeout=timeout,
        )
        node_metric = compute_prf(prediction.nodes, gold_nodes)
        edge_metric = compute_prf(prediction.edges, gold_edges)
        row = metric_row_from_metrics(
            experiment="P3_stress_dense_lab_narrative",
            condition=f"chunk_size_{chunk_size}",
            model=model,
            seconds=seconds,
            error="; ".join(errors),
            node_metric=node_metric,
            edge_metric=edge_metric,
        )
        row["chunk_size"] = chunk_size
        row["chunks"] = len(chunk_rows(data.lab_events, chunk_size))
        add_postprocess_metrics(row, prediction, gold_nodes, gold_edges)
        rows.append(row)
        print_progress(row)
    write_csv(output_dir / "p3_stress_dense_lab_narrative.csv", rows)
    return rows


def run_p4(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P4：基础 raw vs unified LabEvent 输入对比。"""

    gold_nodes, gold_edges = lab_event_gold(data, include_items=False)
    rows = []
    for condition, text in [
        ("raw_text", p4_basic_raw_text(data)),
        ("unified_text_structure", p4_basic_unified_text(data)),
    ]:
        row, _ = evaluate_once(model, p4_lab_event_prompt(text), gold_nodes, gold_edges, timeout)
        row["experiment"] = "P4_raw_vs_unified_lab_event"
        row["condition"] = condition
        rows.append(row)
        print_progress(row)
    write_csv(output_dir / "p4_raw_vs_unified_lab_event.csv", rows)
    return rows


def run_p4_stress(data: HospitalGoldData, output_dir: Path, model: str, timeout: int) -> list[dict[str, Any]]:
    """P4 stress：raw noisy text 和 canonical unified structure 对比。"""

    gold_nodes, gold_edges = mixed_gold(data)
    rows = []
    for condition, text in [
        ("raw_noisy_text", p4_stress_raw_text(data)),
        ("unified_text_structure", p4_stress_unified_text(data)),
    ]:
        row, _ = evaluate_once(model, p4_stress_prompt(text), gold_nodes, gold_edges, timeout)
        row["experiment"] = "P4_stress_raw_noisy_vs_unified"
        row["condition"] = condition
        rows.append(row)
        print_progress(row)
    write_csv(output_dir / "p4_stress_raw_noisy_vs_unified.csv", rows)
    return rows


def run_p5(data: HospitalGoldData, output_dir: Path, models: list[str], timeout: int) -> list[dict[str, Any]]:
    """P5：不同本地小模型对比。"""

    gold_nodes, gold_edges, input_text = mini_gold(data)
    rows = []
    for model in models:
        row, _ = evaluate_once(model, strict_mini_prompt(input_text), gold_nodes, gold_edges, timeout)
        row["experiment"] = "P5_model_comparison"
        row["condition"] = "mixed_mini_graph"
        rows.append(row)
        print_progress(row)
    write_csv(output_dir / "p5_model_comparison.csv", rows)
    return rows


def run_p6(data: HospitalGoldData, output_dir: Path, models: list[str], timeout: int, repetitions: int) -> list[dict[str, Any]]:
    """P6：重复运行稳定性实验。"""

    gold_nodes, gold_edges, input_text = mini_gold(data)
    rows = []
    for model in models:
        predictions: list[GraphPrediction] = []
        node_f1_values: list[float] = []
        edge_f1_values: list[float] = []
        errors: list[str] = []
        seconds = 0.0

        for run_number in range(1, repetitions + 1):
            row, prediction = evaluate_once(model, strict_mini_prompt(input_text), gold_nodes, gold_edges, timeout)
            row["experiment"] = "P6_repeated_run_detail"
            row["condition"] = f"run_{run_number}"
            rows.append(row)
            print_progress(row)
            seconds += float(row["seconds"])
            node_f1_values.append(float(row["node_f1"]))
            edge_f1_values.append(float(row["edge_f1"]))
            if prediction:
                predictions.append(prediction)
            if row["error"]:
                errors.append(row["error"])

        node_overlaps = [jaccard(left.nodes, right.nodes) for left, right in combinations(predictions, 2)]
        relation_agreements = [jaccard(left.edges, right.edges) for left, right in combinations(predictions, 2)]
        graph_overlaps = [jaccard(graph_facts(left), graph_facts(right)) for left, right in combinations(predictions, 2)]
        summary = {
            "experiment": "P6_repeated_run_stability",
            "condition": "summary",
            "model": model,
            "repetitions": repetitions,
            "seconds": round(seconds, 2),
            "error": "; ".join(errors),
            "mean_node_overlap": mean(node_overlaps) if node_overlaps else 1.0,
            "mean_relation_agreement": mean(relation_agreements) if relation_agreements else 1.0,
            "mean_graph_overlap": mean(graph_overlaps) if graph_overlaps else 1.0,
            "mean_node_f1": mean(node_f1_values),
            "node_f1_std": population_std(node_f1_values),
            "mean_edge_f1": mean(edge_f1_values),
            "edge_f1_std": population_std(edge_f1_values),
        }
        rows.append(summary)
        print(f"[P6 summary] {model}: graph_overlap={summary['mean_graph_overlap']} edge_f1={summary['mean_edge_f1']}")
    write_csv(output_dir / "p6_repeated_run_stability.csv", rows)
    return rows


# -----------------------------
# P2 辅助函数
# -----------------------------


def full_gold_from_csv(data: HospitalGoldData) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    """构造完整 admission gold graph。"""

    nodes = {("Patient", data.patient["patient_id"]), ("Admission", data.admission["admission_id"])}
    nodes |= {("Diagnosis", row["diagnosis_id"]) for row in data.diagnoses}
    nodes |= {("Procedure", row["procedure_id"]) for row in data.procedures}
    nodes |= {("Medication", row["medication_id"]) for row in data.medications}
    nodes |= {("LabEvent", row["lab_event_id"]) for row in data.lab_events}
    used_item_ids = {row["lab_item_id"] for row in data.lab_events}
    nodes |= {("LabItem", row["lab_item_id"]) for row in data.lab_items if row["lab_item_id"] in used_item_ids}

    edges = {(data.patient["patient_id"], "has_admission", data.admission["admission_id"])}
    edges |= {(data.admission["admission_id"], "has_diagnosis", row["diagnosis_id"]) for row in data.diagnoses}
    edges |= {(data.admission["admission_id"], "has_procedure", row["procedure_id"]) for row in data.procedures}
    edges |= {(data.admission["admission_id"], "prescribed_medication", row["medication_id"]) for row in data.medications}
    edges |= {(data.admission["admission_id"], "has_lab_event", row["lab_event_id"]) for row in data.lab_events}
    edges |= {(row["lab_event_id"], "is_test_of", row["lab_item_id"]) for row in data.lab_events}
    return nodes, edges


def full_source_text(data: HospitalGoldData) -> str:
    """把完整 source rows 串成 one-shot 输入。"""

    sections = [
        ("Patient", [data.patient]),
        ("Admission", [data.admission]),
        ("Diagnoses", data.diagnoses),
        ("Procedures", data.procedures),
        ("Medications", data.medications),
        ("LabEvents", data.lab_events),
        ("LabItems", data.lab_items),
    ]
    lines = []
    for title, rows in sections:
        lines.append(f"[{title}]")
        for row in rows:
            lines.append("- " + ", ".join(f"{key}={value}" for key, value in row.items() if value))
    return "\n".join(lines)


def strict_full_prompt(input_text: str) -> str:
    """完整 one-shot KG 抽取 prompt。"""

    return f"""Extract the complete hospital knowledge graph from the input.
Return JSON only.
Allowed labels: Patient, Admission, Diagnosis, Procedure, Medication, LabEvent, LabItem.
{RELATION_RULES}
Use exact IDs from the input. Do not invent IDs.
Return JSON shape:
{{"nodes":[{{"node_id":"...","label":"...","name":"..."}}],"edges":[{{"source_id":"...","relation":"...","target_id":"..."}}]}}

Input:
{input_text}"""


def layered_prompts(
    data: HospitalGoldData,
) -> list[tuple[str, str, set[tuple[str, str]], set[tuple[str, str, str]]]]:
    """构造 P2 分层抽取的每一层 prompt 和对应 gold subset。"""

    admission = data.admission
    layers = []

    patient_admission_text = "[Patient]\n- " + ", ".join(f"{k}={v}" for k, v in data.patient.items() if v)
    patient_admission_text += "\n[Admission]\n- " + ", ".join(f"{k}={v}" for k, v in admission.items() if v)
    layers.append((
        "patient_admission",
        strict_full_prompt(patient_admission_text),
        {("Patient", data.patient["patient_id"]), ("Admission", admission["admission_id"])},
        {(data.patient["patient_id"], "has_admission", admission["admission_id"])},
    ))

    def rows_text(title: str, rows: list[dict[str, str]]) -> str:
        return f"[{title}]\n" + "\n".join(
            "- " + ", ".join(f"{key}={value}" for key, value in row.items() if value)
            for row in rows
        )

    layers.append((
        "diagnoses",
        strict_full_prompt(rows_text("Diagnoses", data.diagnoses)),
        {("Diagnosis", row["diagnosis_id"]) for row in data.diagnoses},
        {(admission["admission_id"], "has_diagnosis", row["diagnosis_id"]) for row in data.diagnoses},
    ))
    layers.append((
        "procedures",
        strict_full_prompt(rows_text("Procedures", data.procedures)),
        {("Procedure", row["procedure_id"]) for row in data.procedures},
        {(admission["admission_id"], "has_procedure", row["procedure_id"]) for row in data.procedures},
    ))
    layers.append((
        "medications",
        strict_full_prompt(rows_text("Medications", data.medications)),
        {("Medication", row["medication_id"]) for row in data.medications},
        {(admission["admission_id"], "prescribed_medication", row["medication_id"]) for row in data.medications},
    ))
    layers.append((
        "lab_events",
        lab_event_only_prompt(lab_rows_as_table(data.lab_events)),
        {("LabEvent", row["lab_event_id"]) for row in data.lab_events},
        {(admission["admission_id"], "has_lab_event", row["lab_event_id"]) for row in data.lab_events},
    ))
    used_item_ids = {row["lab_item_id"] for row in data.lab_events}
    lab_item_rows = [row for row in data.lab_items if row["lab_item_id"] in used_item_ids]
    lab_item_text = rows_text("LabEvents", data.lab_events) + "\n" + rows_text("LabItems", lab_item_rows)
    layers.append((
        "lab_items",
        lab_event_item_prompt(lab_item_text),
        {("LabItem", row["lab_item_id"]) for row in lab_item_rows},
        {(row["lab_event_id"], "is_test_of", row["lab_item_id"]) for row in data.lab_events},
    ))
    return layers


# -----------------------------
# 结果汇总、绘图和命令行
# -----------------------------


def metric_row_from_metrics(
    experiment: str,
    condition: str,
    model: str,
    seconds: float,
    error: str,
    node_metric: Metric,
    edge_metric: Metric,
) -> dict[str, Any]:
    """根据已计算好的节点/边指标构造 CSV 行。"""

    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "condition": condition,
        "model": model,
        "seconds": round(seconds, 2),
        "error": error,
        "node_precision": node_metric.precision,
        "node_recall": node_metric.recall,
        "node_f1": node_metric.f1,
        "node_tp": node_metric.tp,
        "node_fp": node_metric.fp,
        "node_fn": node_metric.fn,
        "node_predicted": node_metric.predicted,
        "node_gold": node_metric.gold,
        "edge_precision": edge_metric.precision,
        "edge_recall": edge_metric.recall,
        "edge_f1": edge_metric.f1,
        "edge_tp": edge_metric.tp,
        "edge_fp": edge_metric.fp,
        "edge_fn": edge_metric.fn,
        "edge_predicted": edge_metric.predicted,
        "edge_gold": edge_metric.gold,
    }


def chunk_rows(rows: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    """按指定 chunk size 分割 source rows。"""

    return [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]


def run_chunks(
    model: str,
    chunks: list[list[dict[str, str]]],
    prompt_builder: Any,
    timeout: int,
) -> tuple[GraphPrediction, float, list[str]]:
    """逐 chunk 调用模型，并合并每个 chunk 的预测结果。"""

    merged = GraphPrediction(nodes=set(), edges=set(), raw_json={})
    total_seconds = 0.0
    errors: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        try:
            obj, seconds = call_ollama(model=model, prompt=prompt_builder(chunk), timeout=timeout)
            prediction = parse_prediction(obj)
            merged.nodes |= prediction.nodes
            merged.edges |= prediction.edges
            total_seconds += seconds
            print(f"  chunk {index}/{len(chunks)}: {seconds:.2f}s nodes={len(prediction.nodes)} edges={len(prediction.edges)}")
        except Exception as error:
            total_seconds += timeout
            errors.append(f"chunk {index}: {type(error).__name__}: {error}")
            print(f"  chunk {index}/{len(chunks)} ERROR: {type(error).__name__}: {error}")
    return merged, total_seconds, errors


def print_progress(row: dict[str, Any]) -> None:
    """在终端打印一行简短进度。"""

    print(
        f"[{row.get('experiment')}] {row.get('condition')} "
        f"model={row.get('model')} nodeF1={row.get('node_f1')} edgeF1={row.get('edge_f1')} "
        f"error={'yes' if row.get('error') else 'no'}",
        flush=True,
    )


def save_markdown_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """保存一个简单 Markdown 汇总，方便组员打开查看。"""

    lines = [
        "# Hospital Gold Pilot Experiment Results",
        "",
        "中文说明：本文件由脚本自动生成。结果是本地 Ollama 模型在当前机器上的复现实验输出。",
        "",
        "| Experiment | Condition | Model | Raw Node F1 | Raw Edge F1 | Normalized Node F1 | Normalized Edge F1 | Error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if "node_f1" not in row or "edge_f1" not in row:
            continue
        lines.append(
            f"| {row.get('experiment', '')} | {row.get('condition', '')} | {row.get('model', '')} "
            f"| {row.get('raw_node_f1', row.get('node_f1', ''))} "
            f"| {row.get('raw_edge_f1', row.get('edge_f1', ''))} "
            f"| {row.get('normalized_node_f1', '')} "
            f"| {row.get('normalized_edge_f1', '')} | {row.get('error', '')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_make_plots(output_dir: Path) -> None:
    """如果本机安装了 matplotlib，就根据 CSV 生成几张核心 SVG 图。"""

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("未安装 matplotlib，跳过 SVG 图生成；CSV 和 Markdown 已保存。")
        return

    def read_rows(name: str) -> list[dict[str, str]]:
        path = output_dir / name
        return read_csv_rows(path) if path.exists() else []

    def grouped_bar(filename: str, title: str, groups: list[str], series: list[tuple[str, list[float]]]) -> None:
        x = np.arange(len(groups))
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for index, (label, values) in enumerate(series):
            ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=label)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=10, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend()
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    p1_stress = read_rows("p1_stress_prompt_constraint.csv")
    if p1_stress:
        grouped_bar(
            "p1_stress_prompt_constraint.svg",
            "P1 Stress: Prompt Constraint",
            [row["condition"] for row in p1_stress],
            [
                ("Node F1", [float(row["node_f1"]) for row in p1_stress]),
                ("Edge F1", [float(row["edge_f1"]) for row in p1_stress]),
            ],
        )

    p3_stress = read_rows("p3_stress_dense_lab_narrative.csv")
    if p3_stress:
        grouped_bar(
            "p3_stress_chunk_size.svg",
            "P3 Stress: Chunk Size",
            [row["condition"] for row in p3_stress],
            [
                ("Node F1", [float(row["node_f1"]) for row in p3_stress]),
                ("Edge F1", [float(row["edge_f1"]) for row in p3_stress]),
            ],
        )

    p4_stress = read_rows("p4_stress_raw_noisy_vs_unified.csv")
    if p4_stress:
        grouped_bar(
            "p4_stress_raw_vs_unified.svg",
            "P4 Stress: Raw Noisy vs Unified",
            [row["condition"] for row in p4_stress],
            [
                ("Node F1", [float(row["node_f1"]) for row in p4_stress]),
                ("Edge F1", [float(row["edge_f1"]) for row in p4_stress]),
            ],
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run hospital gold pilot KG extraction experiments.")
    parser.add_argument("--gold-dir", default="gold_case_admission_29366372", help="gold case 数据目录")
    parser.add_argument("--output-dir", default="", help="结果输出目录；默认自动生成 results/hospital_gold_repro_<timestamp>")
    parser.add_argument("--model", default="llama3.1:8b", help="主要实验使用的 Ollama 模型")
    parser.add_argument(
        "--models",
        default="llama3.1:8b,qwen2.5:7b,llama3.2:latest",
        help="P5/P6 模型列表，用逗号分隔",
    )
    parser.add_argument("--timeout", type=int, default=120, help="单次 Ollama 调用 timeout 秒数")
    parser.add_argument("--repetitions", type=int, default=3, help="P6 重复运行次数")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "base", "stress", "p1", "p2", "p3", "p4", "p5", "p6"],
        help="选择运行哪些实验：all/base/stress 或单个 P 实验",
    )
    return parser.parse_args()


def resolve_gold_dir(value: str) -> Path:
    """Resolve hospital gold-case path for both repo and delivery-package layouts."""

    requested = Path(value)
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        cwd = Path.cwd()
        script_root = Path(__file__).resolve().parents[1]
        candidates.extend(
            [
                cwd / requested,
                script_root / requested,
                cwd / "data" / "hospital_gold" / requested.name,
                cwd.parent / "data" / "hospital_gold" / requested.name,
                script_root / "data" / "hospital_gold" / requested.name,
                script_root.parent / "data" / "hospital_gold" / requested.name,
            ]
        )
    for candidate in candidates:
        if (candidate / "gold_case_schema.json").exists():
            return candidate.resolve()
    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"找不到 gold case 目录，已检查:\n{checked}")


def main() -> int:
    """主入口：按 suite 执行实验并保存结果。"""

    global POSTPROCESS_CONTEXT

    args = parse_args()
    try:
        gold_dir = resolve_gold_dir(args.gold_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("results") / f"hospital_gold_repro_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_gold_data(gold_dir)
    POSTPROCESS_CONTEXT = load_hospital_postprocess_context(gold_dir)
    model_list = [item.strip() for item in args.models.split(",") if item.strip()]
    all_rows: list[dict[str, Any]] = []

    print(f"Gold dir: {gold_dir}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Suite: {args.suite}")
    print("请确认 Ollama 已运行；如未运行，请先执行: ollama serve")

    if args.suite in {"all", "base", "p1"}:
        all_rows += run_p1(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "stress", "p1"}:
        all_rows += run_p1_stress(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "base", "p2"}:
        all_rows += run_p2(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "base", "p3"}:
        all_rows += run_p3(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "stress", "p3"}:
        all_rows += run_p3_stress(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "base", "p4"}:
        all_rows += run_p4(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "stress", "p4"}:
        all_rows += run_p4_stress(data, output_dir, args.model, args.timeout)
    if args.suite in {"all", "base", "p5"}:
        all_rows += run_p5(data, output_dir, model_list, args.timeout)
    if args.suite in {"all", "base", "p6"}:
        all_rows += run_p6(data, output_dir, model_list[:2], args.timeout, args.repetitions)

    write_csv(output_dir / "all_results.csv", all_rows)
    save_markdown_summary(output_dir, all_rows)
    try_make_plots(output_dir)
    print(f"\n完成。结果目录: {output_dir.resolve()}")
    print(f"总表: {(output_dir / 'all_results.csv').resolve()}")
    print(f"Markdown 汇总: {(output_dir / 'summary.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
