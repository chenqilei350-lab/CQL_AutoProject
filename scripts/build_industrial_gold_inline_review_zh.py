#!/usr/bin/env python3
"""Build a Chinese inline-review document for industrial gold candidates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRE_REVIEW = ROOT / "data/industrial_gold_candidates/industrial_gold_pre_review_codex_2026-07-31.jsonl"
TRANSLATIONS = ROOT / "data/industrial_gold_candidates/industrial_source_translations_zh_draft_2026-07-31.jsonl"
OUTPUT = ROOT / "docs/industrial_gold_inline_review_zh_2026-07-31.md"


NAME_ZH = {
    "adjust height": "调节高度",
    "attach connector plates": "安装连接板",
    "attach crank": "安装摇柄",
    "attach feet": "安装支脚",
    "attach lower plate": "安装下板",
    "attach monitor": "安装显示器",
    "attach power connection": "连接电源",
    "attach side plates": "安装侧板",
    "attach upper cover": "安装上盖",
    "attach wooden plate": "安装木板",
    "attach zip ties": "安装扎带",
    "bolt": "螺栓",
    "box": "盒子",
    "cable": "线缆",
    "camera": "摄像机",
    "clamp": "夹具",
    "clamp piece": "夹紧工件",
    "clean up": "清理现场",
    "connect legs": "连接桌腿",
    "connector": "连接件",
    "deliver and unload trolley": "运送并卸载手推车",
    "disassemble object": "拆卸物体",
    "disconnect bar": "断开横杆",
    "drilling machine": "电钻",
    "feet": "支脚",
    "find devices": "查找设备",
    "find drilling bit": "寻找钻头",
    "find missing part(s)": "寻找缺失部件",
    "gloves": "手套",
    "inspect": "检查",
    "leg": "桌腿",
    "load trolley": "装载手推车",
    "mark position": "标记位置",
    "mark tabletop": "标记桌面",
    "monitor": "显示器",
    "open box": "打开盒子",
    "open tool box": "打开工具箱",
    "organize": "整理",
    "pick up drilling machine": "拿起电钻",
    "pick up pen": "拿起笔",
    "pick up screwdriver": "拿起螺丝刀",
    "pick up tool": "拿起工具",
    "place down tool": "放下工具",
    "place object": "放置物体",
    "place padding": "放置衬垫",
    "place part(s)": "放置部件",
    "place piece": "放置工件",
    "plate": "板件",
    "plug in": "插入连接",
    "plug in cable(s)": "插入线缆",
    "plug in monitor": "连接显示器",
    "plug in power": "接通电源",
    "preparation": "准备工作",
    "press power button": "按下电源按钮",
    "put on gloves": "戴上手套",
    "read instruction": "阅读说明",
    "remove bolt": "拆下螺栓",
    "remove crank": "拆下摇柄",
    "remove feet": "拆下支脚",
    "remove gloves": "摘下手套",
    "remove lower plate": "拆下下板",
    "remove object": "移走物体",
    "remove part(s)": "拆下部件",
    "remove tabletop": "拆下桌面",
    "remove upper cover": "拆下上盖",
    "return clamp": "归还夹具",
    "screw": "螺钉",
    "screw in": "拧入螺钉",
    "screwdriver": "螺丝刀",
    "separate legs": "分离桌腿",
    "set the camera": "放置摄像机",
    "set the phone": "放置手机",
    "table": "桌子",
    "tabletop": "桌面板",
    "trolley": "手推车",
    "turn on": "开机",
    "unclamp": "松开夹具",
    "unplug monitor": "断开显示器",
    "unplug power": "断开电源",
    "unscrew": "拧松螺钉",
    "wood": "木材",
}

TYPE_ZH = {
    "Action": "动作",
    "Tool": "工具",
    "Object": "对象",
    "Worker": "工人",
    "Procedure": "流程",
    "Scene": "场景",
}

RELATION_ZH = {
    "BEFORE": "先于",
    "ACTS_ON": "作用于",
    "USES_TOOL": "使用工具",
    "PART_OF": "属于流程",
    "WARNING_FOR": "警告对应于",
}

DECISION_ZH = {
    "accept": "保留",
    "fix": "需要修正",
    "reject": "建议删除",
}

REVIEW_NAME_ZH = {
    "S02-N09": "连接板",
    "S02-N11": "木板",
    "S03-N12": "下板",
    "S04-N12": "下板",
    "S05-N09": "侧板",
    "S05-N10": "连接板",
    "S05-N11": "木板",
    "S06-N10": "下板",
    "S08-N11": "工具箱",
    "S09-N09": "螺钉",
    "S10-N10": "螺钉",
    "S11-N10": "螺钉",
    "S12-N09": "下板",
    "S13-N01": "检查流程",
    "S19-N11": "螺钉",
    "S20-N11": "螺钉",
}

WARNING_ISSUE_ZH = {
    "forgot gloves": "忘记戴手套",
    "only one screw": "只使用了一颗螺钉",
    "not possible": "无法执行",
    "screws loose": "螺钉松动",
    "didn't return clamp": "未归还夹具",
    "aligned wrong": "对齐错误",
    "didn't secure clamp": "未固定夹具",
    "didn't clamp": "未夹紧",
    "used hammer": "使用了锤子",
    "used hands": "用手操作",
    "wrong piece": "夹错工件",
    "too late": "执行太晚",
    "didn't connect": "未连接",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bold_translation(text: str) -> list[str]:
    text = text.replace("Lanon钥匙", "内六角扳手（原转录为 Lanon key）")
    text = text.replace("套筒扳手", "内六角扳手")
    text = text.replace("REM卡", "RAM 卡（原转录为 REM）")
    lines: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n", text.strip()):
        paragraph = paragraph.strip()
        if paragraph:
            lines.extend([f"**{paragraph}**", ""])
    return lines


def chinese_name(name: str, label: str | None = None, review_id: str | None = None) -> str:
    if review_id and review_id in REVIEW_NAME_ZH:
        return REVIEW_NAME_ZH[review_id]
    normalized = name.lower().strip()
    if normalized == "screw" and label == "Action":
        return "拧紧螺钉"
    return NAME_ZH.get(normalized, name)


def evidence_zh(evidence: str, fallback_name: str, label: str | None = None) -> str:
    match = re.match(r"(Action|Keystep)\s+(\d+):\s*(.*?)\s*(\([^)]*\))?\.?$", evidence.strip())
    if not match:
        return f"原始证据：{evidence}"
    prefix = "动作" if match.group(1) == "Action" else "步骤"
    number = match.group(2)
    name = chinese_name(match.group(3).strip(), label) or chinese_name(fallback_name, label)
    timestamp = match.group(4) or ""
    return f"{prefix}{number}：{name} {timestamp}".strip()


def node_line(node: dict[str, Any]) -> str:
    decision = DECISION_ZH[node["codex_proposed_decision"]]
    node_type = TYPE_ZH.get(node["label"], node["label"])
    name = chinese_name(node["name"], node["label"], node["review_id"])
    fix = f"；建议修正：{node['codex_proposed_fix']}" if node["codex_proposed_fix"] else ""
    return (
        f"【{node['review_id']}｜{node_type}｜{name}｜{decision}】{fix}  "
        f"原因：{node['codex_pre_review_reason']}"
    )


def edge_line(edge: dict[str, Any], node_by_id: dict[str, dict[str, Any]]) -> str:
    decision = DECISION_ZH[edge["codex_proposed_decision"]]
    source = node_by_id.get(edge["source"], {"name": edge["source"]})
    target = node_by_id.get(edge["target"], {"name": edge["target"]})
    source_name = chinese_name(source["name"], source.get("label"), source.get("review_id"))
    target_name = chinese_name(target["name"], target.get("label"), target.get("review_id"))
    relation = RELATION_ZH.get(edge["relation"], edge["relation"])
    fix = f"；建议修正：{edge['codex_proposed_fix']}" if edge["codex_proposed_fix"] else ""
    return (
        f"【{edge['review_id']}｜{source_name} → {relation} → {target_name}｜{decision}】{fix}  "
        f"原因：{edge['codex_pre_review_reason']}"
    )


def addition_zh(addition: dict[str, str]) -> str:
    proposal = addition["proposal"]
    warning_match = re.match(
        r"新增 Warning (\d+) 节点，并通过 WARNING_FOR 连接到文本中指定的 step："
        r"step '([^']+)' has issue '([^']+)'\.",
        proposal,
    )
    if not warning_match:
        return proposal
    warning_number, step, issue = warning_match.groups()
    step_zh = chinese_name(step, "Action")
    issue_zh = WARNING_ISSUE_ZH.get(issue, issue)
    return (
        f"新增警告{warning_number}节点“{issue_zh}”，并建立："
        f"警告{warning_number} → 警告对应于 → {step_zh}。"
    )


def build() -> None:
    scenes = read_jsonl(PRE_REVIEW)
    translations = {row["scene_id"]: row for row in read_jsonl(TRANSLATIONS)}
    if set(translations) != {row["scene_id"] for row in scenes}:
        raise ValueError("The translation set does not match the candidate scene set.")

    output: list[str] = [
        "# 20条工业 Gold Candidate 中文原文内嵌审核稿",
        "",
        "> 加粗部分是英文 source text 的中文翻译草稿；普通文字是节点、关系、建议和原因。英文原文仍是最终 evidence。",
        "",
        "反馈示例：`S07-N04 保留`、`S10-E04 改成 PART_OF`、`S19-A03 删除`。",
        "",
        "## 全局审核项",
        "",
        "- **G01**：第一轮正式 Gold 建议只评价 annotation-derived text；如果使用完整 transcript，必须继续补全所有事实。",
        "- **G02**：高层步骤包含内部动作时，使用 `PART_OF`，不要生成错误的 `BEFORE`。",
        "- **G03**：复合对象保持完整名称，例如 connector plates、wooden plate。",
        "- **G04**：S19/S20 需要扩展 Warning 节点和 `WARNING_FOR` 关系后才能成为完整 Gold。",
        "",
    ]

    for scene_index, scene in enumerate(scenes, 1):
        scene_code = f"S{scene_index:02d}"
        translation = translations[scene["scene_id"]]["source_text_zh"]
        nodes = scene["candidate_gold_nodes"]
        edges = scene["candidate_gold_edges"]
        node_by_id = {node["id"]: node for node in nodes}
        edges_by_source: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            edges_by_source.setdefault(edge["source"], []).append(edge)

        output.extend([
            f"# {scene_code}｜`{scene['scene_id']}`",
            "",
            f"类别：`{scene['category']}`",
            "",
            "## 加粗中文翻译",
            "",
        ])
        output.extend(bold_translation(translation))
        output.extend(["## 按动作原文直接标注", ""])

        action_nodes = [node for node in nodes if node["label"] == "Action"]
        other_nodes = [node for node in nodes if node["label"] != "Action"]
        for node in action_nodes:
            output.extend([
                f"**{evidence_zh(node['evidence_text'], node['name'], node['label'])}**",
                "",
                node_line(node),
                "",
            ])
            for edge in edges_by_source.get(node["id"], []):
                output.extend([edge_line(edge, node_by_id), ""])

        output.extend(["## 工具和对象节点", ""])
        for node in other_nodes:
            output.extend([
                f"**证据翻译：{evidence_zh(node['evidence_text'], node['name'], node['label'])}**",
                "",
                node_line(node),
                "",
            ])

        additions = scene.get("codex_proposed_additions", [])
        if additions:
            output.extend(["## 建议新增事实", ""])
            for addition in additions:
                output.extend([
                    f"【{addition['review_id']}｜建议新增】{addition_zh(addition)}  ",
                    f"原因：{addition['reason']}",
                    "",
                ])

        output.extend(["## 本场景反馈区", "", f"- `{scene_code}` 总体意见：", "- 需要修改的编号：", "", "---", ""])

    OUTPUT.write_text("\n".join(output), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
