#!/usr/bin/env python3
"""Build a numbered, non-final pre-review of industrial gold candidates.

The output is deliberately marked as a Codex proposal.  It never overwrites the
candidate templates and must not be used as final gold before human approval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/industrial_gold_candidates/industrial_gold_candidate_templates.jsonl"
OUTPUT_JSONL = ROOT / "data/industrial_gold_candidates/industrial_gold_pre_review_codex_2026-07-31.jsonl"
OUTPUT_MD = ROOT / "docs/industrial_gold_pre_review_codex_zh_2026-07-31.md"


NODE_OVERRIDES: dict[str, dict[str, str]] = {
    "S01-N10": {"decision": "reject", "reason": "tabletop 已经是明确目标；把 tabletop 再拆成 table 会产生重复且不精确的实体。"},
    "S01-N12": {"decision": "reject", "reason": "证据是 pick up screwdriver；screw 是从 screwdriver 错误截取出的子串。"},
    "S02-N09": {"decision": "fix", "fix": "名称改为 connector plates。", "reason": "复合名词 connector plates 不应只保留为泛化的 plate。"},
    "S02-N10": {"decision": "reject", "reason": "connector 是 connector plates 的错误拆词结果，与修正后的对象重复。"},
    "S02-N11": {"decision": "fix", "fix": "名称改为 wooden plate。", "reason": "wood 是从 wooden plate 截取出的错误对象名称。"},
    "S03-N12": {"decision": "fix", "fix": "名称改为 lower plate。", "reason": "原标注动作明确为 attach lower plate，泛化为 plate 会丢失区分信息。"},
    "S04-N12": {"decision": "fix", "fix": "名称改为 lower plate。", "reason": "原标注动作明确为 attach lower plate。"},
    "S05-N09": {"decision": "fix", "fix": "名称改为 side plates。", "reason": "避免把 side plates、connector plates 和 wooden plate 合并成同一个 plate。"},
    "S05-N10": {"decision": "fix", "fix": "名称改为 connector plates。", "reason": "connector 是复合对象 connector plates 的不完整名称。"},
    "S05-N11": {"decision": "fix", "fix": "名称改为 wooden plate。", "reason": "wood 是复合对象 wooden plate 的错误拆词结果。"},
    "S06-N10": {"decision": "fix", "fix": "名称改为 lower plate。", "reason": "动作标注明确指定 lower plate。"},
    "S07-N04": {"decision": "reject", "reason": "与 S07-N03 时间重叠且名称相同；当前没有 actor 信息可以证明是两个不同动作。"},
    "S07-N12": {"decision": "reject", "reason": "mark tabletop 的明确目标是 tabletop；额外 table 节点会造成重复。"},
    "S08-N11": {"decision": "fix", "fix": "名称改为 tool box。", "reason": "open tool box 的目标是复合对象 tool box，而不是泛化的 box。"},
    "S09-N09": {"decision": "fix", "fix": "名称改为 screws，并将 evidence 改为 transcript 中包含 screws 的原句。", "reason": "Action 2: screw in 只表达动作；物理螺钉需要使用 transcript 中的明确证据。"},
    "S10-N01": {"decision": "fix", "fix": "类型改为 Procedure，名称保留 preparation。", "reason": "该时间段包含 put on gloves 和 pick up tool，更像高层步骤而不是与子动作平行的原子 Action。"},
    "S10-N04": {"decision": "fix", "fix": "类型改为 Procedure，名称保留 disassemble object。", "reason": "该时间段包含多个 unscrew/remove/organize 子动作，应表示为高层 Procedure。"},
    "S10-N10": {"decision": "fix", "fix": "名称改为 screws，并使用 transcript 中 unscrewing the three screws here 作为证据。", "reason": "避免仅从动词 unscrew 推断物理对象。"},
    "S11-N10": {"decision": "fix", "fix": "名称改为 screws，并使用 transcript 中 unscrew these tiny screws 作为证据。", "reason": "需要用原文明确支持物理 screw 实体。"},
    "S12-N09": {"decision": "fix", "fix": "名称改为 lower plate。", "reason": "原标注动作明确为 remove lower plate。"},
    "S13-N01": {"decision": "fix", "fix": "类型改为 Procedure，名称改为 inspection procedure。", "reason": "该动作覆盖后续多个具体动作，与 Keystep 1 的 inspection 区间一致。"},
    "S19-N11": {"decision": "fix", "fix": "名称改为 screws。", "reason": "Warnings 明确出现 only one screw 和 screws loose，复数名称更符合证据。"},
    "S20-N11": {"decision": "fix", "fix": "名称改为 screws。", "reason": "Warnings 明确出现 screws loose 和 only one screw。"},
}


EDGE_OVERRIDES: dict[str, dict[str, str]] = {
    "S01-E04": {"decision": "reject", "reason": "remove tabletop 的区间包含 pick up screwdriver，不能标为 BEFORE。"},
    "S01-E09": {"decision": "reject", "reason": "目标 table 节点建议删除，tabletop 已经是准确目标。"},
    "S01-E10": {"decision": "reject", "reason": "pick up screwdriver 是拿起工具，不等于使用该工具完成该动作。"},
    "S01-E11": {"decision": "reject", "reason": "目标 screw 是从 screwdriver 错误拆出的节点。"},
    "S02-E09": {"decision": "reject", "reason": "connector 拆词节点建议删除。"},
    "S02-E10": {"decision": "reject", "reason": "attach wooden plate 不应连接到 connector plates；正确目标由 S02-E11 保留。"},
    "S02-E13": {"decision": "reject", "reason": "connector 拆词节点建议删除。"},
    "S02-E14": {"decision": "reject", "reason": "attach wooden plate 不应连接到 connector plates；正确目标由 S02-E15 保留。"},
    "S05-E09": {"decision": "reject", "reason": "attach connector plates 不应指向 side plates。"},
    "S05-E11": {"decision": "reject", "reason": "attach wooden plate 不应指向 side plates。"},
    "S07-E03": {"decision": "reject", "reason": "两个 put on gloves 动作时间重叠，不能标 BEFORE。"},
    "S07-E04": {"decision": "reject", "reason": "关系起点 S07-N04 建议作为重复节点删除。"},
    "S07-E06": {"decision": "reject", "reason": "pick up pen 与 mark tabletop 时间重叠，严格时间定义下不构成 BEFORE。"},
    "S07-E07": {"decision": "reject", "reason": "mark position 是 mark tabletop 区间内的子动作，不构成 BEFORE。"},
    "S07-E10": {"decision": "reject", "reason": "起点 S07-N04 建议删除。"},
    "S07-E12": {"decision": "reject", "reason": "目标 table 节点建议删除。"},
    "S08-E03": {"decision": "reject", "reason": "disconnect bar 位于 separate legs 的时间区间内，是嵌套动作而不是后续动作。"},
    "S08-E05": {"decision": "reject", "reason": "open tool box 位于 preparation 区间内，不能标为 preparation BEFORE open tool box。"},
    "S08-E11": {"decision": "reject", "reason": "pick up drilling machine 是拿起工具，不等于使用钻机完成该动作。"},
    "S09-E08": {"decision": "fix", "fix": "保留 ACTS_ON，但把目标名改为 screws，并换成 transcript 中明确包含 screws 的 evidence。", "reason": "候选动作合理，但当前 evidence 只写 screw in，不足以证明物理对象。"},
    "S10-E01": {"decision": "reject", "reason": "preparation 建议改为 Procedure，且时间上包含 put on gloves。"},
    "S10-E03": {"decision": "reject", "reason": "disassemble object 建议改为 Procedure，且与 pick up tool 时间重叠。"},
    "S10-E04": {"decision": "reject", "reason": "disassemble object 是包含 unscrew 的高层过程，不是其前序原子动作。"},
    "S10-E09": {"decision": "fix", "fix": "目标名改为 screws，使用 transcript 的明确 evidence。", "reason": "关系语义合理，但当前对象证据需要修正。"},
    "S10-E10": {"decision": "fix", "fix": "目标名改为 screws，使用 transcript 的明确 evidence。", "reason": "关系语义合理，但当前对象证据需要修正。"},
    "S11-E09": {"decision": "fix", "fix": "目标名改为 screws，使用 transcript 中明确的 screw evidence。", "reason": "避免仅从动作标签推断对象。"},
    "S13-E01": {"decision": "reject", "reason": "S13-N01 是覆盖后续动作的高层 inspection procedure，不是 action_2 的前序动作。"},
    "S19-E09": {"decision": "fix", "fix": "目标改为新增的 piece 对象；clamp piece 的动作目标是 piece。", "reason": "clamp 是工具/装置，不是 clamp piece 动作的主要被作用对象。"},
    "S20-E09": {"decision": "fix", "fix": "目标改为新增的 piece 对象；clamp piece 的动作目标是 piece。", "reason": "clamp 是工具/装置，不是 clamp piece 动作的主要被作用对象。"},
}


SCENE_NOTES: dict[int, list[str]] = {
    1: ["存在宏动作 remove tabletop 与内部子动作重叠；不能把列表顺序全部转换成 BEFORE。"],
    2: ["connector plates 和 wooden plate 被错误拆成多个泛化词，需要合并复合对象。"],
    3: ["建议新增 upper cover 对象及 action_5/action_7 的 ACTS_ON；当前候选只覆盖 lower plate。"],
    4: ["建议新增 upper cover 对象及 action_7 的 ACTS_ON。"],
    5: ["side plates、connector plates、wooden plate 必须保持为三个不同对象。"],
    6: ["建议新增 upper cover 对象并连接 action_3/action_6；当前节点不完整。"],
    7: ["存在重复手套动作和宏动作/子动作重叠；建议补 phone、pen 等明确对象后再接受。"],
    8: ["存在 separate legs/preparation 宏区间；建议补 bar、drilling bit 等明确对象。"],
    9: ["建议新增 Procedure=repair，并用 PART_OF 连接该区间内动作。"],
    10: ["建议把 preparation 和 disassemble object 转为 Procedure，再用 PART_OF 表达层级。"],
    11: ["建议新增 Procedure=disassemble object，并用 PART_OF 连接该区间内动作。"],
    12: ["建议补 upper cover 和 crank 对象；相邻区间存在 0.02 秒边界重叠，BEFORE 可按 metadata 容差保留。"],
    13: ["最外层 inspect 是高层过程；建议补 power connection、power button 等明确对象。"],
    14: ["动作顺序清楚，但 find/place missing parts 的具体对象未标明；避免把不同零件合并成一个泛化 part。"],
    15: ["建议补 devices 对象，并连接所有 find devices 动作；trolley 可作为 ACTS_ON 的 support 目标。"],
    16: ["建议补 devices 对象，并连接所有 find devices 动作。"],
    17: ["建议补 devices 对象，并连接所有 find devices 动作。"],
    18: ["建议补 devices 对象，并连接所有 find devices 动作。"],
    19: ["该样本的核心是12条 warning，但候选图完全没有 Warning 节点和 WARNING_FOR 边；在扩展 schema 前不能作为完整 Gold。"],
    20: ["该样本的核心是12条 warning，但候选图完全没有 Warning 节点和 WARNING_FOR 边；在扩展 schema 前不能作为完整 Gold。"],
}


ADDITIONS: dict[int, list[dict[str, str]]] = {
    3: [{"proposal": "新增 Object=upper cover；action_5/action_7 ACTS_ON upper cover。", "reason": "动作标签直接支持。"}],
    4: [{"proposal": "新增 Object=upper cover；action_7 ACTS_ON upper cover。", "reason": "动作标签直接支持。"}],
    6: [{"proposal": "新增 Object=upper cover；action_3/action_6 ACTS_ON upper cover。", "reason": "动作标签直接支持。"}],
    8: [{"proposal": "新增 Object=bar；action_4 ACTS_ON bar。", "reason": "disconnect bar 明确支持。"}, {"proposal": "新增 Tool=drilling bit。", "reason": "find drilling bit 明确支持该工具实体，但不等于已使用。"}],
    9: [{"proposal": "新增 Procedure=repair；action_1 至 action_8 PART_OF repair。", "reason": "Keystep 1 的 repair 时间区间覆盖这些动作。"}],
    10: [{"proposal": "action_2/action_3 PART_OF preparation；action_5 至 action_8 PART_OF disassemble object。", "reason": "时间区间和 keystep metadata 明确支持层级。"}],
    11: [{"proposal": "新增 Procedure=disassemble object；action_1 至 action_8 PART_OF 该 Procedure。", "reason": "Keystep 1 覆盖全部候选动作。"}],
    12: [{"proposal": "新增 Object=upper cover 和 Object=crank，并补对应 ACTS_ON。", "reason": "动作标签直接支持。"}],
    13: [{"proposal": "新增 Object=power connection、Object=power button，并补 plug/unplug/press 的 ACTS_ON。", "reason": "动作标签直接支持。"}],
    15: [{"proposal": "新增 Object=devices；action_2/action_5 ACTS_ON devices。", "reason": "find devices 动作标签直接支持。"}],
    16: [{"proposal": "新增 Object=devices；action_2/action_4/action_6/action_8 ACTS_ON devices。", "reason": "find devices 动作标签直接支持。"}],
    17: [{"proposal": "新增 Object=devices；action_2/action_5 ACTS_ON devices。", "reason": "find devices 动作标签直接支持。"}],
    18: [{"proposal": "新增 Object=devices；action_2/action_5 ACTS_ON devices。", "reason": "find devices 动作标签直接支持。"}],
    19: [{"proposal": "新增 Object=piece，并把 clamp/place 等动作连接到 piece。", "reason": "keystep 和 warning 文本明确出现 piece。"}],
    20: [{"proposal": "新增 Object=piece，并把 clamp/place 等动作连接到 piece。", "reason": "keystep 和 warning 文本明确出现 piece。"}],
}


def warning_additions(raw_text: str) -> list[dict[str, str]]:
    additions: list[dict[str, str]] = []
    for line in raw_text.splitlines():
        if not line.startswith("Warning "):
            continue
        prefix, payload = line.split(":", 1)
        additions.append(
            {
                "proposal": f"新增 {prefix} 节点，并通过 WARNING_FOR 连接到文本中指定的 step：{payload.strip()}",
                "reason": "这是 mistake_detection 样本的显式人工 warning 标注。",
            }
        )
    return additions


def decision_for(item_id: str, overrides: dict[str, dict[str, str]]) -> dict[str, str]:
    return overrides.get(item_id, {"decision": "accept", "reason": "候选事实由 action/keystep 标注或明确文本直接支持。"})


def build() -> None:
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    reviewed_rows: list[dict[str, Any]] = []
    md: list[str] = [
        "# Industrial Gold Candidate Codex 预审核稿",
        "",
        "> 本文件不是最终 Gold。它是基于当前候选文件和可见 source text 的第一轮建议，必须由人工确认。",
        "",
        "反馈格式示例：`S07-N04 保留`、`S10-E04 改为 PART_OF`、`S19-A03 删除`。",
        "",
        "编号规则：`Sxx-Nxx`=节点，`Sxx-Exx`=关系，`Sxx-Axx`=建议新增事实。",
        "",
        "## 全局问题",
        "",
        "- **G01** Gold 范围：原始 transcript 往往比候选图覆盖范围更大；若实验输入保留完整 transcript，必须继续补全 Gold，否则模型抽出的真实额外事实会被误算为 FP。",
        "- **G02** 动作层级：多个样本混合高层 keystep 与内部 action，不能简单按列表顺序生成全部 BEFORE。",
        "- **G03** 复合对象：connector plates、wooden plate 等复合对象被拆词，会直接破坏 Edge F1 的端点匹配。",
        "- **G04** Warning Schema：S19/S20 的核心 warning 信息完全没有进入当前图 Schema。",
        "",
    ]

    for scene_index, row in enumerate(rows, 1):
        scene_code = f"S{scene_index:02d}"
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for node_index, node in enumerate(row["candidate_gold_nodes"], 1):
            item_id = f"{scene_code}-N{node_index:02d}"
            proposal = decision_for(item_id, NODE_OVERRIDES)
            nodes.append({**node, "review_id": item_id, "codex_proposed_decision": proposal["decision"], "codex_proposed_fix": proposal.get("fix", ""), "codex_pre_review_reason": proposal["reason"]})
        for edge_index, edge in enumerate(row["candidate_gold_edges"], 1):
            item_id = f"{scene_code}-E{edge_index:02d}"
            proposal = decision_for(item_id, EDGE_OVERRIDES)
            edges.append({**edge, "review_id": item_id, "codex_proposed_decision": proposal["decision"], "codex_proposed_fix": proposal.get("fix", ""), "codex_pre_review_reason": proposal["reason"]})

        additions = list(ADDITIONS.get(scene_index, []))
        if scene_index in {19, 20}:
            additions.extend(warning_additions(row["raw_text"]))
        numbered_additions = [
            {"review_id": f"{scene_code}-A{idx:02d}", **addition}
            for idx, addition in enumerate(additions, 1)
        ]
        pre_status = "blocked_by_warning_schema" if scene_index in {19, 20} else "needs_human_confirmation"
        reviewed_rows.append(
            {
                **row,
                "candidate_gold_nodes": nodes,
                "candidate_gold_edges": edges,
                "codex_proposed_additions": numbered_additions,
                "codex_scene_pre_review_status": pre_status,
                "codex_scene_notes": SCENE_NOTES.get(scene_index, []),
                "human_review_status": "pending",
                "human_reviewer_note": "",
            }
        )

        md.extend([
            f"## {scene_code} - `{row['scene_id']}`",
            "",
            f"- 类别：`{row['category']}`",
            f"- 场景建议状态：`{pre_status}`",
        ])
        for note in SCENE_NOTES.get(scene_index, []):
            md.append(f"- 场景问题：{note}")
        md.extend(["", "### 节点", ""])
        for node in nodes:
            fix = f"；建议修正：{node['codex_proposed_fix']}" if node["codex_proposed_fix"] else ""
            md.append(
                f"- **{node['review_id']}** `{node['codex_proposed_decision']}` | "
                f"`{node['id']}` `{node['label']}` `{node['name']}`{fix}。理由：{node['codex_pre_review_reason']}"
            )
        md.extend(["", "### 关系", ""])
        for edge in edges:
            fix = f"；建议修正：{edge['codex_proposed_fix']}" if edge["codex_proposed_fix"] else ""
            md.append(
                f"- **{edge['review_id']}** `{edge['codex_proposed_decision']}` | "
                f"`{edge['source']} -[{edge['relation']}]-> {edge['target']}`{fix}。理由：{edge['codex_pre_review_reason']}"
            )
        if numbered_additions:
            md.extend(["", "### 建议新增或补充", ""])
            for addition in numbered_additions:
                md.append(f"- **{addition['review_id']}** `add` | {addition['proposal']} 理由：{addition['reason']}")
        md.append("")

    OUTPUT_JSONL.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in reviewed_rows) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSONL}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    build()
