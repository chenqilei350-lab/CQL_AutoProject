# 医院黄金案例实验复现说明

这个目录里有两个可运行脚本：

- `scripts/run_hospital_gold_pilot_experiments_zh.py`：中文注释版完整实现。
- `scripts/run_hospital_gold_pilot_experiments_en.py`：英文注释版入口，复用同一套实现，保证结果一致。

## 1. 运行前准备

确认你在项目根目录：

```bash
cd /Users/qileichen/Desktop/aut-kg-extraction-pipeline
```

确认 gold case 数据目录存在：

```bash
ls gold_case_admission_29366372
```

确认 Ollama 已安装并启动：

```bash
ollama serve
```

如果已经有 Ollama 后台服务在跑，这条命令可能提示端口已占用，可以不用重复启动。

确认模型已下载：

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull llama3.2:latest
```

如果只想跑主要实验，至少需要：

```bash
ollama pull llama3.1:8b
```

## 2. 最推荐的快速运行方式

只跑 P1/P3/P4 stress 实验：

```bash
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite stress
```

这个版本最适合给组员演示，因为它能明显展示：

- schema constraint（模式约束）的作用；
- chunking（分块）对可运行性的影响；
- unified text structure（统一文本结构）的作用。

## 3. 跑完整实验

```bash
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite all
```

完整实验包括：

| Suite | 内容 |
| --- | --- |
| P1 | 最小严格抽取 sanity check |
| P1 Stress | loose prompt vs strict schema prompt |
| P2 | one-shot full graph vs layered extraction |
| P3 | LabEvent chunk size |
| P3 Stress | dense narrative 下的 chunk size |
| P4 | raw vs unified LabEvent input |
| P4 Stress | raw noisy text vs unified canonical structure |
| P5 | 不同小模型对比 |
| P6 | repeated-run stability |

注意：`--suite all` 会比较慢，因为 P2 one-shot 和 P3 stress 大块输入可能会等到 timeout。

## 4. 只跑某一个实验

```bash
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite p1
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite p2
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite p3
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite p4
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite p5
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite p6
```

## 5. 使用英文注释版入口

英文版入口命令完全一样：

```bash
python3 scripts/run_hospital_gold_pilot_experiments_en.py --suite stress
```

英文入口会调用同一个实验实现，因此中英文入口产生的结果一致。

## 6. 输出文件在哪里

默认输出到：

```text
results/hospital_gold_repro_<timestamp>/
```

里面会包含：

| 文件 | 用途 |
| --- | --- |
| `all_results.csv` | 所有实验的总表 |
| `summary.md` | Markdown 汇总 |
| `p1_*.csv` | P1 相关数据 |
| `p2_*.csv` | P2 相关数据 |
| `p3_*.csv` | P3 相关数据 |
| `p4_*.csv` | P4 相关数据 |
| `p5_*.csv` | P5 模型对比 |
| `p6_*.csv` | P6 稳定性 |
| `*.svg` | 如果本机安装了 matplotlib，会自动生成核心图 |

指定输出目录：

```bash
python3 scripts/run_hospital_gold_pilot_experiments_zh.py --suite stress --output-dir results/my_stress_run
```

## 7. 常用参数

```bash
python3 scripts/run_hospital_gold_pilot_experiments_zh.py \
  --suite stress \
  --model llama3.1:8b \
  --timeout 120 \
  --output-dir results/stress_demo
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--suite` | 选择运行 all/base/stress/p1/p2/p3/p4/p5/p6 |
| `--model` | P1/P2/P3/P4 使用的主模型 |
| `--models` | P5/P6 使用的模型列表，用逗号分隔 |
| `--timeout` | 单次 Ollama 调用的超时时间 |
| `--repetitions` | P6 重复运行次数 |
| `--gold-dir` | gold case 数据目录 |
| `--output-dir` | 输出目录 |

## 8. 结果解释注意事项

这些实验分两类：

| 类型 | 解释 |
| --- | --- |
| base experiments | 基础 pilot 实验，用于验证流程 |
| stress experiments | 压力实验，用于暴露变量影响和失败模式 |

论文或组会中应避免说：

> unified text structure always improves KG generation.

更稳妥的说法是：

> In stress-test conditions with noisy input and implicit IDs, unified text structure improved schema-conformant graph extraction by making canonical IDs, labels, and relation directions explicit.

中文：

> 在带噪声且 ID 不显式的压力测试条件下，统一文本结构通过显式给出规范 ID、实体类型和关系方向，提高了符合 schema 的知识图谱抽取质量。

