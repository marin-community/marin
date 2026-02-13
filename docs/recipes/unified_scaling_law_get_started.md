# Get Started: Unified Model Scaling Law 分析

本文档说明如何为 **Unified Image-Text Model**（TASK_OVERVIEW.md 中的统一多模态模型）做 scaling law 分析，并如何与现有 `lib/marin/src/marin/scaling_laws` 和实验管道对接。

---

## 1. 前置条件（对齐 TASK_OVERVIEW 的 Milestones）

Scaling law 分析依赖「已完成训练且写出指标」的 run。在跑 unified 的 scaling 分析之前，需要先具备：

| Milestone | 内容 | 与 scaling 分析的关系 |
|-----------|------|------------------------|
| **M1** | LLM scaling 方法论（Chinchilla / IsoFLOP） | 已有：`isoflop_analysis.py`、`experiments/isoflop_sweep.py`、`exp2166_scaling_ladder_analysis.py` |
| **M2** | Eval 基础设施（text + image understanding + image generation） | 训练 run 需在 summary 里写出至少一个用于拟合的 eval 指标（如 L_text / L_U / L_G 或对应 key） |
| **M3** | 多模态数据 tokenization（text + TokLIP visual） | 已有起点：`experiments/unified/vlm_tokenize_captions.py`；需能产出可训练的 cache |
| **M4** | 稳定多模态训练管道 | 训练需写 `tracker_metrics.jsonl`，且 summary 含 `throughput/total_tokens`、`throughput/total_gflops`、`parameter_count` 及选定的 eval 指标 key |

**结论**：要「get started」做 unified 的 scaling law 分析，**最少**需要：
- 一条可跑通的多模态训练 pipeline（Levanter 或等价），且
- 该 pipeline 在结束时写入与现有 LLM isoflop 兼容的 metrics（见下节）。

---

## 2. 训练 Run 需要写出哪些指标

现有 scaling 分析从 **`tracker_metrics.jsonl`** 里读每条记录的 `config` 和 `summary`。
`experiments/isoflop_sweep.transform_levanter_metrics` 依赖的 **summary 字段**包括：

| 字段 | 含义 | 用于 |
|------|------|------|
| `throughput/total_tokens` | 该 run 总训练 token 数 | IsoFlopRecord.tokens |
| `throughput/total_gflops` | 总训练 FLOPs（GFLOPs） | 乘 1e9 后 round 到 bucket，IsoFlopRecord.flops |
| `parameter_count` | 模型参数量 | IsoFlopRecord.params |
| 一个 eval 指标 key（如 `eval/paloma/c4_en/bpb`） | 用于拟合 loss 的指标 | IsoFlopRecord.metric |

Unified 场景下你可以：

- **先做单指标分析**：选一个主指标（例如 L_text 对应的 eval key，或某个 understanding/generation 的 eval loss key），保证训练时把它写入 `summary`，然后复用现有 `fit_scaling_laws` 流程。
- **后续扩展**：TASK_OVERVIEW 的 RQ1/RQ2 需要 L_text、L_U、L_G 等多指标，可在同一套 `tracker_metrics.jsonl` 里写入多个 key，再在「transform」层按需选择或分别拟合。

**建议**：在 unified 训练里挂上与 Levanter `train_lm` 相同的 throughput / parameter 统计（如 `log_performance_stats` + `parameter_count` 写 summary），并至少写一个 eval 指标到 summary，这样就能直接复用 `read_eval_records` + 下面的 transform 步骤。

---

## 3. 具体步骤：从零到第一次 Unified Scaling 分析

### Step 1：确认训练输出与 run 命名

- 每个 unified 训练 run 的**输出目录**里要有 **`tracker_metrics.jsonl`**（可由 WandB backfill 生成，见 `eval_metrics_reader.read_eval_records`）。
- 若希望和现有 isoflop 工具链一致，run 的**目录名**建议带出「预算 + 模型规模 + batch + 实验名」，例如：
  `unified-{budget}-N{params}-B{batch}-{experiment_name}`
  这样可以用类似 `parse_isoflop_run_name` 的逻辑从路径解析出 `label`，用于分组拟合（见 Step 3）。

### Step 2：准备训练 run 路径列表

与现有 isoflop 完全一致：给 `read_eval_records` 一个「训练 run 输出路径」的列表（本地或 `gs://` 等 fsspec 路径均可）：

```python
from marin.scaling_laws.eval_metrics_reader import read_eval_records

training_runs = [
    "path/to/unified-run-1e18-...",
    "path/to/unified-run-3e18-...",
    # ...
]
raw_records = read_eval_records(training_runs=training_runs)
```

缺文件时会自动从 WandB backfill。

### Step 3：从 raw 转成 IsoFlopRecord（Unified 版 transform）

现有 `transform_levanter_metrics` 是为「纯文本 Levanter run」写的，依赖的 key 是 `throughput/total_tokens`、`throughput/total_gflops`、`parameter_count` 和可配置的 `metric_key`。

你要做的是**为 unified 写一个类似的 transform**（可放在 `experiments/unified/` 下），例如：

- 输入：`read_eval_records` 返回的 `raw_records`（每个元素是带 `config`、`summary`、`run_path` 的 dict）。
- 输出：`list[IsoFlopRecord]`，其中：
  - `tokens` / `flops` / `params`：从 `summary` 的上述 key 解析，`flops` 用 `round_flops_to_bucket(total_gflops * 1e9)`。
  - `metric`：从你选定的 eval key 读（例如 L_text 或某个 understanding/generation 的 loss key）。
  - `label`：从 run 路径或 config 解析（如实验名、r₁/r₂ 等），用于分组拟合。

若 unified 训练里 summary 的 key 与 Levanter 一致，可以**直接复用** `transform_levanter_metrics`，只改 `metric_key` 和（可选）run 命名解析；否则写一个 `transform_unified_metrics(raw_records, metric_key=..., label_map=...)` 即可。

### Step 4：调用现有 scaling 分析

一旦有了 `list[IsoFlopRecord]`，后面与文本 isoflop 完全一致：

```python
from marin.scaling_laws import fit_scaling_laws

records = transform_unified_metrics(raw_records, metric_key="eval/...", label_map=...)
result = fit_scaling_laws(records)
# result.minima_records, result.scaling_fits, result.fit_curves
```

可把结果写成 `isoflop_analysis_result.json`（参考 `run_isoflop_analysis_step` 里的格式），便于后续画图和预测最优配置。

### Step 5（可选）：Unified 的 ScalingRecipe 与 predict_optimal_config

若要像 exp2166 那样「用拟合结果预测最优训练配置」并启动训练，需要：

- 一个实现 **`ScalingRecipe`** 的类（例如 `Unified2025Recipe`），提供：
  - `vocab_size`（Qwen3 + TokLIP 的 unified vocab 大小，见 TASK_OVERVIEW）
  - `candidates_for_budget(budget, seq_len)`：对给定 FLOP 预算枚举候选 (N, batch, steps, lr, ...)，返回 `CandidateConfig` 列表
  - `estimate_memory_bytes(candidate)`：用于 TPU 选型（如 `pick_v5p_type`）
- 若使用现有 Qwen 等模型配置，需实现 **`ModelConfiguration`**（`flops_per_token`、`total_trainable_params`）；unified 的 `flops_per_token` 需把 **visual token** 的 FLOP 算进去（与 seq_len / 每图 token 数一致）。

实现后即可：

```python
from marin.scaling_laws import predict_optimal_config

candidate = predict_optimal_config(
    scaling_fits=result.scaling_fits,
    target_flops=1e20,
    label="your-unified-label",
    recipe=Unified2025Recipe(...),
    seq_len=4096,
)
```

再用 `candidate` 驱动训练配置（参考 `run_optimal_training`）。

### Step 6（可选）：画图与上传 WandB

与文本 isoflop 相同：用 `result` 的 `minima_records`、`fit_curves`、`scaling_fits` 构建 DataFrame 和图表，然后调用：

- `create_isoflop_plot` / `create_scaling_plot`
- `save_plots` / `upload_plots_to_wandb`

详见 `lib/marin/src/marin/scaling_laws/scaling_plots.py`。

---

## 4. 建议的文件与入口

| 内容 | 建议位置 |
|------|----------|
| Unified 的 run 名解析 + transform（raw → IsoFlopRecord） | `experiments/unified/isoflop_metrics.py`（或类似名字） |
| Unified ScalingRecipe + ModelConfiguration | `experiments/unified/recipe.py` 或扩展现有 `experiments/isoflop_sweep.py` 中的 recipe |
| 分析 step（读 run → transform → fit_scaling_laws → 写结果） | `experiments/unified/run_unified_isoflop_analysis.py`，内部调 `run_isoflop_analysis_step` 或仿写一个用 `transform_unified_metrics` 的版本 |
| 训练 step（unified 多模态训练，写 tracker_metrics.jsonl） | 随 M4 在 `experiments/unified/` 下实现，保证 summary 含上述 key |

---

## 5. 与 TASK_OVERVIEW 研究问题的对应

- **RQ1（Multimodal Tax）**：先做 text-only 的 D*_text、N*，再在固定 D*_text 下加 visual，比较 L_text；需要多组 run 和可能的多个 metric key（L_text / L_U / L_G），transform 里按实验组设置 `label` 即可。
- **RQ2（Data Mixture）**：r₁、r₂ 等作为 `label` 或额外维度，同一套 `fit_scaling_laws` 按 label 分组拟合；若要做「最优 r₁*」可在上层对多组 scaling_fits 做比较。
- **RQ5（Loss–Benchmark 相关性）**：在 summary 里同时记录 eval loss 与 benchmark 分数，分析时从 raw_records 里取出，单独做相关性分析，不改变现有 scaling 拟合流程。

---

## 6. 小结：最小可行路径

1. **确保**：有一条 unified 训练 pipeline，且写出 `tracker_metrics.jsonl`，summary 含 `throughput/total_tokens`、`throughput/total_gflops`、`parameter_count` 和至少一个 eval 指标 key。
2. **实现**：Unified 的 run 名解析 + `transform_unified_metrics`（或复用 `transform_levanter_metrics` 并统一 key）。
3. **运行**：`read_eval_records` → transform → `fit_scaling_laws` → 写 `isoflop_analysis_result.json`（可封装成一步 `run_unified_isoflop_analysis_step`）。
4. **扩展**：需要「预测最优配置并启动训练」时，实现 Unified 的 `ScalingRecipe`（及可选 `ModelConfiguration`），再用 `predict_optimal_config` + 现有 TPU/训练入口。

参考实现：**`experiments/isoflop_sweep.run_isoflop_analysis_step`** 和 **`experiments/exp2166_scaling_ladder_analysis.py`**。
