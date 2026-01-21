# Levanter LLM Evaluation 详解

本文档详细介绍 Levanter 框架中语言模型评估（LLM Evaluation）的实现机制。

## 目录

1. [概述](#1-概述)
2. [核心模块架构](#2-核心模块架构)
3. [评估流程详解](#3-评估流程详解)
4. [评估指标](#4-评估指标)
5. [Tagged Evaluation（多数据集标注评估）](#5-tagged-evaluation多数据集标注评估)
6. [LM Eval Harness 集成](#6-lm-eval-harness-集成)
7. [配置详解](#7-配置详解)
8. [使用示例](#8-使用示例)
9. [关键代码路径](#9-关键代码路径)

---

## 1. 概述

Levanter 提供了两种主要的评估模式：

### 1.1 训练期间评估（Training-time Evaluation）

在模型训练过程中，通过回调函数（callback）定期触发评估。这种方式可以：
- 监控训练进度
- 跟踪 loss 和 perplexity 的变化
- 支持 EMA（指数移动平均）模型评估

### 1.2 独立评估（Standalone Evaluation）

使用独立脚本对已保存的模型进行评估，支持：
- Levanter 检查点
- HuggingFace 检查点
- 多种评估指标和任务

### 1.3 评估类型

| 评估类型 | 主要用途 | 核心模块 |
|---------|---------|---------|
| Loss/Perplexity 评估 | 基础语言模型能力评估 | `eval.py` |
| LM Eval Harness | 标准 benchmark 评估 | `eval_harness.py` |
| 多数据集标注评估 | 按 domain 聚合统计 | `DomainTaggedDataset` |

---

## 2. 核心模块架构

### 2.1 模块依赖关系

```
┌─────────────────────────────────────────────────────────┐
│                    main/eval_lm.py                       │
│                   (独立评估脚本入口)                       │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌───────────┐ ┌────────────────────┐
│    eval.py      │ │ metrics.py│ │  eval_harness.py   │
│ (核心评估引擎)   │ │ (指标管理) │ │ (LM Eval Harness)  │
└────────┬────────┘ └───────────┘ └─────────┬──────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐              ┌────────────────────┐
│ TaggedEvaluator │              │ LevanterHarnessLM  │
│ DomainTaggedDS  │              │ _LmEvalHarnessWork │
└────────┬────────┘              └─────────┬──────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
              ┌─────────────────┐
              │   LmHeadModel   │
              │   (模型推理)     │
              └─────────────────┘
```

### 2.2 核心文件说明

| 文件路径 | 功能描述 |
|---------|---------|
| `lib/levanter/src/levanter/eval.py` | 核心评估引擎，包含 `TaggedEvaluator`、`EvalResult`、`DomainTaggedDataset` |
| `lib/levanter/src/levanter/eval_harness.py` | EleutherAI LM Eval Harness 集成，包含 `LevanterHarnessLM`、`LmEvalHarnessConfig` |
| `lib/levanter/src/levanter/main/eval_lm.py` | 独立评估脚本入口，定义 `EvalLmConfig` |
| `lib/levanter/src/levanter/metrics.py` | 指标管理系统，定义 `ReductionType` |
| `lib/levanter/src/levanter/callbacks/__init__.py` | 回调框架，注册评估回调 |

---

## 3. 评估流程详解

### 3.1 训练期间评估流程

训练期间评估通过 `cb_tagged_lm_evaluate` 回调函数实现：

```
┌──────────────────────────────────────────────────────────┐
│              cb_tagged_lm_evaluate (回调创建)              │
│                                                          │
│  参数:                                                    │
│  - EvalBatch: 评估批次大小                                │
│  - tagged_eval_sets: 带标签的评估数据集                    │
│  - tokenizer: 分词器（用于 BPB 计算）                      │
│  - eval_current: 是否评估当前模型                          │
│  - eval_ema: 是否评估 EMA 模型                            │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              TaggedEvaluator 初始化                        │
│                                                          │
│  1. 创建 DomainTaggedDataset（合并多个数据集）              │
│  2. 创建 DataLoader                                       │
│  3. 计算 bytes_per_token（用于 BPB）                       │
│  4. 构建标签层级结构（hierarchy）                           │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              eval_callback 执行（每个评估周期）              │
│                                                          │
│  1. 获取当前 step                                         │
│  2. 如果 eval_current: 评估当前模型参数                    │
│  3. 如果 eval_ema: 评估 EMA 模型                          │
│  4. 通过 levanter.tracker.log 记录结果                    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              TaggedEvaluator.evaluate(model)              │
│                                                          │
│  for batch, tags in DataLoader:                          │
│      1. 计算 per-token loss                               │
│      2. 按 tag 维度聚合 loss 和 tokens                    │
│      3. 更新 RunningMean 统计                             │
│      4. 计算 BPB（如果有 tokenizer）                       │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              EvalResult 返回                               │
│                                                          │
│  - micro_avg_loss: 所有 tokens 的平均 loss                │
│  - macro_avg_loss: 各 domain 平均 loss 的平均              │
│  - tag_macro_losses: 各 tag 的 macro loss                 │
│  - tag_micro_losses: 各 tag 的 micro loss                 │
│  - micro_bpb / macro_bpb: bits-per-byte 指标              │
└──────────────────────────────────────────────────────────┘
```

### 3.2 独立评估流程

使用 `eval_lm.py` 脚本进行独立评估：

```python
# 执行命令示例
python -m levanter.main.eval_lm \
    --config_path config.yaml \
    --checkpoint_path /path/to/checkpoint
```

流程步骤：

1. **初始化配置**：加载 `EvalLmConfig`
2. **加载数据集**：根据 `eval_on_train` 选择训练集或验证集
3. **加载模型**：支持 Levanter 检查点或 HuggingFace 检查点
4. **创建 TaggedEvaluator**：初始化评估器
5. **执行评估**：调用 `eval_model()` 函数
6. **记录结果**：通过 tracker 记录到 WandB 等平台

---

## 4. 评估指标

### 4.1 基础指标

#### Loss（交叉熵损失）

```python
# 位置: eval.py 第 339 行
losses = compute_next_token_loss(m, batch, reduction=None, reduction_axis=())
```

- **计算方式**：对每个 token 计算交叉熵损失
- **用途**：评估模型预测下一个 token 的能力

#### Perplexity（困惑度）

```python
perplexity = exp(loss)
```

- **计算方式**：loss 的指数
- **用途**：衡量模型对文本的建模质量，值越低越好

### 4.2 高级指标

#### Bits-Per-Byte (BPB)

BPB 是一种字节级别的评估指标，消除了不同分词器之间的差异：

```python
# 位置: eval.py 第 366-368 行
# log loss -> bits 转换公式: log2(e) * loss
bpb_per_tag = this_loss_per_tag / hax.maximum(bytes_per_tag, 1) * jnp.log2(jnp.e)
bpb = this_loss / hax.maximum(this_bytes, 1) * jnp.log2(jnp.e)
```

**byte_length_of_token 计算**（位置: `utils/hf_utils.py`）：

```python
def byte_length_of_token(tokenizer, idx: int) -> int:
    """计算 token 的 UTF-8 字节长度"""
    # 1. 特殊 tokens: 返回 0
    # 2. 十六进制格式 <0xHH>: 直接转换
    # 3. 普通 tokens: 通过 decode 获取字节长度
```

#### Micro 与 Macro 平均

| 类型 | 计算方式 | 特点 |
|------|---------|------|
| **Micro Average** | 所有 tokens 加权平均 | 反映整体性能 |
| **Macro Average** | 各 domain 平均的平均 | 对小数据集更公平 |

```python
# Micro: 按 token 数量加权
micro_avg_loss = total_loss / total_tokens

# Macro: 各 domain 平均后再平均
macro_avg_loss = mean([loss_domain1, loss_domain2, ...])
```

### 4.3 指标聚合策略（ReductionType）

```python
# 位置: metrics.py
class ReductionType(Enum):
    MEAN = "mean"      # 求平均值 (loss, accuracy)
    SUM = "sum"        # 累加 (counts, totals)
    MAX = "max"        # 最大值
    MIN = "min"        # 最小值
    LAST = "last"      # 保留最后值 (learning_rate)
```

---

## 5. Tagged Evaluation（多数据集标注评估）

### 5.1 DomainTaggedDataset 类

`DomainTaggedDataset` 用于管理多个带标签的数据集，支持按 domain 聚合统计：

```python
# 位置: eval.py 第 54-157 行
class DomainTaggedDataset(AsyncDataset[tuple[T, hax.NamedArray]]):
    """
    持有多个数据集，每个都有自己的 domain tag。
    同时索引标签以便于聚合。
    """

    def __init__(
        self,
        datasets: Sequence[tuple[AsyncDataset[T], Sequence[str]]],
        max_examples_per_dataset: Optional[int] = None
    ):
        # datasets: [(dataset1, ["domain1", "domain1/subdomain"]), ...]
        # tag_to_index: {"domain1": 0, "domain1/subdomain": 1}
        # Tag axis: hax.Axis("tag", len(tag_index))
```

### 5.2 分层标签支持

标签支持层级结构，使用 "/" 作为分隔符：

```python
# 示例标签结构
tags = [
    "wikipedia",
    "wikipedia/en",
    "wikipedia/zh",
    "books",
    "books/fiction",
    "books/nonfiction"
]

# 层级聚合示例
# "wikipedia" 的 macro loss = mean(loss_wikipedia_en, loss_wikipedia_zh)
# "wikipedia" 的 micro loss = weighted_mean(loss_wikipedia_en, loss_wikipedia_zh)
```

### 5.3 标签层级处理

```python
# 位置: eval.py 第 313-323 行
hierarchy: dict[str, list[int]] = {}
for tag, index in self.dataset.tag_to_index.items():
    parts = tag.split("/")
    for i in range(1, len(parts)):
        parent = "/".join(parts[:i])
        if parent not in hierarchy:
            hierarchy[parent] = []
        hierarchy[parent].append(index)
```

---

## 6. LM Eval Harness 集成

### 6.1 架构设计

Levanter 通过 `LevanterHarnessLM` 类集成 EleutherAI 的 LM Evaluation Harness：

```
┌─────────────────────────────────────────────────────────┐
│              run_lm_eval_harness()                       │
│                                                         │
│  1. 构建 task dictionary                                 │
│  2. 创建 _LmEvalHarnessWorker                            │
│  3. 执行评估                                             │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
┌──────────────────┐        ┌──────────────────────┐
│   Process 0      │        │   Process 1..N       │
│  (主进程)         │        │   (工作进程)          │
│                  │        │                      │
│  LevanterHarnessLM        │  worker_message_loop │
│  运行 Harness    │◄──────►│  等待请求            │
└──────────────────┘        └──────────────────────┘
```

### 6.2 分布式执行机制

由于 LM Eval Harness 不是确定性的，Levanter 采用主从模式：

```python
# 位置: eval_harness.py 第 146-296 行
class _LmEvalHarnessWorker:
    """
    工作进程类：
    - 进程 0: 运行主 Harness，分发请求
    - 其他进程: 运行消息循环，等待 loglikelihood 请求
    """

    def worker_message_loop(self):
        """工作进程消息循环"""
        while True:
            message = self._receive_message()
            if message == _Message.STOP:
                return
            elif message == _Message.LOGLIKELIHOOD:
                payload = self._receive_payload()
                self.process_loglikelihood(payload)
```

### 6.3 LevanterHarnessLM 实现

```python
# 位置: eval_harness.py 第 335-856 行
class LevanterHarnessLM(TemplateLM):
    """Levanter 实现的 LM Eval Harness 接口"""

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """计算 log-likelihood"""
        # 1. 打包请求
        # 2. 分批处理
        # 3. 返回 (log_prob, is_correct) 对

    def generate_until(self, requests) -> List[str]:
        """文本生成（使用 InferenceEngine）"""
        # 注意: 目前仅支持单主机
```

### 6.4 支持的任务类型

| 任务类型 | 方法 | 说明 |
|---------|------|------|
| loglikelihood | `loglikelihood()` | 计算补全的 log 概率 |
| generate_until | `generate_until()` | 生成文本直到停止条件 |
| loglikelihood_rolling | 未实现 | 滚动窗口 log-likelihood |

### 6.5 Loglikelihood 计算详解

```python
# 位置: eval_harness.py 第 180-227 行
def _eval_loglikelihood(model: LmHeadModel, packed_example: LmExample):
    """计算 log-likelihood"""

    # 1. 前向传播获取 logits
    logits = model(packed_example.tokens, attn_mask=packed_example.attn_mask)

    # 2. 计算 per-token loss
    loss = next_token_loss(
        Pos=Pos, Vocab=model.Vocab,
        logits=logits,
        true_ids=packed_example.tokens,
        loss_weight=packed_example.loss_weight,
        reduction=None,
    )

    # 3. 计算贪心预测正确性
    pred_targets = hax.argmax(logits, axis=model.Vocab)
    targets = hax.roll(packed_example.tokens, -1, axis=Pos)
    is_correct = targets == pred_targets

    # 4. 按 segment 提取结果
    return segments, -per_segment_loss, per_segment_correct
```

---

## 7. 配置详解

### 7.1 EvalLmConfig（独立评估配置）

```python
# 位置: main/eval_lm.py 第 34-50 行
@dataclass
class EvalLmConfig:
    # 检查点配置
    checkpoint_path: Optional[str] = None      # Levanter 检查点路径
    hf_checkpoint: Optional[RepoRef] = None    # HuggingFace 检查点

    # 训练器配置
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # 数据配置
    data: SingleDatasetLMConfigBase | LMMixtureDatasetConfig = field(...)

    # 评估配置
    max_eval_length: int = 2048    # 最大序列长度
    model: LmConfig = field(default_factory=LlamaConfig)
    eval_on_train: bool = False    # 是否在训练集评估

    # 额外分析选项
    log_entropy: bool = False      # 是否计算熵
    log_top2_gap: bool = False     # 是否计算 top2 gap
    log_param_stats: bool = False  # 是否计算参数统计
```

### 7.2 LmEvalHarnessConfig（Harness 评估配置）

```python
# 位置: eval_harness.py 第 955-1001 行
@dataclass(frozen=True)
class LmEvalHarnessConfig:
    # 任务配置
    task_spec: list[TaskConfig | str]    # 评估任务列表

    # 评估参数
    max_examples: int | None = None      # 最大样本数
    max_length: int | None = None        # 最大序列长度
    bootstrap_iters: int = 0             # Bootstrap 迭代次数

    # 模板配置
    apply_chat_template: bool = False    # 是否应用 chat 模板
    fewshot_as_multiturn: bool = False   # Few-shot 作为多轮对话

    # 日志配置
    log_samples: bool = False            # 记录样本
    sample_logging: SampleLoggingConfig  # 样本日志配置

    # 生成参数
    generation_kwargs: dict = field(
        default_factory=lambda: {
            "max_gen_toks": 256,
            "temperature": 0.0,
            "n": 1,
            "seed": None
        }
    )
```

### 7.3 TaskConfig（任务配置）

```python
# 位置: eval_harness.py 第 900-952 行
@dataclass(frozen=True)
class TaskConfig:
    task: str                           # 任务名称 (e.g., "mmlu")
    task_alias: str | None = None       # 任务别名
    num_fewshot: int | None = None      # Few-shot 示例数

    # Jinja2 模板配置
    use_prompt: str | None = None       # PromptSource 提示名称
    description: str | None = None      # 任务描述
    doc_to_text: str | None = None      # 样本转文本模板
    doc_to_choice: str | None = None    # 样本转选项模板

    # 分隔符配置
    target_delimiter: str | None = None    # 输入输出分隔符
    fewshot_delimiter: str | None = None   # Few-shot 示例分隔符
```

---

## 8. 使用示例

### 8.1 训练期间评估

```python
from levanter.eval import cb_tagged_lm_evaluate
from levanter.data.text import LMMixtureDatasetConfig

# 准备数据集
data_config = LMMixtureDatasetConfig(...)
tagged_eval_sets = data_config.tagged_eval_sets(Pos)

# 创建评估回调
eval_callback = cb_tagged_lm_evaluate(
    EvalBatch=trainer.EvalBatch,
    tagged_eval_sets=tagged_eval_sets,
    tokenizer=tokenizer,
    device_mesh=trainer.mesh,
    axis_mapping=trainer.compute_axis_mapping,
    eval_current=True,
    eval_ema=True,
    prefix="eval",
)

# 在训练器中注册回调
trainer.add_callback(eval_callback, every=eval_interval)
```

### 8.2 独立脚本评估

```yaml
# config.yaml
checkpoint_path: /path/to/checkpoint
max_eval_length: 2048
eval_on_train: false

model:
  type: llama
  hidden_dim: 4096
  num_heads: 32
  num_layers: 32

data:
  tokenizer: "meta-llama/Llama-2-7b-hf"
  validation_urls:
    - "gs://bucket/validation/*.jsonl"

trainer:
  eval_batch_size: 8
```

```bash
# 执行评估
python -m levanter.main.eval_lm --config_path config.yaml
```

### 8.3 LM Eval Harness 评估

```python
from levanter.eval_harness import (
    LmEvalHarnessConfig,
    TaskConfig,
    run_lm_eval_harness,
)

# 配置任务
config = LmEvalHarnessConfig(
    task_spec=[
        TaskConfig(task="hellaswag", num_fewshot=0, task_alias="hellaswag_0shot"),
        TaskConfig(task="hellaswag", num_fewshot=10, task_alias="hellaswag_10shot"),
        TaskConfig(task="mmlu", num_fewshot=5, task_alias="mmlu_5shot"),
        "arc_easy",  # 也可以直接使用字符串
    ],
    max_examples=1000,  # 限制样本数（用于调试）
    apply_chat_template=False,
)

# 运行评估
results = run_lm_eval_harness(
    config=config,
    model=model,
    tokenizer=tokenizer,
    EvalBatch=Batch,
    axis_resources=compute_axis_mapping,
    mp=trainer.mp,
)

# 结果包含:
# - results["results"]: 各任务的详细结果
# - results["averages"]: macro 和 micro 平均
```

### 8.4 常用评估任务示例

```python
# 定义常用任务配置
CORE_TASKS = [
    TaskConfig("hellaswag", num_fewshot=0, task_alias="hellaswag_0shot"),
    TaskConfig("hellaswag", num_fewshot=10, task_alias="hellaswag_10shot"),
    TaskConfig("mmlu", num_fewshot=0, task_alias="mmlu_0shot"),
    TaskConfig("mmlu", num_fewshot=5, task_alias="mmlu_5shot"),
    TaskConfig("arc_easy", num_fewshot=10),
    TaskConfig("arc_challenge", num_fewshot=10),
    TaskConfig("boolq", num_fewshot=10),
    TaskConfig("winogrande", num_fewshot=5),
]
```

---

## 9. 关键代码路径

### 9.1 文件路径汇总

| 功能模块 | 文件路径 |
|---------|---------|
| 核心评估引擎 | `lib/levanter/src/levanter/eval.py` |
| LM Eval Harness 集成 | `lib/levanter/src/levanter/eval_harness.py` |
| 独立评估脚本 | `lib/levanter/src/levanter/main/eval_lm.py` |
| 指标管理系统 | `lib/levanter/src/levanter/metrics.py` |
| 回调框架 | `lib/levanter/src/levanter/callbacks/__init__.py` |
| 统计工具 | `lib/levanter/src/levanter/utils/stat_utils.py` |
| 字节计算工具 | `lib/levanter/src/levanter/utils/hf_utils.py` |
| 数据配置 | `lib/levanter/src/levanter/data/text.py` |

### 9.2 关键函数索引

| 函数/类 | 位置 | 功能 |
|--------|------|------|
| `cb_tagged_lm_evaluate()` | eval.py:165 | 创建训练期间评估回调 |
| `eval_model()` | eval.py:227 | 执行单次模型评估 |
| `TaggedEvaluator` | eval.py:277 | 多数据集标注评估器 |
| `TaggedEvaluator.evaluate()` | eval.py:380 | 执行完整评估流程 |
| `DomainTaggedDataset` | eval.py:54 | 多数据集管理 |
| `EvalResult` | eval.py:38 | 评估结果数据类 |
| `run_lm_eval_harness()` | eval_harness.py:1157 | 运行 LM Eval Harness |
| `LevanterHarnessLM` | eval_harness.py:335 | Harness LM 适配器 |
| `LevanterHarnessLM.loglikelihood()` | eval_harness.py:500 | 计算 log-likelihood |
| `_LmEvalHarnessWorker` | eval_harness.py:146 | 分布式工作进程 |
| `LmEvalHarnessConfig` | eval_harness.py:955 | Harness 配置类 |
| `TaskConfig` | eval_harness.py:900 | 任务配置类 |

### 9.3 数据结构

#### EvalResult

```python
@dataclass
class EvalResult:
    micro_avg_loss: float              # 所有 tokens 的平均 loss
    macro_avg_loss: float              # domain 级别的平均 loss
    tag_macro_losses: dict[str, float] # 各 tag 的 macro loss
    tag_micro_losses: dict[str, float] # 各 tag 的 micro loss
    total_eval_loading_time: float     # 数据加载时间
    micro_bpb: Optional[float]         # 微观 bits-per-byte
    macro_bpb: Optional[float]         # 宏观 bits-per-byte
    tag_macro_bpb: Optional[dict]      # tag 级 BPB
    tag_micro_bpb: Optional[dict]      # tag 级 BPB
```

#### RunningMean

```python
# 位置: utils/stat_utils.py
class RunningMean(eqx.Module):
    mean: Arrayish       # 当前 mean 估计
    total: Arrayish      # 总权重

    def add(self, x, total):
        """增量更新平均值（Welford 算法）"""
        delta = x - self.mean
        new_total = self.total + total
        ratio = total / new_total
        new_mean = self.mean + delta * ratio
        return RunningMean(new_mean, new_total)
```

---

## 附录：常见问题

### Q1: 如何添加自定义评估任务？

可以通过 `TaskConfig` 定义自定义任务配置，或直接使用 LM Eval Harness 支持的任务名称。

### Q2: BPB 和 Perplexity 有什么区别？

- **Perplexity**: 基于 token 级别的 loss，受分词器影响
- **BPB**: 基于字节级别，不受分词器影响，更适合跨模型比较

### Q3: Micro 和 Macro 平均何时使用？

- **Micro**: 适合评估整体性能
- **Macro**: 适合评估在不同 domain 上的均衡性能

### Q4: 如何在多 GPU/TPU 上运行评估？

Levanter 自动处理分布式评估，只需配置正确的 `trainer.device_mesh` 即可。
