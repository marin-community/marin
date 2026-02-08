# Technical Report: Best Practices for LoRA-Based DPO Fine-Tuning

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: How DPO with LoRA Works](#2-background-how-dpo-with-lora-works)
3. [Optimal LoRA Hyperparameters for DPO](#3-optimal-lora-hyperparameters-for-dpo)
4. [DPO-Specific Hyperparameters](#4-dpo-specific-hyperparameters)
5. [Architecture Decisions: Which Layers and Modules](#5-architecture-decisions-which-layers-and-modules)
6. [Reference Model Handling Strategies](#6-reference-model-handling-strategies)
7. [Loss Function Selection](#7-loss-function-selection)
8. [Training Stability and Common Failure Modes](#8-training-stability-and-common-failure-modes)
9. [Rank-Stabilized LoRA (rsLoRA) for DPO](#9-rank-stabilized-lora-rslora-for-dpo)
10. [End-to-End Configuration Reference](#10-end-to-end-configuration-reference)
11. [Decision Flowchart: Picking Your Parameters](#11-decision-flowchart-picking-your-parameters)
12. [References](#12-references)

---

## 1. Introduction

Direct Preference Optimization (DPO) [Rafailov et al., 2023] eliminates the need for a separate reward model by directly optimizing a policy from preference data using a classification loss derived from the Bradley-Terry model. Low-Rank Adaptation (LoRA) [Hu et al., 2021] reduces trainable parameter count by decomposing weight updates into low-rank matrices, cutting memory by 3x or more.

Combining LoRA with DPO is the dominant practical approach for aligning large language models on consumer and research-grade hardware. However, the interaction between LoRA's constrained parameter space and DPO's preference optimization dynamics introduces unique challenges not present in either technique alone.

This report synthesizes findings from the original LoRA paper, Biderman et al. (2024) on forgetting behavior, the Thinking Machines LoRA study (2025), Hugging Face's preference tuning benchmarks, the rsLoRA paper, and Unsloth's empirical hyperparameter guide into actionable best practices.

---

## 2. Background: How DPO with LoRA Works

### 2.1 The DPO Objective

DPO optimizes the policy model π_θ using pairs of (chosen, rejected) responses for a given prompt. The loss is:

```
L_DPO(π_θ; π_ref) = -E[ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

Where:
- `π_ref` is the frozen reference model (the SFT checkpoint)
- `y_w` is the preferred (chosen) response
- `y_l` is the dispreferred (rejected) response
- `β` controls divergence from the reference policy
- `σ` is the sigmoid function

### 2.2 LoRA in the DPO Context

With LoRA, only the low-rank decomposition matrices A and B are trainable:

```
W = W_0 + (α/r) · B · A
```

Where `W_0` is the frozen pretrained weight, `r` is the rank, and `α` is the scaling factor. In DPO, the reference model π_ref can be implemented by simply disabling the LoRA adapter and using the frozen base weights — avoiding the need to load a second full copy of the model.

### 2.3 The SFT-then-DPO Pipeline

DPO assumes the training data is in-distribution for the policy. The standard pipeline is:

1. **SFT stage**: Fine-tune with LoRA on instruction-following data
2. **Merge or carry forward**: Either merge the SFT adapter into the base model, or load it as the reference adapter
3. **DPO stage**: Apply a new LoRA adapter on top for preference optimization

---

## 3. Optimal LoRA Hyperparameters for DPO

### 3.1 Rank (r)

| Scenario | Recommended Rank | Rationale |
|----------|-----------------|-----------|
| Small preference dataset (<5K pairs) | 16–64 | Lower information content; higher ranks risk overfitting |
| Medium dataset (5K–50K pairs) | 64–128 | Sufficient capacity without excessive parameters |
| Large dataset (50K+ pairs) | 128–256 | Matches full fine-tuning performance [Thinking Machines, 2025] |
| RL-based (GRPO, online DPO) | 1–16 | Low information per episode; even rank-1 matches FullFT [Thinking Machines, 2025] |

**Key finding**: Biderman et al. (2024) showed that full fine-tuning learns perturbations with rank 10–100x greater than typical LoRA configurations. For DPO, where updates are more subtle than SFT, ranks of 64–256 are generally sufficient. However, the marginal benefit of going beyond 128 diminishes for most preference datasets.

**Practical starting point**: Begin with rank 64. If reward margins plateau early, increase to 128 or 256. If training loss drops below 0.2 quickly, decrease rank to reduce overfitting.

### 3.2 Alpha (α) and the Scaling Factor

The effective learning rate of LoRA is scaled by `α/r`. This ratio determines how much the adapter updates influence the final weights.

**Recommendations**:

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| Standard | α = r (ratio = 1) | Default safe choice |
| Aggressive | α = 2r (ratio = 2) | When underfitting; large datasets |
| Conservative | α = r/2 (ratio = 0.5) | Overfitting; small datasets |
| rsLoRA | α/√r (set `use_rslora=True`) | High ranks (>64); see Section 9 |

**For DPO specifically**: Use α = 32 with standard parametrization [Thinking Machines, 2025]. If using rsLoRA with high rank (e.g., r=256), set α = 256 and enable `use_rslora=True`, which applies `α/√r` scaling automatically.

**Overfitting signal**: If training loss drops below 0.2, halve the alpha value to reduce the adapter's influence [Unsloth, 2025].

### 3.3 Learning Rate

This is the single most critical hyperparameter for LoRA-DPO. DPO requires dramatically lower learning rates than SFT.

| Training Type | Recommended LR | Range |
|---------------|---------------|-------|
| SFT with LoRA | 2e-4 | 1e-4 to 5e-4 |
| **DPO with LoRA** | **5e-6 to 5e-7** | **1e-7 to 5e-5** |
| DPO full fine-tuning | 5e-7 | 1e-7 to 5e-6 |

**Key insight from Thinking Machines (2025)**: LoRA requires ~10x higher learning rates than full fine-tuning for the same task. However, DPO itself requires ~10–100x lower learning rates than SFT. These two forces roughly cancel, landing DPO+LoRA in the 5e-7 to 5e-6 range.

**Hugging Face alignment experiments** used `5e-7` for DPO with paired preference data and found it stable across models.

**Tuning strategy**:
1. Start at `5e-6`
2. If reward margins diverge or oscillate: reduce to `1e-6` or `5e-7`
3. If reward margins are flat after 100+ steps: increase to `1e-5`
4. Never exceed `5e-5` for DPO — this almost always causes divergence

### 3.4 Learning Rate Schedule

```
scheduler_type: cosine
warmup_ratio: 0.1  (or 5–10% of total training steps)
```

Cosine annealing with warmup is the consensus choice. Linear decay also works but cosine provides slightly better end-of-training performance. The warmup phase is critical for DPO stability — skipping it can cause early reward divergence.

### 3.5 Batch Size and Gradient Accumulation

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Per-device batch size | 2–4 | Memory-constrained default |
| Gradient accumulation steps | 4–8 | Achieves effective batch size of 8–32 |
| **Effective batch size** | **8–16** | Sweet spot for DPO |

**Critical warning**: LoRA shows greater sensitivity to batch size increases than full fine-tuning, especially on small datasets [Thinking Machines, 2025]. Large batch sizes (>32 effective) can degrade DPO performance with LoRA. Start with effective batch size 16 and only increase if training is too noisy.

For the AOT loss variant, larger batch sizes are explicitly recommended to compute meaningful quantile statistics.

### 3.6 Epochs

**Recommendation: 1 epoch** for DPO.

DPO is prone to overfitting on preference data, especially with LoRA. Hugging Face's experiments used a single epoch across all models. Going beyond 1 epoch risks:
- The model memorizing preference pairs rather than learning generalizable preferences
- Reward hacking (chosen reward increases but actual quality degrades)
- A training loss that drops below 0.2 (overfitting indicator)

If your dataset is very small (<1K pairs), you may cautiously try 2 epochs with increased regularization (weight decay, dropout).

### 3.7 LoRA Dropout

| Scenario | Dropout |
|----------|---------|
| Default | 0.0 |
| Small dataset / overfitting suspected | 0.05–0.1 |
| Large dataset | 0.0 |

Standard practice is no dropout. The rsLoRA DPO configuration from Hugging Face used `dropout=0.1` with rank 256, suggesting dropout becomes more useful at higher ranks.

### 3.8 Weight Decay

```
weight_decay: 0.01
```

Range: 0.01–0.1. Biderman et al. (2024) found that LoRA inherently mitigates forgetting better than regularization techniques like weight decay and dropout. For DPO, light weight decay (0.01) is sufficient.

### 3.9 Optimizer

```
optimizer: adamw_torch  (or adamw_8bit for memory savings)
```

AdamW is the standard. For memory-constrained setups, 8-bit Adam (via bitsandbytes) works without measurable quality loss. Paged AdamW (`paged_adamw_32bit`) is useful when training near GPU memory limits.

---

## 4. DPO-Specific Hyperparameters

### 4.1 Beta (β) — The KL Penalty

β is the most important DPO-specific hyperparameter. It controls how much the trained policy can diverge from the reference model.

| β Value | Behavior | When to Use |
|---------|----------|-------------|
| 0.01 | Very aggressive update; large divergence allowed | Strong base model; high-quality preference data |
| 0.1 | Standard; most commonly used default | General-purpose alignment |
| 0.2–0.5 | Conservative; stays close to reference | Noisy preference labels; small datasets |
| 0.5–0.9 | Very conservative; minimal change | Fine-grained style adjustments only |

**Critical finding from Hugging Face [2024]**: β is highly model-dependent and must be tuned via sweep.

- **Zephyr-7B**: Best performance at β = 0.01 (the lowest tested)
- **OpenHermes-7B**: Best at β = 0.6 for DPO, β = 0.01 for IPO

**Recommended sweep strategy**:
1. Coarse sweep: β ∈ {0.01, 0.1, 0.3, 0.5}
2. Fine-grained sweep in the 0.01–0.2 range (this range is undertested but often contains the optimum)
3. Monitor `rewards/margins` — it should increase steadily. If it spikes and plateaus, β is too low; if it's flat, β is too high.

### 4.2 Label Smoothing

```
label_smoothing: 0.0  (default)
```

For noisy preference data (e.g., AI-generated labels), set `label_smoothing` to 0.1–0.3. This implements Conservative DPO (cDPO) [Mitchell, 2023], which assumes labels are noisy with some probability. For Robust DPO, the same parameter models noise probability.

### 4.3 Max Prompt/Completion Length

```
max_prompt_length: 512
max_length: 1024  (or 2048 for longer responses)
```

Truncation strategy matters: too short and you lose context; too long and you waste compute on padding. Match these to your dataset's actual length distribution.

---

## 5. Architecture Decisions: Which Layers and Modules

### 5.1 Target Modules

**Apply LoRA to ALL linear layers**, not just attention:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP / FFN
]
```

**Thinking Machines (2025) finding**: "Attention-only LoRA significantly underperforms." Including MLP layers is essential for DPO, where the model needs to adjust both how it attends to preference signals and how it transforms representations into outputs.

For Mixture-of-Experts models, also target MoE layers and consider enabling `output_router_logits=True` with `router_aux_loss_coef=0.001` for balanced expert utilization during DPO.

### 5.2 Bias

```
bias: "none"
```

Training bias terms adds minimal capacity but increases memory and slows training. Not recommended for DPO.

### 5.3 Modules to Save

If you need the LM head to adapt (common when the vocabulary or output distribution shifts significantly), add it to `modules_to_save`. For standard DPO on the same vocabulary, this is unnecessary.

---

## 6. Reference Model Handling Strategies

The reference model in DPO is crucial — it provides the baseline log-probabilities against which the policy is compared. With LoRA, you have three strategies:

### 6.1 Strategy 1: Null Reference (Disable Adapter)

**How it works**: Load one model with a LoRA adapter. During reference forward passes, disable the adapter so the base model acts as π_ref.

```python
# In TRL, just don't pass ref_model:
trainer = DPOTrainer(model=model, ref_model=None, ...)
```

**Pros**: Minimal memory; only one model copy
**Cons**: Reference model is the *base* model, not the SFT model. If your SFT adapter was merged in, this works perfectly. If not, the reference doesn't match the SFT policy.

**Best for**: When you've merged the SFT adapter into the base weights before DPO.

### 6.2 Strategy 2: Dual Adapter Loading

**How it works**: Load the SFT adapter twice under different names — one trainable ("train"), one frozen ("reference").

```python
model = PeftModel.from_pretrained(base_model, sft_adapter_path,
                                   is_trainable=True, adapter_name="train")
model.load_adapter(sft_adapter_path, adapter_name="reference")

training_args = DPOConfig(
    model_adapter_name="train",
    ref_adapter_name="reference",
)
```

**Pros**: True SFT reference without merging; memory overhead is only one adapter's worth
**Cons**: Slightly more complex setup; marginal memory increase

**Best for**: QLoRA workflows where merging degrades quality (see Section 6.4).

### 6.3 Strategy 3: Two Full Models

**How it works**: Load two separate model instances.

**Pros**: Maximum flexibility
**Cons**: Doubles memory; wasteful with LoRA

**Avoid this approach** unless you have specific reasons (e.g., reference model is a different architecture).

### 6.4 The QLoRA Merge Problem

Merging a QLoRA adapter into a 4-bit quantized base model introduces quantization errors. Benjamin Marie (2024) demonstrated measurable quality loss from this merge. The recommended workflow:

1. Dequantize the base model to full precision
2. Merge the SFT adapter
3. Re-quantize for DPO training

Or simply use **Strategy 2** (dual adapter) to avoid the merge entirely.

### 6.5 TR-DPO: Syncing the Reference Model

TR-DPO [Azar et al., 2024] periodically syncs the reference model with the policy during training:

```python
training_args = DPOConfig(
    sync_ref_model=True,
    ref_model_sync_steps=64,
    ref_model_mixup_alpha=0.9,
)
```

This helps when training for multiple epochs or on large datasets where the policy drifts far from the initial reference. Consider this if `rewards/margins` grows unboundedly.

---

## 7. Loss Function Selection

### 7.1 Standard Sigmoid (Default DPO)

```python
loss_type = "sigmoid"
```

The original DPO loss. Robust, well-understood, and the best default. Start here.

### 7.2 IPO (Identity Preference Optimization)

```python
loss_type = "ipo"
```

Addresses theoretical overfitting concerns in DPO. β has a different interpretation (reciprocal of the log-likelihood gap). IPO averages over log-likelihoods rather than summing them. Note: early TRL implementations had a bug where the loss was summed, not averaged — ensure you're on TRL ≥ v0.8.0.

**When to use**: When you suspect overfitting on preference labels or have noisy data.

### 7.3 Hinge Loss (RSO)

```python
loss_type = "hinge"
```

β acts as the reciprocal of the margin. More aggressive separation of chosen/rejected.

### 7.4 Combined Losses (MPO)

```python
loss_type = ["sigmoid", "bco_pair", "sft"]
loss_weights = [0.8, 0.2, 1.0]
```

Combines preference optimization with supervised fine-tuning on chosen responses. Adding an SFT component (RPO-style) prevents catastrophic forgetting of the SFT formatting during DPO.

### 7.5 Recommendation

Use **sigmoid** as default. If reward accuracy plateaus above 0.9 but generation quality degrades, try **IPO**. If you observe formatting regression (e.g., the model stops using proper chat format), add an **SFT** loss component with weight 0.5–1.0.

---

## 8. Training Stability and Common Failure Modes

### 8.1 Reward Margin Divergence

**Symptom**: `rewards/margins` increases rapidly, then training loss spikes or NaN occurs.
**Cause**: Learning rate too high or β too low.
**Fix**: Reduce learning rate by 5x; increase β; add warmup.

### 8.2 Flat Reward Margins

**Symptom**: `rewards/margins` stays near zero throughout training.
**Cause**: Learning rate too low, β too high, or rank too low.
**Fix**: Increase learning rate by 2–5x; decrease β; increase rank.

### 8.3 Reward Accuracy ≈ 1.0 Early

**Symptom**: `rewards/accuracies` hits 1.0 within the first 10% of training.
**Cause**: Overfitting. The model memorizes preferences rather than learning generalizable behavior.
**Fix**: Reduce rank; reduce epochs; increase label smoothing; use IPO loss.

### 8.4 Double Descent in LoRA

**Symptom**: Training loss exhibits a characteristic dip, rise, then dip pattern.
**Cause**: LoRA at moderate-to-high ranks can exhibit unstable training dynamics.
**Fix**: Chen et al. (2025) propose LoRA-MGPO (Momentum-Guided Perturbation Optimization). Practically, using rsLoRA with careful learning rate warmup and cosine decay mitigates this.

### 8.5 Formatting Regression

**Symptom**: After DPO, the model loses chat formatting, stops following the template, or outputs garbled text.
**Cause**: DPO optimizes preferences without explicit formatting supervision.
**Fix**: Add an SFT loss component (`loss_type=["sigmoid", "sft"]`); ensure your preference dataset includes proper chat templates.

### 8.6 Chosen Log-Probability Collapse

**Symptom**: Both chosen and rejected log-probabilities decrease, but rejected drops faster.
**Cause**: The model learns to assign low probability to everything, differentiating only by relative margin.
**Fix**: Add an SFT loss on chosen responses (RPO-style); consider WPO (weighted preference optimization) with `use_weighting=True`.

---

## 9. Rank-Stabilized LoRA (rsLoRA) for DPO

### 9.1 The Problem with Standard LoRA at High Ranks

Standard LoRA scales updates by `α/r`. As rank increases, this scaling shrinks the effective update magnitude, causing performance to plateau — not because the capacity is wasted, but because the learning signal is suppressed.

### 9.2 rsLoRA Solution

rsLoRA replaces the scaling with `α/√r`, preventing the learning signal from being artificially dampened at higher ranks.

```python
from peft import LoraConfig

config = LoraConfig(
    r=256,
    lora_alpha=256,
    use_rslora=True,   # Applies α/√r scaling
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "o_proj", "up_proj", "gate_proj"],
    task_type="CAUSAL_LM",
)
```

### 9.3 DPO-Specific Results

On MT-Bench with OpenChat 3.5 + DPO:

| Configuration | Turn 1 | Turn 2 | Average |
|--------------|--------|--------|---------|
| Base model (no DPO) | 8.21 | 7.38 | 7.79 |
| LoRA rank 16 | 8.34 | 7.53 | 7.93 |
| LoRA rank 256 | 8.30 | 7.63 | 7.96 |
| **rsLoRA rank 256** | **8.43** | **7.75** | **8.09** |

rsLoRA rank 256 nearly doubled the improvement over the base model compared to standard LoRA rank 16, with only 13 extra minutes of training on 8x A100 GPUs.

### 9.4 When to Use rsLoRA

- **Always use for ranks > 64** — standard LoRA scaling suppresses learning at high ranks
- **Particularly effective for DPO**, where the model needs sufficient capacity to learn nuanced preference distinctions
- **Minimal cost**: The scaling change has zero computational overhead

---

## 10. End-to-End Configuration Reference

### 10.1 Recommended Default Configuration

```python
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# LoRA Configuration
peft_config = LoraConfig(
    r=64,                  # Start here; increase to 128-256 for large datasets
    lora_alpha=64,         # α = r for standard; use rsLoRA for high ranks
    lora_dropout=0.0,      # Increase to 0.05-0.1 if overfitting
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
    use_rslora=False,      # Set True for r > 64
)

# DPO Configuration
training_args = DPOConfig(
    # Output
    output_dir="./dpo-output",

    # Core DPO params
    beta=0.1,                         # Sweep: {0.01, 0.1, 0.3, 0.5}
    loss_type="sigmoid",              # Default DPO loss

    # Learning rate
    learning_rate=5e-6,               # Range: 5e-7 to 5e-5
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Batch
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # Effective batch = 16

    # Training duration
    num_train_epochs=1,

    # Precision
    bf16=True,

    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Sequence lengths
    max_prompt_length=512,
    max_length=1024,

    # Logging
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,

    # Memory optimization
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,        # Uses disabled adapter as reference
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
```

### 10.2 High-Rank rsLoRA Configuration (Large Dataset)

```python
peft_config = LoraConfig(
    r=256,
    lora_alpha=256,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
    use_rslora=True,
)

training_args = DPOConfig(
    beta=0.1,
    learning_rate=1e-6,           # Lower LR for high-rank stability
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=True,
    weight_decay=0.01,
    gradient_checkpointing=True,
)
```

### 10.3 Small Dataset / Overfitting-Prone Configuration

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    beta=0.3,                      # Conservative: stay close to reference
    loss_type="ipo",               # IPO is more robust to overfitting
    learning_rate=5e-6,
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
    weight_decay=0.05,             # Stronger regularization
    label_smoothing=0.1,           # Assume noisy labels
)
```

---

## 11. Decision Flowchart: Picking Your Parameters

### Step 1: Determine Your Rank

```
Dataset size < 5K pairs  → r = 16–32
Dataset size 5K–50K      → r = 64–128
Dataset size > 50K       → r = 128–256 (use rsLoRA)
RL-based (online DPO)    → r = 4–16
```

### Step 2: Set Alpha

```
r ≤ 64   → α = r (standard scaling)
r > 64   → α = r + use_rslora=True
Overfitting? → halve α
```

### Step 3: Set Learning Rate

```
Start at 5e-6
Training unstable?     → reduce to 5e-7
No learning signal?    → increase to 1e-5
Never exceed 5e-5
```

### Step 4: Set Beta

```
Strong SFT base + clean data   → β = 0.01–0.1
General purpose                → β = 0.1
Noisy labels / small data      → β = 0.3–0.5
Always validate with sweep
```

### Step 5: Monitor and Adjust

```
rewards/margins increasing steadily      → good; continue
rewards/margins flat                     → increase LR or decrease β
rewards/margins spiking/diverging        → decrease LR or increase β
rewards/accuracies = 1.0 too early       → reduce rank, add regularization
training loss < 0.2                      → overfitting; halve α or stop
formatting breaks                        → add SFT loss component
```

---

## 12. References

1. **Hu, E., et al. (2021).** LoRA: Low-Rank Adaptation of Large Language Models. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

2. **Rafailov, R., et al. (2023).** Direct Preference Optimization: Your Language Model is Secretly a Reward Model. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

3. **Biderman, D., et al. (2024).** LoRA Learns Less and Forgets Less. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)

4. **Hayou, S., et al. (2024).** LoRA+: Efficient Low Rank Adaptation of Large Models. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)

5. **Kalajdzievski, D. (2023).** Rank-Stabilized LoRA (rsLoRA): Unlocking the Potential of LoRA Fine-Tuning. [Hugging Face Blog](https://huggingface.co/blog/damjan-k/rslora)

6. **Thinking Machines (2025).** LoRA Best Practices. [Blog Post](https://thinkingmachines.ai/blog/lora/)

7. **Ivison, H., et al. (2024).** Tulu 3: Pushing Frontiers in Open Language Model Post-Training. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)

8. **Tunstall, L., et al. (2024).** Preference Tuning LLMs with Direct Preference Optimization Methods. [Hugging Face Blog](https://huggingface.co/blog/pref-tuning)

9. **Unsloth (2025).** LoRA Hyperparameters Guide. [Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

10. **Meng, F., Wang, Z., & Zhang, M. (2024).** PiSSA: Principal Singular Values and Singular Vectors Adaptation. [arXiv:2404.02948](https://arxiv.org/abs/2404.02948)

11. **Allen-Zhu, Z. & Li, Y. (2024).** Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws. [arXiv:2404.05405](https://arxiv.org/abs/2404.05405)

12. **Malladi, S., et al. (2022).** A Kernel-Based View of Language Model Fine-Tuning. [arXiv:2210.05643](https://arxiv.org/abs/2210.05643)

13. **Chen, Y., et al. (2025).** LoRA-MGPO: Mitigating Double Descent in Low-Rank Adaptation via Momentum-Guided Perturbation Optimization. [arXiv:2502.14538](https://arxiv.org/abs/2502.14538)

14. **Azar, M. G., et al. (2024).** TR-DPO: Trust Region Direct Preference Optimization. [arXiv:2404.09656](https://arxiv.org/abs/2404.09656)

15. **Mitchell, E. (2023).** A Note on DPO with Noisy Preferences & Relationship to RLHF. [cDPO Paper](https://ericmitchell.ai/cdpo.pdf)

16. **Mangrulkar, S., et al. (2022).** PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. [GitHub](https://github.com/huggingface/peft)

17. **Marie, B. (2024).** Don't Merge Your LoRA Adapter into a 4-bit LLM. [Medium](https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997)

18. **HuggingFace TRL (2025).** DPO Trainer Documentation. [Docs](https://huggingface.co/docs/trl/main/en/dpo_trainer)

19. **Dubey, A., et al. (2024).** The Llama 3 Herd of Models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

20. **Shao, Z., et al. (2024).** DeepSeekMath: Pushing the Limits of Mathematical Reasoning. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
