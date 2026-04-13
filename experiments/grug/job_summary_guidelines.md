# Job Summary Guidelines

When summarizing a training job, include the following sections. Each section should be updated to match the actual code used for the specific job — do not copy these defaults verbatim if the job differs.

Lead with a one-sentence summary of what the job is (e.g., model size, compute budget, purpose), followed by a sentence listing the sections: "Below are sections on model, optimizer, training, and scaling principles."

## 1. Model

### Hyperparameters

- **hidden_dim**: Model hidden dimension
- **num_layers**: Number of transformer blocks
- **num_heads**: Number of attention heads
- **num_kv_heads**: Number of key/value heads (differs from num_heads when using GQA)
- **head_dim**: Dimension per attention head
- **intermediate_dim**: MoE expert FFN intermediate dimension
- **shared_expert_intermediate_dim**: Shared (dense) expert FFN intermediate dimension
- **num_experts**: Total number of routed experts
- **num_experts_per_token**: Number of experts selected per token (top-k)
- **vocab_size**: Tokenizer vocabulary size
- **max_seq_len**: Maximum sequence length
- **sliding_window**: Sliding window attention size
- **active_params**: Total active parameters per forward pass (excluding embed, including lm_head)
- **total_params**: Total parameters including all experts, embed, and lm_head

### Features

- **GQA (Grouped Query Attention)**: Reduces KV heads relative to Q heads by a ratio (e.g., 4:1). The number of KV heads is the largest divisor of num_heads that is <= num_heads // gqa_ratio. KV projections are broadcast to match Q heads during attention.
- **QB Routing (Query-Based Load Balancing)**: Router selects top-(K+1) experts on biased logits. The (K+1)-th expert's logit serves as the threshold alpha. A per-expert bias beta is computed via sharded top-k on (logits - alpha), then averaged across devices via pmean. Biases are applied as stop-gradient offsets to balance expert selection without affecting gradients.
- **Sigmoid Combine Weights**: After expert selection via biased logits, combine weights are computed as sigmoid of the unbiased logits for the selected experts (not softmax).
- **GatedNorm**: Learnable per-dimension gating applied after RMSNorm. Compensates for AdamH's bounded activation norms. Uses a low-rank (rank 128) down-up projection with SiLU activation and sigmoid gate.
- **XSA (Exclusive Self Attention)**: After computing attention output y = attn(Q, K, V), subtracts the component of y parallel to V: z = y - (y·v / ||v||²) v, per head. This removes the "copy" component from attention, forcing each head to learn exclusively new information rather than passing through the value unchanged.
- **Attention Gate**: A learnable projection matrix (hidden_dim, num_heads) that produces a per-head, per-token gate via `2 * sigmoid(x @ attn_gate)`. This gate scales the attention output per head before the output projection. Initialized to zero so gating starts at zero and is learned during training.
- **Shared Expert**: A dense FFN that runs in parallel with the routed MoE FFN. Its output is added to the routed output before the residual connection.
- **Sliding Window Attention**: Alternates between short (window/2) and long (window) attention masks. Every 4th layer uses the long window; others use the short window.
- **QK Gain**: After QK normalization and RoPE, Q is scaled by a constant multiplier (qk_mult). This controls the sharpness of attention logits independently of head dimension.
- **RoPE (Rotary Position Embeddings)**: Applied to Q and K projections for position encoding.
- **Normalization placement**: Pre-norm architecture. Each sub-block input goes through RMSNorm then GatedNorm before the sub-block (attention or MLP). Additionally, embeddings are normalized (RMSNorm + GatedNorm) after the embedding lookup, and a final RMSNorm + GatedNorm is applied before the output projection. Q and K are also RMS-normalized inside attention before RoPE.

## 2. Optimizer

### Hyperparameters

- **adam_lr**: Learning rate for Adam parameter groups
- **adamh_lr**: Learning rate for AdamH parameter groups. Typically adam_lr * (13/3).
- **beta1**: First moment decay
- **beta2**: Second moment decay
- **epsilon**: Adam epsilon
- **weight_decay**: Weight decay coefficient
- **max_grad_norm**: Gradient clipping norm
- **warmup**: Warmup fraction or step count
- **decay**: Fraction of training for LR decay phase
- **min_lr_ratio**: Minimum LR as fraction of peak LR
- **lr_schedule**: Schedule type (e.g., "cosine", "linear")
- **z_loss_weight**: Router z-loss coefficient for load balancing

### Features

- **AdamH (Adam with Hessian-scaled updates)**: Applies Newton-style scaling to weight matrix updates. Used for attention and expert weight matrices. The learning rate is typically higher than standard Adam (13/3 ratio).
- **Multi-group Optimizer**: Different parameter groups get different optimizers and learning rates (adam_lr vs adamh_lr). AdamH is used for weight matrices (attention, experts), Adam for everything else.
- **Router Z-Loss**: Auxiliary loss that penalizes large router logits to stabilize training. Added to the main loss scaled by z_loss_weight.

## 3. Training

- **hardware**: TPU type and count (e.g., v4-32, v4-128, v4-512)
- **num_train_steps**: Total training steps
- **batch_size**: Global batch size (number of sequences per step)
- **seq_len**: Sequence length per example
- **total_tokens**: batch_size * seq_len * num_train_steps
- **compute_budget**: Approximate FLOPs budget (tokens * 3 * FLOPs_per_token)
- **mixed_precision**: Policy string (e.g., params=float32, compute=bfloat16, output=bfloat16)
- **expert_parallelism**: Number of expert-parallel shards (1 = replicated)
- **data**: Training mixture description

## 4. Scaling Principles

All optimizer hyperparameters are derived from a reference point (B0=32, T0=1.4e9 tokens) and scaled based on batch size B and total tokens T.

Define the scaling ratio: `r/r0 = (B * T0) / (B0 * T)`

### Learning rates

- **adamh_lr** = `lr0 * sqrt(B/B0) * (T0/T)^0.3`
  - Scales with square root of batch size and inverse 0.3-power of token count
  - Reference: lr0 = 0.003701
- **adam_lr** = `adam_lr0 * sqrt(r/r0)`
  - Scales with square root of the scaling ratio
  - Reference: adam_lr0 = 0.002356

### Momentum and precision

- **beta1** = 0.9062 (fixed, does not scale)
- **beta2** = `clip(beta2_0^(B/B0), 0.95, 0.9999)`
  - Exponentiated by batch ratio to maintain constant token half-life
  - Reference: beta2_0 = 0.999
- **epsilon** = `epsilon0 * sqrt(r0/r)`
  - Scales inversely with square root of scaling ratio
  - Reference: epsilon0 = 1e-15

### Architecture scaling

- **num_layers** = `round(hidden_dim / (64 + 4*log2(hidden_dim) - 9))`
- **num_heads** = `hidden_dim / 128`
- **intermediate_dim** = `hidden_dim / 2` (MoE expert FFN)
- **shared_expert_intermediate_dim** = `hidden_dim` (dense shared expert)
- **initializer_std** = `0.5 / sqrt(hidden_dim)`
- **num_experts** = 64, **num_experts_per_token** = 4 (fixed)

### Schedule

- **warmup** = 10% of training
- **decay** = last 20% of training (linear decay to 0)
- **lr_schedule** = linear

## 5. Outputs

- **GCS checkpoint path**: Full GCS path to the checkpoint directory
- **WandB run link**: URL to the wandb run
- **Final eval metrics**:
  - eval/paloma/c4_en/bpb
  - eval/paloma/macro_loss
  - eval/uncheatable_eval/macro_loss
- **Throughput** (compute mean over last 100 logged steps from wandb history):
  - throughput/mfu
  - throughput/tokens_per_second

Mark fields as "pending" if the run is not yet complete.
