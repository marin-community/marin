# Job Summary: d=2560 MoE on 10T tokens

Big-batch d=2560 MoE training run on ~10T tokens (~1.2e23 FLOPs), with the May Recipe baked in. Below are sections on model, optimizer, training, scaling principles, and outputs.

## 1. Model

### Hyperparameters

- **hidden_dim**: 2560
- **num_layers**: 26 (heuristic-derived: `round(2560 / (64 + 4·log2(2560) − 9))`)
- **num_heads**: 20 (`hidden_dim / 128`)
- **num_kv_heads**: 5 (GQA 4:1; largest divisor of 20 ≤ 5)
- **head_dim**: 128
- **intermediate_dim**: 1280 (`ceil(d/2/128)·128`)
- **shared_expert_intermediate_dim**: 2560 (= d)
- **num_experts**: 256
- **num_experts_per_token**: 4
- **vocab_size**: 128,256
- **max_seq_len**: 4096
- **sliding_window**: 4096
- **active_params**: 2.34B (2.01B core + 0.33B lm_head; excludes embed per guidelines)
- **total_params**: 67.1B (66.4B core + 0.33B embed + 0.33B lm_head)

### Features

- **GQA**: 4:1 ratio (20 Q heads, 5 KV heads).
- **QB Routing**: stop-gradient bias updated each step from previous step's QB-β statistics (applied inside JIT).
- **Sigmoid Combine Weights**: sigmoid on unbiased router logits for selected experts.
- **GatedNorm**: rank-128 low-rank gating after RMSNorm, pre-attention and pre-MLP.
- **XSA**: subtracts V-parallel component from attention output per head.
- **Attention Gate**: learned per-head sigmoid gate on attention output, initialized to zero.
- **Shared Expert**: dense FFN (d=2560) in parallel with routed experts per layer.
- **Sliding Window**: every 4th layer (plus last) uses full window (4096); others use half (2048).
- **Half-RoPE on every layer**: RoPE applied only to the first half of Q/K per head (`q[..., :head_dim/2]`, `k[..., :head_dim/2]`). The second half is rope-free on every layer.
- **PKO (Partial Key Offset)**: on every-4th + last layer (7 of 26 layers here), the rope-free second half of K is additionally shifted by 1 position with doc-start zeroing.
- **QK Gain**: `qk_mult = 1.3`.
- **Split w_gate / w_up**: separate weight matrices for SwiGLU gate and up projections (`split_w_gate_up=True`).
- **Routing renorm sum**: 2.5 (combined-feature baseline).
- **Router z-loss**: disabled (`router_z_loss_coef = 0`).
- **Expert dispatch**: ring (May Recipe default; `moe_implementation` not exposed in current model config).

## 2. Optimizer

### Hyperparameters

(Using the refit `MoeAdamHHeuristic` from #5951.)

- **adam_lr**: 0.000432  (= `0.06602 · 1e13^-0.395 · 2560^-0.150 · √(2048·4096)`)
- **muonh_lr (= adamh_lr)**: 0.001870  (= adam_lr · 13/3)
- **beta1**: 0.9062
- **beta2**: 0.95 (clamped; `0.999^(2048/32) = 0.938` → clipped up to 0.95)
- **epsilon**: 1.06e−14  (= `1e-15 · √((32·1e13)/(2048·1.4e9))`)
- **weight_decay**: 0
- **max_grad_norm**: None (1pct-noclip schedule)
- **warmup**: 1% (= 11,921 steps)
- **decay**: None (linear schedule decays to 0 over full run)
- **min_lr_ratio**: 0.0
- **lr_schedule**: linear
- **z_loss_weight**: 0.0

### Features

- **MuonH** (`GrugMoeMuonHConfig`): scale-invariant Newton-Schulz updates on the `muonh` group (matmuls + GatedNorms), Frobenius hyperball step on top. Used in place of AdamH on weight matrices.
- **Multi-group optimizer**: 3 groups —
  - `muonh`: attn matmuls, MoE expert matmuls (gate/up/down), shared expert, GatedNorms.
  - `adamh`: lm_head / output_proj (LR = `muonh_lr`).
  - `adam`: token_embed, router weight + bias, attn_gate, 1-D norm weights (LR = `adam_lr`).

## 3. Training

- **hardware**: v4-1024 us-central2 (= 512 chips).
- **num_train_steps**: 1,192,093  (= `1e13 / (2048 · 4096)`)
- **batch_size**: 2,048 sequences
- **seq_len**: 4,096
- **tokens_per_batch**: 8.39M (= `2048 · 4096`)
- **total_tokens**: 1.00e13 (= 10T)
- **compute_budget**: ~1.21e23 FLOPs (= `6 · 2.01e9 · 1e13`)
- **mixed_precision**: `params=float32, compute=bfloat16, output=bfloat16`
- **expert_parallelism**: 4
- **data**: Nemotron mix (block-shuffled) with default validation sets
- **checkpoints**: every 10 minutes (rolling temporary), permanent every 5,000 steps

## 4. Scaling Principles

All optimizer hyperparameters derived from the refit MuonH May Recipe heuristic (issue #5951, R² = 0.996 on 17 cells).

### Learning rates (current heuristic)

- **adam_lr** = `0.06602 · tokens^-0.395 · dim^-0.150 · √(B·seq_len)`, capped at 0.05.
- **muonh_lr** = `(13/3) · adam_lr`.

### Momentum and precision

- **beta1** = 0.9062 (fixed)
- **beta2** = `clip(0.999^(B/32), 0.95, 0.9999)` → 0.95 at B=2048
- **epsilon** = `1e-15 · √((32·T) / (B·1.4e9))`

### Architecture scaling

- **num_layers** = `round(hidden_dim / (64 + 4·log2(hidden_dim) − 9))` → 26 for d=2560
- **num_heads** = `hidden_dim / 128`
- **num_kv_heads** = 4:1 GQA, largest divisor of num_heads ≤ num_heads/4
- **intermediate_dim** = `ceil(hidden_dim / 2 / 128) · 128`
- **shared_expert_intermediate_dim** = hidden_dim
- **initializer_std** = `0.5 / sqrt(hidden_dim)` ≈ 0.00988
- **num_experts** = 256, **num_experts_per_token** = 4 (fixed for May Recipe; differs from the 64-expert v16 baseline used in the 1e23 summary)

### Schedule

- **warmup** = 1% of training (1pct-noclip baseline)
- **decay** = linear all the way to 0 (no separate decay phase)
- **max_grad_norm** = None (no clipping)

## 5. Outputs

- **GCS checkpoint path**: `gs://marin-us-central2/grug/moe_may_d2560_10T/marin-big-run-moe_may_d2560_10T-XXXXXX/checkpoints` (pending — actual hash assigned at submission)
- **WandB run link**: `https://wandb.ai/marin-community/marin_moe/runs/marin-big-run-moe_may_d2560_10T` (pending)
- **Final eval metrics**: pending (run not submitted yet)
  - eval/paloma/c4_en/bpb
  - eval/paloma/macro_loss
  - eval/uncheatable_eval/macro_loss
- **Throughput** (mean over last 100 logged steps): pending
