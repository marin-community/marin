# Job Summary: 1e23 MoE (d5120, 129B total / 16B active)

Full-scale 1e23 FLOPs MoE training run on v4-1024 with ring expert dispatch. Below are sections on model, optimizer, training, and scaling principles.

## 1. Model

### Hyperparameters

- **hidden_dim**: 5120
- **num_layers**: 48
- **num_heads**: 40
- **num_kv_heads**: 10 (GQA 4:1)
- **head_dim**: 128
- **intermediate_dim**: 2560 (MoE expert FFN)
- **shared_expert_intermediate_dim**: 5120
- **num_experts**: 64
- **num_experts_per_token**: 4
- **vocab_size**: 128,256
- **max_seq_len**: 4096
- **sliding_window**: 4096
- **total_params**: 129.1B
- **active_params**: 15.9B

### Features

- **GQA**: 4:1 ratio (40 Q heads, 10 KV heads)
- **QB Routing**: stop-gradient bias updated each step from previous step's QB-β statistics (applied inside JIT)
- **Sigmoid Combine Weights**: sigmoid on unbiased router logits for selected experts
- **GatedNorm**: rank-128 low-rank gating after RMSNorm, pre-attention and pre-MLP
- **XSA**: subtracts V-parallel component from attention output per head
- **Attention Gate**: learned per-head sigmoid gate on attention output, initialized to zero
- **Shared Expert**: dense FFN (d=5120) in parallel with routed experts per layer
- **Sliding Window**: every 4th layer uses full window (4096); others use half (2048)
- **QK Gain**: qk_mult = 1.3
- **RoPE**: standard rotary embeddings (theta=10000)
- **ArrayStacked + scan**: all 48 blocks stacked via `jax.lax.scan` for reduced compile time and HBM
- **Expert dispatch**: ring implementation (not ragged_all_to_all)
- **Capacity factor**: 1.0

## 2. Optimizer

### Hyperparameters

- **adam_lr**: 0.001339
- **adamh_lr**: 0.005803 (adam_lr × 13/3)
- **beta1**: 0.9062
- **beta2**: 0.95 (clamped; formula gives 0.95 at bs=2048)
- **epsilon**: 3.356e-15
- **weight_decay**: 0
- **max_grad_norm**: 1.0
- **warmup**: 10% of training
- **decay**: None (linear schedule decays to 0)
- **min_lr_ratio**: 0.0
- **lr_schedule**: linear
- **z_loss_weight**: 1e-4

### Features

- **AdamH**: scale-invariant updates for 2D+ weight matrices (attention, expert weights). Recursive vmap for ArrayStacked >3D tensors.
- **Multi-group optimizer**: AdamH for weight matrices, Adam for biases/norms/router/1D params
- **Router Z-Loss**: penalizes large router logits for stability

## 3. Training

- **hardware**: v4-1024 (512 chips)
- **num_train_steps**: 120,332
- **batch_size**: 2048
- **seq_len**: 4096
- **total_tokens**: ~1.01T
- **compute_budget**: 1e23 FLOPs
- **mixed_precision**: params=float32, compute=bfloat16, output=bfloat16
- **expert_parallelism**: 4
- **data**: Nemotron mix (block-shuffled) with default validation sets
- **checkpoints**: gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep4_ring-8f37ad/checkpoints (every 30 min, keep every 10,000 steps)

## 4. Scaling Principles

All optimizer hyperparameters derived from reference point B0=32, T0=1.4e9 tokens.

### Learning rates

- **adamh_lr** = 1.63 × tokens^(-0.2813) × dim^(-0.3678) × sqrt(B) × (13/3), capped at 0.05
- **adam_lr** = adamh_lr / (13/3)

### Momentum and precision

- **beta1** = 0.9062 (fixed)
- **beta2** = clip(0.999^(B/32), 0.95, 0.9999) → 0.95 at B=2048
- **epsilon** = 1e-15 × sqrt((B0 × T) / (B × T0))

### Architecture scaling

- **num_layers** = 48 (hardcoded for 1e23; formula gives 49 for d=5120)
- **num_heads** = hidden_dim / 128 = 40
- **num_kv_heads** = 10 (largest divisor of 40 ≤ 10)
- **intermediate_dim** = ceil(hidden_dim / 2 / 128) × 128 = 2560
- **shared_expert_intermediate_dim** = hidden_dim = 5120
- **initializer_std** = 0.5 / sqrt(5120) ≈ 0.00699

## 5. Outputs

- **GCS checkpoint path**: gs://marin-us-central2/grug/moe_1e23_d5120_bs2048_ep4_ring-8f37ad/checkpoints
- **WandB run link**: https://wandb.ai/marin-community/dial_moe/runs/moe_1e23_d5120_bs2048_ep4_ring
- **Final eval metrics**: pending (run in progress, step ~3,863 / 120,332)
- **Throughput** (mean over last 100 logged steps):
  - throughput/mfu: 14.23%
  - throughput/tokens_per_second: 194,443
