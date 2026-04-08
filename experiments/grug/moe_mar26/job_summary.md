# Job Summary: moe-v7-1e22-d3200

MoE language model (4.7B active / 34.6B total params) trained at 1e22 FLOP budget on 326B tokens from Nemotron mix. Below are sections on model, optimizer, training, scaling principles, and outputs.

## 1. Model

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_dim | 3200 |
| num_layers | 32 |
| num_heads | 25 |
| num_kv_heads | 25 (MHA, no GQA) |
| head_dim | 128 |
| intermediate_dim | 1600 |
| shared_expert_intermediate_dim | 3200 |
| num_experts | 64 |
| num_experts_per_token | 4 |
| vocab_size | 128,256 |
| max_seq_len | 4096 |
| sliding_window | 4096 |
| active_params | ~4.7B |
| total_params | ~34.6B |
| qk_mult | 1.3 |
| initializer_std | 0.00884 |

### Features

- **QB Routing**: Top-(K+1) expert selection on biased logits with sigmoid combine weights. Per-expert bias beta computed via sharded top-k, averaged across devices via pmean. Biases are stop-gradient offsets.
- **GatedNorm**: Low-rank (128) down-up projection with SiLU + sigmoid gate, applied after RMSNorm before each sub-block.
- **XSA**: Subtracts the V-parallel component from attention output: z = y - (y.v / ||v||^2) v per head.
- **Attention Gate**: Learned projection (3200, 25) producing per-head per-token gates via 2*sigmoid(x @ attn_gate). Initialized to zero.
- **Shared Expert**: Dense FFN (dim=3200) in parallel with routed MoE. Output added to routed output before residual.
- **Sliding Window Attention**: Every 4th layer uses full window (4096), others use half window (2048).
- **QK Gain**: Q scaled by 1.3 after QK normalization and RoPE.
- **RoPE**: Rotary position embeddings on Q and K.
- **Normalization**: Pre-norm with RMSNorm + GatedNorm before each sub-block, after embedding, and before output projection. QK normalized inside attention.

## 2. Optimizer

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| adamh_lr | 0.00405 |
| adam_lr | 0.000863 |
| beta1 | 0.9062 |
| beta2 | 0.968 |
| epsilon | 2.73e-15 |
| max_grad_norm | 1.0 |
| warmup | 10% |
| decay | 20% (linear to 0) |
| lr_schedule | linear |
| z_loss_weight | 1e-4 |

### Features

- **AdamH**: Newton-style scaling for weight matrices (attention, experts). LR ~4.7x higher than Adam groups.
- **Multi-group Optimizer**: AdamH for weight matrices, Adam for everything else (router, embed, norms).
- **Router Z-Loss**: Penalizes large router logits for training stability.

## 3. Training

| Parameter | Value |
|-----------|-------|
| hardware | v4-512 |
| num_train_steps | 77,724 |
| batch_size | 1024 |
| seq_len | 4096 |
| total_tokens | 326B |
| compute_budget | ~1e22 FLOPs |
| mixed_precision | params=float32, compute=bfloat16, output=bfloat16 |
| expert_parallelism | 4 |
| data | Nemotron mix (block shuffle) with default validation sets |

## 4. Scaling Principles

All optimizer hyperparameters derived from reference point (B0=32, T0=1.4e9):

- **adamh_lr** = 0.003701 * sqrt(1024/32) * (1.4e9/3.26e11)^0.3 = 0.00405
- **adam_lr** = 0.002356 * sqrt(r/r0) = 0.000863, where r/r0 = (1024 * 1.4e9) / (32 * 3.26e11) = 0.1374
- **beta2** = clip(0.999^(1024/32), 0.95, 0.9999) = 0.968
- **epsilon** = 1e-15 * sqrt(1/0.1374) = 2.73e-15
- **num_layers** = round(3200 / (64 + 4*log2(3200) - 9)) = 32

## 5. Outputs

| Field | Value |
|-------|-------|
| GCS checkpoint path | gs://marin-us-central2/grug/moe-v7-1e22-d3200-5a4518/checkpoints |
| WandB run (v1) | https://wandb.ai/marin-community/dial_moe/runs/moe-v7-1e22-d3200 |
| WandB run (v2) | https://wandb.ai/marin-community/dial_moe/runs/moe-v7-1e22-d3200-v2 |
| eval/paloma/c4_en/bpb | 0.775 (v1 @ step 69k) |
| eval/paloma/macro_loss | 2.563 (v1 @ step 69k) |
| eval/uncheatable_eval/macro_loss | 2.135 (v1 @ step 69k) |
| throughput/mfu (mean last 100 steps) | 22.7% |
| throughput/tokens_per_second (mean last 100 steps) | 482,449 |
