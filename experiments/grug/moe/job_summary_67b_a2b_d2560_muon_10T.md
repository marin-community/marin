# Job Summary: 67B / 2B-active MoE 10T pretrain (d=2560, MuonH, v4-2048)

~10.07T-token MuonH MoE pretrain at d=2560 / 67B total params / 2.01B active per token (run_id `moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T`; batch ramp from 4,096 → 8,192 at step 15,000, with total steps reduced so total tokens stay near 10T). Below are sections on model, optimizer, training, and scaling principles.

## 1. Model

### Hyperparameters

- **hidden_dim**: 2560
- **num_layers**: 26
- **num_heads**: 20
- **num_kv_heads**: 5 (GQA 4:1)
- **head_dim**: 128
- **intermediate_dim**: 1280 (MoE expert FFN; `ceil(hidden_dim / 2 / 128) * 128`)
- **shared_expert_intermediate_dim**: 2560 (= hidden_dim)
- **num_experts**: 256
- **num_experts_per_token**: 4
- **vocab_size**: 128,256
- **max_seq_len**: 8,192
- **sliding_window**: 2,048
- **total_params**: 67.1B
- **active_params**: 2.01B per token (2.34B w/ lm_head)

### Features

- **GQA**: 4:1 ratio (20 Q heads, 5 KV heads, head_dim 128)
- **QB Routing**: stop-gradient bias updated each step from previous step's QB-β statistics (applied inside JIT)
- **Sigmoid Combine Weights**: sigmoid on unbiased router logits for the K=4 selected experts (not softmax). Combine weights are then renormalized to sum to **2.5** across the K selected experts (`_ROUTING_RENORM_SUM = 2.5` in `model.py:51`), which effectively rescales the routed-expert contribution per token by 2.5× compared to a standard sum-to-1 (softmax-style) normalization.
- **GatedNorm**: rank-128 low-rank gating after RMSNorm, pre-attention and pre-MLP
- **Shared Expert**: dense FFN (d=2560) in parallel with routed experts per layer
- **Sliding Window**: every 4th layer + final layer uses full window (8,192); the other 19 layers use the short window (2,048). Layout: 7 long / 19 short (`i % 4 == 3 or i == num_layers - 1`).
- **disable_long_rope**: long (full-attention) layers skip rotary embedding entirely. Short (windowed) layers still apply half-RoPE.
- **disable_pko**: no per-key offset on long layers. Plain attention only.
- **RoPE**: partial rotary — only half of `head_dim` is rotated (the leading 64 of 128 dims), the rest is passed through unchanged. Applied on short (windowed) layers only; long (full-attention) layers skip RoPE entirely per `disable_long_rope`.
- **ArrayStacked + scan**: all 26 blocks stacked via `jax.lax.scan` (`use_array_stacked_blocks=True`) for reduced compile time and HBM footprint

## 2. Optimizer

### Hyperparameters

- **muonh_lr**: 0.003733 → **0.005279** at step 15,000 (×√2 with the BS doubling)
- **adam_lr**: 0.000861 → **0.001218** at step 15,000 (= muonh_lr / (13/3); also ×√2)
- **beta1**: 0.9062 (unchanged across the BS ramp)
- **beta2**: 0.95 — clamped both before and after the ramp (raw formula 0.999^(B*S/131072) gives 0.774 pre-ramp and 0.599 post-ramp; both clamp to 0.95)
- **epsilon**: ~8.01 × 10⁻¹⁹ → **~5.66 × 10⁻¹⁹** at step 15,000 (÷√2 with the BS doubling)
- **weight_decay**: 0
- **max_grad_norm**: None (no clipping in MuonH 1pct-noclip schedule)
- **warmup**: 0.01 (1% of training; ~3,000 steps)
- **decay**: None (linear decay over the whole post-warmup period)
- **min_lr_ratio**: 0.05 (decay to 5% of peak by end-of-training)
- **lr_schedule**: linear
- **z_loss_weight**: 1e-4 (logit z-loss; router z-loss = 0)

### Features

- **MuonH (Muon + Frobenius hyperball)**: Newton-Schulz orthogonalization on weight matrices followed by a scale-invariant Frobenius-hyperball projection. Three LR groups:
  - **muonh** — 2D weight matrices and stacked MoE expert weights (NS + hyperball)
  - **adamh** — `lm_head` / `output_proj`. AdamH = Adam moments + Frobenius hyperball (`scale_by_adamh`); no Newton-Schulz. Uses the same LR schedule as muonh.
  - **adam** — biases, RMSNorm scales, router weights, gated-norm scales, and (via `rmsnorm_to_adam=True`) stacked RMSNorm scales
- **Multi-group optimizer**: separate optax chains per group routed by parameter path (see `experiments/grug/moe/optimizer.py:create_mask`).
- **Logit Z-Loss**: penalizes large output-logit norm to stabilize training (no router z-loss in this run).

## 3. Training

- **hardware**: v4-2048 (1,024 chips), us-central2, production priority, --no-preemptible
- **num_train_steps**: 157,500 (reduced from 300,000 to absorb the batch-size doubling and hold total tokens ~constant)
- **batch_size**: 4,096 → 8,192 sequences (step change at step 15,000 ≈ 5% of training)
- **seq_len**: 8,192
- **tokens_per_step**: 33,554,432 (~33.5M) for steps 0–14,999; 67,108,864 (~67.1M) for steps 15,000–157,499
- **total_tokens**: ~1.007 × 10¹³ (~10.07T) — 5.03 × 10¹¹ from the BS=4,096 phase (15,000 steps) + 9.56 × 10¹² from the BS=8,192 phase (142,500 steps)
- **compute_budget**: ~1.69 × 10²³ FLOPs (SWA-aware, with lm_head)
- **mixed_precision**: `params=float32,compute=bfloat16,output=bfloat16`
- **mesh**: `(replica_dcn=16, data=64, expert=1, model=1)`. EP=1, `replica_axis_size=16`. Batch is sharded across (replica_dcn, data, expert) → 1,024 shards.
- **expert_parallelism**: 1 (no expert sharding; routed experts replicated across the expert axis)
- **data**: datakit MoE mixture (`_datakit_data_config`, simulated epoching disabled) with default paloma validation sets
- **eval cadence**: every 3,000 steps; `eval_batch_size=1024`, `max_eval_batches=1` (8.39M tokens per eval)
- **checkpoints**: permanent every 3,000 steps; temp save every 60 minutes; auto-resume on preemption

## 4. Scaling Principles

All optimizer hyperparameters derived from the MuonH May Recipe refit (issue #5951, 17 finished cells across d ∈ {512, 768, 1024, 1280} × R ∈ {4, 10, 20, 60, 120, 240} × lr_mult ∈ {0.4, 0.7, 1.0, 1.3, 1.6}, R² = 0.996). Reference seq_len = 4096; conversions assume `tokens_per_batch = batch_size * seq_len`.

### Learning rates

- **muonh_lr** = `18.31 · T^(-0.395) · d^(-0.150) · sqrt(B)`, capped at 0.05
  - Equivalent to `lr_coeff · T^(-0.395) · d^(-0.150) · sqrt(tokens_per_batch)` with `lr_coeff = 0.06602` (= 18.31 / ((13/3) · 64))
  - At T = 1.007e13, d = 2560, B*S = 33.55M → **muonh_lr ≈ 0.003733**
- **adam_lr** = muonh_lr / (13/3) ≈ **0.000861**
- **adamh_lr** = muonh_lr ≈ **0.003733** (the `adamh` group — `lm_head` / `output_proj` — uses the same LR schedule as muonh, via `scale_by_adamh`: Adam moments + Frobenius hyperball, no Newton-Schulz)

### Momentum and precision

- **beta1** = 0.9062 (fixed)
- **beta2** = `clip(0.999^(tokens_per_batch / 131072), 0.95, 0.9999)`
  - At B*S = 33.55M → raw 0.999^256 ≈ 0.774 → clamped to **0.95**
- **epsilon** = `9.676e-18 · sqrt((B0 · T) / (tokens_per_batch · T0))` with B0=32, T0=1.4e9
  - At T = 1.007e13, tokens_per_batch = 33.55M → **epsilon ≈ 8.01e-19**

### Architecture scaling

- **num_layers** = `round(hidden_dim / (64 + 4·log2(hidden_dim) - 9))` = 26 for d=2560
- **num_heads** = `hidden_dim / 128` = 20
- **num_kv_heads** = largest divisor of num_heads ≤ num_heads // 4 = 5 (GQA 4:1)
- **intermediate_dim** = `ceil(hidden_dim / 2 / 128) · 128` = 1280
- **shared_expert_intermediate_dim** = hidden_dim = 2560
- **initializer_std** ≈ `0.5 / sqrt(2560)` ≈ 0.00988
- **num_experts** = 256, **num_experts_per_token** = 4 (fixed; differs from the 1e23 d=5120 run which used 64 experts)

### Schedule

- **warmup** = 1% of training (~3,000 steps)
- **decay** = none (linear schedule decays over the whole post-warmup period)
- **min_lr_ratio** = 0.05 (decay to 5% of peak)
- **lr_schedule** = linear

## 5. Outputs

- **GCS checkpoint path**: `gs://marin-us-central2/grug/moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T-<hash>/checkpoints` (pending — populated on first save)
- **WandB run link**: https://wandb.ai/marin-community/marin_moe/runs/moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T
- **iris job**: `/larry/iris-run-job-20260627-154212`
- **Final eval metrics** (pending — run in progress):
  - eval/paloma/macro_loss (projected: 2.269 @ 0.8 of training; see issue #6044)
  - eval/paloma/c4_en/bpb
  - eval/uncheatable_eval/macro_loss
- **Throughput** (pending — fill from mean over last 100 logged steps):
  - throughput/mfu
  - throughput/tokens_per_second
