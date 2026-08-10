# grug-moe

Current best recipe for training Mixture-of-Experts models in the grug
template. Model, optimizer, train loop, and launch wiring all live in
`experiments/grug/moe/` so the MoE variant can iterate independently from
the dense base template.

## Architecture

**Experts.**
- `num_experts = 256` routed pool; `num_experts_per_token = 4` active per token.
- One always-on **shared** dense MLP per block, in parallel with the routed
  experts (contributes to every token).

**Router.**
- Linear projection of `hidden_dim → num_experts` in **fp32** (cast back at
  the end). Top-k, softmax, and QB statistics all run in fp32.
- A `stop_gradient` bias term gets added to the router logits before top-k;
  the bias is updated each step from the previous step's QB-β statistics
  (see `train.py::_apply_qb_betas`).
- **QB load balancing**: per-expert β is the top-k logit threshold averaged
  across the batch. On the next step, `router_bias := -β`, pushing
  rarely-selected experts up and over-selected experts down. Replaces an aux
  load-balancing loss — the bias mechanism is invisible to gradients.
- **Combine weights**: sigmoid on the *unbiased* router logits of the K
  selected experts, then **renormalised to sum to 2.5**
  (`_ROUTING_RENORM_SUM`).

**Attention** (`CausalSelfAttention.__call__`).
- **GQA**: default ratio 4:1 (`num_kv_heads = num_heads / 4`).
- **Half-RoPE**: rotary embeddings applied only to the first half of Q/K per
  head (`q[..., :head_dim/2]`, `k[..., :head_dim/2]`); the second half is
  rope-free on every layer.
- **PKO (Partial Key Offset)** is wired up but disabled by default
  (`disable_pko=True`). When enabled it would shift the rope-free second half
  of K back by one position on every-4th + last "long" layers (zero at doc
  starts, then rms-norm); short layers always skip PKO.
- **NoPE on long layers**: every-4th + last layers run with rotary embedding
  skipped entirely (`disable_long_rope=True`); short layers keep half-RoPE.
- **Sliding window**: long layers run full causal attention
  (`sliding_window = None`). Short layers run `cfg.sliding_window` (default
  2048).

To reduce risk of long-context extension, we have excluded PKO and use NoPE on
long layers.
- **XSA (Exclusive Self-Attention)**: after attention, subtract the component
  of each head's output parallel to its `aligned_v`: `z = y − (yᵀv / ‖v‖²)·v`
  per head. Followed by a headwise sigmoid gate.

**Norms.**
- **GatedNorm**: rank-128 low-rank gate on RMS-normalised input pre-attention
  and pre-MLP. Acts as a learned per-token gate over the hidden dimension.

**Optimizer + schedule** ([`heuristic.py`](./heuristic.py) +
[`optimizer.py`](./optimizer.py)).
- **MuonH** (`GrugMoeMuonHConfig`, registered as `grug_moe_muonh_v1`):
  Newton-Schulz orthogonalisation + Frobenius-hyperball scale-invariant
  updates on the matrix + GatedNorm group.
  - `adamh` (lm_head only).
  - `adam` (token_embed, router, attn_gate, biases, 1-D norm weights).
- **No gradient clipping** (`max_grad_norm = None`).
- **1% warmup**, linear decay to 0, `min_lr_ratio = 0`.
- LR scaling (fit on the May Recipe sweep, issue #5951, R²=0.996):
  `muonh_lr = 18.31 · tokens^-0.395 · dim^-0.150 · sqrt(B)`
  (equivalently `adam_lr = 0.06602 · tokens^-0.395 · dim^-0.150 · sqrt(tpb)`).

**Loss.**
- Cross-entropy on the next-token logits.
- **Final-logit z-loss** (`GrugTrainerConfig.z_loss_weight = 1e-4` by
  default): adds `z_loss_weight · mean(logsumexp(logits)²)` to stabilise the
  lm-head softmax.
- **Router z-loss off** by default (`router_z_loss_coef = 0.0`).
- No auxiliary load-balancing loss; QB router bias does the balancing.

**Other.**
- **Expert parallelism**: `ragged_all_to_all` or ring-based via
  `levanter.grug.grug_moe.moe_mlp` (default: ring). Default capacity factor 1.0.

## Scaling heuristic

[`MoeHeuristic`](./heuristic.py) turns `(budget, hidden_dim)` into model +
optimizer hyperparameters via `build_model_config(hidden_dim, seq_len)` and
`build_optimizer_config(batch_size, tokens, hidden_dim, seq_len)`. May Recipe
refit on the MuonH LR sweep (issue #5951; 17 cells, R²=0.996). All formulas
anchor at **seq_len = 4096** and write batch effects in terms of
`tokens_per_batch = batch_size · seq_len`.

- **LR**: `adam_lr = 0.06602 · tokens^-0.395 · dim^-0.150 · sqrt(B)`,
  `muonh_lr = (13/3) · adam_lr`.
- **Compute budget**: `C = 3 · flops_per_token(no_lm_head) · tokens`.
- **Epsilon**: `epsilon_coeff · sqrt(tokens / tokens_per_batch)`.
- **Beta1**: fixed at 0.9062.
- **Beta2**: `clip(0.999^(tpb/131072), 0.95, 0.9999)` (constant-token half-life).
- **Layer count**: `num_layers ≈ dim / (64 + 4·log2(dim) − 9)` (rounded).
- **GQA**: largest divisor of `num_heads ≤ num_heads / 4`.

For the earlier AdamH-tuned heuristic (64 experts, used on the May 2026 1e23
hero run), see the pre-rename
[`heuristic.py` on main](https://github.com/marin-community/marin/blob/8586719b524bf7743ec5034403c7e834505fe73e/experiments/grug/moe/heuristic.py).

## Compute-optimal baseline

For each hidden dim we picked a compute budget at the parabola optimum of
its isoflop curve (V2 / May Recipe: drop-1e18 fit, issue #6074; V1 / v16:
issue #4447) and ran the cell at that budget. These are the baseline runs
that ablation experiments compare against.

### May Recipe (drop-1e18 fit, issue #6074) — current baseline

Reference runs on **v4-32 us-central2 with EP=1**, MuonH optimizer
(`muonh_lr` from `heuristic.MoeHeuristic.build_optimizer_config`), 1pct-noclip
schedule, no permanent step-interval checkpoints.

The Paloma macro numbers below were measured under the **previous defaults**:
`seq_len = 4096`, PKO on for long layers, partial half-RoPE on every layer
(no NoPE on long layers), and final-logit z-loss off (`z_loss_weight = 0`).

| Budget   | Dim    | Layers | bs  | Steps    | Tokens  | Paloma macro | v4-32 tok/s | Runtime | Run |
|----------|--------|--------|-----|----------|---------|--------------|-------------|---------|-----|
| 3.82e17  | d512   | 6      | 32  | 10,980   | 1.44e9  | **3.5422**   | 433,986     | 1.25h   | [moe_may_compute_opt_d512_ep1](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d512_ep1) |
| 2.81e18  | d768   | 8      | 64  | 16,875   | 4.42e9  | **3.2273**   | 294,726     | 4.83h   | [moe_may_compute_opt_d768_ep1](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d768_ep1) |
| 1.16e19  | d1024  | 11     | 128 | 16,080   | 8.43e9  | **3.0195**   | 219,720     | 11.72h  | [moe_may_compute_opt_d1024_ep1](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d1024_ep1) |
| 3.46e19  | d1280  | 13     | 256 | 14,325   | 1.50e10 | **2.8857**   | 171,912     | 25.68h  | [moe_may_compute_opt_d1280_ep1](https://wandb.ai/marin-community/marin_moe/runs/moe_may_compute_opt_d1280_ep1) |

Fitted scaling law on these 4 cells (`L_inf=1.6` pinned, α pinned to the v16
exponent of 0.0941):

```
loss(C) = 1.6 + 88.32 · C^-0.0941
```

~2.22× equal-TPS compute-equivalent speedup vs v16 baseline at every budget.

### v16 baseline (historical reference)

Older AdamH MoE sweep (`group=isoflop-moe-v16` in `marin-community/dial_moe`,
issue #4447). Kept here because the v16 scaling law
(`loss = 1.6 + 95.18 · C^-0.0941`) is the reference curve in
`experiments/grug/moe/agent.md` for gate-1 / gate-2 effective speedup
calculations. Empirically validated at 1e21: predicted macro 2.606, measured
d2560-v2 run came in at **2.599**.

| Budget   | Dim      | Layers | Paloma macro | Tokens  | v5p-8 avg tok/s | v5p-8 runtime | Run |
|----------|----------|--------|-------------|---------|-----------------|---------------|-----|
| 2.19e17  | d512     | 6      | **3.8104**  | 8.37e8  | 405,630         | 0.6h          | [moe-v16-compute-opt-d512-2.19e+17](https://wandb.ai/marin-community/dial_moe/runs/moe-v16-compute-opt-d512-2.19e+17) |
| 1.70e18  | d768     | 8      | **3.4339**  | 2.71e9  | 273,532         | 2.8h          | [moe-v16-compute-opt-d768-1.70e+18](https://wandb.ai/marin-community/dial_moe/runs/moe-v16-compute-opt-d768-1.70e+18) |
| 9.00e18  | d1024    | 11     | **3.1605**  | 6.63e9  | 175,165         | 10.5h         | [moe-v16-compute-opt-d1024-9.00e+18](https://wandb.ai/marin-community/dial_moe/runs/moe-v16-compute-opt-d1024-9.00e+18) |
| 2.83e19  | d1280    | 13     | **3.0065**  | 1.24e10 | 128,277         | 26.8h         | [moe-v16-compute-opt-d1280-2.83e+19](https://wandb.ai/marin-community/dial_moe/runs/moe-v16-compute-opt-d1280-2.83e+19) |

## Promotion criteria

Changes can be promoted to this recipe when they demonstrate some combination
of the following. Typically point 1 is sufficient.

1. **Passes gate 1 and gate 2** as defined in [`agent.md`](./agent.md) —
   effective speedup > 1 at all compute-optimal baseline points, and lower
   projected macro_loss at 1e21 and 1e23.
2. **Low curvature around the minimum of each isoflop curve** — stable
   behavior across under- and over-trained regimes, in particular the
   overtrained regime.
3. **Stability and scaling improvements** — better routing balance, controlled
   norm growth, fewer activation outliers. Anything that makes the recipe more
   robust to scaling, even if loss is neutral at small scale.

Most promotable changes will land in one of three files:

- [`model.py`](./model.py) — architecture tweaks (routing, norms, attention,
  activation functions, expert layout, etc.).
- [`heuristic.py`](./heuristic.py) — scaling heuristics (LR formula
  coefficients, depth/width formula, GQA ratio, per-batch-size epsilon/beta2
  scaling).
- [`optimizer.py`](./optimizer.py) — optimizer internals (MuonH / AdamH
  components, parameter-group partitioning, per-group learning rates,
  weight decay).

Some discretionary factors may influence the promotion decision even when the
loss criteria are met — for example, impact on training memory footprint,
inference latency / KV-cache size, serving compatibility, or interaction effects with other promotable changes.

## Files

- [`model.py`](./model.py) — `GrugModelConfig` + transformer implementation.
- [`optimizer.py`](./optimizer.py) — `GrugMoeAdamHConfig` and
  `GrugMoeMuonHConfig` wrappers with expert-param-group awareness.
- [`train.py`](./train.py) — `GrugTrainState`, `train_step`, `_apply_qb_betas`,
  `run_grug` (dispatches a Fray job).
- [`heuristic.py`](./heuristic.py) — `MoeHeuristic` (MuonH, May Recipe
  refit) and `build_from_heuristic` entry point. Current default.
- [`launch.py`](./launch.py) — `GrugMoeLaunchConfig`, the `grug_moe_baseline()`
  lazy `Checkpoint`, and `StepRunner` wiring.
- [`launch_cw_scale.py`](./launch_cw_scale.py) — CoreWeave scale-test launcher.
- [`adamh.py`](./adamh.py) — shared AdamH utilities.
- [`test_optimizer.py`](./test_optimizer.py) — unit tests for the AdamH
  parameter-group mask.
- [`agent.md`](./agent.md) — agent guide for running ablation experiments on Iris.
