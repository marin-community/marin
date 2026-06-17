# grug-moe

Current best recipe for training Mixture-of-Experts models in the grug
template. Model, optimizer, train loop, and launch wiring all live in
`experiments/grug/moe/` so the MoE variant can iterate independently from
the dense base template.

## Architecture

Describes the May Recipe defaults (V2) — what the model + optimizer + schedule
actually run as on `moe_may_pr` today. The architecture choices are hardcoded
in [`model.py`](./model.py); only shape/size knobs live in `GrugModelConfig`.

All layers are MoE. No dense initial layers, no auxiliary load-balancing loss
(QB bias does the balancing), no z-losses (router or final-logit).

**Experts.**
- `num_experts = 256` routed pool; `num_experts_per_token = 4` active per token.
- One always-on **shared** dense MLP per block, in parallel with the routed
  experts (contributes to every token).
- **MoEMLP** stores per-expert `w_gate` and `w_up` as separate `(e, d, i)`
  tensors and concatenates on the forward pass.

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
- **PKO (Partial Key Offset)** on the every-4th + last "long" layers: shift
  the rope-free second half of K back by one position, zero at document
  starts (`segment_ids` change), then rms-norm. Short layers skip PKO.
- **Sliding window**: long layers run full causal attention
  (`sliding_window = None`). Short layers run `cfg.sliding_window` (default
  2048).
- **XSA (Exclusive Self-Attention)**: after attention, subtract the component
  of each head's output parallel to its `aligned_v`: `z = y − (yᵀv / ‖v‖²)·v`
  per head. Followed by a headwise sigmoid gate.

**Norms.**
- **GatedNorm**: rank-128 low-rank gate on RMS-normalised input pre-attention
  and pre-MLP. Acts as a learned per-token gate over the hidden dimension.

**Optimizer + schedule** ([`heuristic_v2.py`](./heuristic_v2.py) +
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

**Other.**
- **Expert parallelism**: `ragged_all_to_all` or ring-based via
  `levanter.grug.grug_moe.moe_mlp` (default: ring). Default capacity factor 1.0.

## What changed from V1

V1 was the first Marin MoE formula, used on the May 2026 1e23
``129B / A29B MoE V1`` hero run (`heuristic_v1.MoeHeuristicV1`, AdamH on the
v16 architecture). Everything below is what moved between V1 and the V2
defaults documented in the Architecture section above.

- **Optimizer**: MuonH (V2) vs AdamH (V1) on the matrix + GatedNorm group.
- **Expert pool**: 256 experts (V2) vs 64 experts (V1), K=4 in both — bigger
  pool, same active path.
- **Routing renormalisation**: sigmoid combine weights now renorm to sum 2.5
  (V2). V1 did not renormalise.
- **Split `w_gate` / `w_up`** in MoEMLP (V2). V1 stored them fused.
- **Half-RoPE on every layer** (V2). V1 applied RoPE to the full head_dim.
- **PKO** on every-4th + last layer (V2). V1 had no PKO.
- **Sliding-window pattern**: long layers fully causal, short layers at
  `cfg.sliding_window` (V2). V1 used `cfg.sliding_window` on long layers and
  halved it on short layers, with no last-layer special case.
- **Router z-loss off** (V2: `router_z_loss_coef = 0.0`). V1: `0.001`.
- **Final-logit z-loss off** (V2: `GrugTrainerConfig.z_loss_weight = 0.0`).
  V1: `1e-4`.
- **No gradient clipping** (V2: `max_grad_norm = None`). V1: `1.0`.
- **1% warmup** (V2). V1: 10%.
- **LR refit** (`heuristic_v2.py`, `MoeHeuristicV2`):
  V1: `adam_lr = 1.63 · tokens^-0.2813 · dim^-0.3678 · sqrt(B)`.
  V2: `adam_lr = 0.06602 · tokens^-0.395 · dim^-0.150 · sqrt(B)`.

## Scaling heuristic

Two heuristic classes turn `(budget, hidden_dim)` into model + optimizer
hyperparameters. Both expose `build_model_config(hidden_dim, seq_len)` and
`build_optimizer_config(batch_size, tokens, hidden_dim, seq_len)`.

- **[`MoeHeuristicV1`](./heuristic_v1.py)** — original v16 AdamH fit
  (186 runs, R²=0.995). Used on the May 2026 1e23 `129B / A29B MoE V1` hero
  run; kept as the reference curve for `agent.md` effective-speedup
  comparisons against v16.
- **[`MoeHeuristicV2`](./heuristic_v2.py)** — May Recipe refit, MuonH
  optimizer (issue #5951; 17 cells, R²=0.996). **Current default** for
  compute-optimal cells and ablation comparisons.

LR scaling differs between the two; everything else (compute-budget convention,
epsilon, beta1/beta2, layer count, GQA) is shared. All formulas anchor at
**seq_len = 4096** and write batch effects in terms of
`tokens_per_batch = batch_size · seq_len`.

- **V1 LR**: `adam_lr = 1.63 · tokens^-0.2813 · dim^-0.3678 · sqrt(B)`,
  `adamh_lr = (13/3) · adam_lr`.
- **V2 LR**: `adam_lr = 0.06602 · tokens^-0.395 · dim^-0.150 · sqrt(B)`,
  `muonh_lr = (13/3) · adam_lr`.
- **Compute budget**: `C = 3 · flops_per_token(no_lm_head) · tokens`.
- **Epsilon**: `epsilon_coeff · sqrt(tokens / tokens_per_batch)`.
- **Beta1**: fixed at 0.9062.
- **Beta2**: `clip(0.999^(tpb/131072), 0.95, 0.9999)` (constant-token half-life).
- **Layer count**: `num_layers ≈ dim / (64 + 4·log2(dim) − 9)` (rounded).
- **GQA**: largest divisor of `num_heads ≤ num_heads / 4`.

V1 also exposes a top-level `build_from_heuristic(budget, hidden_dim,
target_steps=2**14)` helper that returns the full `(model, optimizer,
batch_size, num_steps)` tuple (used by `canary_ferry.py`). For V2, instantiate
`MoeHeuristicV2()` and call `build_model_config` / `build_optimizer_config`
directly — `launch.py` does this for the baseline.

## Compute-optimal baseline

For each hidden dim we picked a compute budget at the parabola optimum of
its isoflop curve (V2 / May Recipe: drop-1e18 fit, issue #6074; V1 / v16:
issue #4447) and ran the cell at that budget. These are the baseline runs
that ablation experiments compare against.

### May Recipe (drop-1e18 fit, issue #6074) — current baseline

Reference runs on **v4-32 us-central2 with EP=1**, MuonH optimizer
(`muonh_lr` from `heuristic_v2.MoeHeuristicV2.build_optimizer_config`), 1pct-noclip
schedule, no permanent step-interval checkpoints.

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
- [`heuristic_v2.py`](./heuristic_v2.py) — scaling heuristics (LR formula
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
- [`heuristic_v1.py`](./heuristic_v1.py) — `MoeHeuristicV1` (AdamH, v16 fit)
  and `build_from_heuristic` entry point.
- [`heuristic_v2.py`](./heuristic_v2.py) — `MoeHeuristicV2` (MuonH, May
  Recipe refit). Current default.
- [`launch.py`](./launch.py) — `GrugMoeLaunchConfig`, baseline `ExecutorStep`,
  and `executor_main` wiring.
- [`adamh.py`](./adamh.py) — shared AdamH utilities.
- [`agent.md`](./agent.md) — agent guide for running ablation experiments on Iris.
