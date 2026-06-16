# grug-moe

Current best recipe for training Mixture-of-Experts models in the grug
template. Model, optimizer, train loop, and launch wiring all live in
`experiments/grug/moe/` so the MoE variant can iterate independently from
the dense base template.

## Architecture

All layers are MoE. No dense initial layers, no load-balancing loss (router
z-loss only). The architecture choices are hardcoded in
[`model.py`](./model.py); only shape/size knobs live in `GrugModelConfig`.

- **Router**: linear projection (fp32) of `hidden_dim → num_experts`. Routing
  uses a `stop_gradient` bias term that is updated each step from the previous
  step's QB-β statistics (see `train.py::_apply_qb_betas`).
- **QB load balancing**: at each step, the router computes per-expert β from
  the top-k logit threshold and stores it on `GrugTrainState.pending_qb_betas`.
  On the next step, `router_bias := -β` to push rarely-selected experts up and
  over-selected experts down. This replaces a traditional aux load-balancing
  loss — the bias mechanism does the balancing and is invisible to gradients.
  The full QB computation lives in [`model.py#L350-L385`](./model.py#L350).
- **Top-k**: `num_experts_per_token = 4` active experts out of `num_experts = 64`.
- **Combine weights**: sigmoid on unbiased router logits for the selected
  experts (not softmax).
- **Shared expert**: one always-on dense MLP per block in parallel with the
  routed experts (contributes to every token).
- **GatedNorm**: rank-128 low-rank gating on RMS-normalized input pre-attention
  and pre-MLP. Acts as a learned per-token gate over the hidden dimension.
- **XSA (Exclusive Self-Attention)**: after attention, subtract the component
  of each head's output that is parallel to its `aligned_v`. `z = y − (yᵀv / ‖v‖²)·v`
  per head. Followed by a headwise sigmoid gate.
- **RoPE**: standard rotary embeddings (no scaling by default).
- **GQA**: grouped-query attention. Default ratio in the heuristic is 4:1
  (`num_kv_heads = num_heads / 4`).
- **Sliding-window attention**: every 4th layer uses full `sliding_window`;
  others use half. Specifically, layer `i` uses the long mask iff `i % 4 == 3`.
- **Fp32 router path**: router logits cast to fp32 before top-k, softmax, and
  QB statistics.
- **Expert parallelism**: `ragged_all_to_all` or ring-based via
  `levanter.grug.grug_moe.moe_mlp` (default: ring). Default capacity factor 1.0.

## May Recipe Updates

Diff vs the v16 baseline architecture above (all changes are baked into
`model.py` and `optimizer.py`; the `GrugModelConfig` knobs to toggle them
were intentionally dropped so the recipe is the recipe):

- **Half-RoPE on every layer**: rotary embeddings applied only to the first
  half of Q/K per head (`q[..., :head_dim/2]`, `k[..., :head_dim/2]`); the
  second half is rope-free on every layer. v16 applied RoPE to the full
  head dimension.
- **PKO (Partial Key Offset)** on every-4th + last layer: shift the
  rope-free second half of K forward by 1 position, then zero at doc-start
  boundaries (using `segment_ids`), then rms-norm — `pko_first_bos_zero`
  ordering. v16 had no PKO.
- **Sliding-window pattern**: every-4th + last layer use `sliding_window=None`
  (full causal up to `max_seq_len`); other layers use
  `cfg.sliding_window` (default 2048). v16 used `cfg.sliding_window` on
  long layers and halved it on short layers, with no last-layer special case.
- **Split `w_gate` / `w_up`** in MoEMLP: stored as separate `(e, d, i)`
  tensors and concatenated on the forward pass. v16 stored them fused.
- **Routing renormalization**: sigmoid combine weights are renormalized to
  sum to 2.5 across the K selected experts (`_ROUTING_RENORM_SUM = 2.5`).
  v16 did not renormalize.
- **MuonH optimizer** (`GrugMoeMuonHConfig`, registered as
  `grug_moe_muonh_v1`): Newton-Schulz orthogonalization + Frobenius
  hyperball scale-invariant updates on the weight-matrix + GatedNorm group.
  v16 used `GrugMoeAdamHConfig` (AdamH) on the same group.
- **256 experts** at k=4 (vs v16's 64 experts at k=4): bigger expert pool,
  same active path (4 routed + shared per token).
- **Router z-loss disabled**: `router_z_loss_coef = 0.0` (v16: 0.001).
- **Final-logit z-loss disabled**: `GrugTrainerConfig.z_loss_weight = 0.0`
  (v16: 1e-4) — `logsumexp_weight` resolves to `None` so the fused
  cross-entropy never applies the logit-stabilization term.
- **No gradient clipping**: `max_grad_norm = None` on the MuonH config
  (v16 used `max_grad_norm = 1.0`).
- **Warmup 1%** of training (v16 used 10%).
- **LR refit** in `heuristic_muonh.py` (`MoeMuonHHeuristic`): refit on the
  MuonH-on-May-Recipe LR sweep (17 cells, R²=0.996, issue #5951):
  `muonh_lr = 18.31 · tokens^-0.395 · dim^-0.150 · sqrt(B)`
  (equivalently `adam_lr = 0.06602 · tokens^-0.395 · dim^-0.150 · sqrt(tpb)`).
  v16 used `adam_lr = 1.63 · tokens^-0.2813 · dim^-0.3678 · sqrt(B)`.

## Scaling heuristic

The [`MoeAdamHHeuristic`](./heuristic.py) in `heuristic.py` turns a compute
budget and a `hidden_dim` into a full `(model_config, optimizer_config,
batch_size, num_steps)` tuple. Formulas were fit on the v16 LR sweep (186 runs,
R²=0.995) and are all anchored at **seq_len = 4096**.

- **Adam LR**: `adam_lr = 1.63 · tokens^(-0.2813) · dim^(-0.3678) · sqrt(B)`
- **AdamH LR**: `lr = (13/3) · adam_lr`
- **Compute budget**: `C = 3 · flops_per_token(no_lm_head) · tokens`
- **Epsilon**: `epsilon_base · sqrt(r0/r)` where `r = (B·T0)/(B0·T)`
- **Beta1**: fixed at 0.9062
- **Beta2**: `clip(0.999^(B/B0), 0.95, 0.9999)` (constant-token half-life)
- **Layer count**: `num_layers ≈ dim / (64 + 4·log2(dim) − 9)` (rounded)
- **GQA**: largest divisor of `num_heads ≤ num_heads / 4`

`build_from_heuristic(budget, hidden_dim, target_steps=2**14)` is the main
entry point — `launch.py` uses it to produce the baseline step. Callers that
want full manual control pass `GrugModelConfig` and `GrugMoeAdamHConfig`
directly to `GrugMoeLaunchConfig`.

## v16 isoflop sweep

From the v16 sweep (`group=isoflop-moe-v16` on wandb, project `dial_moe`).
See [issue #4447](https://github.com/marin-community/marin/issues/4447) for
the full sweep context, per-cell results, and extrapolation tables. All runs
use the architecture described above, QB routing, shared expert, GQA 4:1,
seq_len 4096. The sweep tested multiple hidden dims at each compute budget
(1e18–3e20) to find the optimal model size per budget.

Scaling laws fit on the v16 sweep optima:

- `N*(C) = 1.09e-2 · C^0.535`
- `T*(C) = 1.60e+1 · C^0.464`
- Paloma macro: `1.6 + 95.18 · C^(-0.0941)` (L∞ pinned at 1.6)

Projections:

| Budget | Projected macro |
|--------|-----------------|
| 1e21   | 2.606           |
| 1e23   | 2.252           |

The **measured** 1e21 d2560-v2 run came in at macro **2.599**.

## Compute-optimal baseline

Using `N*(C)` from the isoflop sweep, we inverted to find the optimal compute
budget for each hidden dim, then ran each at its predicted optimal budget. These
are the baseline runs that ablation experiments compare against.

### May Recipe (drop-1e18 fit, issue #6074) — current baseline

Reference runs on **v4-32 us-central2 with EP=2**, MuonH optimizer
(`muonh_lr` from `heuristic_muonh.MoeMuonHHeuristic.build_muonh_config`), 1pct-noclip
schedule, no permanent step-interval checkpoints.

| Budget   | Dim    | Layers | bs  | Steps    | Tokens  | Paloma macro | v4-32 tok/s | Runtime | Run |
|----------|--------|--------|-----|----------|---------|--------------|-------------|---------|-----|
| 3.82e17  | d512   | 6      | 32  | 10,980   | 1.44e9  | **3.5438**   | 530,704     | 1.0h    | [marin-big-run-moe_may_compute_opt_d512](https://wandb.ai/marin-community/marin_moe/runs/marin-big-run-moe_may_compute_opt_d512) |
| 2.81e18  | d768   | 8      | 64  | 16,875   | 4.42e9  | **3.2330**   | 357,696     | 4.0h    | [marin-big-run-moe_may_compute_opt_d768](https://wandb.ai/marin-community/marin_moe/runs/marin-big-run-moe_may_compute_opt_d768) |
| 1.16e19  | d1024  | 11     | 128 | 16,080   | 8.43e9  | **3.0297**   | 255,040     | 11.5h   | [marin-big-run-moe_may_compute_opt_d1024](https://wandb.ai/marin-community/marin_moe/runs/marin-big-run-moe_may_compute_opt_d1024) |
| 3.46e19  | d1280  | 13     | 256 | 14,325   | 1.50e10 | **2.8963**   | 192,300     | 23.5h   | [marin-big-run-moe_may_compute_opt_d1280](https://wandb.ai/marin-community/marin_moe/runs/marin-big-run-moe_may_compute_opt_d1280) |

Fitted scaling law on these 4 cells (`L_inf=1.6` pinned):

```
loss(C) = 1.6 + 88.90 · C^-0.0941
```

Same exponent α as the v16 README curve, ~2.12× equal-TPS compute-equivalent
speedup vs v16 baseline at every budget.

### v16 baseline (historical reference)

Older AdamH MoE sweep (`group=isoflop-moe-v16` in `marin-community/dial_moe`).
Kept here because the v16 scaling law (`loss = 1.6 + 95.18 · C^-0.0941`) is the
reference curve in `experiments/grug/moe/agent.md` for gate-1 / gate-2 effective
speedup calculations.

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
- [`heuristic.py`](./heuristic.py) — scaling heuristics (LR formula coefficients,
  depth/width formula, GQA ratio, per-batch-size epsilon/beta2 scaling).
- [`optimizer.py`](./optimizer.py) — optimizer internals (AdamH components,
  parameter-group partitioning, per-group learning rates, weight decay).

Some discretionary factors may influence the promotion decision even when the
loss criteria are met — for example, impact on training memory footprint,
inference latency / KV-cache size, serving compatibility, or interaction effects with other promotable changes.

## Large run model sizing

Conservative sizing for large runs using equal compute allocation between
parameters and tokens (exponent fixed to 0.5):

```
N*(C) = 0.0543 · C^0.5    (active params, no lm_head)
T*(C) = 3.290  · C^0.5    (tokens)
token:active_param ratio = 60.6 (constant across all scales)
total_params ≈ 8 × active_params  (for E=64, K=4 with shared expert)
```

These formulas fix the exponent to 0.5 for both N and T (equal scaling),
fit on v16 isoflop sweep parabola optima. The free-fit exponents
(N: 0.546, T: 0.464) are slightly param-heavy, but the fixed 0.5 is more
conservative and easier to reason about.

| Budget | Active params | Total params | Tokens  | Predicted macro | Approx dim | Layers |
|--------|--------------|-------------|---------|-----------------|------------|--------|
| 1e21   | 1.7B         | ~14B        | 104B    | 2.606           | d2400      | 24     |
| 1e22   | 5.4B         | ~43B        | 329B    | 2.410           | d3520      | 35     |
| 1e23   | 17.2B        | ~137B       | 1.0T    | 2.252           | d5170      | 50     |
| 1e24   | 54.3B        | ~434B       | 3.3T    | 2.125           | d7590      | 71     |
| 1e25   | 171.7B       | ~1.4T       | 10.4T   | 2.023           | d11150     | 102    |

Predicted macro uses `loss(C) = 1.6 + 95.18 · C^(-0.0941)`.

## Files

- [`model.py`](./model.py) — `GrugModelConfig` + transformer implementation.
- [`optimizer.py`](./optimizer.py) — `GrugMoeAdamHConfig` wrapper on top of
  `AdamHConfig` with expert-param-group awareness.
- [`train.py`](./train.py) — `GrugTrainState`, `train_step`, `_apply_qb_betas`,
  `run_grug` (dispatches a Fray job).
- [`heuristic.py`](./heuristic.py) — `MoeAdamHHeuristic` and
  `build_from_heuristic` entry point.
- [`launch.py`](./launch.py) — `GrugMoeLaunchConfig`, baseline `ExecutorStep`,
  and `executor_main` wiring.
- [`adamh.py`](./adamh.py) — shared AdamH utilities.
- [`agent.md`](./agent.md) — agent guide for running ablation experiments on Iris.
