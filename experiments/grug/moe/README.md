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
