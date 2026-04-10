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

## v16 isoflop sweep: best runs per compute budget

From the v16 sweep (`group=isoflop-moe-v16` on wandb, project `dial_moe`).
See [issue #4447](https://github.com/marin-community/marin/issues/4447) for
the full sweep context, per-cell results, and extrapolation tables. Rankings
below are by **Paloma macro loss** at the final eval step. All runs use the
architecture described above, QB routing, shared expert, GQA 4:1, seq_len
4096. Budget → best run:

| Budget | Best dim | Layers | Paloma macro | c4_en BPB | Run |
|--------|----------|--------|-------------|-----------|-----|
| 1e18   | d768     | 10     | **3.5273**  | 1.0658 | [isoflop-moe-v16-1e+18-d768](https://wandb.ai/marin-community/dial_moe/runs/isoflop-moe-v16-1e%2B18-d768) |
| 3e18   | d768     | 10     | **3.3398**  | 1.0122 | [isoflop-moe-v16-3e+18-d768](https://wandb.ai/marin-community/dial_moe/runs/isoflop-moe-v16-3e%2B18-d768) |
| 1e19   | d1024    | 11     | **3.1494**  | 0.9541 | [isoflop-moe-v16-1e+19-d1024](https://wandb.ai/marin-community/dial_moe/runs/isoflop-moe-v16-1e%2B19-d1024) |
| 3e19   | d1536    | 14     | **3.0066**  | 0.9123 | [isoflop-moe-v16-3e+19-d1536-v2](https://wandb.ai/marin-community/dial_moe/runs/isoflop-moe-v16-3e%2B19-d1536-v2) |
| 1e20   | d1536    | 14     | **2.8509**  | 0.8665 | [isoflop-moe-v16-1e+20-d1536-v2](https://wandb.ai/marin-community/dial_moe/runs/isoflop-moe-v16-1e%2B20-d1536-v2) |
| 3e20   | d2048    | 18     | **2.7222**  | 0.8289 | [isoflop-moe-v16-3e+20-d2048](https://wandb.ai/marin-community/dial_moe/runs/isoflop-moe-v16-3e%2B20-d2048) |

Derived scaling laws (fit on 1e18–3e20 optima):

- `N*(C) = 1.09e-2 · C^0.535`
- `T*(C) = 1.60e+1 · C^0.464`
- Paloma macro: `1.6 + 95.18 · C^(-0.0941)` (irreducible L∞ pinned to 1.6)
- c4_en BPB: `0.4814 + 25.97 · C^(-0.0915)` (free-fit asymptote)

Projections:

| Budget | Projected macro | Projected c4_en BPB |
|--------|-----------------|---------------------|
| 1e21   | 2.606           | 0.7923              |
| 1e23   | 2.252           | 0.6854              |

The **measured** 1e21 d2560-v2 run came in at macro **2.599** / bpb **0.7923**.

## Promotion criteria

Changes can be promoted to this recipe when they demonstrate:

1. **Lower loss at the same runtime** on the rungs of the 1e18 – 3e20 compute
   ladder (measured on the optima above, at the same token count / step count).
2. **Lower projected c4_en BPB at 1e21 and 1e23 FLOPs**, using the scaling-law
   fit above (L∞ pinned at 1.6 for Paloma macro). Re-fit the power law on the
   candidate's ladder and compare projections head-to-head.

Most promotable changes will land in one of three files:

- [`model.py`](./model.py) — architecture tweaks (routing, norms, attention,
  activation functions, expert layout, etc.).
- [`heuristic.py`](./heuristic.py) — scaling heuristics (LR formula coefficients,
  depth/width formula, GQA ratio, per-batch-size epsilon/beta2 scaling).
- [`optimizer.py`](./optimizer.py) — optimizer internals (AdamH components,
  parameter-group partitioning, per-group learning rates, weight decay).

Some discretionary factors may influence the promotion decision even when the
loss criteria are met — for example, impact on training memory footprint,
inference latency / KV-cache size, serving compatibility, or interactions
with unrelated in-flight work.

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
