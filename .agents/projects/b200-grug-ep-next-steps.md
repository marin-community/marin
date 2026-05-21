# B200 Grug EP Next Steps

## Current Decision Point

The `mlp_dim=4096`, `topk=8`, EP8 sweep found a large gap at the first completed
shape:

- `tokens=131072`, `hidden=5120`, `mlp_dim=4096`, `shared_expert_dim=0`:
  DeepEP is about `40.3%` faster than the current ring path.
- `tokens=131072`, `hidden=5120`, `mlp_dim=4096`, `shared_expert_dim=2048`:
  DeepEP is about `43.9%` faster than the current ring path.

This is no longer a parity-validation problem. It is now a concrete regression
case for the current path.

The operating scale for the next iteration is EP4 on the same global model
shape: `tokens=131072`, `hidden=5120`, `mlp_dim=4096`, `experts=64`, `topk=8`.
EP4 still shows a `14-17%` DeepEP advantage, but completes quickly enough for
probe runs and code hillclimbing.

Latest result: the existing `stream_ring` variant is within the target parity
window on the EP4 anchor: about `7.3%` slower than DeepEP without shared experts
and about `5.8%` slower with `shared_expert_dim=2048`. Treat `stream_ring` as
the EP4 development baseline before starting a fresh transport rewrite.

Follow-up: replacing stream-ring's local scatter-add combine with Sonic's
fixed-top-k Triton combine did not improve the EP4 anchor. The Sonic-combine
variant was about `1%` slower than `stream_ring`, so the remaining gap is not
primarily the local combine primitive.

EP8 follow-up: `stream_ring` is still a large improvement over the old current
path, but lands just outside the target parity window at the `131072`,
`hidden=5120`, `mlp_dim=4096`, `topk=8` anchor: about `10.4%` slower than DeepEP
without shared experts and about `10.7%` slower with `shared_expert_dim=2048`.

Current experiment: test whether JAX `ragged_all_to_all` can express the
DeepEP-like token-to-rank protocol. The benchmark-only `token_ragged_a2a`
variant uses DeepEP layout metadata for the first smoke, but performs dispatch
and combine with JAX ragged all-to-all.

## Recipe-Derived Anchor Shapes

The large-run MoE sizing table in `experiments/grug/moe/README.md` gives:

| Budget | Approx dim | Layers | Active width | Shared width | Recipe top-k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1e24` | `d7590` | `71` | `hidden/2`, rounded to 128 | `hidden` | `4` |
| `1e25` | `d11150` | `102` | `hidden/2`, rounded to 128 | `hidden` | `4` |

The benchmark-relevant rounded candidates are:

| Budget | Hidden candidates | Expert `mlp_dim` | Shared expert dim |
| --- | --- | ---: | ---: |
| `1e24` | `7552` or `7680` | `3840` | hidden |
| `1e25` | `11136` or `11264` | `5632` | hidden |

For this transport study, keep `topk=8` as the stress setting, and optionally
run recipe `topk=4` as a control after the gap is understood.

## Plan

1. Finish and log job `49779`.
   - If `262144`, `hidden=5120`, `mlp_dim=4096` fails for the current path,
     treat `131072` as the fitting debug anchor.
   - If it completes, use both `131072` and `262144` to check gap scaling.

2. Freeze a small debug matrix around the fitting bad case.
   - Primary development shape: `tokens=131072`, `hidden=5120`,
     `mlp_dim=4096`, `experts=64`, `topk=8`, `ep=4`, fwd+bwd.
   - Confirming shape: same config with `ep=8` before making headline claims.
   - Run both `shared_expert_dim=0` and `shared_expert_dim=2048`.
   - Repeat current and DeepEP once after each meaningful code change.

3. Decompose the bad case before changing kernels.
   - Run the existing probe kernels for DeepEP transport and expert-compute
     phases.
   - Capture XLA profiles for current and DeepEP on the `131072` anchor.
   - Separate time into routing/counting, token exchange, first ragged GEMM,
     activation/gating, second ragged GEMM, and output combine.

4. Promote `stream_ring` before writing new kernels.
   - Make `stream_ring` the EP4 development baseline.
   - Do not pursue the Sonic-style local combine swap further for this shape;
     it tested slightly slower than the scatter-add combine.
   - EP8 now shows a residual `~10-11%` gap rather than the old `40-44%` gap.
     Promote `stream_ring`, but keep one profile/probe task for the remaining
     EP8 transport difference.
   - First probe the `token_ragged_a2a` benchmark variant. If it works and is
     competitive, it is a cleaner path than committing directly to DeepEP
     transport FFI.

5. Test recipe-shaped anchors after the debug shape is decomposed.
   - `1e24` proxy: start with `hidden=7680`, `mlp_dim=3840`,
     `shared_expert_dim=7680`, `tokens=32768` and `65536`, `topk=8`.
   - `1e25` proxy: start with `hidden=11264`, `mlp_dim=5632`,
     `shared_expert_dim=11264`, `tokens=8192`, `16384`, and `32768`,
     `topk=8`.
   - Include `shared_expert_dim=0` for each proxy to separate shared-expert
     cost from EP transport and expert GEMM cost.

6. Only then pick the implementation track.
   - If current loses mostly in exchange/combine, port the DeepEP-style
     transport schedule directly.
   - If current loses mostly in ragged GEMM, focus on the JAX/CUTLASS/Triton
     GMM path and grouping layout.
   - If current loses mostly in backward memory pressure, reduce materialized
     all-gather state before touching GEMM kernels.

## Stop Criteria

- Short-term: current path is within `5-10%` of DeepEP on the fitting
  `131072`, `hidden=5120`, `mlp_dim=4096`, `topk=8` fwd+bwd anchor.
- Recipe-shaped: current path is within `5-10%` on the largest fitting `1e24`
  proxy and has a clear failure mode for the `1e25` proxy if it does not fit.
