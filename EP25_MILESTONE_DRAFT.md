# EP64 MoE milestone: gather-dispatch adjoint, drop fidelity, and the 25% ledger

Draft for review. Not posted. All MFU figures use the 2.5 PFLOP/s per-GB200 bf16-dense
denominator; p50 is over the measured steps of a 120-step run on one GB200 rack (16 nodes x 4
GPUs, EP64) at the d5120 / 8-of-256 / 48-layer / seq-4096 / batch-1024 operating point.

## TL;DR

- A custom scatter-add adjoint for the fixed-a2a gather dispatch and combine gathers raises p50
  MFU from 20.61% to 24.04% on a matched 120-step A/B (+3.43pp, +16.6%), bands non-overlapping.
  It composes with leg-batched expert GEMMs, which reach 25.39% p50 on a separate 120-step run.
- Those numbers route with load balancing off, where the routed experts drop 85-89% of assignments
  early in training (real router collapse, verified exact, not a metric artifact). QB load-balancing
  costs -1.44pp and settles at ~6% drops at cf1.0 (22.60% p50); the only measured 3%-compliant
  config is QB + cf1.15 at 20.85%. The throughput frontier (24.04) and strict fidelity (20.85) sit
  ~3.2pp apart; closing that gap is the top open fidelity direction.
- Headline MFU is drop-insensitive: the fixed path's expert GEMMs run on capacity-sized buffers
  regardless of how many tokens drop, so the drop fraction is a fidelity metric, not a speed one.
  The -1.44pp QB cost is the router aux-loss, not the drops.

## 1. Speed: custom adjoint for the gather-dispatch backward

The EP64 fixed-capacity a2a builds its send buffer with an int32 assignment scatter followed by an
activation gather (`send_x = padded_x[token_sources]`), and its combine with a second gather
(`send_output[linear_indices]`). Under autodiff both transpose to a generic scatter-add, which the
XProf trace flagged as one of the two largest backward costs (the other being the SendRecv legs).

Both gathers have exact structured transposes expressible from the forward int32 index
composition: the dispatch backward is a segment-sum over each token's top-k send slots (top-k
fan-out), and the combine backward is an injective gather along the slot->assignment inverse (no
accumulation). A `custom_vjp` routes both through those forms. On the operating-point shape the
backward HLO drops from 544 scatter ops to zero (gathers only). Kernel-level gradient parity vs
autodiff holds at rtol=atol=1e-5 for the input, combine weights, and both expert weight tensors,
with identical dropped-token counts.

Matched 120-step A/B, back-to-back, checkpointing disabled:

| dispatch backward | p10 | p50 | p90 | loss @ step 119 |
|---|--:|--:|--:|--:|
| XLA autodiff (scatter-add) | 20.51 | 20.61 | 20.69 | 5.738 |
| custom adjoint (gather + segment-sum) | 23.73 | 24.04 | 24.75 | 5.711 |

+3.43pp p50, p10/p90 bands non-overlapping. Loss differs by 0.027 at step 119, consistent with
independent-run RNG divergence between two separate jobs rather than a numerics change; the kernel
test establishes semantic equivalence. Leg-batched expert GEMMs, benchmarked separately, reach
25.39% p50 on a 120-step run; the adjoint and leg-batching are independent and stack.

## 2. Fidelity: routed-token drops are high with load balancing off

The benchmark configs run `qb_routing` off, i.e. no load-balancing loss on the router. Under the
fixed capacity layout (64 sender shards x 256 experts, capacity 2048 per bucket = mean load at
cf=1.0), the routed experts drop the large majority of assignments early in training.

The drop metric (`SCALE_REPORT_DROPS`, sums the per-layer capacity-overflow count, divides by the
global assignment count `B*S*topk*L`) is exact, verified two ways: the integer cross-check at the
operating point (1,438,043,460 dropped / 1,610,612,736 total = 0.8929, matching the logged
fraction), and shard-invariance against a numpy reference of the true global drops (metric == ref,
ratio 1.000, at both 2 and 4 expert shards). An earlier "65-68%" reading elsewhere was an
overflow-rate-of-buckets semantics, not the token-drop fraction; capacity == mean makes roughly
half of all buckets overflow their tail while the dropped tail itself is a smaller fraction.

120-step frontier at the operating point, p50 MFU and `drop_fraction` at step 119, over
capacity factor x QB (QB = `SCALE_MOE_QB=1`):

| config | p50 MFU | drop @119 | loss tail |
|---|--:|--:|--:|
| QB off, cf1.0 | 24.04 | collapsed, 0.17-0.79 over run | 5.711 |
| QB off, cf1.15 | 22.13 | 0.649 | — |
| QB on, cf1.0 | 22.60 | 0.083 | 5.767 |
| QB on, cf1.15 | 20.85 | 0.037 | 5.788 |

QB is the drop lever; capacity factor is not. QB on takes the drop fraction from collapse to 0.083
at cf1.0 and 0.037 at cf1.15. The loss-tail column is the end-of-run value; at matched steps both QB
legs sit below every QB-off leg. Raising capacity without QB buys no fidelity: cf1.15 QB-off pays
-1.91pp (24.04 -> 22.13) and still drops 0.649. Prices: QB costs -1.44pp (24.04 -> 22.60), the
router aux-loss overhead; cf1.15 under QB costs another -1.75pp (22.60 -> 20.85); combined -3.19pp.
The QB cost is a router cost, not a drop cost: the fixed-path expert GEMMs run on capacity-sized
buffers regardless of drops.

A separate 30-step run measures the early trajectory: at matched steps 23-29 the fraction falls
0.85 -> 0.25 QB-on against 0.89 -> 0.85 QB-off (3.4x); the longer runs below continue that decline.
The always-on shared expert processes every token, so loss descends normally (5.84 -> 5.71) even
under QB-off collapse.

A 350-step QB-on cf1.0 run resolves the steady state (loss healthy to 3.335): drops 0.885 at step
5, 0.271 at 60, 0.175 at 119, 0.089 at 250, 0.064 at 349, tail-100 mean 7.3%, with a halving time
that grows past ~150 steps. cf1.0 QB-on levels toward ~6% and does not cross the 3% bar; the
120-step extrapolation toward <3% is falsified. Step-119 drop is draw-variable (0.175 here vs 0.083
in the frontier leg), so the steady-state ~6% is the reliable figure, not any single step-119 point.
The only measured config under a strict 3% bar is QB + cf1.15 (0.037 at step 119, 20.85% p50); cf1.0
carries ~6% steady drops at 22.60% (the ~3% reference is the known-acceptable rate at 8 buckets from
a prior 1e23 run).

rav's drop-report jobs currently log no drop metric: the per-layer count is computed but never
emitted, because the grug training loop logs `train/loss` through callbacks and never logs the
returned metrics dict. The fix (`2d4a87395`, explicit `tracker.log`) exists in two worktrees and
should land so his runs report drops.

## 3. Sealed negatives

- Rotation ppermute decomposition of the fixed a2a: -9.46pp.
- Weight-prefetch overlap: null, scheduler-gated (LHS/auto-PGLE inert on this workload).
- Token-chunk pipelining of dispatch/FFN: -1.96pp.
- FP8 permutation-leg wire (QDQ decomposition): -2.02pp.
- TE-at-tip NCCL_EP: #3231's collective-stream pin crashes 64-GPU first execution; with the pin
  shimmed out the tip wheel is functionally the old wheel (~17% vs 18.05% a2a anchor).

## 4. Transport comparison

At the operating-point shape, QB off, single draw:

| transport | MFU | drop @119 |
|---|--:|--:|
| fixed + gather + adjoint | 24.04 (p50) | collapsed |
| ragged, one-shot kernel off (`ep25d2-rack-ragged-120`) | 12.38 (mean) | 0.433 |
| ring_cute EP64 | DNF | — |

Ragged runs at roughly half of fixed+adjoint on speed, and its receiver-side capacity still drops
43% under the same QB-off collapse, so it is not a fidelity refuge; QB is the drop lever on every
transport. ring_cute did not finish: OOM at 141.79 GiB in `jit_train_step`. Its EP4/EP8 backend-ladder
wins do not transfer to e256/EP64, and fitting it would be a memory-engineering project of its own.
The ragged one-shot-off control that #7279 asked for is on record at ~12.4%. Caveats: the ragged
number is a cumulative mean against the p50 used for the fixed path, one draw each, QB off, and the
0.433 is end-of-run against the fixed path's step-29 0.846, so drops are not matched-step here.

## 5. Goal ledger

The QB-off numbers (24.04 adjoint, 25.39 with leg-batching) are bench artifacts: they route under
router collapse and only look good because MFU ignores drops. QB-on cf1.0 is 22.60% p50 but settles
at ~6% drops, above a strict 3% bar. The only measured 3%-compliant config is QB + cf1.15 at 20.85%.
So the throughput frontier (24.04) and the strict-fidelity config (20.85) sit ~3.2pp apart, and
closing that gap with faster or better-tuned QB balancing is the highest-leverage remaining fidelity
work, alongside leg-batching composition and fa4-lse.

- Speed, QB-off bench: adjoint 24.04, +leg-batching 25.39.
- Speed, QB-on cf1.0: 22.60, ~6% steady drops (not 3%-compliant).
- Speed, strict 3%-fidelity (QB cf1.15): 20.85, drops 0.037.
- Path to >=25% at strict fidelity: close the ~3.2pp gap (faster QB balancing) plus leg-batching and
  fa4-lse.

> TODO: fa4-lse A/B (d3) for the remaining path-to-25% pp.

## 6. Reproduction

- Branch `agent/ep25-d1-adjoint`, based on `rav/ep-2` @ `fe21ea495` ("Reproduce 17% EP64 MFU on
  GB200"). No pushes have happened; landing the adjoint and the drop metric needs a PR from this
  branch.
- Commits: gather-dispatch reconstruction `45ce02d20`; structured `custom_vjp` for both gathers
  `c9e30f848`; drop metric `4fbc89152` + tracker-logging fix `2d4a87395`.
- Env knobs: `SCALE_A2A_GATHER_DISPATCH=1` (gather dispatch), `SCALE_A2A_CUSTOM_ADJOINT=1` (the
  custom adjoint; requires gather dispatch), `SCALE_REPORT_DROPS=1` (drop metric),
  `SCALE_MOE_QB=1` (QB load-balancing).
- Jobs: speed A/B `ep25d1-adj-control-120-0724-1707` (autodiff) and `ep25d1-adj-custom-120-0724-2216`
  (custom adjoint); drops `ep25d1-drops-30-0724-2318` (QB off) and `ep25d1-qbon-drops-30-0725-0027`
  (QB on); leg-batching `rav/ep64-batched-expert-stability-120-v1-20260724-2353`.
