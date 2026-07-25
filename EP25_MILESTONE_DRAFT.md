# EP64 MoE milestone: gather-dispatch adjoint, drop fidelity, and the 25% ledger

Draft for review. Not posted. All MFU figures use the 2.5 PFLOP/s per-GB200 bf16-dense
denominator; p50 is over the measured steps of a 120-step run on one GB200 rack (16 nodes x 4
GPUs, EP64) at the d5120 / 8-of-256 / 48-layer / seq-4096 / batch-1024 operating point.

## TL;DR

- A custom scatter-add adjoint for the fixed-a2a gather dispatch and combine gathers raises p50
  MFU from 20.61% to 24.04% on a matched 120-step A/B (+3.43pp, +16.6%), bands non-overlapping.
  It composes with leg-batched expert GEMMs, which reach 25.39% p50 on a separate 120-step run.
- The benchmark configs to date route with load balancing off. The routed experts drop 85-89% of
  assignments early in training (real router collapse, not a metric artifact). Turning QB
  load-balancing on cuts that 3.4x (0.49 -> 0.25 by step 29) at no measurable MFU cost. Whether
  QB-on reaches the <3% fidelity bar at steady state is not yet measured.
- Headline MFU is drop-insensitive: the fixed path's expert GEMMs run on capacity-sized buffers
  regardless of how many tokens drop, so the drop fraction is a fidelity metric, not a speed one.

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

Matched steps 23-29, `drop_fraction`, custom adjoint + cf1.0:

| step | QB off | QB on |
|--:|--:|--:|
| 23 | 0.893 | 0.492 |
| 26 | 0.862 | 0.362 |
| 29 | 0.846 | 0.249 |

QB on (`SCALE_MOE_QB=1`) cuts the drop fraction 3.4x and steepens its decline (slope -0.035/step
vs -0.008/step). Step time is unchanged: QB-on 24.11% vs QB-off 24.52% p50 on the two 30-step drop
jobs, within the +/-2-4pp placement variance seen across allocation draws. Capacity factor without
QB is pure cost: d4's cf1.15 QB-off leg measured 22.13% p50 (-1.91pp for +15% capacity) with drops
still 0.86 peak / 0.65 late.

Two caveats. The always-on shared expert processes every token, so training loss descends normally
(5.84 -> 5.71) despite the routed drops. And 30 steps only reaches 0.249 QB-on; whether QB-on
crosses under the ~3% reference (the known-acceptable rate at 8 buckets from a prior 1e23 run) at
steady state is not measured here.

> TODO: <3% steady-state crossing, from d4's 120-step QB-on leg (cf1.0, then cf1.15 if needed).

## 3. Sealed negatives

- Rotation ppermute decomposition of the fixed a2a: -9.46pp.
- Weight-prefetch overlap: null, scheduler-gated (LHS/auto-PGLE inert on this workload).
- Token-chunk pipelining of dispatch/FFN: -1.96pp.
- FP8 permutation-leg wire (QDQ decomposition): -2.02pp.
- TE-at-tip NCCL_EP: #3231's collective-stream pin crashes 64-GPU first execution; with the pin
  shimmed out the tip wheel is functionally the old wheel (~17% vs 18.05% a2a anchor).

## 4. Transport comparison

Ragged all-to-all with the one-shot kernel off, at the operating-point shape, QB off, single draw
(job `ep25d2-rack-ragged-120`): mean MFU ~12.38% (cumulative mean, not p50), final loss 5.708,
`drop_fraction` 0.433 at end of run. That is roughly half of fixed+adjoint's 24.04% p50 on speed,
and ragged's receiver-side capacity still drops 43% of assignments under the same QB-off router
collapse, so it is not a fidelity refuge either. QB load-balancing is the drop lever on every
transport, not the transport choice. Caveats: cumulative mean vs the p50 used elsewhere, one
allocation draw, QB off; the 0.433 is end-of-run (~step 119) against the fixed path's step-29
0.846, so the transports are not compared at matched steps here.

> TODO: ring_cute EP64 arm (running); fill the fixed / ragged / ring table at matched steps before
> the final pass.

## 5. Goal ledger

- 25% MFU at the operating point: adjoint (+3.43pp -> 24.04) + leg-batching (25.39). Met on speed.
- Fidelity: requires QB load-balancing on (drops 0.85 -> 0.25 in 30 steps at ~zero MFU cost);
  the <3% steady-state number is pending (section 2 TODO). Not yet met on fidelity.

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
