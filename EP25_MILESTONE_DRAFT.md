# EP64 MoE milestone: gather-dispatch adjoint, drop fidelity, and the 25% ledger

Draft for review. Not posted. All MFU figures use the 2.5 PFLOP/s per-GB200 bf16-dense
denominator; p50 is over the measured steps of a 120-step run on one GB200 rack (16 nodes x 4
GPUs, EP64) at the d5120 / 8-of-256 / 48-layer / seq-4096 / batch-1024 operating point.

Measurement caveat: at fixed step accounting a run that drops more assignments reads higher MFU,
because dropped assignments gather the zero pad row and do less real work. The g=2 gain probe below
posted 23.4% while dropping 68%. MFU is comparable only within a matched drop regime. The section 1
adjoint A/B is matched (both legs QB off, identical routing and drop counts), so its +3.43pp holds;
the QB-off vs QB-on MFU gap is cross-regime, so the -1.44pp QB cost is an upper bound on the router
overhead, not a clean measurement.

## TL;DR

- A custom scatter-add adjoint for the fixed-a2a gather dispatch and combine gathers raises p50
  MFU from 20.61% to 24.04% on a matched 120-step A/B (+3.43pp, +16.6%), bands non-overlapping.
  It composes with leg-batched expert GEMMs, which reach 25.39% p50 on a separate 120-step run.
- Those numbers route with load balancing off, where the routed experts drop 85-89% of assignments
  early in training (real router collapse, verified exact, not a metric artifact). QB load-balancing
  costs -1.44pp and settles at ~6% drops at cf1.0 (22.60% p50). No configuration has yet been
  measured under a strict 3% bar: QB + cf1.15 read 3.7% when last measured and its steady state was
  never taken, and same-step spill (section 6) reaches 3.66%. Closing that gap is the top open
  fidelity direction.
- MFU is not comparable across drop regimes: the expert GEMMs run on fixed capacity-sized buffers,
  but a run that drops more still reads higher MFU because the dropped rows carry no real work. The
  drop fraction is a fidelity metric, and cross-regime MFU gaps (like the -1.44pp QB cost) are upper
  bounds, not clean measurements.

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
-1.91pp (24.04 -> 22.13) and still drops 0.649. Prices, as upper bounds since these are different
drop regimes (see the measurement caveat): QB costs at most -1.44pp (24.04 -> 22.60); cf1.15 under
QB costs another -1.75pp (22.60 -> 20.85); combined -3.19pp.

A separate 30-step run measures the early trajectory: at matched steps 23-29 the fraction falls
0.85 -> 0.25 QB-on against 0.89 -> 0.85 QB-off (3.4x); the longer runs below continue that decline.
The always-on shared expert processes every token, so loss descends normally (5.84 -> 5.71) even
under QB-off collapse.

A 350-step QB-on cf1.0 run resolves the steady state (loss healthy to 3.335): drops 0.885 at step
5, 0.271 at 60, 0.175 at 119, 0.089 at 250, 0.064 at 349, tail-100 mean 7.3%, with a halving time
that grows past ~150 steps. cf1.0 QB-on levels toward ~6% and does not cross the 3% bar; the
120-step extrapolation toward <3% is falsified. Step-119 drop is draw-variable (0.175 here vs 0.083
in the frontier leg), so the steady-state ~6% is the reliable figure, not any single step-119 point.
QB + cf1.15 is 3%-compliant, at 2.60% measured as a 100-step tail of a 350-step run (20.416% p50).
The 0.037 in the table above is the same configuration read at step 119 of a 120-step run, i.e. at
end of anneal and in a different drop regime; at 3.7% it does not itself clear the bar. The two are
consistent measurements of one configuration, not a disagreement — fewer drops means more real work
at the same step accounting, which reads as lower MFU, so the 350-step number is both the compliant
one and the slower one. Quote cf1.15 with its run length and drop level attached. cf1.0 carries
~7.1% steady drops at 22.062%. The ~3% reference is the known-acceptable rate at 8 buckets from a
prior 1e23 run.

grug's QB is an implicit proportional controller: it applies a 1x router-bias residual per step, not
DeepSeek's +/-gamma integral accumulation. Doubling the gain (g=2, `SCALE_QB_GAIN`, commit
`58c9a19eb`) is a categorical negative: drops sit at 0.67-0.72 for all 350 steps (an overshoot limit
cycle) and loss ends +0.091 worse, far outside draw variance. g=1 plateaus at ~6% and g=2 diverges,
so the ~6% residual is not global-bias under-correction. The leading hypothesis is sender-local
bucket hotspots: at 64 senders x 256 experts the per-bucket capacity is enforced sender-locally, and
global router bias cannot see or fix a single sender overloading one expert. That also explains
cf1.15 roughly halving the drops (more sender-local headroom). Unprobed follow-ups: damped gain
(g<1), DeepSeek-style integral accumulation, and per-sender bias -- the last is kernel-level and the
only one aimed at the hypothesized cause.

rav's drop-report jobs currently log no drop metric: the per-layer count is computed but never
emitted, because the grug training loop logs `train/loss` through callbacks and never logs the
returned metrics dict. The fix (`2d4a87395`, explicit `tracker.log`) exists in two worktrees and
should land so his runs report drops.

## 3. Sealed and below-bar results

- Rotation ppermute decomposition of the fixed a2a: -9.46pp.
- Weight-prefetch overlap: null, scheduler-gated (LHS/auto-PGLE inert on this workload).
- Token-chunk pipelining of dispatch/FFN: -1.96pp.
- FP8 permutation-leg wire (QDQ decomposition): -2.02pp.
- TE-at-tip NCCL_EP: #3231's collective-stream pin crashes 64-GPU first execution; with the pin
  shimmed out the tip wheel is functionally the old wheel (~17% vs 18.05% a2a anchor).
- fa4-lse primal output: matched A/B (0.75 mem fraction, fresh caches) control 20.465% (3 draws) vs
  fa4-lse + host offload 20.648% (2 draws), +0.18pp, below the 0.5pp bar. The on-device variant is
  dead on memory (+32.7 GiB saved activations do not fit); host offload over Grace C2C is the only
  viable form, saving ~70ms of a 13.2s step. The d2560-derived ~1pp estimate does not transfer to
  d5120 EP64, where attention is a small slice. Recommendation: keep behind a flag as a free ~0.2pp
  if offload proves robust at d6144; do not count it toward the 25% bridge.

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

The goal (>=25% at the operating point without regressing fidelity) is not met at honest fidelity.
The best honest config is QB-on cf1.0 at 22.60% p50, which settles at ~6% steady drops, above a
strict 3% bar; the only measured 3%-compliant config is QB + cf1.15, at 20.416% over 350 steps. The QB-off frontier
of 24.04 (adjoint) and 25.39 (with leg-batching) is a matched-regime speed result, but at ~85% early
drops it is a bench number, not a shippable config, and it reads high partly because dropped
assignments do less real work (see the measurement caveat).

| config | drop regime | p50 MFU |
|---|---|--:|
| QB-off bench, adjoint | ~85% early | 24.04 |
| QB-off bench, adjoint + leg-batching | ~85% early | 25.39 |
| QB-on cf1.0 (best honest) | ~6% steady | 22.60 |
| QB-on cf1.15 (strict 3%) | 0.0260 tail-100 | 20.416 |

The strict-fidelity config and the throughput frontier are ~3.2pp apart. Ranked follow-ups to close
it:

- Sender-local balancing (kernel-level): a per-sender router bias aimed at the localized cause the
  gain probe implicates. Global-bias gain tuning is falsified (g=2 diverges).
- Leg-batching + QB-on composition: measure the two together; both are measured alone but not stacked.
- DeepSeek-style integral or damped (g<1) QB: cheaper than kernel work, unprobed.
- MXFP8 on the expert GEMMs: an explicit speed/quality call, not a free win.
- Fused fp8 epilogues.

fa4-lse (+0.18pp, section 3) is below the bar and does not count toward the bridge.

## 6. Same-step spill, and what it implies about the cost of fidelity

An assignment whose expert bucket is full is dropped and contributes nothing. Spill re-offers it,
within the same step, to the next-ranked expert the same token selected, taking only that bucket's
remaining headroom. `SCALE_A2A_SPILL=m` runs m such attempts. Bucket layout and capacity are
unchanged. Because spill is expressed purely as a rewrite of the `(linear_indices, keep)` pair that
the dispatch already builds, and both custom-adjoint VJPs are generic in exactly that pair, it
required no adjoint changes; gradients still match autodiff at 1e-5 with spill on.

350-step legs, QB on, cf1.0, one operating point, all from the same allocation draw, drops as a
true 100-step tail:

| m | p50 MFU | drop fraction (tail-100) | loss @349 |
|--:|--:|--:|--:|
| 0 | 22.062 | 0.0710 | 3.336 |
| 2 | 21.872 | 0.0414 | 3.323 |
| 3 | 21.849 | 0.0366 | 3.320 |

Spill halves the residual drop fraction for 0.213pp of MFU, and loss improves at every m rather
than degrading. The loss result is the one to weigh: a spilled assignment computes `w_k * E_j(x)`
for an expert the router itself ranked for that token, instead of contributing zero, so the
substitution is strictly closer to the intended MoE output than dropping. It does not reach a 3%
bar at cf1.0 (m=3 leaves 3.66%), and the drop-recovery mechanism is bounded by top-k: an assignment
can only be re-offered to experts the token already selected, so `m_max = topk - 1`.

**A routing model is trustworthy on one axis and not the other.** Predicting a configuration that
has not been run means trusting a model, and the useful question is which part of it to trust. With
live 350-step measurements on both axes, the error splits cleanly:

| measurement | model | measured | ratio |
|---|--:|--:|--:|
| cf1.0, m=0 | 0.0692 | 0.0710 | 1.03x |
| cf1.15, m=0 | 0.0230 | 0.0260 | 1.13x |
| cf1.0, m=2 | 0.0304 | 0.0414 | 1.36x |
| cf1.0, m=3 | 0.0237 | 0.0366 | 1.54x |

The capacity response is accurate; the spill response is optimistic, and grows more so with each
attempt. The mechanism explains the split: the model assumes a spilled assignment finds a free
bucket among the token's remaining choices independently each time, but a token's alternative
experts are correlated with its first choice, so later attempts find fewer free buckets than an
idealised count predicts. That is why the error is absent at m=0, small on capacity, and compounding
in m. A prediction that corrects the spill axis and takes the capacity axis at face value is usable;
one that trusts the model whole is not. Both the split and its mechanism were only visible because
live points existed on both axes — a single-axis sweep would have shown a model that looked either
fine or uniformly optimistic, and neither reading would have been right.

**The capacity price is a tiling cliff, not a slope.** Three same-length measurements pin it:
capacity factor 1.00 to 1.05 costs 1.18pp of MFU, while 1.05 to 1.15 costs 0.127pp per +0.05.
The cause is alignment. Capacity is `ceil(capacity_factor * assignments_per_shard / num_experts)`
and becomes the M dimension of the expert GEMM. At capacity factor 1.0 that is exactly 2048, or 16
tiles of 128; at 1.05 it is 2151, which is odd; at 1.15 it is 2356, divisible only by 4. The cost is
paid on leaving alignment and almost nothing is paid for growing capacity afterwards — consistent
with a collective-bound step, where extra padded compute is cheap but a badly shaped GEMM is not.

The fix is not to prefer one lucky capacity factor. It is to round capacity up to a tile-aligned
value whatever factor is requested, so a run asking for 1.05 silently gets 2176 rather than 2151.
The extra capacity is itself a fidelity gain, since a larger bucket drops fewer assignments, so the
alignment is free on both axes. `SCALE_CAPACITY_TILE=N` does this; the recommended production
setting is 128, and buckets at or below one tile are left alone so small configurations are
unaffected.

**Why this was cheap, and what that predicts.** The step is collective-volume-bound: exposed
collective time almost exactly fills compute idle, so the remaining speed lever is reducing
collective bytes. Spill adds no collective bytes and no matmul work — the expert GEMMs run on
capacity-sized buffers whether or not a slot is filled — it adds one segment-rank over the
assignment indices per attempt. It is therefore cheap by construction, not by luck. The general
form is worth stating, because it ranks work that has not been tried yet: on this stack, a
mechanism that spends index or compute work to buy fidelity is close to free, while a mechanism
that adds bytes to the collectives is expensive no matter how well implemented. Spill's 0.213pp and
the collective-bound ceiling are the two measurements behind that claim.

Three extrapolations failed in this work, and they failed the same way. A steady-state estimate
read from a truncated log window; drop fractions compared at equal step numbers across runs of
different lengths; and a capacity price interpolated linearly between two capacity factors. Each
looked like a smooth quantity and each was actually crossing a discretization boundary — a fixed
log-window size, a schedule defined over total steps, a capacity rounded to an integer that then
has to tile a GEMM. The generalisation is worth more than any of the three: on this stack, the
interpolations that break are the ones that cross a boundary the hardware or the harness has
quantised, and the cheap defence is to measure at both sides of the boundary rather than to fit
through it.

A second error ran in the opposite direction to the usual one. Drop fractions from runs of
different lengths were compared at the same step number and the disagreement was attributed to
allocation variance — the same configuration reads 0.083 at step 119 of a 120-step run and 0.175 at
step 119 of a 350-step run, and that was recorded as noise. It is not noise: the LR schedule spans
`num_train_steps`, so step 119 is 99% through the first run and 34% through the second, at roughly
6% and 68% of peak LR. A churning router drops more; an annealed one drops less. Filing that
systematic effect as variance inflated the apparent uncertainty of the whole drop record and hid
real structure. The correction made the data better than it had been claimed to be, which is worth
saying plainly because the usual direction of error is the reverse and a reader will not be looking
for this one. The practical rules — compare only at equal fractions of the LR schedule, prefer a
tail window to any single step, state the run length beside every drop number — are in
`experiments/grug/moe/agent.md`.

One error nearly escaped, and its shape is worth recording because it is not the kind that
provenance checks catch. An earlier version of this document, and a summary derived from it, both
stated that QB + cf1.15 was "the only measured 3%-compliant config" while reporting that same
config's drop fraction as 0.037 two sections below. Every number involved was correct and
correctly attributed; 0.037 was a real measurement. The fabrication was the interpretation, and
it survived review because the sentence was plausible and the figure it rested on was sound —
catching it required doing the one comparison of 0.037 against 0.03 that nobody did. The claim
also crossed a hand-off: the section reporting the measurement and the ledger sentence
generalising from it were written by different people, each treating the other's layer as
already checked. A derived claim needs re-deriving at every hop it crosses. In a chain of
summaries the dangerous residue is not a wrong number, which any audit trail will expose, but a
wrong reading of a right number, which none of them will.

A note on the numbers above. Three of them moved during the work, each time against the result's
favour: a steady-state estimate taken from a truncated log window read 10% low until refetched as a
true 100-step tail; a routing model that predicted spill's benefit proved 1.36-1.54x optimistic
against live measurement; and the MFU cost was first measured against another run's baseline and
rose from 0.153pp to 0.213pp once measured against its own allocation draw. The qualitative result
survived all three. On the last of these: the cross-draw comparison happened to be nearly right, and
was still not safe to rely on — draw variance was 0.060pp where the effect was 0.2pp, which is the
regime in which being nearly right is indistinguishable from being lucky.

## 7. Reproduction

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
- Spill (section 6): `SCALE_A2A_SPILL=m`, commit `1224ccb02`; 350-step legs
  `ep25d1-qbon-cf115-350-0726-0313` (m=0, same draw), `ep25d1-spill2-cf100-350-0726-0028` (m=2),
  `ep25d1-spill3-cf100-350-0726-0149` (m=3). Capacity factor knob `SCALE_CAPACITY_FACTOR`,
  commit `595958b83` (agent d4 added an independent implementation of the same knob under the same
  name in `3e149490f`; reconcile rather than double-apply when merging).
- Steady-state figures are 100-step tails. `iris job logs` truncates to the most recent 1000 lines
  unless `--max-lines` is passed, which biases any statistic over a still-trending metric; see
  `lib/iris/OPS.md`.
- Fidelity frontier and gain probe (d4): 350-step steady state
  `ep25d4-qb-cf100-drops-350-v1`; QB gain g=2 `ep25d4-qbgain2-cf100-350-v1` (`SCALE_QB_GAIN`, commit
  `58c9a19eb`).
