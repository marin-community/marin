# Derisking queue

Experiments that must run before the plan in [`sequence.md`](sequence.md) is
trusted at hero-run scale. **None of these were run for this brief** — it is
compiled entirely from the existing record.

## Standing protocol for every experiment here

The record contains several claims that were later reversed, and the reversals share
a shape. These rules are adopted from the corrections in
[#7279 c5084892846](https://github.com/marin-community/marin/issues/7279#issuecomment-5084892846)
and the ledger caveats in
[c5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482).

1. **Lock the denominator.** 2.5 PFLOP/s GB200 bf16-dense. Report tok/s alongside
   MFU always, because reported MFU is window-blind and inflates with sequence
   length (×1.08 at 4k, ×2.14 at 65k).
2. **Report the drop fraction beside every MFU figure.** MFU is not comparable
   across drop regimes — a run that drops more reads higher MFU for less real work.
   A comparison without matched drops yields an upper bound, not a measurement.
3. **Compare drop fractions only at the same fraction of the LR schedule**, prefer a
   tail window over any single step, and state run length beside every drop figure.
   The schedule is defined over `num_train_steps`, so step 119 of a 120-step run and
   step 119 of a 350-step run are at completely different LR positions.
4. **Several placement draws per arm.** Placement swings measured ±2–4pp at EP64;
   multi-rack GB200 gangs get only a SOFT `nvlink.domain.preferred` constraint. Any
   margin claim under ~2pp is not established by a single draw.
5. **Measure inside the real step with rematerialization on.** Isolated wins on this
   stack have repeatedly failed to survive end to end, in both directions.
6. **Pre-register the prediction and a falsification threshold** for anything with a
   mechanistic story. The delayed-scaling probe is the model: it fired both
   falsification clauses at 17× the threshold, which is what made it cheap to stop.
7. **No `jax.debug.print` inside a rematerialized scan body.** A bare print touching
   no tensor costs 1.41× compiled temp memory — at 48 layers, a 300+ GiB allocation
   request. Instrument through the metrics path or in a small probe config.
8. **A finished job is not a passing job.** The training loop breaks rather than
   raises on NaN, so Iris has reported a NaN run as *succeeded* while writing a
   poisoned checkpoint. Check the loss trajectory and the drop metric, not the exit
   status.
9. **Fix the config baseline before comparing anything.** Every arm should carry
   `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (the record flags the default allocator
   as a silent-deadlock and OOM source at 64×GB200; the fixed arm measured 23.32% /
   362K tok/s), `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`
   on JAX 0.11, and `--xla_gpu_experimental_parallel_collective_overlap_limit=4`.
   Note `TF_GPU_ALLOCATOR` is *not* read by JAX — setting it does nothing.

---

## P0 — blocking. Without these the plan rests on unmeasured comparisons.

### D-1. Measure the drop rate of the FSDP baselines

**Why.** Both EP-versus-FSDP comparisons in the record are one-sided on fidelity:
the 23.1% (d6144 4-of-128) and 19.2% (d5120 8-of-256) FSDP figures have
**unmeasured** drop rates, and this is named as the largest uncertainty on both
results. The EP numbers are being penalised for honesty the baselines never paid.

**What.** Land commit A1 (the tracker-logging fix), then re-run each FSDP baseline
config for one job with `SCALE_REPORT_DROPS=1`.

**Cost.** Two jobs. One line of code.

**Falsifies.** If the FSDP baselines also drop 6–13%, the current EP-vs-FSDP
comparison is fair and the EP line is further ahead than reported. If they drop
<3%, the EP line's advantage is smaller than claimed and possibly negative.

**This is the cheapest high-value experiment in the queue and it should run first.**

### D-2. Re-establish the EP64 stack end to end at compliant fidelity, on merged code

**Why.** The Phase D wins were each measured against a different baseline on a
different branch, at QB-off (D2, D3) or on a 20-step screen (D4). Nobody has run
the *composed* stack — C1–C4 + D1–D4 + E2, with QB on — at once.

**What.** 350-step run at d5120 8-of-256 EP64, one rack, batch 1024, seq 4096,
QB on, cf 1.0625, spill m=3, custom adjoint, padded Muon, PGLE + overlap-limit 4.
Three placement draws. Report p50 MFU over the steady tail and true tail-100 drops.

**Predict before running.** The additive ledger says roughly 20.7% (C2's best
compliant point) + 1.78pp (D4) ≈ 22.5%, less whatever D4's gain fails to transfer
across architectures. State the number, then measure.

**Falsifies.** Whether the Phase D gains compose. Several of them are described as
independent and stacking, and none of that has been tested.

### D-3. Resolve the leg-batching contradiction — **code recovered 2026-07-29; sign still open**

**Why.** The same idea has measured **+1.35pp and −3.66pp**. The 25.39% figure is
QB-off on an unmatched run; an independent reconstruction with QB on and matched
drops measured control 22.66% against batched G=2 **19.00%**, bands
non-overlapping, with the batched path bit-exact against the loop. G=4 never
produced a step.

**The recovery is done, and it dissolves most of the contradiction.** Full findings in
[`d3-legbatching-report.md`](d3-legbatching-report.md) on `agent/deri-d3-recover`.

The claim that the patch was never committed is **wrong**. The exact submitted source
survives as Iris bundle
`0483b2f207323fb3cd79ec326b7592546aabb0812ef8c058be95bd6c8049cd43` (ZIP SHA-256 verified
against the bundle ID), and the same mechanism was committed the next morning in
[`98737aecf`](https://github.com/marin-community/marin/commit/98737aecfa5cd05b9bffe09c96754c96d7177f06).

**The two measurements were never testing the same change.** The recovered positive
implementation defines `SCALE_A2A_BATCH_EXPERT_GEMMS`, requires
`SCALE_A2A_PACK_DISPATCH=1` and `SCALE_A2A_PACK_COMBINE=1`, and keeps the already-packed
one-dispatch/one-combine schedule — it replaces only the per-local-expert GEMM loop with
batched `jnp.matmul`. The reconstruction changes GEMM batching **and** collective
scheduling: its G=2 arm runs two dispatches and two combines and concatenates.

So the −3.66pp is credible evidence that the *reconstruction* regresses, and says nothing
about compute-only batching. Equally, 25.39% is not causal evidence for a win: QB-off,
unmatched baseline from another build, no matched drop accounting, a different FLOPs
denominator, and it did not contain the custom adjoint the issue comment said it stacked
with.

**Standing conclusion: the record supports neither sign.** 25.39% remains barred from any
planning document as an achievable number. Settling the sign needs a new matched rack
A/B, and only after the treatment passes the EP64 compile/runtime/memory gate that the
reconstructed G=4 path already failed.

### D-4. Multi-rack EP64

**Why.** **EP64 has no multi-rack measurement at all.** Every 20T-token projection
in #7201 for the EP line (~65–75 days) scales a one-rack number by 12 with a 7%
weak-scaling penalty, and the record already shows that penalty is wrong for FSDP
MoE — the measured 1→2-rack drop was ~19% relative, not 7%. The mesh puts `expert`
innermost, so adding racks grows data-axis traffic while the MoE all-to-all stays
intra-rack; the shape of that curve is unknown.

**What.** The D-2 configuration at 2 racks and, if available, 4. Report measured
weak-scaling, not an assumed penalty.

**This is the single largest unquantified risk to the hero-run schedule.**

---

## P1 — high value, independent of the merge plan.

### D-5. Confirm `overlap_limit=4` clears the inline all-to-all ops at the hero shape

**Largely already answered — verify, do not re-investigate.** The mechanism is
identified: `GpuCompiler::RunPostSchedulingPipelines` runs
`GpuConvertAsyncCollectivesToSync`, which tags an async-start whose done is
separated only by no-ops as `is_sync=true`. A schedule-dump census shows MoE SYNC
all-to-all going 10 (at the default limit of 1) → 3 → **0 at limit 4** → 1 at 8.
That is the same class as the "three of twelve at 0.0% overlap, 422 ms of every
15.3 s step" observation at the d6144 shape.

**What.** Run `experiments/grug/moe/schedule_report.py` at the d6144 operating
point (about three minutes on one node, no rack time) and confirm the count goes to
zero at limit 4. Note the setting is **not monotone** — limit 2 measured worse than
1 on the ECHO line.

**Cost.** One single-node dump. This replaces what was previously scoped as an
open profiling investigation.

### D-6. Do the FSDP-line levers transfer to EP?

**Why.** Larry asked directly, and the answer so far is mostly no — of the FSDP
levers, EP64 has ported one cleanly (padded Muon, on an unmerged branch), has PGLE
in a degraded form, and has never tested the rest.

**What is already known, so do not re-derive it:**

- **PGLE is not a reliable EP win, and it is not unmeasured either.** +1.1pp on
  FSDP; on EP it is +0.47pp combined with the overlap limit. The ECHO line *has*
  measured it: v152 auto-PGLE reached 22.801% p50, and v153 (v152 plus
  `overlap_limit=4`) reached 23.286% over steps 5–19, **+0.427pp**. Manual PGLE was
  *rejected* there — it matched only 217 of 535 instructions and came in 0.235pp
  below the AutoPGLE leg. The headline padded-Muon A/B ran with PGLE **off**
  because the ~16-minute compile made preemption certain. So the question is
  narrowed, not open: PGLE helps on EP, by roughly a third of what it gives FSDP,
  and the auto/manual choice inverts between the two stacks.
- **Two shared experts is memory-blocked at EP64**, not merely untested: the one
  attempt (two 8192-wide) failed before step 0 at 89.49 GiB. Splitting the shared
  width does not reduce the FSDP gather peak.
- **Host offload is split by model size** — used on the d6144 EP64 leg, but
  *rejected* at d5120, where it needed a 135 GiB pinned-host arena and landed at
  19.694%.

**The GatedNorm / attention-gate / XSA trio is CLOSED, negative, as of 2026-07-29.**
Two matched 350-step draws at d5120 8-of-256 EP64 against a two-draw shared control.

| arm | tok/s (tail 250–349) | reported MFU | drops |
|---|---|---|---|
| control | 321,082–322,258 (mean 321,670) | 20.73–20.80% | 1.42–1.47% |
| trio | 316,587–317,919 (mean 317,253) | 20.44–20.52% | 2.04–2.06% |

Bands are non-overlapping and the trio mean is **1.37% below** control, missing the
pre-registered +1.0% gate in both draws. Ranked on tok/s: reported MFU is shown but not
used, because `lm_flops_per_token` omits the work XSA and the attention gate add. Drops
are 42.1% worse, which was not the registered ranking metric. No conditional
single-variable arms were submitted, per the pre-registration.

**Reconciliation that matters more than the verdict:** GatedNorm, XSA and the attention
gate are **unconditional on `main`** — `experiments/grug/moe/model.py:6` states the
architecture as "QB-routed MoE with GatedNorm, XSA, sigmoid combine weights", and the
config docstring lists them as hardcoded. So this is not an unported lever we declined;
it is a cost `main` already pays at EP64. Whether the trade is correct is a *quality*
question, and nothing in this experiment measures quality.

**DECISION (2026-07-29): standardize on trio-ON for all arms.** Since the series lands on
`main`, and `main` runs the trio unconditionally, trio-on is the configuration production
will actually use. Every arm from here carries `SCALE_GATED_NORM=1`, `SCALE_ATTN_GATE=1`
and `SCALE_XSA=1` unless the experiment is specifically about them. The D-2 composed draw
already does.

Two consequences to carry forward. **The trio-off control band (321,670 tok/s) is no
longer the baseline** — the trio-on band, 316,587–317,919 tok/s (mean 317,253), is. Do not
compare a new trio-on arm against the old control. And **the FSDP line gates the trio the
same way**: `origin/b200-300B-tune` has
`gated_norm=os.environ.get("SCALE_GATED_NORM") == "1"` and the same for the other two, all
defaulting off. So whether the FSDP figures in the comparison table were taken trio-on is
a submit-command question the branch cannot answer. Establish it before any EP-vs-FSDP
claim; this is the same unmatched-conditions problem as D-1's unmeasured drop rates,
raised independently by Larry Dial on 2026-07-29.

**What is genuinely open.** Muon shape-grouping (+0.09pp on FSDP) only. That is below
the ~2pp threshold this project's own protocol sets for repeated placement draws, so it
needs a reason to believe before it earns rack time.

**What.** A single-variable A/B on the D-2 stack for shape-grouping, ≥2 draws given the
sub-2pp margin. The trio arm is done and closed; do not resubmit it.

### D-7. Attack the drop residual somewhere other than the router controller

**The controller family is closed. Do not re-run it.** All four variants are
measured: g=2 pins drops at 0.675–0.793 for 350 steps with loss +0.091 worse;
g<1 cannot beat g=1 by construction (same fixed point, different approach rate);
DeepSeek-style integral plateaus at ~0.60 (γ=0.001) and ~0.46 (γ=0.01) against
g=1's 0.073; and **sender-local bias — the mechanism aimed at the standing
hypothesis — came in at tail-100 0.0856 against global's 0.0732, statistically
identical.**

That last result matters most: it **overturned** the sender-local-hotspot
explanation for the ~6% residual. The revised reading on the record is that the
residual is **batch-stochastic within-batch burstiness**, invisible to any
one-step-delayed bias controller of either kind. Supporting arithmetic: at bucket
mean 2048 uniform routing floors at 0.88–0.91%, and observed 6–8% implies σ
329–411 against a Poisson 45.3 — **routing is 7–9× more clustered than
independent-uniform.**

**What is left.** Same-step spill (already landed as E2 in the plan) works
*because* it acts within the step rather than across steps, which is consistent
with the burstiness reading. Directions that remain open, none probed:

- Raising spill's ceiling — it is capped at `top_k − 1`, so 8-of-256 keeps
  improving through m=7 while 4-of-256 is flat from m=3. **This is an
  architecture-selection input, not a kernel knob.**
- Anything that reduces within-batch burstiness at the source (token ordering,
  batch composition), which nobody has looked at.
- Accepting ~1.4% with spill + cf1.0625 and spending the effort elsewhere.

**Payoff.** The ~3.2pp gap between the throughput frontier and strict fidelity is
still the largest single prize, but the cheap routes to it are now exhausted.

### D-8. Does the EP-aware NS fix also unblock EP32?

**Why.** Both EP32 arms OOM at the reference config on a ~104 GiB temporary,
attributed to the same SPMD involuntary-full-remat fallback (XLA b/433785288) that
commit C2 fixes for the optimizer stack — but for the microbatch input resharding
rather than the expert stack. Nobody has retested EP32 since.

**Cost.** One job on the D-2 stack with `SCALE_EXPERT_AXIS=32`.

**Note.** Even if it runs, EP32 is unlikely to be the right operating point (it pays
dispatch cost without the memory relief). This experiment is worth running for the
*diagnosis*, not because EP32 is a candidate.

### D-12. Does `all_but_moe` remat fit at the production shape?

**Why.** [#7489](https://github.com/marin-community/marin/pull/7489) (slim Sonic CuTe
residuals plus the `all_but_moe` rematerialization split) is the one remat lever with a
measured win, but its memory behaviour is only characterised at shapes we do not run.
Every measurement in the record is **26 layers**; production is **48**, and the
mechanism is layer-stacking — the homogeneous scan stacks all layers, which is why
`attn_only` was called *structurally* infeasible rather than merely tight.

The sharding-width story is favourable and already measured: at 8-way, d5120
`all_but_moe` does not fit (`recompute_all` alone peaks at 157 GiB); at 32-way it fits
and wins, 16.90% against `recompute_all`, +0.74pp, while `none` OOMs. Production EP64 is
wider still, and multi-rack only adds data-parallel replicas — per-device activation
memory is roughly invariant and optimizer state shards wider — so the binding constraint
is within one rack. If it fits on one rack it fits on twelve.

What is unmeasured is layers × per-device tokens at the real shape. A desk estimate
against the one probe with exact buffer shapes (8-way d2560/26L, `all_but_moe` temp
134.99 GiB, irreducible pins `gu` + `out_dispatch` at 2 × 32.5 GiB and 262144
tokens/device) scales those terms by 0.25 for tokens (65536/device at batch 1024 × seq
4096 ÷ 64) × 1.85 for layers, intermediate unchanged at 1280 — so they *shrink*, and it
fits with room. That estimate rests on reconstructed shapes and the token factor carries
most of it. Measure rather than trust it.

**What.** Run the existing compile-only AOT memory probe, `remat_oom_probe.py` (the tool
behind the 283.63 vs 36.89 GiB comparison, jobs 3264/3265), at d5120 / 48L / EP64 for
`recompute_all` and `all_but_moe`. Report peak temp arena for each.

**Cost.** Compile only — minutes, no training steps, no rack draw.

**Falsifies.** Whether #7489 is adoptable at the operating point at all. Note the answer
is a package deal: slim residuals alone cost **−0.28pp under `recompute_all`** (backward
re-gathers with no memory dividend cashed), so the residuals cannot land ahead of the
remat mode without paying for nothing in between.

**Reading the failure mode.** At high HBM occupancy an OOM here surfaces as a BFC
fragmentation wedge or clique stall, not a clean `RESOURCE_EXHAUSTED`. Check
`bfc_allocator` before blaming collectives.

---

## P2 — conditional, and only if the precision decision is taken.

**Precondition for everything in this tier.** The expert-only MXFP8 port already
measured **−2.582pp p50** in a matched, QB-on, drop-reported 120-step A/B at the
EP64 operating point, and **−0.313pp** at d6144/i3072 (`24d411b38`,
`fac261215e`). The recorded verdict is "do not adopt at this operating point", and
the recorded condition for reopening is *a new matched all-QB-on end-to-end pair*
on a **materially different mechanism** — not a re-run of the same one. Do not
spend rack time here unless something in that list changes.

### D-9. FP8 dispatch wire in the real step

**Why.** [#7665](https://github.com/marin-community/marin/issues/7665) measures
1.286× fwd / 1.144× fwd+bwd at EP64 with bit-exact weight gradients, and it is the
only lever whose gain **grows** with EP degree — which matters because the step is
collective-volume-bound and reducing collective bytes is the only remaining lever
in that family. But every number is one layer in isolation: no scan, no remat, no
optimizer, no competing collectives.

**Blocked on more than a port.** The wire needs a quantized consumer downstream,
and the consumer available today is the expert-GEMM port that measured −2.582pp at
this exact shape. The wire's +1.144pp fwd+bwd at EP64 is a layer-level number that
would have to survive the real step *and* more than cover that loss. Sequence this
only behind a mechanism that changes the consumer's sign.

**Success criterion** (already pre-registered on the issue): a positive layer-level
A/B **inside the real step with rematerialization on**.

### D-10. The MFU-versus-EP-degree curve for the *hybrid* MXFP8 recipe

**Why.** The hybrid recipe measured 1.308× (d5120, EP8), +7.22% (d2560, EP8, 66B
tokens) and 0.749× (d6144, **EP1**), and these are unreconciled. The plausible but
unmeasured explanation is that the expert-GEMM share of the step varies with model
size and EP degree.

**Scope this narrowly.** The *expert-only* port's sign at EP64 is already settled
and negative (see the precondition above), so this experiment is only worth running
for the hybrid grouped-plus-dense recipe, and only if something else revives the
direction. It is not on the critical path.

**What.** One configuration, EP1 → EP8 → EP16 → EP64, bf16 against hybrid MXFP8,
matched arms, ≥2 draws at each point.

### D-11. Root-cause the hybrid `w_down` NaN

**Why.** It is currently masked by a guard whose underlying cause was never found,
and the guard has not been re-validated after any kernel or XLA version change. It
sits under the only rigorous quality result in the workstream.

**What.** HLO-level analysis of the hybrid graph's liveness around `dw2`. A
diagnostic *consumer* of `dw2` fixes it while `optimization_barrier` does not,
which is a strong hint.

---

## Sealed. Do not re-open without new information.

Listed here so the queue is not re-derived. Full citations in
[`evidence.md` Group E](evidence.md).

The entire **scheduling family** — rotation `ppermute` (−9.46pp), token-chunk
pipelining (−1.96pp), weight prefetch (null), PGLE/LHS beyond the flag set already
adopted. The step is collective-**volume**-bound: exposed collective time almost
exactly fills compute idle, reproduced at two shapes.

Also: `ring_cute` at e256/EP64 (DNF, OOM 141.79 GiB); ragged a2a with one-shot off
at EP64 (12.38%); TransformerEngine NCCL_EP (ties, ~1.1–1.3pp behind `a2a_cute`);
latent MoE at d6144 EP64 (−0.23 to −1.72pp); FP8 QDQ permutation-leg wire without a
quantized consumer (−2.02pp) and its delayed-scaling remedy (falsified at 17× the
threshold); SM comm/compute partitioning (falsified three times); QB gain g=2
(diverges); auto-PGLE (crashes multi-host); fa4-lse as a primal output (+0.18pp,
below bar); native dense MXFP8 (0.64–0.81×); uniform dense+grouped MXFP8 (misses the
gate in-repo, aborts deterministically at EP2/EP4); NVFP4 (ruled out on risk);
JAX-Toolbox (0.505% time-adjusted regression); MLA (neutral at matched head dims,
negative otherwise); source-push; NVSHMEM/CuTe transport.
