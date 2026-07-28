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
the *composed* stack — C1–C4 + D1–D4 + E1 + E2 — at once.

**What.** 350-step run at d5120 8-of-256 EP64, one rack, batch 1024, seq 4096,
QB on, cf 1.0625, spill m=3, custom adjoint, padded Muon, PGLE + overlap-limit 4.
Three placement draws. Report p50 MFU over the steady tail and true tail-100 drops.

**Predict before running.** The additive ledger says roughly 20.7% (C2's best
compliant point) + 1.78pp (D4) ≈ 22.5%, less whatever D4's gain fails to transfer
across architectures. State the number, then measure.

**Falsifies.** Whether the Phase D gains compose. Several of them are described as
independent and stacking, and none of that has been tested.

### D-3. Compose leg-batched expert GEMMs with QB-on

**Why.** B5 posts 25.39% and is the only remaining item that could bridge to 25% at
honest fidelity, but it was measured QB-off on a separate run — not a matched A/B
— and it has never been stacked with QB. This is listed as an open follow-up in the
25% ledger itself.

**What.** Matched A/B, leg-batching on/off, both arms QB-on cf1.0625 with spill,
350 steps, ≥2 draws.

**Cost.** Two runs.

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

### D-5. Why do three of twelve all-to-all ops run on the compute stream?

**Why.** 422 ms of every 15.3 s step, **31% of all exposed collective time**,
consistent across all four GPUs, unexplained, and larger than latent MoE's entire
best case. It needs no architecture change.

**What.** Dump the HLO for the operating-point graph and name the three ops and
their scheduling annotations, rather than inferring from profile names — that
inference was wrong three times in the FP8 investigation, one rack leg each.

**Cost.** One profile plus HLO analysis. No new runs.

### D-6. Do the FSDP-line levers transfer to EP?

**Why.** Larry asked directly. PGLE was +1.1pp on FSDP and only +0.47pp (with the
overlap limit) under EP; the ECHO screen ran with PGLE off entirely. Two shared
experts (+0.29pp) and Muon shape-grouping (+0.09pp) are unmeasured under EP.

**What.** Three single-variable A/Bs on the D-2 stack: PGLE on/off; 1 vs 2 shared
experts; Muon shape-grouping on/off. Each ≥2 draws given the sub-2pp margins.

### D-7. Sender-local router balancing

**Why.** QB-on at cf1.0 plateaus at ~6–7% drops and does not improve past ~150
steps. Gain tuning is falsified (g=2 diverges into an overshoot limit cycle with
loss +0.091 worse), so the residual is not global-bias under-correction. The
leading hypothesis is sender-local bucket hotspots: capacity is enforced
sender-locally at 64 senders × 256 experts and a global router bias cannot see one
sender overloading one expert. cf1.15 roughly halving drops is consistent with it.

**What.** Cheapest first: DeepSeek-style integral accumulation, and a damped gain
(g < 1) — both are configuration-scale. Only then the kernel-level per-sender bias,
which is the one aimed at the hypothesised cause.

**Payoff.** Closing the ~3.2pp gap between the throughput frontier and strict
fidelity is worth more than any remaining scheduling work.

### D-8. Does the EP-aware NS fix also unblock EP32?

**Why.** Both EP32 arms OOM at the reference config on a ~104 GiB temporary,
attributed to the same SPMD involuntary-full-remat fallback (XLA b/433785288) that
commit C2 fixes for the optimizer stack — but for the microbatch input resharding
rather than the expert stack. Nobody has retested EP32 since.

**Cost.** One job on the D-2 stack with `SCALE_EXPERT_AXIS=32`.

**Note.** Even if it runs, EP32 is unlikely to be the right operating point (it pays
dispatch cost without the memory relief). This experiment is worth running for the
*diagnosis*, not because EP32 is a candidate.

---

## P2 — conditional, and only if the precision decision is taken.

### D-9. FP8 dispatch wire in the real step

**Why.** [#7665](https://github.com/marin-community/marin/issues/7665) measures
1.286× fwd / 1.144× fwd+bwd at EP64 with bit-exact weight gradients, and it is the
only lever whose gain **grows** with EP degree — which matters because the step is
collective-volume-bound and reducing collective bytes is the only remaining lever
in that family. But every number is one layer in isolation: no scan, no remat, no
optimizer, no competing collectives.

**Blocked on.** MXFP8 expert GEMMs do not exist on the EP64 fixed-a2a stack —
expert GEMMs there are bf16 `jnp.einsum`. The port is a prerequisite, and it is
substantial.

**Success criterion** (already pre-registered on the issue): a positive layer-level
A/B **inside the real step with rematerialization on**.

### D-10. The MFU-versus-EP-degree curve for MXFP8 expert GEMMs

**Why.** The same hybrid recipe measured 1.308× (d5120, EP8), +7.22% (d2560, EP8,
66B tokens) and 0.749× (d6144, **EP1**). These are unreconciled. The plausible but
unmeasured explanation is that the expert-GEMM share of the step varies with model
size and EP degree. Without this curve the production sign of MXFP8 is unknown.

**What.** One configuration, EP1 → EP8 → EP16 → EP64, bf16 against MXFP8, matched
arms, ≥2 draws at each point.

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
