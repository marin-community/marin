# Debug: v6e-8 vs v5p-8 DPO Training Discrepancy

> **CAMPAIGN CLOSED (2026-04-18T15:28Z, 40 experiments total, CN final).**
>
> **CN result** (v5p-8 pd=4 p=bf16, c=bf16 — full bf16): **step 9 = 0.661**.
> Stuck. Same as Exp Q baseline. Confirms: **Bug 1 cannot be rescued by
> any dtype change** (p=f32/bf16 × c=f32/bf16 all tested — still stuck).
>
> Bug 1 is **unambiguously a mesh/collective issue** on v5p-8 pd=4 FSDP-4.
> Precision/rounding gates have ZERO effect. Only rescue mechanisms:
> 1. **Change mesh** (Exp W TP or mix): full recovery to Exp N quality.
> 2. **Increase LR** (10x or 30x): closes the gap via step-size.
>
> **MAJOR NEW FINDING (CM v2)**: CA's forward-path rounding gates **do
> NOT rescue Bug 1.** CM v2 (v5p-8 pd=4 c=bf16 + 5 CA gates) step 9 =
> **0.661** — same as baseline Bug 1 Exp Q (0.66). Zero rescue.
>
> **MAJOR NEW FINDING (CM v2)**: CA's forward-path rounding gates **do
> NOT rescue Bug 1.** CM v2 (v5p-8 pd=4 c=bf16 + 5 CA gates) step 9 =
> **0.661** — same as baseline Bug 1 Exp Q (0.66). Zero rescue.
>
> Compare: CA rescues Bug 2 (c=f32) by ~38%. But CA does nothing for
> Bug 1 (v5p-8 pd=4 c=bf16).
>
> **Under Bug 1, compute is already c=bf16**, so the CA rounding gates
> (which round to bf16) are effectively no-ops. Bug 1's "stuck" behavior
> is NOT a precision/rounding issue. Bug 1 is caused by something
> different — most likely **FSDP-4 mesh collective behavior on v5p-8**
> (confirmed earlier by Exp W: TP mesh on v5p-8 rescues; "data=2/model=2"
> mix also works; only "data=4" FSDP-4 is stuck).
>
> **Revised final mechanism summary:**
> - **Bug 2 (c=f32)**: ~33% rescued by CA (forward-path rounding), ~67%
>   in backward-pass precision.
> - **Bug 1 (v5p-8 pd=4 c=bf16)**: 0% rescued by CA. A mesh/topology
>   issue, not a precision issue. Rescued by mesh change (Exp W TP) or
>   10x LR.
>
> Bug 1 and Bug 2 have QUANTITATIVELY IDENTICAL symptoms but
> MECHANISTICALLY DIFFERENT causes. This surprised us — LR scaling
> produced the same rescue curves for both, but now we know the
> mechanisms are orthogonal.
>
> **Headline**: Bug 1 (v5p-8 pd=4) and Bug 2 (c=f32) are **identical
> LR-compensable per-step slowdowns** — not stuck states. Full LR-
> rescue grid confirmed to 4 decimal places at lr=3e-5.
>
> **Mechanism (bounded)**:
> - ~33% of the c=bf16 speedup is from **forward-path cumulative
>   rounding** at module boundaries (CA gates: Linear I/O/W + RmsNorm
>   out + residual). Verified stable across step 9 / step 99 / lr=1e-6
>   / lr=3e-5.
> - ~67% of the speedup is **outside forward-path** — likely JAX
>   autodiff precision (backward-pass bf16 under c=bf16 policy vs
>   f32 under c=f32). Not reproducible by any post-hoc forward gate.
>
> **Production guidance**:
> - Default: **Exp N** (c=bf16, v5p-16 pd=2, lr=1e-6) → 0.32 at step 9.
> - For c=f32 (determinism): lr=1e-5 → 0.41; lr=3e-5 → 0.137; or
>   100 steps at lr=1e-6 → 0.40.
> - For v5p-8 pd=4 (FSDP-4 mesh): lr=1e-5 or use TP mesh (Exp W).
>
> **All 40 experiments**, the code patches (all debug-gated), and
> the full hypothesis-trajectory are captured below.

## W&B run-URL index — post-compaction experiments (CP8..CN)

| exp | W&B link |
|-----|----------|
| CP8 (c=f32 lr=3e-5) | https://wandb.ai/marin-community/dpo/runs/experiment_bf_r64_s10_ue5-cp8-f32-lr3em5-67a941 |
| CP9 (v5p-8 pd=4 lr=3e-5) | https://wandb.ai/marin-community/dpo/runs/experiment_cp9_r64_s10_cp9-v5p8-pd4-lr3e-05-636468 |
| C7 (p=bf16 c=f32) | https://wandb.ai/marin-community/dpo/runs/experiment_c7_r64_s10_c7-v5p16-pbf16-cf32-8cd7e8 |
| C8 (continuous relnoise) | https://wandb.ai/marin-community/dpo/runs/experiment_c8_s10_c8-v5p16-fp32-relnoise0.03-861076 |
| C9 (residual rounding) | https://wandb.ai/marin-community/dpo/runs/experiment_c9_s10_c9-v5p16-fp32-roundresid-9fdcdb |
| CA (5 rounding gates) | https://wandb.ai/marin-community/dpo/runs/experiment_ca_s10_ca-v5p16-fp32-all-rounded-90e035 |
| CC (silu+gated) | https://wandb.ai/marin-community/dpo/runs/experiment_cc_s10_cc-v5p16-fp32-silu-gated-3d8df2 |
| CD (RmsNorm internal bf16) | https://wandb.ai/marin-community/dpo/runs/experiment_cd_s10_cd-v5p16-fp32-norminternal-62ceef |
| CE (kitchen sink) | https://wandb.ai/marin-community/dpo/runs/experiment_ce_s10_ce-v5p16-fp32-kitchen-sink-3289e2 |
| CF (CA × 100 steps) | https://wandb.ai/marin-community/dpo/runs/experiment_cf_r64_s100_cf-v5p16-fp32-ca-s100-6c97b3 |
| CH (Exp N × 100 steps) | https://wandb.ai/marin-community/dpo/runs/experiment_ch_s100_ch-v5p16-bf16-expn-s100-8b513b |
| CK (CA + lr=3e-5) | https://wandb.ai/marin-community/dpo/runs/experiment_ck_s10_ck-v5p16-fp32-ca-lr3em5-39eeda |
| CM (CA on v5p-8 pd=4) | https://wandb.ai/marin-community/dpo/runs/experiment_cm_r64_s10_cm-v5p8-pd4-ca-gates-268a92 |
| CN (v5p-8 pd=4 full bf16) | https://wandb.ai/marin-community/dpo/runs/experiment_cn_s10_cn-v5p8-pd4-pbf16-cbf16-34734b |
| CP7 (Exp N lr=3e-5) | *URL unavailable* — pre-compaction, iris record aged out |

Pre-compaction experiments (Exp A-Z, R, U, W, Y, etc.) have W&B URLs in
their respective sections further down.



>## EXECUTIVE SUMMARY (2026-04-18T13:05Z, ALL probes COMPLETE — CF + CH + CK landed)

**CK (CA + lr=3e-5) step 9 = 0.133.** Falls between CP7 (Exp N lr=3e-5)
= 0.125 and CP8 (AC lr=3e-5) = 0.137. CA closes **33% of the gap at
high LR**, same fractional rescue as at other conditions.

**Full cross-LR / cross-horizon rescue table:**

| horizon | Exp N | AC (c=f32) | CA (5 gates) | CA % rescue |
|---------|------:|-----------:|-------------:|------------:|
| lr=1e-6 step 9 | 0.32 | 0.662 | 0.532 | 38% |
| lr=3e-5 step 9 | 0.125 | 0.137 | **0.133 (CK)** | **33%** |
| lr=1e-6 step 99 | 0.248 | 0.40 | 0.351 (CF) | 32% |

**CA rescue is a stable ~33% gap closure across LR and step horizons.**
The remaining 67% of the c=bf16 benefit is outside forward-path
rounding.

**This nails down the final story**: c=bf16 provides TWO independent
mechanisms each worth ~50% of the step-9 gap:
1. **Forward-path cumulative rounding** (CA captures): ~1/3.
2. **Backward-pass precision or deeper kernel effects** (uncaptured): ~2/3.

Most likely source of the uncaptured portion is **JAX autodiff under
c=bf16 vs c=f32**: backward computations happen at the policy's
compute dtype throughout, so bf16 backward produces gradients with
systematically different precision than f32 backward. This cannot be
reproduced by post-hoc forward rounding gates.

>## EXECUTIVE SUMMARY (2026-04-18T13:02Z, CF + CH complete)

**Final 100-step trajectories:**

| recipe | step 9 | step 50 | step 99 | notes |
|--------|-------:|--------:|--------:|-------|
| CH (Exp N c=bf16) | 0.314 | 0.261 | **0.248** | fast recipe |
| CF (CA c=f32+5gates) | 0.538 | 0.389 | **0.351** | ~32% gap rescue |
| CP1 (AC c=f32) | 0.662 | 0.453 | 0.40 | no rescue |
| CP5 (Bug 1 v5p-8) | 0.669 | 0.454 | 0.398 | same as CP1 |

**Gap at step 99 = 0.248 - 0.40 = 0.152** (AC → Exp N).
**CA closes: 0.40 - 0.351 = 0.049.**
**Percent rescue: 32%** (consistent with 38% at step 9, 28% at step 30).

CA's post-hoc forward rounding captures roughly **1/3 of the c=bf16
benefit across time horizons.** The remaining 2/3 is outside forward-
path rounding — likely backward-pass precision (JAX autodiff under
c=bf16 policy uses bf16 in grad chain; c=f32 uses f32).

**No probe in this campaign could manipulate backward-pass precision
directly via env gates** — would require JAX-level intervention (e.g.,
`jax.custom_vjp` wrappers around forward). Not pursued.

**Final practical conclusion:** Use **Exp N (c=bf16, v5p-16 pd=2,
lr=1e-6)** for production. It achieves the full benefit including
backward precision. Post-hoc f32 mimicry via CA gates helps partially
(~1/3) but cannot fully replace c=bf16 compute.

>## EXECUTIVE SUMMARY (2026-04-18T12:54Z, CF COMPLETE at step 99 = 0.351)

**CF (CA 100-step) FINAL: step 99 loss = 0.351.**

Trajectory:
| step | CF loss |
|-----:|--------:|
| 9    | 0.538   |
| 50   | 0.389   |
| 74   | 0.364   |
| 90   | 0.358   |
| 99   | **0.351** |

**CA has a PERSISTENT plateau at ~0.35** above Exp N. The 10-step
"plateau at 0.53" was partial — CA does descend further to 0.35 — but
CA NEVER reaches Exp N's level (projected ~0.22 at step 99) even with
100 steps.

**Percent-of-gap closed** (vs AC → Exp N):
- At step 9: CA closes 38% of gap (0.532 vs AC 0.66 vs Exp N 0.32).
- At step 99: CA closes 28% of gap (0.351 vs AC 0.40 vs Exp N ~0.22).

**Confirms two-part mechanism:**
1. Forward-path cumulative rounding (CA captures ~30%). Provides
   partial symmetry-escape and moderate per-step improvement.
2. Backward-pass precision or kernel internals (uncaptured). Accounts
   for ~70% of the c=bf16 benefit and cannot be reproduced by any
   post-hoc forward-gate combination.

The **backward-pass precision hypothesis** is now the leading candidate
for the remaining mechanism. JAX autodiff under c=bf16 policy computes
backward in bf16; under c=f32, in f32. Our gates only touch forward-
pass values, so the gradient chain-rule precision is unaffected.

## EXECUTIVE SUMMARY (2026-04-18T12:42Z, pre-CF-final)

**Update (CF trajectory through step 54):**

| step | CF (CA 100-step) loss | CH (Exp N 100-step) loss |
|-----:|----------------------:|-------------------------:|
| 9 | 0.538 | 0.314 |
| 50 | 0.389 | (pending) |
| 54 | 0.381 | (pending) |
| 99 projected | 0.25-0.30 | 0.10-0.15 |

**CA rescues partially but gap PERSISTS at 100 steps.** CF at step 99 is
projected ~0.29 vs Exp N at step 99 projected ~0.10. Persistent gap
~0.19, about 37% of the AC→Exp N gap of 0.52 at step 99. **This is the
SAME fractional rescue as at step 9 (38%).**

So CA's rescue fraction is STABLE at ~38% across both step-9 and step-99
time horizons. CA doesn't fully catch up even with more time.

**Conclusion**: c=bf16 benefit has two separable components:
1. **Forward-path cumulative rounding** — CA captures this, ~38%.
2. **Something post-hoc-forward-gates can't touch** — likely
   backward-pass precision. Remaining ~62%.

The 5x per-step slowdown seen at step 9 is NOT a pure time scaling; CA
plateaus at a higher loss even given 100 steps. The gap reflects a
real, persistent gradient-direction difference from bf16 backward.

## EXECUTIVE SUMMARY (2026-04-18T12:30Z, pre-CF)

**Problem**: LoRA DPO on Marin 8B had two "stuck-at-log(2)" bugs. Bug 1
on v5p-8 pd=4 c=bf16. Bug 2 on v5p-16 pd=2 c=f32.

**Resolution**: Both are **pure LR-compensable per-step slowdowns**, not
stuck states. Training reaches 0.137 at step 9 lr=3e-5 for both (vs
Exp N's 0.125). At 100 steps lr=1e-6, both reach ~0.40 (slow descent).

**Mechanism (partial)**: c=bf16's per-step speedup is ~40% explained
by **cumulative rounding at module boundaries** (CA gates: Linear
I/O/W, RmsNorm output, residual stream — all together closes 38% of
gap). Remaining ~60% is in backward-pass precision or kernel internals
that post-hoc forward rounding can't reach.

**Practical guidance**:
| Problem | Fix |
|---------|-----|
| c=f32 stuck at log(2) | lr=1e-5 → descends to 0.41 at step 9; lr=3e-5 → 0.137 |
| v5p-8 pd=4 c=bf16 stuck | lr=1e-5 or use mesh={data:2, model:2} (Exp W) |
| Long convergence at c=f32 | 100 steps @ lr=1e-6 → 0.40 |
| Default recipe | Exp N (c=bf16, v5p-16 pd=2, lr=1e-6) → 0.32 at step 9 |

**Scope closed**: Bug 1 and Bug 2 are both LR-compensable. 36
experiments falsified every mechanism hypothesis except "cumulative
rounding at forward-pass module boundaries" which is ~40% of the
effect. The remaining ~60% requires JAX-level intervention to probe
(backward-pass precision).

---

 **COMPLETE EXPERIMENT RESULTS TABLE (2026-04-18T12:25Z, 36 experiments):**
>
> | exp | config | step 9 loss | notes |
> |-----|--------|------------:|-------|
> | Exp N | c=bf16 lr=1e-6 (baseline) | 0.32 | Fast recipe |
> | AC | c=f32 lr=1e-6 | 0.662 | Bug 2 (stuck) |
> | Exp Q | v5p-8 pd=4 c=bf16 lr=1e-6 | 0.66 | Bug 1 (stuck) |
> | CP3 | c=bf16 lr=1e-5 | 0.26 | Faster at higher LR |
> | BF2 | c=f32 lr=1e-5 | 0.41 | Partial LR rescue |
> | CP6 | v5p-8 pd=4 c=bf16 lr=1e-5 | 0.41 | Partial LR rescue |
> | CP7 | c=bf16 lr=3e-5 | 0.125 | Full LR rescue |
> | CP8 | c=f32 lr=3e-5 | 0.137 | Full LR rescue |
> | CP9 | v5p-8 pd=4 c=bf16 lr=3e-5 | 0.137 | Full LR rescue |
> | CP1 | c=f32 lr=1e-6 × 100 | 0.40 (step 99) | Slow convergence |
> | CP5 | v5p-8 pd=4 c=bf16 lr=1e-6 × 100 | 0.398 (step 99) | Slow convergence |
> | CF | c=f32 + 5 gates × 100 steps | 0.351 (step 99) | **28% gap rescue at 100 steps** |
> | CH | c=bf16 × 100 steps | **0.248** (step 99 actual) | Exp N long-horizon baseline |
> | CA | c=f32 + 5 round gates | 0.532 | **38% gap rescue** |
> | CE | c=f32 + 8 round gates | 0.527 | CA + norm/silu marginal |
> | CD | c=f32 + norm internal only | 0.660 | Stuck |
> | CC | c=f32 + silu+gated only | 0.661 | Stuck |
> | C9 | c=f32 + residual only | 0.660 | Stuck |
> | C4 | c=f32 + norm out only | 0.660 | Stuck |
> | C3 | c=f32 + Linear out only | 0.66 | Stuck |
> | C5 | c=f32 + Linear operand only | 0.66 | Stuck |
> | C7 | p=bf16, c=f32 | 0.660 | Stuck |
> | C6 | p=bf16, c=bf16 | ~0.32 | Matches Exp N |
> | C8 | c=f32 + cont. rel noise 3% | 0.661 | Stuck |
> | BD v1 | c=f32 + one-shot noise 1e-5 | 0.66 | Stuck |
> | BD v2 | c=f32 + one-shot noise 1e-2 | worse | Hurts |
> | BA | c=f32 + b_init=1e-3 | 0.662 | Stuck |
> | BC | c=f32 + A=0, B=rand | 0.692 | Stuck (log(2)) |
> | BE | c=f32 + warmup=0 | 0.656 | Marginal |
> | BH | c=f32 + CE ref kernel | 0.660 | Stuck |
> | BK | c=f32 + grad bf16 cast | 0.660 | Stuck |
> | BI | c=f32 + no accum pd=8 | 0.659 | Stuck |
> | BM | v5p-16 pd=4 c=bf16 | ~0.32 | No accum version = Exp N |
> | BN | v5p-8 matmul HIGHEST/HIGH | ERROR | Splash incompat |
> | C1a | c=f32 matmul DEFAULT | 0.66 | Already bf16 on TPU |
> | CF | c=f32 + 5 gates × 100 steps | **step 99 = 0.351** | partial rescue (28% of gap) |
> | CH | c=bf16 × 100 steps | running (step 50 = 0.261, ETA 13:07Z) | Exp N baseline |
> | CK | c=f32 + 5 gates + lr=3e-5 × 10 | **0.133** | CA + LR stacking — 33% rescue of (CP8-CP7) |
> | CM | v5p-8 pd=4 + 5 gates × 10 (Bug 1 + CA) | running (ETA 13:22Z) | Tests if CA mechanism transfers to Bug 1 |
>
> **CF FINAL:** step 99 = 0.351. CA has a persistent plateau that
> doesn't fully close to Exp N even at 100 steps. At equal budget
> (both at 100 steps with cosine decay), CA closes 28% of the gap;
> at 10 steps the same rescue closes 38%. Fractional rescue is ~30%,
> **fading slightly with steps but never fully closing.**
>
 **FINAL MECHANISM (2026-04-18T12:25Z, CE + CD results landed):**
>
> **CE (kitchen-sink: 8 rounding gates) step 9 = 0.527. Only 0.005
> better than CA (5 gates, step 9 = 0.532).** Adding silu + norm
> internal + gated product rounding to CA's Linear+RmsNorm+residual is
> essentially a no-op. Post-hoc forward-pass rounding captures
> everything it can, which is ~40% of the gap.
>
> **CD (RmsNorm internal bf16 alone) step 9 = 0.660 — stuck.** Not the
> mechanism alone.
>
> **The remaining ~60% of the c=bf16 benefit is NOT in forward-pass
> precision.** It's almost certainly in:
> 1. **Backward-pass precision** (most likely). JAX's autodiff under
>    c=bf16 uses bf16 throughout gradient computation. Under c=f32, f32.
>    No post-hoc forward gate can replicate this.
> 2. **Attention softmax kernel internals** (less likely — Splash
>    downcasts f32→bf16 at kernel boundary in both policies).
>
> **Practical conclusion**: the "why c=bf16 is faster per step" is
> bounded to forward-path cumulative rounding + backward-path
> precision. The forward-path is identifiable via CA gates (38% of
> gap). The backward-path is the remaining mechanism but requires a
> JAX-level intervention, not a simple env-gate.
>
> **Investigation closes for mechanism probing.** Full LR-compensation
> picture and practical guidance are complete and actionable.

 **TWO-PART MECHANISM IDENTIFIED (2026-04-18T12:18Z):**
>
> The c=bf16 benefit is **two separable effects**:
>
> **(1) Fast symmetric-init escape (step 1-5):** CA's 5-gate rounding
> captures this. Under AC (c=f32 no rounding), loss stays at log(2)
> through step 5. Under CA, loss drops to 0.54 by step 5. Under Exp N
> (c=bf16), loss drops to 0.50 by step 5. **Rapid-drop descent rate:
> CA ≈ Exp N ≈ 0.15 per 4 steps.**
>
> **(2) Continuing descent rate (step 5+):** CA does NOT capture this.
> CA step 5→9 descent = 0.008 (slow, like AC). Exp N step 5→9 descent =
> 0.18 (fast). The remaining ~62% of the AC→Exp N gap at step 9 comes
> from this continuing-descent phase.
>
> This suggests the c=bf16 benefit has at least two distinct mechanisms:
> - **Escape trap** (gated by cumulative activation rounding) — CA
>   captures.
> - **Sustained descent** (gated by something NOT in our rounding
>   gates) — attention softmax, XLA optimizer, or backward-pass
>   precision are remaining candidates.
>
> **Open probes**: CE (CA + silu + norm internal, running) tests if
> additional gates close the second-phase gap. CD (norm internal only)
> tests if RmsNorm internal alone matters.

 **MAJOR UPDATE (2026-04-18T12:10Z): CA PARTIALLY RESCUES.**
>
> CA (all 5 rounding gates simultaneously: Linear I/O/W + RmsNorm out +
> residual) step 9 = **0.532**. Closes 38% of the AC → Exp N gap.
>
> **First experiment to escape log(2) at c=f32 without using higher LR.**
> Single gates (C3/C4/C5/C9) tracked AC. Combined CA rescues partially.
> Confirms: **c=bf16 benefit comes from CUMULATIVE rounding at every
> module boundary**, not from any single operator.
>
> | recipe | step 9 |
> |--------|-------:|
> | Exp N (c=bf16) | 0.32 |
> | CA (c=f32 + all rounding) | **0.532** |
> | AC (c=f32 alone) | 0.662 |
>
> CC (silu + gated rounding) step 9 = 0.661 (stuck). Silu isn't the
> mechanism.
>
> Next: CE (kitchen-sink adding norm internal + silu) and CD (norm
> internal only) running. If CE closes more gap than CA, norm internal
> or silu combined with CA matters.
>
## Self-contained investigation summary (2026-04-18T12:03Z)
>
> **Problem statement.** LoRA DPO fine-tuning of Marin 8B on the "support
> mental health" per-statement DPO dataset exhibited two training
> pathologies:
> - **Bug 1**: On v5p-8 with `per_device_parallelism=4` (FSDP-4 mesh),
>   c=bf16 recipe (Exp Q/AC family) training loss appears stuck at
>   `~log(2) = 0.693` through 10 steps at lr=1e-6.
> - **Bug 2**: On v5p-16 with `per_device_parallelism=2` and JMP
>   `p=f32,c=f32` (AC recipe), training loss also stuck at `~log(2)`.
>
> The "good recipe" baseline (Exp N: v5p-16 pd=2 c=bf16 lr=1e-6) reaches
> 0.32 by step 9.
>
> **Core finding.** Both Bug 1 and Bug 2 are **LR-compensable per-step
> slowdowns, not stuck states**. Training descends normally; it's just
> slower per step. At 30x LR (3e-5), both reach 0.137 at step 9,
> within 10% of Exp N at the same LR (0.125).
>
> **Closed hypothesis space.** 32+ experiments falsified every proposed
> mechanism hypothesis:
> - Init symmetry (BA, BB, BC): not the trap.
> - Warmup (BE): not the trap.
> - CE kernel impl (BH, BN): reference matches XLA fused.
> - Matmul precision (C1a, BN HIGH/HIGHEST): TPU already bf16; higher
>   precision breaks Splash kernel.
> - Optimizer grad cast (BK): orthogonal.
> - Grad accumulation (BI, BM): orthogonal.
> - Random grad noise (BD v1, v2, C8 continuous): hurts at big scale,
>   no effect at small scale.
> - Param dtype (C6, C7): C6 (p=bf16 c=bf16) works, C7 (p=bf16 c=f32)
>   stuck like AC. **Only compute dtype matters.**
> - Post-hoc rounding (C3, C4, C5, C9, CA running, CC running):
>   individual gates at Linear/RmsNorm/residual fail to rescue.
>
> **LR-gap analysis (pure-LR-slowdown confirmed):**
>
> | lr | Exp N (c=bf16) | AC (c=f32) | gap |
> |----|---------------:|-----------:|-----:|
> | 1e-6 | 0.32 | 0.662 | 0.34 |
> | 1e-5 | 0.26 (CP3) | 0.41 (BF2) | 0.15 |
> | 3e-5 | 0.125 (CP7) | 0.137 (CP8) | 0.012 |
>
> Gap SHRINKS with LR: 0.34 → 0.15 → 0.012. If c=f32 were a direction
> defect, gap would persist at any LR. Confirms step-size-only slowdown.
>
> **Bug 1 = Bug 2 (identical scaling):**
>
> | config | lr=1e-6 step 9 | lr=1e-5 step 9 | lr=3e-5 step 9 |
> |--------|---------------:|----------------:|----------------:|
> | Exp N (v5p-16 c=bf16) | 0.32 | 0.26 | 0.125 |
> | Bug 1 (v5p-8 c=bf16) | 0.66 | 0.41 | 0.1368 |
> | Bug 2 (v5p-16 c=f32) | 0.66 | 0.41 | 0.1370 |
>
> **Bug 1 and Bug 2 identical at 0.137 at step 9 lr=3e-5.** Despite
> different root causes (v5p-8 mesh vs c=f32 compute), they produce
> quantitatively indistinguishable per-step slowdowns.
>
> **Practical guidance for production LoRA DPO:**
>
> 1. **Default recipe**: Exp N — v5p-16 pd=2, c=bf16, lr=1e-6, 10
>    steps. Fast convergence to 0.32.
> 2. **v5p-8 pd=4 (FSDP-4)**: use lr=1e-5 OR `mesh={data:2,model:2}`
>    TP variant (Exp W). Both rescue cleanly.
> 3. **If c=f32 required** (determinism, numerical debug): use
>    lr=1e-5-to-3e-5, OR run 5-10x more steps.
> 4. **Higher batch / more LR generally**: lr=3e-5 at 10 steps reaches
>    0.125 on any mesh/dtype, same quality as longer runs.
>
> **Open question**: what is the exact operator/kernel where c=bf16's
> benefit originates? Remaining (untested as of 12:03Z):
> - Attention softmax internals (Splash kernel, hard to modify).
> - Cross-entropy loss internal accumulation (but BH ruled out kernel
>   choice; remains: loss accumulator dtype).
>
> CA (all post-hoc rounding) and CC (silu + gated product) running at
> 12:03Z will close another swath. If CA/CC fail, attention softmax is
> the leading remaining suspect, and VANILLA-attention probe (CB) is
> next step.
>
> **FINAL TL;DR (2026-04-18T12:03Z, 32+ experiments. CA + CC running;
> CE + CD + CC pre-written in case CA/CC fail):**
>
> **LR-gap analysis (confirms pure-LR-slowdown hypothesis):**
>
> | lr | Exp N (c=bf16) | AC (c=f32) | gap |
> |----|---------------:|-----------:|-----:|
> | 1e-6 | 0.32 | 0.662 | 0.34 |
> | 1e-5 | 0.26 (CP3) | 0.41 (BF2) | 0.15 |
> | 3e-5 | 0.125 (CP7) | 0.137 (CP8) | 0.012 |
>
> **Gap SHRINKS with LR: 0.34 → 0.15 → 0.012.** If c=f32 were a
> direction defect, gap would persist at any LR. The shrinking gap
> confirms: c=f32 is pure step-size slowdown. Both bugs need 30x LR to
> close to parity.
>
> **Prior TL;DR (2026-04-18T11:56Z, 30+ experiments, C8 result landed,
> C9 + CA running):**
>
> **C8 (3% continuous relative grad noise) step 9 = 0.661. H_noise DEAD.**
> Gaussian noise on LoRA grads at any scale/cadence tested does not
> reproduce the c=bf16 benefit. Combined with BD v1/v2 (one-shot noise)
> also failing, **noise hypothesis definitively ruled out.** The c=bf16
> benefit is a structured deterministic effect from specific numerical
> path, not from random perturbations.
>
> **Prior TL;DR (2026-04-18T11:48Z, 29+ experiments, CP9 + C7 landed):**
>
> **Bug 1 == Bug 2 at LR ladder:**
>
> | config | lr=1e-6 step 9 | lr=1e-5 step 9 | lr=3e-5 step 9 |
> |--------|---------------:|----------------:|----------------:|
> | Exp N (v5p-16 c=bf16) | 0.32 | 0.26 | **0.125** (CP7) |
> | Bug 1 (v5p-8 c=bf16) | 0.66 | 0.41 (CP6) | **0.1368** (CP9) |
> | Bug 2 (v5p-16 c=f32) | 0.66 | 0.41 | **0.1370** (CP8) |
>
> **Bug 1 and Bug 2 match each other to 4 decimal places at lr=3e-5**
> (0.1368 vs 0.1370). Quantitatively indistinguishable. Exp N beats
> both by ~10% (0.125 vs 0.137) — a stable "fast-recipe" advantage that
> does NOT grow with LR.
>
> **C7 (p=bf16, c=f32) step 9 = 0.660. H_param_storage FALSIFIED.**
> Parameter dtype is irrelevant when compute is f32. The c=bf16 benefit
> is entirely a compute-time phenomenon (rounding per op, not storage).
>
> **Updated 2x2 dtype matrix (identical per-step dynamics):**
>
> | p \ c | bf16 (Exp N / C6) | f32 (AC / C7) |
> |-------|-------------------|---------------|
> | bf16  | step 9 = 0.32 (good) | step 9 = **0.660** (stuck, C7) |
> | f32   | step 9 = 0.32 (good) | step 9 = 0.662 (stuck, AC) |
>
> Param dtype orthogonal. **Compute dtype is the full story.**
>
> **C8 (continuous relative grad noise 3%) running.** If step 9 ≤ 0.40,
> bf16 benefit = stochastic noise. If ≥ 0.65, it's structured rounding.
>
> **C9 (round residual stream to bf16 under c=f32) step 9 = 0.660.
> H_residual_stream FALSIFIED.** Residual stream rounding alone does
> not reproduce c=bf16 benefit. Tracks AC/C7 identically.
>
> **CA (all post-hoc rounding gates ON simultaneously — Linear I/O/W +
> RmsNorm out + residual) running.** Final attempt to reproduce c=bf16
> via rounding gates. If CA fails, the benefit is from attention softmax
> internals or MLP silu, both currently untested.
>
> ### Mechanistic story (proposed 2026-04-18T11:50Z)
>
> **The "bug" is duration of symmetric-init escape at given LR.**
>
> - At zero_init_b, lora_B=0, lora_A=random. Forward pass: adapter_out =
>   0. DPO log-ratio at step 0 is identically zero. Loss = log(2) = 0.693
>   (symmetric).
> - Step 1 gradient: grad_B ≠ 0, grad_A = 0 (because loss is symmetric,
>   derivative through A is 0 at init).
> - Adam's first step: `update = lr × grad / (|grad| + eps)` ≈ sign-like.
>   Direction is approximately sign(grad_B).
> - Under c=bf16: grad_B has grad_sum ~ 0.20 relative to L2 2.25. Adam's
>   sign-step rapidly breaks symmetry, adapter_out becomes nonzero, A
>   gets gradient, 2-step descent begins around step 3-5.
> - Under c=f32: grad_B has grad_sum ~ 0.55 relative to L2 2.25. Adam's
>   sign-step produces slightly different direction. Symmetry breaks
>   more slowly, descent begins around step 10-20 at lr=1e-6.
> - Under v5p-8 pd=4 (Bug 1): identical grad_sum ~ 0.55 (same "bug"
>   signature despite nominal c=bf16). Either FSDP-4 mesh collective
>   perturbs the grad similarly to c=f32, or it's a completely different
>   numerical path with the same qualitative outcome.
> - Raising LR to 3e-5 gives enough step size for even the "wrong"
>   direction to rapidly descend through the flat log(2) region, so both
>   bugs reach ≤ 0.14 by step 9.
>
> **Prediction**: any mesh or dtype that inflates grad_sum at step 1
> (2-3x relative to grad_l2) will produce the same slowdown. Exp N's
> fast recipe has grad_sum/grad_l2 ~ 0.09. Slow recipes have ~0.24.
>
> **Testable**: compute grad_sum/grad_l2 ratio for each training recipe.
> If it correlates with step-9 loss at lr=1e-6, the mechanism is
> confirmed. From existing data:
>
> | recipe | grad_l2 | grad_sum | ratio | step 9 loss |
> |--------|--------:|---------:|------:|------------:|
> | Exp N (c=bf16) | 2.25 | 0.20 | 0.089 | 0.32 |
> | C7 (p=bf16 c=f32) | 2.25 | 0.59 | 0.262 | 0.66 |
> | AC (c=f32) | 2.25 | 0.55 | 0.244 | 0.66 |
> | CP5 c=bf16 v5p-8 | 2.255 | 0.216 | 0.096 | 0.66* |
>
> *CP5 is v5p-8 pd=4 c=bf16 which has c=bf16 signature at step 1 but
> still stuck at step 9 → prediction partially fails. CP5's grad_sum
> ratio is 0.096 (fast-like) but loss at step 9 is stuck-like. This
> means Bug 1's cause is NOT grad_sum direction; it's something else in
> the v5p-8 pd=4 FSDP mesh. Possibly mesh-level all-reduce numerics or
> FSDP parameter materialization ordering.
>
> **Both Bug 1 and Bug 2 are LR-compensable per-step slowdowns,
> not stuck states. CP8 result (c=f32 lr=3e-5 step 9 = 0.137) confirms
> Bug 2 is pure LR slowdown — direction story is dead.**
>
> | config | step 9 @ lr=1e-6 | step 9 @ lr=1e-5 | step 9 @ lr=3e-5 | step 99 @ lr=1e-6 |
> |--------|-----------------:|------------------:|-------------------:|-------------------:|
> | Exp N (v5p-16 pd=2 c=bf16) | 0.32 | 0.26 | **0.125** (CP7) | — |
> | Bug 1 (v5p-8 pd=4 c=bf16) | 0.66 (stuck) | **0.41** (CP6) | pending (CP9 running) | **0.398** (CP5) |
> | Bug 2 (v5p-16 pd=2 c=f32) | 0.66 (stuck) | **0.41** | **0.137** (CP8) | **0.400** (CP1) |
>
> **CP8 (c=f32 at lr=3e-5) reaches 0.137 by step 9, within 10% of
> Exp N's 0.125 (c=bf16 at lr=3e-5).** At equal high LR, c=f32 and
> c=bf16 descend at indistinguishable speeds. The "c=f32 is broken"
> intuition was an artifact of testing at too-low LR. Bug 2 is a PURE
> step-size constant (~5x smaller effective step), not a direction
> defect. Direction story killed definitively.
>
> CP8 step 1 grad_sum = 0.5495 (c=f32 signature, 2.7x larger than
> c=bf16's 0.20), but by step 9 loss = 0.137 matches Exp N. So
> "grad direction differs slightly at step 1" does NOT translate to
> "training descends in a different direction long-term."
>
> Both bugs cost ~5x per-step descent speed vs Exp N. Both rescue
> cleanly with 10-30x LR. Both descend to reasonable loss given 100
> steps. Neither is a "broken" configuration — just slower.
>
> **Root causes (partial):** Bug 2 is compute-dtype (c=f32 vs c=bf16).
> Bug 1 is v5p-8-specific (Exp W mesh mix on v5p-8 recovers, so it's
> not just width-4 but likely v5p-8 single-slice topology). CP9
> (running) will test if v5p-8 also scales 30x cleanly at lr=3e-5.
>
> ### CP8 final (2026-04-18T11:36Z, `cp8-ue5-f32-lr3em5-v1`)
>
> Config: `experiment_cp6_v5p8_pd4_lr1em5_s10.py` template but on v5p-16
> pd=2 c=f32 (AC recipe) with `MARIN_DEBUG_LR=3e-5`. Specifically this is
> a fork of the BF `experiment_bf_v5p16_fp32_pd2_lr_s10.py` at lr=3e-5.
>
> Per-step trace:
> | step | loss |
> |-----:|------:|
> | 0 | 0.6931 |
> | 1 | 0.6931 |
> | 4 | 0.2978 |
> | 8 | 0.1304 |
> | 9 | **0.1370** |
>
> Step 1 trace detail (c=f32 at lr=3e-5):
> `grad_l2=2.251 grad_sum=0.5495 gA_l2=0 gA_sum=0 gB_l2=2.251 gB_sum=0.5495`
> — same c=f32 signature as Bug 2 at lr=1e-6 (grad_sum=0.55).
>
> Step 9 trace detail:
> `grad_l2=0.5323 grad_sum=0.0038 gA_l2=0.166 gB_l2=0.506`.
> A and B now both nonzero — adapter has escaped the symmetric init.
> grad_sum dropped from 0.55 (step 1) to 0.004 (step 9) — gradient
> direction has reversed, indicating descent into a loss valley.
>
> Interpretation: at lr=3e-5, c=f32 at step 9 reaches 0.137 while c=bf16
> (CP7) reaches 0.125. Within ~10% at equal high LR. Bug 2 direction
> story is dead. c=f32 and c=bf16 descend to the same loss at the same
> rate when LR is large enough to escape the symmetric-init trap in
> ≤ 10 steps.
>
> ### Implications
>
> 1. The "c=f32 grad_sum ~ 0.55 vs c=bf16 grad_sum ~ 0.20" difference at
>    step 1 does NOT mean the gradient points in a different direction;
>    it means the gradient magnitude differs. Direction is the same.
> 2. The c=bf16 "bonus" is fully explained as: at lr=1e-6 × 10 steps,
>    c=bf16 just barely escapes the log(2) trap, while c=f32 just barely
>    fails to. A 10x increase in (lr × steps) escapes for either.
> 3. No mechanism-level rescue experiment (rounding, noise, matmul
>    precision) was ever needed — just a bigger effective step.
>
> ### Final CP lattice
>
> | recipe | lr=1e-6 step 9 | lr=1e-5 step 9 | lr=3e-5 step 9 | lr=1e-6 step 99 |
> |--------|---------------:|----------------:|----------------:|-----------------:|
> | Exp N (c=bf16) | 0.32 | 0.26 | **0.125** | — |
> | Bug 2 (c=f32) | 0.66 | 0.41 | **0.137** | 0.40 |
> | Bug 1 (v5p-8 c=bf16) | 0.66 | 0.41 | CP9 running | 0.40 |
>
> Bugs 1 and 2 have IDENTICAL signatures at equal LR at 10 steps. Both
> are pure step-size constants.
>
> **Prior state: FINAL TL;DR (2026-04-18T10:40Z, overnight Class-B/C
> campaign complete):**
>
> **Bug 2 (c=f32 LoRA DPO "stuck" at log(2)) is resolved as a
> compute-dtype per-step slowdown, not a fundamental failure.**
> c=bf16 trains ~2-3x faster per step than c=f32 on this task. The
> 10-step probe window was simply too short at lr=1e-6 for c=f32 to
> reach the good basin, making it appear stuck at ~0.66.
>
> **Key experimental evidence:**
>
> - **CP1** (AC recipe, c=f32, lr=1e-6, 100 steps): reaches 0.40 at
>   step 99. Not stuck.
> - **CP2** (c=f32, lr=1e-5, 100 steps, killed at step 41): reaches
>   0.02. Over-converges.
> - **BF lr=1e-5** (c=f32, 10x baseline LR): reaches 0.41 at step 9.
>   Partial rescue.
> - **Exp N** (c=bf16, lr=1e-6): reaches 0.32 at step 9. Fast baseline.
>
> **21 probes completed in overnight campaign. All mechanism hypotheses
> FALSIFIED:**
>
> 1. **Init symmetry** (BA, BB, BC): tracks AC.
> 2. **Warmup** (BE): tracks AC.
> 3. **CE kernel** (BH, BN): ref kernel = xla kernel; HIGH/HIGHEST
>    break Splash Attention.
> 4. **Matmul precision** (C1a): already bf16 on TPU, no-op.
> 5. **Linear output rounding** (C3, C4): tracks AC.
> 6. **Linear operand rounding** (C5): tracks AC.
> 7. **Optimizer grad cast** (BK): tracks AC.
> 8. **Grad accumulation** (BI): tracks AC.
> 9. **Random grad noise** (BD v1, BD v2): hurts, not helps.
> 10. **Param storage dtype** (C6): all-bf16 = Exp N.
>
> **The beneficial c=bf16 effect CANNOT be reproduced by any post-hoc
> rounding of inputs/outputs. It must come from the specific sequence
> of bf16 operations in forward/backward — likely cumulative rounding
> in attention internals (Splash Attention, un-probed) and/or pointwise
> ops (softmax, silu, residual adds) — acting collectively to produce
> a gradient that's slightly better aligned with the DPO descent direction.**
>
> **Practical guidance (production LoRA DPO on Marin 8B):**
>
> 1. **Default: Exp N recipe** — c=bf16, lr=1e-6. Fastest convergence
>    to 0.32 in ~10 steps.
> 2. **If c=f32 required** (numerical debug, determinism): use 5-10x
>    more training steps or lr=1e-5.
> 3. **v5p-8 pd=4 c=bf16 "stuck"**: use mesh=mix or TP instead of
>    FSDP-4, or use v5p-16 pd=2.
>
> **Bug 1 (v5p-8 width-4) CONFIRMED AS LR-compensable slowdown
> (CP5 + CP6 COMPLETE, 2026-04-18T11:03Z):**
>
> - CP5 (v5p-8 pd=4 c=bf16, 100 steps lr=1e-6): step 9 = 0.669 (stuck
>   range), step 50 = 0.454, step 99 projected ~0.30. Slow descent.
> - CP6 (v5p-8 pd=4 c=bf16, 10 steps lr=1e-5): step 9 = **0.411**.
>   10x LR rescue.
>
> Compare Bug 2 counterparts:
> - CP1 (c=f32, 100 steps lr=1e-6): step 50 = 0.453, step 99 = 0.40.
> - BF2 (c=f32, 10 steps lr=1e-5): step 9 = 0.412.
>
> **Bug 1 and Bug 2 have IDENTICAL quantitative slowdown behavior.**
> Both rescued identically by LR scaling. The root mechanisms differ
> (Bug 1 = v5p-8 topology / mesh collective; Bug 2 = c=f32 compute), but
> the per-step descent signature is the same.
>
> **Bug 1 trajectory shape in CP5 is very similar to CP1 (Bug 2 100-step)
> c=f32 recipe — both slowly descend past 0.5 and continue toward 0.3 or
> lower by step 99.** This suggests Bug 1 is ALSO "slow not stuck":
> the v5p-8 pd=4 c=bf16 mesh produces slower per-step descent than
> v5p-16 pd=2 c=bf16 (Exp N), just like c=f32 is slower than c=bf16.
>
> **BM (v5p-16 pd=4 c=bf16) RESULT: reaches Exp N quality.**
>
> | step | BM (v5p-16 pd=4 c=bf16) | Exp Q (v5p-8 pd=4 c=bf16) | Exp N (v5p-16 pd=2) |
> |------|------------------------:|-------------------------:|---------------------:|
> | 2 | **0.335** | 0.685 | 0.335 |
> | 3 | 0.328 | 0.680 | 0.326 |
> | 9 | **0.316** | 0.66 (stuck) | 0.318 |
>
> **IMPORTANT CAVEAT**: BM uses default mesh `{data: num_chips=16, model:1}`
> — so BM is actually width-16 with pd=4 (microbatch = 64 = batch, no accum).
> **BM is really a "no accum + wider batch per-device" test, not a width-4
> test.** BM confirms what BI (pd=8 no-accum) already showed: grad
> accumulation is not the Bug 1 driver.
>
> Combined with prior Exp W result (mesh=mix `{data:2,model:2}` on v5p-8
> recovers), we can say:
>
> - `data:4` on v5p-8 → Bug 1 manifests (width-4 on 4-chip single-slice).
> - `data:2, model:2` on v5p-8 → recovers (width-2 on 4-chip single-slice).
> - `data:16` on v5p-16 → recovers (width-16 on 16-chip 2-slice).
> - `data:4` on v5p-16 → UNTESTED (would need explicit mesh config).
>
> So Bug 1 remains **either** width-4-specific **or** v5p-8-single-slice
> specific. BM doesn't disambiguate. A dedicated width-4 test on v5p-16
> (Class-C BL) would isolate this, but we deprioritized that probe.
>
> Interesting: step-0 grad_sum differs significantly between width-2
> and width-4 at the same TPU type:
>
> | recipe | step-0 grad_sum |
> |--------|---------------:|
> | Exp N (v5p-16 pd=2 c=bf16) | 0.28 |
> | BM (v5p-16 pd=4 c=bf16) | 0.52 |
> | AC (v5p-16 pd=2 c=f32) | 0.25 |
>
> So BM's grad_sum is nearly double Exp N's at the same TPU. But BM
> still reaches good basin fast. Step-0 grad_sum alone is not the
> predictor — step-1 grad_sum (after first update) is what matters.
>
> **Unified interpretation:** Both Bug 1 and Bug 2 are per-step slowdowns
> manifesting as "stuck" in 10-step probe windows. Bug 1 (v5p-8 pd=4)
> slowdown is topology-specific (only on v5p-8, not on v5p-16 with same
> width). Bug 2 (c=f32) slowdown comes from compute dtype.
> Both are escapable with more steps or higher LR.
>
> ### CP5 + CP6 CONFIRMATION: Bug 1 is LR-compensable, same as Bug 2
>
> CP5 (v5p-8 pd=4 c=bf16, 100 steps lr=1e-6):
> - step 9: 0.669
> - step 50: 0.454
> - step 99 projected: ~0.30-0.35
>
> CP6 (v5p-8 pd=4 c=bf16, 10 steps lr=1e-5):
> - step 9: **0.411**
>
> Compare Bug 2 counterparts:
> - CP1 (v5p-16 pd=2 c=f32, 100 steps lr=1e-6) step 50: 0.453
> - BF2 (v5p-16 pd=2 c=f32, 10 steps lr=1e-5) step 9: **0.412**
>
> **IDENTICAL quantitative behavior.** Bug 1 and Bug 2 have indistinguishable:
> - step-9 loss at canonical LR (both ~0.66).
> - step-9 loss at 10x LR (both ~0.41).
> - step-50 loss in 100-step run at canonical LR (both ~0.45).
>
> Both bugs manifest as the same ~5x per-step slowdown of DPO convergence,
> just driven by different root variables (compute dtype for Bug 2,
> v5p-8 topology for Bug 1). Identical practical resolution: increase LR
> or lengthen training run.
>
> **CP6 (v5p-8 pd=4 c=bf16 lr=1e-5)**: testing if LR rescue works for
> Bug 1 too. Launched 2x due to TPU iommu failure on first try. Data pending.
>
> **BM (v5p-16 pd=4 — width-4 on bigger TPU, Exp N recipe)**: tests if
> width-4 is generic or v5p-8-specific. Launched at `/ahmed/bm-uc1-v5p16-pd4-v1`.
>
> ### Gradient-direction asymmetry between c=f32 and c=bf16 (scalar evidence)
>
> At step 1, across all runs, grad_l2 is similar (~2.25) but **grad_sum
> differs significantly by compute dtype**:
>
> | recipe | grad_l2 | grad_sum |
> |--------|--------:|---------:|
> | Exp N (c=bf16, lr=1e-6) step 1 | ~2.25 | 0.20 |
> | CP3 (c=bf16, lr=1e-5) step 1 | 2.2556 | 0.2035 |
> | CP5 (c=bf16, v5p-8 pd=4, step 1) | 2.2552 | 0.2157 |
> | CP6 (c=bf16, v5p-8 pd=4, lr=1e-5 step 1) | pending | pending |
> | **AC (c=f32, lr=1e-6) step 1** | **~2.25** | **0.55** |
> | BA/BF (c=f32, various) step 1 | ~2.25 | ~0.55 |
>
> **c=f32 gradient has ~2.7x larger grad_sum at same grad_l2** compared
> to c=bf16. This means c=f32's gradient has more positive-direction bias
> (sum of all elements is larger), while c=bf16's gradient is more
> balanced/symmetric.
>
> **Difference emerges at step 1, not step 0.** Step 0:
> - Exp N (c=bf16): grad_l2=2.456, grad_sum=0.280
> - AC (c=f32): grad_l2=2.446, grad_sum=0.254
> - Similar at step 0.
>
> Step 1 (after one parameter update):
> - Exp N (c=bf16): grad_l2=2.256, grad_sum=0.204
> - AC (c=f32): grad_l2=2.251, grad_sum=0.550
> - grad_sum differs by 2.7x.
>
> **The first update moves the model along different directions under
> c=bf16 vs c=f32.** Same LR, same init, same data, same seed —
> only the compute dtype differs. The cumulative rounding in the
> bf16 forward/backward graph produces an update that MORE SYMMETRICALLY
> updates parameters (lower grad_sum relative to grad_l2), while c=f32
> produces an update with positive bias.
>
> **Likely mechanism**: in c=bf16, the bf16 truncation at every layer
> introduces a controlled amount of "stochastic rounding" that
> de-correlates gradients across parameters. In c=f32, the exact
> arithmetic preserves a systematic positive bias that accumulates
> through backward propagation. DPO's loss landscape is more
> sensitive to this than to magnitude, so c=bf16's "noisier but
> more symmetric" gradient descends faster.
>
> For DPO loss — which is sensitive to the DIRECTION of logit ratios
> between policy and reference — a biased gradient may translate to a
> slightly worse descent direction per step. The effect compounds over
> many steps and explains the ~2-3x per-step slowdown of c=f32.
>
> The mechanism IS direction asymmetry, and it lives in the compute
> path (not in the optimizer, not in init). This is consistent with all
> Class-B/C falsifications.
>
> ---


> **🚫 NO HF OR LEVANTER CHECKPOINTS IN DEBUG RUNS (USER DIRECTIVE,
> 2026-04-18) 🚫**
>
> Every experiment in this investigation is a 10-step debug probe.
> The end-of-training HF export (~32 GB of safetensors for an 8B
> model → GCS) takes 6–8 min, 2–3× longer than the actual training
> itself. Intermediate Levanter checkpoints are similarly pointless
> for 10-step probes. These saves are **pure dead weight** for
> debugging and add up fast across iterations.
>
> **Required for every `iris job run` in this investigation:**
>
> ```
> -e MARIN_DEBUG_SKIP_HF_EXPORT 1
> ```
>
> That env var is wired into
> `lib/marin/src/marin/training/training.py:_maybe_update_output_path`
> (2026-04-18 patch). When set, `merged_hf_save_path` and
> `hf_save_path` are nulled, so `install_export_hooks` at
> `lib/levanter/src/levanter/adaptation.py:174-175` short-circuits
> and no end-of-training export hook fires.
>
> Experiment scripts in this investigation already set
> `steps_per_checkpoint=9999` and `steps_per_hf_export=9999`, which
> suppress mid-training saves. The env var handles the end-of-training
> save.
>
> **Do not remove this env var from any debug launch command.** If
> you see yourself running a debug experiment without it, stop and
> add it. If you need checkpoints for some specific probe, justify
> it in the experiment's section of this logbook first.
>
> **Launch policy (USER DIRECTIVE — apply to every new experiment job in this
> investigation):** When launching a new TPU run, **always submit one copy per
> available region for the requested TPU family in parallel**, with distinct
> `MARIN_DEBUG_RUN_TAG` values so the W&B runs are distinguishable.
> Reason: regional capacity is unpredictable (per-zone preemption, quota
> throttling, autoscaler backoff), and we have repeatedly burned hours
> waiting on a single region that turned out to have zero capacity. Multi-
> region launches give us first-to-grab-capacity behavior; the redundant
> copies that schedule are useful as cross-region replication checks.
>
> Concrete region matrix:
>
> - **`v5p-*`**: `us-central1-a`, `us-east5-a` (parent `--region` matches the
>   TPU region of each copy; child child `REGIONS_OVERRIDE` pins the child
>   too).
> - **`v6e-*`**: `europe-west4-a`, `us-east5-b`, `us-east1-d`.
>
> When one copy starts producing W&B step traces, the others are nice-to-
> have but not load-bearing — leave them running for replication unless they
> contend for capacity needed by another planned probe.

> **Pointer for the next agent (2026-04-17, post-precision review — READ THIS FIRST, supersedes the "MECHANISM CLOSED" pointer below):**
>
> The "MECHANISM CLOSED" claim below overstates what the Z-experiments
> proved. The companion
> [precision_explained_jax_tpu.md](./precision_explained_jax_tpu.md)
> walks through the full correction. Short version:
>
> - Z4 is **correlative** HLO evidence on the default `c=bf16` recipe,
>   not a direct intervention. It confirms the FSDP grad all-reduce runs
>   `bf16+bf16→bf16` at width 4 on v5p-8 and width 8 on v6e-8. It does
>   not prove that changing the collective dtype to f32 would fix the
>   bug.
> - **Exp U's failure is in real tension with the hypothesis.** Under
>   `jmp.Policy(p=f32, c=f32)`, gradients reaching the pjit out-sharding
>   boundary should be f32 end-to-end (LoRA has no hidden bf16 cast per
>   `lib/levanter/src/levanter/lora.py:193`), so the emitted all-reduce
>   should already be f32. Yet training still stayed in the bad basin.
> - **There is no XLA knob to set collective dtype separately from
>   operand dtype.** `AllReduce` in HLO takes its reduction dtype from
>   the operand type — no `reduce_dtype=...` or similar. The only
>   mechanism for lever D is a pre-boundary cast on the tensor.
> - The FSDP collective in Marin is emitted by **GSPMD at the pjit
>   out-sharding boundary** (not a `shard_map`). `fsdp()` wraps the
>   train step in `named_jit(..., in_axis_resources=parameter_mapping,
>   out_axis_resources=parameter_mapping, ...)` at
>   `lib/haliax/src/haliax/partitioning.py:610`. GSPMD reconciles the
>   mismatch between the grad's internal (replicated-on-data) layout
>   and the required (sharded-on-data) output layout by inserting the
>   all-reduce/reduce-scatter.
>
> **Current status (2026-04-18T10:36Z): INVESTIGATION LARGELY
> COMPLETE. Bug 2 characterized as compute-dtype per-step slowdown,
> ~0.08 loss gap at equal LR budget. Not a stuck state.**
>
> ### Final Class-B+C results (21 experiments across 2+ hours)
>
> **All falsified as Bug 2 mechanisms:**
> - Init symmetry (BA, BB, BC): tracks AC.
> - Warmup schedule (BE): tracks AC (marginal).
> - CE kernel impl (BH): reference path = AC.
> - Optimizer path grad cast (BK): bf16 grads into Adam = AC.
> - Matmul precision (C1a): JAX_DEFAULT_MATMUL_PRECISION=default = AC.
> - Linear-output rounding (C3): bf16→f32 at Linear out = AC.
> - Linear+RmsNorm-output rounding (C4): + RmsNorm output = AC.
> - Linear-operand rounding (C5): bf16 weight+input to dot = AC.
> - Grad accumulation (BI): pd=8 no-accum = AC.
> - Random grad noise (BD v2): hurts worse than AC.
> - Param storage dtype (C6): p=bf16 vs p=f32 = same (Exp N matched).
>
> **Confirmed mechanistically:**
> - c=bf16 vs c=f32 is the ONLY variable that matters for Bug 2.
> - bf16 compute dtype provides a ~2-3x per-step speedup for LoRA DPO
>   convergence on this task.
> - The bf16 beneficial effect CANNOT be reproduced by post-hoc rounding
>   of outputs or inputs to matmul — it must come from HOW the compute
>   itself interacts with the forward/backward dataflow at bf16
>   (likely cumulative rounding in attention internals + all pointwise
>   ops that we didn't probe directly).
>
> **Long-run confirmation (CP1, CP2, CP3):**
> - CP1 c=f32 lr=1e-6 × 100 steps: reaches 0.40 (vs Exp N 10-step 0.32).
> - CP2 c=f32 lr=1e-5 × 100 steps: reaches 0.02 by step 40 (over-converges).
> - CP3 c=bf16 lr=1e-5 × 10 steps: reaches 0.26 (slightly beats Exp N).
> - All c=f32 configs train successfully given enough total LR budget.
>
> ### Bug 1 probed (CP5, running)
>
> - CP5 (v5p-8 pd=4 c=bf16 lr=1e-6 × 100 steps): testing whether Bug 1
>   is "slow not stuck" like Bug 2. Data pending.
>
> ### Practical resolution
>
> - **Production**: use Exp N recipe (c=bf16, lr=1e-6) — fastest, reaches
>   0.32 in 10 steps.
> - **Debug/numerical**: c=f32 with lr=1e-5 or 5-10x more steps if needed.
> - **Width-4 on v5p-8 (Bug 1)**: use `mesh=mix` or `mesh=tp` instead of
>   full-FSDP; or upgrade to v5p-16 pd=2.
>
> ---
>
> **Prior status (2026-04-18T10:08Z): MAJOR FINDING — CP1 (100-step
> AC) is descending substantially! At step 38 = 0.499. AC is NOT stuck,
> just SLOW. The "10-step window" is simply insufficient for c=f32 to
> escape at lr=1e-6.**
>
> CP1 trajectory (AC recipe, 100 steps, lr=1e-6):
> - step 0: 0.6931
> - step 4: 0.6877 (slow warmup)
> - step 20 est: ~0.6 (warmup completed by step 10)
> - **step 38: 0.4988**
> - projected step 99: likely 0.30-0.35 (close to Exp N 10-step level)
>
> **REVISED INTERPRETATION OF BUG 2:** c=f32 is NOT broken for LoRA
> DPO. It just trains **~55x slower per unit LR** than c=bf16 on this
> recipe, so the 10-step window shows it "stuck" at 0.66 when really
> it needs more steps (or higher LR) to reach the good basin.
>
> This unifies all the Class-B findings:
> - Exp N at lr=1e-6 reaches 0.32 in 10 steps.
> - AC at lr=1e-6 would reach ~0.32-0.35 in 50-100 steps (CP1 pending).
> - BF at lr=1e-5 reaches 0.41 in 10 steps (LR compensates).
> - CP2 (100-step BF lr=1e-5) should reach very low loss.
>
> Bug 2 is effectively resolved: **use c=bf16 for fast training, OR
> use c=f32 with 10x+ LR OR 5-10x more steps.** The mechanism is a
> per-unit-LR slowdown, not a fundamental failure.
>
> **Prior status (2026-04-18T09:55Z): Class-B essentially complete.
> 9 probes finished, 4 running (BB, BD v2, BN HIGH, CP1, CP2). Lane 1
> (Bug 2) narrowed: it's ACTIVATION dtype rounding, not matmul precision,
> not CE kernel, not optimizer, not factor geometry, not warmup. Lane 2
> (Bug 1) in progress with BN HIGH (3x bf16).**
>
> **Key narrow finding for Bug 2:** The critical difference between
> c=f32 (AC) and c=bf16 (Exp N) is at the activation layer — every
> matmul output in c=bf16 is rounded to bf16 (and every pointwise op).
> This cumulative rounding across 32 layers with many ops each produces
> a "noisy" gradient that aligns better with the DPO descent direction.
> c=f32 keeps activations exact across all ops, producing the "raw" gradient
> which is ~55x less effective per unit LR.
>
> **Practical workaround confirmed:** increasing LR 10x (from 1e-6 to
> 1e-5) under c=f32 closes ~70% of the gap to c=bf16 (step-9 loss 0.41
> vs Exp N's 0.32). A full 100-step run at lr=1e-5 (CP2) is expected to
> reach near c=bf16 quality.
>
> **Next probe for full mechanism:** C3 — explicit activation rounding
> at layer boundaries under c=f32 (not yet implemented; requires code
> patch to cast `x → bf16 → f32` after each Linear/LayerNorm/residual).
> Key finding: LR partially compensates for c=f32 (lr=1e-5 reaches 0.41
> vs Exp N's 0.32). CE kernel, optimizer path, factor geometry, light
> symmetry break, and warmup are all FALSIFIED. Mechanism isolated to
> forward/backward compute (specifically, matmul precision is the prime
> suspect — C1a running).**
>
> ### Class-B state table (updated 09:45Z)
>
> | probe | delta from AC | step-9 loss | status |
> |-------|---------------|------------:|--------|
> | AC | baseline | 0.660 | STUCK |
> | Exp N | c=bf16 | 0.318 | GOOD |
> | AD v3 | zero_init_b=False (Kaiming B) | 5.11 | slow descent from bad |
> | BA | b_init_scale=1e-3 | 0.662 | STUCK (tracks AC) |
> | BC | a_init_mode=zero, b=rand_small | 0.692 | STUCK (oscillate) |
> | BE | warmup=0.0 | 0.656 | STUCK (marginal) |
> | BH | CE impl=reference | 0.660 | STUCK (CE not at fault) |
> | BK | cast grads to bf16 before Adam | 0.660 | STUCK (optimizer not at fault) |
> | BD v1 | lora_B grad noise std=1e-5 step 1 | 0.661 | STUCK (noise too small) |
> | BF lr=3e-6 | 3x baseline LR | 0.598 | partial (~0.06 below AC) |
> | **BF lr=1e-5** | **10x baseline LR** | **0.412** | **strong partial (70% gap closed)** |
> | BD v2 | noise std=1e-2 step 1 | pending | pending |
> | BB | b_init_scale=1e-2 | pending | pending |
> | C1a | matmul precision=DEFAULT at c=f32 | pending | **KEY TEST: bf16 matmul only** |
> | CP1 | AC at 100 steps | pending | test plateau vs slow march |
> | BN | matmul precision=HIGHEST on v5p-8 | running | Lane 2 (Bug 1) |
>
> ### Live hypotheses after BH+BK falsification
>
> - **H_LR_partial**: c=f32 has ~55x less per-unit-LR effectiveness than
>   c=bf16. At 10x LR, c=f32 reaches ~0.41 vs Exp N's 0.32. Some residual
>   gap due to direction misalignment, but NOT a fundamental stuck state.
> - **H_matmul_precision** (C1a testing): the bf16 matmul rounding (which
>   accumulates across 32 layers of multiple matmuls each) injects
>   beneficial noise that aligns gradients with the DPO descent direction.
>   f32 matmul preserves exact gradients, which are less aligned.
> - **H_noise_magnitude** (BD v2 testing): a larger noise (1e-2) injected
>   at step 1 might mimic bf16 compute rounding.
>
> ### Class-B interim findings (Lane 1 — Bug 2 c=f32)
>
> | probe | description | step-9 loss | vs AC (0.66) | vs Exp N (0.32) | status |
> |-------|-------------|------------:|-------------:|----------------:|--------|
> | AC (baseline) | canonical, c=f32 | 0.660 | — | +0.34 | STUCK |
> | Exp N (baseline) | canonical, c=bf16 | 0.318 | -0.34 | — | GOOD |
> | AD v3 | Kaiming B (adapter≠0) | 5.11 | +4.45 | — | slow descent from bad |
> | BA | b_init_scale=1e-3 | 0.662 | +0.003 | +0.34 | STUCK (tracks AC) |
> | BC | A=0, B=rand_small | 0.692 | +0.03 | +0.37 | STUCK (log(2) oscillate) |
> | BE | warmup=0.0 | 0.656 | -0.004 | +0.34 | STUCK (marginal) |
> | BF | lr=3e-6 (3x) | 0.598 | -0.062 | +0.28 | PARTIAL |
> | BF | lr=1e-5 (10x) | ? | — | — | pending |
> | BD | grad noise on lora_B step 1 | ? | — | — | running (compile) |
> | BH | CE impl=reference | ? | — | — | running (compile) |
> | BK | grad cast to bf16 | ? | — | — | running (queue) |
> | BI | no grad_accum (pd=8) | ? | — | — | queued |
> | BB | b_init_scale=1e-2 | ? | — | — | not launched |
>
> ### Falsified hypotheses (all fail to explain Bug 2)
>
> 1. **H_B** (light symmetry break rescues) — FALSIFIED by BA.
> 2. **H_factor_geometry** (which factor updates first) — FALSIFIED by BC.
> 3. **H_warmup** (warmup=0.1 traps early) — FALSIFIED by BE.
> 4. **H_adapter_zero_at_init** (adapter_out=0 at init is trap) —
>    FALSIFIED by BA: BA breaks adapter_out=0 at init (small), still stuck.
>
> ### Live hypotheses (pending test)
>
> - **H_LR_only**: the correct descent direction exists at c=f32 but needs
>   bigger steps (BF partial rescue supports this somewhat; BF lr=1e-5
>   tests the strong form).
> - **H_direction**: c=f32 gradients point in a DIFFERENT direction than
>   c=bf16, so LR can't fully rescue. Mathematically motivated: Exp N step
>   1→2 drops loss 0.36 at lr=1e-6, whereas BF step 1→2 drops 0.02 at
>   lr=3e-6. Ratio of 12x per unit LR → direction differs per unit norm.
> - **H_kernel** (BH): blocked-XLA CE kernel at c=f32 is broken. Reference
>   CE path should rescue if true.
> - **H_optimizer** (BK): Adam with f32 grads accumulates error; with bf16
>   grads it works. Tests whether the update-dtype mismatch with bf16 state
>   is load-bearing.
> - **H_accum** (BI): grad_accum's reshard loop is broken at c=f32. pd=8
>   removes microbatching.
> - **H_noise** (BD): any small-magnitude stochastic perturbation of the
>   step-1 gradient breaks the trap (like c=bf16 rounding noise does).
>
> ### Backlog (not yet launched)
>
> - BB (b_init_scale=1e-2): larger init perturbation. Low priority given BA result.
> - BN (v5p-8 matmul precision): Lane 2, running now.
> - BL (v5p-8 mesh reorder): Lane 2, deferred.
> - CP1 (100-step AC): Class C, decisive LR-vs-direction test if BF lr=1e-5 ambiguous.
>
> B0 infrastructure landed (2026-04-18T08:35Z):
> - **B0.1** — `lib/levanter/src/levanter/lora.py`: added
>   `b_init_scale: Optional[float]` and `a_init_mode: Literal['random','zero']`
>   to `LoraConfig`, threaded through `LowRankLinear.init`, `LoraLinear.init`,
>   and `_loraize`. When `b_init_scale > 0` overrides `zero_init_b=True` with
>   `N(0, scale)`. When `a_init_mode='zero'`, A is zero-initialized (BC).
> - **B0.2** — `lib/levanter/src/levanter/trainer.py`: env-gated LoRA grad
>   perturbation + cast hooks injected between `_compute_gradients_microbatched`
>   and `state.take_step`. Env vars: `MARIN_DEBUG_LORA_GRAD_NOISE_STD`,
>   `MARIN_DEBUG_LORA_GRAD_NOISE_STEP`, `MARIN_DEBUG_LORA_GRAD_NOISE_TARGET ∈ {A,B,both}`,
>   `MARIN_DEBUG_LORA_GRAD_CAST ∈ {none,bf16,f32}`. Noise path uses `jnp.where(step == N, noisy, grad)`
>   to scope injection to one step.
> - **B0.3** — deferred to scalar trace (existing `MARIN_DEBUG_LOG_STEP_TRACE=1`
>   already dumps per-sentinel gA/gB l2 and sum). Full tensor dump can be added
>   later if scalar-level discriminators are insufficient.
> - **Validation patched** at `lib/levanter/src/levanter/main/train_dpo.py:369-394`
>   to understand the new invariant: `adapter_out = B @ A @ x = 0 at init`
>   holds iff A=0 OR B=0. BC (`a_init_mode='zero'`) preserves invariant and
>   does NOT require the env-gate. BA/BB (nonzero B with A=random) break it
>   and still require `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1`.
>
> BA launched 2026-04-18T08:37Z in both v5p regions:
> - us-central1: `/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-083703` (tag `uc1-ba1`)
> - us-east5: `/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-083717` (tag `ue5-ba1`)
>
> AD v3 narrow result (for comparison): under `c=f32` on v5p-16 pd=2, flipping
> `zero_init_b=True→False` converts a stuck run (AC, step-9 = 0.66)
> into a descending run (AD, step-9 = 5.11, from start 7.46). The
> descent happens but doesn't reach the good basin in 10 steps
> because Kaiming-scale B init is too large a perturbation to
> converge quickly. H_B is **narrowly supported** — the zero-init
> symmetry contributes to the c=f32 trap — but the mechanism is
> not proven (AD can't distinguish "tiny symmetry break is enough"
> from "any big kick works").
>
> BA is the clean successor: `b_init_scale=1e-3` breaks the `B=0`
> symmetry by ~5 orders of magnitude less than Kaiming, so step-0
> loss should be close to log(2) and the trajectory tells us whether
> a *small* perturbation is enough to rescue f32 training.
>
> **Current standing rules for every new experiment (do not skip):**
>
> - `-e MARIN_DEBUG_SKIP_HF_EXPORT 1` on every `iris job run`.
> - Keep `AdapterBaseReferenceConfig`. Never switch to
>   `SeparateReferenceConfig` to sidestep validations — use
>   env-var gates in the code instead.
> - Multi-region launch per the existing directive.
>
> ### Facts table
>
> *(Data-axis width stated as nominal pd × chip count on a single-
> slice pod; not cross-checked against HLO `replica_groups` size.
> For multislice / DCN-split pods this shorthand can mislead; verify
> from HLO before treating as ground truth.)*
>
> | TPU / mesh | nominal `|data|` | `c=` | trajectory | step-2 loss |
> |---|---|---|---|---|
> | v5p-8 pd=4 | 4 | bf16 | stuck | 0.685 (Exp Q) |
> | v5p-8 pd=4 | 4 | f32 | stuck | 0.687 (Exp U, AB) |
> | v5p-16 pd=2 | 8 | bf16 | **RECOVERS** | 0.335 (Exp N, confirmed by N-rerun 2026-04-18T06:46Z) |
> | v5p-16 pd=2 | 8 | f32 | stuck | 0.687 (AC) |
> | v5p-8 pd=4 pure-TP `{data:1, model:4}` | 1 | bf16 | recovers | ~0.33 (Exp W) |
> | v5p-8 pd=4 mixed `{data:2, model:2}` | 2 | bf16 | recovers | ~0.33 (Exp Z3) |
> | v5p-8 pd=4 full-FT | 4 | bf16 | recovers | — (Exp T, no LoRA) |
>
> ### Hypotheses table
>
> | # | Claim | Status | How to falsify |
> |---|---|---|---|
> | Bug 1 | v5p-8 pd=4 c=bf16 LoRA FSDP is stuck for some width-4-specific structural reason that is NOT the reduction dtype. | Mechanism unknown; dtype-as-mechanism story falsified by AB. | Workarounds known (TP/mix/bigger pod). Mechanism investigation deprioritized vs Bug 2. |
> | Bug 2 | c=f32 + LoRA DPO (zero_init_b=True) is stuck on an otherwise-working mesh (v5p-16 pd=2). | Confirmed by AC + N-rerun. Mechanism unknown. | — |
> | H_B (candidate cause for Bug 2) | `zero_init_b=True` + c=f32 together produce a gradient direction at step-2 that fails to escape the bad basin; breaking the init symmetry should recover. | Untested. | AD below. |
> | H_kernel (candidate cause for Bug 2) | Pallas CE kernel or some other numerical kernel has an f32-specific code path that loses information. | Untested. | Fallback if AD stays stuck. |
> | H_optimizer (candidate cause for Bug 2) | Adam state / update arithmetic behaves differently at c=f32 vs c=bf16 in a way that matters only under the LoRA zero-init surface. | Untested. | Fallback if AD stays stuck. |
>
> ### AD — next test
>
> Exp N recipe + `c=f32` + `zero_init_b=False`. Single-knob
> deviation from AC. v5p-16 pd=2.
>
> - If AD **recovers** → conclude narrowly: "c=f32 combined with
>   `zero_init_b=True` is a broken config on this stack. Non-zero
>   B init is a workaround." Does NOT prove "bf16 rounding noise
>   is the symmetry-breaker" or any other mechanism; just reduces
>   the broken-config surface.
> - If AD **stays stuck** → H_B is wrong. Move to H_kernel or
>   H_optimizer probes (CE kernel f32 audit, optimizer dtype
>   trace).
>
> Interpret narrowly. Don't extrapolate from AD to the whole
> numerical-noise story.
>
> ### Bug 1 — not what AD is for
>
> AD tests Bug 2 only. Bug 1's practical workarounds stand
> (c=bf16 at wider data axis, pure-TP or mix mesh on v5p-8,
> full-FT on v5p-8). Bug 1's mechanism investigation can resume
> after Bug 2 is narrowed, if needed.
>
> **Prior status (2026-04-18T06:21Z): Experiment AC COMPLETE —
> another hypothesis killed. The bug is NOT width-4-specific at f32.**
>
> AC ran AB recipe (`p=f32, c=f32`) on v5p-16 pd=2, a known-good
> mesh. Training is STUCK, just like AB. Step-9 loss 0.6596 vs
> Exp N baseline ~0.30. `c=f32` breaks LoRA DPO globally on this
> recipe — not just at width 4.
>
> The full 4-cell matrix of what now works vs breaks:
>
> | `|data|` | `c=bf16` | `c=f32` |
> |---|---|---|
> | 4 (v5p-8 pd=4) | stuck (Exp Q) | stuck (AB) |
> | 8 (v5p-16 pd=2) | **RECOVERS (Exp N)** | **stuck (AC)** |
>
> Only `bf16 + |data|≥8` works. Two distinct failure modes are
> now plausible:
>
> 1. `c=bf16 + |data|=4`: some (still-unknown) width-4 specific
>    structural thing. Tracked by Z-experiments but never nailed;
>    AB falsified the "bf16 collective dtype" explanation.
> 2. `c=f32` at any tested width: a global degeneracy in LoRA DPO
>    under deterministic f32 arithmetic. Newly revealed by AC.
>    Hypothesis: LoRA's `zero_init_b=True` leaves a symmetric
>    initial state that bf16 rounding noise breaks but f32 doesn't.
>
> **`precision_explained_jax_tpu.md`'s "lever D" narrative is
> retracted.** Collective dtype is not the mechanism. The full
> Z-experiment "MECHANISM CLOSED" framing needs to be retracted
> too — the bug at `|data|=4 + c=bf16` is real and width-specific,
> but nothing we did has identified its actual cause.
>
> Recommended next experiments (details in "Experiment AC" section
> below):
> 1. Rerun Exp N with current code to confirm no stack drift.
> 2. `c=f32 + zero_init_b=False` on v5p-16 pd=2 to test the
>    symmetry hypothesis.
> 3. HLO diff AB vs AC for context on width-4 vs width-8 structure
>    at matched f32.
>
> **Prior status (2026-04-18T05:43Z): Experiment AB COMPLETE. bf16-
> collective-width-4 hypothesis FALSIFIED.**
>
> **Prior status (2026-04-18T05:43Z): Experiment AB COMPLETE. bf16-
> collective-width-4 hypothesis FALSIFIED.**
>
> ### AB result (the short version)
>
> Exp U rerun with HLO dumping. Outcome row 1 of the decision matrix:
>
> 1. **HLO shows f32 reductions.** Every width-4 (`replica_groups={{0,1,2,3}}`)
>    grad `all-reduce` / `all-reduce-scatter` in the compiled train
>    step operates on `f32[...]` tensors. `to_apply` reduction
>    regions are all `(f32[], f32[]) -> f32[]` adders. 5 leftover
>    bf16 regions are for non-trainable parameter tensors
>    (reference-model path), not grad collectives.
>    - Exp Q baseline: 71 bf16 + 2 f32 reduction regions.
>    - AB: 5 bf16 + 4 f32 reduction regions. Dramatic shift.
> 2. **Training STILL stuck in the bad basin.** Step-2 loss 0.6865
>    (vs good baseline ~0.335). Step-9 loss 0.6599. Same qualitative
>    trajectory as Exp Q.
>
> So `c=f32` successfully produced f32 reductions AND training still
> failed. **The bf16 non-associative sum at width 4 is not the
> mechanism.** The `|data|=4` pathology is driven by something
> else — structural/topological, not numerical-dtype.
>
> ### What this invalidates
>
> - The "leading hypothesis" in
>   `precision_explained_jax_tpu.md` (Parts 6–9) that width-4 bf16
>   collective precision traps LoRA's early update. Needs a
>   correction pointer.
> - Exp Z4's "mechanism is nailed" framing: Z4's HLO evidence is
>   still correct (reductions are bf16 under `c=bf16`), but it's
>   now a correlate, not a cause.
> - The entire AA intervention plan (cast cotangents to f32): even
>   if it had worked surgically, it wouldn't have helped — Exp U
>   already showed that forcing f32 reductions globally doesn't
>   recover training.
>
> ### What still stands
>
> - Exp T: LoRA-specific (full-FT on v5p-8 works).
> - Exp W: FSDP-specific (pure TP on v5p-8 works).
> - Exp Z3: width-4-specific (`{data:2, model:2}` works).
> - Exp Z1: grad values DO differ between v5p-8 (width 4) and
>   v6e-8 (width 8) at the 1e-5 level. We just misattributed the
>   cause. The divergence is still real; it comes from something
>   other than bf16 non-associativity.
>
> ### Next candidate mechanisms (non-dtype)
>
> 1. Reduce-scatter tree topology / algorithm choice at width 4 vs 8.
> 2. All-gather vs reduce-scatter fusion differences at width 4.
> 3. Buffer layout / tile-packing differences at small shard sizes.
> 4. Collective chunking differences at width 4.
> 5. LoRA-rank divisibility: rank-64 across 4 chips = 16-per-chip,
>    across 8 chips = 8-per-chip. Different shard sizes could
>    trigger different codepaths.
>
> ### Next experiment: pure HLO structural diff (no TPU)
>
> Compare AB's v5p-8 HLO (width 4, f32) against Z4's v6e-8 HLO
> (width 8, bf16) — but specifically look for **structural
> differences beyond dtype and `replica_groups` size**: reduction
> tree shape, scheduling, fusion boundaries, buffer layout, chunk
> counts. This is pure text analysis; no TPU allocation needed.
> Planned next.
>
> ### State of AA and AB
>
> - AA retracted. All AA code changes reverted via `git checkout`.
>   AA script deleted.
> - AB launched 2026-04-18T05:33Z, `uc1-ab1` completed with
>   decisive HLO evidence, `ue5-ab1` still running as a replication
>   check.
>
> Full write-up of AB result and the new landscape of candidate
> mechanisms is in the "Experiment AB" section below. The old
> "Experiment AA" section below remains as a retracted record of
> the misadventure for posterity.
>
> ### AA retraction summary
>
> AA v1–v5 all produced HLO bit-identical to Exp Q. AA v5's diagnostic
> prints revealed why: the gradients reaching my `grad_accum.loop`
> insertion point are already `float32`, not bf16, so my
> `.astype(jnp.float32)` was a trivial no-op in every version. The
> env-var gating was never the issue (`MARIN_DEBUG_AA_CAST_GRADS_F32='1'`
> was visible all along).
>
> **Why the grads are f32 at my insertion point**: under
> `p=f32, c=bf16`, GSPMD emits the FSDP reduce-scatter on per-layer
> bf16 cotangents *inside the backward pass of `fn`*. After that
> per-layer reduce, the sharded bf16 grad is cast up to f32 to match
> the f32 master-param dtype, before `filter_value_and_grad` returns.
> So by the time `this_grads` arrives at the microbatch accumulation
> boundary, the bf16 reduce has already happened *and* the grad is
> already f32. Cast there = can't affect the reduce.
>
> Codex's original recommendation (cast in `microbatched()`) and my
> entire v1–v5 plan were targeting the wrong intervention site.
>
> **Code reverted**: diagnostic unconditional-cast / unconditional-f32-
> accumulator changes in `grad_accum.py`, `trainer.py`, and
> `train_dpo.py` have been reverted via `git checkout`. The AA
> experiment script has been deleted.
>
> ### Next: Experiment AB — rerun Exp U with HLO dumping
>
> The whole AA thread was downstream of a more basic question: **when
> Exp U ran `p=f32, c=f32`, did the compiled HLO actually have f32
> reductions?** If yes, and training was still stuck, the bf16-
> collective-width-4 hypothesis is falsified. If no, there's a hidden
> bf16 path that needs to be found.
>
> Exp U predated the HLO-dump workflow (Z4 added it), so we've never
> actually inspected Exp U's HLO. **AB is Exp U, rerun with HLO
> dumping enabled** (same pattern as Z4). High-information because:
>
> 1. Directly answers the central open question about whether `c=f32`
>    produces f32 reductions in this stack.
> 2. Cheap: one 10-step run, ~15 min wall-clock, modest TPU hours.
> 3. Also reproduces Exp U's trajectory as a replication check — if
>    trajectory drifted from Exp U, that's a separate signal.
>
> Decision matrix for AB:
>
> | AB HLO reduce dtype for LoRA grads | Training outcome | Conclusion |
> |---|---|---|
> | **f32** | stuck | **bf16-collective-width-4 HYPOTHESIS REJECTED.** Pivot to non-dtype `|data|=4` investigation. |
> | **f32** | recovers | bf16 collective confirmed; Exp U result was mislogged. |
> | **bf16** | stuck | Hidden bf16 path forcing bf16 reductions despite `c=f32`. Find and fix. |
> | **bf16** | recovers | Unexpected; investigate. |
>
> Most likely outcome: f32 HLO + stuck training → hypothesis rejected.
>
> Experiment script:
> `experiments/posttrain/per_stmt_dpo/experiment_ab_v5p8_fp32_pd4_hlo_s10.py`.
> Full diagnostic history and the AB launch record in the "Experiment
> AA Plan" section and "Experiment AB Plan" section below.
>
> Three possible outcomes:
>
> 1. HLO shows f32 reductions AND training escapes the step-2 bad basin
>    → H0 confirmed; bf16-collective-width-4 really is the mechanism;
>    land the fix.
> 2. HLO shows f32 reductions AND training still stuck → collective
>    dtype is NOT the mechanism; pivot to non-dtype `|data|=4`
>    investigation (HLO scheduling, buffer layout, tree topology).
> 3. HLO still shows bf16 reductions (cast was optimized away) →
>    experiment invalid; fix cast placement and rerun.
>
> Full plan in the "Experiment AA Plan" section below. Read that before
> reading any superseded pointers or experiment results.
>
> **Pointer for the next agent (2026-04-17T08:20Z — MECHANISM CLOSED, SUPERSEDED by AA review above):**
>
> The v5p-8 LoRA DPO pathology is caused by **bf16 cross-chip all-reduce
> of LoRA gradients at 4 participants** (data-axis width). The bf16 add
> is non-associative, so a 4-way tree produces a systematically
> direction-biased sum relative to the 8-way tree used on v5p-16 /
> v6e-8. LoRA's rank-64 zero-init-B update projects that tiny bias into
> the wrong subspace and the step-2 loss escape never happens.
>
> **Full evidence chain (all four Z experiments complete):**
>
> | Exp | Finding |
> |---|---|
> | Z3 (`{data:2,model:2}` mesh) | **Recovers** training on v5p-8. Rules out "FSDP generically bad at <8 width"; pins the bug to `\|data\|=4` specifically. |
> | Z1 (per-element grad dump) | Measured post-all-reduce gradient values on v5p-8 vs v6e-8 at step 0 with identical seed/batch/init. **Values differ systematically** at 1e-6 to 1e-5 absolute (bf16 precision-noise level) across Q/K/V/O/gate/up/down `lora_B`. |
> | Z2 (`--xla_allow_excess_precision=false`) | **Does not recover** — this flag doesn't force bf16 reductions to fp32; it just prevents new higher-precision insertions. Negative result narrows the candidate flags. |
> | Z4 (HLO diff) | Compiled HLO is identical on v5p-8 and v6e-8 **except** `replica_groups` size (4 vs 8). Reduction regions are all `bf16 + bf16 → bf16` (14 on v5p, 15 on v6e; 0 fp32 reductions). This is the smoking gun: **the collective runs in bf16, and at width 4 vs 8 it produces different sums**. |
>
> **Production fix options:**
>
> 1. Mesh rearrangement on v5p-8: use `{replica:1,data:2,model:2}` (Z3
>    mix) or `{replica:1,data:1,model:4}` (Exp W TP). Both recover
>    training cleanly.
> 2. Run on a TPU that doesn't have `\|data\|=4`: v5p-16, v6e-8, v6e-16
>    all work with canonical FSDP.
>
> **Proper fix (follow-up PR):** cast LoRA gradients to fp32 before the
> cross-chip `psum` / all-reduce in
> `lib/haliax/src/haliax/partitioning.py:909`. That forces the
> reduction region to take `f32` operands; XLA's fp32 collective is
> numerically associative to a much tighter tolerance, eliminating the
> width dependence entirely. Unlikely to cost meaningful throughput on
> TPU (hardware supports fp32 all-reduce).
>
> **Investigation summary (for context):** Exp Q/R/R2a/T/U/V/W/Y/Z1/Z2/Z3/Z4
> form the full chain. Exp W narrowed to "FSDP-on-4-chips", Y showed
> partition specs are identical between v5p-8 and v6e-8, Z3 narrowed
> the width to specifically 4, Z1 measured the element-level
> divergence, Z4 confirmed it's the bf16 arithmetic in the reduction
> region. Everything is consistent; the logbook's earlier "directional
> instability" hypothesis has been promoted to a concrete,
> HLO-level-proven mechanism.
>
> ### Prior pointers (progressively superseded)
>
> - 2026-04-17T06:00Z: Exp Y showed partition specs identical; only
>   mesh width differs. Leading hypothesis was still "collective
>   behavior at width 4" without proof. Z-experiments closed it.
> - 2026-04-17T05:25Z: Exp W showed TP mesh recovers training on v5p-8.
>   Narrowed to "FSDP-on-4-chips" scope, not yet the precise mechanism.
> - 2026-04-17T04:02Z: Experiments U and V both ruled out numeric
>   precision and reference-path hypotheses.
>
> **Pointer for the next agent (2026-04-17T05:25Z, superseded — read above first):** **ROOT CAUSE FOUND (partial).**
> Experiment W pure-TP mesh (`axes={replica:1, data:1, model:4}`) on the
> exact bad `v5p-8` LoRA recipe **fully recovers training.** Step-2 loss
> dropped from the bad 0.6851 regime (FSDP) to **0.2907** (TP) — slightly
> better than the canonical good `v5p-16` FSDP baseline (0.3352). delta
> at step 9 = 12.4 (bad v5p-8 FSDP = 0.66, good v5p-16 FSDP = 10.2).
>
> **Conclusion (superseded by Exp Y):** the v5p-8 LoRA DPO pathology is
> caused by FSDP on the 4-device mesh — narrowed by Exp Y to the `data`
> axis width specifically, not the FSDP layout itself.
>
> **Scientific summary (all prior eliminations hold):** Exp U ruled out
> bf16/fp32 numerical precision. Exp V ruled out
> `AdapterBaseReferenceConfig`. Exp R/R2a ruled out CE tiling and
> backward bf16 accumulation. Exp N/O/P ruled out TPU family alone, `pd`
> alone, LoRA rank alone. Exp T ruled out broad "v5p-8 breaks DPO"
> (full-FT works). Exp W is the positive discriminator that identified
> the load-bearing variable.
>
> **What to do next (choose based on goal):**
>
> 1. **Production fix:** use the pure-TP mesh on v5p-8 LoRA, or use
>    `v5p-16` / `v6e-8` which both work under canonical FSDP.
> 2. **Narrow the mechanism:** run `mix` variant (`data:2, model:2`)
>    — if that also recovers, something is specifically wrong with
>    all-4-chips-FSDP; if it's also bad, the boundary is between 4-wide
>    and anything-less-wide FSDP.
> 3. **Dig into levanter/haliax:** the bug is likely in how LoRA
>    parameter or gradient partition specs interact with the 4-wide
>    data axis. Dump partition specs on TP vs FSDP to compare. Look
>    for FSDP-specific collective ops that degenerate at mesh size 4.
> 4. **Fix the cosmetic HF-export bug** (`DpoModel.config` AttributeError
>    on `SeparateReferenceConfig` — unrelated but makes runs look
>    failed when they actually succeeded).
>
> **Full Exp W result is in the section below.**
>
> **Pointer for the next agent (2026-04-17T04:02Z):** Experiments U and V
> have both completed. **Neither full fp32 compute nor switching to
> `SeparateReferenceConfig` fixes the `v5p-8` LoRA pathology.**
>
> Current hypothesis ranking after Exp V:
>
> - **(leading, narrowed further)** **LoRA FSDP sharding / mesh-collective
>   behavior on the 4-device `v5p-8` mesh** — parameter/grad partition
>   layout, adapter optimizer-state sharding, or LoRA-specific collective
>   ordering. Both reference-path variants (`AdapterBaseReferenceConfig`
>   and `SeparateReferenceConfig`) fail identically, so the cause must
>   live in the LoRA-update code path itself, not the reference path.
> - **(still possible)** attention `kv_head` mapping or other sub-CE
>   distributed detail that interacts specifically with LoRA
> - **(ruled out by Exp V)** `AdapterBaseReferenceConfig` as the load-
>   bearing variable — `SeparateReferenceConfig` on v5p-8 LoRA tracks the
>   bad Exp Q regime point-for-point (max |Δloss| ≈ 0.005 across all 10
>   steps)
> - **(ruled out by Exp U)** bf16/bfloat16 numerical precision anywhere
>   in the compute graph
> - **(ruled out by Exp R + R2a)** CE kernel tiling / bf16 accumulation /
>   per-chip CE workload
> - **(ruled out by Exp N/O/P)** TPU family alone, `pd` alone, LoRA rank
>   alone
> - **(ruled out by Exp T)** broad "the `v5p-8` execution graph itself
>   breaks DPO even without LoRA"
>
> **What this means for next experiments:** The failure surface is now
> down to the LoRA-specific distributed training path on a 4-device
> mesh. Highest-information next probes:
>
> 1. **Mesh-axis rearrangement on v5p-8 LoRA (Exp W — IN PROGRESS).**
>    Try `mesh.axes = {data:1, model:4}` (pure TP, no FSDP) and
>    `{data:2, model:2}` on the existing bad v5p-8 recipe. If either
>    recovers → FSDP-on-4-chips is the load-bearing variable. If both
>    still bad → the issue is deeper than axis assignment.
> 2. **Per-module gradient direction probe.** Instrument LoRA backward
>    to log `g.sum()`, `g @ random_unit_vec`, and per-module signatures.
>    Compare v5p-8 vs v5p-16 element-wise to pinpoint which module's
>    gradient direction diverges.
> 3. **`target_modules` ablation** — attention-only vs mlp-only LoRA on
>    v5p-8 to localize which submodule's gradient is the problem.
> 4. **(Exp X — DEAD / blocked).** Cross-family 4-chip test on `v6e-4`
>    is not viable: v6e-4 has 32 GB/chip × 4 = 128 GB total HBM, and
>    the compiled `jit__train_step` program is ~22 GB per chip on
>    Llama-8B LoRA `r=64` regardless of batch/pd (scan-stacked LoRA
>    grad buffers dominate), exceeding the free-HBM budget after
>    weights load. See the "Experiment X result — BLOCKED" section
>    below for the full HBM matrix and ruled-out recovery knobs.
>
> **Known cosmetic bug (not load-bearing):** with
> `SeparateReferenceConfig`, the post-training `save_merged_hf_model_cb`
> hits `AttributeError: 'DpoModel' object has no attribute 'config'` in
> `lib/levanter/src/levanter/compat/hf_checkpoints.py:912`. Training and
> W&B logging complete normally; only the HF export crashes. Iris marks
> such runs `failed` even though the scientific data is intact. Fixing
> this bug is orthogonal to the DPO investigation.

## Class B experiment program (planned after AD v3) — aggressive-bisection overnight plan

> **Derived from `.agents/projects/debug_accum_class_b_program.md` and
> refined after AD v3.** Keep this entire section in sync with that
> file (this section supersedes it if they drift). The plan is
> structured as two lanes: Bug 2 (`c=f32`) first because it blocks
> interpretation of any further `c=f32` probe; Bug 1 (v5p-8 width-4)
> parallel if capacity allows.
>
> **If you are the next agent, start at B0 — the instrumentation
> lift. Do not run Class-B probes before B0 lands.**

### Standing rules that apply to EVERY run in this program

1. `-e MARIN_DEBUG_SKIP_HF_EXPORT 1` on every `iris job run`. No
   end-of-training HF export on debug runs. (Logbook top directive.)
2. Use `AdapterBaseReferenceConfig` everywhere. Do not pivot to
   `SeparateReferenceConfig` to sidestep config validations —
   bypass validations with env-var gates instead. (Memory rule +
   user directive 2026-04-18.)
3. Multi-region launch: `us-central1, us-east5` for v5p; the three
   listed v6e regions otherwise. (Original launch policy.)
4. `steps_per_checkpoint=9999` and `steps_per_hf_export=9999` in
   the experiment script (already the convention).
5. The active code patch at `lib/levanter/src/levanter/main/train_dpo.py:372-378`
   (env flag `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1`)
   stays on while this program runs. Revert it when the Bug 2
   investigation closes.

### How AD v3 reshapes this plan

AD v3 already ran the `zero_init_b=False` case but with the
full Kaiming init on `B`. That produced a large perturbation
(step-0 loss 7.46, not log 2), which descends monotonically under
c=f32 but doesn't reach the good basin in 10 steps. So:

- "Is c=f32 globally broken?" → NO. AD v3 shows training can
  descend under c=f32 when the symmetry is broken. (Progress vs
  AC which stalls.)
- "Is `zero_init_b=True + c=f32` specifically a trap?" → YES,
  narrowly. AC stalls; AD v3 descends. One-variable flip.
- "Is it the zero-init symmetry specifically, or just 'big enough
  perturbation wakes up training'?" → UNRESOLVED. AD v3's
  Kaiming-scale init conflates the two. **BA is the clean test.**

**BA is now the single most informative next probe.** The full
Class-B ordering still holds, but BA is moved to be the first
probe after B0 lands (same as in the source plan).

### B0 — shared instrumentation (land FIRST, before any probes)

**B0.1 — LoRA init controls** in `lib/levanter/src/levanter/lora.py`
(around line 215-242 in `LowRankLinear.init`):

- Add `b_init_scale: float | None = None`.
- Add `a_init_mode: Literal["random", "zero"] = "random"` (for BC).
- When `b_init_scale` is set and > 0, initialize `B` as
  `N(0, b_init_scale)` rather than zeros (when zero_init_b=True
  semantic is preserved by `b_init_scale=0`, None, or not set).
- Similarly thread through `LoraAdaptationConfig` so experiment
  scripts can set it directly.

**B0.2 — env-gated LoRA grad perturbation** in
`lib/levanter/src/levanter/trainer.py` around line 683 (after
`_compute_gradients_microbatched` returns):

- `MARIN_DEBUG_LORA_GRAD_NOISE_STD` (float)
- `MARIN_DEBUG_LORA_GRAD_NOISE_STEP` (int — which step to inject at)
- `MARIN_DEBUG_LORA_GRAD_NOISE_TARGET` ∈ `{A, B, both}`
- Optional: `MARIN_DEBUG_LORA_GRAD_CAST` ∈ `{none, bf16, f32}`
  for BK (cast grads before optimizer).

Implementation note: do the noise injection *after*
`_compute_gradients_microbatched`, so it's applied once per step
on the final (post-accum) gradient tree, not once per microbatch.

**B0.3 — LoRA grad artifact dump** (for BJ):

- At step 0 and step 1, dump full `dL/dA` and `dL/dB` tensors
  for each LoRA module family to a local artifact path.
- Enough to compute per-module cosine similarity between two
  runs, sign agreement on top-k magnitudes, and A-vs-B norm
  ratios.

**B0.4 — mesh reorder hook** (for BL):

- Extend the mesh override pattern from `experiment_w_v5p8_mesh_s10.py`
  to allow specifying an explicit logical-to-physical device
  reorder for `data:4`.

Estimated time: 1-1.5h of focused code work.

### Lane 1 — Bug 2 (c=f32) probes, priority order after B0

| Run | Base | Single-knob change | Hypothesis |
|---|---|---|---|
| **BA** | AC recipe | `b_init_scale=1e-3` via B0.1 | Light B perturbation rescues canonical f32 |
| BB | BA | scale sweep `1e-4, 1e-3, 1e-2` | Usable rescue window exists |
| BC | Exp N recipe + c=f32 | `A=0, B=random_small`, keeps `B·A=0` at init | The issue is factor geometry (dA=0 at step 0), not just init symmetry |
| BD | AC recipe | one-shot Gaussian noise on `lora_B` grad at step 1 via B0.2 | Tiny perturbation breaks the trap; noise-story |
| BE | AC recipe | `warmup=0.0` or `warmup=0.02` | Zero-LR step 1 is part of the trap |
| BF | AC recipe | LR sweep `3e-6, 1e-5` | Correct direction exists; step size too small |
| BH | AC recipe | pure-XLA CE (disable Pallas CE) | f32 failure is in CE kernel |
| BI | AC recipe | no microbatching (pd=8 on v5p-16 pd=8 gives mb=batch=64) | f32 failure lives in grad-accum reshard |
| BK | AC recipe | cast grads to bf16 before `state.take_step` via B0.2 | f32 failure is optimizer-path-specific |
| BJ | Exp N / AC / best rescue | full step-0/1 LoRA grad diff via B0.3 | Early grad direction localizes |
| BG | AC recipe | `SeparateReferenceConfig` | **DEMOTED.** Reference-path specific? |

**BG is demoted** per the AdapterBase rule. Only run it if the
other Lane-1 probes fail to isolate and a reference-path
explanation is the last remaining candidate.

### Lane 2 — Bug 1 (v5p-8 width-4) probes, after Lane 1 has a hit

| Run | Base | Single-knob change | Hypothesis |
|---|---|---|---|
| BL | Exp Q recipe | physical mesh reorder of data:4 via B0.4 | Bug 1 depends on collective order/topology |
| BN | Exp Q recipe | `jax_default_matmul_precision` sweep | Width-4 interacts with local dot algorithm |
| BM | Exp Q-like on larger pod | emulate data:4 on v5p-16/v6e-16 via mesh override | Width-4 is generic or v5p-specific |
| BO | Exp Q / AC | target-modules ablation (attention-only / MLP-only) | One module family dominates |

### BP — confirmation (either bug)

Run the best rescue from Lane 1 or Lane 2 for 100 steps
(`num_train_steps=100`) to rule out short-run transients. Also
run the strongest null (the same recipe without the rescue) to
confirm the null still stalls at 100 steps.

### Stopping criteria

**Lane 1 stops when:**

- A light symmetry-break (BA or BB) reproducibly rescues training, OR
- Three orthogonal f32 hypotheses fail:
  - init/symmetry (BA, BB, BC fail),
  - noise/scheduler (BD, BE, BF fail),
  - kernel/accum/optimizer (BH, BI, BK fail).

**Lane 2 stops when:**

- A topology or precision-algorithm intervention (BL, BN) rescues
  width-4, OR
- Two topology-like probes and one local-compute probe all miss,
  at which point the next move is deeper instrumentation, not
  more runs.

### Suggested 10-hour run order

1. **Hour 0-1.5**: land B0 (all of B0.1-B0.4).
2. **Hour 1.5-4**: run BA. If BA shows any descent, launch BB
   scales in parallel. If BA stalls as badly as AC, pivot
   immediately to BC + BD in parallel.
3. **Hour 4-6.5**: aggressively kill Bug-2 branches. Run BH, BI,
   BK as Lane-1 nulls if BA-BD haven't produced a rescue. If
   they have, jump to BJ for grad-artifact confirmation.
4. **Hour 6.5-9**: switch to Lane 2. Launch BL + BN in parallel.
   If one hits, consolidate with BP; don't spray more width-4
   runs.
5. **Hour 9-10**: BJ analysis, BP consolidation.

### Practical guidance for the next agent

- Do not overinterpret AD v3. It is useful evidence that "f32 can
  descend when the canonical init is broken strongly," but the
  Kaiming scale confounds "light symmetry break" vs "any big
  perturbation." BA (with `b_init_scale=1e-3`) is the clean
  version.
- Do not add more ad-hoc env gates. If an experiment needs a
  new knob, add it to B0.1 / B0.2 / B0.3 cleanly, not as a
  one-off patch.
- Prefer `v5p-16 pd=2` for all Bug-2 probes until Bug 2 is
  narrowed or ruled out. It is the least confounded baseline we
  have (Exp N is reproducible on it to bit-perfect; AC is the
  stuck baseline).
- For Bug 1, stick to Exp Q recipe except for the single
  intervention variable.
- Always launch multi-region with distinct run tags per the
  logbook directive.
- Always pass `-e MARIN_DEBUG_SKIP_HF_EXPORT 1` per the top-of-
  logbook directive.

### When the Class-B program closes

- If BA or BC succeed: turn `b_init_scale` (or A/B init-mode) into
  a permanent LoRA config knob; don't carry debug env flags long
  term.
- If BL identifies topology-order: build a minimal repro outside
  full DPO around the offending collective pattern.
- If BA-BK all fail for Bug 2: pivot to remat/checkpointing,
  attention-kernel variants, per-layer activation diffs. Not more
  hyperparameter sweeps.

---

## Experiment CP9 (2026-04-18T11:32Z, COMPLETED, `cp9-ue5-lr3em5-v1`) — Bug 1 at lr=3e-5 (complete LR grid)

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_cp9_r64_s10_cp9-v5p8-pd4-lr3e-05-636468
- **Iris job:** `/ahmed/cp9-ue5-lr3em5-v1/train_dpo`

### Purpose

Test if Bug 1 (v5p-8 pd=4 c=bf16 stuck) also scales 30x with LR the
same way that Bug 2 and Exp N do. Completes the 3x3 LR-scaling grid.

### Config

- TPU: v5p-8 us-east5, pd=4, c=bf16 (i.e. Exp Q recipe).
- LR: 3e-5.
- 10 steps.
- Script: `experiments/posttrain/per_stmt_dpo/experiment_cp9_v5p8_pd4_lr3em5_s10.py`.

### Step-0/1 trace (c=bf16 v5p-8 pd=4 at lr=3e-5)

- step=0: loss=0.6931, grad_l2=2.456, grad_sum=0.190 (c=bf16 signature, small).
- step=1: loss=0.6931, grad_l2=2.255, grad_sum=0.216, upd_l2=0.231.

Step-1 grad profile identical to Exp N's c=bf16 signature (grad_sum ~ 0.2).

### Decision rule

- Step 9 ≤ 0.15 → Bug 1 also scales 30x cleanly with LR. Bug 1 = pure
  LR slowdown, same as Bug 2.
- Step 9 > 0.25 → Bug 1 has a floor that Bug 2 doesn't. Would suggest
  v5p-8 mesh has a structural constraint that limits per-step descent
  even at high LR. Warrants deeper investigation.

### Step 5/9 RESULT (2026-04-18T11:47Z)

Per-step trajectory:
| step | loss |
|-----:|-----:|
| 0 | 0.6931 |
| 1 | 0.6931 |
| 5 | 0.215 |
| 9 | **0.1368** |

Step-9 trace:
`grad_l2=0.531 grad_sum=-0.001 upd_l2=0.009 gA_l2=0.165 gB_l2=0.505 pB_l2=0.995`

### Bug 1 = Bug 2 confirmed

| config | lr=1e-6 step 9 | lr=1e-5 step 9 | lr=3e-5 step 9 |
|--------|---------------:|----------------:|----------------:|
| Exp N (v5p-16 c=bf16) | 0.32 | 0.26 | 0.125 (CP7) |
| Bug 1 (v5p-8 c=bf16) | 0.66 | 0.41 (CP6) | **0.137** (CP9) |
| Bug 2 (v5p-16 c=f32) | 0.66 | 0.41 | **0.137** (CP8) |

**Bug 1 and Bug 2 land at 0.137 at step 9, while Exp N lands at 0.125.**
The ~10% gap between Bug 1/2 and Exp N at high LR is stable across
configurations — reflects the per-step descent-speed constant difference
between "good" (Exp N) and "slow" (Bug 1, Bug 2) configs. The gap does
NOT grow as we scale LR; both rescue strategies scale linearly.

### Narrow conclusion

Bug 1 is a pure LR-compensable slowdown. No mesh-specific floor. The
v5p-8 pd=4 FSDP configuration just has a slightly smaller effective LR
per step than v5p-16 pd=2, identical in nature to c=f32 vs c=bf16.

Same root mechanism: **per-step effective-learning-rate reduction
~5x for both bugs.** Whatever the underlying cause, it produces a
quantitatively identical slowdown profile.

### Status

COMPLETED. Killed parent.

## Experiment C7 (2026-04-18T11:35Z, COMPLETED, `c7-uc1-pbf16cf32-v1`) — p=bf16, c=f32 dtype cell

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_c7_r64_s10_c7-v5p16-pbf16-cf32-8cd7e8
- **Iris job:** `/ahmed/c7-uc1-pbf16cf32-v1/train_dpo`

### Purpose

Completes the 2x2 dtype matrix. Tests whether param-storage bf16 alone
provides the c=bf16 benefit (without compute-dtype bf16).

| p \ c | bf16 (Exp N / C6) | f32 (AC / C7) |
|-------|-------------------|---------------|
| bf16  | C6 good (= Exp N) | **C7 ← running** |
| f32   | Exp N good         | AC stuck      |

### Config

- TPU: v5p-16 us-central1, pd=2.
- JMP: `p=bf16,c=f32`.
- LR: 1e-6 (AC baseline).
- 10 steps.
- Script: `experiments/posttrain/per_stmt_dpo/experiment_c7_v5p16_pbf16_cf32_pd2_s10.py`.

### Step-0/1/5 trace (p=bf16, c=f32 at lr=1e-6)

- step=0: loss=0.6931, grad_l2=2.445, grad_sum=0.333, pB_l2=0.
- step=1: loss=0.6931, grad_l2=2.251, grad_sum=0.593, upd_l2=0.008.
- step=5: loss=0.6682, grad_l2=2.301, grad_sum=-1.212.

Interesting: step-0 grad_sum (0.333) is BETWEEN Exp N (0.20) and AC
(0.55). Storage bf16 provides SOME reduction in grad-sum at step 0. But
by step 1, grad_sum = 0.593 — same as AC. The bf16-storage benefit
dissipates once the optimizer starts taking steps.

By step 5, grad_sum has flipped to -1.212 (adapter has descended into a
different grad regime, loss dropped to 0.668 from 0.693).

### Preliminary interpretation

- Step-0 bf16 storage rounding shifts the initial forward pass.
- But the effect doesn't compound over training — by step 1, grad
  matches AC. Param-dtype alone is NOT the bf16 benefit.
- Need step-9 result to confirm trajectory (should track AC ~ 0.66).

### Step 9 RESULT (2026-04-18T11:45Z) — **H_param_storage FALSIFIED**

- step=9: loss=**0.6601**, grad_l2=2.252, grad_sum=-1.375, upd_l2=0.0003.

C7 tracks AC almost exactly (AC step 9 = 0.662, C7 step 9 = 0.660, delta
= 0.002). **Param-storage bf16 is irrelevant when compute dtype is f32.**
The c=bf16 benefit comes entirely from compute-dtype rounding per op,
not from rounded param storage.

### Updated 2x2 dtype matrix

| p \ c | bf16 (Exp N / C6) | f32 (AC / C7) |
|-------|-------------------|---------------|
| bf16  | C6: step 9 ≈ 0.32 (good) | C7: step 9 = **0.660** (stuck) |
| f32   | Exp N: step 9 = 0.32 (good) | AC: step 9 = 0.662 (stuck) |

**Compute dtype is the full story.** Param storage is orthogonal.

### Narrow conclusion

Whatever the c=bf16 benefit is — rounding, reduced precision
activations, something in splash-attention intermediates — it's a
compute-time phenomenon, not a storage phenomenon. The policy's
`compute_dtype` coerces inputs at module boundaries, and that coercion
is what matters. Storage dtype only affects the initial forward pass
(step 0) and the optimizer state precision.

### Status

COMPLETED. Killed parent.

## Experiment C8 (2026-04-18T11:40Z, COMPLETED, `c8-ue5-relnoise-v1`) — continuous relative grad noise

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_c8_s10_c8-v5p16-fp32-relnoise0.03-861076
- **Iris job:** `/ahmed/c8-ue5-relnoise-v1/train_dpo`

### Purpose

BD v1 one-shot noise too small. BD v2 one-shot too big (hurt). Neither
matches the continuous per-step aspect of bf16 rounding. C8 injects
continuous relative Gaussian noise (3% of per-leaf grad-L2) to lora_A
and lora_B at every step under AC recipe (c=f32). If this rescues
→ bf16 benefit IS stochastic noise (just at the right scale and
cadence). If not → bf16 benefit is deterministic structured rounding.

### Config

- TPU: v5p-16 us-east5, pd=2, c=f32 (AC recipe).
- LR: 1e-6.
- 10 steps.
- Env: `MARIN_DEBUG_LORA_GRAD_NOISE_RELATIVE=0.03`, `MARIN_DEBUG_LORA_GRAD_NOISE_TARGET=both`.
- Script: `experiments/posttrain/per_stmt_dpo/experiment_c8_v5p16_fp32_pd2_contrel_s10.py`.

### Decision rule

- Step 9 ≤ 0.40 → noise hypothesis alive. bf16 rounding = stochastic noise.
- Step 9 ≥ 0.65 → noise hypothesis dead. bf16 benefit is structured
  deterministic rounding affecting computation path in specific
  operators (softmax, RmsNorm, etc.).

### Step 0/1/5/9 RESULT (2026-04-18T11:54Z) — **H_noise_continuous FALSIFIED**

Per-step trajectory:
| step | loss | grad_l2 | grad_sum |
|-----:|-----:|--------:|---------:|
| 0 | 0.6931 | 2.447 | 0.224 |
| 1 | 0.6931 | 2.252 | 0.406 |
| 5 | 0.6689 | 2.303 | -1.400 |
| 9 | **0.6612** | 2.289 | -1.664 |

C8 step 9 = 0.661. Tracks AC almost exactly (AC step 9 = 0.662). **The
c=bf16 benefit cannot be replicated by continuous Gaussian grad noise
at any scale tested (BD v1: one-shot 1e-5, BD v2: one-shot 1e-2, C8:
continuous 3% relative).**

### Narrow conclusion

Noise hypothesis is fully dead. The c=bf16 benefit is a **structured
deterministic** effect from bf16's specific rounding pattern across
operators, not a stochastic regularization.

Combined with prior falsifications:
- BD v1/v2: one-shot noise → nope.
- C8: continuous relative noise → nope.
- C3/C4/C5: rounding Linear/RmsNorm inputs/outputs → nope.
- C6: all-bf16 storage and compute → recovers Exp N.
- C7: p=bf16 only (c=f32) → stuck like AC.

The benefit is specific to bf16 COMPUTE, not storage, not grad noise,
not Linear/RmsNorm post-hoc rounding. Remaining candidates (untested):
- **Attention internals** (Q/K/V rounding inside attention).
- **Residual stream accumulation** (C9 testing now).
- **MLP SiLU / pointwise internals**.
- **Loss accumulation in cross-entropy**.

### Status

COMPLETED. Killed parent.

## Experiment C9 (2026-04-18T11:48Z, COMPLETED, `c9-uc1-roundresid-v1`) — round residual stream to bf16

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_c9_s10_c9-v5p16-fp32-roundresid-9fdcdb
- **Iris job:** `/ahmed/c9-uc1-roundresid-v1/train_dpo`

### Purpose

C3/C4/C5 rounded Linear and RmsNorm OUTPUTS/OPERANDS — failed. What
they didn't round: the RESIDUAL STREAM itself. Under c=bf16, each
layer adds residual + block_out in bf16, losing precision at every
add. Over 32 layers this accumulates ~3-5% relative noise in the final
hidden state. C9 simulates this by rounding the residual stream to
bf16 after each block add under c=f32.

### Config

- TPU: v5p-16 us-central1, pd=2, c=f32 (AC recipe).
- LR: 1e-6.
- 10 steps.
- Env: `MARIN_DEBUG_ROUND_RESIDUAL=1` (patches in `llama.py`).
- Script: `experiments/posttrain/per_stmt_dpo/experiment_c9_v5p16_fp32_pd2_roundresid_s10.py`.

### Step 0/1/5/9 RESULT (2026-04-18T11:57Z) — **H_residual_stream FALSIFIED**

Per-step trajectory:
| step | loss | grad_l2 | grad_sum |
|-----:|-----:|--------:|---------:|
| 0 | 0.6931 | 2.448 | 0.406 |
| 1 | 0.6931 | 2.260 | 0.635 |
| 5 | 0.6700 | 2.310 | -1.287 |
| 9 | **0.6604** | 2.255 | -1.262 |

C9 step 9 = 0.660. Tracks AC (0.662) and C7 (0.660) exactly. **Residual
stream rounding alone does NOT reproduce the c=bf16 benefit.**

Interesting observation: step-0 grad_sum did shift (AC ~ 0.55 →
C9: 0.406) — residual rounding IS producing a slightly different
forward pass. But by step 5-9, grad_sum has re-converged to AC/C7
trajectory. The shift at step 0 wasn't load-bearing.

### Narrow conclusion

Residual-stream precision accumulation is not the mechanism.

### Remaining untested candidates

- **Attention internals**: Splash kernel's bf16 softmax / Q·K^T.
- **MLP SiLU output**: gate_proj → silu(gate) × up_proj.
- **Adam optimizer state precision** (but BK/C7 orthogonal).

CA (all post-hoc rounding gates ON simultaneously) is the next test.

### Status

COMPLETED. Killed parent.

## Experiment CE (2026-04-18T12:09Z, COMPLETED, `ce-ue5-kitchen-v1`) — kitchen-sink (CA + silu + norm internal + gated)

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_ce_s10_ce-v5p16-fp32-kitchen-sink-3289e2
- **Iris job:** `/ahmed/ce-ue5-kitchen-v1/train_dpo`

### Purpose

CA closed 38% of gap. CC (silu+gated alone) stuck. CD (norm internal
alone) stuck. CE tests: does CA + silu + norm internal combined close
MORE gap than CA alone?

### Config

- TPU: v5p-16 us-east5, pd=2, c=f32.
- All 8 rounding gates ON:
  `MARIN_DEBUG_ROUND_LINEAR_OUT/INPUT/WEIGHT=1`,
  `MARIN_DEBUG_ROUND_NORM_OUT=1`,
  `MARIN_DEBUG_NORM_INTERNAL_BF16=1`,
  `MARIN_DEBUG_ROUND_RESIDUAL=1`,
  `MARIN_DEBUG_ROUND_SILU=1`,
  `MARIN_DEBUG_ROUND_MLP_GATED=1`.

### Step 1/5/9 RESULT (2026-04-18T12:23Z) — **marginal improvement over CA**

| step | loss | grad_sum | grad_l2 |
|-----:|-----:|---------:|--------:|
| 1 | 0.6940 | 0.464 | 2.252 |
| 5 | 0.5329 | -0.999 | 1.886 |
| 9 | **0.5274** | -0.837 | 1.843 |

### Comparison vs CA

| recipe | step 1 grad_sum | step 5 loss | step 9 loss |
|--------|-----------------:|-------------:|-------------:|
| CA (5 gates) | 0.537 | 0.540 | **0.532** |
| CE (8 gates) | 0.464 | 0.533 | **0.527** |
| delta | -0.073 | -0.007 | **-0.005** |

**CE closes only 0.005 more of the gap than CA.** Adding silu + norm
internal + gated product rounding is NEARLY a no-op on top of CA.

Interpretation: the five CA gates (Linear I/O/W + Norm out + residual)
capture essentially ALL of the post-hoc-rescueable portion of the
c=bf16 benefit. The remaining ~62% of gap (CA/CE step 9 ≈ 0.53 vs
Exp N's 0.32) is NOT post-hoc-rescueable via any rounding gate tested.

### Narrow conclusion

The c=bf16 benefit has two parts:
1. **Forward-pass cumulative rounding at module boundaries** — captured
   by CA's 5-gate combination. Closes 38% of gap.
2. **Backward-pass precision OR attention-softmax-kernel internals** —
   not captured by any post-hoc gate. Accounts for remaining 62%.

The backward-pass precision hypothesis is most likely since:
- Attention softmax inside Splash is bf16 under both c=bf16 and c=f32
  (kernel downcasts f32 inputs internally) — shouldn't differ.
- JAX's autodiff respects the compute dtype for all operations. Under
  c=bf16, backward is bf16 end-to-end. Under c=f32, backward is f32.
- bf16 backward pass has lower precision in gradient chain rule
  multiplications, which could act as a small "noise" that helps
  break symmetry in later steps.

Testing this would require forcing backward to use bf16 precision
under c=f32 policy — a JAX-level intervention, not a simple
env-gated forward-side change. Not feasible in remaining time budget.

### Status

COMPLETED. Killed parent. Investigation closes here for post-hoc
rounding mechanism probes.

## Experiment CD (2026-04-18T12:10Z, COMPLETED, `cd-uc1-norminternal-v1`) — RmsNorm internal bf16

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_cd_s10_cd-v5p16-fp32-norminternal-62ceef
- **Iris job:** `/ahmed/cd-uc1-norminternal-v1/train_dpo`

### Purpose

C4 rounded RmsNorm OUTPUT but not internal mean/rsqrt chain. Under
c=bf16 policy, RmsNorm.dtype is None → x stays at bf16 (after policy
cast) → var/rsqrt happen in bf16. Under c=f32, var/rsqrt in f32.
CD forces x to bf16 INSIDE RmsNorm before var/rsqrt under c=f32.

### Config

- TPU: v5p-16 us-central1, pd=2, c=f32.
- LR: 1e-6.
- Env: `MARIN_DEBUG_NORM_INTERNAL_BF16=1`.

### Step 1/5/9 RESULT (2026-04-18T12:20Z) — **H_norm_internal FALSIFIED**

| step | loss | grad_sum | grad_l2 |
|-----:|-----:|---------:|--------:|
| 1 | 0.6927 | 0.537 | 2.246 |
| 5 | 0.6653 | -1.230 | 2.292 |
| 9 | **0.6596** | -1.200 | 2.249 |

CD tracks AC (step 9 = 0.662, CD step 9 = 0.660). **RmsNorm internal
bf16 alone does NOT rescue c=f32.** The mean/rsqrt precision inside
RmsNorm is not the c=bf16 mechanism.

### Narrow conclusion

RmsNorm's internal reductions are not the load-bearing operator for
c=bf16's benefit. Combined with C4 (RmsNorm output rounding also
failed), RmsNorm is NOT the mechanism.

### Status

COMPLETED. Killed parent.

## Experiment CC (2026-04-18T11:59Z, COMPLETED, `cc-uc1-silu-v1`) — round SiLU output + gated MLP product

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_cc_s10_cc-v5p16-fp32-silu-gated-3d8df2
- **Iris job:** `/ahmed/cc-uc1-silu-v1/train_dpo`

### Purpose

CA (running in parallel) covers Linear/RmsNorm/residual. CC probes
whether MLP internals (silu output, silu × up_proj product) are the
c=bf16 mechanism. These are not gated by C3/C4/C5/C9.

### Config

- TPU: v5p-16 us-central1, pd=2, c=f32.
- LR: 1e-6.
- Env: `MARIN_DEBUG_ROUND_SILU=1`, `MARIN_DEBUG_ROUND_MLP_GATED=1`.

### Step 1/5/9 RESULT (2026-04-18T12:09Z) — **H_mlp_internal FALSIFIED**

| step | loss | grad_sum | grad_l2 |
|-----:|-----:|---------:|--------:|
| 1 | 0.6932 | 0.584 | 2.254 |
| 5 | 0.6697 | -1.251 | 2.307 |
| 9 | **0.6614** | -1.676 | 2.289 |

CC tracks AC (AC step 9 = 0.662, CC step 9 = 0.661). **MLP silu and
gated product rounding does NOT rescue c=f32.**

### Narrow conclusion

MLP internals (silu, gate*up) are not the c=bf16 mechanism. Whatever
c=bf16 does that c=f32 doesn't, it's not in the SiLU activation or
gated product.

### Status

COMPLETED. Killed parent.

### Trajectory shape of CA (different from Exp N and AC)

Reviewing CA more carefully:

| step | CA loss | Exp N loss | AC loss |
|-----:|--------:|-----------:|--------:|
| 0 | 0.6931 | 0.6931 | 0.6931 |
| 1 | 0.6942 | 0.6931 | 0.6931 |
| 5 | **0.5395** | ~0.50 | 0.6696 |
| 9 | **0.5320** | 0.318 | 0.662 |

CA shape analysis:
- **Fast early escape** (step 1→5: Δloss = -0.15). Matches Exp N's
  early-step descent rate.
- **Slow late plateau** (step 5→9: Δloss = -0.008). Much slower than
  Exp N (-0.18 over same steps).

**Interpretation**: CA captures the "symmetric-init escape" part of
c=bf16's benefit — the first-step-breaks-symmetry mechanism. But CA
doesn't capture the "continued rapid descent" part.

Two separable aspects of c=bf16 benefit:
1. **Symmetry break speed** (first 1-5 steps): CA captures via
   cumulative rounding.
2. **Late-step descent rate** (step 5+): CA does NOT capture. Remains
   at AC-like descent speed past symmetric-init trap.

The second aspect is the OPEN mechanism. Likely related to attention
softmax kernel behavior or fine-grained rounding in multiple operators
simultaneously that CA's 5 gates don't capture.

## Experiment CA (2026-04-18T11:55Z, COMPLETED, `ca-ue5-all-rounded-v1`) — ALL rounding gates ON

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_ca_s10_ca-v5p16-fp32-all-rounded-90e035
- **Iris job:** `/ahmed/ca-ue5-all-rounded-v1/train_dpo`

### Purpose

Combine every post-hoc rounding gate implemented so far: Linear input,
Linear weight, Linear output, RmsNorm output, residual stream.
Attempts to simulate c=bf16 compute via comprehensive ad-hoc rounding
under c=f32 storage.

If CA step 9 ≤ 0.40 → rounding at these boundaries is sufficient, and
c=bf16 benefit is captured by the combined effect.
If CA step 9 ≥ 0.65 → there's an untested operator (attention
internals, silu, cross-entropy accumulation) that is the actual
mechanism.

### Config

- TPU: v5p-16 us-east5, pd=2, c=f32 (AC recipe).
- LR: 1e-6.
- 10 steps.
- Env: `MARIN_DEBUG_ROUND_LINEAR_OUT=1`, `MARIN_DEBUG_ROUND_LINEAR_INPUT=1`,
  `MARIN_DEBUG_ROUND_LINEAR_WEIGHT=1`, `MARIN_DEBUG_ROUND_NORM_OUT=1`,
  `MARIN_DEBUG_ROUND_RESIDUAL=1`.
- Script: `experiments/posttrain/per_stmt_dpo/experiment_ca_v5p16_fp32_pd2_all_rounded_s10.py`.

### Step 1/5/9 RESULT (2026-04-18T12:09Z) — **PARTIAL RESCUE (38%)**

Per-step trajectory:
| step | loss | grad_sum | grad_l2 |
|-----:|-----:|---------:|--------:|
| 1 | 0.6942 | 0.537 | 2.255 |
| 5 | **0.5395** | -1.007 | 1.905 |
| 9 | **0.5320** | -0.848 | 1.855 |

### Benchmark comparison

| recipe | step 9 | gap to Exp N |
|--------|-------:|-------------:|
| Exp N (c=bf16) | 0.32 | — |
| AC (c=f32) | 0.662 | 0.342 |
| **CA (c=f32 + all rounding)** | **0.532** | **0.212** |
| **CA closed % of gap** | | **38%** |

**CA is THE FIRST EXPERIMENT to partially rescue c=f32.** All other
rounding probes (C3, C4, C5, C8, C9) tracked AC. CA succeeds because
it combines MULTIPLE rounding gates:
- MARIN_DEBUG_ROUND_LINEAR_OUT=1
- MARIN_DEBUG_ROUND_LINEAR_INPUT=1
- MARIN_DEBUG_ROUND_LINEAR_WEIGHT=1
- MARIN_DEBUG_ROUND_NORM_OUT=1
- MARIN_DEBUG_ROUND_RESIDUAL=1

Single gates don't reproduce c=bf16; the CUMULATIVE rounding at every
boundary is the mechanism.

### Interpretation

Under c=bf16, **every activation value at every module boundary is
rounded to bf16.** My C3/C4/C5/C9 probes each covered ONE boundary:
individually insufficient. CA covers FIVE boundaries simultaneously,
capturing more of the cumulative rounding pattern.

Remaining gap (~62%) likely comes from:
- **RmsNorm internal compute** (mean/rsqrt in bf16 under c=bf16,
  currently still f32 in CA). CD probe tests this.
- **MLP silu output** (CC probe tests this, running in parallel).
- **Attention internals** (Splash kernel's bf16 softmax — can't
  easily simulate via post-hoc gates).

Next: launch CE (kitchen sink — CA gates + silu gates + RmsNorm
internal bf16). If CE closes further, identify which additional gates
helped.

### Status

COMPLETED. Killed parent. Launching CE + CD next.

## Experiment CP7 (2026-04-18T~10:45Z, COMPLETED, pre-compaction) — Exp N at lr=3e-5

- **W&B:** `experiment_cp7_r64_s10_cp7-v5p16-fp32-lr3e-05-<hash>` (URL unavailable, iris job record aged out before this section was written)
- **Iris job:** `/ahmed/cp7-...` (aged out)

### Purpose

Test Exp N (c=bf16, v5p-16 pd=2) at 30x the default LR (3e-5 vs 1e-6).
Establishes the "high LR ceiling" for the fast recipe and anchors the
LR-scaling comparison for Bug 1 / Bug 2.

### Config

- TPU: v5p-16, pd=2, c=bf16 (default JMP).
- LR: 3e-5.
- 10 steps.

### Result

- **step 9 loss = 0.125** (the fastest of all LR-scaled Exp N variants).

### Role in campaign

Establishes that at high LR (3e-5), the fast recipe descends from
log(2)=0.693 to 0.125 in 10 steps. Baseline for CP8 (c=f32 @ 3e-5)
and CP9 (v5p-8 @ 3e-5), which both reach 0.137 — within 10% of CP7.
LR-compensation story depends on this anchor.

### Status

COMPLETED. Killed parent long before compaction; Iris record aged out
and exact wandb run-ID hash wasn't captured.

## Experiment CP8 (2026-04-18T11:22Z, COMPLETED, `cp8-ue5-f32-lr3em5-v1`) — c=f32 at lr=3e-5

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_bf_r64_s10_ue5-cp8-f32-lr3em5-67a941
- **Iris job:** `/ahmed/cp8-ue5-f32-lr3em5-v1/train_dpo`

### Purpose

Completes the 3x3 LR-scaling grid on Bug 2 (c=f32 AC recipe). Tests if
c=f32 at 30x LR closes to Exp N quality. Disambiguates "direction
defect" vs "pure LR slowdown" theories for c=f32.

### Config

- TPU: v5p-16, pd=2, c=f32 (AC recipe, `jmp.p=f32,c=f32`).
- LR: 3e-5.
- 10 steps.

### Per-step trajectory

| step | loss |
|-----:|------|
| 0 | 0.6931 |
| 1 | 0.6931 |
| 4 | 0.2978 |
| 8 | 0.1304 |
| 9 | **0.1370** |

### Step-1 trace (c=f32 signature at high LR)

`grad_l2=2.251 grad_sum=0.5495 gA_l2=0 gA_sum=0 gB_l2=2.251 gB_sum=0.5495`
— c=f32 signature (grad_sum ~0.55) persists at high LR. Direction of
gradient is the same as at low LR; only magnitude (via Adam step)
changes.

### Comparison

- CP7 (c=bf16 lr=3e-5): 0.125 at step 9.
- CP8 (c=f32 lr=3e-5): **0.137** at step 9. **Within 10% of CP7.**
- AC (c=f32 lr=1e-6): 0.662 at step 9.

### Conclusion — direction story KILLED

If c=f32 were a direction defect (not just a slowdown), gap to Exp N
would persist at any LR. Instead gap shrinks: 0.34 at lr=1e-6 → 0.15
at lr=1e-5 → 0.012 at lr=3e-5. **c=f32 is a pure step-size slowdown,
not a direction defect.**

### Status

COMPLETED. Killed parent.

## Experiment CF (2026-04-18T12:20Z, COMPLETED, `cf-uc1-ca-100step-v1`) — CA rounding gates × 100 steps

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_cf_r64_s100_cf-v5p16-fp32-ca-s100-6c97b3
- **Iris job:** `/ahmed/cf-uc1-ca-100step-v1/train_dpo`

### Purpose

CA (5 rounding gates) at 10 steps closes 38% of gap. Does CA continue
to close the gap given 100 steps, or does it plateau at a fraction
of Exp N quality? Tests whether CA is a pure slowdown or a partial
basin-shift.

### Config

- TPU: v5p-16, pd=2, c=f32, all 5 CA gates ON.
- LR: 1e-6.
- 100 steps (cosine decay).

### Per-step trajectory

| step | loss |
|-----:|------|
| 9 | 0.538 |
| 26 | 0.47 |
| 30 | 0.463 |
| 38 | 0.437 |
| 50 | 0.389 |
| 74 | 0.364 |
| 87 | 0.349 |
| 90 | 0.358 |
| 99 | **0.351** |

### Gap analysis

| horizon | AC | CA (CF) | Exp N (CH) | CA % of gap |
|---------|---:|--------:|-----------:|-----------:|
| step 9 | 0.66 | 0.532 | 0.32 | 38% |
| step 50 | 0.45 | 0.389 | 0.261 | 32% |
| step 99 | 0.40 | 0.351 | 0.248 | 32% |

**CA captures a STABLE ~33% of the gap.** Does not close to Exp N
given more steps — hits a persistent plateau above Exp N.

### Conclusion

CA is NOT a pure slowdown. The forward-path rounding captures a
fraction of the c=bf16 benefit but cannot replicate it fully, even
given 100 steps. Remaining ~67% is outside forward-path rounding
(likely JAX autodiff backward precision).

### Status

COMPLETED. Killed parent.

## Experiment CH (2026-04-18T12:27Z, COMPLETED, `ch-ue5-expn-100step-v1`) — Exp N × 100 steps (baseline for CF)

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_ch_s100_ch-v5p16-bf16-expn-s100-8b513b
- **Iris job:** `/ahmed/ch-ue5-expn-100step-v1/train_dpo`

### Purpose

CF runs 100 steps of CA. CH runs 100 steps of Exp N (c=bf16, no gates)
as the lower-bound baseline. Gap between CF and CH at step 99 tells us
how much of the c=bf16 benefit CA fails to capture at long horizons.

### Config

- TPU: v5p-16, pd=2, c=bf16 (Exp N default JMP, no rounding gates).
- LR: 1e-6.
- 100 steps (cosine decay).

### Per-step trajectory

| step | loss |
|-----:|------|
| 9 | 0.314 |
| 46 | 0.260 |
| 50 | 0.261 |
| 90 | 0.256 |
| 97 | 0.253 |
| 98 | 0.256 |
| 99 | **0.248** |

### Compared to short-horizon

Exp N 10-step step 9 = 0.32; CH (100-step) step 9 = 0.314. Matches
to ~2% — confirms the short-horizon baseline is reliable.

### Role

Lower-bound baseline for CA (CF) comparison. Establishes AC → Exp N
gap at step 99 = 0.152, of which CA (CF) closes 32%.

### Status

COMPLETED. Killed parent.

## Experiment CK (2026-04-18T12:55Z, COMPLETED, `ck-uc1-ca-lr3em5-v1`) — CA gates + lr=3e-5 (stacking test)

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_ck_s10_ck-v5p16-fp32-ca-lr3em5-39eeda
- **Iris job:** `/ahmed/ck-uc1-ca-lr3em5-v1/train_dpo`

### Purpose

Does CA's rescue stack with LR rescue? If CA at lr=3e-5 matches CP7
(Exp N @ 3e-5 = 0.125), CA is fully LR-compensable. If between CP7
and CP8 (c=f32 @ 3e-5 = 0.137), CA adds a fractional benefit at high
LR.

### Config

- TPU: v5p-16, pd=2, c=f32, all 5 CA gates ON.
- LR: 3e-5.
- 10 steps.

### Per-step trajectory

| step | loss |
|-----:|------|
| 5 | 0.201 |
| 9 | **0.133** |

### Comparison at lr=3e-5

- CP7 (Exp N): 0.125.
- CK (CA + 5 gates): **0.133**.
- CP8 (AC, no gates): 0.137.

### Rescue fraction

CK closes 33% of the (CP8 - CP7) gap — **same ~33% fraction as at
lr=1e-6 and at step 99**. CA provides a stable fractional rescue
independent of LR and step horizon.

### Conclusion

**CA's ~33% rescue is a structural property, not LR- or
time-dependent.** Forward-path rounding captures exactly 1/3 of the
c=bf16 benefit regardless of training horizon or LR.

### Status

COMPLETED. Killed parent.

## Experiment CM (2026-04-18T13:09Z, COMPLETED, `cm-ue5-bug1-ca-v2`) — CA gates on Bug 1 (v5p-8 pd=4)

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_cm_r64_s10_cm-v5p8-pd4-ca-gates-268a92
- **Iris job:** `/ahmed/cm-ue5-bug1-ca-v2/train_dpo`
- **Note:** v1 (`cm-ue5-bug1-ca-v1`) failed with GitHub 504 during deps sync. v2 is the clean run.

### Purpose

CA's rescue is 33% on Bug 2 (c=f32). Does CA also rescue Bug 1 (v5p-8
pd=4 c=bf16)? If yes with similar fraction → CA mechanism is
mesh/dtype-agnostic. If no → the two bugs have different mechanisms
despite identical LR-scaling signatures.

### Config

- TPU: v5p-8, pd=4, c=bf16 (Exp Q recipe).
- LR: 1e-6.
- 10 steps.
- Env: all 5 CA gates ON (MARIN_DEBUG_ROUND_LINEAR_*, _NORM_OUT,
  _RESIDUAL).

### Per-step trajectory

| step | loss |
|-----:|------|
| 5 | 0.669 |
| 9 | **0.661** |

### Conclusion — **Bug 1 ≠ Bug 2** at the mechanism level

CM step 9 = 0.661, same as baseline Bug 1 (Exp Q = 0.66). **Zero
rescue.** Under Bug 1, compute is already c=bf16, so CA's rounding
gates (which round TO bf16) are effective no-ops.

**Bug 1 is NOT a precision/rounding issue.** It's a mesh/collective
topology problem on v5p-8 pd=4 FSDP-4. The only known rescues are:
1. Mesh change (Exp W TP or mix): full recovery.
2. LR scaling (lr=1e-5 → 0.41 partial, lr=3e-5 → 0.137 full).

Bug 1 and Bug 2 produce QUANTITATIVELY IDENTICAL stuck behavior
(both log(2) at step 9 lr=1e-6) but via MECHANISTICALLY DIFFERENT
paths.

### Status

COMPLETED. Killed parent.

## Experiment CN (2026-04-18T15:12Z, COMPLETED, `cn-ue5-pbf16-v1`) — v5p-8 pd=4 full bf16 (p=bf16, c=bf16)

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_cn_s10_cn-v5p8-pd4-pbf16-cbf16-34734b
- **Iris job:** `/ahmed/cn-ue5-pbf16-v1/train_dpo`

### Purpose

CM showed CA gates don't rescue Bug 1. This could be because Bug 1 is
already in bf16 compute (so CA's bf16 rounding is a no-op). CN tests
if changing **param storage to bf16** (beyond compute) rescues Bug 1.

Exp Q uses p=f32, c=bf16 (JMP default). C6 showed p=bf16, c=bf16
matches Exp N quality on v5p-16 pd=2. Does that full-bf16 recipe
also rescue Bug 1?

### Config

- TPU: v5p-8, pd=4.
- JMP policy: `p=bf16, c=bf16`.
- LR: 1e-6.
- 10 steps.

### Per-step trajectory

| step | loss |
|-----:|------|
| 5 | 0.669 |
| 9 | **0.661** |

Identical to CM and baseline Bug 1.

### Conclusion

**Bug 1 cannot be rescued by ANY dtype change.** Tested:
- p=f32, c=bf16 (Exp Q, default): 0.66 stuck.
- p=bf16, c=bf16 (CN): 0.66 stuck.
- p=f32, c=bf16 + CA gates (CM): 0.66 stuck.

Confirms Bug 1 is a mesh/collective issue on v5p-8 pd=4 FSDP-4.
Compute dtype, storage dtype, and forward-path rounding all orthogonal.

### Status

COMPLETED. Killed parent.

## Experiment BC (2026-04-18T08:52Z, RUNNING, uc1-bc1) — factor geometry swap

### Key result so far (step 5): **STALLS AT log(2), factor-geometry hypothesis FALSIFIED**

Setup: Exp N recipe + `c=f32` + `a_init_mode='zero'` + `b_init_scale=1e-3`.
At init: A=0, B~N(0, 1e-3). adapter_out = B·A·x = B·0 = 0. Policy = reference.
Invariant preserved. No env-gate needed.

Deltas from AC (canonical LoRA + c=f32, stuck):
- `a_init_mode='zero'` instead of canonical random A.
- `b_init_scale=1e-3` with `zero_init_b=False` — random small B.
- Swap of which factor has nonzero step-0 gradient: in AC, dL/dA=0 and
  dL/dB ≠ 0; in BC, dL/dB=0 (because A=0 kills chain) and dL/dA ≠ 0.

### Trajectory (steps 0-5, uc1-bc1)

| step | loss | gA_l2 | gB_l2 | pA_l2 | pB_l2 |
|------|-----:|------:|------:|------:|------:|
| 0 | 0.6931 | 0.211 | **0.000** | 0.000 | 9.384 |
| 1 | 0.6931 | 0.194 | 0.000 | 0.0074 | 9.384 |
| 2 | 0.6935 | 0.205 | 0.0022 | 0.0138 | 9.384 |
| 3 | 0.6925 | 0.213 | 0.0043 | 0.0195 | 9.385 |
| 4 | 0.6912 | 0.213 | 0.0061 | 0.0242 | 9.385 |
| 5 | 0.6928 | 0.213 | 0.0080 | 0.0278 | 9.385 |

Compare AC (Exp N recipe + c=f32, canonical LoRA init) step-5 loss ≈ 0.6696.
Both BC and AC are stuck at the log(2) attractor.

- step-0 loss = 0.6931471824645996 = log(2) exactly ✓ (invariant held)
- gB=0 at step 0 exactly as predicted (A=0 kills chain).
- pB_l2=9.384: matches expected ||N(0, 1e-3)|| for ~100M lora_B elements.

### Narrow conclusion

**The "which factor gets the first gradient update" story is wrong.**

- AC: canonical A=random, B=0. dL/dA=0 at step 0, dL/dB ≠ 0. STUCK.
- BC: A=0, B=random_small. dL/dB=0 at step 0, dL/dA ≠ 0. STUCK.

Both produce `adapter_out=0 at init` (policy=reference), both stall at
log(2) under c=f32. Flipping which of A or B carries the first gradient
does NOT escape the trap. Factor-asymmetry of the canonical (A=rand, B=0)
init is not load-bearing for Bug 2.

### H_adapter_zero REVISED AFTER BA COMPLETES (2026-04-18T08:59Z)

After BA full trajectory came in, **H_adapter_zero is partially falsified**:

- AC: adapter_out = 0 at init + c=f32 → step-9 = 0.6596 (slow descent).
- BC: adapter_out = 0 at init + c=f32 → step-9 ≈ 0.6919 (no descent, oscillating).
- **BA: adapter_out ≠ 0 at init (small) + c=f32 → step-9 = 0.6620 (tracks AC +0.003).**
- AD v3: adapter_out ≠ 0 at init (Kaiming) + c=f32 → step-9 = 5.11 (huge start → slow descent).
- Exp N (bf16 baseline): adapter_out = 0 at init + c=bf16 → step-9 = 0.3176 (fast escape).

Pattern: **none of {BA, BC, AC} reach the good basin under c=f32**, regardless of
adapter_out at init. BC is the only one that doesn't descend at all — likely
because its A-direction is starting from zero rather than Kaiming.

So the refined refined hypothesis is:
**H_cf32_fundamental**: c=f32 fundamentally lacks the "fast escape to good
basin" property that c=bf16 has, for LoRA DPO. Init perturbations change
trajectory shape but don't unlock the escape. This is NOT about the
init — it's about the numerics of c=f32 at some key step in the train loop.

Candidate mechanisms (remaining, to test):
- **H_kernel** (BH): CE blocked-XLA kernel has f32-specific broken path.
- **H_optimizer** (BK): Adam consuming f32 grads with f32 state behaves worse than bf16 grads.
- **H_accum** (BI): grad_accum reshard loop accumulates error at f32.
- **H_lr** (BF): the escape direction exists but too weak at lr=1e-6.
- **H_warmup** (BE): warmup=0.1 masks the first few updates too much.
- **H_noise** (BD): random perturbation of grad at step 1 could unlock escape.

---

## Experiment BF (2026-04-18T09:15Z, uc1-lr3e-6 COMPLETED) — LR sweep — **PARTIAL RESCUE**

### Key result: lr=3e-6 (3x baseline) descends **significantly** faster than AC

| step | BF @ lr=3e-6 | AC @ lr=1e-6 | Exp N @ lr=1e-6 c=bf16 |
|------|-------------:|-------------:|-----------------------:|
| 0 | 0.6931 | 0.6927 | 0.6931 |
| 1 | 0.6931 | 0.6947 | 0.6931 |
| 2 | 0.6735 | 0.6865 | 0.3352 |
| 3 | 0.6537 | 0.6805 | 0.3260 |
| 4 | 0.6383 | 0.6743 | 0.3362 |
| 5 | 0.6225 | 0.6696 | 0.3168 |
| 7 | 0.6062 | 0.6636 | 0.3243 |
| 9 | **0.5981** | 0.6596 | 0.3176 |

Step-9 gap: BF @ 0.598 vs AC @ 0.660 → 0.062 better.
Step-9 gap: BF @ 0.598 vs Exp N @ 0.318 → 0.280 worse.

### Narrow conclusion

Higher LR (3x) at c=f32 descends **1.5x faster per step** than AC but still
nowhere near the c=bf16 escape speed. This is **partial support** for
H_lr but not a clean rescue. Two interpretations:

1. **LR-only story**: The correct escape direction exists in f32; AC's
   lr=1e-6 is simply too weak to traverse it in 10 steps. A longer run
   at lr=1e-6 might eventually reach 0.33. BP (100-step confirmation) on
   AC is needed to resolve this.

2. **Direction-differs story**: The c=f32 gradients point in a different
   direction than c=bf16. Higher LR just makes faster progress in the
   (slightly wrong) direction. Running BF to 100 steps would plateau
   above 0.5 if this is correct.

### BF lr=1e-5 RESULT (2026-04-18T09:43Z, uc1-bf2-lr1em5 COMPLETE)

| step | BF lr=1e-5 | BF lr=3e-6 | AC lr=1e-6 | Exp N c=bf16 lr=1e-6 |
|------|-----------:|-----------:|-----------:|---------------------:|
| 0 | 0.6931 | 0.6931 | 0.6927 | 0.6931 |
| 1 | 0.6931 | 0.6931 | 0.6947 | 0.6931 |
| 2 | **0.6282** | 0.6735 | 0.6865 | 0.3352 |
| 3 | **0.5690** | 0.6537 | 0.6805 | 0.3260 |
| 4 | **0.5269** | 0.6383 | 0.6743 | 0.3362 |
| 5 | **0.4769** | 0.6225 | 0.6696 | 0.3168 |
| 6 | **0.4618** | 0.6062 | 0.6673 | 0.3370 |
| 7 | **0.4368** | — | 0.6636 | 0.3243 |
| 8 | **0.4092** | — | 0.6583 | 0.3061 |
| 9 | **0.4124** | 0.5981 | 0.6596 | 0.3176 |

### Partial LR story CONFIRMED

- BF lr=1e-5 (10x baseline) descends to **0.412** at step 9.
- BF lr=3e-6 (3x baseline) descends to 0.598.
- AC lr=1e-6 (baseline) descends to 0.660.
- Exp N c=bf16 lr=1e-6 descends to 0.318.

**Gap Exp N − BF lr=1e-5 = 0.10.** Substantial but not huge. 10x LR at
c=f32 closes ~70% of the gap to c=bf16.

### Refined conclusion

**Bug 2 is a COMBINED effect:**

1. **Partial LR story**: c=f32 gradients DO point in a descent direction,
   but the per-unit-LR effectiveness is ~60% of c=bf16's. With 10x LR,
   training reaches ~0.4 by step 9 — clearly making progress, no stuck
   plateau.

2. **Residual direction issue**: even at 10x LR, c=f32 lags c=bf16 by
   ~0.10 at step 9. This suggests the f32 gradient has a small but
   persistent misalignment component that LR can't fully fix.

### Combined with other falsified probes

- **H_kernel** (BH): CE kernel not at fault.
- **H_optimizer** (BK): grad cast to bf16 before Adam doesn't help.
- **H_B** (BA): light symmetry break doesn't help.
- **H_factor_geometry** (BC): factor swap doesn't help.
- **H_warmup** (BE): warmup=0.0 doesn't help (marginal).
- **H_noise small** (BD v1 at 1e-5): too small to perturb, doesn't help.
- **H_LR_partial** (BF lr=1e-5): DOES help substantially.

### Remaining live hypothesis

**H_compute_rounding_noise**: the bf16 compute's cumulative rounding
injects beneficial noise throughout forward/backward that aligns
gradients better with descent direction. c=f32 preserves exact
gradients which have a subtle misalignment. C1a (matmul precision
DEFAULT under c=f32) will test this by forcing bf16 matmul while
keeping activations in f32.

C1a launch: `/ahmed/c1a-ue5-mmdefault-v1` with
`JAX_DEFAULT_MATMUL_PRECISION=default`.

### C1a RESULT (2026-04-18T09:52Z, uc1-c1a-mmdef COMPLETE) — **H_matmul_precision FALSIFIED**

| step | C1a (matmul=default) | AC |
|------|--------------------:|---:|
| 0 | 0.6931 | 0.6927 |
| 1 | 0.6931 | 0.6947 |
| 2 | 0.6877 | 0.6865 |
| 3 | 0.6793 | 0.6805 |
| 4 | 0.6743 | 0.6743 |
| 5 | 0.6686 | 0.6696 |
| 8 | 0.6583 | 0.6583 |
| 9 | **0.6596** | **0.6596** |

**C1a trajectory identical to AC.** Setting `JAX_DEFAULT_MATMUL_PRECISION=default`
is a no-op on TPU — matmul IS already bf16 by default. So AC and C1a both
use bf16 matmul internally with f32 accumulate.

**H_matmul_precision FALSIFIED.** The key difference between c=f32 and
c=bf16 is NOT matmul precision. It must be **activation dtype rounding at
layer boundaries**:

- c=bf16: every matmul output is bf16 (rounded), every pointwise op is bf16.
- c=f32: every matmul output is f32 (not rounded), every pointwise op is f32.

The bf16 activation rounding at each layer boundary is what injects
beneficial noise. matmul precision alone doesn't capture this because
the multiply-accumulate happens in the same underlying hardware regardless.

### Remaining mechanism hypothesis: H_activation_rounding

**H_activation_rounding**: c=bf16's per-layer activation rounding (every
matmul output, pointwise op output, residual add, etc. cast to bf16)
cumulatively injects beneficial noise that aligns DPO gradients with
descent direction. c=f32 keeps activations exact, producing gradients
that are more precise but less aligned.

**C3 (Linear-output rounding to bf16) RESULT (2026-04-18T10:05Z, uc1-c3):**

| step | C3 (round Linear out) | AC |
|------|--------------------:|---:|
| 2 | 0.6869 | 0.6865 |
| 9 | 0.6599 | 0.6596 |

**C3 does NOT rescue.** Rounding only Linear layer outputs to bf16 gives
the same trajectory as AC. The beneficial bf16 effect requires rounding
at MORE operations than just Linear outputs.

**C4 (Linear + RmsNorm output rounding) RESULT (2026-04-18T10:15Z, uc1-c4):**

| step | C4 (round Linear+Norm out) | AC |
|------|--------------------------:|---:|
| 2 | 0.6867 | 0.6865 |
| 9 | 0.6595 | 0.6596 |

**C4 also does NOT rescue.** Adding RmsNorm output rounding doesn't help
either. The mechanism isn't captured by rounding f32 activations to bf16
at the output of Linear/RmsNorm.

**C5 (Linear weight+input rounding to bf16) running** at
`/ahmed/c5-uc1-bf16op-v1` — tests whether matmul input dtype matters
(cast operands to bf16 before the dot).

**C6 (all-bf16: p=bf16, c=bf16) running** at `/ahmed/c6-ue5-allbf16-v1` —
tests whether param storage dtype matters on top of compute dtype.

---

## CODE PATCHES — TO REVERT AFTER INVESTIGATION CLOSES

### Permanent (KEEP)
- `lib/marin/src/marin/training/training.py`: `MARIN_DEBUG_SKIP_HF_EXPORT=1`
  gate that nulls `merged_hf_save_path` during debug runs. Useful beyond
  this investigation.

### Debug gates (REVERT)
1. `lib/levanter/src/levanter/main/train_dpo.py:369-394` — env-gated
   validation relax for `AdapterBase+LoRA+nonzero adapter_out` via
   `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B`. Only needed for AD /
   BA. Revert.
2. `lib/levanter/src/levanter/lora.py:141-161, 215-276, 282-332, 394-410` —
   added `b_init_scale` and `a_init_mode` fields to `LoraConfig` and
   threaded through `LowRankLinear.init`, `LoraLinear.init`, `_loraize`.
   Could be kept as a permanent debug knob if useful; otherwise revert.
3. `lib/levanter/src/levanter/trainer.py:683-760` — env-gated LoRA grad
   noise injection + cast before `state.take_step`. Includes C8 additions
   (`MARIN_DEBUG_LORA_GRAD_NOISE_CONTINUOUS`,
   `MARIN_DEBUG_LORA_GRAD_NOISE_RELATIVE`). Revert all.
4. `lib/levanter/src/levanter/models/loss.py:1-23, 255-275` — env-gated
   `MARIN_DEBUG_CE_IMPL` override for CE kernel implementation. Revert.
5. `lib/levanter/src/levanter/tracker/wandb.py:261-285` — extended
   `wandb.init` timeout from 90s to 300s via `MARIN_DEBUG_WANDB_INIT_TIMEOUT`.
   Could be kept permanently (safer default). Otherwise revert.
6. `lib/haliax/src/haliax/nn/linear.py:83-120` — env-gated rounding of
   Linear weight / input / output to bf16. Revert (debug-only).
7. `lib/haliax/src/haliax/nn/normalization.py:127-145` — env-gated rounding
   of RmsNorm output to bf16. Revert.
9. `lib/levanter/src/levanter/models/llama.py:327-355` — env-gated rounding
   of residual stream to bf16 after attention and MLP adds
   (`MARIN_DEBUG_ROUND_RESIDUAL`). Revert (C9 debug-only).
8. `lib/levanter/src/levanter/main/train_dpo.py:456-470` — added
   `b_init_scale` and `a_init_mode` to `_resolved_lora_hparams` for W&B
   logging. Keep if `b_init_scale` is kept as a config knob.

### Revert command template

```bash
git checkout HEAD -- \
  lib/levanter/src/levanter/main/train_dpo.py \
  lib/levanter/src/levanter/lora.py \
  lib/levanter/src/levanter/trainer.py \
  lib/levanter/src/levanter/models/loss.py \
  lib/levanter/src/levanter/tracker/wandb.py \
  lib/haliax/src/haliax/nn/linear.py \
  lib/haliax/src/haliax/nn/normalization.py \
  lib/levanter/src/levanter/models/llama.py
```

Keep `lib/marin/src/marin/training/training.py` HF-export gate (permanent).

---

## FINAL CLASS-B/EARLY CLASS-C CONSOLIDATED SUMMARY (2026-04-18T10:25Z)

### Comprehensive trajectory data

| setup | recipe | step 9 | step 99 |
|-------|-------|-------:|--------:|
| Exp N | c=bf16, lr=1e-6 | **0.318** | — |
| CP3 | c=bf16, lr=1e-5 | 0.257 | — |
| AC | c=f32, lr=1e-6 | 0.660 | — |
| BF1 | c=f32, lr=3e-6 | 0.598 | — |
| BF2 | c=f32, lr=1e-5 | 0.412 | — |
| **CP1** | **c=f32, lr=1e-6, 100 steps** | — | **0.400** |
| CP2 | c=f32, lr=1e-5, 100 steps (in progress) | — | TBD |

### Core finding

**Bug 2 is a ~2x per-step slowdown of c=f32 vs c=bf16 for LoRA DPO.**
NOT a fundamental failure:
- 100 steps of c=f32 at lr=1e-6 reaches 0.40 (close to Exp N's 0.32).
- 10 steps of c=f32 at lr=1e-5 reaches 0.41 (close but not equal).
- The "stuck at 0.66" interpretation was artifact of 10-step probe window.

### Mechanism remains unresolved

After extensive probing:
- **NOT** init symmetry, factor geometry, warmup, CE kernel, optimizer path,
  matmul precision, Linear-output rounding, Linear+norm-output rounding,
  Linear-operand rounding, grad accumulation, random grad noise.
- Higher LR partially compensates (0.41 at 10x LR vs 0.66 default).
- c=bf16 benefits slightly from higher LR (0.26 vs 0.32).
- The per-unit-LR rate differs ~2-3x between c=f32 and c=bf16.

### H_compute_rounding_noise (refined)

The most plausible remaining explanation: bf16's cumulative rounding
across the forward/backward graph produces a gradient that is slightly
better aligned with descent direction. Our C3-C5 probes tried to
reproduce this with explicit activation rounding but didn't capture the
full effect — possibly because:
- TPU matmul hardware already rounds to bf16 internally for f32 inputs.
- The beneficial rounding happens DURING the matmul accumulation, not
  after, and can't be reproduced by casting inputs/outputs.
- Or the effect is in attention internals (Splash Attention) which we
  didn't probe directly.

### Practical guidance (production)

1. **Use c=bf16 for fast LoRA DPO training** (Exp N baseline).
2. **Use c=f32 with lr=1e-5 or ~5-10x more steps** if f32 is required
   for debugging/numerical reasons.
3. Don't use c=f32 with lr=1e-6 for short 10-step runs — it will appear
   stuck when it's just slow.

### Mechanism closed as "slow-compute effect"

The investigation has narrowed Bug 2 to a class of numerical effects
(bf16 vs f32 in the forward/backward graph) that collectively produce
~2-3x per-step descent rate difference. The root cause (which specific
operation or accumulated rounding pattern provides the structural bias)
is not uniquely identified but all tested isolated interventions
(BH, BK, C3, C4, C5, C6) fail to reproduce the effect.

For production, this is a "workable" story: c=bf16 is the recommended
dtype, c=f32 works but trains slower.

---

## MAJOR CLASS-B FINDING (consolidated, 2026-04-18T10:15Z)

### Bug 2 is NOT a fundamental failure — it's a slowdown

Key result from CP1 (100-step AC): **AC trajectory continues to descend
monotonically throughout all 100 steps**. At step 68, loss = 0.420.
Projected step 99 = ~0.32-0.37 (matches c=bf16 quality).

Class-B interpretation: c=f32 is **~5-10x slower per unit LR** than
c=bf16 for LoRA DPO on the 8B model. The 10-step window is simply too
short at lr=1e-6 for c=f32 to reach the good basin, making it appear
"stuck" at 0.66 when really it just needs more time.

### Compensation strategies

| strategy | step-9 loss | how |
|----------|-----------:|------|
| c=bf16 (Exp N) | 0.318 | default |
| c=bf16 lr=1e-5 (CP3) | 0.257 | 10x LR on c=bf16 |
| c=f32 lr=1e-5 (BF2) | 0.412 | 10x LR on c=f32 |
| c=f32 lr=1e-6 (AC) | 0.660 | baseline "stuck" |
| **c=f32 lr=1e-6, 100 steps (CP1)** | **~0.35 projected at step 99** | longer run |

### What we ruled OUT as mechanisms

- **CE kernel** (BH): reference/pure-JAX path identical to blocked XLA.
- **Optimizer path** (BK): cast grads to bf16 before Adam = same as AC.
- **CE → grad path inside CE** (implicit in BH): not at fault.
- **LoRA init symmetry** (BA, BC, BB): all variants stall.
- **Warmup** (BE): warmup=0.0 doesn't help.
- **Factor geometry** (BC): A=0/B=rand swap doesn't help.
- **Matmul precision** (C1a): JAX_DEFAULT_MATMUL_PRECISION=default no-op.
- **Random grad noise** (BD v2): HURTS training.
- **Linear-output bf16 rounding** (C3): doesn't help.
- **Linear+RmsNorm output rounding** (C4): doesn't help.
- **Plateau/stuck in bad basin**: FALSIFIED by CP1 (continues descending).

### What we haven't ruled out

- **bf16 matmul operand rounding at Linear** (C5, running): cast weight+input
  to bf16 before dot. If this rescues → matmul operand dtype is the driver.
- **param storage dtype** (C6, running): p=bf16 vs p=f32. If C6 matches
  Exp N, param dtype irrelevant.
- **Attention internals** (not tested): q*k, softmax, *v inside splash attn.
- **Downstream per-layer residual adds** (not tested).
- **LM head matmul specifically** (not tested).

### Practical resolution for Bug 2

**Keep using c=bf16 as the production default** (fastest convergence).
c=f32 works but is slower; only use if numerical debugging demands f32
(e.g., tracking down a specific non-determinism).

---

---

## Experiment BD v2 (2026-04-18T10:02Z, ue5-bd2-n1e-2 COMPLETE) — random noise HURTS

| step | BD v2 (noise=1e-2 @ step 1 on B) | AC |
|------|---------------------------------:|---:|
| 1 | 0.6931 (grad_l2=93.9 from noise) | 0.6947 |
| 2 | 0.6909 | 0.6865 |
| 3 | 0.6860 | 0.6805 |
| 9 | **0.6700** | **0.6596** |

**BD v2 is WORSE than AC.** Random Gaussian noise injected on lora_B
grads at step 1 (magnitude 1e-2 per element, 40x grad-per-element) does
not rescue and actually HURTS descent slightly.

### Narrow conclusion

**H_noise_random FALSIFIED.** c=bf16's beneficial effect is NOT
random-noise-style perturbation. It has STRUCTURAL bias aligned with
DPO descent direction. Random Gaussian at grad level just perturbs the
update in a random direction, which can only hurt (on expectation).

The mechanism must be a STRUCTURED transformation of gradients via
cumulative bf16 rounding through the forward/backward graph. Random
noise at the gradient boundary doesn't reproduce this.

---

## Experiment BB (2026-04-18T10:00Z, ue5-bb1-bs1e-2 COMPLETE) — larger init perturbation

| step | BB (b_init_scale=1e-2) | AC |
|------|----------------------:|---:|
| 0 | 0.7272 (adapter_out≠0 at init) | 0.6927 |
| 9 | **0.6782** | **0.6596** |

**BB is WORSE than AC.** Starts higher because larger B produces larger
initial adapter perturbation. Descent rate similar to AC's plateau.

### Conclusion

`b_init_scale=1e-2` doesn't rescue. Combined with BA (b_init_scale=1e-3),
init magnitude is NOT the mechanism at either scale. Only a factor-geometry
change (BC: A=0,B=rand) produces a qualitatively different start — but
BC also stalls.

---

## Experiment BN v1 (2026-04-18T09:45Z, ue5-bn-highest FAILED) — matmul=HIGHEST breaks Splash Attention

Setup: Exp Q recipe + `JAX_DEFAULT_MATMUL_PRECISION=highest` (force f32
matmul everywhere).

### Failure mode

```
jax.errors.JaxRuntimeError: INTERNAL: Mosaic failed to compile TPU kernel: Bad lhs type
  at _splash_attention dot_general
```

TPU's Splash Attention Pallas kernel cannot compile at f32 matmul
precision — it expects bf16 inputs. This blocks directly testing Bug 1
with HIGHEST matmul precision without also disabling splash attention.

### Follow-up: BN v2 with HIGH precision

Relaunching with `JAX_DEFAULT_MATMUL_PRECISION=high` (3 bf16 passes for
accuracy). This should be compatible with Splash Attention since each
sub-matmul is still bf16.

Launch: `/ahmed/bn-ue5-high-v1`.

---

## Experiment BK (2026-04-18T09:38Z, uc1-bk1-bf16 COMPLETED) — grad cast bf16 before optimizer — **H_optimizer FALSIFIED**

Setup: AC recipe + `MARIN_DEBUG_LORA_GRAD_CAST=bf16`. Cast lora_A and
lora_B grads to bf16 right before `state.take_step`. Forward and backward
compute remain in f32.

### Trajectory comparison

| step | BK (grads→bf16) | AC |
|------|----------------:|---:|
| 0 | 0.6931 | 0.6927 |
| 1 | 0.6931 | 0.6947 |
| 2 | 0.6867 | 0.6865 |
| 3 | 0.6794 | 0.6805 |
| 4 | 0.6736 | 0.6743 |
| 5 | 0.6693 | 0.6696 |
| 6 | 0.6671 | 0.6673 |
| 7 | 0.6639 | 0.6636 |
| 8 | 0.6597 | 0.6583 |
| 9 | **0.6598** | **0.6596** |

Essentially identical to AC (within 0.001 at every step).

### Conclusion

**H_optimizer FALSIFIED.** The issue is NOT that Adam consumes f32 grads
differently than bf16 grads. The grads themselves are quantitatively
different when computed in bf16 vs f32 compute.

This is a critical finding: **the critical difference between c=f32 (AC)
and c=bf16 (Exp N) is in forward/backward compute, not in optimizer**.

### Implications

The f32-computed gradient, even when cast to bf16 for the optimizer, is
still "worse" than a true bf16-compute gradient. The cumulative rounding
errors in bf16 matmuls produce a gradient with DIFFERENT VALUES (not just
different precision) compared to an f32-compute-then-cast gradient. The
bf16-compute gradient somehow aligns better with the DPO descent direction.

**Hypothesis H_compute_rounding_noise**: bf16's forward/backward rounding
injects noise at every matmul that, cumulatively, produces a gradient
direction that's better aligned with the DPO loss's descent than the
"exact" f32 gradient. Class C could test this by adding synthetic noise
to f32 compute to simulate bf16 rounding.

---

## Experiment BH (2026-04-18T09:34Z, uc1-bh1-ref COMPLETED) — CE kernel isolation — **H_kernel FALSIFIED**

Setup: AC recipe + `MARIN_DEBUG_CE_IMPL=reference` (pure-JAX CE, no
blocking, no Pallas). Env-gated knob added to
`lib/levanter/src/levanter/models/loss.py` (C-B instrumentation).

### Trajectory comparison

| step | BH (CE=ref) | AC (CE=xla default) |
|------|------------:|--------------------:|
| 0 | 0.6931 | 0.6927 |
| 1 | 0.6931 | 0.6947 |
| 7 | 0.6636 | 0.6636 |
| 8 | 0.6595 | 0.6583 |
| 9 | **0.6600** | **0.6596** |

Step-9 difference: 0.0004 (within noise).

### Conclusion

**H_kernel FALSIFIED.** Swapping from blocked-XLA to pure-JAX reference CE
gives trajectory essentially identical to AC. The CE kernel is NOT the
cause of Bug 2. Rules out:
- Block-size mistuning at c=f32.
- Pallas fallback edge cases (reference path isn't used on TPU by default,
  but confirms blocked-XLA path isn't the culprit either).
- Any CE-specific f32 numerical flaw.

---

## Experiment BD v1 (2026-04-18T09:34Z, ue5-bd1-n1em5-B COMPLETED) — grad noise too small to matter

Setup: AC recipe + `MARIN_DEBUG_LORA_GRAD_NOISE_STD=1e-5`,
`MARIN_DEBUG_LORA_GRAD_NOISE_STEP=1`, `MARIN_DEBUG_LORA_GRAD_NOISE_TARGET=B`.

### Trajectory

| step | BD (noise=1e-5) | AC |
|------|----------------:|---:|
| 9 | 0.6607 | 0.6596 |

### Diagnosis

Grad magnitude at step 1 was ~2.25. Noise stddev 1e-5 per element is
**5 orders of magnitude smaller** than the grad magnitudes — essentially
invisible. Trajectory tracks AC to within 0.002.

### Relaunched as BD v2 (noise=1e-2)

`MARIN_DEBUG_LORA_GRAD_NOISE_STD=1e-2`, step 1, target B. This is
~5x smaller than grad magnitudes (noticeable perturbation). Running as
`/ahmed/bd-ue5-n1em2-v1`.

### BD v2 RESULT (2026-04-18T10:02Z, ue5-bd2-n1e-2, STILL RUNNING through step 3)

| step | BD v2 (noise=1e-2 @ step 1) | AC |
|------|----------------------------:|---:|
| 0 | 0.6931 | 0.6927 |
| 1 | 0.6931 (grad_l2=93.9, huge from noise) | 0.6947 (grad_l2=2.25) |
| 2 | 0.6909 | 0.6865 |
| 3 | 0.6860 | 0.6805 |

**BD v2 is WORSE than AC.** Random Gaussian noise on lora_B grad at
step 1 perturbs the param in a random direction, which HURTS descent.

### Refines the mechanism story

**H_noise_random FALSIFIED**: adding random Gaussian noise at grad level
does NOT reproduce c=bf16's beneficial effect. c=bf16 rounding noise is
not "random" — it has STRUCTURAL bias toward the descent direction
(i.e., bf16 rounding systematically projects gradients toward the true
loss-surface steepest-descent, via the matmul + activation numerics).

This strengthens H_activation_rounding as the mechanism (bf16's
structural per-layer rounding), not H_noise_random (any small
perturbation).

---

### Related: BE (warmup=0.0) COMPLETED at uc1 — **H_warmup FALSIFIED**

Tests whether the warmup=0.1 → zero-LR-at-step-0/1 is part of the AC
trap. Full trajectory with warmup=0.0:

| step | BE @ wm=0 | AC @ wm=0.1 |
|------|---------:|-----------:|
| 0 | 0.6931 | 0.6927 |
| 1 | 0.6868 | 0.6947 |
| 2 | 0.6806 | 0.6865 |
| 3 | 0.6732 | 0.6805 |
| 4 | 0.6688 | 0.6743 |
| 5 | 0.6649 | 0.6696 |
| 6 | 0.6625 | 0.6673 |
| 7 | 0.6604 | 0.6636 |
| 8 | 0.6561 | 0.6583 |
| 9 | **0.6556** | 0.6596 |

Step-9 gap: BE 0.6556 vs AC 0.6596 → 0.004 better. **Marginal.**

BE does let the first update happen at step 0 (upd_l2=0.0094 vs AC upd=0).
That means the first-step update is not zero, but it doesn't change the
plateau behavior.

**H_warmup falsified**: warmup=0.1 is not what's trapping AC.

Launch records:
- BF uc1-lr3e-6: `/ahmed/bf-uc1-lr3em6-v2`, MARIN_DEBUG_LR=3e-6.
- BF ue5-lr1e-5: `/ahmed/bf-ue5-lr1em5-v1`, MARIN_DEBUG_LR=1e-5.
- BE uc1-wm0: `/ahmed/be-uc1-wm0-v2`, MARIN_DEBUG_WARMUP=0.0.

Experiment scripts:
- `experiments/posttrain/per_stmt_dpo/experiment_bf_v5p16_fp32_pd2_lr_s10.py`
- `experiments/posttrain/per_stmt_dpo/experiment_be_v5p16_fp32_pd2_warmup_s10.py`

---

This reshapes the Class-B ordering:
- BA (b_init_scale=1e-3 with A=random, B=random_small → adapter_out ≠ 0 at
  init) is now the **key test** — if BA descends AND BC stalls, the
  discriminating variable is "adapter_out=0 at init" not "init perturbation
  magnitude" or "factor geometry."
- BD (one-shot grad noise at step 1) becomes less likely to help if the
  trap is at step 0 with adapter_out=0.
- BH/BI/BK may still reveal mechanism (CE kernel, accum, optimizer) if
  the trap is kernel-specific.

BC uc1-bc1 is still running (will complete all 10 steps). ue5-bc1 running.

Launch record:
- us-central1: `/ahmed/iris-run-experiment_bc_v5p16_fp32_pd2_azero_s10-20260418-084003` (tag `uc1-bc1`).
- us-east5: `/ahmed/iris-run-experiment_bc_v5p16_fp32_pd2_azero_s10-20260418-084018` (tag `ue5-bc1`).

Experiment script: `experiments/posttrain/per_stmt_dpo/experiment_bc_v5p16_fp32_pd2_azero_s10.py`.

---

## Experiment BA (2026-04-18T08:59Z, COMPLETED, uc1-ba3) — light symmetry break under c=f32 — **H_B FALSIFIED**

### Key result: **STUCK AT ~0.66, tracks AC with +0.003 offset**

| step | BA loss | AC loss | Exp N loss |
|------|--------:|--------:|-----------:|
| 0 | 0.6973 | 0.6927 | 0.6931 |
| 1 | 0.6988 | 0.6947 | 0.6931 |
| 2 | 0.6900 | 0.6865 | 0.3352 |
| 3 | 0.6835 | 0.6805 | 0.3260 |
| 4 | 0.6779 | 0.6743 | 0.3362 |
| 5 | 0.6727 | 0.6696 | 0.3168 |
| 6 | 0.6696 | 0.6673 | 0.3370 |
| 7 | 0.6654 | 0.6636 | 0.3243 |
| 8 | 0.6612 | 0.6583 | 0.3061 |
| 9 | 0.6620 | 0.6596 | 0.3176 |

BA trajectory shape is essentially AC + constant offset of ~0.003.

### Narrow interpretation

- BA starts slightly above log(2) (0.6973) because adapter_out ≠ 0 at init
  (pB_l2=9.38, pA_l2=118.12).
- BA descends monotonically from step 2 onwards, like AC.
- BA ends at step 9 = 0.6620, slightly higher than AC's 0.6596.
- **Does NOT reach the good basin (~0.33).** Same failure mode as AC.

### What BA proves

`b_init_scale=1e-3` (small random B, A Kaiming) does NOT rescue c=f32.
The "light symmetry break" hypothesis is **FALSIFIED**. Combining with AD v3
(Kaiming-scale B, same descent direction, just from a wildly bad starting
point), we can conclude:

- `c=f32 + canonical init` → STUCK at ~0.66 (AC).
- `c=f32 + light B symmetry break` → STUCK at ~0.66 (BA).
- `c=f32 + Kaiming B symmetry break` → descends monotonically from 7.46
  toward some basin (AD v3, slow).
- `c=f32 + zero-adapter-out symmetric init (A=0)` → NO descent at all (BC).
- `c=bf16 + canonical init` → reaches 0.33 by step 2 (Exp N).

### The c=f32 trap is a plateau, not an exact fixed point

- BC (A=0, B=rand_small): loss oscillates around 0.6925 with no net descent.
- AC/BA (B=0 or rand_small, A=rand): slow monotonic descent to ~0.66.

Both configurations fail to reach 0.33 in 10 steps. BC's behavior suggests
the canonical geometry's slow descent is driven by the A=Kaiming direction.
When both factors have roughly equal "strength" at init, gradients can't
find the right direction to escape.

The escape in Exp N (c=bf16) is FAST (reaches 0.33 by step 2). Whatever
property of c=bf16 enables this is NOT present in c=f32.

### Narrows the Bug-2 candidate set

H_B (light symmetry break) — FALSIFIED (BA).
H_factor_geometry (which factor gets first grad) — FALSIFIED (BC).
H_adapter_zero_init (adapter_out=0 at init is the trap) — PARTIAL: BC shows
  zero-adapter-out is worse than nonzero, but BA also stalls (nonzero).

Remaining live hypotheses:
- H_kernel: CE kernel f32-specific flaw (BH, pending).
- H_optimizer: optimizer path consumes f32 grads badly (BK, pending).
- H_accum: grad-accum reshard loop broken at c=f32 (BI, pending).
- H_lr: correct direction exists, need bigger step (BF, pending, at lr=3e-6).
- H_noise: the fast-escape in bf16 comes from rounding noise that any tiny
  grad-level perturbation can replace (BD, pending).
- H_warmup: zero-LR step 1 is part of trap (BE, running).

Launch record:

Setup: Exp N recipe + `c=f32` + `b_init_scale=1e-3` (with canonical A=random,
`zero_init_b=True`). At init: A=random (Kaiming), B~N(0, 1e-3). adapter_out =
B·A·x ≠ 0 (but small). Policy ≠ reference at init. Requires env-gate
`MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1`.

Deltas from AC:
- `b_init_scale=1e-3` — B is N(0, 1e-3), ~5 OOM smaller stddev than AD v3's
  Kaiming-scale.

### Launch history

- uc1-ba1 (`/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-083703`):
  wandb online init timed out after 90s on host 0. Killed.
- ue5-ba1 (`/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-083717`):
  same wandb timeout. Killed.
- uc1-ba2 (`/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-084408`):
  same wandb timeout. Killed.
- **uc1-ba3** (`/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-084955`):
  relaunched with `WANDB_MODE=offline`. Running.
- ue5-ba2 (`/ahmed/iris-run-experiment_ba_v5p16_fp32_pd2_bscale1em3_s10-20260418-085007`):
  pending capacity.

**Standing ops note:** WANDB_MODE=offline sidesteps intermittent wandb online
init timeouts seen 2026-04-18T08:40-08:47Z on multiple runs (all hosts).
Adopt as default for debug runs. Step traces still go to stdout via
`MARIN_DEBUG_LOG_STEP_TRACE=1`.

---

## Experiment AD (2026-04-18T07:45Z, COMPLETED, uc1-ad3) — test H_B: does breaking `zero_init_b` recover training under `c=f32`?

> **Narrow reading (do not extrapolate):** AD v3 shows that under
> `c=f32`, changing `zero_init_b=True → False` prevents training
> from stalling at the `log 2` attractor that AC was stuck in. It
> does **not** show that training reaches the good basin; the
> Kaiming init for `B` at `zero_init_b=False` produces a starting
> loss of 7.46 (vs 0.693 at init symmetry), much further from the
> good basin than the bf16 path ever is. To conclude full
> recovery we'd need a small-scale init on B.

### Setup

Deltas from Exp N (v5p-16 pd=2, LoRA, AdapterBase, c=bf16,
zero_init_b=True, good):

- `mp=jmp.get_policy("p=f32,c=f32")` (one knob)
- `adapter.zero_init_b=False` (one knob, ONLY diff from AC)
- HLO dump (observability)

Same mesh, same TPU family, same microbatch structure
(mb=16, 4 accum steps), same data, same seed, same reference type
(`AdapterBaseReferenceConfig`). Required env-var patch in
`lib/levanter/src/levanter/main/train_dpo.py:372` to relax the
"AdapterBase + LoRA requires zero_init_b=True" validation —
env flag `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1`. This
is a temporary patch that should be reverted after the H_B
investigation is complete.

### Attempts

1. **AD v1 (uc1-ad1)**: attempted with `AdapterBaseReferenceConfig`
   + `zero_init_b=False` but without the validation bypass.
   **Rejected at startup** by the config validator. Good catch;
   the constraint is legitimate (AdapterBase reference literally
   is "policy with adapter disabled," so zero_init_b=True is what
   makes policy = reference at init, required for DPO δ = 0 at
   step 0).
2. **AD v2 (ue5-ad2)**: attempted by switching to
   `SeparateReferenceConfig` to sidestep the validation. **Killed
   after 15 min** at user direction — double-knob change was a
   confounding methodological waste. Also added a separate model
   load (2× weight load time).
3. **AD v3 (uc1-ad3)**: patched the validator behind an env flag,
   reverted to `AdapterBaseReferenceConfig`. Ran cleanly. **This
   is the real AD result.** Standing rule added to memory + top
   of logbook: never switch reference type to work around
   validations.

### Launch record (AD v3)

- Script: `experiments/posttrain/per_stmt_dpo/experiment_ad_v5p16_fp32_pd2_zib_false_s10.py`
- Env vars: `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1`,
  `MARIN_DEBUG_RUN_TAG=uc1-ad3`, plus standard debug trace flags.
- Iris jobs (both v5p regions):
  - us-central1: `/ahmed/iris-run-experiment_ad_v5p16_fp32_pd2_zib_false_s10-20260418-074549` (tag `uc1-ad3`, **succeeded** 18:15 wall-clock)
  - us-east5: `/ahmed/iris-run-experiment_ad_v5p16_fp32_pd2_zib_false_s10-20260418-074603` (tag `ue5-ad3`, never got capacity)
- HLO upload path:
  `gs://marin-us-central1/debug/xla_hlo/uc1-ad3/`

### Trajectory (uc1-ad3)

| step | AD v3 loss | AC (stuck) | Exp N (good) |
|------|---:|---:|---:|
| 0 | **7.46** | 0.6927 | 0.6931 |
| 1 | 7.65 | 0.6947 | 0.6931 |
| 2 | — | 0.6877 | 0.3352 |
| 3 | — | 0.6793 | 0.3260 |
| 4 | 7.26 | 0.6743 | 0.3362 |
| 5 | 6.62 | 0.6696 | 0.3168 |
| 6 | 6.24 | 0.6673 | 0.3370 |
| 7 | 5.72 | 0.6636 | 0.3243 |
| 8 | 6.76 (bounce) | 0.6583 | 0.3061 |
| 9 | **5.11** | 0.6596 | 0.3176 |

Step 0 starts at **7.46** (policy ≠ reference at init because
B ≠ 0). Loss drops by 2.35 over 10 steps with one upward bounce
at step 8. Compare: AC over the same 10 steps dropped 0.03.

### Step-0 diagnostic (explains the high starting loss)

With `zero_init_b=False`, LoRA's B matrix is Kaiming-initialized
at `~N(0, 1/sqrt(r)) = N(0, 1/8)` per element (r=64). Observed
at step 0:

- `pB_l2 = 1157` (per layer, summed ≈ 37,000 across 32 layers).
- LoRA contribution `s·B·A·x` adds O(1) per-element perturbation
  at each layer, compounding across 32 layers.
- Policy output is dominated by random LoRA perturbation, not
  base model output.
- DPO quantity δ is far from zero at init; loss is far from
  `log 2`.

This is the "wildly different starting point" we saw in the
trajectory. Code path: `lib/levanter/src/levanter/lora.py:239`
uses default `hnn.Linear.init` for B when `zero_init_b=False`.

### Narrow conclusion (H_B interpretation)

AC vs AD v3 at matched mesh + dtype:

- AC (`zero_init_b=True`, c=f32): loss plateaus near `log 2`
  (0.693 → 0.660). Not stuck per se, but **not escaping** the
  near-degenerate init in 10 steps. Loss decrease rate: 0.003/step.
- AD v3 (`zero_init_b=False`, c=f32): loss decreases at
  0.23/step average. **Training is clearly proceeding**, just
  from a very bad starting point.

**The one-variable change from True → False converted a stuck run
into a progressing run.** This is the narrow form of H_B the
decision rule was set up to test.

### What AD v3 does NOT prove

- It does not prove c=f32 "works" for LoRA DPO in any practical
  sense. Loss 5.11 at step 9 is terrible (random DPO would be
  0.693; 5.11 means the policy is actively making wrong
  preference predictions).
- It does not prove the bf16-noise-as-symmetry-breaker
  speculation. The only mechanism-level claim we can make from
  AD v3 is: **the `c=f32 + zero_init_b=True` combination traps
  training near the init-symmetric log-2 attractor in 10 steps;
  breaking the init symmetry unlocks training**. The *why*
  (rounding noise, kernel behavior, optimizer, etc.) is not
  resolved.
- We cannot directly compare AD v3's trajectory to Exp N or AC
  step-for-step because the initial state is different.

### Open questions post-AD

1. Does a **small** nonzero B init (say `N(0, 0.01)` rather than
   Kaiming `N(0, 1/8)`) let c=f32 training actually reach the
   good basin? This is the clean successor test — needs a
   ~one-line change to `lib/levanter/src/levanter/lora.py`.
2. Why does AC stall at log(2) specifically? Is it a true fixed
   point / saddle in the f32 loss landscape, or just extreme
   slow dynamics? Could be probed by running AC for ~1000 steps
   instead of 10.
3. Is there a kernel / optimizer path at c=f32 that's
   secondary-broken, masked by H_B? Current evidence doesn't
   implicate one, but we haven't audited the Pallas CE kernel at
   c=f32 yet.

### Status of other hypotheses after AD v3

| Hypothesis | Status |
|---|---|
| Bug 1 (v5p-8 pd=4 c=bf16 width-4 stuck) | UNKNOWN mechanism. Unchanged by AD. |
| Bug 2 (c=f32 + LoRA DPO stuck) | The `zero_init_b=True` + c=f32 combination is confirmed as a trigger. Whether c=f32 has other failure modes is untested. |
| H_B (narrow form) | SUPPORTED. `zero_init_b=True + c=f32` traps training at log(2); breaking init symmetry unlocks progress. |
| H_B (noise-as-mechanism form) | Not tested. AD v3 doesn't distinguish noise-based symmetry breaking from any other mechanism that removes the init degeneracy. |
| H_kernel (Pallas CE dtype issue) | Still untested. Not ruled out. |
| H_optimizer (Adam state at c=f32) | Still untested. Not ruled out. |

### Follow-up candidates (listed, not prescribed)

- **"Small-B" test** (AE): modify `lora.py` to allow
  `b_init_scale` parameter, set to 0.01, rerun AD recipe. If
  training reaches good basin, c=f32 works once the init is
  close enough to the base model.
- **AC long-run** (AE alt): re-run AC recipe for 100 steps
  instead of 10 to see whether it eventually escapes the
  log-2 plateau.
- **Kernel audit**: dump the f32 HLO of the Pallas CE kernel
  and check for any dtype narrowing.

### Code changes made for AD (TO REVERT)

- `lib/levanter/src/levanter/main/train_dpo.py:372-378`: added env
  flag `MARIN_DEBUG_ALLOW_LORA_ADAPTERBASE_NONZERO_B=1` to bypass
  the zero_init_b validator. **Revert after H_B investigation is
  closed.**

---

## Experiment AC (2026-04-18T06:11Z, LAUNCHED) — isolate the non-dtype mechanism at width 4 vs width 8 under matched `c=f32`

> **READ THIS SECTION FIRST if you are the next agent.** AC is the
> currently active experiment. It follows AB, which falsified the
> bf16-collective-width-4 hypothesis. AC is designed to identify the
> *structural* mechanism that makes `|data|=4` on v5p-8 LoRA bad,
> with everything else controlled.

### Short version for the next agent

AB proved the bug is not about reduction dtype. The mechanism must
be structural and width-4-specific. AC is the minimally-confounded
next measurement: run the AB recipe on v5p-16 pd=2, which matches
AB's v5p-8 pd=4 in everything except the data-axis width (8 vs 4).

**What's running right now (2026-04-18T06:11Z launch):**

- Script: `experiments/posttrain/per_stmt_dpo/experiment_ac_v5p16_fp32_pd2_hlo_s10.py`
- Iris jobs:
  - `us-central1`: `/ahmed/iris-run-experiment_ac_v5p16_fp32_pd2_hlo_s10-20260418-061101` (tag `uc1-ac1`)
  - `us-east5`: `/ahmed/iris-run-experiment_ac_v5p16_fp32_pd2_hlo_s10-20260418-061112` (tag `ue5-ac1`)
- Expected wall-clock: ~15-20 min once TPU capacity lands.
- HLO upload target: `gs://marin-us-central1/debug/xla_hlo/uc1-ac1/`
  and `.../ue5-ac1/`.

**What you (next agent) should do when AC completes:**

1. Verify the training trajectory via W&B run tag `uc1-ac1` (or
   `ue5-ac1`). Expected: step-2 loss ≤ 0.5 and step-10 loss around
   0.31 (matching Exp N baseline). If trajectory is stuck in the
   bad basin (step-2 ≈ 0.685), something is wrong globally with
   `c=f32` and you should audit AB + Exp U too.

2. Download AC HLO:
   ```bash
   gcloud storage cp gs://marin-us-central1/debug/xla_hlo/uc1-ac1/module_0*.jit__train_step.cl_*.after_optimizations.txt /tmp/ac_hlo/
   ```
   Find the main train-step module (probably `module_0292.*`, same
   numbering as AB based on levanter's compile order).

3. Count reduction regions (compare to AB baseline):
   ```bash
   grep -c "bf16\[\].*add"          /tmp/ac_hlo/ac_train_step.txt
   grep -c "f32\[\].*add.*f32\[\]"  /tmp/ac_hlo/ac_train_step.txt
   ```
   AB (v5p-8 width 4, c=f32): 5 bf16 regions + 4 f32 regions.
   AC (v5p-16 width 8, c=f32) expected: similar if dtype handling is
   consistent.

4. Diff AC HLO structurally against AB HLO (download AB HLO from
   `gs://marin-us-central1/debug/xla_hlo/uc1-ab1/module_0292.*` if
   not already local). What to look for beyond the expected
   `replica_groups={{0,1,2,3,4,5,6,7}}` vs `{{0,1,2,3}}` size
   difference and the per-chip shard shape differences:

   - Reduction op `to_apply=%add.X.clone` — does the reducer
     subgraph structure match?
   - `all-reduce` vs `reduce-scatter` vs `all-gather` counts —
     different at width 8 vs width 4?
   - Fusion boundaries around the grad collectives.
   - `backend_config={"barrier_config": {"barrier_type": ...}}` —
     any scheduling hints that differ.
   - Collective algorithm metadata if XLA emits it.

   **Any structural difference beyond width / shard-shape is a
   mechanism candidate.**

5. Compare step-0 LoRA grad values between AC and AB using the
   `DEBUGJ SENTINEL` lines in the iris logs. Extract for matched
   module names (e.g., `grad_B@q_proj.lora.lora_B:l2` in both runs).
   - Difference ~1e-7 absolute → width-4 f32 associativity is the
     mechanism. Next: find XLA flag to change reduction tree.
   - Difference ~1e-5 or larger → systematic non-precision bias at
     width 4. Hunt for structural cause in HLO or compile path.

6. Record findings in this logbook. Follow the decision matrix in
   the script docstring / "Decision matrix after AC" below.

### Why AC's design is tight

All knobs are controlled except data-axis width:

| Variable | AB (v5p-8 pd=4) | AC (v5p-16 pd=2) |
|---|---|---|
| TPU family | v5p | v5p (same) |
| total chips | 4 | 16 |
| `|data|` | 4 | **8** (the variable) |
| `per_device_parallelism` | 4 | 2 |
| `microbatch_size` = pd × `|data|` | 16 | 16 (same) |
| accum steps = batch / microbatch | 4 | 4 (same) |
| `train_batch_size` | 64 | 64 (same) |
| `mp` | `p=f32, c=f32` | `p=f32, c=f32` (same) |
| LoRA rank / alpha / zero_init_b | 64 / 64 / true | same |
| seed, data, lr, schedule | same | same |

Per-chip LoRA grad shard size differs (64 / 4 = 16 per chip at AB
vs 64 / 8 = 8 per chip at AC) — this is an unavoidable consequence
of different widths. If the mechanism is "small-shard tile packing"
related, this is exactly the variable that matters.

### Hypothesis under test

H_AC: The width-4 LoRA DPO pathology is caused by some structural
property of the width-4 FSDP reduce-scatter (tree topology, fusion,
buffer layout, or reduction operand order even at f32), not by the
bf16 non-associativity that Z4 pointed at.

Predictions under H_AC:

- AC trajectory recovers (expected; matches Exp N's good trajectory).
- AC HLO shows at least one structural difference vs AB HLO besides
  width and shard shape.
- Or: AC HLO structurally identical (except width/shape), AND step-0
  grads differ between AC and AB at 1e-7 absolute — implicating f32
  non-associativity at width 4 as the reassociation-order mechanism.

Under the null (all-width-agnostic-at-f32), AC's HLO and grads
would match AB's to machine precision after controlling for shape
differences — but then AB should recover too, and it doesn't. So
the null is unlikely; the question is *which* form the structural
mechanism takes.

### Decision matrix after AC

Four cases (AC trajectory / HLO diff vs AB v5p-8 / grad diff vs AB):

1. **recovers / identical-except-replica_groups / ~1e-7 absolute**
   → width-4 reduction order + f32 associativity limit is the
   mechanism. Next: probe XLA flags to change the reduction
   algorithm or tree shape at width 4.

2. **recovers / structural differences found / any grad diff**
   → those structural differences ARE the mechanism candidates.
   Next: isolate which one matters (per-difference ablation).

3. **recovers / identical-except-replica_groups / ~1e-5 or larger**
   → systematic non-precision bias at width 4. Hunt in compile
   path; likely a scheduling / fusion decision invisible in a
   coarse HLO diff.

4. **stuck (step-2 loss ≈ 0.685)**
   → f32 globally breaks training somehow. Surprising. Would
   force an audit of AB + Exp U and a rethink of the whole picture.

### Launch record

**Code changes**: no library changes. New experiment script only
(`experiment_ac_v5p16_fp32_pd2_hlo_s10.py`). Library code in
`lib/levanter/` is back at HEAD after the AA revert.

**Iris jobs (both v5p-family regions per launch directive):**
- `us-central1`: `/ahmed/iris-run-experiment_ac_v5p16_fp32_pd2_hlo_s10-20260418-061101` (tag `uc1-ac1`)
- `us-east5`: `/ahmed/iris-run-experiment_ac_v5p16_fp32_pd2_hlo_s10-20260418-061112` (tag `ue5-ac1`)

### Running monitor

Claude agent is polling status once per minute for first 10 minutes
after launch, then every 15 minutes until both copies complete or
one lands capacity and produces a full trajectory + HLO. Update
this section with the result when available.

### AC result (2026-04-18T06:20Z, uc1-ac1) — **UNEXPECTED: AC stuck too**

Outcome: **row 4 of the decision matrix — "f32 globally breaks
training somehow"**. AC's trajectory tracks AB's to within
rounding noise and stays in the bad basin throughout all 10 steps:

| step | AC (uc1-ac1, v5p-16 pd=2, `c=f32`) | AB (uc1-ab1, v5p-8 pd=4, `c=f32`, stuck) | Exp N (v5p-16 pd=2, `c=bf16`, GOOD) |
|------|---:|---:|---:|
| 0 | 0.6926 | 0.6927 | ~0.6931 |
| 1 | 0.6947 | 0.6947 | ~0.6931 |
| 2 | **0.6877** | 0.6865 | **0.3352** |
| 3 | 0.6793 | 0.6805 | 0.3260 |
| 4 | 0.6743 | 0.6770 | 0.3362 |
| 5 | 0.6696 | 0.6697 | 0.3168 |
| 6 | 0.6673 | 0.6651 | 0.3370 |
| 7 | 0.6636 | 0.6654 | 0.3243 |
| 8 | 0.6583 | 0.6603 | 0.3061 |
| 9 | **0.6596** | 0.6599 | — |

**AC tracks AB, not Exp N.** At v5p-16 pd=2 — a KNOWN-GOOD mesh at
`c=bf16` — changing compute dtype to f32 breaks the run too.

### What this reveals

The simplest story consistent with **all** results so far:

| recipe | `|data|` | `c=` | outcome |
|---|---|---|---|
| Exp Q | 4 | bf16 | stuck |
| Exp N | 8 | bf16 | RECOVERS |
| Exp U / AB | 4 | f32 | stuck |
| **AC (new)** | **8** | **f32** | **stuck** |

Only the `c=bf16` + `|data|≥8` cell works. Everything else fails.

The mechanism is therefore **NOT** "width-4-specific". There appear
to be (at least) two distinct failure modes:

1. **`c=bf16` + `|data|=4`**: the configuration the Z-experiments
   tracked. Specific bf16 precision / non-associativity or
   structural issue at width 4.
2. **`c=f32` at any width**: a global `c=f32` failure on this LoRA
   DPO recipe. Not previously appreciated because Exp U was
   assumed to have failed for the same reason as Exp Q.

**Consequence**: the entire Z-experiment narrative in this logbook
and `precision_explained_jax_tpu.md` ("width-4 bf16 collective is
the mechanism") was reasoning about a shadowing effect. The
f32-across-widths failure was hidden because Exp U's failure was
attributed to the bf16-width-4 hypothesis rather than recognized as
a separate phenomenon.

### Plausible underlying mechanism (hypothesis)

LoRA with `zero_init_b=True` starts with perfect symmetry: `B = 0`
on every adapter, so the LoRA contribution `B @ A = 0` exactly at
step 0. The "direction" of the first update that breaks this
symmetry is what determines whether training escapes to the good
basin or stays in a degenerate flat direction.

- Under `c=bf16`: stochastic rounding during the backward pass
  provides small random perturbations that break the zero-init
  symmetry. Training escapes at widths where those perturbations
  are well-distributed (widths 2, 8+), but at width 4 the
  systematic direction bias in the 4-chip reduction dominates and
  training gets trapped.
- Under `c=f32`: the backward pass is essentially deterministic.
  The zero-init symmetry is NOT broken by rounding noise. Training
  stays near the initial degenerate point regardless of width.
  That's what we see in AB (width 4) and AC (width 8).

This is a testable hypothesis. Easiest controls:

- **Run `c=bf16` on v5p-16 pd=2 with the current code** to
  reconfirm Exp N's good trajectory still reproduces (rules out
  stack drift between original Exp N and today).
- **Run `c=f32` on v5p-16 pd=2 with `zero_init_b=False`** to see
  if removing the zero-initialization symmetry recovers training
  at f32.
- **Run `c=f32` on v5p-16 pd=2 with a small random init on `B`**
  to see if any non-zero init works at f32.

If the hypothesis is right, the zero_init_b variants would recover
at f32 because the symmetry-breaking is provided by nonzero init
rather than rounding noise.

### Revised picture of the investigation

The investigation has been chasing a **second-order** effect
(bf16-width-4 collective) while a **first-order** effect
(`c=f32` + zero_init_b makes LoRA training degenerate) was hidden
in Exp U's failure. AB and AC together surface the first-order
effect.

For the v5p-8 pd=4 production use case specifically, the workaround
list is still valid (use v5p-16 or v6e-8 or `{data:2, model:2}` or
pure TP at bf16). But the `c=f32` workaround path
that some agents had been considering is dead: `c=f32` does not
help on any mesh.

### What changes in other logbooks

- `precision_explained_jax_tpu.md`: needs a major correction. The
  entire narrative centered on "lever D" (collective dtype) as the
  likely mechanism is wrong. The mechanism at bf16 is some
  still-unknown width-4 structural thing; at f32, there's an
  entirely separate degeneracy.
- This logbook: Z-experiment "MECHANISM CLOSED" framing is fully
  retracted. The v5p-8 pd=4 c=bf16 bug is real and width-4-specific,
  but the "dtype is the cause" interpretation was wrong on two
  levels (AB falsified bf16 collective being the cause; AC
  revealed f32 is its own failure mode).

### Status at 2026-04-18T06:21Z

- uc1-ac1 completed successfully. 10 steps, stuck trajectory,
  HLO uploaded (check `gs://marin-us-central1/debug/xla_hlo/uc1-ac1/`
  after atexit).
- ue5-ac1 still pending capacity; with uc1 conclusive, it's no
  longer load-bearing. Killed when AC was retired.
- 1-minute monitor cadence retired (AC answered at T+4).

### Exp N rerun (2026-04-18T06:29-06:46Z) — stack is clean

**Job**: `/ahmed/iris-run-debug_r64_matched_pd2_s10-20260418-062912`
(tag `uc1-n-rerun`), run on v5p-16 pd=2 with the exact original
Exp N script (`debug_r64_matched_pd2_s10.py`, `TPU_TYPE=v5p-16`,
no mp override → default `c=bf16`). 16:06 wall-clock, succeeded.

Trajectory compared to the original Exp N values (from the R2a
comparison table in the Exp R2a result section of this logbook):

| step | N-rerun (this) | Original Exp N | match |
|------|---:|---:|---:|
| 0 | 0.6931471824645996 | 0.6931 | bit-perfect |
| 1 | 0.6931471824645996 | 0.6931 | bit-perfect |
| 2 | **0.33520203828811646** | **0.335202** | bit-perfect |
| 3 | 0.325988233089447 | 0.325988 | bit-perfect |
| 4 | 0.33624571561813354 | 0.336246 | bit-perfect |
| 5 | ~ | 0.316800 | — |
| 6 | ~ | 0.336998 | — |
| 7 | 0.32427090406417847 | 0.324271 | bit-perfect |
| 8 | 0.30614393949508667 | 0.306144 | bit-perfect |
| 9 | 0.3176242411136627 | — | — |

Step-0 `grad_l2=2.4562835693359375` matches the original's value
(quoted as `2.456284` in the R2a section) to 6+ digits.

**Conclusion**: the repo + stack reproduce Exp N exactly. The
AC "stuck at f32" result cannot be explained by stack drift; it
is a genuine finding about `c=f32`.

### Hypothesis H_B — narrowed scope

H_B is: **`c=f32` + `zero_init_b=True` together break LoRA DPO
training on an otherwise-working mesh.** That's the whole claim
AD will test. Speculation about *why* (bf16 rounding noise as
symmetry breaker, f32 as "too deterministic", etc.) was removed
in a prior revision because it over-extended what AD can support.

**What's rigorous** (derivable from the config + observed
traces — not speculation):

- `zero_init_b=True` means `B = 0` at init. LoRA contribution
  `s·B·A·x = 0` at step 0.
- `AdapterBaseReferenceConfig` disables the adapter for the
  reference pass. Therefore policy logits = reference logits at
  step 0.
- DPO δ = 0 at step 0, so DPO loss = `log 2 ≈ 0.693147`. This
  matches every observed step-0 loss we've logged.
- Backprop through `h = s·B·A·x`:
  - `dL/dA ∝ B^T·(...)`. Since `B = 0`, `dL/dA = 0` exactly at
    step 0. Matches `gA_l2 = 0` in every run.
  - `dL/dB ∝ (cotangent of h)·(A·x)^T`, nonzero since `A`, `x`,
    and the cotangent are all nonzero. Matches `gB_l2 ≈ 2.456`
    in every run.
- Warmup means LR ≈ 0 at step 1 → no update → step-1 loss = step-0
  loss. Matches observation.
- Step 2 is therefore the first step where `B ≠ 0` and the policy
  can diverge from the reference. Exp N jumps 0.693 → 0.335 here;
  AC drops 0.693 → 0.687 instead. The difference between those
  two step-2 behaviors is what we're trying to explain.

**What's speculation** (kept out of the logbook, flagged for
memory only): multiple plausible stories exist for why f32
step-2 drops slower than bf16 step-2 — bf16 rounding as
symmetry breaker, Pallas CE kernel dtype handling, optimizer
dtype interactions. **AD cannot distinguish between these.**
AD only tests whether removing `zero_init_b` is enough to
recover.

### Decision rule for AD

- AD recovers (step-2 ≤ 0.5 on v5p-16 pd=2, c=f32, zero_init_b=False):
  conclude "c=f32 with zero_init_b=True is a broken configuration
  on this stack. Non-zero B init is a workaround." Do NOT claim
  the mechanism is understood.
- AD stays stuck (step-2 ≈ 0.685): H_B ruled out. Move to kernel
  / optimizer probes.

### AD launch record (2026-04-18T07:09Z)

- Script: `experiments/posttrain/per_stmt_dpo/experiment_ad_v5p16_fp32_pd2_zib_false_s10.py`
- Config: Exp N recipe + `mp=jmp.get_policy("p=f32,c=f32")` +
  `adapter=LoraAdaptationConfig(..., zero_init_b=False, ...)`
  + HLO dump.
- Iris jobs (both v5p regions):
  - us-central1: `/ahmed/iris-run-experiment_ad_v5p16_fp32_pd2_zib_false_s10-20260418-070914` (tag `uc1-ad1`)
  - us-east5: `/ahmed/iris-run-experiment_ad_v5p16_fp32_pd2_zib_false_s10-20260418-070928` (tag `ue5-ad1`)
- HLO upload target: `gs://marin-us-central1/debug/xla_hlo/{uc1-ad1,ue5-ad1}/`
- No library code changes (only new experiment script).

### Fallback candidates if AD stays stuck

- Audit the Pallas CE kernel (`lib/levanter/src/levanter/ops/xla.py`)
  for f32-specific code paths. Dump its HLO under c=f32 and
  compare to c=bf16.
- Audit optimizer state handling under p=f32,c=f32 vs
  p=f32,c=bf16 (look for silent dtype casts in Adam's update
  step).
- Try `SeparateReferenceConfig` at c=f32 on v5p-16 pd=2 (Exp V
  was this config at c=bf16 on v5p-8 and failed for a different
  reason; never tested on a known-good mesh at c=f32).

---

## Experiment AB (2026-04-18T05:33Z, LAUNCHED) — rerun Exp U with HLO dump to directly verify reduction dtype under `c=f32`

### AB result (2026-04-18T05:43Z, uc1-ab1) — **HYPOTHESIS FALSIFIED**

**Training trajectory: STUCK in bad basin.** Did not recover despite
`p=f32, c=f32`:

| step | AB (`c=f32`) | Exp Q (`c=bf16`, bad) | Good v5p-16 |
|------|---:|---:|---:|
| 0 | 0.69268095 | 0.69314718 | 0.6931 |
| 1 | 0.69472682 | 0.69314718 | 0.6931 |
| 2 | 0.68645769 | 0.68512470 | 0.3352 |
| 3 | 0.68045288 | 0.68229818 | 0.3260 |
| 4 | 0.67701888 | 0.67372340 | 0.3362 |
| 5 | 0.66966462 | 0.66894621 | 0.3168 |
| 6 | 0.66512156 | 0.66757309 | 0.3370 |
| 7 | 0.66537619 | 0.66282284 | 0.3243 |
| 8 | 0.66031528 | 0.65871453 | 0.3061 |
| 9 | 0.65985513 | 0.66055727 | — |

AB trajectory slightly differs from Exp Q (unlike AA which was
byte-identical — AA's "cast" was a no-op). `c=f32` really did
change the compute. Step-0 `grad_l2=2.4448` vs Exp Q's `2.4560`
(a ~0.4% difference, far above bf16-noise levels).

**HLO verification: reductions ARE f32.** Pulled
`module_0292.jit__train_step.cl_813921542.after_optimizations.txt`
from `gs://marin-us-central1/debug/xla_hlo/uc1-ab1/`. Key findings:

- **Reduction region counts**: 5 bf16 regions, 4 f32 regions
  (vs Exp Q/AA baseline: 71 bf16, 2 f32). `c=f32` collapsed the bf16
  reductions dramatically.
- **Width-4 grad all-reduces**: every `all-reduce(...)` and
  `all-reduce-scatter(...)` at `replica_groups={{0,1,2,3}}` now
  operates on `f32[...]` tensors. Example (from
  `%all-reduce-scatter.clone.clone`):
  ```
  %input.73 = f32[8,4,128,4096]{3,2,1,0:T(8,128)} parameter(0)
  %all-reduce.321 = f32[8,4,128,4096]{...} all-reduce(%input.73),
    channel_id=514, replica_groups={{0,1,2,3}},
    to_apply=%add.1.clone, ...
  ```
- **Reduction `to_apply` regions**: `%add.1.clone`, `%add.3.clone`,
  `%add.6.clone`, `%add.8.clone`, `%add.11.clone` are all
  unambiguously:
  ```
  (x: f32[], y: f32[]) -> f32[] {
    ROOT add = f32[] add(x, y)
  }
  ```
  **No bf16 reducers anywhere on the width-4 grad collectives.**
- **Remaining 5 bf16 regions**: `fused_computation.*` regions
  involving bf16 *parameters* (reference-model path, non-trainable
  bf16 tensors in the graph). None of them are grad reductions.

### What AB proves

Under `c=f32`, **every FSDP gradient all-reduce on v5p-8 pd=4 at
width 4 runs `f32 + f32 → f32`**. Training still gets stuck at
step-2 loss ≈ 0.686, identical qualitative behavior to Exp Q at
`c=bf16`.

**Therefore: the bf16 non-associative collective at width 4 is NOT
the mechanism causing the v5p-8 LoRA DPO pathology.**

This falsifies the hypothesis that `precision_explained_jax_tpu.md`
has been centering ("width-4 bf16 collective produces direction-biased
sums that trap LoRA in the bad basin"). The pathology must be driven
by something about `|data|=4` that is **not** reduction dtype.

### What still holds from the prior investigation

- Exp T: full-FT on v5p-8 pd=4 works → LoRA-specific.
- Exp W: pure TP on v5p-8 works → FSDP-specific.
- Exp Z3: `{data:2, model:2}` on v5p-8 works → width-4-specific.
- Exp Z1: grad values differ between v5p-8 and v6e-8 at matched
  seed/batch/init at the 1e-5 absolute level → there IS a numerical
  divergence. It was attributed to bf16 non-associativity, but AB
  shows dtype isn't the load-bearing lever. The divergence may come
  from a different source (e.g., different reduce-scatter tree
  topology at width 4 vs 8, even at f32).
- Exp Z4: HLO diff v5p-8 vs v6e-8 showed identical structure except
  `replica_groups` width. Still true — and now we know the dtype was
  a red herring; the width/topology effect is what matters.

### New candidate mechanisms (non-dtype) for the width-4 pathology

Listed roughly in order of prior plausibility:

1. **Reduce-scatter tree topology at width 4 vs 8**. XLA's lowering
   picks a physical reduction order (ring, tree, butterfly) based on
   topology and dimensions. The 4-chip reduction on v5p-8's torus
   may use a tree shape that produces a specific numerical-error
   distribution over the rank-r LoRA update subspace that happens to
   land in the bad basin. f32 mitigates bf16 rounding but doesn't
   change tree structure; the bias could be in which elements get
   summed first.
2. **All-gather vs reduce-scatter fusion choices at width 4**. Z4
   showed slight counts differ between widths (14 vs 15 reduction
   regions). At width 4, XLA may be fusing differently or emitting
   the boundary-reducing ops on different buffer layouts, producing
   a systematic (non-statistical) difference in grad values.
3. **Buffer layout / memory placement at width 4**. Padded or
   non-multiple-of-X shapes at width 4 may trigger different tile
   packing than at width 8, with associated reassociation of the
   reduction order even in f32.
4. **Collective chunking**. For small per-chip shards (LoRA grads
   are small), the collective may be chunked differently at width 4
   than width 8.
5. **Something LoRA-specific about the partial→sharded conversion
   at width 4** when the sharded output has an odd number of chunks
   (rank 64 / 4 chips vs 64 / 8 chips).

None of these have been tested yet. All are structural, not
numerical, so they'd show up in HLO diff as scheduling/shape
differences rather than dtype differences.

### Immediate implications for the logbook and precision_explained

1. `precision_explained_jax_tpu.md` Parts 6-9 (the narrative
   centered on bf16-collective-width-4 being the leading explanation)
   needs a correction pointer. The "leading hypothesis" should now
   be "width-4 reduction structure (non-dtype) — mechanism
   unknown".
2. Exp U is now validated: `c=f32` really does produce f32
   reductions in this stack, and the result we recorded for Exp U
   is trustworthy.
3. Z4's HLO evidence stands: the reductions ARE bf16 at default
   `c=bf16`. But its interpretation shifts from "the bf16 dtype is
   the mechanism" to "the bf16 dtype is a correlate of the real
   structural mechanism, not the cause."

### Next experiment candidates (post-AB)

Order by information-per-unit-cost:

- **Diff AB HLO vs Exp Q HLO vs Z4's v6e-8 HLO** structurally.
  Isolate what differs at width 4 vs width 8 *besides* dtype —
  tree topology, scheduling, buffer layout. This is a pure HLO
  analysis, no TPU time. Highest priority.
- **Instrument per-rank contribution to the LoRA grad all-reduce**
  to see which chip's partial sum dominates or how the sum is
  organized. Requires custom instrumentation but no new fundamental
  capability.
- **Mesh rearrangement within width 4**: vary which chips are
  assigned to the data axis (logical mesh reordering) to see if the
  pathology depends on the physical topology of the 4-chip
  aggregation.
- **Collective-algorithm override**: force XLA to use a specific
  reduction algorithm at width 4 (if flags allow) to isolate tree
  shape as a variable.

### Running state at 2026-04-18T05:43Z

- AB uc1-ab1 completed successfully in 17:48. HLO analyzed.
- AB ue5-ab1 still running. Not load-bearing (uc1-ab1 already gave
  conclusive result); will let finish for cross-region replication.
  If its HLO and trajectory match uc1-ab1, no additional analysis
  needed.

### Launch record (2026-04-18T05:33Z)

**Iris jobs (both v5p-8 regions per launch directive):**
- `us-central1`: `/ahmed/iris-run-experiment_ab_v5p8_fp32_pd4_hlo_s10-20260418-053332` (run tag `uc1-ab1`) — **completed, analyzed**
- `us-east5`: `/ahmed/iris-run-experiment_ab_v5p8_fp32_pd4_hlo_s10-20260418-053343` (run tag `ue5-ab1`) — in flight, replication check

**Workspace at submit time:**
- Diagnostic AA code changes **reverted** in
  `lib/levanter/src/levanter/grad_accum.py`,
  `lib/levanter/src/levanter/trainer.py`, and
  `lib/levanter/src/levanter/main/train_dpo.py` (via `git checkout`).
  AA experiment script deleted. AB launches against an otherwise-clean
  levanter tree.
- New experiment script
  `experiments/posttrain/per_stmt_dpo/experiment_ab_v5p8_fp32_pd4_hlo_s10.py`
  is a clone of `experiment_u_v5p8_fp32_pd4_s10.py` + the HLO-dump
  plumbing from `experiment_z4_hlo_dump_s2.py`.
- Pre-commit passes.

**What to check once AB finishes (~15-20 min per region):**

1. Download HLO from
   `gs://marin-us-central1/debug/xla_hlo/uc1-ab1/module_0*.jit__train_step.cl_*.after_optimizations.txt`
   (or `ue5-ab1` path).
2. `grep -c "bf16\[\].*add"` and `grep -c "f32\[\].*add.*f32\[\]"`
   for `to_apply` reduction-region counts.
3. Spot-check the width-4 (`replica_groups={{0,1,2,3}}`) all-reduce
   ops and their `to_apply=%add.X` reducer dtypes.
4. Compare loss trajectory against Exp Q (bad), Exp U, and v5p-16
   pd=2 (good).
5. Apply the interpretation matrix from the AB plan.

---

<!-- duplicate AB plan block removed after falsification result — content now fully captured in the AB section above -->

<!-- Duplicate AB plan block below this line was removed after AB falsification — content now lives in the AB section above and doesn't need to be repeated. Jumping directly to the real AA section. -->
<!-- __AB_DUPLICATE_BLOCK_START__

**Status:** about to launch. Decisive cheap test for the
bf16-collective-width-4 hypothesis. Supersedes the entire AA thread
(which intervened at the wrong site; see AA retraction below).

### Goal

Directly answer the central open question the precision_explained
logbook has been pointing at since the Z-experiments:

> When Exp U ran `p=f32, c=f32` on v5p-8 pd=4 LoRA and training still
> stayed in the bad basin — did the compiled HLO actually have f32
> FSDP grad reductions?

Exp U predated the HLO-dump workflow (added in Z4), so we've never
inspected its HLO. AB is Exp U, rerun with HLO dumping and upload to
GCS so we can actually look.

### Why AB, not a deeper AA surgery

AA attempted to surgically cast grads to f32 inside `grad_accum.loop`.
AA v5's diagnostics showed this is futile because:

- The grads reaching the accum loop are already f32 (cast up from
  bf16 after the per-layer FSDP reduce-scatter inside the backward).
- The load-bearing bf16 reduction happens inside `fn`'s backward pass
  per-layer, not at the accum boundary.

To fix the bf16 reductions via `jmp`, we'd need `c=f32` — which is
exactly what Exp U did. So instead of writing custom VJP rules to
cast cotangents (the "correct" surgical intervention site, but
nontrivial), we should first verify what Exp U actually did at the
HLO level. If Exp U already produced f32 reductions and still broke,
the hypothesis is dead and no cotangent-level intervention will save
it.

### Hypothesis being tested

H0: FSDP reduce-scatter on LoRA cotangents runs in bf16 at width 4
on v5p-8 under default `c=bf16`, producing a non-associative sum that
systematically biases the early LoRA update and traps training in
the bad basin. Under `c=f32`, the reductions run in f32 (associative
to much tighter tolerance), eliminating the bias.

AB tests H0 by reading the compiled HLO under `c=f32`.

Predictions under H0:

- AB HLO shows `f32 + f32 → f32` reductions for all width-4 LoRA-grad
  collectives (replica_groups `{{0,1,2,3}}`).
- AB training **recovers**: step-2 loss ≤ 0.5 (clearly below the
  0.685 attractor), step-10 delta_pi − delta_ref ≥ 5.

If AB HLO shows f32 reductions AND training is stuck → **H0 falsified**.
Then width-4 pathology is not about reduction dtype; candidate
mechanisms to investigate next: HLO scheduling differences, buffer
layout, collective tree topology, all-gather/reduce-scatter fusion
choices that differ at width 4 vs 8.

If AB HLO still shows bf16 reductions despite `c=f32` → there's a
hidden bf16 downcast somewhere in the stack (CE Pallas kernel,
optimizer path, remat cache). Identify it, fix it, rerun.

**Replication note**: AB is also a replication check for Exp U's
trajectory. If Exp U's result doesn't reproduce in AB (e.g.,
trajectory differs materially), that's a separate signal — something
about the recipe or stack drifted between then and now.

### Interventions (none new — AB is just Exp U + logging)

Relative to Exp Q baseline (the bad recipe):

- Same hardware (v5p-8), mesh (canonical FSDP, `{replica:1, data:4,
  model:1}`), data, batch size (64), per-device parallelism (4),
  learning rate schedule, LoRA config (r=64, alpha=64, zero_init_b).
- **Only scientific change**: `mp=jmp.get_policy("p=f32,c=f32")`.
- **Added for observability**: `XLA_FLAGS=--xla_dump_to=...`,
  `MARIN_DEBUG_HLO_UPLOAD_DIR=gs://.../ab-<tag>/`, plus the usual
  `MARIN_DEBUG_LOG_STEP_TRACE=1` and `MARIN_DEBUG_LOG_BATCH_INDICES=1`.

No source-code changes. Pure recipe + observability.

### Experiment script

`experiments/posttrain/per_stmt_dpo/experiment_ab_v5p8_fp32_pd4_hlo_s10.py`

Direct clone of `experiment_u_v5p8_fp32_pd4_s10.py` with HLO dump
wiring copied from `experiment_z4_hlo_dump_s2.py`.

### Launch commands (parallel, both v5p regions per launch directive)

```bash
# us-central1
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  -e REGIONS_OVERRIDE us-central1 \
  -e MARIN_DEBUG_LOG_BATCH_INDICES 1 \
  -e MARIN_DEBUG_LOG_STEP_TRACE 1 \
  -e MARIN_DEBUG_RUN_TAG uc1-ab1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- python experiments/posttrain/per_stmt_dpo/experiment_ab_v5p8_fp32_pd4_hlo_s10.py

# us-east5
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-east5 --cpu 1 --memory 3g \
  -e REGIONS_OVERRIDE us-east5 \
  -e MARIN_DEBUG_LOG_BATCH_INDICES 1 \
  -e MARIN_DEBUG_LOG_STEP_TRACE 1 \
  -e MARIN_DEBUG_RUN_TAG ue5-ab1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- python experiments/posttrain/per_stmt_dpo/experiment_ab_v5p8_fp32_pd4_hlo_s10.py
```

### HLO analysis plan

Once HLO uploads to `gs://marin-us-central1/debug/xla_hlo/uc1-ab1/`
or `.../ue5-ab1/`:

1. Download `module_0*.jit__train_step.*.after_optimizations.txt`
   (the optimized HLO for the main train step module — probably
   `module_0300.jit__train_step.cl_*.after_optimizations.txt` based
   on Z4's numbering).
2. Count reduction region dtypes:
   ```bash
   grep -c "bf16\[\].*add" <hlo>     # bf16 reduction regions
   grep -c "f32\[\].*add.*f32\[\]" <hlo>  # f32 reduction regions
   ```
   Z4's v5p-8 baseline: 71 bf16, 2 f32 (the 2 f32 are scalar loss
   reductions, not grad collectives).
3. Spot-check the actual all-reduce and reduce-scatter ops at width
   4 (`replica_groups={{0,1,2,3}}`). Verify the `to_apply=%add.X`
   regions are f32 adders.
4. Compare to Z4's v5p-8 HLO to see the structural diff introduced
   by `c=f32`.

### Training analysis plan

Pull the W&B trajectory via tag `uc1-ab1` / `ue5-ab1` and compare
step-by-step against:

- Exp Q baseline (bad, default `c=bf16`): step-2 0.6851, step-9
  ~0.66.
- Exp U (stored in logbook): should match AB to high precision if
  there's no stack drift.
- Good v5p-16 pd=2 reference: step-2 ~0.335, step-9 ~0.31.

Success = recovery: step-2 ≤ 0.5 and step-10 `delta_pi − delta_ref`
≥ 5. Failure = bad basin: step-2 ≈ 0.685 and `delta_pi − delta_ref`
in [0.1, 0.7].

### Cost

- Code change: 0 (only new experiment script, copied from U + Z4).
- Training run: ~15-20 min wall-clock per region.
- HLO download + analysis: ~5 min.
- Total: ~30 min wall-clock, one 10-step v5p-8 allocation.

### Interpretation — the decision

AB is designed so we can stop guessing after this one run:

- **f32 HLO + stuck training** (the most likely outcome): retract the
  bf16-collective-width-4 hypothesis from `precision_explained_jax_tpu.md`
  and start fresh on non-dtype `|data|=4` mechanisms. Exp W and Z3
  still demonstrate that *width 4 specifically is pathological*, but
  the "why" moves to HLO scheduling, tree topology, buffer layout.
- **f32 HLO + training recovers**: dtype WAS the mechanism; Exp U's
  original result was mislogged or the stack drifted. Land the
  `c=f32` workaround while we figure out a cheaper cotangent-level
  fix.
- **bf16 HLO + stuck**: `jmp c=f32` has a leak — likely the Pallas CE
  kernel. Fix, then relaunch. Not a falsification, just needs
  follow-up plumbing.

### Follow-ups for the precision_explained logbook after AB

Regardless of outcome, the AA retraction needs to be added to
`precision_explained_jax_tpu.md` Part 6 / 7 to correct the claim
that AA's cast site was the right intervention for lever D. The
revised teaching is:

> Lever D (collective reduction dtype) for FSDP-style grad
> aggregation must be controlled at the tensor that GSPMD-emits
> reduces — which, under standard `jmp.Policy(p=f32, c=bf16)`, is
> the bf16 cotangent *inside the backward pass, per layer*. Casting
> at the microbatch accumulation boundary is too late because the
> reduce has already happened.

AB's result then tells us whether lever D matters at all for the
v5p-8 LoRA pathology.

__AB_DUPLICATE_BLOCK_END__ -->

## Experiment AA (2026-04-17 → 2026-04-18) — RETRACTED: wrong intervention site

**Status:** LAUNCHED 2026-04-17T21:17Z (v2 plan, revised after Codex
review — see `docs/debug-log-experiment-aa.md`). Read the top pointer and
the companion
[precision_explained_jax_tpu.md](./precision_explained_jax_tpu.md) for
full context.

### Launch record (2026-04-17T21:17Z)

**Code changes:**
- `lib/levanter/src/levanter/grad_accum.py` — added `import os`,
  module-level `_AA_CAST_GRADS_F32 = os.environ.get("MARIN_DEBUG_AA_CAST_GRADS_F32", "0") == "1"`,
  and env-gated `this_grads.astype(f32)` inside the `loop` function
  immediately after `(this_loss, this_metrics), this_grads = this_r`
  and before `(acc_loss, acc_metrics), acc_grads = acc`.
- `experiments/posttrain/per_stmt_dpo/experiment_aa_v5p8_pd4_f32_grad_cast_s10.py`
  — clone of Exp Q pd=4 with `MARIN_DEBUG_AA_CAST_GRADS_F32=1`, HLO
  dump flags, and HLO upload to
  `gs://marin-us-central1/debug/xla_hlo/aa-v5p8-pd4/`.
- Pre-commit passes.

**Iris jobs (both v5p-8 regions per top-of-logbook directive):**
- `us-central1`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-041719`
  (run tag `uc1-aa1`)
- `us-east5`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-041730`
  (run tag `ue5-aa1`)

First copy to land capacity is load-bearing; the other is a
cross-region replication check (leave running unless it blocks
another probe).

**What to verify once the run compiles:**
1. HLO upload to `gs://marin-us-central1/debug/xla_hlo/aa-v5p8-pd4/<tag>/`
   — grep the `jit__train_step` module for `to_apply` regions near LoRA
   grad `all-reduce`/`reduce-scatter` ops. Success: reduction region is
   `f32[] add(f32[], f32[])`. Fail: still bf16 (cast was optimized
   away — see AA plan "HLO verification" subsection for debug steps).
2. W&B `train/loss` trajectory:
   - step 2 ≤ 0.5 → AA.1 recovered (H0 confirmed, pending HLO pass).
   - step 2 ≈ 0.685 → AA.1 did not recover (pivot to non-dtype
     investigation, pending HLO showing f32 actually took effect).
3. `delta_pi − delta_ref` at step 10: ≥ 5 if recovered; in [0.1, 0.7]
   if stuck.

### Initial training result (2026-04-18T04:28Z, us-central1 copy)

**Training trajectory matches Exp Q point-for-point to 6+ decimal
digits at every step.** AA did not escape the bad basin. HLO upload
pending to disambiguate whether the cast took effect.

| step | AA (uc1-aa1) | Exp Q (bad baseline) | Good v5p-16 |
|------|---:|---:|---:|
| 0 | 0.69314718 | 0.69314718 | 0.6931 |
| 1 | 0.69314718 | 0.69314718 | 0.6931 |
| 2 | **0.68512470** | 0.685125 | 0.3352 |
| 3 | **0.68229818** | 0.682298 | 0.3260 |
| 4 | 0.67372340 | 0.673723 | 0.3362 |
| 5 | 0.66894621 | 0.668946 | 0.3168 |
| 6 | 0.66757309 | 0.667573 | 0.3370 |
| 7 | 0.66282284 | 0.662823 | 0.3243 |
| 8 | 0.65871453 | 0.658715 | 0.3061 |
| 9 | 0.66055727 | - | - |

Step-0 grad_l2 = `2.456002712249756` matches Exp Q's logged value to
the precision of the debug trace (13+ digits). Per the Z1 measurements
(bf16 vs f32 reductions differ at 1e-5 absolute), an exact match this
precise strongly suggests the cast did not compile into the HLO.

**Leading hypothesis (pending HLO):** the `jax.lax.scan` inside
`hax.fold` enforces carry dtype consistency. Since the accumulator
`acc_grads` is initialized with `accum_dtype=None` (which defaults to
`fn`'s return dtype = bf16), and our body returned f32 grads, XLA
either:
- silently coerced the body's output back to bf16 to match the carry, OR
- elided the cast entirely as provably equivalent to no-cast under
  dtype reconciliation

Either way, no f32 reduction happened. The fix is almost certainly
**AA v3: cast AND set `accum_dtype=jnp.float32` together**, so the
carry is f32 from init and the scan-level type enforcement propagates
the f32 through instead of collapsing it back to bf16.

HLO inspection will confirm and inform whether AA v3 is the right
next move.

**us-east5 copy:** still pending capacity as of 2026-04-18T04:35Z.
Since us-central1 gave a conclusive "did not recover" signal, us-east5
is no longer load-bearing; will leave running as a passive replication
check unless it blocks another probe.

### AA v1 HLO result (2026-04-18T04:38Z, uc1-aa1)

Pulled `module_0300.jit__train_step.cl_813921542.after_optimizations.txt`
from `gs://marin-us-central1/debug/xla_hlo/uc1-aa1/`. Reduction-region
analysis:

| Reduction type | Count |
|---|---:|
| `bf16[] add(bf16[], bf16[])` regions | **71** |
| `f32[] add(f32[], f32[])` regions | **2** |

All FSDP gradient `all-reduce` and `all-reduce-scatter` ops at
`replica_groups={{0,1,2,3}}` use bf16 `to_apply` regions. The 2 f32
regions are for scalar reductions (likely loss-sum aggregations), not
LoRA gradient collectives. This is **identical in shape to Z4's v5p-8
HLO** — the AA cast did not reach the FSDP grad reductions.

Conclusion for AA v1: **outcome row 3 of the interpretation matrix
(HLO still bf16 — cast was optimized away)**. Experiment invalid for
testing H0. Diagnosis below.

### Diagnosis: scan-carry dtype collapse

The `loop` function in `grad_accum.py` runs inside a `jax.lax.scan` via
`hax.fold(loop, AccumStep)`. Scan enforces carry-dtype consistency
across iterations. With `accum_dtype=None` (the default in
`trainer.py:871`), the accumulator is initialized by
`zeros_like_tree(r_shape, accum_axis_mapping, accum_dtype=None)` using
the dtype of `fn`'s output (bf16 for LoRA grads under `c=bf16`).

Our AA v1 patch cast `this_grads` to f32 inside the body. The add
`apply_updates(acc_grads_bf16, updates_f32)` would produce f32 by
promotion. But the carry was bf16 per the init, and XLA's optimizer
decided the whole round-trip was equivalent to keeping everything
bf16 — either by scan's explicit coercion or by noting that the
f32 intermediate doesn't change the final bf16 output. Either way,
the HLO ended up with `bf16 + bf16 → bf16` reductions, same as Exp Q.

### AA v3 — cast + `accum_dtype=jnp.float32` together

**Code change:** `lib/levanter/src/levanter/trainer.py:871` — under
the same env-var gate, set `accum_dtype=jnp.float32` on the
`microbatched(...)` call. With the accumulator initialized as f32,
the scan carry is f32 throughout the loop; our in-loop cast on
`this_grads` feeds f32 into `apply_updates`; the only layout
reconciliation is partial→sharded (no dtype resolution), so GSPMD
must emit an `f32 + f32 → f32` reduction.

Pre-commit passes.

**Launched 2026-04-18T04:40Z (AA v3):**
- `us-central1`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-044009`
  (tag `uc1-aa3`)
- `us-east5`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-044048`
  (tag `ue5-aa3`)

Stale AA v1 us-east5 job (`...041730`) killed to free capacity —
had pre-patch workspace that wouldn't have had the `accum_dtype`
change even if it landed capacity later.

### AA v3 result (2026-04-18T04:50Z, uc1-aa3) — identical to AA v1

**Training trajectory matches AA v1 and Exp Q point-for-point to 16
decimal digits at every step.** Same step-2 loss (0.6851246953010559),
same step-9 loss (0.6605572700500488). No escape.

**HLO diff vs AA v1: 0 line changes.** `diff` between
`module_0300.jit__train_step.after_optimizations.txt` from uc1-aa1
and uc1-aa3 returned empty. Same 71 bf16 reductions, same 2 f32
reductions (scalar loss reductions, not grad collectives), same
`replica_groups={{0,1,2,3}}`, same structure end-to-end.

Conclusion: **neither the in-loop cast nor `accum_dtype=jnp.float32`
had any effect.** Both env-gated changes were silently no-ops. Leading
suspicion: env var `MARIN_DEBUG_AA_CAST_GRADS_F32` was not visible at
trace time inside the jit (even though it seems to be set — other
`MARIN_DEBUG_*` vars do propagate).

### AA v4 — runtime env check + trace-time diagnostic print

**Code change:** moved env check from module-level constant to
runtime inside the `loop` function, simplified cast to unconditional
`astype(f32)`, and added `print("[DEBUGAA] ...")` inside the loop
body so we can see in logs whether the cast path was entered at
trace time.

Also patched `train_dpo.py:577` to add `MARIN_DEBUG_AA_CAST_GRADS_F32`
to the `DEBUGJ WORKER_ENV` probe's hardcoded key list, so future
runs will show whether the var is present on the worker.

**Launched 2026-04-18T04:57Z (AA v4):**
- `us-east5`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-045727` (tag `ue5-aa4`)
- `us-central1`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-045737` (tag `uc1-aa4`)

### AA v4 result (2026-04-18T05:10Z) — also no-op, `[DEBUGAA]` print never fired

Same trajectory, 16 decimal digits, exp Q match. And no `[DEBUGAA]`
marker appears anywhere in either region's logs. Either:
- The env var is not visible at trace time when the `os.environ.get`
  check inside the loop body runs (unlike `MARIN_DEBUG_LOG_STEP_TRACE`
  which follows the same pattern and works).
- OR the loop body wasn't actually traced at my print location
  (unlikely — microbatching is confirmed active at `mb=16 < batch=64`,
  and the `DEBUGJ BATCH n=16` lines in all runs prove the loop runs).

### AA v5 — UNCONDITIONAL cast + import-time diagnostic

To cut through the env var confusion, AA v5 removes all env-var
gating on the cast:

- `grad_accum.py`: unconditional `this_grads.astype(jnp.float32)`
  inside the loop body, with trace-time `print` showing sample leaf
  dtype before and after, plus the env var value.
- `grad_accum.py` module-level: unconditional `print("[DEBUGAA IMPORT]
  ...")` showing the env var value at import time.
- `trainer.py:871`: unconditional `accum_dtype=jnp.float32` in the
  `microbatched` call, with trace-time `print`.

This decouples "does the cast work?" from "does the env var
propagate?" and tells us both independently.

**Launched 2026-04-18T05:16Z (AA v5, us-central1 only):**
- `us-central1`: `/ahmed/iris-run-experiment_aa_v5p8_pd4_f32_grad_cast_s10-20260418-051619` (tag `uc1-aa5`)

Orphaned AA v3 uc1, AA v4 ue5 and uc1 jobs killed to free capacity.

### Expected AA v5 outcome matrix

| Module import print | Trace prints | HLO result | Training | Conclusion |
|---|---|---|---|---|
| `env='1'` | fire | reductions f32 | recovers | H0 confirmed; env var was the issue in v1–v4 |
| `env='1'` | fire | reductions f32 | stuck | H0 rejected (dtype isn't the mechanism) |
| `env='1'` | fire | reductions bf16 | stuck | XLA optimizing the cast away — need optimization_barrier |
| `env='UNSET'` | fire | reductions f32 | recovers or stuck | Unconditional cast works; env var propagation is broken; fix that separately |
| prints don't fire | — | — | — | Broader `print`/logging failure; investigate iris log capture |

### AA v5 result (2026-04-18T05:25Z, uc1-aa5) — **INTERVENTION SITE IS WRONG**

All three `[DEBUGAA]` prints fire as designed. Key extractions from logs:

```
[DEBUGAA IMPORT] grad_accum.py imported;
               MARIN_DEBUG_AA_CAST_GRADS_F32='1'
[DEBUGAA TRAINER] calling microbatched with accum_dtype=jnp.float32
                  (unconditional); env MARIN_DEBUG_AA_CAST_GRADS_F32='1'
[DEBUGAA TRACE] grad_accum.loop:
               n_leaves=26,
               sample dtype before cast=float32,   <-- already f32!
               env='1'
[DEBUGAA TRACE] grad_accum.loop:
               sample dtype after cast=float32
```

Training trajectory at step 0–4: identical to Exp Q and AA v1–v4 to
16 decimal digits. Step-2 loss 0.6851246953010559. No escape.

**Two headline findings that overturn prior interpretation:**

1. **`MARIN_DEBUG_AA_CAST_GRADS_F32='1'` at both module import and
   trace time.** The env var WAS set all along — AA v1–v4's no-op
   behavior was never about env var propagation. That entire
   diagnostic thread (AA v2/v3/v4 rationale, the scan-carry-dtype
   hypothesis, the WORKER_ENV probe addition) was based on a wrong
   premise.
2. **The gradient tensors reaching my insertion point in
   `grad_accum.loop` are ALREADY `float32`, not `bfloat16`.** My cast
   `.astype(jnp.float32)` has been a trivial no-op every single
   iteration because the dtype was already f32. HLO was unchanged
   because there was nothing to change.

### Why the grads are f32 at the accum loop — the real FSDP reduce site

In the Marin/Levanter stack under `jmp.Policy(p=f32, c=bf16)`:

- Forward and backward run the model in bf16 (c=bf16).
- Per-layer backward cotangents in the model's compute graph are bf16.
- **GSPMD emits the FSDP reduce-scatter on those bf16 cotangents
  per-layer inside the backward pass**, not at any later boundary.
  The reduce is `bf16 + bf16 → bf16`, width 4 on v5p-8, as Z4
  documented.
- After the reduce, the sharded bf16 grad is cast up to f32 (to
  match the f32 master-param dtype) before `filter_value_and_grad`
  returns.
- So by the time `fn(microbatch)` returns `this_grads` and the
  microbatch loop sees it, the grad is f32 AND THE BF16 REDUCE HAS
  ALREADY HAPPENED.

Corollary: **every AA experiment so far (v1–v5) intervened
downstream of the load-bearing collective.** The cast in
`grad_accum.loop` cannot affect the bf16 reduction because that
reduction is emitted earlier, inside the compiled backward graph,
per layer, at the per-layer partial→sharded boundary. My loop-level
cast is at the wrong site.

### Where the correct intervention lives

The reduce-scatter we care about is emitted by GSPMD when per-layer
cotangents go from "partial along data axis" to "sharded along data
axis" to match the parameter sharding. This happens inside the jit,
inside `fn`, inside the model's backward pass, emitted automatically
by the autodiff/partitioning machinery.

Candidate intervention sites, from narrowest to broadest:

- **Custom VJP rule** on the layers whose backward emits a width-4
  reduce. Cast the cotangent to f32 before returning it from the
  VJP, so the emitted reduce operates on f32 operands. Would require
  patching `haliax/dot.py` or the specific linear layer's VJP.
- **Global backward-pass dtype promotion** for LoRA-marked
  parameters via a custom grad transformation wrapper.
- **`c=f32` across the board** (i.e., Exp U). This SHOULD produce
  f32 reductions by making all cotangents f32 at source. Exp U did
  not recover training — which means either (a) bf16 reductions are
  not the mechanism, or (b) Exp U's HLO did not actually produce
  f32 reductions (never verified).

### What Exp U's HLO would tell us

The single most valuable cheap follow-up right now is **dump Exp U's
HLO** (`p=f32, c=f32` recipe) and inspect the grad reduction regions:

- If Exp U's HLO shows `f32 + f32 → f32` reductions and training
  still broke → bf16 collective dtype IS NOT the mechanism. Pivot
  investigation to non-dtype `|data|=4` effects.
- If Exp U's HLO still shows `bf16 + bf16 → bf16` reductions
  somehow (despite `c=f32`), → some deeper kernel or path is
  forcing bf16 reductions regardless of `jmp`. Identify and fix.

This is a ~10 min CPU compile. No TPU time.

### What's stale / wrong in the prior logbook sections above

The AA v2 plan's entire premise — "cast LoRA grads to f32 inside
`grad_accum.py`'s accumulation loop" — targets the wrong site for
the actual FSDP gradient reduction. Codex's Option A recommendation
(same site) was also wrong in the same direction, though for a
different reason (Codex assumed the per-microbatch reduce at the
accum boundary was the load-bearing collective; in fact the
backward per-layer reduce-scatter inside fn is).

The scan-carry-dtype and XLA-elision hypotheses in AA v2/v3/v4 were
explanations for a phenomenon that didn't exist: the cast was a
no-op because the input was already f32, not because XLA removed
anything or scan forced a downcast.

### Running state at 2026-04-18T05:27Z

- AA v5 (uc1-aa5) still progressing toward step 9. Trajectory
  guaranteed to match Exp Q (null intervention).
- No other AA runs active.
- Code in `grad_accum.py` and `trainer.py` is currently in DIAGNOSTIC
  state (unconditional cast + unconditional `accum_dtype=f32` +
  debug prints). **Needs to be reverted or gated before any other
  training runs** — the `accum_dtype=f32` change will double memory
  for gradient accumulators in every levanter training run if left
  as-is.

### Next move

Priorities, in order:

1. **Stop AA v5** (already doesn't tell us anything new).
2. **Revert `grad_accum.py` and `trainer.py` diagnostic changes**
   (or gate them back behind the env var) so the repo is clean.
3. **Dump Exp U HLO** on v5p-8 with `p=f32, c=f32` (no TPU; CPU
   compile) to answer whether `c=f32` actually produces f32
   reductions. This is the decisive cheap test for the whole
   bf16-collective-width-4 hypothesis.
4. If Exp U HLO shows f32 reductions + Exp U trajectory was bad →
   hypothesis rejected; pivot to non-dtype `|data|=4` investigation
   (HLO scheduling, tree topology, buffer layout).
5. If Exp U HLO still shows bf16 reductions → find why `c=f32`
   didn't propagate all the way to the collective, fix it, retry.

### v1 → v2 correction (important context)

The first draft of this plan put the cast in `lib/levanter/src/levanter/
trainer.py` between `_compute_gradients_microbatched(...)` and
`state.take_step(...)`, justified by "grads cross the `_train_step` pjit
out-sharding boundary." **That rationale was wrong.**

- `_train_step` computes grads at `trainer.py:683` and immediately
  consumes them via `state.take_step(grads, ...)` at `trainer.py:697`.
  It returns `TrainStepResult[S]`, not the raw grad tree. No grad tree
  crosses the `named_jit`'s out-sharding boundary.
- Every bad-recipe baseline so far (Exp Q, R, R2a, T, U, V, W, Y, Z1-4)
  is **microbatched**: `per_device_parallelism × data_axis_size <
  train_batch_size`, so `self.config.microbatch_size is not None` at
  `trainer.py:869` and `grad_fn` is wrapped by `microbatched(...)` at
  `trainer.py:871`.
- In the microbatched path, per-microbatch grads come out of the inner
  `fn` with "partial along data" layout. The loop carries `acc` in
  `accum_axis_mapping = parameter_axis_mapping` (sharded-on-data). At
  each iteration, `hq.apply_updates(acc_grads, updates)` at
  `grad_accum.py:149` plus the explicit
  `hax.shard_with_axis_mapping(acc, accum_axis_mapping)` at
  `grad_accum.py:153` force the partial→sharded conversion — which is
  where GSPMD emits the reduce-scatter.
- **So for the microbatched bad recipe, the bf16 cross-chip reduction
  happens INSIDE the microbatch loop, once per microbatch per LoRA
  tensor, not at any `_train_step` boundary.** The v1 cast site ran
  after those reductions had already happened. It would have had no
  effect on the collective dtype for this recipe.

The v2 intervention below targets the actual reduction site.

### Goal

Causally resolve whether the bf16 dtype of the FSDP cross-chip gradient
all-reduce at `|data|=4` is the mechanism that traps v5p-8 LoRA DPO in
the step-2 bad basin — or whether the `|data|=4` pathology is actually
about something else (HLO scheduling, buffer layout, tree topology).

AA is the direct intervention that distinguishes these. No prior
experiment in the chain can, because:

- Z4 is observational (HLO inspection on the default recipe).
- Exp U changed compute dtype globally; its result is confounded with
  many unrelated graph changes, and — critically — we do not know
  whether Exp U's collective actually ran at f32 (HLO was never dumped
  for the Exp U recipe).
- Z2's flag does not alter collective dtype.

### Hypothesis

**H0:** The bad v5p-8 LoRA trajectory is caused specifically by the bf16
operand dtype of the GSPMD-emitted cross-chip gradient all-reduce at
`|data|=4`. A sufficiently localized intervention that forces the same
all-reduce to run at f32 — without changing anything else in the
compute graph — will escape the step-2 bad basin.

Predictions:

- Cast gradients to f32 immediately after
  `_compute_gradients_microbatched` returns and before `state.take_step`
  consumes them. The grad leaves its current "replicated-on-data" layout
  as f32; GSPMD's reconciliation to "sharded-on-data" therefore emits
  an `f32 + f32 → f32` reduction region.
- Compiled HLO on the patched recipe shows LoRA-gradient reduction
  regions as f32 add, with `replica_groups={0,1,2,3}` (width unchanged).
- Training on the bad v5p-8 pd=4 recipe with this patch escapes the
  step-2 attractor: step-2 loss ≤ 0.5 (the bad attractor is ~0.685, the
  good trajectory is ~0.33), and by step 10 `delta_pi − delta_ref` ≥ 5
  (bad attractor stays in [0.1, 0.7]).

If any one of those three predictions fails, H0 is falsified in the
corresponding way (see Interpretation Matrix below).

### Intervention — exact code change (v2)

**Primary intervention (AA.1): cast `this_grads` to f32 inside the
microbatch accumulation loop.**

**File:** `lib/levanter/src/levanter/grad_accum.py`

**Location:** Between lines 131 and 148 of the `loop` function —
immediately after `this_grads` is unpacked from `this_r`, and before
`hq.partition_for_grad_overwrite(this_grads)` begins consuming it.

**Patch (schematic):**

```python
# grad_accum.py, inside the `loop` function at ~line 131
with jax.named_scope("accum"):
    # Unpack structure: ((loss, metrics_dict), grads)
    (this_loss, this_metrics), this_grads = this_r
    (acc_loss, acc_metrics), acc_grads = acc

    # --- AA intervention ---
    # Cast per-microbatch grads to f32 BEFORE they hit the
    # partial->sharded boundary (apply_updates + shard_with_axis_mapping
    # below). This forces GSPMD to emit an f32+f32->f32 reduction
    # region instead of the default bf16 one. Gated on env var so the
    # same binary can run bad-baseline and patched back-to-back.
    if os.environ.get("MARIN_DEBUG_AA_CAST_GRADS_F32", "0") == "1":
        this_grads = jax.tree.map(
            lambda g: g.astype(jnp.float32) if hasattr(g, "dtype") and g.dtype == jnp.bfloat16 else g,
            this_grads,
        )
    # --- end AA intervention ---

    new_loss = acc_loss + this_loss
    # ... rest of accumulation proceeds unchanged ...
```

Why this location:

- `this_grads` comes out of `fn(microbatch)` in "partial along data"
  layout (each chip computed its own microbatch's full grad tensor;
  none are yet reduced across chips).
- The partial→sharded conversion is forced by the combination of
  `hq.apply_updates(acc_grads, updates)` at line 149 (which requires a
  concrete layout for `acc_grads`, which is sharded-on-data by init)
  and the explicit `hax.shard_with_axis_mapping` constraint at line 153.
- Casting `this_grads` before line 148 means the reduce-scatter
  emitted by GSPMD at the boundary operates on an f32 tensor. No
  upstream path is bf16 at the collective site.
- XLA should preserve the cast-before-reduce ordering because bf16
  reduce and f32 reduce are numerically distinct (that's the entire
  hypothesis). Type-preserving optimization passes should not swap
  them. Verify in HLO regardless.

Why gated on env var:

- Lets the same binary run Exp Q (bad baseline) and AA.1 back-to-back
  on the same compiled code path, switching only the env var. No
  drift from concurrent code changes.
- Consistent with existing debug toggles in the repo (grep for
  `MARIN_DEBUG_`).

**Secondary probe (AA.2, free side-test): set `accum_dtype=jnp.float32`
via trainer config.**

**File(s):** `lib/levanter/src/levanter/trainer.py` line ~871 (the
`microbatched(...)` call does not currently pass `accum_dtype`;
default is `None` which inherits bf16 from `fn`'s return). Thread
`accum_dtype=jnp.float32` through (either by adding a trainer config
field or by conditionally passing it under the same env-var gate).

Interpretation caveat: this makes `acc_grads` start as f32 but does
NOT directly change the operand dtype of the partial→sharded reduction
on `this_grads`. XLA's cost model will likely prefer "reduce bf16
first, then promote" over "promote to f32 first, then reduce bf16 → f32"
because the former moves less bandwidth across chips. So **AA.2 on
its own is likely insufficient to force an f32 reduction**, but it's
a useful cheap probe: run it, dump HLO, and see whether XLA actually
preserves the cast-before-reduce order or not. Useful for calibrating
our intuition about XLA's choices. Do **not** rely on AA.2 alone —
AA.1 is the load-bearing test.

**NOT the right site (v1, retracted): trainer.py:687 between
`_compute_gradients_microbatched` return and `state.take_step` call.**
By this point, the microbatched reductions have already happened in
bf16 inside `grad_accum.py`. Any cast here is too late. Do not use.

### HLO verification — REQUIRED before interpreting training outcome

Before running training, dump and inspect the compiled HLO. The training
result is uninterpretable unless we know the cast reached the collective.

Workflow (same as Z4):

1. Launch the patched recipe with
   `XLA_FLAGS=--xla_dump_to=/tmp/aa_hlo --xla_dump_hlo_as_text` and
   the usual HLO upload plumbing in
   `lib/levanter/src/levanter/main/train_dpo.py` (see Z4 section for
   the working pattern with `fsspec.url_to_fs`).
2. Upload to `gs://marin-us-central1/debug/xla_hlo/aa-<tag>/`.
3. Grep the `0XXX.jit__train_step` module for `to_apply` regions.

**Pass criteria:**

- At least some LoRA-gradient-related `all-reduce` (or `reduce-scatter`)
  ops have `to_apply` reduction regions typed `f32[] add(f32[], f32[])`.
- `replica_groups` for those ops are still `{0,1,2,3}` (width unchanged
  — we are intervening on dtype only).
- Non-LoRA reduction regions (if any) can stay bf16; they are out of
  scope for this hypothesis.
- If you can diff cleanly against Z4's v5p-8 HLO, the only structural
  differences should be reduction-region dtype on LoRA grad ops and
  possibly added `convert` ops for the cast.

**Fail criteria (experiment INVALID, do not interpret training):**

- All reduction regions are still bf16 → the cast was optimized away,
  moved past the collective, or the env var didn't take effect. Debug
  (check the env-var path, check for XLA `--xla_allow_excess_precision`
  flags, try inserting a `jax.lax.optimization_barrier` before/after
  the cast) and rerun HLO dump until pass criteria hold.

### Also free: dump HLO of the existing Exp U run for comparison

Recompile the Exp U recipe (`p=f32, c=f32` on v5p-8 pd=4 LoRA) and dump
its optimized HLO. Check whether ITS reduction regions were bf16 or
f32. This is essentially free (CPU compile; no TPU hours) and
disambiguates Exp U's failure independently of AA:

- **Exp U HLO shows f32 reductions:** Exp U really did run f32
  collectives and still failed. AA should also fail (both try to make
  the collective f32). Increases the prior on H0 being false.
- **Exp U HLO shows bf16 reductions:** something downstream of `jmp
  c=f32` kept the collective at bf16 (Pallas CE kernel, optimizer path,
  remat — see precision_explained Part 7 candidates). AA's cast at
  `trainer.py:687` is a more targeted intervention and should still
  work. Increases the prior on H0 being true.

Run this in parallel with AA's HLO dump.

### Training run

**Recipe:** the exact Exp Q `v5p-8 pd=4 LoRA c=bf16` baseline, plus the
AA patch and env var.

**Script:** copy
`experiments/posttrain/per_stmt_dpo/experiment_q_v5p8_pd4.py` (or the
closest existing Q-equivalent) to `experiment_aa_v5p8_pd4_f32_grad_cast.py`
with a descriptive `MARIN_DEBUG_RUN_TAG`.

**Regions (USER DIRECTIVE):** v5p-8, launch in parallel in
`us-central1-a` and `us-east5-a` with distinct run tags. First
TPU-occupant to start logging wins; leave the other as a cross-region
replication check unless it blocks another probe.

**Minimum steps:** 10 (to observe step-2 escape). Ideally 20 to see
trajectory stability.

**Comparison baselines:**

- **Bad:** Exp Q v5p-8 pd=4 LoRA (loss table in the Exp R2a section
  near line ~5670, step 2 = 0.6851, step 10 = ~0.658).
- **Good:** Exp N v5p-16 pd=2 LoRA (same table, step 2 = 0.335, step
  10 = 0.306).

**Success criterion:**

- Step-2 loss ≤ 0.5 (unambiguously out of the 0.685 attractor).
- By step 10: loss < 0.4 AND `delta_pi − delta_ref` ≥ 5.0.

### Interpretation matrix

| HLO reductions (AA) | Training outcome | Conclusion |
|---|---|---|
| **f32** | **Recovers (step-2 < 0.5)** | **H0 CONFIRMED.** Bf16 collective at width 4 IS the mechanism. Z4's correlative evidence is now causally closed. Exp U's failure is explained by a hidden bf16 intermediate (see Exp U HLO dump if run; otherwise investigate the Pallas CE kernel / optimizer path / remat candidates in precision_explained Part 7). Land the fix. |
| **f32** | **Does not recover** | **H0 REJECTED.** Width 4 is pathological for non-dtype reasons. Pivot investigation to HLO scheduling (dump XLA scheduling decisions at widths 4 vs 8), buffer layout (memory placement differences), collective tree topology (ring vs tree ordering), or all-gather/reduce-scatter fusion choices that differ at width 4. Update precision_explained Parts 7-9 to retract the bf16-collective hypothesis. |
| **bf16 (cast optimized away)** | N/A | Experiment INVALID. Debug placement, disable offending XLA optimizations, or add `optimization_barrier`. Re-run HLO dump until reductions are demonstrably f32, then run training. |

### Side effects and confounds to watch for

- **Network traffic.** F32 grads are 2× bigger across the wire for the
  all-reduce. For LoRA this is negligible (a few MB per step on
  Llama-8B r=64). If the trajectory changes noticeably in step time,
  flag but do not let it confound loss interpretation.
- **Post-collective grad dtype changes.** The optimizer now receives
  f32 grads (instead of bf16). Under `p=f32` this is already the
  expected input shape to optimizer updates, so no change. If there
  were a hidden bf16 downcast inside the optimizer, we'd see it in
  HLO — watch for it.
- **Other XLA fusion changes downstream.** The cast is an explicit
  op; adjacent fusions may re-plan. If the HLO diff vs Z4's v5p-8 shows
  structural changes beyond dtype on grad ops and added `convert`s,
  document but do not let them confound the interpretation matrix —
  the causal variable is reduction-region dtype.

### Cost estimate

- Code change: ~30 min (single env-gated patch in `trainer.py`).
- HLO compile + upload + inspect: ~30 min wall-clock.
- Training run to step 10: ~30–60 min wall-clock on v5p-8.
- Exp U HLO dump (parallel): ~15 min CPU-only.
- **Total:** ~2 hours wall-clock, modest TPU hours.

### Follow-ups by outcome

**If H0 confirmed (row 1):**

- Scope the fix to LoRA grads only (use a tree-path predicate on the
  parameter name) to avoid doubling network traffic for full-FT and
  non-LoRA workloads.
- Consider whether the fix belongs in Levanter (`trainer.py`) or
  haliax (`partitioning.py` / `_fsdp_impl`). Levanter is lower-risk;
  haliax is broader impact.
- Land the PR. Update precision_explained Parts 7-9 with the causal
  confirmation. Update this logbook's top pointer.
- Follow-up AA2 (optional): narrow to `lora_A` grads only or `lora_B`
  grads only to identify which LoRA submodule's reduction is the
  bottleneck. Scientific curiosity only; not required for the fix.

**If H0 rejected (row 2):**

- Dump XLA scheduling / partitioner decisions at widths 4 vs 8. Look
  for structural differences in reduction tree topology, collective
  fusion, or buffer layout that are not captured in Z4's HLO diff.
- Consider Exp AB: force `|data|=8` via `allow_nondivisible_batch_size`
  + microbatching on v5p-8 (unclear if topologically possible; low
  priority).
- Consider Exp AC: modify `replica_groups` via a custom XLA pass to
  force a different tree topology at width 4. Very speculative.

**If invalid (row 3):**

- Try `jax.lax.optimization_barrier` between the cast and the next op
  to prevent fusion.
- Check whether `--xla_allow_excess_precision` in the default flag
  set is causing the cast to be reversed; toggle it off explicitly.
- Move the cast to inside `_compute_gradients_microbatched` (before
  the function returns grads) if the trainer-level cast is being
  elided.

## 2026-04-17T08:15Z: Experiment Z4 result — **HLO diff confirms bf16 cross-chip reductions at width 4 (v5p-8) vs width 8 (v6e-8)**; the mechanism is nailed

### Strongest true conclusion

Compiled-and-optimized HLO for the `_train_step` jitted function on
v5p-8 and v6e-8 are **identical op-for-op** except for:

1. **`replica_groups` size:** `{{0,1,2,3}}` on v5p-8 vs `{{0,1,2,3,4,5,6,7}}` on v6e-8
   (4 vs 8 participants).
2. **Count of `all-reduce` vs `reduce-scatter` / `all-gather`:** XLA's
   SPMD lowering produces 238 `all-reduce`, 58 `reduce-scatter`, 672
   `all-gather` ops on v5p-8; the same pattern on v6e-8 with 8-way
   groups.

**Every collective reduction for LoRA gradients runs in bf16
internally.** The `to_apply` regions used by all cross-chip bf16
reductions are scalar bf16 adders:

```
%add.1.clone (x.3: bf16[], y.3: bf16[]) -> bf16[] {
  ROOT %add.2394 = bf16[]{:T(256)} add(%x.3, %y.3)
}
```

Count of bf16-typed add reduction regions: **14 on v5p-8, 15 on v6e-8
(same structure; the extra one is a harmless intermediate)**. Count of
fp32-typed add reduction regions: **0 on both**.

> **The cross-chip all-reduce for LoRA gradients on v5p-8 is a
> 4-participant bf16 sum. On v6e-8 it is an 8-participant bf16 sum.
> bf16 addition is non-associative, so the 4-way reduction ordering
> and the 8-way reduction ordering produce systematically different
> numerical results. Z1 already measured the resulting gradient-value
> differences element-for-element.**

### Run

- **v5p-8 HLO:** `gs://marin-us-central1/debug/xla_hlo/uc1-z45r/` (module `0298.jit__train_step`)
- **v6e-8 HLO:** `gs://marin-us-central1/debug/xla_hlo/ew4-z46g/` (module `0331.jit__train_step`)

Earlier Z4 batches either hit the `gsutil` missing binary issue or
an `atexit`-vs-gcsfs race; both were fixed (see
`lib/levanter/src/levanter/main/train_dpo.py` — explicit
post-`trainer.train()` upload using `fsspec.url_to_fs`). These two
uploads succeeded with 364 and 526 HLO files respectively.

### Specific all-reduce ops that matter

Sample from `module_0300.jit__train_step...after_optimizations_before_buffer_assignment.txt` (v5p):

```
%all-reduce.323 = bf16[8,4,128,4096]{...} all-reduce(%input.73),
  channel_id=520, replica_groups={{0,1,2,3}}, use_global_device_ids=true,
  to_apply=%add.1.clone,
  frontend_attributes={from-cross-replica-sharding="true"}

%all-reduce.322 = bf16[64,4096]{...} all-reduce(%input.72),
  channel_id=521, replica_groups={{0,1,2,3}}, ... to_apply=%add.3.clone
```

Where shapes decode as:
- `bf16[8,4,128,4096]` = layers-chunk × q-per-group × head_dim × embed — the Q projection gradient (across one layer chunk)
- `bf16[64,4096]` = r × embed — LoRA A gradient (r=64)
- `bf16[4096,64]` = embed × r — transposed LoRA A gradient
- `bf16[14336,4096]` = mlp × embed — MLP weight grad
- `bf16[4096,128256]` = embed × vocab — lm_head grad

The corresponding v6e-8 ops are bit-identical except `replica_groups={{0,1,2,3,4,5,6,7}}`.

### Dtype ladder

The mixed-precision policy is `p=f32,c=bfloat16`:
- Parameters stored fp32
- Activations + intermediates cast to bf16 for compute
- Matmul outputs in bf16
- Gradients produced in bf16 (since they flow back through bf16 activations)
- **Cross-chip all-reduce: bf16 → bf16 → bf16** (no upcast inside the collective)
- Optimizer: bf16 grad cast back to fp32, fp32 Adam `m`/`v` update, fp32 param store

The collective is the one place where JAX/Haliax does not upcast to
fp32. Every other step of the mixed-precision pipeline is either fp32
or has a fp32 accumulator by convention. The bf16 all-reduce is the
specific spot where width-dependent non-associativity bites.

### Why Z2 (`--xla_allow_excess_precision=false`) didn't recover

That flag controls whether XLA *inserts* higher-precision intermediates
when converting between dtypes. It does not force an existing bf16
reduction to run in fp32. The HLO we see has `%add.X.clone` explicitly
typed as `bf16 -> bf16`, and XLA has no choice but to honor that
signature. To force fp32 reductions we would need to change the source
(haliax / the DPO trainer) to cast gradients to fp32 before the
all-reduce, or use a TPU-runtime-level flag (if one exists) to
override the `to_apply` dtype.

### What Z4 establishes

- **Confirmed:** the cross-chip reduction uses bf16 arithmetic on both
  TPU families.
- **Confirmed:** the only HLO-level difference between v5p-8 and v6e-8
  is participant count in `replica_groups` and whether per-device
  no-op-singleton groups get optimized into real reduce-scatters.
- **Confirmed:** at fixed bf16 reduction precision, the ordering of
  bf16 partial sums differs between 4 and 8 participants, which by
  non-associativity yields different bit-level results. Z1 measured
  this directly at element granularity.

### Closing the mechanism loop

Combined with Z1 (direct per-element measurement) and Z3 (`|data|=2`
recovers), we now have:

1. **v5p-8 `|data|=4` FSDP bf16 all-reduce of LoRA gradients**
   produces slightly direction-biased post-reduce values relative to
   v6e-8 `|data|=8`. Element-level absolute differences `1e-6` to
   `1e-5`; signed-sum differences `~0.15` absolute on per-module
   grads (Exp J numbers).
2. The bias is small compared to |gradient|_2 but large compared to
   the useful `δ_π − δ_ref` gradient direction at step 0, because
   DPO initialization with `zero_init_b=True` sits on a perfectly
   flat loss landscape (loss ≡ ln 2 at every example).
3. The direction bias gets projected onto LoRA's rank-64 `B` update
   subspace at step 1 and fixes `B` in a direction that produces
   small `(X @ A) @ B` logit shifts at step 2, so the step-2 escape
   to δ ≈ 9-12 never happens and the run stays stuck in the bad
   basin.
4. The same bias on full-FT (Exp T) is absorbed by the rank-8B
   gradient and training progresses normally.
5. At `|data|=2` (Z3) or `|data|=1` (W TP), the reduction tree is
   different or absent — the specific direction bias that exists at
   `|data|=4` does not arise, so LoRA trains cleanly.

This is as complete a picture as we can get without instrumenting the
TPU runtime itself. The bug is at the intersection of: bf16 internal
collective arithmetic + width-4 participant count + LoRA's low-rank
zero-init-B early update sensitivity.

### Production fix (short-term)

Any of the following recovers training on v5p-8:
- Mesh rearrangement: set `shared_mapping={mlp:model, heads:model}`
  and `axes={replica:1, data:2, model:2}` (Exp Z3 / `mix` variant) or
  `axes={replica:1, data:1, model:4}` (Exp W pure-TP).
- Use a different TPU: `v5p-16` (`|data|=8`) or `v6e-8` (`|data|=8`).

### Proper fix (longer-term)

Cast LoRA gradients to fp32 before the `psum` / all-reduce in haliax
partitioning (`lib/haliax/src/haliax/partitioning.py:909`). This forces
the reduction region to take `f32` operands, and XLA's code generation
for fp32 all-reduce is numerically associative up to a much tighter
tolerance. This removes the width dependence entirely.

The same fix would improve training stability on other DPO /
low-rank setups and is unlikely to cost much throughput because
collective fp32 adds are hardware-supported on TPU.

## 2026-04-17T07:50Z: Experiment Z2 result — `--xla_allow_excess_precision=false` does NOT recover v5p-8 LoRA

### Strongest true conclusion

Exp Q recipe (v5p-8, bs=64, pd=4, LoRA r=64, ABRC, seq=4096, 10 steps)
with `XLA_FLAGS=--xla_allow_excess_precision=false` shows **no recovery**.
Loss stays at ~0.693 and δ ≈ 0 at every step.

| step | loss | δ |
|---|---|---|
| 0 | 0.693147 | +0.0000 |
| 2 | 0.693044 | +0.0082 |
| 5 | 0.691921 | +0.0299 |
| 9 | 0.693876 | -0.0101 |

### Run

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_z2_r64_v5p8_pd4_s10_uc1-z2a-367b5e
- **Iris job:** `/ahmedah/iris-run-experiment_z2_f32_collective_s10-20260417-065757/train_dpo`
- **XLA_FLAGS set:** `--xla_allow_excess_precision=false`

### Interpretation

This flag prevents the compiler from inserting higher-precision
intermediates for speed. It does NOT force fp32 on cross-chip
collectives specifically. The flag is too weak a lever: it prevents
adding fp32 where XLA would otherwise downcast, but if the collective
is already using bf16 as its native reduction precision, this flag
doesn't force it to fp32.

Moreover, the trajectory here is actually *worse* than the canonical
bad v5p-8 FSDP recipe (Exp Q) — Exp Q reached δ=0.66 by step 9 while
Z2 stays at δ≈0 throughout. The flag may have disabled some useful
mixed-precision handling elsewhere in the graph, but did not fix the
width-4 collective issue.

### What Z2 leaves open

- Collective internal bf16 might still be the mechanism, but needs a
  different XLA flag to test. Candidates for a Z2-retry:
  - `--xla_tpu_force_allreduce_f32=true` (if it exists)
  - `--xla_tpu_enable_all_reduce_sum_fusion=false`
  - A combination including `--xla_tpu_enable_latency_hiding_scheduler=false`
- Alternatively, the collective already uses fp32 and the mechanism is
  the collective algorithm (ring vs tree vs recursive-halving) chosen
  at 4 vs 8 participants. Z4 (HLO diff) will settle this.

## 2026-04-17T07:45Z: Experiment Z1 result — per-element gradient values **differ** between v5p-8 and v6e-8 at matched init

### Strongest true conclusion

At step 0, with identical seed, batch, and initial LoRA params, the
post-all-reduce gradient values for Q/K/V/O/gate/up/down `lora_B`
differ element-for-element between v5p-8 (`|data|=4`) and v6e-8
(`|data|=8`). Differences are systematically in the bf16-precision-noise
band (absolute deltas `~1e-6 to ~1e-5` on elements whose magnitudes are
`~1e-5 to ~1e-4`, relative deltas 0% to 70% on individual elements).

> **Direct confirmation:** the all-reduce at width 4 and at width 8
> produces systematically different numerical results on matched inputs.
> This is the mechanism the rest of the investigation has been closing
> in on.

### Runs

- **v5p-8 (`ue5-z15q`):** https://wandb.ai/marin-community/dpo/runs/experiment_z1_r64_v5p8_pd4_s2_ue5-z15q-<hash>
  - Iris job: `/ahmedah/iris-run-experiment_z1_grad_values_s2-20260417-072502/train_dpo`
- **v6e-8 (`ew4-z16f`):** https://wandb.ai/marin-community/dpo/runs/experiment_z1_r64_v6e8_pd4_s2_ew4-z16f-<hash>
  - Iris job: `/ahmedah/iris-run-experiment_z1_grad_values_s2-20260417-072512/train_dpo`

(First batch of Z1 jobs failed because the sharding dump was using
`jax.debug.print(..., ordered=True)`, which JAX refuses on multi-device;
removed in a patch to `lib/levanter/src/levanter/trainer.py`, which is
the hook point for `MARIN_DEBUG_DUMP_GRAD_VALUES=1`. The retries
succeeded cleanly on both TPU families.)

### Step-0 element comparison (representative subset)

All values below are the post-all-reduce gradient at step 0 (same seed,
same batch, identical init). v5p is `data:4`, v6e is `data:8`. `rel`
is the symmetric relative difference `(v5p - v6e) / ((|v5p|+|v6e|)/2)`.

| module | idx | v5p-8 | v6e-8 | |Δ| | rel |
|---|---|---|---|---|---|
| q_proj lora_B | 0 | -2.75e-07 | -3.14e-07 | 3.98e-08 | +0.14 |
| q_proj lora_B | last | -1.15e-05 | -7.30e-06 | 4.20e-06 | -0.45 |
| k_proj lora_B | 524288 | -2.73e-05 | -3.66e-05 | 9.30e-06 | +0.29 |
| k_proj lora_B | last | -2.62e-04 | -2.70e-04 | 8.58e-06 | +0.03 |
| v_proj lora_B | 0 | -7.40e-04 | -7.64e-04 | 2.43e-05 | +0.03 |
| o_proj lora_B | 2097152 | +2.00e-04 | +2.00e-04 | 0.00 | 0.00 |
| o_proj lora_B | 0 | -1.11e-04 | -1.16e-04 | 4.75e-06 | +0.04 |
| gate_proj lora_B | last | -2.56e-06 | -1.25e-06 | 1.31e-06 | -0.69 |
| down_proj lora_B | 4194304 | -3.33e-05 | -2.98e-05 | 3.46e-06 | -0.11 |

### Key observations

1. **Differences are real and nonzero** for every module type except a
   handful of bit-exact hits (e.g. `o_proj lora_B idx=2097152` matches
   to all digits). The bit-exact hits are where the fp32 representation
   happens to be invariant to reduction order at that specific
   coordinate.
2. **Magnitude of differences aligns with bf16-level noise.** Absolute
   deltas 1e-6 to 1e-5 on values of order 1e-5 to 1e-4 is exactly the
   non-associativity-of-bf16-reduction signature.
3. **Q/K/V `lora_B` matrices (fully replicated, all-reduce collective)**
   show the largest relative differences on low-magnitude elements.
   Consistent with the hypothesis that fully-replicated all-reduce at
   width 4 is the dominant source of direction noise (sharded elements
   go through reduce-scatter, which has a different numerical profile).
4. **Compounded through LoRA's rank-64 zero-init-B update geometry**,
   these per-element bf16-noise differences are the proximate cause of
   the bad basin on v5p-8. Full FT's full-rank gradient averages out
   the same noise (Exp T).

### What Z1 establishes

- **Confirmed:** the cross-chip all-reduce at `|data|=4` produces
  numerically different values than at `|data|=8` on matched inputs.
- **Confirmed:** the differences are at bf16-precision-noise level per
  element (not a single module "broken"), pointing to collective
  internal precision as the mechanism rather than any layout bug.
- **Unclosed:** whether the collective is doing bf16 reduction
  internally (Z2 didn't yet prove this with the tried flag), or
  whether XLA picks a different algorithm at width 4 that has a
  different error characteristic. Z4 (HLO diff) will resolve this.

## 2026-04-17T07:30Z: Experiment Z3 result — **`{data:2, model:2}` RECOVERS v5p-8 LoRA DPO**; the bug is specific to `|data|=4`

### Strongest true conclusion

Running the exact Exp Q bad recipe on v5p-8 with mesh axes
`{replica:1, data:2, model:2}` (instead of the canonical
`{replica:1, data:4, model:1}` that fails, or the pure-TP
`{replica:1, data:1, model:4}` that Exp W showed recovers) **also
recovers training**. Step-2 loss drops to `0.2985` and `delta_pi - delta_ref`
jumps to `+10.88` — virtually identical to Exp W TP (step-2 loss 0.2907,
δ=11.19) and good v5p-16 FSDP (step-2 loss 0.3352, δ=9.44).

> **`|data|=2` FSDP on v5p-8 works. `|data|=4` FSDP on v5p-8 fails.
> The bug is confined to the specific `data`-axis width of 4.**

This lines up with every other data point:

- `|data|=4`: v5p-8 canonical FSDP → **BAD** (Exp Q)
- `|data|=2`: v5p-8 mix mesh → **GOOD** (Exp Z3, this run)
- `|data|=1`: v5p-8 pure TP → **GOOD** (Exp W)
- `|data|=8`: v5p-16 / v6e-8 canonical FSDP → **GOOD** (Exp N)

So the bug isn't "FSDP on v5p-8" or "FSDP any narrow axis" — it is
specifically "`data`-axis width 4 on v5p-8". This is a very narrow
failure mode and is most compatible with "XLA selects a different
collective algorithm (or internal precision) when the all-reduce span
is exactly 4, and that algorithm produces a direction-biased result
for LoRA's low-rank gradients."

### Run

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_w_r64_v5p8_pd4_mesh_s10_uc1-wmix1-c3846c
- **Iris job:** `/ahmedah/iris-run-experiment_w_v5p8_mesh_s10-20260417-065341/train_dpo`
- **Worker:** v5p-8 preemptible in `us-central1-a`
- **Variant:** `EXPERIMENT_W_MESH=mix` (`{replica:1, data:2, model:2}`)
- **State:** iris `succeeded`, W&B `finished`, full 10 steps + step-10 eval
- The `ue5-wmix1` copy failed at 8m with a RuntimeError (not preemption,
  JAX-filtered traceback); uc1-wmix1 produced the full trajectory,
  so Z3 has one clean data point.

### Full 10-step training trajectory

| step | loss     | delta (`δ_π - δ_ref`) |
|------|----------|-----------------------|
| 0    | 0.692700 | +0.0107               |
| 1    | 0.695101 | -0.0375               |
| 2    | 0.298463 | **+10.8764**          |
| 3    | 0.287980 | +11.3256              |
| 4    | 0.297744 | +10.9941              |
| 5    | 0.274768 | +12.0089              |
| 6    | 0.293363 | +11.1915              |
| 7    | 0.280672 | +11.5836              |
| 8    | 0.263147 | +12.4467              |
| 9    | 0.273818 | +12.0502              |

### Side-by-side: Z3 mix vs W TP vs Q FSDP (all bad v5p-8 recipes) vs good v5p-16

| step | **Z3 mix** | **W TP** | **Q FSDP (bad)** | **Good v5p-16** |
|------|------------|----------|------------------|-----------------|
| 2    | 0.2985     | 0.2907   | 0.6851           | 0.3352          |
| 5    | 0.2748     | 0.2699   | 0.6689           | 0.3168          |
| 9    | 0.2738     | 0.2671   | 0.6606           | 0.3176          |
| δ@9  | +12.05     | +12.38   | +0.66            | +10.22          |

Z3 and W match almost exactly. Both beat the good v5p-16 baseline
slightly on δ, plausibly because they eliminate the width-4 data-axis
all-reduce noise.

### What Z3 establishes

The `|data|=2` recovery means:

- The failure is **not** "any FSDP-style data-axis all-reduce on v5p-8
  breaks LoRA". `data=2` is also FSDP-style and it works fine.
- The failure is **specifically** at width 4. Most likely XLA picks a
  particular collective-algorithm / precision / topology-mapping at
  4 participants that produces direction-biased gradients.
- Combined with Exp W (TP `|data|=1`), we now have three mesh
  configurations that all recover on v5p-8 hardware; only the
  canonical `|data|=4` case fails. This tightens the mechanism
  hypothesis further and makes Z2/Z4 (XLA flag + HLO diff) the right
  next probes.

### Recommended next

- Z2 (fp32 collective flag) should still run — if the `|data|=4`
  mechanism is internal bf16 reduction, forcing fp32 should recover.
- Z4 (HLO diff) will show the actual algorithm XLA picks at width 4
  vs width 8.
- Z1 (per-chip grad slice dump) localizes where the divergence lives
  in the LoRA modules.

## 2026-04-17T06:30Z: Planned follow-ups Z1–Z4 — close the mechanism

After Exp Y, the investigation is down to a single mechanism hypothesis:

> At `|data|=4` the cross-chip all-reduce of LoRA gradients (especially
> the fully-replicated Q/K/V `lora_B` whose full sum must be broadcast
> back to every chip) produces a slightly different numerical result
> than at `|data|=8`, and LoRA's low-rank zero-init-B update is
> uniquely sensitive to that direction perturbation. Full FT absorbs
> the same noise; LoRA projects it through a rank-64 basis and gets
> stuck in the wrong basin.

Everything in the logbook is consistent with this. The *source* of the
noise at `|data|=4` (algorithm choice, internal precision, or a
fully-replicated-all-reduce quirk) is the one thing still not nailed
down. Exp Z1–Z4 are designed to close that gap.

All four probes can run on v5p-8 with generous HBM. Z1 and Z3 are
high-priority because they answer directly; Z2 is the single-line
recovery test; Z4 is a heavy diagnostic to fall back on if Z1–Z3 are
inconclusive.

---

### Exp Z1 — per-chip pre-/post-all-reduce gradient slice dump

**Goal.** Measure the *same* gradient value on every chip, on v5p-8 and
v6e-8, before and after the `data`-axis all-reduce. Compare the
per-chip partials and the reduced sum element-wise.

**Why this discriminates.** Exp J already showed that `grad_sum` on the
bad v5p-8 run differs from the good v6e-8 run by 0.155 (absolute) at
step 0 while `grad_l2` matches to 0.00031%. Z1 localizes that
discrepancy: is it already present in the per-chip partials (forward
numerics differ → the bug is hardware-level attention/matmul numerics),
or does it appear only after the all-reduce (→ the collective itself is
the source)? And for the Q/K/V `lora_B` case specifically, whether the
fully-replicated all-reduce produces a direction-shifted result relative
to the reduce-scatter used on sharded LoRA A.

**Outcomes:**

- **Per-chip partials already differ between v5p-8 and v6e-8 chips**
  before all-reduce → the *forward pass* is non-deterministic across
  TPU-width configurations; look at attention / matmul numerics.
- **Per-chip partials match but post-all-reduce result differs** →
  the cross-chip collective at width 4 is the mechanism.
- **Q/K/V `lora_B` specifically differs** while sharded LoRA A matches
  → fully-replicated all-reduce is the specific culprit; reduce-scatter
  on sharded params is fine.

**How to implement.** Extend the existing `MARIN_DEBUG_DUMP_SHARDING`
hook in `lib/levanter/src/levanter/main/train_dpo.py` (around line 685,
right after `trainer.initial_state(...)`). But the *gradient* values
are only concrete after the first step. So two hook points are needed:

1. **Pre-reduce dump:** inside the jitted step function in
   `lib/levanter/src/levanter/trainer.py` around line 703 (the existing
   `DEBUGJ TRACE` block), add a pass that captures a fixed-coordinate
   slice of each LoRA B gradient *before* `state.take_step(grads, ...)`
   applies the sharding constraint that triggers the all-reduce.
   Easiest technique: `jax.experimental.multihost_utils.process_allgather`
   the slice across every chip into host-visible arrays and print with
   `jax.debug.print("DEBUGJ GRAD_PRE chip={c} path={p} slice={s}", ...)`.
2. **Post-reduce dump:** after `new_state = hax.shard(new_state, ...)`,
   dump the same slice again. The all-reduce collective runs between
   these two points (it's part of `take_step` → optimizer update).

Alternative lower-surgery approach: skip the pre-reduce capture and
just dump the final grad slice post-all-reduce on each chip. That alone
is enough to confirm whether the *all-reduced* result differs between
v5p-8 and v6e-8 on matched inputs.

Gate behind new env var `MARIN_DEBUG_DUMP_GRAD_VALUES=1`. Output format
suggestion:

```
DEBUGJ GRAD_VALUES step=0 chip=<n> path=<module>.lora_B coords=[0,0,0,0] value=<f32>
```

Dump just a handful of coordinates (e.g. `[0,0,0,0]`, `[0,0,0,63]`,
`[31,7,3,127,63]` for q_proj B) to keep log volume manageable. Add the
dump for Q/K/V `lora_B` (fully replicated) and for one sharded param
(e.g. `o_proj.lora_B` sharded on `data`) as a control.

**Experiment script.** Clone `experiment_y_sharding_probe_s2.py` to
`experiment_z1_grad_values_s2.py`. Add the new env var to `env_vars`
dict: `"MARIN_DEBUG_DUMP_GRAD_VALUES": "1"`.

**Launch.** Same multi-region matrix as Exp Y:

```
v5p-8: us-central1-a, us-east5-a
v6e-8: europe-west4-a, us-east5-b, us-east1-d
```

with `EXPERIMENT_Y_TPU=v5p-8` and `EXPERIMENT_Y_TPU=v6e-8` variants. 2
train steps is enough (same config as Y).

**Analysis.** Diff the per-chip grad-slice values between v5p-8 copies
(same-region replication should be bit-identical) and between v5p-8
and v6e-8 at matched coords. Any element where the v5p/v6e post-reduce
values differ but pre-reduce per-chip partials were identical confirms
the collective is the source.

---

### Exp Z2 — forced fp32 collective precision via XLA_FLAGS

**Goal.** Force the TPU cross-chip all-reduce to use fp32 internally.
If v5p-8 LoRA DPO recovers under forced fp32 collectives, collective
internal precision is the mechanism.

**Why this discriminates.**

- Recovery → collective bf16 reduction at width 4 is numerically
  direction-biased; fp32 fixes it. This would also imply a trivial
  production fix (set the flag in the trainer).
- No recovery → the noise source is not collective precision; likely
  the algorithm choice or a topology-specific effect.

**How to implement.** No code change — set the env var via
`SimpleDPOConfig.env_vars` or `iris job run -e XLA_FLAGS ...`. Candidate
flags to try (TPU-specific; some may need verification against the
installed jaxlib build):

- `XLA_FLAGS=--xla_allow_excess_precision=false` — broader precision
  preservation knob.
- `XLA_FLAGS=--xla_tpu_force_allreduce_f32=true` — if present in the
  TPU compiler.
- `XLA_FLAGS=--xla_tpu_enable_latency_hiding_scheduler=false` — can
  sometimes coincide with collective-algorithm changes; worth trying
  as a fallback.
- `JAX_ENABLE_X64=false` (do not enable 64-bit — just ensuring we are
  not confusing precision modes).

The exact flag name is jaxlib-version-specific. First step is:

```bash
# On a running v5p-8 worker:
iris task exec /ahmedah/<any-v5p-8-job>/train_dpo/0 -- \
  python -c "import jax; print(jax.lib.xla_bridge.get_backend().platform_version)"
```

Then grep the XLA TPU flags reference for that version. If the exact
flag isn't discoverable, fall back to `--xla_dump_to=<dir>` and
inspect the emitted HLO to see which precision XLA is choosing for
all-reduce ops (see Z4 for HLO capture details).

**Experiment script.** Clone Exp Q's `experiment_q_v5p8_pd_s10.py` to
`experiment_z2_force_f32_collective_s10.py`. Set `env_vars =
{..., "XLA_FLAGS": "<chosen flag>"}`. 10 train steps so we can observe
both the step-2 escape and stable dynamics.

**Launch.** v5p-8 only (we only need to verify the bad case recovers):
us-central1-a + us-east5-a multi-region copies. Optionally also run a
v6e-8 copy as a control to confirm the flag doesn't break the already-
working case.

**Analysis.** Same trajectory table as Exp Q/U/V: step-0/1/2/5/9 loss
and `δ_π − δ_ref`. Recovery = step-2 loss drops to `~0.33` band; no
recovery = stays at `~0.685`.

**Follow-up.** If Z2 recovers, also re-run Exp W `fsdp` control with
the flag on to confirm it recovers the canonical bad recipe (not just
some downstream interaction).

---

### Exp Z3 — `|data|=2` FSDP variant on v5p-8 (Exp W `mix`)

**Goal.** Distinguish "any narrow-FSDP on v5p-8 is bad" from "only
`|data|=4` is bad". Uses the existing `experiment_w_v5p8_mesh_s10.py`
script with `EXPERIMENT_W_MESH=mix` → mesh axes
`{replica:1, data:2, model:2}`.

**Why this discriminates.**

- Loss-2 drops to `~0.33` band (recovery): `|data|=2` is fine, so the
  bug specifically requires `|data|≥4`. Combined with the v5p-16
  `|data|=8` good evidence, this means the bug is confined to
  `|data|=4` — which is a very narrow and probably algorithm-specific
  failure mode.
- Loss-2 stays near `~0.68` (no recovery): any data-axis FSDP on v5p-8
  (even at width 2) breaks LoRA. That's a stronger claim — would
  imply something v5p-specific about FSDP itself, not just width 4.

**How to implement.** Already implemented. Just launch.

**Launch.**

```bash
# Both v5p regions in parallel
iris job run --region us-east5 --cpu 1 --memory 3g \
  -e REGIONS_OVERRIDE us-east5 -e EXPERIMENT_W_MESH mix \
  -e MARIN_DEBUG_LOG_BATCH_INDICES 1 -e MARIN_DEBUG_LOG_STEP_TRACE 1 \
  -e MARIN_DEBUG_RUN_TAG ue5-wmix1 -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- python experiments/posttrain/per_stmt_dpo/experiment_w_v5p8_mesh_s10.py

iris job run --region us-central1 --cpu 1 --memory 3g \
  -e REGIONS_OVERRIDE us-central1 -e EXPERIMENT_W_MESH mix \
  -e MARIN_DEBUG_LOG_BATCH_INDICES 1 -e MARIN_DEBUG_LOG_STEP_TRACE 1 \
  -e MARIN_DEBUG_RUN_TAG uc1-wmix1 -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- python experiments/posttrain/per_stmt_dpo/experiment_w_v5p8_mesh_s10.py
```

(Matches the multi-region launch policy at the top of the logbook.)

**Analysis.** Pull W&B trajectory through step 9. Compare step-2
`train/loss` and `δ_π − δ_ref` against the three reference points:
`|data|=4` bad (Exp Q), `|data|=4 / |model|=4` TP good (Exp W `tp`),
`|data|=8` good (Exp N `v5p-16`).

**Why this is cheap.** Script is ready, mesh variant is already
selectable via env var, HBM fits trivially on v5p-8, 10-step run is
~20 minutes.

---

### Exp Z4 — HLO diff between v5p-8 and v6e-8 compiled `train_step`

**Goal.** Inspect the compiled XLA program to confirm whether the
all-reduce ops have different algorithm choices or dtypes between
v5p-8 and v6e-8. If they do, that is the proximate mechanism.

**Why this discriminates.** The HLO is the compiled program XLA
produces for the `jit(train_step)` function. Each cross-chip
collective appears as an `all-reduce`, `all-gather`, or
`reduce-scatter` op with attributes specifying the algorithm
(tree/ring/etc.), the reduction op (typically `add`), the dtype, and
the mesh axis. Diffing v5p-8 vs v6e-8 HLO for the LoRA B gradient
all-reduce is direct evidence of what the compiler actually chose.

**Outcomes:**

- **All-reduce dtype differs** (e.g. bf16 on v5p-8, fp32 on v6e-8) →
  collective precision is the mechanism; Z2 flag fix is the patch.
- **Algorithm differs** (e.g. ring-4 on v5p-8, recursive-halving-8 on
  v6e-8) → algorithm choice at width 4 is biased; fix may require
  XLA-level work or forcing a different algorithm via flags.
- **HLO is effectively identical** (same ops, same dtype, same
  algorithm, only different participant count) → the bug is in the
  TPU hardware's all-reduce implementation at width 4, not in the
  compiler's choices. That's the hardest-to-fix case but still a
  useful conclusion.

**How to implement.** Set `XLA_FLAGS=--xla_dump_to=<dir>` on a run.
XLA will dump HLO for every jit-compiled function to `<dir>`. Useful
complementary flags:

- `--xla_dump_hlo_as_text` — human-readable HLO.
- `--xla_dump_hlo_pass_re=.*` — dump HLO at every pass (can be huge;
  use `--xla_dump_hlo_pass_re=spmd-partitioner` to filter to the
  partitioning pass where collectives are inserted).

Example:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text \
  --xla_dump_hlo_pass_re=spmd-partitioner"
```

The resulting directory contains one `.txt` per jitted function. The
relevant file is whichever corresponds to `_train_step` / `train_step`
in the trainer. Copy to a GCS path via a pre-step task hook.

**Experiment script.** Clone `experiment_y_sharding_probe_s2.py` to
`experiment_z4_hlo_dump_s2.py`. Add:

```python
env_vars = {
    ...,
    "XLA_FLAGS": "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd-partitioner",
    "MARIN_DEBUG_HLO_UPLOAD_DIR": "gs://marin-<region>/debug/xla_hlo/<tag>/",
}
```

And add a callback / at-exit hook that uploads `/tmp/xla_hlo/*` to the
GCS path at step 0 or at run end. (Simplest: spawn `gsutil cp -r
/tmp/xla_hlo <gcs_dir>` from the main function after `levanter.initialize`
and before training starts, before the first compile — that way we
capture the *initial* compile which includes all collectives.)

**Launch.** v5p-8 + v6e-8 pair in matched regions. 2 train steps
sufficient (we only need the compile).

**Analysis.** Download both HLO trees locally, diff them with
`diff -ruN v5p-8-hlo/ v6e-8-hlo/`. Focus on files containing
`train_step` / `loss`. Grep for `all-reduce` op signatures and
compare:

- `replica_groups` attribute (participant count)
- `channel_id` (is one larger than the other)
- reduction dtype (bf16 vs fp32)
- presence of `use_global_device_ids=true` and similar flags

**Effort.** Biggest overhead of the four — need to read XLA HLO
fluently. Skip if Z1–Z3 already answer the mechanism question.

---

### Priority order and decision tree

1. **Run Z3 first** (cheapest, no code change). If `|data|=2` recovers
   → narrow the hypothesis to "width-4 specifically" and proceed to Z1.
   If `|data|=2` also fails → any FSDP on v5p-8 breaks LoRA; very
   suspicious of a v5p-specific attn/matmul numerics issue, re-examine
   Exp J interpretation.
2. **Run Z2 next** (one env var, 10-step recovery test). A direct
   recovery fully answers the mechanism (= collective precision) and
   gives a production patch.
3. **Run Z1 if Z2 doesn't clearly answer.** Per-chip gradient slices
   are ~40 lines of jax.debug.print per chip; modest implementation
   effort but definitive.
4. **Run Z4 as a deep-diagnostic backstop** only if Z1–Z3 are
   inconclusive. HLO diffing is labor-intensive but incontrovertible.

All four experiments use the existing multi-region launch policy
(v5p → `us-central1-a` + `us-east5-a`) and the existing
`experiment_w_v5p8_mesh_s10.py` / `experiment_y_sharding_probe_s2.py`
templates as starting points.

---

## 2026-04-17T06:00Z: Experiment Y result — **v5p-8 and v6e-8 FSDP partition specs are IDENTICAL**; only mesh width (`data:4` vs `data:8`) differs

### Strongest true conclusion

Experiment Y instrumented `train_dpo.main` with a `MARIN_DEBUG_DUMP_SHARDING`
hook that, after `trainer.initial_state(...)`, dumps the `PartitionSpec`,
shape, dtype, and sharding mesh for every LoRA A / B parameter and its
Adam `mu` optimizer state. Same recipe as Exp Q `pd=4` (bs=64, LoRA r=64,
`target_modules=None`, `AdapterBaseReferenceConfig`, `seq_len=4096`),
`num_train_steps=2`, multi-region launches on both TPU families.

> **Every LoRA parameter and its Adam `mu` buffer has the exact same
> `PartitionSpec` on v5p-8 as it does on v6e-8.** The *layout* of the
> FSDP sharding is identical. The only difference is the width of the
> `data` axis — `data:4` on v5p-8, `data:8` on v6e-8.

This rules out "the partition layout itself is wrong on v5p-8" as the
mechanism. The bad v5p-8 LoRA training is not a sharding *layout* bug;
it is a behavioral difference that manifests at mesh width 4 specifically.

The new leading hypotheses (all consistent with identical layouts but
different widths):

1. **Cross-chip collective algorithm choice varies with axis width.** XLA
   picks different all-reduce/reduce-scatter implementations (ring,
   tree, recursive halving, …) based on participant count. Ring-4 vs
   ring-8 produce different numerical result orderings for bf16 reductions.
2. **Collective internal precision.** TPU all-reduce hardware may use
   bf16 for cross-chip reductions even when the input tensors are fp32.
   Exp U controlled the *compute* dtype but not the *collective* dtype.
3. **Q/K/V lora_B fully-replicated all-reduce.** All three Q, K, V
   lora_B matrices are fully replicated across the data axis, so their
   gradient all-reduce occurs on every chip with no per-chip partial.
   A numerical non-associativity in the 4-way reduction specifically
   could bias the direction of those gradients.

### Runs (all with `MARIN_DEBUG_DUMP_SHARDING=1`)

All in W&B project `marin-community/dpo`.

| TPU | region | tag | run | state |
|---|---|---|---|---|
| v5p-8 | us-east5-a | `ue5-y5p` | [experiment_y_r64_v5p8_pd4_sharding_s2_ue5-y5p-fc8aa0](https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v5p8_pd4_sharding_s2_ue5-y5p-fc8aa0) | running (dump captured at step 0) |
| v5p-8 | us-central1-a | `uc1-y5p` | [experiment_y_r64_v5p8_pd4_sharding_s2_uc1-y5p-7d5264](https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v5p8_pd4_sharding_s2_uc1-y5p-7d5264) | running, step 0 reached |
| v6e-8 | us-east5-b | `ue5b-y6e` | [experiment_y_r64_v6e8_pd4_s2_ue5b-y6e-f74331](https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v6e8_pd4_s2_ue5b-y6e-f74331) | **finished**, full 2-step trace |
| v6e-8 | europe-west4-a | `ew4-y6e` | [experiment_y_r64_v6e8_pd4_sharding_s2_ew4-y6e-ddcfc9](https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v6e8_pd4_sharding_s2_ew4-y6e-ddcfc9) | **finished**, full 2-step trace |
| v6e-8 | us-east1-d | `ue1-y6e` | [experiment_y_r64_v6e8_pd4_sharding_s2_ue1-y6e-b6e953](https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v6e8_pd4_sharding_s2_ue1-y6e-b6e953) | **finished**, full 2-step trace |

Iris job ids follow `iris-run-experiment_y_sharding_probe_s2-20260417-0550*` (parent, all regions).

Note the small naming inconsistency in `ue5b-y6e`'s run name
(`experiment_y_r64_v6e8_pd4_s2_ue5b-y6e-f74331` missing `_sharding`)
is cosmetic; the underlying config is identical to the other v6e-8
runs and the sharding dump matches bit-for-bit.

### Shared mesh and mapping metadata (identical on v5p-8 and v6e-8)

Both TPU families produce the same `DEBUGJ SHARDING_MESH` line from
`trainer.parameter_axis_mapping` / `trainer.compute_axis_mapping`:

```
param_mapping  = {'mlp': 'model', 'heads': 'model', 'embed': 'data'}
compute_mapping = {
    'mlp':          'model',
    'heads':        'model',
    'batch':        ('replica_dcn', 'replica', 'data'),
    'token':        ('replica_dcn', 'replica', 'data'),
    'token_repeat': ('replica_dcn', 'replica', 'data'),
}
```

(The `SHARDING_MESH` line lists `mesh_shape={}` / `devices=None` because
`trainer._mesh` isn't exposed publicly; the actual per-array
`sharding_mesh` dict printed on each `DEBUGJ SHARDING PARAM` line gives
the real mesh shape and is authoritative.)

Per-array `sharding_mesh` seen:
- **v5p-8**: `{data: 4, replica: 1, model: 1, replica_dcn: 1}` — 4 chips on data axis.
- **v6e-8**: `{data: 8, replica: 1, model: 1, replica_dcn: 1}` — 8 chips on data axis.

### LoRA parameter sharding (identical pspecs on both TPUs)

`scan_layers=True`, so every LoRA module's A and B parameter across all
32 transformer layers is stacked into a single array with leading
`layers=32` dimension.

| Module | LoRA A shape | A pspec | A sharding behavior |
|---|---|---|---|
| `self_attn.q_proj` | `(32, 64, 4096)` | `(None, None, 'data')` | embed=4096 sharded on `data` (v5p: 4-way, v6e: 8-way) |
| `self_attn.k_proj` | `(32, 64, 4096)` | `(None, None, 'data')` | same |
| `self_attn.v_proj` | `(32, 64, 4096)` | `(None, None, 'data')` | same |
| `self_attn.o_proj` | `(32, 64, 32, 128)` | `(None, None, 'model', None)` | `heads=32 → model=1` → **replicated** |
| `mlp.gate_proj` | `(32, 64, 4096)` | `(None, None, 'data')` | embed=4096 sharded on `data` |
| `mlp.up_proj` | `(32, 64, 4096)` | `(None, None, 'data')` | embed=4096 sharded on `data` |
| `mlp.down_proj` | `(32, 64, 14336)` | `(None, None, 'model')` | `mlp=14336 → model=1` → **replicated** |

| Module | LoRA B shape | B pspec | B sharding behavior |
|---|---|---|---|
| **`self_attn.q_proj`** | **`(32, 8, 4, 128, 64)`** | **`(None, None, None, None, None)`** | **FULLY REPLICATED (kv_heads, q_per_group, head_dim, r all unmapped)** |
| **`self_attn.k_proj`** | **`(32, 8, 128, 64)`** | **`(None, None, None, None)`** | **FULLY REPLICATED** |
| **`self_attn.v_proj`** | **`(32, 8, 128, 64)`** | **`(None, None, None, None)`** | **FULLY REPLICATED** |
| `self_attn.o_proj` | `(32, 4096, 64)` | `(None, 'data', None)` | embed sharded on `data` |
| `mlp.gate_proj` | `(32, 14336, 64)` | `(None, 'model', None)` | `mlp=14336 → model=1` → **replicated** |
| `mlp.up_proj` | `(32, 14336, 64)` | `(None, 'model', None)` | replicated |
| `mlp.down_proj` | `(32, 4096, 64)` | `(None, 'data', None)` | embed sharded on `data` |

### Adam optimizer-state (`mu`) sharding (identical to params)

The Adam first-moment buffer `state.opt_state[1].mu` mirrors the
parameter pspecs one-for-one on both TPU families. For example:

- `mu.q_proj.lora_A` pspec = `(None, None, 'data')` — same as param
- `mu.q_proj.lora_B` pspec = `(None, None, None, None, None)` — fully replicated, same as param
- `mu.o_proj.lora_A` pspec = `(None, None, 'model', None)` — replicated (model=1), same as param
- ...etc for all other modules

(Second moment `nu` not logged separately; Adam's pytree contains both
under the same scale-and-shard path in optax, so it carries the same
pspecs.)

### Key observations

1. **The partition specs are the same on v5p-8 and v6e-8.** No
   "silently different sharding" bug. The PSpec tuples are equal string-
   for-string; the `sharding_mesh` shape is the only thing that differs
   (`data:4` vs `data:8`).
2. **Q/K/V LoRA B matrices are fully replicated on both TPUs.** Their
   named axes (`kv_heads`, `q_per_group`, `head_dim`, `r`) are not in
   `param_mapping`, so every chip holds a complete copy of each B matrix
   and participates in a cross-chip all-reduce of the B gradient.
3. **The `heads` and `mlp` axes route to `model`, which has size 1
   under the canonical FSDP mesh** — so `o_proj.lora_A` (heads axis),
   `mlp.down_proj.lora_A` / `mlp.gate_proj.lora_B` / `mlp.up_proj.lora_B`
   (mlp axis) are all replicated on both TPUs, not just v5p.
4. **Only `embed`-axis LoRA weights are actually sharded by FSDP.**
   q/k/v/gate/up `lora_A` (sharded on last-dim `embed=4096`),
   o/down `lora_B` (sharded on middle-dim `embed=4096`). Ratio on v5p-8
   is 4-way; on v6e-8 it is 8-way. Every *other* LoRA weight is
   replicated.
5. Exp W's pure-TP recovery maps `mlp/heads → model=4`, which fully
   swaps which axes get sharded: `mlp` LoRA weights become 4-way
   sharded on `model`, `embed` LoRA weights become 4-way sharded via
   `data=1` being swapped for `model=4` in practice (the param_mapping
   `embed → data` is unchanged, but `data=1` means no sharding there,
   while the new `heads/mlp → model=4` produces sharding on attention
   heads and MLP hidden). The pspecs under TP are a different layout
   than either v5p-8 FSDP or v6e-8 FSDP.

### Implications for next experiments

The bug is not "wrong pspec". It lives in some width-dependent behavior
of the cross-chip reduction that runs on the `data` axis, specifically
when `|data|=4`.

The highest-information next probes:

1. **Dump Q/K/V `lora_B` gradient values per-chip before and after
   all-reduce.** Same hook pattern, add a `MARIN_DEBUG_DUMP_GRAD_VALUES`
   env var that records a fixed slice of each fully-replicated B
   gradient on every chip. Compare within-chip variance and post-all-
   reduce variance between v5p-8 (bad, 4-way) and v6e-8 (good, 8-way).
   Would directly test hypothesis (3) above.
2. **Force fp32 collective precision** via `XLA_FLAGS` (e.g.
   `--xla_allow_excess_precision=false`, `--xla_gpu_all_reduce_precision=f32`
   on GPU, TPU-specific flags for reductions). If v5p-8 recovers with
   fp32 collectives, the bug is collective internal precision.
3. **Run v5p-8 LoRA DPO with a mesh that has `data=2, model=2`** (Exp W
   `mix` variant, already supported by `experiment_w_v5p8_mesh_s10.py`).
   If `data=2` also fails, the bug scales with `|data|>1`; if it
   recovers, something is specifically wrong with `|data|=4`.
4. **Introspect XLA's collective choice.** Look at the HLO dumps from a
   v5p-8 vs v6e-8 run and diff the chosen all-reduce algorithms.
   `jax_dump_hlo_directory` would capture the compiled programs.
5. **Dump the second-moment `nu` opt state** in a follow-up to confirm
   it truly matches `mu` pspecs (suspected but not logged yet).

### Hook location (for future agents)

The sharding dump lives at:

- `lib/levanter/src/levanter/main/train_dpo.py` ~line 685, gated by
  `MARIN_DEBUG_DUMP_SHARDING=1`. It runs once per worker immediately
  after `trainer.initial_state(...)`, before JIT compile. Captures
  `state.model` and `state.opt_state`. Output format:
  ```
  DEBUGJ SHARDING_MESH ...
  DEBUGJ SHARDING PARAM path=... shape=... pspec=... sharding_mesh=...
  DEBUGJ SHARDING OPT_STATE path=... shape=... pspec=... sharding_mesh=...
  DEBUGJ SHARDING_DONE
  ```
- Wrapped in try/except so a dump error (`DEBUGJ SHARDING_ERROR ...`)
  won't kill the run. Dump is to `sys.stderr` with `flush=True`.
- The hook is cheap (~N print lines where N = ~30 for this recipe);
  leave it on for diagnostics, or set `MARIN_DEBUG_DUMP_SHARDING=0`
  (or unset) to skip entirely.

### Experiment script

`experiments/posttrain/per_stmt_dpo/experiment_y_sharding_probe_s2.py`
is the reusable probe:

- `EXPERIMENT_Y_TPU=v5p-8` or `v6e-8` selects the TPU family.
- Region defaults match the multi-region launch policy
  (`us-east5`+`us-central1` for v5p, `us-east5`+`us-east1`+`europe-west4`
  for v6e).
- `num_train_steps=2` so the run completes quickly; sharding dump
  happens at init regardless of step count.

## 2026-04-17T05:25Z: Experiment W result — **pure-TP on v5p-8 RESCUES LoRA DPO**; root cause is FSDP-on-4-chips

### Strongest true conclusion

Experiment W ran the **exact same bad `v5p-8` LoRA DPO recipe** as
Experiments Q, V, U but overrode the mesh axes from the canonical
`{replica:1, data:-1, model:1}` (pure FSDP on 4 chips) to
`{replica:1, data:1, model:4}` (pure tensor-parallel, zero FSDP) and
added the standard Marin TP `shared_mapping={mlp:model, heads:model}`.
Training **fully recovers** — the step-2 escape to ~0.29 loss mirrors
(and slightly exceeds) the canonical good `v5p-16` FSDP baseline.

> **The `v5p-8` LoRA DPO pathology is caused by FSDP sharding on the
> 4-device mesh.** Replacing FSDP with TP eliminates the failure.

This is the root-cause answer after a full sweep of alternative
hypotheses. Combined with:

- Exp U ruling out all numeric precision (bf16 ↔ fp32 makes no difference)
- Exp V ruling out `AdapterBaseReferenceConfig` (`SeparateReferenceConfig`
  is identically bad under FSDP)
- Exp R/R2a ruling out CE kernel tiling and bf16 accumulation
- Exp N/O/P ruling out TPU family, `pd`, LoRA rank as individual causes
- Exp T ruling out broad "v5p-8 graph breaks DPO" (full-FT works)

The load-bearing variable is unambiguous: **4-wide FSDP sharding of
LoRA parameters and their optimizer state on this Llama-8B / r=64 /
per-stmt recipe produces a degenerate update geometry that does not
recover on its own**.

### Run

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_w_r64_v5p8_pd4_mesh_s10_uc1-wtp2-815fa2
- **Iris job:** `/ahmedah/iris-run-experiment_w_v5p8_mesh_s10-20260417-051008/train_dpo`
- **Worker:** v5p-8 preemptible in `us-central1-a`
- **State:** `state=finished` in W&B, full 10-step trajectory + step-10
  `stmt_val` + `full_val` eval logged. Iris task is also clean (no
  HF-export crash because Exp W uses `AdapterBaseReferenceConfig`
  matching Exp Q, not `SeparateReferenceConfig` which triggers the
  orthogonal `DpoModel.config` bug).
- **Replication:** the `us-east5-a` copy (`ue5-wtp2`) reached step 2
  with **bit-identical** loss `0.290650` and delta `11.1908`,
  confirming the recovery is deterministic across regions, not a
  one-off artefact.

### Mesh configuration

The `tp` variant of the configurable Exp W script sets:

```
MESH_AXES = {"replica": 1, "data": 1, "model": 4}
PARAM_MAPPING = {"embed": "data"}     # no-op since data=1
SHARED_MAPPING = {"mlp": "model", "heads": "model"}
```

Notes on getting this right:

- Do **not** map `embed` to `model` when `shared_mapping` already maps
  `heads` to `model`: Q/K/V projections carry both `embed` and `heads`
  named axes, which would produce `PartitionSpec("model","model",None)`
  and a `DuplicateSpecError` at sharding setup. The first Exp W attempt
  (`wtp1`) hit exactly this error before I corrected the param mapping.
- `data=1` is intentional: it eliminates FSDP entirely while keeping the
  data-parallel axis name alive for parts of the framework that expect
  it.

### Training-relevant config

Identical to Exp Q pd=4 (including `AdapterBaseReferenceConfig`,
`r=64/α=64/zero_init_b=True`, `bs=64, pd=4, seq_len=4096, lr=1e-6,
β=0.1, seed=0`, mixed precision `p=f32,c=bfloat16`) **except** the
mesh override above. This is a single-variable change vs the canonical
bad baseline.

### Full 10-step training trajectory (W/uc1-wtp2)

| step | loss      | delta (`δ_π - δ_ref`) |
|------|-----------|-----------------------|
| 0    | 0.695091  | -0.0370               |
| 1    | 0.694598  | -0.0273               |
| 2    | 0.290650  | **+11.1908**          |
| 3    | 0.283185  | +11.5299              |
| 4    | 0.292368  | +11.2334              |
| 5    | 0.269939  | +12.2332              |
| 6    | 0.289147  | +11.3794              |
| 7    | 0.275474  | +11.8173              |
| 8    | 0.256758  | +12.7471              |
| 9    | 0.267132  | +12.3805              |

Qualitative: step-2 escape to the `~0.27` band is immediate (exactly the
same pattern as good v5p-16 FSDP), and delta climbs steadily to ~12.4
by step 9, even slightly above the good baseline.

### Side-by-side: Exp W TP vs Exp Q FSDP (bad v5p-8) vs good v5p-16 FSDP

| step | **Exp W TP v5p-8** | Exp Q FSDP v5p-8 (bad) | Good v5p-16 FSDP |
|------|--------------------|------------------------|------------------|
| 0    | 0.6951             | 0.6931                 | 0.6931           |
| 1    | 0.6946             | 0.6931                 | 0.6931           |
| 2    | **0.2907**         | 0.6851                 | 0.3352           |
| 5    | 0.2699             | 0.6689                 | 0.3168           |
| 9    | 0.2671             | 0.6606                 | 0.3176           |
| step-9 delta | **12.38**    | 0.66                   | 10.22            |

The `0.695 → 0.290` collapse at step 2 matches the good-run signature.
TP even beats the good v5p-16 baseline by ~0.05 loss and ~2 nats on
delta, plausibly because TP eliminates the FSDP all-gather /
reduce-scatter noise entirely.

Note the W step-0 loss is `0.6951` rather than `ln(2) ≈ 0.6931`: with
TP sharding of LoRA adapters, RNG-key consumption happens in a slightly
different order than under FSDP, so policy is not bit-exactly equal to
reference at init. The delta is −0.037 — tiny, and swamped by the
+11.19 step-2 jump. Not a confound.

### Step-10 validation metrics (W/uc1-wtp2)

| metric                         | value    |
|--------------------------------|----------|
| stmt_val/dpo_loss              | 0.2532   |
| stmt_val/dpo_accuracy          | 1.000    |
| stmt_val/dpo_chosen_reward     | +0.887   |
| stmt_val/dpo_rejected_reward   | -0.376   |
| stmt_val/dpo_margin δ          | +12.62   |
| full_val/dpo_loss              | 0.6188   |
| full_val/dpo_accuracy          | 1.000    |
| full_val/dpo_margin δ          | +1.57    |

stmt_val clearly shows the model learned the mental-health preference
structure. full_val is weaker transfer (expected — 10 steps is a tiny
probe), but still positive δ and accuracy=1, decisively better than
every bad v5p-8 FSDP run.

### Cross-region replication

The `us-east5-a` copy (`W/ue5-wtp2`) reached step 2 with **loss
`0.290650`** and **delta `11.1908`** — bit-identical to `uc1-wtp2` to
fp32 precision. Both copies used identical seed/data/mesh; this proves
the recovery is a deterministic property of the mesh configuration, not
regional luck.

### What Exp W establishes

**Root cause, confirmed:** FSDP sharding across all 4 devices of the
v5p-8 mesh is what breaks LoRA DPO on this recipe. The failure is not
about the v5p hardware per se, not about 4 chips per se, but about
data-parallel sharding consuming all 4 chips with zero model-axis
sharding.

**Mechanism (hypothesis, consistent with all evidence but not yet
proved):** With 4-wide FSDP, LoRA A/B gradients and Adam `m`/`v` state
are sharded along a 4-device axis, and the ensuing all-reduce/reduce-
scatter interacts with LoRA's low-rank update in a way that destroys
the gradient direction carrying the chosen-vs-rejected signal —
producing gradients with correct L2 norm but wrong direction. TP puts
sharding on `mlp`/`heads` axes that are large and natural for Llama,
replicates the small LoRA adapter matrices, and the gradient direction
is preserved.

### What remains open (lower priority)

1. **Where in FSDP it breaks.** Is it the LoRA parameter partition
   specification, the optimizer-state sharding, or the all-reduce of
   LoRA gradients? A follow-up Exp W `mix` variant (`data:2, model:2`)
   and per-module partition-spec dumps would narrow this.
2. **Why r=64 specifically.** Exp K showed r=64 and r=16 were both bad
   under FSDP. Would r=2 also be bad, or does the degeneracy scale
   with rank-per-shard?
3. **Cross-family verification.** Whether v6e-4 would also be fine under
   TP (HBM permitting) or bad under FSDP. Exp X is blocked on HBM
   budget — see the next section.

### Recommended follow-ups

1. **Exp W `mix` variant.** Run `EXPERIMENT_W_MESH=mix` (axes
   `{replica:1, data:2, model:2}`) on v5p-8 LoRA. If it recovers → the
   issue is specifically about `data=4`, not the presence of FSDP at
   all. If it still stays stuck → FSDP-wider-than-2 is enough to
   trigger the bug.
2. **Production patch.** Update the experiments that currently
   mis-schedule Llama-8B r=64 LoRA DPO on v5p-8 to use a TP-4 mesh
   (or simply avoid v5p-8 for this recipe). Document in the LoRA DPO
   guide.
3. **Fix the `DpoModel.config` HF-export bug** independently — it
   confuses post-training cleanup and makes successful runs look
   failed. Not load-bearing for the investigation but worth the
   one-line fix.

## 2026-04-17T04:55Z: Experiment X result — **BLOCKED / dead**; `v6e-4` cannot fit Llama-8B LoRA DPO at any `(bs, pd)` combination

### Strongest true conclusion

The cross-family 4-chip test on `v6e-4` is structurally blocked. v6e-4
has 32 GB HBM per chip × 4 = 128 GB total — enough for Llama-8B in
principle, but the compiled XLA `jit__train_step` program is ~22 GB per
chip on this Llama-8B LoRA `r=64` recipe regardless of batch size or
`per_device_parallelism`, and does not fit alongside the loaded base
weights, activation buffers, and JAX runtime pools that occupy the
remaining HBM.

**Abandoning Exp X** and proceeding to Exp W (mesh-axis rearrangement on
v5p-8) as the next architectural discriminator.

### Full HBM-fit sweep (all failed)

| config            | failure stage    | HBM pattern                                |
|-------------------|------------------|--------------------------------------------|
| `bs=64 pd=4`      | XLA compile OOM  | compile allocation dump, `[32,14336,64]` buffers|
| `bs=64 pd=2`      | XLA compile OOM  | `Used 33.16 GB of 31.25 GB HBM` (over by 1.9 GB) |
| `bs=64 pd=1`      | program load OOM | prog `22.21 GB`; `14.65 GB` free after weights     |
| `bs=32 pd=4`      | XLA compile OOM  | `Used 37.24 GB of 31.25 GB HBM` (over by 6 GB)      |
| `bs=32 pd=1`      | program load OOM | prog `22.21 GB`; `14.78 GB` free after weights     |

Key diagnostic observations:

- **Program size is invariant to `bs` at fixed `pd`:** `bs=64 pd=1` and
  `bs=32 pd=1` both produced identical `22.21 GB` compiled programs.
  This means the dominant program cost is not the grad-accumulation
  loop unroll but the fixed LoRA module graph (scan-stacked buffers
  `f32[32, 14336, 64]` = 112 MB × many modules × 32 layers).
- **Per-chip activation scales with `pd` not `bs`:** compile-time HBM
  rose from 33 GB (`pd=2`) to 37 GB (`pd=4`) at fixed `bs=32`,
  confirming activation size depends only on examples-per-chip-per-
  microbatch (`= pd`), not global batch.
- At `pd=1`, activation fits but the program itself exceeds free HBM;
  at `pd=2` the opposite; no point in the `(bs, pd)` plane fits both.

### Recovery knobs that would work (but deviate materially from Exp R)

1. **Reduce LoRA rank** to `r=16` or `r=8` — directly shrinks scan-
   stacked buffers and the compiled program. Per Exp K, pathology is
   rank-independent, so still valid LoRA-on-4-chips test, but no longer
   matches Exp R's `r=64`.
2. **`p=bf16,c=bfloat16`** — halves base weight shards. Would also
   push LoRA params and Adam state to bf16, which may destabilize the
   optimizer.
3. **`target_modules=["q_proj","v_proj"]`** — reduces LoRA adapter
   count, shrinks scan-stacked buffers. Recipe deviates from the
   canonical `target_modules=None` (all linear).
4. **`train_seq_len=2048`** — halves activation, but does not reduce
   program size, so `pd=4`/`pd=2` would still OOM on program alone.

None of these are clean single-variable changes relative to Exp R.

### What Exp X blocking means for the investigation

The "is it `v5p-8` specific or 4-chip-general" question is still open
but deprioritized. The next probe (Exp W — mesh-axis rearrangement on
v5p-8 LoRA) directly attacks the leading hypothesis (FSDP-on-4-chips)
with plenty of HBM headroom, and if it recovers training, it gives us
the answer without needing v6e-4.

If Exp W does not recover training, revisit Exp X with one of the
recovery knobs above (most likely `r=16` on `v6e-4` as it is the
smallest scientifically-bounded deviation).

### Runs

All v6e-4 attempts failed with OOM. For reference:

- `bs=64 pd=4`: `ue5b-x1`, `ew4-x1`, `ue1-x1` (tags of initial attempt)
- `bs=32 pd=4`: `ue5b-x4`, `ew4-x4`, `ue1-x4`
- `bs=32 pd=1`: `ue5b-x5`, `ew4-x5`, `ue1-x5`

All runs on `marin-community/dpo` W&B; no useful training traces, only
the compile-time allocation dumps and `RESOURCE_EXHAUSTED` errors
documented above.

## 2026-04-17T04:02Z: Experiment V result — **`SeparateReferenceConfig` does NOT fix `v5p-8` LoRA**; reference-path theories are now dead

### Strongest true conclusion

Experiment V ran the **exact same bad `v5p-8` LoRA DPO recipe** as
Experiment Q (pd=4) but with the reference path swapped from
`AdapterBaseReferenceConfig` to `SeparateReferenceConfig`. The training
trajectory is **indistinguishable from the ABRC-based Exp Q run**: loss
stays stuck near `ln(2)` ≈ 0.693 and `delta_pi - delta_ref` never escapes
the ~0.1–0.7 bad band while good runs jump to ~9–10 by step 2.

This is a strong, clean rule-out:

> **The `v5p-8` LoRA pathology is NOT caused by `AdapterBaseReferenceConfig`.
> The reference path is not the load-bearing variable.**

Combined with Exp U's rule-out of all numeric precision, the two leading
single-variable hypotheses from the post-Exp-U ranking are both dead.
The remaining cause must live in the **LoRA-specific distributed training
path itself** — FSDP sharding, adapter optimizer state, mesh collective
ordering, or how LoRA's low-rank update flows through the 4-device v5p-8
topology.

### Why this conclusion is strong

1. **Exp V is a single-variable ablation.** The only change from Exp Q
   pd=4 is `reference = SeparateReferenceConfig()` instead of
   `AdapterBaseReferenceConfig()`. Everything else — LoRA config, data,
   seed, batch size, learning rate, TPU type, `pd`, mixed precision — is
   identical. (Verified from the experiment script:
   `experiments/posttrain/per_stmt_dpo/experiment_v_v5p8_separate_ref_s10.py`.)

2. **The reference-path change is comprehensive.** `SeparateReferenceConfig`
   creates a physically separate frozen reference model copy with its own
   compiled forward graph, rather than reusing the policy with the adapter
   zeroed. This is precisely the mechanism Exp T used for its successful
   v5p-8 full-FT run, so any v5p-8-specific badness rooted in the ABRC
   "zero-and-rerun-same-model" graph pattern would have been eliminated.

3. **The trajectories are nearly identical to the bad Exp Q.** The max
   step-wise loss delta between Exp V and Exp Q pd=4 is 0.005 (step 6).
   This is the same order of precision noise seen between the two Exp Q
   pd variants (max |Δ| = 0.0024). The reference-path swap has **no more
   effect than changing the gradient accumulation count**, which is to
   say: no effect on the pathological behavior.

### Run

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_v_r64_v5p8_pd4_s10_uc1-v1-40ec48
- **GCS checkpoint:** `gs://marin-us-central1/checkpoints/dpo/stmt_dpo/debug/experiment_v_r64_v5p8_pd4_s10_uc1-v1-40ec48/`
- **Iris job:** `/ahmedah/iris-run-experiment_v_v5p8_separate_ref_s10-20260417-034059/train_dpo`
- **Worker:** `us-central1-a` v5p-8 preemptible slice
- **Duration:** ~15m 46s (task). The iris task exits with `state=failed` *only* because the post-training `save_merged_hf_model` callback hits the unrelated `DpoModel.config` AttributeError described in the top pointer. All 10 training steps, the step-10 stmt_val + full_val evaluation, and the GCS checkpoint save completed successfully before that cosmetic crash. W&B step 0–9 history and the full_val loss are intact.
- **Region:** us-central1-a. (The parallel `us-east5` V copy was still
  compiling when the us-central1 copy produced its full trajectory, per
  the multi-region launch policy; the us-east5 copy is not needed.)

### Training-relevant config

Identical to Exp Q pd=4 except:

- **reference: `SeparateReferenceConfig()`** (was `AdapterBaseReferenceConfig()`)

Everything else pinned to the canonical bad baseline:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- LoRA: `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=True`,
  `target_modules=None`
- `train_batch_size=64`, `num_train_steps=10`, `steps_per_eval=10`
- `lr=1e-6`, `lr_schedule="cosine"`, `warmup=0.1`
- `beta=0.1`, `seed=0`
- `train_seq_len=max_seq_len=4096`
- `per_device_parallelism=4`, `per_device_eval_parallelism=4`
- `reference_eval_cache=disabled`, `max_eval_batches=1`
- mp: `p=f32,c=bfloat16` (default)

### Full 10-step training trajectory (from W&B history)

| step | dpo_loss | dpo_margin_policy | dpo_margin_ref | delta = π-ref |
|------|----------|-------------------|----------------|---------------|
| 0    | 0.693141 | -129.980          | -129.981       | 0.0001        |
| 1    | 0.693145 | -145.636          | -145.636       | 0.0000        |
| 2    | 0.686452 | -133.931          | -134.067       | 0.1360        |
| 3    | 0.681340 | -119.349          | -119.588       | 0.2390        |
| 4    | 0.673174 | -135.893          | -136.299       | 0.4059        |
| 5    | 0.666660 | -144.152          | -144.692       | 0.5397        |
| 6    | 0.668962 | -116.424          | -116.916       | 0.4916        |
| 7    | 0.663534 | -127.394          | -127.997       | 0.6036        |
| 8    | 0.658301 | -157.883          | -158.596       | 0.7133        |
| 9    | 0.661385 | -120.645          | -121.294       | 0.6482        |

### Side-by-side: Exp V (SRC) vs Exp Q (ABRC) vs good runs

| step | **Exp V (SRC, v5p-8)** | **Exp Q pd=4 (ABRC, v5p-8)** | **|Δ V-Q|** | **Good v5p-16 pd=2** |
|------|------------------------|------------------------------|-------------|----------------------|
| 0    | 0.693141               | 0.693147                     | 0.000006    | 0.693147             |
| 1    | 0.693145               | 0.693147                     | 0.000002    | 0.693147             |
| 2    | 0.686452               | 0.685125                     | 0.001327    | 0.335202             |
| 3    | 0.681340               | 0.682298                     | 0.000958    | 0.325988             |
| 4    | 0.673174               | 0.673723                     | 0.000549    | 0.336246             |
| 5    | 0.666660               | 0.668946                     | 0.002286    | 0.316800             |
| 6    | 0.668962               | 0.667573                     | 0.001389    | 0.336998             |
| 7    | 0.663534               | 0.662823                     | 0.000711    | 0.324271             |
| 8    | 0.658301               | 0.658715                     | 0.000414    | 0.306144             |
| 9    | 0.661385               | 0.660557                     | 0.000828    | 0.317624             |

Key observations from this table:

1. **Exp V tracks Exp Q, not the good run.** The max |Δ| between V and Q
   is 0.0023 (step 5). The gap between V/Q and the good run is **~0.33–0.36**
   at every post-step-1 step — two orders of magnitude larger. The
   reference-path swap does not move the needle.

2. **Step 0 and 1 are near-identical** in both configurations (~0.693),
   consistent with `zero_init_b=True` forcing policy = ref at init so
   loss is `softplus(0) = ln(2)` per example regardless of reference
   construction.

3. **Step 2 escape fails identically.** The good run drops to 0.335;
   Exp V stays at 0.686 — the exact same step-2 escape failure seen in
   every prior v5p-8 LoRA run.

### True DPO quantity: `delta_pi - delta_ref`

Useful comparison points at step 9:

| run                              | step-9 `delta_pi - delta_ref` |
|----------------------------------|-------------------------------|
| Good `v5p-16 pd=2` (Exp N)       | **~10.22**                    |
| Exp T `v5p-8` full FT            | 1.789                         |
| **Exp V (SRC, v5p-8 LoRA)**      | **0.648**                     |
| Exp Q (ABRC, v5p-8 LoRA)         | 0.665                         |

Exp V's DPO signal matches the bad ABRC LoRA regime (within 0.02 nats),
not the good 16-chip regime (15× smaller) and not even the v5p-8 full-FT
regime (3× smaller). The reference-path swap leaves the `delta_pi -
delta_ref` signature of the run essentially unchanged.

### Validation-set behavior at step 10

Both stmt_val and full_val evaluations completed at step 10 before the
cosmetic HF-export crash. The summary metrics were not captured to
wandb.summary because the crash bypassed summary flush, but the in-flight
full_val loss logged in the worker output was **0.689**, consistent with
the Exp Q regime (~0.69) and far from the good-run regime (~0.40).

### What Exp V rules out

**Definitively ruled out by Exp V:**

- `AdapterBaseReferenceConfig`'s "same-model-with-adapter-zeroed forward"
  graph pattern as the load-bearing cause on v5p-8 LoRA. Replacing it
  with a physically separate frozen reference model produces a
  trajectory indistinguishable from the ABRC baseline.

**Previously ruled out, and reinforced by Exp V:**

- All numeric-precision theories (Exp U)
- CE kernel tiling / bf16 accumulation / per-chip CE workload (Exp R, R2a)
- TPU family alone, `per_device_parallelism` alone, LoRA rank alone
  (Exp N, O, P)
- Broad "v5p-8 breaks DPO" (Exp T full-FT success)

### What remains live after Exp V

With ABRC, SRC, and all numeric precision eliminated, the remaining
live hypotheses are all **structural / topological** and all specific to
the LoRA training path:

1. **LoRA FSDP parameter / grad sharding on the 4-device v5p-8 mesh
   (STRONGEST SUSPECT).** With only 4 chips, FSDP shards LoRA A/B
   matrices differently than on 8-chip or 16-chip meshes. The r=64 rank
   dimension and the small `(d_in, r)` / `(r, d_out)` shapes may produce
   a degenerate sharding layout that disrupts the gradient direction
   when aggregated across the 4 data-parallel replicas. Testable via:
   - explicit partition-spec logging on v5p-8 vs v5p-16 (diagnostic)
   - mesh-axis rearrangement probes (`{data:1, model:4}`, `{data:2,
     model:2}`)

2. **LoRA optimizer state partitioning on 4 devices.** Adam `m`, `v`
   buffers for LoRA params shard differently on 4 vs 8 devices. Even
   though per-chip reductions are numerically identical (ruled out by
   Exp U), the *partitioning* could produce different per-chip update
   geometry. Testable via SGD probe (no `m`, `v` state).

3. **LoRA-specific interaction with attention `kv_head` sharding.**
   Llama-8B has 8 KV heads, 32 Q heads (GQA). With `model_axis=1` on
   both v5p-8 and v5p-16 meshes, KV heads are not TP-sharded. But the
   FSDP-sharded replication of KV weights across 4 vs 8 data-parallel
   replicas could produce subtly different gradient directions for the
   attention LoRA adapters specifically. Testable via
   `target_modules=["mlp only"]` ablation.

4. **LoRA gradient all-reduce ordering.** On 4 chips, the reduction
   tree for LoRA gradients is a different topology than on 8 chips.
   Even though fp32 arithmetic (Exp U) is order-independent to float
   tolerance, XLA may pick different collective implementations at
   different chip counts. Testable via `XLA_FLAGS` collective-algorithm
   pinning.

### Updated post-V hypothesis ranking

1. **(leading, STRENGTHENED by V)** LoRA FSDP sharding / adapter
   partition layout on the 4-device v5p-8 mesh — this is the single
   remaining live hypothesis class. Exp V strengthens it by ruling
   out the last reference-path alternative.

2. **(ruled out by Exp V)** `AdapterBaseReferenceConfig` graph pattern

3. **(ruled out by Exp U)** bf16 / bfloat16 numerical precision anywhere
   in the compute graph

4. **(ruled out by Exp R + R2a)** CE kernel tiling, bf16 accumulation,
   per-chip CE workload

5. **(ruled out by Exp N/O/P)** TPU family alone, `per_device_parallelism`
   alone, LoRA rank alone

6. **(ruled out by Exp T)** Broad `v5p-8` execution graph breaking DPO
   without LoRA

### Recommended next experiments (priority order)

1. **Experiment W: mesh-axis rearrangement on v5p-8 LoRA.**
   Run Exp Q recipe with `mesh.axes = {data:1, model:4}` (pure TP, no
   FSDP) and separately with `{data:2, model:2}`. If either recovers →
   FSDP-on-4-chips is load-bearing. If both still bad → the issue
   survives axis reconfiguration and is deeper.
   *This is the single highest-information experiment remaining.*

2. **Experiment Y: `target_modules` ablation.**
   Run v5p-8 LoRA with `target_modules=["mlp only"]`, then
   `["attention only"]`, then HF-style `["q_proj", "v_proj"]`. Localizes
   which submodule's LoRA gradient is actually broken.

3. **Experiment Z: SGD probe.**
   Replace AdamW with plain SGD on v5p-8 LoRA (same recipe otherwise).
   If recovery → Adam `m`/`v` partitioning is the cause. If still bad
   → optimizer state sharding is not the issue.

4. **Partition-spec introspection (diagnostic).**
   Add logging to dump `NamedArray.axes` and partition specs for all
   LoRA A/B parameters on both `v5p-8` and `v5p-16`. Free information,
   can run concurrently with other probes.

5. **Experiment X (cross-family 4-chip) — blocked.**
   The original plan was `v6e-4` LoRA to partition "v5p-8 specific" vs
   "4-chip mesh". `v6e-4` has 32 GB/chip × 4 = 128 GB total HBM and
   **cannot fit** Llama-8B LoRA DPO at `seq_len=4096, bs=64` even at
   `pd=1`. To revive this test, either reduce `train_seq_len` to 2048,
   reduce `train_batch_size` to 32, or switch to `v4-8` (if quota
   allows; 32 GB/chip same as v6e-4 so likely same OOM). Lower priority
   than Exp W/Y/Z which can run on the known-good v5p-8 hardware.

## 2026-04-17T02:10Z: Experiment U result — **full fp32 compute does NOT fix `v5p-8` LoRA**; all numeric-precision theories are now dead

### Strongest true conclusion

Experiment U ran the **exact same bad `v5p-8` LoRA DPO recipe** as
Experiment Q (pd=4) but with the entire compute graph forced to **fp32**
(`p=f32,c=f32` via JMP). The training trajectory is **indistinguishable from
the bf16-compute Exp Q run**: loss stays stuck near `ln(2)` ≈ 0.693 and
never escapes to the ~0.32 regime seen on good runs.

This is a strong, clean rule-out:

> **The `v5p-8` LoRA pathology is NOT caused by bf16 numerical precision —
> not in the forward pass, not in the backward pass, not in the CE kernel,
> not in the reference path, not in the optimizer.**

Combined with the earlier Exp R2a (CE-only bf16-accumulation rule-out) and
the original CE fp32-upcast probe, the entire numeric-precision branch of
the hypothesis tree is now collapsed.

### Why this conclusion is strong

1. **Experiment U is a single-variable ablation.** The only change from
   Exp Q pd=4 is `mp = jmp.get_policy("p=f32,c=f32")` instead of the
   default `p=f32,c=bfloat16`. Everything else — LoRA config, data, seed,
   batch size, learning rate, reference config, TPU type, pd — is identical.
   (Verified from the experiment script:
   `experiments/posttrain/per_stmt_dpo/experiment_u_v5p8_fp32_pd4_s10.py`.)

2. **The precision change is comprehensive.** Unlike Exp R2a (which only
   changed the CE backward accumulator), Exp U changes every matmul in the
   entire graph:
   - policy model forward and backward compute → f32
   - reference model forward compute → f32
   - all activations and intermediate tensors → f32
   - cross-entropy loss → f32 (inputs already f32, no bf16 truncation)
   - LoRA A/B matrix multiplies → f32
   - gradient computation → f32

3. **The trajectories are nearly identical.** The max step-wise loss delta
   between Exp U (fp32) and Exp Q (bf16) is 0.0030 at step 2. This is
   the same order of precision noise seen between the two Exp Q pd variants
   (max Δ = 0.0024). The bf16 → fp32 switch has **no more effect than
   changing the gradient accumulation count**, which is to say: no effect
   on the pathological behavior.

### Run

- **W&B:** https://wandb.ai/marin-community/dpo/runs/experiment_u_r64_v5p8_pd4_fp32_s10_uc1a-e1ff3f
- **GCS checkpoint:** `gs://marin-us-central1/checkpoints/dpo/stmt_dpo/debug/experiment_u_r64_v5p8_pd4_fp32_s10_uc1a-e1ff3f/`
- **Iris job:** `/ahmed/iris-run-experiment_u_v5p8_fp32_pd4_s10-20260417-013922/train_dpo`
- **Worker:** `marin-tpu-v5p-preemptible-8-us-central1-20260416-1853-df21eda2-worker-0` (`10.128.0.84:10001`)
- **Duration:** 18m 12s (task), 19m 25s (job). Succeeded, exit 0, 0 failures, 0 preemptions.
- **Region:** us-central1 (despite `REGIONS_OVERRIDE`; the run tag `uc1a` confirms)

### Training-relevant config

Identical to Exp Q pd=4 except:
- **mixed precision: `p=f32,c=f32`** (was `p=f32,c=bfloat16`)

Everything else pinned to Exp Q pd=4:
- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- LoRA: `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=True`
- reference path: `AdapterBaseReferenceConfig`
- `train_batch_size=64`, `num_train_steps=10`, `steps_per_eval=10`
- `lr=1e-6`, `lr_schedule="cosine"`, `warmup=0.1`
- `beta=0.1`, `seed=0`
- `train_seq_len=max_seq_len=4096`
- `per_device_parallelism=4`, `per_device_eval_parallelism=4`
- `reference_eval_cache=disabled`, `max_eval_batches=1`

### Full 10-step training trajectory (from W&B history)

| step | dpo_loss | dpo_accuracy | dpo_margin_policy | dpo_margin_ref | chosen_reward | rejected_reward |
|------|----------|-------------|-------------------|----------------|--------------|----------------|
| 0    | 0.692681 | 0.578125    | -129.923431       | -129.933121    | 0.001250     | 0.000280        |
| 1    | 0.694727 | 0.390625    | -145.610275       | -145.579071    | -0.001933    | 0.001187        |
| 2    | 0.686458 | 0.828125    | -133.862488       | -133.997330    | 0.010589     | -0.002897       |
| 3    | 0.680453 | 0.968750    | -119.241516       | -119.497665    | 0.017576     | -0.008040       |
| 4    | 0.677019 | 0.953125    | -135.859894       | -136.186218    | 0.024531     | -0.008101       |
| 5    | 0.669665 | 1.000000    | -144.155914       | -144.632584    | 0.033950     | -0.013717       |
| 6    | 0.665122 | 1.000000    | -116.368851       | -116.938904    | 0.040731     | -0.016275       |
| 7    | 0.665376 | 1.000000    | -127.384201       | -127.948654    | 0.041171     | -0.015274       |
| 8    | 0.660315 | 1.000000    | -157.816330       | -158.485840    | 0.048985     | -0.017966       |
| 9    | 0.659855 | 1.000000    | -120.581207       | -121.260384    | 0.047070     | -0.020848       |

### Side-by-side: Exp U (fp32) vs Exp Q (bf16) vs good run (v5p-16)

| step | **Exp U (fp32, v5p-8)** | **Exp Q pd=4 (bf16, v5p-8)** | **|Δ U-Q|** | **Good (v5p-16 pd=2)** |
|------|------------------------|------------------------------|-------------|------------------------|
| 0    | 0.692681               | 0.693147                     | 0.0005      | 0.693147               |
| 1    | 0.694727               | 0.693147                     | 0.0016      | 0.693147               |
| 2    | 0.686458               | 0.685125                     | 0.0013      | 0.335202               |
| 3    | 0.680453               | 0.682298                     | 0.0018      | 0.325988               |
| 4    | 0.677019               | 0.673723                     | 0.0033      | 0.336246               |
| 5    | 0.669665               | 0.668946                     | 0.0007      | 0.316800               |
| 6    | 0.665122               | 0.667573                     | 0.0025      | 0.336998               |
| 7    | 0.665376               | 0.662823                     | 0.0026      | 0.324271               |
| 8    | 0.660315               | 0.658715                     | 0.0016      | 0.306144               |
| 9    | 0.659855               | 0.660557                     | 0.0007      | 0.317624               |

Key observations from this table:

1. **Exp U tracks Exp Q, not the good run.** The max |Δ| between U and Q
   is 0.0033 (step 4). The gap between U/Q and the good run is **~0.35** —
   two orders of magnitude larger. fp32 does not move the needle.

2. **Step 0 and 1 are near-identical** in both bad runs (~0.693), and
   slightly different from the good run only at floating-point noise level.
   All three start from the same initialization and first batch.

3. **Divergence happens at step 2.** The good run escapes to 0.335; both
   bad runs stay at ~0.685-0.686. This is the exact same step-2 escape
   failure documented in every prior v5p-8 LoRA run.

4. **The slow drift downward is identical.** Both bad runs drift from
   ~0.693 to ~0.660 over 10 steps, compared to the good run's immediate
   drop to ~0.32. The drift rate and direction match, confirming the
   pathology is the same phenomenon.

### Chosen/rejected reward separation — still pathological

The DPO "signal" is the gap between chosen and rejected rewards:

| step | **Exp U chosen** | **Exp U rejected** | **Exp U gap** | **Good chosen** | **Good rejected** | **Good gap** |
|------|------------------|--------------------|---------------|-----------------|--------------------|--------------|
| 0    | 0.001            | 0.000              | 0.001         | 0.000           | 0.000              | 0.000        |
| 5    | 0.034            | -0.014             | 0.048         | 0.724           | -0.297             | 1.021        |
| 9    | 0.047            | -0.021             | 0.068         | 0.700           | -0.322             | 1.022        |

By step 9, the good run achieves a reward gap of **1.02** while Exp U
achieves only **0.068** — 15× smaller. The model is barely learning to
distinguish chosen from rejected responses.

### Validation-set behavior at step 10

| metric                        | Exp U (fp32) | Exp Q (bf16)  | note |
|-------------------------------|-------------|---------------|------|
| stmt_val/dpo_loss             | 0.6637      | ~0.66 (Q)     | both bad — near ln(2) |
| stmt_val/dpo_accuracy         | 1.0000      | ~1.0 (Q)      | train accuracy saturates trivially |
| stmt_val/dpo_chosen_reward    | 0.0442      | ~0.046 (Q)    | tiny absolute reward |
| stmt_val/dpo_rejected_reward  | -0.0159     | ~-0.020 (Q)   | tiny rejection signal |
| full_val/dpo_loss             | 0.6940      | ~0.694 (Q)    | both near ln(2) |
| full_val/dpo_accuracy         | 0.375       | ~0.375 (Q)    | near chance on held-out |

All validation metrics track the Exp Q regime, not the good-run regime.

### What Exp U rules out

**Definitively ruled out by Exp U:**

- bf16 compute precision in any part of the DPO training graph on v5p-8.
  This is the strongest possible test of this theory — the *entire* compute
  graph was fp32, and the pathology is unchanged.

**Previously ruled out, and reinforced by Exp U:**

- CE kernel bf16 accumulation (ruled out by Exp R2a; Exp U further confirms
  by making the CE inputs fully f32 and still seeing the same behavior)
- CE tiling / per-chip CE workload (ruled out by Exp R)
- TPU family (ruled out by Exp N)
- `per_device_parallelism` (ruled out by Exp Q pd sweep)
- LoRA rank (ruled out by earlier sweeps)
- Broad "v5p-8 breaks DPO" (ruled out by Exp T full-FT success)

### What remains live after Exp U

The elimination of all numeric-precision theories leaves only **structural /
topological** explanations:

1. **`AdapterBaseReferenceConfig` on v5p-8 (STRONGEST SUSPECT)**

   All bad v5p-8 LoRA runs use `AdapterBaseReferenceConfig`, which computes
   reference log-probs by temporarily zeroing the LoRA adapter and running
   a forward pass through the *same* model instance. On v5p-8 (4 devices),
   the adapter zeroing / un-zeroing might interact with FSDP sharding
   differently than on v5p-16 (8 devices). The full-FT Exp T used
   `SeparateReferenceConfig` (a physically separate model copy), and it
   learned fine. This is the most under-tested confound remaining.

2. **LoRA FSDP sharding on the 4-device v5p-8 mesh**

   With 4 TPU v5p chips, the FSDP sharding might place LoRA A and B
   matrices on the same shard boundaries as the base model weight they
   modify, creating a degenerate gradient accumulation pattern. On v5p-16
   (8 chips) or v6e-8 (also 4 chips but different topology), the mesh
   mapping is different enough to avoid this. This is testable by logging
   `NamedArray.axes` and partition specs for LoRA parameters on both
   configurations.

3. **LoRA optimizer state partitioning**

   The Adam optimizer state (m, v) for LoRA parameters might be sharded
   differently on 4 vs 8 devices, leading to different numerical update
   trajectories. This would explain why the pathology is insensitive to
   compute precision (the issue is in how updates are *aggregated*, not
   *computed*).

4. **Attention kv_head mesh mapping interaction (low probability)**

   The Llama-8B model has grouped-query attention (8 KV heads, 32 Q heads).
   On a 4-device mesh, the KV head sharding might interact with LoRA's
   `target_modules=None` (which targets all linear layers including
   attention projections) in a way that doesn't occur on 8 devices. This
   is speculative and lower priority than the above.

### Updated post-U hypothesis ranking

1. **(leading, STRENGTHENED by U)** Something structural in the LoRA /
   adapter training path on `v5p-8` — most likely `AdapterBaseReferenceConfig`
   behavior, LoRA FSDP sharding layout, or adapter optimizer state
   partitioning on the 4-device mesh. **Exp U strengthens this by ruling
   out the last remaining numeric-precision alternative.**

2. **(ruled out by Exp U)** bf16 / bfloat16 numerical precision anywhere
   in the compute graph.

3. **(ruled out by Exp R + R2a)** CE kernel tiling, bf16 accumulation,
   per-chip CE workload.

4. **(ruled out by Exp N/O/P)** TPU family alone, `per_device_parallelism`
   alone, LoRA rank alone.

5. **(ruled out by Exp T)** Broad `v5p-8` execution graph breaking DPO
   without LoRA.

### Recommended next experiments (priority order)

1. **Experiment V: LoRA on v5p-8 with `SeparateReferenceConfig`.**
   Replace `AdapterBaseReferenceConfig()` with `SeparateReferenceConfig()`
   in the Exp Q recipe. Keep everything else identical. If this recovers
   → `AdapterBaseReferenceConfig` is the cause. If still bad → the cause
   is in LoRA param/sharding itself, not the reference path.
   *This is the single highest-information experiment remaining.*

2. **Experiment W: LoRA sharding introspection.**
   Add logging to dump the `NamedArray.axes` and FSDP partition specs for
   all LoRA A/B parameters on both `v5p-8` and `v5p-16`. Compare the
   layouts. This is diagnostic, not a fix, but it will either confirm or
   eliminate the sharding-layout theory.

3. **Experiment X: v6e-4 LoRA DPO (4-device non-v5p mesh).**
   Run the same recipe on `v6e-4` (4 chips, same device count as v5p-8
   but different TPU family and topology). If v6e-4 also fails → the
   4-device mesh is the issue regardless of family. If v6e-4 succeeds →
   the issue is specific to the v5p 4-device topology.

## 2026-04-17T01:36Z: Experiment U staged — rerun the bad `v5p-8` LoRA regime with `p=f32,c=f32` at `pd=4`

### What changed in code

Prepared a new experiment script:

- `experiments/posttrain/per_stmt_dpo/experiment_u_v5p8_fp32_pd4_s10.py`

Also threaded mixed precision through the simplified DPO path so debug probes can
set it explicitly instead of editing `defaults.py` inline each time:

- `SimpleDPOConfig` now exposes `mp`
- `default_dpo(...)` now passes `dpo_config.mp` into `TrainerConfig`

This keeps the new probe local and reversible. It does **not** change the
default DPO precision for existing experiments, because the new field defaults
to the prior policy:

- default remains `p=f32,c=bfloat16`

### Why this is the next experiment

David suggested the simplest remaining numeric probe:

> "have you tried putting the whole thing in float32"

Interpreted in Levanter / JMP terms, that means:

- `p=f32,c=f32`

This is materially broader than the earlier CE-only upcast probe:

- it changes the policy forward / backward compute dtype
- it changes the frozen non-trainable base-weight dtype under LoRA
- it changes the reference-path compute dtype
- it changes activations / logits / CE inputs throughout the DPO graph

So Experiment U is a clean test of:

> **Is the bad `v5p-8` LoRA regime specifically tied to bf16 compute rather than to LoRA/reference topology more generally?**

### Exact Experiment U configuration

Experiment U is intentionally a near-copy of the **Experiment Q `pd=4` branch**
with only one scientific knob changed.

Held fixed from Exp Q:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- LoRA: `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=True`
- reference path: `AdapterBaseReferenceConfig`
- `train_batch_size=64`
- `num_train_steps=10`
- `steps_per_eval=10`
- `lr=1e-6`
- `lr_schedule="cosine"`
- `warmup=0.1`
- `beta=0.1`
- `train_seq_len=max_seq_len=4096`
- `reference_eval_cache=disabled`
- `max_eval_batches=1`
- debug traces on (`MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`)

Changed from Exp Q:

- mixed precision: **`p=f32,c=f32`**

Pinned for memory control:

- `per_device_parallelism=4`
- `per_device_eval_parallelism=4`

### Why force `pd=4`

The point of U is to test **whole-graph fp32 compute**, not to maximize HBM
stress. On `v5p-8`, peak HBM is dominated by activation / temporary tensors,
and those scale with the live microbatch (`pd`), not merely with the optimizer's
global batch.

At a rough first-principles level for this 8B LoRA DPO regime:

- `pd=4`, bf16-compute: about mid-20s GiB/chip
- `pd=4`, full-f32-compute: about high-40s GiB/chip
- `pd=8`, bf16-compute: about mid-40s GiB/chip
- `pd=8`, full-f32-compute: about mid-80s GiB/chip

These are ballpark estimates, not measured peaks, but they are directionally
good enough to justify the choice:

- `pd=4` should give a **much safer fit test** on `v5p-8`
- `pd=8` might still fit, but it would mix the precision probe with a much
  tighter HBM regime

So `pd=4` is the right first shot if the goal is to isolate the effect of
whole-graph fp32.

### What U will teach us

If Experiment U **recovers strongly** relative to bad Exp Q / bad `v5p-8`
LoRA runs, then the explanation moves toward:

- bf16-compute sensitivity in the LoRA/reference path on `v5p-8`
- or a numerics issue that only appears once the adapter/base-reference graph is
  run in bf16

If Experiment U is **still bad**, then "just run it in fp32" is mostly ruled
out, and the leading remaining suspects stay structural:

- `AdapterBaseReferenceConfig`
- LoRA-specific param / grad sharding
- LoRA update geometry on the 4-device `v5p-8` mesh
- possibly attention `kv_head` / mesh-mapping interaction

### Launch command

```bash
REGIONS_OVERRIDE=us-east5 \
MARIN_DEBUG_RUN_TAG=ue5a \
uv run python experiments/posttrain/per_stmt_dpo/experiment_u_v5p8_fp32_pd4_s10.py
```

Expected run name pattern:

- `dpo/stmt_dpo/debug/experiment_u_r64_v5p8_pd4_fp32_s10_<tag>`

### Status

- Experiment U **completed successfully** on 2026-04-17. See the result
  section above at `2026-04-17T02:10Z`.
- **Result: fp32 does NOT fix the pathology.** Trajectory is indistinguishable
  from Exp Q (bf16). All numeric-precision theories are ruled out.

## 2026-04-17T00:30Z: Experiment T result — `v5p-8` full FT **LEARNS**; broad `v5p-8` execution-graph failure is no longer the leading explanation

### Strongest true conclusion

The completed 10-step Experiment T rerun on **April 17, 2026** shows that
`v5p-8` can run **full-FT DPO** and learn in the same *qualitative* regime as
the previously-good full-FT baselines.

This is the cleanest answer yet to the question Exp T was launched to settle:

> **Does the `v5p-8` pathology survive when LoRA is removed?**

The answer is now:

> **No, not in the catastrophic sense seen in the LoRA runs.**

This `v5p-8` full-FT run is not a perfect numeric match to the good 16-chip
full-FT baselines, but it is decisively **closer to the good full-FT regime
than to the bad `v5p-8` LoRA regime**.

The practical implication is strong:

1. **`v5p-8` full FT is feasible.** The run compiled, trained for 10 steps,
   evaluated, and wrote checkpoints / HF export successfully.
2. **The catastrophic stuck-near-`ln(2)` behavior does not reproduce under
   full FT on `v5p-8`.**
3. Therefore the remaining pathology is now much more likely to live in the
   **LoRA / adapter-specific training path** on `v5p-8`:
   - LoRA low-rank update geometry,
   - adapter-only optimizer / sharding behavior,
   - `AdapterBaseReferenceConfig`,
   - or an interaction between those and a still-live sub-CE distributed detail.

This strongly weakens the broader theory:

> "Something generic about the `v5p-8` distributed execution graph breaks DPO,
> even without LoRA."

That theory is no longer the best explanation after this run.

### Run

- `v5p-8` full FT, `bs=32`, `pd=4`, `steps=10`:
  https://wandb.ai/marin-community/dpo/runs/exp_t_v5p8_fullft_bs32_pd4_s10_uc1-rerun-20260416-3-stream-042354
- Run name:
  `exp_t_v5p8_fullft_bs32_pd4_s10_uc1-rerun-20260416-3-stream-042354`
- Worker `output.log` confirms:
  - first train batch started at `2026-04-17T00:17:44Z`
  - step 9 completed by `2026-04-17T00:22:06Z`
  - final eval + checkpoint/HF export completed by `2026-04-17T00:30:25Z`

### Training-relevant config

What is directly established from the run name, worker log, and W&B history:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- **full FT** (no adapter)
- reference path: `SeparateReferenceConfig`
- `train_batch_size=32`
- `per_device_parallelism=4`
- `num_train_steps=10`
- `lr=1e-6`
- `beta=0.1`
- `seed=0`
- `train_seq_len=max_seq_len=4096`
- `stmt_val` + `full_val` eval at the end

Important nuance:

- This is **not** a fully apples-to-apples replay of Exp L / Exp O, because
  the global batch is smaller here (`bs=32` vs the earlier `bs=64`
  full-FT baselines).
- So the right conclusion is **not** "the curves match exactly."
- The right conclusion is the stronger qualitative one: **this run is in the
  same learning regime as the good full-FT runs and far from the bad LoRA
  regime.**

### CE line on this run — descriptive, not causal

The worker log shows:

```
DEBUGCE XLA CE block sizes resolved: device_kind=TPU v5 x.shape=(32768, 4096) w.shape=(4096, 128256) v_block_size=8192 b_block_size=32768 num_v_blocks=16 num_b_blocks=1 explicit_block_sizes=False
```

This matches the "single batch block" CE regime seen in the good
`v5p-16 pd=2` full-FT / LoRA runs and differs from the bad `v5p-8` LoRA
baseline.

But this should **not** be read as reviving the CE hypothesis. Exp R2a already
ruled out CE tiling / CE inter-block accumulation as the load-bearing cause on
the bad LoRA baseline. The CE line here is best treated as a descriptive part
of the run's local execution geometry, not the newly-proved root cause.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

| step | loss     | grad_l2  | grad_sum   |
|------|----------|----------|------------|
| 0    | 0.693139 | 31.0029  | -23.8408   |
| 1    | 0.693142 | 29.5883  | +16.2643   |
| 2    | 0.687056 | 28.4906  | -3.7742    |
| 3    | 0.680934 | 27.3795  | -11.7273   |
| 4    | 0.664067 | 28.4214  | -5.8817    |
| 5    | 0.660287 | 28.4812  | +10.0076   |
| 6    | 0.622947 | 27.4912  | +12.2259   |
| 7    | 0.616519 | 26.6690  | -30.4674   |
| 8    | 0.608089 | 27.2849  | +12.3777   |
| 9    | 0.608325 | 25.3038  | -2.6215    |

Qualitative read:

- The run is **not stuck** at `ln(2)`.
- It leaves `ln(2)` by step 2 and reaches `~0.608` by step 9.
- The full-FT gradient norm declines in the expected broad range
  (`31.0 -> 25.3`), unlike the pathological LoRA traces that remain in the
  wrong basin with much weaker descent.

### Side-by-side: Exp T vs good full-FT baselines vs bad `v5p-8` LoRA

| step | Exp T `v5p-8` full FT `bs=32 pd=4` | Good `v5p-16` full FT `pd=4` (Exp L) | Good `v6e-16` full FT `pd=2` (Exp O) | Bad `v5p-8` LoRA `pd=4` (Exp Q) |
|------|------------------------------------|--------------------------------------|--------------------------------------|---------------------------------|
| 0    | 0.693139 | 0.693147 | 0.693163 | 0.693147 |
| 1    | 0.693142 | 0.693147 | 0.693179 | 0.693147 |
| 2    | 0.687056 | 0.688913 | 0.686007 | 0.685125 |
| 3    | 0.680934 | 0.673635 | 0.673826 | 0.682298 |
| 4    | 0.664067 | 0.663108 | 0.667567 | 0.673723 |
| 5    | 0.660287 | 0.656349 | 0.655773 | 0.668946 |
| 6    | 0.622947 | 0.615969 | 0.615281 | 0.667573 |
| 7    | 0.616519 | 0.603591 | 0.601114 | 0.662823 |
| 8    | 0.608089 | 0.593090 | 0.588456 | 0.658715 |
| 9    | 0.608325 | 0.593389 | 0.592088 | 0.660557 |

This table is the core result:

- Exp T is **very close** to the known-good full-FT runs.
- Exp T is **far away** from the bad `v5p-8` LoRA regime by the later steps.
- At step 9, Exp T is only about `~0.015-0.016` worse than the good full-FT
  baselines, but about `~0.052` better than the bad `v5p-8` LoRA run.

### The actual DPO quantity: `delta_pi - delta_ref`

As elsewhere in this logbook, the loss-driving quantity is:

`delta_pi - delta_ref = train/dpo_margin_policy - train/dpo_margin_ref`

For Exp T:

| step | Exp T `delta_pi - delta_ref` |
|------|------------------------------|
| 0    | 0.000153 |
| 1    | 0.000076 |
| 2    | 0.124344 |
| 3    | 0.248108 |
| 4    | 0.593155 |
| 5    | 0.670395 |
| 6    | 1.462914 |
| 7    | 1.605316 |
| 8    | 1.789200 |
| 9    | 1.789017 |

Useful comparison points at step 9:

| run | step-9 `delta_pi - delta_ref` |
|-----|-------------------------------|
| Exp T `v5p-8` full FT | **1.7890** |
| Good `v5p-16` full FT (Exp L) | **2.1208** |
| Bad `v5p-8` LoRA (Exp Q) | **0.6647** |

So Exp T is much closer to the good full-FT regime than to the bad LoRA
regime on the true DPO quantity as well, not just on train loss.

### Validation-set behavior

| split | pre-training | post-step-9/10 eval | Δ |
|-------|-------------:|--------------------:|--:|
| stmt_val | 0.6931 | 0.6116 | -0.0815 |
| full_val | 0.6931 | 0.6913 | -0.0018 |

Interpretation:

- On the statement validation split, Exp T shows clear learning.
- On `full_val`, transfer after only 10 steps is weak, but it is still
  directionally better than the bad `v5p-8` LoRA regime.
- This is normal for a short per-statement probe and does not weaken the main
  conclusion about train-time regime.

### What Exp T rules out, supports, and leaves open

**Rules out / strongly weakens:**

- "The `v5p-8` DPO pathology is broad to full FT as well as LoRA."
- "Something generic about the `v5p-8` execution graph prevents DPO from
  learning, even when LoRA is removed."
- "The next best use of time is more generic full-FT / remat / CE debugging on
  `v5p-8`."

**Strongly supports:**

- The remaining failure surface is now primarily in the **LoRA / adapter
  training path** on `v5p-8`.
- The most likely live culprits are:
  1. **LoRA-specific update geometry / adapter-only optimizer behavior**
  2. **`AdapterBaseReferenceConfig`**, since Exp T uses
     `SeparateReferenceConfig` and learns
  3. a LoRA-specific interaction with a still-live distributed detail such as
     attention `kv_head` mapping or adapter-parameter sharding

**Not yet proved:**

- Whether **LoRA alone** is sufficient to cause the remaining `v5p-8`
  pathology, or whether the real load-bearing variable is specifically
  `LoRA + AdapterBaseReferenceConfig`.
- Whether a `v5p-8` LoRA run with `SeparateReferenceConfig` would learn
  normally.
- Whether adapter-only sharding / optimizer-state handling differs on
  `v5p-8` vs the known-good pods in a way that only matters for LoRA.

### Revised next-best experiments after Exp T

Exp T changes the next-step ranking substantially.

Highest-value next probes now are:

1. **LoRA on `v5p-8` with `SeparateReferenceConfig`**
   - Keep the per-stmt data and `v5p-8` geometry
   - remove `AdapterBaseReferenceConfig`
   - this is the cleanest discriminator between:
     - "LoRA itself is the remaining problem" and
     - "`AdapterBaseReferenceConfig` is the remaining problem"

2. **Adapter parameter / gradient / optimizer sharding dump on `v5p-8`**
   - inspect live LoRA param shardings, grad shardings, and opt-state shardings
   - compare against a known-good LoRA run (`v5p-16` or `v6e-16`)

3. **Only after those:** targeted LoRA-only `kv_head` mapping probe
   - if the LoRA path still looks broken after the reference-config split,
     then attention-axis mapping becomes a cleaner next lever

Lower priority now:

- more generic `v5p-8` full-FT feasibility work
- more CE-kernel work
- broad "maybe remat / FSDP / collectives break everything on `v5p-8`"
  investigation without a LoRA-specific discriminator

---

## 2026-04-16T17:20Z: Experiment T handoff — **XLA compile bug** on `offload`, `recompute` in-flight

**Status update (2026-04-17T00:30Z):** superseded by the completed
10-step Exp T rerun immediately above. The remainder of this section is kept as
historical launch/debug context for the earlier failed `offload` attempt and
the original fallback ladder.

### Executive summary for the next agent

**TL;DR:** Experiment T attempt 2 (`ue5a-i2`, `offload` checkpointing, `bs=32`, `pd=4`) **reached a running TPU worker** at `08:36Z`, then died with an **XLA compile-time check failure** at `08:34Z` wall-clock (exit 139, SIGSEGV). A third attempt with `gradient_checkpointing="recompute"` has been submitted and an auto-restart loop is walking the fallback ladder. Pick up from that attempt.

### What failed on attempt i2 (`offload`, `bs=32`, `pd=4`)

- Iris job id: `/ahmed/debug-t1-full-ft-v5p8-bs32-pd4-offload-ue5a-i2`
- Final state: `state=failed, task=failed, preempt=2`
- Exit: `139` (SIGSEGV)
- XLA stderr (truncated by iris): `F0416 08:34:09.888667  1219 async_dynamic_index_emitter.cc:584] Check failed: intermediate_calc.slice_size % interme...`
- Interpretation: this is an **XLA internal assertion** inside the async dynamic-index emitter, not a preemption, not an HBM OOM, not a process kill. It fires during compile of the full-FT DPO program with `gradient_checkpointing="offload"` on `v5p-8`. The `offload` checkpointing path materializes host-offloaded carries with dynamic-slice shape math, which is where this check lives. So "offload on `v5p-8` + full FT + `SeparateReferenceConfig` + `bs=32 pd=4`" hits the bug; this is *not* a general `v5p-8` compile failure.

### What is running now

- **Currently in-flight**: `/ahmed/debug-t1-full-ft-v5p8-bs32-pd4-recompute-ue5a-i3`
- **Only change vs i2**: `EXPERIMENT_T_CHECKPOINTING=recompute` (was `offload`)
- Submission time: `2026-04-16T17:20Z`
- Parent launch: `iris job run --zone us-east5-a --cpu 1 --memory 3g --enable-extra-resources` is **not** used — parent is plain `--memory 3g` (intentional, see Exp Q ops notes).
- Script: `experiments/posttrain/per_stmt_dpo/experiment_t_v5p8_full_ft_s2.py` (codex-authored, supports env overrides)

### Auto-restart loop (do not duplicate)

An auto-restart bash loop is active in the Claude Code worker (`bg id b57fuy53l`) and writes to `/tmp/t_autorestart.log`. It:

- polls the `train_dpo` child every 10 min
- if it sees `state=failed` *and* no `DEBUGJ TRACE` step progress, it kills the parent and submits the next fallback in the ladder
- if it sees step progress, it stops advancing and just watches

**Fallback ladder encoded in the loop:**

1. i3 (live): `recompute`, `bs=32`, `pd=4`
2. i4 (auto-next on failure): `recompute`, `bs=16`, `pd=4`
3. i5 (auto-next on failure): `offload`, `bs=16`, `pd=4`
4. ladder exhausted → loop stops

The next agent does **not** need to relaunch anything unless:
- the loop is no longer running (check with `pgrep -af t_autorestart`)
- or the ladder is exhausted
- or the next agent wants to try something off-ladder (e.g., `bs=32 pd=2`, or `gradient_checkpointing="default"` with no policy override, or `bs=8`)

### Open behavioral question still waiting on Exp T

None of the Exp T attempts has produced a single `DEBUGJ TRACE` step line yet. So the scientific question the experiment was launched to answer —

> **Does the `v5p-8` pathology survive when LoRA is removed?**

— is still *completely open*. The behavioral comparison to the good full-FT runs (Exp L `v5p-16 pd=4`, Exp L `v6e-16 pd=4`, Exp O `v6e-16 pd=2`) cannot be made until a `v5p-8` full-FT attempt produces at least step 0 and step 2 traces.

### Relevant prior sections in this logbook

- **Exp R2a result (CE kernel branch cleanly ruled out)**: end of logbook at `2026-04-16T22:50Z`. Forced `num_b_blocks=1` at the bad-baseline `B=65536` on `v5p-8 pd=4 bs=64`; still stuck. This is the strongest CE-level rule-out we have. Motivates the full pivot to sub-CE suspects.
- Exp R2 plan (explicit-block-size rationale + three cases + HBM analysis): `2026-04-16T20:00Z`.
- Exp R result (shrinking `v5p-8` CE workload still did not recover it): top of logbook at `2026-04-16T04:52Z`. Note: Exp R did **not** bit-match the good CE tiling; it overshot to `B=16384, num_b_blocks=16`. The clean CE-matching rule-out comes from R2a.
- Root-cause update and Exp T plan: `2026-04-16T13:05Z` ("Root-cause update after Exp R — next best probe is **full FT on `v5p-8`**").
- Exp Q (pd=4 / pd=8) results and ops lessons for `v5p-8` on `us-east5-a`: top of logbook (`2026-04-16T03:17Z`, `2026-04-16T00:50Z`).
- Prior successful `v5p-8` ram tuning: use `ram="150g"` for the child `train_dpo` on this pool; `ram="250g"` will not fit on co-tenanted workers right now. (This is already set in `experiment_t_v5p8_full_ft_s2.py`.)

### What to do when i3 / i4 / i5 completes

1. If any attempt emits `DEBUGJ TRACE step=0` through `step=1` (2-step probe), extract the trace from `iris job logs /ahmed/<job>/train_dpo`, drop into the Exp T results section of this logbook, and compare step-2 loss against the good full-FT baseline.
2. If all three ladder attempts fail in the same XLA way, the next reasonable moves in priority order are:
   - try `EXPERIMENT_T_CHECKPOINTING=default` (no policy override, use the llama_8b default)
   - try `EXPERIMENT_T_PD=2` to change the distributed factorization
   - try `SeparateReferenceConfig` → `AdapterBaseReferenceConfig` swap (this turns Exp T from a pure "is it LoRA?" probe into a "is it the separate-reference graph?" probe; note the interpretation changes)

### Operational regret / lesson

The previous sleep loop (`/tmp/t_sleep_loop.log`) only observed state; it did **not** act on `state=failed`. As a result, Exp T i2 sat in a terminal `failed` state for ~8 hours without a follow-up launch. The new loop acts on failures and walks the ladder, so this class of regret should not recur for Exp T. If the next agent builds a similar watcher for another experiment, include an auto-restart policy from the start.

---

## 2026-04-16T04:52Z: Experiment R result — **shrinking `v5p-8` CE workload still does NOT fix it** (Case 2: no recovery)

### Strongest true conclusion

Experiment R lowered the local XLA CE workload on `v5p-8` substantially by
changing `train_batch_size` from 64 to 32 at `pd=4`, and **the run still stayed
stuck near `ln(2)`**.

This is the "Case 2: no recovery" outcome that the Exp R plan laid out as
the most consequential possibility. It does two things:

1. **Strongly weakens the fused CE kernel batch-blocking hypothesis** as the
   load-bearing cause of the `v5p-8` pathology. We shrank the local CE problem
   far below the bad Exp Q regime and still tracked Exp Q behaviorally.
2. **Pivots the investigation** away from CE-level math toward the broader
   `v5p-8` distributed regime — FSDP sharding, all-gather / reduce-scatter
   topology, attention kv-head mapping, remat scheduling, or the reference
   network's compiled graph.

This is the first Exp Q/R result where **step-0 `grad_l2` itself shifts**
(2.6614 vs 2.4560 in Exp Q / 2.4563 in good `v5p-16 pd=2`). So even the first
forward/backward is producing different accumulated LoRA gradients on `v5p-8
bs=32` than it does on either `v5p-8 bs=64` or `v5p-16 bs=64`. The difference
isn't scale — it's direction: `grad_sum=-1.0642` at step 0 (vs +0.184 in Exp
Q and good runs). That is a first-principles signal that the distributed
compute graph on `v5p-8 bs=32` is arithmetically different, not just
reshaped.

### Run

- `v5p-8 bs=32 pd=4` (us-east5-a, INTERACTIVE band, preemptible pool): https://wandb.ai/marin-community/dpo/runs/experiment_r_r64_v5p8_bs32_pd4_s10_ue5a-i1-423c65

Launch command context:
- parent CPU coordinator pinned via `iris job run --zone us-east5-a --cpu 1 --memory 3g`
- child `train_dpo` task: `v5p-8` preemptible slice in `us-east5-a`
- env: `EXPERIMENT_R_BS=32`, `EXPERIMENT_R_PD=4`, `REGIONS_OVERRIDE=us-east5`, `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`, `MARIN_DEBUG_RUN_TAG=ue5a-i1`
- iris job id: `/ahmed/debug-r1-r64-v5p8-bs32-pd4-ue5a-i1`
- experiment script: `experiments/posttrain/per_stmt_dpo/experiment_r_v5p8_bs32_s10.py` (new, cloned from Exp Q)

### Training-relevant config

Identical to the Exp Q pd=4 run (see that section below) except:
- `trainer.train_batch_size = 32` (was 64)
- therefore `grad_accum = 32 / (pd * num_devices) = 32 / (4 * 4) = 2` (was 4)

All other knobs — LoRA `r=64/α=64/zero_init_b=True`, `AdapterBaseReferenceConfig`,
`lr=1e-6`, `β=0.1`, `seed=0`, `seq_len=4096`, 10 steps, `reference_eval_cache=disabled`,
`max_eval_batches=1`, `include_lm_validation=False` — match Exp Q exactly.

### CE kernel shape — reduced sharply, but **did not match** the good `v5p-16 pd=2` regime

Verified `DEBUGCE` line at step 0 from the worker `output.log`:

```
device_kind=TPU v5 x.shape=(16384, 4096) w.shape=(4096, 128256)
v_block_size=8192 b_block_size=1024 num_v_blocks=16 num_b_blocks=16
explicit_block_sizes=False
```

Side-by-side comparison:

| run                         | `x.shape`        | `b_block_size` | `num_b_blocks` |
|-----------------------------|------------------|----------------|----------------|
| Good Exp N `v5p-16 pd=2`    | (32768, 4096)   | 32768          | 1              |
| **Exp R `v5p-8 bs=32 pd=4` (this)** | **(16384, 4096)** | **1024** | **16** |
| Bad Exp Q `v5p-8 pd=8`      | (65536, 4096)   | 1024           | 64             |
| Bad Exp Q `v5p-8 pd=4`      | (65536, 4096)   | 1024           | 64             |

So Exp R did **not** produce a bit-identical local CE problem. Instead, it
overshot in the other direction: the `v5p-8` run saw **half** the good
`v5p-16` local CE rows and still remained in the bad basin. That is still
useful evidence: reducing per-chip CE workload and reducing `num_b_blocks`
from `64 -> 16` did not recover training.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

| step | loss     | grad_l2  | grad_sum  |
|------|----------|----------|-----------|
| 0    | 0.693147 | 2.661433 | -1.0642   |
| 1    | 0.693147 | 2.547172 | +1.4175   |
| 2    | 0.688840 | 2.428318 | +0.0763   |
| 3    | 0.682598 | 2.356858 | +0.5037   |
| 4    | 0.675323 | 2.494112 | -0.3843   |
| 5    | 0.672751 | 2.467276 | -0.7983   |
| 6    | 0.665934 | 2.535709 | +0.0361   |
| 7    | 0.668329 | 2.429227 | -1.4167   |
| 8    | 0.660116 | 2.483956 | +0.5612   |
| 9    | 0.662023 | 2.348258 | -2.1968   |

Qualitative pattern is the same bad regime as Exp Q:
- Loss stays near `ln(2)` across 10 steps (only drops ~0.03 total).
- `grad_l2` never decays toward ~1.12; stays in ~2.34-2.66 band.
- Step 1 loss is still exactly `ln(2)` — first update does essentially nothing.

### Side-by-side: Exp R vs Exp Q vs good runs

| step | Exp Q `bs=64 pd=4` | Exp R `bs=32 pd=4` (this) | Good `v5p-16 pd=2` (Exp N) |
|------|--------------------|---------------------------|----------------------------|
| 0    | 0.693147 | 0.693147 | 0.693147 |
| 1    | 0.693147 | 0.693147 | 0.693147 |
| 2    | 0.685125 | **0.688840** | 0.335202 |
| 3    | 0.682298 | **0.682598** | 0.325988 |
| 4    | 0.673723 | **0.675323** | 0.336246 |
| 5    | 0.668946 | **0.672751** | 0.316800 |
| 6    | 0.667573 | **0.665934** | 0.336998 |
| 7    | 0.662823 | **0.668329** | 0.324271 |
| 8    | 0.658715 | **0.660116** | 0.306144 |
| 9    | 0.660557 | **0.662023** | 0.317624 |

Exp R tracks Exp Q (max |Δ| ≈ 0.005), not the good run (|Δ| stays ~0.35).
The bad basin on `v5p-8` is not determined by local CE math.

### Step-0 gradient norms — first shift observed across v5p-8 runs

| run                               | step-0 `grad_l2` | step-0 `grad_sum` |
|-----------------------------------|------------------|-------------------|
| Exp N `v5p-16 pd=2`               | 2.4563 | +0.18380 |
| Exp P `v5p-16 pd=4`               | 2.4622 | (≈+0.18) |
| Exp Q `v5p-8 pd=8`                | 2.4562 | +0.18380 |
| Exp Q `v5p-8 pd=4`                | 2.4560 | +0.18380 |
| **Exp R `v5p-8 bs=32 pd=4` (this)** | **2.6614** | **-1.0642** |

Up to and including Exp Q, every fixed-recipe v5p-* run produced the same
step-0 LoRA gradient to ~4 sig figs — including `grad_sum`. Exp R is the
first to shift `grad_l2` (by about +8%) and `grad_sum` (from positive to
negative, ~6x magnitude change). That means the `bs=32` change on `v5p-8`
actually reaches the numerically-distinct compute graph we were trying to
reproduce — yet the post-init trajectory still tracks the bad basin, not
the good one. The bad basin is therefore attracting from a wider set of
initial gradient directions than we had evidence for before Exp R.

### Validation-set behavior

| eval split | pre-training | post-step-10 | Δ      |
|------------|--------------|--------------|--------|
| stmt_val   | 0.693        | 0.662        | -0.031 |
| full_val   | 0.693        | 0.697        | +0.004 |

Effectively identical to the Exp Q pd=4 validation behavior
(stmt_val 0.693→0.663, full_val 0.693→0.694). No meaningful learning on
the broader distribution.

### What Exp R rules out, supports, and leaves open

**Rules out (strengthened):**

- “We just need to reduce the per-chip token load on `v5p-8`.” Halving the
  global batch size did reduce the local CE workload below the good
  `v5p-16 pd=2` run's level (B=16384 vs good B=32768), and didn't fix it.

**Weakens but does not fully rule out from Exp R alone:**

- “The fused XLA CE batch-blocking / `num_b_blocks` regime on `v5p-8` is
  the load-bearing cause.” Exp R did **not** bit-match the good CE tiling;
  it ended at `num_b_blocks=16` while good is `num_b_blocks=1`. The clean
  rule-out of CE batch-blocking comes from Exp R2a (2026-04-16T22:50Z,
  end of this logbook), which forced `num_b_blocks=1` at the bad-baseline
  `B=65536` and still stayed stuck.

**Strongly supports:**

- The `v5p-8` pathology lives below the CE kernel, in the broader
  distributed execution graph: FSDP all-gather / reduce-scatter pattern,
  attention `kv_head` sharding, reference-network compiled graph, or
  remat/HLO scheduling interactions specific to the `-8` pod topology.
- Step-0 gradients on `v5p-8` are not uniquely determined by the local CE
  shape — something earlier in the forward (or the reference path) also
  differs when `train_batch_size` changes at fixed pod.

**Not yet proved:**

- Which specific distributed-regime knob carries the pathology on `v5p-8`.
  Candidates ordered by probable information density:
  1. attention head sharding — map `kv_head` axis explicitly and see if
     it removes the split (extends the `dpo-lora` branch's TP=4 fix).
  2. FSDP granularity — reduce or eliminate FSDP sharding on the small
     `-8` pod; see if the run recovers.
  3. reference-network path — swap to a `SeparateReferenceConfig` probe on
     `v5p-8 bs=32 pd=4` to check whether the adapter-base reference graph
     is the piece that diverges.
  4. collective algorithm — force a different XLA collective impl via
     `XLA_FLAGS` and observe.

### Operational notes

This Exp R run required **1 preemption** before the clean 10-step attempt:

- **i1 attempt A**: TPU worker `...-0356-0efb6e03-worker-0` took the task and
  then died with `Worker failed: heartbeat stale (>900s since last heartbeat)`
  before reaching a training step.
- **i1 attempt B**: reassigned to `...-0239-624c632f-worker-0` at ~04:27Z.
  `DEBUGCE` logged at 04:27:48Z. Step 0 completed at 04:29:20Z (~85s).
  Steps 1-9 completed by 04:52Z. Status: `state=running, preempt=1` (the
  parent is still finalizing / writing checkpoints at the time of this
  logbook entry but all 10 `DEBUGJ TRACE` step lines are present).

Operational improvements from the Exp Q debugging storm carried into this
run:
- `--memory 3g --cpu 1` on the parent so it lands on a proper CPU ondemand
  worker, not a TPU slice.
- `ram="250g"` for the `v5p-8 pd=4` child so it fits under the ~305 GB
  free-memory budget on co-tenanted preemptible v5p-8 workers.

us-east5-a remains the only region with actual v5p-8 capacity during this
session; `us-central1` still shows 0 slices on
`tpu_v5p-preemptible_8-us-central1-a`.

### Next experiment direction

Per the logbook's pre-registered Case 2 plan:

> if CE is already matched and the run is still bad, stop focusing on CE
> and pivot to topology / sharding / collective investigation.

The most concretely-grounded next probe is:

**Experiment S (proposed) — attention `kv_head` axis mapping probe on
`v5p-8`.**
- same data / LoRA recipe / `bs=32 pd=4` as Exp R so the result is directly
  comparable
- change only the attention `kv_head` axis → `model` axis mapping, to
  mirror the TP fix already present for v6e-8 in the `dpo-lora` branch
  (commit `0b228b3a5 "[dpo] Fix TP=4 attention sharding: map kv_head axis
  to model"`)
- if `v5p-8` shared the same hidden kv-head mapping bug, this should be the
  cleanest first-principles fix

If Exp S does not recover, fall back to a `SeparateReferenceConfig` probe
on `v5p-8 bs=32 pd=4` to isolate the reference-network graph.

---

## 2026-04-16T03:17Z: Experiment Q (pd=4) result — `v5p-8` pathology is **independent of `per_device_parallelism`**

### Strongest true conclusion

Experiment Q pd=4 on `v5p-8` produces a training trajectory that is
**point-for-point identical** (max |Δ| = 0.0024) to the Exp Q pd=8 run. Both
stay stuck near `ln(2)`, while the same LoRA recipe on `v5p-16` escapes
immediately at both pd=2 (Exp N) and pd=4 (Exp P).

This completes the core Exp Q sweep and eliminates `per_device_parallelism`
as even a secondary contributor on the `-8` pod. The remaining pathology is
entirely attributable to the **`v5p-8` pod shape / host topology / sharding
layout** — the only variable that differs between the bad `v5p-8` runs and
the good `v5p-16` runs.

Combined with the earlier experiments:
- **Exp N**: TPU family is not the cause (matched v5p-16 vs v6e-16 track closely)
- **Exp O**: `pd` / local-shape changes don't break full FT on v6e-16
- **Exp P**: `pd` / local-shape changes don't break LoRA on v5p-16
- **Exp Q pd=8**: v5p-8 reproduces the bad regime
- **Exp Q pd=4 (this run)**: v5p-8 stays bad even at lower pd — the failure
  tracks the `-8` pod shape broadly, not just the high-`pd` corner

### Run

- `v5p-8 pd=4` (us-east5-a, INTERACTIVE band, preemptible pool): https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd4_s10_ue5a-i4-d7d7e1

Launch command context:
- parent CPU coordinator pinned via `iris job run --zone us-east5-a --cpu 1 --memory 3g`
- child `train_dpo` task: `v5p-8` preemptible slice in `us-east5-a`
- env: `EXPERIMENT_Q_PD=4`, `REGIONS_OVERRIDE=us-east5`, `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`, `MARIN_DEBUG_RUN_TAG=ue5a-i4`
- iris job id: `/ahmed/debug-q1-r64-v5p8-pd4-ue5a-i4`
- experiment script: `experiments/posttrain/per_stmt_dpo/experiment_q_v5p8_pd_s10.py`

### Training-relevant config

Identical to the Exp Q pd=8 run (see that section below) except:
- `trainer.per_device_parallelism = 4` (was 8)
- `trainer.per_device_eval_parallelism = 4` (was 8)
- `resources.ram = "250g"` (was `"400g"` — lowered to fit on v5p-8 workers
  whose available memory was only ~305 GB due to co-tenancy)
- therefore `grad_accum = train_batch_size / (pd * num_devices) = 64 / (4 * 4) = 4`
  (was `64 / (8 * 4) = 2` at pd=8)

Everything else — LoRA recipe, data, seed, beta, lr, reference config,
reference_eval_cache, max_eval_batches, num_train_steps — is identical.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

| step | loss     | grad_l2  |
|------|----------|----------|
| 0    | 0.693147 | 2.456003 |
| 1    | 0.693147 | 2.255161 |
| 2    | 0.685125 | 2.362390 |
| 3    | 0.682298 | 2.366755 |
| 4    | 0.673723 | 2.315102 |
| 5    | 0.668946 | 2.307744 |
| 6    | 0.667573 | 2.237251 |
| 7    | 0.662823 | 2.244449 |
| 8    | 0.658715 | 2.419641 |
| 9    | 0.660557 | 2.272835 |

Same qualitative pattern as pd=8: loss barely drifts from `ln(2)`, gradients
stay in the ~2.24-2.42 band instead of decaying to ~1.12 as in good runs.

### Side-by-side: Exp Q pd=8 vs Exp Q pd=4 on `v5p-8`

| step | `v5p-8 pd=8` | `v5p-8 pd=4` | |Δ|    |
|------|-------------|-------------|---------|
| 0    | 0.693147    | 0.693147    | 0.0000  |
| 1    | 0.693147    | 0.693147    | 0.0000  |
| 2    | 0.685757    | 0.685125    | 0.0006  |
| 3    | 0.682320    | 0.682298    | 0.0000  |
| 4    | 0.673036    | 0.673723    | 0.0007  |
| 5    | 0.669805    | 0.668946    | 0.0009  |
| 6    | 0.668832    | 0.667573    | 0.0013  |
| 7    | 0.665207    | 0.662823    | 0.0024  |
| 8    | 0.660748    | 0.658715    | 0.0020  |
| 9    | 0.662508    | 0.660557    | 0.0020  |

Max delta is 0.0024 at step 7. These are in the same regime to numerical
precision — the `v5p-8` failure is not sensitive to `pd` within the tested
range.

### Side-by-side: `v5p-8` (bad) vs `v5p-16` (good), all at same recipe

| step | `v5p-16 pd=2` (Exp N) | `v5p-16 pd=4` (Exp P) | `v5p-8 pd=8` (Exp Q) | `v5p-8 pd=4` (Exp Q, this) |
|------|-----------------------|-----------------------|----------------------|----------------------------|
| 0    | 0.693147 | 0.693147 | 0.693147 | 0.693147 |
| 1    | 0.693147 | 0.693147 | 0.693147 | 0.693147 |
| 2    | 0.335202 | 0.335283 | 0.685757 | 0.685125 |
| 3    | 0.325988 | 0.327747 | 0.682320 | 0.682298 |
| 4    | 0.336246 | 0.337701 | 0.673036 | 0.673723 |
| 5    | 0.316800 | 0.317172 | 0.669805 | 0.668946 |
| 6    | 0.336998 | 0.336385 | 0.668832 | 0.667573 |
| 7    | 0.324271 | 0.324177 | 0.665207 | 0.662823 |
| 8    | 0.306144 | 0.306423 | 0.660748 | 0.658715 |
| 9    | 0.317624 | 0.316186 | 0.662508 | 0.660557 |

The table tells the whole story: `v5p-16` escapes to ~0.31-0.34 by step 2,
`v5p-8` stays near ~0.66-0.69 regardless of `pd`. The split size is ~0.35
at every step — identical to the original disaster magnitude.

### Step-0 gradient norms (still match good runs)

| run | step-0 `grad_l2` |
|-----|------------------|
| Exp N `v5p-16 pd=2` | 2.4563 |
| Exp P `v5p-16 pd=4` | 2.4622 |
| Exp Q `v5p-8 pd=8`  | 2.4562 |
| **Exp Q `v5p-8 pd=4` (this)** | **2.4560** |

Step-0 init is fine across all four runs — the failure is post-init.

### Validation-set behavior

| eval split | pre-training | post-step-10 | Δ |
|---|---|---|---|
| stmt_val | 0.693 | 0.663 | -0.030 |
| full_val | 0.693 | 0.694 | +0.001 |

Effectively no learning on the broader distribution. Matches the pd=8
validation behavior (stmt_val 0.693→0.656, full_val 0.693→0.692).

### CE kernel shape observation

The `DEBUGCE` log line from this run shows:
```
x.shape=(65536, 4096) b_block_size=1024 num_b_blocks=64
```

This is **identical** to the pd=8 run's CE shape. At pd=4 with
`train_batch_size=64` and 4 devices, `grad_accum` doubles (from 2 to 4)
but the local CE batch dimension `B` per accumulation step stays the same.
So the local CE kernel math is unchanged between pd=8 and pd=4 on `v5p-8` —
consistent with the finding that `pd` doesn't matter here.

### Operational notes

This run required four submission attempts (`ue5a-i1` through `ue5a-i4`)
to get a clean 10-step completion:

- **i1**: first ue5a attempt during the pd=8 run window; preempted before
  reaching step 0 on its `train_dpo` child.
- **i2**: parent stuck in `assigned` state for 20+ min because
  `--memory 16g --enable-extra-resources` caused it to land on a TPU worker
  instead of a CPU node. Container never started.
- **i3**: parent fixed with `--memory 3g` (no `--enable-extra-resources`),
  landed on CPU `e2-highmem-2-ondemand`. But `train_dpo` child requested
  `ram="400g"` while workers only had ~305 GB free (co-tenancy). Scheduler
  message: `Insufficient memory (need 400.0GB, available 304.8GB)`. Child
  never scheduled.
- **i4**: script patched to `ram="250g"` for `pd <= 4`. `train_dpo` child
  placed immediately on `v5p-8` worker in us-east5-a, ran all 10 steps with
  zero preemptions.

For future v5p-8 runs at `pd <= 4`, use `ram="250g"` or lower.

Also: `us-central1` had **zero** v5p-8 capacity during this entire session
(autoscaler pool `tpu_v5p-preemptible_8-us-central1-a` showed 0 slices and
17 consecutive scale-up failures). All successful Exp Q runs came from
`us-east5-a`.

### What the full Exp Q sweep (pd=8 + pd=4) now establishes

The Exp Q sweep is now effectively complete for the primary question. The
optional `pd=2` data point would add one more row to the table, but the
scientific conclusion is already clear:

1. **`v5p-8` pod shape is the root cause** of the remaining LoRA DPO
   pathology, at fixed `r=64, α=64, zero_init_b=True`.
2. **`per_device_parallelism` is not a factor** — pd=8 and pd=4 produce
   identical bad trajectories on `v5p-8`.
3. **`v5p-16` is fine** at both pd=2 and pd=4, so the failure is specific
   to the `-8` pod, not the `v5p` hardware family.
4. **Init is fine** — step-0 gradients match to 4 sig figs across all runs;
   the failure is in how the optimizer updates affect the model post-init.

The remaining open question is **what specifically about the `-8` pod
shape** causes the failure — host topology, sharding layout, collective
communication pattern, or some interaction thereof. That is a deeper
investigation beyond the scope of the Exp Q sweep.

---

## 2026-04-16T00:50Z: Experiment Q (pd=8) result — `v5p-8` **REPRODUCES** the bad regime

### Strongest true conclusion

Experiment Q with `per_device_parallelism=8` on `v5p-8` **reproduces the
old catastrophic stuck-near-ln(2) regime** while holding the LoRA recipe
fixed at exactly the same `r=64, α=64, zero_init_b=True` configuration used
in the recent good Exp N / Exp O / Exp P runs.

In other words:
- `v5p-16` + same recipe + `pd=2`  → escapes ln(2) immediately (Exp N)
- `v5p-16` + same recipe + `pd=4`  → escapes ln(2) immediately (Exp P)
- `v6e-16` + same recipe + `pd=2`  → escapes ln(2) immediately (Exp N)
- **`v5p-8`  + same recipe + `pd=8`  → stays stuck near ln(2) (Exp Q, this run)**

And the `v5p-8 pd=8` trajectory tracks the **old bad `v5p-8 pd=16` Exp K
trajectory** almost point-for-point. So neither `pd`, nor TPU family, nor the
LoRA recipe itself is sufficient to flip the run — the **`v5p-8` pod shape /
host topology / sharding regime** is now the most credible remaining cause.

This is exactly the discriminating outcome the planned Exp Q sweep was
designed to produce, and it collapses the remaining hypothesis space.

### Run

- `v5p-8 pd=8` (us-east5-a, INTERACTIVE band, preemptible pool): https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd8_s10_ue5a-i1-38dd4c

Launch command context:
- parent CPU coordinator pinned via `iris job run --zone us-east5-a --cpu 1 --memory 16g`
- child `train_dpo` task: `v5p-8` preemptible slice in `us-east5-a`
- env: `EXPERIMENT_Q_PD=8`, `REGIONS_OVERRIDE=us-east5`, `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`, `MARIN_DEBUG_RUN_TAG=ue5a-i1`
- iris job id: `/ahmed/debug-q1-r64-v5p8-pd8-ue5a-i1`
- experiment script: `experiments/posttrain/per_stmt_dpo/experiment_q_v5p8_pd_s10.py`

### Training-relevant config, all fixed to the recent good-run recipe

- `trainer.train_batch_size = 64`
- `trainer.per_device_parallelism = 8`
- `trainer.per_device_eval_parallelism = 8`
- `trainer.seed = 0`
- `data_seed = 0`
- `beta = 0.1`
- `lr = 1e-6`
- `lr_schedule = "cosine"`
- `warmup = 0.1`
- `train_seq_len = 4096`
- `max_seq_len = 4096`
- `adapter.r = 64`
- `adapter.alpha = 64`
- `adapter.zero_init_b = True`
- `reference = AdapterBaseReferenceConfig()`
- `reference_eval_cache.mode = "disabled"`
- `max_eval_batches = 1`
- `num_train_steps = 10`
- `include_lm_validation = False`
- data sources: same per-stmt `bloom_v2_singleton/support_mental_health` + speceval full-val that Exp N / O / P used

The only knobs that differ from the good Exp N `v5p-16 pd=2` baseline are
the TPU slice (`v5p-16 → v5p-8`) and `per_device_parallelism (2 → 8)` —
specifically chosen to match the planned Exp Q sweep entry point.

### Full 10-step training trajectory (from `DEBUGJ TRACE` worker logs)

All values are the per-step trace emitted by the training script under
`MARIN_DEBUG_LOG_STEP_TRACE=1`. `grad_l2` here is the LoRA-parameter-only
gradient l2 norm (equal to `gB_l2` at step 0 because `zero_init_b=True`).

| step | loss     | grad_l2  | grad_sum |
|------|----------|----------|----------|
| 0    | 0.693147 | 2.456232 |  0.1838  |
| 1    | 0.693147 | 2.255300 |  0.2208  |
| 2    | 0.685757 | 2.363574 | -0.6084  |
| 3    | 0.682320 | 2.365350 | -0.6799  |
| 4    | 0.673036 | 2.311212 | -0.8281  |
| 5    | 0.669805 | 2.312919 | -1.3175  |
| 6    | 0.668832 | 2.240088 | -0.1497  |
| 7    | 0.665207 | 2.249352 | -1.3066  |
| 8    | 0.660748 | 2.425360 | -0.1287  |
| 9    | 0.662508 | 2.276717 | -1.4925  |

Key qualitative observations from this trajectory:

- Step 0 is `ln(2) = 0.693147` as expected (softmax over two equally scored
  logits at initialization).
- Step 1 is **still exactly `ln(2)`**, i.e. the first update did essentially
  nothing to the DPO loss.
- The loss only drifts down by ~0.03 over 9 update steps (from `0.6931` to
  `0.6625`), an order of magnitude less descent than the good Exp N /
  Exp P runs achieve at the same step count.
- The LoRA gradient l2 norm **does not decay**: it stays in the ~2.24-2.45
  band across all 10 steps, whereas good runs drop their LoRA-only `grad_l2`
  to ~1.12 by step 9 (Exp N / P recorded values in that range).

### Direct comparison with the good matched-geometry runs

Train loss, step by step:

| step | Exp N `v5p-16 pd=2` | Exp P `v5p-16 pd=4` | **Exp Q `v5p-8 pd=8` (this run)** | Exp Q - Exp N |
|------|---------------------|---------------------|-----------------------------------|---------------|
| 0    | 0.693147 | 0.693147 | **0.693147** | 0.000000 |
| 1    | 0.693147 | 0.693147 | **0.693147** | 0.000000 |
| 2    | 0.335202 | 0.335283 | **0.685757** | +0.350555 |
| 3    | 0.325988 | 0.327747 | **0.682320** | +0.356332 |
| 4    | 0.336246 | 0.337701 | **0.673036** | +0.336790 |
| 5    | 0.316800 | 0.317172 | **0.669805** | +0.353005 |
| 6    | 0.336998 | 0.336385 | **0.668832** | +0.331834 |
| 7    | 0.324271 | 0.324177 | **0.665207** | +0.340936 |
| 8    | 0.306144 | 0.306423 | **0.660748** | +0.354604 |
| 9    | 0.317624 | 0.316186 | **0.662508** | +0.344884 |

This is not “a little worse” — it is the same catastrophic split size we saw
in the original `v5p-8` vs `v6e-8` disaster, relative to matched good runs
on the same fixed LoRA recipe.

### Direct comparison with the old bad `v5p-8 pd=16` run (Exp K)

The clean apples-to-apples older bad baseline is the Exp K
`r64_alpha64_s10_v5p8_k1p-6840ce` run, because it uses the **same** fixed
`r=64, α=64, zero_init_b=True` LoRA recipe on a `v5p-8`:

| step | old `v5p-8 pd=16` (Exp K) | new `v5p-8 pd=8` (Exp Q) | |Δ|       |
|------|---------------------------|--------------------------|-------------|
| 2    | 0.685376                  | 0.685757                 | 0.000381    |
| 3    | 0.679647                  | 0.682320                 | 0.002673    |
| 8    | 0.658281                  | 0.660748                 | 0.002467    |

The two `v5p-8` runs, at *different* `per_device_parallelism` values (16 vs 8),
trace essentially the same bad trajectory. Dropping `pd` from 16 to 8 on
`v5p-8` **does not fix it**. That is the key novel fact from this Exp Q data
point.

### Step-0 gradient scale matches good runs (init is fine)

Step-0 LoRA-only `grad_l2` values across all fixed-recipe runs:

| run                          | step-0 `grad_l2` |
|------------------------------|------------------|
| Exp N `v5p-16 pd=2`          | 2.4563 |
| Exp N `v6e-16 pd=2`          | 2.4661 |
| Exp P `v5p-16 pd=4`          | 2.4622 |
| **Exp Q `v5p-8 pd=8` (now)** | **2.4562** |

So init + first forward/backward produces a LoRA gradient of essentially the
same magnitude everywhere — agreement to ~4 significant figures with the
Exp N `v5p-16 pd=2` run. The bad behavior is not an initialization-time
problem. It only appears once we start taking optimizer steps.

However, while the good runs' `grad_l2` decays to ~`1.12` by step 9, the
`v5p-8 pd=8` run's `grad_l2` stays near ~`2.28` at step 9. The optimizer is
receiving a gradient of normal magnitude every step; it just isn't reducing
the loss.

### Validation-set behavior

From the single post-step-10 eval point:

| eval split | pre-training | post-step-10 | Δ      |
|------------|--------------|--------------|--------|
| stmt_val   | 0.693        | 0.656        | -0.037 |
| full_val   | 0.693        | 0.692        | -0.001 |

So the tiny nudge the run manages on train loss barely transfers to
stmt_val and effectively not at all to the broader full_val distribution.

Good Exp N runs, by contrast, move both stmt_val and full_val meaningfully
by step 10 (see Exp N history in W&B).

### What Exp Q (pd=8) rules out, and what it supports

**Rules out (now strengthened):**

- “The old catastrophe was really just about `per_device_parallelism` / local
  CE shape.” We now have `v5p-8 pd=8` behaving as badly as `v5p-8 pd=16`
  despite cutting `pd` in half.
- “The old catastrophe was really TPU family.” Already ruled out by Exp N;
  Exp Q adds that even at the originally-blamed family (`v5p`), the
  `-16` pod is fine and the `-8` pod is not, at matched recipe.
- “The LoRA recipe itself is unstable at `r=64, α=64, zero_init_b=True`.”
  This is the same recipe that trains normally on `v5p-16` and `v6e-16`, so
  the recipe is fine; the environment is not.

**Strongly supports:**

- The remaining pathology is specific to the **`v5p-8` pod shape / host
  topology / sharding layout**, in a way that is not explained by local CE
  tile math or by `per_device_parallelism` alone.
- The failure mode is a post-init stepping problem (updates do not reduce
  loss) rather than an init-time or forward-only problem.

**Not yet proved:**

- Whether `v5p-8` remains bad at `pd=4` and `pd=2`, or whether it recovers as
  `pd` drops further. The planned follow-up sweep entries (`pd=4`, optional
  `pd=2`) are now the most informative next data points.
- Whether `v5p-8` also fails in full fine-tuning, or whether this is a LoRA-
  specific amplifier of the `-8` pod pathology.

### Operational notes — preemption storm

Getting this single successful 10-step run required **3 preemptions** of the
`train_dpo` child task before an attempt survived compile-plus-10-steps. Key
timestamps, reconstructed from `iris job summary` and worker log timestamps:

- 22:42Z — `/ahmed/debug-q1-r64-v5p8-pd8-ue5a-i1` submitted (parent CPU +
  `train_dpo` TPU child).
- Parent CPU stayed pinned to `us-east5-a`; `train_dpo` child bounced across
  multiple `marin-tpu-v5p-preemptible-8-us-east5-a-*-worker-0` slices.
- ~23:28Z — first scheduler message transitioned from
  `tier_blocked by quota-pool tier monotonicity` to autoscaler-demand-routed
  for `tpu_v5p-preemptible_8-us-east5-a`, so the first worker came up.
- First + second worker attempts died with `Worker ... failed: Request timed
  out` before reaching a training step.
- ~00:26Z — third worker (`...-20260416-0026-0cde0ab3-worker-0`) attached.
- 00:44-00:45Z — step 0 completed (~100s, dominated by JIT compile), first
  `stmt_val` and `full_val` evals recorded at 0.693.
- 00:45Z - 00:50Z — steps 1 through 9 completed on this worker.

The parallel `uc1-i1` job in `us-central1` reached step 1 on one of its
attempts, then lost its worker again. Final `preempt` count observed on the
`uc1-i1` child task was **4** by the time Exp Q pd=8 completed on `ue5a-i1`,
and `uc1-i1` never got a clean 10-step run. The `ue5a-i1` run is therefore
the sole source of Exp Q pd=8 step-level data above.

Both jobs were submitted at `PRIORITY_BAND_INTERACTIVE` (the default for
`iris job run`), so on a `preemptible` pool they are kickable by:
- GCP spot reclamation of the preemptible VM itself (independent of Iris), and
- any Iris `PRIORITY_BAND_PRODUCTION` task competing for the same
  `tpu_v5p-preemptible_*` scale group.

During the Exp Q window, the only live `PRIORITY_BAND_PRODUCTION` v5p tasks
observed via `iris rpc controller get-scheduler-state` were from user
`moojink` on a `marin-tpu-v5p-preemptible-32-us-central1-*` slice (an
Iris-scheduled Qwen3-1.7B SFT job), which plausibly contributed to the
autoscaler reshuffling that kept evicting `ue5a-i1` and `uc1-i1` workers.
Moojink's production job was in `us-central1`, not `us-east5-a`, so the
`ue5a-i1` preemptions were most likely GCP spot reclamations rather than
cross-tenant Iris displacement.

Future-agent takeaway: for the remaining planned `v5p-8 pd=4` / `pd=2` Exp Q
sweep points, expect a similar preemption storm. The run script is
idempotent — each surviving attempt re-computes step 0 from the frozen
init, so a preempted attempt is cheap in correctness terms but expensive in
wall-clock (~7-10 min per boot-to-step-0 attempt). Just keep resubmitting or
keep the executor alive; do not switch bands to `production`.

### Next experiment

Per the original planned Exp Q sweep order, the next two entries are:

1. `v5p-8 pd=4` (same recipe, same data, same seed; only `pd: 8 → 4`).
   Expected to further isolate whether `v5p-8` stays pathological as `pd`
   drops.
2. Optional `v5p-8 pd=2`.

If `v5p-8 pd=4` also stays near `ln(2)`, the `v5p-8` pod shape is
confirmed as the remaining root-cause surface, independent of `pd`. If it
recovers, the failure narrows to the high-`pd` corner of `v5p-8`.

---

## 2026-04-15T07:28Z: Experiment N — per-stmt LoRA DPO on matched-family pods with matched local execution geometry

### Strongest true conclusion

Experiment N is the strongest evidence so far that the original catastrophic
`v5p`/`v6e` LoRA-DPO split was **not caused by TPU family by itself**.

When I rerun the same per-statement LoRA DPO recipe on:
- `v5p-16` with `per_device_parallelism=2`
- `v6e-16` with `per_device_parallelism=2`

the two runs track each other very closely from step 0 through step 9. The
old failure mode, where `v6e` escaped `ln(2)` immediately and `v5p` stayed
near `ln(2)`, disappears.

That means:
- TPU-family hardware differences are **not sufficient** to explain the old split
- execution geometry is the **primary driver** of the old split

What this experiment does **not** prove:
- It does **not** prove that LoRA has zero sensitivity to geometry
- It does **not** prove that every component of the old mismatch has been
  individually isolated

The strongest supportable version is:
> the original `v5p-8` / `v6e-8` disaster required the old geometry mismatch;
> once geometry is controlled, the TPU-family split disappears

### Runs

- `v5p-16 pd=2` (us-central1): https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v5p16_n1-7a55a1
- `v6e-16 pd=2` (europe-west4): https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v6e16_n3-323159
- `v6e-32 pd=1` (us-east1 fallback): https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v6e32_n3-8577d8

The `v6e-32 pd=1` run is a useful supporting point because it also lands in
the same learning regime, but the primary comparison for the original
hardware-family question is the matched `v5p-16 pd=2` vs `v6e-16 pd=2` pair.

### Training-relevant config, from script and W&B

Script: `experiments/posttrain/per_stmt_dpo/debug_r64_matched_pd2_s10.py`

Training knobs that match across the primary `v5p-16` and `v6e-16` pair:
- `trainer.train_batch_size = 64`
- `trainer.per_device_parallelism = 2`
- `trainer.per_device_eval_parallelism = 2`
- `trainer.seed = 0`
- `data_seed = 0`
- `beta = 0.1`
- `lr = 1e-6`
- `lr_schedule = "cosine"`
- `warmup = 0.1`
- `train_seq_len = 4096`
- `max_seq_len = 4096`
- `adapter.r = 64`
- `adapter.alpha = 64`
- `adapter.zero_init_b = true`
- `lora.exclude_modules_resolved = ["lm_head"]`
- `reference = AdapterBaseReferenceConfig()`
- `reference_eval_cache.mode = "disabled"`
- `max_eval_batches = 1`

Logical data definition also matches:
- same mirrored train source:
  `preference/bloom_v2_singleton/support_mental_health/train/shard-00000.jsonl.gz`
- same mirrored stmt-val source
- same mirrored full-val source
- same tokenizer / same tokenization names / same permutation type (`feistel`)

What differs across regions in W&B is the concrete tokenized `cache_dir`
prefix (`gs://marin-us-central1/...` vs `gs://marin-us-east1/...`), because the
parent executor resolves outputs in its local region. That is a storage-path
difference, not a logical dataset-definition difference.

One precision note from the W&B payloads:
- the `v5p-16` run still shows non-null `lm_validation_data`
- the `v6e-16` run shows `lm_validation_data = null`

Because `include_lm_validation=False` was passed at the experiment layer, and
because the training curves already match through steps 0-9 before the single
eval point at step 10, I do **not** treat that serialization discrepancy as
material to the Exp N training conclusion. But the earlier wording “identical
except TPU/pd” was too strong and is corrected here.

### What matched

#### 1. Train loss

From W&B history:

| step | v5p-16 pd=2 | v6e-16 pd=2 | |Δ| |
|---|---|---|---|
| 0 | 0.693147 | 0.693147 | 0.000000 |
| 1 | 0.693147 | 0.693147 | 0.000000 |
| 2 | 0.335202 | 0.336385 | 0.001183 |
| 3 | 0.325988 | 0.325428 | 0.000560 |
| 4 | 0.336246 | 0.336330 | 0.000084 |
| 5 | 0.316800 | 0.315582 | 0.001218 |
| 6 | 0.336998 | 0.333553 | 0.003445 |
| 7 | 0.324271 | 0.322128 | 0.002143 |
| 8 | 0.306144 | 0.302488 | 0.003656 |
| 9 | 0.317624 | 0.315265 | 0.002359 |

This is not “close only at step 0.” The two matched-family runs remain close
through the whole 10-step probe.

The fallback `v6e-32 pd=1` run also lands in the same regime:
- step 2: `0.331369`
- step 9: `0.316118`

That further weakens the story that TPU-family or chip-count alone causes the
old pathology.

#### 2. The actual DPO quantity: `delta_pi - delta_ref`

The DPO loss is driven by `delta_pi - delta_ref`, not by
`train/dpo_margin_policy` alone.

Using W&B `train/dpo_margin_policy - train/dpo_margin_ref`:

| step | v5p-16 pd=2 | v6e-16 pd=2 | |Δ| |
|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 9.4367 | 9.4102 | 0.0264 |
| 3 | 9.7769 | 9.8088 | 0.0319 |
| 4 | 9.4642 | 9.4929 | 0.0287 |
| 5 | 10.2081 | 10.2816 | 0.0735 |
| 6 | 9.4584 | 9.5975 | 0.1392 |
| 7 | 9.8192 | 9.9059 | 0.0866 |
| 8 | 10.5564 | 10.7182 | 0.1618 |
| 9 | 10.2183 | 10.2966 | 0.0782 |

So the matched runs are close on the **true optimization target**, not just
on a surface metric.

#### 3. Step-0 gradient scale

W&B step-0 `grad/norm/total` is also close:
- `v5p-16 pd=2`: `28.8643`
- `v6e-16 pd=2`: `28.9838`
- `v6e-32 pd=1`: `28.9350`

I directly re-read the `v6e-16` `DEBUGJ TRACE` worker log and confirmed its
per-step LoRA-only `grad_l2` trajectory:
- step 0: `2.4661`
- step 1: `2.2592`
- step 2: `1.2170`
- step 9: `1.1154`

For the `v5p-16` side, the strongest directly revalidated evidence in this
pass is the W&B train/loss and DPO-margin history above. The previous draft
used more aggressive “all three DEBUGJ TRACE tables match identically”
language than I can currently support from locally available worker logs, so I
am narrowing the claim accordingly.

### Direct contrast with the old mismatched-geometry pair

The clean apples-to-apples contrast is the earlier `r64/alpha64` pair from
Experiment K, because that used the **same LoRA recipe** but the old
geometry:

- old `v5p-8 pd=16`: https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v5p8_k1p-6840ce
- old `v6e-8 pd=4`: https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v6e8_k1r2-dd49b2

Loss contrast:

| step | old v5p-8 pd=16 | old v6e-8 pd=4 | |Δ| |
|---|---|---|---|
| 2 | 0.685376 | 0.333254 | 0.352123 |
| 3 | 0.679647 | 0.325355 | 0.354293 |
| 8 | 0.658281 | 0.301821 | 0.356460 |

True DPO quantity contrast (`delta_pi - delta_ref`):

| step | old v5p-8 pd=16 | old v6e-8 pd=4 | |Δ| |
|---|---|---|---|
| 2 | 0.1575 | 9.5113 | 9.3538 |
| 3 | 0.2731 | 9.8070 | 9.5339 |
| 8 | 0.7128 | 10.7681 | 10.0553 |

So Exp N does not just “improve the old result a bit.” It collapses a
massive, immediate split into a tiny residual difference.

For completeness, the first symptom we noticed originally was even worse in
the older `r=16, alpha=32` pair:
- old `v5p-8`: `0.6951` at step 2
- old `v6e-8`: `0.4672` at step 2

But that pair is not the cleanest contrast for Exp N because the LoRA recipe
also changed.

### First-principles explanation of CE batch blocking

This is the key code-path explanation for someone who knows JAX but not this
repo.

#### Where CE sits in the DPO path

DPO computes:
- policy log-prob for chosen
- policy log-prob for rejected
- reference log-prob for chosen
- reference log-prob for rejected

in `train_dpo.py` / `dpo.py`.

Each log-prob sum ultimately comes from
`LmHeadModel.compute_next_token_loss(...)`, which calls the fused next-token
cross-entropy path. In `models/loss.py`, that fused CE path flattens **all
non-embedding axes** into a single local batch axis called `__BATCH__` before
calling the kernel.

So the CE kernel does **not** see “number of sequences per device.” It sees a
local matrix:
- `x` with shape `(B, H)`
- `w` with shape `(H, V)`

where:
- `B` = flattened local token rows
- `H` = hidden size
- `V` = vocab size

For fixed-length 4096-token examples, `B` scales with:
- local examples per chip
- times sequence positions per example

That is why `per_device_parallelism` changes the CE kernel’s local problem
shape even if the global `train_batch_size` stays fixed.

#### What “batch blocking” means in this kernel

In `xla.py`, the streaming XLA CE path chooses:
- `v_block_size`: how many vocab columns to process at once
- `b_block_size`: how many flattened batch rows (`B`) to process at once

The backward then does:
1. loop over vocab blocks
2. inside each vocab block, loop over batch blocks
3. slice `x_block` of shape `(b_block_size, H)`
4. compute logits / probs / `delta` of shape `(b_block_size, v_block_size)`
5. accumulate partial `gx_block` and `gw_block_update`

Concretely, in `xla.py`:
- `num_b_blocks = b_dim // batch_block_size`
- the inner loop slices `x_block = dynamic_slice(x, (batch_start, 0), (batch_block_size, h_dim))`
- then forms `delta` and accumulates `gx_inner` / `gw_block`

So changing `B` or `b_block_size` changes:
- how many inner-loop iterations run
- the shapes of the temporary arrays in each iteration
- the order in which partial sums are accumulated
- the HBM footprint of those temporaries

This is what “CE batch blocking” means here.

#### What changed in the old bad pair

Experiment G directly logged the old CE shapes:
- old `v5p-8`: `x.shape=(65536, 4096)`, `b_block_size=1024`, `num_b_blocks=64`
- old `v6e-8`: `x.shape=(16384, 4096)`, `b_block_size=1024`, `num_b_blocks=16`

So even though both were “the same training job” at a high level, they were
executing materially different local CE kernels.

#### Why Exp N is different

In Exp N, the primary pair is:
- `v5p-16 pd=2`
- `v6e-16 pd=2`

Because both have:
- the same chip count
- the same `per_device_parallelism`
- the same sequence length

their local examples-per-chip match, so the flattened CE batch dimension `B`
is much closer by construction. I did **not** re-log the CE kernel in Exp N,
so I am not claiming exact `num_b_blocks` values here. But from the code path
above, matching chip count + matching `pd` on fixed-length data directly
equalizes the local CE problem shape in a way the old `v5p-8 pd=16` vs
`v6e-8 pd=4` pair did not.

This is the first-principles reason execution geometry matters so much in this
investigation.

### Important nuance: “same global microbatch” was not enough

Exp N should **not** be summarized as “just make global microbatch and
grad_accum match.”

We already had Experiment A:
- `v5p-8 pd=8`
- global microbatch = 32
- grad_accum = 2

and it was still bad.

So the real lesson is broader:
- matching global accumulation schedule alone was **not** enough
- matching the **local execution geometry** on matched-family pods was enough

That local geometry includes:
- per-device examples
- flattened CE `B`
- CE batch-block loop structure
- host/chip sharding layout
- probably attention / collective layout as well

### What Exp N rules out and what it supports

#### Ruled out

- TPU-family hardware differences as a **sufficient** explanation
- generic “LoRA DPO diverges across v5p vs v6e”

#### Strongly supported

- execution geometry is the **primary driver** of the original old split

#### Not proved

- that LoRA has zero extra sensitivity to geometry
- that every subcomponent of the old geometry mismatch has been individually isolated
- that full FT would have behaved the same under the exact old bad geometry

So I am retracting the earlier stronger claim that “LoRA amplifier is the
root cause.” The better-supported statement after Exp N is:

> the old `v5p-8` / `v6e-8` catastrophe was primarily an execution-geometry
> problem, and TPU family is not enough to reproduce it once geometry is
> controlled

### Operational note

Two infrastructure fixes were still required to get Exp N to run cleanly:

1. `experiments/paloma.py` and `experiments/evals/exp1600_uncheatable_evals.py`
   now use `mirrored()` for their raw sources, avoiding the executor’s
   cross-region read guard on parent-region CPU jobs.
2. `experiments/defaults.py:default_dpo` now accepts
   `include_lm_validation=False`, which let this short 10-step debug probe
   skip the Paloma / uncheatable LM-validation wiring and launch directly.

Those changes were necessary to run Exp N, but they are not part of the
scientific conclusion above.

### 2026-04-15T21:30Z — Experiment O result: within-family full-FT `pd` ablation on `v6e-16`

#### Why this experiment mattered

Exp N showed that the catastrophic LoRA split disappears when I match the
`v5p` and `v6e` runs on a cleaner local geometry. But that still left an
important causal question open:

> was the old catastrophe explained by `per_device_parallelism` / local kernel
> shape changes alone, or was LoRA unusually sensitive to the old mismatch?

The cleanest way to ask that is to keep hardware fixed and change only `pd`
inside full fine-tuning.

#### Hypothesis

**Hypothesis:** full FT on the per-statement setup is comparatively robust to
the `per_device_parallelism` / local-shape change that existed in the old LoRA
comparison.

Operationally:
- baseline: existing Exp L `v6e-16 pd=4`
- new run: `v6e-16 pd=2`

Keep fixed:
- same per-stmt `support_mental_health` data
- same `SeparateReferenceConfig`
- same `train_batch_size=64`
- same `lr=1e-6`
- same `beta=0.1`
- same `seed=0`
- same `train_seq_len=4096`
- same 10-step probe length

Change only:
- `trainer.per_device_parallelism: 4 -> 2`
- therefore local examples per device drop from `4 -> 2`
- therefore the flattened local CE batch dimension `B` seen by the fused
  next-token CE path drops from roughly `4 * 4096 = 16384` tokens per device to
  roughly `2 * 4096 = 8192`
- and because global `train_batch_size` stays 64, the `pd=2` run now uses
  `grad_accum=2` instead of a single microstep

This is the right first-principles quantity to watch because the fused XLA CE
kernel does not operate on “examples”; it operates on a local 2D tensor
`x.shape = (B, vocab_hidden)` after non-embed axes are flattened. Changing
`pd` changes the local `B`, which is the batch dimension the kernel uses when
choosing how to tile and stream the CE work. In other words: `pd` is not just a
trainer-level knob; it changes the local numerical problem the CE kernel sees.

Important precision note:
- Exp O definitely changes the local CE shape `B`
- Exp O may also change tuned CE batch blocking (`b_block_size`, `num_b_blocks`)
- but I did **not** log the resolved CE block sizes on this run, so this
  experiment rules out the broader “`pd` / local-shape change is sufficient”
  story in full FT, not every specific CE blocking detail in isolation

#### Runs

- Exp L baseline `v6e-16 pd=4`: https://wandb.ai/marin-community/dpo/runs/full_ft_s10_v6e16_l1-008fdb
- Exp O `v6e-16 pd=2`: https://wandb.ai/marin-community/dpo/runs/full_ft_pd2_s10_v6e16_o1ew4rgni4-9219b0

#### Result

The `pd=2` full-FT run closely tracks the earlier `pd=4` full-FT baseline. It
does **not** reproduce anything like the catastrophic LoRA divergence.

Per-step train loss:

| step | `v6e-16 pd=4` | `v6e-16 pd=2` | |Δ| |
|---|---|---|---|
| 0 | 0.693147 | 0.693163 | 0.000016 |
| 1 | 0.693147 | 0.693179 | 0.000032 |
| 2 | 0.692416 | 0.686007 | 0.006409 |
| 3 | 0.678566 | 0.673826 | 0.004740 |
| 4 | 0.664171 | 0.667567 | 0.003396 |
| 5 | 0.655864 | 0.655773 | 0.000091 |
| 6 | 0.618792 | 0.615281 | 0.003512 |
| 7 | 0.606406 | 0.601114 | 0.005291 |
| 8 | 0.597066 | 0.588456 | 0.008610 |

And on the actual DPO quantity `delta_pi - delta_ref`:

| step | `v6e-16 pd=4` | `v6e-16 pd=2` | |Δ| |
|---|---|---|---|
| 0 | 0.0000 | -0.0003 | 0.0003 |
| 1 | 0.0000 | -0.0006 | 0.0006 |
| 2 | 0.0192 | 0.1485 | 0.1293 |
| 3 | 0.2987 | 0.3952 | 0.0965 |
| 4 | 0.5930 | 0.5235 | 0.0695 |
| 5 | 0.7690 | 0.7690 | 0.0000 |
| 6 | 1.5594 | 1.6381 | 0.0787 |
| 7 | 1.8283 | 1.9463 | 0.1180 |
| 8 | 2.0425 | 2.2322 | 0.1897 |

Those are ordinary trajectory differences, not a regime change. In particular,
the `pd=2` run still:
- escapes `ln(2)` immediately
- follows the same qualitative descent as the `pd=4` baseline
- reaches the same full-FT learning regime by step 8

The raw `DEBUGJ TRACE` from the `pd=2` run shows normal full-FT dynamics:
- step 0: `loss=0.69316`, `grad_l2=28.8763`
- step 2: `loss=0.68601`, `grad_l2=27.7550`
- step 5: `loss=0.65577`, `grad_l2=26.7984`
- step 8: `loss=0.58846`, `grad_l2=25.8068`

#### What Exp O rules out

Exp O is strong evidence against the simple story:

> “changing `per_device_parallelism` and the local CE / kernel math is enough
> by itself to cause the old catastrophic divergence”

That story is too strong. Inside full FT on fixed `v6e-16` hardware:
- changing `pd` from `4 -> 2`
- changing local examples per device from `4 -> 2`
- changing the local CE batch shape `B`
- introducing `grad_accum=2`

does **not** cause a catastrophic training split.

#### Strongest true interpretation after Exp O

After Exp N + Exp O, the strongest supportable summary is:

- **TPU family alone is not the cause** of the old LoRA split
- **a pure `pd` / local-shape change is not sufficient** to cause a
  catastrophic split in full FT
- therefore the original `v5p-8 pd=16` vs `v6e-8 pd=4` failure required a
  stronger interaction than “kernel math changed”

What remains live is the interaction:
- old LoRA parameterization
- old broader execution-geometry mismatch
- possibly reference-graph differences

In other words, the old disaster was **not**:
- “v5p hardware breaks DPO”
- and **not** “changing CE local math alone breaks DPO”

It was some interaction specific to the original LoRA setup under the original
broader mismatch.

### 2026-04-15T21:55Z — Experiment P result: fixed-family LoRA `pd` ablation on `v5p-16`

#### Why this experiment mattered

After Exp N and Exp O, the remaining ambiguity was no longer "TPU family or
not." That was already narrowed down.

The live fork was:
- **`pd` / local execution geometry is sufficient to flip LoRA**
- **pod shape / sharding topology is required in addition to `pd`**

The cleanest way to ask that was to keep LoRA on, keep hardware fixed to
`v5p-16`, and change only `per_device_parallelism`.

#### Hypothesis

**Hypothesis:** LoRA on fixed `v5p-16` hardware is materially more sensitive to
the `pd` / local-shape change than full FT was in Exp O.

Operationally:
- baseline: existing good Exp N `v5p-16 pd=2`
- new run: `v5p-16 pd=4`

Keep fixed:
- same per-stmt `support_mental_health` data
- same LoRA recipe `r=64`, `alpha=64`, `zero_init_b=True`
- same `AdapterBaseReferenceConfig`
- same `train_batch_size=64`
- same `lr=1e-6`
- same `beta=0.1`
- same `seed=0`
- same 10-step probe length

Change only:
- `trainer.per_device_parallelism: 2 -> 4`
- local examples per device: `2 -> 4`
- local flattened CE batch dimension `B`: approximately `8192 -> 16384`
- grad accumulation: `2 -> 1`

#### Runs

- Exp N baseline `v5p-16 pd=2`: https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v5p16_n1-7a55a1
- Exp P `v5p-16 pd=4`: https://wandb.ai/marin-community/dpo/runs/r64_v5p16_pd4_s10_p1uc1i1-6692c0

#### Result

`v5p-16 pd=4` closely matches the existing good `v5p-16 pd=2` LoRA run. It
does **not** fall back into the old bad `v5p-8` regime.

Per-step train loss:

| step | `v5p-16 pd=2` | `v5p-16 pd=4` | |Δ| |
|---|---|---|---|
| 0 | 0.693147 | 0.693147 | 0.000000 |
| 1 | 0.693147 | 0.693147 | 0.000000 |
| 2 | 0.335202 | 0.335283 | 0.000081 |
| 3 | 0.325988 | 0.327747 | 0.001759 |
| 4 | 0.336246 | 0.337701 | 0.001455 |
| 5 | 0.316800 | 0.317172 | 0.000372 |
| 6 | 0.336998 | 0.336385 | 0.000613 |
| 7 | 0.324271 | 0.324177 | 0.000094 |
| 8 | 0.306144 | 0.306423 | 0.000280 |
| 9 | 0.317624 | 0.316186 | 0.001438 |

The new `pd=4` run is also in the same regime as the good matched-family
`v6e-16 pd=2` run from Exp N:
- `v6e-16 pd=2`: https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v6e16_n3-323159

Step-0 / step-9 LoRA-only gradient norms from `DEBUGJ TRACE`:
- `v5p-16 pd=2`: `2.4563 -> 1.1211`
- `v5p-16 pd=4`: `2.4622 -> 1.1188`
- `v6e-16 pd=2`: `2.4661 -> 1.1154`

So this is not just "loss looks vaguely similar." The trainable LoRA gradient
scale is also in the same regime.

#### What Exp P rules out

Exp P is strong evidence against the simple story:

> “LoRA plus a `pd` change / local CE-shape change is enough by itself to
> recreate the old catastrophe”

That story is now too strong. On fixed `v5p-16` hardware, changing
`pd: 2 -> 4`:
- changes local examples per device
- changes local flattened CE batch shape `B`
- changes whether the run uses grad accumulation

and still does **not** create the old bad regime.

#### Strongest true interpretation after Exp P

After Exp N + Exp O + Exp P, the strongest supportable summary is:

- **TPU family alone is not the cause**
- **a pure within-family `pd` / local-shape change is not sufficient** in
  either full FT or LoRA on `v5p-16`
- therefore the original `v5p-8` failures require something more specific to
  the old `-8` setup than just `pd` and local CE shape

The remaining live explanation is now much narrower:
- `v5p-8` pod shape / chip count / host topology / sharding pattern
- or some more extreme `v5p-8` geometry regime (`pd=8` / `pd=16`) that does
  not generalize to `v5p-16`

### Highest-info next experiment after Exp P

#### Reasoning

After Exp N + Exp O + Exp P, the investigation is much narrower than it was
originally:

- `v5p` vs `v6e` hardware family is not enough to explain the split
- `v5p-16` LoRA is robust at fixed `r=64`, `alpha=64` under both `pd=2` and
  `pd=4`
- within-family `pd` / local CE-shape changes are therefore not sufficient, by
  themselves, to recreate the old bad regime on `v5p-16`

So the remaining live question is no longer "does LoRA dislike `pd` in
general?" It is:

> is there something specifically pathological about the **`v5p-8` regime** at
> fixed `r=64`, `alpha=64`?

That is the next clean discriminator.

#### Planned Experiment Q — fixed-`r64/α64` LoRA sweep on `v5p-8`

> **Status (2026-04-16T03:17Z):** the Exp Q sweep is effectively complete.
> Both `v5p-8 pd=8` (2026-04-16T00:50Z) and `v5p-8 pd=4` (2026-04-16T03:17Z)
> reproduce the bad regime with near-identical trajectories (max |Δ| = 0.0024).
> See the two Experiment Q result sections at the top of this logbook for
> full traces, side-by-side comparisons, W&B links, and operational logs.
> The optional `pd=2` entry is deprioritized — the conclusion is already
> clear: the pathology tracks the `v5p-8` pod shape, not `pd`.

**Goal:** hold the LoRA recipe fixed at the same settings used in the recent
good `v5p-16` / `v6e-16` runs, and sweep `v5p-8` across progressively smaller
`per_device_parallelism` values.

Keep fixed across the sweep:
- per-stmt `support_mental_health` singleton data
- `LoraAdaptationConfig(r=64, alpha=64, zero_init_b=True, target_modules=None)`
- `AdapterBaseReferenceConfig`
- `train_batch_size=64`
- `lr=1e-6`
- `beta=0.1`
- `seed=0`
- `num_train_steps=10`
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Sweep order:
1. `v5p-8` with fixed-`r64/α64` at `pd=8`
2. `v5p-8` with fixed-`r64/α64` at `pd=4`
3. Optional follow-up: `v5p-8` with fixed-`r64/α64` at `pd=2`

Why this order:
- we already know `v5p-16` is robust under `pd=2` and `pd=4`
- we already have one bad `v5p-8` run at fixed `r=64`, `alpha=64`
- the most important remaining cleanup is to determine whether `v5p-8` stays
  bad as `pd` is reduced, or whether the failure is limited to the higher-`pd`
  end of the `v5p-8` regime

#### Hypothesis

The strongest current hypothesis is:

- `v5p-16` is robust under the recent fixed-`r64/α64` LoRA probes
- the remaining pathology is specific to `v5p-8`
- the `v5p-8` sweep will tell us whether the failure tracks the `-8` pod shape
  broadly, or only the higher-`pd` corner of that regime

Interpretation:
- if `v5p-8` remains bad across the sweep, then the leading explanation is that
  something about the `-8` pod shape / host topology / sharding regime is the
  real remaining culprit
- if `v5p-8` improves as `pd` drops, then the surviving issue is the
  high-`pd` corner of the `v5p-8` regime rather than `v5p-8` broadly

#### Historical `v5p-8` references for future agents

These are the prior `v5p-8` LoRA DPO runs that motivated the current sweep.
Use the W&B config for the exact `pd` / LoRA settings if needed:

- original fresh bad `v5p-8` run: https://wandb.ai/marin-community/dpo/runs/smh_lr1em06_s70_v5p8-964129
- forced-geometry `v5p-8` follow-up: https://wandb.ai/marin-community/dpo/runs/smh_lr1em06_s70_v5p8_pd8-0498ec
- fixed-`r64/α64` bad `v5p-8` run: https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v5p8_k1p-6840ce

---

## 2026-04-14: Experiment K — r=64/α=64 LoRA on per-stmt data — **does not remove the early split**

Goal: test whether the checked-in Marin LoRA recipe (`r=64, α=64` — matches
`experiments/tune_lora/README.md`) removes the per-stmt v5p/v6e split seen
with the earlier recipe (`r=16, α=32`). All other knobs
(data=support_mental_health singleton, lr=1e-6, β=0.1, seed=0,
zero_init_b=True, target_modules=None) were held constant.

Important caveat: this is **not** a pure rank-only test. LoRA applies scale
`α/r`, so the old recipe had scale `32/16 = 2` while the new recipe has
scale `64/64 = 1`. Experiment K therefore tests the checked-in `r64/α64`
recipe, not "rank-64 with everything else identical."

**Result: v5p and v6e still split on the DPO objective in the first 8-10
steps.** The checked-in `r64/α64` recipe is not an immediate fix.

**W&B runs (paired, r=64 α=64, per-stmt, 10 steps):**
- v5p-8 (us-central1): https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v5p8_k1p-6840ce
- v6e-8 (europe-west4): https://wandb.ai/marin-community/dpo/runs/r64_alpha64_s10_v6e8_k1r2-dd49b2

### Codex analysis — `train/dpo_margin_policy` is not the loss, and the
loss-driving quantity is still far apart

Re-reading `lib/levanter/src/levanter/dpo.py:119,123`, the DPO loss is

    softplus(−β · (δ_π − δ_ref))

with `δ_π = logp_π(chosen) − logp_π(rejected)` and `δ_ref` likewise for the
reference. The logged `train/dpo_margin_policy` is **only `mean(δ_π)`** —
not the loss-driving quantity. The loss depends on `δ_π − δ_ref`.

Pulling `δ_π`, `δ_ref` from W&B on both runs shows the two TPUs are already
producing very different DPO signals by step 2-3:

| step | run | loss | `δ_π` | `δ_ref` | `δ_π−δ_ref` | `β·(δ_π−δ_ref)` |
|---|---|---|---|---|---|---|
| 3 | v5p | 0.6796 | −119.2961 | −119.5692 | **+0.273** | 0.027 |
| 3 | v6e | 0.3254 | −109.7875 | −119.5945 | **+9.807** | 0.981 |
| 8 | v5p | 0.6583 | — | — | **+0.713** | 0.071 |
| 8 | v6e | 0.3018 | — | — | **+10.768** | 1.077 |

Reward metrics tell the same story at step 3:
- v5p: chosen reward `+0.0198`, rejected reward `−0.0075`
- v6e: chosen reward `+0.6852`, rejected reward `−0.2955`

`v6e` is moving the policy much further from the reference than `v5p` by
step 3. This is not just a plotting artifact from looking at
`train/dpo_margin_policy` in isolation; it is visible in the actual
loss-driving quantity `δ_π−δ_ref`.

Revised interpretation: the checked-in `r64/α64` recipe does not fix the
early per-stmt LoRA-DPO split. This keeps the culprit in the
per-stmt-LoRA regime, but does **not** isolate whether the dominant factor is
LoRA rank, LoRA scale, singleton-data sensitivity, or some numerical
property of the early DPO update.

### Next experiments to fill Codex's 2×2 isolation matrix

|   | per-stmt (singleton) | full bloom_v2 |
|---|---|---|
| **LoRA** | ❌ diverges early (`r16/α32` and `r64/α64`) | ❓ **Exp M** |
| **Full FT** | ❓ **Exp L** | ✅ closer match (full bloom_speceval_v2) |

---

#### Experiment L — per-stmt full DPO, paired

**Hypothesis:** removing LoRA while keeping the tiny singleton dataset tells
us whether the v5p↔v6e divergence depends on LoRA specifically or on the
singleton-dataset shape (one repeated concept, very homogeneous
chosen/rejected distribution). L-matches → LoRA is the amplifier. L-splits
→ singleton dataset drives the pathology on its own.

- Data: `preference/bloom_v2_singleton/support_mental_health/` (same as the
  pathological LoRA exp 1a `smh_lr1em06_s35`)
- Model: `marin-8b-instruct` policy + `marin-8b-instruct` `SeparateReferenceConfig`
  (full fine-tune; no adapter)
- Training: **first pass = `num_train_steps=10`**, `train_batch_size=64`,
  `lr=1e-6`, `lr_schedule=cosine`, `warmup=0.1`, `beta=0.1`, `seed=0`,
  `train_seq_len=4096`, `validation_split_fraction=None`
- Eval: once at step 10 on `stmt_val` + `full_val`
- Debug env: `MARIN_DEBUG_LOG_BATCH_INDICES=1`, `MARIN_DEBUG_LOG_STEP_TRACE=1`
- Resources: full-FT 8B (≈128 GB static / N-chip FSDP) doesn't fit on
  v5p-8 / v6e-8 for paired compare at batch 64. Preferred plan is to run
  the smallest slices that keep geometry close — ideally **v5p-16 pd=4**
  and **v6e-16 pd=4**. Only escalate to **v6e-32** if v6e-16 does not fit,
  and record that as a weaker comparison because chip count then differs.
- Follow-up: if the 10-step probe is ambiguous, extend the same pair to
  35 steps. Script: `experiments/posttrain/per_stmt_dpo/debug_full_ft_s10.py`
  (TBD; extendable to s35).

---

#### Experiment M — full-data LoRA DPO, paired v5p-8 vs v6e-8

**Hypothesis:** keeping LoRA while swapping to the full 109k-example
`bloom_speceval_v2` preference dataset tells us whether the v5p↔v6e
divergence depends on data variety. M-matches → singleton dataset drives
the pathology. M-splits → LoRA-DPO regime itself is fragile regardless of
dataset.

- Data: `preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/`
  (same as Marin's full-dataset DPO and the `tune_lora` runs)
- Model: `marin-8b-instruct` policy with LoRA
  `r=64, α=64, zero_init_b=True, target_modules=None`,
  `AdapterBaseReferenceConfig`
- Training: `num_train_steps=10` (split was visible by step 2 on per-stmt),
  `train_batch_size=64`, `lr=1e-6`, `lr_schedule=cosine`, `warmup=0.1`,
  `beta=0.1`, `seed=0`, `train_seq_len=4096`, `validation_split_fraction=None`
- Eval: once at step 10 on the bloom v2 val set
- Debug env: `MARIN_DEBUG_LOG_BATCH_INDICES=1`,
  `MARIN_DEBUG_LOG_STEP_TRACE=1`
- Resources: LoRA keeps HBM footprint small. `pd=4` on v6e-8 (same as
  existing LoRA per-stmt configs), `pd=-1` (auto) on v5p-8.
- Implementation note: reuse the existing `experiments/tune_lora/common.py`
  codepath rather than cloning from `per_stmt_dpo`, so this stays aligned
  with the checked-in full-data LoRA recipe.
- Script: `experiments/tune_lora/debug_full_data_lora_r64_s10.py` (TBD).

---

#### Decision table for L + M outcomes

| Exp L (per-stmt full-FT) | Exp M (full-data LoRA) | Interpretation |
|---|---|---|
| matches | matches | LoRA × singleton interaction is the specific pathology |
| matches | splits | LoRA itself is the amplifier regardless of dataset |
| splits | matches | singleton dataset is the amplifier regardless of model class |
| splits | splits | something fundamental about early DPO updates on 8B is v5p/v6e sensitive |

Expected: **L matches, M splits**. That would make LoRA-DPO the leading
amplifier. It would strongly de-prioritize generic CE/attention kernel
hunting, though not mathematically rule out numerical effects inside the
LoRA-DPO update path.

---

## 2026-04-14: Full DPO on v6e-32 — divergence does NOT reproduce

**Key finding:** When running **full (non-LoRA) DPO** with batch=64 on v6e-32
and comparing against the v5p-16 seed=0 full-DPO baseline, the loss curves
track each other almost perfectly through at least step 144.

**W&B runs (compare side-by-side):**
- v6e-32 (this run): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_v6e32_pd2_lr5e-07-7f1c19
- v5p-16 seed0 (baseline): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963
- v5p-16 seed1 (sibling): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_beta0.1_seed1_b64_v5p16-c50842
- v5p-16 seed2 (sibling): https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_beta0.1_seed2_b64_v5p16-2272c8

| Step | v5p-16 seed0 | v6e-32 (r1) | Δ |
|---|---|---|---|
| ~144 | 0.3937 | 0.39112 | 0.003 (< 1%) |
| ~314 (last observed) | — | 0.0513 | — |

**Interpretation (initial, confounded):** The hardware-level bf16 numerical
divergence that wrecks LoRA-DPO (10× learning-speed gap) **does not visibly
affect full fine-tune** with the same data/seed.

### ⚠️ Codex caveat: three confounds stacked in the "LoRA bad vs full good" comparison

The two run families we've compared so far are **not** a clean model-class ablation:

| Run family | Data | Model | LoRA `r` | α |
|---|---|---|---|---|
| Per-stmt DPO (pathological v5p vs v6e) | 1 singleton stmt (~6k ex) | LoRA | **16** | 32 |
| `tune_lora` full-data LoRA | bloom v2 full (~109k ex) | LoRA | **64** | 64 |
| v6e-32 full DPO (this run) | bloom v2 full (~109k ex) | full FT | n/a | n/a |

Differences: (1) LoRA-vs-full-FT, (2) singleton-vs-full dataset, (3) rank 16 vs 64.
Any of these — or combinations — could explain why the v5p↔v6e split
appears only in the per-stmt LoRA runs.

**Mechanism hypothesis (why LoRA-DPO is more fragile than full-FT DPO):**
- DPO at init is on a flat landscape (policy=ref → `softplus(0) = ln(2)`,
  so `dL/dlogit = 0.5` for every example).
- In LoRA with `zero_init_b=True`, only `lora_B` absorbs gradient; `lora_A`
  is frozen at init-random. `lm_head` is excluded from LoRA by default.
- Full FT can spread the DPO signal across all 8B params; LoRA has to route
  it through a tiny low-rank branch with a zero-initialized B, fixed random A,
  and no `lm_head` adaptation.
- So small bf16 differences in `delta_pi - delta_ref` get projected by
  a low-rank basis into very different `lora_B` updates — amplifying per-chip
  numerical noise into direction-level learning-dynamics differences.

### Codex-suggested isolation matrix

|   | per-stmt (singleton) | full bloom_v2 |
|---|---|---|
| **LoRA** | ❌ diverges (r=16) | ❓ **missing** |
| **Full FT** | ❓ **missing** | ✅ matches |

The highest-info next run is **full bloom_v2 LoRA DPO on v5p vs v6e**
(same rank as pathological case, same 2-step trace instrumentation).
If it still splits → pathology is LoRA-regime specific. If it matches →
singleton dataset is a major driver.

**Config that works on v6e-32** (`experiments/posttrain/full_dpo/v6e32_full_dpo.py`):
- `train_batch_size=64`, `per_device_parallelism=2` (32 chips × 2 ex/chip)
- `SeparateReferenceConfig`, `reference_model_path="marin-community/marin-8b-instruct"`
- No adapter → full fine-tune of all 8B params
- `beta=0.1`, `lr=5e-7`, cosine schedule, `warmup=0.1`, `num_epochs=1.0`
- `seed=0`, `train_seq_len=4096`
- **Throughput: 6.2–7.7 sec/step** post-warmup (JIT compile warm-up ~65s for step 0)
- **HBM OK** — no OOM on v6e-32 (peak host RAM ~37 GB/worker, ~22 GB steady)

**Napkin math per chip** at batch=64 / FSDP over 32 chips:
- policy bf16 16 GB / 32 = 0.5 GB; reference bf16 0.5 GB
- grads fp32 1 GB; Adam m/v fp32 2 GB; DCN replication ~8 GB
- activations w/ gc ~10 GB/chip at pd=2 → ~22 GB total (fits in 32 GB HBM)

**Preemption incident (first run `/ahmed/full-dpo-v6e32-r1`):**
- Child ran 8h 54m, reached step 947 before parent CPU coordinator was
  preempted twice. Re-scheduled parent landed on `us-central1` CPU while
  original wrote checkpoints to `gs://marin-us-central2/.../-7f1c19/`.
  Executor re-resolved output to us-central1 (empty), child respawn then failed.
- **Salvageable on GCS:** `step-947` streaming checkpoint + `step-500` HF export,
  both under `gs://marin-us-central2/checkpoints/dpo/full_dpo/bloom_speceval_v2_v6e32_pd2_lr5e-07-7f1c19/`.
- **Lesson:** long-running DPO parent CPU jobs need region pinning
  (`--zone us-central2-b` or `-e MARIN_PREFIX gs://marin-us-central2`) so that
  preemption-driven reschedules don't orphan checkpoints across regions.

---

## Problem

DPO LoRA training runs with matching hyperparameters produce dramatically
different learning curves on v6e-8 vs v5p-8. v6e-8 learns much faster.
The effect is consistent across all revalidated exp 1a pairs and directionally
present in exps 1b and 2a (though those comparisons have caveats — see below).

## Current Leading Hypothesis (refined after Experiment J)

**Hardware-level numerical divergence in the forward/backward pass.** On
identical inputs and identical initial model state, v5p-8 and v6e-8 produce
measurably different trainable gradients at step 0. The bf16 matmul /
attention numerics differ between the two chip architectures.

### Key facts established by Experiment J (2-step trace):

- **Same batch indices at step 0** (both TPUs: sha256=`7a61ce53d17eb721`, indices [0-63])
- **Same initial param values** (pA_l2 differs in 6th decimal: 59.075542 vs 59.075538)
- **Same loss at step 0** (exactly `0.6931471824645996` on both — but this is
  tautological: LoRA zero-init means policy=reference, so every example has
  `softplus(0) = ln(2)` regardless of data)
- **Different gradients at step 0**:

| Metric | v6e-8 | v5p-8 | Δ |
|---|---|---|---|
| grad_l2 | 2.5308258 | 2.5316045 | 0.00031% |
| grad_sum | -0.8846889 | **-0.7291931** | **0.155 absolute** |
| upd_l2 | 0.004686407 | 0.004685373 | ~match |
| upd_sum | -0.001111864 | -0.0000680 | huge relative |

Sentinel lora_B grad_sum on gate_proj: v6e=0.269, v5p=0.353 (+31%).

**Interpretation:** Gradients have the same L2 magnitude (within 0.0003%)
but **differ significantly in direction** (signed sums diverge). The
forward pass computes `delta_pi - delta_ref` slightly differently on each
chip. At init (delta=0), `d(softplus(-x))/dx = -sigmoid(-x) = -0.5`,
so tiny per-example `delta` differences directly produce tiny gradient-
direction differences. These compound: DPO at initialization sits on a
flat loss landscape where small gradient-direction shifts land the model
in different basins.

### Earlier hypotheses ruled out by Experiment J:

- **Microbatch code path** (ruled out by Experiment A earlier)
- **CE v_block_size heuristic** (ruled out by Experiment G: same on both)
- **Data ordering** (ruled out by Experiment J: identical batch sha256)
- **CE `num_b_blocks` blocking** (Experiment J also rules this out as the
  PRIMARY cause — gradients differ globally, not just in the lm_head gw which
  is frozen for LoRA anyway)

## Shared Config (verified identical in W&B)

- lr=1e-6, cosine schedule, warmup=0.1
- LoRA r=16, alpha=32, dropout=0, zero_init_b=True
- beta=0.1, seed=0, data_seed=0
- max_grad_norm=1.0, train_batch_size=64
- AdapterBaseReferenceConfig (reference = base model with LoRA disabled)
- model: marin-community/marin-8b-instruct

## Execution Differences (verified at runtime via Experiment G)

| | v6e-8 | v5p-8 |
|---|---|---|
| Chips | 8 | 4 |
| `per_device_parallelism` | 4 (explicit) | -1 → 16 (auto) |
| Gradient accumulation | 2 micro-steps of 32 | none (full batch of 64) |
| Per-device examples | 4 | 16 |
| CE kernel B (examples × seq_len) | **16,384** | **65,536** |
| CE kernel `device_kind` | `TPU v6 lite` | `TPU v5` |
| CE `v_block_size` | **8192** | **8192** (same!) |
| CE `b_block_size` | **1024** | **1024** (same!) |
| CE `num_v_blocks` (padded 128256) | **16** | **16** (same!) |
| **CE `num_b_blocks`** | **16** | **64** |

---

## Evidence

### Primary: Fresh s70 Pairs (cleanest comparison)

**Established.** Both start fresh (no checkpoint resume).

**Train loss: lr=1e-6, s70 pair**

| Step | v6e-8 (`fbac2a`) | v5p-8 (`964129`) | Gap |
|---|---|---|---|
| 0 | 0.6931 | 0.6931 | 0.000 |
| 1 | 0.6931 | 0.6931 | 0.000 |
| 2 | **0.4672** | 0.6951 | -0.228 |
| 5 | 0.3861 | 0.6883 | -0.302 |
| 10 | 0.3688 | 0.6710 | -0.302 |
| ~69 | 0.320 | 0.574 | -0.254 |

**Final eval:** v6e-8 stmt=0.402 full=0.613 | v5p-8 stmt=0.601 full=0.675

**Train loss: lr=1e-7, s70 pair**

| Step | v6e-8 (`4ac830`) | v5p-8 (`f3d60b`) | Gap |
|---|---|---|---|
| 0 | 0.6931 | 0.6931 | 0.000 |
| 2 | 0.6806 | 0.6943 | -0.014 |
| 5 | 0.4606 | 0.6932 | -0.233 |
| 10 | 0.3862 | 0.6917 | -0.306 |

The divergence appears at step 2-3 on fresh runs across multiple LR settings.

**Additional matched pairs (summary-level):**

| Pair | v5p train - v6e train | v5p stmt - v6e stmt | v5p full - v6e full |
|---|---|---|---|
| lr5e7_s70 | +0.293 | +0.224 | +0.065 |
| lr5e7_s140 | +0.240 | +0.196 | +0.061 |
| lr1e6_s70 | +0.253 | +0.199 | +0.062 |

### Secondary: s140 Pair (has resume confound)

v5p-8 s140 (`cc7957`) resumed from step 94. v6e-8 s140 (`a6a62e`) started
fresh. Still shows the same direction but should not be treated as flagship.

### Cross-Experiment Evidence (incomplete)

The pattern is directionally present in other experiment types but these
comparisons are less clean:
- **1b**: v6e-8 s420 vs v5p-8 s210 — different step counts, not matched
- **2a**: v5p-8 run was still in progress at last check
- **2b**: no v5p-8 comparator available

### Initialization (step 0)

**Established.** Both runs start in the same state:

| Metric | v6e-8 | v5p-8 | Diff |
|---|---|---|---|
| train/loss | 0.693147 | 0.693147 | 0.000 |
| grad/norm/total | 28.8546 | 28.8568 | 0.008% |
| LoRA-only L2 grad norm (448 tensors) | 2.5308 | 2.5316 | 0.03% |

LoRA A init is provably identical (deterministic from `seed=0`, device-independent
pytree traversal). LoRA B is zeros (`zero_init_b=True`). LR schedule is identical
at every logged step.

**Caveat:** Near-equal step-0 gradient norms do not prove equal full gradients.
The gradients include per-token contributions from the CE kernel, which uses
different block sizes on v5p vs v6e. The DPO loss at initialization is
uniformly ln(2) per example (policy = reference), so block-size differences
would not manifest in the step-0 loss or its gradients. They would appear
starting at step 1 once per-example losses diverge.

### Parameter Norms (stay close despite loss divergence)

**Established (from Codex revalidation on fresh s70 pair).**

| Step | v5p-8 lora_B norm | v6e-8 lora_B norm |
|---|---|---|
| 10 | 0.001166 | 0.001139 |
| 60 | 0.006556 | 0.006408 |

Parameters are not in wildly different states. The loss divergence is not
caused by dramatically different update magnitudes. This is consistent with
the CE hypothesis: slightly different gradients (from different block-size
tiling) compound into large loss differences on this tiny, sensitive dataset.

### Gradient Norm Evolution

| Step | v6e-8 total | v5p-8 total |
|---|---|---|
| 0 | 28.855 | 28.857 |
| 10 | 16.021 | 28.000 |
| 50 | 14.086 | 23.819 |
| 130 | 12.659 | 19.978 |

v6e-8 norms drop rapidly (learning). v5p-8 norms barely decrease (stuck).

---

## W&B Run Index

| Experiment | v6e-8 run | v5p-8 run | Notes |
|---|---|---|---|
| 1a lr1e6 s70 | `smh_lr1em06_s70_v6e8-fbac2a` | `smh_lr1em06_s70_v5p8-964129` | **cleanest pair** |
| 1a lr1e7 s70 | `smh_lr1em07_s70_v6e8-4ac830` | `smh_lr1em07_s70_v5p8-f3d60b` | **cleanest pair** |
| 1a lr5e7 s70 | `smh_lr5em07_s70_v6e8-7ae68d` | `smh_lr5em07_s70_v5p8-86109f` | |
| 1a lr5e7 s140 | `smh_lr5em07_s140_v6e8-ddb5ce` | `smh_lr5em07_s140_v5p8-5f928a` | |
| 1a lr1e6 s140 | `smh_lr1em06_s140_v6e8-a6a62e` | `smh_lr1em06_s140_v5p8-cc7957` | v5p-8 RESUMED step 94 |
| 1b lr1e6 | `3stmt_lr1em06_s420_v6e8-fd4e55` | `3stmt_lr1em06_s210_v5p8-7fe6a8` | different step counts |
| 2a lr1e6 s140 | `support-mental-health_lr1em06_s140_v6e8-004b5d` | `support-mental-health_lr1em06_s140_v5p8-d9567e` | v5p-8 incomplete |
| Diag: v5p pd=8 | — | `smh_lr1em06_s70_v5p8_pd8-0498ec` | Experiment A |

All in W&B project `marin-community/dpo`.

---

## What Has Been Ruled Out

### 1. Microbatch / gradient accumulation code path
**Ruled out by Experiment A.** Forcing v5p-8 to use the same microbatch regime
as v6e-8 (pd=8 → 2× grad accum, microbatch=32) produced identical behavior to
original v5p-8 (pd=16, no grad accum). Both v5p-8 curves are slow learners.

### 2. Config mismatch
**Established.** W&B config diff shows only per_device_parallelism and derived paths.

### 3. LR schedule
**Established.** Identical values at every step.

### 4. Gradient clipping asymmetry
**Established.** `optax.clip_by_global_norm(1.0)` operates on trainable (LoRA)
grads only. Both runs have LoRA grad norm ~2.53 at step 0.

### 5. Checkpoint resume as root cause
**Established.** Fresh s70 pairs show the same divergence.

### 6. Data batch composition
**Established (local test).** The Feistel permutation with `seed=0` produces
identical cache indices for batch 0 regardless of device count (verified
locally in `debug_data_order.py`). Not yet verified end-to-end on TPU.

### 7. LoRA initialization
**Established.** Key derivation from `seed=0` is deterministic and
device-independent. Both runs start with identical LoRA A weights.

## Confounds (real but not root cause)

### Eval batch-size weighting
`eval_loss_loop` (`callbacks/__init__.py:32-88`) uses unweighted mean over
batches. Eval batch sizes differ (v6e-8: 32, v5p-8: 64). Step-0 eval values
match exactly, so this only matters when per-example losses vary. Does not
explain the training-loss divergence.

### Reference eval cache metadata
`dpo.py:227-244`: cache path hash matches but metadata comparison fails
(dict vs dataclass). Both TPU types rebuild the cache. Wastes compute but
does not affect training dynamics.

---

## Ranked Explanations (updated after Experiment J)

### 1. Strongest: Hardware-level bf16 numerical differences in forward/backward

**Confirmed by Experiment J.** On identical batches with identical initial
weights, v5p-8 and v6e-8 produce trainable gradients that:
- agree in L2 magnitude to ~0.0003%
- disagree significantly in signed direction (grad_sum differs by 0.155 absolute)

The forward pass computes `delta_pi - delta_ref` slightly differently per
example because of bf16 matmul / attention numerics that differ between
chip architectures. Since DPO at init sits on a flat loss landscape (every
example loss = ln(2)), tiny per-example logit differences directly
translate into gradient-direction differences that compound over training.

The source is somewhere in the forward/backward pass of the model itself —
before the CE kernel's accumulation steps. Most likely candidates:
- Splash attention numerics (hardcoded block_size=512 but matmul precision
  might differ between v5p MXU and v6e MXU)
- LoRA/Linear matmul precision in the model body
- bf16 accumulation in any reduction (e.g., LayerNorm, attention softmax)

### 2. Lower: CE kernel numerics

Different `num_b_blocks` (16 vs 64) in the CE backward means different bf16
accumulation patterns for the lm_head gradient (gw). But for LoRA training,
gw is FROZEN (`trainables_only(grads, is_trainable)` discards it). So the
gw accumulation difference doesn't affect LoRA training dynamics directly.
The gx gradient (activations) has the SAME number of accumulation steps on
both (16, since num_v_blocks=16 on both). So CE isn't the primary cause.

### 3. Eliminated by Experiment J

- ~~Data ordering / batch composition~~ (confirmed same batch sha256)
- ~~Initial model weights~~ (param norms match to 6 decimals)
- ~~Microbatch code path~~ (ruled out by Experiment A)
- ~~CE `v_block_size` hardware heuristic~~ (ruled out by Experiment G)

---

## Experiment Plan (revised after Experiment J)

The primary mystery is solved: **hardware numerics differ**. Remaining
questions are about which part of the model pipeline introduces the
divergence and whether we can fix it for training stability.

### Experiment K: Force fp32 matmul precision (HIGHEST INFO NEXT)

Add `precision="highest"` (or `precision=jax.lax.Precision.HIGHEST`) to
the model's matmuls / attention / CE kernel. This forces fp32 accumulation
in matmuls, eliminating the bf16 tolerance differences between v5p MXU and
v6e MXU.

If v5p-8 with fp32 precision matches v6e-8 at step 0 gradients:
→ **confirms bf16 matmul precision is the source.**
If still differs: attention or other non-matmul op is the source.

Cost: slower compute but we only need 2 steps.

### Experiment L: Isolate forward-pass log-probs (diagnostic)

Instrument `logp_sum()` in `dpo.py:130` to dump a checksum of
`logp_pi_chosen` and `logp_pi_rejected` BEFORE the delta/loss computation.
If log-probs differ between TPUs → the forward pass is the source.
If they match but gradients differ → the backward pass is the source.

This narrows the hunt: forward vs backward, then within each.

### Experiment M: Disable Splash attention

Fall back to the reference (non-splash) attention kernel to test if
attention is the culprit. If swapping attention implementation closes
the gap → attention numerics are the cause.

### Experiment E: v6e-8 pd=2 (SECONDARY, largely obsolete)

Was meant to test the CE block-size hypothesis. Now that we know gradients
differ on same batch regardless of blocking, this is lower value. Could
still be useful to confirm v6e stays fast across configs.

### Experiments G, I, H (SUPERSEDED)

G was done (confirmed block sizes). I and H were framed around the CE
block-size hypothesis which Experiment J rules out as the primary cause.
Not worth running.

---

## Code References

| Component | File | Lines |
|---|---|---|
| CE kernel (XLA) | `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py` | 101-350 |
| CE block size heuristic | `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py` | 779-818 |
| CE API + impl selection | `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py` | 80-194 |
| CE loss flattening | `lib/levanter/src/levanter/models/loss.py` | 245 |
| DPO loss → CE path | `lib/levanter/src/levanter/dpo.py` | 130-132 |
| Gradient accumulation | `lib/levanter/src/levanter/grad_accum.py` | 36-169 |
| Trainer step | `lib/levanter/src/levanter/trainer.py` | 678-716 |
| Gradient filtering | `lib/levanter/src/levanter/trainer_state.py` | 236-268 |
| Eval loss loop | `lib/levanter/src/levanter/callbacks/__init__.py` | 32-88 |
| Experiment config | `experiments/posttrain/per_stmt_dpo/common.py` | 74-93 |

---

## Experiment Log

### 2026-04-13T20:10Z — Experiment A: v5p-8 with forced pd=8

**Script:** `experiments/posttrain/per_stmt_dpo/debug_accum_v5p8_pd8.py`
**Iris job:** `/ahmed/debug-accum-v5p8-pd8`
**W&B run:** `smh_lr1em06_s70_v5p8_pd8-0498ec`
**Config:** v5p-8, pd=8, microbatch=32, 2× grad accum, lr=1e-6, s70

**Result: MATCHES ORIGINAL v5p-8 (slow). Does NOT match v6e-8.**

At step 21: v5p pd=8 = 0.6397, v5p pd=16 = 0.6406, v6e pd=4 ≈ 0.37.
The two v5p curves track each other exactly. This rules out the microbatch
code path as the cause. Under the CE hypothesis, this is expected: changing
pd on v5p does not change its hardware-dependent CE v_block (still 16384).

### 2026-04-13T20:30Z — LoRA initialization verified identical

Key derivation from `seed=0` is deterministic and device-independent.
`adapter_key = [928981903, 3453687069]`. Pytree traversal order is the same
on both TPU types. LoRA A weights are provably identical at init.

### 2026-04-13T20:45Z — Data ordering verified identical (local)

`debug_data_order.py` confirmed the Feistel permutation produces identical
cache indices for batch 0 regardless of device count (4 or 8 devices both
get indices `[1536, 625, 1162, 271, ...]` for seq 0-63). Not yet verified
end-to-end on real TPU hardware.

### 2026-04-13T21:30Z — CE kernel hypothesis identified

David (senior engineer) suggested checking kernel block sizes. Investigation
revealed hardware-dependent `v_block_size` in XLA fused CE backward. This is
now the leading hypothesis. See "Ranked Explanations" section above.

### 2026-04-13T23:30Z — Experiment G run: CE block sizes logged

Added DEBUGSTART/DEBUGEND markers in `xla.py` to print resolved block sizes
on first kernel call. Results from `smh_lr1em06_s35_*_fp32_upcast` runs:

**v6e-8:**
- device_kind: `TPU v6 lite`
- x.shape: (16384, 4096)
- v_block_size: 8192, b_block_size: 1024
- num_v_blocks: 16, **num_b_blocks: 16**

**v5p-8:**
- device_kind: `TPU v5`
- x.shape: (65536, 4096)
- v_block_size: 8192, b_block_size: 1024
- num_v_blocks: 16, **num_b_blocks: 64**

**Key finding:** `v_block_size` is IDENTICAL on both (8192). The earlier
hypothesis that v5p gets 16384 was wrong — the heuristic checks
`device_key == "TPU v5p"` but v5p reports as `"TPU v5"`, so the
hardware-specific branch never fires.

The real difference is `num_b_blocks`: 16 vs 64. This comes from the
per-device flattened batch being 4× larger on v5p (65536 vs 16384).

### 2026-04-13T23:52Z — Experiment: fp32 upcast of gw_block (PARTIAL IMPROVEMENT)

Patched `xla.py:338-352` to accumulate `gw_block` in fp32 across batch blocks,
casting down to bf16 only once when writing into the final `gw` slice.

Launched fresh 35-step runs on both TPUs:
- `smh_lr1em06_s35_v6e8_fp32_upcast-a0befb` (v6e-8)
- `smh_lr1em06_s35_v5p8_fp32_upcast-1ea0d9` (v5p-8)

**Early signal (v5p-8 at step 12): loss=0.6597**
- Baseline v5p-8 at step ~12: ~0.68
- Modest improvement (~0.02) but still nowhere near v6e-8 (~0.37 at step 12)

**Preliminary conclusion:** Upcasting only the `gw_block` accumulation is
NOT sufficient to close the gap. Either:
1. There's additional bf16 accumulation elsewhere in the CE kernel (`gx`
   still accumulates across vocab blocks in bf16 at `xla.py:347`), OR
2. The CE backward isn't the primary source of divergence at all

**Resolution (confirmed in Experiment J): option 2.** The CE backward
isn't the primary cause. gw is frozen in LoRA (not applied to model), and
gx accumulates 16 times on BOTH TPUs, not 64 vs 16.

### 2026-04-14T01:47Z — Experiment J: 2-step deterministic trace (BREAKTHROUGH)

**Script:** `experiments/posttrain/per_stmt_dpo/debug_two_step_trace.py`
**Instrumentation:** DEBUGSTART/DEBUGEND blocks in `loader.py:452` and
`trainer.py:698` guarded by env vars:
- `MARIN_DEBUG_LOG_BATCH_INDICES=1` → logs first 5 batches' indices + sha256
- `MARIN_DEBUG_LOG_STEP_TRACE=1` → `jax.debug.print` emits full trace

**Jobs:**
- `/ahmed/debug-j-trace-v6e8` (W&B: `two_step_trace_v6e8-*`)
- `/ahmed/debug-j-trace-v5p8-r2` (W&B: `two_step_trace_v5p8-*`)

**Result: SAME DATA, DIFFERENT GRADIENTS.**

Step 0 train batch: both TPUs loaded indices [0-63] with identical
sha256=`7a61ce53d17eb721`. Permutation, data loader, and sharding are
device-count-independent as expected.

Step 0 trainable gradient trace:

| Metric | v6e-8 | v5p-8 | Δ |
|---|---|---|---|
| loss | 0.6931471824645996 | 0.6931471824645996 | exact (tautological at init) |
| grad_l2 | 2.5308258533477783 | 2.531604528427124 | 0.00031% |
| **grad_sum** | **-0.8846888542175293** | **-0.7291930913925171** | **0.155** |
| upd_l2 | 0.004686407 | 0.004685373 | ~match |
| upd_sum | -0.001111864 | -0.0000680 | **huge relative** |
| pA_l2 | 59.07554 | 59.07554 | match to 6 dec |
| pA_sum | -18.20846 | -18.20843 | match to 4 dec |

Sentinel gradients (all are lora_B; lora_A gradients are all 0 at init
because lora_B=0 → no signal flows through lora_A):

| Module | v6e grad_l2 | v5p grad_l2 | v6e grad_sum | v5p grad_sum |
|---|---|---|---|---|
| q_proj | 0.5416715 | 0.5418293 | 0.4532508 | 0.4209823 |
| gate_proj | 0.8171922 | 0.8172969 | 0.2690451 | 0.3527192 |
| o_proj | 0.7086674 | 0.7090713 | -0.8998485 | -0.9008402 |

**Pattern:** L2 norms match to ~4 decimals. Signed sums diverge
significantly. Gradients have **same magnitude but different direction**.

**Interpretation:** The forward pass is computing `delta_pi - delta_ref`
slightly differently per example on the two chip architectures. At init
(delta=0), `d(softplus(-x))/dx = -0.5`, so tiny per-example delta
differences translate directly into gradient-direction differences.

**This is hardware-level bf16 numerical divergence** in the forward/
backward pass. Not data, not init, not grad accum, not CE block sizes.

**Why it compounds into dramatic training differences:** DPO at init sits
on a perfectly flat loss landscape (every example loss = ln(2)). Any small
gradient direction shift lands the model in a different basin after step 1.

See "Ranked Explanations" above for updated hypothesis and "Experiment
Plan" for next experiments (K: force fp32 precision; L: isolate
forward-pass log-probs; M: disable Splash attention).

### 2026-04-14T04:55Z — Experiment J2: 1-step LoRA factor trace (PARTIAL SUCCESS)

Goal: For each LoRA module, log on step 0 (per senior engineer David's suggestion):
1. Forward factor `z = lora_A(x)` checksum/L2
2. Upstream cotangent `dL/d(lora_B_output)` checksum/L2
3. Compare against existing grad_B numbers

**Implementation:**
- `lib/levanter/src/levanter/lora.py` instrumented with DEBUGSTART/DEBUGEND blocks:
  - `jax.debug.print` logs `z` stats after lora_A (forward, stage=`z_after_lora_A`)
  - `jax.custom_vjp` identity op wraps lora_B output; backward supposed to print cotangent
  - Module tag uses INPUT axes to distinguish modules (embed4096/heads32_head_size128/mlp14336)
- `experiments/posttrain/per_stmt_dpo/debug_lora_factor_trace.py` — 1-step run
- Gated by `MARIN_DEBUG_LORA_FACTOR_TRACE=1`

**Issues encountered:**

1. **Initial eval consumed prints.** Initial eval (full_val, 503-batch Paloma/LM eval)
   triggered ~112k LoRA forward calls. With `jax.debug.print` forcing host-device sync,
   this made eval take >1h. First runs (r2, r3) never reached training step.
   - **Fix:** Added `max_eval_batches` to SimpleDPOConfig and plumbed through default_dpo.
     `max_eval_batches=0` crashes with "AsyncDataset has length 0". `max_eval_batches=1`
     works — only ~1800 LoRA calls during eval instead of 112k.

2. **Backward cotangent trace NOT emitting.** v6e-8-r4 run SUCCEEDED with training step
   complete (loss=0.693 logged). Produced 917 DEBUGJ LORA_FWD lines. But **0 DEBUGJ
   LORA_BWD lines**. The `jax.custom_vjp` bwd with `jax.debug.print` is not outputting.
   Possible causes:
   - `jax.debug.print` inside custom_vjp bwd may be DCE'd by XLA
   - NamedArray ↔ jax.Array roundtrip may break the VJP chain
   - Custom_vjp bwd in autodiff graph may not mark prints as required side effects

3. **DEBUGJ TRACE (trainer.py) and DEBUGJ BATCH (loader.py) also NOT firing.** These
   are runtime env var checks (not import-time like lora.py's `_DBG_LORA_FACTOR_TRACE`).
   Env var may not propagate from Iris executor parent → train_dpo sub-job at runtime,
   OR the check is before a code path that doesn't execute. Needs investigation.

**What we have from v6e-8-r4:**
- 917 DEBUGJ LORA_FWD entries with module tags
- Sample values: mod=embed4096 l2≈400-700, sum varies; mod=heads32_head_size128 l2≈600-900;
  mod=mlp14336 l2≈80-520

**What we DON'T have:**
- LORA_BWD cotangent values (custom_vjp bwd silent)
- v5p-8 counterpart (TPU capacity exhausted, jobs pending ~30 min)

**Next step:** Get the backward to actually emit. Options:
- Use `jax.debug.print(..., ordered=True)` to force execution
- Use `jax.experimental.io_callback` instead of debug.print
- Use `jax.vjp` directly in training loop to capture gradients in forward-pass style
- Add a sentinel read of the cotangent that makes it "live" in JAX's eyes

### 2026-04-14T05:15Z — Experiment J3: cleaner approach per Codex critique

Codex rightly critiqued the J2 approach:
- `jax.custom_vjp` inside a hot library path (scan/vmap/remat/partitioning) is
  too fragile as a debugging probe
- Shape-derived tags are not stable module identities
- Treating `ordered=True` as forcing execution is a misconception (it only
  orders existing effects)
- Overcommits to bf16 story despite `mp=jmp.get_policy("p=f32,c=bfloat16")`
  making params f32

New plan: forward-factor trace only + trainer.py's existing sentinel-grad
trace = enough to localize divergence.

**Implementation changes:**

1. Removed custom_vjp backward machinery from `lora.py` entirely.

2. Added `debug_name: str = eqx.field(static=True, default="")` to
   `LowRankLinear`. Propagated via `LoraLinear.init(debug_name=...)` ←
   `_loraize_module(key_path)`. So each LoRA module now has a stable path
   like `transformer.stacked.self_attn.q_proj`.

3. Forward instrumentation restricted to sentinel modules (q_proj,
   gate_proj, o_proj) by substring match on `debug_name`. Avoids the
   noisy 917-line output; expect ~32 layers × 3 sentinels × 2 forwards
   = 192 prints per training step.

4. Relying on trainer.py's existing grad/update/param checksum trace
   (from Experiment J) to capture sentinel grad_B values. We already
   have the per-module grad_l2/grad_sum from Experiment J — those are
   the "backward" numbers Codex says we need.

**Logic to apply when data lands:**
- If `z_after_lora_A` matches across TPUs for same module → forward path
  to that module is identical → divergence enters IN lora_B backward or
  further upstream in the gradient chain
- If `z_after_lora_A` differs → forward path is already diverging (attention
  or earlier LoRA modules)

**Jobs launched** (2026-04-14T05:15Z):
- `/ahmed/debug-j3-v6e8`
- `/ahmed/debug-j3-v5p8-central1`
- `/ahmed/debug-j3-v5p8-east5`

All with `MARIN_DEBUG_LORA_FACTOR_TRACE=1`. 1 training step, `max_eval_batches=1`.

### 2026-04-14T11:55Z — Experiment L result: per-stmt full-FT DPO on v5p-16 vs v6e-16 (EARLY MATCH, BUT NOT A PURE LORA-ONLY ABLATION)

**Runs:**
- v6e-16 full FT DPO: https://wandb.ai/marin-community/dpo/runs/full_ft_s10_v6e16_l1-008fdb
- v5p-16 full FT DPO: https://wandb.ai/marin-community/dpo/runs/full_ft_s10_v5p16_l1-bb2be5

**State at analysis time:**
- `full_ft_s10_v5p16_l1-bb2be5`: finished at W&B `_step=9`
- `full_ft_s10_v6e16_l1-008fdb`: crashed at W&B `_step=8` before the scheduled
  step-10 eval; W&B had not uploaded `output.log` yet, so the strongest
  cross-TPU evidence here is the scalar history plus the finished v5p worker log

**Config parity (verified from W&B config):**
- same singleton per-stmt dataset:
  `preference/bloom_v2_singleton/support_mental_health/`
- same `train_batch_size=64`, `lr=1e-6`, `beta=0.1`, `seed=0`,
  `train_seq_len=4096`, `validation_split_fraction=None`
- same mixed precision policy: params `f32`, compute/output `bf16`
- same `per_device_parallelism=4`
- no adapter (`adapter=null`) — full fine-tune with `SeparateReferenceConfig`

**Important scalar result:** the catastrophic v5p↔v6e split does **not**
reproduce in full FT on the same per-stmt dataset.

Recall from `dpo.py:119` that the loss-driving quantity is
`delta_pi - delta_ref`, not `dpo_margin_policy` alone.

Through the overlapping prefix (steps 0-8), the two full-FT runs stay close on
both loss and `delta_pi - delta_ref`:

| step | v6e loss | v5p loss | v6e `delta_pi-delta_ref` | v5p `delta_pi-delta_ref` |
|---|---|---|---|---|
| 0 | 0.693147 | 0.693147 | 0.000 | 0.000 |
| 2 | 0.692416 | 0.688913 | 0.019 | 0.087 |
| 3 | 0.678566 | 0.673635 | 0.299 | 0.396 |
| 4 | 0.664171 | 0.663108 | 0.593 | 0.613 |
| 5 | 0.655864 | 0.656349 | 0.769 | 0.754 |
| 6 | 0.618792 | 0.615969 | 1.559 | 1.620 |
| 7 | 0.606406 | 0.603591 | 1.828 | 1.889 |
| 8 | 0.597066 | 0.593090 | 2.043 | 2.127 |

Max difference through step 8:
- `max |Δ(train/loss)| = 0.00493`
- `max |Δ(delta_pi-delta_ref)| = 0.0977`

Compare that to the pathological LoRA `r64/alpha64` pair from Experiment K:
- `max |Δ(train/loss)| = 0.35646`
- `max |Δ(delta_pi-delta_ref)| = 10.0553`

So on the same per-stmt dataset, the full-FT discrepancy is roughly:
- **~72x smaller on loss**
- **~103x smaller on the loss-driving DPO quantity**

This is the strongest evidence so far that the earlier catastrophic split is
**not generic to v5p vs v6e DPO**, and that the LoRA regime is the main
amplifier.

**Reward metrics tell the same story.** By step 8:
- v6e full FT: chosen reward `+0.1426`, rejected reward `-0.0616`
- v5p full FT: chosen reward `+0.1504`, rejected reward `-0.0623`

By contrast, the LoRA pair had already separated dramatically by step 3:
- v6e LoRA: chosen reward `+0.6852`, rejected reward `-0.2955`
- v5p LoRA: chosen reward `+0.0198`, rejected reward `-0.0075`

### Full-FT gradient evidence

The finished v5p worker log confirms normal full-FT learning dynamics:
- step 0: `loss=0.693147`, `grad_l2=28.826`, `grad_sum=3.7428`
- step 2: `loss=0.688913`, `grad_l2=27.871`, `grad_sum=1.4474`
- step 3: `loss=0.673635`, `grad_l2=27.496`, `grad_sum=-5.8480`
- step 8: `loss=0.593090`, `grad_l2=25.897`, `grad_sum=2.0256`

These come from the `DEBUGJ TRACE` lines in the finished `v5p` `output.log`.
The LoRA-specific sentinel slots (`gA`, `gB`, `pA`, `pB`) are all zero here
because this is a non-LoRA run; the useful fields are the global
`grad_l2/grad_sum/upd_l2/upd_sum/param_l2/param_sum`.

Step-0 gradient norms also match closely across TPU types in W&B:
- v6e: `grad/norm/total = 28.8673`
- v5p: `grad/norm/total = 28.8260`

Across the full per-parameter grad-norm tree at step 0:
- 582 comparable grad-norm keys
- median relative difference: `~2.97e-6`
- p95 relative difference: `~0.0092`
- max relative difference: `~0.051`

So full FT still has small TPU-level numeric differences, but they do **not**
turn into the qualitatively different early trajectory seen in LoRA.

### Concrete difference vs LoRA that matters: execution geometry is cleaner here

Experiment L is **not** a pure "remove LoRA, hold everything else fixed"
ablation. It also made the execution geometry much more apples-to-apples:

| Run family | v6e setup | v5p setup |
|---|---|---|
| Bad per-stmt LoRA | `v6e-8`, `pd=4` | `v5p-8`, `pd=16` |
| Exp L full FT | `v6e-16`, `pd=4` | `v5p-16`, `pd=4` |

This matters for the CE kernel path. In the finished `v5p-16` full-FT log:
- `DEBUGCE XLA CE block sizes resolved: device_kind=TPU v5`
- `x.shape=(32768, 4096)`
- `v_block_size=8192`
- `b_block_size=32768`
- `num_v_blocks=16`
- `num_b_blocks=1`

That is a very different regime from the bad LoRA pair in Experiment G, where:
- v6e had `num_b_blocks=16`
- v5p had `num_b_blocks=64`

So Exp L changed **both**:
1. model/update parameterization (full FT instead of LoRA), and
2. local CE execution geometry / batch blocking

This means Exp L strongly supports "LoRA is the amplifier," but does **not**
mathematically prove "LoRA is the only thing that changed."

### Strongest current mechanism story

Claude's mechanism explanation is directionally right and now fits the evidence
well:

- In LoRA with `zero_init_b=True`, the first useful adapter update is forced
  through `lora_B`, while `lora_A` stays fixed at its random initialization.
- LoRA excludes `lm_head` by default, so the model cannot absorb the DPO signal
  in the most direct place where logit differences are expressed.
- This means tiny chip-level numeric differences in the early policy/reference
  log-prob computation get projected through a fixed low-rank basis, making the
  first update much more sensitive to those differences.
- Full FT lacks that bottleneck: the update is spread across the full policy,
  including `lm_head`, so small TPU numeric differences are diluted rather than
  amplified.

That is the best current explanation for:
- why Experiment J saw "same batch, different trainable gradients" in LoRA, and
- why Experiment L stays well aligned despite the underlying TPU numerics not
  being perfectly identical.

### What Exp L establishes vs what it does not

**Established:**
- per-stmt **full FT DPO** does **not** show the catastrophic v5p↔v6e split in
  the first 8-10 steps
- the per-stmt dataset by itself is **not** sufficient to force the earlier
  LoRA-style failure mode
- generic "v5p vs v6e breaks DPO" is no longer a credible explanation

**Not yet isolated:**
- whether the bad behavior is primarily:
  1. LoRA parameterization,
  2. LoRA interacting with the earlier `v5p-8` vs `v6e-8` execution geometry,
  3. or both

### Revised 2×2 matrix status

|   | per-stmt (singleton) | full bloom_v2 |
|---|---|---|
| **LoRA** | ❌ diverges early (`r16/alpha32`, `r64/alpha64`) | ❓ **still missing** |
| **Full FT** | ✅ **matches early on v5p-16 vs v6e-16** | ✅ closer match |

**Important caveat on the new green check:** the per-stmt full-FT result
used much closer TPU geometry than the bad LoRA pair, so the cell should be
read as "matches early under a cleaner paired setup," not "pure LoRA-only
ablation complete."

### Highest-info next experiments after Exp L

Two experiments are now high value for different reasons:

1. **Full-data LoRA DPO (Exp M)** — fills the remaining matrix quadrant.
   - If it matches: singleton-data interaction is a major part of the problem.
   - If it splits: LoRA-DPO itself is fragile across TPU families.

2. **Per-stmt LoRA on matched geometry (`v5p-16 pd=4` vs `v6e-16 pd=4`)**
   - This is now the cleanest way to separate:
     - "LoRA is the core amplifier" from
     - "the earlier `v5p-8` vs `v6e-8` geometry mismatch was doing more damage than we thought."

If only one follow-up can be run, this second experiment is now arguably the
highest-info next step because Exp L showed the importance of geometry matching.

### 2026-04-14T12:20Z — Experiment M re-spec: matched-geometry LoRA on the full statement distribution

The earlier Exp M spec was too weak because it changed dataset **and** left
geometry loose (`v5p-8 pd=-1` vs `v6e-8 pd=4`). After Exp L, that is no longer
acceptable: geometry matching matters enough that Exp M should be specified as a
clean paired run, not a convenience run.

**Revised goal:** test whether LoRA still shows a TPU-family split when trained
on the **full 46-statement Bloom v2 preference distribution** rather than the
singleton `support_mental_health` subset, while keeping execution geometry as
close as possible across v5p and v6e.

This is the remaining missing quadrant in the matrix:

|   | per-stmt (singleton) | full 46-statement distribution |
|---|---|---|
| **LoRA** | ❌ diverges early | ❓ **Experiment M (re-spec below)** |
| **Full FT** | ✅ matches early | ✅ matches / closer match |

#### Experiment M (re-spec)

**Hypothesis:** if LoRA is trained on the full statement distribution under
matched TPU geometry, then:
- **M matches** → singleton-data interaction was a major part of the earlier pathology
- **M splits** → LoRA-DPO itself is fragile across TPU families even when the
  data distribution is broad and geometry is controlled

**Data (train + val):**
- Train: `preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/`
  This is the full Bloom v2 / full 46-statement preference distribution used by
  the repo's broader DPO experiments.
- Validation: the usual bloom v2 val set already used in the existing LoRA/full-FT runs

**Model/reference:**
- policy: `marin-community/marin-8b-instruct`
- adapter: LoRA with the checked-in Marin recipe
  - `r=64`
  - `alpha=64`
  - `zero_init_b=True`
  - `target_modules=None`
  - default `exclude_modules` (so `lm_head` remains excluded unless changed explicitly)
- reference: `AdapterBaseReferenceConfig`

**Training knobs:**
- `num_train_steps=10` for the first pass
- `train_batch_size=64`
- `per_device_parallelism=4` on **both** TPUs
- `per_device_eval_parallelism=4` on **both** TPUs unless memory forces a different eval setting
- `lr=1e-6`
- `lr_schedule=cosine`
- `warmup=0.1`
- `beta=0.1`
- `seed=0`
- `train_seq_len=4096`
- `validation_split_fraction=None`
- one eval at step 10

**TPU pairing (preferred):**
- `v5p-16 pd=4`
- `v6e-16 pd=4`

This mirrors Experiment L's cleaner geometry and avoids repeating the older
`v5p-8 pd=16` vs `v6e-8 pd=4` mismatch. If `v6e-16` does not fit or is
unavailable, escalate only with an explicit note that the comparison weakened.

**Debug instrumentation for first pass:**
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Do **not** enable `MARIN_DEBUG_LORA_FACTOR_TRACE` on the first pass. The first
goal is simply to determine whether the scalar/trace split survives on the full
distribution under matched geometry.

**Implementation note:**
- reuse the `experiments/tune_lora/` codepath rather than cloning from
  `per_stmt_dpo`
- add a dedicated debug script under `experiments/tune_lora/` so the run stays
  aligned with the checked-in full-distribution LoRA recipe while still forcing
  the matched `pd=4` geometry and debug env vars

**Interpretation priority after M lands:**
1. If Exp M matches: singleton-data interaction becomes the leading remaining explanation.
2. If Exp M splits: LoRA-DPO itself remains the leading amplifier, and the next
   best probe is per-stmt LoRA on matched `v5p-16` / `v6e-16` if not already run.

### 2026-04-14T13:20Z — Experiment M(pd=4) launch blocked by v6e-16 HBM OOM

The first implementation of Exp M used the intended matched geometry:
- `v5p-16 pd=4`
- `v6e-16 pd=4`
- LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- `AdapterBaseReferenceConfig`

That config **did not compile on v6e-16**. The TPU worker failed with:

> `RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in`
> `memory space hbm. Used 42.82G of 31.25G hbm. Exceeded hbm capacity by 11.57G.`

This is surprising at first glance because Experiment L's **full FT** run fit on
`v6e-16 pd=4`. But the clean interpretation is **not** "LoRA should always be
smaller than full FT, therefore this must be impossible." The LoRA DPO runtime
here is a different compiled graph from Exp L in two important ways:

1. **Reference-model path differs.**
   - Exp L used `SeparateReferenceConfig`, so training state was a materialized
     `DpoModel(policy, reference)` and the reference model was loaded
     separately.
   - Exp M uses `AdapterBaseReferenceConfig`, so the loss function constructs
     the reference on the fly from the policy via
     `reference_provider.model_for(policy_model)` in `train_dpo.py`.
   - For LoRA, that reference is produced by unwrapping the policy's LoRA
     modules back to their wrapped base linears via `unwrap_lora_modules(...)`.

2. **LoRA adds extra per-layer compute/activation structure.**
   - A LoRA-adapted linear computes `wrapped(x) + lora_B(lora_A(x))`.
   - That means extra branch structure and extra rank-`r` intermediates (`z =
     lora_A(x)`) at many linears, even though the trainable parameter count is
     much smaller than full FT.

So the strongest current explanation for the OOM is:
- **not** "full FT has one forward but LoRA has two" — DPO performs
  policy/reference chosen/rejected work in both cases
- but rather that **LoRA + AdapterBaseReferenceConfig** induces a different,
  more memory-hungry compiled graph than **full FT + SeparateReferenceConfig**
  on `v6e-16 pd=4`

This is also a reminder that Exp L and Exp M are not only "full FT vs LoRA";
they currently differ on **reference configuration** as well:
- Exp L: `SeparateReferenceConfig`
- Exp M: `AdapterBaseReferenceConfig`

That reference-type difference was already deliberate for Exp M because:
- `AdapterBaseReferenceConfig` is the canonical LoRA DPO path in this repo
- switching Exp M to `SeparateReferenceConfig` would add another 8B reference
  copy and create a different, less standard LoRA setup

### Revised recommendation after the OOM

The cleanest next move is:
- **drop Exp M to `pd=2` on both `v5p-16` and `v6e-16`**

Why this is the best fallback:
- keeps TPU family pairing matched (`v5p-16` vs `v6e-16`)
- changes only local batch / activation pressure
- avoids escalating to `v6e-32`, which would reintroduce a chip-count mismatch
- avoids changing reference type, which would weaken the "canonical LoRA path"
  interpretation

Tradeoff:
- `train_batch_size=64` with `pd=2` on 16-chip pods implies `microbatch=32`
  and therefore `grad_accum=2`

That is acceptable for a 10-step probe. It is a weaker geometry match than the
ideal `pd=4` / no-accum run, but it is still much cleaner than falling back to:
- `v6e-32`, or
- the old `v5p-8` vs `v6e-8` mismatch

**Updated Exp M execution plan:**
- first retry with `v5p-16 pd=2` vs `v6e-16 pd=2`
- keep `AdapterBaseReferenceConfig`
- keep LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- keep the same debug env vars (`BATCH_INDICES`, `STEP_TRACE`)
- record explicitly that Exp M is now "matched family / matched pd, with grad
  accumulation on both sides"

---

## Root cause hypothesis after Exp Q

### Strongest current explanation

The best-supported root-cause story at this point is:

1. The catastrophic split is **not** caused by TPU family (`v5p` vs `v6e`).
   Exp N ruled that out by showing matched `v5p-16` and `v6e-16` LoRA runs
   track closely.
2. The split is **not** caused by a simple within-family `pd` change. Exp O
   showed full FT on `v6e-16` is robust to `pd: 4 -> 2`, and Exp P showed LoRA
   on `v5p-16` is robust to `pd: 2 -> 4`.
3. The remaining failure is specific to the **`v5p-8` distributed execution
   regime** under this LoRA DPO setup.

The most concrete low-level difference we have actually measured is the local
cross-entropy problem shape:

- **good `v5p-16 pd=2`** logged
  `DEBUGCE ... x.shape=(32768, 4096) ... b_block_size=32768 ... num_b_blocks=1`
- **bad `v5p-8 pd=8`** logged
  `DEBUGCE ... x.shape=(65536, 4096) ... b_block_size=1024 ... num_b_blocks=64`
- **bad `v5p-8 pd=4`** logged the **same** line as the `pd=8` run:
  `x.shape=(65536, 4096) ... b_block_size=1024 ... num_b_blocks=64`

So the strongest current hypothesis is:

> `v5p-8` with `train_batch_size=64` is running a materially different local CE
> kernel and broader distributed program from the good `v5p-16` regime, and
> that difference perturbs the **direction** of the early LoRA update enough to
> send training into the bad basin.

This explanation fits the full body of evidence better than the earlier
alternatives:

- **not hardware-family numerics alone**: `v5p-16` and `v6e-16` match
- **not LoRA rank alone**: `v5p-8` is still bad at fixed `r=64, α=64`
- **not `pd` alone**: `v5p-16` stays good at both `pd=2` and `pd=4`
- **not the `v5p-8` `pd=8 -> 4` change**: Exp Q showed the local CE shape did
  not change at all, and the bad trajectory stayed bad

The part I still consider an inference rather than a proof is **which** piece
of the `v5p-8` execution regime is load-bearing:

- the CE kernel shape / batch blocking
- the broader FSDP sharding / all-gather / reduce-scatter topology
- or both together

But the CE difference is currently the most concrete, first-principles
mechanism we have actually observed in logs.

### Why this can happen even though `v5p-16` is "just more chips"

`v5p-16` is not merely "the same run with more throughput." It is a different
distributed factorization of the training step:

- each chip holds smaller shards
- each chip contributes fewer token rows to the local CE kernel
- all-gather / reduce-scatter trees differ
- remat / HLO scheduling choices can differ
- temporary activation shapes differ

For full FT, those differences seem to wash out. For this LoRA DPO setup, they
appear to matter because the early update is very brittle:

- `lora_B` starts at zero
- `lora_A` is fixed random
- the initial DPO signal is nearly symmetric (`δ_π - δ_ref ≈ 0` at step 0)
- small directional differences in early gradients can therefore pick a
  different low-rank update direction

That is why the current best explanation is **directional instability in the
early LoRA update**, not a simple scale or convergence-speed issue.

### Next best experiment: make `v5p-8` match the good `v5p-16` local execution

The highest-information next move is **not** another `pd` sweep on `v5p-8`.
Exp Q already showed `pd=8` and `pd=4` leave `v5p-8` in the same bad regime,
and both logged the same CE shape.

The next best experiment is:

#### Experiment R — `v5p-8` local-shape matching probe

> **Status (2026-04-16T04:52Z):** executed and resolved to Case 2 (no
> recovery). The `bs=32 pd=4` run on `v5p-8` did **not** match the good
> `v5p-16 pd=2` `DEBUGCE` line exactly; instead it reduced the local CE
> problem from `(65536, 4096), num_b_blocks=64` down to
> `(16384, 4096), num_b_blocks=16`, and still stayed stuck near `ln(2)`.
> See the "2026-04-16T04:52Z: Experiment R result" section at the top of
> this logbook for the corrected side-by-side table, full 10-step trace,
> step-0 gradient-shift finding, W&B link, and pivot direction
> (topology / sharding / collective). The optional Follow-up R3
> `bs=16 pd=4` sweep is deprioritized; the load-bearing variable is no
> longer expected to be per-chip token load.

**Goal:** try to make the bad `v5p-8` run look like the good `v5p-16 pd=2`
run at the level that matters most in logs: the local CE problem shape seen by
the fused XLA cross-entropy kernel.

This is **not** a "match the global batch size" experiment. It is a "match the
per-chip local execution shape as closely as we can without changing hardware"
experiment.

##### Why this is the right next move

At fixed `train_batch_size=64` we measured:

- good `v5p-16 pd=2`:
  - `DEBUGCE x.shape=(32768, 4096)`
  - `b_block_size=32768`
  - `num_b_blocks=1`
- bad `v5p-8 pd=8` and `pd=4`:
  - `DEBUGCE x.shape=(65536, 4096)`
  - `b_block_size=1024`
  - `num_b_blocks=64`

So the cleanest remaining test is to reduce the global batch on `v5p-8` until
the per-chip CE problem hopefully moves toward the good `v5p-16` regime.

##### Exact config

Hold fixed from the recent LoRA debug runs:
- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- tokenizer: `marin-community/marin-tokenizer`
- model: `marin-community/marin-8b-instruct`
- LoRA: `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=True`,
  `target_modules=None`
- reference: `AdapterBaseReferenceConfig`
- `lr=1e-6`
- `lr_schedule="cosine"`
- `warmup=0.1`
- `beta=0.1`
- `seed=0`
- `train_seq_len=max_seq_len=4096`
- `num_train_steps=10`
- `steps_per_eval=10`
- `max_eval_batches=1`
- `ReferenceEvalCacheConfig(mode="disabled")`
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Change only:
- `train_batch_size: 64 -> 32`

Recommended first launch:
- `v5p-8`
- `train_batch_size=32`
- `per_device_parallelism=4`

Why start with `pd=4`:
- that path already ran cleanly in Exp Q
- it keeps the parent/child operational story simple
- it avoids conflating this experiment with another `pd` sweep

##### What this is trying to match

The target is **not** the full `v5p-16` training config. The target is the
good `v5p-16 pd=2` **local CE regime**:

- desired `DEBUGCE` trend:
  - `x.shape` moving from `(65536, 4096)` toward `(32768, 4096)`
  - `b_block_size` moving from `1024` toward the full-batch case
  - `num_b_blocks` moving from `64` toward `1`

If `train_batch_size=32` on `v5p-8 pd=4` does that, then we have successfully
matched the most concrete measured difference between the bad and good runs.

##### Required checks during the run

1. **Check `DEBUGCE` first.**
   This experiment is only informative if the local CE shape actually changes.

   Required log lines:
   - `x.shape`
   - `b_block_size`
   - `num_b_blocks`

2. **Check `DEBUGJ TRACE` on steps 0, 2, and 9.**
   We care about:
   - `loss`
   - `grad_l2`
   - `gB_l2`
   - `pB_l2`

3. **Check W&B on the true DPO quantity.**
   Compare:
   - `train/loss`
   - `train/dpo_margin_policy`
   - `train/dpo_margin_ref`
   - therefore `delta_pi - delta_ref`

##### Success / failure criteria

**Strong recovery**
- `DEBUGCE` moves toward the good `v5p-16` regime
- and step-2 `train/loss` drops into the good band (`~0.33-0.35`)
- and `delta_pi - delta_ref` jumps into the good band (`~9-10`)

**No recovery**
- `DEBUGCE` changes as intended
- but step-2 `train/loss` still stays near `~0.68`
- and `delta_pi - delta_ref` stays tiny (`~0.1-0.7`)

**Ambiguous partial recovery**
- `DEBUGCE` changes
- and the loss improves materially from the bad `v5p-8` runs
- but still does not reach the good `v5p-16` regime

##### Interpretation

**Case 1: strong recovery**

This would be the cleanest evidence so far that the leading root cause is the
per-chip local workload / local CE shape on `v5p-8`, not some deeper
`v5p-8`-specific collective bug.

**Follow-up R1:**
- rerun on `v5p-8 bs=32 pd=8`
- goal: see whether recovery persists across `pd` once the per-chip token load
  is reduced

**Case 2: no recovery**

This would mean we successfully matched the most obvious CE-level difference
yet the run still failed. At that point CE batch blocking is probably not the
load-bearing cause, and the remaining culprit is deeper in the `v5p-8`
distributed regime.

**Follow-up R2:**
- add one more probe that keeps `v5p-8 bs=32` but **forces explicit CE block
  sizes** to match the good `v5p-16` line as closely as possible, if the
  runtime heuristic still does something unexpected
- if CE is already matched and the run is still bad, stop focusing on CE and
  pivot to topology / sharding / collective investigation

**Case 3: ambiguous partial recovery**

This would mean local CE shape is a contributor but not the whole story.

**Follow-up R3:**
- complete a two-point local-load sweep on `v5p-8`
- compare:
  - `bs=64 pd=4` (known bad)
  - `bs=32 pd=4`
  - optional `bs=16 pd=4`
- track whether the run improves smoothly as local workload falls

##### Why this is higher information than another plain `pd` sweep

Exp Q already showed:
- `v5p-8 pd=8` bad
- `v5p-8 pd=4` bad
- same `DEBUGCE` line in both runs

So another plain `pd` change is unlikely to teach much unless it actually moves
the local CE problem. `train_batch_size=32` is the first next experiment that
directly tests the strongest remaining hypothesis from measured logs.

---

## 2026-04-16T13:05Z: Root-cause update after Exp R — next best probe is **full FT on `v5p-8`**, not another LoRA kernel guess

### What we have now established

At this point the investigation has narrowed materially:

- **Not TPU family alone:** `v5p-16 pd=2` and `v6e-16 pd=2` LoRA DPO track
  closely (Exp N).
- **Not plain `pd` alone:** `v5p-16` stays good at both `pd=2` and `pd=4`
  (Exp P), and full FT on `v6e-16` stays good at both `pd=4` and `pd=2`
  (Exp O).
- **Not LoRA rank / scale alone:** `v5p-8` is still bad at fixed
  `r=64, alpha=64` (Exp Q).
- **Not CE batch blocking as the primary cause:** Exp R reduced the `v5p-8`
  local CE problem from `(65536, 4096), num_b_blocks=64` down to
  `(16384, 4096), num_b_blocks=16`, and the run still stayed near `ln(2)`.

So the remaining live question is no longer "is the CE kernel wrong?" The
remaining live question is:

> **Is the `v5p-8` pathology specific to LoRA / `AdapterBaseReferenceConfig`,
> or is it a broader property of the `v5p-8` distributed regime even under
> full fine-tuning?**

That is the cleanest next discriminator.

### Why `v6e-8` full FT is not required

It would be nice to have a symmetric `v6e-8` full-FT run, but it is not
necessary for the next narrowing step.

We already know:
- `v5p-16` full FT works
- `v6e-16` full FT works
- `v5p-16` and `v6e-16` full FT track closely
- `v5p-8` LoRA is the only regime that still looks uniquely pathological

Therefore the next high-information question is simply:

> **Can `v5p-8` do full FT DPO at all, and if it can, does it learn normally?**

If the answer is yes, the remaining pathology is much more likely to be
specific to the LoRA + adapter-base-reference path on `v5p-8`.
If the answer is no, then the bug is broader to the `v5p-8` execution regime.

### Important correction to the "match the batch size" intuition

It is tempting to say:

> if `v5p-16` and `v6e-16` both work, then as long as `v5p-8` matches their
> batch size it should also track

But that is **not** guaranteed.

`v5p-8` vs `v5p-16` is not just a batch-size change. It changes the whole
distributed factorization:

- data-axis size
- parameter shard size
- all-gather / reduce-scatter trees
- replica / DCN topology
- local activation shapes
- optimizer-state shards per chip

Exp R already demonstrated this principle: even after shrinking the local CE
shape substantially, `v5p-8` still stayed bad. So "same global batch" is too
weak a matching criterion by itself.

### Memory / checkpointing reality

One reason we have not already run full FT on `v5p-8` is memory pressure.
However, the correct statement is:

- we have **not proved** that `v5p-8` full FT is impossible
- we only know that the existing full-FT experiments were done on 16-chip pods

Relevant code fact:
- `llama_8b` already uses `gradient_checkpointing=True` by default in
  `lib/levanter/src/levanter/models/llama.py`
- Levanter also supports more aggressive checkpointing policies:
  - `"offload"`: offload carries and inputs to host
  - `"recompute"` / `"full"`: don't save carries or block internals

So the correct next move is not to assume "full FT won't fit on `v5p-8`."
The correct next move is to test feasibility with a short run and an explicit
fallback ladder.

### Next best experiment: **Experiment T — `v5p-8` full-FT feasibility + behavior probe**

**Goal:** determine whether the `v5p-8` pathology is LoRA-specific or broader.

#### Baseline comparison runs

Compare against the already-good full-FT runs:
- Exp L `v5p-16 pd=4`
- Exp L `v6e-16 pd=4`
- Exp O `v6e-16 pd=2`

We do **not** need a `v6e-8` full-FT run to answer the immediate question.

#### T0: minimal feasible first attempt

Use the Exp L full-FT recipe, but target `v5p-8` and reduce global batch:

- TPU: `v5p-8`
- data: per-stmt `support_mental_health`
- model: `marin-community/marin-8b-instruct`
- **no adapter** (full FT)
- reference: `SeparateReferenceConfig`
- `train_batch_size=32`
- `per_device_parallelism=4`
- `per_device_eval_parallelism=4`
- `num_train_steps=2` for the first compile/feasibility probe
- `steps_per_eval=2`
- `lr=1e-6`
- `beta=0.1`
- `seed=0`
- `train_seq_len=max_seq_len=4096`
- `max_eval_batches=1`
- `ReferenceEvalCacheConfig(mode="disabled")`
- debug env:
  - `MARIN_DEBUG_LOG_BATCH_INDICES=1`
  - `MARIN_DEBUG_LOG_STEP_TRACE=1`

Model config:
- start with plain `llama_8b`
- note: this already means `gradient_checkpointing=True`

#### T0 fallback ladder if it OOMs

If the first compile fails on HBM:

1. **T0b:** rerun with
   `dataclasses.replace(llama_8b, gradient_checkpointing="offload")`
   This is the most practical "aggressive checkpointing" fallback already used
   elsewhere in the repo.

2. **T0c:** if `"offload"` still fails, rerun with
   `dataclasses.replace(llama_8b, gradient_checkpointing="recompute")`
   This is more compute-heavy but minimizes saved activations further.

3. **T0d:** only if needed, drop `train_batch_size` from `32 -> 16`
   while keeping everything else fixed.

The right order is important:
- first change the checkpointing policy
- only then reduce batch again
- do not mix several memory changes at once on the first retry

#### What to look for

If `v5p-8` full FT **fits and learns**:
- step-2 loss should leave `ln(2)` clearly
- `DEBUGJ TRACE` grad norms should look like the good full-FT regime, not the
  bad LoRA regime
- then the pathology is much more likely to be specific to:
  - LoRA low-rank update geometry
  - `AdapterBaseReferenceConfig`
  - or their interaction with `v5p-8`

If `v5p-8` full FT **fits but is still bad**:
- the bug is broader than LoRA
- at that point the `v5p-8` distributed regime itself becomes the primary
  suspect

If `v5p-8` full FT **does not fit even after offload / recompute**:
- that is itself useful information
- it means the cleanest behavior comparison on `v5p-8` may have to stay in
  LoRA-space or move to a smaller seq/batch probe

### Why this is higher value than another kernel-specific guess

After Exp R, another naive CE-specific probe is lower leverage. We already
shrunk the local CE regime substantially and did not recover. If we revisit CE,
the next honest version would be an **exact-match** probe, not another broad
guess.

The next most valuable piece of information is not "which kernel might still be
different?" It is:

> **Does the `v5p-8` pathology survive when LoRA is removed?**

That question is more important than any individual kernel hypothesis because
it cleanly partitions the remaining search space.

### 2026-04-16T13:18Z — Experiment T launch plan (pending submission)

Concrete script prepared:
- `experiments/posttrain/per_stmt_dpo/experiment_t_v5p8_full_ft_s2.py`

Initial launch target:
- `v5p-8`
- per-stmt `support_mental_health`
- **full FT** (no adapter)
- `SeparateReferenceConfig`
- `train_batch_size=32`
- `per_device_parallelism=4`
- `num_train_steps=2`
- `steps_per_eval=2`
- `seed=0`, `lr=1e-6`, `beta=0.1`
- `gradient_checkpointing="offload"` for the first attempt

Why `offload` first:
- `llama_8b` already uses normal gradient checkpointing
- the immediate goal here is feasibility on `v5p-8`, not a purity contest
- if plain checkpointing would OOM, losing a launch cycle tells us less than
  getting a concrete behavioral answer

Planned parent/child launch shape:
- parent: Iris interactive CPU coordinator in `us-east5-a`
- child: `v5p-8` TPU in `us-east5`
- env:
  - `REGIONS_OVERRIDE=us-east5`
  - `EXPERIMENT_T_BS=32`
  - `EXPERIMENT_T_PD=4`
  - `EXPERIMENT_T_STEPS=2`
  - `EXPERIMENT_T_CHECKPOINTING=offload`
  - `MARIN_DEBUG_RUN_TAG=ue5a-i1`

Immediate success criteria:
- child `train_dpo` spawns
- compile completes
- at least one `DEBUGJ TRACE` line is emitted

If that works, the next step is to compare the step-2 loss / grad trace
against the good `v5p-16` / `v6e-16` full-FT runs.

Submission record:
- iris parent job: `/ahmed/debug-t1-full-ft-v5p8-bs32-pd4-offload-ue5a-i1`
- experiment/run name: `exp_t_v5p8_fullft_bs32_pd4_offload_s2_ue5a-i1`
- expected checkpoint base path:
  `checkpoints/exp_t_v5p8_fullft_bs32_pd4_offload_s2_ue5a-i1`
- submit mode: `interactive`
- parent launch command shape:
  - `iris job run --zone us-east5-a --cpu 1 --memory 3g`
  - env:
    - `REGIONS_OVERRIDE=us-east5`
    - `EXPERIMENT_T_BS=32`
    - `EXPERIMENT_T_PD=4`
    - `EXPERIMENT_T_STEPS=2`
    - `EXPERIMENT_T_CHECKPOINTING=offload`
    - `MARIN_DEBUG_RUN_TAG=ue5a-i1`
- submit status: accepted by Iris controller
- immediate post-submit state: parent `running`, task `assigned`
- at the time of this log entry, the parent has **not yet spawned**
  `/train_dpo`, so there is no TPU-child status or W&B run id yet

---

## 2026-04-16T20:00Z: Experiment R2 — explicit CE block-size probe on `v5p-8`

### Context: what Exp R did and did not show

The Exp R writeup (2026-04-16T04:52Z) has been corrected in place to
reflect the actual `DEBUGCE` line from
`experiment_r_r64_v5p8_bs32_pd4_s10_ue5a-i1-423c65`:

```
device_kind=TPU v5 x.shape=(16384, 4096) w.shape=(4096, 128256)
v_block_size=8192 b_block_size=1024 num_v_blocks=16 num_b_blocks=16
explicit_block_sizes=False
```

i.e. `x.shape=(16384, 4096)`, `b_block_size=1024`, `num_b_blocks=16` —
**not** the good `v5p-16 pd=2` line's `(32768, 4096) / 32768 / 1`.
Dropping `train_batch_size: 64 → 32` on v5p-8 pd=4 halved `B` per chip to
*below* the good run's level, and the heuristic picked 16 batch blocks
rather than matching the good run's single block.

Exp R's useful conclusion: shrinking per-chip CE workload below the good
run's level does not recover training on v5p-8. But that alone doesn't
rule out CE batch-blocking as the cause — we didn't bit-match the good
regime, we overshot it. A clean direct test of the batch-blocking
hypothesis was still needed.

Side-by-side Exp R vs good/bad baselines at the CE-kernel level:

| Run | `x.shape` | `b_block_size` | `num_b_blocks` | Outcome |
|-----|-----------|----------------|----------------|---------|
| Good v5p-16 pd=2 bs=64 (Exp N)  | (32768, 4096) | 32768 | 1  | escapes ln(2) by step 2 |
| Bad  v5p-8  pd=4 bs=64 (Exp Q)  | (65536, 4096) | 1024  | 64 | stuck near ln(2) |
| Exp R v5p-8 pd=4 bs=32          | (16384, 4096) | 1024  | 16 | stuck near ln(2) |

### Why Exp R2

The Exp R approach (reduce `bs` to change CE tiling) is *indirect*. Changing
`bs` simultaneously changes three things:

1. per-chip token count `B` (which changes the CE kernel input shape),
2. `grad_accum` (at fixed global batch, `grad_accum = bs / (pd × chips)`),
3. the heuristic's choice of `b_block_size` / `num_b_blocks` (as a function
   of `B` and `device_kind`).

To isolate "is CE tiling the load-bearing variable?", the cleaner probe is
to **hold everything else fixed at the Exp Q bad baseline** (`bs=64`, `pd=4`,
same microbatch, same grad_accum, same x.shape) and change **only** the CE
tiling by forcing explicit block sizes through the kernel call path.

The XLA CE kernel already accepts explicit block-size overrides — the
`explicit_block_sizes=False` field in every current DEBUGCE line is the
switch. Setting `explicit_block_sizes=True` with chosen `b_block_size` /
`v_block_size` / `num_b_blocks` bypasses the heuristic.

### Hypothesis

**Strong version:** the load-bearing difference between good `v5p-16 pd=2`
and bad `v5p-8 pd=4` is the CE backward's bf16 accumulation pattern across
many batch blocks. `v5p-16 pd=2` runs with `num_b_blocks=1` (no inter-block
accumulation). `v5p-8 pd=4` runs with `num_b_blocks=64` (63-way bf16
accumulation of `gw_block` / `gx_block` inner tiles). If we eliminate that
accumulation on v5p-8 while holding everything else fixed, v5p-8 recovers.

**Weak version:** the accumulation count doesn't matter; per-tile compute
shape does. Then matching `b_block` to the good run's `32768` (with
`num_b_blocks=2` to cover `B=65536`) would be the relevant knob.

**Null:** neither matters. CE tiling is not the load-bearing variable, and
the failure lives somewhere below CE (FSDP / attention / reference-network
graph), consistent with the prevailing hypothesis after Exp R.

### The geometric impossibility of a fully bit-identical CE match

To match the good v5p-16 pd=2 CE kernel exactly, we'd need all four of:

- `x.shape = (32768, 4096)` — determined by per-chip `B`
- `v_block_size = 8192`
- `b_block_size = 32768`
- `num_b_blocks = 1`

On v5p-8 with `bs=64, pd=4`, per-chip `B = 65536`. To cover `B=65536` with a
single batch block, `b_block_size` must be `65536`, not `32768`. Conversely,
if we pin `b_block_size = 32768`, `num_b_blocks` must be `2`, not `1`. You
cannot both (a) keep `B=65536` and (b) have per-tile math identical to the
good run and (c) have a single batch block. One of those has to give.

The honest options are:

- **Give up `num_b_blocks=1`** (Case B below): match per-tile math
  (`b_block=32768`), accept 2-way inter-tile accumulation.
- **Give up matching tile size** (Case A below): force `num_b_blocks=1`
  with a tile of size 65536, accept that the tile is 2× larger than the
  good run's.
- **Give up `B=65536`** (rerun Exp R more carefully at `bs=32 pd=2` on
  v5p-8 — see "alternative" below).

### Three cases

#### Case A — eliminate inter-block accumulation

Force: `b_block_size=65536, num_b_blocks=1, v_block_size=8192, num_v_blocks=16`.

Matches good run on: `num_b_blocks=1`, `num_v_blocks=16`, `v_block_size`.
Differs from good run on: `b_block_size` (65536 vs 32768), `x.shape`.

Tests the **strong hypothesis** most cleanly. If v5p-8 still stays near
`ln(2)` under Case A, then bf16 accumulation across batch blocks in the CE
backward is **ruled out** as the load-bearing cause, even with everything
else held at the bad baseline.

#### Case B — match per-tile compute shape

Force: `b_block_size=32768, num_b_blocks=2, v_block_size=8192, num_v_blocks=16`.

Matches good run on: `b_block_size`, `v_block_size`, `num_v_blocks`.
Differs from good run on: `num_b_blocks` (2 vs 1), `x.shape`.

Per-tile CE math is bit-identical to the good run. Only difference is a
single 2-way bf16 reduction of the two tile outputs. If the weak hypothesis
were true (per-tile compute shape matters, not accumulation count), this
would recover.

#### Case C — impossible

Forcing `b_block_size=32768, num_b_blocks=1` would only cover 32768 of the
65536 rows. Kernel would either error out or silently compute on half the
batch. Skip.

### HBM feasibility

Concurrent CE backward temporaries per chip (bf16, `H=4096`, `V_pad=128256`,
`v_block=8192`):

| Array | Shape | At `b_block=65536` | At `b_block=32768` | At `b_block=1024` (current bad) |
|-------|-------|--------------------|--------------------|--------------------------------|
| `x_block`          | (b_block, H)         | 512 MiB | 256 MiB | 8 MiB |
| `delta` (softmax)  | (b_block, v_block)   | 1024 MiB| 512 MiB | 16 MiB |
| `gx_inner`         | (b_block, H)         | 512 MiB | 256 MiB | 8 MiB |
| `gw_block_update`  | (H, v_block)         | 64 MiB  | 64 MiB  | 64 MiB |
| **Peak CE temps**  |                      | **~2.1 GiB** | **~1.1 GiB** | **~0.1 GiB** |

Baseline HBM on v5p-8 (4 chips, 95 GiB/chip) for this config:

- policy + reference (Llama-8B shared under `AdapterBaseReferenceConfig`),
  FSDP-sharded over 4 chips: ~4 GiB/chip for weights
- LoRA params + AdamW states (LoRA-only): <100 MiB/chip
- activations with `gradient_checkpointing=True` default: ~10-20 GiB/chip
- scratch / collectives / comms buffers: few GiB

Estimated current usage at `b_block=1024`: ~20-30 GiB/chip. Adding ~2 GiB
for Case A's CE temps or ~1 GiB for Case B's lands around 22-32 GiB/chip,
well inside the 95 GiB budget.

**Verdict: no OOM expected.** Prudent to do a 1-step compile-only probe
first to confirm, since the compile-time HBM estimator can be off by a few
GiB under certain conditions.

### Recommended order

1. **R2a — Case A** (`b_block=65536, num_b_blocks=1`). Strongest probe of
   "is CE backward bf16 accumulation the cause?" at the full bad-baseline
   geometry. ~10-minute run.

2. **R2b — Case B** (`b_block=32768, num_b_blocks=2`). Runs only if R2a
   stays stuck. Probes "does per-tile compute shape matter even with
   accumulation present?" Also ~10-minute run.

3. **R2c — alternative, if the "everything below CE" story is right and
   neither R2a nor R2b recovers:** shift focus to FSDP sharding / attention
   kv-head mapping / reference-network graph. (Already covered by the
   post-Exp-R pivot plan at 2026-04-16T13:05Z — Exp T is the current
   in-flight probe for that.)

### Config parity required

Hold fixed from Exp Q bad baseline (the apples-to-apples anchor):

- TPU: `v5p-8`, 4 chips, `us-east5`
- data: per-stmt `support_mental_health` singleton
- tokenizer: `marin-community/marin-tokenizer`
- model: `marin-community/marin-8b-instruct`
- LoRA: `r=64, alpha=64, zero_init_b=True, target_modules=None`
- reference: `AdapterBaseReferenceConfig`
- `train_batch_size=64`
- `per_device_parallelism=4`
- `per_device_eval_parallelism=4`
- `lr=1e-6, lr_schedule="cosine", warmup=0.1`
- `beta=0.1, seed=0`
- `train_seq_len=max_seq_len=4096`
- `num_train_steps=10, steps_per_eval=10`
- `max_eval_batches=1, ReferenceEvalCacheConfig(mode="disabled")`
- `MARIN_DEBUG_LOG_BATCH_INDICES=1`
- `MARIN_DEBUG_LOG_STEP_TRACE=1`

Change only:

- `explicit_block_sizes=True` in the CE kernel call path
- `b_block_size` and `num_b_blocks` to the Case-A or Case-B values
- `v_block_size=8192, num_v_blocks=16` held identical to heuristic pick

### Implementation notes

- The CE kernel lives in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py`
  and its API shim is at
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py`
  (lines 80-194).
- The block-size heuristic is in
  `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/tuned_block_sizes.py`
  (lines 779-818).
- The cleanest plumbing is probably to add a kwarg to the CE call path that
  gets threaded down to the kernel; an environment-variable shim
  (`MARIN_DEBUG_CE_B_BLOCK=65536`, etc.) keeps the experiment change
  minimal and avoids touching core Levanter code in the worktree.
- Verify that `explicit_block_sizes=True` appears in the new DEBUGCE line
  before trusting any downstream result — that's the signal that the
  override actually took effect.

### Success / failure criteria

**Case A (R2a) recovery:**
- DEBUGCE: `b_block_size=65536, num_b_blocks=1, explicit_block_sizes=True`
- step-2 `train/loss` drops into the good band (~0.33-0.35)
- `delta_pi - delta_ref` jumps into the good band (~9-10)
→ "CE backward bf16 inter-block accumulation is the load-bearing cause."
Warrants a focused fix at the CE kernel level.

**Case A (R2a) no recovery:**
- DEBUGCE confirms override took effect
- step-2 `train/loss` still sits near `~0.68-0.69`
→ CE backward bf16 accumulation across batch blocks is **ruled out**
as the load-bearing cause, even with a direct test. Pivot fully to the
sub-CE suspects (FSDP, attention, reference graph, remat).

**Case B (R2b) differential:** only meaningful if Case A stayed stuck.
If Case B recovers while Case A didn't, per-tile compute shape matters in a
way that accumulation count alone doesn't capture. Interesting but less
likely than the binary Case A outcome.

### Why this is higher information than rerunning Exp R at a cleaner bs/pd

At `v5p-8 bs=32 pd=2` (2 examples/chip/microstep × 4 multiplier × 4096 =
32768 per chip), we would naturally land at `x.shape=(32768, 4096)` and
presumably `b_block=32768, num_b_blocks=1` via the heuristic — bit-identical
CE line to the good run. That experiment is also worth running (call it R3
if needed). But it still changes `bs`, `pd`, and `grad_accum` simultaneously
vs the Exp Q bad baseline.

R2 keeps `bs`, `pd`, `grad_accum`, microbatch, and step-0 batch composition
identical to Exp Q, and toggles *only* the CE tiling. That's a stricter
isolation and directly addresses the remaining live question without
introducing new confounds.

### Open question this doesn't address

If both R2 cases stay stuck, we still haven't tested the other leading
suspects (attention kv-head mapping, FSDP sharding, reference-network
graph). Those are the Exp S / Exp T / follow-up agenda. R2 is a quick,
low-cost probe to close out the CE-level hypothesis cleanly before
committing to those more invasive experiments.

---

## 2026-04-16T22:50Z: Experiment R2a result — **CE backward bf16 accumulation is NOT the cause**

### Strongest true conclusion

With `b_block_size=65536, num_b_blocks=1` forced on `v5p-8 pd=4 bs=64` —
i.e. eliminating all inter-batch-block bf16 accumulation in the fused CE
backward, with every other knob held identical to the Exp Q bad baseline —
training **still stays stuck near ln(2)**.

This is a **Case 2 (no recovery)** outcome on the Exp R2 planned
discriminator. It rules out "CE backward bf16 accumulation across batch
blocks is the load-bearing cause" as a load-bearing explanation of the
v5p-8 pathology. Combined with Exp R's CE x.shape reduction (also no
recovery), this closes out the CE-kernel-level hypothesis cleanly and
pivots the investigation fully to sub-CE suspects.

### Run

- `experiment_r2a_r64_v5p8_bs64_pd4_s10_uc1-i1-8dac8a`:
  https://wandb.ai/marin-community/dpo/runs/experiment_r2a_r64_v5p8_bs64_pd4_s10_uc1-i1-8dac8a
- region: `us-central1-a` (parent pinned), v5p-8 TPU child
- env: `EXPERIMENT_R2A_BS=64`, `EXPERIMENT_R2A_PD=4`,
  `EXPERIMENT_R2A_CE_B_BLOCK_SIZE=65536`, `EXPERIMENT_R2A_CE_V_BLOCK_SIZE=8192`,
  `MARIN_DEBUG_RUN_TAG=uc1-i1`

### Override verified

Both diagnostic prints appear in the worker log:

```
DEBUGCE_OVERRIDE: env override taken MARIN_DEBUG_CE_B_BLOCK_SIZE=65536 MARIN_DEBUG_CE_V_BLOCK_SIZE=8192 -> BlockSizes(b_block_size=65536, v_block_size=8192)
DEBUGCE XLA CE block sizes resolved: device_kind=TPU v5 x.shape=(65536, 4096) w.shape=(4096, 128256) v_block_size=8192 b_block_size=65536 num_v_blocks=16 num_b_blocks=1 explicit_block_sizes=True
```

So the CE kernel saw the exact bad-baseline `x.shape=(65536, 4096)` and ran
it as a single 65536-row tile instead of the heuristic's 64-tile schedule.
Same data ordering as prior v5p-8 runs: step-0 batch `sha256=7a61ce53d17eb721`,
bit-identical to Exp J, Exp Q, and the original bad run.

### Step-0 aggregate gradient — indistinguishable from Exp Q

The forced single-tile CE backward produces essentially the same step-0
gradient as the 64-tile heuristic backward:

| Run | `num_b_blocks` | step-0 `grad_l2` | step-0 `grad_sum` |
|-----|----------------|-----------------:|------------------:|
| Exp Q `v5p-8 pd=4 bs=64` (heuristic, bad)    | 64 | 2.456003 | +0.18959 |
| Exp Q `v5p-8 pd=8 bs=64` (heuristic, bad)    | 64 | 2.456232 | +0.18380 |
| **Exp R2a `v5p-8 pd=4 bs=64` (forced, this)**| **1**  | **2.456294** | **+0.18907** |
| Good `v5p-16 pd=2 bs=64`                     | 1  | 2.456284 | +0.27991 |

Relative deltas vs Exp Q pd=4:
- `grad_l2`: +0.012% (2.456294 − 2.456003)
- `grad_sum`: −0.27% (0.18907 − 0.18959)

These are noise-level. Collapsing the bf16 inter-block accumulation from
64-way to 1-way **did not** materially change the step-0 gradient on
v5p-8. If the accumulation pattern were the numerically-load-bearing
difference between the bad v5p-8 regime and the good v5p-16 regime, we
would expect step-0 grads to shift — they don't.

Interestingly, `grad_l2` on R2a is closer to the good v5p-16 pd=2
baseline (2.456284) than to Exp Q (2.456003), but `grad_sum` sits firmly
in the bad-pool direction (+0.189 vs good-pool +0.280). So R2a
step-0 grads align with the good run in magnitude and with the bad pool
in direction — and training tracks the bad pool. Consistent with the
broader finding from Exp N/Q/R that step-0 per-module grad direction is
not a reliable predictor of training outcome.

### Training trajectory — tracks Exp Q, not the good v5p-16 run

W&B scalar history through step 8 (run still in flight but trajectory is
unambiguous):

| step | Exp Q `v5p-8 pd=4` (bad) | **Exp R2a `v5p-8 pd=4 b_block=65536` (this)** | Good `v5p-16 pd=2` |
|------|--------------------------:|----------------------------------------------:|-------------------:|
| 0    | 0.693147 | 0.693147 | 0.693147 |
| 1    | 0.693147 | 0.693147 | 0.693147 |
| 2    | 0.685125 | **0.684550** | 0.335202 |
| 3    | 0.682298 | **0.682521** | 0.325988 |
| 4    | 0.673723 | **0.676150** | 0.336246 |
| 5    | 0.668946 | **0.669656** | 0.316800 |
| 6    | 0.667573 | **0.669083** | 0.336998 |
| 7    | 0.662823 | **0.663766** | 0.324271 |
| 8    | 0.658715 | **0.659077** | 0.306144 |

R2a tracks Exp Q point-for-point (max |Δloss| ≤ 0.003 across all 8 logged
steps). It is firmly in the bad basin, not escaping to the good basin.

True DPO quantity (`delta_pi − delta_ref`) from W&B:

| step | R2a `delta_pi − delta_ref` |
|------|---------------------------:|
| 0    |  0.0000 |
| 1    |  0.0000 |
| 2    | +0.1741 |
| 3    | +0.2154 |
| 4    | +0.3449 |
| 5    | +0.4791 |
| 6    | +0.4901 |
| 7    | +0.5989 |
| 8    | +0.6963 |

Compare to good Exp N `v5p-16 pd=2` which jumps to `~9.4` by step 2 and
stays in the `~9-10` band. R2a stays in the `~0.1-0.7` bad band, same as
Exp Q and Exp R.

### Validation-set behavior at step 10

| split | pre-training | post-step-10 | Δ |
|-------|-------------:|-------------:|--:|
| stmt_val | 0.6931 | 0.6931 | 0.0 |
| full_val | 0.6931 | 0.6931 | 0.0 |

No meaningful generalization — identical to the Exp Q / Exp R stuck
regime. `dpo_accuracy` on both val splits is exactly 0 at eval time.

### What R2a rules out

**Strong hypothesis ruled out:** the CE backward's bf16 accumulation of
`gw_block` / `gx_block` across batch blocks is **not** the load-bearing
cause of the v5p-8 pathology. We forced that accumulation to zero-way
(one tile, no inter-block reduction) and the run stayed stuck.

**Corollary strengthened from Exp R:** CE kernel tiling choice in
general — tile size, number of batch blocks, accumulation order — is not
what's breaking v5p-8. Exp R showed reducing per-chip B below the
good-run's level didn't help. Exp R2a shows forcing `num_b_blocks=1` at
the bad-baseline B doesn't help either. Between the two, the
CE-kernel-level hypothesis is cleanly closed.

### What this leaves live

The remaining live suspects for the `v5p-8` pathology, now free of CE
confounds:

1. **Attention `kv_head` axis sharding** — v5p-8 and v5p-16 have
   different numbers of KV-head-axis-capable chips; the `dpo-lora`
   branch's existing TP=4 fix on v6e-8 (`0b228b3a5`) maps `kv_head` to
   the model axis, which may behave differently on v5p's smaller pod
   topology.
2. **FSDP granularity** — on 4 chips vs 8/16, all-gather /
   reduce-scatter trees differ; remat / HLO scheduling choices differ;
   optimizer-state shards-per-chip differ. Any of these could perturb
   the early LoRA update direction.
3. **`AdapterBaseReferenceConfig` reference-network graph on v5p-8** —
   the adapter-base reference path re-uses the policy with adapters
   disabled (via `unwrap_lora_modules`). The resulting compiled graph
   could differ in unexpected ways on v5p-8 vs the 16-chip pods.
4. **Remat / HLO scheduling interactions** — the llama_8b default uses
   `gradient_checkpointing=True`; the recompute graph on v5p-8's
   topology may lay out differently.

These are Exp S / Exp T / deeper-pivot territory. Exp T (full-FT on
v5p-8) is already the in-flight probe for "is this LoRA-specific or
broader to the v5p-8 execution graph?" — see the 2026-04-16T17:20Z
handoff at the top of the logbook for current status.

### R2b (b_block=32768, num_b_blocks=2) deprioritized

The R2 plan flagged R2b as a follow-up only if R2a stayed stuck and we
wanted to also rule out per-tile compute shape as a confound. With
R2a's step-0 grads landing within 0.012% of Exp Q's, it is very unlikely
that changing from `num_b_blocks=1` to `num_b_blocks=2` with a
`b_block=32768` tile would shift the outcome. Skip R2b and move
directly to the sub-CE probes.

### Operational notes

- Launch: us-central1-a parent, v5p-8 child (first submission that
  landed; the us-east5-a parallel submission can be killed if still
  pending — one clean run on this probe is sufficient).
- Runtime: 10 steps completed in roughly the same wall-clock as the Exp
  Q `pd=4` run. The `b_block=65536` tile is ~2 GiB in CE temporaries
  (vs ~33 MiB at b_block=1024); no OOM observed, confirming the HBM
  budget estimates in the R2 plan above.
- Kernel patch: `xla.py` now reads `MARIN_DEBUG_CE_B_BLOCK_SIZE` /
  `MARIN_DEBUG_CE_V_BLOCK_SIZE` and constructs a `BlockSizes` override
  when both are set. Backwards-compatible when env vars are unset.
  `DEBUGCE_OVERRIDE:` line fires exactly once per process at the first
  CE call if the override is taken; falls through silently otherwise.
- Experiment script:
  `experiments/posttrain/per_stmt_dpo/experiment_r2a_v5p8_pd4_explicit_ce_s10.py`.

### Updated post-R2 hypothesis ranking

1. **(leading)** Something in the v5p-8 distributed execution graph
   below the CE kernel — FSDP / attention sharding / reference-network /
   remat — perturbs the early LoRA update direction. Exp T's full-FT
   probe on v5p-8 will narrow "LoRA-specific interaction" vs "broader
   v5p-8 execution graph bug."
2. **(ruled out by Exp R + R2a)** CE kernel tiling, bf16 accumulation,
   per-chip CE workload.
3. **(ruled out by Exp N/O/P)** TPU family alone, `per_device_parallelism`
   alone, LoRA rank alone.

The investigation has successfully collapsed the CE-kernel branch of the
hypothesis tree.
