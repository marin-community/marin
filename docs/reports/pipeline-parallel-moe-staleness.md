# Pipeline Parallelism for MoE Training: Staleness, Correction, and a Throughput Verdict

## TL;DR

We asked whether a throughput-optimal (asynchronous) pipeline-parallel schedule is
worth adopting for MoE pretraining, given that such schedules apply **stale
gradients** — the weights used in the backward pass are several optimizer steps
behind. We studied the staleness in software, by injecting a controlled
per-parameter gradient delay into a single `jax.jit`'d training step, so we could
measure the convergence cost without first building pipeline parallelism. The base
optimizer is Muon.

What we found:

- **Modeling pipeline staleness as a single global delay overstates its cost by
  2–3×.** A real 1F1B/asynchronous pipeline keeps the last stage fresh (delay
  τ=0) and the delay grows toward the input embedding. With the faithful
  per-stage profile, a 6-stage pipeline plateaus +0.086 above synchronous
  training; a uniform τ=5 model plateaus +0.282.
- **The staleness cost is a recoverable token tax, not a quality floor.** Muon
  needs 1.33× the tokens of synchronous training to reach matched loss at 6k
  steps, falling to 1.16× at 15k and still descending. Adam (our `adamh` variant)
  needs 2.44× under the same delay.
- **Weight prediction is the corrector that works, and the right velocity to
  extrapolate is Muon's orthogonalized update.** Evaluating the gradient at
  predicted weights `Ŵ = w − τ·lr·ΔW` cuts the tax to 1.23×. Predicting along the
  *raw momentum* (before Newton-Schulz) instead is worse than no correction at
  all. DC-ASGD-style curvature correction recovers ≤3% and is inert on Muon.
- **A v6e/DCN throughput model turns the 1.16× tax into a net 1.08–1.73× useful
  speedup** for the multi-slice batch sizes where cross-slice gradient traffic is
  exposed (DCN comm/compute ≥ 0.5). It is a wash or a loss when training is deeply
  compute-bound.

These are simulation results on a 130M-class MoE; we have not yet run a real
pipeline schedule. The harness is in
[`experiments/grug/moe_delay/`](https://github.com/marin-community/marin/tree/main/experiments/grug/moe_delay).

## Why this question

Synchronous pipeline schedules exist and are bit-exact to single-device training.
Zero-Bubble (Qi et al. 2023) and 1F1B/PipeDream-Flush (Narayanan et al. 2021) pay
only a pipeline *bubble*, not a staleness cost, and are what Megatron-LM and
TorchTitan ship. If a synchronous schedule is fast enough, there is no staleness
problem to solve.

The reason to look past them is communication. Synchronous data/replica
parallelism all-reduces the gradients across slices, with traffic proportional to
the parameter count — large for an MoE, where every expert parameter is reduced.
Pipeline parallelism instead passes only stage-boundary activations, with traffic
proportional to `batch × hidden`, independent of depth or parameter count. When
cross-slice (DCN) bandwidth is the bottleneck, the asynchronous schedules that
remove the bubble become attractive — and they are the ones that introduce
staleness. So the question is whether the throughput win pays for the staleness
cost.

## Setup

**Model.** A small Mixture-of-Experts decoder from our `grug` training template:
hidden 512, 64 experts, top-4 routing, 6 layers, ~14M active / ~290M total
parameters, sequence length 2048, trained on a Nemotron-CC-centric mixture. It
fits on one v6e-8. Small enough for fast iteration, large enough to be sharded
across the model axis (which matters below).

**Delay injection.** The `grug` training step is a single `jax.jit`'d function. We
wrap the optimizer so it keeps a depth-τ FIFO of past gradients (and the weights
they were computed at) in the optimizer state, and feeds the inner optimizer the
stale gradient each step. At τ=0 this is bit-identical to the unwrapped optimizer.
This reproduces constant-delay asynchronous SGD exactly, and the FIFO plus any
correction statistics are the "O(weights) extra optimizer state" we are budgeting
for. No pipeline parallelism is built.

**Per-stage delay.** A pipeline does not apply one global delay. Splitting the
model into P stages, an asynchronous/1F1B schedule applies stage `s`'s gradient
`(P−1−s)` steps late: the last stage is fresh, the first stage (input embedding)
is stalest. We inject this with a per-parameter delay map (block `i` → stage
`(i·P)//L`, delay `(P−1)−stage`), not a single τ. The `pp6` configuration below is
this profile with one stage per layer.

**Metric.** Token overhead: a delayed run reaches some final loss at step `s`; the
synchronous run reaches that same loss at step `s₀`; the overhead is `s/s₀`
(tokens = steps × a fixed batch). It answers "how many extra tokens to reach the
same loss," which is what feeds the throughput trade-off. We also report the raw
final-loss gap to synchronous training at a fixed step budget.

## How stale is pipeline parallelism, really?

**Takeaway: the per-stage delay profile changes the throughput verdict; modeling
the pipeline as one global delay overstates the cost 2–3×.**

On a 6k-step iso-loss cohort, the faithful 6-stage profile (`pp6`) plateaus +0.086
above synchronous Muon. A uniform τ=5 delay — the same model treated as one global
delay — plateaus +0.282, 3.3× worse.

| staleness model | final-loss gap to sync | token overhead |
|---|---|---|
| per-stage `pp6` (faithful) | +0.086 | 1.33× |
| uniform τ=5 | +0.282 | 2.42× |

The difference is that a real pipeline keeps the late layers fresh, and the late
layers carry most of the loss reduction. Studies that model pipeline staleness as
a single delay (the natural choice when you have not separated the stages) roughly
double-to-triple the apparent cost. This is what flips the throughput verdict
below from "never worth it" to "worth it where the communication is exposed."

## The cost of staleness

**Takeaway: Muon degrades smoothly with delay and far less than Adam, and the gap
is a token tax that shrinks with budget.**

Uncorrected, Muon's loss gap to synchronous training grows roughly linearly in the
delay, with no cliff:

| τ | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Muon gap | +0.14 | +0.35 | +0.85 | +1.50 | +2.57 |
| Adam (`adamh`) gap | +0.74 | — | +2.21 | — | — |

The per-step slope (gap/τ ≈ 0.07–0.11) is roughly constant, so the cost is a
forecastable function of pipeline depth. Muon's τ=8 gap reproduces to ~1% across
two seeds (+0.850 / +0.873); Adam's is both larger and noisier (+2.21 / +1.90).

Reframed as a token tax on the realistic `pp6` profile, Muon needs 1.33× the
tokens of synchronous training to reach matched loss at 6k steps; Adam needs
2.44×. The tax shrinks with budget: from 6k to 15k steps the uncorrected Muon
overhead falls 1.33× → 1.16× (gap +0.086 → +0.050), and both runs are still
descending at 15k. These are slow-catch-up trajectories, not converged quality
floors — the delayed run is behind, not permanently worse.

## Cheap correction: what works, what does not

**Takeaway: weight prediction is the only corrector that helps Muon, and it must
extrapolate the orthogonalized update — not the raw momentum.**

We tested two O(weights) corrector families from the asynchronous-training
literature.

**DC-ASGD curvature correction (Zheng et al. 2017) is dead here.** The method adds
`λ·(g⊙g)·Δw` to the stale gradient, using the squared gradient as a diagonal
Hessian proxy. The appealing "near-free" version reuses Adam's second moment `v_t`
as that curvature term. It recovers ≤3% of Adam's τ=8 gap at its best λ, is worse
with the EMA `v_t` than with the instantaneous `g²`, and is completely inert on
Muon — Newton-Schulz renormalizes the magnitude that the correction adjusts.

**Weight prediction works, and it is Muon-specific.** Instead of patching the
stale gradient, evaluate it at predicted weights `Ŵ = w − τ·lr·ΔW`, extrapolating
the most recent applied update forward by the delay (SpecTrain, Chen et al. 2018;
XPipe, Guan et al. 2019). For Muon, `ΔW` is the orthogonalized update. This closes
~46% of Muon's τ=8 gap (+0.850 → +0.471, reproducible to ~1% across seeds) and
cuts the `pp6` token tax from 1.33× to 1.23×. The same correction does not
reliably help Adam (0–14%, within seed noise): Muon's orthogonalized update is a
clean, well-scaled velocity to extrapolate, and Adam's raw adaptive step is not.

### Pre- vs post-orthogonalization prediction

Muon maintains a momentum buffer `M`, orthogonalizes it with Newton-Schulz, and
steps along `−lr·NS(M)`. The weight predictor above extrapolates `NS(M)` (the
applied, post-orthogonalization update). An alternative is to extrapolate the raw
momentum `M` (pre-orthogonalization), on the hypothesis that `M` is smoother
step-to-step than its orthogonalized image and so a lower-variance predictor of
where the weights are drifting. We built that variant as a pipeline-specific Muon
optimizer — an EMA of the stale gradient kept in the wrapper, pointed in the
descent direction and rescaled to the realized step size — plus three Muon-aware
ways to gate or clamp it: a Cautious-Optimizer sign gate (Liang et al. 2024,
`wp_cautious`), a LARS-style per-leaf trust-ratio clamp (You et al. 2017,
`wp_trust`), and a soft cosine-agreement gate (`wp_confidence`).

The hypothesis is wrong. Predicting along the raw momentum is worse than
predicting along the orthogonalized update, and worse than applying no correction.

| corrector (per-stage `pp6`, 6k steps) | final-loss gap | token overhead |
|---|---|---|
| post-orth `weight_pred` (extrapolate `NS(M)`) | +0.067 | **1.23×** |
| `wp_trust` (raw `M` + trust-ratio clamp) | +0.076 | 1.29× |
| `wp_confidence` (raw `M` + cosine gate) | +0.078 | 1.29× |
| uncorrected `pp6` | +0.086 | 1.33× |
| `wp_preorth` (raw `M`, unclamped) | +0.097 | 1.41× |
| `wp_cautious` (raw `M` + sign gate) | +0.098 | 1.41× |

The mechanism: the weights move along `−lr·NS(M)`, and Newton-Schulz substantially
rotates the direction relative to `M`. Extrapolating `M` therefore predicts
weights the optimizer will not visit, and the forward evaluation lands at a worse
point than no prediction. The two variants that recover to just below the
uncorrected baseline — the trust clamp and the cosine gate — do so by suppressing
the prediction toward zero. The hard per-coordinate sign gate does not help at
all, which says the failure is a whole-direction rotation, not a handful of
wrong-sign coordinates. The right thing to do with a raw-momentum prediction is to
use less of it; the right velocity to extrapolate is the orthogonalized update.

These six arms are single-seed (seed 0). The post-orth-vs-raw-momentum gap (0.067
vs 0.076–0.098) is large relative to the ~1% seed reproducibility we measured for
Muon weight prediction; the `wp_trust`/`wp_confidence` near-tie is within plausible
seed noise.

## Throughput verdict

**Takeaway: at the 1.16× converged tax, asynchronous pipeline parallelism is net
positive once cross-slice gradient traffic is exposed, and a wash when training is
deeply compute-bound.**

A parametric v6e/DCN model
([`pp_throughput_model.py`](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_delay/pp_throughput_model.py))
compares synchronous data parallelism (all-reduce gradients, volume ∝ N_total)
against pipeline parallelism (pass stage-boundary activations, volume ∝
batch·hidden). The decisive quantity is the DCN comm/compute ratio. For a 300B-total
/ 20B-active MoE over 8 slices, the gradient all-reduce hides under compute at
large batch (batch 8M, ratio 0.25 — pipeline parallelism is moot), and is exposed
at smaller batch. Applying the converged 1.16× token tax:

| per-step batch | DCN comm/compute | net useful speedup of async PP |
|---|---|---|
| 8M | 0.25 | ~1.0× (compute-bound; comm hides) |
| 4M | 0.5 | 1.08× |
| 2M | 1.0 | 1.30× |
| 1M | 2.0 | 1.73× |

The break-even token overhead at batch 2M is 1.50×: the converged 1.16× clears it,
and the uniform-τ 2.42× would fail it. The per-stage staleness model is what makes
the difference between adopting and rejecting asynchronous pipeline parallelism in
this regime.

## Limitations and next steps

- **This is a software simulation.** We inject delay into a synchronous step; we
  have not run a real pipeline schedule. The simulation reproduces constant-delay
  async SGD exactly, but a real schedule also changes microbatch statistics and
  numerics. The next step is a minimal real pipeline schedule to confirm the
  simulator predicts reality (tracked as a follow-up).
- **One model size.** All results are on the 130M-class MoE. The throughput model
  extrapolates to a 300B/20B MoE, but the staleness measurements do not — the tax
  could move with depth, width, or expert count.
- **The corrector sweep is single-seed.** The qualitative finding
  (raw-momentum prediction is worse than orthogonalized-update prediction) is
  robust to the seed noise we measured; the fine ranking among the raw-momentum
  variants is not.
- **Long-horizon prediction degrades.** Constant-velocity weight prediction fades
  as the delay grows (53% / 45% / 19% of the gap closed at τ=2 / 8 / 16).
  Realistic per-stage delays (τ ≈ 1–8) sit where it works best, but deeper
  pipelines would need a better predictor.

## References

- Jordan et al. 2024, Muon optimizer.
- Qi et al. 2023, Zero Bubble Pipeline Parallelism.
- Narayanan et al. 2021, Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW).
- Zheng et al. 2017, Asynchronous Stochastic Gradient Descent with Delay Compensation (DC-ASGD).
- Chen et al. 2018, Efficient and Robust Parallel DNN Training through Model Parallelism on Multi-GPU Platform (SpecTrain).
- Guan et al. 2019, XPipe: Efficient Pipeline Model Parallelism for Multi-GPU DNN Training.
- Liang et al. 2024, Cautious Optimizers.
- You et al. 2017, Large Batch Training of Convolutional Networks (LARS).
- Full experiment log, tables, and per-run links: [tracking issue #6431](https://github.com/marin-community/marin/issues/6431).
