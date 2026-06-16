# Delayed-Gradient Pipeline Parallelism for Grug MoE — Experimental Strategy

**Status:** proposal / research plan
**Date:** 2026-06-16
**Weaver issue:** #191
**Owner:** (unassigned)
**Testbed:** `experiments/grug/moe/` — d512 MoE compute-optimal baseline on TPU v6e-4

---

## TL;DR

We want to know whether pipeline parallelism (PP) is worth adopting for Grug MoE
training, and — if we use a *throughput-optimal* PP schedule — whether we can
recover synchronous-SGD quality under the resulting **delayed/stale gradients**
cheaply, using only **O(weights)** extra optimizer state, with **Muon** as the
base optimizer.

Three findings reframe the work:

1. **There is a synchronous escape hatch.** Zero-Bubble PP (ZB-1p/ZB-2p/ZBV) and
   1F1B/PipeDream-Flush are **bit-exact to synchronous SGD** — they pay only a
   (near-zero) *bubble* cost, not a staleness cost, and they are what Megatron-LM
   and TorchTitan actually ship. If we adopt one of these, the entire
   delayed-gradient problem disappears. **So the first question is empirical: does
   async buy enough throughput in our v6e/DCN-bound regime to justify any staleness
   R&D at all?**

2. **We can answer the staleness questions without building PP.** The grug
   `train_step` is a single `jax.jit`'d function (`experiments/grug/moe/train.py:326-329`).
   We can inject a controlled gradient delay τ in software — a FIFO of past
   gradients (and weight snapshots) carried in `GrugTrainState` — which exactly
   reproduces constant-delay async SGD. **This decouples the optimizer research
   from the heavy PP engineering** and lets us run the whole study on a single
   v6e-4 in fast, cheap iterations.

3. **Muon-under-delay is an unexplored void.** There is *no* published work on
   Muon (or any orthogonalized/spectral-norm optimizer) under pipeline staleness.
   The only adjacent evidence — MuLoCo/Dion — is delayed-*communication*
   (DiLoCo-style), a different regime, and the specific claims that "Muon is more
   delay-robust than Adam" were **refuted** in our verification pass. This is
   genuinely novel research with a clear mechanism hypothesis (below) and a cheap
   testbed.

The two realistically-implementable O(W) corrector families from the literature are
**weight prediction / extrapolation** (SpecTrain, XPipe, PipeOptim) and
**curvature correction** (DC-ASGD). Both reuse statistics an Adam/Muon trainer
already (almost) has. **The headline Muon-specific idea is to make the
weight-prediction increment ΔW be Muon's orthogonalized momentum step.** Every
method here is validated only at small/CNN or ≤12-layer-transformer scale, so all
are "promising but unvalidated at LLM/MoE scale" — exactly the gap this plan
closes.

---

## 1. Motivation and the strategic fork

PP is attractive for Grug in two regimes we care about:

- **Cross-slice / DCN-bound:** the grug mesh already has a `replica_dcn` axis for
  multi-slice replication (`lib/levanter/src/levanter/grug/sharding.py:109`). PP
  over DCN sends only thin stage-boundary activations instead of replicating
  gradients, which can beat FSDP/replication when inter-slice bandwidth is the
  bottleneck.
- **Memory-bound:** PP shards layers across devices, lowering per-device weight +
  optimizer-state footprint — relevant as we scale total params (the E=64
  recipe's total is ~8× active).

But PP forces a fork:

```
                         ┌── SYNCHRONOUS schedule (ZB / 1F1B) ── bit-exact to sync SGD,
                         │     pay a bubble. NO optimizer change needed.
  Adopt PP for Grug? ───┤
                         └── ASYNC schedule (2BW / PipeMare / XPipe) ── higher
                               throughput, but gradients are τ steps stale.
                               Needs delay tolerance OR O(W) correction.
```

The naive instinct is to build PP, discover staleness hurts, then bolt on a fix.
That is backwards and expensive. Instead:

> **Measure the staleness sensitivity of the *actual Grug MoE recipe* with a
> software delay simulator first. The result tells us whether we need async at
> all, and if so which corrector is worth implementing — before we write a line
> of pipeline scheduling code.**

---

## 2. Where Marin is today (codebase reality)

| Capability | Status | Evidence |
|---|---|---|
| Data / FSDP / tensor / expert parallelism | present (SPMD) | `lib/levanter/src/levanter/utils/mesh.py`, `grug/sharding.py:109` |
| Sequence/context parallelism | present | `ResourceAxis.CONTEXT`, `gpt2_small_fast_context.yaml` |
| **Pipeline parallelism** | **absent** | `lib/haliax/src/haliax/nn/scan.py:449` — `Stacked` has a TODO: *"we can probably make this module support pipeline parallelism, but that's a whole project in itself"* |
| Gradient accumulation / microbatching | present, **fully synchronous** | `lib/levanter/src/levanter/grad_accum.py` (`microbatched()` folds over microbatches, single optimizer step) |
| Any async / delayed / stale-gradient handling | **absent** | grep of `lib/levanter/` finds nothing |
| Muon (raw-array grug variant) | present | `lib/levanter/src/levanter/optim/grugmuon.py` (`GrugMuonConfig`) |
| Grug MoE optimizer | `GrugMoeAdamHConfig` (AdamH, 3 param groups) | `experiments/grug/moe/optimizer.py` |

Key structural facts we exploit:

- **The grug train loop is hand-rolled, not Levanter's `Trainer`.** The single
  jit'd `train_step` (`experiments/grug/moe/train.py:305-366`) computes grads and
  applies the optimizer in one place — a clean seam for delay injection.
- **`GrugTrainState` already carries an optional O(W) buffer** (`ema_params`,
  `train.py:249`). Adding a `grad_fifo` / `weight_snapshot` field is an
  established pattern, and its sharding is automatic (state is resharded to the
  parameter axis mapping each step).
- **Muon keeps a *single* per-parameter buffer** (`ScaleByMuonState.momentum_buffer`).
  It has no second moment. Correctors are added as extra `optax.GradientTransformation`s
  in the `muon`/`adamw` chains of `GrugMuonConfig.build` (`grugmuon.py:82-121`).
- **Embeddings are untied** (`token_embed` + `output_proj`, `model.py:492/495`)
  and route to AdamW; the ~110M of 2D/3D backbone + expert matrices route to
  Muon. **O(W) delay-correction buffers only need to cover the Muon path.**

---

## 3. Research synthesis

*(Full deep-research report with adversarial verification: 20 primary sources,
94 claims extracted, 25 verified, 22 confirmed / 3 refuted. References in §10.
All efficacy numbers below are from the cited primary sources unless noted.)*

### 3.1 Pipeline schedules and their staleness depth τ

| Schedule | Synchronous? | Staleness τ | Production use |
|---|---|---|---|
| GPipe (sync flush) | ✅ bit-exact | 0 (bubble cost) | legacy |
| 1F1B / PipeDream-Flush | ✅ bit-exact | 0 | **Megatron-LM, TorchTitan** |
| Interleaved 1F1B | ✅ bit-exact | 0 | **Megatron-LM** |
| **Zero-Bubble (ZB-1p/2p/ZBV)** | ✅ **bit-exact** (loss verified bit-identical to 1F1B) | 0 | Megatron-style; splits backward into B (input-grad) + W (weight-grad) to fill bubbles |
| PipeDream (async, weight-stashing) | ❌ | τ = D−rank−1 (max D−1 at stage 0, 0 at last) | research |
| PipeDream-2BW (double-buffered) | ❌ | τ ≈ 1 (uniform) | research |
| PipeMare | ❌ | τ_fwd ∝ (P−i), τ_bkwd = 0 (fwd/bkwd discrepancy) | research |
| XPipe / SpecTrain / PipeOptim | ❌ | s = ⌊k/2⌋+N−k−1 (fwd) etc. | research |

**Takeaways:** (a) the throughput-optimal *synchronous* schedules are bit-exact —
verified empirically by the Zero-Bubble authors recording bit-identical losses vs
1F1B. (b) Async staleness **grows linearly with pipeline depth and is largest at
the earliest stages** — so any correction must be **stage-dependent** (this
validates a per-layer staleness age τ as a real signal, not a gimmick). (c)
PipeDream-2BW is effectively the **τ=1 uniform-delay** case; deeper interleaving
pushes τ up.

### 3.2 The quality gap vs sync SGD

- At **small transformer scale**, weight prediction essentially closes the gap:
  SpecTrain reports ~0 accuracy gap on a small Transformer (60.14% vs 60.33%
  data-parallel) and PipeMare recovers near-baseline BLEU on a **12-layer NMT
  Transformer** (34.1/27.0 vs GPipe 34.5/27.5; with warmup 34.5/27.8) **where
  naive PipeDream-style fine-grained pipelining diverges to BLEU 0.0.**
- **Single-source optimism caveat:** SpecTrain's "no gap" is self-reported; the
  independent XPipe reproduction found SpecTrain *does* drop ~0.5% and that
  PipeDream is better than SpecTrain claimed. Treat all "matches sync SGD" numbers
  as **upper bounds**.
- **Memory:** PipeMare weight+optimizer memory is ~1.25× baseline vs PipeDream
  weight-stashing's 2.06–2.39×.
- **When staleness washes out:** large batch + small LR + late training (small
  steps, slowly-moving weights) → staleness matters less; early training and
  high-curvature directions → staleness compounds. **This is exactly what our
  delay-injection sweep measures directly.**

### 3.3 O(W) delayed-gradient correction families (ranked)

| Method | Extra state | Statistic maintained | Efficacy | Complexity | Scale validated |
|---|---|---|---|---|---|
| **Weight prediction** (PipeOptim/XPipe/SpecTrain) | ≤2 weight copies; **no new per-param stats** | reuses optimizer's own ΔW (momentum / Adam m̂,v̂) | best; ~0 gap at small transformer scale | low–med | CNN + small Transformer |
| **DC-ASGD** (diagonal-Hessian) | 1 stale-weight copy; curvature = λ·g⊙g | g⊙g (≈ Adam v_t) + the stale weight | recovers near-seq SGD on CNNs (CIFAR ASGD 9.27→8.19% vs seq 8.65%) | med | **CNN-only** |
| **Gap-aware / staleness-scaled LR** | scalar(s) / per-layer τ | τ | partial; cheap damping | trivial | mixed |
| **PipeDream weight stashing** | up to D weight copies (**> O(W)**) | exact fwd/bkwd weight match | exact but expensive | med | small |
| **PipeDream-2BW** | 2 weight copies | none (tolerates τ=1) | "washes out" bet | med | GPT-scale (throughput, not a controlled quality A/B) |
| **TiMePReSt** (tolerate, no correction) | none | none | concedes lower statistical efficiency (more steps) | low | CNN-only |
| **Error-feedback / cautious masking** | 0–1 buffer | sign agreement of stale update vs current momentum | untested under PP staleness | low | — |

**Weight prediction is the leading contender** precisely because it adds **no new
per-parameter statistics** — it extrapolates the future weights a gradient will
land on using the optimizer's *own* update increment:

```
PipeOptim:  Ŵ_{t+s} ≈ W_t − lr·s·ΔW_t,   ΔW = m̂/(√v̂+ε)  (Adam)  or  momentum (SGDM)
XPipe:      Ŵ_t     = W_t − s·lr·ΔW_t,    ΔW = bias-corrected Adam moments
SpecTrain:  Ŵ_{t+s} = W_t − s·η·v_{t−1}   (momentum velocity only)
```

PipeOptim/XPipe explicitly **generalize the predictor to the active optimizer's
update rule** — *which is the exact seam a Muon variant plugs into.*

### 3.4 Curvature / second-moment state as a (near-free) corrector

DC-ASGD's correction:

```
w_{t+τ+1} = w_{t+τ} − η·( g(w_t) + λ·g(w_t)⊙g(w_t)⊙(w_{t+τ} − w_t) )
```

The curvature term `λ·g⊙g` is **structurally identical to Adam/RMSProp's
pre-bias-correction second moment v_t**. DC-ASGD-a even normalizes λ by
`√(MeanSquare+ε)` with an EMA (m=0.95) of g² — i.e. RMSProp. **Implication:** an
Adam-based trainer's existing v_t can double as the delay-compensation curvature
term at near-zero extra cost. **No published work has done this for a
preconditioned optimizer** (open question). For Muon, which has no v_t, we either
maintain a *shadow* second moment used only for correction, or borrow Dion's
power-iteration preconditioner.

### 3.5 Muon under delay — the void

- **No paper** addresses Muon / Scion / orthogonalized optimizers under pipeline
  staleness.
- MuLoCo (Muon-in-DiLoCo, scaled to 15B) and Dion (distributed orthonormalization,
  validated 160M–3B, 2–3× over AdamW) are **delayed-communication** (periodic
  averaging) — a different delay regime. **The claims that Muon's pseudo-gradients
  stay more aligned than AdamW as delay grows were refuted 0-3.** Do **not** build
  on a "Muon is naturally staleness-tolerant" premise.
- **Mechanism hypothesis (ours, testable):** Newton-Schulz orthogonalization
  renormalizes singular values to ≈1, so the Muon update's *magnitude* is
  decoupled from the stale gradient's *scale*. A stale momentum buffer that is
  wrong in norm but roughly right in *subspace* yields a nearly-correct Muon
  update. The error that *survives* orthogonalization is the **angular / subspace
  drift** of the momentum buffer, not its magnitude. **Prediction:** Muon degrades
  gracefully under *scale* staleness but is sensitive to *directional* staleness;
  the right corrector therefore targets the directional component, and the natural
  weight-predictor is `Ŵ_{t+s} = W_t − s·lr·ΔW_muon` with `ΔW_muon` the
  orthogonalized momentum step. This is the headline experiment.

---

## 4. The plan: a software delay-injection testbed

**Central de-risking move.** Build a delay simulator inside the existing
synchronous grug loop. No pipeline scheduling, no activation send/recv — just a
gradient (and weight-snapshot) FIFO in the train state.

### 4.1 Faithful constant-delay model

At step *t* with current weights `w_t`:

1. Compute `g_t = ∇loss(w_t)` (unchanged).
2. Push `(g_t, w_t)` into a depth-τ FIFO.
3. Pop `(g_{t−τ}, w_{t−τ})` and feed the **stale** gradient to the optimizer:
   `updates, opt_state = optimizer.update(g_{t−τ}, opt_state, w_t)`.
4. `w_{t+1} = w_t + updates`.

This exactly reproduces constant-delay-τ async SGD (the PipeDream-2BW case is
τ=1). Cost: O(τ·W) for the gradient ring + O(τ·W) for weight snapshots. For τ≤8
on the ~110M Muon-path that is ≤3.5 GB fp32 — trivial on v6e-4 (128 GB).

### 4.2 Staleness profiles (knobs)

- **Uniform τ** ∈ {0, 1, 2, 4, 8, 16} — models 2BW (τ=1) through deep interleaving.
  τ=0 is the synchronous control.
- **Per-layer τ** = `D − stage(layer) − 1` — models real async-1F1B stage-dependent
  staleness (validated formula §3.1). Implemented by bucketing grug's scanned
  layers into P pipeline stages and delaying each stage's gradient slice
  accordingly. This is the realistic PP emulation and the testbed for
  stage-dependent corrections.
- **Forward/backward discrepancy** (PipeMare-style τ_bkwd=0) — optional, lower
  priority.

### 4.3 What the testbed measures (cheaply, before any PP)

- **Staleness tolerance curve:** Paloma macro / train loss vs τ at the d512 MoE
  compute-optimal point — *how much does staleness actually cost Grug?*
- **Correction efficacy:** the same curve with each O(W) corrector enabled — *which
  corrector closes the gap, and by how much, per unit of extra state?*
- **Muon vs Adam degradation:** update-direction cosine vs the synchronous update
  as a function of τ, separated into magnitude vs angular error — *tests the §3.5
  mechanism hypothesis directly.*
- **Go/no-go input for PP:** combined with a throughput model (bubble cost of sync
  ZB/1F1B vs async on v6e DCN), the tolerance curve tells us whether async + a
  corrector beats just-pay-the-bubble.

---

## 5. The O(W) delayed-Muon optimizer-state design

We layer correctors as `optax.GradientTransformation`s on top of `grug_muon`
(`grugmuon.py:82-121`). Each maintains state structurally identical to the params,
sharded automatically. The corrector consumes `(stale_grad, current_params,
stale_weight_snapshot, τ)` and emits a corrected gradient/update.

### 5.1 Statistic menu → mechanism → cost

The user's proposed statistics map onto concrete, literature-grounded mechanisms:

| Statistic (O(W) unless noted) | Mechanism it enables | Cost | Grounding |
|---|---|---|---|
| **stale age τ** (scalar or per-layer) | stage-dependent correction strength; staleness-scaled LR | O(1)–O(layers) | τ formula §3.1 |
| **EMA of g²** (shadow second moment) | DC-ASGD curvature term λ·g⊙g; Muon lacks v_t so we add it *only* for correction | 1×O(W) | §3.4 |
| **stale weight snapshot w_{t−τ}** | ΔW = w_t − w_{t−τ} for DC-ASGD Taylor term | τ×O(W) | DC-ASGD |
| **structured update direction ΔW_muon** | weight-prediction extrapolation `Ŵ = w − s·lr·ΔW_muon` (**headline**) | reuse momentum (0 new) | PipeOptim/XPipe |
| **EMA of ΔW²** (per-param sensitivity) | damp stale updates on fast-moving / high-sensitivity params; alt curvature proxy | 1×O(W) | novel; sensitivity proxy |
| **curvature / Hessian-diag proxy** | g⊙g, or secant `(g_t−g_{t−1})/(w_t−w_{t−1})` (Hessian-vector-free) | 1–2×O(W) | DC-ASGD / secant |

### 5.2 "What else could we throw at it" — additional O(W) signals

Beyond the above, candidates worth a cheap ablation (none tested under PP staleness
in the literature — all open):

- **Predicted-vs-actual update discrepancy** (PipeMare-flavored): store the
  predicted ΔW, compare to the realized one next step, use the residual to adapt
  correction strength or trust. 1×O(W).
- **Cautious sign-agreement mask** (Levanter already has `optim/cautious.py`): zero
  the components of the stale update whose sign disagrees with the *current*
  momentum sign — kills the most-wrong stale components nearly for free.
- **LARS/LAMB trust ratio under delay:** per-layer `‖w‖/‖corrected update‖` to bound
  the step when staleness inflates the update norm. O(layers).
- **Per-parameter gradient SNR:** EMA of g and g² → `SNR = |E[g]|/√Var[g]`. Low-SNR
  directions are noise-dominated (staleness ≈ harmless); high-SNR directions are
  where stale errors compound → correct selectively. 2×O(W).
- **Momentum half-life vs τ coupling:** Muon's momentum buffer already mixes
  gradients across steps; under large τ, shortening the momentum half-life (or
  Nesterov look-ahead) is a zero-extra-state lever worth sweeping.

### 5.3 Concrete correctors to implement (in priority order)

1. **`muon_weight_prediction`** — apply `grug_muon` to the stale grad, then
   *un-apply* / *pre-apply* `s·lr·ΔW_muon` so the orthogonalized step targets the
   predicted future weights. **Zero new per-param state.** The headline,
   Muon-native, literature-leading method. (Implementation note: because Muon's ΔW
   is the post-orthogonalization step, the predictor must extrapolate in *weight*
   space using that step — this is the precise novelty the research flags as
   untested.)
2. **`dc_asgd_correction`** — Taylor term `λ·v_t⊙(w_t − w_{t−τ})` reusing a shadow
   EMA-of-g² as v_t and the stale-weight snapshot. **Tests the "curvature reuse =
   near-free correction" open question.** Pairs with Adam-path params for free
   (AdamH already has v_t).
3. **`staleness_lr_scale`** — per-layer τ-dependent step damping; trivial baseline
   and a strong control (does cheap damping alone recover most of the gap?).
4. **`cautious_stale_mask`** — sign-agreement masking of the stale update.
5. (stretch) **secant diagonal curvature**, **SNR-gated correction**.

The **pre- vs post-orthogonalization** placement of the correction is itself an
ablation: correcting the *momentum buffer before* Newton-Schulz (fixing direction,
letting orthogonalization renormalize) vs correcting the *update after*.

---

## 6. Experiment matrix and phasing

Aligned to the grug MoE **gate-1/gate-2** promotion methodology
(`experiments/grug/moe/README.md`, `agent.md`): a change promotes when effective
speedup > 1 at the compute-optimal baselines and projected macro improves at 1e21
/ 1e23. Here the bar is different — **recover sync-SGD quality under delay** — so we
report the *gap to the τ=0 control*, not absolute speedup.

**Testbed:** d512 MoE compute-optimal baseline (E64/K4, 6 layers, ~14M active /
~110M Muon-path / ~240M total incl. untied embeddings), macro 3.81 @ 8.37e8 tokens,
~0.6 h on v5p-8 → comfortably a single v6e-4 run. An E16/K2 trim is available for
sub-15-min smoke iterations. **Base optimizer: `grug_muon`** (the focus), with
`grug_moe_adamh_v2` as the Adam comparison arm.

| Phase | Question | Runs | Exit criterion |
|---|---|---|---|
| **P0 — Throughput reality check** | Does async beat sync ZB/1F1B on v6e DCN at all? | analytic + microbench (no training) | bubble-cost vs async-gain estimate for d512 and one larger point |
| **P1 — Delay simulator** | Build & validate the FIFO injector; τ=0 reproduces baseline bit-for-bit | 1 control + harness tests | τ=0 matches current baseline loss curve |
| **P2 — Staleness tolerance curve** | How much does staleness cost Grug MoE? | τ ∈ {0,1,2,4,8,16}, uniform + per-layer, Muon & Adam | macro-vs-τ curve; identify the τ where gap > noise |
| **P3 — Muon weight prediction** | Does `Ŵ=w−s·lr·ΔW_muon` recover sync quality? (headline) | corrector × τ-sweep × {pre,post}-orth | gap to τ=0 closed to within run-to-run noise |
| **P4 — Curvature reuse (DC-ASGD)** | Can v_t double as the curvature term near-free? | DC-ASGD on Adam-path (free v_t) + shadow-v_t Muon | quantify gap-closure per unit extra state |
| **P5 — Cheap-control ablations** | Does τ-scaled LR / cautious masking alone suffice? | staleness_lr_scale, cautious_stale_mask | rank correctors by gap-closed ÷ extra-state |
| **P6 — Extra-stat ablations** | Do EMA-ΔW², SNR-gating, secant curvature add anything? | one-factor sweeps | keep only signals that beat P3/P4 |
| **P7 — Real-PP spike** (conditional on P0/P2) | Stand up a minimal PP schedule and confirm the simulator predicted reality | 1–2 engineering spikes | measured PP loss tracks simulated τ-curve |

P0–P6 run entirely on the software simulator (cheap, single v6e-4). P7 is gated on
P0 (async worth it) **and** P2/P3 (a corrector that works) — we only pay the
`Stacked` PP engineering cost once the optimizer answer is in hand.

---

## 7. Metrics and success criteria

- **Primary:** Paloma macro (and train loss) **gap to the τ=0 synchronous
  control** at the d512 compute-optimal point. A corrector "works" if it closes the
  gap to within run-to-run seed noise (estimate noise with ≥2 seeds of the control).
- **Diagnostic:** update-direction cosine vs sync, decomposed into magnitude vs
  angular error, as a function of τ (tests the Muon mechanism hypothesis).
- **Cost-normalized:** gap-closed ÷ bytes-of-extra-state — the whole point is *O(W)*
  efficiency, so a method that needs 3×O(W) must beat one needing 0.
- **Robustness:** does the corrector hold across the under-/over-trained regimes
  (low isoflop-curve curvature, per the MoE README promotion criteria)?
- **Go/no-go:** P0 throughput model × P2 tolerance curve → a single recommendation:
  *sync-ZB*, *async + corrector X*, or *PP not worth it for Grug yet*.

---

## 8. Risks, caveats, and open questions

- **Scale caveat (dominant).** Every corrector in the literature tops out at
  CNN or ≤12-layer transformer scale; none on a modern decoder LLM, none on MoE,
  none in JAX, none with Muon. Our results will be the first datapoint — treat
  external "matches sync" numbers as upper bounds.
- **The simulator is not PP.** It models *gradient delay* faithfully but not the
  forward/backward *weight-version discrepancy* of some schedules (PipeMare
  τ_bkwd=0), nor activation-staleness effects. P7 validates that the simulated
  τ-curve predicts real PP; until then, conclusions are about delayed gradients,
  not about a specific schedule.
- **Muon-robustness is unproven, not assumed.** The "more robust" claims were
  refuted; we test, not presume. If Muon turns out *more* sensitive to directional
  staleness, that is itself a publishable, decision-relevant result.
- **MoE-specific staleness.** Router/expert dynamics under delay are unstudied: a
  stale router bias (QB-β is already applied one step late, `train.py:309`) plus
  stale expert grads could interact. The testbed should log router balance vs τ.
- **Sync escape hatch may win.** If P0 shows ZB/1F1B's bubble is cheap in our
  regime, the correct answer is "implement synchronous PP, skip the staleness R&D."
  That is a perfectly good — and cheaper — outcome, and the plan is structured to
  surface it first.
- **Open questions carried from the research** (each maps to a phase): does Muon's
  magnitude-normalization help/hurt/wash out vs an Adam predictor (P3); can v_t
  serve as the DC-ASGD curvature term for a preconditioned optimizer (P4); is async
  even worth it on v6e DCN (P0); do the extra O(W) stats help under PP staleness
  (P6).

---

## 9. Issue breakdown

The phases map to GitHub issues (filed under this plan; labels `agent-generated`):

1. **Delay-injection testbed** — FIFO grad/weight-snapshot in `GrugTrainState`,
   uniform + per-layer τ, τ=0 bit-reproduces baseline. (P1)
2. **Staleness tolerance curve for Grug MoE** — macro-vs-τ at d512, Muon & Adam,
   uniform & per-layer. (P2)
3. **Muon weight-prediction corrector** (headline) — `Ŵ=w−s·lr·ΔW_muon`, pre/post
   orthogonalization ablation. (P3)
4. **DC-ASGD curvature reuse** — v_t-as-curvature, near-free correction; Adam-path
   free, shadow-v_t Muon. (P4)
5. **Cheap-control correctors** — τ-scaled LR damping + cautious stale-mask. (P5)
6. **Extra O(W) statistic ablations** — EMA-ΔW², SNR-gating, secant curvature. (P6)
7. **v6e PP throughput model** — bubble cost of sync ZB/1F1B vs async gain;
   go/no-go input. (P0)
8. **Real-PP validation spike** (conditional) — minimal schedule on `Stacked`,
   confirm simulator predicts reality. (P7)

---

## 10. References

Primary sources (adversarially verified; vote = refute-votes / 3):

- Zero-Bubble PP — arXiv:2401.10241 (bit-exactness verified empirically).
- Controllable-memory PP / ZBV — arXiv:2405.15362.
- PipeOptim (optimizer-dependent weight prediction) — arXiv:2312.00839 (IEEE TPDS 2025).
- XPipe (Adam weight prediction) — arXiv:1911.04610.
- SpecTrain (SGDM weight prediction) — arXiv:1809.02839.
- PipeMare (async PP, discrepancy correction) — arXiv:1910.05124 (MLSys 2021).
  *(Its strongest "matches sync" summary claim was refuted 1-2; the 12-layer NMT
  numbers stand.)*
- DC-ASGD (delay compensation, diagonal Hessian) — arXiv:1609.08326 (ICML 2017).
- DC-S3GD (decentralized DC-ASGD) — arXiv:1911.02516 (SC19).
- TiMePReSt (tolerate-the-mismatch baseline) — arXiv:2410.14312 (FGCS 2025).
- PipeDream-2BW — Narayanan et al., ICML 2021.
- MuLoCo (Muon-in-DiLoCo, to 15B) — arXiv:2505.23725. *(Muon-more-robust claims
  refuted 0-3; it is delayed-communication, not pipeline staleness.)*
- Dion (distributed orthonormalization) — arXiv:2504.05295.

Codebase anchors: `experiments/grug/moe/{train,model,optimizer,heuristic}.py`,
`experiments/grug/moe/README.md`, `lib/levanter/src/levanter/optim/{grugmuon,muon,cautious}.py`,
`lib/levanter/src/levanter/grug/sharding.py`, `lib/haliax/src/haliax/nn/scan.py:449`.
