# Delphi Midtraining Endpoint Prediction — Retrospective & Analysis

*Author: Claude (Opus 4.8), 2026-05-30. Working doc for Ahmed; written off the Ahmed/Kaiyue
Slack thread about the 1e22 extrapolation miss.*

This is the full record of what I investigated, what I built, the numbers, my reasoning, and
my opinionated take on what's actually going on and what to do next. It is deliberately long
and includes the dead-ends, because the dead-ends are informative.

---

## 0. TL;DR

- The 1e22 `math_val_loss` miss (10–20%) is **real, systematic, and one-directional**: every
  scaling form fit on the small ladder predicts a **higher** loss than the 1e22 runs actually
  achieve. The big runs *beat* the extrapolation.
- It is a **genuine acceleration**: the log-log slope steepens at the top of the ladder
  (~−0.10 → −0.12/−0.14/−0.15 for p67m33/p50m50/p33m67 over 1e21→1e22), and the steepening is
  **monotonically worse the more math is in the mix**.
- **It is not a missing irreducible-loss floor.** A Chinchilla floor bends the curve the wrong
  way and is the *worst* form (−15.8% at 1e22). This matches Ahmed's "Chinchilla 18% > log-log
  15%."
- **The best modeling-only fix** is a pooled law that separates model size `N` from math tokens
  `D_math` (`pooled_mechanistic`): **8.97% at 1e22 vs 10.70% baseline**, and near-unbiased at
  1e21. But it only helps modestly, and the residual is still mix-ordered.
- **No form fit on the small ladder closes the gap**, because the acceleration only manifests
  *at* 1e22 and is not identifiable from data below it.
- **Kaiyue's "predict the improvement" (Δ) idea, done naively, is worse (17%)** — the step-0
  baseline is inconsistent across the ladder and differencing amplifies error. It needs real
  control runs.
- **The single biggest threat to validity** (Section 9): the small ladder and the 1e21/1e22
  targets may not be the *same midtraining protocol* (fresh-warmup "false" midtrain vs
  resumed-mid-cooldown "true" midtrain). If so, part of the 1e22 "over-performance" is a
  protocol artifact, not scaling. **Verify this before trusting any extrapolation.**
- Practical horizon: the current power law is trustworthy to **~3× compute (through 1e21,
  ≤~2%)** and degrades to ~10% at **10× (1e22)**.

---

## 1. The question

From the thread: the new Delphi midtraining fits predict 1e21's final math validation loss
well (~1%) but miss 1e22 by 12–20%. The error **grows with the midtraining math fraction**
(p33m67 worst, p67m33 best), and the predictions are **over-pessimistic** (real < predicted).
Ahmed floated: train small models longer; maybe the same LR multiplier shouldn't be applied
across the ladder. Kaiyue asked whether we're doing log-linear without a Bayesian-optimal
(irreducible) loss, and suggested predicting the *improvement* from midtraining instead of the
absolute loss.

The goal I set: **find better ways to get predictability**, grounded in the data, and ship the
best modeling-only improvement.

---

## 2. The setup, decoded

**What's being predicted.** The final `math_val_loss` = `eval/nemotron_cc_math_v1/4plus/loss`
of a midtraining run, as a function of compute scale, data mix, and LR.

**The ladder.** Seven small "isoflop-bucket winner" scales `3e18 → 3e20` (12 cells each: 3
mixes × 4 LRs, all complete), plus held-out `1e21` (12/12) and `1e22` (11/12 observed; one
cell is an unfinished `best_prefix` run, excluded everywhere).

**Mix `pXXmYY`** = `XX`% web/pretrain-style tokens, `YY`% math tokens in the midtraining mix.
So `p33m67` is the most math-heavy, `p67m33` the least.

**LR** = a multiplier ∈ {0.33, 0.50, 0.67, 0.83} applied to a (nearly scale-invariant) base
peak LR (1e21 base ≈ 7.4e-3, 1e22 ≈ 7.2e-3).

**Per-scale model size and tokens** (from the canonical `experiments/delphi_models.py` HF repo
names; midtrain tokens = `K=0.20 × pretrain tokens`, the global Delphi rule):

| scale | N (params) | pretrain tokens | midtrain tokens | math tokens (p33m67) |
|---|---:|---:|---:|---:|
| 3e18 | 447M | 1.2B | 0.24B | 0.16B |
| 9e18 | 550M | 2.9B | 0.58B | 0.39B |
| 2e19 | 837M | 3.6B | 0.72B | 0.48B |
| 3e19 | 998M | 5.0B | 1.00B | 0.67B |
| 9e19 | 1.4B | 10.6B | 2.12B | 1.42B |
| 2e20 | 1.9B | 14.8B | 2.96B | 1.98B |
| 3e20 | 2.5B | 18.6B | 3.72B | 2.49B |
| **1e21** | **3.4B** | **46.3B** | **9.26B** | **6.2B** |
| **1e22** | **9.7B** | **160B** | **32.0B** | **21.4B** |

Note the scale of the jump: 1e22 has a **9.7B-param model trained on ~21B math tokens** in the
math-heavy mix, vs the small ladder's sub-2.5B params and sub-2.5B math tokens. The held-out
points are not a gentle step — they're a different regime.

**Schedule.** WSD: 10% warmup, ~70% stable, 20% linear decay to 0; cooldown fraction held
constant across scales.

**The canonical fit** (in `delphi_small_final_loss_scaling.py`) is a per-recipe log-log linear
law `L = A·C^α` on `scale_flops` (the pretrain-FLOPs label), no floor. A floor+power
(`E + A·C^-α`) exists only as a diagnostic.

---

## 3. Diagnosis (read-only, before writing any code)

I reproduced the miss and dissected it on the cached CSVs.

### 3.1 Reproduce the miss

Per-recipe single power law, fit on `3e18–3e20`, predict held-out:

| target | mean abs % | signed mean % | by mix (signed) |
|---|---:|---:|---|
| 1e21 | 1.06 | −0.99 | p67m33 +0.1, p50m50 −1.1, p33m67 −2.0 |
| 1e22 | 10.70 | −10.70 | p67m33 −6.1, p50m50 −11.1, p33m67 −15.0 |

One decade up (1e21) is excellent; two decades (1e22) is bad, **all negative** (pessimistic),
**ordered by math fraction**.

### 3.2 The smoking gun: the slope steepens

Local log-log slope between consecutive scales (avg over LR):

| mix (math frac) | …2e20→3e20 | 3e20→1e21 | **1e21→1e22** |
|---|---:|---:|---:|
| p67m33 (0.33) | −0.106 | −0.095 | **−0.120** |
| p50m50 (0.50) | −0.110 | −0.105 | **−0.136** |
| p33m67 (0.67) | −0.116 | −0.112 | **−0.149** |

The exponent is ~−0.10 across the ladder, then **accelerates** at the very top — the curve
bends *down* (improves faster), and **more math → more bend**. A single straight line in
log-log cannot represent this, so it under-predicts the high-end improvement → predicts too
high a loss.

### 3.3 Why it is NOT a floor (geometry)

A Chinchilla floor `E + A·C^-α` is **concave-up** in log-log: it *flattens* toward `log E` at
high compute. The data does the **opposite** (it steepens). So adding a floor pushes the
prediction *further* above the truth. Empirically confirmed: `per_recipe_floor` is the worst
form at 1e22 (−15.8%). This is exactly why Ahmed's Chinchilla fit (18%) was worse than log-log
(15%). **The missing ingredient is curvature/acceleration, not a floor.**

### 3.4 Why it is mix-ordered: it's a math-data effect

The bend grows with math fraction. The natural reading: the **math skill** is what's
accelerating. On the small ladder the math-token budget is tiny (0.16–2.5B), so the math
capability is data-starved and improves shallowly; by 1e22 the math-heavy mix has ~21B math
tokens and a 9.7B model — enough for the skill to "come online" and drop loss faster than the
starved small-ladder trajectory predicts.

---

## 4. What I tried (five forms, head-to-head)

I added five candidate endpoint forms and scored them on the **observed** held-out cells. All
fit on the small ladder only; the one unfinished 1e22 cell is excluded.

1. **`per_recipe_power`** — current canonical: one log-log line per (mix, LR). *Baseline.*
2. **`per_recipe_floor`** — Chinchilla `E + A·C^-α` per recipe. *(Kaiyue's "Bayesian-optimal
   loss" question — tested explicitly.)*
3. **`pooled_curvature`** — pooled log-quadratic in compute (per-recipe intercept + shared
   `b·logC + c·(logC)²`). A curve that can bend.
4. **`pooled_broken_power`** — smoothly-broken power law (BNSL-style) in compute, per-recipe
   amplitude, shared break + two slopes. The "robust extrapolator."
5. **`pooled_mechanistic`** — pooled `log L = b₀ + b₁·logN + b₂·log(D_math) + b₃·lr`. Separates
   the model-size and math-token axes the single FLOPs axis conflates.

### 4.1 Held-out results

| form | 1e21 mean abs % | 1e22 mean abs % | 1e22 signed % |
|---|---:|---:|---:|
| **pooled_mechanistic** | 1.11 | **8.97** | −8.97 |
| per_recipe_power (baseline) | 1.06 | 10.70 | −10.70 |
| pooled_broken_power | 1.37 | 10.77 | −10.77 |
| pooled_curvature | 2.15 | 14.83 | −14.83 |
| per_recipe_floor (Chinchilla) | 2.32 | 15.76 | −15.76 |

`pooled_mechanistic` wins at 1e22 and is the only form **unbiased** at 1e21 (+0.17% signed vs
everyone else negative). Its residual still shrinks-but-keeps the mix ordering (p33m67 −14.0,
p50m50 −9.2, p67m33 −3.6, vs baseline −15.0 / −11.1 / −6.1).

### 4.2 Concrete: predicted vs observed loss at 1e22 (lr0.67)

Percentages are abstract; here are the actual loss values:

| mix | form | predicted | observed | error |
|---|---|---:|---:|---:|
| p33m67 | per_recipe_power | 0.644 | 0.560 | −15.2% |
| p33m67 | pooled_mechanistic | 0.639 | 0.560 | −14.3% |
| p67m33 | per_recipe_power | 0.706 | 0.665 | −6.1% |
| p67m33 | pooled_mechanistic | 0.689 | 0.665 | −3.6% |

### 4.3 Why N × D_math helps — and a subtle reason it's the *only* axis trick that can

A crucial methodological point I verified: **within a single recipe**, swapping the x-axis from
compute `C` to midtrain-tokens or math-tokens is a **no-op** for prediction. Along the ladder
`N ∝ C^0.38` and `D ∝ C^0.60` (clean powers), so any of these axes is just a linear rescaling
of `log C` — same fit, same extrapolation. So "use math tokens as the x-axis" does *nothing*
per-recipe.

The only way the math-token axis can help is by **pooling across mixes**, where at a fixed
scale `D_math` differs by mix (0.33/0.50/0.67 × midtrain tokens). Pooling lets the math-data
exponent be estimated from all 12 recipes jointly and gives math-heavy cells a steeper
effective trajectory. That is exactly what `pooled_mechanistic` exploits, and why a per-recipe
reparametrization would not.

### 4.4 Why curvature/broken-power don't rescue 1e22

`pooled_curvature` and `pooled_broken_power` *can* bend, but fit on the **straight** small
ladder they have nothing to anchor the bend to — the acceleration is below the noise floor
until 1e22, which is held out. So they either don't bend (collapse to the power law) or bend
based on small-ladder wiggle and extrapolate wildly (`pooled_broken_power` hits 26% in the
CV fold that trains on ≤1e21). **You cannot learn the 1e22 bend from data that stops at 3e20.**

(Aside: I first gave `pooled_mechanistic` a `(log D_math)²` curvature term. It *overfit* the
straight small ladder and made 1e22 worse — 10.6% vs 8.9% for the linear version. I dropped it.
This is the same lesson: curvature fit from below doesn't transfer.)

---

## 5. Kaiyue's idea: predict the improvement Δ

Tested directly. Δ = (step-0 baseline loss) − (final loss). Fit a power law to Δ on the small
ladder, predict Δ at the targets, reconstruct `final = baseline_observed − Δ_pred`.

| target | improvement-decomposition mean abs % | (vs absolute-loss baseline) |
|---|---:|---:|
| 1e21 | 2.80 | 1.06 |
| 1e22 | **17.21** | 10.70 |

**Worse, not better.** Two reasons:

1. **Inconsistent baseline.** The step-0 baselines are 2.247 (3e18) → 1.582 (3e20) → 1.46
   (1e21) → 1.265 (1e22). But (see Section 9) the small-ladder runs appear to be *false*
   midtraining (fresh warmup from a fully-pretrained checkpoint → step-0 = fully-pretrained
   loss) while 1e21/1e22 appear to be *true* midtraining (resumed mid-cooldown → step-0 = a
   partially-cooled model). So Δ is measured against two different definitions of "before."
2. **Differencing amplifies error.** baseline ≈ 1.2–2.2, final ≈ 0.5–1.5 — similar magnitudes.
   A small absolute error in predicted Δ becomes a large % error in `baseline − Δ`.

The instinct is sound — isolating the midtraining effect *should* be cleaner — but it needs a
**consistent reference**, i.e. real 0%-math control runs, which don't exist in the data.

---

## 6. Ahmed's LR hypothesis

"Maybe we shouldn't use the same LR multiplier across the ladder." The data doesn't support
this as the cause of the 1e22 miss:

- All four LRs over-shoot **together**; the bias is **mix-ordered, not LR-ordered**.
- The base peak LR is already nearly scale-invariant (7.4e-3 → 7.2e-3), and LR is second-order
  (within-mix spread ~0.01–0.015 vs the mix spread ~0.1).

A finer LR sweep is still worth doing for **prescription** (what LR to use at 1e23) and to fit
the best-LR *envelope*, but it won't fix the extrapolation bias.

---

## 7. Trustworthy horizon (rolling-origin CV)

Fit up to a cutoff scale, predict the **next** scale up; report mean abs % vs extrapolation
distance. Baseline (`per_recipe_power`) and the field:

| test | × multiple | per_recipe_power | pooled_mechanistic | pooled_broken_power | per_recipe_floor |
|---|---:|---:|---:|---:|---:|
| 3e19 | 1.5× | 0.16 | 0.69 | 0.20 | 1.94 |
| 3e20 | 1.5× | 0.19 | 0.67 | 0.44 | 0.83 |
| 2e20 | 2.2× | 1.27 | 0.66 | 1.27 | 0.67 |
| 9e19 | 3.0× | 0.12 | 0.51 | 0.29 | 2.58 |
| 1e21 | 3.3× | 1.06 | 1.33 | 1.37 | 2.32 |
| **1e22** | **10×** | **9.94** | **9.16** | **26.04** | **11.18** |

**Reading:** within-ladder and 1-decade hops are all ≤~2.5%. The cliff is at 10× (1e22), where
everything is ~9–11% (broken-power unstable at 26%). Note the mechanistic and curvature forms
edge out the power law here (9.16 / 9.55 vs 9.94) *because 1e21 is in this fold's training set*
— once you have an anchor near the target, the bend becomes partly visible. **That is the whole
story in one line: the fix for 1e22 is an anchor near 1e22, not a cleverer curve.**

Practical rule baked into the report: **trust to ~3×; treat beyond that as directional.**

---

## 8. What I shipped

**`scripts/analysis/delphi_small_final_loss_scaling.py`**
- Canonical per-scale `SCALE_PARAMS_B`, `SCALE_PRETRAIN_TOKENS_B`, `MATH_FRACTION`
  (verified against `experiments/delphi_models.py`).
- `attach_scaling_features`, the `EndpointForm` registry, and the five forms.
- `compare_forms_heldout` + `summarize_form_comparison`, `rolling_origin_cv`,
  `select_best_form`, `full_ladder_grid`.
- New `summary.md` sections ("Form Comparison On Held-Out", "Trustworthy Extrapolation
  Horizon"); best form overlaid (dashed) on `endpoint_math_val_loss.html`.

**`scripts/analysis/build_delphi_midtraining_interactive_report.py`**
- New report section **"Endpoint Form Comparison & Trustworthy Horizon"** (the form table +
  CV table + horizon sentence; payload keys `formComparison`, `cvByDistance`,
  `renderFormComparison`). The "Endpoint Scaling Law" plot now overlays the `pooled_mechanistic`
  fit.

**New outputs:** `endpoint_form_comparison.csv`, `endpoint_form_comparison_summary.csv`,
`endpoint_cv_by_distance.csv`.

All pre-commit clean; report regenerates byte-stable; ~17 MB page budget preserved.

---

## 9. Threats to validity (read this before believing anything above)

1. **Protocol mismatch (the big one).** There appear to be two midtraining experiment files:
   `exp_delphi_math_10b_midtrain.py` ("false" midtrain — load weights only, fresh warmup) and
   `exp_delphi_true_midtrain.py` ("true" midtrain — resume from pretrain mid-cooldown). The
   small-ladder run names (`…-k0p20-…`) and the held-out names (`…-9p25b/32p07b-…`) differ, and
   the step-0 baselines behave differently. **If the small ladder and 1e21/1e22 were produced
   by different protocols, the extrapolation is comparing two different training procedures**,
   and a chunk of the systematic 1e22 offset could be a protocol artifact rather than scaling.
   I did not fully confirm which scales used which file. **This is the first thing to check.**
2. **Tiny lever arm, 2-decade extrapolation.** 7 points per recipe, exponents estimated over
   ~2.5 decades, extrapolated 1.5 decades. Estimation variance alone is non-trivial; I have not
   bootstrapped CIs on the exponents (worth adding).
3. **One 1e22 cell is a forecast, not observed** (`p50m50-lr67`, excluded — that's why n=11).
4. **The mechanistic win is modest** (10.7→9.0%) and rests on pooling assumptions (shared
   exponents across mixes/LR). It is better and less biased, but not a "solved" predictor.

---

## 10. My take (opinionated)

- **I'm confident** the 1e22 miss is a real, mix-ordered acceleration, not noise and not a
  floor. The slope-steepening and the monotone-in-math-fraction ordering are too clean to be
  chance.
- **My leading mechanistic hypothesis:** `math_val_loss` is governed by a **math-token
  data-scaling law** that the compute ladder has never isolated (N and D always co-move). The
  small ladder is in a data-starved regime for the math skill; 1e22 is not. The "acceleration"
  is the math skill's data-scaling law having a steeper effective slope once you're past the
  starved regime — possibly a soft emergence/breakpoint in `D_math`.
- **But I genuinely can't rule out the protocol confound** (Section 9.1). If false-vs-true
  midtrain differs systematically, the whole "acceleration" could be partly an artifact of the
  held-out runs using a more favorable schedule. This would *also* explain why the over-
  performance is so one-directional. I'd put real probability on this mattering.
- **Therefore the honest headline is not "use the mechanistic fit," it's: you cannot reliably
  extrapolate this 2 decades from the small ladder, full stop.** The mechanistic fit is the
  best available band-aid (~9%, unbiased at 1e21); the report's horizon is the real deliverable.
- **What actually buys predictability is data, not a cleverer functional form:**
  1. **Iso-N, vary-`D_math` sweep** at one or two small scales (e.g. 3e20: K ∈ {0.1, 0.2, 0.4,
     0.8}). This is Ahmed's "train small models longer," done correctly — not to add compute,
     but to **measure the math-token exponent at fixed model size**, the quantity the ladder
     confounds. Cheap, and the single most informative experiment.
  2. **0%-math control runs** at ≥3 scales incl 1e21/1e22, with the **same protocol** as the
     targets. Makes Kaiyue's improvement decomposition viable *and* lets us settle the
     protocol-confound question.
  3. **Finer LR sweep at 1e21** — for prescription and the best-LR envelope, not the bias.
- If forced to pick one next action: **confirm the small-ladder vs 1e21/1e22 protocol is
  identical.** If it isn't, fix that before any more curve-fitting — it may dissolve half the
  puzzle.

---

## 11. Reproduce

```bash
# fits, comparison, CV, summary, plots (uses cached endpoints; fetches held-out from W&B)
uv run python scripts/analysis/delphi_small_final_loss_scaling.py --use-cache

# interactive report
uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py \
  --output midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html
```

Key files:
- Fits/comparison/CV: `scripts/analysis/delphi_small_final_loss_scaling.py`
- Report: `scripts/analysis/build_delphi_midtraining_interactive_report.py`
- Numbers: `midtrain_analysis_outputs/small_final_loss_scaling/endpoint_form_comparison*.csv`,
  `endpoint_cv_by_distance.csv`, `summary.md`
- Source data: `midtrain_analysis_outputs/small_final_loss_scaling/{endpoints,extrapolation_targets}.csv`
- Canonical N/tokens: `experiments/delphi_models.py`
- Full thread/handoff: `.agents/logbooks/delphi_midtraining_visualization.md`
