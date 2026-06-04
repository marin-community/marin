# Codex Logbook: Delphi Midtraining Endpoint Prediction

Objective: critically analyze `midtrain_pred_retro_claude.md`, develop independent claims and hypotheses, and test whether 1e22-scale performance can be predicted from smaller models using absolute loss, deltas, anchors, or other transformations.

Operating rule: append-only. Log attempts, ideas, commands, artifacts, outcomes, and negative results.

## 2026-05-30 19:08 ET — codex-001 — State Restore And Claim Audit

### Inputs Inspected

- `midtrain_pred_retro_claude.md` exists and is an untracked 371-line retrospective.
- `midtrain_pred_retro_codex_logbook.md` did not exist before this entry, so this file starts the append-only Codex log.
- Existing related artifacts are present under `midtrain_analysis_outputs/small_final_loss_scaling/`, including endpoint CSVs, held-out prediction CSVs, form-comparison CSVs, trajectory-prefix CSVs, and the interactive report.
- The worktree already has modified files unrelated to this initial logbook creation:
  - `.agents/logbooks/delphi_midtraining_visualization.md`
  - `.agents/logbooks/midtraining_delphi.md`
  - `scratch/20260530_ocdbt_verify_result.json`
  - `scripts/analysis/build_delphi_midtraining_interactive_report.py`
  - `scripts/analysis/delphi_small_final_loss_scaling.py`

### Claude Retro: Claims To Verify Or Challenge

1. The 1e22 final `math_val_loss` miss is real, systematic, one-directional, and pessimistic: small-ladder fits predict higher loss than observed.
2. The miss grows monotonically with math fraction, not LR.
3. Local log-log slopes steepen between 1e21 and 1e22, especially for math-heavy mixes.
4. A Chinchilla-style irreducible floor bends the wrong way and worsens predictions.
5. The best current modeling-only variant is a pooled `N` plus `D_math` law, but it only reduces 1e22 mean absolute percentage error from roughly 10.7% to roughly 9.0%.
6. Naive improvement/delta prediction is worse because baselines are inconsistent and differencing amplifies error.
7. No form trained only below 1e21 can reliably infer the 1e22 bend; the missing ingredient is a near-target anchor or new experimental axis.
8. Protocol mismatch between the small ladder and 1e21/1e22 may be a first-order confound.

### Initial Independent Hypotheses

H1. Absolute final loss extrapolation from the small ladder alone probably cannot predict 1e22 without a structural prior or anchor; the target is too far from the observed scale range and all axes co-move.

H2. Delta prediction may still be viable if the delta is normalized by a consistent baseline proxy or predicted as a residual relative to a shared absolute-loss law, rather than using raw step-0 minus final loss.

H3. A useful predictor may be neither absolute nor raw delta, but a calibrated extrapolation-error model: fit small-to-intermediate residuals as a function of extrapolation distance, math fraction, and slope drift, then use 1e21 as an anchor for 1e22.

H4. If 1e21 is allowed as an anchor, predicting the 1e21-to-1e22 delta or slope may be meaningfully better than fitting only `3e18..3e20`; this is still operationally useful if one can afford a 1e21 scout before a 1e22 run.

H5. If 1e21 is not allowed, a trustworthy result may be a calibrated uncertainty interval rather than a point prediction: the small ladder might predict rank ordering and rough range, but not absolute 1e22 loss.

### Next Experiments

1. Reproduce the endpoint tables from cached CSVs and verify Claude's headline numbers.
2. Measure sign, mix ordering, LR ordering, local slopes, and scale-dependent residuals directly from `endpoints.csv` and `extrapolation_targets.csv`.
3. Compare absolute-loss fits, residual/delta fits, anchored slope extrapolations, and uncertainty intervals.
4. Generate plots under a Codex-specific output directory so the analysis is inspectable without modifying prior report artifacts.

## 2026-05-30 19:18 ET — codex-002 — Artifact And Protocol Audit

### Commands

- `uv run python` inspection over:
  - `midtrain_analysis_outputs/small_final_loss_scaling/endpoints.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_targets.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/extrapolation_predictions.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/endpoint_form_comparison*.csv`
  - `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_*`
  - `midtrain_analysis_outputs/midtrain_trajectory_deltas.csv`
- `rg` and targeted config parsing over `midtrain_wandb_data/runs/*/config.json`.
- Read relevant comments/docstrings in `experiments/exp_delphi_math_10b_midtrain.py`, `experiments/exp_delphi_true_midtrain.py`, `experiments/delphi_models.py`, and the midtraining logbooks.

### Findings

- The cached summary reproduces the core endpoint numbers:
  - `per_recipe_power` small-ladder fit: 1e21 mean abs percent error 1.06%, 1e22 10.70%, signed 1e22 mean -10.70%.
  - `pooled_mechanistic`: 1e22 mean abs percent error 8.97%, signed -8.97%.
  - `per_recipe_floor`: 1e22 mean abs percent error 15.76%, signed -15.76%.
- `trajectory_points.csv` includes both small-CV and held-out-large trajectories for the math/eval/Paloma metrics, so raw improvement, relative improvement, and prefix-calibrated prediction are testable without new downloads.
- Important challenge to Claude Section 9: cached configs for representative 1e21 and 1e22 held-out runs show:
  - `checkpoint_init_mode: MODEL_ONLY`
  - `initialize_from_checkpoint_path: mirror://adamh-scaling-ladder-.../checkpoints/step-*`
  - `trainer.load_checkpoint_path: None`
  - fresh trainer steps (`4411` for 1e21, `7647` for 1e22)
  - `optimizer.rewarmup: 0`, `optimizer.cooldown: None`
- That is the old/fake CPT protocol, not `exp_delphi_true_midtrain.py` natural resume. The true-midtraining file explicitly says true runs should be named `true-midtrain-*` and use no `initialize_from_checkpoint_path`; the held-out names are `delphi-1e21-*` / `delphi-1e22-*`.
- So the specific claim "small ladder is false midtrain but 1e21/1e22 are true midtrain" is not supported by local configs. The stronger, narrower confound remains: schedules and launch generations differ. In particular, the legacy 1e21/1e22 runs use fixed warmup-token logic (`500` warmup steps at 1e21, `250` at 1e22), while the newer small K=0.20 ladder docs emphasize fractional warmup policy. This can affect endpoint curvature and step-0 baselines, but it is not the same as true-vs-false midtraining.

### Updated Hypotheses

H6. The protocol confound should be reframed as "different CPT launcher/schedule generation" rather than "false vs true midtraining." This weakens Claude's strongest invalidation of delta prediction, but does not rescue raw deltas automatically.

H7. Because `trajectory_points.csv` has baseline values, the next decisive check is not just raw `baseline - final`, but relative improvement and final/baseline ratio. If those fail too, the delta family is genuinely weak under the current data.

H8. The most likely useful prediction product may be: absolute point predictions are biased at 1e22, but ranking and recipe selection are predictable; early large-run prefixes can calibrate final loss much better than small-only extrapolation.

## 2026-05-30 19:37 ET — codex-003 — Independent Prediction Experiments

### Artifact

Added and ran:

```bash
uv run python scripts/analysis/codex_midtrain_prediction_retro.py
```

Outputs:

- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/summary.md`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/prediction_experiments.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/prediction_experiment_summary.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/local_slopes.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/rank_predictability.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/bootstrap_power_intervals.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/prefix_prediction_summary.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/heldout_protocol_audit.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/baseline_metric_summary.csv`
- `midtrain_analysis_outputs/codex_midtrain_prediction_retro/envelope_predictions.csv`
- HTML plots in the same directory:
  - `math_loss_endpoints.html`
  - `local_slopes_by_mix.html`
  - `baseline_1e22_error_heatmap.html`
  - `prediction_experiment_summary.html`
  - `rank_selection_1e22.html`
  - `prefix_prediction_1e22.html`
  - `bootstrap_intervals_1e22.html`
  - `baseline_metric_errors.html`
  - `envelope_prediction_summary.html`

Validation:

- `uv run python -m py_compile scripts/analysis/codex_midtrain_prediction_retro.py` passed.
- `uv run ruff check ...` could not run because `ruff` is not installed in this environment.

### Experiments Tried

1. Reproduced small-only per-recipe absolute power law.
2. Pooled absolute variants:
   - shared slope with per-recipe intercept
   - mix-dependent slope
   - mix/LR-dependent slope
   - mechanistic `log_n + log_dmath + lr`
   - mechanistic plus `log_dmath * math_frac`
3. 1e21-anchored predictors:
   - anchor at observed 1e21 with small-ladder slope
   - anchor at observed 1e21 with last segment slope (`3e20->1e21`)
   - recipe-specific slope trend
   - repeat the observed slope drift from small fit to `3e20->1e21`
4. Delta-style predictors from trajectory baselines:
   - raw improvement: `baseline - final`
   - relative improvement: `(baseline - final) / baseline`
   - final/baseline ratio
5. Mix-gap predictors:
   - predict `p50/p33 - p67` gaps from small scales, with predicted p67 or observed p67 anchor.
6. Rank/selection analysis:
   - Spearman rank and regret of the predicted best recipe.
7. Residual-bootstrap intervals:
   - bootstrap small-ladder per-recipe residuals and test whether held-out endpoints fall inside 95% intervals.
8. Early-prefix prediction:
   - summarize existing trajectory-prefix methods specifically for held-out 1e22 math loss.
9. Metric specificity:
   - compare baseline small-only power errors for math, aggregate eval, Paloma macro, Paloma C4, and train loss.
10. Envelope prediction:
   - predict best LR per mix and best overall loss rather than every LR cell.

### Main Results

Small-only absolute prediction:

- `absolute_per_recipe_power_small_only`: 1e21 = 1.06% MAPE, 1e22 = 10.70% MAPE, signed 1e22 = -10.70%.
- `absolute_pooled_mechanistic_linear`: 1e22 = 9.03% MAPE.
- `absolute_pooled_mechanistic_mix_interaction`: 1e21 improves to 0.82% and 1e22 is 9.00%, but it does not materially beat the prior mechanistic result.
- Pooled mix/LR slope variants do not help 1e22; they stay around 10.7%.

Anchoring at 1e21:

- `anchor_1e21_small_power_slope`: 1e22 = 9.61%.
- `anchor_1e21_last_segment_slope`: 1e22 = 7.58%.
- `anchor_1e21_recipe_slope_trend`: 1e22 = 8.31%.
- `anchor_1e21_repeat_slope_drift`: 1e22 = 5.62%, the best non-prefix point predictor tried so far.

Interpretation: if a 1e21 scout exists, the useful signal is not just the absolute 1e21 loss; it is the change in local slope near the top of the ladder. Repeating the last observed slope drift is crude and retrospective, but it shows that the 1e22 gap is partially predictable once the 1e21 anchor reveals acceleration.

Delta predictors:

- Raw improvement reproduces Claude's failure: 1e22 = 17.21%.
- Relative improvement is better but still bad: 1e22 = 13.25%.
- Final/baseline ratio is also bad: 1e22 = 15.58%.

Interpretation: delta framing is not rescued by simple normalization. The problem is not just baseline offset; the improvement law itself changes shape at 1e22.

Rank/order:

- The small-only absolute predictor has 1e22 Spearman rank 0.973 and chooses `p33m67-lr67`, the actual best observed recipe, with 0% selection regret.
- Several anchored and delta variants also choose the true best recipe.

Interpretation: absolute loss is poorly calibrated, but recipe ordering is highly predictable. This matters if the operational decision is "which mix/LR should we run?" rather than "what exact loss will we get?"

Envelope:

- Predicting best-LR-per-mix from small scales does not fix absolute calibration: 1e22 best-LR-per-mix MAPE = 11.04%; best overall error = 15.35%.
- Anchoring the envelope at 1e21 still leaves 9.86% best-LR-per-mix MAPE and 12.98% best-overall error.

Interpretation: envelope fitting makes the target more decision-like but does not solve final-loss prediction.

Prefix calibration:

- Existing trajectory-prefix methods are far better once any 1e22 prefix is available:
  - 10% prefix: best method `template_by_recipe`, 3.08% MAPE.
  - 70% prefix: 3.35% MAPE.
  - 80% prefix: 2.02% MAPE.
  - 90% prefix: 0.74% MAPE.
- The non-monotone prefix curve (20-60% worse than 10% in these summaries) needs more inspection; it may reflect method selection/templates and schedule phase rather than a real information regression.

Metric specificity:

- At 1e22, small-only power errors by metric:
  - `math_val_loss`: 10.70%, signed -10.70% (observed better than predicted).
  - `eval_loss`: 2.35%, signed +2.35% (observed worse than predicted).
  - `paloma_macro_loss`: 3.67%, signed +3.67%.
  - `paloma_c4_loss`: 3.91%, signed +3.91%.
  - `train_loss`: 10.48%, signed -9.47%.

Interpretation: the acceleration is math-specific, not a broad "large runs are better than expected" effect. The data suggest specialization improves faster than expected while general/Paloma retention is slightly worse than expected.

Bootstrap:

- Small-ladder residual-bootstrap 95% intervals cover 5/12 1e21 cells and 0/11 1e22 cells.
- 1e22 interval widths are only about 3% of observed loss, far below the 6-15% systematic miss.

Interpretation: ordinary fit uncertainty is misleadingly tight because small-ladder residuals do not contain the regime change. The honest uncertainty has to include model-form/regime uncertainty, not just regression residuals.

### Current Claims

C1. I agree with Claude that the 1e22 `math_val_loss` miss is real, systematic, pessimistic, and mix-ordered.

C2. I agree that an irreducible floor is the wrong geometry for this miss.

C3. I partially agree on the mechanistic `N x D_math` law: it is a modest improvement, not a solution. My added interaction term does not materially change that.

C4. I disagree with the specific "small false vs large true midtraining" caveat. Cached held-out configs show the large runs are also `MODEL_ONLY` CPT. The remaining protocol caveat is schedule/launcher mismatch, especially warmup fraction.

C5. Delta prediction is not viable in the naive forms tried here, even after normalizing by baseline. It can predict ranking but not calibrated 1e22 absolute loss.

C6. Small-only prediction can choose the best recipe, but cannot calibrate the absolute 1e22 math loss. This distinction should be explicit.

C7. If a 1e21 anchor is allowed, slope-drift extrapolation can reduce 1e22 error from 10.7% to 5.6%. This is the best non-prefix point predictor so far, but it is still biased pessimistic.

C8. If an early 1e22 prefix is allowed, trajectory-template calibration is much stronger than pure scale extrapolation. A 10% prefix already gives about 3.1% MAPE, and late prefixes fall below 2%.

### Next Ideas

1. Inspect why 10% prefix beats 20-60% prefix in the current template summaries.
2. Test whether anchored slope drift can be estimated from held-out folds without using 1e22, rather than just retrospectively repeating the last drift.
3. Break 1e22 residuals into math-gain versus Paloma-retention tradeoff to see whether the "better math, worse retention" pattern explains the absolute miss.
4. Add a concise final synthesis once these checks stop changing the conclusion.

## 2026-05-30 19:51 ET — codex-004 — Slope-Drift CV And Prefix-Fraction Diagnostics

### Commands

Updated `scripts/analysis/codex_midtrain_prediction_retro.py` to regenerate:

- `anchored_slope_drift_cv.csv`
- `anchored_slope_drift_cv_summary.csv`
- `prefix_fraction_diagnostics.csv`
- `prefix_fraction_large_vs_small.csv`
- `anchored_slope_drift_cv.html`
- `prefix_fraction_large_vs_small.html`

Then ran:

```bash
uv run python scripts/analysis/codex_midtrain_prediction_retro.py
uv run python -m py_compile scripts/analysis/codex_midtrain_prediction_retro.py
```

Both completed successfully.

### Slope-Drift CV Result

Rolling-origin anchored prediction, where the model sees an anchor at the previous scale and predicts the next scale:

| test | jump | small slope | last-segment slope | repeat-drift |
|---|---:|---:|---:|---:|
| 1e21 | 3.3x | 1.01% | **0.85%** | 2.65% |
| 1e22 | 10.0x | 9.61% | 7.58% | **5.62%** |

Across earlier within-ladder tests, repeat-drift is often worse:

- `3e19`: small slope 0.20%, last segment 1.34%, repeat-drift 2.48%.
- `9e19`: small slope 0.12%, last segment 1.76%, repeat-drift 3.47%.
- `3e20`: small slope 0.53%, last segment 1.18%, repeat-drift 1.83%.
- `2e20`: repeat-drift is slightly best at 1.15%, but all three are close.

Interpretation: repeat-drift is not a robust general extrapolator. It helps at 1e22 because 1e22 is exactly the first place where slope acceleration becomes large. This is evidence that a 1e21 anchor reveals a regime transition, not proof that "repeat the last drift" is a reliable law.

### Prefix Fraction Result

Computed fraction of final math improvement already achieved by each prefix:

`fraction = (baseline_loss - prefix_loss) / (baseline_loss - final_loss)`

Comparison of held-out large runs to the small-run mean:

| scale | prefix | large mean fraction | small mean fraction | large - small |
|---|---:|---:|---:|---:|
| 1e22 | 0.10 | 0.555 | 0.549 | +0.006 |
| 1e22 | 0.20 | 0.661 | 0.692 | -0.031 |
| 1e22 | 0.30 | 0.733 | 0.772 | -0.039 |
| 1e22 | 0.50 | 0.834 | 0.870 | -0.037 |
| 1e22 | 0.70 | 0.907 | 0.937 | -0.030 |
| 1e22 | 0.80 | 0.946 | 0.965 | -0.018 |
| 1e22 | 0.90 | 0.980 | 0.987 | -0.007 |

Interpretation: the 10% prefix works unusually well because 1e22 and small runs have almost identical normalized-improvement fraction at that point. From 20-70%, 1e22 has achieved 3-4 percentage points less of its final improvement than small runs at the same nominal prefix, so small-template methods overestimate the endpoint improvement and become pessimistic. By 80-90%, the fraction mismatch shrinks again.

This explains the non-monotone prefix summary without invoking noise: the large run's normalized trajectory is slightly delayed relative to the small-run template through the middle of the run.

### Updated Synthesis

C9. The strongest actionable prediction mode is "small models give trajectory shape, early 1e22 prefix calibrates amplitude," but the shape is not perfectly invariant. The mid-run large-scale trajectory lags the small template.

C10. A 1e21 anchor plus slope drift is useful evidence for curvature but should not be sold as a stable law. The honest production rule is closer to: once the last observed segment starts steepening, widen uncertainty and expect the next decade to beat the small-only power law on math loss.

C11. The Claude retro's "trustworthy to ~3x" claim remains directionally right, but the exact extrapolation-distance language needs care: small-ladder-to-1e22 is 33x beyond the 3e20 fit max, while 1e21-to-1e22 anchored prediction is 10x.

## 2026-05-30 20:00 ET — codex-005 — Math/Retention Residual Tradeoff

### Commands

Updated `scripts/analysis/codex_midtrain_prediction_retro.py` to write:

- `tradeoff_residuals_by_cell.csv`
- `tradeoff_residuals_by_mix.csv`
- `math_vs_paloma_residual_tradeoff.html`

Then ran:

```bash
uv run python scripts/analysis/codex_midtrain_prediction_retro.py
uv run python -m py_compile scripts/analysis/codex_midtrain_prediction_retro.py
```

Both completed successfully.

### Result

Baseline small-only power signed percent residuals by mix:

| target | mix | math | eval | paloma macro | paloma c4 |
|---|---|---:|---:|---:|---:|
| 1e21 | p33m67 | -2.00 | +1.14 | +1.57 | +1.59 |
| 1e21 | p50m50 | -1.08 | +1.16 | +1.35 | +1.43 |
| 1e21 | p67m33 | +0.11 | +1.42 | +1.31 | +1.42 |
| 1e22 | p33m67 | -15.00 | +2.03 | +3.90 | +4.08 |
| 1e22 | p50m50 | -11.14 | +2.23 | +3.67 | +3.90 |
| 1e22 | p67m33 | -6.07 | +2.77 | +3.43 | +3.74 |

Sign convention: negative means observed loss is lower/better than predicted; positive means observed loss is higher/worse than predicted.

Interpretation:

- The 1e22 surprise is a specialization/retention tradeoff shift, not a uniform quality gain.
- Math improves much faster than the small-ladder law predicts.
- Paloma macro/C4 are worse than predicted, with the largest retention miss in the math-heavy mix.
- Aggregate `eval_loss` is also worse than predicted, but its mix ordering differs from Paloma; do not over-read aggregate eval as a pure retention metric.

### Updated Claims

C12. The best explanation is not simply "large models overperform scaling laws." It is "math specialization overperforms, while general/Paloma retention underperforms mildly." Any predictive model for 1e22 should be multi-objective; predicting only math loss hides a retention cost.

C13. The mix ordering strengthens the math-token hypothesis: more math fraction gives more unpredicted math gain and more unpredicted Paloma damage.

C14. A practically useful forecast should report at least three things separately: calibrated math-loss interval, recipe rank/regret, and retention-risk interval.

## 2026-05-30 20:08 ET — codex-006 — Final Synthesis

### Direct Answer

Can we predict 1e22 performance from smaller models?

**Absolute 1e22 math loss from the small ladder alone: no, not to single-digit precision.** The standard per-recipe power law misses by 10.70% MAPE and every small-only variant I tried remains around 9-11%. The residual is systematic, not noise, and bootstrap intervals from small residuals are falsely tight.

**Absolute 1e22 math loss with a 1e21 anchor: partially.** Anchoring on observed 1e21 and using the last-segment slope reduces the miss to 7.58%; repeating the observed slope drift reduces it to 5.62%, but rolling-origin CV shows repeat-drift is not generally reliable. I would use this as an uncertainty-widening signal, not a production law.

**Deltas/improvements from step-0 baselines: no in the simple forms tested.** Raw improvement is worse than absolute loss (17.21% at 1e22). Relative improvement and final/baseline ratio are also worse than absolute loss (13.25% and 15.58%). Delta methods do preserve ranking well, but not calibrated loss.

**Recipe choice/ranking: yes.** The small-only absolute predictor has 1e22 Spearman rank 0.973 and picks the actual best observed recipe, `p33m67-lr67`, with zero regret. If the operational question is which recipe to run, the small ladder is much more useful than the absolute-loss MAPE suggests.

**Early 1e22 prefix to final 1e22 endpoint: yes, much better.** Existing trajectory-template predictions using a 10% prefix reach 3.08% MAPE; 80% prefix is 2.02%, 90% prefix 0.74%. This is not "from smaller models only," but it is the strongest practical predictability story: small models supply a trajectory template, the large prefix calibrates amplitude.

**Retention/general performance: needs separate prediction.** Math loss is better than predicted while Paloma/C4 retention is worse than predicted. A single scalar "performance" forecast is misleading unless it includes both specialization gain and retention damage.

### My Claims

1. The 1e22 math-loss overperformance is real and mix-ordered.
2. It is best described as a math-specialization acceleration, not a general large-model quality boost.
3. A floor/irreducible-loss fix has the wrong curvature for this failure.
4. Separating `N` and `D_math` helps modestly, but the existing ladder does not identify the 1e22 regime transition.
5. Naive delta prediction fails even after baseline normalization.
6. Ranking is much more predictable than calibrated absolute loss.
7. A 1e21 anchor reveals useful slope steepening, but slope-drift rules are not robust enough to trust blindly.
8. Early large-run prefixes are the strongest practical route to endpoint prediction.
9. Claude's "large runs may be true-midtraining while small runs are false-midtraining" caveat is not supported by cached held-out configs. The real caveat is schedule/launcher mismatch, especially warmup fraction.
10. Future forecasts should be multi-objective: math loss, recipe ranking/regret, and retention risk.

### Hypotheses Worth Testing With New Runs

1. Iso-N / vary-`D_math` sweep: fixes the central identifiability problem by measuring the math-token exponent without changing model size.
2. 0%-math controls under the same launcher/schedule: makes improvement decomposition meaningful and separates base-model quality from midtraining effect.
3. 1e21 scout then 1e22 forecast: operationally test whether slope steepening at 1e21 reliably predicts the next decade.
4. Prefix-calibrated large-run forecasting: freeze a rule before seeing the endpoint and test whether 10-30% prefixes can predict final math and retention jointly.
5. Retention-aware envelope fitting: choose recipe by a weighted math/Paloma objective, not by math loss alone.

### Stop Condition

I am stopping because the explored branches now agree on the same conclusion:

- small-only calibrated absolute prediction is not solved;
- deltas do not rescue it;
- anchors and prefixes help, with prefixes clearly better;
- ranking is reliable;
- retention moves in the opposite direction from math specialization.

Additional work would require either new runs or turning this retrospective into a polished report, not more ad-hoc curve fitting.
