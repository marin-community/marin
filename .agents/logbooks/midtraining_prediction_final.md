# Delphi Midtraining Prediction Final Retrospective

Date: 2026-06-28

Scope: Delphi p33m67/p50m50/p67m33 math midtraining, K=0.20 iso-FLOP, iso-token
controls, Nemotron math validation contamination, and the final clean-seen eval
reports.

Status: final narrative and artifact index for the prediction/debugging thread.

## Executive Summary

The original symptom was real in the old validation target: a scaling law fit on
the small K=0.20 ladder through 3e20 predicted the 1e21 endpoints reasonably, but
predicted 1e22 math validation loss far too high. On the public interactive
dashboard, p33m67 K=0.20 missed 1e22 by roughly +18%: the model did much better
than expected on the old `nemotron_cc_math_v1/4plus` validation anchor.

That was not solved by trying many more fitting forms. We tried per-recipe power
laws, Chinchilla floor+power, pooled mechanistic fits, log-log fits, parameter
and data-token axes, base-model rows at zero midtraining tokens, and separate
base/improvement components. These helped describe iso-token runs, but on the
old K=0.20 target they still left a large 1e22 error.

Two confounds were the actual issue.

First, K=0.20 is not "FLOPs only." It includes midtraining. Because it is an
iso-FLOP fraction of the base pretraining compute, the absolute midtraining
token budget grows with base scale. In the p33m67 ladder it grows from about
0.245B tokens at 3e18 to about 32B tokens at 1e22. Once we ran iso-token controls,
holding the midtraining token budget fixed, the old 1e22 miss mostly disappeared:
fixed-budget iso-token ladders were smooth, monotone, and had small negative
1e22 errors around -3% to -4%.

Second, the old math validation anchor was contaminated for the thing we were
measuring. It was not exact duplicate contamination by document id; exact hashes
looked clean. The problem was fuzzy near-duplicate and same-source leakage inside
the math corpus, combined with scale-dependent exposure during midtraining. The
validation split was a deterministic Feistel carve-out from
`nemotron_cc_math_v1/4plus`, but the train remainder still contained many near
twins of the validation documents. At Jaccard >= 0.75, the scan found
9,757 / 57,243 validation docs and 9.53M / 51.20M validation tokens implicated.
Actual exposure replay showed this got worse with scale and with math fraction.

The final clean-seen evals changed the story. When we evaluated the K=0.20 ladder
on a validation set decontaminated against the actual seen training documents
for 1e22 p33m67 K=0.20, the 1e22 prediction errors collapsed to low single
digits. The clean K=0.20 lr0.50 series went from old full 4plus +18.56% error
to clean-seen +2.83%. The dropped contaminated complement retained a large
absolute miss, +0.0999 loss at 1e22, nearly the same absolute miss as old full
4plus (+0.1042). That is the strongest final evidence that the rough scaling was
an eval-target confound, not a broken law of midtraining.

The base models before midtraining were not the source of the anomaly. The
step-0/base math-loss Chinchilla plot fits the base-model math loss through
3e20 and predicts held-out 1e21/1e22 with small errors (+0.7% and +2.4% in the
Chinchilla view). The bad +12% to +19% endpoint errors appear after math
midtraining on the old target. Base-loss rows were also included in later
fit-family reports, and separate base/improvement components did not fix the
old K=0.20 target.

The shuffle work is a related but different lesson. Earlier Bison/Mantis work
found a harmful affine/LCG shuffle choice for cooldown training and moved to a
Feistel shuffle. The Delphi validation carve-out here used a full fixed Feistel
permutation before taking the validation tail, and the training stream used a
block shuffle with Feistel block permutation for I/O locality. The shuffle choice
made validation identity deterministic and was not the observed cause of this
prediction miss. The failure came from what was inside the corpus after the split:
near-duplicate math documents remained in the training side and were exposed
more heavily at larger midtraining budgets.

## Main Conclusion

The smooth midtraining scaling story is:

1. The old validation target made 1e22 look unusually good because the model had
   much more opportunity to see near-duplicate math pages during midtraining.
2. K=0.20 iso-FLOP compounded the problem by giving larger base models far more
   midtraining tokens.
3. When we either fix the midtraining token budget or change the validation set
   to an actual-seen clean target, the endpoint fits become smooth again.
4. When we do both, the remaining errors are small enough to look like ordinary
   extrapolation/model-form error rather than a qualitative scaling break.

## Artifact Map

### Public And Original Dashboards

| Artifact | What it contains | Why it matters |
|---|---|---|
| <https://ahmeda14960.github.io/delphi-midtraining/?v=c86be93c> | Frozen public interactive report titled `Delphi Midtraining Interactive Scaling`. It contains curve prediction, per-cell within-run prediction, endpoint scaling, target MAE config search, and joint trajectory fits. | This is the original public artifact the thread started from. It shows the old raw math validation anomaly. |
| [midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html](../../midtrain_analysis_outputs/small_final_loss_scaling/delphi_midtraining_interactive.html) | Local source-family interactive report. The exact `c86be93c` bytes are not present locally, but this is the local report family generated by the same script lineage. | Use this when working locally. Use the public URL for the frozen June 7 public snapshot. |
| [midtrain_analysis_outputs/small_final_loss_scaling/summary.md](../../midtrain_analysis_outputs/small_final_loss_scaling/summary.md) | Small-ladder endpoint fit summary. | Best compact text summary of the original 3e18->3e20 fit and 1e21/1e22 held-out miss. |
| [midtrain_analysis_outputs/small_final_loss_scaling/endpoint_math_val_loss.html](../../midtrain_analysis_outputs/small_final_loss_scaling/endpoint_math_val_loss.html) | Raw math validation endpoint scaling. | Shows the old target where 1e22 appears too good. |
| [midtrain_analysis_outputs/small_final_loss_scaling/endpoint_eval_loss.html](../../midtrain_analysis_outputs/small_final_loss_scaling/endpoint_eval_loss.html) | Aggregate eval/loss endpoint scaling. | Useful contrast: aggregate eval/loss did not show the same pathological math-val miss. |
| [midtrain_analysis_outputs/small_final_loss_scaling/endpoint_paloma_macro_loss.html](../../midtrain_analysis_outputs/small_final_loss_scaling/endpoint_paloma_macro_loss.html) | Paloma macro retention endpoint scaling. | Confirms this was not a universal validation instability. |
| [midtrain_analysis_outputs/small_final_loss_scaling/first_math_loss_chinchilla.html](../../midtrain_analysis_outputs/small_final_loss_scaling/first_math_loss_chinchilla.html) | Base step-0 math loss vs pretraining compute, plus endpoint overlays. | Key evidence that the base models before midtraining were smooth on this target. |
| [midtrain_analysis_outputs/index.html](../../midtrain_analysis_outputs/index.html) | Earlier local plot index. | Pointer to the first broader dashboard set. |
| [midtrain_analysis_outputs/analysis_summary.md](../../midtrain_analysis_outputs/analysis_summary.md) | Earlier analysis summary. | Historical context for the initial prediction approach. |

The public `?v=c86be93c` report embeds 4,673 trajectory points, 23 held-out
targets, 84 endpoints, 96 scaling fits, and tens of thousands of prediction rows.
For p33m67 K=0.20 on the old raw 4plus target, the public page's Chinchilla-style
endpoint fit through 3e20 gave these 1e22 misses:

| Series | Old 1e22 actual | Prediction | Percent error | Loss error |
|---|---:|---:|---:|---:|
| p33m67 lr0.33 | 0.572544 | 0.681570 | +19.04% | +0.109026 |
| p33m67 lr0.50 | 0.561019 | 0.665204 | +18.57% | +0.104185 |
| p33m67 lr0.67 | 0.559539 | 0.661742 | +18.27% | +0.102203 |
| p33m67 lr0.83 | 0.563027 | 0.663669 | +17.88% | +0.100642 |

### Earlier Plot Set

These are the older local plot HTMLs from the first dashboard family:

| Plot | Meaning |
|---|---|
| [midtrain_analysis_outputs/plots/current_eval_loss_by_scale.html](../../midtrain_analysis_outputs/plots/current_eval_loss_by_scale.html) | Current aggregate eval loss by scale. |
| [midtrain_analysis_outputs/plots/raw_validation_curves_dashboard.html](../../midtrain_analysis_outputs/plots/raw_validation_curves_dashboard.html) | Raw validation curves across runs. |
| [midtrain_analysis_outputs/plots/final_validation_loss_vs_model_flops.html](../../midtrain_analysis_outputs/plots/final_validation_loss_vs_model_flops.html) | Endpoint validation loss vs base compute. |
| [midtrain_analysis_outputs/plots/endpoint_math_loss_by_recipe.html](../../midtrain_analysis_outputs/plots/endpoint_math_loss_by_recipe.html) | Math loss endpoint by recipe. |
| [midtrain_analysis_outputs/plots/endpoint_eval_loss_by_recipe.html](../../midtrain_analysis_outputs/plots/endpoint_eval_loss_by_recipe.html) | Aggregate eval endpoint by recipe. |
| [midtrain_analysis_outputs/plots/1e22_math_loss_predictions.html](../../midtrain_analysis_outputs/plots/1e22_math_loss_predictions.html) | Original 1e22 math-loss forecasts. |
| [midtrain_analysis_outputs/plots/1e22_eval_loss_predictions.html](../../midtrain_analysis_outputs/plots/1e22_eval_loss_predictions.html) | Original 1e22 aggregate eval forecasts. |
| [midtrain_analysis_outputs/plots/normalized_math_loss_collapse_finished_1e20_1e21.html](../../midtrain_analysis_outputs/plots/normalized_math_loss_collapse_finished_1e20_1e21.html) | Normalized math-loss collapse diagnostic. |
| [midtrain_analysis_outputs/plots/math_vs_paloma_pareto.html](../../midtrain_analysis_outputs/plots/math_vs_paloma_pareto.html) | Math improvement vs Paloma retention tradeoff. |

Important caveat: some early `1e20` rows in this older analysis lineage used a
historical non-canonical base. The canonical Delphi model selection issue was
fixed later by using registry/HF collection paths rather than ad hoc GCS name
search.

### Retrospective Prediction Dashboards

| Artifact | What it contains | Headline |
|---|---|---|
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/summary.md](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/summary.md) | Independent prediction retrospective over cached endpoint/trajectory exports. | Small-only 1e22 math MAPE stayed bad at 10.70%; anchored 1e21 slope-drift improved to 5.62% but still could not explain the raw target. |
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/math_loss_endpoints.html](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/math_loss_endpoints.html) | Endpoint math loss view. | Shows the old endpoint target. |
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/baseline_1e22_error_heatmap.html](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/baseline_1e22_error_heatmap.html) | Error heatmap by mix/LR. | The raw target error is mix dependent, worst for p33m67. |
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/prediction_experiment_summary.html](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/prediction_experiment_summary.html) | Candidate prediction methods. | More methods helped, but did not make the old raw 1e22 target smooth. |
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/rank_selection_1e22.html](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/rank_selection_1e22.html) | Rank selection at 1e22. | Despite bad absolute error, rank selection still picked p33m67-lr67 with zero regret in most methods. |
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/prefix_prediction_1e22.html](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/prefix_prediction_1e22.html) | Prefix calibration at 1e22. | Late prefixes got close; this is not enough for early planning. |
| [midtrain_analysis_outputs/codex_midtrain_prediction_retro/baseline_metric_errors.html](../../midtrain_analysis_outputs/codex_midtrain_prediction_retro/baseline_metric_errors.html) | Error by metric. | At 1e22, math_val_loss was the outlier: 10.70% old-target MAPE, while aggregate eval and Paloma metrics were much lower. |

### Fit-Family And Resource-Axis Reports

| Artifact | What it contains | Headline |
|---|---|---|
| [sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report.html](../../sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report.html) | Broad fit family report with endpoint Chinchilla, log-log, parameter axes, data axes, base rows, and separate base/improvement forms. | Best combined old-target held-out endpoint MAE was 1.99%, but the focus K=0.20 1e22 prediction was still +13.81% error. |
| [sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_summary.csv](../../sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_summary.csv) | Summary table for the broad report. | Use for exact model-family ranking. |
| [sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only.html](../../sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only.html) | Same model families, iso-token only. | Best iso-token-only held-out endpoint MAE was 0.93%, and tok8b 1e22 focus error was +2.41%. |
| [sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla.html](../../sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla.html) | Two-resource Chinchilla variants. | Early attempt to separate pretraining compute and midtraining tokens. |
| [sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla.html](../../sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla.html) | Parameter-aware Chinchilla variants. | Added model parameter count. |
| [sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla.html](../../sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla.html) | Parameter and data-token Chinchilla variants. | Best resource-aware old-target report before clean eval. |

The best all-series old-target row in the broad fit-family report was:

| Model | Train rows | Held-out rows | Held-out MAE | K=0.20 1e22 actual | Prediction | Error |
|---|---:|---:|---:|---:|---:|---:|
| Chinchilla endpoints: N + D_pre + D_mid | 42 | 12 | 1.99% | 0.56102 | 0.638473 | +13.81% |

The best iso-token-only row was:

| Model | Train rows | Held-out rows | Held-out MAE | tok8b 1e22 actual | Prediction | Error |
|---|---:|---:|---:|---:|---:|---:|
| Chinchilla endpoints: N + D_pre + D_math | 35 | 10 | 0.93% | 0.72012 | 0.737454 | +2.41% |

This contrast was one of the first strong signs that model form was not the
main problem. The same fitting machinery worked on fixed-token data.

### Iso-Token And Token-Budget Plots

| Artifact | What it contains | Headline |
|---|---|---|
| [sk_midtrain_analysis_fable/delphi_isotoken_all_budgets_vs_isoflop.png](../../sk_midtrain_analysis_fable/delphi_isotoken_all_budgets_vs_isoflop.png) | All old-target iso-token budgets compared with old K=0.20 iso-FLOP. | Fixed budgets have small 1e22 errors; K=0.20 has the large +18.6% error. |
| [sk_midtrain_analysis_fable/isoflop_miss_is_token_artifact.png](../../sk_midtrain_analysis_fable/isoflop_miss_is_token_artifact.png) | Simplified explanatory static plot. | The old "miss" follows token-budget confounding. |
| [sk_midtrain_analysis_fable/error_vs_tokens.png](../../sk_midtrain_analysis_fable/error_vs_tokens.png) | Error vs token budget. | Shows the old-target error structure by token budget. |
| [sk_midtrain_analysis_fable/isotoken_endpoints.csv](../../sk_midtrain_analysis_fable/isotoken_endpoints.csv) | Endpoint table for iso-token runs. | Source data for old-target fixed-token fits. |
| [sk_midtrain_analysis_fable/isoflop_k020_endpoints.csv](../../sk_midtrain_analysis_fable/isoflop_k020_endpoints.csv) | Endpoint table for old K=0.20 iso-FLOP. | Source data for the red comparison line. |
| [sk_midtrain_analysis_fable/isotoken_scaling_fits.csv](../../sk_midtrain_analysis_fable/isotoken_scaling_fits.csv) | Old-target iso-token Chinchilla fits. | Exact fitted errors for the static figures. |

### Clean-Seen Final Reports

| Artifact | What it contains | Headline |
|---|---|---|
| [sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report.html](../../sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report.html) | K=0.20 p33m67 clean-seen fit-family report. | Best clean-seen held-out MAE 1.44%; 1e22 errors 0.74% to 3.90% across LR. |
| [sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_summary.csv](../../sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_summary.csv) | Summary table for the clean-seen K=0.20 report. | Exact fit ranking and 1e22 prediction columns. |
| [sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report.html](../../sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report.html) | Old 4plus vs clean-seen K=0.20 report. | Shows why only two primary targets should be exposed: old 4plus and clean-seen. |
| [sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.html](../../sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.html) | Unified clean-seen report for iso-token 1B/2B/4B/8B plus K=0.20. | Clean-seen fixed-budget 1e22 errors are -2.3% to -2.8%; clean K=0.20 is +2.83%. |
| [sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_vs_old_endpoint_scaling.png](../../sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_vs_old_endpoint_scaling.png) | Cleaner static old-vs-clean endpoint summary. | Use this for presentations: old target vs clean target by budget. |
| [sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling.png](../../sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling.png) | Final seen-partition plot: old full, retained clean, dropped contaminated. | Strongest final evidence: clean retained has small error; dropped contaminated retains large absolute miss. |
| [sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv](../../sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv) | Numeric summary for the seen-partition plot. | Use for exact absolute and percent errors. |

### Contamination Worktree Artifacts

The relevant contamination worktree is:

```text
/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam
```

Key logs and plans:

| Path | What it contains |
|---|---|
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/logbooks/nemotron_math_data.md` | Main contamination, decontamination, clean-seen, and seen-partition logbook. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/logbooks/nemotron_math_pplx_gap.md` | Per-document/per-token perplexity-gap mechanism study. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/projects/nemotron_math_val_decontamination.md` | Clean validation set project plan and canonical clean-seen pointers. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/projects/decon_val_eval_plan.md` | Original eval-only decontamination plan. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/docs/reports/marin-32b-retro.md` | Background on Bison/Mantis shuffle and math-data cleanup. |

Key contamination plots:

| Plot | Meaning |
|---|---|
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/4plus_jaccard_histograms.png` | 4plus-only validation/train fuzzy-overlap histogram. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/jaccard_histograms.png` | Earlier union-source Jaccard histogram. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/lsh_scurve_bandings.png` | LSH sensitivity/banding diagnostic. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/decon_val_loss_vs_cutoff_p33m67_lr0.33.png` | Old anchor vs paranoid decon cutoff sweep. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/decon_val_loss_vs_cutoff_p33m67_lr0.33_4plus.png` | 4plus-only corrected cutoff sweep. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/decon_val_loss_vs_cutoff_p33m67_lr0.33_4plus_vs_union.png` | 4plus-only vs union decon comparison. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/ppl_gap_per_band_vs_scale.png` | Loss by Jaccard band vs model scale. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/ppl_gap_per_doc_vs_scale.png` | Per-doc loss curves showing memorization of high-J docs. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/ppl_gap_per_token_j088-000bf3fd.png` | Token-level collapse on a high-J validation document. |
| `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/perplexity_gap_explorer.html` | Interactive document-level mechanism explorer. |

## Timeline

### 1. Initial LR/Mix Prediction Work

The first midtraining predictor work used 10B-token math CPT sweeps to learn
trajectory prediction and choose LR/mix settings. Early predictors were useful
for rank and late-prefix planning but not enough for blind endpoint extrapolation.
The thread then moved toward endpoint scaling: fit the small ladder, hold out
1e21/1e22, and ask whether the small-ladder law predicts the large models.

Important historical bug: some early "1e20" analysis used a wrong non-canonical
base (`isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5`) instead of canonical
Delphi (`isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6`). That contaminated
some early discussion but was not the final cause of the 1e22 K=0.20 miss. The
lesson was to use the registry/HF collection for Delphi model identity, not GCS
name search.

### 2. Original Small-Ladder Endpoint Fit

The small ladder had complete K=0.20 cells from 3e18 through 3e20. The analysis
fit endpoint forms on that ladder and scored held-out 1e21/1e22.

From [summary.md](../../midtrain_analysis_outputs/small_final_loss_scaling/summary.md):

| Form | 1e21 mean abs % | 1e22 mean abs % | Signed 1e22 mean |
|---|---:|---:|---:|
| pooled N x D_math + LR | 1.11% | 8.97% | -8.97% |
| per-recipe power | 1.06% | 10.70% | -10.70% |
| pooled broken power | 1.37% | 10.77% | -10.77% |
| pooled log-quadratic | 2.15% | 14.83% | -14.83% |
| per-recipe Chinchilla floor | 2.32% | 15.76% | -15.76% |

The sign convention there was observed minus predicted, so negative means the
model predicted too high a loss. The important point is that 1e21 was fine and
1e22 was not.

For old math validation in p33m67, the simple per-recipe small-ladder fit had
1e22 errors around -15% in that sign convention. Later Chinchilla-style
K=0.20-focused reports used prediction-minus-actual and showed the same issue
as +18% to +19%.

### 3. Base Models Before Midtraining

The base-model question was whether the scaling problem was already present
before midtraining. The answer from the local step-0 report is no: on the old
math validation target, base math loss vs pretraining compute was smooth.

Artifact:

- [first_math_loss_chinchilla.html](../../midtrain_analysis_outputs/small_final_loss_scaling/first_math_loss_chinchilla.html)
- [delphi_first_math_loss_scaling.py](../../scripts/analysis/delphi_first_math_loss_scaling.py)

The script treats `step=0` math eval loss as the base model's loss before any
midtraining data is seen. It averages across mix/LR rows within each scale, then
fits the same Chinchilla floor+power form through 3e20 and holds out 1e21/1e22.

The figure annotation reports:

| Series | Chinchilla held-out 1e21 | Chinchilla held-out 1e22 |
|---|---:|---:|
| base step-0 math loss | +0.7% | +2.4% |
| best-LR endpoint p33m67 | +2.9% | +18.6% |
| best-LR endpoint p50m50 | +2.4% | +16.4% |
| best-LR endpoint p67m33 | +1.7% | +12.9% |

This says the old target may have contained near-duplicates as a corpus property,
but the smoothness failure was not already a base-model scaling failure. The
failure emerged after midtraining exposure, and it was strongest in the most
math-heavy mix.

Later fit-family work also included base rows explicitly. The separate
base/improvement Chinchilla mode fit base rows at D=0 and then a saturating
midtraining improvement. It did not fix the old K=0.20 1e22 focus case; in the
broad report it had +15.79% focus error, worse than the best endpoint-only
resource models. That made "bad base component" a weak explanation.

### 4. The "FLOPs Only" Mislabel Was Wrong

We removed the "FLOPs only" framing because K=0.20 includes midtraining. It is
a fixed fraction of the base model's pretraining compute spent on midtraining.
Therefore larger base models receive many more midtraining tokens.

For p33m67 K=0.20, the approximate token budget spread was:

| Scale | K=0.20 midtraining tokens |
|---|---:|
| 3e18 | 0.245B |
| 1e22 | 32.1B |

That is about a 130x token-budget spread. Since p33m67 is 67% math, the math
token budget also grows with scale. A law fit only on base compute is therefore
mixing at least three variables:

1. Base model scale.
2. Pretraining compute/data.
3. Absolute midtraining math exposure.

This matters even without contamination. With contamination, it is worse because
larger models also get more opportunities to see near-duplicates of the old
validation documents.

### 5. Iso-Token Controls

The iso-token experiment fixed the midtraining token budget and swept the base
scale. The first 1B ladder already made the point: losses were monotone, no
1e22 crossover appeared, and the old K=0.20 +18.6% error became about -3.8% at
fixed 1B tokens.

The final old-target all-budget plot is:

- [delphi_isotoken_all_budgets_vs_isoflop.png](../../sk_midtrain_analysis_fable/delphi_isotoken_all_budgets_vs_isoflop.png)

Old-target fixed-token 1e22 errors were small and negative across budgets:

| Series | Old-target 1e22 error |
|---|---:|
| iso-token 500M | about -3% to -4% |
| iso-token 1B | about -3.8% |
| iso-token 2B | about -3.6% |
| iso-token 4B | about -3.7% |
| iso-token 8B | about -3.1% |
| K=0.20 iso-FLOP | +18.6% |

The exact clean-seen version is even clearer and is covered below.

### 6. Many Fitting Forms Did Not Save The Old Target

The broad fit-family work tried the natural next models:

- Endpoint Chinchilla fits over `N`, `D_pre`, `D_mid`, `D_math`, `C_pre`,
  `C_mid`, and `C_mid_math`.
- Log-log endpoint fits.
- Separate base loss plus midtraining improvement.
- Base rows with D=0.
- Parameters and data together.
- Iso-token-only versions of the same reports.

The result was not "we need a cleverer old-target fit." The result was:

1. These fits work well on iso-token data.
2. They still miss old K=0.20 1e22.
3. The old K=0.20 target is therefore a data/eval issue, not just an
   underconstrained functional form.

Best broad old-target fit:

| Model | Held-out endpoint MAE | K=0.20 1e22 error |
|---|---:|---:|
| Chinchilla endpoints: N + D_pre + D_mid | 1.99% | +13.81% |

Best iso-token-only fit:

| Model | Held-out endpoint MAE | tok8b 1e22 error |
|---|---:|---:|
| Chinchilla endpoints: N + D_pre + D_math | 0.93% | +2.41% |

This was a critical transition: the code and fitting helpers were good enough
to fit smooth series, but the old K=0.20 series was not a clean smooth series.

### 7. Contamination Investigation

The old validation set came from a deterministic Feistel carve-out from the
math cache. That gave deterministic identity but not fuzzy decontamination.

Original split properties:

- Math cache: `gs://marin-us-east5/tokenized/nemotron_cc_math_v1/4plus-2c5519`
- Validation sequences: 12,500
- `shuffle_before_trainval_split: true`
- Full Feistel index remap with fixed `PRNGKey(0)` before slicing validation.
- Training stream used block shuffle with Feistel block permutation.

Exact duplicate scan found no exact duplicate doc hashes. That was misleading.
Fuzzy MinHash/LSH and exact 5-char-shingle Jaccard verification showed substantial
near-duplicate overlap.

Key counts from the contamination log:

| Jaccard cutoff | Val docs with train near-dup | Share of val docs |
|---|---:|---:|
| >= 0.50 | 20,831 / 57,243 | 36.4% |
| >= 0.75 | 9,757 / 57,243 | 17.0% |

At Jaccard >= 0.75, the implicated validation mass was:

| Unit | Count |
|---|---:|
| Validation docs | 9,757 / 57,243 |
| Validation windows | 6,839 / 12,500 |
| Validation tokens | 9.53M / 51.20M |

Actual exposure replay showed scale dependence for p33m67 K=0.20:

| Scale | Combined exposed val tokens |
|---|---:|
| 3e18 | 0.635M |
| 1e21 | 10.282M |
| 1e22 | 20.165M |

Across mixes at 1e22, the exposure tracked math fraction:

| Mix | Exposed val tokens |
|---|---:|
| p33m67 | 20.165M |
| p50m50 | 17.281M |
| p67m33 | 14.015M |

This matched the old-target anomaly: p33m67 had the largest 1e22 miss, p50m50
was next, p67m33 was smaller.

### 8. Paranoid Decon Evals

The first decontamination evals used cutoff datasets such as `j050`, `j075`,
and `j090`, meaning "keep documents below this max train Jaccard cutoff" in the
paranoid set convention.

Important cache roots:

```text
gs://marin-us-east5/tokenized/nemotron_math_val_decon/j050/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon/j075/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon/j090/validation/
```

Corrected 4plus-only decon roots:

```text
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j050/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j055/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j060/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j065/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j070/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j075/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j080/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j085/validation/
gs://marin-us-east5/tokenized/nemotron_math_val_decon_4plus/j090/validation/
```

For p33m67 lr0.33, the old anchor vs decon result inverted with scale:

| Scale | Old anchor | j050 | Anchor - j050 |
|---|---:|---:|---:|
| 3e18 | 1.4720 | 1.3597 | +0.1123 |
| 1e21 | 0.8104 | 0.7612 | +0.0492 |
| 1e22 | 0.5727 | 0.6126 | -0.0399 |

The 1e21->1e22 improvement on original val was 0.2377 nats, but only 0.1486
nats on j050. That means about 37% of the apparent original-val gain from 1e21
to 1e22 was contamination-driven in this slice.

The fine cutoff sweep showed the high-Jaccard behavior was not just random noise:
at 1e22 the model did unusually well on high-J documents, while at small scale
those documents were hard. That is what a memorization/exposure effect predicts.

### 9. Mechanism: High-J Documents Become Easy At 1e22

The document/per-token study answered why the decon gap behaved strangely. High
Jaccard documents are intrinsically harder at small scale, but at 1e22 they can
become the easiest documents because the model has effectively seen close twins.

Key artifacts:

- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/ppl_gap_per_band_vs_scale.png`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/ppl_gap_per_doc_vs_scale.png`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/ppl_gap_per_token_j088-000bf3fd.png`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/perplexity_gap_explorer.html`

Band-level examples:

| Band | 1e21 loss | 1e22 loss | Drop |
|---|---:|---:|---:|
| clean | 0.647 | 0.536 | -0.110 |
| j088 curated | 1.460 | 0.728 | -0.732 |

The j088 drop was about 6.7x the clean drop. The flagship high-J document
`000bf3fd` ("Unit 14: Time is Money") went from high loss at small scale to
near-memorized at 1e22. At token level, the median token loss collapsed from
about 1.99 to 0.003 nats, and 78% of tokens were below 0.05 nats at 1e22.

The first binary exposure hypothesis was too simple. Later cross-reference
analysis showed that the amount of exposure mattered more than max Jaccard alone:
near-duplicate multiplicity had stronger correlation with 1e22 loss than max-J.

### 10. Actual-Seen Clean Validation

The paranoid cutoff sets were useful but not the final answer, because "has a
near-duplicate somewhere in the corpus" is not the same as "this model actually
saw the near-duplicate during this run." The final clean-seen set was built
against actual seen training documents for the 1e22 p33m67 K=0.20 run.

Canonical clean-seen cache:

```text
gs://marin-us-east5/tokenized/nemotron_math_val_clean_seen_1e22_p33m67_k020/validation
```

Canonical manifest:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/seen_docs/1e22_p33m67_k020_math/val_decon/clean_val_against_contaminated_seen_docs/manifest.json
```

Clean-seen size:

| Unit | Count |
|---|---:|
| Kept docs | 3,367 |
| Tokens | 2,265,243 |
| Eval sequences | 553 |

Scope caveat: this set is clean against the actual Datakit contamination from
the 1e22 p33m67 K=0.20 training run. For other mixes it is a fixed benchmark,
not a per-mix actual-seen clean set.

K=0.20 p33m67 clean-seen eval sweep output:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.json
```

The sweep completed 36/36 jobs. The best clean-seen 1e22 row was lr0.67:

| Run family | 1e22 clean_seen_loss |
|---|---:|
| p33m67 K=0.20 lr0.67 | 0.8233972191810608 |

The clean-seen K=0.20 fit-family report then showed:

| Fit | Held-out MAE | 1e22 lr0.33 | 1e22 lr0.50 | 1e22 lr0.67 | 1e22 lr0.83 |
|---|---:|---:|---:|---:|---:|
| pooled Chinchilla + LR quadratic: N + D_pre | 1.44% | +3.90% | +2.47% | +1.33% | +0.74% |

That is the cleanest direct answer for the original K=0.20 ladder: after changing
the validation target to actual-seen clean, the 1e22 errors are low single digit.

### 11. Clean-Seen Iso-Token Re-Evals

The iso-token clean-seen evals were run for 1B, 2B, 4B, and 8B. The corrected
1B 1e22 row is from:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/delphi-1e22-p33m67-tok1b-lr50-a008/step-237/metrics.jsonl/eval_results.json
```

Canonical all-isotoken summaries:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.json
```

The final summary had all 36 rows and `partial_rows: []`.

Clean-seen 1e22 endpoint errors from the unified report:

| Series | Clean 1e22 actual | Prediction | Error |
|---|---:|---:|---:|
| iso-token 1B | 0.996622 | 0.972886 | -2.38% |
| iso-token 2B | 0.958299 | 0.936194 | -2.31% |
| iso-token 4B | 0.926756 | 0.901372 | -2.74% |
| iso-token 8B | 0.894756 | 0.869519 | -2.82% |
| K=0.20 iso-FLOP | 0.824991 | 0.848331 | +2.83% |

Old-target comparison from the same report:

| Series | Old 1e22 actual | Prediction | Error |
|---|---:|---:|---:|
| iso-token 1B | 0.917658 | 0.883257 | -3.75% |
| iso-token 2B | 0.856535 | 0.825790 | -3.59% |
| iso-token 4B | 0.794952 | 0.765342 | -3.72% |
| iso-token 8B | 0.720246 | 0.698201 | -3.06% |
| K=0.20 iso-FLOP | 0.561143 | 0.665313 | +18.56% |

This is the cleanest combined view:

1. Fixed-token scaling is smooth on old and clean targets.
2. K=0.20 old target is pathological.
3. K=0.20 clean target is smooth.

### 12. Seen-Partition Complement

The final high-information check evaluated the dropped contaminated complement
directly. This answered: if retained clean fixes the error, does the dropped set
retain the bad behavior?

Dropped complement cache:

```text
gs://marin-us-east5/tokenized/nemotron_math_val_dropped_seen_1e22_p33m67_k020
```

Output root:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_seen_partition_1e22_k020_lr50
```

The clean and dropped document sets partitioned the original validation docs:

| Set relation | Count |
|---|---:|
| clean docs | 3,367 |
| dropped docs | 53,876 |
| intersection | 0 |
| union | 57,243 |

Final 1e22 row from the eval run:

| Metric | Loss |
|---|---:|
| dropped contaminated loss | 0.665261 |
| old 4plus anchor loss | 0.561143 |
| eval/loss aggregate | 0.634742 |

Fit summary:

| Target | 1e22 actual | Prediction | Loss error | Percent error |
|---|---:|---:|---:|---:|
| old full 4plus | 0.561143 | 0.665313 | +0.104170 | +18.56% |
| retained clean | 0.824991 | 0.848331 | +0.023340 | +2.83% |
| dropped contaminated | 0.665261 | 0.765120 | +0.099859 | +15.01% |

The absolute-error view is the key. Old full and dropped contaminated both miss
by about 0.10 loss, while retained clean misses by about 0.023. The percent
errors differ partly because the denominators differ. This supports the
contamination hypothesis more directly than the percent plot alone.

Important caveat: old full 4plus is not simply a weighted aggregate of the new
clean and dropped eval targets in these outputs. The old full 1e22 loss
(`0.561143`) is below both the dropped loss (`0.665261`) and retained clean loss
(`0.824991`), so it is a distinct eval target/implementation path, not just
`clean U dropped` averaged in the same way. Do not interpret the three rows as
a perfect additive decomposition of the same metric. Interpret them as three
diagnostic targets measured in comparable sweeps.

## Shuffle Narrative

There were two separate shuffle stories.

### Bison/Mantis Shuffle Issue

The 32B Bison/Mantis work found that a cheap affine/linear-congruential
permutation could create correlated data ordering in cooldown training. Mantis
moved to Feistel shuffle and a cleaner math mix, and the training pathology
disappeared. That lesson is documented in:

```text
/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/docs/reports/marin-32b-retro.md
```

That is an important operational lesson, but it is not the explanation for the
Delphi K=0.20 prediction miss.

### Delphi Validation Split

For Delphi midtraining:

- The validation carve-out used full Feistel permutation with fixed seed before
  taking the validation tail.
- The training stream used block shuffle with Feistel block permutation for I/O
  locality.
- The validation identity was stable and byte-identical across the sweep.

The split was randomized and deterministic, but it was not fuzzy-decontaminated
against the training side. That is the distinction that matters. The shuffle did
not fail to randomize; the corpus contained near-duplicate math documents, and
random splitting put near twins on both sides.

Therefore the final statement should be:

> The shuffle choice did not break the Delphi base models or the validation
> identity. The validation set was stable. What failed was treating a randomly
> carved math validation split from a redundant corpus as a clean held-out target
> after scale-dependent math midtraining exposure.

## Corrected Hypotheses And Mistakes

### Corrected: "FLOPs Only"

K=0.20 is not FLOPs-only. It includes midtraining. Removing the "FLOPs only"
label was correct because it hid the token-budget confound.

### Corrected: Mode Inconsistency

At one point the old 1e21/1e22 anomaly was suspected to come from mode
inconsistency between CPT and true cooldown. Later audit corrected this: the
K=0.20 1e21/1e22 large runs were CPT/model-only fresh-optimizer runs like the
small K=0.20 ladder. Mode inconsistency was not the main explanation.

### Corrected: Base Loss Component

Adding a base-loss component was a reasonable modeling attempt. It did not solve
the raw K=0.20 miss. The step-0 base math loss itself extrapolated smoothly, and
separate base/improvement forms did not rescue the old target.

### Corrected: Binary Exposure

The first contamination interpretation was too binary: did the model see a
near-duplicate or not? The mechanism study showed dose matters. Near-duplicate
multiplicity and actual exposure count better explain very low 1e22 losses on
some high-J validation documents.

### Corrected: Paranoid Cutoff Semantics

The decon `jXXX` sets and the per-band PPL plots use different semantics:

- Decon `jXXX` means cumulative keep set: keep docs whose max train Jaccard is
  below `0.XX`.
- PPL band `jXXX` means a disjoint band around that Jaccard level.

Mixing those interpretations caused confusion during the middle of the thread.

### Corrected: Stale Iso-Token 1B Row

The 1B iso-token 1e22 row was stale until the corrected run completed:

```text
delphi-1e22-p33m67-tok1b-lr50-a008
step-237
```

The final clean-seen isotoken summary includes all 36 rows and has no partial
rows.

### Corrected: Too Many Old Validation Targets In One Plot

Early old-vs-clean plots exposed old `eval_loss`, old macro loss, old 4plus
anchor, and clean-seen together. That was confusing. The clean comparison should
focus on:

1. old 4plus validation anchor (`anchor_4plus_loss` / old 4plus), and
2. new clean-seen validation (`clean_seen_loss`).

The diagnostic CSVs can keep extra columns, but the explanatory plots should not
lead with them.

## Why The Final Evidence Supports The Hypothesis

The contamination/token-budget hypothesis predicts:

1. Fixed-token runs should not show the huge 1e22 old-target acceleration.
2. A clean eval target should reduce the K=0.20 1e22 extrapolation error.
3. The contaminated/dropped subset should retain a large error.
4. The effect should be strongest in the most math-heavy mix.
5. High-J documents should become disproportionately easy at 1e22.
6. Base step-0 scaling should remain smooth if the effect is caused by
   midtraining exposure rather than the base model.

What we observed:

| Prediction | Observed? | Evidence |
|---|---|---|
| Fixed-token runs are smooth | Yes | Old-target iso-token errors around -3% to -4%; clean-seen iso-token errors around -2.3% to -2.8%. |
| Clean target reduces K=0.20 error | Yes | lr0.50 K=0.20 1e22 old +18.56%; clean +2.83%. |
| Dropped subset retains large miss | Yes | dropped contaminated +0.0999 loss error, close to old full +0.1042. |
| Effect tracks math dose | Yes | p33m67 had highest actual exposure and largest old-target miss; p67m33 smaller. |
| High-J docs become easy at 1e22 | Yes | PPL band/doc/token studies show high-J docs collapse much more than clean docs. |
| Base step-0 scaling is smooth | Yes | Base step-0 Chinchilla held-out errors +0.7% at 1e21 and +2.4% at 1e22. |

That is why the final narrative is strong: multiple independent checks point in
the same direction, and the final clean-seen/dropped-partition experiment is a
direct diagnostic, not only a modeling change.

## Recommended Current Reading Order

For someone trying to understand the whole journey:

1. Open the original public report:
   <https://ahmeda14960.github.io/delphi-midtraining/?v=c86be93c>
2. Read the compact original endpoint summary:
   [summary.md](../../midtrain_analysis_outputs/small_final_loss_scaling/summary.md)
3. Open the base step-0 plot:
   [first_math_loss_chinchilla.html](../../midtrain_analysis_outputs/small_final_loss_scaling/first_math_loss_chinchilla.html)
4. Open the old-target iso-token plot:
   [delphi_isotoken_all_budgets_vs_isoflop.png](../../sk_midtrain_analysis_fable/delphi_isotoken_all_budgets_vs_isoflop.png)
5. Read the broad fit-family summary:
   [delphi_midtraining_fit_family_report_summary.csv](../../sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_summary.csv)
6. Open the contamination histograms and PPL mechanism plots in the contamination
   worktree:
   `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/plots/`
7. Open the clean-seen K=0.20 report:
   [delphi_k020_clean_seen_fit_family_report.html](../../sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report.html)
8. Open the clean-seen iso-token unified report:
   [delphi_isotoken_clean_seen_unified_report.html](../../sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.html)
9. Open the seen-partition final plot:
   [delphi_k020_seen_partition_scaling.png](../../sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling.png)

## Source Logbooks

Local midtrain worktree:

- [delphi_midtraining_visualization.md](delphi_midtraining_visualization.md)
- [debug_midtrain.md](debug_midtrain.md)
- [midtraining_delphi.md](midtraining_delphi.md)
- [midtraining_redesign.md](midtraining_redesign.md)
- [true_midtraining.md](true_midtraining.md)
- [delphi_true_cooldown.md](delphi_true_cooldown.md)
- [nemotron_math_data.md](nemotron_math_data.md)

Contamination worktree:

- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/logbooks/nemotron_math_data.md`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/logbooks/nemotron_math_pplx_gap.md`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/projects/nemotron_math_val_decontamination.md`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/.agents/projects/decon_val_eval_plan.md`
- `/Users/ahmed/code/marin/.claude/worktrees/nemotron_contam/docs/reports/marin-32b-retro.md`

## Canonical Data Roots

Old K=0.20 clean-seen summary:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.json
```

Clean-seen iso-token summary:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.json
```

Seen-partition eval root:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_seen_partition_1e22_k020_lr50
```

Clean-seen validation cache:

```text
gs://marin-us-east5/tokenized/nemotron_math_val_clean_seen_1e22_p33m67_k020/validation
```

Dropped seen validation cache:

```text
gs://marin-us-east5/tokenized/nemotron_math_val_dropped_seen_1e22_p33m67_k020
```

Perplexity-gap mechanism root:

```text
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/perplexity_gap/
```

## Remaining Caveats

1. The clean-seen cache is canonical for actual seen docs in the 1e22 p33m67
   K=0.20 run. It is not automatically per-mix clean for p50m50 or p67m33.
2. The clean-seen set is much smaller than the old validation anchor. It is the
   correct diagnostic for this contamination question, but future production
   validation should be built clean from the start with enough mass.
3. The old full 4plus anchor, retained clean target, and dropped target are not
   a perfect additive decomposition of one identical eval implementation. Use
   them as diagnostic targets, not as exact mixture arithmetic.
4. Parameter/data-aware fits can still be collinear because Delphi model size,
   pretraining compute, and pretraining tokens are coupled by design.
5. A 3e20->1e22 extrapolation is still a long jump. Clean validation fixes the
   confound, but it does not remove the general risk of long-horizon scaling
   extrapolation.

## Final One-Sentence Narrative

The Delphi midtraining scaling law looked rough because the old math validation
target combined an iso-FLOP token-budget confound with scale-dependent exposure
to near-duplicate math validation content; after fixed-token controls and
actual-seen clean validation, the midtraining endpoint fits are smooth again.
