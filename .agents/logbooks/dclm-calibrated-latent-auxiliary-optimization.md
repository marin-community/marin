# DCLM-Calibrated Latent Auxiliary Optimization

## Scope

Goal: test whether auxiliary smooth metrics and latent representations can improve heldout DCLM Core v2 prediction and ranking beyond direct DCLM-only smooth proxies, while keeping the final optimization target DCLM-anchored.

Primary target: DCLM Core v2 hard macro and component scores on the 300M swarm.

Operational constraint: first phase uses existing 300M observations only. No new training run should be proposed until the modeling validation shows a credible heldout DCLM gain over direct DCLM baselines and proportional.

## Starting Artifacts

- Updated DCLM matrix and proportional-noise rows: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/raw_metric_matrix_300m_dclm_updated_20260615/`
- Direct all-22 DCLM smooth DSP diagnostics: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_all22_smooth_dsp_300m_20260614_repeatcopy128/`
- Existing DCLM latent-factor DSP diagnostics: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_latent_factor_dsp_300m_20260614_repeatcopy128/`

## Initial Read

The updated matrix has complete hard DCLM Core v2 columns, complete selected dense smooth proxies for all 22 DCLM components, and 10 proportional-noise rows. Existing direct DCLM smooth DSP fits are reasonably fit in OOF diagnostics but only weakly coupled to hard DCLM. Existing unsupervised/posthoc latent-factor targets modestly improve same-panel hard coupling but are not yet credible submission targets because the best raw optima are far from proportional and the hard-DCLM support remains weak.

## Approaches To Test

1. Direct DCLM-only baseline: selected smooth proxy per component, macro z-score/rank-INT objectives, reliability-weighted variants, and hard-DCLM rank diagnostics.
2. DCLM-calibrated auxiliary latent factors: learn latent factors from reliable smooth metrics, then calibrate them to DCLM with heldout validation.
3. Supervised latent regression: PLS/CCA-style representations that are explicitly trained to predict DCLM components rather than generic eval variance.
4. Stacked residual model: direct DCLM DSP prediction plus auxiliary latent residual correction, fit only through out-of-fold predictions.
5. Guardrail optimization: optimize a DCLM-calibrated target while enforcing non-regression constraints on reliable DCLM components and high-SNR auxiliary metrics.

## Validation Rules

- Compare against direct DCLM-only baselines, not just against hard-DCLM raw noise.
- Use row-heldout validation for DCLM hard macro and smooth macro ranking.
- Use component-heldout validation to catch overfitting to DCLM component idiosyncrasies.
- Report whether auxiliary metrics improve heldout DCLM prediction/ranking; if not, keep them as guardrails only.
- Penalize far extrapolative optima unless they are supported by nearest-observed candidates or a planned validation run.

## Log

### 2026-06-15/16

- Created Fieldbook experiment `exp_01kv75sgfcev9q8vz70qr5e38g`.
- Starting CC design review before modeling code. Requested focus: statistical soundness of auxiliary latent factors, DCLM anchoring, heldout validation, and stop/go criteria.

### 2026-06-16 - CC design review

CC reviewed the existing DCLM matrix, direct smooth DSP diagnostics, and latent-factor diagnostics in read-only mode via `ctc ask` session `5f7d411c-d719-4685-b4dc-02577f7fcbdc`.

Key review points:
- Unsupervised auxiliary latent factors are not a sound objective by themselves; they are likely to optimize generic smooth capability rather than DCLM Core v2.
- The first gate should be a hard-DCLM noise-floor/reliability audit using the 10 proportional-noise rows.
- Auxiliary metrics should enter only through leakage-free DCLM-supervised calibration, residualization, or guardrails, and must beat direct DCLM-only baselines on heldout DCLM.
- Far extrapolative raw optima should be rejected unless supported by trust-region diagnostics and validation runs.

Immediate next test: T0 noise-floor/reliability audit for DCLM hard macro and per-component hard metrics.

### 2026-06-16 - T0 noise-floor gate

Ran `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_dclm_auxiliary_noise_floor.py`.

Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_calibrated_auxiliary_noise_floor_20260616/`.

Macro result:
- Hard DCLM macro signal SD across 241 signal rows excluding `baseline_stratified`: `0.009450`.
- Proportional-noise SD across 10 proportional repeats: `0.008230`.
- Signal/noise SD ratio: `1.148`.
- Reliability proxy: `0.242`.
- Best observed hard macro over proportional: `0.000491`, only `0.060` proportional-noise SDs.

Interpretation: the hard DCLM macro is target-limited/noise-limited at 300M. Direct hard-macro optimization is not credible on this panel. The next useful modeling test should use reliability-filtered or component-aware DCLM targets, then check whether auxiliary smooth metrics improve heldout DCLM prediction over DCLM-only smooth proxies and a general-factor decoy.

### 2026-06-16 - T1 anchored regression diagnostic

Ran `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_dclm_auxiliary_anchored_regression.py`.

Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_calibrated_auxiliary_anchored_regression_20260616/`.

Feature groups:
- `dclm_selected_smooth_22`: one selected dense smooth proxy per DCLM component.
- `dclm_all_complete_smooth`: all normalized, complete smooth/proxy columns for DCLM component task prefixes; 67 features.
- `non_dclm_auxiliary_smooth`: normalized, complete smooth/proxy columns outside DCLM task prefixes; 70 features.
- Combination groups with DCLM plus non-DCLM auxiliary features.

Main result:
- Official hard macro is predicted much better by `dclm_all_complete_smooth` than by `dclm_selected_smooth_22`.
- `dclm_all_complete_smooth` ridge/PLS10 reaches OOF Spearman about `0.786` and R2 about `0.62`.
- `dclm_selected_smooth_22` ridge reaches OOF Spearman about `0.279` and R2 about `0.09`.
- `non_dclm_auxiliary_smooth` ridge is similarly weak: OOF Spearman about `0.270` and R2 about `0.09`.
- Adding non-DCLM auxiliaries to DCLM-all does not improve official hard macro prediction and slightly lowers fit quality.
- Reliability-weighted hard macro is also best predicted by DCLM-all smooth features: OOF Spearman about `0.834` and R2 about `0.69`.

Interpretation: the promising direction is DCLM-task smooth enrichment and hard-calibrated denoising, not a generic auxiliary latent factor. Non-DCLM smooth metrics should be treated as guardrails unless stricter heldout validation shows they improve DCLM prediction.

Next test: build an out-of-fold hard-calibrated DCLM denoised target from `dclm_all_complete_smooth`, then fit and compare DSP on that target with trust-region constraints.

### 2026-06-16 - CC follow-up on T0/T1

CC reviewed the T0/T1 outputs in read-only mode via `ctc ask` session `e8a6067b-1c04-48a6-b68d-e6eab31c24a0`.

Key critique:
- T0 says the hard macro has low repeatability under the proportional-noise denominator, while T1 says DCLM-all smooth features predict hard macro with high row-heldout R2. This is either legitimate denoising or shared same-eval finite-sample coupling.
- Do not build a DSP target from the T1 denoiser until this fork is resolved.
- Best clean test would split eval examples so smooth features and hard accuracy use disjoint examples. If per-example logs are unavailable, run leave-task-out prediction: predict each DCLM component's hard score using smooth features from other DCLM tasks only.

Corrected framing: DCLM at 300M should be treated first as a guardrail/constraint surface around proportional. A DCLM-improving optimization target requires cleaner validation than current row-heldout same-task smooth-hard fits.

### 2026-06-16 - T2 leave-task-out diagnostic

Ran `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_dclm_leave_task_out_prediction.py`.

Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dclm_leave_task_out_prediction_20260616/`.

Aggregate result:
- Same-task smooth proxies are a strong upper bound: median component Spearman about `0.681` and median R2 about `0.473` for the PLS3 variant on the 12 components with enough same-task smooth columns; ridge over same-task smooth features reaches median Spearman about `0.488` and median R2 about `0.279` across all 22 components.
- Other DCLM-task smooth proxies transfer only weakly overall: ridge median Spearman about `0.201` and median R2 about `0.031`; PCA5+ridge reaches median Spearman about `0.211` and median R2 about `0.040`.
- Non-DCLM auxiliary smooth metrics are weaker still: ridge median Spearman about `0.148` and median R2 about `0.015`.
- Combining other-DCLM and non-DCLM auxiliaries reaches median Spearman about `0.235` and median R2 about `0.064`.

Component-level nuance:
- Cross-task transfer is meaningful for broad/general components such as HellaSwag 0/10 shot, ARC Easy, Lambada, BigBench QA Wikidata, and CoQA.
- Cross-task transfer is poor for several BigBench, commonsense, and reading-comprehension components.

Interpretation: the T1 hard-macro fit from `dclm_all_complete_smooth` is partly same-task smooth-hard reconstruction rather than evidence for a robust latent DCLM axis. This supports using DCLM smooth features as a denoising/guardrail surface, not yet as a standalone optimization target. Non-DCLM auxiliary metrics should remain guardrails unless a cleaner validation split shows they improve DCLM prediction.

Next decision: do not submit a new DCLM-optimized mixture from this evidence alone. The next credible tests are either an eval-example split for DCLM hard-vs-smooth calibration, if per-example logs can be recovered, or a trust-region candidate search that treats DCLM smooth surfaces as constraints around proportional and validates any candidate with heldout/scaled DCLM.

### 2026-06-16 - Readiness-weighted aggregate DSP diagnostic

Ran `experiments/domain_phase_mix/exploratory/two_phase_many/fit_readiness_weighted_aggregate_dsp.py`.

Output: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/readiness_weighted_aggregate_dsp_20260616/`.

This tests whether the heteroskedastic/readiness diagnostics from proportional-controllability improve a v4-style aggregate objective. The original collaborator-style 5-factor aggregate is the baseline. Four readiness variants reweight the original selected metric panel:

- `strict_steerable`: direct local/finite-effect steerable metrics only.
- `steerable_guardrail`: direct steerable metrics plus partial-weight robust guardrail metrics.
- `steerable_guardrail_stabilized`: direct/guardrail metrics plus stable Paloma and uncheatable BPB anchors.
- `broad_screened`: stabilized target plus tiny weight on fragile screened metrics.

Main fit result:

| target | active metrics | OOF Spearman | OOF R2 | raw optimum TV to proportional | raw optimum phase max weights |
|---|---:|---:|---:|---:|---:|
| `original_factor` | 0 | 0.8996 | 0.8017 | 0.8163 | 0.2672 / 0.4496 |
| `strict_steerable` | 2 | 0.8694 | 0.6923 | 0.9578 | 0.9131 / 0.9854 |
| `steerable_guardrail` | 3 | 0.8908 | 0.8014 | 0.7378 | 1.0000 / 0.3034 |
| `steerable_guardrail_stabilized` | 26 | 0.9055 | 0.8535 | 0.6536 | 0.2014 / 0.8990 |
| `broad_screened` | 35 | 0.9148 | 0.8638 | 0.6801 | 0.2653 / 0.7095 |

Interpretation:

- Readiness weighting does improve surrogate fit: `broad_screened` beats the original factor on OOF Spearman and R2.
- Direct-readiness-only optimization is not deployable. With only two active metrics, `strict_steerable` collapses to near-single-domain mixtures and overstates predicted gain.
- Stable BPB anchors are necessary. They prevent the objective from becoming a narrow local-control target and produce the best fit among the tested variants.
- The unconstrained raw optima are still extrapolative. Even the best screened/stabilized variants have TV around `0.65-0.68` from proportional and nearest observed mixture `baseline_proportional`, so they should not be submitted directly.

Decision: readiness-weighted aggregation is useful as an objective-construction diagnostic, but not yet a validation-ready mixture generator. The next optimization step should construct trust-region/path candidates from proportional toward the `broad_screened` and `steerable_guardrail_stabilized` optima, then compare predicted target gains and guardrail regressions before any MoE scaling validation.
