# DSP perturbation inclusion ablation

Objective: `eval/uncheatable_eval/bpb`.

Question: does adding the 55 proportional perturbation rows at a scale improve DSP generalization enough to justify including similar local interventions in future swarms?

Evaluation design:
- `original_only_oof`: standard OOF on the original swarm rows.
- `original_only_external`: model fit on original rows, evaluated on perturbation rows as an external local-intervention holdout.
- `augmented_oof_original_rows`: OOF on original rows, where all perturbation rows are always available in the training folds.
- `augmented_oof_perturbation_rows`: OOF on perturbation rows, where all original rows are always available in the training folds.
- Negative `rmse_delta_vs_baseline` means the augmented fit improved RMSE against the relevant baseline.

## Canonical DSP

| display_scale   | variant                        | eval_panel        | evaluation                      |          n |     rmse |   spearman |   rmse_delta_vs_baseline |   rmse_delta_paired_bootstrap_ci025 |   rmse_delta_paired_bootstrap_ci975 |
|:----------------|:-------------------------------|:------------------|:--------------------------------|-----------:|---------:|-----------:|-------------------------:|------------------------------------:|------------------------------------:|
| 60M/1.2B        | dsp_phase_benefit_penalty_nnls | original_rows     | original_only_oof               | 242.000000 | 0.009969 |   0.850313 |               nan        |                          nan        |                          nan        |
| 60M/1.2B        | dsp_phase_benefit_penalty_nnls | original_rows     | augmented_oof_original_rows     | 242.000000 | 0.009732 |   0.859136 |                -0.000238 |                           -0.000716 |                            0.000277 |
| 60M/1.2B        | dsp_phase_benefit_penalty_nnls | perturbation_rows | original_only_external          |  55.000000 | 0.007144 |   0.785065 |               nan        |                          nan        |                          nan        |
| 60M/1.2B        | dsp_phase_benefit_penalty_nnls | perturbation_rows | augmented_oof_perturbation_rows |  55.000000 | 0.006263 |   0.836147 |                -0.000881 |                           -0.001616 |                            0.000145 |
| 60M/1.2B        | dsp_phase_benefit_penalty_nnls | combined_rows     | augmented_random_oof            | 297.000000 | 0.009755 |   0.891721 |               nan        |                          nan        |                          nan        |
| 100M/6B         | dsp_phase_benefit_penalty_nnls | original_rows     | original_only_oof               | 242.000000 | 0.008835 |   0.898476 |               nan        |                          nan        |                          nan        |
| 100M/6B         | dsp_phase_benefit_penalty_nnls | original_rows     | augmented_oof_original_rows     | 242.000000 | 0.008841 |   0.902808 |                 0.000006 |                           -0.000290 |                            0.000306 |
| 100M/6B         | dsp_phase_benefit_penalty_nnls | perturbation_rows | original_only_external          |  55.000000 | 0.007833 |   0.763997 |               nan        |                          nan        |                          nan        |
| 100M/6B         | dsp_phase_benefit_penalty_nnls | perturbation_rows | augmented_oof_perturbation_rows |  55.000000 | 0.007006 |   0.770491 |                -0.000827 |                           -0.001322 |                           -0.000281 |
| 100M/6B         | dsp_phase_benefit_penalty_nnls | combined_rows     | augmented_random_oof            | 297.000000 | 0.008305 |   0.905552 |               nan        |                          nan        |                          nan        |

## Effective-exposure empirical comparator

| display_scale   | variant                             | eval_panel        | evaluation                      |          n |     rmse |   spearman |   rmse_delta_vs_baseline |   rmse_delta_paired_bootstrap_ci025 |   rmse_delta_paired_bootstrap_ci975 |
|:----------------|:------------------------------------|:------------------|:--------------------------------|-----------:|---------:|-----------:|-------------------------:|------------------------------------:|------------------------------------:|
| 60M/1.2B        | dsp_effective_exposure_penalty_nnls | original_rows     | original_only_oof               | 242.000000 | 0.007280 |   0.903576 |               nan        |                          nan        |                          nan        |
| 60M/1.2B        | dsp_effective_exposure_penalty_nnls | original_rows     | augmented_oof_original_rows     | 242.000000 | 0.007316 |   0.911618 |                 0.000035 |                           -0.000398 |                            0.000531 |
| 60M/1.2B        | dsp_effective_exposure_penalty_nnls | perturbation_rows | original_only_external          |  55.000000 | 0.004947 |   0.805556 |               nan        |                          nan        |                          nan        |
| 60M/1.2B        | dsp_effective_exposure_penalty_nnls | perturbation_rows | augmented_oof_perturbation_rows |  55.000000 | 0.004769 |   0.830592 |                -0.000178 |                           -0.000446 |                            0.000299 |
| 60M/1.2B        | dsp_effective_exposure_penalty_nnls | combined_rows     | augmented_random_oof            | 297.000000 | 0.006967 |   0.943314 |               nan        |                          nan        |                          nan        |
| 100M/6B         | dsp_effective_exposure_penalty_nnls | original_rows     | original_only_oof               | 242.000000 | 0.007106 |   0.919645 |               nan        |                          nan        |                          nan        |
| 100M/6B         | dsp_effective_exposure_penalty_nnls | original_rows     | augmented_oof_original_rows     | 242.000000 | 0.007911 |   0.902334 |                 0.000805 |                           -0.000338 |                            0.002703 |
| 100M/6B         | dsp_effective_exposure_penalty_nnls | perturbation_rows | original_only_external          |  55.000000 | 0.005113 |   0.789971 |               nan        |                          nan        |                          nan        |
| 100M/6B         | dsp_effective_exposure_penalty_nnls | perturbation_rows | augmented_oof_perturbation_rows |  55.000000 | 0.003454 |   0.762987 |                -0.001659 |                           -0.002049 |                           -0.001175 |
| 100M/6B         | dsp_effective_exposure_penalty_nnls | combined_rows     | augmented_random_oof            | 297.000000 | 0.007782 |   0.896481 |               nan        |                          nan        |                          nan        |

## Interpretation

Use perturbation rows if they improve perturbation-row OOF materially without degrading original-row OOF. If they only improve local perturbation rows and do not help original-row OOF, they are still useful as targeted gradient/intervention diagnostics, but should not replace broad randomized swarm points.
