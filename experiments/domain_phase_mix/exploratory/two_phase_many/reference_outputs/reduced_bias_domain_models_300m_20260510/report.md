# 300M Reduced-Bias Domain GRP Models

## Setup

These local fits remove GRP family partitions, CC quality pairing, and scalar quality discounts.
Each of the 39 domains gets its own saturation rate and overexposure threshold. For fixed nonlinear
shape parameters, the linear benefit and penalty heads are solved by NNLS variable projection.

Compared variants:

- `domain_no_phase_penalty_157`: Raw total exposure, per-domain saturation and penalty, no phase term.
- `domain_phase_benefit_penalty_158`: Raw total exposure with global phase-1 benefit premium; penalties see raw exposure only.
- `domain_effective_exposure_penalty_158`: Domain-wise analogue of old GRP: phase-1 multiplier enters exposure for both benefit and penalty.
- `domain_phase_benefit_penalty_premium_159`: Raw total exposure with separate global phase-1 benefit and penalty premia.

## Results

| variant                                  |   total_param_count |   train_rmse |   cv_rmse |   oof_spearman |   cv_foldmean_regret_at_1 |   lower_tail_optimism |   raw_nearest_observed_tv |   raw_nearest_observed_value |   phase0_max_weight |   phase1_max_weight |
|:-----------------------------------------|--------------------:|-------------:|----------:|---------------:|--------------------------:|----------------------:|--------------------------:|-----------------------------:|--------------------:|--------------------:|
| old_grp_effective_exposure_no_l2         |                  42 |     0.010102 |  0.011141 |       0.804042 |                  0.007789 |              0.005019 |                  0.500000 |                     0.968801 |            0.025641 |            0.393906 |
| domain_no_phase_penalty_157              |                 157 |     0.014187 |  0.019480 |       0.306467 |                  0.005647 |              0.010248 |                  0.500000 |                     0.968801 |            0.939376 |            0.138239 |
| domain_phase_benefit_penalty_158         |                 158 |     0.005656 |  0.008848 |       0.899626 |                  0.000201 |              0.004497 |                  0.500000 |                     0.968801 |            0.956713 |            0.436980 |
| domain_effective_exposure_penalty_158    |                 158 |     0.004941 |  0.008024 |       0.905364 |                  0.000498 |              0.004637 |                  0.500000 |                     0.968801 |            0.203549 |            0.694892 |
| domain_phase_benefit_penalty_premium_159 |                 159 |     0.007269 |  0.010232 |       0.865899 |                  0.000976 |              0.006074 |                  0.500000 |                     0.968801 |            0.329440 |            0.376328 |

## Best Observed Rows By Prediction

| variant                                  | best_pred_observed_run                   |   best_pred_observed_pred_bpb |   best_pred_observed_actual_bpb |   best_pred_observed_actual_rank |   pred_top8_mean_actual_bpb |   pred_top8_best_actual_bpb |   actual_best_bpb | actual_best_run   |
|:-----------------------------------------|:-----------------------------------------|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|----------------------------:|------------------:|:------------------|
| domain_effective_exposure_penalty_158    | baseline_olmix_loglinear_uncheatable_bpb |                      0.952344 |                        0.956062 |                                3 |                    0.961868 |                    0.955440 |          0.955440 | run_00125         |
| domain_no_phase_penalty_157              | baseline_olmix_loglinear_uncheatable_bpb |                      0.957310 |                        0.956062 |                                3 |                    0.969375 |                    0.955440 |          0.955440 | run_00125         |
| domain_phase_benefit_penalty_158         | run_00125                                |                      0.955372 |                        0.955440 |                                1 |                    0.962583 |                    0.955440 |          0.955440 | run_00125         |
| domain_phase_benefit_penalty_premium_159 | baseline_olmix_loglinear_uncheatable_bpb |                      0.955297 |                        0.956062 |                                3 |                    0.963649 |                    0.955440 |          0.955440 | run_00125         |

## Interpretation

- Best CV RMSE row: `domain_effective_exposure_penalty_158` with cv_rmse=0.008024.
- Best OOF Spearman row: `domain_effective_exposure_penalty_158` with oof_spearman=0.905364.
- The strongest reduced-bias variants are good observed-row rankers, but their unconstrained raw optima are still off-manifold.
- Raw optima are diagnostic only; high nearest-observed TV or collapsed phase weights mean non-deployable.
- This script uses finite-difference L-BFGS-B for nonlinear variable-projection fitting because the NNLS inner solve is nonsmooth; fitted-mixture optimization uses analytic gradients.
