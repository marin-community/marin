# Variable-Scale DSP on Marin ND Scaling Data

## Data

- Source: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`
- Metric: `eval/uncheatable_eval/bpb`
- Rows: `641` labeled rows for grouped OOF.

## Form

This is a screening evaluation: the per-domain DSP geometry
`rho_i`, `tau_i`, and `gamma` is frozen from `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_apple_repetition_variants_300m_20260514/dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls/model.json`.
The fit retunes the global scale exponents, optional scale-amplitude exponents, and the profiled linear head.

The baseline scale trajectory is modeled by:

$$g(N,D)=E+C(N/N_0)^{-\beta}+B(N/N_0)^\delta(D/D_0)^{-\alpha}.$$

DSP mixture features are centered against proportional at the same exact `(N,D)`:

$$\Delta S_i=S_i(w)-S_i(w_{\mathrm{prop}}), \qquad \Delta P_i=P_i(w)-P_i(w_{\mathrm{prop}}).$$

The tested additive form is:

$$\hat y(N,D,w)=g(N,D)-n^{\kappa_b}\sum_i a_i\Delta S_i+n^{\kappa_p}\sum_i p_i\Delta P_i.$$

The exposure-scaled variant additionally replaces DSP exposure `z_i` by `(D/D0)^omega z_i`.

## Grouped OOF Results

| model                           |   parameter_count |     rmse |      mae |   spearman |   pearson |   regret_at_1 |   top8_overlap |   predicted_actual_std_ratio |
|:--------------------------------|------------------:|---------:|---------:|-----------:|----------:|--------------:|---------------:|-----------------------------:|
| dsp_vs_centered_no_amp          |               163 | 0.030166 | 0.016952 |   0.910872 |  0.923518 |      0.041499 |       0.625000 |                     0.950052 |
| dsp_vs_centered_shared_amp      |               164 | 0.030462 | 0.017252 |   0.902250 |  0.922436 |      0.041499 |       0.750000 |                     0.962049 |
| dsp_vs_centered_split_amp       |               165 | 0.030662 | 0.017321 |   0.901084 |  0.921546 |      0.041499 |       0.750000 |                     0.965119 |
| dsp_vs_centered_exposure_scaled |               164 | 0.030166 | 0.016709 |   0.910978 |  0.923750 |      0.000000 |       0.750000 |                     0.957921 |

## Leave-One-Scale-Out Results

| model                           | scale      |   n |     rmse |   spearman |   regret_at_1 |   top8_overlap |
|:--------------------------------|:-----------|----:|---------:|-----------:|--------------:|---------------:|
| dsp_vs_centered_no_amp          | 130m_2p6b  |  39 | 0.014523 |   0.579150 |      0.000000 |       0.625000 |
| dsp_vs_centered_no_amp          | 1_2b_24b   |   4 | 0.039330 |  -0.600000 |      0.041499 |       1.000000 |
| dsp_vs_centered_no_amp          | 300m_6b    | 268 | 0.019658 |   0.540083 |      0.000000 |       0.625000 |
| dsp_vs_centered_no_amp          | 520m_10p4b |  37 | 0.017234 |   0.733760 |      0.004697 |       0.750000 |
| dsp_vs_centered_no_amp          | 60m_1p2b   | 293 | 0.044569 |   0.749729 |      0.060809 |       0.000000 |
| dsp_vs_centered_shared_amp      | 130m_2p6b  |  39 | 0.221720 |  -0.201822 |      0.000000 |       0.375000 |
| dsp_vs_centered_shared_amp      | 1_2b_24b   |   4 | 0.021445 |  -0.600000 |      0.041499 |       1.000000 |
| dsp_vs_centered_shared_amp      | 300m_6b    | 268 | 0.017289 |   0.555527 |      0.000000 |       0.625000 |
| dsp_vs_centered_shared_amp      | 520m_10p4b |  37 | 0.012336 |   0.835941 |      0.004697 |       0.625000 |
| dsp_vs_centered_shared_amp      | 60m_1p2b   | 293 | 0.044588 |   0.750513 |      0.060809 |       0.000000 |
| dsp_vs_centered_split_amp       | 130m_2p6b  |  39 | 0.222945 |  -0.203644 |      0.000000 |       0.375000 |
| dsp_vs_centered_split_amp       | 1_2b_24b   |   4 | 0.021647 |  -0.600000 |      0.041499 |       1.000000 |
| dsp_vs_centered_split_amp       | 300m_6b    | 268 | 0.017269 |   0.553867 |      0.000000 |       0.750000 |
| dsp_vs_centered_split_amp       | 520m_10p4b |  37 | 0.011928 |   0.834282 |      0.004697 |       0.625000 |
| dsp_vs_centered_split_amp       | 60m_1p2b   | 293 | 0.044126 |   0.752290 |      0.060809 |       0.000000 |
| dsp_vs_centered_exposure_scaled | 130m_2p6b  |  39 | 0.018267 |   0.651417 |      0.000000 |       0.500000 |
| dsp_vs_centered_exposure_scaled | 1_2b_24b   |   4 | 0.002413 |   1.000000 |      0.000000 |       1.000000 |
| dsp_vs_centered_exposure_scaled | 300m_6b    | 268 | 0.021236 |   0.605513 |      0.000000 |       0.250000 |
| dsp_vs_centered_exposure_scaled | 520m_10p4b |  37 | 0.020534 |   0.844476 |      0.000000 |       0.500000 |
| dsp_vs_centered_exposure_scaled | 60m_1p2b   | 293 | 0.044749 |   0.723265 |      0.060809 |       0.000000 |

## Decoded Parameter Summary

|    alpha |     beta |    delta | variant                                                       |   gamma_benefit |   gamma_saturation |   gamma_penalty |   kappa_benefit |   kappa_penalty |     omega |   rho_min |   rho_median |   rho_max |   tau_min |   tau_median |   tau_max |   r1_min |   r1_median |    r1_max | frozen_domain_geometry   |   fitted_nonlinear_param_count | model                           |   parameter_count | optimizer_success   | optimizer_message                                    |
|---------:|---------:|---------:|:--------------------------------------------------------------|----------------:|-------------------:|----------------:|----------------:|----------------:|----------:|----------:|-------------:|----------:|----------:|-------------:|----------:|---------:|------------:|----------:|:-------------------------|-------------------------------:|:--------------------------------|------------------:|:--------------------|:-----------------------------------------------------|
| 0.797097 | 0.083030 | 0.502540 | dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |        4.242625 |           1.000000 |        1.000000 |        0.000000 |        0.000000 |  0.000000 |  0.003385 |     0.104621 |  1.665686 | -2.000000 |     0.396748 |  6.843084 | 0.005455 |    0.247333 | 16.300788 | True                     |                              3 | dsp_vs_centered_no_amp          |               163 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 0.448663 | 0.115275 | 0.084759 | dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |        4.242625 |           1.000000 |        1.000000 |       -0.691868 |       -0.691868 |  0.000000 |  0.003385 |     0.104621 |  1.665686 | -2.000000 |     0.396748 |  6.843084 | 0.005455 |    0.247333 | 16.300788 | True                     |                              4 | dsp_vs_centered_shared_amp      |               164 | True                | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL     |
| 0.514499 | 0.130458 | 0.181012 | dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |        4.242625 |           1.000000 |        1.000000 |       -0.661605 |       -0.797873 |  0.000000 |  0.003385 |     0.104621 |  1.665686 | -2.000000 |     0.396748 |  6.843084 | 0.005455 |    0.247333 | 16.300788 | True                     |                              5 | dsp_vs_centered_split_amp       |               165 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 0.020000 | 0.020000 | 0.001000 | dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |        4.242625 |           1.000000 |        1.000000 |        0.000000 |        0.000000 | -0.677295 |  0.003385 |     0.104621 |  1.665686 | -2.000000 |     0.396748 |  6.843084 | 0.005455 |    0.247333 | 16.300788 | True                     |                              4 | dsp_vs_centered_exposure_scaled |               164 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Interpretation

- Centering DSP features against proportional at the same exact `(N,D)` is important: it leaves the global scale head to model the proportional trajectory and asks DSP to model relative mixture effects.
- All four variants materially improve grouped OOF rank over the standalone repetition-aware mixture scaling-law adaptations tested separately.
- The split-amplitude variant has the best grouped OOF RMSE and regret among this screen, but its leave-130M and leave-60M behavior is worse than the no-amplitude version.
- The no-amplitude centered form is the most stable leave-one-scale-out candidate in this screen.
- A full retune of DSP domain geometry across ND would require analytic/autodiff gradients or substantially more optimizer time; finite-difference L-BFGS over 80+ nonlinear domain parameters is not practical for quick iteration.
