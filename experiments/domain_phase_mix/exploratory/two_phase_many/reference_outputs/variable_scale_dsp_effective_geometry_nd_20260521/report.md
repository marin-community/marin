# Variable-Scale DSP on Marin ND Scaling Data

## Data

- Source: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`
- Metric: `eval/uncheatable_eval/bpb`
- Rows: `641` labeled rows for grouped OOF.

## Form

This is a screening evaluation: the per-domain DSP geometry
`rho_i`, `tau_i`, and `gamma` is frozen from `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510/dsp_effective_exposure_penalty_nnls/model.json`.
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
| dsp_vs_centered_no_amp          |               163 | 0.026801 | 0.015884 |   0.928529 |  0.940347 |      0.041499 |       0.625000 |                     0.971486 |
| dsp_vs_centered_shared_amp      |               164 | 0.027115 | 0.015713 |   0.923339 |  0.939119 |      0.041499 |       0.750000 |                     0.976795 |
| dsp_vs_centered_split_amp       |               165 | 0.026334 | 0.015310 |   0.927391 |  0.942553 |      0.023235 |       0.750000 |                     0.975816 |
| dsp_vs_centered_exposure_scaled |               164 | 0.026580 | 0.015663 |   0.930388 |  0.941444 |      0.041499 |       0.625000 |                     0.974999 |

## Leave-One-Scale-Out Results

| model                           | scale      |   n |     rmse |   spearman |   regret_at_1 |   top8_overlap |
|:--------------------------------|:-----------|----:|---------:|-----------:|--------------:|---------------:|
| dsp_vs_centered_no_amp          | 130m_2p6b  |  39 | 0.011688 |   0.822065 |      0.000000 |       0.625000 |
| dsp_vs_centered_no_amp          | 1_2b_24b   |   4 | 0.016118 |   0.800000 |      0.000000 |       1.000000 |
| dsp_vs_centered_no_amp          | 300m_6b    | 268 | 0.018335 |   0.742569 |      0.000000 |       0.500000 |
| dsp_vs_centered_no_amp          | 520m_10p4b |  37 | 0.012619 |   0.861546 |      0.004697 |       0.750000 |
| dsp_vs_centered_no_amp          | 60m_1p2b   | 293 | 0.044511 |   0.747165 |      0.035741 |       0.375000 |
| dsp_vs_centered_shared_amp      | 130m_2p6b  |  39 | 0.073287 |  -0.035020 |      0.000000 |       0.250000 |
| dsp_vs_centered_shared_amp      | 1_2b_24b   |   4 | 0.017477 |   0.400000 |      0.041499 |       1.000000 |
| dsp_vs_centered_shared_amp      | 300m_6b    | 268 | 0.016832 |   0.747214 |      0.000000 |       0.500000 |
| dsp_vs_centered_shared_amp      | 520m_10p4b |  37 | 0.009769 |   0.901138 |      0.004697 |       0.625000 |
| dsp_vs_centered_shared_amp      | 60m_1p2b   | 293 | 0.044845 |   0.747852 |      0.035741 |       0.375000 |
| dsp_vs_centered_split_amp       | 130m_2p6b  |  39 | 0.071066 |  -0.059717 |      0.043722 |       0.250000 |
| dsp_vs_centered_split_amp       | 1_2b_24b   |   4 | 0.019148 |   0.800000 |      0.023235 |       1.000000 |
| dsp_vs_centered_split_amp       | 300m_6b    | 268 | 0.023482 |   0.759040 |      0.000000 |       0.500000 |
| dsp_vs_centered_split_amp       | 520m_10p4b |  37 | 0.010369 |   0.948554 |      0.004697 |       0.875000 |
| dsp_vs_centered_split_amp       | 60m_1p2b   | 293 | 0.051167 |   0.747130 |      0.112129 |       0.375000 |
| dsp_vs_centered_exposure_scaled | 130m_2p6b  |  39 | 0.010102 |   0.870648 |      0.000000 |       0.625000 |
| dsp_vs_centered_exposure_scaled | 1_2b_24b   |   4 | 0.033784 |   0.800000 |      0.023235 |       1.000000 |
| dsp_vs_centered_exposure_scaled | 300m_6b    | 268 | 0.021898 |   0.745262 |      0.000000 |       0.500000 |
| dsp_vs_centered_exposure_scaled | 520m_10p4b |  37 | 0.012964 |   0.899004 |      0.004697 |       0.875000 |
| dsp_vs_centered_exposure_scaled | 60m_1p2b   | 293 | 0.069215 |   0.734129 |      0.327265 |       0.125000 |

## Decoded Parameter Summary

|    alpha |     beta |    delta | variant                             |   gamma_benefit |   gamma_saturation |   gamma_penalty |   kappa_benefit |   kappa_penalty |    omega |   rho_min |   rho_median |   rho_max |   tau_min |   tau_median |   tau_max |   r1_min |   r1_median |   r1_max | frozen_domain_geometry   |   fitted_nonlinear_param_count | model                           |   parameter_count | optimizer_success   | optimizer_message                                    |
|---------:|---------:|---------:|:------------------------------------|----------------:|-------------------:|----------------:|----------------:|----------------:|---------:|----------:|-------------:|----------:|----------:|-------------:|----------:|---------:|------------:|---------:|:-------------------------|-------------------------------:|:--------------------------------|------------------:|:--------------------|:-----------------------------------------------------|
| 0.552153 | 0.097611 | 0.322588 | dsp_effective_exposure_penalty_nnls |        0.000000 |          14.362237 |       14.362237 |        0.000000 |        0.000000 | 0.000000 |  0.021949 |     1.856430 |  2.000000 | -1.212493 |     1.700192 |  6.111360 |      nan |         nan |      nan | True                     |                              3 | dsp_vs_centered_no_amp          |               163 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 0.492231 | 0.102152 | 0.145835 | dsp_effective_exposure_penalty_nnls |        0.000000 |          14.362237 |       14.362237 |       -0.436509 |       -0.436509 | 0.000000 |  0.021949 |     1.856430 |  2.000000 | -1.212493 |     1.700192 |  6.111360 |      nan |         nan |      nan | True                     |                              4 | dsp_vs_centered_shared_amp      |               164 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 0.451696 | 0.108237 | 0.007699 | dsp_effective_exposure_penalty_nnls |        0.000000 |          14.362237 |       14.362237 |       -0.578646 |       -0.202417 | 0.000000 |  0.021949 |     1.856430 |  2.000000 | -1.212493 |     1.700192 |  6.111360 |      nan |         nan |      nan | True                     |                              5 | dsp_vs_centered_split_amp       |               165 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| 0.671081 | 0.121557 | 0.339495 | dsp_effective_exposure_penalty_nnls |        0.000000 |          14.362237 |       14.362237 |        0.000000 |        0.000000 | 0.086643 |  0.021949 |     1.856430 |  2.000000 | -1.212493 |     1.700192 |  6.111360 |      nan |         nan |      nan | True                     |                              4 | dsp_vs_centered_exposure_scaled |               164 | True                | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## Interpretation

- Centering DSP features against proportional at the same exact `(N,D)` is important: it leaves the global scale head to model the proportional trajectory and asks DSP to model relative mixture effects.
- All four variants materially improve grouped OOF rank over the standalone repetition-aware mixture scaling-law adaptations tested separately.
- The split-amplitude variant has the best grouped OOF RMSE and regret among this screen, but its leave-130M and leave-60M behavior is worse than the no-amplitude version.
- The no-amplitude centered form is the most stable leave-one-scale-out candidate in this screen.
- A full retune of DSP domain geometry across ND would require analytic/autodiff gradients or substantially more optimizer time; finite-difference L-BFGS over 80+ nonlinear domain parameters is not practical for quick iteration.
