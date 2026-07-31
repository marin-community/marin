# MCT-DSP Hybrid ND Evaluation

## Data

- Source: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`
- Metric: `eval/uncheatable_eval/bpb`
- Frozen DSP geometry: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510/dsp_effective_exposure_penalty_nnls/model.json`
- MCT reference: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/joint_model_refreshed_20260426/mct_lrq_no_barrier_canonical`

## Tested Forms

- `mct_dsp_anchor`: MCT centered scale scaffold plus frozen effective-exposure DSP anchor.
- `mct_dsp_split_amp`: adds centered benefit/penalty interaction terms with fitted N amplitudes.
- `mct_dsp_tau_shift`: shifts all DSP penalty thresholds by global `eta_N log(N/N0)+eta_D log(D/D0)`.
- `mct_dsp_apple_sat`: applies Apple-style shared-r1 repetition discount to saturation exposure only.

## Validation Metrics

| model                     | fit_protocol   | split                  |   n |     rmse |      mae |   spearman |   pearson |   regret_at_1 |   top8_overlap |
|:--------------------------|:---------------|:-----------------------|----:|---------:|---------:|-----------:|----------:|--------------:|---------------:|
| mct_lrq69_drop_no_barrier | seed7          | seed7_holdout          |  61 | 0.010075 | 0.007798 |   0.970703 |  0.990858 |      0.000000 |       0.750000 |
| mct_lrq69_drop_no_barrier | seed7          | fixed340_holdout       |  27 | 0.006143 | 0.005058 |   0.923687 |  0.940046 |      0.000000 |       0.750000 |
| mct_lrq69_drop_no_barrier | seed7          | random_supplement      |  34 | 0.012334 | 0.009973 |   0.868908 |  0.979299 |      0.000000 |       0.875000 |
| mct_lrq69_drop_no_barrier | leave900out    | all900_leave_scale_out |   4 | 0.010177 | 0.009138 |   1.000000 |  0.858613 |      0.000000 |       1.000000 |
| mct_dsp_anchor            | seed7          | seed7_holdout          |  61 | 0.014404 | 0.011201 |   0.959439 |  0.981819 |      0.000000 |       0.625000 |
| mct_dsp_anchor            | seed7          | fixed340_holdout       |  27 | 0.015417 | 0.011808 |   0.865079 |  0.862736 |      0.020488 |       0.625000 |
| mct_dsp_anchor            | seed7          | random_supplement      |  34 | 0.013546 | 0.010719 |   0.833155 |  0.962262 |      0.000000 |       0.750000 |
| mct_dsp_anchor            | leave900out    | all900_leave_scale_out |   4 | 0.015967 | 0.014545 |   0.400000 |  0.592800 |      0.000000 |       1.000000 |
| mct_dsp_split_amp         | seed7          | seed7_holdout          |  61 | 0.038992 | 0.022090 |   0.852089 |  0.851972 |      0.073465 |       0.375000 |
| mct_dsp_split_amp         | seed7          | fixed340_holdout       |  27 | 0.031967 | 0.026324 |   0.697192 |  0.691288 |      0.035817 |       0.500000 |
| mct_dsp_split_amp         | seed7          | random_supplement      |  34 | 0.043774 | 0.018727 |   0.658060 |  0.523538 |      0.151714 |       0.750000 |
| mct_dsp_split_amp         | leave900out    | all900_leave_scale_out |   4 | 0.140587 | 0.092484 |  -0.600000 | -0.255044 |      0.041499 |       1.000000 |
| mct_dsp_tau_shift         | seed7          | seed7_holdout          |  61 | 0.012815 | 0.009888 |   0.967002 |  0.985593 |      0.000000 |       0.750000 |
| mct_dsp_tau_shift         | seed7          | fixed340_holdout       |  27 | 0.012993 | 0.010288 |   0.884615 |  0.891244 |      0.000000 |       0.750000 |
| mct_dsp_tau_shift         | seed7          | random_supplement      |  34 | 0.012672 | 0.009570 |   0.867074 |  0.966927 |      0.000000 |       0.750000 |
| mct_dsp_tau_shift         | leave900out    | all900_leave_scale_out |   4 | 0.023111 | 0.017579 |   0.800000 |  0.669820 |      0.000000 |       1.000000 |
| mct_dsp_apple_sat         | seed7          | seed7_holdout          |  61 | 0.014392 | 0.011194 |   0.959439 |  0.981857 |      0.000000 |       0.625000 |
| mct_dsp_apple_sat         | seed7          | fixed340_holdout       |  27 | 0.015398 | 0.011802 |   0.865079 |  0.862741 |      0.020488 |       0.625000 |
| mct_dsp_apple_sat         | seed7          | random_supplement      |  34 | 0.013539 | 0.010711 |   0.833155 |  0.962311 |      0.000000 |       0.750000 |
| mct_dsp_apple_sat         | leave900out    | all900_leave_scale_out |   4 | 0.015964 | 0.014625 |   0.400000 |  0.585368 |      0.000000 |       1.000000 |

## Fitted Low-Dimensional Interaction Parameters

| model             | fit_protocol   |   objective | success   | message                                              |   kappa_benefit |   kappa_penalty |     eta_n |    eta_d |         r1 |
|:------------------|:---------------|------------:|:----------|:-----------------------------------------------------|----------------:|----------------:|----------:|---------:|-----------:|
| mct_dsp_anchor    | seed7          |    0.021340 | True      | no nonlinear parameters                              |        0.000000 |        0.000000 |  0.000000 | 0.000000 | inf        |
| mct_dsp_anchor    | leave900out    |    0.020715 | True      | no nonlinear parameters                              |        0.000000 |        0.000000 |  0.000000 | 0.000000 | inf        |
| mct_dsp_split_amp | seed7          |    0.019220 | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |       -0.000005 |        0.087676 |  0.000000 | 0.000000 | inf        |
| mct_dsp_split_amp | leave900out    |    0.018871 | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |       -0.522017 |        0.000006 |  0.000000 | 0.000000 | inf        |
| mct_dsp_tau_shift | seed7          |    0.020769 | True      | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL     |        0.000000 |        0.000000 | -0.296738 | 0.490650 | inf        |
| mct_dsp_tau_shift | leave900out    |    0.020086 | True      | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL     |        0.000000 |        0.000000 | -0.271519 | 0.498104 | inf        |
| mct_dsp_apple_sat | seed7          |    0.021341 | True      | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL     |        0.000000 |        0.000000 |  0.000000 | 0.000000 | 100.000000 |
| mct_dsp_apple_sat | leave900out    |    0.020713 | True      | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL     |        0.000000 |        0.000000 |  0.000000 | 0.000000 |  12.805052 |

## Interpretation

- The hybrid forms are competitive with the variable-scale DSP screen, but they do not beat canonical MCT on the established MCT validation splits.
- `mct_dsp_tau_shift` is the best hybrid on the seed7/fixed340/random split family, but it generalizes poorly to leave-900-out.
- The plain anchor and Apple-style saturation discount are the safest leave-900-out hybrids, but they are still materially worse than MCT there.
- The split-amplitude interaction improves train fit and then collapses on every held-out split; do not promote it without much stronger regularization.
- The main limitation is that DSP domain geometry is frozen from the 300M fit. A real promotion attempt would need analytic/autodiff gradients for full ND retuning and then raw-optimum/perturbation-gradient diagnostics.
