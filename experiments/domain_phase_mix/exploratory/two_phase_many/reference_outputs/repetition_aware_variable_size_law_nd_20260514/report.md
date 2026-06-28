# Repetition-Aware Variable-Size Law on Marin ND Scaling Data

## Data

- Source: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`
- Metric: `eval/uncheatable_eval/bpb`
- Rows: `641` labeled rows for grouped OOF.

## Tested Forms

Base paper form:

$$L = E + C/N^{\beta} + B N^{\delta}/D_{\mathrm{eff}}^{\alpha} + \gamma h.$$

For Marin's 39-domain, two-phase mixtures, domain exposure is aggregated as:

$$h_i = 0.8w_{0i} + 0.2w_{1i}, \qquad r_i = w_{0i}c_{0i} + w_{1i}c_{1i},$$

$$D_{i,\mathrm{eff}} = h_iD\,\frac{r_{i,\mathrm{eff}}}{r_i}, \qquad r_{i,\mathrm{eff}} = r_i \; (r_i \le 1), \quad 1 + r_1(1-e^{-(r_i-1)/r_1}) \; (r_i > 1).$$

The multi-domain variants use either equal domain value, positive per-domain value weights inside `D_eff`, or an additional signed linear mixture head.

## Grouped OOF Results

| model                                      |   parameter_count |     rmse |      mae |   spearman |   pearson |   regret_at_1 |   top8_overlap |   predicted_actual_std_ratio |
|:-------------------------------------------|------------------:|---------:|---------:|-----------:|----------:|--------------:|---------------:|-----------------------------:|
| scale_only_chinchilla_size                 |                 6 | 0.033012 | 0.019557 |   0.808008 |  0.907140 |      0.041499 |       0.750000 |                     0.908205 |
| raml_uniform_domain_value_shared_r1        |                 7 | 0.028837 | 0.018319 |   0.863451 |  0.930028 |      0.046898 |       0.625000 |                     0.939539 |
| raml_domain_value_shared_r1                |                46 | 0.026197 | 0.017432 |   0.891984 |  0.942695 |      0.041499 |       0.625000 |                     0.956454 |
| raml_domain_value_linear_mix_shared_r1     |                85 | 0.084952 | 0.020873 |   0.887290 |  0.695448 |      0.023235 |       0.625000 |                     1.504600 |
| raml_domain_value_linear_mix_per_domain_r1 |               123 | 0.025702 | 0.016884 |   0.895723 |  0.946630 |      0.023235 |       0.625000 |                     1.004216 |

## Leave-One-Scale-Out Results

| model                                      | scale      |   n |     rmse |   spearman |   regret_at_1 |   top8_overlap |
|:-------------------------------------------|:-----------|----:|---------:|-----------:|--------------:|---------------:|
| scale_only_chinchilla_size                 | 130m_2p6b  |  39 | 0.206345 |   0.839878 |      0.036042 |       0.625000 |
| scale_only_chinchilla_size                 | 1_2b_24b   |   4 | 0.019892 | nan        |      0.041499 |       1.000000 |
| scale_only_chinchilla_size                 | 300m_6b    | 268 | 0.034270 |   0.267422 |      0.034341 |       0.500000 |
| scale_only_chinchilla_size                 | 520m_10p4b |  37 | 0.036546 |   0.832889 |      0.031030 |       0.500000 |
| scale_only_chinchilla_size                 | 60m_1p2b   | 293 | 0.060558 | nan        |      0.011993 |       0.125000 |
| raml_uniform_domain_value_shared_r1        | 130m_2p6b  |  39 | 0.033151 |   0.789615 |      0.036042 |       0.500000 |
| raml_uniform_domain_value_shared_r1        | 1_2b_24b   |   4 | 0.021259 |  -0.800000 |      0.046898 |       1.000000 |
| raml_uniform_domain_value_shared_r1        | 300m_6b    | 268 | 0.029094 |   0.197265 |      0.034341 |       0.375000 |
| raml_uniform_domain_value_shared_r1        | 520m_10p4b |  37 | 0.032283 |   0.786675 |      0.031030 |       0.500000 |
| raml_uniform_domain_value_shared_r1        | 60m_1p2b   | 293 | 0.057185 |   0.474040 |      0.060017 |       0.000000 |
| raml_domain_value_shared_r1                | 130m_2p6b  |  39 | 0.037779 |   0.878138 |      0.036042 |       0.500000 |
| raml_domain_value_shared_r1                | 1_2b_24b   |   4 | 0.019644 |  -0.600000 |      0.041499 |       1.000000 |
| raml_domain_value_shared_r1                | 300m_6b    | 268 | 0.046608 |   0.429327 |      0.034341 |       0.500000 |
| raml_domain_value_shared_r1                | 520m_10p4b |  37 | 0.029565 |   0.827643 |      0.031030 |       0.500000 |
| raml_domain_value_shared_r1                | 60m_1p2b   | 293 | 0.045598 |   0.671964 |      0.008197 |       0.625000 |
| raml_domain_value_linear_mix_shared_r1     | 130m_2p6b  |  39 | 0.278896 |   0.348178 |      0.029756 |       0.375000 |
| raml_domain_value_linear_mix_shared_r1     | 1_2b_24b   |   4 | 0.056485 |   0.800000 |      0.023235 |       1.000000 |
| raml_domain_value_linear_mix_shared_r1     | 300m_6b    | 268 | 0.035426 |   0.316957 |      0.021420 |       0.125000 |
| raml_domain_value_linear_mix_shared_r1     | 520m_10p4b |  37 | 0.040721 |   0.876008 |      0.000000 |       0.625000 |
| raml_domain_value_linear_mix_shared_r1     | 60m_1p2b   | 293 | 0.061049 |   0.586829 |      0.327265 |       0.250000 |
| raml_domain_value_linear_mix_per_domain_r1 | 130m_2p6b  |  39 | 0.039851 |   0.757287 |      0.029756 |       0.375000 |
| raml_domain_value_linear_mix_per_domain_r1 | 1_2b_24b   |   4 | 0.059091 |   1.000000 |      0.000000 |       1.000000 |
| raml_domain_value_linear_mix_per_domain_r1 | 300m_6b    | 268 | 0.024057 |   0.439071 |      0.021420 |       0.375000 |
| raml_domain_value_linear_mix_per_domain_r1 | 520m_10p4b |  37 | 0.029151 |   0.880512 |      0.000000 |       0.625000 |
| raml_domain_value_linear_mix_per_domain_r1 | 60m_1p2b   | 293 | 0.053528 |   0.585479 |      0.327265 |       0.250000 |

## Decoded Nonlinear Parameter Summary

|    alpha |     beta |    delta |         r1 |   tau_min |   tau_median |   tau_max | model                                      |   parameter_count |     r1_min |   r1_median |     r1_max |
|---------:|---------:|---------:|-----------:|----------:|-------------:|----------:|:-------------------------------------------|------------------:|-----------:|------------:|-----------:|
| 1.335165 | 0.020000 | 0.589107 | inf        |  1.000000 |     1.000000 |  1.000000 | scale_only_chinchilla_size                 |                 6 | nan        |  nan        | nan        |
| 2.000000 | 0.020000 | 1.038160 |   9.471579 |  1.000000 |     1.000000 |  1.000000 | raml_uniform_domain_value_shared_r1        |                 7 | nan        |  nan        | nan        |
| 1.783074 | 0.020000 | 1.163305 |   0.570589 |  0.036692 |     0.800469 | 37.455171 | raml_domain_value_shared_r1                |                46 | nan        |  nan        | nan        |
| 0.945745 | 0.025736 | 0.436066 |  50.742072 |  0.016764 |     1.836366 | 13.349094 | raml_domain_value_linear_mix_shared_r1     |                85 | nan        |  nan        | nan        |
| 0.699462 | 0.063250 | 0.374865 | nan        |  0.026241 |     0.846921 | 78.222885 | raml_domain_value_linear_mix_per_domain_r1 |               123 |   0.010513 |    5.154232 | 100.000000 |

## Interpretation

- The literal variable-size law is too low-dimensional for Marin mixtures if domains are treated as equally valuable.
- Positive per-domain value weights are the most stable improvement over the literal paper form.
- A signed linear mixture head plus per-domain repetition constants gives the best grouped OOF fit here, but the shared-r1 signed-head variant is badly miscalibrated and the best variant hits optimizer budget limits.
- These RAML adaptations are useful baselines/backbones for scale/repetition structure, not replacements for DSP-style mixture structure without additional optimum and perturbation-geometry validation.
