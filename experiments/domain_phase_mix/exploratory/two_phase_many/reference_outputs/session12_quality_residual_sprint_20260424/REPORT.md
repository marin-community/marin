# Session 12 Quality Residual Sprint

This local sprint fits small residual heads on top of MCT-LRQ predictions using only the corresponding training split.
Mixture-only residual variants preserve the fixed-mixture scaling-law shape because they add only an extra `R(w)` term.
The `uD`/`uND` variants are diagnostic only: they can improve prediction but are less clean structurally unless constrained.

## Best Metrics

### fixed340_holdout

| model                                    | fit_protocol   |   params | preserves_fixed_mixture_scaling_law   |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   std_ratio |
|:-----------------------------------------|:---------------|---------:|:--------------------------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|------------:|
| mct_lrq74_balanced_base                  | seed7          |       74 | True                                  |  27 | 0.006073 |   0.931013 |                -0.000029 |               0.953371 |    1.013185 |
| mct_lrq74_drop_base                      | seed7          |       74 | True                                  |  27 | 0.006080 |   0.931013 |                 0.000318 |               0.965338 |    1.024914 |
| mct_lrq74_balanced_ud_quality_ridge0.001 | seed7          |       83 | False                                 |  27 | 0.006560 |   0.935287 |                -0.001615 |               1.021256 |    1.083450 |
| mct_lrq74_balanced_ud_quality_ridge0.01  | seed7          |       83 | False                                 |  27 | 0.006560 |   0.935287 |                -0.001615 |               1.021324 |    1.083512 |
| mct_lrq74_balanced_ud_quality_ridge0.1   | seed7          |       83 | False                                 |  27 | 0.006561 |   0.935287 |                -0.001619 |               1.021996 |    1.084129 |
| mct_lrq74_balanced_ud_quality_ridge1     | seed7          |       83 | False                                 |  27 | 0.006572 |   0.935287 |                -0.001653 |               1.027526 |    1.089258 |
| mct_lrq74_drop_ud_quality_ridge0.001     | seed7          |       83 | False                                 |  27 | 0.006595 |   0.935287 |                -0.001318 |               1.038463 |    1.101182 |
| mct_lrq74_drop_ud_quality_ridge0.01      | seed7          |       83 | False                                 |  27 | 0.006595 |   0.935287 |                -0.001318 |               1.038532 |    1.101246 |

### seed7_holdout

| model                                             | fit_protocol   |   params | preserves_fixed_mixture_scaling_law   |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   std_ratio |
|:--------------------------------------------------|:---------------|---------:|:--------------------------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|------------:|
| mct_lrq74_balanced_base                           | seed7          |       74 | True                                  |  61 | 0.010074 |   0.971391 |                 0.001061 |               1.038405 |    1.047825 |
| mct_lrq74_drop_base                               | seed7          |       74 | True                                  |  61 | 0.010084 |   0.971391 |                 0.001215 |               1.036703 |    1.046186 |
| mct_lrq74_balanced_ud_quality_ridge10             | seed7          |       83 | False                                 |  61 | 0.011552 |   0.971285 |                -0.000033 |               1.067231 |    1.078220 |
| mct_lrq74_drop_ud_quality_ridge10                 | seed7          |       83 | False                                 |  61 | 0.011558 |   0.971074 |                 0.000126 |               1.065843 |    1.076945 |
| mct_lrq74_balanced_anchor_plus_ud_quality_ridge10 | seed7          |       97 | False                                 |  61 | 0.011909 |   0.976626 |                 0.001564 |               1.039691 |    1.052961 |
| mct_lrq74_drop_anchor_plus_ud_quality_ridge10     | seed7          |       97 | False                                 |  61 | 0.011912 |   0.976679 |                 0.001655 |               1.038241 |    1.051560 |
| mct_lrq74_drop_anchor_plus_ud_quality_ridge1      | seed7          |       97 | False                                 |  61 | 0.012167 |   0.973242 |                 0.001351 |               1.053305 |    1.066486 |
| mct_lrq74_balanced_anchor_plus_ud_quality_ridge1  | seed7          |       97 | False                                 |  61 | 0.012202 |   0.972977 |                 0.001263 |               1.054594 |    1.067802 |

### all900_leave_scale_out

| model                                                   | fit_protocol   |   params | preserves_fixed_mixture_scaling_law   |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   std_ratio |
|:--------------------------------------------------------|:---------------|---------:|:--------------------------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|------------:|
| mct_lrq74_drop_anchor_plus_ud_und_quality_ridge0.1      | leave900out    |      105 | False                                 |   4 | 0.005654 |   1.000000 |                -0.001643 |               0.741982 |    0.755616 |
| mct_lrq74_balanced_anchor_plus_ud_und_quality_ridge0.1  | leave900out    |      105 | False                                 |   4 | 0.005699 |   1.000000 |                -0.002059 |               0.747751 |    0.761197 |
| mct_lrq74_balanced_anchor_plus_ud_quality_ridge1        | leave900out    |       97 | False                                 |   4 | 0.007212 |   0.800000 |                -0.002337 |               0.829534 |    0.892975 |
| mct_lrq74_drop_anchor_plus_ud_quality_ridge1            | leave900out    |       97 | False                                 |   4 | 0.007394 |   0.800000 |                -0.002231 |               0.815341 |    0.882250 |
| mct_lrq74_drop_anchor_plus_ud_und_quality_ridge0.01     | leave900out    |      105 | False                                 |   4 | 0.007938 |   1.000000 |                -0.006376 |               0.852076 |    0.877849 |
| mct_lrq74_balanced_anchor_plus_ud_und_quality_ridge0.01 | leave900out    |      105 | False                                 |   4 | 0.008098 |   1.000000 |                -0.006696 |               0.854035 |    0.877333 |
| mct_lrq74_balanced_base                                 | leave900out    |       74 | True                                  |   4 | 0.010434 |   1.000000 |                -0.001962 |               0.528150 |    0.606958 |
| mct_lrq74_drop_base                                     | leave900out    |       74 | True                                  |   4 | 0.010495 |   1.000000 |                -0.001671 |               0.513265 |    0.587847 |

## Interpretation

- MCT-LRQ is already quality-split aware through its LRQ anchor.
- The compact mixture-only residuals are the relevant structural probes; they should be judged against the base MCT rows.
- If a `uD` residual wins, treat it as evidence for quality-dependent scale elasticity, not as a clean promotion by itself.

- `fixed340_holdout`: best structural residual `mct_lrq74_balanced_qsplit_core_ridge10` has RMSE `0.007990`, which is `0.001918` worse than the best MCT base row `mct_lrq74_balanced_base`.
- `seed7_holdout`: best structural residual `mct_lrq74_balanced_qsplit_support_ridge10` has RMSE `0.013784`, which is `0.003710` worse than the best MCT base row `mct_lrq74_balanced_base`.
- `all900_leave_scale_out`: best structural residual `mct_lrq74_balanced_qsplit_support_ridge10` has RMSE `0.028899`, which is `0.018465` worse than the best MCT base row `mct_lrq74_balanced_base`.

Conclusion: the clean quality/support residual is not a promotion. It lowers training error, but it overfits and degrades held-out fixed-340M, seed-7, and 900M transfer. The useful signal is that MCT-LRQ already contains the quality split; extra anchor residual width is currently harmful unless it is incorporated into a jointly fitted structural body.
