# GRP Target Transform Robustness Audit

Dataset: 300M/6B qsplit-core signal rows (`n=240`) with variable-subset run_00097 noise rows (`n=10`).

GRP no-L2 uses a non-negative least-squares linear head and `reg=0`. Therefore positive affine
target transforms should be exactly safe: the target can be shifted or rescaled by a positive
constant without changing prediction ranks or nonlinear-parameter selection. Negation is not safe
unless it is the deliberate conversion of a higher-is-better metric into the canonical lower-is-better
model target.

## Positive Affine Checks

| metric_slug                 | transform_slug   |   snr_ratio |   fixed_pred_spearman_vs_canonical_pred |   selected_pred_spearman_vs_canonical_pred | selected_same_params_as_canonical   |
|-----------------------------|------------------|-------------|-----------------------------------------|--------------------------------------------|-------------------------------------|
| uncheatable_bpb             | canonical        |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| uncheatable_bpb             | affine_large     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| uncheatable_bpb             | affine_small     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| mmlu_sl_choice_prob_norm    | canonical        |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| mmlu_sl_choice_prob_norm    | affine_large     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| mmlu_sl_choice_prob_norm    | affine_small     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| mmlu_sl_choice_logprob_norm | canonical        |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| mmlu_sl_choice_logprob_norm | affine_large     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| mmlu_sl_choice_logprob_norm | affine_small     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| agentic_success_bpb         | canonical        |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| agentic_success_bpb         | affine_large     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |
| agentic_success_bpb         | affine_small     |    1.000000 |                                1.000000 |                                   1.000000 | True                                |

## Non-Affine / Wrong-Sign Diagnostics

| metric_slug                 | transform_slug      |   snr_ratio |   fixed_pred_spearman_vs_canonical_pred |   selected_pred_spearman_vs_canonical_pred |   selected_pred_spearman_vs_actual_canonical | canonical_predicted_best_run   | selected_predicted_best_run   |
|-----------------------------|---------------------|-------------|-----------------------------------------|--------------------------------------------|----------------------------------------------|--------------------------------|-------------------------------|
| uncheatable_bpb             | wrong_sign_negation |    1.000000 |                                0.708732 |                                   0.759064 |                                     0.717047 | run_00125                      | run_00211                     |
| uncheatable_bpb             | rank_normal         |    1.603498 |                                0.996685 |                                   0.996685 |                                     0.895882 | run_00125                      | run_00125                     |
| mmlu_sl_choice_prob_norm    | wrong_sign_negation |    1.000000 |                                0.546479 |                                   0.565061 |                                     0.137350 | run_00125                      | run_00224                     |
| mmlu_sl_choice_prob_norm    | rank_normal         |   -0.936419 |                                0.989656 |                                   0.971354 |                                     0.092470 | run_00125                      | run_00076                     |
| mmlu_sl_choice_logprob_norm | wrong_sign_negation |    1.000000 |                                0.558809 |                                   0.658727 |                                     0.291413 | run_00063                      | run_00060                     |
| mmlu_sl_choice_logprob_norm | rank_normal         |    0.355507 |                                0.970867 |                                   0.970867 |                                     0.414338 | run_00063                      | run_00063                     |
| agentic_success_bpb         | wrong_sign_negation |    1.000000 |                                0.677903 |                                   0.740366 |                                     0.711956 | run_00223                      | run_00211                     |
| agentic_success_bpb         | rank_normal         |    1.612209 |                                0.998179 |                                   0.998179 |                                     0.908991 | run_00223                      | run_00223                     |

## Interpretation

- Positive affine scaling, including z-scoring by a positive standard deviation, is safe for GRP no-L2.
- Naive negation keeps variance SNR fixed, but it reverses the monotonic direction and breaks the NNLS head's sign prior.
- Rank-normal and other monotone nonlinear transforms preserve target order but change spacings; GRP is not rank-only, so prediction rankings and selected optima can change.
- SNR invariance alone is not sufficient. The target orientation and spacing relative to the GRP inductive bias matter.
