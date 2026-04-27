# 60M Benchmark-Aggregate GRP Fits

## Coverage

|                                                  |   non_null_rows |
|:-------------------------------------------------|----------------:|
| lm_eval/mmlu_5shot/acc                           |             242 |
| lm_eval/gsm8k/exact_match,flexible-extract       |               3 |
| lm_eval/humaneval/pass@1,create_test             |               3 |
| benchmark_accuracy_mean                          |               3 |
| lm_eval/olmo_base_easy_overlap/macro_bpb         |             242 |
| olmo_base_easy_overlap_accuracy_mean             |             242 |
| olmo_base_easy_overlap_accuracy_zmean            |             242 |
| olmo_base_easy_overlap_accuracy_task_logit_mean  |             242 |
| olmo_base_easy_overlap_accuracy_task_probit_mean |             242 |
| olmo_base_easy_overlap_accuracy_task_rank_zmean  |             242 |

## Fit Summary

| family_scheme   | display_name                                            |   n |   metric_oof_rmse |   metric_oof_spearman |   best_observed_metric |   predicted_observed_metric |   predicted_observed_regret |   raw_predicted_optimum_metric |   raw_nearest_observed_metric |   raw_nearest_observed_regret |   raw_nearest_observed_tv |   raw_phase1_broad_text_share |   raw_phase1_tech_code_share |   raw_phase1_reasoning_share |
|:----------------|:--------------------------------------------------------|----:|------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|-------------------------------:|------------------------------:|------------------------------:|--------------------------:|------------------------------:|-----------------------------:|-----------------------------:|
| default         | OLMoBaseEval easy-overlap mean accuracy raw-probability | 242 |          0.003013 |              0.635977 |               0.380723 |                    0.380723 |                    0.000000 |                       0.395072 |                      0.380723 |                      0.000000 |                  0.601468 |                      0.999660 |                     0.000043 |                     0.000297 |
| qa_reasoning    | OLMoBaseEval easy-overlap mean accuracy raw-probability | 242 |          0.002975 |              0.653515 |               0.380723 |                    0.380723 |                    0.000000 |                       0.402106 |                      0.371146 |                      0.009577 |                  0.812697 |                      0.998467 |                     0.000000 |                     0.001532 |
| qa_tech         | OLMoBaseEval easy-overlap mean accuracy raw-probability | 242 |          0.002988 |              0.639593 |               0.380723 |                    0.380723 |                    0.000000 |                       0.399023 |                      0.380723 |                      0.000000 |                  0.682779 |                      0.600114 |                     0.399818 |                     0.000068 |
| synthetic_tech  | OLMoBaseEval easy-overlap mean accuracy raw-probability | 242 |          0.002965 |              0.652270 |               0.380723 |                    0.380723 |                    0.000000 |                       0.452184 |                      0.363993 |                      0.016730 |                  0.863305 |                      0.976769 |                     0.023231 |                     0.000000 |

## Family Partition Result

The raw mean-accuracy fit is sensitive to how synthetic QA is assigned:

- `qa_reasoning` gives the best rank fit (`Spearman=0.6535`) but worsens raw-optimum quality. Its unrestricted optimum is far from the observed panel (`TV=0.813`) and the nearest observed row loses `0.0096` accuracy to the best observed row.
- `synthetic_tech` has similar rank fit (`Spearman=0.6523`) but the raw optimum is even more over-optimistic (`0.452` predicted accuracy) and the nearest observed row is poor.
- `qa_tech` is the best tradeoff among the tested partitions. It slightly improves RMSE over default, still selects `baseline_olmix_loglinear` as the best observed row, and makes phase 1 less degenerate at the family level (`60%` broad text / `40%` tech-code instead of ~`100%` broad text).
- `qa_tech` still does not solve raw optimization. Its raw optimum is still optimistic (`0.399` predicted versus `0.3807` best observed) and remains far from the observed panel (`TV=0.683`).

Current recommendation: keep `qa_tech` as the best accuracy-aggregate family partition candidate, but do not use unconstrained raw optimization. This is a useful modeling knob, not a deployment-ready optimizer.

## Notes

- MMLU is already complete for the 242-row fit swarm.
- Full benchmark mean is only fit once MMLU, GSM8K, and HumanEval are non-null for every row.
- Accuracy objectives include both raw probability and logit-probability fits when complete.
- OLMoBaseEval easy-overlap metrics are read from the qsplit240 and selected-baseline reruns when available.
- OLMoBaseEval objectives use the rows covered by qsplit240, selected-baseline, and optional Olmix BPB reruns.
