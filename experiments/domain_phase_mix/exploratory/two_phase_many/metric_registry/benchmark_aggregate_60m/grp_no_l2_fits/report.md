# 60M Benchmark-Aggregate GRP Fits

## Coverage

|                                                  |   non_null_rows |
|:-------------------------------------------------|----------------:|
| lm_eval/mmlu_5shot/acc                           |             242 |
| lm_eval/gsm8k/exact_match,flexible-extract       |               3 |
| lm_eval/humaneval/pass@1,create_test             |               3 |
| gsm8k_humaneval_accuracy_mean                    |               3 |
| benchmark_accuracy_mean                          |               3 |
| olmo_base_easy_overlap_no_mmlu_accuracy_mean     |             242 |
| all_suite_accuracy_mean                          |               3 |
| lm_eval/mmlu_5shot/bpb                           |             242 |
| olmo_base_easy_overlap_no_mmlu_bpb_mean          |             242 |
| all_suite_bpb_mean                               |             242 |
| lm_eval/olmo_base_easy_overlap/macro_bpb         |             242 |
| olmo_base_easy_overlap_accuracy_mean             |             242 |
| olmo_base_easy_overlap_accuracy_zmean            |             242 |
| olmo_base_easy_overlap_accuracy_task_logit_mean  |             242 |
| olmo_base_easy_overlap_accuracy_task_probit_mean |             242 |
| olmo_base_easy_overlap_accuracy_task_rank_zmean  |             242 |

## Fit Summary

| family_scheme   | display_name                                                             |   n |   metric_oof_rmse |   metric_oof_spearman |   best_observed_metric |   predicted_observed_metric |   predicted_observed_regret |   raw_predicted_optimum_metric |   raw_nearest_observed_metric |   raw_nearest_observed_regret |   raw_nearest_observed_tv |   raw_phase1_broad_text_share |   raw_phase1_tech_code_share |   raw_phase1_reasoning_share |
|:----------------|:-------------------------------------------------------------------------|----:|------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|-------------------------------:|------------------------------:|------------------------------:|--------------------------:|------------------------------:|-----------------------------:|-----------------------------:|
| default         | OLMoBaseEval easy-overlap macro BPB                                      | 242 |          0.058440 |              0.351772 |               1.702057 |                    1.764695 |                    0.062638 |                       1.635910 |                      1.799418 |                      0.097361 |                  0.692489 |                      0.692363 |                     0.269165 |                     0.038472 |
| default         | OLMoBaseEval easy-overlap mean accuracy excluding MMLU raw-probability   | 242 |          0.003440 |              0.580359 |               0.390879 |                    0.390879 |                    0.000000 |                       0.407994 |                      0.390879 |                      0.000000 |                  0.599398 |                      0.976119 |                     0.001764 |                     0.022116 |
| default         | OLMoBaseEval easy-overlap mean accuracy excluding MMLU logit-probability | 242 |          0.003423 |              0.588909 |               0.390879 |                    0.390879 |                    0.000000 |                       0.407305 |                      0.390879 |                      0.000000 |                  0.592445 |                      0.999821 |                     0.000029 |                     0.000150 |
| default         | OLMoBaseEval easy-overlap mean accuracy raw-probability                  | 242 |          0.003082 |              0.612064 |               0.380723 |                    0.380723 |                    0.000000 |                       0.391770 |                      0.380723 |                      0.000000 |                  0.558259 |                      0.997486 |                     0.000288 |                     0.002226 |
| default         | OLMoBaseEval easy-overlap mean accuracy logit-probability                | 242 |          0.003073 |              0.614160 |               0.380723 |                    0.380723 |                    0.000000 |                       0.393373 |                      0.380723 |                      0.000000 |                  0.601727 |                      0.999930 |                     0.000010 |                     0.000060 |
| default         | OLMoBaseEval easy-overlap mean accuracy arcsin-sqrt probability          | 242 |          0.003098 |              0.605373 |               0.380723 |                    0.380723 |                    0.000000 |                       0.392809 |                      0.380723 |                      0.000000 |                  0.598409 |                      0.998508 |                     0.000176 |                     0.001316 |
| default         | OLMoBaseEval easy-overlap mean accuracy probit-probability               | 242 |          0.003078 |              0.613095 |               0.380723 |                    0.380723 |                    0.000000 |                       0.388251 |                      0.380723 |                      0.000000 |                  0.522734 |                      0.999804 |                     0.000166 |                     0.000030 |
| default         | OLMoBaseEval easy-overlap mean accuracy rank-normal diagnostic           | 242 |          0.778761 |              0.600424 |               2.867887 |                    2.867887 |                    0.000000 |                       6.053879 |                      2.867887 |                      0.000000 |                  0.574154 |                      0.960085 |                     0.000424 |                     0.039491 |
| default         | OLMoBaseEval easy-overlap per-task z-scored mean accuracy                | 242 |          0.289311 |              0.545082 |               1.419174 |                    1.419174 |                    0.000000 |                       2.798697 |                      1.419174 |                      0.000000 |                  0.576076 |                      0.999924 |                     0.000009 |                     0.000067 |
| default         | OLMoBaseEval easy-overlap per-task mean logit accuracy                   | 242 |          0.015480 |              0.600567 |              -0.521698 |                   -0.521698 |                    0.000000 |                      -0.454193 |                     -0.521698 |                      0.000000 |                  0.597018 |                      0.999894 |                     0.000002 |                     0.000104 |
| default         | OLMoBaseEval easy-overlap per-task mean probit accuracy                  | 242 |          0.009269 |              0.593096 |              -0.320593 |                   -0.320593 |                    0.000000 |                      -0.287789 |                     -0.320593 |                      0.000000 |                  0.599440 |                      0.951870 |                     0.000069 |                     0.048061 |
| default         | OLMoBaseEval easy-overlap per-task rank-normal mean accuracy             | 242 |          0.297990 |              0.543219 |               1.286747 |                    1.195636 |                    0.091111 |                       2.496246 |                      0.554511 |                      0.732236 |                  0.711008 |                      0.918128 |                     0.032310 |                     0.049562 |

## Notes

- MMLU is already complete for the 242-row fit swarm.
- GSM8K and HumanEval are generation-task accuracy metrics; this collection does not expose BPB/logprob for them.
- Accuracy objectives include raw probability and logit-probability fits.
- The full benchmark accuracy mean is suite-level: MMLU, GSM8K, and HumanEval receive one vote each.
- The all-suite accuracy mean is suite-level: OLMoBase easy-overlap excluding MMLU, MMLU, GSM8K, and HumanEval receive one vote each.
- The all-suite BPB mean averages MMLU BPB with the available non-MMLU OLMoBase easy-overlap BPB tasks; GSM8K/HumanEval BPB is unavailable.
- OLMoBaseEval easy-overlap metrics are read from the qsplit240 and selected-baseline reruns when available.
- OLMoBaseEval objectives use the rows covered by qsplit240, selected-baseline, and optional Olmix BPB reruns.
