# 60M Benchmark-Aggregate GRP Fits

## Coverage

|                                                  |   non_null_rows |
|:-------------------------------------------------|----------------:|
| lm_eval/mmlu_5shot/acc                           |             242 |
| lm_eval/gsm8k/exact_match,flexible-extract       |             242 |
| lm_eval/humaneval/pass@1,create_test             |             242 |
| gsm8k_humaneval_accuracy_mean                    |             242 |
| benchmark_accuracy_mean                          |             242 |
| olmo_base_easy_overlap_no_mmlu_accuracy_mean     |             242 |
| all_suite_accuracy_mean                          |             242 |
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

| family_scheme   | display_name                                                                                      |   n |   metric_oof_rmse |   metric_oof_spearman |   best_observed_metric |   predicted_observed_metric |   predicted_observed_regret |   raw_predicted_optimum_metric |   raw_nearest_observed_metric |   raw_nearest_observed_regret |   raw_nearest_observed_tv |   raw_phase1_broad_text_share |   raw_phase1_tech_code_share |   raw_phase1_reasoning_share |
|:----------------|:--------------------------------------------------------------------------------------------------|----:|------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|-------------------------------:|------------------------------:|------------------------------:|--------------------------:|------------------------------:|-----------------------------:|-----------------------------:|
| default         | Mean(OLMoBase easy-overlap without MMLU, MMLU, GSM8K, HumanEval) suite accuracy raw-probability   | 242 |          0.002365 |              0.323591 |               0.170817 |                    0.170353 |                    0.000464 |                       0.174993 |                      0.170353 |                      0.000464 |                  0.578458 |                      0.987968 |                     0.009880 |                     0.002153 |
| default         | Mean(OLMoBase easy-overlap without MMLU, MMLU, GSM8K, HumanEval) suite accuracy logit-probability | 242 |          0.002368 |              0.300137 |               0.170817 |                    0.170353 |                    0.000464 |                       0.180991 |                      0.170353 |                      0.000464 |                  0.651504 |                      0.963827 |                     0.036166 |                     0.000007 |
| default         | Mean(MMLU, OLMoBase easy-overlap without MMLU) BPB                                                | 242 |          0.052643 |              0.327617 |               1.775225 |                    1.844719 |                    0.069494 |                       1.752570 |                      1.815086 |                      0.039861 |                  0.634342 |                      0.652839 |                     0.276536 |                     0.070624 |
| default         | OLMoBaseEval easy-overlap mean accuracy excluding MMLU raw-probability                            | 242 |          0.003304 |              0.627767 |               0.390879 |                    0.390879 |                    0.000000 |                       0.408704 |                      0.390879 |                      0.000000 |                  0.576920 |                      0.999501 |                     0.000303 |                     0.000196 |
| default         | OLMoBaseEval easy-overlap mean accuracy excluding MMLU logit-probability                          | 242 |          0.003309 |              0.628353 |               0.390879 |                    0.390879 |                    0.000000 |                       0.409543 |                      0.390879 |                      0.000000 |                  0.591031 |                      0.999981 |                     0.000004 |                     0.000016 |

## Notes

- MMLU is already complete for the 242-row fit swarm.
- GSM8K and HumanEval are generation-task accuracy metrics; this collection does not expose BPB/logprob for them.
- Accuracy objectives include raw probability and logit-probability fits.
- The full benchmark accuracy mean is suite-level: MMLU, GSM8K, and HumanEval receive one vote each.
- The all-suite accuracy mean is suite-level: OLMoBase easy-overlap excluding MMLU, MMLU, GSM8K, and HumanEval receive one vote each.
- The all-suite BPB mean averages MMLU BPB with the available non-MMLU OLMoBase easy-overlap BPB tasks; GSM8K/HumanEval BPB is unavailable.
- OLMoBaseEval easy-overlap metrics are read from the qsplit240 and selected-baseline reruns when available.
- OLMoBaseEval objectives use the rows covered by qsplit240, selected-baseline, and optional Olmix BPB reruns.
