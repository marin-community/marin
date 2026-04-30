# Benchmark Proxy Optimization: Research Logbook

## Scope
- Goal: test whether smooth perplexity/BPB proxies can improve 300M benchmark mixture optimization.
- Primary metrics: OOF rank/regret against mean choice probability and mean accuracy, excluding MMLU-Pro.
- Constraints: local-only modeling, no new training/eval launches.

## Experiment Log
### 2026-04-28 - Perplexity proxy sprint
- Hypothesis: GRP models smooth BPB/perplexity surfaces better than bounded benchmark probabilities, so a learned BPB proxy may produce better selectors or saner optima.
- Command: `uv run --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_perplexity_proxy_benchmark.py`
- Result summary:
| candidate_id                                                       | candidate_kind      | target                            | feature_set       |   choice_spearman |   accuracy_spearman |   raw_nearest_observed_tv |   raw_nearest_observed_choice |   raw_nearest_observed_accuracy |
|:-------------------------------------------------------------------|:--------------------|:----------------------------------|:------------------|------------------:|--------------------:|--------------------------:|------------------------------:|--------------------------------:|
| direct_grp_mean_choice_prob_norm_no_mmlu_pro                       | direct_grp_target   | mean_choice_prob_norm_no_mmlu_pro | direct_target     |          0.606439 |            0.505600 |                  0.874540 |                      0.253744 |                        0.261124 |
| mean_choice_prob_norm_no_mmlu_pro__eval_plus_lm_eval__ridge__logit | scalar_proxy_target | mean_choice_prob_norm_no_mmlu_pro | eval_plus_lm_eval |          0.597067 |            0.534188 |                  0.563840 |                      0.262990 |                        0.282983 |
| mean_accuracy_no_mmlu_pro__eval_plus_lm_eval__ridge__identity      | scalar_proxy_target | mean_accuracy_no_mmlu_pro         | eval_plus_lm_eval |          0.549853 |            0.515479 |                  0.873947 |                      0.259233 |                        0.278741 |
| direct_grp_mean_accuracy_no_mmlu_pro                               | direct_grp_target   | mean_accuracy_no_mmlu_pro         | direct_target     |          0.540114 |            0.501532 |                  0.850180 |                      0.253744 |                        0.261124 |
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_perplexity_proxy_benchmark_20260428/`.

### 2026-04-29 - Suite-balanced 300M benchmark objective
- Hypothesis: the flat 119-column objective overweights MMLU subject variants, so a suite-balanced target should be a better benchmark optimization target.
- Command: `uv run --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_suite_balanced_benchmark.py`
- Result summary:
| candidate_id                                                                         | candidate_kind           | target                                         | feature_set       |   choice_spearman |   accuracy_spearman |   raw_nearest_observed_tv |   raw_nearest_observed_choice |   raw_nearest_observed_accuracy |
|:-------------------------------------------------------------------------------------|:-------------------------|:-----------------------------------------------|:------------------|------------------:|--------------------:|--------------------------:|------------------------------:|--------------------------------:|
| direct_grp_suite_balanced_choice_prob_norm_snr_difficulty                            | direct_grp_target        | suite_balanced_choice_prob_norm_snr_difficulty | direct_target     |          0.579173 |            0.433893 |                  0.644523 |                      0.319882 |                        0.391932 |
| suite_balanced_choice_prob_norm_snr_difficulty__eval_plus_lm_eval__elasticnet__logit | component_proxy_ensemble | suite_balanced_choice_prob_norm_snr_difficulty | eval_plus_lm_eval |          0.543059 |            0.448470 |                  0.811242 |                      0.318058 |                        0.399181 |
| suite_balanced_choice_prob_norm_snr_difficulty__eval_plus_lm_eval__elasticnet__logit | scalar_proxy_target      | suite_balanced_choice_prob_norm_snr_difficulty | eval_plus_lm_eval |          0.482586 |            0.451814 |                  0.516395 |                      0.319594 |                        0.397025 |
| direct_grp_suite_balanced_accuracy_snr_difficulty                                    | direct_grp_target        | suite_balanced_accuracy_snr_difficulty         | direct_target     |          0.443099 |            0.639237 |                  0.800600 |                      0.318058 |                        0.399181 |
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_suite_balanced_benchmark_20260429/`.

### 2026-04-30 01:57 UTC - Smooth-proxy coverage audit
- Hypothesis: benchmark optimization should use smooth proxies where possible; generation exact-match/pass@1 should remain validation/reporting unless a teacher-forced proxy exists.
- Commands:
  - `uv run python - <<'PY' ... current_300m_lm_eval_task_proxy_audit.csv ... PY`
  - `uv run iris --cluster marin job summary /calvinxu/dm-300m-english-lite-snr-r3-east5-20260429-221817`
  - `uv run iris --cluster marin job logs /calvinxu/dm-300m-english-lite-snr-r3-east5-20260429-221817 | tail -120`
- Result:
  - Current `metrics_wide.csv` has 129 non-empty 300M lm-eval task keys.
  - 120/129 already have smooth proxy columns.
  - The 9 missing smooth columns are derived MMLU category rows plus legacy `sciq_0shot`; these do not require new evaluation.
  - The separately collected GSM8K/HumanEval overlay has no smooth proxies by construction: vLLM generation artifacts only contain exact-match/pass@1 and stderr.
  - English-lite child logs confirm MCQ/loglikelihood tasks emit `bpb`, `logprob`, `choice_logprob_norm`, and `choice_prob_norm`, but the parent failed with 10 child failures and needs failure-only recovery/collection.
- Interpretation:
  - The real smooth-proxy gap is GSM8K and HumanEval, not MMLU or English-lite.
  - MMLU should use the SL-Verb smooth columns as the preferred proxy source; standard MMLU can stay as a diagnostic to avoid double-counting MMLU.
  - GSM8K/HumanEval need separate teacher-forced scoring jobs because vLLM TPU generation does not surface prompt logprobs.
- Artifact: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/300m_eval_accounting_20260429/smooth_proxy_audit_and_plan.md`.

### 2026-04-29 - 300M GSM8K/HumanEval SNR accounting
- Hypothesis: GSM8K/HumanEval should be included in reporting, but only used for optimization if their 300M signal is not dominated by fixed-mixture seed noise.
- Commands:
  - `gcloud storage cp gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_300m_gsm8k_humaneval_evals_20260429_0045/pinlin_calvin_xu/data_mixture/ngd3dm2_300m_gsm8k_humaneval_evals_20260429_0045/collect_results-23f14b/300m_gsm8k_humaneval_eval_results.csv experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_gsm8k_humaneval_completion/300m_gsm8k_humaneval_eval_results.csv`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/build_eval_signal_to_noise_all_metrics_300m.py --extra-results-csv experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_gsm8k_humaneval_completion/300m_gsm8k_humaneval_eval_results.csv`
- Result summary:
| metric | qsplit signal rows | noise rows | mean | top | SNR | recommendation |
|:--|--:|--:|--:|--:|--:|:--|
| GSM8K flexible exact match | 238 | 10 | 2.01% | 2.96% (`run_00197`) | 0.89 | report/downweight only |
| GSM8K strict exact match | 238 | 10 | 1.52% | 2.58% (`run_00142`) | 1.09 | downweight |
| HumanEval pass@1 | 238 | 10 | 1.19% | 4.27% (`run_00131`) | 1.98 | downweight |
- Interpretation: HumanEval is just under the keep threshold and can be a downweighted suite-level reporting/optimization component. GSM8K is weaker: flexible exact match is below SNR 1, and strict exact match is barely above SNR 1.
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/300m_gsm8k_humaneval_snr_20260429/`.

### 2026-04-29 - Suite-balanced 300M benchmark objective
- Hypothesis: the flat 119-column objective overweights MMLU subject variants, so a suite-balanced target should be a better benchmark optimization target.
- Command: `uv run --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_suite_balanced_benchmark.py`
- Result summary:
| candidate_id                                                                         | candidate_kind           | target                                         | feature_set       |   choice_spearman |   accuracy_spearman |   raw_nearest_observed_tv |   raw_nearest_observed_choice |   raw_nearest_observed_accuracy |
|:-------------------------------------------------------------------------------------|:-------------------------|:-----------------------------------------------|:------------------|------------------:|--------------------:|--------------------------:|------------------------------:|--------------------------------:|
| direct_grp_suite_balanced_choice_prob_norm_snr_difficulty                            | direct_grp_target        | suite_balanced_choice_prob_norm_snr_difficulty | direct_target     |          0.579173 |            0.433893 |                  0.644523 |                      0.319882 |                        0.391932 |
| suite_balanced_choice_prob_norm_snr_difficulty__eval_plus_lm_eval__elasticnet__logit | component_proxy_ensemble | suite_balanced_choice_prob_norm_snr_difficulty | eval_plus_lm_eval |          0.543059 |            0.448470 |                  0.811242 |                      0.318058 |                        0.399181 |
| suite_balanced_choice_prob_norm_snr_difficulty__eval_plus_lm_eval__elasticnet__logit | scalar_proxy_target      | suite_balanced_choice_prob_norm_snr_difficulty | eval_plus_lm_eval |          0.482586 |            0.451814 |                  0.516395 |                      0.319594 |                        0.397025 |
| direct_grp_suite_balanced_accuracy_snr_difficulty                                    | direct_grp_target        | suite_balanced_accuracy_snr_difficulty         | direct_target     |          0.443099 |            0.639237 |                  0.800600 |                      0.318058 |                        0.399181 |
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_suite_balanced_benchmark_20260429/`.
