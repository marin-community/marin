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

### 2026-04-30 - Selected-proxy task optimization
- Hypothesis: optimizing the current selected smooth proxies, with MMLU collapsed into one group, gives a more task-aligned objective than uncheatable BPB or flat MMLU-heavy averages.
- Command: `uv run --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_selected_proxy_tasks.py`
- Result summary:
| candidate_id                           |   proxy_spearman |   accuracy_reference_spearman |   raw_nearest_observed_tv |   raw_nearest_observed_choice |   raw_nearest_observed_accuracy |
|:---------------------------------------|-----------------:|------------------------------:|--------------------------:|------------------------------:|--------------------------------:|
| selected_proxy_keep_only_task_balanced |         0.794495 |                      0.666993 |                  0.856152 |                      1.030174 |                        0.336953 |
| selected_proxy_flat_weighted           |         0.789469 |                      0.605306 |                  0.756175 |                      0.646875 |                        0.309923 |
| selected_proxy_task_balanced           |         0.645939 |                      0.332818 |                  0.887648 |                      0.048946 |                        0.346176 |
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_selected_proxy_tasks_20260430/`.

### 2026-04-30 - Selected-proxy task optimization
- Hypothesis: optimizing the current selected smooth proxies, with MMLU collapsed into one group, gives a more task-aligned objective than uncheatable BPB or flat MMLU-heavy averages.
- Command: `uv run --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_selected_proxy_tasks.py`
- Result summary:
| candidate_id                           |   proxy_spearman |   accuracy_reference_spearman |   raw_nearest_observed_tv |   raw_nearest_observed_choice |   raw_nearest_observed_accuracy |
|:---------------------------------------|-----------------:|------------------------------:|--------------------------:|------------------------------:|--------------------------------:|
| selected_proxy_snr_gt2_task_balanced   |         0.796770 |                      0.609597 |                  0.789914 |                      0.797770 |                        0.297153 |
| selected_proxy_keep_only_task_balanced |         0.794495 |                      0.666993 |                  0.856152 |                      1.030174 |                        0.336953 |
| selected_proxy_flat_weighted           |         0.789469 |                      0.605306 |                  0.756175 |                      0.646875 |                        0.309923 |
| selected_proxy_task_balanced           |         0.645939 |                      0.332818 |                  0.887648 |                      0.048946 |                        0.346176 |
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_selected_proxy_tasks_20260430/`.

### 2026-05-01 - GRP-style modeling of IRT scores
- Hypothesis: denoised IRT/factor targets provide a better task-optimization surrogate target than raw `mean(choice_prob_norm)`.
- Command: `uv run --with matplotlib --with torch python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/fit_grp_300m_irt_targets.py`
- Result summary:
| candidate_id                                  |   target_spearman |   accuracy_reference_spearman |   raw_nearest_observed_tv |   raw_nearest_observed_accuracy |   top8actual_hull_nearest_observed_tv |
|:----------------------------------------------|------------------:|------------------------------:|--------------------------:|--------------------------------:|--------------------------------------:|
| eval_bpb_irt_variable_theta_3                 |          0.962704 |                     -0.013742 |                  0.603826 |                        0.345227 |                              0.363450 |
| eval_bpb_irt_fixed_theta_1                    |          0.954189 |                      0.150912 |                  0.861336 |                        0.340461 |                              0.079849 |
| eval_bpb_irt_variable_theta_2                 |          0.946392 |                      0.148037 |                  0.826003 |                        0.350830 |                              0.102115 |
| eval_bpb_irt_variable_accuracy_weighted_theta |          0.942025 |                      0.398867 |                  0.709710 |                        0.354364 |                              0.092616 |
| task_proxy_irt_fixed_theta_2                  |          0.933074 |                      0.254359 |                  0.895261 |                        0.339675 |                              0.000000 |
| eval_bpb_irt_fixed_accuracy_weighted_theta    |          0.931335 |                      0.394169 |                  0.847947 |                        0.351567 |                              0.098285 |
| hybrid_irt_variable_theta_2                   |          0.931230 |                      0.222006 |                  0.848661 |                        0.347952 |                              0.059986 |
| hybrid_irt_fixed_theta_2                      |          0.929362 |                      0.328594 |                  0.888173 |                        0.339675 |                              0.040629 |
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_300m_irt_targets_20260501/`.

### 2026-05-10 - DSP fits to issue #5416 aggregate
- Hypothesis: the newer DSP family should be tested directly on the issue #5416 signed factor/IRT aggregate rather than only on the earlier task-proxy IRT targets.
- Command: `uv run --no-project --with matplotlib --with numpy --with pandas --with scipy --with scikit-learn --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_issue5416_aggregate_300m.py`
- Target construction: current 242-row 300M raw metric matrix, run_00097 variable-subset noise baseline, 26 selected signed columns, Horn-selected factor count `5`; fit target is `-issue5416_aggregate` so DSP remains a minimization model.
- Result summary:
| variant | params | score CV RMSE | score CV R2 | score OOF Spearman | raw nearest TV |
|:--|--:|--:|--:|--:|--:|
| `dsp_phase_benefit_saturation_penalty_nnls` | 160 | 0.148561 | 0.859189 | 0.918927 | 0.715133 |
| `dsp_saturation_penalty_split_nnls` | 159 | 0.153063 | 0.850526 | 0.903048 | 0.809242 |
| `dsp_effective_exposure_penalty_nnls` | 158 | 0.174260 | 0.806258 | 0.880500 | 0.734133 |
| `dsp_phase_benefit_penalty_nnls` | 158 | 0.179998 | 0.793290 | 0.884568 | 0.740535 |
| `dsp_no_phase_penalty_nnls` | 157 | 0.411557 | -0.080658 | 0.329355 | 0.598074 |
- Interpretation: DSP fits the issue #5416 aggregate substantially better than the earlier selected-proxy GRP targets. All fitted variants select observed best `run_00125` as the best observed point, but raw optima remain off-manifold and should not be deployed without constraints or validation.
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_issue5416_aggregate_300m_20260510/`.

### 2026-05-11 - Pareto-aware effective-exposure validation launch
- Goal: train all seven Pareto-aware effective-exposure DSP optima from `experiments/domain_phase_mix/exploratory/two_phase_many/pareto_aware_effective_exposure_issue5416.py` at historical `300m_6b` (displayed `100M/6B`) before deciding which candidates deserve downstream eval completion.
- Candidate source artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/pareto_aware_effective_exposure_issue5416_20260511/candidate_summary.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/pareto_aware_effective_exposure_issue5416_20260511/candidate_weights_long.csv`
- Launcher: `experiments/domain_phase_mix/launch_pareto_effective_exposure_issue5416_validation.py`.
- Local launch artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/pareto_effective_exposure_issue5416_validation_20260511/`.
- GCS candidate copies:
  - `gs://marin-us-east5/experiments/domain_phase_mix/pareto_effective_exposure_issue5416_validation_20260511/candidate_summary.csv`
  - `gs://marin-us-east5/experiments/domain_phase_mix/pareto_effective_exposure_issue5416_validation_20260511/candidate_weights_long.csv`
- Launch fixes required before a live run could dispatch correctly:
  - `uv.lock` had ambiguous `torch==2.10.0+cu128` CUDA dependency entries, causing remote `uv` parse failure. Fixed by pinning `nvidia-cudnn-cu12==9.10.2.21` and `nvidia-nccl-cu12==2.27.5` with the PyPI source; `uv lock --check` passed afterward.
  - `resolve_executor_step` dropped `ExecutorStep.resources` when building the `StepSpec`, so resourceful training steps executed in the parent and collided in JAX distributed initialization. Fixed by propagating `resources=step.resources`; verified with a direct `ResourceConfig.with_tpu("v5p-8", regions=["us-east5"], zone="us-east5-a")` smoke test and `py_compile`.
- Failed attempt: `/calvinxu/dm-pareto-dsp-issue5416-validation-20260511-105708`.
  - This attempt was stopped after child jobs repeatedly failed with `No accelerator found` / missing `libtpu.so`.
  - Root cause: `StepRunner._run_iris_job` unwrapped `RemoteCallable` when an explicit `StepSpec.resources` override was present, but did not preserve `RemoteCallable.env_vars` or `RemoteCallable.pip_dependency_groups`. The child jobs therefore had TPU resources but installed a non-TPU environment.
  - Fix: preserve the remote callable environment and dependency groups in the explicit-resource path. Verified with a local monkeypatch smoke test that `_run_iris_job` passes `env_vars` and `pip_dependency_groups` through while still overriding resources.
- Additional launch fix:
  - Plain `ExecutorStep(fn=run_levanter_train_lm, resources=v5p-8)` steps were not `RemoteCallable`s, so the previous fix was insufficient: explicit TPU resources still did not imply TPU dependency groups. Fixed `StepRunner._run_iris_job` to infer `tpu` / `gpu` dependency groups from explicit `StepSpec.resources`, and to merge them with any remote callable groups.
  - Added `--candidate-id` filtering to the launcher so transient failed candidates can be retried without regenerating all seven validation runs.
- Failed/stopped attempts:
  - `/calvinxu/dm-pareto-dsp-issue5416-validation-20260511-111210`: stopped after old-style training steps still launched without TPU extras and failed with `No accelerator found`.
  - `/calvinxu/dm-pareto-dsp-issue5416-validation-20260511-112211`: launched all seven after the TPU-extra fix. Six children were killed quickly by TPU runtime `/dev/vfio` device-busy errors. The surviving `hard_item_guardrail` child passed TPU setup, started from scratch, skipped lm-eval harness, and reached validation/data loading; continue tracking that child even though the parent will likely be marked failed because the other six children were killed.
- Retry job: `/calvinxu/dm-pareto-dsp-issue5416-validation-retry6-20260511-113855`.
  - Candidate subset: `aggregate_only`, `hard_group_guardrail`, `group_dro`, `item_cvar25`, `item_maximin`, `mean_plus_tail_penalty`.
  - Concurrency reduced to `max_concurrent=2` because the 7-way launch produced six TPU device-busy kills. This is a concrete scheduler/runtime reason to avoid full parallelism for this retry.
  - Startup status: `aggregate_only` passed TPU autodiscovery, started from scratch, skipped lm-eval harness, and entered data loading; `hard_group_guardrail` is pending on v5p-8 capacity in `us-east5-a`.
- Intended validated candidates: `aggregate_only`, `hard_item_guardrail`, `hard_group_guardrail`, `group_dro`, `item_cvar25`, `item_maximin`, and `mean_plus_tail_penalty`.

### 2026-05-11 - DSP canonical-form naming update
- Decision: use `dsp_effective_exposure_penalty_nnls` as the canonical DSP form going forward.
- Formula:
  `z_i(w) = c_{0i} w_{0i} + gamma c_{1i} w_{1i}` and
  `y_hat(w) = beta_0 - sum_i a_i (1 - exp(-rho_i z_i(w))) + sum_i p_i softplus(log(1 + z_i(w)) - tau_i)^2`.
- Parameter count with 39 domains: per-domain `(rho_i, tau_i, a_i, p_i)` plus global `(beta_0, gamma)`, for `4M + 2 = 158`.
- Code/documentation updates:
  - Added the canonical DSP formulation to the top of `experiments/domain_phase_mix/exploratory/two_phase_many/pareto_aware_effective_exposure_issue5416.py`.
  - Updated the self-contained collaborator implementation `experiments/domain_phase_mix/exploratory/two_phase_many/collaborator_scaling_data_packet_20260430/standalone_code/dsp_exact.py` so `--variant canonical` maps to `dsp_effective_exposure_penalty_nnls`; the old `dsp_phase_benefit_penalty_nnls` form remains available as `--variant benefit_gain`.

### 2026-05-18 - Clean-slate aggregate metric iteration
- Goal: rebuild the aggregate task metric from the current 300M raw metric matrix, including David/#5005 raw-PPL additions, without inheriting the issue #5416 projection quirks.
- Notebook: `experiments/domain_phase_mix/exploratory/two_phase_many/aggregate_metric_clean_slate_20260518.py`.
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_metric_clean_slate_20260518/`.
- Construction changes:
  - Exclude hard accuracy, diagnostic metrics, `lm_eval/averages/*`, top-level `eval/macro_*`, nested `*_macro_*`, and top-level Paloma/Uncheatable aggregate BPB/loss rows from aggregate items.
  - Pool `Generation proxies` and `Raw PPL task train` into `Generative/task-train proxies`, so HumanEval teacher-forced NLL and GSM8K-train raw PPL are not dropped as singleton suites.
  - Make `suite_balanced_mean_no_dsp` the true primary no-DSP aggregate: it is suite-balanced and role/SNR/MoE-R2 weighted, but not suite-gated by DSP controllability. The old suite-gated sensitivity is now `suite_balanced_mean_dsp_suite_gated`.
  - Add robust value-transform sensitivities: clipped-z and rank-z variants.
- Current selected primary aggregate: `82` smooth items across `10` suites.
- Key observed-row result:
  - Primary `suite_balanced_mean_no_dsp`: best observed row `run_00200`, proportional rank `75/242`, top-minus-proportional `0.8385`.
  - Role-balanced primary: best observed row `run_00200`, proportional rank `91/242`.
  - Optimize-only primary: best observed row `run_00200`, proportional rank `158/242`.
- Effective-exposure DSP check on 12 aggregate candidates:
  - The DSP-suite-gated sensitivity fits best among scalar aggregates (`OOF Spearman 0.9047`) but is partly circular by construction.
  - The true primary no-DSP aggregate fits worse (`OOF Spearman 0.8418`, normalized CV RMSE `0.5224`) after including low-controllability MMLU/MMLU-SL suites.
  - Rank/clipped robust transforms do not fix raw optimum behavior; raw optima remain far from the observed manifold (`nearest observed TV` mostly `0.46`-`0.77`) and often concentrate a phase heavily.
  - `suite_balanced_mean_dsp_weighted` has the least-collapsed raw mixture among the inspected scalar aggregate variants, but it is DSP-weighted and lower rank-fit (`OOF Spearman 0.8217`).
- Interpretation:
  - The aggregate construction is now less circular and more suite-complete, but that makes the modeling problem visibly harder.
  - The current evidence favors using these aggregates for observed-row ranking and constrained/trust-region search, not raw unconstrained DSP optima.
  - The main unresolved tension is that MMLU/MMLU-SL are task-relevant but weakly controllable under the current partition/model, so including them lowers DSP fit quality rather than producing a clean deployable optimum.
- Review:
  - Claude Code Opus 4.7 Max was invoked with `env -u ANTHROPIC_API_KEY`; account preflight showed `plambdafour@proton.me` / `stripe_subscription`.
  - First review found macro leakage, accidental DSP suite gating in the primary aggregate, and stale-cache hazards; these were fixed.
  - Second review reported no remaining correctness blockers. Minor cleanup applied afterward: also exclude `micro*` suite aggregates and remove an unused suite-weight accumulator.

### 2026-05-18 - aggregate candidate review and raw-optimum plots
- Goal: decide which clean-slate aggregate variants are useful enough to keep, and visualize each raw DSP optimum mixture in full.
- Notebook update:
  - Added a `Predicted Optimum Mixtures` section to `experiments/domain_phase_mix/exploratory/two_phase_many/aggregate_metric_clean_slate_20260518.py`.
  - The section includes a full heatmap for all raw DSP optima by candidate, phase, and domain, plus a dropdown-selected full per-domain phase bar chart with `weight / effective epochs` end labels.
  - Sparse/extrapolative raw optima are now explicitly marked in the heatmap row labels and selected-candidate title.
- Current diagnostic result:
  - All `12/12` cached raw DSP optima are sparse/extrapolative under the notebook criterion (`nearest observed TV >= 0.5` or a phase-domain max weight `>= 0.5`).
  - Primary `suite_balanced_mean_no_dsp`: OOF Spearman `0.8418`, normalized CV RMSE `0.5224`, raw nearest-observed TV `0.7348`, raw max weights `0.9776/0.5164`.
  - Best fit is the DSP-suite-gated sensitivity `suite_balanced_mean_dsp_suite_gated`: OOF Spearman `0.9047`, normalized CV RMSE `0.3928`, but it is partially circular because suite eligibility uses DSP controllability.
  - The least-collapsed-looking raw mixture is `suite_balanced_mean_dsp_weighted` by phase max weights (`0.1759/0.3234`), but it has poor `cv_regret_at_1=0.8243`, lower OOF Spearman `0.8217`, and uses DSP weighting.
- Claude Code review take:
  - Keep `suite_balanced_mean_no_dsp` as the primary aggregate because it is the least circular and suite-complete.
  - Keep `suite_balanced_clipped_mean_no_dsp` as a robustness twin.
  - Keep `suite_balanced_mean_dsp_suite_gated` as a labeled sensitivity showing what happens when we trust DSP for suite eligibility.
  - Treat suite-factor, unbalanced reliability-weighted, optimize-only, role-balanced, and rank-z variants as diagnostics rather than primary optimization targets.
  - Do not validate any raw unconstrained DSP optimum directly. Use constrained/trust-region optimization, a consensus observed-near mixture, or validate `run_00200`-like candidates first.

### 2026-05-18 - factor aggregate sprint for non-sparse raw DSP optima
- Goal: find a clean-slate aggregate whose unregularized raw effective-exposure DSP optimum is qualitatively non-sparse, rather than relying on explicit TV penalties or trust-region optimization.
- CC ideation feedback:
  - The previous suite-level factor variants were too coarse and mostly rank-1; use item-level factor analysis on the selected smooth items instead.
  - Diagnose optimizer starts and raw-optimum stability before attributing sparsity only to aggregate construction.
  - Keep raw unregularized DSP as the acceptance target; TV regularization remains diagnostic only.
- Notebook and fitting updates:
  - Added item-level factor candidates to `experiments/domain_phase_mix/exploratory/two_phase_many/aggregate_metric_clean_slate_20260518.py`.
  - Initial item-factor family uses the current 82 selected smooth items; Horn parallel analysis selected `K=7`.
  - Added stricter item-filter iterations with the sprint loop capped at `10` item-filter iterations. The current implementation realizes 4 item-filter iterations and 12 stricter factor candidates: controllability `>=0.65`, controllability `>=0.80`, optimize-role controllability `>=0.65`, and task-proxy optimize-role controllability `>=0.65`, each with balanced projection plus factor-balance penalties `0.5` and `1.0`.
  - Extended `standalone_code/dsp_exact.py` and `fit_effective_exposure_dsp_aggregate_candidates_300m.py` with observed-mixture seeded raw optimization, entropy/effective-support diagnostics, fitted `gamma`/`tau` diagnostics, and top-8 Jaccard stability across raw-optimization seeds.
- Full DSP refit command:
  - `uv run --with numpy --with pandas --with scipy --with scikit-learn python experiments/domain_phase_mix/exploratory/two_phase_many/fit_effective_exposure_dsp_aggregate_candidates_300m.py --workers 16`
  - Settings: `27` candidates, `maxiter=100`, `optimum_starts=200`, `stability_seeds=5`, `stability_starts=80`, `max_observed_starts=242`.
- Acceptance gates:
  - `oof_spearman >= 0.88`
  - `raw_nearest_observed_tv <= 0.40`
  - `max_phase_weight <= 0.40`
  - `min_phase_support_gt_1e3 >= 8`
  - multi-seed top-8 domain Jaccard `>= 0.70`
- Result:
  - `0/27` candidates passed all raw-optimum gates.
  - Best gate-ish stricter factor candidate by rank fit: `item_factor_controllable080_balanced_projection`, with OOF Spearman `0.8951`, nearest-observed TV `0.7874`, max phase weight `0.8418`, min support `9`, and top-8 Jaccard `0.7333`; it fails because the raw optimum is still far off-manifold and concentrated.
  - Best gate-ish candidate by max-weight geometry: `item_factor_task_proxy_controllable065_balanced_projection`, with OOF Spearman `0.8880`, nearest-observed TV `0.6023`, max phase weight `0.2827`, min support `8`, and top-8 Jaccard `0.3414`; it avoids single-domain collapse but remains off-manifold and optimizer-unstable.
  - Strongest raw rank fit among item-factor candidates: `item_factor_horn_balanced_projection`, OOF Spearman `0.9066`, but it collapses phase 1 to `0.9430` on `dolma3_cc/literature_high` and fails TV, max-weight, support, and stability gates.
  - Factor-balance penalties improve stability for some candidates but usually trade off rank fit and still do not bring raw optima close enough to the observed manifold.
- Interpretation:
  - Item-level factors improve target construction diagnostics but do not solve the raw-unconstrained DSP sparsity problem.
  - The current evidence still argues against validating a raw aggregate-DSP optimum directly.
  - Next useful path is likely constrained/trust-region or observed-near optimization, plus using these factor aggregates as latent scoring coordinates rather than as unconstrained raw-optimum targets.
