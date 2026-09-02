# Debugging log for 300M SNR signal_n missingness

## Overview

Investigate why many rows in `eval_signal_to_noise_all_metrics_300m_current.csv` have `signal_n < 242`.

## Initial Status

The rebuilt 300M all-metrics SNR table has 242 signal rows overall, but per-metric `signal_n` varies:

- `242`: all `eval/*`, all teacher-forced smooth-proxy metrics, and most English-lite metrics.
- `241`: all MMLU metrics.
- `238`: hard generation GSM8K/HumanEval metrics.

## Cause 1: MMLU is missing from the local wide table for one signal row

All MMLU metrics have `signal_n=241` because the 300M signal row
`baseline_olmix_loglinear_uncheatable_bpb` has no `lm_eval/mmlu_5shot/*` metrics in
`metric_registry/metrics_wide.csv`.

This is not a missing eval job. The attached artifact exists at:

`gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b/baseline_olmix_loglinear_uncheatable_bpb-7ac5e9/lm_eval_artifacts/lm_eval_harness_results.20000.json`

and contains top-level MMLU plus subject-level MMLU and MMLU-SL-Verb keys. The local problem is that the
subject metrics are not being overlaid into the current SNR input.

## Cause 2: hard GSM8K/HumanEval skipped four baseline rows

The hard GSM8K/HumanEval result CSV has four 300M baseline rows marked `not_launched`:

- `baseline_olmix_loglinear_uncheatable_bpb`
- `baseline_proportional`
- `baseline_stratified`
- `baseline_unimax`

This makes the hard generation metrics report `signal_n=238`. The skip reason in the CSV is
`launch_decision=skip_existing`, but `existing_artifact_count=0`, so this should be treated as launcher
state/skip accounting that needs follow-up before using hard GSM8K/HumanEval as complete 242-row metrics.

The data itself exists in `paper_plots/img/baseline_scaling_downstream_eval_metrics_merged.csv`. Adding
that CSV to the SNR overlay makes hard GSM8K/HumanEval report `signal_n=242`.

## Results

The baseline_stratified east5 recovery fixed English-lite and teacher-forced smooth-proxy coverage:

- `300m_english_lite_eval_results_merged.csv`: `253/253 collected`.
- `300m_generative_smooth_proxy_eval_results.csv`: `253/253 collected`.

The SNR builder now includes the known local downstream eval overlays by default and enriches overlay rows
from referenced `lm_eval_artifacts/lm_eval_harness_results*.json` files. This fixes both missingness
classes without launching any new jobs:

- The baseline-scaling downstream eval table supplies the skipped hard GSM8K/HumanEval baseline rows.
- The attached 300M Olmix MMLU artifact supplies subject-level MMLU and MMLU-SL-Verb metrics.

After regenerating `eval_signal_to_noise_all_metrics_300m_current*`:

- `signal_rows = 242`
- `noise_rows = 10`
- `n_metrics = 629`
- every metric has `signal_n = 242`
- every metric has `noise_n = 10`

## Proxy Selection Policy

The keep/drop table should not optimize directly against hard metrics such as `acc`, `exact_match`, or
`pass@1` when a smoother proxy is available. Those hard labels are retained as the accuracy target for
correlation and SNR accounting, but the selected optimization proxy is now chosen from smooth metrics:

- `choice_prob_norm`
- `choice_logprob_norm`
- `bpb`
- `nll`
- `perplexity`
- `loss`
- `choice_logprob`
- `logprob`

For each task, the builder selects the smooth proxy with the highest score:

`abs(Spearman(proxy, accuracy)) * SNR_shrink(proxy) * SNR_shrink(accuracy) * smoothness_weight`

Teacher-forced GSM8K and HumanEval proxy tasks are grouped back to their hard target task:

- `teacher_forced/gsm8k_5shot_gold_solution/*` and
  `teacher_forced/gsm8k_5shot_answer_hash/*` are grouped with `lm_eval/gsm8k/*`.
- `teacher_forced/humaneval_10shot_canonical_solution/*` is grouped with `lm_eval/humaneval/*`.

With this policy, no hard metric is selected as an optimization proxy. The current recommendation counts
are:

- `keep`: 9
- `downweight`: 45
- `report_only`: 30

Notable selected proxies:

- `gsm8k`: `teacher_forced/gsm8k_5shot_gold_solution/nll`, downweighted because correlation is only
  moderate despite strong smooth-proxy SNR.
- `humaneval`: `teacher_forced/humaneval_10shot_canonical_solution/bpb`, kept because both proxy SNR and
  proxy-to-accuracy correlation pass the threshold.
- `swag_0shot`: `mcq_smooth/swag_0shot/choice_logprob_norm`, kept because both proxy SNR and
  proxy-to-accuracy correlation pass.
- `medmcqa_5shot`, `truthfulqa_mc1_0shot`, and `truthfulqa_mc2_0shot`: downweighted because the new
  MCQ smooth proxy has useful but marginal SNR or correlation.
- `sciq_5shot`: report-only because its best smooth proxy has SNR < 1.

## Missing `selected_proxy_metric` Rows

Rows with no selected proxy are intentional under the smooth-proxy policy:

- `eval` has many smooth BPB/loss metrics but no hard accuracy target to correlate against, so it is
  `report_only` with `missing_accuracy_target`.
- The previous `missing_smooth_proxy` rows for `medmcqa_5shot`, `sciq_5shot`, `swag_0shot`,
  `truthfulqa_mc1_0shot`, and `truthfulqa_mc2_0shot` are resolved by the MCQ smooth-proxy completion
  job.

The builder now emits `accuracy_metric_count`, `smooth_proxy_metric_count`, and `available_metric_kinds`
in the task summary and keep/drop outputs to make this distinction explicit.

## MMLU category smooth-proxy derivation

The SNR builder now derives MMLU category-level smooth metrics locally from subject-level metrics before
SNR computation. It reads the canonical `MMLU_SUBJECT_TO_CATEGORY` assignment from
`experiments/evals/olmo_base_easy_overlap.py` without importing the eval config module, then averages
subject-level:

- `choice_prob_norm`
- `choice_logprob_norm`
- `choice_logprob`
- `logprob`
- `bpb`

This gives `mmlu_stem_5shot`, `mmlu_humanities_5shot`, `mmlu_social_sciences_5shot`, and
`mmlu_other_5shot` smooth proxy candidates with `signal_n=242` / `noise_n=10`. Current recommendations:

- `mmlu_humanities_5shot`: downweight, selected `choice_prob_norm`.
- `mmlu_other_5shot`: downweight, selected `choice_prob_norm`.
- `mmlu_social_sciences_5shot`: report-only, selected `choice_prob_norm` has SNR < 1.
- `mmlu_stem_5shot`: report-only, selected `choice_prob_norm` has SNR < 1.

## MCQ smooth-proxy completion launcher

Added `experiments/domain_phase_mix/launch_300m_mcq_smooth_proxy_evals.py` for the remaining hard-only
English-lite tasks:

- `medmcqa_5shot`
- `sciq_5shot`
- `swag_0shot`
- `truthfulqa_mc1_0shot`
- `truthfulqa_mc2_0shot`

The launcher uses the same 253-row 300M SNR candidate population as English-lite and scores every answer
choice with Levanter loglikelihood. It writes metrics under `mcq_smooth/<task>/...`. Single-gold tasks get
`choice_prob_norm`, `choice_logprob_norm`, `choice_logprob`, `choice_prob`, `logprob`, `bpb`, and `nll`.
TruthfulQA MC2 gets normalized probability mass over all true choices and intentionally omits ambiguous
single-gold `bpb`/`nll`.

Dry-run status:

- `253` candidates.
- `253` launchable rows.
- No duplicate checkpoint roots.
- All rows use exact `hf/step-22887`.
- The historical central1 `baseline_stratified` row keeps its original checkpoint root for metric joins,
  but scores the existing east5 HF mirror.

Final collection status:

- Iris parent `/calvinxu/dm-300m-mcq-smooth-proxy-20260430-075906` reported failed because five child
  steps failed after collection, but the collect step wrote a complete CSV.
- `300m_mcq_smooth_proxy_eval_results.csv` has `253` rows.
- All `253` rows have all expected MCQ smooth metrics present.
- Adding this CSV to the default SNR overlays resolves the formerly hard-only MCQ rows.

## Future Work

- [x] Ingest or flatten the existing Olmix attached MMLU artifact into the SNR overlay so subject-level MMLU reaches `signal_n=242`.
- [x] Rebuild SNR after those two gaps are filled.
- [x] Derive MMLU category smooth proxies locally from subject-level metrics.
- [x] Launch and collect MCQ smooth proxies for `medmcqa_5shot`, `sciq_5shot`, `swag_0shot`, and TruthfulQA.
- [ ] Fix the GSM8K/HumanEval completion launcher collector so future `skip_existing` rows copy existing metrics into the collected result CSV instead of depending on SNR-side overlay recovery.
