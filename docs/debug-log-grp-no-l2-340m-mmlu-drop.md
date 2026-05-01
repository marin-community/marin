# Debugging log for GRP no-L2 340M MMLU drop

## Initial Status

The MMLU scaling plot showed a sharp drop for `GRP no-L2` at corrected
`340M/10.4B`: MMLU accuracy is `25.60%`, below its `100M/6B` value (`26.84%`)
and far below its `900M/24B` value (`29.60%`).

## Hypothesis 1: Plot or merge bug

I traced the plotted value through:

- `baseline_scaling_downstream_eval_metrics_merged.csv`
- `baseline_scaling_downstream_eval_metrics_all_sources.csv`
- the collected result CSV at
  `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_baseline_scaling_downstream_evals_postrefresh_20260425-025012/collect_results-39a3d8/baseline_scaling_downstream_eval_results.csv`
- the underlying result artifact at
  `gs://marin-us-east5/evaluation/lm_evaluation_harness_levanter/lmeval_debug_baseline_scaling_grp_no_l2_520m_10p4b_mmlu_postrefresh-324348/results.json`

## Result

No plot/merge bug found. The merged value exactly matches the source result:

- `lm_eval/mmlu_5shot/acc = 0.2560176613`
- `lm_eval/averages/macro_avg_acc = 0.2569356332`
- `lm_eval/mmlu_5shot/bpb = 2.0835322016`
- `lm_eval/mmlu_5shot/choice_logprob = -1.436706`

The launcher state also points to the expected checkpoint:

- checkpoint root:
  `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_520m_10p4b/baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b-4c051e`
- HF checkpoint:
  `.../hf/step-19835`
- task suite: `mmlu`
- task aliases: `mmlu_5shot`

## Hypothesis 2: Incomplete or malformed MMLU result

I inspected the result JSON. It contains all expected MMLU subtasks:

- 57 leaf MMLU categories
- 14,042 total leaf examples
- aggregate `mmlu_5shot` metrics
- group/macro/micro averages

No missing categories or malformed aggregate were found.

## Result

The drop is broad, not a single-task artifact. GRP `340M` is worse than GRP
`100M` on many categories. The largest negative category deltas vs GRP `100M`
include:

- `professional_medicine`: `20.2%` vs `43.0%`
- `astronomy`: `19.7%` vs `32.9%`
- `logical_fallacies`: `23.3%` vs `35.0%`
- `high_school_computer_science`: `23.0%` vs `33.0%`
- `college_mathematics`: `20.0%` vs `29.0%`

This looks like a real multiple-choice accuracy regression, not a broken row.

## Hypothesis 3: The 340M point used a different GRP mixture

I compared the persisted `run_manifest.json` files for the GRP no-L2 baseline
scaling runs at:

- `20M/2.6B`
- `100M/6B`
- `340M/10.4B`
- `900M/24B`

All four run manifests have the same:

- `candidate_run_name = baseline_genericfamily_power_family_penalty_no_l2_raw_optimum`
- `candidate_source_experiment =
  pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb`
- phase-0 weights
- phase-1 weights

The phase-weight L1 distance between every pair of these four manifests is
exactly `0.0`. This rules out the main apples-to-apples concern for the GRP
line: the `340M` point did not use a different GRP mixture.

## Interpretation

Current evidence says this is probably not a plot/source bug. It is a
metric-specific failure mode:

- GRP no-L2 `340M` has very strong perplexity (`eval/uncheatable_eval/bpb =
  0.8422`), better than the other baselines at that scale.
- Its MMLU BPB/logprob also improves with scale, so the model is not globally
  worse at MMLU text likelihood.
- But its multiple-choice accuracy and normalized choice probability are bad at
  `340M`, then recover sharply at `900M`.

This suggests GRP no-L2 at `340M` may have a calibration/choice-ranking issue
that is not visible in perplexity.

## Remaining Confirmation

The clean confirmation would be to rerun only this MMLU eval once:

- checkpoint: `.../baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b-4c051e/hf/step-19835`
- suite: `mmlu_5shot`

If the rerun reproduces `~25.6%`, treat the drop as real. If it moves back into
the `28-30%` range, investigate nondeterminism or eval harness instability.
