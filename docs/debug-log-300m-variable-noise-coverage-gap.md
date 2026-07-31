# Debugging log for 300M variable-noise metric coverage gap

## Overview

Close the remaining fixed-vs-variable noise coverage gap in the 300M raw metric matrix and SNR outputs.

## Initial status

The rebuilt WSC273 variable-subset metrics fixed the IRT BPB item set: `variable_noise_available=43` and
`missing_variable=[]`. A broader all-metric comparison still showed 10 metrics present under fixed-subset
noise and absent under variable-subset noise:

- GSM8K hard exact-match metrics.
- HumanEval hard pass@1 metrics.
- MMLU-Pro top-level metrics.
- SocialIQA hard/smooth metrics.

## Hypothesis: local collection is stale for GSM8K/HumanEval

Remote GCS prefixes show partial variable GSM8K/HumanEval successes under:

- `ngd3dm2_300m_noise_gsmhe_20260501`
- `ngd3dm2_300m_noise_gsmhe_retry3_20260501`
- `ngd3dm2_300m_noise_gsmhe_retry4_20260501`

The local variable GSM8K/HumanEval CSV was just an uncollected state table with
`missing_executor_status`, because retry eval keys had different suffixes from the local state rows.

## Changes to make

- Update `launch_300m_gsm8k_humaneval_evals.py` collection to fall back to unique run-name matches when
  exact eval keys differ across retry submissions.
- Include local GSM8K/HumanEval result CSVs in candidate coverage so future state generation skips rows
  already collected locally.

## Hypothesis: MMLU-Pro is missing from the noise parity backfill

The noise parity launcher filled standard MMLU, SL-Verb MMLU, ARC Easy, PIQA, and HellaSwag aliases, but
not MMLU-Pro. Fixed-subset rows had MMLU-Pro from an older source; variable-subset rows did not.

## Changes to make

- Add `mmlu_pro_5shot` to `launch_300m_noise_parity_evals.py`.
- Add `--task-alias` to the parity launcher so this gap can be filled without rerunning the other parity
  aliases.

## Hypothesis: SocialIQA failed in the first variable English-lite run

The initial variable English-lite run has failed/running statuses; later successful recovery outputs were
created without SocialIQA. A targeted SocialIQA-only English-lite run should fill this gap without
rerunning the full suite.

## Changes to make

- Add a default overlay path for a targeted variable SocialIQA-only result CSV.
- Add `--task-alias` to the English-lite launcher so SocialIQA can be retried alone without the brittle
  shell-expanded exclusion list.

## Results

Submitted targeted remote gap-fill jobs on 2026-05-01:

- `/calvinxu/dm-300m-noise-gsmhe-gapfix-r3-20260501-182137`: three remaining variable GSM8K/HumanEval
  rows, regenerated from `MARIN_300M_CANDIDATE_PANELS=variable_subset_noise_300m_6b`.
- `/calvinxu/dm-300m-noise-socialiqa-gapfix-r3-20260501-182137`: ten variable SocialIQA-only rows using
  `--task-alias socialiqa_5shot`.
- `/calvinxu/dm-300m-noise-mmlupro-gapfix-r3-20260501-182137`: ten variable MMLU-Pro-only parity rows
  using `--task-alias mmlu_pro_5shot`.

The first SocialIQA-only attempt hit a systematic loader bug: upstream `social_iqa` uses the legacy
`social_i_qa` dataset script, which modern `datasets` refuses to execute. I stopped that job and replaced
the SocialIQA task with an equivalent inline JSON-backed task using pre-materialized AI2 train/dev JSONL
files at `gs://marin-us-east5/raw/eval-datasets/socialiqa-json-v1/`.

Resubmitted SocialIQA as `/calvinxu/dm-300m-noise-socialiqa-gapfix-r4-20260501-183426`.

While waiting for completion, I also fixed a collection footgun in
`launch_300m_english_lite_evals.py`: `--collect-from-prefix` now rebuilds state from the current
`--task-alias` and panel filters instead of blindly loading the last local state CSV. This matters for
targeted recovery jobs because stale state would either miss the targeted rows or accidentally collect
the wrong task family.

The targeted jobs completed successfully:

- GSM8K/HumanEval gap fill: 3/3 collected into
  `300m_gsm8k_humaneval_eval_results_variable_subset_noise.csv`.
- SocialIQA JSON gap fill: 10/10 collected into
  `300m_english_lite_eval_results_variable_subset_noise_socialiqa_only.csv`.
- MMLU-Pro gap fill: 10/10 collected into
  `300m_noise_parity_eval_results_variable_subset_mmlupro_only.csv`.

The MMLU-Pro collection initially reported 10 `missing_executor_status` rows even though GCS showed
successful children. Root cause: `launch_300m_noise_parity_evals.py` did not have the stable run-name
fallback used by the GSM/HumanEval and English-lite collectors. I added the same fallback there and
re-collected successfully.

Rebuilt:

- `raw_metric_matrix_300m.csv`: 242 signal rows.
- `noise_baseline_run00097_fixed_subset_300m.csv`: 10 fixed-subset rows.
- `noise_baseline_run00097_variable_subset_300m.csv`: 10 variable-subset rows.
- `raw_metric_matrix_300m_with_noise.csv`: 262 rows.
- fixed and variable SNR tables and keep/drop tables.
- IRT/factor-analysis outputs.
- fixed-vs-variable SNR plots.

Validation:

- The 10 formerly fixed-only metrics are now present and non-null for all 10 variable-subset noise rows.
- Fixed and variable SNR metric sets both contain 1060 metrics; `fixed_only=[]` and `variable_only=[]`.
- Every SNR row has `signal_n=242` and `noise_n=10`.
- IRT coverage is complete for the 43-item BPB set: `fixed_noise_available=43`,
  `variable_noise_available=43`, `missing_fixed=[]`, `missing_variable=[]`.
