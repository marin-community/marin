# StarCoder Heteroskedastic SNR: Research Logbook

## Scope
- Goal: estimate whether within-mixture noise varies across the two-phase StarCoder mixture landscape.
- Primary metrics: `eval/paloma/dolma_100_programing_languages/bpb`, `eval/uncheatable_eval/bpb`, `eval/loss`, and historical StarCoder lm-eval metrics.
- Constraints: reuse the historical central1 two-phase StarCoder/Nemotron proxy setup; do not relaunch all 143 dense historical points; make the panel append-only so additional anchors can be added later.
- Noise scope: this first panel estimates data-ordering/batching variance at fixed model initialization convention, because `trainer_seed=None` and `simulated_epoch_subset_seed=None`; total nuisance variance would require a later factorial seed panel.

## Baseline
- Date: 2026-05-23
- Code refs: `experiments/domain_phase_mix/two_phase_starcoder_experiment.py`, `experiments/domain_phase_mix/two_phase_starcoder_experiment_v5.py`, `experiments/domain_phase_mix/starcoder_metadata.py`.
- Data refs: `experiments/domain_phase_mix/exploratory/paper_plots/data/two_phase_starcoder_combined_143_from_wandb.csv`.
- Baseline numbers: 143 completed two-phase rows with complete historical metrics, but no exact repeated `(phase_0_starcoder, phase_1_starcoder)` coordinate groups at 6 decimal places.

## Experiment Log
### 2026-05-23 03:05 - repeat-anchor launcher setup
- Hypothesis: a small set of repeated anchors can estimate heteroskedastic within-mixture variance more cleanly than residuals from the dense historical StarCoder surface.
- Command: `uv run python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --dry-run`
- Config: 10 default anchors, 5 repeats per anchor, 50 training runs total, central1 `v5p-8`, historical seed convention `trainer_seed=None`, `data_seed=run_id`, `simulated_epoch_subset_seed=None`.
- Result: pending validation and Claude Code review before live submission.
- Interpretation: this directly targets heteroskedastic SNR while preserving the ability to append more anchors later via `--extra-anchor-csv`.
- Append discipline: add future anchors only with `--extra-anchor-csv`; do not reorder or replace the default anchors. If changing `--repeats`, use a new `--run-id-base` or a new cohort so run-id arithmetic stays unambiguous.
- Next action: run local tests/dry-run, obtain CC review, then submit from Iris if no blockers.

### 2026-05-23 03:34 - first parent submission failure
- Hypothesis: the CC-reviewed parent should dispatch 50 training children on central1.
- Command: `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-20260523-0333 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8`
- Config: 10 anchors, 5 repeats, local artifact dir defaulting under the repo.
- Result: parent `/calvinxu/dm-starcoder-hetero-snr-20260523-0333` failed before child dispatch with `FileNotFoundError` for the historical StarCoder CSV under `/app`.
- Interpretation: the launcher depended on a local source CSV that is not included in the Iris workspace bundle. Freeze the two observed anchors and source-panel SHA in code for the default panel, while still using the local CSV when present.
- Next action: patch launcher, test the missing-default-source fallback, get CC review, and resubmit.

### 2026-05-23 03:39 - retry1 parent submission failure
- Hypothesis: frozen default anchors remove the source-CSV packaging dependency.
- Command: `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-retry1-20260523-0340 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523`
- Config: same 50-run panel; `/tmp` local artifact dir to avoid parent container writeability risk.
- Result: parent `/calvinxu/dm-starcoder-hetero-snr-retry1-20260523-0340` passed anchor/materialization but failed in `cache_eval_datasets` with `ModuleNotFoundError: No module named 'lm_eval'`; no children dispatched.
- Interpretation: the parent job needs the project `eval` extra because the eval-dataset cache step imports `lm_eval.tasks`.
- CC review: use workspace-prefixed extras `--extra marin:tpu --extra marin:eval`, not bare `--extra eval`. Confirmed central1 cache manifest exists at `gs://marin-us-central1/raw/eval-datasets/code-tasks/.eval_datasets_manifest.json`.
- Next action: resubmit with `--extra marin:tpu --extra marin:eval`, `/tmp` local artifact dir, and a fresh parent name.

### 2026-05-23 03:46 - retry2 parent submission failure
- Hypothesis: installing `marin:eval` in the parent should let the cache step complete and dispatch training children.
- Command: `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-retry2-20260523-0352 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523`
- Config: same 50-run panel and extras recommended by CC.
- Result: parent `/calvinxu/dm-starcoder-hetero-snr-retry2-20260523-0352` got through materialization and `lm_eval` import, then failed in `cache_eval_datasets` after caching 21/22 datasets because `winograd_wsc:wsc273` failed under the current HF/datasets stack; no children dispatched.
- Interpretation: this pilot does not need WSC to answer the heteroskedastic SNR question. Keep the core BPB, commonsense, and code evals, but exclude only `wsc273` from the inline eval harness and analysis metric list. WSC can be backfilled separately if needed.
- Next action: patch `EVAL_TASKS` to use `CORE_TASKS` minus `wsc273`, test, get CC review, and resubmit with the same reviewed extras.

### 2026-05-23 03:56 - retry3 launch review
- Hypothesis: excluding only `wsc273` removes the failing HF dataset cache edge while preserving the metrics needed for the heteroskedastic SNR pilot.
- Command: `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk -p '<focused retry3 review prompt>'`
- Config: local checks passed before review: `uv run python -m py_compile experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py`, `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py`, and `uv run python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --dry-run`.
- Result: CC found no launch blockers and cleared retry3. Review specifically checked central1 locality, retry2 cache behavior, run-id/name/output collisions, rematerialization risk, and known pre-dispatch failures.
- Interpretation: retry3 should re-execute the same cache step body without `wsc273`; the 21 datasets cached during retry2 should either be skipped by per-dataset existence checks or reverified in place.
- Caveat: `eval_tasks` is not part of the cache step hash, so later re-including `wsc273` will require an explicit out-of-band cache refresh or cache-version bump; simply removing it from `EXCLUDED_INLINE_EVAL_TASKS` may not rematerialize that dataset.
- Next action: submit retry3 with `--extra marin:tpu --extra marin:eval`, central1 placement, `/tmp` local artifact dir, and `max_concurrent=8`.

### 2026-05-23 04:07 - retry3 children dispatched
- Hypothesis: retry3 should clear parent-side cache/materialization and create training children.
- Command: `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-retry3-20260523-0356 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523`
- Config: same 10 anchors x 5 repeats panel; `wsc273` excluded from inline eval; central1 `v5p-8`; `max_concurrent=8`.
- Result: parent `/calvinxu/dm-starcoder-hetero-snr-retry3-20260523-0356` wrote executor metadata at `gs://marin-us-central1/experiments/launch_starcoder_heteroskedastic_snr-d16f97.json`, warmed the 21-dataset eval cache, and reported `has_children=true`. The first 8 checkpoint children were submitted and are pending due to central1 TPU availability, not due to launcher/cache failure.
- Interpretation: the known pre-dispatch blockers are cleared. Remaining risk is ordinary scheduling/TPU availability and downstream training/eval failures.
- Fieldbook: created experiment `exp_01ksa8e32nbpq6f57f0pj4scta` (`StarCoder heteroskedastic SNR repeat anchors`), populated 50 datapoint runs, recorded the failed parent attempts and live retry3 parent, linked retry lineage, linked retry3 to all 50 training runs, and attached local manifests/logbook.
- Next action: check child progress later; if individual children fail, prefer resubmitting the same parent/graph or using executor skip semantics instead of writing narrow one-off failure-only launchers unless the failure mode requires a code/config change.

### 2026-05-23 16:18 - retry4 submitted after child dependency fix
- Failure:
  Retry3 parent dispatched all 50 training children, but every child failed inside the inline eval harness with `ModuleNotFoundError: No module named 'lm_eval'`.
- Root cause:
  The Iris parent had `marin:eval`, but the child TPU training steps were old-style `ExecutorStep` callables. The StepRunner inferred only the `tpu` dependency group for child Fray submissions, so the child env lacked the `eval` extra.
- Fix:
  `experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py` now wraps configured training steps with `RemoteCallable.pip_dependency_groups=["eval"]`, preserving the original resources and output paths. The StepRunner merges this with resource-inferred `tpu`.
- Validation:
  `uv run python -m py_compile experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py experiments/grug/moe/launch_v4_path_logprob_evals.py experiments/grug/moe/eval_logprob.py`

  `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_grug_logprob_eval.py -q`

  `uv run python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --dry-run --local-artifact-dir /tmp/starcoder_hetero_snr_20260523_retry4_check`
- Claude Code review:
  Reviewed with `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`; preflight showed `plambdafour@proton.me`, `stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. Verdict: no blockers.
- Submitted:
  `/calvinxu/dm-starcoder-hetero-snr-retry4-20260523-161834`

  Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-retry4-20260523-161834 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523`
- Immediate status:
  Parent was `JOB_STATE_RUNNING`.
- Fieldbook:
  Added retry job `job_01ksbj7ayrk3nzskzhc79cc0qa`, linked as a retry of retry3, and added a debug note on `exp_01ksa8e32nbpq6f57f0pj4scta`.

### 2026-05-23 22:23 - retry4 systematic lm-eval tokenizer failure
- Status:
  retry4 has `34` failed children, `8` running children, and `1` pending child.
  Failed children reached the inline lm-eval phase, then crashed in
  `LevanterHarnessLM.loglikelihood` with
  `dataclasses.FrozenInstanceError: cannot assign to field 'pad_token_id'`.
- Root cause:
  Marin's HF-backed tokenizer wrapper is a frozen dataclass with a read-only
  `pad_token_id` property. The eval harness tried to set
  `self.tokenizer.pad_token_id = self.tokenizer.eos_token_id` when no pad token
  was present.
- Local fix:
  `lib/levanter/src/levanter/eval_harness.py` now resolves an effective pad
  token id locally and passes it to request packing/padding statistics without
  mutating the tokenizer.
- Validation:
  - `uv run python -m py_compile lib/levanter/src/levanter/eval_harness.py experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py`
  - `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py -q`
  - Ad-hoc frozen-tokenizer check for `_effective_pad_token_id`.
  - Full `lib/levanter/tests/test_eval_harness.py` collection remains blocked
    by a pre-existing missing `_enable_hf_offline_mode_for_eval_cache` symbol,
    unrelated to this patch.
- Claude Code review:
  `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, session
  `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: no blockers for the patch
  or retry plan. CC recommended waiting for retry4's remaining old-bundle
  children to terminate before submitting retry5, because those children cannot
  pick up the local patch.
- Next action:
  Once retry4 has no running/pending children, submit retry5 with the same
  launcher/output roots so succeeded rows skip and failed rows rerun under the
  fixed eval harness bundle.

### 2026-05-24 01:36 - retry5 submitted with frozen-tokenizer fix
- Status before submit:
  retry4 had reached a terminal state with `51` failed tasks and no running or
  pending children. The systematic failure was still
  `dataclasses.FrozenInstanceError: cannot assign to field 'pad_token_id'`.
- Validation:
  - `uv run python -m py_compile lib/levanter/src/levanter/eval_harness.py experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py`
  - `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py -q`
  - `uv run python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --dry-run --local-artifact-dir /tmp/starcoder_hetero_snr_20260524_retry5_review`
- Claude Code review:
  `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`
  after preflight showed `plambdafour@proton.me`, `stripe_subscription`, and
  no inherited `ANTHROPIC_API_KEY`. Verdict: no blockers; retry5 cleared.
- Submitted:
  `/calvinxu/dm-starcoder-hetero-snr-retry5-20260524-0135`

  Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-retry5-20260524-0135 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523`
- Fieldbook:
  Added retry job `job_01kscj34k6xpqpgjqava5x7fhk`, linked as a retry of
  retry4, and added a debug note on `exp_01ksa8e32nbpq6f57f0pj4scta`.
- Next action:
  Check that the parent dispatches children and that the first rerun reaches
  inline lm-eval without the frozen-tokenizer crash.

### 2026-05-24 02:35 - retry6 submitted after MarinTokenizer tok_encode fix
- Status:
  retry5 progressed past the frozen `pad_token_id` mutation, but the first five
  children failed in `LevanterHarnessLM.generate_until` with
  `TypeError: 'HfMarinTokenizer' object is not callable`.
- Root cause:
  `tok_encode` still assumed an HF-style callable tokenizer. The live
  StarCoder eval path uses `HfMarinTokenizer`, which implements `encode_batch`
  but is not callable.
- Fix:
  `lib/levanter/src/levanter/eval_harness.py` now preserves the callable HF
  tokenizer path and uses `encode_batch` for non-callable Marin tokenizers,
  preserving the scalar vs batch return shape.
- Validation:
  - `uv run pytest tests/test_levanter_eval_harness_tokenizer.py -q`
  - `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_levanter_eval_harness_tokenizer.py -q`
  - `uv run python -m py_compile lib/levanter/src/levanter/eval_harness.py experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py tests/test_levanter_eval_harness_tokenizer.py`
  - `uv run python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --dry-run --local-artifact-dir /tmp/starcoder_hetero_snr_20260524_retry6_review`
- Claude Code review:
  `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`
  after preflight showed `plambdafour@proton.me`, `stripe_subscription`, and
  no inherited `ANTHROPIC_API_KEY`. Verdict: no blockers; retry6 cleared.
- Submitted:
  `/calvinxu/dm-starcoder-hetero-snr-retry6-20260524-0235`

  Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-starcoder-hetero-snr-retry6-20260524-0235 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523`
- Fieldbook:
  Marked retry5 `killed`, added retry6 job `job_01kscnfehrhvkyzt4e2ebjegsk`,
  linked it as a retry of retry5, and added a debug note on
  `exp_01ksa8e32nbpq6f57f0pj4scta`.
- Next action:
  Smoke-check retry6 parent dispatch and then verify the first child reaches
  generation eval without the tokenizer-callability crash.

### 2026-05-24 - region-local policy update

Retry6 is now terminal with `86` failed jobs and `6` killed jobs. A stop request
found no running jobs under
`/calvinxu/dm-starcoder-hetero-snr-retry6-20260524-0235`.

This launcher is central1-by-design: the code-task eval cache and tokenizer
cache point at `gs://marin-us-central1`. It was not itself a cross-region
placement bug because retry6 used explicit `--region us-central1 --zone
us-central1-a`.

The current policy is region-local, not east5-only for this historical panel.
StarCoder can run in central1 as long as every parent, child, cache, tokenizer,
checkpoint, state, and executor path remains central1-local.

Before any further StarCoder retry, validate the exact live command with:

`uv run python -m experiments.domain_phase_mix.east5_launch_safety --expected-region us-central1 --expected-zone us-central1-a --expected-bucket-prefix gs://marin-us-central1 --command '<iris job run ...>'`

The remaining blocker is not region locality; it is the terminal retry6 job
state and any still-unfixed eval/runtime failure behind those failed children.

### 2026-05-24 18:24 - retry7 submitted after scalar tok_decode fix

Retry6 failed systematically in Levanter eval-harness generation with
`TypeError: argument 'ids': 'int' object cannot be converted to 'Sequence'`.
The remaining runtime bug was scalar decode of `eot_token_id`, after the earlier
retry6 patch had already fixed non-callable Marin tokenizer encoding.

Fix:
`lib/levanter/src/levanter/eval_harness.py` now routes generation-path decode
through `LevanterHarnessLM.tok_decode`, which wraps scalar integer token IDs
before calling `tokenizer.decode` and preserves list/array passthrough behavior.

Validation:
- `uv run pytest tests/test_levanter_eval_harness_tokenizer.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_east5_launch_safety.py -q`
  passed with `23 passed`.
- `uv run ruff check lib/levanter/src/levanter/eval_harness.py tests/test_levanter_eval_harness_tokenizer.py experiments/domain_phase_mix/east5_launch_safety.py tests/test_domain_phase_mix_east5_launch_safety.py`
  passed.
- `uv run python -m py_compile lib/levanter/src/levanter/eval_harness.py tests/test_levanter_eval_harness_tokenizer.py experiments/domain_phase_mix/east5_launch_safety.py tests/test_domain_phase_mix_east5_launch_safety.py`
  passed.
- `uv run python -m experiments.domain_phase_mix.east5_launch_safety --expected-region us-central1 --expected-zone us-central1-a --expected-bucket-prefix gs://marin-us-central1 --command-file /tmp/starcoder_retry7_command.txt --json`
  passed with no errors or warnings.

Claude Code review:
`env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`
after preflight showed `plambdafour@proton.me`, `stripe_subscription`, and no
inherited `ANTHROPIC_API_KEY`. Verdict: no blockers; submit retry7.

Submitted:
`/calvinxu/dm-starcoder-hetero-snr-retry7-20260525-0119`

Command:
`uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-starcoder-hetero-snr-retry7-20260525-0119 --region us-central1 --zone us-central1-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523 --tpu-region us-central1 --tpu-zone us-central1-a --eval-datasets-cache-path gs://marin-us-central1/raw/eval-datasets/code-tasks`

Initial Iris status:
parent accepted and `JOB_STATE_PENDING`; pending reason was central1 CPU worker
capacity, not a validation or region-locality failure.

Fieldbook:
added retry job `job_01ksebsrse7jbjrf0qc8899rmz`, linked as a retry of
retry6, resolved stale east5-only policy notes, and added a debug note on
`exp_01ksa8e32nbpq6f57f0pj4scta`.

Next action:
check that the parent dispatches children and that the first child reaches
generation eval without the scalar decode crash.

### 2026-05-25 04:23 - east5 retry9 submitted after central1 parent stall

Retry7 remained pending in central1 before child dispatch. Iris reported the
parent needed `16GB` memory while central1 had only `14.1GB` available and was
waiting on CPU worker scale-up.

Region-local audit:
- `gs://marin-us-east5/tokenized/nemotron_cc/{hq_actual,hq_synth,medium_high,medium,medium_low,low_actual}-*`
  all had successful caches.
- `gs://marin-us-east5/tokenized/dolma/starcoder-8b6089` exists; this is the
  actual StarCoder domain used by `STARCODER_DOMAIN`.
- `gs://marin-us-east5/raw/tokenizers/meta-llama--Meta-Llama-3.1-8B` exists.
- `gs://marin-us-east5/raw/eval-datasets/code-tasks` exists.

Fix:
`experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py` is now
region-parametric. `tpu_region` drives `MARIN_PREFIX`,
`MARIN_TOKENIZER_CACHE_PATH`, eval-cache defaulting, and checkpoint output
paths. The launcher rejects eval-cache overrides that point at a different
Marin bucket prefix than the selected TPU region.

Validation:
- `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py -q`
  passed with `8 passed`.
- `uv run ruff check experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py`
  passed.
- `uv run python -m py_compile experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py`
  passed.
- `uv run python -m experiments.domain_phase_mix.east5_launch_safety --expected-region us-east5 --expected-zone us-east5-a --expected-bucket-prefix gs://marin-us-east5 --command-file /tmp/starcoder_retry9_east5_command.txt --json`
  passed with no errors or warnings.

Claude Code review:
Used `env -u ANTHROPIC_API_KEY claude --resume
d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max`
after subscription preflight. Patch review and focused retry9 command review
both returned no blockers.

Actions:
- Stopped retry7:
  `/calvinxu/dm-starcoder-hetero-snr-retry7-20260525-0119`.
- Submitted retry8 on east5 with the old `16GB/1CPU/20GB` parent; it remained
  pending on east5 CPU VM capacity, so it was stopped before child dispatch.
- Submitted retry9:
  `/calvinxu/dm-starcoder-hetero-snr-retry9-east5-20260525-0420`.

Retry9 command:
`uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-starcoder-hetero-snr-retry9-east5-20260525-0420 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 0.5 --memory 8GB --disk 5GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --base-name-prefix pinlin_calvin_xu/data_mixture/t2s_heteroskedastic_snr_20260523_east5_retry9 --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523_east5_retry9 --tpu-region us-east5 --tpu-zone us-east5-a --eval-datasets-cache-path gs://marin-us-east5/raw/eval-datasets/code-tasks`

Initial retry9 status:
parent is `JOB_STATE_RUNNING`; task moved from `building` to `running`.
Logs show only east5 GCS paths so far and no disk-full/OOM during startup. The
parent is currently refreshing/downloading the code-task eval cache into the
east5 cache path before child dispatch.

Fieldbook:
updated retry7 to `killed`, recorded retry8 as `killed`, recorded retry9 as
`running`, and added a debug note on `exp_01ksa8e32nbpq6f57f0pj4scta`.

Next action:
check that retry9 completes eval-cache preparation and dispatches 50 `v5p-8`
children in `us-east5-a`. If the parent fails with disk pressure, resubmit the
same command with `--disk 10GB`.

### 2026-05-25 18:40 - retry13 running after east5 cache-manifest repair

Retry9 failed before child dispatch with parent exit `137` while preparing or
uploading the east5 code-task eval cache. Retrying with larger parent memory did
not solve the scheduling problem:

- retry10 used an east5-a parent with `1CPU/32GB/20GB`; Iris could not schedule
  it because only `16GB` was available.
- retry11 used an east5-a parent with `1CPU/16GB/20GB`; it still remained
  pending on CPU capacity.
- retry12 used an east5-a parent with `0.5CPU/8GB/20GB`; it still remained
  pending on saturated east5-a CPU worker capacity.

The root cause for retry9 was not missing datasets. Audit showed the current
StarCoder eval tasks were already cached under
`gs://marin-us-east5/raw/eval-datasets/code-tasks`, but the manifest was legacy:
it lacked `cache_layout_version`, `includes_hf_hub_cache`, and
`includes_hf_modules_cache`. That made `CacheManifest.supports_full_offline_task_loading()`
false, so the parent took the expensive cache-materialization path again.

Fix:
backed up the old manifest to
`gs://marin-us-east5/raw/eval-datasets/code-tasks/.eval_datasets_manifest.pre_repair_20260525_retry11.json`
and uploaded a repaired manifest with:

- `cache_layout_version=2`
- `includes_hf_hub_cache=true`
- `includes_hf_modules_cache=true`
- current task names
- existing cached dataset entries and `failed_datasets=0`

Verification:
`load_cache_manifest("gs://marin-us-east5/raw/eval-datasets/code-tasks")`
reported `failed_datasets=0`, `cached_datasets=22`, `layout=2`, `hub=True`,
`modules=True`, and `fast_path=True`.

Validation:
- `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_levanter_eval_harness_tokenizer.py -q`
  passed with `11 passed`.
- `uv run python -m py_compile experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py lib/levanter/src/levanter/eval_harness.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_levanter_eval_harness_tokenizer.py`
  passed.

Claude Code review:
Used `env -u ANTHROPIC_API_KEY claude --resume
d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max`
after subscription preflight showed `plambdafour@proton.me`,
`stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. CC reviewed the
retry10 command, the manifest repair plus retry11, the retry12 reduced-parent
command, and finally the retry13 region-only-parent command. Verdict for retry13:
no blockers. CC noted that `east5_launch_safety.py` is stricter than necessary
for this case because it requires a parent zone, while the safe property here is
parent region `us-east5`, child TPU zone `us-east5-a`, and all GCS paths under
`gs://marin-us-east5`.

Actions:
- Stopped retry10:
  `/calvinxu/dm-starcoder-hetero-snr-retry10-east5-20260525-1818`.
- Stopped retry11:
  `/calvinxu/dm-starcoder-hetero-snr-retry11-east5-20260525-1824`.
- Stopped retry12:
  `/calvinxu/dm-starcoder-hetero-snr-retry12-east5-20260525-1835`.
- Submitted retry13 with an east5 region-pinned CPU parent and explicit east5-a
  child TPUs:
  `/calvinxu/dm-starcoder-hetero-snr-retry13-east5-region-parent-20260525-1838`.

Retry13 command:
`uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-starcoder-hetero-snr-retry13-east5-region-parent-20260525-1838 --region us-east5 --enable-extra-resources --cpu 0.5 --memory 8GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --base-name-prefix pinlin_calvin_xu/data_mixture/t2s_heteroskedastic_snr_20260523_east5_retry13 --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523_east5_retry13 --tpu-region us-east5 --tpu-zone us-east5-a --eval-datasets-cache-path gs://marin-us-east5/raw/eval-datasets/code-tasks`

Current retry13 status:
parent is `JOB_STATE_RUNNING` and has dispatched the first 8 child jobs.
Children are `JOB_STATE_PENDING` on `tpu_v5p-preemptible_8-us-east5-a`
availability with quota-pool tier monotonicity, not failing due to launcher,
cache, or cross-region data issues.

Fieldbook:
recorded retry10, retry11, and retry12 as `killed`; recorded retry13 as
`running`; linked retry lineage through the previous failed jobs; and added a
debug note on `exp_01ksa8e32nbpq6f57f0pj4scta`.

Next action:
monitor retry13 until the first child reaches `JOB_STATE_RUNNING`. If children
remain pending only on TPU capacity, no code or submission fix is needed.

### 2026-05-25 23:43 - retry15 train-only submitted after inline eval failures

Retry13 was stopped after the first child hit an inline eval-harness
`compute_axis_resources` error in `generate_until`. After patching that Levanter
path and getting CC review, retry14 was submitted with the same east5
region-only parent pattern:
`/calvinxu/dm-starcoder-hetero-snr-retry14-east5-region-parent-20260525-2006`.

Retry14 then exposed a broader problem with inline lm-eval during training for
this StarCoder/Nemotron configuration. Current Iris state before stopping was 7
failed children, 7 running children, 2 pending children, plus the parent.
Observed failures included `SIGSEGV` in JAX distributed startup
(`add_port [::]:8482`) and XLA memory/OOM-like failures in generation prefill.
Fieldbook still showed zero recorded checkpoint/eval coverage for the 50
datapoints, so continuing retry14 was likely to spend more TPU time before
failing at the inline eval boundary.

Fix:
`experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py` now has
`--skip-inline-eval-harness`, which sets `LEVANTER_SKIP_EVAL_HARNESS=1` on child
training jobs after the region-local eval-cache dependency is attached. This
keeps normal training, Levanter validation/perplexity, final checkpoint writing,
and final HF export, while suppressing the brittle inline lm-eval/generation
metrics. Those lm-eval/code metrics must be backfilled later from the preserved
checkpoints.

The regional launch safety checker now has an explicit
`--allow-region-only-parent` mode. The default still rejects missing parent
zones, but this opt-in mode allows the known safe shape used here: Iris parent
`--region us-east5`, child TPU `--tpu-region us-east5 --tpu-zone us-east5-a`,
and all Marin GCS paths under `gs://marin-us-east5`.

Validation:
- `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_east5_launch_safety.py -q`
  passed with `24 passed`.
- `uv run python -m py_compile experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py experiments/domain_phase_mix/east5_launch_safety.py tests/test_domain_phase_mix_east5_launch_safety.py`
  passed.
- `uv run python -m experiments.domain_phase_mix.east5_launch_safety --expected-region us-east5 --expected-zone us-east5-a --expected-bucket-prefix gs://marin-us-east5 --allow-region-only-parent --command-file /tmp/starcoder_retry15_train_only_command.txt --json`
  returned `ok=true`, `errors=[]`, parent region `us-east5`, no parent zone,
  child TPU region `us-east5`, child TPU zone `us-east5-a`, and only
  `gs://marin-us-east5/raw/eval-datasets/code-tasks` among explicit Marin GCS
  paths. The one warning records the intentional region-only parent.

Claude Code review:
Used `env -u ANTHROPIC_API_KEY claude --resume
d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max`
after subscription preflight showed `plambdafour@proton.me`,
`stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. CC reviewed the
train-only patch, exact retry15 command, and the region-only-parent safety patch;
both reviews returned no blockers and recommended stopping retry14 and submitting
retry15.

Actions:
- Stopped retry14:
  `/calvinxu/dm-starcoder-hetero-snr-retry14-east5-region-parent-20260525-2006`.
- Submitted retry15:
  `/calvinxu/dm-starcoder-hetero-snr-retry15-east5-train-only-20260525-2340`.

Retry15 command:
`uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-starcoder-hetero-snr-retry15-east5-train-only-20260525-2340 --region us-east5 --enable-extra-resources --cpu 0.5 --memory 8GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --base-name-prefix pinlin_calvin_xu/data_mixture/t2s_heteroskedastic_snr_20260523_east5_retry15_train_only --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523_east5_retry15_train_only --tpu-region us-east5 --tpu-zone us-east5-a --eval-datasets-cache-path gs://marin-us-east5/raw/eval-datasets/code-tasks --skip-inline-eval-harness`

Initial retry15 status:
parent submitted successfully and is `JOB_STATE_RUNNING` with the parent task in
`building`.

Fieldbook:
updated retry14 to `killed`, added retry15 as `running` with retry lineage,
archived the old retry14 live-child-health failure validation, added a passing
retry15 train-only submission validation, resolved retry14 next-action notes, and
added a new next-action to monitor retry15 dispatch and final checkpoint export.

Next action:
monitor retry15 until it dispatches child TPU jobs in `us-east5-a`, then check
that train-only children reach final HF export. After checkpoints land, launch a
follow-up eval-only job to backfill the suppressed lm-eval/code metrics.

### 2026-05-26 - retry15 live status refresh

Live Iris check of
`/calvinxu/dm-starcoder-hetero-snr-retry15-east5-train-only-20260525-2340`
showed the parent remains running and has dispatched child TPU jobs in
`us-east5-a`.

Current visible state:

- `36` child trainings succeeded.
- `9` child trainings are running.
- `2` child jobs failed with us-east5 GCS egress-bandwidth 429s.
- The parent is still running.

Do not submit a retry while the parent is active. After retry15 becomes
terminal, retry the failed/missing rows with CC review and preserve east5
locality. The 429s look like transient regional GCS bandwidth pressure rather
than a code bug.

Fieldbook:

- Refreshed retry15's running job timestamp.
- Archived the older retry15 child-health validation.
- Added an updated warning validation with the current counts.

### 2026-05-26 - retry15 terminal with two transient GCS 429 failures

Live Iris check showed
`/calvinxu/dm-starcoder-hetero-snr-retry15-east5-train-only-20260525-2340`
is terminal failed because `2` child training steps failed.

Final visible state:

- `48` child trainings succeeded.
- `2` child trainings failed.
- Both failures were us-east5 GCS egress-bandwidth 429s, not deterministic code
  failures.

Failed children:

- `t2s_snr_a04_r01`
- `t2s_snr_a03_r03`

Interpretation:

The remaining failure is a narrow transient-infrastructure gap, but the current
StarCoder launcher does not expose an exact `--only-run-name-file` or equivalent
selector. Do not rerun the full 50-row graph to recover 2 rows. The next fix
should add an exact run-name subset retry path to
`experiments/domain_phase_mix/launch_starcoder_heteroskedastic_snr.py`, dry-run
only these two repeats, get CC review, then submit a 2-row retry.

Fieldbook:

- Marked retry15 as `failed` with the 2-row GCS 429 reason.
- Archived stale live-health warning.
- Added a warning validation recording `48` succeeded and `2` failed.

### 2026-05-26 - retry16 old-prefix executor-skip retry submitted

Rationale:

The initial plan was to add an exact run-name selector and submit only
`t2s_snr_a04_r01` and `t2s_snr_a03_r03`. After revisiting executor semantics,
we switched to the cleaner operational pattern: resubmit the full retry15 graph
with the same `--base-name-prefix` and let executor cache skip completed rows.

Evidence:

- Dry-run with
  `--base-name-prefix pinlin_calvin_xu/data_mixture/t2s_heteroskedastic_snr_20260523_east5_retry15_train_only`
  prepared the same `10` anchors x `5` repeats = `50` training runs.
- A known succeeded row,
  `t2s_snr_a00_r00`, has `.artifact.json` and final `hf/step-3813` files under
  its deterministic output root.
- The two failed rows, `t2s_snr_a04_r01` and `t2s_snr_a03_r03`, only have
  partial executor metadata and `checkpoints/eval_metrics.jsonl`, with no
  `.artifact.json`.
- Safety gate passed with `ok=true`, parent region `us-east5`, child TPU
  region/zone `us-east5/us-east5-a`, and only `gs://marin-us-east5` explicit
  Marin GCS paths. The parent intentionally remains region-only to avoid
  `us-east5-a` CPU capacity stalls.
- Validation passed:
  `uv run pytest tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_domain_phase_mix_east5_launch_safety.py -q`
  reported `24 passed`, and `uv run python -m py_compile` passed on the launcher
  and safety files.

Claude Code review:

Used `env -u ANTHROPIC_API_KEY claude --resume
d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max`
after subscription preflight showed `plambdafour@proton.me`,
`stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. CC verdict: no
blockers; submit retry16. It confirmed executor skip-on-`.artifact.json`
semantics should skip the 48 completed rows and rerun only the two failed roots
because `force_run_failed=True` is the default.

Submitted:

`/calvinxu/dm-starcoder-hetero-snr-retry16-east5-old-prefix-skip-20260526-1245`

Command:

`uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-starcoder-hetero-snr-retry16-east5-old-prefix-skip-20260526-1245 --region us-east5 --enable-extra-resources --cpu 0.5 --memory 8GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_starcoder_heteroskedastic_snr --base-name-prefix pinlin_calvin_xu/data_mixture/t2s_heteroskedastic_snr_20260523_east5_retry15_train_only --max-concurrent 8 --local-artifact-dir /tmp/starcoder_hetero_snr_20260523_east5_retry16_old_prefix_skip --tpu-region us-east5 --tpu-zone us-east5-a --eval-datasets-cache-path gs://marin-us-east5/raw/eval-datasets/code-tasks --skip-inline-eval-harness`

Fieldbook:

Added retry16 as a running retry of retry15, resolved the obsolete exact-selector
next action, resolved the retry15 monitoring next action, added a passing
submission validation, and opened a retry16 monitoring next action.

Next action:

Monitor retry16. Expected behavior is immediate executor skipping for the 48
completed rows and fresh/recovered execution only for `t2s_snr_a04_r01` and
`t2s_snr_a03_r03`. If either row fails while resuming from a partial checkpoint,
diagnose before deleting any row-specific output root.

Initial status:

The first post-submit Iris check confirmed the expected executor-cache behavior:
the retry16 prefix has the running parent plus exactly two child jobs, one for
`t2s_snr_a04_r01` and one for `t2s_snr_a03_r03`. No children were dispatched for
the 48 rows that already had `.artifact.json`.

### 2026-05-26 - preliminary 48-row variance readout

Read small metadata/eval files from the east5 checkpoint roots for retry15's
old-prefix graph. Current clean panel:

- `48` final-step rows with `.artifact.json` and `latest_step=3813`.
- `2` failed retry15 rows have only partial `eval_metrics.jsonl` at steps
  `2000` and `3000`; they are excluded from the variance readout and are the
  two rows retry16 is currently rerunning.
- Inline lm-eval was intentionally skipped, so this readout covers Levanter
  train/eval metrics only, not downstream lm-eval/code metrics.

Artifacts:

- `experiments/domain_phase_mix/exploratory/reference_outputs/starcoder_heteroskedastic_snr_20260523/collected_train_only_metrics_live.csv`
- `experiments/domain_phase_mix/exploratory/reference_outputs/starcoder_heteroskedastic_snr_20260523/preliminary_key_metric_by_anchor.csv`
- `experiments/domain_phase_mix/exploratory/reference_outputs/starcoder_heteroskedastic_snr_20260523/preliminary_heteroskedastic_metric_summary.csv`

Headline within-anchor standard deviations:

| Metric | Quiet anchor std | Loud anchor std | Max/min |
| --- | ---: | ---: | ---: |
| `eval/bpb` | `0.000375` (`observed_global_best`) | `0.05335` (`starcoder_only`) | `142x` |
| `eval/paloma/dolma_100_programing_languages/bpb` | `0.000760` (`late_code_moderate`) | `0.04725` (`starcoder_only`) | `62x` |
| `eval/uncheatable_eval/bpb` | `0.000640` (`late_code_moderate`) | `0.05150` (`starcoder_only`) | `80x` |
| `eval/paloma/macro_bpb` | `0.000453` (`proportional`) | `0.06511` (`starcoder_only`) | `144x` |

Even excluding the pathological `starcoder_only` anchor, within-anchor standard
deviation still varies substantially:

- `eval/bpb`: `6.8x`.
- `eval/paloma/dolma_100_programing_languages/bpb`: `13.0x`.
- `eval/uncheatable_eval/bpb`: `11.6x`.
- `eval/paloma/macro_bpb`: `13.5x`.

Interpretation:

- This is strong preliminary evidence that noise is heteroskedastic across the
  StarCoder mixture landscape for smooth train/eval BPB metrics.
- The high-StarCoder pathological region is both worse on mean BPB and much
  noisier. Near the useful late-code/moderate-code region, noise is much lower.
- The top three code-BPB anchors are close enough that ranking them precisely
  still needs the final two repeats and possibly more local repeats:
  `observed_global_best` mean `0.9128`, `observed_p0_zero_slice_best` mean
  `0.9130`, and `late_code_moderate` mean `0.9152`.
- This first panel estimates data-ordering/batching variance under the current
  fixed trainer/init convention, not total nuisance variance across all seeds.

### 2026-05-26 22:42 PDT - heteroskedastic landscape plot

Confirmed retry16 succeeded, refreshed the two stale local collection rows from
their final `eval_metrics.jsonl` records, and built an interactive Plotly
version of the StarCoder two-phase landscape that overlays the completed
repeat-anchor panel.

Command:

`uv run --with kaleido==0.2.1 python -m experiments.domain_phase_mix.exploratory.paper_plots.starcoder_two_phase_heteroskedastic_landscape`

Artifacts:

- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_heteroskedastic_landscape.html`
- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_heteroskedastic_landscape.png`
- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_heteroskedastic_anchor_summary.csv`

Validation:

- `uv run python -m py_compile experiments/domain_phase_mix/exploratory/paper_plots/starcoder_two_phase_heteroskedastic_landscape.py tests/test_starcoder_heteroskedastic_landscape.py`
- `uv run pytest tests/test_starcoder_heteroskedastic_landscape.py -q` reported `3 passed`.
- Retry16 Iris status:
  `uv run iris --cluster=marin job summary /calvinxu/dm-starcoder-hetero-snr-retry16-east5-old-prefix-skip-20260526-1245 --json`
  reported parent `state=succeeded`, `exit_code=0`.
- Refreshed `t2s_snr_a03_r03` and `t2s_snr_a04_r01` from their final
  `checkpoints/eval_metrics.jsonl` records. All 10 anchors now have 5 rows at
  `latest_step=3813` in
  `collected_train_only_metrics_live.csv`.

Plot semantics:

- Background surface: the historical 143-row two-phase StarCoder landscape for
  `eval/paloma/dolma_100_programing_languages/bpb`.
- Left overlay: anchor points colored by `log10` within-anchor repeat variance
  for the selected metric.
- Right overlay: anchor points colored by absolute contrast SNR versus the
  proportional anchor, defined as
  `|mean(anchor)-mean(proportional)| / pooled_standard_error`.
- Marker z-position is always the repeat mean for the Code BPB target metric,
  so both overlay panels remain on the same landscape geometry.

Data caveat:

- The current figure uses all `50` final-step rows at `latest_step=3813`.
- Earlier preliminary summaries used `48` final-step rows and excluded two
  partial-step retry15 rows. Those rows have now been refreshed from retry16's
  successful final outputs.

### 2026-05-27 18:25 PDT - 3D confidence bars

Updated the StarCoder heteroskedastic landscape plot to draw vertical 3D
confidence bars at the evaluated repeat anchors.

Command:

`uv run --with kaleido==0.2.1 python -m experiments.domain_phase_mix.exploratory.paper_plots.starcoder_two_phase_heteroskedastic_landscape`

Artifacts:

- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_heteroskedastic_landscape.html`
- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_heteroskedastic_landscape.png`
- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_heteroskedastic_anchor_summary.csv`

Plot semantics:

- Marker z-position remains the repeat mean for the Code BPB target
  `eval/paloma/dolma_100_programing_languages/bpb`, so both 3D panels stay on
  the original StarCoder landscape geometry.
- The vertical bars show approximate normal 95% CIs for that Code-BPB
  z-position, using `1.96 * std / sqrt(n)` from the repeat panel.
- Marker color and size still use the selected dropdown metric: left panel for
  within-anchor repeat variance, right panel for contrast SNR versus the
  proportional anchor.
- This means the CI bars quantify uncertainty in the plotted z-coordinate, not
  a per-selected-metric y-axis. A per-metric CI plot would need a different
  z-axis or a separate 2D view.

Validation:

- `uv run python -m py_compile experiments/domain_phase_mix/exploratory/paper_plots/starcoder_two_phase_heteroskedastic_landscape.py tests/test_starcoder_heteroskedastic_landscape.py`
- `uv run pytest tests/test_starcoder_heteroskedastic_landscape.py -q` reported `3 passed`.
- Regeneration reported all `50` final-step rows and `0` excluded partial rows.

### 2026-05-27 18:45 PDT - local gradient-SNR landscape

Built a dedicated local-SNR plot for the StarCoder two-phase panel.

Definition:

For metric \(b\), estimate a local weighted-linear response surface
\(\hat\mu_b(w)\) from the historical 143 completed StarCoder mixtures. At each
repeat anchor \(w\), compute
\[
\mathrm{SNR}^{\nabla}_b(w)
= \frac{r^2\lVert \nabla\hat\mu_b(w)\rVert_2^2}{\hat\sigma_b^2(w)}
\]
with \(r=0.05\) in StarCoder mixture units, bandwidth `0.18`, and
\(\hat\sigma_b^2(w)\) from the five repeat runs at that anchor. This is a local
detectability diagnostic: expected squared first-order movement under a small
mixture step divided by one-run repeat variance.

Command:

`uv run --with kaleido==0.2.1 python -m experiments.domain_phase_mix.exploratory.paper_plots.starcoder_two_phase_local_snr_landscape`

Artifacts:

- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_local_snr_landscape.html`
- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_local_snr_landscape.png`
- `experiments/domain_phase_mix/exploratory/paper_plots/img/starcoder_two_phase_local_snr_anchor_summary.csv`

Plot semantics:

- Left panel: selected metric mean surface from the historical 143 runs, with
  repeat anchors colored/sized by `log10_snr_power`.
- Right panel: anchor-only local SNR plot with
  `z = log10(r^2 ||grad mu(w)||^2 / sigma^2(w))`; stems run from zero to the
  local SNR value, and point color shows `log10` repeat variance.
- The plot intentionally stays anchor-only for SNR because noise was measured
  at only 10 locations. Interpolating a full SNR surface would overstate
  certainty.

Validation:

- `uv run python -m py_compile experiments/domain_phase_mix/exploratory/paper_plots/starcoder_two_phase_local_snr_landscape.py tests/test_starcoder_heteroskedastic_landscape.py`
- `uv run pytest tests/test_starcoder_heteroskedastic_landscape.py -q` reported `6 passed`.
- Regeneration reported `12` plotted metrics, `50` final-step repeat rows, and
  `0` excluded partial rows.

Initial readout:

Highest maximum local gradient-SNR among the plotted metrics was on broad
Uncheatable/Paloma text metrics, especially
`eval/uncheatable_eval/wikipedia_english/bpb`,
`eval/uncheatable_eval/bbc_news/bpb`, and
`eval/uncheatable_eval/arxiv_computer_science/bpb`. Code-specific metrics
(`dolma_100_programing_languages`, GitHub Python/CPP) still have high median
local SNR, but lower maximum local gradient magnitude than the broad metrics in
this two-dimensional StarCoder/Nemotron family.
