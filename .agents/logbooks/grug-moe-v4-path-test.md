# Grug-MoE v4 Path Test: Research Logbook

## Scope
- Goal: evaluate the one-dimensional path from the proportional Grug-MoE mixture to the v4 mixture.
- Path: `w(t) = (1 - t) * w_proportional + t * w_v4`, with `t in {0.25, 0.50, 0.75}`.
- Existing endpoints: `t=0` is the completed `grug_moe_mix` proportional track; `t=1` is the completed `grug_moe_mix_v4` track.
- Scales: d512/2.19e17, d768/1.70e18, d1024/9.00e18, d1280/2.83e19, d1536/9.00e19.
- Primary readout: Grug validation/perplexity during training, followed by the Grug logprob eval suite used by `grug_moe_mix_dashboard_20260517`.

## Baseline
- Existing dashboard: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/dashboard.html`.
- Endpoint data source: `gs://marin-us-east5/grug/{grug_moe_mix,grug_moe_mix_v4}_d*-*`.
- Existing eval source: `gs://marin-us-east5/evaluation/grug_logprob`.
- Known t=1 coverage issue: `grug_moe_mix_v4_d1536-9.00e+19` has executor-successful but metric-empty `logprob_gsm8k_5shot` and `logprob_humaneval_10shot` results. These need a targeted eval retry before treating t=1 as full-metric complete.

## Experiment Log

### 2026-05-20 17:35 - Training Dry Run
- Hypothesis: intermediate path points will reveal whether v4’s gains/tradeoffs are monotone and whether a trust-region interpolation beats both endpoints on the aggregate.
- Command:
  `uv run python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5 --dry_run True`
- Config:
  15 training specs, `max_concurrent=8`, `v5p-8`, `us-east5-a`, `TARGET_BUDGET=6325183647689`, `TARGET_STEPS=2**14`.
- Result:
  Dry run inspected 15 Grug path training steps and emitted expected output roots:
  `gs://marin-us-east5/grug/grug_moe_mix_v4_path_{t025,t050,t075}_d*-*`.
- Validation:
  Local artifacts under `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_v4_path_test_20260520/` have 15 manifest rows and 234 weight rows. Each candidate phase sums to 1.0.
- Interpretation:
  The launcher is ready for live parent submission. The data config uses the current canonical top-level domain runtime-cache builder with phase weights recovered from the existing GCS endpoints.
- Next action:
  Submit the 15-run training parent, then run the same Grug logprob eval suite on the 15 new checkpoints plus the two malformed t=1 v4 d1536 eval cells.

### 2026-05-20 17:36 - Training Parent Submitted
- Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-grug-moe-v4-path-train-20260520-1736 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 8GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5`
- Job:
  `/calvinxu/dm-grug-moe-v4-path-train-20260520-1736`
- Expected children:
  15 Grug training steps, with executor max concurrency 8. Each child dispatches a `v5p-8` training worker in `us-east5-a`.
- Initial status:
  Parent accepted by Iris; first summary showed the parent task in `building`.
- Next action:
  Recheck parent logs/summary until the 15 child training jobs have either completed or surfaced actionable failures. After all checkpoints land, launch the Grug logprob suite for the path rows and the two malformed v4 d1536 endpoint eval cells.

### 2026-05-20 17:40 - Parent Relaunch After GCS Client Fix
- Failure:
  `/calvinxu/dm-grug-moe-v4-path-train-20260520-1736` failed during parent startup before dispatching children because the Iris CPU parent image does not provide `gcloud`.
- Fix:
  Replaced launcher GCS reads with `fsspec.open` and hardcoded the two endpoint executor roots needed for proportional/v4 d512 endpoint weights:
  `gs://marin-us-east5/grug/grug_moe_mix_d512-2.19e+17-e6a48f` and
  `gs://marin-us-east5/grug/grug_moe_mix_v4_d512-2.19e+17-6aa7c8`.
- Validation:
  Re-ran dry-run successfully with:
  `uv run python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5 --dry_run True`.
- Relaunch command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-grug-moe-v4-path-train-20260520-1740 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 8GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5`
- Job:
  `/calvinxu/dm-grug-moe-v4-path-train-20260520-1740`
- Initial status:
  Parent accepted by Iris; first summary showed the parent `pending`.

### 2026-05-20 17:54 - Claude Code Launcher Review
- Command:
  `env -u ANTHROPIC_API_KEY claude --model claude-opus-4-7 --effort max --permission-mode dontAsk ...`
- Auth preflight:
  Local Claude OAuth account was `plambdafour@proton.me`, billing type `stripe_subscription`, and `ANTHROPIC_API_KEY` was removed from the Claude environment.
- Verdict:
  No hard blockers. Claude flagged one main comparability caveat: this path launcher uses the current canonical `TokenizedMixtureGroup` / `lm_varying_mixture_data_config` wiring, while the original v4/proportional endpoint branch used a `ConcatDatasetComponent`-style construction. These should be statistically close for size-proportional internal partition sampling, but they are not bit-identical data loaders.
- Other review notes:
  The path launcher also uses `shuffle=True`, whereas the endpoint path likely used the default `BlockShuffleConfig`, and it does not explicitly set `auto_build_caches=False`. Both are non-blocking for final metrics but should be documented when interpreting endpoint-vs-path loss curves.
- Decision:
  Do not stop the already-dispatched training parent. Treat this as a documented confound. If the path results look inconsistent, run a small t=0 calibration through the same path launcher wiring before drawing strong endpoint-relative conclusions.
- Eval follow-up requirements:
  The full-metrics eval launcher should set Grug eval `capacity_factor=8.0` explicitly and must not treat the two malformed v4 d1536 GSM8K/HumanEval `results.json` files as complete. Force a new eval hash or use a fresh output prefix for those two endpoint retry cells.

### 2026-05-20 18:03 - Training Relaunch After Missing `asyncio` Fix
- Failure:
  The `17:40` parent dispatched children, but several failed at dataset construction with:
  `NameError: name 'asyncio' is not defined` from `lib/levanter/src/levanter/data/text/datasets.py` in `LazyAsyncDataset.__init__`.
- Fix:
  Added the missing `import asyncio` to `lib/levanter/src/levanter/data/text/datasets.py`.
- Validation:
  `uv run python -m py_compile lib/levanter/src/levanter/data/text/datasets.py experiments/grug/moe/model.py experiments/grug/moe/launch_v4_path_test.py`

  `uv run python - <<'PY' ... LazyAsyncDataset(lambda: None, finite_length=1) ... PY`

  `uv run python experiments/grug/moe/launch_v4_path_test.py --prefix gs://marin-us-east5 --dry_run True`

  The retry dry-run emits 15 fresh `grug_moe_mix_v4_path_r1_*` steps. Local artifacts still have 15 manifest rows and 234 candidate-weight rows; every candidate phase sums to 1.0.
- Recovery:
  Stopped `/calvinxu/dm-grug-moe-v4-path-train-20260520-1740` to avoid wasting TPU time on the known-bad image. Submitted a fresh parent:
  `/calvinxu/dm-grug-moe-v4-path-train-r1-20260520-1803`.
- Current status:
  Initial summary shows the retry parent running/assigned with no failures yet. First status sweep shows the first 8 retry executor steps as `RUNNING` and the remaining 7 not yet dispatched, matching `max_concurrent=8`.

### 2026-05-22 02:57 - Dashboard Path Section Added
- Parent checked:
  `/calvinxu/dm-grug-moe-v4-path-train-r1-20260520-1803`.
- Current training status:
  9 of 15 path runs are executor-successful: `t=0.25`, `t=0.50`, and `t=0.75` have completed at d512, d768, and d1024. The d1280 and d1536 rows for all three `t` values remain running. No child failures were observed in this sweep.
- Follow-up benchmark eval status:
  No `evaluation/grug_logprob/grug_moe_mix_v4_path_r1_*` result JSONs were found yet, so the intermediate path checkpoints do not yet have the MoE dashboard benchmark suite.
- Dashboard update:
  `experiments/domain_phase_mix/exploratory/two_phase_many/build_grug_moe_mix_dashboard.py` now adds a Path Test section. It plots per-task loss-like metrics against training FLOPs, with one track per mixture on the v4 path (`t=0` proportional, `t=1` v4, and intermediate `t` curves when available). The regenerated dashboard is:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/dashboard.html`.
- Interpretation:
  The current path plot is endpoint-only, not a completed path test. The next action is to launch or collect Grug logprob evals for the 9 successful path checkpoints, and later for the remaining 6 once training completes.

### 2026-05-22 03:20 - Missing Streaming Eval Follow-Up Identified
- Problem:
  The path-training parent does not automatically trigger the Grug MoE dashboard eval suite as each checkpoint finishes. This is why the path dashboard still shows only the `t=0` proportional and `t=1` v4 endpoint tracks: the path eval metrics CSV is empty, not merely partially plotted.
- Fix in progress:
  Restored a Grug-native logprob evaluator at `experiments/grug/moe/eval_logprob.py` and added a rerunnable missing-cell launcher at `experiments/grug/moe/launch_v4_path_logprob_evals.py`.
- Dry-run:
  `uv run python experiments/grug/moe/launch_v4_path_logprob_evals.py --dry-run`

  Result: 9 successful path checkpoints, 22 benchmark tasks, and 198 missing eval cells to launch. This is the expected `9 * 22` first wave for the currently completed d512/d768/d1024 path rows.
- Submission gate:
  Per operator preference, no live eval submission should happen until Claude Code reviews the launcher and dry-run manifest. Future live job submissions should use this as a standard gate when practical.

### 2026-05-22 05:35 - First Wave Path Eval Submission
- Launcher hardening:
  Before review/submission, patched `experiments/grug/moe/launch_v4_path_logprob_evals.py` to set an explicit `ExecutorMainConfig(prefix="gs://marin-us-east5", max_concurrent=64)`, record `max_concurrent` in the summary artifact, emit LF CSVs, and avoid treating arbitrary `gcloud storage cat` failures as missing files.
- Validation:
  `uv run python -m py_compile experiments/grug/moe/eval_logprob.py experiments/grug/moe/launch_v4_path_logprob_evals.py`

  `uv run python experiments/grug/moe/launch_v4_path_logprob_evals.py --dry-run`

  Dry-run result: 9 successful path checkpoints, 22 tasks, 198 launch cells, 0 skipped cells, `max_concurrent=64`.
- Claude Code review:
  Invoked with `env -u ANTHROPIC_API_KEY claude --model claude-opus-4-7 --effort max ...` after preflight confirmed local OAuth account `plambdafour@proton.me`, billing `stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. Verdict: no blockers. Non-blocking notes were tokenizer pad fallback sanity, hash-suffix length consistency, and multi-host listener cleanup if this ever stops being single-host.
- Extra sanity check:
  `load_tokenizer(marin_tokenizer)` returns `HfMarinTokenizer`, `pad_token_id=None`, `eos_token_id=128001`, and is a dataclass, so the existing pad fallback branch is reachable and structurally valid.
- Submitted job:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval-20260522-053429`
- Initial Iris status:
  Parent is `running`, with the sole parent task in `building` on a us-east5-a worker. This parent should dispatch the 198 missing eval cells for the currently successful path checkpoints. Rerun the same launcher later as d1280/d1536 path trainings finish.

### 2026-05-22 05:38 - Eval Parent Failed On `gcloud`; Fsspec Fix
- Failed parent:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval-20260522-053429`
- Failure:
  The parent failed before dispatching children with `FileNotFoundError: [Errno 2] No such file or directory: 'gcloud'`. This repeated the earlier training-parent lesson: Iris parent images cannot assume the `gcloud` CLI exists.
- Fix:
  Replaced `gcloud storage ls/cat` discovery in `experiments/grug/moe/launch_v4_path_logprob_evals.py` with `fsspec.open` and `marin.utils.fsspec_glob`, so parent-side GCS discovery uses repo/runtime dependencies instead of a shell binary.
- Validation:
  `uv run python -m py_compile experiments/grug/moe/eval_logprob.py experiments/grug/moe/launch_v4_path_logprob_evals.py`

  `uv run python experiments/grug/moe/launch_v4_path_logprob_evals.py --dry-run`

  Dry-run still returns 9 successful checkpoints, 22 tasks, 198 launch cells, 0 skipped cells, and `max_concurrent=64`.
- Focused Claude Code review:
  Re-reviewed only the fsspec replacement with `env -u ANTHROPIC_API_KEY claude --model claude-opus-4-7 --effort max ...`. Verdict: no blockers.
- Relaunch:
  Submitted `/calvinxu/dm-grug-moe-v4-path-logprob-eval2-20260522-053916`.
- Relaunch status:
  The parent is running and has dispatched the first concurrency window: 64 child eval jobs are pending for `v5p-8` capacity in `us-east5-a`. Pending reason is currently TPU capacity/quota-pool availability, not a launcher error. The parent remains alive and should continue dispatching up to `max_concurrent=64` as children finish.

### 2026-05-22 16:25 - Eval4 Retry After SL-Verb Task Fix
- Problem:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval3-20260522-153344` produced useful
  successful cells, but it also failed every non-MMLU SL-Verb alias because
  `lm-eval` does not register `boolq_sl_verb`, `commonsense_qa_sl_verb`, or
  `medmcqa_sl_verb` task names. The same parent was also running on a
  preemptible TPU worker and had already been preempted once.
- Fix:
  Patched `experiments/grug/moe/eval_logprob.py` so:
  - `boolq_sl_verb_10shot` is a dynamic alias over base `boolq`.
  - `csqa_sl_verb_5shot` is a dynamic alias over base `commonsense_qa`, with
    rendered choices `A. <text>` through `E. <text>` and integer target derived
    from `answerKey`.
  - `medmcqa_sl_verb_5shot` is a dynamic alias over base `medmcqa`, with
    rendered choices `A. <opa>` through `D. <opd>`.
  - `_lm_eval_spec` now emits dynamic dict specs when `task_alias` is set, even
    when no other task kwargs are present, so aliases survive into lm-eval.
- Validation:
  `uv run python -m py_compile experiments/grug/moe/eval_logprob.py experiments/grug/moe/launch_v4_path_logprob_evals.py tests/test_grug_logprob_eval.py`

  `uv run ruff check experiments/grug/moe/eval_logprob.py tests/test_grug_logprob_eval.py`

  `uv run pytest tests/test_grug_logprob_eval.py -q`

  The test suite now has a rendered-task regression test that loads the exact
  dynamic specs through `lm_eval.tasks.get_task_dict` and verifies the rendered
  choices and integer targets for BoolQ, CommonsenseQA, and MedMCQA.
- Claude Code review:
  Invoked with `env -u ANTHROPIC_API_KEY`, Opus 4.7, max effort, after auth
  preflight confirmed `plambdafour@proton.me`, `stripe_subscription`, and no
  inherited `ANTHROPIC_API_KEY`. The first review asked for explicit
  task-kwargs rendering coverage. After adding that test, the focused follow-up
  verdict was: no blockers, clear to proceed.
- Live action:
  Stopped eval3 after it reached `119` succeeded cells. Corrected dry-run after
  stopping found `11` successful training checkpoints, `22` tasks, `121` skipped
  cells, and `121` launch cells.
- Active retry:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval4-20260522-1625`

  Submitted with a nonpreemptible CPU parent in `us-east5`; child eval tasks
  remain `v5p-8` in `us-east5-a` through the evaluator resource config. Initial
  status was pending on east5 on-demand CPU capacity, not a launcher failure.

### 2026-05-22 16:33 - Eval5 Parent Capacity Adjustment
- Issue:
  Eval4 remained pending before dispatch because east5 on-demand highmem CPU
  parent capacity was unavailable (`us-east5-a=at_max_slices`,
  `us-east5-b=backoff`).
- Claude Code review:
  Reviewed the capacity adjustment with `env -u ANTHROPIC_API_KEY`, Opus 4.7,
  max effort. Verdict: no blockers. The review called out that the parent only
  performs small GCS metadata/result discovery and writes manifests; checkpoint
  tensor reads remain in the child eval tasks pinned to `us-east5-a`.
- Action:
  Stopped eval4 before dispatch and submitted
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval5-20260522-1633` with
  `cpu=1`, `memory=2GB`, default `disk=5GB`, `--no-preemptible`, and no parent
  region/zone pin.
- Current status:
  Eval5 parent is running and has dispatched the first concurrency window:
  `9` eval children running and `55` pending, with output paths under
  `gs://marin-us-east5/evaluation/grug_logprob/...`. The first launched cells
  include the corrected `boolq_sl_verb_10shot`, `csqa_sl_verb_5shot`, and
  `medmcqa_sl_verb_5shot` aliases.

### 2026-05-23 16:18 - Eval6 Single-Cell Retry
- Failure:
  Eval5 produced `119` successful eval cells and one failed cell:
  `grug_moe_mix_v4_path_r1_t050_d512-2.19e+17` on
  `medmcqa_sl_verb_5shot`. The child failed with the known infra-like
  `SIGSEGV` / `add_port.cc` port-listener error, not with a task-rendering or
  checkpoint-layout error.
- Dry-run:
  `uv run python experiments/grug/moe/launch_v4_path_logprob_evals.py --dry-run --only-run-id grug_moe_mix_v4_path_r1_t050_d512-2.19e+17 --only-task-alias medmcqa_sl_verb_5shot --assume-missing --max-concurrent 1`

  Result: exactly one launch cell, checkpoint layout `legacy_moe_flat`, no skipped cells, and east5 checkpoint locality.
- Claude Code review:
  Reviewed with `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`; preflight showed `plambdafour@proton.me`, `stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. Verdict: no blockers.
- Submitted:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval6-medmcqa-retry-20260523-161834`

  Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-grug-moe-v4-path-logprob-eval6-medmcqa-retry-20260523-161834 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --only-run-id grug_moe_mix_v4_path_r1_t050_d512-2.19e+17 --only-task-alias medmcqa_sl_verb_5shot --assume-missing --max-concurrent 1`
- Immediate status:
  Parent was `JOB_STATE_RUNNING`; the single child was pending on east5 TPU capacity.
- Fieldbook:
  Added retry job `job_01ksbj7b03zdzy5mvnjfnbch49`, linked as a retry of eval5, and added a debug note on `exp_01ks41d0zkx21kb4w8kyqfz2mb`.

### 2026-05-23 16:38 - Eval7 Nonpreemptible Parent Retry
- Issue:
  Eval6 parent accepted the one-cell graph, but the first child was killed with
  `Parent task preempted`; the parent then requeued a replacement child. Because no useful child work
  remained in progress, continuing on the preemptible CPU parent risked repeating the same failure mode.
- Claude Code review:
  Reviewed the stop/relaunch decision with `env -u ANTHROPIC_API_KEY`, Opus 4.7, max effort, session
  `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: stop eval6 and relaunch the same one-cell retry
  with `--no-preemptible`; no blockers.
- Action:
  Stopped `/calvinxu/dm-grug-moe-v4-path-logprob-eval6-medmcqa-retry-20260523-161834`.
  Submitted `/calvinxu/dm-grug-moe-v4-path-logprob-eval7-medmcqa-retry-20260523-1638`.

  Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-grug-moe-v4-path-logprob-eval7-medmcqa-retry-20260523-1638 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --only-run-id grug_moe_mix_v4_path_r1_t050_d512-2.19e+17 --only-task-alias medmcqa_sl_verb_5shot --assume-missing --max-concurrent 1`
- Immediate status:
  Eval7 parent was accepted and pending on nonpreemptible east5 CPU capacity.
- Fieldbook:
  Marked eval6 `killed`, added eval7 job `job_01ksbkbjachzvps7j42wp2nsf1`, and added a debug note on
  `exp_01ks41d0zkx21kb4w8kyqfz2mb`.

### 2026-05-23 16:45 - Eval8 Parent Capacity Recovery
- Issue:
  Eval7 was correct semantically but its nonpreemptible parent was pinned to
  `us-east5-a` and remained pending on parent CPU capacity. No child eval work
  had started.
- Claude Code review:
  Reviewed the capacity recovery with `env -u ANTHROPIC_API_KEY`, Opus 4.7,
  max effort, session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: stop
  eval7 and relaunch with a nonpreemptible but placement-unconstrained parent;
  keep child checkpoint/eval locality inside the launcher.
- Action:
  Stopped `/calvinxu/dm-grug-moe-v4-path-logprob-eval7-medmcqa-retry-20260523-1638`.
  Submitted `/calvinxu/dm-grug-moe-v4-path-logprob-eval8-medmcqa-retry-20260523-1645`
  with `--no-preemptible --cpu 1 --memory 2GB --disk 10GB`, no parent
  `--region/--zone`, and the same one-cell evaluator command.
- Current status:
  Eval8 parent is running and has one child running:
  `grug_moe_mix_v4_path_r1_t050_d512-2.19e+17` /
  `medmcqa_sl_verb_5shot`.
- Fieldbook:
  Marked eval7 `killed`, added eval8 job
  `job_01ksbkr58p2rg9qgbph31scjs5`, and added a debug note on
  `exp_01ks41d0zkx21kb4w8kyqfz2mb`.

### 2026-05-24 - cross-region audit

Rohith's egress report included smaller Grug eval activity. CC review found the
active path-training parent
`/calvinxu/dm-grug-moe-v4-path-train-r1-20260520-1803` is east5-local:
the submitted command used `--zone us-east5-a`, the launcher writes to
`gs://marin-us-east5/grug`, and children are running in `us-east5-a`. Leave the
active training parent running.

However, the earlier capacity-recovery pattern that relaunched eval parents
without parent `--region/--zone` is no longer acceptable for east5 workloads.
Future Grug eval/training submissions must use explicit parent
`--region us-east5 --zone us-east5-a`, east5 GCS paths, and CC review before
live submission.

### 2026-05-25 00:43 - Eval9 GSM8K/HumanEval Numeric Retry
- Issue:
  The path dashboard lacked intermediate GSM8K/HumanEval tracks because the
  path follow-up eval artifacts for `logprob_gsm8k_5shot` and
  `logprob_humaneval_10shot` existed but were semantically empty:
  representative `results.json` payloads contained only `{"alias": ...}` and
  no numeric `bpb,none` / `nll,none` metrics. The dashboard collector was
  correct to emit no rows; the launcher was wrong to treat any `results.json`
  as a valid completed cell.
- Fix:
  Patched `experiments/grug/moe/launch_v4_path_logprob_evals.py` so existing
  results are classified as `valid_existing_result` only when they contain
  usable numeric metrics. For `logprob_gsm8k_5shot` and
  `logprob_humaneval_10shot`, valid means both `bpb` and `nll` are present.
  Invalid existing results relaunch under a nested `numeric_retry1` subpath to
  avoid executor cache collisions with the alias-only artifacts. Also patched
  `experiments/grug/moe/eval_logprob.py` to support this optional retry
  output-attempt suffix.
- Tests:
  `uv run pytest tests/test_grug_v4_path_logprob_launcher.py tests/test_grug_logprob_eval.py -q`
  passed with `13` tests. `uv run python -m py_compile` on the touched files
  passed, and `uv run ruff check` on the touched files passed.
- Dry-run:
  `uv run python experiments/grug/moe/launch_v4_path_logprob_evals.py --dry-run --only-task-alias logprob_gsm8k_5shot --only-task-alias logprob_humaneval_10shot --max-concurrent 64`

  Result: `12` successful training checkpoints, `24` launch cells, `0` skipped
  cells, `22` invalid existing alias-only cells, and `2` genuinely missing
  cells for `t=0.25`, `d1280`.
- Claude Code review:
  Reviewed with `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-7 --effort max --permission-mode dontAsk`; preflight showed `plambdafour@proton.me`, `stripe_subscription`, and no inherited `ANTHROPIC_API_KEY`. Verdict: no blockers.
- Launch-safety gate:
  The Iris parent command passed
  `uv run python -m experiments.domain_phase_mix.east5_launch_safety --command ... --json`
  with `ok=true`, parent region `us-east5`, and parent zone `us-east5-a`.
- Submitted:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval9-gsm-hum-numeric-retry-20260525-004254`

  Command:
  `uv run iris --cluster=marin job run --no-wait --job-name dm-grug-moe-v4-path-logprob-eval9-gsm-hum-numeric-retry-20260525-004254 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --only-task-alias logprob_gsm8k_5shot --only-task-alias logprob_humaneval_10shot --max-concurrent 64`
- Immediate status:
  Parent is `JOB_STATE_RUNNING`; child eval jobs have been dispatched and are
  pending on east5 TPU capacity.
- Fieldbook:
  Added and updated job `job_01ksf1d2zeths0xkm48ggx5z6d` on experiment
  `exp_01ks41d0zkx21kb4w8kyqfz2mb`, with status `running` and attributes for
  the `24` launch cells, `22` invalid existing cells, `2` missing cells, and
  retry attempt `numeric_retry1`.

### 2026-05-25 - Path dashboard refresh with numeric retry data
- Issue:
  The available `numeric_retry2` GSM8K/HumanEval results had landed in nested
  executor paths such as
  `.../logprob_gsm8k_5shot/numeric_retry2-*/results.json`, but the dashboard
  collector only used a broad recursive glob that missed those nested retry
  outputs in practice. The cached path plots therefore still omitted the
  intermediate `t=0.25/0.50/0.75` GSM8K and HumanEval tracks.
- Fix:
  Patched
  `experiments/domain_phase_mix/exploratory/two_phase_many/build_grug_moe_mix_dashboard.py`
  to explicitly list both direct task-child results and one-level nested retry
  results, then de-duplicate metric rows by logical task/scale/metric while
  preferring the highest `numeric_retryN` attempt.
- Validation:
  `uv run pytest tests/test_grug_moe_mix_dashboard.py -q` passed with `4`
  tests, and `uv run python -m py_compile` passed for the dashboard builder and
  test file.
- Refresh:
  Refreshed `grug_moe_v4_path_eval_metrics_long.csv`,
  `grug_moe_path_task_scaling.csv`,
  `grug_moe_path_task_deltas.csv`,
  `img/path_task_scaling_facets.html`, and
  `img/path_task_delta_facets.html` under
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_mix_dashboard_20260517/`.
- Coverage:
  Path eval metrics now contain `1335` rows across `22` tasks. GSM8K and
  HumanEval each have `24` path metric rows, corresponding to `4` completed
  non-`d1536` scales for each of `t=0.25`, `t=0.50`, and `t=0.75`, with both
  `bpb` and `nll`. The `d1536` intermediate points remain absent because those
  path training runs have not succeeded.

### 2026-05-25 - Path plot interpolation color ladder
- Change:
  Recolored the path-test interpolation tracks in
  `build_grug_moe_mix_dashboard.py` with a fixed red-to-green ladder:
  proportional is red, `t=0.25` orange, `t=0.50` yellow, `t=0.75` light green,
  and v4 green. This replaces the previous categorical palette, which made the
  path order visually arbitrary.
- Artifacts:
  Regenerated `img/path_task_scaling_facets.html` and
  `img/path_task_delta_facets.html` from cached path CSVs; no GCS refresh or job
  submission was needed.
- Validation:
  `uv run pytest tests/test_grug_moe_mix_dashboard.py -q`, `uv run ruff check`
  on the dashboard/test files, and `uv run python -m py_compile` all passed.

### 2026-05-25 - Path response correlation analysis
- Hypothesis:
  The path \(w(t)=(1-t)p+t w_{v4}\) should reveal whether the v4 endpoint is a
  broadly improving controllable direction, a coverage tradeoff against
  proportional, or a trust-region direction whose intermediate mixtures are
  preferable for some tasks.
- Command:
  `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/analyze_grug_moe_path_response.py`
- Inputs:
  `grug_moe_path_task_deltas.csv` from the cached dashboard output and the
  representability/path-test framing in
  `/Users/calvinxu/Downloads/Representability in Stratified Sampling.md`.
- Method:
  Used `delta_oriented` so positive always means better than proportional at the
  same scale. The headline cut uses strict-common completed path scales
  `d512,d768,d1024` and excludes incomplete non-verb MMLU-SL aliases. For each
  task, computed pooled Pearson/Spearman correlation between interpolation
  coordinate `t` and task delta, scale-fixed slope, per-scale slopes, endpoint
  delta, and best observed mean `t`.
- Result:
  Wrote outputs under
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/`.
  The strict-common classification over 20 tasks was:
  - endpoint improves: 4 tasks
    (`logprob_humaneval_10shot`, `logprob_gsm8k_5shot`, `boolq_10shot`,
    `truthfulqa_mc1_0shot`);
  - interior peak: 9 tasks;
  - worsens with `t`: 6 tasks
    (`hellaswag_0shot`, `hellaswag_5shot`, `wsc273_0shot`,
    `boolq_sl_verb_10shot`, `csqa_5shot`, `mmlu_sl_verb_5shot`);
  - mixed/flat: 1 task (`copa_0shot`).
- Interpretation:
  The strongest aligned improvements are GSM8K and HumanEval, as expected.
  HellaSwag and several broad/SL-verb tasks move against the v4 direction,
  consistent with a real coverage tradeoff rather than a uniformly Pareto
  improving endpoint. Several tasks peak at `t=0.25` or `t=0.75`, which supports
  evaluating trust-region interpolations rather than only proportional versus
  v4. This is still a single-seed/path-correlation analysis, so small native
  deltas should be interpreted alongside metric SNR/noise estimates.
- Fieldbook:
  Added report/table/plot artifacts and a research note to experiment
  `exp_01ks41d0zkx21kb4w8kyqfz2mb`.
