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

### 2026-05-26 - Live status refresh
- Training:
  Live Iris check of
  `/calvinxu/dm-grug-moe-v4-path-train-r1-20260520-1803` showed `14/15`
  training children succeeded. The only remaining training datapoint is
  `t=0.75,d1536`, which is still running. No failed-like child jobs are visible
  under the active training parent.
- Eval:
  The t=0.25,d1280 eval retry
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval10-t025-d1280-full-suite-20260525-2028`
  is complete: parent succeeded, with `21` succeeded task rows and `20` earlier
  killed duplicates superseded by replacement children.
- Fieldbook:
  Refreshed the active training parent timestamp and archived the stale eval10
  live-warning validation. No new submission is needed until the final d1536
  training checkpoint lands.

### 2026-05-26 - Eval11 d1536 full-suite submission
- Training status:
  Live Iris check showed
  `/calvinxu/dm-grug-moe-v4-path-train-r1-20260520-1803` completed
  successfully with `15/15` child trainings succeeded.
- Dry-run:
  `uv run python experiments/grug/moe/launch_v4_path_logprob_evals.py --dry-run --max-concurrent 64`

  Result: `15` successful training checkpoints, `22` tasks, `66` launch cells,
  `264` skipped cells, `264` valid existing cells, and `0` invalid existing
  cells. This is the missing d1536 path coverage.
- Safety:
  The proposed Iris parent command passed
  `uv run python -m experiments.domain_phase_mix.east5_launch_safety --json`
  with parent region `us-east5`, parent zone `us-east5-a`, and no warnings.
- Claude Code review:
  Reviewed with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, resumed Marin
  session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`; preflight showed
  `plambdafour@proton.me`, `stripe_subscription`, and no inherited
  `ANTHROPIC_API_KEY`. Verdict: no blockers; submit as written.
- Submitted:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval11-d1536-full-suite-20260526-1225`

  Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-grug-moe-v4-path-logprob-eval11-d1536-full-suite-20260526-1225 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 16GB --disk 20GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --max-concurrent 64`
- Fieldbook:
  Marked the path training parent as `succeeded`, added eval11 as a running
  job, resolved the previous d1536/eval10 next actions, and added an eval11
  monitoring next action.

### 2026-05-26 - Eval11b d1536 parent resource retry
- Status:
  The eval11 parent remained `JOB_STATE_PENDING` with zero children dispatched.
  The pending reason was parent-side CPU/memory capacity on `us-east5-a`
  highmem workers, not an eval launcher failure.
- Action:
  Stopped eval11 before any child dispatch, then submitted a smaller parent:
  `/calvinxu/dm-grug-moe-v4-path-logprob-eval11b-d1536-full-suite-lite-20260526-1231`.
- Claude Code review:
  Ran a focused review in the Marin CC session after the capacity stall. Verdict:
  no blockers; stop eval11 and submit eval11b with `0.5` CPU, `8GB` memory, and
  `15GB` disk. Safety gate again passed for `us-east5/us-east5-a`.
- Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-grug-moe-v4-path-logprob-eval11b-d1536-full-suite-lite-20260526-1231 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 0.5 --memory 8GB --disk 15GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --max-concurrent 64`
- Current state:
  Post-submit check shows eval11b is also pending with zero children dispatched
  due `us-east5-a` highmem CPU/memory capacity. This is a scheduler capacity
  wait; no failed eval cells exist yet.
- Fieldbook:
  Marked eval11 as `killed`, archived the stale eval11 submission validation,
  resolved the eval11 monitoring note, added eval11b as a queued retry, and
  opened an eval11b monitoring next action.

### 2026-05-26 - Endpoint eval12 coverage backfill
- Coverage diagnosis:
  After refreshing the dashboard, the path/intermediate rows were complete, but
  endpoint coverage still had two classes of gaps. `grug_moe_mix_v4_d1536`
  had alias-only invalid `logprob_gsm8k_5shot` and
  `logprob_humaneval_10shot` outputs, and `mmlu_sl_0shot` /
  `mmlu_sl_5shot` were missing for proportional d1536 plus all v4 endpoint
  scales. The `mmlu_sl_verb_*` rows were already present.
- Code changes:
  Extended `experiments/grug/moe/launch_v4_path_logprob_evals.py` so explicit
  base endpoint checkpoint roots use the current checkpoint layout while path
  roots keep the legacy flat layout. Also fixed explicit-root existing-result
  discovery so valid endpoint cells are skipped.
- Dry-run:
  The endpoint dry-run over six endpoint checkpoint roots and four task aliases
  produced `24` candidate cells, `14` launch cells, `10` skipped cells, `2`
  invalid-existing retries, `10` valid existing cells, and checkpoint layout
  `current`.
- Safety and review:
  `east5_launch_safety` passed with parent region `us-east5`, parent zone
  `us-east5-a`, and all GCS roots under `gs://marin-us-east5`. Claude Code was
  invoked with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, resumed Marin
  session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`; preflight showed
  `plambdafour@proton.me`, `stripe_subscription`, and no inherited
  `ANTHROPIC_API_KEY`. Verdict: no blockers.
- Submitted:
  `/calvinxu/dm-grug-moe-v4-endpoint-logprob-eval12-20260526-2312`

  Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-grug-moe-v4-endpoint-logprob-eval12-20260526-2312 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 0.5 --memory 8GB --disk 20GB --extra marin:tpu --extra marin:eval -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --max-concurrent 16 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_d1536-9.00e+19-54df08 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d512-2.19e+17-6aa7c8 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d768-1.70e+18-556077 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1024-9.00e+18-7938f2 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1280-2.83e+19-fd54fd --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1536-9.00e+19-5ae83d --only-task-alias mmlu_sl_0shot --only-task-alias mmlu_sl_5shot --only-task-alias logprob_gsm8k_5shot --only-task-alias logprob_humaneval_10shot --retry-attempt endpoint_retry1`
- Current state:
  Post-submit Iris summary shows the parent accepted and currently pending with
  `0/1` completed parent tasks.
- Fieldbook:
  Started a session for `exp_01ks41d0zkx21kb4w8kyqfz2mb`, added the eval12
  job as running, recorded the dry-run manifest/summary artifacts, added pass
  validation for dry-run plus CC/safety review, and opened a next-action note to
  monitor eval12 and refresh the dashboard/analysis once it completes.

### 2026-05-26 - Endpoint eval13 capacity recovery
- Status:
  Eval12 remained pending with zero children dispatched. Iris reported parent
  capacity pressure on the `us-east5-a` highmem CPU slice:
  `cpu_vm_e2_highmem_2_ondemand-us-east5-a=at_max_slices`.
- Claude Code review:
  Reviewed a capacity-only recovery with `env -u ANTHROPIC_API_KEY`, Opus 4.7
  max effort, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`.
  Verdict: no blockers; stop eval12 before it dispatches, then resubmit the
  same 14 logical cells with a standard-tier parent. CC suggested using `15GB`
  disk instead of `10GB` for uv install headroom.
- Action:
  Stopped `/calvinxu/dm-grug-moe-v4-endpoint-logprob-eval12-20260526-2312`
  before any child dispatch. Submitted
  `/calvinxu/dm-grug-moe-v4-endpoint-logprob-eval13-20260526-2335` with
  `1` CPU, `2GB` memory, `15GB` disk, no parent `--extra` constraints, and the
  same launcher arguments.
- Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-grug-moe-v4-endpoint-logprob-eval13-20260526-2335 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 2GB --disk 15GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --max-concurrent 16 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_d1536-9.00e+19-54df08 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d512-2.19e+17-6aa7c8 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d768-1.70e+18-556077 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1024-9.00e+18-7938f2 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1280-2.83e+19-fd54fd --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1536-9.00e+19-5ae83d --only-task-alias mmlu_sl_0shot --only-task-alias mmlu_sl_5shot --only-task-alias logprob_gsm8k_5shot --only-task-alias logprob_humaneval_10shot --retry-attempt endpoint_retry1`
- Current state:
  Eval13 parent started immediately and dispatched the 14 cells. Initial child
  poll showed two transient `add_port.cc:83` SIGSEGV failures
  (`grug_moe_mix_d1536` / `mmlu_sl_5shot` and `grug_moe_mix_v4_d1024` /
  `mmlu_sl_5shot`); wait for the parent to settle before any focused retry.
- Fieldbook:
  Marked eval12 as `killed`, added eval13 as a running retry, and recorded a
  warning validation with the initial child failure signal.

### 2026-05-26 - Endpoint eval14 layout repair
- Eval13 result:
  Eval13 dispatched all 14 cells but failed them. Most children failed with
  `FileNotFoundError` for missing `params/blocks/*/mlp/expert_mlp/*` arrays.
  This proved the explicit endpoint roots are still legacy-flat MoE checkpoints,
  not current nested-`expert_mlp` checkpoints.
- Code fix:
  Kept `BASE_RUN_ID_RE` endpoint parsing in
  `experiments/grug/moe/launch_v4_path_logprob_evals.py`, but removed the
  incorrect switch to `CURRENT_CHECKPOINT_LAYOUT`. Endpoint and path roots now
  use `legacy_moe_flat` for this launcher.
- Validation:
  `uv run python -m py_compile experiments/grug/moe/launch_v4_path_logprob_evals.py`
  passed, and
  `uv run pytest tests/test_grug_v4_path_logprob_launcher.py tests/test_grug_logprob_eval.py -q`
  passed with `15` tests. The repaired dry-run still has `14` launch cells,
  `10` skipped cells, `2` invalid direct-logprob retries, and now reports
  checkpoint layout `legacy_moe_flat`.
- Claude Code review:
  Reviewed with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, resumed Marin
  session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`. Verdict: no blockers; use
  `endpoint_retry2` for the two invalid direct-logprob cells and submit eval14.
- Submitted:
  `/calvinxu/dm-grug-moe-v4-endpoint-logprob-eval14-20260526-2345`.

  Command:
  `uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-grug-moe-v4-endpoint-logprob-eval14-20260526-2345 --region us-east5 --zone us-east5-a --enable-extra-resources --cpu 1 --memory 2GB --disk 15GB -e WANDB_API_KEY "$WANDB_API_KEY" -- python experiments/grug/moe/launch_v4_path_logprob_evals.py --max-concurrent 16 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_d1536-9.00e+19-54df08 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d512-2.19e+17-6aa7c8 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d768-1.70e+18-556077 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1024-9.00e+18-7938f2 --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1280-2.83e+19-fd54fd --checkpoint-root gs://marin-us-east5/grug/grug_moe_mix_v4_d1536-9.00e+19-5ae83d --only-task-alias mmlu_sl_0shot --only-task-alias mmlu_sl_5shot --only-task-alias logprob_gsm8k_5shot --only-task-alias logprob_humaneval_10shot --retry-attempt endpoint_retry2`
- Fieldbook:
  Marked eval13 as failed with the layout root cause, added eval14 as the active
  retry, and recorded the CC-reviewed eval14 submission validation.
- Initial health:
  First child-status poll showed all 14 eval14 children in running/building/pending
  states with zero failures, and filtered recent logs did not repeat the
  eval13 `expert_mlp` checkpoint-layout error.

### 2026-05-27 - Endpoint eval15 completion and path analysis refresh
- Eval14 result:
  Eval14 fixed the checkpoint-layout issue and completed `10/14` endpoint cells,
  but failed the remaining cells on Hugging Face Hub rate limiting. The eval14
  submission did not pass `HF_TOKEN`, so the children were likely using the
  unauthenticated hub quota despite the local shell having a token available.
- Retry:
  Submitted `/calvinxu/dm-grug-moe-v4-endpoint-logprob-eval15-20260527-0010`
  after CC no-blockers review. The retry passed both `WANDB_API_KEY` and
  `HF_TOKEN`, reduced `--max-concurrent` to `2`, used `endpoint_retry3`, and
  kept all checkpoint roots in `gs://marin-us-east5`.
- Result:
  Eval15 succeeded. All `4/4` child cells completed. A post-success dry-run with
  a fresh retry tag reports `launch_cells=0`, `skipped_cells=24`,
  `invalid_existing_cells=0`, and `valid_existing_cells=24`, confirming that the
  endpoint backfill is complete.
- Dashboard:
  Regenerated the Grug-MoE dashboard cache and plots. The path dashboard now has
  complete endpoint cells for `mmlu_sl_0shot`, `mmlu_sl_5shot`,
  `logprob_gsm8k_5shot`, and `logprob_humaneval_10shot`, including the d1536
  t=0/t=1 endpoint rows.
- Path-response analysis:
  Regenerated
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/`.
  `coverage_summary.csv` now reports `20/20` complete task paths for each hidden
  dimension `512`, `768`, `1024`, `1280`, and `1536`, with zero missing rows.
  Strict-common analysis uses all five hidden dimensions and `20` headline
  tasks. Current classification counts are `endpoint_improves=4`,
  `interior_peak=6`, `mixed_or_flat=6`, and `worsens_with_t=4`.
- Fieldbook:
  Recorded the eval15 success validation, archived superseded endpoint-retry
  warnings/failures as validations, refreshed drifted local dry-run artifacts,
  recorded the refreshed dashboard/path-analysis artifacts, resolved the active
  endpoint-monitoring notes, and wrote a fresh experiment checkpoint. Fieldbook
  status now has no blocking failed jobs, no failed validations, no validation
  warnings, and current freshness for this experiment.

### 2026-05-27 - Standardized path effect-size analysis
- Motivation:
  Native path deltas are oriented but not comparable across tasks because BPB,
  accuracy, and normalized choice-probability metrics use different units.
- Implementation:
  Extended
  `experiments/domain_phase_mix/exploratory/two_phase_many/analyze_grug_moe_path_response.py`
  to compute per-task empirical metric scales from the Grug-MoE dashboard/path
  oriented values, attach standardized deltas, and export standardized CSVs and
  HTML plots.
- Outputs:
  New artifacts under
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grug_moe_path_response_analysis_20260525/`:
  `task_metric_scales.csv`, `standardized_path_task_deltas.csv`,
  `task_t_standardized_mean_deltas.csv`,
  `task_endpoint_standardized_delta_summary.csv`,
  `task_t_standardized_delta_heatmap.html`,
  `task_endpoint_standardized_delta_ranking.html`, and
  `task_t_standardized_mean_delta_facets.html`.
- Result:
  At `t=1`, standardized endpoint deltas have `11` positive tasks and `9`
  negative tasks. Mean positive endpoint z-delta is `0.409`; mean absolute
  negative endpoint z-delta is `0.293`; max gain is about `+0.846`
  (`medmcqa_5shot`) and max deterioration is about `-1.161`
  (`boolq_sl_verb_10shot`). The endpoint therefore looks positive on average
  in empirical standardized units, but not Pareto-safe: the largest single
  standardized regression is larger than the largest single standardized gain.
- Caveat:
  This is an empirical native-unit scale normalization, not a repeated-seed
  noise standardization. It should be interpreted as an effect-size diagnostic,
  not as a statistical-significance estimate.
- Validation:
  `uv run pytest tests/test_grug_moe_mix_dashboard.py tests/test_grug_moe_path_response_analysis.py -q`
  passed with `8` tests, and
  `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/analyze_grug_moe_path_response.py`
  passed.
- Fieldbook:
  Recorded the standardized-analysis artifacts, added a pass validation, wrote a
  checkpoint, and refreshed drifted artifact metadata. Fieldbook status for the
  Grug-MoE v4 path experiment now has no blocking failed jobs, no failed
  validations, no validation warnings, and current freshness.
