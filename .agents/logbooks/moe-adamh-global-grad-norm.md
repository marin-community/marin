# MoE AdamH Global Gradient Normalization: Research Logbook

## Scope

- Goal: test whether normalizing the full gradient tree to RMS 1 before
  optimizer moment updates improves Grug MoE training relative to the current
  AdamH recipe.
- Primary metrics: final `eval/paloma/macro_loss`, effective speedup versus
  the compute-optimal MoE baseline.
- Secondary metrics: `throughput/tokens_per_second`,
  `throughput/total_tokens`, training stability.
- Constraints: compare at the README compute-optimal d512, d768, d1024, and
  d1280 budgets; run gate 1 before gate 2.
- Issue: https://github.com/marin-community/marin/issues/5182

## Baseline

- Date: 2026-04-25
- Code refs:
  - `experiments/grug/moe/README.md`
  - `experiments/grug/moe/adamh.py`
  - `experiments/grug/moe/optimizer.py`
  - `experiments/grug/moe/launch.py`
- Baseline numbers: compute-optimal d512, d768, d1024, and d1280 table in
  `experiments/grug/moe/README.md`.
- Prior related result: #5180 tested per-module gradient normalization. It
  improved d768 but failed d512, so gate 1 failed overall.

## Experiment Log

### 2026-04-25 18:10 - Kickoff

- Hypothesis: one global RMS normalization preserves relative gradient scales
  across modules while controlling the absolute gradient scale entering AdamH,
  avoiding the d512 regression from per-module reweighting.
- Command: local implementation and unit tests.
- Config:
  - optimizer: `GrugMoeAdamHGlobalGradientNormConfig`
  - normalization: scale the full gradient tree to combined RMS 1 before
    AdamH/Adam moment updates.
  - gate 1 launch:
    `GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE=gate1 python -m experiments.grug.moe.launch_adamh_global_grad_norm`
  - gate 2 launch:
    `GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE=gate2 python -m experiments.grug.moe.launch_adamh_global_grad_norm`
- Result: pending implementation.
- Interpretation: gate 1 should run d512 and d768 first; gate 2 should run
  only if both small-scale effective speedups exceed 1.0.
- Next action: add focused optimizer tests, implement the global-normalized
  optimizer and launch wiring, then submit gate 1 to Iris.

### 2026-04-25 18:12 - MOE-AGGN-001 gate 1 submitted

- Hypothesis: global gradient RMS normalization can recover the d512 point
  while preserving the d768 improvement seen in #5180.
- Command:
  `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --memory=2G --disk=4G --cpu=1 --extra=cpu --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE gate1 -- python -m experiments.grug.moe.launch_adamh_global_grad_norm`
- Config:
  - commit: `6c6675b3e9e633ef5344b1ee83b4931db2cd18a8`
  - issue: https://github.com/marin-community/marin/issues/5182
  - PR: https://github.com/marin-community/marin/pull/5183
  - Iris parent job: `/kaiyue/iris-run-job-20260426-011219`
  - data browser:
    https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/launch_adamh_global_grad_norm-374368.json
  - d512 step:
    `grug/moe-adamh-global-grad-norm/d512-2p19e17_b769d61c`
  - d512 output:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d512-2p19e17-2406f6`
  - d512 W&B:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-global-grad-norm-d512-2p19e17
  - d768 step:
    `grug/moe-adamh-global-grad-norm/d768-1p70e18_e21b2843`
  - d768 output:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d768-1p70e18-591709`
  - d768 W&B:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-global-grad-norm-d768-1p70e18
- Result: submitted to Iris and dispatched both Fray TPU child jobs. Initial
  allocations hit the TPU device-busy/no-accelerator bad-node signature, then
  Iris retried automatically. At startup stabilization both child jobs were
  `JOB_STATE_RUNNING` with `failure_count=0`.
- Interpretation: gate 1 is now a live Iris run. Continue babysitting until
  both d512 and d768 reach terminal state, then compare against the README
  baselines using the `agent.md` effective-speedup formula.
- Next action: monitor to terminal state; launch gate 2 only if both gate-1
  effective speedups exceed 1.0.

### 2026-04-25 18:53 - MOE-AGGN-001 preemption recovery fix

- Hypothesis: the gate-1 parent launcher inherited its storage prefix from the
  launcher worker region, so a parent preemption could move the executor prefix
  from `gs://marin-us-central1` to `gs://marin-us-east5` and make child jobs
  miss existing checkpoints.
- Command:
  `.venv/bin/iris --config lib/iris/examples/marin.yaml job stop /kaiyue/iris-run-job-20260426-011219`
- Config:
  - observed saved checkpoint:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d512-2p19e17-2406f6/checkpoints/step-4000`
  - observed restart search path:
    `gs://marin-us-east5/grug/moe-adamh-global-grad-norm/d512-2p19e17-2406f6/checkpoints`
  - fix: default `MARIN_PREFIX` to `gs://marin-us-central1` before the
    launch file builds import-time data configs.
- Result: stopped the self-submitted gate-1 parent and both child jobs after
  the prefix mismatch appeared. Focused optimizer tests and launch selector
  smoke still passed after the prefix fix.
- Interpretation: gate 1 should be resubmitted in `us-central1` with
  `MARIN_PREFIX=gs://marin-us-central1`; d512 can restore from the existing
  central1 checkpoint, and future parent preemptions should not change the
  storage prefix.
- Next action: lint, commit, push, then resubmit gate 1 with a pinned launcher
  region and continue monitoring to terminal state.

### 2026-04-25 18:56 - MOE-AGGN-001 gate 1 resubmitted

- Hypothesis: pinning the launcher to `us-central1` and setting
  `MARIN_PREFIX=gs://marin-us-central1` will keep executor output paths stable
  across parent restarts and allow checkpoint restore.
- Command:
  `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --memory=2G --disk=4G --cpu=1 --extra=cpu --region us-central1 --no-preemptible -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX gs://marin-us-central1 -e GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE gate1 -- python -m experiments.grug.moe.launch_adamh_global_grad_norm`
- Config:
  - commit: `dabbc2b52`
  - Iris parent job: `/kaiyue/iris-run-job-20260426-015624`
  - data browser:
    https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/launch_adamh_global_grad_norm-7d0e46.json
  - d512 output:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d512-2p19e17-2406f6`
  - d768 output:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d768-1p70e18-591709`
- Result: d512 restored from
  `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d512-2p19e17-2406f6/checkpoints/step-3000`;
  d768 restored from
  `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d768-1p70e18-591709/checkpoints/step-1000`.
- Interpretation: the storage-prefix issue is fixed for the live run. The
  incomplete d512 step-4000 save was skipped automatically; step-3000 is the
  latest loadable checkpoint.
- Next action: continue monitoring gate 1 to terminal state.

### 2026-04-25 19:11 - MOE-AGGN-001 launch dry-run fix

- Hypothesis: the launch file can keep the manual `us-central1` default while
  still honoring CI's `--prefix` dry-run argument if it mirrors that CLI prefix
  into `MARIN_PREFIX` before importing the shared MoE launch data config.
- Command:
  `WANDB_MODE=disabled uv run python experiments/grug/moe/launch_adamh_global_grad_norm.py --dry_run True --executor_info_base_path "$tmp" --prefix "$tmp"`
- Result: dry-run output paths now stay under the temporary executor prefix,
  including raw dataset overrides. The runpy dry-run path used by
  `tests/test_dry_run.py` also passed, and
  `uv run --with pytest --with pytest-timeout pytest tests/test_grug_moe_adamh_global_norm.py -q`
  passed.
- Interpretation: this should fix the PR CI timeout and temporary-directory
  collision from commit `dabbc2b52` without changing the live Iris run, which
  already passes `MARIN_PREFIX=gs://marin-us-central1`.
- Next action: run pre-commit, commit and push the PR fix, then continue
  monitoring gate 1.

### 2026-04-25 19:36 - MOE-AGGN-001 d512 gate-1 result

- Result: d512 finished successfully on Iris.
- Metrics:
  - W&B run:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-global-grad-norm-d512-2p19e17
  - global_step: `6386`
  - eval/paloma/macro_loss: `3.807736158370972`
  - throughput/tokens_per_second: `406222.9291489406`
  - effective speedup vs README baseline: `1.0143777998139154`
- Interpretation: the d512 gate-1 point passes the `>1.0` threshold. Gate 1
  is still blocked on d768, which remains running.
- Next action: continue monitoring d768 to terminal state, then compute the
  gate-1 decision.

### 2026-04-25 22:01 - MOE-AGGN-001 gate 1 passed; gate 2 launched

- Result: d768 finished successfully on Iris.
- Metrics:
  - W&B run:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-global-grad-norm-d768-1p70e18
  - global_step: `10342`
  - eval/paloma/macro_loss: `3.4274773597717285`
  - throughput/tokens_per_second: `275023.52176416403`
  - effective speedup vs README baseline: `1.043646604830193`
- Gate 1 decision:
  - d512 effective speedup: `1.0143777998139154`
  - d768 effective speedup: `1.043646604830193`
  - both points exceed `1.0`, so gate 2 should run.
- Command:
  `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --memory=2G --disk=4G --cpu=1 --extra=cpu --region us-central1 --no-preemptible -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX gs://marin-us-central1 -e GRUG_MOE_ADAMH_GLOBAL_GRAD_NORM_GATE gate2 -- python -m experiments.grug.moe.launch_adamh_global_grad_norm`
- Config:
  - commit: `6d26ec0ea`
  - Iris parent job: `/kaiyue/iris-run-job-20260426-050129`
  - data browser:
    https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/launch_adamh_global_grad_norm-e8d8ad.json
  - d1024 output:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d1024-9p00e18-9f3972`
  - d1024 W&B:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-global-grad-norm-d1024-9p00e18
  - d1280 output:
    `gs://marin-us-central1/grug/moe-adamh-global-grad-norm/d1280-2p83e19-ed47e1`
  - d1280 W&B:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-global-grad-norm-d1280-2p83e19
  - issue update, submitted with raw GitHub REST:
    https://github.com/marin-community/marin/issues/5182#issuecomment-4321300353
- Startup status: parent and both child jobs reached `JOB_STATE_RUNNING`.
  The first allocations hit the TPU device-busy/no-accelerator signature and
  Iris retried automatically. Both children later reached `Progress on:train`.
- Next action: monitor gate 2 to terminal state.

### 2026-04-25 22:10 - MOE-AGGN-001 d1280 overflow fix

- Result: the first gate-2 attempt was stopped after d1280 reproduced a code
  failure twice.
- Evidence:
  - Iris parent job:
    `/kaiyue/iris-run-job-20260426-050129`
  - d1024 remained running on the old commit with `failure_count=0`.
  - d1280 reached the first train step, then failed in
    `normalize_gradients_to_unit_rms` with:
    `OverflowError: An overflow was encountered while parsing an argument to a jitted computation, whose argument path is x2.`
  - The failing expression was `square_sum / num_elements`; d1280 has enough
    parameters for the Python integer element count to exceed signed int32
    when staged by JAX.
- Fix:
  - Convert the accumulated element count to a floating JAX scalar before the
    RMS division.
  - Added a regression test using a tiny fake gradient leaf with
    `size = 2**31`.
- Validation:
  - The new regression failed before the fix with the same overflow.
  - `uv run --with pytest --with pytest-timeout pytest tests/test_grug_moe_adamh_global_norm.py -q`
    passed after the fix.
  - `WANDB_MODE=disabled uv run python experiments/grug/moe/launch_adamh_global_grad_norm.py --dry_run True --executor_info_base_path "$tmp" --prefix "$tmp"`
    passed after the fix.
- Command:
  `.venv/bin/iris --config lib/iris/examples/marin.yaml job stop /kaiyue/iris-run-job-20260426-050129`
- Interpretation: gate 2 must be resubmitted on a fixed commit; the d1024
  partial run was also stopped so both gate-2 points use the same code.
- Next action: run pre-commit, commit and push the overflow fix, resubmit
  gate 2, update the issue via raw GitHub REST, and continue monitoring.
