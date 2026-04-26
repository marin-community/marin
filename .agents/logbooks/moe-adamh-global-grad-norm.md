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
  - fix: default `launch_adamh_global_grad_norm.py` to
    `ExecutorMainConfig(prefix=os.environ.get("MARIN_PREFIX", "gs://marin-us-central1"))`.
- Result: stopped the self-submitted gate-1 parent and both child jobs after
  the prefix mismatch appeared. Focused optimizer tests and launch selector
  smoke still passed after the prefix fix.
- Interpretation: gate 1 should be resubmitted in `us-central1` with
  `MARIN_PREFIX=gs://marin-us-central1`; d512 can restore from the existing
  central1 checkpoint, and future parent preemptions should not change the
  storage prefix.
- Next action: lint, commit, push, then resubmit gate 1 with a pinned launcher
  region and continue monitoring to terminal state.
