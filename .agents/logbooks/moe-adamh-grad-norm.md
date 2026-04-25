# MoE AdamH Gradient Normalization: Research Logbook

## Scope

- Goal: test whether normalizing each module's gradients to RMS 1 before AdamH
  moment updates improves Grug MoE training relative to the current AdamH
  recipe.
- Primary metrics: final `eval/paloma/macro_loss`, effective speedup versus
  the compute-optimal MoE baseline.
- Secondary metrics: `throughput/tokens_per_second`,
  `throughput/total_tokens`, routing balance/stability metrics.
- Constraints: compare at the README compute-optimal d512, d768, d1024, and
  d1280 budgets; run gate 1 before gate 2.
- Issue: https://github.com/marin-community/marin/issues/5180

## Baseline

- Date: 2026-04-25
- Code refs:
  - `experiments/grug/moe/README.md`
  - `experiments/grug/moe/adamh.py`
  - `experiments/grug/moe/optimizer.py`
  - `experiments/grug/moe/launch.py`
- Baseline numbers: compute-optimal d512, d768, d1024, and d1280 table in
  `experiments/grug/moe/README.md`.

## Experiment Log

### 2026-04-25 11:45 - Kickoff

- Hypothesis: module-wise gradient RMS normalization reduces scale mismatch
  between attention, shared expert, and routed expert AdamH groups without
  changing AdamH's projected parameter update rule.
- Command: local implementation and unit tests.
- Config:
  - optimizer: `GrugMoeAdamHGradientNormConfig`
  - normalization: each module's gradient leaves are scaled to combined RMS 1
    before AdamH moment updates.
  - gate 1 launch: `GRUG_MOE_ADAMH_GRAD_NORM_GATE=gate1 python -m experiments.grug.moe.launch_adamh_grad_norm`
  - gate 2 launch: `GRUG_MOE_ADAMH_GRAD_NORM_GATE=gate2 python -m experiments.grug.moe.launch_adamh_grad_norm`
- Result: implementation passes focused optimizer tests and pre-commit locally.
- Interpretation: ready to run gate 1 against d512 and d768 baselines.
- Next action: create the GitHub experiment issue, push the branch, and submit
  the gate 1 Iris job.

### 2026-04-25 11:46 - MOE-AGNH-001 gate 1 submitted

- Hypothesis: the gradient-normalized AdamH variant can match or improve the
  d512 and d768 compute-optimal MoE baselines without hurting throughput.
- Command:
  `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --memory=2G --disk=4G --cpu=1 --extra=cpu --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e GRUG_MOE_ADAMH_GRAD_NORM_GATE gate1 -- python -m experiments.grug.moe.launch_adamh_grad_norm`
- Config:
  - commit: `d97fdc3131750bfaab552d459fd14e51bf8a2860`
  - issue: https://github.com/marin-community/marin/issues/5180
  - PR: https://github.com/marin-community/marin/pull/5181
  - Iris parent job: `/kaiyue/iris-run-job-20260425-184632`
  - data browser:
    https://marin.community/data-browser/experiment?path=gs%3A//marin-us-east5/experiments/launch_adamh_grad_norm-7614bf.json
  - W&B d512:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-grad-norm-d512-2p19e17
  - W&B d768:
    https://wandb.ai/understanding-sam/marin_moe/runs/moe-adamh-grad-norm-d768-1p70e18
- Result: submitted to Iris and dispatched both Fray TPU child jobs. The first
  TPU allocation hit a device-busy bad-node signature; Iris retried
  automatically and both child jobs reached `JOB_STATE_RUNNING`.
- Interpretation: gate 1 is healthy enough to monitor for first eval/final
  Paloma macro loss rather than resubmit immediately.
- Next action: babysit the d512 and d768 jobs to terminal state, compare final
  metrics with the README baselines, then decide whether to launch gate 2.

### 2026-04-25 12:40 - MOE-AGNH-001 d512 final

- Hypothesis: d512 should show effective speedup greater than 1 versus the
  README compute-optimal baseline.
- Command: W&B API summary pull for
  `understanding-sam/marin_moe/moe-adamh-grad-norm-d512-2p19e17`.
- Config:
  - budget: `2.19e17`
  - baseline Paloma macro loss: `3.8104`
  - baseline throughput: `405630` tokens/s
  - variant Paloma macro loss: `3.815110206604004`
  - variant throughput: `406982.727149109` tokens/s
  - variant total tokens: `837156864`
- Result: d512 Iris child job succeeded. Effective speedup is `0.980893`;
  loss delta is `+0.004710` and throughput delta is `+0.333%`.
- Interpretation: d512 does not clear gate 1. Since the d768 gate 1 child job
  is already running, continue it to terminal state for the complete gate 1
  comparison, but do not launch gate 2 unless the decision criteria are
  explicitly revised.
- Next action: continue babysitting d768, then close out the issue with the
  gate 1 table.
