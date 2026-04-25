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
