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
