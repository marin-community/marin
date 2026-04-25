# MoE Depth MuP LR Sweep: Research Logbook

## Scope

- Goal: test whether depth MuP residual scaling makes the current Grug MoE
  recipe less learning-rate-sensitive across model scale.
- Primary metrics: final `eval/paloma/macro_loss`, LR optimum curvature per
  scale.
- Secondary metrics: `throughput/tokens_per_second`,
  `throughput/total_tokens`, effective speedup versus the compute-optimal MoE
  baseline.
- Issue: https://github.com/marin-community/marin/issues/5178
- Prior LR sweep: https://github.com/marin-community/marin/issues/4225

## Baseline

- Date: 2026-04-25
- Code refs:
  - `experiments/grug/moe/README.md`
  - `experiments/grug/moe/model.py`
  - `experiments/grug/moe/heuristic.py`
- Baseline numbers: compute-optimal d512, d768, d1024, and d1280 table in
  `experiments/grug/moe/README.md`.

## Experiment Log

### 2026-04-25 11:25 - Kickoff

- Hypothesis: scaling each layer's residual updates by `1 / sqrt(num_layers)`
  will reduce depth-dependent residual growth and flatten the LR optimum across
  MoE scales.
- Command: local implementation only so far.
- Config:
  - architecture: current `experiments/grug/moe` recipe
  - intervention: depth MuP residual update scale
  - sweep scales: d512, d768, d1024, d1280 compute-optimal budgets from the
    README
  - LR multipliers: planned around the existing v16 LR formula
- Result: pending.
- Interpretation: pending.
- Next action: add tests for explicit residual scale behavior and sweep step
  construction, then implement the model and launch wiring.

### 2026-04-25 11:45 - Implementation scaffold

- Hypothesis: the smallest safe implementation is an opt-in model config flag,
  keeping the baseline recipe unchanged while the sweep enables depth MuP.
- Command:
  - `uv run --with pytest --with pytest-timeout python -m pytest tests/test_grug_moe_depth_mup.py`
  - `uv run --with pytest --with pytest-timeout python -m pytest tests/test_grug_variant_contracts.py -k moe`
  - `uv run --with pyrefly pyrefly check experiments/grug/moe/model.py experiments/grug/moe/heuristic.py experiments/grug/moe/depth_mup_lr_sweep.py tests/test_grug_moe_depth_mup.py`
  - `./infra/pre-commit.py --fix experiments/grug/moe/model.py experiments/grug/moe/heuristic.py experiments/grug/moe/depth_mup_lr_sweep.py experiments/grug/moe/README.md experiments/grug/moe/agent.md tests/test_grug_moe_depth_mup.py .agents/logbooks/moe-depth-mup-lr-sweep.md`
- Config:
  - `GrugModelConfig.depth_mup_residual_scaling=True`
  - residual update scale: `1 / sqrt(num_layers)`
  - sweep module: `experiments/grug/moe/depth_mup_lr_sweep.py`
  - sweep grid: d512, d768, d1024, d1280 x 9 LR multipliers
- Result:
  - red phase: 3 expected failures for missing config field, block scale, and
    sweep module
  - green phase: 3 depth-MuP tests passed after implementation
  - MoE contract slice: 3 passed
  - targeted Pyrefly: 0 errors
  - targeted pre-commit: OK
- Interpretation: local behavior is wired correctly; no TPU jobs launched yet.
- Next action: run broader contract checks, then decide whether to submit the
  sweep jobs.
