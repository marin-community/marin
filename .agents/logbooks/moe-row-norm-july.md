---
topic: moe-row-norm-july
issue: https://github.com/marin-community/marin/issues/8131
baseline_issue: https://github.com/marin-community/marin/issues/6882
description: Evaluate factorized output scales with row-wise MuonH on the exact July MoE baseline.
author: kaiyuew
---

# MoE Row Norm on July Baseline: Research Logbook

## Current TL;DR

The first Gate-1 launch matched July hyperparameters but was not descended from
the `july_baseline` branch. Its d512 result is not decision-grade, and its d768
cell was stopped at step 6,257. A later matched control was stopped at step 707
for the same ancestry error. The corrected variant is now based directly on
July commit `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d` and snapshot at
`82482dc554893a8b1025ed6561d8224e98013e66`.

## Scope

- Goal: determine whether writing each linear as `v * (Wx)`, preserving each
  mathematical output-row norm of `W`, and preserving each scale-vector L2
  norm improves effective training speed.
- Primary metrics: final `eval/paloma/macro_loss`, final-100-step mean
  `throughput/tokens_per_second`, and effective speedup.
- Constraints: exact issue #6882 July baseline ancestry and recipe; `W` retains
  July initialization; every `v` starts at one; the LM-head `W` remains AdamH;
  router and attention-gate `W` retain Adam.

## Baseline

- Date: 2026-07-04.
- Code ref: `july_baseline` commit `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
- d512: Paloma macro loss 3.5667, 352,609 tokens/s, batch 16, 10,980
  steps, 8192-token context, v5p-8.
- W&B: https://wandb.ai/marin-community/dial_moe/runs/moe_may_july_baseline_d512

## Experiment Log

### 2026-08-10 17:00 PDT - Non-July launch invalidated

- Hypothesis: matching current model and optimizer config fields was sufficient
  to compare with the July baseline.
- Command: `git merge-base --is-ancestor marin/july_baseline HEAD` and direct
  branch/file comparison.
- Config: prior branch `codex/moe-row-norm-8131`; July head `52d8a9eb8`.
- Result: the ancestry check failed; merge-base was `4aac51f5`. The completed
  d512 row-norm run ended at loss 3.5934 and 344,365 tokens/s but is not a valid
  July-branch result. The d768 row-norm run was stopped at step 6,257, and the
  mistakenly launched control was stopped at step 707. All W&B histories were
  preserved.
- Interpretation: config equality does not establish codebase equality after
  substantial changes to Grug and Levanter. These runs are excluded from the
  decision.
- Next action: rebuild directly from the July head and relaunch a paired d512
  variant/control.

### 2026-08-10 17:12 PDT - Exact-July implementation snapshot

- Hypothesis: the scale/direction factorization can preserve the exact July
  function at initialization while changing only optimizer geometry.
- Command:
  - `uv run --package marin-core --group test pytest -q experiments/grug/moe_row_norm/test_optimizer.py`
  - `uv run --package marin-core --group test pytest -q tests/test_grug_variant_contracts.py`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run python -m experiments.grug.moe_row_norm.launch --dry_run true`
- Config: d512, batch 16, 10,980 steps, sequence length 8192, 256 experts,
  top-4, 4:1 GQA, no PKO, no long-window RoPE, z-loss 0, v5p-8.
- Result: 7 focused tests and 16 Grug contract tests passed; lint, formatting,
  and type checks passed. The test independently matches initialized weights
  and logits to the July model. The dry run resolves exactly the paired d512
  checkpoints plus their July data dependencies.
- Interpretation: three independent checks now bind the run to July: git base
  ancestry, exact recipe equality, and initialized numerical parity.
- Next action: submit W&B runs `MOE-ROW-NORM-JULY-001-d512` and
  `MOE-ROW-NORM-JULY-CTRL-001-d512`, verify both advance, then compare terminal
  metrics.
