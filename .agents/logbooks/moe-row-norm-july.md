---
topic: moe-row-norm-july
issue: https://github.com/marin-community/marin/issues/8131
baseline_issue: https://github.com/marin-community/marin/issues/6882
description: Evaluate factorized output scales with row-wise MuonH on the exact July MoE baseline.
author: kaiyuew
---

# MoE Row Norm on July Baseline: Research Logbook

## Current TL;DR

The corrected exact-July d512 pair completed successfully. Row-norm finished at
Paloma macro loss 3.59609 and 348,644 final-100-step tokens/s, versus 3.57940
and 355,680 tokens/s for the paired untouched July control. Its effective
speedup is 0.8965x, so the variant fails at d512 and should not advance to d768.
Both runs descend directly from July commit
`52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.

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

### 2026-08-10 19:14 PDT - Corrected pair advancing

- Hypothesis: an exact-codebase control separates the row-norm intervention
  from drift between the July branch and current Marin.
- Iris parent: `/kaiyuew/moe-row-norm-july-d512-pair-8131`.
- W&B:
  - Variant: https://wandb.ai/marin-community/dial_moe/runs/MOE-ROW-NORM-JULY-001-d512
  - Control: https://wandb.ai/marin-community/dial_moe/runs/MOE-ROW-NORM-JULY-CTRL-001-d512
- Config: the serialized runtime configs match on all selected model, data,
  optimizer-scalar, trainer, and hardware fields. They differ only in the
  intended model/optimizer implementation and checkpoint destination.
- Result: both runs advanced across two observations with finite loss. The
  second observation was variant step 267, loss 5.5504, 352,273 tokens/s;
  control step 369, loss 5.2661, 355,829 tokens/s. The step offset comes from
  the variant's longer initial compilation.
- Interpretation: the corrected launch passed the fourth check: live runtime
  config equality and healthy paired execution.
- Next action: monitor both jobs through terminal state and compare final
  Paloma macro loss and final-100-step throughput.

### 2026-08-10 20:39 PDT - Corrected pair terminal

- Hypothesis: row-wise direction updates plus norm-preserving output scales
  improve quality enough to offset any throughput cost.
- Iris parent: `/kaiyuew/moe-row-norm-july-d512-pair-8131` (succeeded).
- Result:
  - Row-norm: final Paloma macro loss 3.5960889; mean throughput over steps
    10,880--10,979 was 348,644 tokens/s.
  - Exact July control: final Paloma macro loss 3.5793972; mean throughput over
    steps 10,880--10,979 was 355,680 tokens/s.
  - Delta: row-norm was 0.0166917 loss worse and 1.98% slower.
  - Effective speedup against the paired control: 0.8965x at the d512
    3.82e17-FLOP budget.
  - The historical July run was loss 3.5667 at 352,609 tokens/s. The new
    exact-code control was 0.01270 loss worse and 0.87% faster, demonstrating
    enough run-to-run drift that the paired control is the primary comparator.
- Checkpoints:
  - `gs://marin-us-central1/grug/moe_row_norm_july_d512-a38388/checkpoints/step-10980`
  - `gs://marin-us-central1/grug/moe_row_norm_july_baseline_control_d512-cf2086/checkpoints/step-10980`
  - Both metadata files report step 10,980 and `is_temporary: false`.
- Interpretation: the hypothesis is rejected at d512. The quality penalty
  narrowed late but never reversed, and the intervention also reduced
  throughput. This cell fails the first Gate-1 scale.
- Next action: do not launch d768 for this variant. Retain the implementation
  and histories as a negative result.
