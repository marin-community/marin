---
topic: inkling-relative-position-gate1
issue: https://github.com/marin-community/marin/issues/7208
description: Evaluate Inkling-style relative attention bias against the July MoE baseline.
author: kaiyuew
---

# Inkling relative position Gate 1: Task Logbook

## Current TL;DR

The two-cell Gate 1 launcher is locally validated and ready for Iris submission.
The fixed comparator is the July Baseline in #6882, not the newer baseline table
in `experiments/grug/moe/README.md`.

## Scope

- Goal: determine whether learned input-dependent relative attention bias has effective wall-clock speedup greater than 1 at both Gate 1 scales.
- Primary metrics: final `eval/paloma/macro_loss`, last-100-step `throughput/tokens_per_second`, and final `throughput/total_tokens`.
- Constraints: exact #6882 d512/d768 cells; v5p-8; 8192-token sequences; no PKO; zero final-logit z-loss.
- Coordinating issue: https://github.com/marin-community/marin/issues/7208

## Baseline

- Date: 2026-07-15
- Code refs: #6882, branch `july_baseline` at `52d8a9eb8d9434cf1dcaaee060edeadc60dfff9d`.
- d512: budget 3.82e17, batch 16, 10,980 steps, macro loss 3.5667, 352,609 tok/s.
- d768: budget 2.81e18, batch 32, 16,875 steps, macro loss 3.2272, 249,954 tok/s.

## Hypothesis Queue

### Active

- `MOE-RPE-001`: the relative-position d512 cell has effective speedup greater than 1 versus #6882. Next test: run the d512 cell.
- `MOE-RPE-002`: the relative-position d768 cell has effective speedup greater than 1 versus #6882. Next test: run the d768 cell.

### Blocked

None.

### Falsified / Dead End

None.

### Promoted

None.

## Entry Log

### 2026-07-15 12:04 PDT - Gate 1 launch snapshot

- Hypothesis: replacing positional encoding with Inkling-style learned input-dependent relative bias yields effective speedup greater than 1 at both July Baseline Gate 1 cells.
- Commit Hash: `94bbf165ef23e8221e819cfc8002a7de633e2b29`
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe_relative_position.launch`
- Config: `MOE-RPE-001-d512` uses batch 16 for 10,980 steps; `MOE-RPE-002-d768` uses batch 32 for 16,875 steps. Both use sequence length 8192, seed 0, MuonH, 256 experts/top-4, the #6882 Nemotron mixture, disabled PKO, zero final-logit z-loss, and W&B `marin-community/dial_moe` group `MOE-RPE-gate1-issue-7208`.
- Result: local config contract, dense value/gradient parity, 8192-token lowering probe, dry-run DAG, and repository lint passed. Iris submission is pending.
- Interpretation: the launch snapshot is reproducible and matches the requested comparator; no experiment claim is available before the runs finish.
- Next action: push the snapshot and submit both cells through Iris.

### 2026-07-15 12:06 PDT - Final formatted launch snapshot

- Hypothesis: unchanged from `MOE-RPE-001` and `MOE-RPE-002` above.
- Commit Hash: `a496bc044`
- Command: `.venv/bin/iris --cluster=marin job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe_relative_position.launch`
- Config: unchanged from the prior entry.
- Result: applied the repository's Black formatting to two experiment files and reran `./infra/pre-commit.py --changed-files --fix` successfully. No behavior changed.
- Interpretation: use this commit, rather than the earlier pre-format snapshot, as the reproducible launch revision.
- Next action: push the snapshot and submit both cells through Iris.
