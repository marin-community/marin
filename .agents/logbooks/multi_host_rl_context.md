# Multi-Host RL Context

Branch: `multi_host_rl`
Issue: `#4287` `Experiment: Multi-host RL trainer weight sync on v6e-16`
Base: `origin/iris_rl` at `59601ab7660013797b6ae7f095d5b9c7e9615151`

## Purpose

This note is the branch-local entry point for the multi-host RL trainer thread.
It exists so future work on `v6e-16` trainer export and per-step weight sync
does not have to reconstruct the distributed failure history from the full Iris
RL migration logbook.

## Current Read

- Single-host direct `e4ms2` is stable and already validated by a successful
  `500/500` direct Iris run.
- The remaining scaling blocker is trainer-side weight export and per-step
  weight sync on multi-host `v6e-16`.
- The distributed trainer problem is not "Levanter multi-host training is
  broken." The problem is specifically the RL export / serialization path that
  must materialize trainer state for rollout workers every step.

## What Has Already Been Proven

### Stable single-host baseline

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-clean-nodelprevtmp-train`

Use that run as the correctness and robustness anchor. Multi-host work should
be framed as a scaling follow-up, not as a revalidation of the base Iris RL
pipeline.

### Relevant multi-host probes

`iris-rl-v6e-e1d`:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-v6e-e1d-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-v6e-e1d-rollout-0`

`iris-rl-v6e-e5b`:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-v6e-e5b-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-v6e-e5b-rollout-0`

### Distinct bugs that were already exposed

1. `v6e-8` bootstrap export OOM
   - `copy_and_flatten(...)` during bootstrap serve hit HBM pressure on
     `v6e-8`

2. Multi-host deadlock in trainer weight serve
   - only process `0` performed `copy_and_flatten(...)` / `device_get(...)`
   - on multi-host trainers those operations trigger cross-host collectives
   - the fix was to make all processes participate in the collective-affecting
     part of `serve_weights()`

3. Non-addressable-array materialization failure
   - direct host materialization of non-fully-addressable global arrays was
     invalid
   - the fix used multi-host-safe host gathering for those leaves

4. New `sequential_host_flatten` probe is semantically invalid on sharded
   multi-host trainers
   - eager `hsd.to_state_dict(model)` on the concrete trainer object ends up
     iterating non-fully-addressable global arrays in Python
   - this is different from the old jitted path and currently blocks the
     low-peak export experiment

## Main Conclusion So Far

The old jitted `copy_and_flatten(...)` path is distributed-safe enough to get
farther, but remains memory-tight. The new lower-memory
`sequential_host_flatten` direction is interesting, but the current eager
implementation is invalid for sharded multi-host trainers.

The immediate branch goal is therefore not "just try a larger TPU." It is:

1. preserve distributed-safe export semantics
2. recover a lower-peak host-materialization / serialization path
3. keep per-step weight sync

## Read These First

### Primary narrative

- `.agents/logbooks/iris-rl-multihost-trainer.md`
  - canonical focused write-up for the `v6e` / distributed trainer thread
- `.agents/logbooks/iris-rl-codex.md`
  - high-level migration context and where the multihost thread split off
- `.agents/logbooks/iris-rl-claude.md`
  - chronological evidence for the early `v6e` attempts and the deadlock
    diagnosis

### Detailed debug notes

- `docs/debug-log-iris-rl-multihost-trainer.md`
- `docs/debug-log-v6e-multihost-weight-transfer-materialization.md`
- `docs/debug-log-v6e-rollout-sync-oom.md`

### Code to inspect

- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
- `lib/marin/src/marin/rl/weight_transfer/checkpoint.py`
- `lib/marin/src/marin/rl/weight_transfer/base.py`
- `lib/marin/src/marin/rl/weight_transfer/jax.py`
- `experiments/xp_iris_rl_regression_direct_gcs_prod.py`
  - region-aware launcher introduced during the `v6e` thread

### Tests and nearby integration points

- `tests/rl/test_weight_transfer.py`
- `tests/rl/test_rl_experiment_utils.py`

## Specific Hypotheses Worth Rechecking

1. The real remaining problem is export-time materialization and serialization,
   not steady-state trainer sharding.
2. Eager `hsd.to_state_dict(model)` on a concrete multi-host trainer is the
   wrong abstraction boundary.
3. The best next design likely keeps the distributed-safe state-dict conversion
   inside JIT, then performs host materialization and Arrow serialization more
   sequentially afterward.
4. Bigger TPU shapes only help if they actually reduce per-chip export pressure
   for the leaves that are currently being amplified back toward near-full bf16
   sizes.

## Things Not To Rediscover

- Direct single-host Iris RL already works.
- Checkpoint stability is no longer the main blocker; see
  `.agents/projects/ckpt_rl.md` for that thread.
- The original coordinator / retry / W&B resume issues were fixed before this
  branch split and are not the main subject of this branch.

## Suggested Immediate Next Step

Use this branch to answer one question cleanly:

"What export path keeps per-step weight sync and remains valid on a sharded
multi-host trainer, while reducing the peak memory pressure that still makes
`v6e-16` trainer export fragile?"
