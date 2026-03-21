# Alternating RL Logbook

Reference design doc: [`.agents/projects/alternating-multihost-rl.md`](/Users/ahmed/code/marin/.claude/worktrees/enchanted-crafting-wave/.agents/projects/alternating-multihost-rl.md)

This logbook is the running record for all ongoing experiments, smoke tests, failures, measurements, and follow-up fixes required to get alternating single-allocation multi-host RL working properly on real TPUs.

## Scope

Track:

- single-host smoke tests
- multi-host smoke tests
- compile-cache behavior across phase boundaries
- vLLM bootstrap and host-local sampling behavior
- materialization correctness and throughput
- Levanter training resume/export behavior
- controller/container lifecycle failures
- algorithmic behavior such as policy lag and `steps_per_phase` sensitivity

Do not use this file for speculative design notes that belong in the main design doc. Use it for execution history, observed behavior, concrete hypotheses, and next actions.

## Current Status

- Code status: controller/runtime/materialization/training path implemented in this worktree
- Local validation status:
  - alternating unit tests passing
  - repo pre-commit passing
- Next milestone:
  - first real TPU smoke test

## Success Criteria

We consider alternating RL viable only if all of the following are demonstrated on real TPU runs:

1. One full phase completes end to end: sampling -> materialization -> training -> export -> next policy manifest.
2. Warm-cache phase restarts reuse XLA caches well enough that boundary overhead is acceptable.
3. Multi-host training consumes materialized batches correctly with the current `hax.shard()` loader path.
4. Multi-phase runs complete without controller/state drift or container lifecycle failures.
5. Learning dynamics remain usable at some practical `steps_per_phase`.

## Experiment Template

Copy this block for each real run:

```md
## Run <ID>

- Date:
- Owner:
- Goal:
- TPU:
- Hosts:
- Image:
- Code revision:
- Command:
- Phase quotas:
- Expected outcome:

### Result

- Status:
- Completed phases:
- Wall-clock:
- Boundary overhead:
- Warm-cache behavior:
- Sampling throughput:
- Training throughput:
- Export result:

### Evidence

- Run state path:
- Policy manifest path:
- Materialized manifest path:
- Logs:
- Metrics:

### Findings

-

### Follow-up

- [ ]
```

## Hypotheses To Validate

### H1: Warm cache makes phase restarts acceptable

We expect persistent `JAX_COMPILATION_CACHE_DIR` and `VLLM_XLA_CACHE_PATH` to convert later phase starts from cold compile behavior into mostly warm-cache starts.

Evidence needed:

- first phase startup time
- second phase startup time
- whether the second phase still triggers large compilations

### H2: The materialized-batch loader is correct on multi-host TPU

The current design has each host read the same `TrainingBatch` pickle and then call `hax.shard(...)`. This is simple, but it must be validated on real distributed TPU runtime.

Evidence needed:

- no shape/sharding errors
- no silent gradient corruption symptoms
- stable multi-host training behavior on a tiny run

### H3: A practical `steps_per_phase` sweet spot exists

We expect there to be a workable range where:

- policy lag is not too large
- phase-boundary overhead is amortized enough

Evidence needed:

- at least a small sweep over `steps_per_phase`
- notes on wall-clock overhead and training behavior

## Run Ledger

## Open Issues

- Warm-cache reuse is not yet validated on real TPU.
- Multi-host loader correctness is not yet validated on real TPU.
- First real vLLM TPU bootstrap in the target image is not yet validated under the alternating controller.

## Decisions

- 2026-03-21: Use the alternating controller/runtime/state architecture in this worktree as the base implementation.
- 2026-03-21: Treat this mode as a single-allocation execution mode, not a replacement for concurrent two-TPU RL.
- 2026-03-21: Require compile-cache persistence from day 1.
