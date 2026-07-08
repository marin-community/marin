# JaxPP Grug Schedule Coverage Plan

## Current State

- Coordinating issue: https://github.com/marin-community/marin/issues/7024
- Logbook: `.agents/logbooks/jaxpp-grug-moe.md`
- Working implementation:
  - `implementation="explicit_mpmd"` partitions `Transformer` weights and optimizer state into contiguous pipeline stages.
  - Explicit 4-stage single-microbatch MPMD runs on 4x 8xH100 CoreWeave east02.
  - Explicit 4-stage GPipe runs with 4 microbatches. Best completed GPipe rung: d2560, 24 layers, 128 experts, top-k 4, batch 32, seq 128, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`, mean MFU `0.3793670762020138`.
  - Explicit `std_1f1b` runs with 4 microbatches. Best 64-expert 4x8 run so far: four physical/logical stages, d2560, 24 layers, top-k 4, batch 32, seq 128, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.70`, mean MFU `0.4853452360321486`.
  - Explicit 4-stage `std_1f1b` is no longer limited to two stages; 4-layer, 24-layer 8-expert, 24-layer 64-expert, 24-layer 128-expert, and 24-layer 256-expert d2560 runs succeeded on 4x 8xH100 with four physical/logical stages.
  - Largest completed 4-stage `std_1f1b` rung: 256 experts, 24 layers, top-k 4, batch 32, seq 128, `61,354,855,424` params, loss `9.045845985412598`, mean MFU `0.23384911314610907`.
  - Stage-local explicit MPMD init fixed the prior 192/256 optimizer-state initialization OOM by splitting parameters before `optimizer.init` and by localizing Optax scalar state to each stage mesh.
  - The highest MFU datapoint remains the 64-expert single-microbatch explicit MPMD rung, mean MFU `0.7529189015282038`; it is not a microbatch schedule comparison.
- Automatic JaxPP schedule path:
  - `std_1f1b` no longer fails on token-batch after-loop placement after moving microbatch reshape outside the JaxPP trace.
  - With `GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1` and `GRUG_JAXPP_PATCH_CONST_SHARDINGS=1`, it gets past the prior `jaxpp/sharding_inference.py:613` assertion on the reduced 4-layer/8-expert smoke.
  - It now fails in JaxPP input placement: `_maybe_shard_inputs` tries to `device_put` stage-local expert weights using a global `NamedSharding` whose mesh still includes the non-addressable `pipeline` axis.
  - Non-explicit mesh fallback is incompatible with Grug `reshard(..., PartitionSpec(...))` init.

## Main Gap

The user asked to try relevant JaxPP schedules. Explicit GPipe and explicit `std_1f1b` now run, including a 4-stage `std_1f1b` 64-expert comparison, but automatic JaxPP schedules still cannot execute Grug's explicitly sharded params end to end.

## Viable Next Paths

1. Get a less compile-dominated 256-expert performance number.
   - The 24-layer 256-expert capacity proof used 2 steps and has only one MFU sample.
   - A longer synthetic run at the same shape would make the MFU number more meaningful.
   - This stays on the proven `jaxpp.experimental.mpmd` API and avoids automatic JaxPP input-placement internals.

2. Fix automatic JaxPP input placement.
   - The reduced 4x8 failure has advanced from sharding inference to input placement:
     `/dlwh/iris-run-job-20260708-204723/grug-train-jaxpp-auto-std-l4-e8-constshard4-20260708-2052`.
   - Current failure: `device_put` rejects a global `NamedSharding(mesh=Mesh('pipeline': 4, ...), spec=P('expert', 'data', 'model'))` because it does not represent addressable devices for the local stage process.
   - Likely fix candidates:
     - convert automatic compiled step input shardings to addressable stage-local mesh shardings before `GlobalMpmdFunction.__call__`;
     - skip JaxPP `_maybe_shard_inputs` for arrays already resharded via `spmd_to_mpmd_reshard`;
     - make the const-sharding patch produce per-stage shardings when JaxPP extracts ClosedJaxpr consts.

## Recommended Next Step

For deliverable schedule coverage, the explicit path now has GPipe plus 4-stage `std_1f1b` MFU data through the requested 24-layer 256-expert shape. For upstream-quality automatic schedules, investigate JaxPP's `_maybe_shard_inputs` path with the v4 reduced reproducer above.

## Validation Ladder

1. Local syntax:
   - `uv run python -m py_compile experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py`
   - `./infra/pre-commit.py --fix experiments/grug/moe/train.py experiments/grug/moe/model.py experiments/grug/moe/launch.py experiments/grug/moe/launch_cw_jaxpp_may_d2560.py experiments/grug/moe/run_cw_jaxpp_may_d2560.sh .agents/logbooks/jaxpp-grug-moe.md`
2. Tiny automatic smoke:
   - d2560, 4 layers, 8 experts, top-k 1, batch 32, seq 128, 4 stages, 4 microbatches, `GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS=1`, `GRUG_JAXPP_PATCH_CONST_SHARDINGS=1`.
3. Explicit schedule regression:
   - d2560, 24 layers, 64 experts, top-k 4, batch 64, seq 128, GPipe and two-stage `std_1f1b`.
4. MFU comparison:
   - Compare explicit single-microbatch baseline, explicit GPipe, and explicit `std_1f1b`.
   - Treat automatic schedule runs as compile/smoke evidence until they emit loss.
