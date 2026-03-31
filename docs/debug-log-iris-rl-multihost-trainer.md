# Debugging Log For Iris RL Multihost Trainer

Goal:

- explain why the RL trainer/export path is failing or becoming memory-tight on
  multi-host `v6e-16`
- recover a distributed-safe export path that keeps per-step sync semantics

## Initial status

Current situation:

- the old `tree_jit` Arrow Flight export path is distributed-safe enough to run
  on the multi-host trainer after earlier fixes, but it appears memory-tight
- the new `sequential_host_flatten` experiment dies immediately on multi-host
  bootstrap weight serve

Observed TPU probe failures:

- `/ahmed/iris-rl-v6e-e5b-exportdbg-001`
- `/ahmed/iris-rl-v6e-e1d-exportdbg-001`

Observed stack shape:

- `train_worker.py` calls `serve_weights(-1, state.model)`
- `arrow_flight.py` enters `_export_weights_sequential_host_flatten(...)`
- eager `hsd.to_state_dict(model)` reaches Haliax scan `_unstack_state_dict(...)`
- eager iteration over a non-fully-addressable global `jax.Array` raises:

```text
AssertionError
assert self.is_fully_replicated or self.is_fully_addressable
```

## Hypothesis 1: eager state-dict conversion is invalid on the concrete sharded trainer

The new sequential path broke multi-host correctness by moving
`hsd.to_state_dict(model)` out of the old jitted path and into eager Python.

Expected consequence:

- scanned layers try to unstack by iterating a concrete global array
- eager iteration is only valid for fully replicated or fully addressable
  arrays
- multi-host trainer leaves are neither

## Changes to make

No code changes landed yet in this debug log. Planned debugging actions:

1. add a focused regression test that demonstrates:
   - old jitted `copy_and_flatten(...)` path survives
   - eager `hsd.to_state_dict(model)` path fails on the same sharded structure
2. prove whether the difference is:
   - JIT tracing semantics
   - Haliax scan eager unstack behavior
   - or both

## Results

Current conclusion:

- this is not evidence that Levanter multi-host training is broken
- it is evidence that our new eager serialization variant is not valid for the
  current sharded trainer representation

## Hypothesis 2: the right fix is a distributed-safe low-peak export, not eager leaf walking

The correct path likely keeps distributed-safe state-dict / flatten work under
JIT or another collective-safe mechanism, then reduces peak pressure later:

- sequential host materialization
- sequential Arrow batch creation
- discard-as-you-go serialization

This would preserve:

- multi-host correctness
- per-step sync semantics

while targeting the real old problem:

- trainer export HBM pressure

## Future Work

- [ ] Add regression coverage for eager-vs-jitted export on a sharded scanned model
- [ ] Quantify where the old `tree_jit` path creates its largest live buffers
- [ ] Recover a valid low-peak export path and rerun a short `v6e-16` trainer probe
- [ ] Only after that, evaluate whether `v6e-32` repartitioning is necessary
