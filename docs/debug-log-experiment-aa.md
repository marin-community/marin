# Debugging log for experiment AA

Assess whether Experiment AA in `.agents/logbooks/debug_accum_tpu_type.md` is a sound causal test of the v5p-8 LoRA failure mechanism.

## Initial status

The AA plan proposes casting gradients to `f32` in `_train_step` after `_compute_gradients_microbatched(...)` returns and before `state.take_step(...)`, with the stated goal of forcing the FSDP cross-chip gradient all-reduce to run in `f32`.

## Hypothesis 1

The AA plan identifies the wrong boundary. `_train_step` does not return gradients, so there is no `_train_step` output-boundary reshard for the grad tree.

## Changes to make

No product-code changes. Read the trainer, trainer state, partitioning, grad accumulation, and experiment-Q code paths to verify where gradients are produced, sharded, and consumed.

## Results

- `_train_step` is wrapped in `named_jit(...)`, but it computes `grads` and immediately passes them into `state.take_step(...)`; it returns `TrainStepResult`, not gradients.
- The bad `Experiment Q` `pd=4` baseline is microbatched.
- In the microbatched path, `microbatched(...)` reshards the accumulator with `hax.shard_with_axis_mapping(acc, accum_axis_mapping)` inside the accumulation loop.
- Therefore, for the actual bad recipe, a plausible grad collective site already exists inside gradient accumulation before AA's proposed cast point.
- Conclusion: AA's scientific goal makes sense, but the specific patch location/rationale in the logbook is not clean for the `pd=4` baseline. A better intervention site is inside `microbatched(...)` immediately before `hax.shard_with_axis_mapping(...)`, or via `accum_dtype=jnp.float32`.

## Future Work

- [ ] Rework AA so the cast sits immediately upstream of the actual grad reshard/all-reduce site in the microbatched path
- [ ] Dump HLO for the revised intervention and verify the targeted reduction regions are `f32`
- [ ] If needed, run a non-microbatched control to isolate outer-train-step reshard effects
