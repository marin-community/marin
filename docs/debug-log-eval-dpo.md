# Debugging log for eval_dpo

Fix `experiments/eval_dpo.py` so it profiles standalone DPO evaluation using the real Levanter model-init, sharding, and profiler paths.

## Initial status

`experiments/eval_dpo.py` was added as a standalone eval profiler, but it diverged from current Levanter APIs and the DPO training path:

- `prepare_model_init_context()` and `load_model_from_source()` were called with missing required arguments.
- `TreeCache.load()` was called without the required cache exemplar and metadata.
- The script used `compute_axis_mapping` where model loading expects `parameter_axis_mapping`.
- The script bypassed the usual mesh and axis-mapping context.
- The script manually managed JAX tracing instead of using `profile_ctx()`.

## Hypothesis 1

The script was copied from older or partially remembered APIs, so it no longer matches the current model-init and cache-loading helpers.

## Changes to make

- Update `experiments/eval_dpo.py` to call `prepare_model_init_context()` and `load_model_from_source()` with the current required arguments.
- Load the tokenizer explicitly and use `preprocessor_for_preference_format()` plus `TreeCache.load(..., exemplar=..., options=...)`.
- Reuse the Bloom SpecEval v2 config, tokenizer, and model definitions instead of hardcoding parallel copies.

## Results

The script now uses the current model-init and preference-cache APIs and pulls model/tokenizer/beta settings from the existing Bloom SpecEval v2 experiment config.

## Hypothesis 2

Even with corrected helper calls, the profile would still be misleading if the script did not run under the same mesh, dtype, and profiler context as real DPO eval.

## Changes to make

- Run the script under `trainer_config.use_device_mesh()` and `hax.axis_mapping(parameter_axis_mapping)`.
- Match DPO mixed precision with `jmp.get_policy("p=f32,c=bfloat16")`.
- Use `profile_ctx()` for trace lifecycle and artifact upload.
- Explicitly finish the tracker before exit.

## Future Work

- [ ] Factor the DPO eval loss body into a shared helper so this script and `train_dpo.py` do not duplicate it.
- [ ] Replace the hardcoded validation cache path with a more robust config or CLI argument.
- [ ] Add a small smoke-test mode that profiles a bounded number of eval batches locally.

## Results

`experiments/eval_dpo.py` now mirrors the real DPO eval path much more closely: same model config, same policy/reference load path, same mixed precision, same eval loop, and the repo’s standard profiler helper.

Verification so far:

- `uv run python -m py_compile experiments/eval_dpo.py` passes.
- `uv run python - <<'PY' import experiments.eval_dpo as m; print(...) PY` imports the module successfully.

## Hypothesis 3

The remote TPU crash is caused by running the eval function under the wrong thread-local axis mapping. The script left the global mapping set to `parameter_axis_mapping`, which shards `embed` across data shards. The fused CE kernel then enters `shard_map(...)` with local activations shaped like `embed(256)` while still comparing against the model contract axis `embed(4096)`, producing the `Axis embed present in both specs with different sizes` failure.

## Changes to make

- Stop holding `hax.axis_mapping(parameter_axis_mapping)` around the full eval script.
- Run the standalone eval function with `hax.named_jit(axis_resources=compute_axis_mapping)` so eval executes under compute sharding, matching other standalone eval entrypoints like `eval_lm.py`.

## Results

`experiments/eval_dpo.py` now loads parameters with `parameter_axis_mapping` but runs eval batches under `compute_axis_mapping`, which should keep `batch` sharded and `embed` unsharded during fused next-token loss.
