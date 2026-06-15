# Debugging log for DeepEP remat

Goal: make the Grug DeepEP MoE path work with remat enabled on CoreWeave.

## Initial status
DeepEP FFI builds in the CUDA-devel image. With MAY_REMAT=save_moe, tracing fails before runtime with JAX `FfiEffect` inside full-block checkpoint/remat. With remat disabled, tracing gets past that failure but DeepEP runtime dispatch still fails on H100 shared-memory kernel setup.

## Hypothesis 1
Full-block remat wraps DeepEP dispatch/combine FFI calls. Split remat so side-effecting transport stays outside checkpoint regions, and checkpoint only pure attention/expert compute.

## Changes to make
- Thread remat mode into the MoE expert MLP dispatcher.
- For DeepEP, wrap only the pure local expert up/down computation in `jax.checkpoint`.
- For the experiment model, avoid full-block checkpointing when `moe_implementation=deepep`; checkpoint attention separately and call the MLP outside the block checkpoint.

## Results for hypothesis 1
- `uv run --package marin-levanter pytest tests/test_grug_variant_contracts.py::test_grug_moe_remat_mode_controls_checkpoint_boundary -q`: passed.
- `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`: passed before adding the explicit remat-mode API tests.
- `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`: passed after adding the remat-mode API tests.
- `./infra/pre-commit.py --files experiments/grug/moe/model.py lib/levanter/src/levanter/grug/_moe/common.py lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/src/levanter/grug/_moe/ep_deepep.py lib/levanter/src/levanter/grug/_moe/ep_ring.py lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py lib/levanter/tests/grug/test_grugformer_moe.py docs/debug-log-deepep-remat.md --fix`: passed.
- CoreWeave job `deepep-remat-grug-20260615-000805` reached one-step Grug training with `moe_implementation=deepep` and `remat_mode=save_moe`. It did not hit the previous JAX remat error (`Effects not supported in partial-eval of checkpoint/remat: [FfiEffect()]`). It failed later at the existing H100 DeepEP dispatch runtime issue: `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess` in `/tmp/DeepEP/csrc/kernels/intranode.cu:521`.

## Hypothesis 2
The H100 runtime failure uses the upstream DeepEP `intranode.cu` launch pattern even though the build cache key is computed from patched source bytes. `_prepare_intranode_source` only wrote generated patched source when the thread-count override changed the source, so the default H100 path compiled the original upstream file while the cache expected the patched launch pattern.

## Changes to make
- Always compare upstream `intranode.cu` with `_intranode_source_bytes`.
- Compile `build/generated/intranode.cu` whenever the patched bytes differ from upstream bytes.
- Keep returning upstream `intranode.cu` only when no source patch is needed.

## Results for hypothesis 2
- `./infra/pre-commit.py --files lib/levanter/src/levanter/kernels/deepep/transport_ffi.py --fix`: passed.
- CoreWeave job `deepep-runtime-grug-20260615-002358` completed a one-step Grug DeepEP training smoke on one 8xH100 node with `MAY_MOE_IMPLEMENTATION=deepep` and `MAY_REMAT=save_moe`.
- The run reached `model remat save_moe moe deepep`, trained one step, and did not hit the previous `cudaFuncSetAttribute` shared-memory setup failure.
- `uv run --package marin-levanter pytest lib/levanter/tests/kernels/test_deepep_availability.py -q`: passed.
- `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grugformer_moe.py -q`: passed.
- `./infra/pre-commit.py --files experiments/grug/moe/model.py lib/levanter/src/levanter/grug/_moe/common.py lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/src/levanter/grug/_moe/ep_deepep.py lib/levanter/src/levanter/grug/_moe/ep_ring.py lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py lib/levanter/src/levanter/kernels/deepep/transport_ffi.py lib/levanter/tests/grug/test_grugformer_moe.py lib/levanter/tests/kernels/test_deepep_availability.py docs/debug-log-deepep-remat.md --fix`: passed.
