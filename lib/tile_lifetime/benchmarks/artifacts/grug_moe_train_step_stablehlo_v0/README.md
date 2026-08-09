# Ordinary Grug MoE train-step StableHLO

This artifact establishes that Shuttle can begin from an ordinary Grug MoE
training program without first introducing opaque model-semantic calls.

The probe constructs one small Grug MoE layer, executes ordinary JAX
`value_and_grad`, applies an optimizer update, and asks JAX for StableHLO. The
frontend configuration deliberately keeps math visible:

- reference tensor-algebra attention;
- scatter-based routed MoE;
- `jax.lax.ragged_dot_general` rather than Pallas/Sonic/DeepEP execution;
- no FA4, MoK, MSA, or other complete semantic kernel.

Both SGD and AdamW variants contain zero `stablehlo.custom_call` operations.
The SGD module is 329,403 characters and contains 82 `dot_general`, 96
`reduce`, two `sort`, 16 `scatter`, and ten `all_reduce` operations. AdamW is
466,782 characters and retains the same 82 contractions while exposing the
optimizer maps and reductions explicitly.

This is a structural fixture, not a performance result. It uses CPU lowering,
`B=2`, `S=4`, hidden/intermediate size 32, one layer, four experts, and top-2
routing so that the fixture stays inspectable. The next integration experiment
must capture the post-SPMD GPU `HloModuleProto` and prove that at least one
forward/backward region remains recoverable there.

Reproduce with:

```bash
uv run --frozen --package marin-tile-lifetime python \
  lib/tile_lifetime/benchmarks/grug_moe_train_step_hlo.py \
  --optimizer adamw \
  --stablehlo-output /tmp/grug-train-step.mlir \
  --summary-output /tmp/grug-train-step.json
```

Files:

- `train-step.mlir` and `summary.json`: SGD train step.
- `train-step-adamw.mlir` and `summary-adamw.json`: AdamW train step.
- The generating script lives at
  `lib/tile_lifetime/benchmarks/grug_moe_train_step_hlo.py`.

