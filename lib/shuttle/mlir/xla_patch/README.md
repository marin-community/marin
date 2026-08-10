# Pinned XLA StableHLO transform hook

This directory contains a proposed XLA patch for the native Shuttle MLIR
extension boundary. It applies to OpenXLA revision
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`, the revision used by the pinned
JAX 0.10.1 stack.

The patch adds a generic, public StableHLO module-transform registry. When
`xla_shuttle_enable` is true, `MlirToXlaComputation` runs the registered
transforms after Shardy and CHLO cleanup and immediately before
`ConvertStablehloToHloWithOptions`. The registry clones the module, passes the
opaque `xla_shuttle_options` string to each transform, verifies the resulting
MLIR, and commits the clone only after every transform succeeds. Enabling the
hook without a registered transform is a compilation error.

The patch does not contain the Shuttle dialect or pass pipeline. A
Shuttle-enabled jaxlib must link a registration translation unit that calls:

```cpp
xla::StablehloModuleTransformRegistry::Global().Register(
    "shuttle", RunShuttleStablehloPipeline);
```

The registration library must be strongly linked, for example with an
`alwayslink = True` Bazel target when registration occurs during static
initialization. `RunShuttleStablehloPipeline` owns option-schema validation,
source-coverage checks, semantic-erasure checks, and verification that no
Shuttle operation or attribute reaches XLA's StableHLO-to-HLO conversion.

Apply and test the patch from an exact XLA checkout:

```bash
test "$(git rev-parse HEAD)" = 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
git apply --check /path/to/0001-add-stablehlo-module-transform-hook.patch
git apply /path/to/0001-add-stablehlo-module-transform-hook.patch
bazel test \
  //xla/pjrt:stablehlo_module_transform_test \
  //xla/pjrt:mlir_to_hlo_test \
  //xla/pjrt:pjrt_executable_test
```

The artifact is source-level integration work. Marin does not vendor XLA, and
the patch has not been compiled in this repository. End-to-end acceptance
still requires a Shuttle-enabled jaxlib running ordinary
`jax.jit(..., compiler_options=...)` forward and JAX-generated backward
programs.
