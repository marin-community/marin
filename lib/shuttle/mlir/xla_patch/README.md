# Pinned XLA StableHLO transform hook

This directory contains a proposed XLA patch for the native Shuttle MLIR
extension boundary. It applies to OpenXLA revision
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`, the revision used by the pinned
JAX 0.10.1 stack.

The patch adds a generic, public StableHLO module-transform registry. When
`xla_shuttle_enable` is true, `MlirToXlaComputation` runs the registered
`shuttle` transform after Shardy and CHLO cleanup and immediately before
`ConvertStablehloToHloWithOptions`. The registry clones the module, passes the
opaque `xla_shuttle_options` string only to that composite transform, verifies
the resulting MLIR, and commits the clone only after the transform succeeds.
Other registered transforms do not run. The first execution atomically seals
the registry, so a later registration cannot change compilation without a
process or binary change. Enabling the hook without a registered `shuttle`
transform is a compilation error.

PJRT MLIR entry points apply only `xla_shuttle_enable` and
`xla_shuttle_options` before this hook. Other environment option overrides stay
deferred to their existing backend boundary. The patch routes the CPU,
interpreter, StreamExecutor, and deviceless GPU callers through the same narrow
wrapper while retaining the original overrides for later application,
serialization, and cache identity.

The patch does not contain the Shuttle dialect or pass pipeline. A
Shuttle-enabled jaxlib must link a registration translation unit that calls:

```cpp
xla::StablehloModuleTransformRegistry::Global().Register(
    "shuttle", [](mlir::ModuleOp module, absl::string_view options) {
      return mlir::shuttle::runShuttleXlaTransform(module, options);
    });
```

The registration library must be strongly linked. The external
`@shuttle_mlir//:ShuttleXlaRegistryAdapter` target is `alwayslink = True`, owns
canonical option-schema validation, and calls the exact shared Shuttle
pipeline. The pinned JAX composition patch in `../jax_patch` links that adapter
into the final CPU `_jax` extension without adding a Shuttle dependency to
XLA's generic registry or `mlir_to_hlo` targets.

Apply and test the patch from an exact XLA checkout:

```bash
test "$(git rev-parse HEAD)" = 9b635916ecc6df6efee62d8e4b0c7ef87ef84d69
git apply --check /path/to/0001-add-stablehlo-module-transform-hook.patch
git apply /path/to/0001-add-stablehlo-module-transform-hook.patch
git apply --check /path/to/0002-anchor-lit-labels-to-xla-repository.patch
git apply /path/to/0002-anchor-lit-labels-to-xla-repository.patch
bazel test \
  //xla/pjrt:stablehlo_module_transform_test \
  //xla/pjrt:mlir_to_hlo_test \
  //xla/pjrt:mlir_to_hlo_unregistered_transform_test \
  //xla/pjrt:pjrt_executable_test
```

The second patch makes XLA's lit macros safe for calls from an external
repository. String labels created inside a macro resolve in the caller's
repository. Constructing XLA-owned runtime labels with `Label(...)` in
`lit.bzl` anchors them to XLA. The patch covers the runner data, default config,
GPU specifications, CUDA/NCCL dependencies, and Google config while leaving
load statements and caller-supplied labels unchanged.

The internal `//third_party/py/lit:lit` string remains unchanged. In the exact
OSS source path it is overwritten by the `lit_custom_*` target before any rule
consumes it; it is not an XLA-owned runtime label in that execution mode.

The artifact is source-level integration work. Marin does not vendor XLA, and
the patch has not been compiled in this repository. End-to-end acceptance
still requires a Shuttle-enabled jaxlib running ordinary
`jax.jit(..., compiler_options=...)` forward and JAX-generated backward
programs.
