# Pinned JAX jaxlib composition overlay

This patch applies to JAX 0.10.1 commit
`619764c15117fbefc4ba13ab941871cb514c23f6`. That release pins XLA commit
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`.

The patch adds the always-linked `@shuttle_mlir//:ShuttleXlaRegistryAdapter`
library to the final `//jaxlib:_jax` extension. The adapter depends on XLA's
public generic StableHLO transform registry and registers the `shuttle` key at
static initialization. XLA's registry and `mlir_to_hlo` targets do not depend
on Shuttle.

Apply the XLA patches from `../xla_patch` to an exact pinned XLA checkout, then
build in JAX's default WORKSPACE mode with both local repositories injected at
the command line:

```bash
test "$(git rev-parse HEAD)" = 619764c15117fbefc4ba13ab941871cb514c23f6
git apply --check /path/to/0001-link-shuttle-xla-registry-adapter.patch
git apply /path/to/0001-link-shuttle-xla-registry-adapter.patch
bazel build \
  --override_repository=xla=/path/to/patched/xla \
  --override_repository=shuttle_mlir=/path/to/marin/lib/shuttle/mlir \
  //jaxlib:_jax
```

The repository overrides avoid checked-in machine-specific paths. A Bazel
dependency query against the exact JAX release and patched XLA revision proves
the final `_jax` target reaches the always-linked adapter. This checkpoint only
claims the CPU jaxlib composition path. Dynamically loaded GPU PJRT plugins
require their own explicit adapter linkage and registration proof.
