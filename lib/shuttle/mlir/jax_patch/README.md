# Pinned JAX jaxlib composition overlay

This patch applies to JAX 0.10.1 commit
`619764c15117fbefc4ba13ab941871cb514c23f6`. That release pins XLA commit
`9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`.

The patch adds the always-linked `@shuttle_mlir//:ShuttleXlaRegistryAdapter`
library to the final `//jaxlib:_jax` extension. The adapter depends on XLA's
public generic StableHLO transform registry and registers the `shuttle` key at
static initialization. XLA's registry and `mlir_to_hlo` targets do not depend
on Shuttle.

The second patch contains an acceptance-only configuration selected with
`--define=SHUTTLE_TEST_OBSERVER=1`. It compiles a guarded private
`_jax._shuttle_test_observer` binding into the final `_jax` DSO and links the
external observer bridge. Without that define, the binding and bridge target
are absent. The registry adapter remains linked in both configurations.

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

Apply `0002-add-acceptance-observer-bridge.patch` after `0001` only for the
acceptance wheel. The source/query proof expects both patches and checks that
the test bridge is selected only with its build define.

For the reviewed CPU acceptance wheel, first run the guarded source/query
proof, then build a release-versioned jaxlib wheel:

```bash
git apply --check /path/to/0002-add-acceptance-observer-bridge.patch
git apply /path/to/0002-add-acceptance-observer-bridge.patch
python /path/to/verify_acceptance_patch.py \
  --bazel /path/to/bazel-7.7.0 \
  --jax-source /path/to/jax \
  --xla-source /path/to/patched/xla \
  --shuttle-mlir /path/to/marin/lib/shuttle/mlir \
  --output-user-root /path/to/bazel-output
python build/build.py build \
  --wheels=jaxlib \
  --python_version=3.12 \
  --bazel_path=/path/to/bazel-7.7.0 \
  --output_path=/path/to/dist \
  --bazel_options=--override_repository=xla=/path/to/patched/xla \
  --bazel_options=--override_repository=shuttle_mlir=/path/to/marin/lib/shuttle/mlir \
  --bazel_options=--define=SHUTTLE_TEST_OBSERVER=1 \
  --bazel_options=--repo_env=ML_WHEEL_TYPE=release
```

The expected output is one `jaxlib-0.10.1-cp312-*.whl`. Install that wheel,
the matching JAX 0.10.1 source, and `lib/shuttle` into an isolated environment,
then run `shuttle_jaxlib_acceptance.py` with an empty `--work-directory` and a
`--report` path. The checked-in driver owns the two fresh persistent-cache
workers, the cache-disabled concurrency/lifetime worker, and a cache-disabled
context-manager worker that exercises normal and exceptional native capture
teardown. The Iris runner must not replace those semantics with inline Python.

Before building, run the pinned fixture-contract regression suite explicitly:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run pytest -q \
    lib/shuttle/mlir/jax_patch/test_acceptance_contract.py \
    lib/shuttle/mlir/jax_patch/test_verify_acceptance_patch.py \
    lib/shuttle/mlir/jax_patch/test_fixture_audit_gate.py \
    lib/shuttle/mlir/jax_patch/test_bf16_composed_fixture_oracle.py
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python lib/shuttle/mlir/jax_patch/verify_acceptance_fixture_oracles.py
```

The remote preflight runs the default no-write audit through the checked-in
gate, which builds and resolves the test-only normalizer target directly:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python lib/shuttle/mlir/jax_patch/fixture_audit_gate.py \
    --bazel /path/to/bazel-7.7.0 \
    --xla-source /path/to/patched/xla \
    --output-user-root /path/to/bazel-output \
    --repository-cache /path/to/repository-cache \
    --jobs 24 \
    --ram-mb 65536 \
    --python /path/to/jax-0.10.1-python \
    --generator lib/shuttle/mlir/test/Inputs/regenerate-jax-fixtures.py
```

The gate requires the single executable output from
`@shuttle_mlir//:shuttle-test-opt`. Production `shuttle-opt` does not link the
test-only fingerprint pass and is rejected before the fixture audit starts.
The generator includes the failed tool argv, exit code, and bounded stdout and
stderr when a normalizer subprocess fails.

The four composed BF16 primal/VJP fixtures are an inventory boundary, not an
acceptance claim. Audit them separately with the same built normalizer and the
independent hook-boundary verifier:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python lib/shuttle/mlir/jax_patch/fixture_audit_gate.py \
    --bazel /path/to/bazel-7.7.0 \
    --xla-source /path/to/patched/xla \
    --output-user-root /path/to/bazel-output \
    --repository-cache /path/to/repository-cache \
    --jobs 24 \
    --ram-mb 65536 \
    --python /path/to/jax-0.10.1-python \
    --generator lib/shuttle/mlir/test/Inputs/regenerate-jax-bf16-composed-fixtures.py \
    --verifier lib/shuttle/mlir/jax_patch/verify_bf16_composed_fixture_oracle.py
```

The six BF16 row Fold fixtures cover forward, JAX-owned backward, and composed
boundaries at `2048x4096` and the structural `7x13` mutation. They are fixture
inventory only. Their oracle manifest explicitly records that the evaluation
oracle is unpinned and that no hardware evidence exists. Audit the corpus with
JAX and jaxlib 0.10.1:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python \
    lib/shuttle/mlir/test/Inputs/regenerate-jax-bf16-row-fold-scale-fixtures.py
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run python \
    lib/shuttle/mlir/jax_patch/verify_bf16_row_fold_scale_fixture_oracle.py
PYTHONPATH=lib/shuttle/mlir/jax_patch \
  uv run pytest -q \
    lib/shuttle/mlir/jax_patch/test_bf16_row_fold_scale_fixture_oracle.py
```

To refresh an intentional toolchain or source change, run the generator with
`--write`, then the verifier with `--write`, and rerun the test.

The independent verifier hashes a canonical graph inventory instead of MLIR
SSA names. The inventory includes every top-level operand, attribute, result
type, source-result reference, output anchor, and each reducer block's scalar
`stablehlo.add` result and `stablehlo.return` operation reference.

`shuttle_jaxlib_target1_acceptance.py` is the installed-wheel ABI 5 CPU
contract for those six boundaries. It runs each boundary under
`source_ordered` and `fast`, for twelve distinct compiler/cache identities. A
cache-disabled process saves disabled ordinary-JAX BF16 baselines. Separate
populate and reuse processes require exact bitwise `y`, `dx`, and `dgamma`
parity, total observer source coverage, final Shuttle erasure, twelve uniquely
attributed cache entries, and twelve public second-process cache hits. The
current pipelines round-trip these operations, so bitwise parity is required.
A future non-bitwise `fast` rewrite needs a separately reviewed tolerance
revision before any timing run.

The driver is checked locally without claiming wheel execution:

```bash
PYTHONPATH=lib/shuttle/mlir/jax_patch:lib/shuttle/src \
  uv run --project lib/shuttle --group test pytest -q \
    lib/shuttle/mlir/jax_patch/test_target1_acceptance_contract.py \
    lib/shuttle/mlir/jax_patch/test_shuttle_jaxlib_target1_acceptance.py
```

Run the installed-wheel contract only inside a reviewed Linux CPU runner:

```bash
JAX_PLATFORMS=cpu python \
  lib/shuttle/mlir/jax_patch/shuttle_jaxlib_target1_acceptance.py \
  --work-directory /fresh/target1-acceptance \
  --report /fresh/target1-acceptance.json
```

The contract is audited from the checked-in pinned forward and JAX-owned VJP
StableHLO fixtures at XLA's module-transform hook boundary. The fixture audit
reapplies the pinned StableHLO complex-math expander and rejects a changed
hook-boundary digest. This matters for the VJP: XLA moves its scalar constant
ahead of the first dot before Shuttle assigns source ordinals. The contract
requires exact normalized selected regions, the complete algebra/lowered
manifest, the VJP constant/broadcast/subtract exclusion island, phase-specific
provenance erasure, and the final normalized module fingerprint. The validator
does not select behavior from callable names or workload keys.

The repository overrides avoid checked-in machine-specific paths. A Bazel
dependency query against the exact JAX release and patched XLA revision proves
the final `_jax` target reaches the always-linked adapter. This checkpoint only
claims the CPU jaxlib composition path. Dynamically loaded GPU PJRT plugins
require their own explicit adapter linkage and registration proof.
