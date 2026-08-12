# Shuttle ABI 5 CPU ordinary-JAX acceptance plan

This plan separates local source checks from the external Iris upload needed to
build and run the release jaxlib wheel on Linux x86-64. It does not authorize an
upload or job submission. The ABI 5 Marin commit, task-image digest, dependency
lock, and Target 1 wheel-path contract do not exist yet.

## Existing workflow

The reusable source interfaces are:

- `lib/shuttle/mlir/xla_patch/`: the pinned XLA hook patches and revision;
- `lib/shuttle/mlir/jax_patch/`: the pinned JAX composition patches, source and
  query verifier, fixture gates, observer contract, and CPU acceptance driver;
- `lib/shuttle/mlir/jax_patch/shuttle_jaxlib_acceptance.py`: the installed-wheel
  process, cache, observer-lifetime, and bitwise tanh-dot checks; and
- `lib/iris/config/cw-us-east-02a.yaml`: the external CoreWeave Iris destination
  used by the previous CPU run.

`lib/shuttle/mlir/artifacts/native-preflight-20260810-jaxacceptance6/` is sealed
historical evidence. Its runner and manifest pin Marin revision
`6340706df454d699124fc7b676499f5db9cccd4a` and the source and test counts from
that revision. Do not edit, copy over, or relaunch that artifact as ABI 5
evidence. A new run needs a new reviewed runner, manifest, source capsule, and
artifact directory.

The fixed upstream source pins remain:

| Component | Identity |
| --- | --- |
| JAX and jaxlib | `0.10.1`, JAX `619764c15117fbefc4ba13ab941871cb514c23f6` |
| XLA | `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` |
| StableHLO | `806a6844dfd92cca1ce5391c86dca0ef9e952550` |
| LLVM | `9a4faee1068c09efbf837cfb7b0f5693b24635f4` |
| nanobind | `30f12ae6650ecec86042053d522d9af585f269b0` |
| Bazel | `7.7.0` |

The StableHLO, LLVM, and nanobind identities are transitive pins in the pinned
XLA checkout. Record them in the new artifact even though the runner does not
select them independently.

## Local preflight

Run these steps from a clean detached worktree at the exact reviewed ABI 5
commit. They do not submit a job or upload the worktree.

1. Require an empty `git status --short`, confirm the full commit SHA, and
   verify pipeline ABI 5 agrees in Python, C++, observer fixtures, and registry
   tests. Derive the source-capsule file count and hashes from that commit; do
   not copy counts from `jaxacceptance6`.
2. Apply both XLA patches and both JAX patches to fresh checkouts at the pinned
   commits. Run `git apply --check`, reverse-check each applied patch, and run
   `git diff --check` in both patched trees.
3. Run the source-level contracts and fixture oracles:

   ```bash
   PYTHONPATH=lib/shuttle/mlir/jax_patch \
     uv run --project lib/shuttle --group test pytest -q \
       lib/shuttle/mlir/jax_patch/test_acceptance_contract.py \
       lib/shuttle/mlir/jax_patch/test_verify_acceptance_patch.py \
       lib/shuttle/mlir/jax_patch/test_fixture_audit_gate.py \
       lib/shuttle/mlir/jax_patch/test_bf16_composed_fixture_oracle.py \
       lib/shuttle/mlir/jax_patch/test_bf16_row_fold_scale_fixture_oracle.py
   PYTHONPATH=lib/shuttle/mlir/jax_patch \
     uv run --project lib/shuttle --group test python \
       lib/shuttle/mlir/jax_patch/verify_acceptance_fixture_oracles.py
   PYTHONPATH=lib/shuttle/mlir/jax_patch \
     uv run --project lib/shuttle --group test python \
       lib/shuttle/mlir/test/Inputs/regenerate-jax-bf16-row-fold-scale-fixtures.py
   PYTHONPATH=lib/shuttle/mlir/jax_patch \
     uv run --project lib/shuttle --group test python \
       lib/shuttle/mlir/jax_patch/verify_bf16_row_fold_scale_fixture_oracle.py
   ```

4. With the pinned XLA checkout and a locally available Bazel 7.7.0 binary,
   run the Shuttle build gates, fixture audit through the resolved
   `shuttle-test-opt`, native tests, lit suite, four patched XLA tests, and the
   configured JAX dependency query. Register the exact reviewed worktree in the
   fresh XLA workspace before running the fixture gate, whose Bazel calls do not
   accept a repository override:

   ```bash
   if test -f /path/to/patched/xla/WORKSPACE.bazel; then
     XLA_WORKSPACE=/path/to/patched/xla/WORKSPACE.bazel
   else
     XLA_WORKSPACE=/path/to/patched/xla/WORKSPACE
   fi
   printf '\nlocal_repository(\n    name = "shuttle_mlir",\n    path = "/path/to/abi5/lib/shuttle/mlir",\n)\n' \
     >> "${XLA_WORKSPACE}"
   ```

   Then run the native targets:

   ```bash
   cd /path/to/patched/xla
   /path/to/bazel-7.7.0 --output_user_root=/fresh/xla-output build \
     --override_repository=shuttle_mlir=/path/to/abi5/lib/shuttle/mlir \
     --repository_cache=/fresh/repository-cache \
     @shuttle_mlir//:shuttle_ops_inc_gen \
     @shuttle_mlir//:ShuttleDialect \
     @shuttle_mlir//:ShuttlePasses \
     @shuttle_mlir//:ShuttleXlaRegistration \
     @shuttle_mlir//:ShuttleXlaRegistryAdapter \
     @shuttle_mlir//:ShuttleObserverTestBridge \
     @shuttle_mlir//:ShuttlePythonObserverTestBridge \
     @shuttle_mlir//:shuttle-opt
   /path/to/bazel-7.7.0 --output_user_root=/fresh/xla-output test \
     --override_repository=shuttle_mlir=/path/to/abi5/lib/shuttle/mlir \
     --repository_cache=/fresh/repository-cache \
     --cache_test_results=no \
     --test_output=errors \
     @shuttle_mlir//:mlir_tests \
     @shuttle_mlir//:pipeline_observer_test \
     @shuttle_mlir//:observer_test_bridge_test \
     @shuttle_mlir//:xla_registration_test \
     @shuttle_mlir//:xla_registry_adapter_test \
     //xla/pjrt:stablehlo_module_transform_test \
     //xla/pjrt:mlir_to_hlo_test \
     //xla/pjrt:mlir_to_hlo_unregistered_transform_test \
     //xla/pjrt:pjrt_executable_test
   ```

   Run the default and composed fixture gates exactly:

   ```bash
   PYTHONPATH=/path/to/abi5/lib/shuttle/mlir/jax_patch \
     /path/to/jax-0.10.1-python \
       /path/to/abi5/lib/shuttle/mlir/jax_patch/fixture_audit_gate.py \
       --bazel /path/to/bazel-7.7.0 \
       --xla-source /path/to/patched/xla \
       --output-user-root /fresh/xla-output \
       --repository-cache /fresh/repository-cache \
       --jobs 24 \
       --ram-mb 65536 \
       --python /path/to/jax-0.10.1-python \
       --generator /path/to/abi5/lib/shuttle/mlir/test/Inputs/regenerate-jax-fixtures.py
   PYTHONPATH=/path/to/abi5/lib/shuttle/mlir/jax_patch \
     /path/to/jax-0.10.1-python \
       /path/to/abi5/lib/shuttle/mlir/jax_patch/fixture_audit_gate.py \
       --bazel /path/to/bazel-7.7.0 \
       --xla-source /path/to/patched/xla \
       --output-user-root /fresh/xla-output \
       --repository-cache /fresh/repository-cache \
       --jobs 24 \
       --ram-mb 65536 \
       --python /path/to/jax-0.10.1-python \
       --generator /path/to/abi5/lib/shuttle/mlir/test/Inputs/regenerate-jax-bf16-composed-fixtures.py \
       --verifier /path/to/abi5/lib/shuttle/mlir/jax_patch/verify_bf16_composed_fixture_oracle.py
   ```

   Do not pass the row-Fold generator or verifier to that gate: neither accepts
   its unconditional `--normalizer` argument. Their supported direct commands
   are in step 3. Run the configured JAX proof from its checkout:

   ```bash
   cd /path/to/patched/jax
   HERMETIC_PYTHON_VERSION=3.12 \
     PYTHONPATH=/path/to/abi5/lib/shuttle/mlir/jax_patch \
     /path/to/jax-0.10.1-python \
       /path/to/abi5/lib/shuttle/mlir/jax_patch/verify_acceptance_patch.py \
       --bazel /path/to/bazel-7.7.0 \
       --jax-source /path/to/patched/jax \
       --xla-source /path/to/patched/xla \
       --shuttle-mlir /path/to/abi5/lib/shuttle/mlir \
       --output-user-root /fresh/jax-output
   ```

   A macOS build is useful source preflight, but it does not replace the Linux
   x86-64 wheel run.
5. Create the upload candidate in a fresh capsule directory. Include only the
   reviewed ABI 5 `lib/shuttle` source needed by the runner, the new runner, and
   its manifest. Exclude `.git`, `.venv`, Python and pytest caches, existing
   artifacts, repository and Iris configs, `coreweave.yaml`, credentials, and
   unrelated Marin packages. Record each ZIP member's relative path, type, mode,
   byte count, and SHA-256 in a canonical ZIP manifest. Also record an expected
   extraction manifest containing path, type, byte count, and SHA-256; Python
   `zipfile.extractall` does not preserve the stored Unix mode. Record the
   capsule byte count and SHA-256 after creating the same zip Iris will submit
   with `iris.cluster.client.bundle.create_workspace_zip`. Make the reviewed
   capsule immutable between that check and submission.

The preflight stops if the canonical commit is unavailable, the worktree is
dirty, a patch or oracle drifts, the capsule allowlist changes without review,
or the capsule manifest does not reproduce from the exact commit.

## Linux wheel and acceptance boundary

The exact acceptance run requires Linux x86-64. It downloads Bazel, fresh pinned
XLA and JAX checkouts, Python wheels, and Bazel repositories; builds the patched
release `jaxlib-0.10.1-cp312-*-manylinux_*.whl`; installs JAX, jaxlib, and
Shuttle into an isolated environment; and runs the checked-in acceptance
driver with `JAX_PLATFORMS=cpu`. No GPU is requested or inspected.

The reviewed submission must originate inside the capsule directory. Use an
exact reviewed Iris client checkout and record its commit and config digest.
Iris always creates and sends a workspace bundle from the current directory.
`--no-sync` disables task environment setup; it does not disable the upload.
The controller receives `bundle_blob`, writes it to its configured external
bundle store at
`s3://marin-us-east-02a/iris/cw-us-east-02a/state/bundles`, and stages it into
the CoreWeave task.

The current CLI submission shape below is not launch-ready. `--max-retries 0`
sets failure and task-failure limits, but its client leaves the preemption retry
limit at 1,000. `--task-image` overrides the main task container, while the
bundle-fetch init container still uses the cluster default
`ghcr.io/marin-community/iris-task:latest`. Use no submission command until a
reviewed mechanism sets every retry limit to zero and pins both containers by
OCI digest. The submit environment must also reject `.marin.yaml`, unset
`HF_TOKEN`, `WANDB_API_KEY`, `GCS_RESOLVE_REFRESH_SECS`, and inherited
`MARIN_PROVENANCE`, and record the exact non-secret environment sent by Iris.
Iris still generates `MARIN_PROVENANCE`, which contains submitter, timestamp,
and command metadata and must be included in the disclosure review. The
remaining resource shape is:

```bash
env -u HF_TOKEN -u WANDB_API_KEY -u GCS_RESOLVE_REFRESH_SECS -u MARIN_PROVENANCE \
  /path/to/reviewed-iris/.venv/bin/iris \
  --config /path/to/reviewed-iris/lib/iris/config/cw-us-east-02a.yaml \
  job run \
  --no-wait \
  --no-sync \
  --enable-extra-resources \
  --cpu 24 \
  --memory 96GB \
  --disk 250GB \
  --max-retries 0 \
  --timeout 14400 \
  --priority interactive \
  --task-image '<reviewed-cpu-image>:<immutable-tag>@sha256:<digest>' \
  --job-name '<unique-abi5-job-name>' \
  -- bash run_jax_acceptance_preflight.sh
```

Do not run that command until the capsule manifest, bundle ZIP digest, exact
destination, resource request, task and init image digests, every zero retry
limit, submitted environment, and command have independent review and the
trusted user has authorized the upload and job. A minimal authorization, after
those blockers close, is:

> I authorize uploading ABI 5 Shuttle capsule `<zip-sha256>` with reviewed file
> manifest `<manifest-sha256>` and standard Iris launch provenance metadata to
> Iris/CoreWeave `cw-us-east-02a` for one CPU-only acceptance job with 24 CPUs,
> 96 GiB memory, 250 GiB disk, a four-hour timeout, and zero failure, task-failure,
> or preemption retries. Send no tokens or `.marin.yaml` environment. Use only
> reviewed task and init image digests. Do not upload any other repository files
> or request a GPU.

After submission, monitor the single job through terminal state. Preserve the
complete task log, terminal job/task descriptions and events, acceptance JSON,
cache-key attribution, wheel filename and SHA-256, capsule and per-source-file
hashes, runner and manifest hashes, exact Iris client and config identities,
task and init image digests, dependency lock, compiler/tool versions, and every
built or tested target. A successful CPU artifact is architecture and CPU
numerical evidence only. It is not GPU linkage, performance, oracle, or Target
1 acceptance evidence.

The runner must print the task's `IRIS_BUNDLE_ID` and recompute the extracted
member inventory before any build. After submission, require that content ID to
equal the pre-reviewed ZIP SHA-256, the stored central-directory inventory to
equal the pre-reviewed ZIP manifest, and the extracted path/type/size/hash
inventory to equal the expected extraction manifest. A mismatch invalidates the
run even if all compiler tests pass.

## ABI 5 gaps to close before submission

The historical runner is not fully reproducible. Its Bazel version is pinned,
but the downloaded binary digest is recorded after download instead of checked
against an expected digest. It upgrades `pip`, `setuptools`, and `wheel` without
versions and installs ranges for NumPy, SciPy, `ml_dtypes`, `opt_einsum`, and
pytest. It also relies on the cluster's default task image. The ABI 5 runner
must assert the Bazel binary SHA-256, use a reviewed hash-pinned dependency lock,
pin the Python 3.12 patch version and main and init container images by OCI
digest, set failure, task-failure, and preemption retry ceilings explicitly to
zero, and record the resolved OS and compiler identities. The lock must also
cover build-isolated requirements: JAX source requests unpinned `setuptools` and
`wheel`, while Shuttle requests `uv_build>=0.7.19,<0.10.0`. Preinstall exact
hashed build backends and disable build isolation, or apply an equivalently
reviewed build constraint that proves those resolutions.

`shuttle_jaxlib_acceptance.py` currently exercises only the f32 tanh-dot forward
and JAX-owned VJP fixtures. Rebuilding it at ABI 5 proves the CPU `_jax`
composition, observer, teardown, and cache identity paths. It does not execute
the mapped-singleton rowwise-normalization operations that caused the ABI bump.
Before using the run as Target 1 evidence, add a reviewed ordinary-JAX
installed-wheel contract for the forward, JAX-owned backward, and composed BF16
boundaries at `2048x4096` and `7x13`, under `SOURCE_ORDERED` and `FAST`. That
contract must observe total source coverage, final Shuttle erasure, fresh ABI 5
cache identities, and `y`, `dx`, and `dgamma` parity under separately reviewed
numerical limits. Fixture generation or `shuttle-opt` round-tripping alone does
not close this wheel-path gap.

Background-research effort: medium. The local audit used canonical
`c9e5f0734c968b70195d6836f2239ef61d9a2934`, the sealed `jaxacceptance6`
artifact, current JAX/XLA overlay code, and the Iris bundle submission path.
