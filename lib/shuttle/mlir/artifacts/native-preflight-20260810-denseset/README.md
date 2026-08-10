# Native MLIR preflight after the DenseSet correction

The CPU-only preflight passed operation generation, then failed at the narrow
`@shuttle_mlir//:ShuttleDialect` compile gate. Generated attribute and operation
definitions instantiated MLIR builder classes while only forward declarations
were visible in `ShuttleDialect.cc`.

The runner stopped at that first failure. `ShuttlePasses`, `shuttle-opt`, the
MLIR lit suite, and the four patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-denseset`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-denseset`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `98.53` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `d245bc23c181beb08c4b865044ab0d8aaf1279b0`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

The target matrix stopped at the first failure:

| Target | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:ShuttleDialect` | Failed compiling `ShuttleDialect.cc` |
| `@shuttle_mlir//:ShuttlePasses` | Not run |
| `@shuttle_mlir//:shuttle-opt` | Not run |
| `@shuttle_mlir//:mlir_tests` | Not run |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

The TableGen target loaded 84 packages, configured 14,124 targets, and
completed its single action in 18.586 seconds. The dialect target then analyzed
with 1,577 configured targets. Its first compiler diagnostic was:

```text
bazel-out/k8-opt/bin/external/shuttle_mlir/_virtual_includes/shuttle_attrs_inc_gen/shuttle/IR/ShuttleAttrs.cc.inc:84:19: error: variable has incomplete type '::mlir::Builder'
   84 |   ::mlir::Builder odsBuilder(odsParser.getContext());
      |                   ^
```

The same generated attribute file failed at lines 114, 167, and 224. Generated
operation definitions then reported incomplete `mlir::Builder`,
`mlir::OpBuilder`, and `mlir::ImplicitLocOpBuilder` types. The visible headers
provided only forward declarations from `AffineMap.h`, `Dialect.h`, and
`OpDefinition.h`. Clang stopped at its 20-error limit.

This run reached semantic C++ parsing after the DenseSet include correction.
It does not establish that adding any particular MLIR builder header is
sufficient; that source change requires exact-pin review and a later native
validation.

## Toolchain and source proof

- Debian GNU/Linux 13 (`trixie`), Linux
  `6.8.12-680-6063-coreweave-amd64-f81899c8`
- GCC and G++ `14.2.0`; GNU ld `2.44`
- Git `2.47.3`; Python `3.12.13`; XLA hermetic Python `3.11`
- curl `8.14.1`
- Bazel binary SHA-256:
  `fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf`
- Bazel embedded OpenJDK `21.0.5`, Zulu `21.38+21-CA`
- Bazel output base:
  `/app/bazel-output-user-root/7bcae6140cd45c3d4e0ed6da28f91b54`
- Bazel execution root:
  `/app/bazel-output-user-root/7bcae6140cd45c3d4e0ed6da28f91b54/execroot/xla`

The raw log records `release 7.7.0` before either target. XLA fetched the exact
revision, and `git apply --check`, `git apply`, and `git diff --check` succeeded
before Bazel ran. `source-sha256.txt` records the 65 exact canonical files
bundled under `lib/shuttle/mlir`.

## Client proof

The already-approved payload-local ignored symlinks exposed the current
repository `pyproject.toml` and `config` directory for local CoreWeave store
discovery. A read-only query reached the controller and confirmed that the job
did not exist before submission.

The generated workspace archive contained 68 entries and 109,510 bytes. It had
zero `config`, `pyproject.toml`, or `.git` entries. The job was submitted once.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `c9c8112e665c964709093dca9341bdad31e15767968ff9d733d391a19d1284b1`.
- `terminal-summary.txt`: controller `job summary` output after termination.
- `failure-context.txt`: TableGen pass through the complete 20-error dialect
  failure and terminal runner lines.
- `compiler-errors-first-100.txt`: first 100 lines beginning with the failed C++
  action.
- `run_native_preflight.sh`: exact submitted runner.
- `launch-command.txt`: exact submission command.
- `manifest.env`: exact source and resource limits supplied to the runner.
- `client-proof.txt`: absolute client, store-discovery, and archive proof.
- `source-sha256.txt`: SHA-256 output for the 65 bundled canonical files.
- `toolchain.txt`: source, toolchain, output-tree, and target-state facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod, generated files, and Bazel output tree were not retained.
This checkpoint does not authorize a relaunch.
