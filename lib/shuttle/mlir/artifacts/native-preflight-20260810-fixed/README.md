# Native MLIR preflight after the ReturnLike fix

The CPU-only preflight advanced past operation generation, then failed while
compiling `lib/IR/ShuttleDialect.cc` for `@shuttle_mlir//:shuttle-opt`.
The narrow `@shuttle_mlir//:shuttle_ops_inc_gen` target passed. The first C++
diagnostic reported that the generated `ShuttleOps.h.inc` referenced
`mlir::BytecodeOpInterface`, which was not declared in the translation unit.

No `shuttle-opt` binary completed. The MLIR lit suite and the four patched XLA
tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-fixed`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-fixed`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `109.93` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `bba4867bb9943c6c0553e0b910fdea8d6b83e14e`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

The target matrix stopped at the first failure:

| Target | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:shuttle-opt` | Failed compiling `ShuttleDialect.cc` |
| `@shuttle_mlir//:mlir_tests` | Not run |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

The narrow target loaded 84 packages, configured 14,124 targets, and completed
its single action. The next target analyzed after loading two more packages and
configuring 10,235 targets. Its first compiler diagnostic was:

```text
bazel-out/k8-opt/bin/external/shuttle_mlir/_virtual_includes/shuttle_ops_inc_gen/shuttle/IR/ShuttleOps.h.inc:234:282: error: no member named 'BytecodeOpInterface' in namespace 'mlir'
```

Clang then reported `using Op::Op` template errors, missing
`DialectBytecodeReader` and `DialectBytecodeWriter`, and undeclared
`getOperation` and `getProperties` references before reaching the 20-error
limit. These later diagnostics may be cascades from the first missing type.

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
before Bazel ran. `source-sha256.txt` records the 39 bundled files under
`lib/shuttle/mlir`.

## Client correction

The first local submission command exited before controller contact because the
current Iris client could not discover the repository's CoreWeave store metadata
from the minimal payload directory. No job existed from that command.

The approved correction exposed the current repository `pyproject.toml` and
`config` directory to local config discovery through payload-local ignored
symlinks. A read-only controller query then succeeded from the payload
directory. The generated workspace archive contained 43 entries and zero
`config`, `pyproject.toml`, or `.git` entries. The source bundle and runner were
unchanged.

## Evidence

- `raw-attempt.log.gz`: byte-exact controller-retained logs from submission
  timestamp `1786352412775`, compressed with `gzip -n`. Its decompressed
  SHA-256 is
  `58658fc59902db9abf9fc2e25d2f35d28aeafcbe6b7efaa25c864d7fd97b9ae2`.
- `terminal-summary.txt`: controller `job summary` output after termination.
- `compiler-errors-first-100.txt`: the first 100 log lines beginning with the
  failed C++ action.
- `generated-header-context.txt`: retained include stack and the limit on
  generated-header evidence.
- `run_native_preflight.sh`: exact submitted runner.
- `launch-command.txt`: exact submission command.
- `manifest.env`: exact source and resource limits supplied to the runner.
- `client-proof.txt`: absolute client and config-discovery proof.
- `source-sha256.txt`: remote SHA-256 output for the 39 bundled source files.
- `toolchain.txt`: source, toolchain, output-tree, and resource facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The failed pod was terminal before this checkpoint was written, so the generated
`ShuttleOps.h.inc`, XLA checkout, and Bazel output tree were not retained. This
checkpoint does not authorize a relaunch.
