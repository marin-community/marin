# Native MLIR preflight after the bytecode interface fix

The CPU-only preflight passed operation generation, then failed at the narrow
`@shuttle_mlir//:ShuttleDialect` compile gate. Clang could not find
`llvm/ADT/SmallDenseSet.h`, included by `lib/IR/ShuttleDialect.cc`.

The runner stopped at that first failure. `shuttle-opt`, the MLIR lit suite,
and the four patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-bytecode`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-bytecode`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `88.15` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `412928d5aee2325d178b6a0efd1eb8383e46c7c6`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

The target matrix stopped at the first failure:

| Target | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:ShuttleDialect` | Failed compiling `ShuttleDialect.cc` |
| `@shuttle_mlir//:shuttle-opt` | Not run |
| `@shuttle_mlir//:mlir_tests` | Not run |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

The TableGen target loaded 84 packages, configured 14,124 targets, and
completed its single action in 19.101 seconds. The dialect target then analyzed
with 1,577 configured targets. Its first and only compiler diagnostic was:

```text
external/shuttle_mlir/lib/IR/ShuttleDialect.cc:18:10: fatal error: 'llvm/ADT/SmallDenseSet.h' file not found
   18 | #include "llvm/ADT/SmallDenseSet.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~
```

XLA pins LLVM to `9a4faee1068c09efbf837cfb7b0f5693b24635f4`.
At that revision, the `llvm/include/llvm/ADT` directory has no
`SmallDenseSet.h`; `llvm/ADT/DenseSet.h` declares `llvm::SmallDenseSet`.
`include-audit.txt` records the exact-pin source evidence. This artifact does
not claim that replacing the include compiles.

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
before Bazel ran. `source-sha256.txt` records the 52 exact canonical files
bundled under `lib/shuttle/mlir`.

## Client proof

The first read-only pre-submit query exited before controller contact because
the current Iris client could not discover CoreWeave store metadata from the
minimal payload directory. The already-approved correction exposed the current
repository `pyproject.toml` and `config` directory through payload-local ignored
symlinks. The corrected query reached the controller and confirmed that the
job did not exist.

The generated workspace archive contained 55 entries and 81,914 bytes. It had
zero `config`, `pyproject.toml`, or `.git` entries. The job was submitted once;
the source bundle and runner were unchanged.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `87ce9d54bdcebbe3c203921775102dac9f58dee00831faedaa0a9d62f41438ea`.
- `terminal-summary.txt`: controller `job summary` output after termination.
- `failure-context.txt`: TableGen pass, dialect invocation, and complete
  compiler failure context from the raw log.
- `include-audit.txt`: exact pinned LLVM header-name and declaration evidence.
- `run_native_preflight.sh`: exact submitted runner.
- `launch-command.txt`: exact submission command.
- `manifest.env`: exact source and resource limits supplied to the runner.
- `client-proof.txt`: absolute client, store-discovery, and archive proof.
- `source-sha256.txt`: SHA-256 output for the 52 bundled canonical files.
- `toolchain.txt`: source, toolchain, output-tree, and target-state facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were not retained. This checkpoint does
not authorize a relaunch.
