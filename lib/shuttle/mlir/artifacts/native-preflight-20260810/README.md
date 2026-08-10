# Native MLIR preflight negative result

The CPU-only native preflight failed while TableGen generated the Shuttle
operation declarations and definitions. The pinned MLIR TableGen environment
does not define `ReturnLike`, which `shuttle.yield` lists as a trait in
`ShuttleOps.td:69`.

Within the Shuttle target, only TableGen ran. No C++ compilation completed;
the MLIR lit suite and the four patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `92.92` seconds
- Canonical Marin source: `0481d4ef2b9f7139b53784ea3d03790d72d1699c`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- Bazel: `7.7.0`
- Bundle: `36,767` bytes
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Retries configured: `0`
- Accelerators: none requested; no device query ran

Bazel loaded 86 packages, configured 24,359 targets, and reported 2,127
processes before both `-gen-op-decls` and `-gen-op-defs` failed:

```text
external/shuttle_mlir/include/shuttle/IR/ShuttleOps.td:69:34: error: Variable not defined: 'ReturnLike'
                           Pure, ReturnLike, Terminator]> {
                                 ^
```

The target matrix stopped at the first failure:

| Target | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle-opt` | Failed during TableGen |
| `@shuttle_mlir//:mlir_tests` | Not run |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

## Toolchain

- Debian GNU/Linux 13 (`trixie`), Linux
  `6.8.12-680-6063-coreweave-amd64-f81899c8`
- GCC and G++ `14.2.0`
- GNU ld `2.44`
- Git `2.47.3`
- Python `3.12.13`; XLA selected hermetic Python `3.11`
- curl `8.14.1`
- Bazel binary SHA-256:
  `fe7e799cbc9140f986b063e06800a3d4c790525075c877d00a7112669824acbf`
- Bazel embedded OpenJDK `21.0.5`, Zulu `21.38+21-CA`
- Bazel output base:
  `/app/bazel-output-user-root/7bcae6140cd45c3d4e0ed6da28f91b54`
- Bazel execution root:
  `/app/bazel-output-user-root/7bcae6140cd45c3d4e0ed6da28f91b54/execroot/xla`

The log records `release 7.7.0` before the build command. XLA fetched the exact
revision, and `git apply --check`, `git apply`, and `git diff --check` all
succeeded before Bazel ran.

## Evidence

- `raw-attempt.log.gz`: byte-exact controller-retained raw task logs, fetched
  from the exact second-attempt submission timestamp `1786350950173` and
  compressed with `gzip -n`. Its decompressed SHA-256 is
  `9e1a752d34e66462461146d373b87bd5c68f40564e4dfd0b26583a25e50e69d1`.
- `terminal-summary.txt`: controller `job summary` output after termination.
- `run_native_preflight.sh`: exact runner submitted in the second bundle.
- `launch-command.txt`: exact submission command.
- `manifest.env`: exact source, policy, Bazel, and resource limits supplied to
  the runner.
- `source-sha256.txt`: SHA-256 for all 29 bundled files under
  `lib/shuttle/mlir`.
- `toolchain.txt`: exact source, toolchain, output-tree, and resource facts
  extracted from the runner and raw log.
- `monitoring-state.json`: job ownership and both failed-attempt records. The
  first attempt stopped before any build because command options preceded the
  Bazel subcommand; the second attempt used the locally smoke-tested ordering
  and reached TableGen.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The failed pod was terminal before this checkpoint was written, so the remote
XLA checkout and Bazel output tree were not retained. The raw log, runner,
source hashes, and controller summary are the preserved evidence.
