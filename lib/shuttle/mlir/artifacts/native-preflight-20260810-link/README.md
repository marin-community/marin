# Native MLIR preflight after the ArrayAttr correction

The CPU-only preflight passed operation generation and both explicit Shuttle
library compile gates. The `@shuttle_mlir//:shuttle-opt` target compiled, then
failed while linking because the Shuttle dialect constructor and type ID were
undefined.

The runner stopped at that first failure. The MLIR lit suite and the four
patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-arrayattr`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-arrayattr`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `651.05` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `a7c10fa6941c7a53efcfb59d866fbdc827c29ff0`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- LLVM source: `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

The target matrix stopped at the first failure:

| Target | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:ShuttleDialect` | Passed |
| `@shuttle_mlir//:ShuttlePasses` | Passed |
| `@shuttle_mlir//:shuttle-opt` | Failed linking the driver |
| `@shuttle_mlir//:mlir_tests` | Not run |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

Operation generation loaded 84 packages, configured 14,124 targets, and
completed its single action in 18.256 seconds. `ShuttleDialect` configured
1,577 targets and completed 952 actions in 47.774 seconds. `ShuttlePasses`
configured 192 targets and completed 91 actions in 5.655 seconds.

The driver configured 8,466 targets and ran 5,951 processes before the link
failed after 541.934 seconds. The retained linker diagnostics are:

```text
ld.lld: error: undefined symbol: mlir::detail::TypeIDResolver<mlir::shuttle::ShuttleDialect, void>::id
ld.lld: error: undefined symbol: mlir::shuttle::ShuttleDialect::ShuttleDialect(mlir::MLIRContext*)
```

References came from `shuttle-opt.cc` and `Passes.cc`. This run establishes
that the four corrected `ArrayAttr` accesses compile in the dialect library and
that the pass library compiles. It does not establish that the driver links or
that lit and XLA behavior pass.

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

The raw log records `release 7.7.0` before any build target. XLA fetched the
exact revision, and `git apply --check`, `git apply`, and `git diff --check`
succeeded before Bazel ran. `source-sha256.txt` records the 91 exact canonical
files bundled under `lib/shuttle/mlir`.

## Client proof

The approved payload-local ignored symlinks exposed the current repository
`pyproject.toml` and `config` directory for local CoreWeave store discovery. A
read-only query reached the controller and confirmed that the job did not exist
before submission.

The generated workspace archive contained 94 entries and 171,122 bytes. Its
SHA-256 was
`b1cb39bf9ca3616f54996172696bd30dd4587d377536bae5e6302ba6e83fec44`.
It had zero `config`, `pyproject.toml`, or `.git` entries. The job was submitted
once.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `db72010d2e4e40c56f1b74acf6763dfd7a485ac891680a09028a9e3e996cdc43`.
- `terminal-summary.txt`: controller `job summary` output after termination.
- `failure-context.txt`: all four ordered native gates through the terminal
  runner lines.
- `linker-errors.txt`: retained linker diagnostic and target timing.
- `run_native_preflight.sh`: exact submitted runner.
- `launch-command.txt`: exact submission command.
- `manifest.env`: exact source and resource limits supplied to the runner.
- `client-proof.txt`: absolute client, store-discovery, and archive proof.
- `source-sha256.txt`: SHA-256 output for the 91 bundled canonical files.
- `toolchain.txt`: source, toolchain, output-tree, and target-state facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod, generated files, and Bazel output tree were not retained.
This checkpoint does not authorize a relaunch.
