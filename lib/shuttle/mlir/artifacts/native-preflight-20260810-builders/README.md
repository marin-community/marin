# Native MLIR preflight after the Builders header correction

The CPU-only preflight passed operation generation, then failed at the narrow
`@shuttle_mlir//:ShuttleDialect` compile gate. The Builders header correction
closed the generated incomplete-type failure. Four handwritten expressions in
`ShuttleDialect.cc` call `front()` on `mlir::ArrayAttr`, whose exact pinned API
does not provide that member.

The runner stopped at that first failure. `ShuttlePasses`, `shuttle-opt`, the
MLIR lit suite, and the four patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-builders`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-builders`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `98.66` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `cce8dbc849dea7d288308cf34e5b1baa957acfa6`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- LLVM source: `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
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
completed its single action in 18.363 seconds. The dialect target then analyzed
with 1,577 configured targets. Its four distinct diagnostics were:

```text
ShuttleDialect.cc:150:54: error: no member named 'front' in 'mlir::ArrayAttr'
ShuttleDialect.cc:177:55: error: no member named 'front' in 'mlir::ArrayAttr'
ShuttleDialect.cc:254:55: error: no member named 'front' in 'mlir::ArrayAttr'
ShuttleDialect.cc:459:47: error: no member named 'front' in 'mlir::ArrayAttr'
```

Bazel reported the same four errors for the PIC and non-PIC compile actions.
This run establishes that the generated Builder incomplete-type blocker is
closed. It does not establish which exact pinned `ArrayAttr` access expression
should replace `front()`; that source change requires exact-pin review and a
later native validation.

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
before Bazel ran. `source-sha256.txt` records the 78 exact canonical files
bundled under `lib/shuttle/mlir`.

## Client proof

The approved payload-local ignored symlinks exposed the current repository
`pyproject.toml` and `config` directory for local CoreWeave store discovery. A
read-only query reached the controller and confirmed that the job did not exist
before submission.

The generated workspace archive contained 81 entries and 140,879 bytes. Its
SHA-256 was
`3cb712ff8cc4dc2d98b34b059937bb46e963f08a6f04f81a9c89c3142e6aa615`.
It had zero `config`, `pyproject.toml`, or `.git` entries. The job was submitted
once.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `e63370c01161266be60615135ba1aceb6dc7e5adc257a78265de15e08d902241`.
- `terminal-summary.txt`: controller `job summary` output after termination.
- `failure-context.txt`: operation-generation pass through the complete dialect
  failure and terminal runner lines.
- `compiler-errors.txt`: both PIC and non-PIC compiler diagnostics.
- `run_native_preflight.sh`: exact submitted runner.
- `launch-command.txt`: exact submission command.
- `manifest.env`: exact source and resource limits supplied to the runner.
- `client-proof.txt`: absolute client, store-discovery, and archive proof.
- `source-sha256.txt`: SHA-256 output for the 78 bundled canonical files.
- `toolchain.txt`: source, toolchain, output-tree, and target-state facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod, generated files, and Bazel output tree were not retained.
This checkpoint does not authorize a relaunch.
