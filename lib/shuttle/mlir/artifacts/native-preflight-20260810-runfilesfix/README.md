# Passing native Shuttle and XLA preflight

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found exactly the seven
anchored XLA-owned runtime labels. All current Shuttle build gates passed. The
uncached pipeline observer test passed, all 17 lit fixtures passed, and all
four patched XLA targets compiled and executed successfully.

This is the positive native checkpoint for canonical Marin source
`f2ef178dcc13f04c9b89b817674f7158d6add5d8`. It also closes the prior lit
runfiles failure: the three fixtures that require files under `test/Inputs`
passed from the external-repository runfiles tree.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-runfilesfix`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-runfilesfix`
- Controller state: `succeeded`
- Task exit: `0`
- Task duration: `9 minutes and 39.79 seconds`
- Failures: `0`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `f2ef178dcc13f04c9b89b817674f7158d6add5d8`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- LLVM source: `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

| Gate | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:ShuttleDialect` | Passed |
| `@shuttle_mlir//:ShuttlePasses` | Passed |
| `@shuttle_mlir//:ShuttleXlaRegistration` | Passed |
| `@shuttle_mlir//:shuttle-opt` | Passed build and link |
| `bazel build @shuttle_mlir//:pipeline_observer_test` | Passed |
| `bazel test @shuttle_mlir//:pipeline_observer_test` | 1 of 1 passed, uncached |
| `bazel build @shuttle_mlir//:mlir_tests` | Passed |
| `bazel test @shuttle_mlir//:mlir_tests` | 17 of 17 passed, uncached |
| Four patched XLA targets | 4 of 4 passed, uncached |

The four XLA targets were `stablehlo_module_transform_test`,
`mlir_to_hlo_test`, `mlir_to_hlo_unregistered_transform_test`, and
`pjrt_executable_test`. Their combined uncached gate completed 4,609 actions in
254.228 seconds.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `e45dc1cf932a8480b6e8f04d2e896a3f79f7921d80bfb8114e9e2a9eb6ce485f`.
- `terminal-summary.txt` and `controller-terminal-proof.txt`: controller
  terminal state, task result, and release proof.
- `remote-runner-and-gates.txt`: remote patch, seven-label, ordered-gate, lit,
  observer, and XLA runtime evidence.
- `run_native_preflight.sh`, `launch-command.txt`, and `manifest.env`: exact
  submitted command and runner inputs.
- `client-proof.txt`, `bundle-proof.txt`, and
  `controller-collision-proof.txt`: absolute client, external non-secret store
  metadata, config-free bundle, and no-existing-job proof.
- `local-runner-proof.txt`: pre-submit exact-pin runner proof.
- `source-sha256.txt`: hashes for the 54 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, and gate facts.
- `monitoring-state.json`: preparation, single submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were released after success. This
checkpoint records one exact validation and does not authorize a relaunch.
