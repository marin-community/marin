# Native MLIR preflight after the dialect-definition correction

The CPU-only preflight passed operation generation, both Shuttle library
compile gates, and the `shuttle-opt` build and link. The MLIR lit suite then
failed during Bazel analysis because a `//xla` tool label resolved within the
external `@@shuttle_mlir` repository.

No lit test executed. The runner stopped at that first failure, so the four
patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-dialectdefs`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-dialectdefs`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `640.18` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `3239a21e3f81a687e19e638fe8de5e993ca32332`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- LLVM source: `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

| Target | Result |
| --- | --- |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:ShuttleDialect` | Passed |
| `@shuttle_mlir//:ShuttlePasses` | Passed |
| `@shuttle_mlir//:shuttle-opt` | Passed build and link |
| `@shuttle_mlir//:mlir_tests` | Failed during analysis; no test ran |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

Operation generation completed in 19.790 seconds. `ShuttleDialect` completed
952 actions in 50.421 seconds, and `ShuttlePasses` completed 91 actions in
7.019 seconds. The driver configured 8,466 targets and completed 5,951 actions
in 529.397 seconds.

Lit target analysis failed after 1.538 seconds:

```text
ERROR: no such package '@@shuttle_mlir//xla': BUILD file not found in directory 'xla' of external repository @@shuttle_mlir.
ERROR: .../external/shuttle_mlir/BUILD.bazel:126:15: ... referenced by '@@shuttle_mlir//:test/verifier-errors.mlir.test'
ERROR: No test targets were found, yet testing was requested
```

This run establishes that the generated dialect-definition correction closes
the prior `shuttle-opt` link failure. It does not establish lit or XLA test
behavior.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `0d5dfb12d99384c8b58ac3f725d1d7d20757f8897bfdee65c81a043f58b31665`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `failure-context.txt`: four passing native gates through terminal failure.
- `lit-analysis-errors.txt`: driver success and complete lit analysis failure.
- `run_native_preflight.sh`, `launch-command.txt`, and `manifest.env`: exact
  submitted command and runner inputs.
- `client-proof.txt`: absolute client, store-discovery, and config-free bundle
  proof.
- `source-sha256.txt`: hashes for the 104 bundled canonical files.
- `toolchain.txt`: source, toolchain, output-tree, and gate facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The raw log records Bazel `release 7.7.0`, the exact XLA checkout, successful
patch application, and all ordered gate commands. The terminal pod and Bazel
output tree were not retained. This checkpoint does not authorize a relaunch.
