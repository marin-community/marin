# Native MLIR preflight after the first lit-label correction

The CPU-only preflight applied both reviewed XLA patches, reverse-checked each
patch, verified the combined diff, and found the anchored lit runner label in
the patched source. Operation generation, both Shuttle libraries, and the
`shuttle-opt` build and link passed.

The separate `mlir_tests` build gate then failed during Bazel analysis because
the pinned lit macro exposed another caller-relative XLA label. Bazel resolved
`//xla/tsl/cuda` inside the external `@@shuttle_mlir` repository.

No lit test executed. The runner stopped at the first failure, so the four
patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-litfix`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-litfix`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `623.96` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `bb7cdd8e6b09e67a03837616c9c2e2623092ba3b`
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
| `bazel build @shuttle_mlir//:mlir_tests` | Failed during analysis |
| `bazel test @shuttle_mlir//:mlir_tests` | Not run |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

Operation generation completed in 19.677 seconds. `ShuttleDialect` completed
952 actions in 51.004 seconds, and `ShuttlePasses` completed 91 actions in
7.327 seconds. The driver configured 8,466 targets and completed 5,951 actions
in 509.650 seconds.

Lit target analysis failed after 0.668 seconds:

```text
ERROR: no such package '@@shuttle_mlir//xla/tsl/cuda': BUILD file not found in directory 'xla/tsl/cuda' of external repository @@shuttle_mlir.
ERROR: .../external/shuttle_mlir/BUILD.bazel:126:15: ... referenced by '@@shuttle_mlir//:_test/semantic-erasure-errors.mlir.test_tools_on_path'
ERROR: Analysis of target '@@shuttle_mlir//:test/semantic-erasure-errors.mlir.test' failed; build aborted: Analysis failed
```

This run establishes that patch `0002` closes the prior unconditional runner
label failure. It does not establish lit fixture or XLA test behavior. Every
string label reached by Shuttle's exact lit macro mode must be anchored before
another native run.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `44dee7ac935e7b8028f5e77234969b6304b85c447598ff0fd3555521870d3fed`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `remote-patch-proof.txt`: remote application, reverse checks, combined diff,
  and anchored-label proof for both XLA patches.
- `gate-and-failure-context.txt`: four passing native gates through the lit
  analysis failure.
- `run_native_preflight.sh`, `launch-command.txt`, and `manifest.env`: exact
  submitted command and runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client and config-free
  bundle proof.
- `local-runner-proof.txt` and `local-applied-xla.patch.gz`: pre-submit
  exact-pin runner proof and the byte-preserving compressed combined XLA diff.
- `source-sha256.txt`: hashes for the 118 bundled canonical files.
- `toolchain.txt`: source, toolchain, output-tree, and gate facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were not retained. This checkpoint does
not authorize a relaunch.
