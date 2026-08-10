# Native MLIR preflight through XLA options-forwarding execution

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found all seven anchored
XLA-owned runtime labels. Operation generation, both Shuttle libraries,
`shuttle-opt`, the separate `mlir_tests` build, and all 11 lit fixtures passed.

All four XLA test targets compiled and executed. Three targets passed. The
`mlir_to_hlo_test` binary passed eight of nine internal tests and failed
`MlirToHloTest.EnabledModuleTransformReceivesOptions`. Conversion succeeded,
but the post-call lookup of the `test.forwarded_options` module attribute at
`xla/pjrt/mlir_to_hlo_test.cc:110` returned false. The runner stopped at this
first failed gate.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-mainfixture`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-mainfixture`
- Controller state: `failed`
- Task exit: `3`
- Task duration: `13 minutes and 58.55 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `b1e79dd7be5eb8b59fb1b2d02ad3bcdd7a8f4cbe`
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
| `bazel build @shuttle_mlir//:mlir_tests` | Passed |
| `bazel test @shuttle_mlir//:mlir_tests` | 11 passed |
| `//xla/pjrt:stablehlo_module_transform_test` | Passed |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Passed |
| `//xla/pjrt:pjrt_executable_test` | Passed |
| `//xla/pjrt:mlir_to_hlo_test` | Failed; 8 of 9 internal tests passed |

Operation generation completed in 31.250 seconds. `ShuttleDialect` completed
952 actions in 50.226 seconds, and `ShuttlePasses` completed 91 actions in
7.191 seconds. The driver completed 5,951 actions in 540.501 seconds. The
separate lit build completed 143 actions in 16.491 seconds. Lit execution took
2.868 seconds and passed all 11 tests.

The four-test XLA gate compiled and executed every target in 150.214 seconds
and 3,432 actions. The sole failing internal test reported:

```text
[ RUN      ] MlirToHloTest.EnabledModuleTransformReceivesOptions
xla/pjrt/mlir_to_hlo_test.cc:110: Failure
Value of: forwarded_options
  Actual: false
Expected: true
[  FAILED  ] MlirToHloTest.EnabledModuleTransformReceivesOptions (4 ms)
```

The preceding `TF_EXPECT_OK(MlirToXlaComputationWithPjRtOptions(...))` emitted
no failure. The `@main` fixture therefore closed the prior importer failure.
This run does not establish that the transform's options are observable on the
original module after the public conversion call.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `c576af5f2b163b7cdc89579449749cd5e94505b3ae5a65fdb9090ca515bb3b66`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `remote-runner-and-gates.txt`: remote patch, seven-label, ordered-gate, lit,
  and XLA runtime evidence.
- `run_native_preflight.sh`, `launch-command.txt`, and `manifest.env`: exact
  submitted command and runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client, external
  non-secret store metadata, and config-free bundle proof.
- `local-runner-proof.txt`: pre-submit exact-pin runner proof.
- `source-sha256.txt`: hashes for the 30 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, and gate facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were not retained. This checkpoint does
not authorize a relaunch.
