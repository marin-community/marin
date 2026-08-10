# Native MLIR preflight through XLA test execution

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found all seven anchored
XLA-owned runtime labels. Operation generation, both Shuttle libraries,
`shuttle-opt`, the separate `mlir_tests` build, and all 11 lit fixtures passed.

All four XLA test targets compiled and executed. Three targets passed. The
`mlir_to_hlo_test` binary passed eight of nine internal tests and failed
`MlirToHloTest.EnabledModuleTransformReceivesOptions`: its fixture defines
`@negate`, while StableHLO-to-HLO conversion requires a `@main` function. The
subsequent `forwarded_options` failure is secondary because conversion stopped
before the transform could receive the options. The runner stopped at this
first failed gate.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-statusmacros`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-statusmacros`
- Controller state: `failed`
- Task exit: `3`
- Task duration: `13 minutes and 45.7 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `8eb01889a10bcb9f906741afd67df92fe34de69d`
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

Operation generation completed in 21.893 seconds. `ShuttleDialect` completed
952 actions in 51.255 seconds, and `ShuttlePasses` completed 91 actions in
7.035 seconds. The driver completed 5,951 actions in 544.415 seconds. The
separate lit build completed 143 actions in 11.191 seconds. Lit execution took
2.569 seconds and passed all 11 tests.

The four-test XLA gate compiled and executed every target in 152.121 seconds
and 3,432 actions. The sole failing internal test reported:

```text
xla/pjrt/mlir_to_hlo_test.cc:106: Failure
Value of: MlirToXlaComputationWithPjRtOptions(...)
Expected: is OK
  Actual: UNKNOWN: error: conversion requires module with `main` function
...
"func.func"() <{..., sym_name = "negate"}> ({
...
xla/pjrt/mlir_to_hlo_test.cc:110: Failure
Value of: forwarded_options
  Actual: false
Expected: true
```

The other eight `MlirToHloTest` cases passed. This run establishes native
compilation and execution of the patched registry, unregistered-transform,
PJRT executable, and MLIR-to-HLO test targets against the exact pins. It does
not establish a fully passing four-target XLA gate.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `168153032174a61a9876de2e1fe5065f087c85def0503f6fa3ff69a93eadcb3c`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `remote-runner-and-gates.txt`: remote patch, seven-label, ordered-gate, lit,
  and XLA runtime evidence.
- `run_native_preflight.sh`, `launch-command.txt`, and `manifest.env`: exact
  submitted command and runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client, external
  non-secret store metadata, and config-free bundle proof.
- `local-runner-proof.txt` and `local-applied-xla.patch.gz`: pre-submit
  exact-pin runner proof and the byte-preserving compressed combined XLA diff.
- `source-sha256.txt`: hashes for the 30 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, and gate facts.
- `monitoring-state.json`: preparation, submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were not retained. This checkpoint does
not authorize a relaunch.
