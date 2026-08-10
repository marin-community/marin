# Native MLIR current-tree observer preflight

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found all seven anchored
XLA-owned runtime labels. Operation generation, `ShuttleDialect`,
`ShuttlePasses`, `ShuttleXlaRegistration`, `shuttle-opt`, and the standalone
pipeline observer test built successfully. The uncached observer test passed.

The separate `mlir_tests` build also passed. Its uncached execution ran all 17
tests: 14 passed and three failed because the JAX MLIR fixtures under
`test/Inputs/` were absent from the tests' Bazel runfiles. The fixtures were
present in the submitted 54-file source bundle. The `lit_test_suite` excludes
`test/Inputs/**` from `srcs` and does not declare those files as runtime data,
while the three failed fixtures reference them through `%S/Inputs/...`.

The runner stopped at that first failed gate. None of the four patched XLA test
targets ran.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-observerhook`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-observerhook`
- Controller state: `failed`
- Task exit: `3`
- Task duration: `5 minutes and 1.06 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `b120bf9ce86e98341c687c0b97f4837f5e7e46b6`
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
| `@shuttle_mlir//:ShuttleXlaRegistration` | Passed |
| `@shuttle_mlir//:shuttle-opt` | Passed build and link |
| `@shuttle_mlir//:pipeline_observer_test` build | Passed |
| `@shuttle_mlir//:pipeline_observer_test` execution | Passed uncached, 1 of 1 |
| `bazel build @shuttle_mlir//:mlir_tests` | Passed; 17 targets |
| `bazel test @shuttle_mlir//:mlir_tests` | Failed uncached; 14 of 17 passed |
| Four patched XLA targets | Not run after the first failed gate |

The three failed MLIR tests were:

- `coverage-mutation-errors.mlir`
- `vertical-slice-algebra.mlir`
- `vertical-slice-lowering.mlir`

Each reported the same runfiles condition. One representative diagnostic was:

```text
cannot open input file '.../external/shuttle_mlir/test/Inputs/jax-0.10.1-tanh-dot-vjp.mlir': No such file or directory
```

The run does not evaluate the semantic assertions in those three tests or any
of the four XLA tests. It does establish native compilation and execution for
the current observer and registration targets.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `86c8540b78106f9f7e2c41c8b815596b2de74ad5050a9277c3a5957f0015d0ec`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `remote-runner-and-gates.txt`: remote patch, seven-label, ordered-gate,
  observer, lit, and stop-boundary evidence.
- `run_native_preflight.sh`, `launch-command.txt`, and `manifest.env`: exact
  submitted command and runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client, external
  non-secret store metadata, and config-free bundle proof.
- `local-runner-proof.txt`: pre-submit exact-pin runner proof.
- `source-sha256.txt`: hashes for all 54 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, and gate facts.
- `monitoring-state.json`: preparation, single submission, and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were not retained. This checkpoint does
not authorize a relaunch.
