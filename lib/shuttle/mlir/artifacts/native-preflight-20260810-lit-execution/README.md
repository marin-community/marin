# Native MLIR preflight through lit execution

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found all seven anchored
XLA-owned runtime labels. Operation generation, both Shuttle libraries,
`shuttle-opt`, and the separate `mlir_tests` build passed.

The lit suite then executed all 11 fixtures. Eight passed and three failed:
`fail-closed.mlir`, `map-errors.mlir`, and `no-shuttle-errors.mlir`. The runner
stopped at this first failed gate, so the four patched XLA tests did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-litlabels7`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-litlabels7`
- Controller state: `failed`
- Task exit: `3`
- Task duration: `671.6` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `96446afc70502cf3023383f4b2641d0ca2edfe88`
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
| `bazel test @shuttle_mlir//:mlir_tests` | 8 passed, 3 failed |
| `//xla/pjrt:stablehlo_module_transform_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_test` | Not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Not run |
| `//xla/pjrt:pjrt_executable_test` | Not run |

Operation generation completed in 18.205 seconds. `ShuttleDialect` completed
952 actions in 51.021 seconds, and `ShuttlePasses` completed 91 actions in
6.345 seconds. The driver configured 8,466 targets and completed 5,951 actions
in 539.985 seconds. The separate lit build completed 139 actions in 11.355
seconds. Lit execution took 2.813 seconds.

The three fixture failures expose separate issues:

- `fail-closed.mlir` invokes LLVM's `not` utility, but the lit target's tools
  contain only `shuttle-opt` and `FileCheck`. The generated script reported
  `not: command not found`, so FileCheck received the shell error instead of a
  Shuttle diagnostic.
- `map-errors.mlir` expected the unbound-domain and scalar-domain diagnostics.
  The newer result-map permutation check fired first in both cases, so the
  intended diagnostics were not reached.
- `no-shuttle-errors.mlir` emitted the intended error text at different MLIR
  locations: the nested `shuttle.yield`, an unknown fused source location, and
  the module operation. The fixture annotations expected the surrounding
  region or source lines, so `--verify-diagnostics` classified each actual
  diagnostic as unexpected and each annotation as missing.

This run establishes native build, link, lit analysis, and fixture execution
against the exact pins. It does not establish a passing lit suite or any of the
four XLA tests.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `f457503a0233242d5eb753fb40d57845daa73d429723f61c534162d1646dcdf0`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `remote-runner-and-gates.txt`: remote patch, seven-label, and ordered-gate
  evidence.
- `lit-failure-context.txt`: complete retained output around all three failed
  fixtures.
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
