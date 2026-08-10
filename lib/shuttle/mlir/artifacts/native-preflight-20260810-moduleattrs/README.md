# Native MLIR preflight through XLA test compilation

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found all seven anchored
XLA-owned runtime labels. Operation generation, both Shuttle libraries,
`shuttle-opt`, the separate `mlir_tests` build, and all 11 lit fixtures passed.

The XLA hook implementation advanced past the prior `ModuleOp::getAttrs`
failure. The four-test gate then failed compiling
`stablehlo_module_transform_test.cc`: the new test used `TF_ASSERT_OK` and
`TF_EXPECT_OK` without making those macros available to the translation unit.
The runner stopped at this first failed gate. No XLA test executable ran.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-moduleattrs`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-moduleattrs`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `802.26` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `db8cb729be64e66ddbc84a8ab506174804737c64`
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
| `//xla/pjrt:stablehlo_module_transform_test` | Failed to build; did not run |
| `//xla/pjrt:mlir_to_hlo_test` | No status; did not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | No status; did not run |
| `//xla/pjrt:pjrt_executable_test` | No status; did not run |

Operation generation completed in 19.215 seconds. `ShuttleDialect` completed
952 actions in 51.604 seconds, and `ShuttlePasses` completed 91 actions in
6.252 seconds. The driver configured 8,466 targets and completed 5,951 actions
in 532.030 seconds. The separate lit build completed 143 actions in 11.385
seconds. Lit execution took 2.674 seconds and passed all 11 tests.

The four-test XLA gate analyzed all four targets, then failed after 134.122
seconds with 13 diagnostics from one translation unit:

```text
xla/pjrt/stablehlo_module_transform_test.cc:73:3: error:
use of undeclared identifier 'TF_ASSERT_OK'
xla/pjrt/stablehlo_module_transform_test.cc:85:3: error:
use of undeclared identifier 'TF_EXPECT_OK'
...
12 warnings and 13 errors generated.
```

`TF_ASSERT_OK` was unresolved at lines 73, 113, 128, 130, 150, 160, 161,
174, 207, and 229. `TF_EXPECT_OK` was unresolved at lines 85, 132, and 191.
Bazel reported zero of four tests executed: one target failed to build and three
had no status. This run establishes the complete native Shuttle build, link,
and lit suite against the exact pins. It does not establish that any patched
XLA test passes.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `07393ff02c6c2d31ab6e0efb68e906064a419f5320f22ce7fde55f97c1ce24c8`.
- `terminal-summary.txt`: controller terminal summary and release proof.
- `remote-runner-and-gates.txt`: remote patch, seven-label, ordered-gate, lit,
  and compiler-failure evidence.
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
