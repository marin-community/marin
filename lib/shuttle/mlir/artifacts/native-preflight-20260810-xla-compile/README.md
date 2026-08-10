# Native MLIR preflight through XLA hook compilation

The CPU-only preflight applied both reviewed XLA patches in order,
reverse-checked both, verified the combined diff, and found all seven anchored
XLA-owned runtime labels. Operation generation, both Shuttle libraries,
`shuttle-opt`, the separate `mlir_tests` build, and all 11 lit fixtures passed.

The runner then reached the four patched XLA tests. Their shared
`stablehlo_module_transform` dependency failed to compile because the patch
called `getAttrs()` on `mlir::ModuleOp`, which does not expose that method at
the pinned MLIR revision. The runner stopped at this first failed gate. No XLA
test executable ran.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-litruntime`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-litruntime`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `644.17` seconds
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `a23d26e80a7176f2582f6810d3e10ad1ca3f1d11`
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
| `//xla/pjrt:stablehlo_module_transform_test` | Did not build or run |
| `//xla/pjrt:mlir_to_hlo_test` | Did not run |
| `//xla/pjrt:mlir_to_hlo_unregistered_transform_test` | Failed to build its shared dependency; did not run |
| `//xla/pjrt:pjrt_executable_test` | Did not run |

Operation generation completed in 17.721 seconds. `ShuttleDialect` completed
952 actions in 47.832 seconds, and `ShuttlePasses` completed 91 actions in
7.175 seconds. The driver configured 8,466 targets and completed 5,951 actions
in 505.209 seconds. The separate lit build completed 143 actions in 11.347
seconds. Lit execution took 2.451 seconds and passed all 11 tests.

The four-test XLA gate analyzed all four targets, then failed after 19.247
seconds while compiling `xla/pjrt/stablehlo_module_transform.cc`:

```text
xla/pjrt/stablehlo_module_transform.cc:98:33: error: no member named 'getAttrs' in 'mlir::ModuleOp'
   98 |   module->setAttrs(transformed->getAttrs());
      |                    ~~~~~~~~~~~~~^
```

Bazel reported zero of four tests executed: one target failed to build and
three were skipped. This run establishes the complete native Shuttle build,
link, and lit suite against the exact pins. It does not establish that the XLA
hook compiles or that any patched XLA test passes.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `cd3ee132c1d2c1ec7eeb1b557dbf7cd0335c063034d635bbbdba8ea78e90bf69`.
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
