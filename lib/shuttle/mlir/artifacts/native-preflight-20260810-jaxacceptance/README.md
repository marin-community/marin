# Native JAX acceptance preflight negative result

The CPU-only preflight passed the exact-pin patch and dependency proofs, all
seven native Shuttle build gates, four native tests, all 17 MLIR lit tests,
and all four patched XLA tests. The release JAX wheel then failed to compile
the test-only Python observer bridge against the pinned nanobind API.

`nanobind::tuple` does not provide a size constructor, and its indexed
accessor rejects mutation. The bridge used both unsupported operations while
building the Python snapshot. The runner stopped at this first failed gate.
No wheel was produced or installed, and none of the ordinary-JAX acceptance
workers ran.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-jaxacceptance`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-jaxacceptance`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `28 minutes and 47.53 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `370efa95c9ceaac2f80bcd2b2357752659c31715`
- JAX source: `619764c15117fbefc4ba13ab941871cb514c23f6`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- LLVM source: `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

| Gate | Result |
| --- | --- |
| XLA 0001 then 0002 and JAX 0001 then 0002 apply, reverse, and diff proofs | Passed |
| Real configured Bazel dependency-query verifier | Passed |
| Pure acceptance and verifier tests | 34 passed |
| `@shuttle_mlir//:shuttle_ops_inc_gen` | Passed |
| `@shuttle_mlir//:ShuttleDialect` | Passed |
| `@shuttle_mlir//:ShuttlePasses` | Passed |
| `@shuttle_mlir//:ShuttleXlaRegistration` | Passed |
| `@shuttle_mlir//:ShuttleXlaRegistryAdapter` | Passed |
| `@shuttle_mlir//:ShuttleObserverTestBridge` | Passed |
| `@shuttle_mlir//:shuttle-opt` | Passed build and link |
| Four native Shuttle tests | 4 passed uncached |
| `@shuttle_mlir//:mlir_tests` | 17 passed uncached |
| Four patched XLA tests | 4 passed uncached |
| `//jaxlib/tools:jaxlib_wheel` | Failed compiling `ShuttlePythonObserverTestBridge` |
| Wheel install and ordinary-JAX acceptance | Not run |

The wheel build analyzed 44,062 targets and ran 19,354 processes over
899.814 seconds before the compiler reported:

```text
external/shuttle_mlir/lib/Transforms/PythonObserverTestBridge.cc:19:13: error: no matching constructor for initialization of 'nb::tuple'
   19 |   nb::tuple record(11);
      |             ^      ~~
external/shuttle_mlir/lib/Transforms/PythonObserverTestBridge.cc:37:13: error: no matching constructor for initialization of 'nb::tuple'
   37 |   nb::tuple records(events.size());
      |             ^       ~~~~~~~~~~~~~
external/nanobind/include/nanobind/nb_accessor.h:200:23: error: static assertion failed due to requirement '!is_tuple': tuples are immutable!
  200 |         static_assert(!is_tuple, "tuples are immutable!");
      |                       ^~~~~~~~~
```

The runner emitted no wheel path or wheel hash because the wheel target did
not complete. The isolated install, disabled baseline, cache-disabled
concurrency and lifetime worker, cache population worker, and cache reuse
worker all remained unrun. This result establishes the native Shuttle and XLA
gates only; it does not establish ordinary-JAX invocation or cache behavior.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `be761823f3f626921d9781246511c52651d6ca74489e17655747b85bb97ea6e9`.
- `failure-context.txt`: exact first compiler diagnostics and gate boundary.
- `controller-terminal-proof.txt`, `task-describe.txt`, and `task-events.txt`:
  controller terminal and release evidence.
- `remote-runner-and-gates.txt`: remote patch, dependency, native, lit, XLA,
  and wheel-gate evidence.
- `run_jax_acceptance_preflight.sh`, `launch-command.txt`, and `manifest.env`:
  exact submitted command and frozen runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client, external
  non-secret store metadata, and config-free bundle proof.
- `local-runner-proof.txt`: local exact-pin pre-submit proof.
- `source-sha256.txt`: hashes for the 82 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, gate, and no-wheel facts.
- `monitoring-state.json`: sole-monitor submission and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

The terminal pod and Bazel output tree were not retained. This checkpoint does
not authorize a relaunch.
