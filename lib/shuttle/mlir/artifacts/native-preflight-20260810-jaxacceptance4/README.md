# Native JAX acceptance preflight negative result

The CPU-only preflight passed the exact-pin source, patch, configured
dependency, and seven native Shuttle build gates. The new default audit of all
six generated fixtures then selected the production `shuttle-opt` binary as
its normalizer. The requested test-only fingerprint flag is registered only by
`shuttle-test-opt`, so the subprocess exited 1 on the first temporary fixture.

The runner stopped at that first failure. Native tests, 17 MLIR lit tests, four
patched XLA tests, the JAX wheel, and ordinary-JAX acceptance did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-jaxacceptance4`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-jaxacceptance4`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `7 minutes and 5.76 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `2fadce00a8a94512f613fd0ceb24d51525698c27`
- JAX source: `619764c15117fbefc4ba13ab941871cb514c23f6`
- XLA source: `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`
- LLVM source: `9a4faee1068c09efbf837cfb7b0f5693b24635f4`
- Bazel: `7.7.0`
- Resources: `24` CPU, `96GB` memory, `250GB` disk
- Accelerators: none requested; no device query ran

| Gate | Result |
| --- | --- |
| XLA 0001 then 0002 and JAX 0001 then 0002 apply, reverse, and diff proofs | Passed |
| Seven anchored XLA runtime labels | Passed, exact count 7 |
| Real configured Bazel dependency-query verifier | Passed |
| Pure acceptance and verifier tests | 47 passed |
| Acceptance fixture oracle verifier | Passed |
| Seven native Shuttle build gates | Passed |
| Default audit of all six generated fixtures | Failed before checking the first fixture |
| Four native Shuttle tests | Not run |
| `@shuttle_mlir//:mlir_tests` | Not run |
| Four patched XLA tests | Not run |
| `//jaxlib/tools:jaxlib_wheel` | Not run |
| Ordinary-JAX acceptance workers | Not run |

The seven build gates were `shuttle_ops_inc_gen`, `ShuttleDialect`,
`ShuttlePasses`, `ShuttleXlaRegistration`, `ShuttleXlaRegistryAdapter`,
`ShuttleObserverTestBridge`, and `shuttle-opt`. Their Bazel builds all completed
successfully before the audit.

The generator invoked:

```text
/app/sources/xla/bazel-out/k8-opt/bin/external/shuttle_mlir/shuttle-opt \
  --shuttle-test-report-normalized-fingerprint \
  /tmp/shuttle-fixture-audit-2fwg2mhu/fixture.mlir
```

The submitted generator captured subprocess output and raised only
`CalledProcessError`, so the retained log does not contain the tool's stderr.
Source inspection after the terminal result found that `ShuttleTestPasses` owns
the flag and is linked by `@shuttle_mlir//:shuttle-test-opt`; production
`shuttle-opt` intentionally omits it. This is runner/tool-selection evidence,
not a fixture-oracle or normalizer-semantic failure.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; the artifact records compressed and decompressed hashes.
- `failure-context.txt`: exact first failure and unrun gates.
- `controller-terminal-proof.txt`, `task-describe.txt`, and `task-events.txt`:
  controller terminal, attempt, and release evidence.
- `raw-attempt-client.stderr`: retained client tunnel log for the full-log
  fetch.
- `remote-runner-and-gates.txt`: remote patch, dependency, test, build, and
  failure-gate evidence.
- `run_jax_acceptance_preflight.sh`, `launch-command.txt`, and `manifest.env`:
  exact submitted command and frozen runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client, external
  non-secret store metadata, and config-free bundle proof.
- `local-runner-proof.txt`: local exact-pin pre-submit proof summary.
- `source-sha256.txt`: hashes for the 82 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, gate, and terminal facts.
- `monitoring-state.json`: sole-monitor submission and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

This checkpoint does not authorize a relaunch.
