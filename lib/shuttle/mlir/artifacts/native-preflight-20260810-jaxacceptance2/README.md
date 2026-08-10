# Native JAX acceptance preflight negative result

The CPU-only preflight passed the exact-pin source, patch, and configured
dependency proofs; all seven native Shuttle build gates; four uncached native
tests; all 17 MLIR lit tests; and all four patched XLA tests. It then built,
hashed, installed, and imported the release JAX 0.10.1 wheel with the Shuttle
test observer in the final `_jax` DSO.

The checked-in ordinary-JAX acceptance driver completed its disabled baseline
worker. Its cache-disabled concurrency worker compiled four ordinary `jax.jit`
wrappers, retained observer events after closing the capture, and failed when
one invocation did not match exactly one audited forward or VJP contract. The
runner stopped at that first failed gate. The cache population and reuse
workers did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-jaxacceptance2`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-jaxacceptance2`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `58 minutes and 15.31 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `7fd635f8a90a0f8299baf6ff58dcee7722aaa2ef`
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
| Pure acceptance and verifier tests | 40 passed |
| Acceptance fixture oracle verifier | Passed |
| Seven native Shuttle build gates | Passed |
| Four native Shuttle tests | 4 passed uncached |
| `@shuttle_mlir//:mlir_tests` | 17 passed uncached |
| Four patched XLA tests | 4 passed uncached |
| `//jaxlib/tools:jaxlib_wheel` | Passed in 2,123.969 seconds |
| Isolated JAX, jaxlib, and Shuttle install | Passed |
| Disabled ordinary-JAX baseline worker | Passed |
| Cache-disabled concurrency/lifetime worker | Failed contract match |
| Cache population and reuse workers | Not run |

The release wheel was
`/app/wheel-dist/jaxlib-0.10.1-cp312-cp312-manylinux_2_27_x86_64.whl`
with SHA-256
`beeae0e0cd9f7af8e80c92b9a1fc15970a4745eabb54c80be288747da5bac6c7`.
The wheel built in 25,078 actions, installed cleanly, and imported as jaxlib
0.10.1 beside JAX `0.10.1.dev20260518+619764c` and the bundled Shuttle
package.

The first acceptance failure was:

```text
File "/app/lib/shuttle/mlir/jax_patch/shuttle_jaxlib_acceptance.py", line 184, in grouped_event_evidence
  raise AssertionError("observer invocation did not match exactly one audited fixture contract")
AssertionError: observer invocation did not match exactly one audited fixture contract
```

The failing code path did not serialize the retained events or the per-contract
mismatches, so the controller log cannot identify the differing field. The
terminal pod was deleted after release and its `/app/acceptance-work` and
`/app/logs` files are no longer available. The job used `--no-sync`; Iris
retained controller logs but exposed no reverse file-sync artifact for the
wheel. This checkpoint therefore preserves the exact wheel hash, path, build
command, install proof, and contents listing in the raw log, but not the wheel
binary itself.

This result establishes native compilation, native tests, lit coverage,
patched XLA integration, release-wheel production, and isolated installation.
It does not establish the ordinary-JAX observer contract, cache population, or
second-process cache reuse.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; decompressed SHA-256
  `58b834bc4b78f9388b2bef543604ff23171579e1cf015bb32b19f58998ac00a4`.
- `failure-context.txt`: exact acceptance traceback, wheel facts, and first
  failed gate.
- `controller-terminal-proof.txt`, `task-describe.txt`, and `task-events.txt`:
  controller terminal and release evidence.
- `raw-attempt-client.stderr`: retained client tunnel log for the final log
  fetch.
- `remote-runner-and-gates.txt`: remote patch, dependency, native, lit, XLA,
  wheel, install, and acceptance-gate evidence.
- `run_jax_acceptance_preflight.sh`, `launch-command.txt`, and `manifest.env`:
  exact submitted command and frozen runner inputs.
- `client-proof.txt` and `bundle-proof.txt`: absolute client, external
  non-secret store metadata, and config-free bundle proof.
- `local-runner-proof.txt`: local exact-pin pre-submit proof summary.
- `source-sha256.txt`: hashes for the 82 bundled canonical Shuttle files.
- `toolchain.txt`: source, toolchain, output-tree, gate, wheel, and install
  facts.
- `monitoring-state.json`: sole-monitor submission and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

This checkpoint does not authorize a relaunch.
