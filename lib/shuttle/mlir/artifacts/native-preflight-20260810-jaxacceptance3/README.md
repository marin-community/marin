# Native JAX acceptance preflight negative result

The CPU-only preflight passed the exact-pin source, patch, and configured
dependency proofs; all seven native Shuttle build gates; four uncached native
tests; all 17 MLIR lit tests; and all four patched XLA tests. It built, hashed,
installed, and imported the release JAX 0.10.1 wheel with the Shuttle test
observer in the final `_jax` DSO.

The checked-in ordinary-JAX acceptance driver completed its disabled baseline
worker. Its cache-disabled concurrency worker retained three observer events
for invocation 2 and rejected them because the normalized selected-region
membership matched neither audited fixture contract. The bounded diagnostic
recorded every event field and the independent forward and VJP mismatch
reasons. The runner stopped at that first failed gate. Cache population and
reuse did not run.

## Result

- Job: `/dlwh/shuttle-native-mlir-preflight-20260810-jaxacceptance3`
- Dashboard:
  `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-native-mlir-preflight-20260810-jaxacceptance3`
- Controller state: `failed`
- Task exit: `1`
- Task duration: `45 minutes and 27.6 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Canonical Marin source: `ba31d47e354746543bdc179277071de25c48eaed`
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
| Pure acceptance and verifier tests | 43 passed |
| Acceptance fixture oracle verifier | Passed |
| Seven native Shuttle build gates | Passed |
| Four native Shuttle tests | 4 passed uncached |
| `@shuttle_mlir//:mlir_tests` | 17 passed uncached |
| Four patched XLA tests | 4 passed uncached |
| `//jaxlib/tools:jaxlib_wheel` | Passed in 1,949.548 seconds |
| Isolated JAX, jaxlib, and Shuttle install | Passed |
| Disabled ordinary-JAX baseline worker | Passed |
| Cache-disabled concurrency/lifetime worker | Failed selected-region membership contract |
| Cache population and reuse workers | Not run |

The release wheel was
`/app/wheel-dist/jaxlib-0.10.1-cp312-cp312-manylinux_2_27_x86_64.whl`
with SHA-256
`21c30fd5403fa11f6894cb8e845ff43f34d61594e71203751b02e8762e0a4357`.
The wheel built in 25,078 actions, installed cleanly, and imported as jaxlib
0.10.1 beside JAX `0.10.1.dev20260518+619764c` and the bundled Shuttle
package.

The first mismatching invocation had policy `source_ordered` and three phases:
`algebra_coverage`, `lowered_coverage`, and `final_erasure`. The first two
events shared these bounded values:

- selected-region membership: length `373`, SHA-256
  `cc8a395018a9866e14fa882137009717ca604c13ce48ccc6044bac5d5a154df3`
- coverage manifest: length `2843`, SHA-256
  `a626b0c7df55bf83154a5697de85eb3c99989bff0cc77b8f3bf469310db0238d`
- unsupported fingerprint:
  `1a9aad82650111cbc134fcc17d1afcb051f9ae729f6cdfd48105d1e8dc210201`

The final event reported semantic erasure and normalized module fingerprint
`d4dad86c0c4abf2f4a98bdd19879cbfb789c8d6cba8b18fa56decc4589a8ddb5`.
Both fixture comparisons failed only at normalized selected-region membership.
The retained diagnostic does not expose the region-membership contents, so it
does not yet distinguish a stale fixture oracle from pipeline semantic drift.

The terminal pod was deleted after release. The job used `--no-sync`; Iris
retained controller logs but exposed no reverse file-sync artifact for the
wheel. This checkpoint preserves the exact wheel hash, path, build command,
install proof, and contents listing in the raw log, but not the wheel binary.

This result establishes native compilation, native tests, lit coverage,
patched XLA integration, release-wheel production, isolated installation, and
bounded observer-failure diagnostics. It does not establish the ordinary-JAX
observer contract, cache population, or second-process cache reuse.

## Evidence

- `raw-attempt.log.gz`: complete controller-retained log, compressed with
  `gzip -n`; the artifact records its compressed and decompressed hashes.
- `failure-context.txt`: exact three-event diagnostic, wheel facts, and first
  failed gate.
- `controller-terminal-proof.txt`, `controller-wait-proof.txt`,
  `task-describe.txt`, and `task-events.txt`: controller terminal and release
  evidence.
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
- `toolchain.txt`: source, toolchain, output-tree, gate, wheel, install, and
  diagnostic facts.
- `monitoring-state.json`: sole-monitor submission and terminal state.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

This checkpoint does not authorize a relaunch.
