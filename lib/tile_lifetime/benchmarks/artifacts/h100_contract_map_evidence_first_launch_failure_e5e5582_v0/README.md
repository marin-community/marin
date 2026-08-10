# H100 contract-map evidence first-launch failure

The first reviewed execution attempt authenticated and restored the exact source
capsule, then failed while importing the runner. The no-sync task command used
the unqualified `python` command. It resolved outside the image's frozen
`/opt/h100-evidence-runtime` environment, and `tile_lifetime.attention` failed at
`import numpy as np` with `ModuleNotFoundError: No module named 'numpy'`.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-e5e5582`
- Dashboard: `https://iris-cw-us-west-04a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-e5e5582`
- Submitted: `2026-08-10T22:52:43.859000+00:00`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; duration `5.45 seconds`
- Failures: `1`; preemptions: `0`; retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `e5e5582909cd27f5ceae27bd1d7fb4e3a5afecdf`
- Source tree: `052493377ab11599ed344e8bdd25ae052443269e`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:e5e5582909cd27f5ceae27bd1d7fb4e3a5afecdf@sha256:38e4ff36fba55f336782b5430b41b0aa67129c9b7d3cac51753e2cb7d00bfa28`

The launcher and manifest authentication lines were the only successful task
output before the traceback. The source capsule contains 152 members and
4,957,994 expanded bytes. The transported bundle was 1,084,285 bytes, below
Iris's 25MiB limit.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher SHA-256 | Passed in the task |
| Exact manifest SHA-256 | Passed in the task |
| Capsule restoration and member verification | Passed before runner execution |
| Frozen Python runtime selection | Failed; the task used unqualified `python` |
| Runner import | Failed at the first NumPy import |
| H100 architecture and tool preflight | Not run |
| JAX device query | Not run |
| CUDA compilation, profiling, or kernel execution | Not run |
| Evidence bundle production | Not run |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: redacted, typed controller `job_config` fields and the
  exact persisted command argv. No task environment values are retained.
- `controller-summary.txt`: terminal controller and task summary.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: exact source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: immutable image build workflow and OCI identity.
- `failure-analysis.txt`: interpreter and PATH boundary traced through the
  submitted command, Iris K8s task wrapper, capsule launcher, and image.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

This is a negative result. It establishes capsule authentication and the first
runtime failure only. It does not establish H100 availability, compiler or
profiler readiness, numerical parity, repeatability, or timing. This checkpoint
does not authorize a retry or relaunch.
