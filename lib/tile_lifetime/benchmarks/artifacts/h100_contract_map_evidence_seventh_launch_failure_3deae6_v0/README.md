# H100 contract-map evidence seventh-launch failure

The seventh reviewed execution authenticated and restored exact source commit
`3deae6618d52b3cfe1bf993b40f8e5192928be20`, selected the frozen Python
runtime, passed the H100/tool preflight, compiled the generated candidates,
validated the loaded shared-library topology, and entered the first profiled
case worker. The worker accepted ordinary XLA numerically, then rejected the
source-ordered generated `dx` output before timing.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-3deae66-v7`
- Dashboard: `https://iris-cw-us-west-04a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-3deae66-v7`
- Submitted: `2026-08-11T06:59:38.545Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `82.003 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `3deae6618d52b3cfe1bf993b40f8e5192928be20`
- Source tree: `e5cc6ee41bc756bcdc4caa4da96792aef5510101`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:16cb0da00c80a79059ddbb448eb00be309ee49d6@sha256:d011ebdfc00d5b3a423d872b86388126636e4755cc5eb722c078f6c6d2ce598a`

The source capsule contains 153 members and 4,976,307 expanded bytes. The
transported three-file Iris bundle was 1,088,143 bytes, below Iris's 25MiB
limit.

The direct H100 diagnostic for case `contract_map_9836cdbed389db24` reported
source-ordered `dx` maximum ULP distance `29608` against the immutable `1`
limit. Maximum and mean absolute error were `0.00390625` and
`0.0003416987310629338`; mean ULP distance was `28.51408765652952`; nonfinite
count was zero. Three repeats had identical content identities and zero
pairwise drift.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| Frozen Python runtime selection | Passed through `/opt/h100-evidence-runtime/bin/python` |
| H100 and compiler/profiler tool preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| Nsight Systems command acceptance | Passed with `--capture-range-end=stop` |
| Ordinary-XLA numerical gate | Passed before the generated backend |
| First source-ordered numerical gate | Failed with measured scalar and immutable limit |
| Nsight Compute and completed Nsight Systems report | Not produced |
| Timing samples and accepted 24-record evidence bundle | Not produced |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: sanitized controller request fields and exact submitted
  argv. Ambient credentials are intentionally omitted.
- `controller-summary.txt`: terminal job, task, and attempt state.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: image workflow and immutable OCI identity.
- `failure-analysis.txt`: reached execution boundary and exact numerical
  diagnostic.
- `SHA256SUMS`: hashes for every other file in this directory.

This result records H100 measurements for the rejected output. It does not
establish source-ordered numerical acceptance, profiler evidence, timing, or
an accepted bundle.
