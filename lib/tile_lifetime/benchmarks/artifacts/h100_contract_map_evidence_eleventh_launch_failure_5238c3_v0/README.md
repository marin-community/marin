# H100 contract-map evidence eleventh-launch failure

The eleventh reviewed execution authenticated exact source commit
`5238c39e7a506b919fc803046de4a9dc2c29d02f`, passed the H100/tool preflight,
compiled the generated candidates, and completed the first case's profiled
ordinary-XLA/source-ordered/fast execution. The first ordinary-XLA cache
protocol then produced all three compile roots and all three paired cold/hit
roots, but the nine roots did not converge to one persistent-cache content
identity.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-5238c39-v11`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-5238c39-v11`
- Submitted: `2026-08-11T12:27:41.852Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `146.853 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `5238c39e7a506b919fc803046de4a9dc2c29d02f`
- Source tree: `9f8e9e06cfc76055dee55d5a1a34d82d71878377`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`

The source capsule contains 154 members and 5,005,712 expanded bytes. The
three-file Iris bundle was 1,095,166 bytes, below Iris's 25MiB limit.

The wrapper raised:

```text
ValueError: all compile, cold, and hit roots must converge to one cache content identity
```

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| Frozen Python runtime selection | Passed through `/opt/h100-evidence-runtime/bin/python` |
| H100 and compiler/profiler tool preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| First-case three-backend numerical and steady-profile gates | Passed before cache workers |
| First ordinary-XLA three compile and three cold/hit root pairs | Completed before convergence validation |
| Nine-root cache identity convergence | Failed; more than one content identity was observed |
| Nsight Compute, retained backend artifacts, remaining backends/cases, accepted 24-record bundle | Not reached |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: sanitized controller request fields and exact submitted
  argv. Ambient credentials are omitted.
- `controller-summary.txt`: terminal job, task, and attempt state from a narrow
  read-only controller query.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: image workflow and immutable OCI identity.
- `failure-analysis.txt`: reached execution boundary and exact first
  diagnostic.
- `SHA256SUMS`: hashes for every other file in this directory.

The retained task log does not contain the nine individual identities, cache
file counts, byte totals, or final-HLO hashes. This result therefore does not
distinguish semantic nondeterminism, phase-owned metadata, or an overbroad
cache hashing scope.
