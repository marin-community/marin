# H100 contract-map evidence thirteenth-launch failure

The thirteenth reviewed execution authenticated source commit
`fcbe3a3c8ab430dd0fbc6e1e7f789b1342c71387`, completed the first case's
three-backend numerical and Nsight Systems gates, and ran the first
ordinary-XLA cache protocol through all nine workers. The cache convergence
validator failed closed because its rejection diagnostic exceeded the reviewed
4,096-character bound.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-fcbe3a3-v13`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-fcbe3a3-v13`
- Submitted: `2026-08-11T13:42:57.081Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `141.709 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `fcbe3a3c8ab430dd0fbc6e1e7f789b1342c71387`
- Source tree: `3713bc31212acc8eae749e7a94fd373ac73e0eda`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`

The source capsule contains 154 members and 5,017,806 expanded bytes. The
three-file Iris bundle was 1,098,083 bytes, below Iris's 25MiB limit. Its
SHA-256 is the controller-recorded bundle ID
`f6798ef15420ae74df2feba96a6bf7662ce758b4fe8ed55cf95dea09b6d5f282`.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| H100 and compiler/profiler preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| First-case three-backend numerical and steady-profile gates | Passed before cache workers |
| Three compile and three paired cold/hit roots | Completed; every immediate cold/hit root check passed |
| Nine-root cache identity convergence | Failed before a bounded diagnostic could be emitted |
| Nsight Compute, retained artifacts, remaining backends/cases, accepted bundle | Not reached |
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
- `failure-analysis.txt`: reached execution boundary and bounded conclusions.
- `SHA256SUMS`: hashes for every other file in this directory.

No cache key, serialized-executable identity, equality partition, file count,
byte total, or HLO hash reached the retained log. The failure therefore proves
that the diagnostic representation exceeded its bound. It does not establish
whether the nine semantic cache identities converged or how many partitions
were observed.
