# H100 contract-map evidence twelfth-launch failure

The twelfth reviewed execution authenticated source commit
`41cd8512b7333525d22c57b9f1c8d4d789f270a8`, completed the first case's
three-backend numerical and Nsight Systems gates, and reached the first
ordinary-XLA cache protocol. All nine cache workers completed. The exact
cross-root content-identity gate rejected six equality classes.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-41cd851-v12`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-41cd851-v12`
- Submitted: `2026-08-11T12:55:42.888Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `147.172 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `41cd8512b7333525d22c57b9f1c8d4d789f270a8`
- Source tree: `c2e7141aff52332f5c23163ece348b61cd05309e`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`

The source capsule contains 154 members and 5,008,299 expanded bytes. The
three-file Iris bundle was 1,095,764 bytes, below Iris's 25MiB limit. Its
SHA-256 is the controller-recorded bundle ID
`e65c807dbb86f87f80c82e8225be27aa1cd520ef5d9ebd7165911b16c77f2780`.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| H100 and compiler/profiler preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| First-case three-backend numerical and steady-profile gates | Passed before cache workers |
| Three compile and three paired cold/hit roots | Completed; each cold/hit pair was exact |
| Nine-root cache identity convergence | Failed with six equality classes |
| Nsight Compute, retained artifacts, remaining backends/cases, accepted bundle | Not reached |
| Retry or relaunch | Not performed |

## Evidence

- `cache-diagnostic.json`: the bounded structured diagnostic emitted by the
  production validator.
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

All roots contained ten cache files. The three compile roots were distinct.
`compile[1]` and `compile[2]` had the same final-HLO hash but different cache
identities and byte totals. Each cold/hit pair matched its identity, HLO hash,
file count, and byte total exactly. This proves process-local repeatability and
cross-process variation. It does not identify which cache file or field varied.

