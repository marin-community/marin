# H100 contract-map evidence fourteenth-launch failure

The fourteenth reviewed execution authenticated source commit
`fbe098a4e342cfa6c8795459c01d5399fbd7524b`, completed the first case's
three-backend numerical and Nsight Systems gates, and ran the first
ordinary-XLA cache protocol through all nine workers. The semantic cache
identity gate rejected six serialized-executable classes under one target-key
digest.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-fbe098a-v14`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-fbe098a-v14`
- Submitted: `2026-08-11T14:03:09.750Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `142.686 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `fbe098a4e342cfa6c8795459c01d5399fbd7524b`
- Source tree: `d52e91467c927160b49c155af4bfcc185fb6c5a9`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`

The source capsule contains 154 members and 5,017,882 expanded bytes. The
three-file Iris bundle was 1,098,147 bytes, below Iris's 25MiB limit. Its
SHA-256 is the controller-recorded bundle ID
`c86ddf11b78e75cb41579c1238f3634b9a109b361c04b38631d0eb6b6893dd13`.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| H100 and compiler/profiler preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| First-case three-backend numerical and steady-profile gates | Passed before cache workers |
| Three compile and three paired cold/hit roots | Completed; each cold/hit pair was exact |
| Nine-root cache identity convergence | Failed with six serialized-executable classes |
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

All nine roots have the same exact target-key digest. The three fresh compile
roots have distinct serialized-executable hashes. Each cold/hit pair has the
same serialized-executable hash, file count, byte total, and final-HLO hash.
The three pairs differ from one another and from the compile roots.
