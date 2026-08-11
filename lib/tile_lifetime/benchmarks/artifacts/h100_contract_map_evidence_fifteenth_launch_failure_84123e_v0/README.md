# H100 contract-map evidence fifteenth-launch failure

The fifteenth reviewed execution authenticated source commit
`84123e89713445b7b23fc5dc1f16e0675063b0aa`, completed the first case's
canonical-cache setup, numerical and Nsight Systems gates, and three backend
cache protocols. The first Nsight Compute profile produced its report, public
SASS export, and CSV, but the CSV parser treated the units row as a kernel row
and rejected its empty `Kernel Name` field.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-84123e8-v15`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-84123e8-v15`
- Submitted: `2026-08-11T14:54:01.725Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `218.075 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `84123e89713445b7b23fc5dc1f16e0675063b0aa`
- Source tree: `23fd5849ea8a629f6a7ccaf09b653ab3f865921e`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:5238c39e7a506b919fc803046de4a9dc2c29d02f@sha256:2efa83fbf8f2073a4175eef919a9f0c6c2db435d7c1ec9ed79017e1ea0d10cef`

The source capsule contains 154 members and 5,041,535 expanded bytes. The
three-file Iris bundle was 1,102,571 bytes, below Iris's 25MiB limit. Its
SHA-256 is the controller-recorded bundle ID
`8210d457e9dfc319d3627c531df153a4844fbc59e81ab480bcf7be2c6dbb6f80`.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| H100 and compiler/profiler preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| First-case canonical cache preparation and merge | Passed |
| First-case three-backend numerical and steady-profile gates | Passed |
| Three backend fresh-compile and canonical-retrieval cache protocols | Passed internally |
| First Nsight Compute worker, report, and SASS export | Completed before CSV parsing |
| Nsight Compute units-row parsing | Failed before metric/kernel acceptance |
| Cross-consumer cache validation, retained artifacts, remaining cases, accepted bundle | Not reached |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: sanitized controller request fields and exact submitted
  argv. Ambient credentials are omitted.
- `controller-summary.txt`: terminal job, task, and attempt state from narrow
  read-only controller queries.
- `bounded-traceback.log`: exact task output before the 11,992-byte final
  diagnostic row.
- `raw-log-identity.txt`: byte, line, and SHA-256 identities for the complete
  controller-retained task output and omitted final diagnostic.
- `ncu-units-observation.json`: bounded identity fields and required metric
  units recovered from the exact final diagnostic.
- `payload-identity.json`: source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: image workflow and immutable OCI identity.
- `failure-analysis.txt`: reached execution boundary and evidence limit.
- `SHA256SUMS`: hashes for every other file in this directory.

The complete final diagnostic is not checked in because it serializes 274 CSV
columns. Its byte length and SHA-256 are retained, while the bounded observation
contains the eleven empty identity fields and nine metric units needed to
reproduce the parser failure.
