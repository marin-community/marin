# H100 contract-map evidence tenth-launch failure

The tenth reviewed execution authenticated exact source commit
`e9c050131c301b9bf98e93784f41fd31b8e5ef47`, passed the H100/tool preflight,
compiled the generated candidates, passed the three backend numerical gates,
and completed the first profiled case worker. Nsight Systems produced and
exported the first steady-state report. The source parser then found no CUDA
kernel fully contained by the first required NVTX range,
`contract_map.steady.0.ordinary_xla`.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-e9c0501-v10`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-e9c0501-v10`
- Submitted: `2026-08-11T09:35:54.045Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `88.184 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `e9c050131c301b9bf98e93784f41fd31b8e5ef47`
- Source tree: `6192ebdf6f55ba99d019710c53e0c88dc2d028e9`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:3247e17d6c0f0fbf7263b5aee7891d209c978ac9@sha256:1543d62f4537773d09a1d7968139665d4f01b4b193412ea71f9f98a9a7f21e45`

The source capsule contains 154 members and 4,993,325 expanded bytes. The
three-file Iris bundle was 1,092,621 bytes, below Iris's 25MiB limit.

The wrapper raised:

```text
ValueError: Nsight Systems range 'contract_map.steady.0.ordinary_xla' contains no CUDA kernels
```

The failed task did not export its temporary `.nsys-rep` or SQLite file from
Iris. This artifact therefore does not claim the range timestamps, kernel
timestamps, table inventory, or cause of the failed attribution.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| Frozen Python runtime selection | Passed through `/opt/h100-evidence-runtime/bin/python` |
| H100 and compiler/profiler tool preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| Ordinary-XLA, source-ordered, and fast numerical gates | Passed before steady timing |
| Steady schedule and NVTX capture | Case worker completed and serialized its result |
| Nsight Systems profile and SQLite export | Completed before parser entry |
| SQLite schema, range schedule, and kernel identity parsing | Passed through first-range attribution |
| First-range CUDA kernel attribution | Failed; no fully contained kernel was found |
| Accepted timing/profile and 24-record evidence bundle | Not produced |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: sanitized controller request fields and exact submitted
  argv. Ambient credentials are omitted.
- `controller-summary.txt`: terminal job, task, and attempt state from narrow
  read-only controller queries.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: image workflow and immutable OCI identity.
- `failure-analysis.txt`: reached execution boundary and exact first
  diagnostic.
- `SHA256SUMS`: hashes for every other file in this directory.

This result establishes successful export and parser entry for the first
profiled case. It does not establish accepted per-range timing or copy
accounting, remaining-case completion, or an accepted 24-record bundle.
