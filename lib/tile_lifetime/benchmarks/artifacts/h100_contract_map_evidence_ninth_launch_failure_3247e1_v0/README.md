# H100 contract-map evidence ninth-launch failure

The ninth reviewed execution authenticated and restored exact source commit
`3247e17d6c0f0fbf7263b5aee7891d209c978ac9`, selected the frozen Python
runtime, passed the H100/tool preflight, compiled the generated candidates,
validated the loaded shared-library topology, passed the numerical gates, and
completed the first profiled case worker. Nsight Systems produced and exported
the first steady-state report, but the source parser rejected its SQLite schema
because an empty `CUPTI_ACTIVITY_KIND_MEMCPY` table was omitted.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-3247e17-v9`
- Dashboard: `https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-3247e17-v9`
- Submitted: `2026-08-11T08:56:12.784Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `86.043 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `3247e17d6c0f0fbf7263b5aee7891d209c978ac9`
- Source tree: `1b6dafaf1692cfe62ef4317cda96d3418f7a58d6`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:3247e17d6c0f0fbf7263b5aee7891d209c978ac9@sha256:1543d62f4537773d09a1d7968139665d4f01b4b193412ea71f9f98a9a7f21e45`

The source capsule contains 154 members and 4,988,812 expanded bytes. The
transported three-file Iris bundle was 1,091,423 bytes, below Iris's 25MiB
limit.

The wrapper raised:

```text
ValueError: Nsight Systems SQLite export omits CUPTI tables: ('CUPTI_ACTIVITY_KIND_MEMCPY',)
```

The failed task did not retain the temporary exported SQLite database, so this
artifact does not claim its complete table inventory or metadata values.

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
| SQLite trace parsing | Failed on the omitted memcpy table |
| Accepted timing/profile and 24-record evidence bundle | Not produced |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: sanitized controller request fields and exact submitted
  argv. Ambient credentials are intentionally omitted.
- `controller-summary.txt`: terminal job, task, and attempt state.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: image workflow and immutable OCI identity.
- `failure-analysis.txt`: reached execution boundary and exact first
  diagnostic.
- `SHA256SUMS`: hashes for every other file in this directory.

This result establishes that the first case reached the exported Nsight
Systems trace parser after the numerical and profiling worker gates. It does
not establish accepted per-range copy accounting, accepted timing/profile
evidence, remaining-case completion, or an accepted 24-record bundle.
