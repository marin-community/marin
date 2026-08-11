# H100 contract-map evidence eighth-launch failure

The eighth reviewed execution authenticated and restored exact source commit
`675e3399984129c0b76e3d14f9c123fb46fb0518`, selected the frozen Python
runtime, passed the H100/tool preflight, compiled the generated candidates,
validated the loaded shared-library topology, and passed the ordinary-XLA,
source-ordered, and fast numerical gates. The first steady sample then stopped
at the NVTX push for `contract_map.steady.0.ordinary_xla`.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-675e339-v8`
- Dashboard: `https://iris-cw-us-west-04a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-675e339-v8`
- Submitted: `2026-08-11T07:45:46.963Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `86.653 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `675e3399984129c0b76e3d14f9c123fb46fb0518`
- Source tree: `558fd48e110baa4d57d901778d7b230e13ed2ba6`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:16cb0da00c80a79059ddbb448eb00be309ee49d6@sha256:d011ebdfc00d5b3a423d872b86388126636e4755cc5eb722c078f6c6d2ce598a`

The source capsule contains 153 members and 4,982,973 expanded bytes. The
transported three-file Iris bundle was 1,089,601 bytes, below Iris's 25MiB
limit.

The wrapper raised `RuntimeError: NVTX rejected range
'contract_map.steady.0.ordinary_xla'`. Source commit `675e339` accepts NVTX
push levels zero and above and takes this branch only for a negative return.
The exact negative return value was not included in the task diagnostic.
Nsight Systems completed a local `steady_trace.nsys-rep`, but the failed worker
did not produce an accepted profile or evidence bundle.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| Frozen Python runtime selection | Passed through `/opt/h100-evidence-runtime/bin/python` |
| H100 and compiler/profiler tool preflight | Passed |
| Generated compilation and loaded-library topology | Passed before the case worker |
| Ordinary-XLA, source-ordered, and fast numerical gates | Passed before warmup and timing |
| Warmup schedule | Passed before steady timing |
| First steady NVTX range | Failed before the first timed execution |
| Nsight Systems report generation | Completed locally after worker failure; not accepted or retained |
| Nsight Compute and accepted profile | Not produced |
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
- `failure-analysis.txt`: reached execution boundary and exact first
  diagnostic.
- `SHA256SUMS`: hashes for every other file in this directory.

This result establishes that all three backends passed the pre-timing numerical
gates after the source-ordered BF16 fusion repair. It does not establish steady
timing, accepted profiler evidence, or an accepted 24-record bundle.
