# H100 contract-map evidence fifth-launch failure

The fifth reviewed execution attempt authenticated and restored the exact
source capsule, selected the frozen Python runtime, passed the H100/tool
preflight, and compiled every generated candidate. Each retained shared
library exposed the expected six-kernel topology. Nsight Systems accepted the
corrected `--capture-range-end=stop` command and started the first case worker.
The worker then rejected the first numerical result before timing because its
forward mean ULP distance exceeded the immutable ordinary-XLA floor.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-16cb0da-v5`
- Dashboard: `https://iris-cw-us-west-04a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-16cb0da-v5`
- Submitted: `2026-08-11T03:01:41.913Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; attempt duration `81.535 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `16cb0da00c80a79059ddbb448eb00be309ee49d6`
- Source tree: `0c8e14c2e84b1fd849d274c197b9a1f02e7279b7`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:16cb0da00c80a79059ddbb448eb00be309ee49d6@sha256:d011ebdfc00d5b3a423d872b86388126636e4755cc5eb722c078f6c6d2ce598a`

The source capsule contains 152 members and 4,963,276 expanded bytes. The
transported Iris bundle was 1,085,399 bytes, below Iris's 25MiB limit.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher and manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| Frozen Python runtime selection | Passed through `/opt/h100-evidence-runtime/bin/python` |
| H100 and compiler/profiler tool preflight | Passed |
| Every generated shared library, PTX, and cubin compile | Passed |
| Separate cubin and authoritative loaded-library SASS validation | Passed for every generated candidate |
| Nsight Systems command acceptance | Passed with `--capture-range-end=stop` |
| First-case numerical gate | Failed on `numerical.outputs.forward.mean_ulp_distance` |
| Nsight Compute and completed Nsight Systems report | Not produced |
| Timing samples and accepted 24-record evidence bundle | Not produced |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: exact controller request fields and submitted argv.
- `controller-summary.txt`: terminal controller, task, and attempt state.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: image workflow and immutable OCI identity.
- `failure-analysis.txt`: reached execution boundary and exact first failure.
- `SHA256SUMS`: hashes for every other file in this directory.

This negative result establishes source authentication, runtime and H100
preflight, generated candidate compilation, loaded-image topology validation,
and corrected Nsight Systems command acceptance. It does not establish the
rejected numerical value, timing, profiler evidence, or an accepted bundle;
the old diagnostic did not print the measured ULP value or threshold.
