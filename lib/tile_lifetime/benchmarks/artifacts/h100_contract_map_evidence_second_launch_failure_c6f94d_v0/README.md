# H100 contract-map evidence second-launch failure

The second reviewed execution attempt authenticated and restored the exact
source capsule, selected the frozen Python runtime, passed the H100 and tool
preflight, and compiled the first generated candidate's shared library, PTX,
and cubin. It failed before kernel execution while disassembling that cubin.
CUDA 13.2 `cuobjdump --dump-sass` could not find its `nvdisasm` executable in
the image through either `PATH` or `NVDISASM_PATH`.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-c6f94d-v2`
- Dashboard: `https://iris-cw-us-west-04a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-c6f94d-v2`
- Submitted: `2026-08-10T23:38:54.081000+00:00`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; duration `16.06 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `c6f94dda2e1fbb4af06c8f35363c4556c237b9be`
- Source tree: `34a9e5177bef5e206661e7647d5b8d3f9728170d`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:c6f94dda2e1fbb4af06c8f35363c4556c237b9be@sha256:b5cd299addc5e7b313f4e7b18537b765b74b31e4c092e1e257c031a9c0483819`

The source capsule contains 152 members and 4,958,782 expanded bytes. The
transported Iris bundle was 1,084,584 bytes, below Iris's 25MiB limit.

| Gate | Result |
| --- | --- |
| Deterministic name collision check | Passed; no existing job was found |
| Exact launcher SHA-256 | Passed in the task |
| Exact manifest SHA-256 | Passed in the task |
| Capsule restoration and imported-module audit | Passed before generated compilation |
| Frozen Python runtime selection | Passed through the explicit `/opt/h100-evidence-runtime/bin/python` boundary |
| H100 and compiler/profiler tool preflight | Passed |
| First generated shared library, PTX, and cubin compile | Passed |
| First generated cubin disassembly | Failed because `nvdisasm` was absent |
| Nsight Compute and Nsight Systems collection | Not run |
| Kernel execution and numerical checks | Not run |
| Accepted 24-record evidence bundle | Not produced |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: redacted, typed controller `job_config` fields and the
  exact persisted command argv. No task environment values are retained.
- `controller-summary.txt`: terminal controller and task summary.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: exact source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: immutable image build workflow and OCI identity.
- `failure-analysis.txt`: the reached runner boundary and missing executable.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

This negative result establishes the capsule, runtime, device/tool preflight,
and first generated compilation boundary only. It does not establish numerical
parity, repeatability, profiler evidence, kernel timing, or an accepted bundle.

