# H100 contract-map evidence third-launch failure

The third reviewed execution attempt authenticated and restored the exact
source capsule, selected the frozen Python runtime, passed the H100 and tool
preflight, and compiled the first generated candidate's shared library, PTX,
and cubin. It also disassembled the retained cubin. It failed before kernel
execution when the exact six-kernel topology parsed from the authoritative
loaded shared library did not equal the generated topology.

## Result

- Job: `/dlwh/shuttle-h100-contract-map-evidence-dbbd9e-v3`
- Dashboard: `https://iris-cw-us-west-04a.oa.dev/#/job/%2Fdlwh%2Fshuttle-h100-contract-map-evidence-dbbd9e-v3`
- Submitted: `2026-08-11T01:32:22.774Z`
- Controller state: `failed`
- Task: `0/1` completed; task exit `1`; duration `15.32 seconds`
- Failures: `1`; preemptions: `0`; failure retries configured: `0`
- Resources: `8` CPU, `64GB` memory, `100GB` disk, one `H100`
- Container profile: `CONTAINER_PROFILE_DEFAULT`
- Source commit: `dbbd9e4fe53e8ec7ad2c8d409dbaa0351ac064ff`
- Source tree: `fc78276a63b1e64ff7abe3c976c619d0492727f2`
- Immutable image: `ghcr.io/marin-community/iris-task-h100-evidence:dbbd9e4fe53e8ec7ad2c8d409dbaa0351ac064ff@sha256:945f44cca0aa44be922c9d806e7b8e6b98915ed22323cca26ca89f23bf3a4e19`

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
| Separate retained cubin disassembly | Passed |
| Loaded shared-library SASS topology | Failed the exact six-kernel comparison |
| Nsight Compute and Nsight Systems collection | Not run |
| Kernel execution and numerical checks | Not run |
| Accepted 24-record evidence bundle | Not produced |
| Retry or relaunch | Not performed |

## Evidence

- `submission.json`: typed controller request fields and exact persisted argv.
- `controller-summary.txt`: terminal controller and task summary.
- `raw-task.log`: complete controller-retained task output.
- `payload-identity.json`: exact source, capsule, manifest, launcher, and bundle
  identities from the pre-submit proof.
- `image-build-proof.txt`: immutable image build workflow and OCI identity.
- `failure-analysis.txt`: the reached runner boundary and exact first failure.
- `SHA256SUMS`: integrity hashes for every other file in this directory.

This negative result establishes the capsule, runtime, device/tool preflight,
first generated compilation, separate cubin disassembly, and loaded-image
inspection boundary only. It does not establish the actual loaded kernel
topology, numerical parity, repeatability, profiler evidence, kernel timing,
or an accepted bundle.
