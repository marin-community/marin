# [shellbox] Make the Iris machine backend create sandboxes, and declare its network policy

`shellbox.backends.iris.IrisMachineFactory` cannot create a machine at the head of #9623, and when its creation is
fixed, it cannot create one for a TaskCompendium task with the default `environment.network = false`. Taskforge runs
every docker task through this factory on the cluster, so every docker task fails on Iris today. This draft
describes what was measured on 2026-10-05 and the change in [`iris-machine.patch`](iris-machine.patch) (not
applied; `git apply --check` passes from the worktree root).

## What fails today

Measured from inside Iris jobs on `marin` with `lib/taskforge/scripts/iris_machine_probe.py` (evidence under
`lib/taskforge/.evidence/sandbox/`, which is gitignored and local to the machine that ran it):

1. **Every `create` fails within a second.** The readiness poll compares `Task.status().state`, which is now the
   string enum `iris.client.workload.TaskState`, with the proto ints `job_pb2.TASK_STATE_*`. The comparison is never
   true, so the first poll takes the failure branch, which then reads `status.error`, a field `TaskStatus` does not
   have: `AttributeError("'TaskStatus' object has no attribute 'error'")` after 0.27 to 0.87 s, in all five probe
   runs (`iris_machine_probe-0{2..6}.txt`). The unit tests never call `create`, so they pass.
2. **`NetworkPolicy.DENY` is refused**, and TaskCompendium's `EnvironmentSpec.network` defaults to `false`, which
   RolloutEngine (`rolloutengine.machines._task_machine`) maps to `DENY`. So with (1) fixed, a default docker task
   still fails with `Iris does not provide per-job network denial; select NetworkPolicy.ALLOW`.
3. **`NetworkPolicy.ALLOW` is not what a `marin` sandbox gets.** In every gVisor sandbox that started on the GCP
   workers (us-west4-a), `getent hosts` failed (no DNS; `/etc/resolv.conf` points at the host's `127.0.0.53`) and a
   TCP connect to `1.1.1.1:443` failed with `Network is unreachable` (`iris_machine_probe-05.txt`, `-06.txt`,
   `explore/net*_gcp_gvisor.txt`). The same image under the DEFAULT profile on the same pool resolves and connects
   (`explore/net_gcp_default.txt`). The gVisor profile on these workers is network-denied, so the backend
   advertises the one policy it does not provide and refuses the one it does.
4. **Uploads over about 96 KiB fail on docker workers.** `TRANSFER_CHUNK_BYTES = 128 * 1024` puts about 175 KiB of
   base64 into one `sh -c` argument; Linux limits one argument to 128 KiB (`MAX_ARG_STRLEN`), so the worker's
   `docker exec` fails with `[Errno 7] Argument list too long: 'docker'` (probe 05). With 64 KiB chunks a 512 KiB
   upload and download round-trip intact (probe 06: upload 3.3 s, download 1.8 to 1.9 s).
5. **The submitter's `HF_TOKEN` and `WANDB_API_KEY` reach the sandbox.** `iris.cluster.types.EnvironmentSpec` copies
   both from the submitting process into every job unless the job overrides them. The probe job set
   `HF_TOKEN=probe-not-a-secret` and the sandbox saw an 18-byte `HF_TOKEN` (probe 05, 06). A model-controlled
   sandbox must not receive the submitter's credentials.

## Proposed change (the patch)

- Poll readiness against `TaskState` and report `status.error_message`.
- `IrisMachineFactory(network: NetworkPolicy, ...)`, required: it declares the network the target cluster's gVisor
  profile gives every sandbox, because Iris cannot set it per job. `create` refuses any other policy with
  `UnsupportedMachineSpec`. For `marin` that is `NetworkPolicy.DENY`, which makes default TaskCompendium docker tasks
  runnable and makes `network: true` tasks fail up front instead of running without the network they asked for.
- Blank `HF_TOKEN` and `WANDB_API_KEY` in the sandbox job's environment.
- `TRANSFER_CHUNK_BYTES = 64 * 1024`.
- README: the network table row and the Iris paragraph.
- Tests: `create` waits through PENDING and BUILDING to RUNNING and blanks the submitter's key; a task that fails
  before running is reported and its job cancelled; a mismatched policy is refused; a 512 KiB file round-trips
  through an exec provider that enforces the 128 KiB argument limit. All four fail on the current code; the patched
  `tests/test_iris_machine.py` passes (7 tests).

Alternatives considered:

- Enforce DENY inside the sandbox: `unshare -rn` works in these gVisor sandboxes (exit 0, egress then unreachable, the
  filesystem shared), so each command could run in a user and network namespace. It is not needed on `marin`, where
  the profile already denies, and a per-command namespace would not share loopback between commands (a setup
  command's server would be invisible to later commands). A persistent namespace held by a sleeper process and
  entered with `nsenter` would fix that, at the cost of every command running as a mapped root that cannot `chown`
  to other users. Worth doing only for a cluster whose gVisor sandboxes have egress.
- Let RolloutEngine map DENY to ALLOW for Iris. Rejected: it would silently claim a property the engine does not
  check.

Taskforge, after this lands: `taskforge.sandbox.factories.machine_factories(MachineHost.IRIS)` passes
`network=NetworkPolicy.DENY`, and `IRIS_DOCKER` changes to `network={DENY}` and drops its `unavailable` reason. The
capability-table test in `tests/sandbox/test_factories.py` fails until that update is made.

## Measurements with the fix (probe 06, `iris_machine_probe-06.txt`)

The probe job ran on `marin`, pinned to `us-west4-a`; its sandboxes are child jobs, which Iris places in the parent's
zone. Image `docker.io/library/ubuntu:24.04`, 1 CPU, 2 GiB.

| What | Result |
|---|---|
| `create`, sequential, 6 sandboxes | 65.3, 39.7, 5.7, 14.4, 61.0, 21.8 s (probe 05: 20.8, 3.1, 8.5, 12.1, 4.7, 3.9 s) |
| `create`, 8 concurrent | 10.5 to 17.8 s, all succeeded (probe 05, 4 concurrent: 27.7 s each) |
| `run(("true",))` | 0.86 to 1.26 s; five `echo` runs 1.15 to 1.5 s. Each run is at least four exec RPCs (run, two downloads, cleanup). |
| `close` | 0.14 to 0.64 s |
| upload / download 512 KiB | 3.3 s / 1.8 to 1.9 s with 64 KiB chunks |
| `job_ttl=60` | Commands still succeeded at 90 s and 121 s after create; at 153 s the job was `failed` and `run` raised `RuntimeError('Iris exec failed: Task ... is not running (state=TASK_STATE_KILLED)')`. The TTL is enforced with one to two minutes of lag and surfaces as an untyped `RuntimeError`. |

Not fixed by the patch, and reported here for the Iris owners:

- **Placement.** Sandboxes placed on the v4 TPU workers in us-central2-b never start: `OCI runtime start failed:
  ... cannot run with network enabled in root network namespace` (probe 03, 6 of 6 sequential and 4 of 4
  concurrent creates, two different workers). Children inherit the parent's zone, and a sandbox-level zone
  constraint that disagrees is unschedulable (probe 04), so the caller has to place the parent job.
- **CoreWeave.** GVISOR-profile pods on `cw-us-east-02a` stayed in `PodInitializing` until the job timeout (20 min)
  and then failed without a reason (`explore/net_cw_gvisor*`, `explore/net2_cw_gvisor*`). The DEFAULT profile on the
  same cluster starts in under a minute, has egress and DNS, and carries the cluster's object-store keys
  (`AWS_SECRET_ACCESS_KEY`, `CW_KEY_SECRET`) in its environment (`explore/kbuild_cw.txt`, names only).
- **Private images.** A sandbox from the task registry (`envreg.208261-marin-gpu.coreweave.app/...@sha256:...`)
  fails at pull on the GCP workers: the registry requires authentication and Iris passes no pull credential
  (probes 03, 05, 06). Every Taskforge-built image lives there, so this blocks Iris execution of built tasks until
  Iris workers can authenticate to the registry or images are mirrored somewhere they can pull.
- A killed or expired sandbox raises a bare `RuntimeError`; a typed exception (machine gone) would let callers
  classify it without matching text.
