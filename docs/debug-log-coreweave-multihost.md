# Debugging log for CoreWeave multi-host GPU canary

Validate that the `scripts/canary/coreweave_multihost.py` automation script
loads, all imports resolve, and the full test suite passes after adding
multi-host GPU coscheduling support. Then run the real canary on CoreWeave.

## Process notes

- Don't run the full test suite speculatively. Tests are expensive and slow.
  Only run tests when there's a specific regression to verify.
- Always update this log before and after each action.

## Initial status

All code changes implemented across 8 files (3 modified, 5 new). Pre-commit
lint initially failed on one issue.

## Hypothesis 1: Ruff F821 on forward-reference annotation

`resolve_coscheduling` in `iris_backend.py` used a string annotation
`"CoschedulingConfig | None"` for its return type. The file already has
`from __future__ import annotations`, so the string was valid at runtime, but
ruff's F821 check still validates that the name is importable at module scope.

### Changes made

- Added `CoschedulingConfig` to the top-level import from `iris.cluster.types`
  in `iris_backend.py` (line 34).
- Removed the now-redundant local `from iris.cluster.types import
  CoschedulingConfig` inside `resolve_coscheduling`.

### Results

Pre-commit passes cleanly after this fix.

## Hypothesis 2: API mismatches in automation script

The script uses several internal Iris APIs (`_configure_client_s3`,
`_pin_latest_images`, `_build_cluster_images`, `IrisConfig.load`,
`platform.start_controller`, `platform.stop_all`, `platform.shutdown`,
`platform.discover_controller`, `platform.tunnel`). Verified all exist with
matching signatures via codebase search.

### Results

- `uv run python scripts/canary/coreweave_multihost.py --help` → CLI loads OK
- `uv run python -c "from scripts.canary.coreweave_multihost import cli"` → all imports resolve

## Hypothesis 3: kubectl missing --kubeconfig

`validate_prereqs` ran `kubectl cluster-info` without `--kubeconfig`, so it
used the default `~/.kube/config` instead of `~/.kube/coreweave-iris`. Same
issue in `_kubectl` helper — all kubectl calls need the explicit kubeconfig
path.

### Changes made

- Extracted `_kubeconfig_path()` helper that reads `KUBECONFIG` env var with
  fallback to `~/.kube/coreweave-iris`.
- Added `--kubeconfig` to `_kubectl()` helper and `validate_prereqs()`.

### Results

- `uv run python scripts/canary/coreweave_multihost.py validate` → "All prereqs OK"

## Hypothesis 4: Shared namespace with single-host canary

The script and single-host canary used the same `iris-canary` namespace/label.
Concurrent runs or teardown would interfere.

### Changes made

- Created dedicated `lib/iris/examples/coreweave-canary-multihost.yaml` with
  namespace `iris-canary-mh`, label prefix `iris-canary-mh`, separate state dir
  and controller service name.
- Removed h100-16x scale group from `coreweave-canary.yaml` (belongs only in
  multihost config).
- Updated script defaults and workflow to use new config.

### Results

Validate passes with the new config. Namespace is isolated.

## Attempt 1: Real CW run — R2 credentials wrong

**Started**: 2026-03-17 ~12:10 UTC

- Cold boot triggered (no existing controller in `iris-canary-mh`).
- Docker images built and pushed.
- Controller pod CrashLoopBackOff: `PermissionError: Forbidden` on
  `s3://marin-na/iris/state/canary-multihost`.
- Root cause: local env had stale R2 key (`74ef29b1...`) instead of the
  correct one (`ff69fd...`). The K8s secret was created with the wrong key.
- Fix: `source ~/.env`, delete stale secret, reboot cluster.

## Attempt 2: Real CW run — label too long

**Started**: 2026-03-17 ~20:27 UTC

- Warm reboot succeeded. Controller up with correct R2 creds.
- Job submitted: `multihost-canary-mh-canary-20260317-202706`.
- Worker ConfigMap creation failed:
  ```
  metadata.labels: Invalid value:
  "iris-canary-mh-iris-canary-mh-cpu-erapids-20260317-2027-31f6be45":
  must be no more than 63 bytes
  ```
- Root cause: pre-existing bug in `coreweave.py:648`. `prepare_slice_config()`
  sets `config.name_prefix = "{label_prefix}-{sg_name}"`, then `create_slice()`
  prepends `self._label_prefix` again:
  ```python
  slice_id = f"{self._label_prefix}-{scale_group_name}-{suffix}"
  # becomes: "iris-canary-mh-iris-canary-mh-cpu-erapids-..."  (64 bytes, over 63 limit)
  ```
  The old single-host canary happened to fit (54 bytes) so the bug was latent.

### Changes made

- `lib/iris/src/iris/cluster/platform/coreweave.py:648`: removed the redundant
  `self._label_prefix` prefix from `slice_id`:
  ```python
  slice_id = f"{scale_group_name}-{generate_slice_suffix()}"
  ```

## Attempt 3: Real CW run — ghost slice from stale checkpoint

**Started**: 2026-03-17 ~20:32 UTC

- Warm reboot with label fix. Controller started, job submitted with 2 tasks.
- Controller adopted stale slice `iris-canary-mh-iris-canary-mh-h100-16x-...`
  from its S3 checkpoint (leftover from attempt 2's failed state).
- Controller thought h100-16x already had 1 slice booting, so the autoscaler
  never created a new one. 2 GPU tasks stuck pending indefinitely.

### Changes made

- Added `clear_controller_state(config_path)` to the automation script. It
  deletes the S3 checkpoint dir before each warm reboot so the controller
  starts fresh with no ghost slices.
- Called from `warm_reboot()` after deleting worker pods but before re-creating
  the controller.

## Attempt 4: Real CW run — S3 client misconfigured + wrong delete label

**Started**: 2026-03-17 ~20:41 UTC

Two issues found:

1. **R2 S3 client 400 Bad Request**: `clear_controller_state` used `fsspec.core.url_to_fs()`
   which created an S3FileSystem without R2's required `region_name='auto'` and
   explicit credentials. Fixed by creating `_make_s3fs()` that passes
   `key`, `secret`, `client_kwargs={"region_name": "auto"}`, and `skip_instance_cache=True`.

2. **Wrong label in warm_reboot**: Was deleting pods with label
   `{label_prefix}-role=worker` = `iris-canary-mh-role=worker`, but the actual K8s
   labels are `iris-{label_prefix}-managed=true` = `iris-iris-canary-mh-managed=true`.
   Old worker pods survived warm_reboot, and the controller adopted them as ghost slices.
   Fixed label and also delete configmaps.

3. **Verbose logging too noisy**: `-v` flag set `logging.DEBUG` globally, flooding
   output with botocore/s3fs internals. Changed to only set DEBUG on `iris.*` and
   script loggers.

### Changes made

- `scripts/canary/coreweave_multihost.py`: `_make_s3fs()` with R2-compatible config
- `scripts/canary/coreweave_multihost.py`: fixed managed label in `warm_reboot()`
- `scripts/canary/coreweave_multihost.py`: scoped verbose logging to iris loggers

## Attempt 5: Real CW run — after cleanup

**Started**: 2026-03-17 ~20:52 UTC

NodePools and nodes are already provisioned from previous attempts. Both
cpu-erapids and h100-16x nodes are Ready. Manually cleaned stale pods/configmaps.
Re-running with fixed script.

## Hypothesis 6: CoreWeave recovery and multi-VM slice semantics are mismatched

The repeated "ghost slice" failures exposed a deeper recovery problem than the
checkpoint cleanup bug alone. Iris recovery assumes a slice is a durable
platform object that should be rediscovered and adopted after controller
restart. On CoreWeave, the current implementation is a hack: Iris creates a
single synthetic "slice" handle, then bootstraps `num_vms` Kubernetes worker
Pods behind it. That abstraction is close enough for single-VM groups, but it
breaks down for multi-VM groups because recovery, autoscaling, and
coscheduling all reason about the synthetic slice instead of the actual Pods.

### Recovery problem

- Controller restart can adopt stale slice state from checkpoint or leftover
  worker Pods/configmaps even when the synthetic multi-VM slice was only
  partially created.
- The autoscaler then believes a full `num_vms=2` slice is already inflight,
  so it refuses to provision replacement capacity.
- This makes recovery brittle: one bad partial slice can block the entire
  canary indefinitely until state is manually scrubbed.

### Senior engineer investigation

A focused code review of the scheduling path found that the canary currently
depends on two coupled assumptions that are probably wrong for CoreWeave:

1. Multi-host GPU jobs are auto-coscheduled by `pool`.
2. The CoreWeave H100 scale group models a 2-host allocation as one
   `num_vms=2` Iris slice.

That forces Iris to treat CoreWeave like TPU multislice. But CoreWeave is
already provisioning ordinary Kubernetes nodes/Pods, and the synthetic
multi-VM slice adds recovery and scheduling state that the platform does not
really own. The next experiment is to remove the multi-VM slice model for this
canary: configure `h100-16x` as single-VM slices with `max_slices=2`, stop
auto-coscheduling GPU replicas, and let CoreWeave provision two arbitrary H100
workers.

### Changes to make

- Reconfigure the multihost canary scale group to use one VM per slice and
  allow two slices.
- Remove automatic GPU coscheduling in the Fray/Iris submission helpers so the
  canary submits `replicas=2` without a `pool` gang constraint.
- Update CoreWeave docs/tests to match the new scheduling model.

### Expected result

If the hypothesis is correct, the canary should schedule as two independent
single-worker slices. That should eliminate the synthetic multi-VM recovery
path and avoid the "one stale slice blocks all progress" failure mode.

### Changes made

- `lib/iris/examples/coreweave-canary-multihost.yaml`: changed `h100-16x` from
  `num_vms=2, max_slices=1` to `num_vms=1, max_slices=2`.
- `lib/fray/src/fray/v2/iris_backend.py`: removed automatic GPU
  coscheduling-by-pool for multi-replica jobs.
- `lib/iris/src/iris/cli/job.py`: removed the CLI helper's automatic GPU
  coscheduling default.
- Updated the matching regression tests and CoreWeave docs.

### Results

- `uv run pytest lib/fray/tests/test_iris_backend_coscheduling.py` → 5 passed
- `uv run pytest lib/iris/tests/cli/test_job_multinode.py` → 5 passed
- `uv run python scripts/canary/coreweave_multihost.py --help` → CLI still loads

Next step: restart the real canary with the new single-VM-slice configuration
and observe whether the controller provisions two independent H100 workers.

## Hypothesis 7: Keep Iris multi-host slices, but make CoreWeave topology explicit

The "delete the 16x path" experiment was useful for isolating the recovery bug,
but it is not the right long-term design. Iris still wants `num_vms`-sized
gangs for 16x/32x/64x jobs, and CoreWeave can support that if each sub-VM is a
real worker pod with explicit InfiniBand-topology placement.

### Design decision

- Keep CoreWeave multi-host groups as `num_vms > 1` slices.
- Keep one worker process per sub-VM / pod.
- Keep coscheduling for these GPU multi-host jobs.
- Express CoreWeave topology intent through reserved `worker.attributes` keys
  rather than introducing a new proto/config surface right now.
- Use the sentinel value `same-slice` for CoreWeave topology labels to mean:
  "schedule the leader pod first, read its bound node's label value, then pin
  follower pods to that same value."

### Changes to make

- Restore GPU coscheduling defaults for multi-replica jobs.
- Add config validation so CoreWeave GPU groups with `num_vms > 1` must set at
  least one CoreWeave topology label in `worker.attributes`.
- Update the CoreWeave platform so multi-host slices:
  - create one pod/worker per sub-VM,
  - schedule the first pod,
  - read its node labels,
  - schedule follower pods with matching topology selectors.
- Update docs and canary config to use the multi-host topology contract.

### Changes made

- Restored GPU `pool` coscheduling in the Iris CLI and Fray backend so
  multi-host CoreWeave GPU jobs still launch as gangs.
- Added CoreWeave config validation: GPU groups with `num_vms > 1` now require
  at least one topology label in `worker.attributes`.
- Chose the contract `backend.coreweave.cloud/superpod: same-slice` for the
  canary and example configs.
- Updated `CoreweavePlatform` multi-host slice creation to:
  - launch one worker pod per sub-VM,
  - schedule the leader first,
  - read the bound node's topology labels,
  - pin follower pods to the discovered topology domain.
- Extended the fake kubectl test harness with node objects and pod-to-node
  binding so the topology discovery path is covered in unit tests.
- Updated the CoreWeave docs to describe the new `same-slice` multi-host
  topology contract.

### Results

- `uv run pytest lib/iris/tests/cluster/platform/test_coreweave_platform.py -k 'multi_node or node_selector'`
  → 6 passed
- `uv run pytest lib/iris/tests/cluster/platform/test_config.py -k 'coreweave_multihost_gpu or worker_settings_validation'`
  → 2 passed
- `uv run pytest lib/fray/tests/test_iris_backend_coscheduling.py lib/iris/tests/cli/test_job_multinode.py`
  → 10 passed

The implementation now matches the intended design: CoreWeave keeps Iris
multi-host slices, but each sub-VM is an explicit worker pod and follower pods
inherit an IB-topology selector discovered from the leader's bound node.

### Historical note

The previous temporary version also worked in a narrower sense: changing the
canary to single-VM slices and removing GPU coscheduling avoided the synthetic
multi-VM recovery failure mode. We are not keeping that design because it gave
up the 16x/32x/64x Iris slice contract. The current implementation preserves
the gang semantics and adds explicit topology-aware worker placement instead of
removing multi-host slices.

## Attempt 6: Re-test, commit, push, then re-run the real canary

**Started**: 2026-03-17 local evening

- Re-ran the focused regression tests after the topology-aware multi-host
  implementation landed.
- Next actions: commit the task-related changes on the feature branch, push,
  then restart the real CoreWeave canary to see whether the old behavior can be
  replicated or whether the topology-aware implementation clears it.

### Results

- Commit created: `6ee2576f7` (`Add topology-aware CoreWeave multihost slices`)
- Branch pushed: `origin/rjpower/20260317-coreweave-multi-host`
- Pre-commit hooks run during `git commit` all passed, including pyrefly.

Next step: restart `scripts/canary/coreweave_multihost.py -v run` from the
pushed code and inspect whether the previous failure mode reproduces.

## Hypothesis 8: JAX bootstrap env is wrong on the live run

The next likely failure mode is no longer slice placement; it is JAX bootstrap
configuration inside the task. The previous run appears to have missed or
mis-set one or more JAX/Iris distributed env vars. Add targeted logging in
`jax_init.py` so the next live run shows the exact bootstrap inputs before
calling `jax.distributed.initialize()`.

### Changes to make

- Log the relevant Iris/JAX env vars at JAX init time.
- Log the resolved `job_info` values used to choose coordinator, task index,
  and task count.
- Re-run the existing `jax_init` tests to verify the instrumentation does not
  change behavior.

### Changes made

- Added targeted bootstrap logging to `lib/iris/src/iris/runtime/jax_init.py`.
- The log now records:
  - the Iris/JAX env vars present at init time,
  - resolved `job_info.task_index`,
  - `job_info.num_tasks`,
  - `job_info.advertise_host`,
  - the allocated JAX ports map.

### Results

- `uv run pytest lib/iris/tests/test_jax_init.py` → 8 passed

This is instrumentation only; it does not change the coordinator registration
or follower polling flow.

## Attempt 7: Real CW run with JAX bootstrap logging

**Started**: 2026-03-17 local evening

- Restarted `source ~/.env && uv run python scripts/canary/coreweave_multihost.py -v run`
  to capture the new `initialize_jax` bootstrap logs on a live job.
- The run did not reach controller rollout or task startup.

### Results

- Failure occurred during the local Docker build for `iris-worker:31d253b42`.
- The build failed in the Dockerfile step that installs `docker-ce-cli` and
  `docker-buildx-plugin`:
  `E: ... You don't have enough free space in /var/cache/apt/archives/`
- Because the image build failed, the canary never reached the JAX
  initialization path, so there are no new live `jax_init` logs yet.

### New blocker

- Local Docker builder disk space is exhausted. This must be cleared or the
  image build path must be avoided before we can continue reproducing the live
  JAX/bootstrap behavior.

## Attempt 8: Clear Docker cache, rerun canary, inspect live capacity behavior

**Started**: 2026-03-17 local evening

- Cleared old Docker images and builder state with `docker system prune -af`.
- Restarted the real CoreWeave multihost canary.
- The rerun progressed past image build, pushed fresh worker/task images,
  started the CoreWeave controller, and submitted the canary job successfully.
- While watching the live cluster state, noticed the `h100-16x` scale group was
  blocked at `max_slices` even though the multi-host slice had `num_vms: 2` and
  no workers had come up yet.

### Results

- The rerun disproved the prior local blocker: Docker disk exhaustion was not
  the root cause of the live CoreWeave issue.
- Live `NodePool` apply logs showed:
  - `h100-16x` configured with `num_vms: 2`
  - `NodePool iris-canary-mh-h100-16x` applied with `targetNodes: 1`
  - `NodePool iris-canary-mh-h100-16x` applied with `maxNodes: 1`
- That is incorrect for a multihost group. The shared NodePool is sized in
  nodes, not slices, so a single 2-VM slice requires 2 nodes of capacity.
- This explains the UI behavior:
  - Iris believed the one allowed slice was already the cap (`max_slices: 1`)
  - CoreWeave could only ever provision one node
  - the 2-worker slice therefore remained infeasible and showed `0 vms`

## Hypothesis 9: Shared CoreWeave NodePool sizing still uses slice counts

The topology-aware worker creation path was correct, but the shared NodePool
reconciliation still mapped `min_slices`/`max_slices` directly to
`minNodes`/`maxNodes`. That works for single-VM groups and is wrong for any
CoreWeave group with `num_vms > 1`.

### Fix

- Keep Iris slice accounting unchanged.
- Size shared CoreWeave NodePools in nodes:
  - `minNodes = min_slices * num_vms`
  - `maxNodes = max_slices * num_vms`

### Changes made

- Updated `lib/iris/src/iris/cluster/platform/coreweave.py` so
  `ensure_nodepools()` multiplies node counts by `slice_template.num_vms`.
- Added a regression test in
  `lib/iris/tests/cluster/platform/test_coreweave_platform.py` asserting that a
  group with `num_vms: 2` and `max_slices: 1` produces `maxNodes: 2`.

### Results

- `uv run pytest lib/iris/tests/cluster/platform/test_coreweave_platform.py -k 'ensure_nodepools or multi_node or node_selector'`
  → 13 passed

Next step: rerun the real canary from this patched code and confirm the
`h100-16x` NodePool comes up with 2 nodes of capacity, then continue to the
JAX bootstrap logs if the workers start.

## Hypothesis 10: Adopted slices stay stale after controller restart

The CPU worker looked healthy in Fleet after the controller restart, but the
autoscaler UI still showed the adopted CPU slice as `0 vms` / `unknown` /
booting. That suggested the naming was fine and the restore path was failing to
reconcile adopted cloud slices back into their live lifecycle and worker list.

### Changes made

- Updated `lib/iris/src/iris/cluster/controller/scaling_group.py` so unknown
  adopted cloud slices call `describe()` during restore.
- Mapped the returned `CloudSliceState` into autoscaler lifecycle rather than
  defaulting every adopted slice to `BOOTING`.
- Seeded `worker_ids` from the live cloud status during adoption.
- Added restore/reconciliation coverage in
  `lib/iris/tests/cluster/test_snapshot_reconciliation.py`.

### Results

- `uv run pytest lib/iris/tests/cluster/test_snapshot_reconciliation.py`
  → 15 passed
- `uv run pytest lib/iris/tests/cluster/controller/test_autoscaler.py -k 'refresh or restore'`
  → 1 passed

This fix addresses the stale CPU slice view after warm controller restarts.

## Hypothesis 11: Levanter is skipping Iris JAX bootstrap entirely

After multihost placement converged, the live training logs still showed
Levanter skipping distributed initialization:

- `levanter.distributed Not initializing jax.distributed because no distributed config was provided, and no cluster was detected.`

That meant the workers were running as independent single-process jobs rather
than joining one distributed JAX job, even though Iris had launched a 2-worker
multihost slice.

### Changes made

- Updated `lib/levanter/src/levanter/distributed.py` so when Levanter would
  otherwise skip distributed init, but an Iris job context is present, it calls
  `iris.runtime.jax_init.initialize_jax()` directly.
- Added a focused regression test in
  `lib/levanter/tests/test_distributed.py`.

### Results

- `uv run pytest lib/levanter/tests/test_distributed.py` → 2 passed
- `uv run pytest lib/iris/tests/test_jax_init.py` → 8 passed

This should make Iris multihost Levanter jobs run the same JAX bootstrap path
we already instrumented in `jax_init.py`.

## Attempt 9: Real rerun with patched Levanter distributed init

**Started**: 2026-03-17 local evening

- Rebuilt and pushed fresh images with tag `36cdd13f7`.
- Warm-restarted the CoreWeave canary controller and resubmitted the canary.
- Confirmed the H100 multihost scale group now applies a NodePool with
  `maxNodes: 2` and later scales to `targetNodes: 2`.

### Results so far

- The controller created the expected multihost slice:
  `iris-canary-mh-h100-16x-20260317-2216-970fcb0d`.
- The leader pod `vm0` registered successfully and the controller discovered
  `backend.coreweave.cloud/superpod=2` on its node.
- The follower pod `vm1` was created with the expected selector:
  `iris-iris-canary-mh-scale-group=h100-16x`,
  `backend.coreweave.cloud/superpod=2`.
- CoreWeave autoscaling triggered as expected:
  `TriggeredScaleUp ... 1->2 (max: 2)`.
- Live NodePool state confirms we are waiting on infrastructure capacity, not
  an Iris sizing bug:
  - `targetNodes: 2`
  - `currentNodes: 1`
  - `inProgress: 1`

### New findings

- The top-level canary task reached ferry dispatch, then logged:
  `Job grug-train-mh-canary-20260317-221614 already exists, adopting existing job`
- Controller logs show the child multihost Iris job was already submitted with
  2 tasks at `22:16:52`, and later the duplicate launch attempt failed with:
  `already exists and is still running`
- At the same time, controller status remained:
  `2 workers (0 failed), 1 active jobs, 2 pending tasks`
- There are no H100 task pods yet; only the top-level CPU canary task pod is
  running. That means we still have not reached the Levanter training worker
  logs for this rerun.

### Current blocker

- We are now blocked on the second H100 worker pod. `vm1` is still pending
  because it requires a second H100 node in the same discovered
  `backend.coreweave.cloud/superpod=2` domain, and the existing node already
  uses host port `10001` for `vm0`.
- Until `vm1` becomes schedulable and the child Iris job can start its 2 H100
  tasks, we cannot yet verify the patched Levanter JAX bootstrap behavior on
  the live run.

## Hypothesis 12: Fast GPU node loss came from desired capacity collapsing back to 1

The fast H100 host loss did not match a normal nodepool idle timeout. The old
multihost pool had been reconciled with a steady-state `targetNodes: 1`, and
the second node only existed because the pending follower pod triggered a
temporary autoscaling bump to `1->2`. When we killed the worker pods during
restart, that temporary pressure disappeared, so the extra H100 node became
excess capacity and CoreWeave could remove it quickly without waiting for an
idle-node timer.

### Changes made

- Updated `lib/iris/src/iris/cluster/platform/coreweave.py` so existing
  NodePools keep one full slice worth of nodes warm instead of always clamping
  `targetNodes` back to `1`.
- For CoreWeave multihost groups this means:
  - `num_vms: 2` keeps `targetNodes: 2`
  - single-host groups still keep `targetNodes: 1`
- Added regression coverage in
  `lib/iris/tests/cluster/platform/test_coreweave_platform.py` for an existing
  `num_vms: 2` pool reconciling to `targetNodes: 2`.

### Expected result

- During the next restart / churn cycle, the H100 NodePool should retain
  desired capacity for one whole multihost slice instead of shrinking back to a
  single node between follower reschedules.

## Hypothesis 13: The second H100 task attempt is churning pod identities

After both H100 workers came up, the multihost job still did not converge.
The Task History UI showed repeated `BUILDING` entries for logical task `/1`.
Live Kubernetes and worker logs confirm that this is real pod churn, not just
stale UI state.

### Results

- Current live state has:
  - one H100 training task pod `Running`: `iris-task-52fba4efea4e`
  - one H100 training task pod `Pending`: `iris-task-8fc01c138a66`
- Worker `vm1` previously created `iris-task-d6c491a20ad1` for logical task
  `/power/multihost-canary-mh-canary-20260317-221614/grug-train-mh-.../1`
- That pod later disappeared, and the same worker created
  `iris-task-8fc01c138a66` for the same logical task
- The worker log repeatedly reports:
  - `Started Kubernetes task pod iris-task-...`
  - followed by repeated `kubectl ... get pod iris-task-...`
  - followed by `Error from server (NotFound): pods "iris-task-..." not found`

### Interpretation

- The bug is no longer image pull or GPU node provisioning.
- The second H100 task attempt is losing its Kubernetes pod identity and being
  recreated under a fresh random pod name for the same logical Iris task.
- That explains the repeated `BUILDING` rows for task `/1`: they are separate
  task pod attempts, not one stable pod slowly booting.

### Next step

- Inspect the Kubernetes runtime / `TaskAttempt` path to understand why a
  `NotFound` from polling causes a fresh pod to be created for the same logical
  task instead of stabilizing on the existing attempt.

## Attempt 10: Pin task pods to the owning worker node

The multihost task-pod manifest was missing any node binding, so Kubernetes was
free to place a worker-owned task pod on any compatible H100 node. That is the
wrong contract for the Kubernetes runtime: once a task is assigned to a worker,
the task pod should run on that worker's node.

### Changes made

- Updated `lib/iris/src/iris/cluster/runtime/kubernetes.py` so
  `KubernetesRuntime` captures `IRIS_WORKER_NODE_NAME` from the worker pod
  environment and passes it into each `KubernetesContainerHandle`.
- When present, the task pod manifest now sets `spec.nodeName` to that worker
  node.
- Added regression coverage in
  `lib/iris/tests/cluster/runtime/test_kubernetes_runtime.py` asserting that a
  task pod inherits `nodeName=gd92f4a` when `IRIS_WORKER_NODE_NAME=gd92f4a`.

### Expected result

- Task `/0` launched by worker `vm0` will run on `vm0`'s node.
- Task `/1` launched by worker `vm1` will run on `vm1`'s node.
- The scheduler no longer has to solve a generic cluster-wide `8 GPU` placement
  problem for worker-owned task pods.

### Verification

- `uv run pytest lib/iris/tests/cluster/runtime/test_kubernetes_runtime.py -k 'worker_node or gpu_resources or advertise_host or pod_not_found'`
  → 6 passed
- `uv run pytest lib/iris/tests/cluster/platform/test_coreweave_platform.py -k 'ensure_nodepools or multi_node or node_selector'`
  → 14 passed

## Attempt 11: Fix canary teardown so it actually deletes NodePools

After the end-of-day teardown, the controller resources were gone but the
managed CoreWeave NodePools were still present. The teardown implementation was
issuing `kubectl delete nodepool -l <label-prefix>-managed=true`, which does not
match the actual managed label shape used everywhere else:
`iris-<label-prefix>-managed=true`.

### Changes made

- Updated `scripts/canary/coreweave_multihost.py` so `teardown_cluster()`
  deletes NodePools using the correct managed label selector:
  `iris-{label_prefix}-managed=true`.

### Expected result

- `uv run python scripts/canary/coreweave_multihost.py -v teardown` will now
  remove the managed H100 and CPU NodePools instead of leaving them running.
