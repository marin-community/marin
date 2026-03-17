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
