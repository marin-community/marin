# Iris direct-provider gang admission via Kueue (CoreWeave)

Status: design / not yet implemented
Author: design pass, 2026-05-29
Scope: `lib/iris` — the "direct" (Kubernetes) provider only. No change to the
worker-daemon (GCP) scheduling path.

## 1. Problem

Iris has two completely separate scheduling paths:

- **Worker-daemon path (GCP).** The controller's `Scheduler` does true
  atomic, all-or-nothing coscheduling in
  `scheduler.py::_find_coscheduled_assignments` — it finds one `group_by`
  worker group with ≥`num_tasks` workers that all have capacity and assigns
  every task at once, or assigns nothing.

- **Direct path (CoreWeave k8s).** `K8sTaskProvider` is a *direct provider*
  with **no worker rows** (`tasks.py` module docstring; `factory.py` returns
  `workers=None`). `transitions.py::drain_for_direct_provider` promotes up to
  `DIRECT_PROVIDER_PROMOTION_RATE = 128` PENDING tasks/min straight to
  ASSIGNED and emits a `DirectProviderBatch`; the provider `kubectl apply`s
  one independent `Pod` per task. Placement and capacity are delegated to the
  k8s scheduler + cloud autoscaler (see the comment at `transitions.py:273`).

The direct path therefore has **no atomic admission**: for a 64×8 job (64
pods, 8 GPUs each = 512 GPUs) the k8s scheduler can bind 63 pods and leave the
64th Pending behind the autoscaler. Any startup collective (`jax.distributed`
init, NCCL) blocks until the gang completes — or times out. The only gang
guarantee that exists today is on the *teardown* side: the coscheduled sibling
cascade in `transitions.py` (`_terminate_coscheduled_siblings` /
`_requeue_coscheduled_siblings`, invoked from both `_apply_task_transitions`
and `apply_direct_provider_updates`). Atomic *startup* and atomic *re-admission
on retry* are missing.

`_requeue_coscheduled_siblings`'s own docstring (`transitions.py:799`) names the
hazard: after a transient failure its retry "may place its retry on a different
slice from the still-RUNNING siblings, splitting the coscheduled job." On k8s
there is nothing preventing that split.

## 2. Decision: Kueue plain-pod-groups

CoreWeave CKS ships **Kueue** via its own Helm chart, with topology-aware
scheduling enabled by default. Kueue's *plain Pod group* integration is a
near-exact fit for Iris's model:

- It does **gang admission only**, via scheduling gates. Kueue's mutating
  webhook injects a `kueue.x-k8s.io/admission` scheduling gate on any pod that
  carries the queue-name label, and removes the gate only once the whole
  group's internal `Workload` is admitted. Gated pods sit `Pending /
  SchedulingGated` — the same Pending signal the autoscaler already keys on.
- It **does not recreate pods**: "Kueue does not recreate failed pods … when
  preempted, all group pods are deleted; external controllers must create
  replacements." That external controller is Iris. The whole
  `_terminate/_requeue_coscheduled_siblings` cascade stays as-is.

Rejected alternatives:

- **JobSet / k8s Job** — owns pod creation, retries, completion. Duplicates and
  conflicts with Iris's attempt model and the `_delete_stray_pods` pod-diff
  reconciler. No.
- **Volcano / SIG scheduler-plugins PodGroup CRD** — works as a pure admission
  gate, but is *not* what CoreWeave runs. Would mean operating a second
  scheduler ourselves. No.
- **Kueue Job integration (suspend/Workload)** — heavier; overlaps with Iris's
  own admission ordering. The *plain-pod-group* integration is the lifecycle-
  light subset we want.

References:
- https://docs.coreweave.com/products/cks/clusters/coreweave-charts/kueue
- https://kueue.sigs.k8s.io/docs/tasks/run/plain_pods/
- https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/

## 3. Attribute conversion: Iris → Kueue

Separate the two things "gang" means; they map to different keys.

| Iris concept | Kueue expression | Key |
|---|---|---|
| `num_tasks` | pod-group size | annotation `kueue.x-k8s.io/pod-group-total-count: "64"` |
| gang identity (per generation) | pod-group name | label `kueue.x-k8s.io/pod-group-name: iris-pg-{job_hash}-{attempt_id}` |
| target queue/quota | LocalQueue ref | label `kueue.x-k8s.io/queue-name: <pool-queue>` |
| `group_by` = hard physical unit (TPU slice, single NVLink domain) | **required** topology | annotation `kueue.x-k8s.io/podset-required-topology: <topology-label>` |
| `group_by` = soft locality (multi-node GPU spine) | **preferred** topology | annotation `kueue.x-k8s.io/podset-preferred-topology: <topology-label>` |
| `PriorityBand` (PRODUCTION/INTERACTIVE/BATCH) | WorkloadPriorityClass | label `kueue.x-k8s.io/priority-class: iris-{band}` |

`num_tasks` carries the all-or-nothing count. `group_by`'s *atomicity* role is
fully absorbed by `pod-group-total-count`; its only remaining job on k8s is
*locality*, expressed as topology because there are no Iris worker rows on this
path — pods bind to nodes, not workers.

CoreWeave topology label hierarchy (coarse → fine):
`backend.coreweave.cloud/fabric` → `backend.coreweave.cloud/superpod` →
`backend.coreweave.cloud/leafgroup` → `kubernetes.io/hostname`, plus
`ds.coreweave.com/nvlink.domain` for NVLink multi-node.

Hard vs soft:
- A 64-node GPU job will not fit one leafgroup → **preferred** topology at
  `leafgroup` (best-effort; Kueue falls back to a higher tier). This replaces
  the current soft `podAffinity` block.
- A single TPU slice / single NVLink domain must be one domain → **required**
  topology at `ds.coreweave.com/nvlink.domain`.

Mapping table in the provider:

```python
# group_by attribute -> (topology annotation key, node topology label, hard?)
_GROUP_BY_TO_TOPOLOGY: dict[str, tuple[str, str, bool]] = {
    "tpu-name": (_KUEUE_REQUIRED_TOPOLOGY,  "ds.coreweave.com/nvlink.domain",      True),
    "nvlink":   (_KUEUE_REQUIRED_TOPOLOGY,  "ds.coreweave.com/nvlink.domain",      True),
    "pool":     (_KUEUE_PREFERRED_TOPOLOGY, "backend.coreweave.cloud/leafgroup",   False),
}
```
Unknown `group_by` → no topology annotation (admission still gangs via the
total-count); fail loud only if a hard unit is misconfigured.

## 4. Code changes

### 4.1 Proto — plumb coscheduling + priority into `RunTaskRequest`

`RunTaskRequest` (`job.proto:591`) carries `num_tasks` and `constraints` but
not `group_by` or priority. The data already lives in `job_config_table`
(`has_coscheduling`, `coscheduling_group_by`, `priority_band` — see
`reads.py:477-490`). Add:

```proto
message RunTaskRequest {
  ...
  CoschedulingConfig coscheduling = 13;  // empty group_by => not coscheduled
  PriorityBand priority = 14;
}
```

### 4.2 Carry the columns into the dispatch row

`PendingDispatchRow` (`reads.py:76`) does not yet carry coscheduling/priority.
Add three fields and extend `_PENDING_DISPATCH_COLS` (`transitions.py:169`) +
`_pending_dispatch_row` (`transitions.py:188`) to select them from the
`job_config_table` join that `drain_for_direct_provider` already performs:

```python
@dataclass(frozen=True, slots=True)
class PendingDispatchRow:
    ...
    has_coscheduling: bool
    coscheduling_group_by: str        # "" when not coscheduled
    priority_band: int                # job_pb2.PriorityBand
```

Then in `_build_run_request` (`transitions.py`):

```python
if row.has_coscheduling:
    run_req.coscheduling.group_by = row.coscheduling_group_by
run_req.priority = row.priority_band
```

### 4.3 Pod manifest — add Kueue labels/annotations

In `_build_pod_manifest` (`tasks.py:385`), when
`run_req.coscheduling.group_by`:

```python
_KUEUE_POD_GROUP_NAME     = "kueue.x-k8s.io/pod-group-name"
_KUEUE_POD_GROUP_TOTAL    = "kueue.x-k8s.io/pod-group-total-count"
_KUEUE_QUEUE_NAME         = "kueue.x-k8s.io/queue-name"
_KUEUE_PRIORITY_CLASS     = "kueue.x-k8s.io/priority-class"
_KUEUE_REQUIRED_TOPOLOGY  = "kueue.x-k8s.io/podset-required-topology"
_KUEUE_PREFERRED_TOPOLOGY = "kueue.x-k8s.io/podset-preferred-topology"

def _pod_group_name(job_id: str, attempt_id: int) -> str:
    # Coscheduled siblings move in lockstep (the requeue path bumps every
    # sibling's attempt together), so attempt_id is the generation key:
    # a full-gang requeue -> new pod-group-name -> fresh atomic admission.
    return f"iris-pg-{_task_hash(job_id)}-{attempt_id}"
```

```python
if run_req.coscheduling.group_by:
    gen = attempt_id
    labels[_KUEUE_POD_GROUP_NAME] = _pod_group_name(run_req.task_id, gen)
    labels[_KUEUE_QUEUE_NAME] = config.local_queue          # new PodConfig field
    labels[_KUEUE_PRIORITY_CLASS] = _band_to_wpc(run_req.priority)
    annotations = metadata.setdefault("annotations", {})
    annotations[_KUEUE_POD_GROUP_TOTAL] = str(run_req.num_tasks)
    topo = _GROUP_BY_TO_TOPOLOGY.get(run_req.coscheduling.group_by)
    if topo is not None:
        anno_key, node_label, _hard = topo
        annotations[anno_key] = node_label
```

Notes:
- The existing soft `podAffinity` block (`tasks.py:524`) is **removed** —
  `podset-preferred-topology` supersedes it.
- `PodConfig` (`tasks.py:234`) gains `local_queue: str = ""`. The
  `colocation_topology_key` field is repurposed/retired in favor of the
  topology mapping above; set the CW config's value to a leafgroup label if we
  keep it for the non-coscheduled case.
- `_band_to_wpc` maps `PriorityBand` → `"iris-production" | "iris-interactive"
  | "iris-batch"` (UNSPECIFIED → interactive, matching the proto comment).

No PodGroup CRD is created and no `sync()` pre-pass / GC of PodGroups is
needed: Kueue synthesizes and GCs the `Workload` from the pod labels.

### 4.4 `drain_for_direct_provider` — gang-atomic promotion (the critical change)

Kueue only admits once it has observed **all** `pod-group-total-count` pods.
Today `drain_for_direct_provider` (`transitions.py:2904`) promotes a flat
`limit(128)` slice of PENDING rows with no gang awareness. If a gang is split
across drain cycles — or a gang larger than 128 never fits one cycle — Kueue
waits forever for pods Iris has not created, a deadlock.

Change the PENDING-promotion loop to be gang-aware:

1. Partition PENDING coscheduled rows by `(job_id, generation)`.
2. Promote a gang **all-or-none**: only promote a coscheduled group if the
   *entire* group fits in the remaining per-cycle budget. A single gang larger
   than `DIRECT_PROVIDER_PROMOTION_RATE` must be promoted whole — raise/relax
   the cap for that case (the cap exists only to bound API-server pressure, per
   `transitions.py:271`), or treat a coscheduled gang as one promotion unit
   and apply the rate limit per-gang rather than per-pod.
3. Non-coscheduled rows keep the existing flat first-fit promotion.

This mirrors, on the *promotion* side, the all-or-nothing logic that
`_find_coscheduled_assignments` already performs on the worker-path
*assignment* side.

### 4.5 `_poll_pods` / `_task_update_from_pod` — gate-awareness

- A Kueue-gated pod has `phase: Pending` (sub-condition `SchedulingGated`).
  `_task_update_from_pod` (`tasks.py`) already maps `Pending → BUILDING`, so a
  gang awaiting admission shows as BUILDING — correct, never a failure.
- The "Pod not found" grace path (`_poll_pods`, `tasks.py`) is unaffected:
  Iris creates the Pod object itself (delete-then-create in
  `service.py::_apply_pod`), so a gated pod is present in the list immediately;
  it never trips the not-found path.
- **Action item:** when Kueue *preempts* a gang it deletes all its pods. Those
  reads should land as `WORKER_FAILED` (preemption budget), not `FAILED`
  (application budget). Verify `_is_infrastructure_failure` /
  `_INFRASTRUCTURE_FAILURE_REASONS` (`tasks.py:99`) covers the
  `Preempting/Evicted/DeadlineExceeded` reasons Kueue/kubelet stamps on
  preempted gang pods; the not-found-after-grace path currently yields plain
  `FAILED` with `error="Pod not found"` — for a coscheduled task that
  disappeared due to gang preemption we likely want WORKER_FAILED. Decide and
  encode.

### 4.6 `activeDeadlineSeconds` interaction (gotcha)

`_build_pod_manifest` sets `spec.activeDeadlineSeconds` from the job timeout
(`tasks.py`). For a gated pod, k8s starts counting `activeDeadlineSeconds`
from pod creation, *including* time spent gated waiting for admission. A gang
that waits a long time for the autoscaler could hit `DeadlineExceeded` before
it ever runs. Mitigation: do **not** set `activeDeadlineSeconds` on
coscheduled pods (rely on the controller's own timeout accounting), or start
the timeout clock at RUNNING in the controller rather than via k8s. Decide:
prefer dropping `activeDeadlineSeconds` for coscheduled pods.

## 5. Cluster-side prerequisites (one-time, Helm/admin — not Iris code)

- **ResourceFlavor** per GPU type, labelled with the CW topology labels
  (`backend.coreweave.cloud/flavor`, `node.kubernetes.io/instance-type`).
- **ClusterQueue** with the quota and the `Topology` spec (TAS).
- **LocalQueue** per Iris namespace/pool → its name is `PodConfig.local_queue`.
- Three **WorkloadPriorityClass** objects: `iris-production`,
  `iris-interactive`, `iris-batch`.
- The CKS Kueue Helm chart, with TAS enabled (default).

These belong in cluster bootstrap config, not in the controller's RBAC
ensure-block, since they are cluster-scoped quota policy. (Open question 6.3.)

## 6. End-to-end flow after the change (64×8 job)

1. Job launched with `num_tasks=64`, `CoschedulingConfig(group_by=...)`,
   priority band. Persisted to `job_config_table`.
2. `drain_for_direct_provider` promotes the **whole 64-task gang** in one cycle
   (§4.4), each row → `RunTaskRequest` with `coscheduling` + `priority` (§4.1–2).
3. `sync()` applies 64 Pods, each labelled `pod-group-name=iris-pg-{h}-{0}`,
   `pod-group-total-count=64`, queue, priority-class, + topology annotation.
4. Kueue's webhook gates all 64 (`SchedulingGated`, phase Pending → Iris sees
   BUILDING). Kueue forms one `Workload`, size 64.
5. Kueue admits all-or-nothing when 64 slots (respecting TAS topology) are
   free; until then pods stay Pending → drives the CW autoscaler exactly as
   today. On admission Kueue removes the gates; all 64 bind together.
6. Pods run; `_poll_pods` reports RUNNING.
7. On a sibling failure, Iris's existing cascade fires: terminal → terminate
   all siblings; transient → requeue all siblings to PENDING with a bumped
   attempt → new `pod-group-name=...-{1}` → Kueue forms a fresh Workload and
   re-admits the gang atomically. Kueue never resurrects the old generation.
8. On job kill, Iris deletes the pods (pod-diff); Kueue marks the Workload
   finished/GC'd. (Optionally delete the Workload explicitly — open question.)

## 7. Open questions

6.1 **Workload deletion on hard kill.** When Iris kills a coscheduled job,
deleting pods may make Kueue see a failed group. Confirm whether Iris should
additionally delete the Kueue `Workload` object (would require registering the
`workloads.kueue.x-k8s.io` GVK in `K8sResource`), or whether pod deletion +
generation bump is sufficient. Lean: pod deletion suffices for terminal kills;
no Workload management needed.

6.2 **Preemption classification.** Finalize §4.5 — gang-preemption deletes →
WORKER_FAILED vs FAILED. Needs a test against a real Kueue preemption.

6.3 **Where queue/flavor/priority config lives.** ResourceFlavor/ClusterQueue/
LocalQueue/WorkloadPriorityClass ownership: Helm values vs an Iris-managed
bootstrap step. Recommend Helm/admin-owned; Iris only references `local_queue`
+ band names by string.

6.4 **`group_by` → topology label coverage.** Confirm the real CW label names
for the pools Marin uses (`backend.coreweave.cloud/leafgroup` vs `superpod`
granularity) and whether TPU is even on CW k8s or only GCP (if TPU is GCP-only,
the hard-topology branch may be dead on the direct path).

## 8. Test plan

- Unit: `_build_pod_manifest` emits the correct labels/annotations for
  coscheduled vs non-coscheduled, and the hard/soft topology branch per
  `group_by`. Assert on the structured manifest dict (not rendered strings).
- Unit: `drain_for_direct_provider` promotes a gang all-or-none; a gang larger
  than the per-cycle cap is still promoted whole; non-coscheduled promotion
  unchanged. Drive via the existing `transition_driver.py` test harness.
- Integration (fake k8s): a coscheduled job's pods all carry one
  `pod-group-name`; a sibling failure bumps the generation and the new pods
  carry the next `pod-group-name`. Reuse `providers/k8s/fake.py`.
- Manual on a CKS cluster with Kueue: verify gated→admitted transition, that
  Pending-gated reads as BUILDING, and preemption classification (6.2).
