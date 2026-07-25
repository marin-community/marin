---
date: 2026-07-24
system: iris
severity: degraded
resolution: mitigated
pr: none
issue: none
---

# TL;DR

- A singleton 8×H100 inference task stayed in Iris `BUILDING` while its Kubernetes Pod remained `Pending`.
- Kueue rejected admission because none of 32 H100 nodes fit 48 CPU and 1 TiB memory: 13 nodes lacked CPU and 19 lacked memory.
- The cluster ran 1,031 CPU-only interactive Pods, one interactive GPU Pod, and 12 batch GPU Pods. The CPU-only token-count shards filled roughly 31 × 64 GiB on each of 19 H100 nodes.
- Iris stored only `SchedulingGated: Scheduling is blocked due to non-empty scheduling gates`. It could not associate a singleton Pod with its Pod-owned Kueue Workload.
- The working-tree fix indexes Kueue Workloads by Pod UID, surfaces active task diagnostics on the job page and `iris job summary`, and adds regression coverage. The live controller still needs a deploy.
- The job was later killed with `Terminated by user`; it never reached `RUNNING`.

# Original problem report

Job `/loom/eval-20260725-002116-grug-agentic-s3-step1903-4ee6/inference-875f534a0cc2410a9e028c15b5b4be05` showed one task in `BUILDING` on `cw-us-east-02a`. The job page showed no placement reason. The task requested 48 CPU, 1 TiB memory, 512 GiB disk, and 8 H100 GPUs.

# Investigation path

1. `iris --cluster=cw-us-east-02a cluster status` at 2026-07-24 17:27 PDT showed a healthy controller and the Kubernetes backend's expected `0/0` Iris workers.

2. `iris job summary` confirmed that the federated job existed on `cw-us-east-02a` and its only task was in `BUILDING`.

3. The controller task row carried the generic backend status `SchedulingGated: Scheduling is blocked due to non-empty scheduling gates`. Its current attempt identified Pod `iris-loom-eval-20260725-002116-grug-3daaa3c6-0-9b4d0a333d36978c`.

4. The Pod was `Pending`, had no node, and was owned by Kueue Workload `pod-iris-loom-eval-20260725-002116-grug-3daaa3c6-0-9b4d0a333d36978c-2a662`.

5. The Workload's `QuotaReserved=False` condition said: `couldn't assign flavors to pod set main: topology "infiniband" doesn't allow to fit any of 1 pod(s). Total nodes: 32; excluded: resource "cpu": 13, resource "memory": 19`.

6. One memory-blocked H100 node ran 31 token-count Pods, each requesting 2 CPU and 64 GiB memory. Cluster-wide active Iris Pods grouped into 1,031 CPU-only interactive Pods, one interactive GPU Pod, and 12 batch GPU Pods.

7. The target Pod used `iris-interactive` priority 10. Kueue ClusterQueue `iris-cq` had `withinClusterQueue: LowerPriority`; equal-priority token-count Pods were not preemption candidates.

8. `lib/iris/src/iris/cluster/backends/k8s/tasks.py:1423` resolved Workloads only through the gang `pod-group-name`. Kueue singleton Pods lack that label and link their generated Workload through the Pod UID instead.

9. `lib/iris/dashboard/src/components/controller/JobDetail.vue:273` and `lib/iris/src/iris/cli/job.py:1350` did not render `TaskStatus.status_message` on job-level surfaces. The task detail page did render it, but only after a user opened the individual task.

10. A final status check at 2026-07-24 17:43 PDT showed the job as `KILLED` with `Terminated by user`. Its only task never left the scheduling gate.

# User course corrections

- The user explicitly authorized direct `kubectl` probes after the controller-level status proved insufficient. This exposed the Kueue Workload condition that Iris omitted.
- The user requested an Iris code change once the missing scheduling feedback was isolated. The change stayed on reporting and attribution paths; it did not alter Kueue placement or preemption policy.

# Root cause

The immediate placement failure was real resource fragmentation. Kueue v0.18.0 could not find one H100 node with both 48 CPU and 1 TiB free. CPU-only interactive token-count Pods consumed nearly all memory on 19 H100 nodes. GPU work consumed enough CPU on the other 13 nodes to make the requested shape fail TAS admission.

The silent job page was an Iris attribution and presentation bug. Gang Pods carry `kueue.x-k8s.io/pod-group-name`, so Iris could look up their Workload by name. Singleton Pods carry only the queue label; Kueue links the generated Workload through `kueue.x-k8s.io/job-uid` and a Pod owner reference. The resolver at `lib/iris/src/iris/cluster/backends/k8s/tasks.py:1423` ignored those links. The job page and CLI summary then ignored the task-level backend status even when present.

# Fix

`lib/iris/src/iris/cluster/backends/k8s/tasks.py` now builds one Workload index by group name and Pod UID:

```python
return _KueueWorkloadIndex(by_name=by_name, by_pod_uid=by_pod_uid)
```

Singleton Pods resolve through `metadata.uid`; gang Pods retain their group-name path. Any Kueue-managed `SchedulingGated` Pod now receives the Workload's `QuotaReserved` verdict and emits an `iris.task_event` row from `k8s/kueue`.

`lib/iris/dashboard/src/components/controller/JobDetail.vue` groups identical active task diagnostics into a job-level warning and shows each task's backend status in the task list. `lib/iris/src/iris/cli/job.py` includes the same field in the `DIAGNOSTIC` column.

The live job remained governed by the existing queue policy. The code change did not preempt or stop workloads.

# How OPS.md could have shortened this

The `Pod stuck in Pending` section in `lib/iris/docs/coreweave.md` now starts with `iris task describe` and a narrow `QuotaReserved` JSONPath query. It maps `SchedulingGated` to Kueue admission and names topology resource exclusions as the signal to inspect.

The same section now warns that `kubectl describe pod` prints literal environment values. Narrow JSONPath queries preserve the scheduling evidence without exposing task credentials.

# Artifacts

- `.agents/ops/2026-07-24-iris-singleton-kueue-diagnostic-gap.md`
- `lib/iris/src/iris/cluster/backends/k8s/tasks.py`
- `lib/iris/dashboard/src/components/controller/JobDetail.vue`
- `lib/iris/src/iris/cli/job.py`
- Kueue Workload `iris/pod-iris-loom-eval-20260725-002116-grug-3daaa3c6-0-9b4d0a333d36978c-2a662`
