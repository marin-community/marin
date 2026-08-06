# Background Research Brief

- Effort: medium
- Stop rule: stop when another Kubernetes mechanism no longer changes the ranking of hard isolation, borrowable capacity, and scheduler customization
- Date: 2026-08-06

## Question

Can Kubernetes keep CPU-only work from stranding free CoreWeave accelerators while still allowing that work to borrow otherwise-idle CPU on accelerator nodes?

## Current Marin Context

Iris lowers CPU and GPU requirements independently. CPU becomes a normal Pod request, while a GPU becomes an `nvidia.com/gpu` limit and implicit request ([`tasks.py`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L814-L849)). CPU-only Pods may run on any compatible CoreWeave node. Every Iris Pod now passes through the common topology-aware `cw-tas` ResourceFlavor, so lower-priority CPU reservations are visible to Kueue preemption ([`coreweave.md`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/docs/coreweave.md#L153-L180)).

The remaining gap is equal priority. Iris maps the three user bands to Kubernetes priorities `0`, `10`, and `1000` ([`types.py`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/src/iris/cluster/platforms/k8s/types.py#L19-L34)). Kueue's `withinClusterQueue: LowerPriority` policy can reclaim only lower-priority Workloads ([`kueue_manifests.py`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py#L288-L318)). A GPU job and CPU-only spillover job in the same Iris band are therefore peers even when the CPU reservation leaves an H100 idle.

The hardware has room for borrowing. RNO2A and US-EAST-02A H100 nodes expose 128 CPU and eight H100s, or 16 CPU per GPU ([`cw-rno2a.yaml`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/config/cw-rno2a.yaml#L185-L205)). A training request of four CPU per GPU reserves 32 CPU on a full eight-GPU node and leaves roughly 96 CPU for other Pods. A hard GPU-node taint would give up that headroom.

The report has three materially different explanations:

1. Lower-band CPU Workloads blocked the GPU Workload. This should already be recoverable after #7928, so it points to rollout state or a regression.
2. Same-band CPU-only Workloads blocked the GPU Workload. Kueue cannot select equal-priority victims; this is the case the proposed policy addresses.
3. Higher-band or accelerator Workloads blocked the GPU Workload. That follows the current priority contract and should not be overridden by a CPU-per-accelerator rule.

The affected cluster, Workload IDs, and time window are not yet available, so the recommendation remains conditional on confirming the second case.

## Internal Prior Work

[Issue #7916](https://github.com/marin-community/marin/issues/7916) measured the related lower-priority failure. Sixty-three RNO2A H100 nodes had fewer than 48 unreserved CPU cores even though batch Pods held 6,248 CPU cores. [PR #7928](https://github.com/marin-community/marin/pull/7928) and [Echo wiki 80](https://echo.oa.dev/wiki/80) moved CPU-only and GPU Workloads into one TAS flavor so Kueue could model lower-priority CPU reservations as removable. That change deliberately retained CPU spillover.

The current priority model is already expressed at both layers. Pods receive a Kubernetes `PriorityClass`, and Kueue derives Workload priority from it unless Iris stamps an explicit `WorkloadPriorityClass` ([`tasks.py`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L899-L912)). `KueueConfig.priority_classes` and `PodConfig.kueue_priority_classes` already carry an optional band-to-WorkloadPriorityClass map, although no production config uses it ([`config.py`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/src/iris/cluster/config.py#L619-L635)).

## External Prior Art

Kubernetes documents hard taints as the standard way to keep ordinary Pods off nodes with special hardware. A matching toleration admits only the hardware-using Pods ([Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/#example-use-cases)). This gives a hard guarantee but reserves the whole node, not a CPU quantity per accelerator.

Kubelet `kubeReserved`, `systemReserved`, and `reservedSystemCPUs` reserve resources for Kubernetes and OS daemons. They reduce Node Allocatable for every Pod and cannot reserve CPU for GPU workloads ([Reserve Compute Resources for System Daemons](https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/)).

Kubernetes Pod priority evicts only lower-priority Pods when removal would make the pending Pod fit ([Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)). Kueue similarly allows an incoming Workload to preempt a lower-priority Workload in the same ClusterQueue ([Kueue v0.18 preemption](https://kueue.sigs.k8s.io/v0.18/docs/concepts/preemption/)). A Kueue `WorkloadPriorityClass` is independent of Pod priority and is used for queue order and Workload-level preemption ([Kueue v0.18 WorkloadPriorityClass](https://kueue.sigs.k8s.io/v0.18/docs/concepts/workload_priority_class/)). This separation permits a one-point accelerator preference inside each Iris band without changing kube-scheduler's Pod-level band semantics.

Kubernetes also documents low-priority placeholder Pods for reserving a numeric amount of capacity ([Overprovision Node Capacity](https://kubernetes.io/docs/tasks/administer-cluster/node-overprovisioning/)). It conflicts with the current Kueue TAS path: TAS subtracts non-TAS DaemonSet and Deployment Pod usage from node capacity, so Kueue would treat placeholders as fixed before the native scheduler could preempt them ([Kueue TAS capacity calculation](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/#capacity-calculation)).

Kueue's plain-Pod integration implements preemption with `DELETE` calls ([Troubleshooting Pods](https://kueue.sigs.k8s.io/v0.18/docs/tasks/troubleshooting/troubleshooting_pods/#why-did-my-pod-disappear)). A PodDisruptionBudget constrains the eviction API, not direct deletion, so Iris's coordinator PDB cannot exclude a coordinator from Kueue victim selection. A safe priority offset must give the existing coordinator shape the same Workload priority as the accelerator.

Priority offsets also create an upgrade boundary. Iris does not rewrite running Pods during normal reconciliation, so a new `band+1` protected Workload could see an existing same-band GPU or coordinator Workload at the legacy band value as a victim. The classes must be opt-in per cluster, and first activation must reject both unfinished legacy shapes unless a separate migration updates them safely.

Kueue v1beta1 records the selected class in `Workload.spec.priorityClassName` and distinguishes a WorkloadPriorityClass through `spec.priorityClassSource` ([Kueue v1beta1 WorkloadSpec](https://kueue.sigs.k8s.io/v0.18/docs/reference/kueue.v1beta1/#workloadspec)). The activation audit must inspect both fields; the v1beta1 API does not expose `priorityClassRef`.

CoreWeave NodePools support `spec.nodeTaints`, so hard isolation is implementable without a custom admission webhook ([CoreWeave Node Pool reference](https://docs.coreweave.com/products/cks/reference/node-pool)). Iris GPU Pods already tolerate `nvidia.com/gpu`; CPU-only Pods do not ([`tasks.py`](https://github.com/marin-community/marin/blob/f8a0c7cba0c5e29efafe2f0b215eadf1d3d11c5a/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L976-L988)).

## Negative / Failed Leads

- Raising or injecting `cpu = GPUs * N` does not protect free GPUs. The job already requests CPU; the scheduler can still encounter a node whose CPU was consumed first.
- Kubelet system reservations make the reserved CPU unavailable to GPU Pods too.
- A custom extended resource can count synthetic tokens but cannot stop CPU-only Pods from consuming ordinary CPU. Removing the GPU Pod's normal CPU request would oversubscribe the host and break CPU accounting.
- Kueue quota is cluster-wide and currently intentionally non-binding. It does not express a per-node CPU reserve.
- Separate CPU and accelerator ClusterQueues in a borrowing cohort could reserve aggregate CPU quota, but still would not guarantee per-node CPU beside a free accelerator. It also adds cross-queue reclaim policy and duplicates the current admission path.
- A hard `nvidia.com/gpu:NoSchedule` taint works, but on H100 nodes it would keep roughly 96 of 128 CPU cores unavailable to CPU-only work when all eight GPUs run at the reported four-CPU-per-GPU request.
- Placeholder Pods reserve an exact quantity for the native scheduler, but Kueue TAS sees them as fixed non-TAS usage and may never admit the Pod that should preempt them.
- A custom scheduler plugin or DRA driver could model coupled resources, but it adds a new control-plane component and still needs a policy for actual CPU cgroup requests.

## Evidence Map

### Claim: a fixed CPU-per-GPU request is not the missing Kubernetes primitive

- Support:
  - Iris already emits both requests independently in `tasks.py`.
  - Kubernetes schedules against remaining Node Allocatable, regardless of why earlier Pods consumed it.
- Contradictions:
  - A hard node taint prevents prior CPU-only consumption, but it reserves the entire node rather than the requested quantity.
- Directness to Marin: exact current lowering path and H100 node shape.
- Confidence: high.
- Action: keep explicit job CPU requests; solve reclaim policy separately.

### Claim: accelerator-specific Kueue priority is the narrowest borrow-and-reclaim policy

- Support:
  - Unified TAS already makes CPU-only Workloads removable in the relevant node snapshot.
  - WorkloadPriorityClass changes Kueue order and preemption without changing Pod priority.
- Contradictions:
  - Kueue preempts a whole Workload. A CPU gang may lose more work than the single blocking Pod suggests.
  - The same offset must cover coordinator-shaped Workloads because their PDB does not constrain Kueue deletion. This also lets a pending coordinator reclaim ordinary same-band CPU work.
  - Workload priority affects queue order as well as preemption, so protected Workloads overtake older ordinary same-band CPU Workloads.
  - Activating the offset beside legacy GPU or coordinator Workloads could make them victims.
- Directness to Marin: reuses the exact ClusterQueue, TAS flavor, and priority gaps already deployed.
- Confidence: medium-high pending a CoreWeave CI reproduction.
- Action: canary a one-point Workload priority offset before live rollout.

## Recommended Next Experiments

### 1. Reproduce equal-band CPU stranding and reclaim it with Workload priority

- Minimum experiment: on the CoreWeave CI cluster, pin same-band CPU-only Pods to an H100 pool until a GPU Pod cannot fit by CPU, then submit a fresh treatment GPU Pod with a protected WorkloadPriorityClass one point above the band.
- Baseline/control: run the same resource shape using only the current band PriorityClass, then delete the baseline Workloads before treatment.
- Expected signal: baseline remains `QuotaReserved=False`; treatment records a Kueue preemption, evicts only lower-priority CPU Workloads, and admits the H100 Workload.
- Falsifier: the treatment remains unadmitted, evicts an equal/higher-band Workload, or admits but leaves the Pod unschedulable.
- Cost/risk: one CI H100 and disposable CPU tasks; no research workload interruption.
- Sources: Kueue preemption and WorkloadPriorityClass docs above.

### 2. Measure victim quality before production rollout

- Minimum experiment: inspect the selected victims for single-Pod workers, CPU gangs, and protected coordinator-shaped Pods.
- Baseline/control: lower-band preemption behavior from the existing priority classes.
- Expected signal: victims occupy GPU-node CPU and their removal supplies the requested per-node capacity.
- Falsifier: Kueue evicts unrelated CPU-node work, a full CPU gang for negligible GPU-node benefit, or a coordinator whose loss cascades to a larger pipeline.
- Cost/risk: synthetic workloads only in CI.
- Sources: Echo wiki 80 and the unified TAS implementation.

### 3. Reject unsafe activation beside legacy protected Workloads

- Minimum experiment: leave same-band GPU and coordinator Workloads running without a protected class, enable the protected mapping, and restart preflight.
- Baseline/control: restart with the mapping still empty.
- Expected signal: baseline starts normally; treatment fails before dispatch and identifies the legacy Workload.
- Falsifier: treatment dispatches a new protected Workload or selects either legacy Workload as a victim.
- Cost/risk: synthetic CI Workloads only; do not run this against production.
- Sources: the current `sync()` path applies manifests only for new task attempts.

## Hypothesis Queue Update

- Add: an opt-in one-point Kueue-only accelerator and coordinator priority offset reclaims the Pod's real host-resource request while preserving CPU spillover and coordinator protection.
- Revise: hard GPU taints remain the fallback if victim quality is unsafe, not the default.
- Falsify / stop: kubelet reservation, custom extended-resource tokens, and placeholder Pods do not satisfy both borrowability and TAS admission.
- Promote: no implementation until the CI victim-selection and legacy-activation experiments pass.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Iris Pod lowering | Marin code | `lib/iris/src/iris/cluster/backends/k8s/tasks.py` | CPU/GPU requests and existing Kueue/PriorityClass hooks | High | Pinned to `f8a0c7cba` |
| Iris Kueue manifests | Marin code | `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py` | Unified TAS and lower-priority policy | High | Pinned to `f8a0c7cba` |
| Issue #7916 / PR #7928 | issue / PR | GitHub | Measured prior failure and deployed lower-priority fix | High | Direct RNO2A evidence |
| Echo wiki 80 | wiki | https://echo.oa.dev/wiki/80 | Incident synthesis and rejected alternatives | High | Direct RNO2A evidence |
| Kubernetes taints | official docs | kubernetes.io | Hard special-hardware isolation | High | Stable native mechanism |
| Kubernetes node reservation | official docs | kubernetes.io | System-only reservation semantics | High | Does not reserve for workloads |
| Kubernetes priority | official docs | kubernetes.io | Pod-level lower-priority preemption | High | Stable since v1.14 |
| Kueue priority and preemption | official docs | kueue.sigs.k8s.io | Workload-only priority and victim rule | High | Versioned v0.18 docs; behavior still needs a CI reproduction |
| CoreWeave NodePool reference | vendor docs | docs.coreweave.com | `nodeTaints` availability | High | Hard-isolation fallback |

## Handoff

- Suggested issue `Prior work` block: #7916 and #7928 fixed lower-band CPU reclamation by unifying TAS accounting. Equal-band CPU work can still strand GPUs because Kueue only preempts lower priority. Kubernetes has no native fractional CPU reservation for a workload class; use opt-in Kueue workload priority for borrowable capacity or a hard GPU-node taint for isolation.
- Suggested logbook entry: none; this design directory is the durable record.
- Rollout scope: stage H100 first; evaluate GB200 after H100 victim-selection data is available.
- Stop reason: every additional mechanism reduces to hard isolation, priority-based borrowing, or a new scheduler/control-plane component; none changes the recommendation.
