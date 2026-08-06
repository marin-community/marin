# Reclaim Accelerator-Node CPU for Accelerator Workloads

Prevent CPU-only work from leaving expensive accelerators idle while retaining its ability to use otherwise-idle host CPU. Iris should treat CPU on accelerator nodes as borrowable capacity: CPU-only Pods may use it, and an accelerator Workload may reclaim its explicit host-resource request from same-band CPU-only Workloads.

This is separate from choosing a performance floor such as four CPU cores per GPU. Iris already sends a job's CPU and GPU requests to Kubernetes. The missing policy is who yields after CPU-only work consumes the node first.

## Challenges

Kubernetes has no built-in fractional reservation of ordinary CPU for a class of future Pods. Kubelet reservations remove CPU from every Pod, including GPU Pods. A GPU-node taint is a hard all-or-nothing boundary. Synthetic extended resources do not prevent ordinary CPU consumption.

Marin also has two schedulers in the path. Kueue admits a complete Workload against per-node TAS capacity before Kubernetes places its Pods. A native scheduler trick such as low-priority placeholder Pods can fail at the earlier gate because Kueue counts those Pods as fixed node usage.

The relevant current behavior is documented in [research.md](research.md). [PR #7928](https://github.com/marin-community/marin/pull/7928) made lower-priority CPU reservations reclaimable by placing every Iris Pod in one TAS flavor. Same-band CPU and GPU Workloads still share one numeric priority, so `withinClusterQueue: LowerPriority` cannot choose the CPU Workload as a victim.

The observed Larry jobs must be classified before implementation. If their blockers were lower-band CPU Workloads, the result is a #7928 deployment or regression investigation rather than evidence for this policy. Higher-band or accelerator Workloads are expected blockers under the existing priority contract. This design addresses only same-band CPU-only Workloads that leave an otherwise-usable accelerator idle.

## Costs / Risks

- Same-band CPU-only work becomes opportunistic on accelerator hosts. It can restart even though the user did not select a lower Iris priority band.
- Protected Workloads also move ahead of older ordinary CPU Workloads in the same band when both are pending. The policy changes queue order even when no preemption is needed.
- Kueue preempts whole Workloads. If a CPU gang has one Pod on a needed H100 node, reclaim may interrupt the entire gang.
- Single-replica CPU coordinators are currently protected from voluntary eviction with a PodDisruptionBudget. Kueue preempts plain Pods with a delete, which does not consult the eviction subresource or that budget. The priority policy must protect coordinator-shaped Workloads explicitly to avoid cascading into a larger pipeline retry.
- Sustained accelerator demand can starve CPU-only work after the dedicated CPU pools fill. The existing user bands remain the primary ordering: a batch GPU Workload cannot reclaim interactive CPU work.
- This is a reclaim preference, not an unconditional guarantee. Equal protected Workloads, higher-band Pods, system Pods, memory, disk, or an infeasible topology can still leave an accelerator pending.

## Design

Add three Kueue `WorkloadPriorityClass` objects for accelerator and protected coordinator work, with values one point above the corresponding Iris band:

| Iris band | Ordinary CPU-only Workload | Accelerator or coordinator Workload |
| --- | ---: | ---: |
| `BATCH` | 0 | 1 |
| `INTERACTIVE` | 10 | 11 |
| `PRODUCTION` | 1000 | 1001 |

The gaps preserve the existing band ordering. Production CPU work at 1000 still outranks interactive accelerator work at 11, and interactive CPU work at 10 still outranks batch accelerator work at 1. `iris-system` remains 10000.

`K8sTaskProvider` will stamp `kueue.x-k8s.io/priority-class` on Pods requesting `nvidia.com/gpu` and CPU-only Pods matched by the existing `_is_coordinator_task` protection rule. Kueue will use that value to order and preempt Workloads. Giving both shapes the same value prevents a same-band accelerator from selecting a coordinator as a victim. It also lets a pending coordinator reclaim ordinary same-band CPU work, consistent with its existing critical-workload treatment.

The Pod's Kubernetes `priorityClassName` remains `iris-{production,interactive,batch}`, so kube-scheduler and node-pressure behavior continue to reflect the user-selected band. Kueue documents this separation explicitly: `WorkloadPriorityClass` controls the Workload without changing Pod priority.

The existing common `cw-tas` flavor is required. CPU-only Pods use unconstrained TAS, so Kueue knows which node holds their CPU reservation. When an accelerator Workload cannot fit, `withinClusterQueue: LowerPriority` can simulate removing same-band CPU Workloads and choose victims whose removal makes the requested topology fit. It reclaims the accelerator Pod's actual CPU, memory, and disk requests. Iris does not inject `GPUs * N` CPU, change `ResourceSpec`, or add a CPU limit.

Pulumi owns the new cluster-scoped WorkloadPriorityClasses beside the ClusterQueue and ResourceFlavor. `install_kueue.py --with-queues` renders the same objects for non-Pulumi clusters. Merely provisioning the classes does not enable the policy. A new empty-by-default Kueue mapping enables their use explicitly in each cluster config.

Controller preflight verifies every configured class and its numeric value before dispatch starts. It also refuses first activation while an unfinished GPU or coordinator-shaped Workload in the Iris namespace lacks the configured protected class. Without that gate, a newly submitted protected Workload at `band+1` could select an already-running same-band GPU or coordinator Workload at the legacy band value as a victim. The cluster must drain those legacy Workloads or migrate them under a separately reviewed procedure before enabling the mapping. Once enabled, all members of a Pod group receive the same class during construction and redrive.

This ordering gives a reversible rollout: provision inert objects, run the CI canary, enable one drained cluster, then expand by config. Removing the mapping stops assigning the offset to new Workloads; it does not rewrite admitted Workloads.

Hard-tainting GPU NodePools with `nvidia.com/gpu:NoSchedule` remains the fallback. It is Kubernetes' standard recommendation for special hardware and Iris GPU Pods already carry the toleration. It would also keep coordinators off GPU nodes. It is not the default because an eight-H100 node has 128 CPUs while the reported training shape requests 32. Hard isolation would make the other 96 CPUs unavailable to CPU-only work whenever all GPUs are occupied.

Separate CPU and accelerator ClusterQueues in a borrowing cohort could reserve aggregate CPU quota for accelerator Workloads, but would not express per-node capacity. It would also duplicate admission policy and require cross-queue reclaim rules. That is more machinery than the same-queue priority offset and does not improve topology precision.

## Testing

Unit tests will assert that GPU and coordinator Pods receive the one-point-higher WorkloadPriorityClass while their Pod PriorityClass stays unchanged. Other CPU-only Pods retain the current class, and explicit operator WorkloadPriorityClass mappings remain authoritative. Every member of a Pod group must carry the same class. Manifest tests will cover all three class names and values, and preflight will fail before dispatch if a configured class is absent, has the wrong value, or legacy protected-shape Workloads make first activation unsafe.

The behavior test belongs in the existing CoreWeave Kueue smoke path. Pin disposable same-band CPU Workloads to the selected H100 pool until a GPU Workload cannot fit by CPU. Run the baseline without the protected mapping and confirm it remains `QuotaReserved=False`, then delete those Workloads and run a fresh treatment with the mapping. The treatment passes only if Kueue records lower-priority CPU victims, admits the GPU Workload, and the Pod runs. Additional cases verify that a batch GPU cannot evict interactive CPU work, that a same-band coordinator is not a victim, and that activation is rejected while a legacy same-band GPU or coordinator Workload is active.

Roll out to a drained `cw-us-west-04a` first. Before production rollout, inspect victim Workload IDs, node assignments, requested resources, queue latency, and retry outcomes. RNO2A is the first production target because its 128-CPU/eight-H100 shape and prior CPU-spillover incident provide a direct comparison. Defer GB200 until the H100 rollout establishes safe victim selection; GB200 has a different 144-CPU/four-GPU host ratio and rack-level topology. No live cluster changes are part of this design PR.
