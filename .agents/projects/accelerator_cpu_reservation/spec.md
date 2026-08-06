# Accelerator CPU Reclaim Contract

## Scope

This contract adds a Kueue-only priority offset for Iris GPU and protected coordinator Workloads on the Kubernetes backend. It does not change Iris priority bands, Pod-level Kubernetes priority, resource requests, or VM/TPU scheduling.

Implementation starts only after the reported stalled jobs are shown to have same-band CPU-only blockers. Lower-band blockers instead trigger a #7928 rollout or regression investigation; higher-band and accelerator blockers retain their current ordering.

## Cluster-Scoped Objects

The CoreWeave Kueue substrate owns these `kueue.x-k8s.io/v1beta1` `WorkloadPriorityClass` objects:

| Name | Value | Applied to |
| --- | ---: | --- |
| `iris-protected-batch` | 1 | Protected Workloads in `PRIORITY_BAND_BATCH` |
| `iris-protected-interactive` | 11 | Protected Workloads in `PRIORITY_BAND_INTERACTIVE` or an unset/inherited band resolved to interactive |
| `iris-protected-production` | 1001 | Protected Workloads in `PRIORITY_BAND_PRODUCTION` |

Each object is cluster-scoped, non-default, and has a description stating that it gives accelerator and coordinator Workloads reclaim preference over ordinary CPU-only Workloads in the same Iris band. The canonical names and values live in `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py` and are shared by Pulumi and `install_kueue.py`.

## Python Surface

`lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py` adds:

```python
PROTECTED_WORKLOAD_PRIORITY_CLASSES: dict[int, tuple[str, int]]

def build_protected_workload_priority_class_manifests() -> list[dict]:
    """Return one canonical WorkloadPriorityClass manifest per Iris band."""
```

`lib/iris/src/iris/cluster/backends/k8s/tasks.py` extends `PodConfig` with:

```python
protected_kueue_priority_classes: dict[int, str]
```

`KueueConfig` adds an empty-by-default `protected_priority_classes: dict[str, str]` mapping from band name to WorkloadPriorityClass name. The composer validates the band names and lowers the configured entries into `PodConfig.protected_kueue_priority_classes`. An explicit entry in the existing `kueue_priority_classes` mapping remains authoritative for that band and disables the automatic protected class for that Pod.

The canonical mapping is not a `PodConfig` or `KueueConfig` default. Provisioning the class objects is inert until a cluster config opts in, which permits staged activation and rollback.

## Pod Lowering

For a `RunTaskRequest` whose resources request at least one GPU or for which `_is_coordinator_task` returns true:

1. Resolve an unset priority to `PRIORITY_BAND_INTERACTIVE`.
2. If `PodConfig.kueue_priority_classes` has an explicit mapping for the band, stamp that name in the `kueue.x-k8s.io/priority-class` label.
3. Otherwise, if `protected_kueue_priority_classes` has an entry for the band, stamp that name.
4. Continue to stamp `spec.priorityClassName` from `priority_class_names` exactly as today.

For another request with no GPU, retain the existing optional `kueue_priority_classes` behavior and do not stamp a protected class.

The accelerator rule depends on the GPU count derived by `get_gpu_count`. The coordinator rule reuses `_is_coordinator_task`, so it matches the same single-task, accelerator-free requests that receive a PodDisruptionBudget. CPU count, GPU variant, coscheduling mode, and `host_network` do not otherwise change class selection.

Every Pod in a coscheduled group resolves the same effective band and protected mapping. Controller redrive must not produce a group whose members carry different `kueue.x-k8s.io/priority-class` labels.

## Provisioning and Preflight

- `infra/pulumi/src/iac/coreweave/kueue.py` creates all protected WorkloadPriorityClasses before the ClusterQueue-dependent controller rollout.
- `lib/iris/scripts/install_kueue.py --with-queues` includes the same manifests in its printed plan and apply set.
- `K8sControllerProvider.verify_prerequisites()` fetches every configured protected WorkloadPriorityClass before task dispatch and reports a missing class or a value other than `band+1`.
- `K8sResource` adds the cluster-scoped `WORKLOAD_PRIORITY_CLASSES` endpoint for `kueue.x-k8s.io/v1beta1`.

A missing or drifted configured class is a deployment error. Iris does not omit the label or fall back after preflight.

## Activation Gate

Whenever `protected_priority_classes` is non-empty, controller preflight lists unfinished Kueue Workloads in the Iris namespace. It treats a Workload as a protected shape when any PodSet requests `nvidia.com/gpu`, or when its PodSets contain one accelerator-free Pod in total, matching `_is_coordinator_task` after lowering.

For each protected shape in a band where the protected mapping is effective, preflight requires `spec.priorityClassName` to equal the configured class and `spec.priorityClassSource` to equal `kueue.x-k8s.io/workloadpriorityclass`. These are the Kueue v1beta1 Workload fields; there is no `priorityClassRef`. A mismatch fails preflight and identifies the Workload. This includes pending, admitted, and evicted-but-requeued Workloads; finished Workloads do not block activation.

The operator must let legacy GPU and coordinator Workloads finish or use a separately reviewed migration before enabling the mapping. This prevents a new `band+1` Workload from selecting an old same-band protected shape as a lower-priority victim. A first rollout therefore has these phases:

1. Provision the canonical WorkloadPriorityClass objects with no config mapping.
2. Pass the isolated CoreWeave behavior test.
3. Confirm the target namespace has no unfinished legacy GPU or coordinator Workloads.
4. Add the protected mapping to that cluster and restart its controller.

Removing the mapping prevents new Workloads from receiving the protected class. Existing admitted Workloads keep their Kueue priority until they finish; rollback does not mutate them.

## Resource Semantics

- `resources.requests.cpu` remains the caller's explicit `ResourceSpec.cpu_millicores` value.
- Iris adds no CPU limit and no CPU-per-GPU default or minimum.
- NodePool taints and tolerations do not change.
- The `cw-tas` ResourceFlavor and `withinClusterQueue: LowerPriority` policy do not change.
- VM/TPU backends do not consume WorkloadPriorityClass names.

## Persisted and Wire Shapes

No protobuf, database, Finelog schema, or on-disk format changes. Existing task and job priority fields continue to expose the Iris band, not the internal one-point Kueue priority offset.

## Errors

Controller prerequisite validation adds the missing WorkloadPriorityClass names to `PrerequisitesNotProvisionedError`. Pod construction introduces no new error type. Kubernetes or Kueue rejection after a successful preflight propagates through the existing Pod apply and task-event paths.

## Out of Scope

- Choosing or enforcing a CPU-per-GPU performance floor.
- Hard isolation of GPU NodePools with a taint.
- Custom scheduler plugins, Dynamic Resource Allocation drivers, or synthetic extended resources.
- Changes to user-facing Iris priority names, budget demotion, or federation priority accounting.
- In-place migration or relabeling of legacy protected Workloads.
