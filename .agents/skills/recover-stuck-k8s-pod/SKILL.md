---
name: recover-stuck-k8s-pod
description: Diagnose and safely recover stuck terminating Kubernetes pods on Marin CoreWeave clusters, especially node-bound GPU pods. Use for deletion hangs, suspected uninterruptible GPU/NCCL waits, node cordoning or reboot decisions, and force-deletion requests.
---

# Recover a stuck Kubernetes pod

Treat a pod that remains beyond its deletion deadline as a stuck Kubernetes
object. Determine whether the node still owns live work before changing it.
Force deletion removes the API object; it does not prove that the process or
GPU allocation is gone.

## Guardrails

- Start with read-only inspection. Get explicit operator approval before
  cordoning, deleting workloads, rebooting a node, or changing Iris state.
- Never force-delete a node-bound GPU pod while its process might still be
  running. Cordon the node, quiesce Iris retries, and obtain provider-level
  proof of reboot completion first.
- Extract the canonical Iris attempt from the `task` container's
  `IRIS_TASK_ID`. Labels are sanitized or truncated display identifiers; never
  use them as targets for `iris job kick`.
- Do not use a broad `kubectl drain --force`. Inventory sibling pods, owners,
  disruption budgets, volumes, and local storage before approving collateral.
- Record whether the node was already cordoned. Uncordon only a node this
  procedure cordoned, and only after health checks pass.
- Do not restart an Iris cluster as part of pod recovery.

Read `platform.coreweave.kubeconfig_path` and `kube_context` from the canonical
`lib/iris/config/<cluster>.yaml` before operating. Do not copy those values from
an alert, an old incident, or this skill:

```bash
rg -n 'kubeconfig_path|kube_context' lib/iris/config/<cluster>.yaml
```

Set `<kubeconfig>`, `<context>`, `<namespace>`, `<pod>`, and `<node>` explicitly
in every command. Do not rely on the current kubectl context.

## Classify the object

Inspect the pod and its events:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> \
  -n <namespace> get pod <pod> -o yaml
kubectl --kubeconfig <kubeconfig> --context <context> \
  -n <namespace> get events --field-selector involvedObject.name=<pod> \
  --sort-by=.lastTimestamp
```

The Grafana `classification` column chooses the first applicable case:

- `invalid_timestamp`: repair or escalate the malformed object metadata.
- `finalizer`: investigate and repair the owning controller. Do not reboot a
  node solely because an object has a finalizer.
- `terminal`: the workload finished; investigate API or runtime cleanup.
- `unbound`: no node owns the pod; investigate API or scheduler cleanup.
- `node_cleanup`: the nonterminal object is bound to a node with no finalizer.
  Continue with node-level recovery only for this case.

`FailedKillPod` or `ExceededGracePeriod` events strengthen the diagnosis, but
events can expire. Missing logs do not identify a stuck termination, and
Kubernetes/DCGM metrics do not reliably identify an NCCL collective. A process
in `D` state or a kernel stack involving GPU/NCCL code is supporting evidence,
not the alert condition.

## Identify the Iris attempt and blast radius

Extract the exact attempt and inspect all workloads on the node:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> \
  -n <namespace> get pod <pod> \
  -o jsonpath='{.spec.containers[?(@.name=="task")].env[?(@.name=="IRIS_TASK_ID")].value}{"\n"}'
kubectl --kubeconfig <kubeconfig> --context <context> \
  get pods --all-namespaces --field-selector spec.nodeName=<node> -o wide
kubectl --kubeconfig <kubeconfig> --context <context> \
  get node <node> -o jsonpath='{.spec.unschedulable}{"\n"}'
```

Also inspect owner references, pod disruption budgets, PVCs, `emptyDir`, and
unmanaged pods. Split `<attempt>` (for example `/user/job/0:3`) at the task
suffix and check current Iris state:

```bash
uv run iris --cluster=<cluster> job summary <job>
```

Before node recovery, choose an explicit retry policy with the operator:

- Stop the parent job with `iris job stop <job>` when all work must remain
  quiescent.
- Mark only the canonical attempt preempted with
  `iris job kick <attempt> --state preempted` when the operator accepts Iris
  scheduling its retry within the job's preemption budget.

Do not kick by pod label, and do not permit an immediate retry onto a node that
has not yet been cordoned.

## Isolate and try graceful cleanup

With approval, cordon first and then request an ordinary deletion:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> cordon <node>
kubectl --kubeconfig <kubeconfig> --context <context> \
  -n <namespace> delete pod <pod> --wait=false
```

Wait for kubelet cleanup and inspect fresh events. If the object disappears,
verify the process and GPU allocation are gone before considering the node
healthy. If it remains and node cleanup is implicated, continue to reboot.

For additional evidence, an approved privileged node debug session can inspect
the host under `/host`:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> \
  debug node/<node> -it --image=ubuntu --profile=sysadmin
```

Do not kill arbitrary PIDs from the debug shell.

## Reboot the node before force deletion

CoreWeave `cwic` is a prerequisite. If it is unavailable or authentication
fails, stop and use the CoreWeave console or support; do not substitute force
deletion.

```bash
command -v cwic
cwic auth login
cwic node get <node>
cwic node describe <node>
cwic node reboot --force --message "stuck terminating GPU pod <namespace>/<pod>" <node>
```

Use `--force` only when the graceful path has failed and the collateral impact
is approved. A Kubernetes `NotReady`/`Ready` transition alone is not proof of a
physical reboot. Wait for `cwic node describe` to show the provider operation
completed and the node returned through self-test to production. Record the
provider operation, boot identity when available, and affected sibling pods.

Use the Kubernetes `node.kubernetes.io/out-of-service` taint only when the node
is confirmed powered off and the storage implications are understood. It can
force detach volumes and carries data-corruption risk; it is not the normal
path for a wedged GPU pod.

## Remove stale state and restore service

Only after provider-confirmed reboot completion may the remaining API object be
force-deleted:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> \
  -n <namespace> delete pod <pod> --grace-period=0 --force
```

Verify that the original pod UID is gone, no duplicate Iris attempt is active,
GPU allocatable capacity is restored, required daemonsets are healthy, and the
node has passed its workload health checks. If this procedure cordoned a node
that was initially schedulable, uncordon it only after those checks:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> uncordon <node>
```

Record the timeline, commands, canonical attempt, node, provider operation,
collateral workloads, and verification evidence in the incident or task log.

## References

- [Kubernetes pod termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination)
- [kubectl delete force-deletion warning](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_delete/)
- [CoreWeave node reboot](https://docs.coreweave.com/docs/products/cks/nodes/reboot-a-node)
- [CoreWeave node status and self-test](https://docs.coreweave.com/docs/products/cks/nodes/node-status)
- [Kubernetes non-graceful node shutdown](https://kubernetes.io/docs/concepts/cluster-administration/node-shutdown/#non-graceful-node-shutdown)
