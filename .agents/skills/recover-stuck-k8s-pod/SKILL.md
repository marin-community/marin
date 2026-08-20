---
name: recover-stuck-k8s-pod
description: Diagnose and safely recover stuck terminating Kubernetes pods on Marin CoreWeave clusters, especially node-bound GPU pods. Use for deletion hangs, suspected uninterruptible GPU/NCCL waits, node cordoning or reboot decisions, and force-deletion requests.
---

# Recover a stuck Kubernetes pod

Start read-only. Get explicit approval before cordoning, deleting workloads,
rebooting a node, or changing Iris state. Never restart an Iris cluster. Force
deletion removes only the API object, not a process or GPU allocation.

Read `kubeconfig_path` and `kube_context` from
`lib/iris/config/<cluster>.yaml`, and pass explicit kubeconfig, context,
namespace, pod, and node to every command:

```bash
rg -n 'kubeconfig_path|kube_context' lib/iris/config/<cluster>.yaml
kubectl --kubeconfig <kubeconfig> --context <context> -n <namespace> get pod <pod> -o yaml
kubectl --kubeconfig <kubeconfig> --context <context> -n <namespace> get events --field-selector involvedObject.name=<pod> --sort-by=.lastTimestamp
```

Classify by the first applicable Grafana value: `invalid_timestamp` (repair or
escalate metadata), `finalizer` (repair owner; no reboot solely for finalizer),
`terminal`, `unbound`, or `node_cleanup` (nonterminal, node-bound, no finalizer).
FailedKillPod/ExceededGracePeriod support the diagnosis; missing events/logs,
DCGM metrics, and a `D`-state process alone do not.

Extract `IRIS_TASK_ID` from the task container; labels may be truncated and are
never Iris Attempt targets. Inventory all pods on the node, owners, disruption
budgets, PVCs, `emptyDir`, unmanaged pods, and prior cordon state:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> -n <namespace> get pod <pod> -o jsonpath='{.spec.containers[?(@.name=="task")].env[?(@.name=="IRIS_TASK_ID")].value}{"\n"}'
kubectl --kubeconfig <kubeconfig> --context <context> get pods --all-namespaces --field-selector spec.nodeName=<node> -o wide
kubectl --kubeconfig <kubeconfig> --context <context> get node <node> -o jsonpath='{.spec.unschedulable}{"\n"}'
uv run iris --cluster=<cluster> job describe <job>
```

With the operator, choose parent `iris job cancel <job>` or canonical
`iris attempt preempt <attempt>`. Do not permit an immediate retry onto a node
that has not been cordoned.

With approval, cordon then request ordinary deletion:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> cordon <node>
kubectl --kubeconfig <kubeconfig> --context <context> -n <namespace> delete pod <pod> --wait=false
```

Wait for kubelet cleanup and inspect fresh events. If the object disappears,
verify that its process and GPU allocation are gone before treating the node as
healthy. Continue to reboot only when the object persists and `node_cleanup`
evidence still implicates the node.

Node cleanup requires `cwic`. If it is unavailable or unauthenticated, stop and
use the CoreWeave console/support. Do not use broad `kubectl drain --force` or
kill arbitrary PIDs:

```bash
command -v cwic
cwic auth login
cwic node get <node>
cwic node describe <node>
cwic node reboot --force --message "stuck terminating GPU pod <namespace>/<pod>" <node>
```

Use `--force` only after graceful cleanup fails and collateral is approved. A
Kubernetes Ready transition is insufficient; wait for provider completion,
boot identity/self-test, and production return. Use the out-of-service taint
only for confirmed power-off with understood volume/data-corruption impact.

Only then may the object be force-deleted:

```bash
kubectl --kubeconfig <kubeconfig> --context <context> -n <namespace> delete pod <pod> --grace-period=0 --force
```

Verify original UID removal, no duplicate attempt, restored GPU capacity,
healthy daemonsets and workloads, and workload health checks. Uncordon only a
node this procedure cordoned when it was initially schedulable. Record evidence
with `write-ops-log` and link its Echo URL from the PR/issue.
