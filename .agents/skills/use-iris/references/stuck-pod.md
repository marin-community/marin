# Recover a stuck CoreWeave pod

Use only for a specified pod that remains terminating beyond its deadline. Read the canonical kubeconfig, context, and namespace from `lib/iris/config/<cluster>.yaml` and pass them explicitly to every command.

Start read-only:

```bash
kubectl --kubeconfig <file> --context <context> -n <namespace> \
  get pod <pod> -o yaml
kubectl --kubeconfig <file> --context <context> -n <namespace> \
  get events --field-selector involvedObject.name=<pod> --sort-by=.lastTimestamp
kubectl --kubeconfig <file> --context <context> \
  get pods --all-namespaces --field-selector spec.nodeName=<node> -o wide
```

Extract the canonical attempt from the task container's `IRIS_TASK_ID`; never target a sanitized label. Inspect owner references, finalizers, sibling pods, disruption budgets, volumes, local storage, node state, and current Iris job state.

Get explicit approval for the exact retry policy and every mutation. Cordon before permitting an Iris retry. Try ordinary deletion first. Do not use a broad forced drain or kill arbitrary host processes.

Force deletion removes only the API object. For a node-bound GPU process that may still run, obtain provider-confirmed physical reboot completion before force deletion:

```bash
cwic node reboot --force --message "stuck terminating GPU pod <namespace>/<pod>" <node>
kubectl --kubeconfig <file> --context <context> -n <namespace> \
  delete pod <pod> --grace-period=0 --force
```

If `cwic` or authentication is unavailable, stop and use CoreWeave support. Never substitute force deletion. Verify the original UID is gone, no duplicate attempt is active, GPU capacity and daemonsets recovered, and node health passes. Uncordon only if this procedure cordoned the node and all checks pass. Never restart Iris as pod recovery.

Publish the incident with `write-ops-log`, including the canonical attempt, node, provider operation, collateral workloads, and verification evidence.
