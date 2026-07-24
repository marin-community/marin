---
date: 2026-07-24
system: coreweave
severity: outage
resolution: fixed
pr: none
issue: none
---

## TL;DR

- Grafana fired `ControlPlaneCrashLooping cw-us-east-02a`; Kueue's manager had no ready webhook endpoint.
- `kueue-controller-manager-6ff6f884b7-p4n6s` was `OOMKilled` with exit code 137 after reaching its 512 MiB memory limit during restart resync.
- A 2 GiB request and limit restored `cw-us-east-02a`. The fleet audit found the same 512 MiB default on `cw-us-east-08a` and `cw-us-west-04a`; both were raised to 2 GiB.
- Pulumi now defaults every CoreWeave Kueue manager to 2 GiB and rejects lower per-cluster values. Larger overrides remain valid.
- Targeted Pulumi updates changed only the Kueue Helm release in the three affected stacks. All four managers were Ready with serving webhook endpoints after the updates.

## Original problem report

Grafana reported `[FIRING:1] ControlPlaneCrashLooping cw-us-east-02a` at 10:33 AM on 2026-07-24. The operator requested investigation of the Kueue crash loop and authorized Kubernetes changes.

## Investigation path

1. `kubectl get pods -n kueue-system` showed `kueue-controller-manager-6ff6f884b7-p4n6s` at `0/1 CrashLoopBackOff` with 16 restarts. The Kueue webhook EndpointSlice contained only its non-ready address.

2. The Pod's last terminated state was `Reason: OOMKilled`, `Exit Code: 137`. Its request and limit were both 512 MiB. The last observed process ran from `2026-07-24T17:36:50Z` to `17:36:56Z`.

3. The previous RNO2A outage suggested API-client throttling and leader-election loss. That cause was ruled out: `kueue-manager-config` already contained `clientConnection.qps: 100` and `burst: 200`, and the sampled termination was an OOM kill.

4. The cluster contained 820 Kueue Workloads and 1,743 Pods. Previous-container logs showed a dense startup resync until the process was killed. The cluster had no Metrics API, so no historical working-set sample was available.

5. `kubectl set resources` raised the manager request and limit to 2 GiB. Deployment `kueue-controller-manager-77cbc9bcd7-xrxjl` became Ready with zero restarts and restored the serving webhook endpoint.

6. A fleet audit found `cw-rno2a` already at 2 GiB. `cw-us-east-08a` and `cw-us-west-04a` still used 512 MiB, so both were rolled to 2 GiB before they accumulated the same restart load.

7. The Pulumi schema had left `manager_memory_limit` unset by default. `infra/pulumi/src/iac/coreweave/kueue.py:83` passed that value to `build_cks_values`; `lib/iris/src/iris/cluster/platforms/k8s/kueue_manifests.py:244` omitted the resource override when it was `None`, exposing the CoreWeave chart's 512 MiB default.

8. Targeted previews for all four stacks showed one Kueue Helm update in `cw-us-east-02a`, `cw-us-east-08a`, and `cw-us-west-04a`, with 28 unchanged resources per stack. `cw-rno2a` had no diff. All three targeted updates succeeded between `17:52:38Z` and `17:53:11Z`.

## User course corrections

- The operator expanded the recovery from the firing cluster to every Kueue installation, with a minimum of 2 GiB. This found and corrected the same 512 MiB default on two healthy clusters.
- The operator rejected an Iris start/restart prerequisite check and directed the fix to Pulumi only. Pulumi remained the sole owner of the Kueue Deployment; Iris controller lifecycle behavior did not change.

## Root cause

`KueueProvisioningSpec.manager_memory_limit` defaulted to `None`. The Pulumi Kueue component therefore omitted `controllerManager.manager.resources`, and Helm retained the `cks-kueue` chart's 512 MiB request and limit. Kueue retains watched Pods and Workloads in memory during startup resync; 512 MiB was insufficient for the 820-Workload, 1,743-Pod `cw-us-east-02a` cluster.

The 100-QPS, 200-request client limiter from the 2026-07-22 RNO2A incident was already active. This incident was a separate memory-cap failure.

## Fix

The live recovery set each affected manager to:

```yaml
resources:
  requests:
    memory: 2Gi
  limits:
    memory: 2Gi
```

`infra/pulumi/src/iac/config.py:76` now uses `2Gi` as the shared default. The validator at `infra/pulumi/src/iac/config.py:80` rejects lower values while allowing larger cluster-specific overrides. The redundant RNO2A override was removed from `lib/iris/config/cw-rno2a.yaml`.

Targeted `pulumi up` operations updated only these Helm release URNs:

```text
urn:pulumi:<stack>::marin-iac::marin:coreweave:KueueAddon$kubernetes:helm.sh/v3:Release::kueue
```

## How OPS.md could have shortened this

- Add a CoreWeave control-plane crash-loop subsection under `lib/iris/OPS.md` Troubleshooting. Start with `kubectl describe pod` and `kubectl logs --previous`; classify `OOMKilled`/137 separately from exit-code-1 controller errors before changing resources or client rate limits.
- Add `kubectl get deployment -n <namespace> -o yaml` to compare requests and limits with the node's allocatable memory. Note that CoreWeave clusters may not expose the Metrics API, so termination state remains the reliable OOM signal.
- Add the targeted Pulumi workflow from `infra/pulumi/README.md`: preview the exact Helm release URN and reject any NodePool, DNS, or unrelated resource change before recovery.

## Artifacts

- `.agents/ops/2026-07-24-coreweave-kueue-manager-oom.md`
- `infra/pulumi/src/iac/config.py`
- `infra/pulumi/tests/test_config.py`
- `infra/pulumi/README.md`
- Grafana alert: https://grafana.oa.dev/alerting/grafana/k8s-control-plane-crashloop/view?orgId=1
