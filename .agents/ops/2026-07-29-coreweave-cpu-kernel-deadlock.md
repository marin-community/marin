---
date: 2026-07-29
system: coreweave
severity: degraded
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7738
issue: none
---

# CoreWeave US-WEST-04A CPU kernel deadlock

## TL;DR

- CoreWeave reported `KernelDeadlock=True` on the only CPU node in
  `cw-us-west-04a` after a soft lockup in `kswapd0`.
- CoreWeave cordoned the node and requested `production-reboot`, but five
  tenant control-plane pods blocked the lifecycle transition.
- Ready replacements were established on the healthy H100 node before the
  original pods were removed. Traefik required a temporary placement change
  because its CPU affinity and required pod anti-affinity prevented a
  replacement from scheduling.
- CoreWeave began an immediate Redfish reboot after `CWActive` became false.
- The node returned with a new boot ID, passed CoreWeave's verification
  workflow, and returned to `production` without a cordon or taints.
- Grafana diagnostics were extended to include this cluster and expose node
  readiness, cordons, deadlock conditions, and pending lifecycle state. A
  critical alert now fires when `KernelDeadlock=True` persists for five
  minutes.

## Original problem report

GitHub Actions job `90580147865` spent one hour starting the `iris-ci`
controller and was cancelled before running its integration tests. The
controller pod remained Pending:

```text
0/2 nodes are available: 1 node(s) didn't match Pod's node affinity/selector,
1 node(s) were unschedulable.
```

The pod selected the `cpu-erapids` Iris scale group. The cluster's only
matching node, `g8fd930`, was cordoned, while the other node belonged to the
`h100-8x` scale group.

## Investigation path

1. The controller Deployment had zero available replicas because its pod could
   not schedule onto either of the cluster's two nodes.

2. Node `g8fd930` remained `Ready=True` but had
   `spec.unschedulable=true` and
   `node.coreweave.cloud/cordonReason=KernelDeadlock,NLCCPendingExitProduction`.

3. Its structured condition reported
   `KernelDeadlock=True`, reason `CPUSoftLockup`, with
   `watchdog: BUG: soft lockup - CPU#63 stuck for 25s! [kswapd0:428]`.

4. CoreWeave set `PendingPhaseState=True`, reason `production-reboot`, but
   `CWActive=True` named five tenant blockers: cert-manager,
   cert-manager-cainjector, cert-manager-webhook, kueue-controller-manager, and
   Traefik.

5. Each Deployment had one replica. Traefik's PodDisruptionBudget required one
   available replica and allowed no disruption, so deleting the bound pod
   first would have interrupted cluster ingress.

6. The four cert-manager and Kueue Deployments were temporarily scaled to two.
   Their replacements became Ready on healthy node `g145386`, after which only
   the original pods on `g8fd930` were deleted.

7. Traefik's second replica remained Pending because the Deployment required a
   CPU node and required replicas to use different hostnames. Its pod template
   was temporarily moved to the GPU node class. One replacement became Ready
   on `g145386` before the original pod entered termination.

8. After the original tenant pods were gone, CoreWeave reported
   `CWActive=False`, moved the node to `production-reboot`, and initialized
   `NLCCImmediateReboot=True` with reason `NLCCRedfishReboot`.

9. The forced power cycle changed the boot ID from
   `8bb1816f-1e3a-4052-bd3e-f7a35a609e8e` to
   `2452a7c6-9108-45bd-b9fc-e781ba7c5632`. The node returned Ready and
   CoreWeave's HPC verification workflow succeeded.

10. An initial Traefik handback occurred before CoreWeave removed its
    `node.coreweave.cloud/reserved:NoExecute` verification taint. The
    Deployment's `maxUnavailable: 1` strategy removed the GPU replica while
    the CPU replacement was Pending, briefly interrupting Traefik. GPU
    placement was restored immediately.

11. CoreWeave then returned the node to `production` and removed the
    reservation taint. Two GPU replicas protected the final handback; a CPU
    replica became Ready before temporary capacity was removed.

## User course corrections

- The operator redirected the investigation from the individual CI timeout to
  fleet diagnostics and an alert for the underlying node deadlock.
- The operator then requested the steps needed for CoreWeave to reboot the
  node and explicitly approved the staged live recovery.

## Root cause

The CPU node's kernel watchdog detected a soft lockup in `kswapd0`. CoreWeave
correctly cordoned the node and queued a reboot, but its lifecycle controller
would not reboot while tenant pods remained active. Those singleton
control-plane workloads had no pre-existing replicas elsewhere, and Traefik's
placement and disruption constraints required a temporary placement change to
preserve availability.

The Iris controller scheduling failure was a consequence of the provider
cordon, not an Iris controller deadlock.

## Fix

The five tenant workloads were moved off `g8fd930` before CoreWeave initiated
the node reboot. CoreWeave completed the forced power cycle and verification,
then returned the node to `production` without a cordon or reservation taint.
Temporary replica counts were returned to one, and Traefik's original CPU
affinity and toleration were restored. The Iris controller scheduled on the
recovered node and all five singleton control-plane Deployments reported one
Ready replica.

The Grafana bridge now polls `cw-us-west-04a`, uses its `iris-ci` namespace,
omits nonexistent finelog-mirror checks for that cluster, and reports CoreWeave
node lifecycle conditions. `CoreWeaveNodeKernelDeadlock` is a critical alert
with a five-minute hold. The cluster's Pulumi config grants the existing
Grafana observer identity permission to list Nodes.

## How OPS.md could have shortened this

`lib/iris/OPS.md` now documents the provider cordon and pending-reboot signals,
the replacement-before-deletion sequence, PodDisruptionBudget checks, and the
boot-ID verification. It also records the placement-constraint fallback that
was needed for Traefik, the provider reservation-taint restore gate, and the
prohibition against uncordoning provider-cordoned nodes or force-draining
provider DaemonSets.

## Artifacts

- [Failed CoreWeave CI job](https://github.com/marin-community/marin/actions/runs/30451767755/job/90580147865)
- [Grafana diagnostics and alert PR](https://github.com/marin-community/marin/pull/7738)
- `.agents/ops/2026-07-29-coreweave-cpu-kernel-deadlock.md`
