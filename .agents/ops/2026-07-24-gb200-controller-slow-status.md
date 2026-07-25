---
date: 2026-07-24
system: coreweave
severity: degraded
resolution: investigating
pr: none
issue: none
---

## TL;DR

- `iris --cluster=cw-us-east-08a cluster dashboard` established its port-forward and then blocked as designed; stopping the command also closed the dashboard tunnel.
- The Iris controller was healthy on image `6544e1251d`, with zero restarts, 0.49 CPU cores, 2.43 GiB RSS, and responsive status, SQL, and log RPCs.
- Kubernetes node-list calls took 6.2-6.9 seconds every reconcile cycle. Several federation and dashboard RPCs took 1-2.5 seconds, which made the controller appear stalled.
- Kueue's `'iris-pg-c2f15c452d9e2b7f-1' group has fewer runnable pods than expected` warning occurred while Iris tore down a failed 16-pod gang generation. Attempt 2 was admitted with 16 pods and ran normally.
- The GB200 NodePool had 201 Ready nodes against a 216-node target. CoreWeave reported all 12 racks present but was still "awaiting remaining nodes to join their assigned domains."

## Original problem report

The operator reported that the `cw-us-east-08a` Iris controller seemed stalled
after `iris cluster dashboard` printed a local URL and waited. The dashboard
also showed `'iris-pg-c2f15c452d9e2b7f-1' group has fewer runnable pods than
expected`.

## Investigation path

1. `kubectl get pods -n iris` showed
   `iris-controller-757c484648-jgtj6` Ready for 18 hours with zero restarts.
   Kueue's controller was Ready with zero restarts after its earlier memory
   increase.

2. `iris cluster status` and `iris process status` completed in about three
   seconds. The controller reported healthy on image `6544e1251d`, 0.49 CPU
   cores, 2.43 GiB RSS, 84 threads, and 71 open file descriptors. This ruled
   out a dead controller, crash loop, and memory pressure.

3. Controller warning logs showed `Slow list nodes` every minute. Calls took
   6.2-6.9 seconds against the 2-second warning threshold. `FederationSync`,
   `ListJobs`, `LaunchJob`, and `RegisterEndpoint` calls intermittently took
   1-2.5 seconds. No controller exception or failed reconcile loop accompanied
   the latency.

4. The pod-group hash was mapped through
   `lib/iris/src/iris/cluster/backends/k8s/tasks.py:293` to
   `/rav/ep64-packed-sonic-combine-30-v1-20260724-1931/grug-train-ep64-packed-sonic-combine-30-v1-20260724-1931`.
   Attempt 1 contained 16 pods.

5. Task 5 of attempt 1 exited 133 at `2026-07-24T19:39:58Z`. Its JAX
   coordination error said rank 12 connected with a different incarnation.
   Iris marked the other 15 tasks `COSCHED_FAILED` and deleted the generation's
   Kueue Workload.

6. During the staggered pod deletion, Kueue reconciled the remaining labeled
   pods after their Workload was gone. At `19:40:35Z` it emitted
   `ErrWorkloadCompose: 'iris-pg-c2f15c452d9e2b7f-1' group has fewer runnable
   pods than expected`. The old Workload and all attempt-1 pods were gone by
   inspection time.

7. Iris created pod group `iris-pg-c2f15c452d9e2b7f-2`. Kueue admitted and
   ungated all 16 pods. `iris job summary` reported 16 of 16 tasks running.

8. `cw-use08a-gb200` reported 201 current nodes and 216 target nodes. All 201
   Kubernetes Node objects were Ready and schedulable. The NodePool condition
   reported 12 current racks against 12 target racks, with 15 nodes still
   awaiting domain join since `2026-07-22T17:26:06Z`.

## User course corrections

- The operator supplied the exact Kueue pod-group warning after the initial
  controller checks. This redirected the investigation from general RPC
  responsiveness to the failed gang generation and distinguished a cleanup
  warning from an active admission failure.

## Root cause

The dashboard command itself had not stalled. `iris cluster dashboard` owns the
local port-forward and intentionally waits until interrupted. The live
controller was responsive, but CoreWeave Kubernetes node-list operations took
about 6.5 seconds per reconcile cycle and caused visible RPC latency.

The pod-group warning was a consequence of gang cleanup. Iris deleted the
attempt-1 Workload and pods after one task failed. Kueue briefly observed fewer
runnable pods than the immutable 16-pod group total while deletion events
arrived. Attempt 2 used a new group name, as defined by
`lib/iris/src/iris/cluster/backends/k8s/tasks.py:285`, and admitted all 16 pods.
The cause of the first JAX rank reincarnation and the 15 missing GB200 nodes
remained unresolved.

## Fix

No live mutation or controller restart was needed. Iris's existing gang retry
path replaced attempt 1 with attempt 2, and the replacement gang ran with all
16 tasks.

CoreWeave still needed to supply or recover the 15 nodes absent from the
216-node NodePool target. The Kubernetes API latency also remained above Iris's
2-second warning threshold.

## How OPS.md could have shortened this

- In `lib/iris/OPS.md` under **Cluster Lifecycle**, state that `iris cluster
  dashboard` is a foreground port-forward and must remain running while the
  browser uses the printed URL.
- In `lib/iris/OPS.md` under **Troubleshooting**, add `Slow list nodes` as a
  signal for Kubernetes API latency. Pair it with `iris process status` and a
  direct `iris cluster status` check to distinguish slow reconciliation from a
  dead controller.
- In `lib/iris/OPS.md` under **CoreWeave (GPU) Operations**, compare NodePool
  `targetNodes`, `currentNodes`, and `status.conditions[].message`; `AtTarget`
  can be true when all racks exist but individual nodes have not joined.
- In `lib/iris/OPS.md` under **Troubleshooting**, document that
  `ErrWorkloadCompose` during an old pod-group generation's deletion can be a
  transient cleanup warning. Check whether a newer group generation is
  admitted before treating it as a scheduling outage.

## Artifacts

- `.agents/ops/2026-07-24-gb200-controller-slow-status.md`
- `lib/iris/config/cw-us-east-08a.yaml`
- `https://iris-cw-us-east-08a.oa.dev`
