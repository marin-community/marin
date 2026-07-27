---
date: 2026-07-26
system: iris
severity: degraded
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7661
issue: https://github.com/marin-community/marin/issues/7652
---

## TL;DR

- A four-node H100 gang ran for 4.9 hours, then task 0 showed exit 137 and the whole job failed without launching attempt 1.
- CoreWeave's retained CKS audit logs showed that Kueue preempted all four pods for an `iris-interactive` workload.
- Kueue used `TerminationTarget=True`, reason `WorkloadEvictedDueToPreempted`; Iris only recognized Kubernetes `DisruptionTarget`.
- Iris charged the victim as `FAILED`, crossed `max_task_failures=0`, and killed the pending retry and bounced siblings during job finalization.
- The fix classifies Kueue Workload evictions as worker failures, preserves their cause, and retains a one-shot backend/controller task-action history in finelog for up to seven days.

## Original problem report

The user reported that
`/benjaminfeuer/rl-tasktrove-dq-sweep-30b-terminus2-qwen-20260725-163115-1ae770`
failed after task 0 exited 137. Its three coscheduled siblings were bounced, no
attempt 1 pod appeared, and the UI provided no explanation for who deleted the
pod or why the retry disappeared.

## Investigation path

1. The issue's task and attempt rows were read from the Iris controller DB. Task
   0 attempt 0 was `FAILED` with exit 137; siblings were `COSCHED_FAILED`; the
   final task error was `Job exceeded max_task_failures`.
2. `iris.task_event` contained admission, container initialization, and the
   terminal `Error`, but its one-hour retention and backend-only vocabulary did
   not preserve the controller decisions.
3. Current Kubernetes Events no longer contained the pods. The cluster API
   exposed no audit resource, confirming that `kubectl get events` was not a
   durable actor history.
4. Controller logs showed Iris's first `DELETE` for task 0 at
   `2026-07-25 21:27:44`, after the job had already terminalized. Iris cleanup
   did not cause the original pod loss.
5. CoreWeave's US-WEST audit Loki retained the missing record. At
   `2026-07-25 21:26:47`, service account
   `system:serviceaccount:kueue-system:kueue-controller-manager` patched
   `TerminationTarget=True`, reason `WorkloadEvictedDueToPreempted`, then
   deleted all four gang pods.
6. The preemptor Workload UID mapped to
   `/dlwh/iris-run-job-20260725-212435/grug-train-jaxpp-auto-checkjaxpr-std-l4-e8-b32-s128-r4-20260725-1427`.
   Its pods used `iris-interactive`; the victim used `iris-batch`. Kueue's
   prioritization was expected.
7. Code inspection found that
   `lib/iris/src/iris/cluster/backends/k8s/tasks.py:1025` recognized
   `DisruptionTarget` and a small container-reason set, but not Kueue's
   `TerminationTarget`. The generic container `Error` therefore won.

## User course corrections

- Initial access restrictions prevented live infrastructure and Weaver checks.
  The user granted full access, enabling the audit-log reconstruction instead of
  relying on the vanished Kubernetes Events.
- The user asked whether CoreWeave already retained Kubernetes actions before
  adding a Grafana-bridge mirror. That check found the regional audit Loki and
  avoided duplicating the full audit stream.

## Root cause

Kueue evicts admitted Workloads by patching each pod with
`TerminationTarget=True` and a `WorkloadEvicted*` reason before deletion. Iris's
infrastructure-failure classifier at
`lib/iris/src/iris/cluster/backends/k8s/tasks.py:1025` only accepted the native
Kubernetes `DisruptionTarget` condition. The task container consequently appeared as
`reason="Error"`, exit 137 and was classified as an application failure.

The reconcile kernel at
`lib/iris/src/iris/cluster/controller/reconcile/job.py:37` correctly charged a
`FAILED` attempt against the job-wide cumulative failure budget. With
`max_task_failures=0`, recomputation
terminalized the job and killed the just-created pending retry. The controller
made those decisions internally but did not append them to the task's finelog
timeline, so the user saw neither the Kueue actor nor the finalization chain.

## Fix

`lib/iris/src/iris/cluster/backends/k8s/tasks.py:1025` now recognizes true
`TerminationTarget` conditions whose reason begins `WorkloadEvicted`, classifies
them as infrastructure failures, and preserves the condition reason and message
on the attempt and in `iris.task_event`.

The reconcile effect contract in
`lib/iris/src/iris/cluster/controller/reconcile/batches.py:323` now emits compact
task actions for retry, coscheduled sibling cascade, and job finalization. The
commit sink writes those actions to `iris.task_event` after the SQL transaction
commits, including worker-reaping transactions. The namespace retains up to
seven days, and `iris task events /user/job/0` queries the current job
incarnation's attempts in one chronological view.

No data migration was required. `attempt_uid` is an additive nullable finelog
column, so old segments remain readable while new events are separated across
job recreations. Existing one-hour segments age out under the new policy.

## How OPS.md could have shortened this

- The Task Operations section should identify `iris task events` as the first
  command after a pod disappears and explain that it joins backend observations
  with controller actions.
- The CoreWeave section should distinguish ephemeral Kubernetes Events from CKS
  audit Loki and provide a bounded regional query by pod name and time.
- The CoreWeave guide should state that raw audit retention belongs in
  CoreWeave Observe or Telemetry Relay, while Iris retains only task-scoped
  interpreted actions.

## Artifacts

- [Issue diagnosis](https://github.com/marin-community/marin/issues/7652#issuecomment-5084922262)
- [Weaver design](https://loom.rjp.io/s/eip8pzhx/artifacts/iris-7652-design)
