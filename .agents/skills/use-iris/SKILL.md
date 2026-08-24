---
name: use-iris
description: Use Iris to submit, inspect, debug, monitor, or recover jobs and tasks; diagnose scheduling and federation; deploy controllers; or reserve dev GPUs and TPUs. Use for ordinary Iris operations, including babysitting a job, controller rollouts, accelerator sessions, and stuck CoreWeave pods. Use a narrower production-run or Zephyr skill when that workflow is explicitly requested.
---

# Use Iris

Read only the material needed for the request:

- Normal jobs, tasks, scheduling, auth, or CoreWeave: `lib/iris/OPS.md`.
- Federation: `lib/iris/docs/federation.md`.
- Continuous job monitoring: [references/monitor-job.md](references/monitor-job.md).
- Controller deploy or rollback: [references/controller-rollout.md](references/controller-rollout.md).
- Interactive GPU or TPU: [references/dev-accelerators.md](references/dev-accelerators.md).
- Stuck terminating CoreWeave pod: [references/stuck-pod.md](references/stuck-pod.md).
- Logs or measurements: use `query-finelog`.

Resolve cluster facts from `lib/iris/config/<cluster>.yaml`; do not copy live coordinates from memory.

## Common reads

```bash
uv run iris --cluster=<cluster> job describe <job>
uv run iris --cluster=<cluster> task describe <task>
uv run iris --cluster=<cluster> task events <task>
uv run iris --cluster=<cluster> rpc controller list-backends
```

For a pending federated root, inspect all three parent-side views:

```bash
uv run iris --cluster=<parent> job list --prefix <root-job>
uv run iris --cluster=<parent> rpc controller list-peers
uv run iris --cluster=<parent> query \
  "SELECT job_id, peer_id, handoff_state FROM federated_jobs WHERE job_id='<root-job>'"
```

Only root jobs federate; their whole tree stays on the peer. Parent `job describe` is the liveness source, while forwarded logs may lag. CoreWeave tasks normally read regional S3 and GCP tasks read GCS.

## Boundaries

- Start read-only and name the evidence that distinguishes each cause.
- Never run `iris cluster restart` without explicit approval for the named cluster; it kills all workers and jobs.
- Treat a controller restart as a deployment and require an explicitly named target.
- Cancel, complete, fail, preempt, resubmit, or change Kubernetes state only when the request or selected reference authorizes that exact action.
- Avoid `kubectl describe pod` on task pods because it can print environment values.
