---
name: use-iris
description: Operate, inspect, submit, or diagnose Iris jobs, tasks, clusters, scheduling, federation, and CoreWeave behavior. Use for ordinary Iris operational questions and cross-cutting investigations; pair it with a narrower workflow skill when the request is to babysit a job, deploy controllers, reserve hardware, recover a Kubernetes pod, or profile training.
---

# Use Iris

Treat the checked-in operator manuals and cluster configs as the source of truth. Use this skill to select the right state surface and workflow, then load only the relevant section.

## Route the task

- Read `lib/iris/OPS.md` for CLI access, jobs and tasks, scheduler state, CoreWeave inspection, auth, controller state, and live troubleshooting.
- Read `lib/iris/docs/federation.md` for peer routing, root-job handoff, parent/peer state, object-store boundaries, or proxied serving endpoints.
- Read `lib/iris/config/<cluster>.yaml` to resolve a named cluster's backend, Kubernetes context, namespace, and other operational facts. Do not reproduce live coordinates from memory.
- Use `query-finelog` and read `lib/finelog/OPS.md` for logs, resource measurements, profiles, telemetry, or forwarding.
- Use the matching narrow skill for a stateful or mutating workflow: `babysit-job`, `babysit-zephyr`, `deploy-iris-controllers`, `reserve-gpu`, `reserve-tpu`, `recover-stuck-k8s-pod`, or `profile-training`.

## Establish the state boundary

1. Resolve the cluster and canonical job or task ID. Prefer `--cluster=<name>` for checked-in configs and `--config=<path>` for a custom or pinned file.
2. Start with the narrowest read-only Iris view: `job list`, `job describe`, `task describe`, `task events`, `rpc controller list-backends`, or `cluster status` as appropriate.
3. Query controller SQLite only for registry and decisions such as job state, assignments, scheduling, and federation handoff. Use Finelog for time-series measurements and logs.
4. For Kubernetes, inspect projected `kubectl get` views before lower-level objects. Avoid `kubectl describe pod` on task Pods because it can print environment values.
5. Explain which observation separates each hypothesis before proposing a mutation.

## Handle federation

Submit a federated workload through its parent and reason about both controllers:

- Only whole root jobs federate. Pin the coordinator/root with `--target-cluster`; its descendants remain on that peer.
- `rpc controller list-peers` reports reachability, shapes, and advertised availability. The parent's `federated_jobs` table reports queued, promoted, and handed-off state.
- A pre-handoff job has no task on the peer. Read the pending reason before treating the absence as a failure.
- `job describe` on the parent mirrors peer task state and is the liveness source. Logs forward asynchronously through Finelog and may lag.
- Credentials do not cross clouds. CoreWeave task Pods normally read their regional S3 store, while GCP tasks read GCS. Move inputs or choose placement accordingly.

For a pending federated root, always collect the complete parent-side triad before narrowing the cause:

```bash
uv run iris --cluster=<parent> job list --prefix <root-job-id>
uv run iris --cluster=<parent> rpc controller list-peers
uv run iris --cluster=<parent> query \
  "SELECT job_id, peer_id, handoff_state FROM federated_jobs WHERE job_id='<root-job-id>'"
```

Use the job's pending reason, peer reachability and advertised backend shape, handoff state, and availability together. Do not infer a peer failure from the absence of a task before handoff.

## Respect mutation boundaries

- Never run `iris cluster restart` without explicit approval for the named cluster; it destroys all workers and jobs.
- Treat a controller restart as a deployment. Require an explicitly named target and use `deploy-iris-controllers`.
- Job cancel, complete, fail, preempt, resubmit, and direct Kubernetes changes mutate shared state. Perform them only when the request or selected narrow workflow authorizes the exact action.
- A read-only diagnosis does not authorize credential changes, cluster repair, or a speculative restart.

Report the commands inspected, the evidence found, remaining uncertainty, and the next approval boundary.
