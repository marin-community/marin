---
name: babysit-zephyr
description: Launch and babysit Zephyr pipeline jobs on Iris.
---

# Babysit Zephyr

Start, monitor, and recover Zephyr pipeline jobs on Iris. Escalate diagnosis to
debug.

## Job model and commands

A Zephyr job has a *-coord coordinator and a *-workers pool. One job can run
sequential pipelines (p<N> in child names); that is normal. A new hash with
the same p0 is a retry, and old coordinators can linger. Child names follow
'<hash>-p<pipeline>-a<attempt>-{coord,workers}'.

Resolve the user's cluster name to 'lib/iris/config/<name>.yaml' as described by
babysit-job; substitute that path for '<CONFIG>' below.

Dashboard:

~~~bash
uv run iris --config <CONFIG> cluster dashboard
~~~

Submit the user's command with --no-wait; increase the default 1 GB entrypoint
memory for stateful long-running pipelines:

~~~bash
uv run iris --config <CONFIG> job run --region <REGION> --no-wait -- python <SCRIPT>
uv run iris --config <CONFIG> job run --region <REGION> --memory 5GB --no-wait -- python <SCRIPT>
~~~

Record the returned job ID. To stop or restart, ask the user first: job cancel
kills the coordinator and all workers.

~~~bash
uv run iris --config <CONFIG> job cancel <JOB_ID>
~~~

## Monitor

Check child state and resource usage:

~~~bash
uv run iris --config <CONFIG> rpc controller list-tasks --job-id <JOB_ID>
~~~

Healthy state is a RUNNING coordinator with one task and RUNNING workers ramping
toward their target. diskMb updates about every 60 seconds and is always zero
on Kubernetes because the workdir is inside the pod.

Read coordinator progress and, when logs are flooded by worker RPCs, filter
pull_task, Started operation, report_result, registered, and tasks completed
entries:

~~~bash
uv run iris --config <CONFIG> rpc controller get-task-logs \
  --id <COORD_JOB_ID> --max-total-lines 5000 --attempt-id -1 --tail
~~~

When parsing returned entries:

~~~python
for entry in task_logs:
    message = entry.get("data", "")
    if any(term in message for term in ("pull_task", "Started operation", "report_result", "registered", "tasks completed")):
        continue
    print(message)
~~~

Progress must show shards completing and stages advancing. A useful thread dump
when logs are noisy is:

~~~bash
uv run iris --config <CONFIG> rpc controller profile-task \
  --json '{"target":"<COORD_JOB_ID>/0","durationSeconds":1,"profileType":{"threads":{}}}'
~~~

actor-method_0 in _wait_for_stage and _coordinator_loop indicate active work.
Threads all idle in _worker indicate an exited, zombie coordinator.

Smoke-check child jobs and early errors during the first 2–5 minutes. Then check
often enough to observe stage transitions (minutes for short stages, 15–30
minutes for long ones). If workers are killed or the coordinator is zombie,
inspect the newest hash/attempt because StepRunner may retry automatically.
“Terminated by user” is not proof of a human cancellation; inspect parent,
coordinator, and worker logs.

Ask before canceling and resubmitting after failure. Escalate to debug for a
stuck stage, stragglers, repeated worker errors, or controller/RPC failures.
