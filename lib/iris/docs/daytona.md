# Daytona backend (proof of concept)

A cluster config with `daytona_provider` runs every task attempt in its own
[Daytona](https://www.daytona.com/docs) v2 sandbox. Like the Kubernetes backend,
it dispatches tasks directly: there are no Iris workers, and Daytona places the
sandboxes.

```yaml
name: daytona-poc
storage:
  remote_state_dir: file:///var/lib/iris/remote
daytona_provider:
  api_key_env: DAYTONA_EVAL_API_KEY            # variable that holds the key
  default_task_image: ghcr.io/marin-community/iris-task:latest
  controller_address: ""                       # set only if sandboxes can reach the controller
  sandbox_ttl_minutes: 360
```

Install the `daytona` extra with the controller extra and start the controller as
usual:

```bash
uv run --package marin-iris --extra controller --extra daytona \
  python -m iris.cluster.controller.main serve --config daytona-poc.yaml \
  --host 127.0.0.1 --port 10000 --state-dir /tmp/iris-daytona --fresh
uv run iris --controller-url http://127.0.0.1:10000 job run -- python main.py
```

## How an attempt runs

The controller performs all provider I/O, so a sandbox never connects to it:

1. It pulls the task image (or `default_task_image`) as a named Daytona image.
2. It creates a sandbox labelled with the attempt UID, with a fixed shape of
   `cpu / 2` cores (one core is two vCPUs) and the requested memory. Its
   lifecycle policy deletes it after `min(sandbox_ttl_minutes, task timeout +
   10 minutes)`, or after `sandbox_ttl_minutes` for a task without a timeout.
3. It uploads the code bundle and workdir files to `/app` and the job's secrets
   to `/run/rigging/secrets`, then starts the task script (setup commands, then
   the run command) as a sandbox process. The process receives the cluster's
   `defaults.task_env`, the job's variables and the Iris variables, as a
   Kubernetes pod does.
4. A thread follows the process log into finelog under the attempt's key and
   resumes the stream from its cursor after a disconnect; if the stream stays
   down, the exit status still comes from the process record. Reconcile
   reports `RUNNING` once the process start is requested, then the exit status.
5. When the attempt leaves the desired set (finished, cancelled or timed out),
   its launch stops and its sandbox is deleted. A restarted controller deletes
   the cluster's leftover sandboxes and the controller retries their attempts.

The attempt stays `ASSIGNED` until its process start is requested, so a launch
failure before that retries without charging the task. While Daytona is full,
the launch waits with backoff, as a pod waits for a node: nothing but
cancellation ends that wait. A sandbox that has not started 10 minutes after its
create request fails as `WORKER_FAILED` and retries for free. A failed image,
an image not ready within 10 minutes, or an invalid workdir path fails the
attempt as `FAILED`. A start request whose answer is lost fails the attempt
after `RUNNING`, which charges the preemption budget, because the process may
have run.

`iris task exec` and Shellbox's Iris backend run commands in the attempt's
sandbox.

## Limits

- CPU tasks only: jobs that request GPUs or TPUs, or coscheduling, are rejected.
- Tasks cannot reach each other or advertise ports, so actors, coordinators and
  multi-host training do not work.
- Child jobs, endpoints and task tokens need `controller_address` reachable from
  Daytona; with it empty, tasks run without controller access.
- Egress allowlists are unavailable; sandboxes have internet access.
- Logs are copied by the controller, so a controller restart loses the running
  attempts.
