---
name: babysit-job
description: Monitor an Iris job and recover it on failure. Use when asked to babysit or watch a job or run.
---

# Babysit an Iris job

For Zephyr pipelines use `babysit-zephyr`. Otherwise monitor and recover at the
job level. Required before starting: canonical `job_id` (`/<user>/<job>`), Iris
config, and an exact resubmit command containing `--no-wait`. Resolve shorthand
cluster names from `lib/iris/config/`; TPU resubmits need `--extra marin-core:tpu`
and `--tpu <variant>` (`--reserve` alone does not attach devices). Ask for any
missing field. TPU bad-node errors go to `debug`.

Recovery is cancel then resubmit. Cluster actions are out of scope: never
restart, recreate, or mutate a cluster without explicit consent. Keep one
monitor loop per job and assign one owner. Continue until successful terminal
state plus user acknowledgement, a requested stopping point, or an
unrecoverable error. Ferry-scale runs often take 4–5 hours.

After submit/restart, sleep 120 once; otherwise use the 570-second cadence. A
long-running monitor may be polled in ~30-second tool chunks. Check state-file
and stdout/event-log freshness; after two missed cadences report `monitor stale`
separately from `run unhealthy`. A blocked query is inconclusive: cross-check
W&B, logs, checkpoint movement, worker health, and monitor state.

## State and checks

Write `scratch/<YYYYMMDD-HHMM>_monitoring_state.json` and track at least:
`ts`, `job_id`, `config`, `resubmit_command`, and `restart_count`. For resident
`marin-mcp-babysitter`, also record controller URL, cluster, sessions, ports,
and logs; verify with `iris_job_summary` and `iris_tail_logs`. Restart only a
smoke-test tunnel/session on controller connection refusal; never the cluster.
Start it with a stable controller URL and streamable HTTP when this path is
needed:

```bash
uv run --package marin-core marin-mcp-babysitter --controller-url <URL> \
  --cluster <CLUSTER> --transport streamable-http --host 127.0.0.1 --port <PORT>
```

Each cadence:

```bash
uv run iris --config <CONFIG> job logs --since-seconds 900 <JOB_ID> | \
  rg -i -e "loss|error|traceback|exception|resource_exhausted|oom|compiler_base\.cc:2587|program hbm requirement|largest program allocations|ownerdiederror|dead node|node death|autoscaler unsatisfied resources|no accelerator found|failed_precondition|device or resource busy"
uv run iris --config <CONFIG> job list --prefix <JOB_ID>
```

`job list --prefix` requires canonical names. Treat `pending_reason` capacity
wait as scheduler waiting, not a cluster fault. `RUNNING` is only a controller
signal; confirm W&B/checkpoint progress. On terminal or OOM-like signals, get:

```bash
uv run iris --config <CONFIG> job describe <JOB_ID>
```

Record exact final step from config/code, W&B identity/progress, metrics, output
root, and checkpoint path. Training is live while W&B is catching up only when
timestamped progress advances; once W&B appears, its steps/timestamps must move.

## Recovery and completion

Fix only small obvious errors (for example `NameError`, `ImportError`,
`SyntaxError`, or clear `KeyError`) before one recovery. Stop and report OOM,
XLA HBM, distributed/data failures, unclear traces, or a repeated same error.
TPU/XLA HBM patterns include `Program hbm requirement` and `Largest program
allocations in hbm`; noisy shutdown traces alone do not prove failure.

```bash
uv run iris --config <CONFIG> job cancel <JOB_ID>
<RESUBMIT_COMMAND>
```

Capture the new canonical ID, increment `restart_count`, update state, and
resume the cadence. If a fixed `--job-name` reuses an ID, rely on terminal state
and updated metadata rather than assuming it is a new job.

Before completion, verify successful terminal state, W&B final state when used,
and `metadata.json` in the expected checkpoint. Stop obsolete heartbeats and
resident sessions. Handoff includes job ID, latest signal/error, W&B links, and
resubmit metadata.
