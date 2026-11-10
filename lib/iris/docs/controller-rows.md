# Controller rows in Finelog

`iris.job`, `iris.job_config` and `iris.task_attempt` are durable copies of the
controller DB's `jobs`, `job_config` and `task_attempts` rows, which the
controller deletes a few days after a job ends. The controller that executes
a job writes them, after each write transaction that changes the job or
attempt: at submit, when an attempt is created, when it starts (with its node)
and when it finishes. A job handed off to a federation peer is written only by
that peer. On the hub, `cluster` is the writing controller's cluster config
name, never empty. A controller whose Finelog forwards (the CoreWeave
clusters) leaves `cluster` empty on its regional store, because a forwarder
ships only rows whose `cluster` is empty; the forwarder stamps them with its
`forwarding.cluster`, which the controller requires to equal its cluster
name. A controller whose Finelog does not forward sets `cluster` itself:
`marin` writes straight to the hub, and `marin-dev` keeps its own store. So
on a regional store, filter on `cluster = ''`; on the hub, on the cluster
name.

They copy every column of those tables, with these exceptions. Treat the
changed columns as redacted, never as the controller's own values:

| Column | Differs from the controller DB |
| --- | --- |
| `iris.job.placement` | The DB's `jobs.cluster` ("local" or the peer the job was handed to), renamed because `cluster` here is the writing controller |
| `iris.job_config.setup_scripts_json` | Replaces `environment_json`: only the setup scripts, with sensitive `NAME=value` assignments and token-like strings redacted. Environment variables and secrets are not copied, neither names nor values |
| `iris.job_config.entrypoint_json` | The command with the values of flags whose names look sensitive (`--hf-token X`, `--api-key=X`) and prefixed or high-entropy tokens redacted; setup commands as setup scripts. Not caught: a secret passed as a plain positional argument, in an ordinary-looking URL, or as a flag value that starts with `-`. Workdir file contents are never stored |
| `iris.job_config.submit_argv_json` | Every `-e`/`--env-vars` value and `--secret` source redacted, whatever its name (a `--secret` name may be redacted too); then the whole command line as in `entrypoint_json`, since the command may follow the flags without `--` |
| `iris.job_config.accelerator_*` | Added: the device request flattened (below) |

Keys: a job is `(cluster, root_job_id, root_submitted_at_ms, job_id,
submitted_at_ms)`, because job ids are reused after a job ends, and a live
parent can resubmit a finished child under the same name (the child keeps its
parent's `root_submitted_at_ms`; `submitted_at_ms` is its own submit time). All
three tables carry the job key. The two `_ms` columns are epoch milliseconds;
the other time columns are timestamps. An attempt is `attempt_uid`, and it
carries the job key plus `(task_id, attempt_id)`. `node_name` (Kubernetes) or
`worker_id` (worker VMs) places an attempt on a machine.

Joining `iris.task` or `iris.task_event`: `(task_id, attempt_id)` alone is
ambiguous, because a job name that is reused after the job ends restarts its
task and attempt ids. `iris.task_event` carries `attempt_uid`; join on it where
it is set. `iris.task` has no `attempt_uid`, so bound the join by the
attempt's lifetime as well. Match `cluster` too, since the same job name can
run on two clusters at once. On the hub, where this query is meant to run,
`marin`'s own `iris.task` rows have an empty `cluster`:

```sql
SELECT a.attempt_uid, max(t.memory_peak_mb) AS peak_memory_mb
FROM "iris.task_attempt" a
JOIN "iris.task" t
  ON COALESCE(NULLIF(t.cluster, ''), 'marin') = a.cluster
 AND t.task_id = a.task_id AND t.attempt_id = a.attempt_id
 AND t.ts >= a.started_at AND t.ts <= coalesce(a.finished_at, now())
WHERE a.root_job_id = '/user/job' AND a.state = 'TASK_STATE_SUCCEEDED'
  AND a.ts >= now() - INTERVAL '7 days' AND t.ts >= now() - INTERVAL '7 days'
GROUP BY a.attempt_uid;
```

`iris.job_config` has one row per job, written at submit, with the requested
resources. `res_device_json` is the raw device request. Three columns flatten
it: `accelerator_kind` (`gpu` or `tpu`), `accelerator_type` (for example `H100`
or `v5p-64`) and `accelerators_per_task`, counted as the scheduler counts it. A
CPU-only job has NULL, NULL and 0; test `accelerator_kind IS NULL` for
CPU-only, because a TPU request without a count also has 0. The count is per
task; multiply by `iris.job.num_tasks` for the whole job. GPU-hours per root
job over the last 7 days, from each attempt's final row. The config side has a
longer window, because a config row is written at submit and a long job may
have started weeks before its attempts finished:

```sql
SELECT a.root_job_id, c.accelerator_type,
       sum(c.accelerators_per_task * (to_unixtime(a.finished_at) - to_unixtime(a.started_at))) / 3600.0 AS accelerator_hours
FROM (
  SELECT *, row_number() OVER (PARTITION BY attempt_uid ORDER BY ts DESC, seq DESC) AS rn
  FROM "iris.task_attempt"
  WHERE ts >= now() - INTERVAL '7 days'
) a
JOIN "iris.job_config" c
  ON c.cluster = a.cluster AND c.root_job_id = a.root_job_id
 AND c.root_submitted_at_ms = a.root_submitted_at_ms
 AND c.job_id = a.job_id AND c.submitted_at_ms = a.submitted_at_ms
WHERE a.rn = 1 AND a.started_at IS NOT NULL AND a.finished_at IS NOT NULL
  AND c.accelerator_kind = 'gpu'
  AND c.ts >= now() - INTERVAL '37 days'
GROUP BY a.root_job_id, c.accelerator_type
ORDER BY accelerator_hours DESC;
```

`iris.job.tags` is reserved for submit-time job tags as a JSON object
(`json_get(tags, 'category')`); it is NULL until Iris accepts job tags at
submit.

`error` is copied in full, as the controller DB stores it (like
`iris.task_event.message`, it is not truncated).

Rows are appended, never replaced: an entity has one row per state change.
Its current state is its newest row, ordered by `ts DESC, seq DESC`. A row is
the entity's state at the end of its write transaction; `ts` is taken just
before that transaction commits, so it orders the rows of one entity but is
not the exact commit time. Use `started_at` / `finished_at` for when things
happened. Enum columns (`state`, `priority_band`, `preemption_policy`) hold
the proto enum names, such as `TASK_STATE_RUNNING`. The latest state of each
attempt of a job:

```sql
SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY attempt_uid ORDER BY ts DESC, seq DESC) AS rn
  FROM "iris.task_attempt"
  WHERE root_job_id = '/user/job' AND ts >= now() - INTERVAL '7 days'
) WHERE rn = 1;
```

Rows can be lost without blocking the controller: a failed snapshot (logged
as `controller rows: snapshot failed`), a failed enqueue (logged as
`controller rows: write failed`, for that table only) or a failed Finelog
flush drops those rows. To find gaps, compare daily row counts with the
controller DB.

History is kept in the archive (remote retention is `retain_forever`). The
storage policies (5 years; 8 GiB for `iris.task_attempt`, 4 GiB each for the
other two) bound only the hub's local copy; older segments are read back from
the archive.
