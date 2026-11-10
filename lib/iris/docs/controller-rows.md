# Controller rows in Finelog

The `iris.controller_db__*` namespaces are durable copies of the controller DB's
tables of the same names: `jobs`, `job_config`, `tasks` and `task_attempts`. The
controller deletes those rows a few days after a job ends. The other `iris.*`
namespaces, such as `iris.task` and `iris.task_state`, are measurements and
events, not copies of the controller DB. The controller that executes
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
| `iris.controller_db__jobs.placement` | The DB's `jobs.cluster` ("local" or the peer the job was handed to), renamed because `cluster` here is the writing controller |
| `iris.controller_db__job_config.setup_scripts_json` | Replaces `environment_json`: only the setup scripts, **names only** (below). Environment variables and secrets are not copied, neither names nor values |
| `iris.controller_db__job_config.entrypoint_json` | The command and setup commands, **names only** (below). Workdir file contents are never stored |
| `iris.controller_db__job_config.submit_argv_json` | The submitter's command line, **names only** (below) |
| `iris.controller_db__job_config.accelerator_*` | Added: the device request flattened (below) |
| `iris.controller_db__tasks.placement` | The DB's `tasks.cluster`, as for jobs |
| `iris.controller_db__tasks.task_submitted_at` | The DB's `tasks.submitted_at_ms`, renamed because `submitted_at_ms` here is the job key |

**Names only:** command lines and scripts keep only command words, flag names
(`--lr`, `-e`, cut at `=` or after a short flag's letter) and assignment names
(`NAME=`). A command word is the first token of a statement (the line, or after
`&&`, `||`, `;`, `|` or `&`) that is neither a flag nor an assignment, and it is
kept only if it looks like a program name (`python`, `./run.sh`) with no
token-like string in it. So a short, plain-looking value written as the very
first word of a statement is still kept; nothing else in a command is. Every value, positional argument and quoted string,
including a whole `bash -c "..."` body and everything after `--`, is
`<redacted>`. Backslash-continued lines are joined first, every line of a
heredoc body is `<redacted>`, and a line that does not parse is `<unparsed>`. In
`submit_argv_json`, the user command after the first `--` starts a new command
word. So `python train.py --hf-token X --lr 3e-4` is stored as
`python <redacted> --hf-token <redacted> --lr <redacted>`.

**Free text:** the `error` columns of jobs, tasks and attempts, and
`tasks.status_message` (the backend's reason a task is not running yet, such as
a scheduler or Kueue message), stay readable, with token-like strings redacted. That redaction is
best-effort, not names-only: an error can still echo a short value or URL from
a command.

**The one place the copy is not every change:** an `iris.controller_db__tasks`
row is written at submit and when the task's state, current attempt, current
worker, error or exit code changes. A change to any other column alone
(`status_message`, `container_id`, `current_worker_address`, `backend_id`)
writes no row, because Kubernetes rewrites the pending reason on every
reconcile. So those columns hold their value as of the row's change, and
pending-reason history is kept only at those trigger points.

Keys: a job is `(cluster, root_job_id, root_submitted_at_ms, job_id,
submitted_at_ms)`, because job ids are reused after a job ends, and a live
parent can resubmit a finished child under the same name (the child keeps its
parent's `root_submitted_at_ms`; `submitted_at_ms` is its own submit time). All
four tables carry the job key. The two `_ms` columns are epoch milliseconds;
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
FROM "iris.controller_db__task_attempts" a
JOIN "iris.task" t
  ON COALESCE(NULLIF(t.cluster, ''), 'marin') = a.cluster
 AND t.task_id = a.task_id AND t.attempt_id = a.attempt_id
 AND t.ts >= a.started_at AND t.ts <= coalesce(a.finished_at, now())
WHERE a.root_job_id = '/user/job' AND a.state = 'TASK_STATE_SUCCEEDED'
  AND a.ts >= now() - INTERVAL '7 days' AND t.ts >= now() - INTERVAL '7 days'
GROUP BY a.attempt_uid;
```

`iris.controller_db__job_config` has one row per job, written at submit, with the requested
resources. `res_device_json` is the raw device request. Three columns flatten
it: `accelerator_kind` (`gpu` or `tpu`), `accelerator_type` (for example `H100`
or `v5p-64`) and `accelerators_per_task`, counted as the scheduler counts it. A
CPU-only job has NULL, NULL and 0; test `accelerator_kind IS NULL` for
CPU-only, because a TPU request without a count also has 0. The count is per
task; multiply by `iris.controller_db__jobs.num_tasks` for the whole job. GPU-hours per root
job over the last 7 days, from each attempt's final row. The config side has a
longer window, because a config row is written at submit and a long job may
have started weeks before its attempts finished:

```sql
SELECT a.root_job_id, c.accelerator_type,
       sum(c.accelerators_per_task * (to_unixtime(a.finished_at) - to_unixtime(a.started_at))) / 3600.0 AS accelerator_hours
FROM (
  SELECT *, row_number() OVER (PARTITION BY attempt_uid ORDER BY ts DESC, seq DESC) AS rn
  FROM "iris.controller_db__task_attempts"
  WHERE ts >= now() - INTERVAL '7 days'
) a
JOIN "iris.controller_db__job_config" c
  ON c.cluster = a.cluster AND c.root_job_id = a.root_job_id
 AND c.root_submitted_at_ms = a.root_submitted_at_ms
 AND c.job_id = a.job_id AND c.submitted_at_ms = a.submitted_at_ms
WHERE a.rn = 1 AND a.started_at IS NOT NULL AND a.finished_at IS NOT NULL
  AND c.accelerator_kind = 'gpu'
  AND c.ts >= now() - INTERVAL '37 days'
GROUP BY a.root_job_id, c.accelerator_type
ORDER BY accelerator_hours DESC;
```

`iris.controller_db__jobs.tags` holds the job's tags as a JSON object
(`json_get(tags, 'category')`), or NULL for an untagged job; jobs submitted
before tags existed are untagged.

`error` is not truncated, but token-like strings in it are redacted (see
**Free text** above).

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
  FROM "iris.controller_db__task_attempts"
  WHERE root_job_id = '/user/job' AND ts >= now() - INTERVAL '7 days'
) WHERE rn = 1;
```

Rows can be lost without blocking the controller: a failed snapshot (logged
as `controller rows: snapshot failed`), a failed enqueue (logged as
`controller rows: write failed`, for that table only) or a failed Finelog
flush drops those rows. To find gaps, compare daily row counts with the
controller DB.

History is kept in the archive (remote retention is `retain_forever`). The
storage policies (5 years; 8 GiB each for `tasks` and `task_attempts`, 4 GiB
each for `jobs` and `job_config`) bound only the hub's local copy; older segments are read back from
the archive.
