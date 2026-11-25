# Observability rows: controller DB copies in Finelog

The `iris.controller_db__*` namespaces are durable copies of the controller DB's
tables of the same names: `jobs`, `job_config`, `tasks` and `task_attempts`. The
controller deletes those rows a few days after a job ends. The other `iris.*`
namespaces, such as `iris.task` and `iris.task_state`, are measurements and
events, not copies of the controller DB. The controller that executes
a job writes them, after each write transaction that changes the job or
attempt: at submit, when an attempt is created, when it starts (with its node)
and when it finishes. A job handed off to a federation peer is written only by
that peer. `cluster` is the writing controller's cluster config name, never
empty, on its own store and on the hub. A CoreWeave cluster's Finelog
forwards the rows named for its `forwarding.cluster`, and each cluster config
sets that to the cluster's name, which a test checks.

## Which columns are copied

They copy a chosen set of columns, not a mirror. **The rule:** copy a column
when the controller or a fixed vocabulary writes its value (ids, keys, states,
enums, times, counts, resource numbers, policies, placement). Do not copy a
column whose value is text the user wrote. Identifiers the user chose that
name or place the work, such as the job name, user, `task_image` and
`constraints_json`, are copied, since queries group and filter by them;
`coscheduling_group_by` names a worker attribute, from the same vocabulary as
the constraint keys.
`task_image` is an image reference (`registry/path:tag`), which is needed to
know what ran; a container runtime rejects credentials in a reference, and
registry auth comes from the node's credential helper. There are two
exceptions, both below: the free-text `error` and `status_message` columns,
redacted, and the launch's git identity, read from the environment into the
`provenance_*` columns. Rows are kept forever, so any other column that fails
the rule is left out, not redacted and kept.

Not copied, all from `job_config`:

| DB column | Why |
| --- | --- |
| `environment_json` | Environment variables and secrets; only the launch's git identity is read from it, into `provenance_*` (below) |
| `entrypoint_json`, `submit_argv_json` | Command lines and setup commands the user wrote |
| `bundle_id`, `ports_json`, `egress_allow_json` | User-written ids, ports and hosts |
| `name` | Same as `iris.controller_db__jobs.name` |

Not copied from `tasks`: `priority_neg_depth` and `priority_insertion`, the
scheduler's queue order, which no question about past work needs; and
`current_worker_address`, which `iris.worker` records per worker. A column can be
added later when a query needs it, but never removed.

Copied columns that differ from the controller DB:

| Column | Differs from the controller DB |
| --- | --- |
| `iris.controller_db__jobs.placement` | The DB's `jobs.cluster` ("local" or the peer the job was handed to), renamed because `cluster` here is the writing controller |
| `iris.controller_db__job_config.accelerator_*` | Added: the device request flattened (below) |
| `iris.controller_db__job_config.provenance_*` | Added: four fields of `environment_json`'s `MARIN_PROVENANCE` (below) |
| `iris.controller_db__tasks.placement` | The DB's `tasks.cluster`, as for jobs |
| `iris.controller_db__tasks.task_submitted_at` | The DB's `tasks.submitted_at_ms`, renamed because `submitted_at_ms` here is the job key |

**When the controller DB schema changes:**
`test_every_controller_db_column_is_classified` in
`tests/cluster/controller/test_observability_rows.py` lists every column of the four
tables as COPIED (with its Finelog field) or EXCLUDED (with a reason), and fails
on a column it does not list. A new DB column needs a one-line decision there by
the rule above; copying it also adds a nullable field to the row dataclass. A
dropped DB column that was copied keeps its Finelog field, which becomes
`| None = None` and reads NULL from then on: that change adds a retired
classification to the test, which checks the field defaults to None, and lists
the column and the date here. Finelog columns
are never removed or renamed.

**Free text:** the `error` columns of jobs, tasks and attempts, and
`tasks.status_message` (the backend's reason a task is not running yet, such as
a scheduler or Kueue message), stay readable, with token-like strings redacted. That redaction is
best-effort: an error can still echo a short value or URL from a command.

**Launch provenance:** `iris.controller_db__job_config` keeps four fields of
the job's `MARIN_PROVENANCE` environment variable, as parsed by
`rigging.provenance.Provenance.from_json`: `provenance_base_commit`,
`provenance_tree_hash`, `provenance_dirty` and `provenance_branch` (token-like
strings redacted, as for the free text). The rest of that variable is not
copied: the command line and the remote URL are user text that can hold a
credential, and the user and launch time are already in
`iris.controller_db__jobs`. Every other environment variable, and every secret,
is dropped. All four are NULL when the variable is unset or does not parse; the
job's rows are still written.

**The one place a row is not written on every change:** an `iris.controller_db__tasks`
row is written at submit and when the task's state, current attempt, current
worker, error or exit code changes. A change to any other column alone
(`status_message`, `container_id`, `backend_id`)
writes no row, because Kubernetes rewrites the pending reason on every
reconcile. So those columns hold their value as of the row's change, and
pending-reason history is kept only at those trigger points. Likewise, a worker
observation writes an `iris.controller_db__task_attempts` row only when it
changes the attempt row; repeating the attempt's state, as a status-message
refresh does, writes none.

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
as `observability rows: snapshot failed`), a failed enqueue (logged as
`observability rows: write failed`, for that table only) or a failed Finelog
flush drops those rows. To find gaps, compare daily row counts with the
controller DB.

History is kept in the archive (remote retention is `retain_forever`). The
storage policies (5 years; 8 GiB each for `tasks` and `task_attempts`, 4 GiB
each for `jobs` and `job_config`) bound only the hub's local copy; older segments are read back from
the archive.

## Backfilling rows from before the live path

**Rows cannot be removed once written.** Finelog has no row delete, and
dropping a table drops the live rows too. A wrong backfill can only be
filtered out of queries, as `ts < live_start` for its cluster.

The live path writes rows only for changes after a controller with it starts
(`live_start`). `lib/iris/scripts/backfill_controller_rows.py` writes, once per
cluster, one row for each job, job config, task and attempt in a controller DB
checkpoint that has no row yet: its state in the checkpoint, built by the live
path's own queries and row builders, with the same redaction. Its `ts` is the
entity's latest own time (finish, start, or submit), capped at
`live_start - 1 ms`, so a live row for the same key always sorts after it, even
one written during the run; such a key ends up with both rows, the live one
current. A key that already has a row is skipped, so a second run writes nothing.

What a backfilled row cannot hold:

- History: one row per entity, its state in the checkpoint, not one per change.
- Jobs pruned before the checkpoint: the controller deletes terminal jobs 7
  days after they finish, so the backfill reaches about 7 days back.
- `worker_id` of an attempt, and `current_worker_id` of a task, whose worker
  was removed: NULL, because only the live path keeps the cleared value.
- `ts` is the entity's own time, not when its row was written.

Before writing, the script refuses a `--live-start-ms` less than 15 minutes
old, later than the checkpoint, earlier than the live rows' real start (a job
submitted after it has no live config row), or later than it (rows before it
already exist, unless `--rerun` declares an earlier run). It reads Finelog's
keys one table and day at a time, halving a failed window down to an hour, and
stops without writing when that fails. It writes 5,000-row batches, one table
after another, flushing each and stopping at the first that does not land; a
rerun writes the rest.

```bash
# In a job on the cluster, from the checkpoint taken after live_start:
uv run python lib/iris/scripts/backfill_controller_rows.py dry-run \
  --cluster marin --live-start-ms <ms> --checkpoint <checkpoint_dir_url>
# Then run, with the confirmation value the dry run prints, and check:
uv run python lib/iris/scripts/backfill_controller_rows.py run ... \
  --i-understand-rows-cannot-be-removed=marin:<ms>
uv run python lib/iris/scripts/backfill_controller_rows.py check ...
```

`check` exits 1 when any DB entity has no row. Delivery is at least once, so a
retried batch can land twice; the copies are identical, and counts should use
distinct keys.

## Cost to the controller, and turning the copy off

The rows are read just before each write transaction commits and enqueued just
after, both under the controller's write lock, so every other writer waits for
them. The controller emits to `telemetry_v1.iris` (`service = 'iris-controller'`),
always after releasing the lock:

| Name | Kind | Meaning |
| --- | --- | --- |
| `iris_observability_rows_transaction` | event, one per transaction that copies rows | `body_json` holds `job_rows`, `job_config_rows`, `task_rows`, `attempt_rows`, their total `rows`, `snapshot_seconds`, `enqueue_seconds` and their total `seconds` |
| `iris_controller_write_lock_held_seconds` | delta counter, once a minute | Time write transactions held the lock (not maintenance such as WAL checkpoints) |
| `iris_controller_write_lock_wait_seconds` | delta counter, once a minute | Time write transactions waited to acquire it |

The event holds counts and durations only: no job ids, users or names. One event
per transaction pairs each burst's row count with its time, so the per-row cost
and the fixed cost per transaction can be read off, and so can how close a table
gets to the Finelog client's 10,000-row buffer. Expect at most one event per
reconcile tick that changes rows, plus one per submit: on the order of 10,000 a
day on `marin`. The controller's RPC telemetry in
`telemetry_v1.iris.rpc` writes about 9 million rows a day.

The lock counters are summed in a window that closes with the first write
transaction ending 60 s or more after it opened, so a quiet controller emits
longer windows, and the last window before shutdown is not emitted. A
transaction that rolls back after its snapshot emits no row event, though its
lock time is counted. The wait is timed around `acquire()`; it is near zero for
a re-entrant acquire by the thread already holding the lock.

A transaction whose rows take over 100 ms also logs `observability rows: N rows
took M ms under the write lock`, at most once a minute.

Row time against rows, per hour (the slope is the per-row cost):

```sql
SELECT date_trunc('hour', to_timestamp_millis(timestamp_ms)) AS hour,
       count(*) AS transactions,
       max(CAST(json_get(body_json, 'rows') AS BIGINT)) AS max_rows,
       max(CAST(json_get(body_json, 'seconds') AS DOUBLE)) AS max_seconds,
       sum(CAST(json_get(body_json, 'seconds') AS DOUBLE)) AS row_seconds
FROM "telemetry_v1.iris"
WHERE service = 'iris-controller' AND name = 'iris_observability_rows_transaction'
  AND timestamp_ms >= <start_ms> AND timestamp_ms < <end_ms>
GROUP BY 1
ORDER BY 1;
```

Lock time per hour, to compare `row_seconds` against:

```sql
SELECT date_trunc('hour', to_timestamp_millis(timestamp_ms)) AS hour, name, sum(value) AS seconds
FROM "telemetry_v1.iris"
WHERE service = 'iris-controller'
  AND name IN ('iris_controller_write_lock_held_seconds', 'iris_controller_write_lock_wait_seconds')
  AND timestamp_ms >= <start_ms> AND timestamp_ms < <end_ms>
GROUP BY 1, 2
ORDER BY 1, 2;
```

`telemetry_v1.iris` has no rows before this change ships, so check after the
first deploy that both queries return data.

To turn the copy off, set it in the cluster config and restart the controller:

```yaml
finelog:
  config: marin
  controller_rows: false
```

The controller then attaches no row tables: write transactions mark, read and
write no rows, and the controller logs `observability rows are off` at start.
It still emits the lock counters, so lock time before and after the switch can
be compared. Rows for changes made while it is off are not
written later.
