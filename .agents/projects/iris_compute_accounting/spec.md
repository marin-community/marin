# Spec: iris compute accounting

Contracts and the exact queries. Every SQL block in §1 was run against the live
marin finelog (GCS parquet, DuckDB) on 2026-07-19; the output shown is real. §2
onward is the `iris.accounting` capture the Phase-2 queries depend on. No new
proto — the durable store is a finelog dataclass namespace (finelog derives the
schema from the row type, as
[`IrisProvisioning`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/autoscaler/provisioning.py#L39)
does).

## 0. How the report reads finelog

Two read paths, both DuckDB SQL over the same namespaces:

- **Offline (the weekly script):** load the GCS parquet segments for a namespace
  and register them with DuckDB. `gs://marin-us-central2/finelog/marin/<namespace>/seg_L*.parquet`.
  Segment schemas evolve (a `cluster` column was added), so concat with
  `promote_options="permissive"`.
- **Dashboard:** the same SQL through the `infra/grafana` finelog bridge
  (`GET /finelog/{cluster}/query?sql=`), which runs finelog's SELECT-gated
  `Query` RPC. CTEs are fine here; the leading-token guard that blocks `WITH` is
  only on the controller-DB `ExecuteRawQuery`, which this design does not use.

## 1. Queries that run today (Phase 1)

### Q1 — MFU per job (`telltale`)

Levanter forwards `throughput/mfu` as the gauge `levanter_throughput_mfu`, tagged
with `job_id` (the iris job root) and `run` (the wandb run). On a multi-host job
every worker forwards the same value, so collapse replicas per step before
averaging. `process_index` is NULL in the data — do not filter on it.

```sql
WITH per_step AS (
  SELECT job_id, run, ts, any_value(value) AS mfu
  FROM "telltale"
  WHERE name = 'levanter_throughput_mfu'
    AND ts >= {{from}} AND ts < {{to}}
  GROUP BY job_id, run, ts)
SELECT job_id, run, round(avg(mfu), 1) AS mean_mfu, count(*) AS steps
FROM per_step GROUP BY job_id, run ORDER BY steps DESC;
```

Real output:

```
                 job_id                          run          mean_mfu  steps
 /runner/iris-run-job-20260718-074819  canary-tpu-29636354127-1       19.1    139
 /runner/iris-run-job-20260718-113025  grug-multislice-smoke-...       4.5      9
```

`levanter_throughput_mean_mfu`, `_p10_mfu`, `_p50_mfu`, `_p90_mfu` are forwarded
too, so per-job percentiles need no extra math.

### Q2 — preemption events by pool/zone (`iris.provisioning`)

One durable row per slice provisioning outcome; `outcome='preempted'` is a
post-ready runtime loss. It is cluster/pool grain with no user/job key, so it
gives the preemption *event* rate; per-user waste comes from `iris.accounting`.

```sql
SELECT accelerator_variant, zone, count(*) AS preemptions
FROM "iris.provisioning"
WHERE outcome = 'preempted' AND ts >= {{from}} AND ts < {{to}}
GROUP BY 1, 2 ORDER BY preemptions DESC;
```

Real output (last 7 days):

```
 accelerator_variant   zone            preemptions
 v6e-4                 us-east1-d       471
 v6e-4                 us-east5-b       384
 v5p-8                 us-east5-a       311
 v5p-8                 us-central1-a    267
 v6e-8                 us-east5-b       133
```

`worker_count` is 0 on preempted rows, so provisioning yields event counts, not
chip-hours.

### Q3 — active host-hours by user (`iris.task`)

`iris.task` samples per attempt every ~5s; `task_id` is the full job path, so the
second path segment is the user (`JobName.user`). Active time per attempt is its
`ts` span. This measures host-hours (one worker); converting to chip-hours needs
the accelerator count, which `iris.task` lacks.

```sql
WITH attempt AS (
  SELECT split_part(task_id, '/', 2) AS user, task_id, attempt_id, worker_id,
         date_diff('second', min(ts), max(ts)) AS active_s
  FROM "iris.task"
  WHERE ts >= {{from}} AND ts < {{to}}
  GROUP BY 1, 2, 3, 4)
SELECT user, round(sum(active_s) / 3600.0, 1) AS host_hours,
       count(DISTINCT task_id) AS tasks
FROM attempt GROUP BY user ORDER BY host_hours DESC;
```

Real output (last 24h):

```
 user            host_hours  tasks
 eczech          21.6        445
 larry           12.4        257
 michaelryan     4.7         98
 benjaminfeuer   0.6         12
 calvinxu        0.2         4
```

Usernames are real here, so `user_id` is usable for Phase 1. The `local_admin` /
shared-user share is the coverage number Phase 1 reports before Phase 2.

## 2. Durable namespace `iris.accounting` (Phase 2)

New file `lib/iris/src/iris/cluster/controller/accounting.py`.

```python
ACCOUNTING_NAMESPACE = "iris.accounting"


class CapacityType(StrEnum):
    PREEMPTIBLE = "preemptible"
    RESERVED = "reserved"
    UNKNOWN = "unknown"       # backend with no scale_group/zone signal


class TerminalCause(StrEnum):
    """Raw terminal disposition, total over the terminal TaskState members.

        SUCCEEDED -> FINISHED     PREEMPTED -> PREEMPTED
        FAILED -> FAILED          KILLED -> KILLED
        WORKER_FAILED -> WORKER_FAILED
        UNSCHEDULABLE -> UNSCHEDULABLE   COSCHED_FAILED -> COSCHED_FAILED
    An unmapped terminal state raises (house-style fail-fast); it is never
    coerced to a default bucket."""
    FINISHED = "finished"
    FAILED = "failed"
    PREEMPTED = "preempted"
    KILLED = "killed"
    WORKER_FAILED = "worker_failed"
    UNSCHEDULABLE = "unschedulable"
    COSCHED_FAILED = "cosched_failed"


@dataclass
class IrisAccounting:
    """One terminated task attempt. Doubles as the finelog table schema.

    Emitted once, at terminalization, immutable thereafter. Chip-seconds for the
    attempt = active_seconds * device_count, device_count being chips
    (TpuDevice.count per worker summed across the slice). Nullable timestamps are
    stored NULL; a never-started attempt has active_seconds = NULL."""

    key_column: ClassVar[str] = "submitting_user"

    ts: datetime
    cluster: str
    user_id: str                 # JobName.user (second path segment of task_id)
    submitting_user: str         # authenticated principal; "local_admin" if unauth
    job_name: str                # root job path "/user/job"
    job_id: str                  # == telltale.job_id, for the MFU join
    task_id: str
    attempt_id: int
    attempt_uid: str             # read-side dedup key
    accelerator_variant: str     # via device_counts_from_json, e.g. "v5litepod-16"
    device_count: int            # chips in the slice
    zone: str | None
    region: str | None
    capacity_type: str           # CapacityType, from the worker's scale_group
    created_at_ms: int | None
    started_at_ms: int | None
    finished_at_ms: int | None
    active_seconds: float | None
    terminal_cause: str          # TerminalCause
    job_terminal_state: str      # enclosing job's terminal state (separates user-cancel)
    exit_code: int | None
    issue: int | None            # MARIN_ISSUE; None = untagged
```

**Registration.** Add an `accounting_table: Table[IrisAccounting]` field to the
`LogStack` dataclass
([log_stack.py:43](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/log_stack.py#L43))
and `client.get_table(ACCOUNTING_NAMESPACE, IrisAccounting)` to its construction.
The server default retention (no `max_age_seconds`) is confirmed durable and that
assumption is pinned in the module docstring.

**Emit.**

```python
def build_accounting_row(*, attempt, task, job, worker, worker_attrs, cluster, now) -> IrisAccounting:
    """Resolve the full row from rows all live inside the terminalization
    transaction. capacity_type/zone/region from worker/worker_attrs (present
    here, gone after remove_worker); variant + chip count from job.res_device_json
    via device_counts_from_json; issue from job.environment_json[MARIN_ISSUE].
    Pure — no I/O — so it is unit-testable."""
```

Call site: the reconcile commit path that terminalizes an attempt
([ops/worker.py](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/ops/worker.py)
for worker failure/preempt,
[reconcile/task.py](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/reconcile/task.py)
for clean/kill), ordered before `remove_worker`. Delivery guarantee (Open
Question): v1 = best-effort append + a `reconcile_accounting()` daily backstop
that re-derives missing rows for non-preempted attempts still inside DB
retention; hardening = a transactional outbox.

## 3. Phase-2 queries (over `iris.accounting`)

Week-clipped chip-hours by user × capacity type × accelerator:

```sql
SELECT coalesce(nullif(submitting_user, 'local_admin'), user_id) AS user,
       cluster, capacity_type, accelerator_variant,
       round(sum(
         greatest(0, epoch(least(finished_at_ms/1000, {{to_epoch}}))
                   - greatest(started_at_ms/1000, {{from_epoch}}))
         * device_count) / 3600.0, 1) AS chip_hours
FROM (SELECT *, row_number() OVER (PARTITION BY attempt_uid ORDER BY ts) rn
      FROM "iris.accounting") WHERE rn = 1
  AND started_at_ms IS NOT NULL
  AND finished_at_ms/1000 >= {{from_epoch}} AND started_at_ms/1000 < {{to_epoch}}
GROUP BY 1, 2, 3, 4 ORDER BY chip_hours DESC;
```

Chip-hours on preempted attempts (upper bound on waste), excluding user cancels:

```sql
SELECT accelerator_variant,
       round(sum(active_seconds * device_count) / 3600.0, 1) AS exposed_chip_hours,
       count(*) AS attempts
FROM "iris.accounting"
WHERE capacity_type = 'preemptible'
  AND terminal_cause IN ('preempted', 'worker_failed')
  AND NOT (terminal_cause = 'killed' AND job_terminal_state = 'cancelled')
  AND ts >= {{from}} AND ts < {{to}}
GROUP BY 1 ORDER BY exposed_chip_hours DESC;
```

Recovery idle (chip-time re-placing after a preempt), a self-join on ordered
attempts of a task:

```sql
SELECT round(sum((n.started_at_ms - p.finished_at_ms) / 1000.0 * n.device_count)
             / 3600.0, 1) AS recovery_idle_chip_hours
FROM "iris.accounting" p
JOIN "iris.accounting" n
  ON n.task_id = p.task_id AND n.attempt_id = p.attempt_id + 1
WHERE p.terminal_cause IN ('preempted', 'worker_failed')
  AND n.started_at_ms IS NOT NULL AND p.finished_at_ms IS NOT NULL;
```

MFU per user, device-hour-weighted, joined on `job_id` (Q1 collapse + the
chip-hours CTE), with coverage = chip-hours whose job has ≥1 MFU sample over
total chip-hours.

## 4. Dashboard panels (`infra/grafana/dashboards/compute.json`)

Each panel is one finelog-SQL query through the bridge (Infinity datasource,
`{{from}}`/`{{to}}` substituted).

| Panel | Namespace | Phase | Query |
|---|---|---|---|
| MFU per job (table) | `telltale` | 1 | Q1 |
| Preemptions by pool (bar) | `iris.provisioning` | 1 | Q2 |
| Host-hours by user (bar) | `iris.task` | 1 | Q3 |
| Chip-hours by user × capacity (stacked bar) | `iris.accounting` | 2 | §3 chip-hours |
| Preemptible waste upper bound (stat) | `iris.accounting` | 2 | §3 exposed |
| Recovery idle (stat) | `iris.accounting` | 2 | §3 recovery |

The Phase-1 panels render now; the Phase-2 panels return empty until the
namespace exists, then populate with no dashboard change.

## 5. Cron + issue tag

- `.github/workflows/ops-compute-report.yaml`, `on.schedule: '0 15 * * 1'`,
  `workflow_dispatch` (`dry_run`, `iso_week`), reusing the tunnel +
  `DISCORD_WEBHOOK_INTERNAL_DISCUSS` + `GIST_PAT` setup from
  [`ops-storage-report.yaml`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/.github/workflows/ops-storage-report.yaml).
  The Phase-2 emit is inline in the controller, so a daily `reconcile_accounting`
  guard is the only extra scheduled step.
- `MARIN_ISSUE_ENV = "MARIN_ISSUE"` in `step_runner.py`, stamped at
  `_submit_iris_job` when the launch context supplies a number (source is an
  Open Question), read onto `IrisAccounting.issue`.

## File summary

| Path | Phase | What |
|---|---|---|
| `scripts/ops/compute_report.py` | 1 | DuckDB rollup (Q1–Q3) + gist + Discord |
| `.github/workflows/ops-compute-report.yaml` | 1 | weekly cron |
| `infra/grafana/dashboards/compute.json` | 1/2 | dashboard |
| `lib/iris/src/iris/cluster/controller/accounting.py` | 2 | row, enums, `build_accounting_row`, `reconcile_accounting` |
| `lib/iris/src/iris/cluster/controller/log_stack.py` | 2 | register table (edit) |
| `lib/iris/src/iris/cluster/controller/reconcile/*`, `ops/worker.py` | 2 | emit before `remove_worker` (edit) |
| `lib/marin/src/marin/execution/step_runner.py` | 3 | `MARIN_ISSUE` stamp (edit) |

## Out of scope

- Reading placement back from the DB after terminalization — it is already gone.
- True checkpoint-aware waste (needs a Levanter checkpoint-cadence event); Phase
  2 reports the labeled upper bound plus recovery-idle.
- A first-class `labels` map on `LaunchJobRequest`; the env-var tag is the v1
  issue link.
- MFU/utilization for non-Levanter jobs and any TPU hardware duty cycle; none
  exists, coverage is reported.
- Changing `ListUsers` to group on `submitting_user`.
