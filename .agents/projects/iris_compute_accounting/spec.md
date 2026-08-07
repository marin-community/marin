# Spec: iris compute accounting

The queries and the public surface of `scripts/ops/compute_report.py`. Every SQL
block in §1 is what the script runs; the numbers were produced against the live
marin finelog (GCS parquet, DuckDB) for 2026-W29.

## 0. How the report reads finelog

DuckDB over the GCS parquet segments of each namespace, following
`scripts/ops/cross_region.py`: list a namespace's objects, select those whose
mtime is within the window (a segment's mtime ≥ its rows' `ts`, so
`mtime ≥ window.start − slack` is a safe lower bound), download, and
`read_parquet(..., union_by_name=true)` (segment schemas evolve — a `cluster`
column was added). The finelog `remote_log_dir`
(`gs://marin-us-central2/finelog/marin`) resolves from the Iris config via
`load_finelog_config`. The Grafana bridge runs the same SQL server-side.

## 1. Queries (verified)

### Chip-hours by user, preemptible vs reserved (`iris.task`)

`worker_id` is `marin-tpu-<gen>-<capacity>-<slice_chips>-<zone>-...-worker-<n>`.
Per-host chips = a slice's chips divided by its live hosts (SPMD → every host
emits rows). `ondemand` folds into reserved.

```sql
WITH parsed AS (
  SELECT split_part(task_id, '/', 2) AS user,
         regexp_extract(worker_id,
           '^marin-tpu-([a-z0-9]+)-(reserved|preemptible|ondemand)-([0-9]+)-',
           ['generation', 'cap', 'chips']) AS p,
         regexp_replace(worker_id, '-worker-[0-9]+$', '') AS slice_id, worker_id, ts
  FROM task
  WHERE worker_id LIKE 'marin-tpu-%' AND ts >= {{from}} AND ts < {{to}}),
host AS (
  SELECT user,
         CASE WHEN p.cap = 'preemptible' THEN 'preemptible' ELSE 'reserved' END AS capacity,
         p.generation AS generation, slice_id, CAST(p.chips AS BIGINT) AS slice_chips, worker_id,
         date_diff('second', min(ts), max(ts)) AS active_s
  FROM parsed WHERE p.cap <> '' GROUP BY 1, 2, 3, 4, 5, 6),
chip AS (
  SELECT user, capacity, generation,
         active_s * slice_chips / count(*) OVER (PARTITION BY slice_id) / 3600.0 AS chip_hours
  FROM host)
SELECT user,
       coalesce(sum(chip_hours) FILTER (WHERE capacity = 'preemptible'), 0) AS preemptible,
       coalesce(sum(chip_hours) FILTER (WHERE capacity = 'reserved'), 0) AS reserved
FROM chip GROUP BY user ORDER BY sum(chip_hours) DESC;
```

Real output (2026-W29, chip-hours; 132,659 preemptible + 155,573 reserved total):

```
 user           preemptible  reserved
 larry                    0   137695
 michaelryan          59008     5691
 eczech               49026        0
 benjaminfeuer         6125       80
```

The `chip` CTE also feeds the by-capacity/generation table (`GROUP BY capacity,
generation`): reserved v4 155,573; preemptible v5p 67,614, v6e 44,399, v5e 20,282.

### Preemption events by pool/zone (`iris.provisioning`)

```sql
SELECT accelerator_variant, zone, count(*) AS preemptions
FROM provisioning
WHERE outcome = 'preempted' AND ts >= {{from}} AND ts < {{to}}
GROUP BY 1, 2 ORDER BY preemptions DESC;
```

2026-W29: 2,034 events; v6e-4/us-east1-d 409, v5p-8/us-east5-a 303, ….

### MFU per job (`telltale`)

```sql
WITH per_step AS (
  SELECT job_id, run, ts, any_value(value) AS mfu
  FROM telltale
  WHERE name = 'levanter_throughput_mfu' AND ts >= {{from}} AND ts < {{to}}
  GROUP BY job_id, run, ts)
SELECT job_id, run, round(avg(mfu), 1) AS mean_mfu, count(*) AS steps
FROM per_step GROUP BY job_id, run ORDER BY steps DESC;
```

2026-W29: canary-tpu 19.1% over 139 steps; multislice smoke 4.5%.

### Coverage

Share of active worker-seconds on a parseable `marin-tpu-%` worker over all
workers, so the chip-hour totals disclose what they omit (GPU/CoreWeave, CPU).

## 2. `scripts/ops/compute_report.py` public surface

```python
@dataclass(frozen=True)
class UserChipHours:
    user: str
    preemptible: float
    reserved: float
    @property
    def total(self) -> float: ...

@dataclass(frozen=True)
class CapacityGenChipHours:
    capacity: str          # "preemptible" | "reserved"
    generation: str        # "v4" | "v5p" | "v5e" | "v6e" | ...
    chip_hours: float

@dataclass(frozen=True)
class JobMfu: job_id: str; run: str; mean_mfu: float; steps: int
@dataclass(frozen=True)
class PoolPreemptions: accelerator_variant: str; zone: str; preemptions: int

@dataclass(frozen=True)
class WeeklyReport:
    iso_week: str
    window: TimeWindow
    chip_hours: list[UserChipHours]
    by_capacity_gen: list[CapacityGenChipHours]
    chip_hour_coverage: float
    mfu: list[JobMfu]
    preemptions: list[PoolPreemptions]

# Compute — pure, over DuckDB views telltale / provisioning / task:
def chip_hours_by_user(con, window) -> list[UserChipHours]: ...
def chip_hours_by_capacity_gen(con, window) -> list[CapacityGenChipHours]: ...
def chip_hour_coverage(con, window) -> float: ...
def mfu_per_job(con, window) -> list[JobMfu]: ...
def preemptions_by_pool(con, window) -> list[PoolPreemptions]: ...
def build_report(con, iso_week: str) -> WeeklyReport: ...

# Windowing / render / publish:
def iso_week_window(iso_week: str) -> TimeWindow: ...      # [Mon 00:00, next Mon) UTC
def previous_iso_week(today: date) -> str: ...
def render_markdown(report) -> str: ...
def compose_discord_summary(report, gist_url: str) -> str: ...

# CLI: --config, --iso-week, --channel, --dry-run/--no-dry-run
def main(config, iso_week, channel, dry_run) -> None: ...
```

`--dry-run` (default) prints the markdown and posts nothing; `--no-dry-run`
creates a secret gist and posts the summary to `#internal-discuss` via
`scripts/ops/discord.py`.

## 3. Preemption-waste follow-up (terminal cause)

`iris.task` has no per-attempt terminal cause, so it cannot say which chip-hours
were _lost_ to preemption. Two closes, to pick with the team:

1. **Controller SELECT.** In the weekly job, `ExecuteRawQuery`
   (`controller.proto:898`, SELECT-only) over `task_attempts` for the week's
   terminal attempts, `state ∈ {PREEMPTED(10), WORKER_FAILED(7)}`, joined to the
   chip-hours by `(task_id, attempt_id)`. Within the 7-day `job_retention`
   window. No controller change.
2. **Terminal-cause emit.** A durable finelog row `{attempt_uid, terminal_cause,
   ts}` written by the controller at terminalization, joined to `iris.task` for
   placement. Survives retention; touches the controller.

Either yields wasted preemptible chip-hours and (via ordered attempts of a task)
recovery idle.

## 4. Dashboard panels (`infra/grafana/dashboards/compute.json`, stretch)

| Panel | Namespace | Query |
|---|---|---|
| Chip-hours by user × capacity (stacked bar) | `iris.task` | §1 chip-hours |
| Chip-hours by capacity × generation (bar) | `iris.task` | §1 by-cap/gen |
| Preemptions by pool (bar) | `iris.provisioning` | §1 preemptions |
| MFU per job (table) | `telltale` | §1 MFU |

## 5. Cron + issue tag

- `.github/workflows/ops-compute-report.yaml`, `on.schedule: '0 15 * * 1'`,
  `workflow_dispatch` (`iso_week`, `dry_run`), GCP auth for GCS read +
  `DISCORD_WEBHOOK_INTERNAL_DISCUSS` + `GIST_PAT`, modeled on
  `ops-egress-report.yaml`.
- `MARIN_ISSUE_ENV = "MARIN_ISSUE"` stamped at `step_runner._submit_iris_job`
  (stretch); the report reads it via the option-1 controller join.

## File summary

| Path | Status | What |
|---|---|---|
| `scripts/ops/compute_report.py` | done | chip-hours / MFU / preemption rollup + gist + Discord |
| `tests/ops/test_compute_report.py` | done | unit tests over synthetic DuckDB tables |
| `.github/workflows/ops-compute-report.yaml` | done | weekly cron |
| terminal-cause join (§3 option 1) | follow-up | wasted preemptible chip-hours |
| `infra/grafana/dashboards/compute.json` | follow-up | dashboard |
| `MARIN_ISSUE` stamp | follow-up | per-issue attribution |

## Out of scope

- A controller emit for placement — not needed; placement is in the durable
  `iris.task` worker id.
- MFU/utilization for non-Levanter jobs and TPU hardware duty cycle; none exists,
  coverage is reported.
- GPU/CoreWeave chip-hours until their worker-id format is parsed.
