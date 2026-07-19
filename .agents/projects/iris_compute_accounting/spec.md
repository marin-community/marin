# Spec: iris compute accounting

Contracts the design commits to. No new proto — the durable store is a finelog
dataclass namespace (finelog derives the schema from the row type, as
[`IrisProvisioning`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/autoscaler/provisioning.py#L39)
does). Population is a **controller-side emit at terminalization**, not an
external pull (peer review established that placement is destroyed by the time
any pull runs — see `design.md` → Challenges).

## 1. Durable namespace `iris.accounting`

New file `lib/iris/src/iris/cluster/controller/accounting.py`.

```python
ACCOUNTING_NAMESPACE = "iris.accounting"


class CapacityType(StrEnum):
    PREEMPTIBLE = "preemptible"
    RESERVED = "reserved"
    UNKNOWN = "unknown"       # backend with no scale_group/zone signal


class TerminalCause(StrEnum):
    """Raw terminal disposition, one-to-one with the terminal TaskState members.

    Stored raw (not pre-bucketed into "preemption") so the rollup derives named
    metrics in SQL and new TaskState members fail fast instead of silently
    bucketing wrong. Mapping is total over terminal TaskState values:
        SUCCEEDED      -> FINISHED
        FAILED         -> FAILED
        PREEMPTED      -> PREEMPTED
        KILLED         -> KILLED
        WORKER_FAILED  -> WORKER_FAILED
        UNSCHEDULABLE  -> UNSCHEDULABLE
        COSCHED_FAILED -> COSCHED_FAILED
    Any other/unknown terminal state raises (house-style fail-fast), it is not
    coerced to a default bucket.
    """
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
    attempt = active_seconds * device_count where device_count is *chips*
    (TpuDevice.count per worker, summed across the slice — not cores, not hosts).
    Nullable timestamps are stored NULL (never 0); durations are NULL when their
    endpoints are absent, so a never-started attempt has active_seconds = NULL,
    not an epoch-scale value.
    """

    key_column: ClassVar[str] = "submitting_user"  # per-user rollups dominate

    ts: datetime                 # terminalization time
    cluster: str
    user_id: str                 # JobName.user (OS username; noisy)
    submitting_user: str         # authenticated principal; "local_admin" if unauth
    job_name: str                # root job path "/user/job"
    job_id: str                  # == telltale.job_id (the job root) for the MFU join
    task_id: str
    attempt_id: int
    attempt_uid: str             # stable per-attempt id; read-side dedup key
    accelerator_variant: str     # normalized via device_counts_from_json, e.g. "v5litepod-16"
    device_count: int            # chips in this attempt's slice
    zone: str | None             # e.g. "us-central2-b"; None for non-GCE
    region: str | None           # derived from zone; None if zone unknown
    capacity_type: str           # CapacityType value, from the worker's scale_group
    created_at_ms: int | None
    started_at_ms: int | None
    finished_at_ms: int | None
    active_seconds: float | None   # (finished-started)/1000 when both present, else None
    terminal_cause: str            # TerminalCause value
    job_terminal_state: str        # enclosing job's terminal state (distinguishes user-cancel)
    exit_code: int | None
    issue: int | None              # MARIN_ISSUE; None = untagged
```

**Registration.** In
[`log_stack.py`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/log_stack.py#L43),
add an `accounting_table: Table[IrisAccounting]` field to the `LogStack`
dataclass and `client.get_table(ACCOUNTING_NAMESPACE, IrisAccounting)` to its
construction. No `storage_policy`; the server default retention must be
explicitly confirmed to be durable (no `max_age_seconds`) and that assumption
pinned in the module docstring, not merely inherited.

## 2. Emit at terminalization

New `lib/iris/src/iris/cluster/controller/accounting.py` (same module):

```python
def build_accounting_row(
    *,
    attempt: TaskAttemptRow,
    task: TaskRow,
    job: JobRow,
    worker: WorkerRow | None,      # still live at emit; None only if already gone
    worker_attrs: WorkerAttrs | None,
    cluster: str,
    now: datetime,
) -> IrisAccounting:
    """Resolve the full accounting row from rows that are all live inside the
    terminalization transaction. capacity_type/zone/region come from `worker`/
    `worker_attrs` (present here, gone after remove_worker); variant + chip count
    from job.res_device_json via device_counts_from_json; issue from
    job.environment_json[MARIN_ISSUE]. Pure — no I/O — so it is unit-testable."""
```

The call site is the reconcile commit path that terminalizes an attempt
([ops/worker.py](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/ops/worker.py)
for worker failure/preempt;
[reconcile/task.py](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/reconcile/task.py)
for clean/kill), **ordered before `remove_worker`**. Delivery guarantee (Open
Question in `design.md`): v1 = best-effort append after the DB commit +
`reconcile_accounting()` backstop that re-derives missing rows for
non-preempted attempts still inside DB retention; hardening = a transactional
outbox row written in the same DB transaction and flushed to finelog by a
background loop.

## 3. Phase 0 — attribution + definition validation

New `scripts/ops/compute_accounting_probe.py` (throwaway-grade, kept as a
regression check): a single windowed `ExecuteRawQuery`
([controller.proto:898](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/rpc/controller.proto#L898))
over the live ≤7-day window reporting (a) chip-hours by `submitting_user` vs. the
`local_admin`/OS-username share, (b) chip-hours by accelerator with week-boundary
clipping. Query must be a plain windowed `SELECT` (the guard at
[service.py:2868](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/service.py#L2868)
checks the leading token, so `WITH ... SELECT` CTEs are rejected) paginated by
`finished_at_ms` ranges (the RPC returns the full result set unbounded).

## 4. Weekly rollup + MWS drop

New `scripts/ops/compute_report.py`, modeled on `scripts/ops/egress_report.py`.

```python
@dataclass
class OverheadBlock:
    allocated_chip_hours: float
    preemptible_chip_hours: float
    exposed_to_preemption_chip_hours: float   # UPPER BOUND; see design metrics
    recovery_idle_chip_hours: float           # Σ next.started-prev.finished after a preempt
    preemption_events: int                    # slice-deduped where possible; else note per-attempt


@dataclass
class WeeklyRollup:
    iso_week: str                              # "2026-W29"
    by_user: list[UserChipHours]               # user × capacity_type × chip-hours
    by_cluster_accel: list[ClusterAccelChipHours]
    overhead: OverheadBlock
    top_jobs: list[JobChipHours]
    mfu_by_user: list[UserMfu]                 # mean MFU + coverage_pct
    by_issue: list[IssueChipHours]             # empty until MARIN_ISSUE lands


def build_weekly_rollup(con: DuckDBPyConnection, iso_week: str, *, top_n: int = 15) -> WeeklyRollup:
    """Aggregate iris.accounting (+ telltale MFU) for one ISO week. Rules:
    - dedup rows by attempt_uid (QUALIFY row_number() ... — belt-and-braces over write-side dedup);
    - clip active_seconds to the week: overlap of [started, finished) with the ISO-week interval (UTC);
    - roll up on submitting_user, falling back to user_id when submitting_user == LOCAL_ADMIN_SUBMITTER;
    - exclude terminal_cause=KILLED with job_terminal_state=cancelled from exposed_to_preemption;
    - recovery_idle: self-join per task_id ordered by attempt_id, gap after a PREEMPTED/WORKER_FAILED prev;
    - MFU: join telltale.levanter_throughput_mfu on job_id, filter process_index=0, mean per job,
      device-hour-weighted per user; coverage_pct = chip-hours whose job has >=1 MFU sample / total."""


def render_markdown(rollup: WeeklyRollup) -> str: ...
def compose_discord_summary(rollup: WeeklyRollup, gist_url: str) -> str: ...
def main(iso_week: str | None, clusters: list[str], channel: str, dry_run: bool) -> None: ...
```

`--dry-run` prints without gist/post. `iso_week` defaults to the previous
complete ISO week. `clusters` is a list — the report spans clusters for the
cluster/region split; one cluster's read failing degrades that cluster's rows,
not the whole publish. `LOCAL_ADMIN_SUBMITTER` and the `levanter_throughput_mfu`
metric name / `telltale` namespace are as in
[`finelog/telltale.py:21`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/finelog/src/finelog/telltale.py#L21).

## 5. Cron

New `.github/workflows/ops-compute-report.yaml`, `on.schedule: '0 15 * * 1'`
(after the storage report), `workflow_dispatch` with `dry_run` + `iso_week`
inputs, reusing the tunnel + `DISCORD_WEBHOOK_INTERNAL_DISCUSS` + `GIST_PAT`
setup from
[`ops-storage-report.yaml`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/.github/workflows/ops-storage-report.yaml).
The emit (§2) is inline in the controller, so no separate snapshot cron is
needed; the DB reconciliation backstop (§2) runs as a short daily
`workflow_dispatch`/cron guard against lost appends.

## 6. Issue tag

Constant `MARIN_ISSUE_ENV = "MARIN_ISSUE"` in `step_runner.py`; stamped into
`env_vars` at `_submit_iris_job`
([step_runner.py:478](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/marin/src/marin/execution/step_runner.py#L478))
when the launching context supplies one (source is an Open Question), read onto
`IrisAccounting.issue`. Absent → `None`.

## 7. Dashboard (stretch)

New `infra/grafana/dashboards/compute.json` over `iris.accounting` + `telltale`
via the existing finelog SQL bridge. No new datasource.

## File summary

| Path | What |
|---|---|
| `lib/iris/src/iris/cluster/controller/accounting.py` | `IrisAccounting` row, `CapacityType`, `TerminalCause`, `build_accounting_row`, `reconcile_accounting` |
| `lib/iris/src/iris/cluster/controller/log_stack.py` | `accounting_table` field + registration (edit) |
| `lib/iris/src/iris/cluster/controller/reconcile/*`, `ops/worker.py` | emit call at terminalization, before `remove_worker` (edit) |
| `scripts/ops/compute_accounting_probe.py` | Phase-0 attribution/definition probe |
| `scripts/ops/compute_report.py` | weekly DuckDB rollup + gist + Discord |
| `.github/workflows/ops-compute-report.yaml` | weekly cron + daily reconcile guard |
| `lib/marin/src/marin/execution/step_runner.py` | `MARIN_ISSUE` env stamp (edit) |
| `infra/grafana/dashboards/compute.json` | dashboard (stretch) |

## Out of scope

- Pull-only population with no controller change — ruled out: placement is gone
  by the time a pull runs.
- True checkpoint-aware waste (requires a Levanter checkpoint-cadence event) —
  v1 reports the labeled upper bound + recovery-idle instead.
- First-class `labels` map on `LaunchJobRequest` — the env-var tag is the v1
  issue link; a proto label field is a possible follow-up.
- Backfilling MFU/utilization for non-Levanter jobs (vLLM, Zephyr), and any TPU
  hardware duty-cycle collection — none exists; coverage is reported, not
  manufactured.
- Changing `ListUsers` to group on `submitting_user`.
