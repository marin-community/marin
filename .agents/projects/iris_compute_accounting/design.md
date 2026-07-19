# Iris compute accounting

_Why are we doing this? What's the benefit?_

"Where did the compute go this week?" gets answered anecdotally today. We want a
weekly rollup of TPU chip-hours per user, split by cluster/region, accelerator,
and preemptible-vs-reserved, with a preemption-overhead figure and MFU per job,
dropped into the weekly team summary (`#internal-discuss`). One shared, citable
picture for the team.

## Background

The material is spread across the controller DB and several finelog namespaces,
and the pieces that would give a clean weekly join expire the fastest. MFU per
job already lands in the durable finelog `telltale` namespace and is queryable
today. Per-user active time and preemption events are also queryable from
durable finelog. What is missing durably: chip counts, per-attempt placement
(zone/preemptible), and terminal cause. Those live only in the controller DB
behind a 7-day prune, and placement is destroyed even sooner (Challenges). Full
survey with file:line refs in [`research.md`](./research.md); every query below
is verified against the live marin cluster and written out in
[`spec.md`](./spec.md).

## What we can build today, and what needs new capture

Three of the report's numbers run today on existing durable finelog (verified;
queries and real output in `spec.md`):

- **MFU per job** — `telltale.levanter_throughput_mfu`. The 2026-07-18
  canary-tpu run averaged 19.1% MFU over 139 steps.
- **Preemption events by pool/zone** — `iris.provisioning`. Over the last 7
  days v6e-4/us-east1-d saw 471 preemptions, v5p-8/us-east5-a 311.
- **Active host-hours by user** — `iris.task`, whose `task_id` path begins with
  the user (`/eczech/...`). Last 24h: eczech 21.6h, larry 12.4h, michaelryan 4.7h.

That is a useful weekly report immediately. Two numbers it cannot produce:

- **Chip-hours.** `iris.task` carries no chip count, so host-hours cannot be
  converted to chip-hours without the accelerator variant + device count, which
  live only in the job resource spec in the controller DB.
- **Preemptible split, region, and terminal cause per attempt.** These sit on
  the `workers`/attempt rows in the controller DB and expire (Challenges), so
  from finelog alone the report cannot attribute preemption waste to a user or
  split chip-hours by capacity type.

`iris.accounting` (below) closes both gaps.

## Challenges

Placement is destroyed for the attempts we most want to measure. Zone, region,
and preemptible live on the `workers` row (`worker_attributes`,
[schema.py:484](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/schema.py#L484)),
reached through `task_attempts.worker_id`, a `SET NULL` FK
([schema.py:428](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/schema.py#L428)).
`remove_worker` nulls that FK and deletes the worker in the same transaction
that commits a preempted attempt's terminal state
([writes.py:734](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/writes.py#L734)),
and surviving workers are pruned at `worker_retention` = 24h
([controller.py:242](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/controller.py#L242)).
A job that reads the DB after the fact sees NULL placement for the entire
preemption cohort at t=0. The resolved facts have to be captured when the
attempt terminates, while the worker row is still present. That capture is
`iris.accounting`.

## Costs / Risks

- A new persistent finelog namespace is a retention and schema commitment, and
  the emit is new controller work on the terminalization path.
- A best-effort finelog append at terminalization is not atomic with the DB
  commit. A lost append loses that attempt's placement with no way to
  reconstruct it. The delivery guarantee is an Open Question.
- Some compute lands under `local_admin` (in-cluster/loopback submissions) or a
  shared OS username. Phase 1 measures that fraction before we build on it.
- MFU coverage is Levanter-training-only; vLLM and Zephyr emit none, and TPUs
  have no hardware duty-cycle signal.

## Design

**Phase 1 — the report, on existing finelog.** `scripts/ops/compute_report.py`
loads the durable finelog namespaces into DuckDB and emits, per ISO week: MFU
per job, preemption events by pool/zone, and host-hours by user. It publishes a
markdown gist and a compact table to `#internal-discuss`, modeled on
[`scripts/ops/egress_report.py`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/scripts/ops/egress_report.py),
cron'd weekly. No controller change. This ships the transparency win at once and
measures attribution coverage (host-hours under a real username vs.
`local_admin`/shared users) before we commit to a schema.

**Phase 2 — `iris.accounting`, for chip-hours and the preemptible split.** A
durable finelog namespace, one immutable row per terminated attempt, emitted by
the controller at terminalization, before `remove_worker`, where jobs/tasks/
workers are all present. The row records attribution (`user_id`,
`submitting_user`, `job_name`, `job_id`, `issue`), placement (`zone`, `region`,
`capacity_type` ∈ {preemptible, reserved, unknown}, `accelerator_variant`,
`device_count`), timing (created/started/finished, all nullable), and outcome
(`terminal_cause` = raw `TaskState`, plus the enclosing job's terminal state so
user-cancels are separable). The report upgrades host-hours to chip-hours, adds
the capacity-type and region split, ties preemptions to users, and computes
preemption overhead. Schema and emit contract in `spec.md`.

**Metrics (Phase 2), defined to avoid over-counting.** Chip-seconds per attempt
= `active_seconds × device_count`, clipped to the ISO week by the overlap of
`[started, finished)` with the interval. Reported:

- **Allocated chip-hours** by user × cluster × `capacity_type` × accelerator.
- **Chip-hours on preempted attempts** (`capacity_type = preemptible ∧
  terminal_cause ∈ {PREEMPTED, WORKER_FAILED}`), as an upper bound on preemption
  waste — Levanter checkpoints and resumes, so most of a preempted attempt's
  runtime is retained. User-cancel `KILLED` rows are excluded.
- **Recovery idle** = Σ over a task's ordered attempts of
  `next.started − prev.finished` after a preempt, a self-join on
  `iris.accounting`. `iris.provisioning` supplies only the cluster-level
  preemption-event count.
- **MFU** by user / cluster / top-N job from `telltale`, collapsing worker
  replicas per step (`spec.md` Q1), with a coverage % = share of chip-hours
  whose job has ≥1 MFU sample.

**Issue attribution.** Stamp `MARIN_ISSUE=<N>` into `env_vars` at
[`StepRunner._submit_iris_job`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/marin/src/marin/execution/step_runner.py#L478)
— persisted, child-inherited, copied onto the row. The source of `<N>` is an
Open Question. Absent → user-level only.

**Dashboard.** `infra/grafana/dashboards/compute.json` over the same namespaces
through the existing finelog SQL bridge — the Phase-1 panels run at once, the
Phase-2 panels turn on with `iris.accounting`. Panel SQL in `spec.md`.

## Testing

Unit-test the Phase-2 rollup on synthetic `iris.accounting` rows with a hand-built
preemption sequence (attempt 0 PREEMPTED after 100s on 8 chips, attempt 1
FINISHED) and assert allocated vs. exposed-to-preemption vs. recovery-idle, the
capacity_type/accelerator split, and week-boundary clipping across an ISO-week
boundary. Fail fast on an unmapped `TaskState`. Test the emit fires for
WORKER_FAILED/PREEMPTED with placement populated. Reconciliation invariant: for
any window inside DB retention, `iris.accounting` totals must match a direct
controller-DB aggregation by terminal state and accelerator.

## Open Questions

- Delivery guarantee for the emit: best-effort append + reconciliation backstop,
  or a transactional outbox (row written in the terminalization transaction,
  flushed to finelog by a background loop)?
- `capacity_type` source: read the worker's `scale_group`/`worker_attributes` at
  emit — confirm it is populated on every backend, and pick the `unknown`
  fallback for backends with no zone.
- The upper-bound waste number: ship it labeled with recovery-idle alongside, or
  add a Levanter checkpoint-cadence signal to bound actual loss?
- Where does the `MARIN_ISSUE` number come from — an `ExecutorStep` field, a CI
  variable, or human-set env?
- If Phase 1 shows most host-hours are `local_admin`/shared usernames, do we add
  a `MARIN_USER` stamp before the per-user table is worth publishing?
