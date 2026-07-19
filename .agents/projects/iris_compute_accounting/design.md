# Iris compute accounting

_Why are we doing this? What's the benefit?_

"Where did the compute go this week?" gets answered anecdotally today. We want a
weekly rollup of TPU chip-hours per user, split by cluster/region, accelerator,
and preemptible-vs-reserved, with a preemption-overhead figure and MFU per job,
dropped into the weekly team summary (`#internal-discuss`). One shared, citable
picture for the team.

## Background

The material is spread across the controller DB and several finelog namespaces.
The key finding came from querying the live data: everything the issue asks for
except one field is already in durable finelog:

- **`iris.task.worker_id` encodes placement.** A worker id is
  `marin-tpu-v5p-preemptible-32-us-east5-a-...-worker-1`: capacity type,
  generation, slice chip count, zone, and host index are all in the string, and
  `iris.task` carries it durably. So capacity/zone/accelerator per attempt
  survive in finelog even for preempted attempts.
- **`iris.task.task_id`** is `/user/job/...`, so the user is the second path
  segment.
- **MFU per job** is in the `telltale` namespace; **preemption events** are in
  `iris.provisioning`.

Full survey with file:line refs in [`research.md`](./research.md); every query
is verified against the live marin cluster and written out in
[`spec.md`](./spec.md).

## What this changes vs. the first draft

The first draft proposed a new `iris.accounting` namespace populated by a
controller-side emit at terminalization, because the controller DB's `workers`
row (which holds zone/preemptible) is deleted when a preempted attempt
terminates. That reasoning was about the controller DB. It missed that the same
placement is durably recorded in the finelog `iris.task` namespace via the
worker id string. Chip-hours, the preemptible split, and the region split need
no controller change. What is _not_ in `iris.task` is the per-attempt terminal
cause — whether an attempt finished, was preempted, or was killed — so
distinguishing chip-hours _lost_ to preemption from chip-hours _consumed_ on
preemptible capacity is the one remaining gap.

## Design

**A report over durable finelog, no controller change.**
`scripts/ops/compute_report.py` loads the finelog namespaces into DuckDB and
emits, per ISO week:

- **Chip-hours by user, preemptible vs reserved.** Parse capacity/generation/
  slice-chips from each `iris.task` worker id; per-host chips = a slice's chips
  divided by its live hosts (SPMD, so every host of a slice emits rows);
  chip-hours = host active-seconds × chips-per-host. 2026-W29 totalled 132,659
  preemptible + 155,573 reserved chip-hours: larry 137,695 reserved (v4),
  michaelryan 64,699, eczech 49,026 preemptible.
- **Chip-hours by capacity type and generation** — reserved v4 155,573;
  preemptible v5p 67,614, v6e 44,399, v5e 20,282.
- **Preemption events by pool/zone** from `iris.provisioning` — 2,036 events in
  the week, led by v6e-4/us-east1-d.
- **MFU per job** from `telltale.levanter_throughput_mfu` (collapse worker
  replicas per step; `process_index` is null).

It publishes a markdown gist and a compact summary to `#internal-discuss`,
modeled on
[`scripts/ops/egress_report.py`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/scripts/ops/egress_report.py),
cron'd weekly. The report states a coverage figure — the share of active
worker-time on parseable TPU workers — so the chip-hour totals stay honest
(GPU/CoreWeave and CPU workers are not counted yet).

**The remaining gap: preemption waste.** Preemptible chip-hours _consumed_ is a
headline number the report gives today. Attributing chip-hours to the specific
attempts that _ended_ in a preemption (waste, and recovery idle between attempts)
needs a per-attempt terminal cause, which `iris.task` lacks. Two ways to close
it, to decide with the team:

1. Read the controller DB for the week's terminal attempts (via the SELECT-only
   `ExecuteRawQuery`), join `state ∈ {PREEMPTED, WORKER_FAILED}` attempts to
   their chip-hours. Works within the 7-day `job_retention` window
   ([controller.py:239](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/controller.py#L239));
   the weekly job runs well inside it.
2. Emit a tiny per-attempt terminal-cause row to finelog at terminalization (just
   `attempt_uid` + `terminal_cause`), joined to `iris.task` for placement. Durable
   past retention, but touches the controller.

Recommendation: option 1 for v1 — no controller change, and the report already
runs offline against finelog + a controller SELECT.

**Attribution.** `user_id` (the `iris.task` path user) is a real username in the
data (eczech, larry, ...), so it is usable directly. `submitting_user` (the IAP
email) is cleaner but lives in the controller DB; fold it in with option 1 if
wanted.

**Issue attribution (stretch).** Stamp `MARIN_ISSUE=<N>` into `env_vars` at
[`StepRunner._submit_iris_job`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/marin/src/marin/execution/step_runner.py#L478);
the report reads it from the controller DB join. The source of `<N>` is an Open
Question.

**Dashboard (stretch).** `infra/grafana/dashboards/compute.json` — the same
queries through the existing finelog SQL bridge.

## Costs / Risks

- Per-host chips = slice-chips / live-hosts assumes every host of a slice emits
  `iris.task` rows in the window (true for SPMD training); a slice with idle
  hosts would mis-divide. Slice-chips follows the marin variant naming
  (`v5p-32` → 32), so the unit is whatever that convention counts.
- Chip-hour coverage is TPU-only; GPU/CoreWeave and CPU workers are excluded
  until their worker ids are parsed too. The report states coverage explicitly.
- `iris.task` is high volume (~1.5 GB/week); the weekly job downloads a week of
  segments. Fine in GitHub Actions, but not instant.
- MFU coverage is Levanter-training-only; TPUs have no hardware duty-cycle signal.

## Testing

Unit-test the rollups by registering synthetic `iris.task` rows: a single-host
preemptible slice and a two-host reserved slice assert the preemptible/reserved
split and the per-host chip division (slice chips / hosts); a non-TPU worker
asserts the coverage figure; MFU asserts replica-collapse per step. All pass
without finelog access. Verified end-to-end against live finelog for 2026-W29.

## Open Questions

- **Preemption waste**: option 1 (controller SELECT within retention) or option 2
  (tiny terminal-cause emit)? Recommendation: option 1 for v1.
- **Per-host chip count**: is slice-chips / live-hosts accurate enough, or should
  we pin a chips-per-host table per generation (v4/v5p/v5e/v6e)?
- **Coverage**: is TPU-only acceptable for v1, or do we parse GPU/CoreWeave
  worker ids for chip-hours too?
- **`MARIN_ISSUE` provenance**: where does the issue number come from — an
  `ExecutorStep` field, a CI variable, or human-set env?
- **`submitting_user` vs `user_id`**: is the `iris.task` path username enough, or
  do we join the controller DB for the IAP email?
