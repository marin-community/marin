# Compute-report enhancements

The v1 report (`scripts/ops/compute_report.py`) answered "how much did each user
consume." A reviewer reading it could not answer the questions the issue asks:
how much did preemption *cost*, what was the compute *for*, and *where* did it
run. This is the plan for closing that gap, ordered by value-per-effort. Each
item names the data it needs and whether that data is already loaded.

## Shipped in v1 (quick wins, no new data source)

These read the `iris.task` chip-hours base CTE that was already loaded, so they
cost nothing beyond the query:

1. **Waste — chip-hours redone after preemption/failure.** For each task, every
   attempt a later attempt superseded is counted as redone work, split by
   capacity. This is the "what are we paying for preemptible" number the issue
   calls the surprising one. It is an honest **upper bound**: iris.task carries
   no terminal cause, so it over-counts intentional restarts and under-counts a
   final attempt that itself never finished. Labeled as such.
2. **Top jobs by chip-hours** (`/user/jobname` grain) with a per-job distinct
   attempt count as a churn signal. Directly the issue's "top-N jobs."
3. **Chip-hours by region**, parsed from the worker_id zone. The issue's
   "by cluster/region" split.
4. **Headline block** — total chip-hours, waste %, preemption count — so the
   three numbers a reader wants are above the fold in both the gist and Discord.

## Tier 2 — richer finelog rollups (data already durable, new queries)

5. **Reserved-capacity idle.** `iris.worker` heartbeats carry
   `running_task_count`; worker-seconds at 0 on a reserved slice are paid-for
   idle. This is a *different* waste axis from preemption churn and may be larger
   — reserved capacity sitting unused costs the same whether a job is on it or
   not. Needs the `iris.worker` namespace added to the loader.
6. **Provisioning health.** From `iris.provisioning`: stockout rate, preemption
   rate per pool normalized to chip-weighted slice-days (not raw event counts,
   which favor small slices), and `provision_latency` p50/p90 — the re-place
   delay half of churn. Some columns already loaded; latency needs the field.
7. **Re-place delay from attempt gaps.** The wall-clock between an attempt's last
   iris.task sample and the next attempt's first sample of the same task is dead
   time attributable to preemption. Complements the provisioning latency with an
   observed, per-task number. Base CTE already has task/attempt/ts.
8. **Week-over-week deltas.** Load the prior ISO week too and show ± on the
   headline and per-user rows. Cheap; makes "better or worse than last week"
   answerable, which a single-week snapshot cannot.

## Tier 3 — needs a controller-DB snapshot (attribution the log lacks)

9. **Weekly controller snapshot** written to a durable table before the 7-day
   `job_retention` ages rows out. Gives three things finelog cannot:
   - **terminal cause per attempt** — turns the waste *upper bound* (item 1) into
     an attributed preempted-vs-failed-vs-restarted decomposition;
   - **`submitting_user`** — canonical attribution instead of the job-path second
     segment (handles service accounts, shared launchers);
   - **`MARIN_ISSUE` / launch label** — the issue-attribution stretch goal, if we
     land a launch-time `issue=#NNNN` convention.
   This is the "roll up incrementally into a durable table" answer to the issue's
   retention open-question.

## Tier 4 — the dashboard (the issue's "nice dashboard for users")

10. **Static HTML page with charts**, published as an artifact next to the gist.
    Six views fable and the issue motivate:
    - trailing-8-week stacked bars (preemptible vs reserved) — the trend;
    - top-15 jobs, horizontal bar;
    - waste decomposition stacked bar (see below);
    - pool survivability (time-to-preempt distribution per pool);
    - user × region heatmap;
    - MFU vs chip-hours scatter (efficiency vs spend).
    Presentation is a hybrid: ~10-line Discord summary, full tables in the gist,
    charts on the HTML page. rjpower's "preloads into a duckdb + dashboard" —
    the DuckDB is the report; this is the dashboard on top.
11. **TPU per-process utilization** folded in from telltale/prometheus so the MFU
    signal covers more than the two Levanter canary jobs it sees today (reserved
    v4, ~half the fleet, currently gets no efficiency signal).

## Waste as a decomposition, not one number

The issue's "productive vs wasted" open question resolves cleanly if we report
waste as a stacked decomposition rather than a single figure:

| component | source | tier |
| --- | --- | --- |
| redone superseded attempts | iris.task (upper bound) | shipped |
| preempted-attributed | controller snapshot terminal cause | 3 |
| re-place delay | attempt gaps + provision_latency | 2 |
| redone step / cold-compile | telltale step regressions | 4 |
| reserved idle | iris.worker running_task_count=0 | 2 |

Each is a distinct thing the team can act on; collapsing them hides which lever
matters. v1 ships the first row and labels it as the upper bound.

## Cut / deferred

- The 2-canary MFU table stays but is not a headline until item 11 broadens it.
- Raw provisioning event counts — superseded by the chip-weighted rate (item 6).
- Per-issue attribution waits on item 9's launch-label convention.
