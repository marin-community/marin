# Iris compute accounting

_Why are we doing this? What's the benefit?_

"Where did the compute go this week?" is answered anecdotally today. We want a
weekly, per-user (and ideally per-GitHub-issue) rollup of TPU chip-hours — split
by cluster/region, accelerator type, and preemptible-vs-reserved, with an
honest "chip-hours exposed to preemption" number — dropped into the weekly team
summary the team already reads (`#internal-discuss`). One shared, citable
picture instead of guesses.

## Background

The raw material exists but is scattered and — critically — *ephemeral in the
wrong places*. The authoritative per-attempt lifecycle (start/stop/state) lives
in the controller DB `task_attempts` table
([schema.py:423](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/schema.py#L423)),
not finelog, and terminal jobs are pruned after 7 days
([controller.py:239](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/controller.py#L239)).
MFU-per-job is already forwarded durably to the finelog `telltale` namespace and
Grafana reads it, but only for Levanter training. Attribution is first-class but
subtle: the clean key is `submitting_user` (the IAP email), not the OS-username
`user_id` that `ListUsers` returns. Full survey with file:line refs in
[`research.md`](./research.md). This revision incorporates a three-way peer
review (codex + fable + architect) that found two load-bearing defects in the
first draft, addressed below.

## What stats we're missing

| Want | Have today | Missing |
|---|---|---|
| chip-hours per user × cluster × accelerator | per-attempt start/stop in controller DB (7d) | durable store; clean per-user key not wired |
| preemptible-vs-reserved + region split | on the `workers` row (zone, scale_group) | **the workers row is destroyed at preemption** (see Challenges) — must capture at terminalization |
| honest preemption waste | full preempted-attempt runtime is derivable | checkpoints make full runtime a large over-count; true waste needs checkpoint events we don't emit |
| MFU per job / per user | `telltale.levanter_throughput_mfu`, durable | a stable `job_id` join key on the accounting row; process dedup; Levanter-only coverage |
| job → GitHub issue | nothing programmatic | a launch-time link *and* a source for the issue number |

## Challenges

**The placement data is destroyed exactly for the rows we care about.** Zone,
region, and preemptible live only on the `workers` row (via `worker_attributes`,
[schema.py:484](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/schema.py#L484)),
reachable through `task_attempts.worker_id` — a `SET NULL` FK
([schema.py:428](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/schema.py#L428)).
`remove_worker` nulls that FK and deletes the worker in the *same transaction*
that commits the attempt's terminal state
([writes.py:734](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/writes.py#L734)),
and even surviving workers are pruned at `worker_retention` = **24h**
([controller.py:242](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/iris/src/iris/cluster/controller/controller.py#L242)).
So any job that *reads the DB after the fact* — the obvious "weekly query" — gets
`worker_id = NULL` for every preempted/failed attempt, i.e. the entire
preemption cohort, at t=0. This is not a polling-window problem; running more
often cannot fix it. **The resolved placement facts must be captured
synchronously at terminalization, before the worker row is torn down.** That is
the load-bearing new primitive — and it is precisely the "iris event we could
roll up over": one durable per-attempt record, stamped by the controller when
the attempt terminates.

The second challenge is honesty of the headline number (see Design → metrics).

## Costs / Risks

- A new persistent finelog namespace is a retention + schema commitment; the
  emit is a new controller responsibility on the terminalization path.
- A best-effort finelog append at terminalization is not atomic with the DB
  commit; a lost append loses that attempt's placement irrecoverably (the DB
  reconciliation backstop covers non-preempted attempts only — preempted ones
  have no worker row to re-read). Delivery guarantee is an Open Question.
- Attribution collapses to `local_admin` for jobs submitted in-cluster / over
  loopback; the *size* of that bucket is unknown and could dominate. We measure
  it before building (Phase 0).
- MFU coverage is Levanter-training-only; "utilization" is not universal, and
  TPUs have no independent hardware duty-cycle cross-check.

## Design

**A durable `iris.accounting` finelog namespace, one immutable row per
terminated attempt, emitted by the controller at terminalization** — inside the
reconcile commit, before `remove_worker` runs, where jobs/tasks/workers are all
still live. The row records attribution (`user_id`, `submitting_user`,
`job_name`, `job_id` = the telltale job root, `issue`), placement (`zone`,
`region`, `capacity_type` ∈ {preemptible, reserved, unknown}, `accelerator_variant`,
`device_count`), timing (`created_at_ms`, `started_at_ms`, `finished_at_ms`, all
nullable — no `0` sentinels), and outcome (`terminal_cause` = the raw
`TaskState`, plus the enclosing job's terminal state so user-cancels are
distinguishable). Full schema, nullable/mapping rules, and delivery-guarantee
options in [`spec.md`](./spec.md). Because it is durable, the weekly window is
never bounded by controller retention.

**Metrics, defined honestly.** The rollup reads `iris.accounting` into DuckDB and
reports, per ISO week (chip-seconds clipped to the week via overlap of
`[started, finished)` with the interval, so long attempts don't dump into their
finishing week):

- **Allocated chip-hours** by user × cluster × `capacity_type` × accelerator —
  the "where did compute go" number.
- **Chip-hours exposed to preemption** = Σ over rows with
  `capacity_type = preemptible ∧ terminal_cause ∈ {PREEMPTED, WORKER_FAILED}` of
  `active_seconds × device_count`, labeled explicitly as an **upper bound** on
  waste: Levanter checkpoints and resumes, so most of a preempted attempt's
  runtime is usually retained. User-cancel `KILLED` rows (enclosing job cancelled)
  are excluded — they are not preemption churn.
- **Recovery idle** = Σ over each task's ordered attempts of
  `next.started − prev.finished` where `prev` was preempted — the honestly
  computable "chip-time lost re-placing after a preempt." This is a self-join on
  `iris.accounting`, *not* `iris.provisioning` (which has no user/job key and can
  only give a cluster-level provisioning-latency cross-check).
- **MFU** by user / cluster / top-N job, joining `telltale.levanter_throughput_mfu`
  on `job_id`, deduped to `process_index = 0`, device-hour-weighted, with an
  explicit **coverage %** = share of the week's chip-hours whose job has ≥1 MFU
  sample. Partial coverage is reported, never hidden.

**Publish** modeled on
[`scripts/ops/egress_report.py`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/scripts/ops/egress_report.py):
markdown gist + a compact table to `#internal-discuss` via `scripts/ops/discord.py`,
cron'd weekly from a new workflow, reusing the storage-report tunnel machinery.

**Issue attribution.** Stamp `MARIN_ISSUE=<N>` into `env_vars` at
[`StepRunner._submit_iris_job`](https://github.com/marin-community/marin/blob/1f2b88682bf47c4ed848d917208fa32107084c5d/lib/marin/src/marin/execution/step_runner.py#L478)
— persisted, child-inherited, copied onto the row. The *source* of `<N>` (a new
`ExecutorStep` field vs. a CI/env variable) is an Open Question; degrades to
user-level only when unset.

**Phasing.**
0. **Validate the premise, cheaply.** One `ExecuteRawQuery` over the live 7-day
   window: what fraction of chip-hours carry a real `submitting_user` vs.
   `local_admin`/OS-username? Plus an ad-hoc chip-hours-by-user/accelerator
   report to settle the user/accelerator/week-clipping definitions. This needs no
   new infra and no controller change, and it cannot produce
   placement-for-preempted or honest waste — those gate on the emit. If
   attribution coverage is poor, add a `MARIN_USER` stamp on the same channel
   before freezing anything.
1. `iris.accounting` schema + controller terminalization emit + a DB
   reconciliation backstop (re-derive missing non-preempted rows from retained
   attempts).
2. `scripts/ops/compute_report.py` weekly rollup + cron + MWS drop.
3. `MARIN_ISSUE` tagging + `infra/grafana/dashboards/compute.json`.

## Testing

Unit-test the rollup on synthetic rows with a hand-built preemption sequence
(attempt 0 PREEMPTED after 100s on 8 chips, attempt 1 FINISHED) → assert exact
allocated vs. exposed-to-preemption vs. recovery-idle, the user/accelerator/
capacity_type split, and week-boundary clipping for an attempt spanning two ISO
weeks. Fail-fast on an unmapped `TaskState`. Test the emit fires for
WORKER_FAILED/PREEMPTED with placement populated (worker still live at emit).
**Reconciliation invariant** (replacing "eyeball the hero runs"): for any window
still fully inside DB retention, `iris.accounting` totals must match a direct
controller-DB aggregation broken down by terminal state and accelerator — a
regression test on real data, not a vibe check.

## Open Questions

- **Delivery guarantee for the emit.** Best-effort append + DB reconciliation
  backstop (simple; loses placement for a preempted attempt if the append is
  lost), or a transactional outbox (row written to the DB in the terminalization
  transaction, a flusher ships it to finelog)? The outbox is the only true
  exactly-once, but it is more machinery on the controller.
- **`capacity_type` source.** Read the worker's `scale_group`/`worker_attributes`
  at emit (authoritative, present only at terminalization) — confirm it is
  populated for every backend (GCE/TPU vs. k8s/GPU), and decide the `unknown`
  fallback for backends without a zone.
- **"Exposed to preemption" vs. true waste.** Ship the upper-bound number with
  recovery-idle alongside, or invest in a checkpoint-cadence signal (a Levanter
  event) to bound actual loss? Recommendation: ship the upper bound, labeled, in
  v1.
- **Issue-number provenance.** Where does `<N>` come from — a new `ExecutorStep`
  field, a CI variable, or human-set env? Without a source, `by_issue` is a no-op.
- **Attribution coverage (Phase 0 result gates the rest).** If most compute is
  `local_admin`, is the per-user table still worth shipping, and do we add
  `MARIN_USER` first?
