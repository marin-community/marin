# Unified cluster/job Grafana coverage

Weaver #138. Goal: the grafana.oa.dev dashboards lack "at a glance" fleet job
state — you cannot open the site and see what is running, queued, or stuck
across every cluster. Three coupled changes, each grounded in live finelog data
and the existing bridge. Revised after a fable peer review (§8 records what
changed).

## 1. What exists today

`infra/grafana/` is a Cloud Run Grafana plus a Starlette "bridge" that fronts
finelog SQL (per cluster), the live Iris controller RPC (GCE only, by internal
IP), the GitHub API, W&B, and the CoreWeave k8s API. Dashboards: `home`,
`infra`, `fleet`, `iris`, `pipelines`, `training`, `k8s`, `finelog`.

Job coverage is thin and split:

- `infra.json` panel 22 "Jobs by state" hits `/iris/{cluster}/jobs`, one
  `ExecuteRawQuery` returning aggregate counts by state (in-flight + 24h
  terminal). Single-cluster (`${cluster}` variable), GCE-only, buried.
- No per-job listing, no cross-cluster view, no CoreWeave jobs anywhere.
- The stuck signals `iris.task_state` emits (`oldest_pending_age_ms`,
  `oldest_building_age_ms`) feed an alert but no at-a-glance panel.
- The home stat row is all infra; nothing answers "is any job stuck?"

## 2. Dashboard principles applied

Overview-first: add the missing job altitude to the top, not another buried
table. Lead with exceptions: jobs are a queue, so the at-a-glance quantities are
counts by state plus wait age (queue saturation), sorted worst-first. Single
pane: one fleet view grouped by cluster beats flipping `${cluster}`.
Actionability: age thresholds, failed-job links. No drift: reuse the `panelRef`
fragment mechanism for any shared job panel.

## 3. Where job state actually lives (validated against live data)

Validated against the marin hub parquet (`gs://marin-us-central2/finelog/marin/`).
Two facts drive the whole design:

- `iris.task_state` is emitted **only by cluster-view (k8s / CoreWeave)
  controllers** — `TaskStateCollector` starts only when a backend has
  `BackendCapability.CLUSTER_VIEW`, which the GCE/RPC backend does not have
  (`controller.py`). This is deliberate (#7413): "a GCP controller's DB is
  directly queryable via ExecuteRawQuery, so GCP emits nothing." So the hub's
  `iris.task_state` is CoreWeave-only; **marin and marin-dev contribute no rows**.
- marin-dev's finelog is neither the hub nor a forwarder
  (`lib/finelog/config/marin-dev.yaml`), so nothing marin-dev writes reaches the
  hub regardless.

Therefore there is no single finelog query that is the fleet job view. The fleet
view must be assembled by the bridge from three sources it already reaches:

| Cluster        | Live per-job task state          | Terminal / recently-failed |
|----------------|----------------------------------|----------------------------|
| marin (GCE)    | controller RPC (jobs+tasks)      | controller RPC (jobs)      |
| marin-dev (GCE)| controller RPC (jobs+tasks)      | controller RPC (jobs)      |
| cw-* (k8s)     | hub finelog `iris.task_state`    | k8s failed pods (no RPC)   |

The CoreWeave column is unusable until Workstream A: those rows arrive at the hub
with `cluster=NULL` (§4), so they cannot be attributed to a cluster.

## 4. Workstream A — fix CoreWeave iris.* federation stamping

### The bug (confirmed against real data)

On the finelog-marin hub, CoreWeave-origin `iris.*` stats written by the
**controller or k8s backend** — `iris.task_state`, `iris.worker`, `iris.task`,
`iris.task_event`, `iris.profile`, `iris.provisioning` — arrive with the origin
`cluster` column **NULL** instead of `cw-*`. Task-written namespaces (`log`,
`iris.task_status`) stamp correctly.

Proof: the `/benjaminfeuer/glm52-datagen-*` and `grug-agentic` eval jobs' `log`
rows are `cluster=cw-rno2a`; the same jobs' `iris.task_state` rows are
`cluster=NULL` — so they federate, but arrive unstamped. Recent-segment
distribution: `iris.task_state` 100% NULL; `iris.task_status` cw-us-east-02a
173967 / cw-rno2a 4486 / NULL 62964 (the NULL are pre-fix-era rows); `log`
cw-rno2a 2.3M / cw-us-east-02a 1.2M / cw-us-east-08a 32k / NULL 666k. Impact: any
panel grouping iris.* by cluster collapses all CoreWeave into the empty bucket.

### Mechanism

The forwarder (`lib/finelog/rust/src/server/forwarding.rs`) stamps origin cluster
only when the local namespace schema carries the implicit `cluster` column
(`has_origin = schema.column(IMPLICIT_CLUSTER_COLUMN).is_some()`, ~line 367);
missing-column namespaces ship unstamped. The asymmetry that lets it slip: the
hub's log-plane ingest stamps cluster from the authenticated forwarding identity
(`log_service.rs::authorized_cluster`), but the stats-plane ingest
(`stats_service.rs` `WriteRows`) does not — so stats federation depends entirely
on the sender having stamped.

Corrected diagnosis (was "legacy namespace adopted from disk"): `register_table`
applies `with_implicit_cluster` on every register/evolve additively
(`store.rs:356`), and `iris.task_state` was first registered 2026-07-19, two days
after the implicit column landed (#7313, 2026-07-17). A post-#7313 server would
have had the column from first registration. The consistent explanation is that
the **CoreWeave finelog server images predate #7313** (finelog ships as a pinned
wheel/image). That is why five days of controller restarts have not healed it,
and it reorders the verification below.

### Fix

Primary (hub-side, robust): stamp the origin cluster on the hub's `WriteRows`
from the authenticated forwarding identity, mirroring
`log_service.rs::authorized_cluster`. The interceptor already records
`AuthIdentity` for `WriteRows` (auth.rs gates PushLogs/WriteRows/RegisterTable
alike), so the identity is in hand; the omission is log-first history, not
policy. This fixes the class regardless of the sender's schema and — crucially —
needs only a **finelog-marin hub redeploy**, not a CoreWeave redeploy.

Implementation traps the fix must handle (all confirmed against the source):

1. The live failure is a *missing* cluster column, not an empty one. The broken
   namespaces ship batches with no cluster column, which alignment NULL-fills.
   Stamp by injecting/filling the column during align — do not only rewrite empty
   strings.
2. No-op when the receiving namespace's registered schema lacks a cluster column,
   or an added column trips an "unknown column" rejection, which the sender's
   forwarder treats as a permanent poison pill (counted data loss).
3. Mismatch policy: mirror the log plane — a non-empty cluster that disagrees with
   the credential is `permission_denied`. Note `forwarding.rs::push` retries
   `PermissionDenied` forever (only `InvalidArgument` is permanent), so a mismatch
   stalls that namespace. Acceptable — a correct forwarder's stamp equals its
   credential — but state it.
4. The fix fires only on the JWT identity path; the hub stack is CIDR-first over
   RFC1918 (`lib/finelog/config/marin.yaml`). Today CW forwarders reach the hub
   over the public internet and hit the JWT layer (proven by correctly-stamped
   `log`), but routing CW→hub through an internal tunnel would degrade identity to
   `Network` and silently stop stamping. State the invariant; test it.
5. Local-writer invariant: the sender-side forwarder filter is `cluster IS NULL OR
   cluster = ''`. A writer that JWT-authenticates to its own cluster's finelog
   would get rows stamped locally, which the filter then classifies as foreign and
   never forwards. Not currently triggered; document next to both sites (or extend
   the filter to also pass `cluster = <own cluster>`).

No backfill: historical NULL CoreWeave rows stay NULL, so any `NULL→marin` mapping
would misattribute history. The live jobs view uses a 2-minute window so this is
moot there, but `iris.worker`/`iris.task` history panels must not remap NULL.

Secondary (sender-side, confirmed correct in code): `merge_schemas` is additive,
so once a CoreWeave finelog runs a post-#7313 image, the next controller restart
re-registers `iris.task_state` with the cluster-column schema and the forwarder
stamps on the sending side too. Verification order: upgrade a CW finelog image →
restart its controller → confirm the namespace gains the column and stamps. This
needs cluster access the weaver VM lacks; the hub-side fix does not depend on it.

## 5. Workstream B — unified fleet Jobs, via a merged bridge endpoint

Because no single query spans the fleet (§3), the primary deliverable is a bridge
endpoint that merges the three sources into one uniform shape — the
bridge-owns-the-query pattern already used for `/iris/*` and `/k8s/*`:

```
GET /iris/fleet/jobs   ->  one row per active root job, all clusters
    columns: cluster, job, pending, assigned, building, running, active,
             oldest_pending_age_ms, oldest_building_age_ms, source
```

Assembly:

- GCE (marin, marin-dev): each cluster's `IrisSource` runs one `ExecuteRawQuery`
  for active root jobs and their per-state task counts from `tasks`, plus oldest
  pending/dispatch ages from the attempt timestamps. `source="rpc"`. The bridge
  already holds an `IrisSource` per GCE cluster.
- CoreWeave: one hub-finelog query over `iris.task_state`, latest snapshot per
  (cluster, root_job_id) within a **fixed 2-minute trailing window** (3–4
  emission intervals), independent of the Grafana range so finished jobs — which
  stop emitting with no final zero row — cannot linger as "active". `source="finelog"`.
  Correct SQL (note `COALESCE` — a bare `JOIN USING(cluster)` drops NULL keys, and
  today every CW row is NULL):

```sql
WITH recent AS (
  SELECT COALESCE(cluster,'') AS cluster, root_job_id, pending, assigned,
         building, running, oldest_pending_age_ms, oldest_building_age_ms, ts,
         ROW_NUMBER() OVER (PARTITION BY COALESCE(cluster,''), root_job_id
                            ORDER BY ts DESC) AS rn
  FROM "iris.task_state"
  WHERE ts >= {{from}})            -- {{from}} = now - 2 minutes, bridge-fixed
SELECT cluster, root_job_id, pending, assigned, building, running,
       oldest_pending_age_ms, oldest_building_age_ms
FROM recent WHERE rn = 1 AND root_job_id <> ''
```

The endpoint merges both, maps `cluster=''`→`marin` only for hub rows the bridge
knows are marin-local (post-A; pre-A the CW rows are NULL and are labeled
`unknown` rather than misattributed to marin), and sorts worst-first by
`GREATEST(oldest_building_age_ms, oldest_pending_age_ms)`.

### Panels — new `jobs.json`

1. Fleet rollup stat row: total running / pending / building across the fleet and
   count of stuck jobs (either age over threshold), from the merged rows.
2. Active jobs table (fleet-wide): the merged endpoint, columns cluster | job |
   pending | assigned | building | running | oldest pending | oldest building,
   worst-first, age columns thresholded.
3. Per-cluster freshness: last rollup-row age per cluster rendered as an
   exception — "silence ⇒ controller down" is the table's own contract, so a
   stale/absent rollup should show, not silently drop from the sum.
4. Queue depth over time: `pending` and `oldest_pending_age_ms` from the CW rollup
   rows (`root_job_id=''`), `date_bin`ned, one series per cluster.
5. Recently failed/killed (GCE): the existing `ExecuteRawQuery` path,
   `state IN (5,6,7,8) AND finished_at_ms > now-24h` (includes `unschedulable`),
   columns job | state | exit_code | error | finished_at. marin + marin-dev.

### Home additions

A compact fleet-jobs `panelRef` fragment on `home.json`: "jobs running" / "jobs
stuck" stat pair (and optionally the top few stuck jobs), so the landing page
answers "is anything stuck right now" without navigating. Shared fragment ⇒ no
drift from `jobs.json`.

Sequencing: B depends on A for the CoreWeave column; the GCE (RPC) half ships
independently and is correct today. Until A deploys, CoreWeave rows are labeled
`unknown`, and the panel description says so.

## 6. Workstream C — grafana.* finelog history namespace

The bridge's k8s/alert reads are current-state only, so there is no record of
*when* we lost a node or a GB200 rack tray. The bridge holds a finelog
`LogClient` to the hub and can write. Split by value and land the strongest piece
first:

- `grafana.alert_events` (ship first): the Loom alert webhook already receives
  every firing group; also append it here (fingerprint, alertname, severity,
  target_cluster, status, starts_at). Event-driven, naturally single-writer,
  genuinely new data — a durable alert history independent of Grafana's own state
  retention.
- Rack-tray / node-loss history: prefer deriving from data that already
  federates. The k8s backends emit one `iris.worker` row per node (GPU hardware
  included) to the hub; rack-tray history is best added as a `rack` column on
  `IrisWorkerStat` (producer next to its mechanism), not re-observed by the
  bridge. The one thing that federation misses is a tray/node that has stopped
  registering with the k8s API entirely (emits no row) — for exactly that,
  `grafana.rack_trays` (cluster, rack, trays_total, trays_ready per scan) is
  justified, because absence-of-rows is far harder to query than an explicit
  `trays_ready` step. Name that trade-off rather than duplicating `iris.worker`.

Schema notes: the subject-cluster field must **not** be named `cluster` —
`with_implicit_cluster` keeps a declared `cluster` column as-is, so it would
occupy the implicit origin column and make every grafana.* row look foreign to a
future forwarder. Use `target_cluster`. Single-writer is guaranteed by the
deploy (`min=max=1` instances, already set to avoid duplicate alert evaluators) —
assert it as an invariant. Set a bounded `StoragePolicy`. Writes are best-effort
and must never fail a read path.

## 7. Sequencing and risks

A (hub redeploy) unblocks the cluster dimension → B (dashboard) is the visible
win → C (alert history) is independent and lands in parallel.

- A is a finelog Rust change: hub-side stamping fix + Rust test here; going live
  is a finelog release + finelog-marin hub redeploy (not CW). Sender-side
  self-heal needs a CW finelog image upgrade and cluster access to verify.
- B ships the GCE half today; CoreWeave rows label as `unknown` until A deploys.
- C: `grafana.alert_events` first; justify or drop the scan-driven tables against
  the existing `iris.worker` federation.

## 8. What the review changed

- §3 added: marin/marin-dev emit no `iris.task_state` (CLUSTER_VIEW gate);
  marin-dev's finelog does not forward. Killed the "one finelog query = fleet
  view" framing.
- §5 restructured around the merged `/iris/fleet/jobs` endpoint as primary (not
  optional); fixed the NULL-dropping `JOIN USING`, switched to a fixed 2-minute
  freshness window to avoid ghost (finished) jobs, added per-cluster freshness as
  an exception panel, added terminal state 8, and the `active` column.
- §4 fix spec expanded with the missing-column/no-op/mismatch/CIDR/local-writer
  traps; corrected deploy target (hub redeploy suffices) and the root mechanism
  (CW server image predates #7313, not legacy adoption).
- §6 renamed the subject field to `target_cluster`, asserted single-writer via
  `min=max=1`, and reframed rack/node history as mostly an `iris.worker` column
  with `grafana.rack_trays` only for the never-registers case; `alert_events`
  ships first.

## 9. Open questions

1. Prefer the bridge-merge (respects #7413's GCP-emits-nothing) or drop the
   CLUSTER_VIEW gate so GCE also emits `iris.task_state` (uniform substrate, but
   reverses a deliberate maintainer decision and still needs a marin-dev union)?
   This design implements the bridge-merge; the gate change is a maintainer call.
2. Rack/node-loss history: `iris.worker` `rack` column vs bridge `grafana.rack_trays`
   — land both, or only the column plus a never-registers alert?
