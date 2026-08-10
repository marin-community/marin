# Design: multi-backend dashboard + RPC surface

## Problem

One Iris controller now fronts a collection of task backends (a GCP/TPU
worker-daemon backend, a k8s backend, multiple CoreWeave worker-daemon backends).
The web dashboard and the controller RPC surface still assume a single backend:

- Tab visibility is gated on **one** backend's capabilities (`/auth/config` serves
  the representative backend), so a worker-daemon + k8s controller hides one's tabs.
- `GetAutoscalerStatus` shows only the first backend's autoscaler; the other
  backends' slices/capacity are invisible.
- `GetKubernetesClusterStatus` assumes the representative backend *is* the k8s one
  and breaks otherwise.
- No RPC row (`JobStatus`/`TaskStatus`/`WorkerHealthStatus`) carries `backend_id`,
  even though it is in the DB — so the UI cannot show or filter by backend.
- The new "job matched no backend" failure mode is indistinguishable from "no
  capacity."

This design covers (1) the dashboard UX for N backends and (2) the RPC/proto
changes that feed it, and decides which PR each part lands in.

## Principles

- **Single-backend clusters look exactly as today.** Every multi-backend
  affordance is gated on `multiBackend = backends.length > 1`. No new tab, no
  selector, no extra column, no detail-page row for the common case.
- **Backend is a scope/filter dimension, not top-level navigation.** Jobs, users,
  and budgets are controller-global (threaded across backends in one tick); forcing
  a backend choice before you can see a job fights the data model and destroys the
  cross-fleet aggregate view that is the whole point of one controller fronting many
  backends.
- **Additive + back-compatible at the wire.** New proto fields are optional; empty
  default means "all backends" / single-backend identity. No migration (columns
  already exist).
- **Normalize at the boundary.** `backend_id` becomes a first-class field on row
  messages, never parsed from scale-group name strings.
- **Reuse existing components** (`DataTable`, `TabNav` slot, `InfoCard`,
  `FleetOverview`, `ConstraintChip`) rather than inventing widgets.

## UX design

**Backend as scope + one overview tab.** Two affordances, both `multiBackend`-gated:

1. **Backends overview tab** (`/backends`): a responsive grid of `InfoCard`s, one
   per backend. Per card: id + name, health dot, capability chips
   (`workers`/`autoscaler`/`cluster`); kind; users (`all (*)` or `restricted (N)`);
   advertised devices as `ConstraintChip`s; owned scale-group count (→ `/capacity?backend=`);
   worker count (→ `/fleet?backend=`); `running · pending` task counts; and a capacity-health
   rollup (`9/12 pools · 2 degraded`, `healthy`, or `at quota`). Backends are few and
   heterogeneous (advertised attrs differ per backend), so a card grid beats a wide sparse
   table. A banner lists any unroutable jobs.

2. **Scope selector** (`BackendScope.vue`) rendered into the `TabNav` slot: an
   `All backends ▾` `<select>` that writes/clears `?backend=<id>`. Because every tab
   already spreads `...route.query`, the scope persists across tab switches and is
   shareable/back-buttonable for free.
   - **All backends (default):** today's aggregate, plus a **Backend** column on the
     Jobs/Fleet/Capacity tables (rendered only in All mode) — click a value to scope.
   - **One backend:** each tab passes `backendId` into its existing RPC query
     (`JobQuery`/`WorkerQuery`/autoscaler/scheduler reads) for **server-side** filtering
     (correct pagination); the redundant Backend column is hidden.

**Capability-gating fix.** `App.vue` gates `ALL_TABS` on the **union** of capabilities
(served by `/auth/config`), so a worker-daemon + k8s controller shows both Workers and
Cluster. Per-backend capabilities are retained on each `BackendInfo` so the selector can
dim a tab the scoped backend lacks. We reject per-backend *tab duplication*
("Workers (gcp) / Workers (cw)") — it explodes the nav and kills the cross-fleet view.

**Routing failures + `--backend` pins.** The per-job diagnostic reason already renders
via the Jobs Diagnostic column / `JobDetail` banner, but a banner/count/card needs
**structured** data, not parsed `pendingReason` strings (codex). `ListBackends` carries an
`unroutable_job_count` + a small `(job_id, reason)` sample; the dashboard adds one
`danger`-styled **Unroutable** `MetricCard` (distinct from "no capacity," which an
autoscaler can fix), a Backend column on the Pending/Unmet tables, and a pin glyph
(`📌 backend`) + a `Backend` `InfoRow` (pinned chip) on `JobDetail`.

**Scale at many backends + edge cases** (codex): above a threshold (~8) `BackendScope`
becomes a searchable combobox and `/backends` switches from the card grid to a compact
table. A tab scoped to a backend that lacks its capability shows an explicit "this backend
does not support this surface" empty state (not silent dimming). An unknown `?backend=` is
handled deterministically — the frontend clears it against the known roster and the server
returns `INVALID_ARGUMENT` for an unknown `backend_id` filter; never a silent fallback to
all backends.

**Detail pages** gain a `Backend` `InfoRow` (Job/Task/Worker), all `multiBackend`-gated.

See `spec.md` for the exact frontend component list. Mockup of the overview:

```
Backends                                          3 backends · 1 unroutable job
┌ gcp · GCP/TPU       ● [workers][autoscaler] ┐  ┌ eu-k8s · EU cluster ● [cluster] ┐
│ kind    worker-daemon (GCP/TPU)             │  │ kind    kubernetes               │
│ users   all (*)                             │  │ users   restricted (3)           │
│ devices [v5e-4] [v5e-16] [v5p-8]            │  │ devices [h100-8] [h100-1]        │
│ groups  12   workers 84   tasks 312·40      │  │ groups  —    workers —   tasks 18·0 │
│ capacity ● 9/12 pools · 2 degraded          │  │ capacity ● healthy               │
└─────────────────────────────────────────────┘  └──────────────────────────────────┘
```

## RPC / API design

**Correctness fixes (server-only):**
- `/auth/config` serves the **union** capabilities + a per-backend list (keeps the
  legacy `backend` key for back-compat).
- `GetAutoscalerStatus` **unconditionally merges across all** backends' autoscalers (no
  per-backend drill-down in core — that request filter is feature-PR), tagging each
  `ScaleGroupStatus` with `backend_id`. Safe because scale-group names are globally unique
  across backends (they are the single key space of `_scale_group_to_backend`), so the
  per-group `current_demand`/`recent_actions` need no further disambiguation; merged
  `recent_actions` are sorted newest-first and truncated to the existing cap, and
  `last_routing_decision` (a single per-autoscaler snapshot) is left unset in the merged
  view.
- `GetKubernetesClusterStatus` locates the `CLUSTER_VIEW` backend by capability (first by
  sorted backend_id when, hypothetically, more than one exists; the feature PR adds the
  `backend_id` filter and the "require it when >1" rule).
- `ControllerProtocol` widened with `backends` and `backend_id_for_scale_group(...)`.

**Feature surface (lands with the frontend):**
- `backend_id` on `JobStatus`/`TaskStatus`; `backend_id` + `scale_group` on
  `WorkerHealthStatus`; `backend_id` on `Pending/RunningTaskBucket`.
- `backend_id` filters on `JobQuery`/`WorkerQuery` (+ optional drill-down on the
  autoscaler/k8s requests).
- New `ListBackends` RPC returning a `BackendSummary` per backend (capabilities,
  advertised attributes, restricted/allowed-users, owned scale groups, worker/task
  counts, autoscaler presence, capacity-health rollup).
- Thread `backend_id`/`scale_group` into `TASK_DETAIL_COLS`/`WORKER_DETAIL_COLS`/
  `_JOB_ROW_COLUMNS` and the row dataclasses.

Exact proto/signatures in `spec.md`.

## PR placement

The split follows one rule: **is the dashboard *wrong*, or merely *missing detail*?**

- **Core multi-backend PR (#6730, this branch):** the three correctness fixes +
  `ControllerProtocol` widening + the single proto field `ScaleGroupStatus.backend_id`
  (the disambiguator that makes the merged autoscaler view usable — unsafe to land the
  merge without it). These fix active bugs a 2-backend cluster's dashboard would have;
  all server-only, regression-shaped, no frontend dependency. Net new wire surface: one
  field.
- **Dedicated dashboard PR (new, stacked on #6730):** all row-level `backend_id`
  tagging, query filters, `ListBackends`, and the full Vue frontend (overview tab,
  scope selector, columns, detail rows). These are inert until a frontend reads them, so
  proto + UI land and review together.

Rationale: keep #6730 focused (it is already a large contract rework) while ensuring it
doesn't ship a controller whose own dashboard misreports a multi-backend config. The
feature PR is a coherent, independently reviewable unit (proto + service + frontend +
e2e screenshots) that depends only on #6730's backend collection and `backend_id` columns.

Neither PR3 (remote transport) nor PR4 (hardening) is involved — the dashboard is
orthogonal to the remote backend.

## Out of scope

- The `remote` transport's effect on the dashboard (a remote backend renders the same
  as any `CLUSTER_VIEW` backend via `ListBackends`/`GetKubernetesClusterStatus`-style
  surfaces; PR3 adds the transport, not new UI).
- Residual-demand-as-timeseries / capacity forecasting.
- Per-backend auth boundaries (one controller, one auth, shared budgets — backends are
  a routing/ownership dimension, not a tenancy boundary).

## Resolved (codex review)

1. **`ScaleGroupStatus.backend_id` in the core PR — yes.** The merged capacity view is
   actively wrong without it; the autoscaler merge stays in core. The merge is
   *unconditional* (all backends); per-backend drill-down request filters move to the
   feature PR so core adds exactly one wire field.
2. **Dedicated `ListBackends` — yes.** Do not grow `GetSchedulerState` into a cluster summary.
3. **Worker `backend_id`: derive from `scale_group` in-memory — yes, no migration now.**
   Relies on the invariant that scale-group names are globally unique across backends
   (single `_scale_group_to_backend` key space) and fall back to `DEFAULT_BACKEND_ID`.
4. **Restricted backends: show greyed for operators.** The dashboard is operator-facing; if
   backends ever become end-user tenancy boundaries, enforce visibility server-side and hide
   unauthorized backends.

Two contracts codex sharpened that the spec now pins: (a) row `backend_id` is the **literal**
backend id (`DEFAULT_BACKEND_ID` on single-backend clusters, the routed id once pinned, empty
only while a job is genuinely unrouted) — never an overloaded empty sentinel; the UI hides the
column via `multiBackend`, not by reading empty string. (b) `BackendSummary` carries `kind`
and `allowed_user_count` (so "restricted (N)" needs no full allow-list), and the per-backend
health dot is derived frontend-side from `capacity_health` + counts (the controller has no
single stored backend-health signal today; a backend-liveness field is a natural addition when
remote backends land in PR3).
