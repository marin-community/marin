# Cluster-native federation: one `cluster` coordinate, `local` as the default

> **Supersedes the naming half of [`design.md`](./design.md).** design.md §4–§8 stand as the
> data-model/rollout design, but its §5.1 "deterministic `remote_job_id`", the `~`-folded remote
> root (§6.1), and the local-vs-federated framing are **replaced** by this document. Federation
> has **never been deployed** — migrations `0034`–`0036` have never run against a live DB — so we
> are free to rewrite the DDL, protos, and identity model in place rather than layering shims.

## 1. Goal

Make **cluster a first-class coordinate that is always present**, defaulting to `local`, so the
controller has **one uniform path** for local and federated work. A peer is not a special case —
it is just a `cluster` value that isn't `local`. The end state the maintainer asked for:

> minimal special casing around peers in the code base — instead all operations are cluster-centric,
> with the local operations just a default cluster.

Two things drop out of that:

1. **No name rebasing.** The `~`-folded `remote_job_id` (`/<user>/<parent_cluster>~<name>`) and the
   whole `FEDERATION_DELIMITER` machinery are deleted. A job's id is **cluster-invariant**: the peer
   runs, logs, and reports the job under the *same* `job_id` the parent submitted. Identity carries
   no cluster; the `cluster` coordinate travels beside it.
2. **No local/federated branch.** `child_cluster` (a federation-only discriminator that is `""` for
   local) becomes `cluster` (always set, `'local'` by default). Reads filter/​group by `cluster`
   uniformly; the scheduler fold is "the `local` cluster's rows", not "the rows that aren't federated".

## 2. The model

### 2.1 `cluster` — one column, always set, `local` by default

`jobs.cluster` and `tasks.cluster` (renamed from `child_cluster`) are `NOT NULL DEFAULT 'local'`:

| value | meaning | who runs it |
|---|---|---|
| `'local'` | owned by **this** controller | this controller's backends (`backend_id` set) |
| `'<peer>'` | handed off to that peer cluster | the peer's backends (this controller mirrors) |

`'local'` is a **reserved cluster name** — an alias for "this controller" that every controller uses
for its own rows. It is a **compile-time constant**, which is load-bearing: the fold's partial indexes
and the `local_tasks` selectable are defined at schema-import time (`schema.py:375-419`) as
`WHERE cluster = 'local'`, exactly as they are `WHERE child_cluster = ''` today. Storing the
controller's *real* cluster id in the column instead (e.g. `'cw-us-east'`) would make that predicate a
runtime value and **break the static partial index** — so the sentinel stays a constant, and the real
cluster id is resolved from `local` only at the one boundary that needs an absolute name (§4).

Invariant, replacing the `backend_id`⊕`child_cluster` exclusivity: `backend_id != '' ⟹ cluster = 'local'`
(only a local task binds a local backend).

### 2.2 `job_id` is cluster-invariant — no rebasing, no `remote_job_id`

A job's identity (`JobName`, the `jobs`/`tasks` PK, every FK, the finelog key) stays exactly
`/user/job[/child…][/index]` and is **the same string on the parent and the peer**. The parent hands
`/alice/train` to the peer; the peer materialises `/alice/train` in *its own* DAG; the peer's changelog
reports `/alice/train`; the parent mirrors `/alice/train`. There is no second name.

This deletes, wholesale:

- `FEDERATION_DELIMITER = "~"` and the submit-time reservation of `~` (`types.py:95`, `service.py:1153`).
- `JobName.federated_remote_root` / `is_federated_remote` / `split_federated_root` (`types.py:174-214`).
- `manager.encode_remote_job_id` / `_rebase_task_id` / `_rebase_profile_target` (`manager.py:49-74`).
- `federated_jobs.remote_job_id` (the column), `FederatedHandle.remote_job_id`,
  `federated_job_for_remote_id`, `federated_handles_for_peer` keyed by remote id (`reads.py`).
- `JobStatus.remote_job_id` (36), `TaskStatus.remote_job_id` (26), `FederationJobDelta.remote_job_id`
  (→ renamed `job_id`) in the protos; the `remoteJobId` TS fields and the dashboard's client-side
  rebase (`TaskDetail.vue`, `JobDetail.vue`, `LogViewer.vue`).

### 2.3 `cluster` as an addressing coordinate; `JobName` stays identity-only

A user can *address* a specific cluster's job with the grammar

```
/cluster/<name>/<user>/<job>          e.g.  /cluster/cw-us-east/alice/train
```

This is **parsed at the boundary** (CLI/API/dashboard query) into a pair `(cluster, JobName)`; the
cluster is peeled off into the `cluster` coordinate and the identity is the bare `JobName`. Resolution:

- `<name> == 'local'` → the caller's own controller (`cluster='local'`).
- `<name> == '<peer>'` → that peer.
- **no `/cluster/...` prefix** → cluster **unset** → a query wildcard (all clusters); a *submit* with
  no cluster defaults to `local`.

`JobName` gains **no** cluster field — putting an addressing qualifier inside the value type used as a
DB key / log key / FK is the "one field, two meanings" trap AGENTS.md warns against ("normalize inputs
at the boundary, not throughout"; "avoid `X | str` unions"). Cluster is a **sibling coordinate**, not
part of identity, so the mirror can store the peer's `/alice/train` as-is under a different `cluster`
value without any rebasing.

### 2.4 `local` is relative; the fold is "my cluster"

`cluster='local'` is **relative to each controller**. The same job is:

- `cluster='cw'` on the parent (it delegated to cw), with **no task rows** materialised (mirrored on sync);
- `cluster='local'` on cw (cw executes it), a **normal local job** with a `federated_jobs(direction=RECEIVED)`
  sidecar recording the requester.

So a received handoff is *not* a special row on the peer — it is an ordinary `cluster='local'` job plus
a one-row sidecar. Each controller's scheduler folds **its own `cluster='local'` rows**; federated rows
(`cluster='<peer>'`) are structurally excluded by the same `local_tasks` selectable as today, now
`WHERE cluster='local'`. "Local is just a default cluster" is literally true: the fold is
`cluster = <the reserved default>`.

## 3. Uniqueness & collisions

Rebasing gave per-parent namespacing for free (`/alice/cw~train` can't clash with the peer's own
`/alice/train`). Dropping it means `job_id` must be unique **within a cluster's namespace**, enforced at
three points — the first already exists, the other two are small additions:

1. **Submit (parent).** `jobs.job_id` is the PK; `launch_job` already rejects a duplicate name
   (`service.py:1515-1521`). The parent is the sole submission point in practice, so it is the name
   allocator for the whole federation.
2. **Handoff admission (peer).** When a peer *receives* a handoff for `/alice/train`, it distinguishes:
   - no such job → materialise it (`cluster='local'` + `federated_jobs` RECEIVED, requester=parent);
   - a job `/alice/train` **already RECEIVED from the same requester** → idempotent re-drive, KEEP (the
     boot-recovery / retry path — this is what today's `EXISTING_JOB_POLICY_KEEP` covers);
   - a job `/alice/train` that is **local to the peer or received from a different requester** → **reject
     the handoff** with a clear `ALREADY_EXISTS`. The parent surfaces the failure on the handle (a new
     `HANDOFF_REJECTED` terminal state) rather than silently overwriting.
3. **Sync ingest (parent).** The parent mirrors only job_ids it holds a SENT handle for
   (`federated_handle_for(peer_id, job_id)`); a peer reporting an id the parent never handed to it is
   **logged at WARNING and skipped** (the maintainer's "log warning and ignore" — it means a peer/parent
   disagreement, never normal traffic).

**Cross-cluster log keys** are disambiguated by finelog's `cluster` column (§4), not by the key — so even
if two clusters legitimately run the same `/alice/train`, their logs never mix.

> **Semantic cost, stated honestly:** this trades the automatic per-parent namespacing of `~` for
> "job names are unique per cluster namespace, colliding handoffs are rejected." For the deployed
> topology (one parent fanning out to peers through a single name allocator) this is a non-issue; a
> future many-parents-one-peer topology would surface rejections that `~` hid. Acceptable now, and the
> `cluster` coordinate leaves room to reintroduce a composite `(cluster, job_id)` key later without a
> name rewrite.

## 4. Logs: query by `(job_id, cluster)`; resolve `local` once

The peer runs `/alice/train`; its workers push logs under key `/alice/train/0:<attempt>` (unchanged,
`log_keys.py:52`), relayed to the shared global finelog **stamped with the peer's real cluster id**
(`PushLogsRequest.cluster`). A reader fetches a job's logs with `FetchLogs(source=/alice/train/, cluster=X)`.

The `cluster` filter value is `resolve_finelog_cluster(row.cluster)`:

```
resolve_finelog_cluster(c) = self.cluster_id  if c == 'local'  else c
```

This is the **one irreducible place** the `local` alias becomes an absolute id: finelog is a *global*
store shared by many controllers, so each must stamp/read its *real* id (every cluster calling its own
logs `'local'` would collide). Everywhere else stays `local`-relative. The relay stamps `self.cluster_id`;
the read side (dashboard + CLI + any server helper) translates `local → self.cluster_id` and passes
`<peer>` through verbatim — and because `peer_id` in the registry **is** the peer's real cluster id, a
federated job's filter is just its `cluster` column value.

This requires finishing the **#6862 Python wire contract** the finelog `cluster` PR deferred:

- add `cluster` to the Python `logging.proto` + regenerate stubs (Rust already has it);
- the relay/forwarder sets `PushLogsRequest.cluster = self.cluster_id` (`finelog_relay.py`, `forwarder.py`);
- the dashboard `LogViewer` sends `cluster` on `FetchLogs` (resolving `local`), replacing the
  `remoteJobId` key-rebase entirely. The dashboard learns `self.cluster_id` once from bootstrap config.

The `FederatedLogsNotice.vue` "honest empty state" is deleted; federated logs render natively. Issue
**#6883** (native in-parent log serving) is closed by this.

## 5. Schema changes (edit `0034`–`0036` in place — never run)

- `jobs`/`tasks`: `child_cluster` → **`cluster`**, `NOT NULL DEFAULT 'local'`; backfill/`server_default`
  `'local'`. Partial indexes `idx_tasks_*_local` and the `local_tasks` selectable → `WHERE cluster='local'`.
- `federated_jobs`: **drop `remote_job_id`**. Keep `direction` (SENT/RECEIVED), `peer_id`,
  `owner_principal`, `handoff_state` (SENT; add `HANDOFF_REJECTED`), `cancel_intent_version`.
- `federation_sync_state`, `federated_tasks`, `federation_changelog`: unchanged except the `cluster`
  rename ripples. `federation_changelog.job_id` already carries the plain id — no change.
- `task_attempts.attempt_uid`: drop the `"{peer_id}~{uid}"` namespacing (`writes.py:944`) — with
  cluster-unique job names the mirrored uid `/alice/train/0:1` is already unique; the peer-prefix was
  only there because of `~`. (Verify against the global unique index before removing.)

Because the DDL has never run, we **rewrite `0034`–`0036`** to land the final shape rather than adding a
`0037` that renames what `0034` just created. (Open question 10.a: squash vs. keep the three-migration
history — recommend rewriting in place for a clean first-run.)

## 6. Proto changes

- Drop `JobStatus.remote_job_id (36)` and `TaskStatus.remote_job_id (26)`.
- `JobStatus.child_cluster (35)` / `TaskStatus.child_cluster (25)` → **`cluster`**, documented as
  always-set (`'local'` default). `JobQuery.child_cluster (11)` → `cluster` (unset = all).
- `FederationJobDelta.remote_job_id (1)` → **`job_id`** (the peer already sends its own local id here —
  `service.py:3134` — which now *is* the shared id, so this is a rename, not a semantics change).
- Regenerate `job_pb2` / `controller_pb2` (+ `.pyi`) and the finelog `logging_pb2`.

## 7. Cleanup inventory (the actual work)

| Area | File(s) | Change |
|---|---|---|
| Identity | `cluster/types.py` | delete `FEDERATION_DELIMITER`, `federated_remote_root`, `is_federated_remote`, `split_federated_root` |
| Handoff/proxy | `federation/manager.py` | delete `encode_remote_job_id`/`_rebase_*`; `handoff.name = local_job_id.to_wire()`; cancel/exec/profile target the plain `job_id` |
| Store specs | `federation/store.py` | drop `remote_job_id` from `HandoffSpec`/`CancelTarget`; key by `job_id` |
| Mirror | `controller/federation_store.py` | match deltas by `job_id`; collision → warn+skip; drop reverse-lookup by remote id |
| Reads | `controller/reads.py` | `FederatedHandle` drops `remote_job_id`; `federated_handle_for(peer_id, job_id)`; `cluster` rename |
| Writes | `controller/writes.py` | `cluster` column; drop attempt-uid `~` prefix; mirror unchanged otherwise |
| Service | `controller/service.py` | drop `remote_job_id` population (job/task status, exec/profile); **peer-side handoff-admission collision check**; `resolve_finelog_cluster`; drop `~` submit guard |
| Controller wiring | `controller/controller.py` | `cluster_id` is the real id; expose to log path + bootstrap |
| Finelog Python | `finelog/proto/logging.proto`, client, `forwarder.py`, `controller/finelog_relay.py` | add `cluster` field; relay stamps `self.cluster_id` |
| Dashboard | `dashboard/src/...` | delete `FederatedLogsNotice.vue`, `remoteJobId` (rpc.ts), the rebase in `TaskDetail.vue`/`JobDetail.vue`/`LogViewer.vue`; `LogViewer` sends `cluster`; `cluster` column/tag flat namespace; `child_cluster`→`cluster` in TS |
| Tests | `tests/cluster/**` | update `test_types.py` (drop rebase cases), `test_federation_handoff.py`, `test_federation_exec_proxy.py`, `test_migration_0035/0036`; add peer-side collision-reject test + cluster-filter log test |

## 8. Special-casing removed (the payoff)

- No `if is_federated_remote(...)` / `split_federated_root(...)` name gymnastics.
- No `remote_job_id` threaded through handle → spec → manager → peer → proto → TS → dashboard.
- No dashboard client-side key rebase; no `FederatedLogsNotice` empty-state branch.
- No `child_cluster == ""` "is it local?" checks — everything is `cluster == 'local'` vs a value, and
  most reads don't check at all (they group/filter by `cluster`).
- The fold seam is unchanged in shape (still one `local_tasks` selectable), only the sentinel moves
  `'' → 'local'`.

## 9. Sequencing (this reworks merged code — flag to maintainer)

The `~`/`remote_job_id` machinery is **on `main`** (federation PRs 1–4 merged); this branch is PR5
(dashboard). This cleanup therefore edits merged federation internals, not just the dashboard. Options:

- **(A)** Fold the cleanup into this PR (#6884) — it already reworks federation naming (commit
  `c1884067c`), so the reviewer sees the identity model settle in one place before it ships.
- **(B)** Land a dedicated "cluster-native federation" PR stacked before/with the dashboard.

**Decided (maintainer, 2026-07-03): (A)** — fold all cleanups into #6884. "Better big and fix
everything than split it out." Nothing is deployed and the changes are tightly coupled to the
dashboard log serving that triggered this PR.

## 10. Open questions for the reviewer

a. **Migration hygiene** — rewrite `0034`–`0036` in place (clean first-run, no rename-of-a-rename) vs.
   append `0037`. Recommend rewrite (never deployed).
b. **`cluster` coordinate vs. `JobName` field** — this doc keeps `JobName` identity-only and carries
   cluster as a sibling coordinate (§2.3). Confirm that's the right cut vs. a `JobName.cluster` field.
c. **Where `local → self.cluster_id` lives** — one helper at the finelog boundary (§4). Is there any
   *other* absolute-id boundary (cross-cluster addressing, peer-to-peer) that also needs it?
d. **Peer-side collision policy** — reject with `ALREADY_EXISTS` + a `HANDOFF_REJECTED` handle state
   (§3.2). Confirm vs. silently ignoring.
e. **Sidecar shape** — keep `federated_jobs.direction` (one table, SENT/RECEIVED) vs. split into two
   tables now that identity is uniform. Separation-of-concerns call.
f. **Naming** — should the config section / `FederationManager` / `PeerRouter` / `ListPeers` be renamed
   cluster-centric (e.g. `clusters:` / `ClusterRouter`), or does "peer" still name the *relationship*
   (a cluster I delegate to) cleanly while `cluster` names the *coordinate*? Lean: keep "peer" for the
   relationship, "cluster" for the coordinate — but the reviewer should judge whether that's a
   distinction that pays for itself.

## 11. Review resolutions (2026-07-03) — the implementation contract

A critical review hardened the plan. The sentinel flip `"" → 'local'` is **not a rename**: today the
code encodes "is local?" as the *falsy empty string* (`if job.child_cluster`), so a truthy `'local'`
inverts ~15 branches. These are the binding decisions the cleanup must honour.

### 11.1 Blockers folded in

- **B1 — Truthiness, not equality (the load-bearing fix).** The discriminator is read in boolean
  position all over, relying on `"" == local == falsy`. A bare `grep child_cluster == ''` misses them.
  Every one must become an explicit federated check via a single helper:
  - Python: `service.py:352, 1557, 1566, 1686-1687, 1801` — `if …child_cluster` → `if is_federated(…cluster)`.
  - Dashboard: `TaskDetail.vue:332 (v-if="!task.childCluster"), :366, :554`; `JobsTab.vue:482, :665`;
    `JobDetail.vue:967` — via a TS `isFederated()` helper.
- **B2 — Python param defaults.** `ops/job.py:138` and `writes.py:189` default `child_cluster: str = ""`.
  A local submit uses the default, so leaving it `""` writes `cluster=""` and **the row fails the
  `WHERE cluster='local'` fold → local tasks never schedule**. Flip both defaults to `LOCAL_CLUSTER`.
- **B3 — Wire carries the literal `'local'`.** Population sites `service.py:292, 1593, 1705, 3099` use
  `child_cluster or ""`. **Decision: emit `'local'` on the wire, delete the `or ""` fallbacks**, and flip
  the dashboard truthiness (B1) — do *not* remap `local→""` at the boundary (that reintroduces the
  special-case we are deleting).

### 11.2 Majors folded in

- **M1 — Peer-side admission is federation-aware, not `KEEP`.** `EXISTING_JOB_POLICY_KEEP` returns the
  existing job regardless of owner, so a real collision binds to the wrong job and the parent hangs.
  Add a federation admission branch, run when `request.HasField("federation")`, at **both** the outer
  existence check (`service.py:1312`) and the inner-tx re-check (`service.py:1515`): existing row is
  `federated_jobs(direction=RECEIVED)` with `peer_id == requester_id` ⇒ idempotent KEEP; **any other
  existing row ⇒ reject `ALREADY_EXISTS`** (covers local, other-requester, and SENT-handle collisions).
- **M2 — A rejected handoff terminalizes.** `_deliver_handoff` (`manager.py:317-329`) catches every
  `ConnectError` as "will retry", so an `ALREADY_EXISTS` re-drives forever. Special-case
  `exc.code == Code.ALREADY_EXISTS` → write `HANDOFF_REJECTED` and drop from `pending_handoffs()`.
- **M3 — Log read topology is the hub, made explicit.** The dashboard reads logs through a *dumb*
  reverse proxy (`proxy/system.log-server/finelog.logging.LogService`, `useRpc.ts:13`), so there is **no
  server-side place to resolve `local`** — the dashboard learns `self.cluster_id` from bootstrap and
  sends `cluster` itself. And a `cluster='<peer>'` filter only returns rows if `system.log-server`
  points at the **shared global finelog the peers relay to** — which is the intended topology (the
  maintainer: "the parent dashboard's finelog _is_ the global store"). Documented as a hard assumption.
- **M4 — `'local'` is a reserved name.** Fail-fast config validation must reject a `cluster_id` or a
  peer named `'local'`; the sentinel and the real-cluster-id namespace (`requester_id`/`peer_id` on the
  wire are *always* the real id, `manager.py:334,371`) share a keyspace and must stay disjoint.

### 11.3 Typed coordinate + shared resolver (§10.b/c resolved)

- `LOCAL_CLUSTER = "local"` module constant + `is_local(cluster)` / `is_federated(cluster)` helpers,
  home in `cluster/types.py` (the identity module). Bare `str` value type (consistent with `peer_id`
  everywhere and the proto `string` fields); the helpers, not a `NewType` cast wall, are what collapse
  B1. `'local'` stays a compile-time constant so the static partial indexes hold.
- `resolve_finelog_cluster(c) = self.cluster_id if is_local(c) else c` is a **shared** helper (used by
  relay stamp + read-side), not finelog-local — `local→self` is an absolute-id boundary that recurs
  (finelog, config validation, cross-cluster addressing).

### 11.4 Expanded cleanup inventory (added to §7)

1. Truthiness readers (B1) — Python + Vue.
2. `insert_job` / `insert_job_and_config` default params (B2).
3. `or ""` proto-population fallbacks → `'local'` (B3): `service.py:292, 1593, 1705, 3099`.
4. Replay golden JSONs (~14, `"child_cluster": ""`) + `test_federation_fold_exclusion.py:242` assert → `'local'`.
5. `config.py` reservation of `'local'` (M4).
6. Delete `JobName.with_root_job` (`types.py:158`) — orphaned once `_rebase_*` go.
7. Stale docstrings: `controller.py:263`, `job.proto:280-284/393-398`, `types.py:91-95`.
8. Document the `CLUSTER_CONSTRAINT_KEY="cluster"` overlap (`constraints.py:1002`) — the submit-time
   `--cluster` *pin* is a third role of the word, distinct from the column and the query param.

## 12. Shared implementation contract (every agent conforms to this EXACTLY)

The cleanup is split across parallel agents; these are the pinned interfaces they meet at. The literal
`'local'` must be byte-identical everywhere (schema `server_default`, migration DDL, `local_tasks`
predicate, partial indexes, Python param defaults, `LOCAL_CLUSTER`).

**Cluster coordinate (already landed in `types.py`):**
- `iris.cluster.types.LOCAL_CLUSTER = "local"`; `is_local(c) -> c == LOCAL_CLUSTER`;
  `is_federated(c) -> c != LOCAL_CLUSTER`.
- DB column on `jobs`/`tasks` is **`cluster`**, `NOT NULL DEFAULT 'local'`. Row values are `'local'` or a
  peer id — **never** `""`. A *query filter* may be unset (= all clusters); that's a filter concept, not
  a stored value.

**Protos** (rename keeps the field number; retire deleted numbers with `reserved`):
- `job.proto`: `TaskStatus.child_cluster (25) → cluster (25)`, delete `remote_job_id (26)` + `reserved 26;`.
  `JobStatus.child_cluster (35) → cluster (35)`, delete `remote_job_id (36)` + `reserved 36;`. Refresh docstrings.
- `controller.proto`: `JobQuery.child_cluster (11) → cluster (11)`; `FederationJobDelta.remote_job_id (1) → job_id (1)`.
- Regen: `uv run python lib/iris/scripts/generate_protos.py`.
- `finelog/proto/logging.proto`: add `string cluster = 3;` to `PushLogsRequest` and `string cluster = 9;`
  to `FetchLogsRequest` (match the Rust proto numbers). Regen: `cd lib/finelog && buf generate`.

**Wire values:** `cluster` proto fields carry the literal row value (`'local'` or peer). Delete every
`… or ""` fallback at population sites — the wire says `'local'`, not `""`.

**Handoff:** `federation/store.py` `HandoffState`: `PENDING_HANDOFF=0`, `HANDED_OFF=1`, add `HANDOFF_REJECTED=2`.

**Logs (read rule — no self-id needed on the dashboard):** the cluster filter is
`isFederated(cluster) ? cluster : <unset>`. A federated job filters by its peer id (targets the peer's
relayed rows in the shared hub store); a local job sends **no** cluster filter (its key is
federation-unique, so an unfiltered read returns exactly its rows). This assumes the hub topology
(`system.log-server` points at the shared global finelog the peers relay to) — the maintainer confirmed it.

**Logs (write/relay side):** the finelog forwarder gains a `cluster` argument
(`push_batch(key, entries, *, cluster: str = "")`), and the **iris relay** (`finelog_relay.py`, backend
scope) stamps `cluster = self.cluster_id` (the controller's real id) when forwarding to a remote hub.
Local single-cluster pushes keep `cluster=""` (unchanged). `resolve_finelog_cluster` is therefore only a
*write-side* concern (stamp the real id); there is no read-side `local→self` resolution to build.

**Config:** `config.py` rejects a `cluster_id` (this controller's `name`) or any peer named `'local'` —
fail-fast, so the sentinel and the real cluster-id namespace stay disjoint.

### 11.5 Confirmed safe

Dropping the `attempt_uid` `~`-prefix (`writes.py:947`) is safe: local uids are random
`secrets.token_hex(8)`, the peer mints its own, the parent never runs a handed-off job locally, and a
job is handed to exactly one peer — so a peer's raw uid cannot collide in the parent's
`idx_task_attempts_uid`. Migrations `0034`–`0036` are rewritten in place (never deployed), collapsing
the `remote_job_id` create/rename dance since no live DB has the column.
