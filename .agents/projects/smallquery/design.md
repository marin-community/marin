# smallquery

_Why are we doing this? What's the benefit?_

We have large Parquet datasets on regional GCS and CoreWeave object storage, and no
self-serve way to run ad-hoc SQL over them — today that means hand-writing a zephyr pipeline or
pulling data into a notebook. smallquery gives "BigQuery on our own hardware": a user submits a
SQL query through an API or dashboard, it runs distributed across **spare CPU/RAM on
preemptible TPU VMs** (capacity we already pay for and largely waste), and the result lands in
object storage to stream or download. The win is turning idle cluster headroom into an
interactive analytics engine that sits next to the data instead of egressing it.

## Challenges

The hard part is that our cheapest compute is the most hostile substrate for a query engine.
Preemptible TPU VMs are **lost at any instant with no graceful drain signal** — Iris detects
loss only reactively, via the worker's Reconcile RPC going UNREACHABLE
(`lib/iris/docs/architecture.md`, `lib/iris/src/iris/scheduling/scheduler.py:61-137`). Their
local disk is tiny and slow, and the default worker scratch is `/dev/shm/iris` *tmpfs* —
RAM-backed (`lib/iris/src/iris/config.py:378`) — so any engine that **spills or shuffles
through local disk is disqualified**. And there is no node-level data-locality primitive in
Iris; "colocate compute with data" can only mean region/zone placement via the constraint
system (`lib/iris/src/iris/rpc/job.proto:480-518`), which is enough to avoid cross-region
egress but nothing finer.

Critically, **no OSS distributed engine survives both constraints as a turnkey system**
(`research.md` §4): upstream Ballista shuffles to executor local disk and restarts the whole job
on executor loss; smallpond assumes a fast shared filesystem (3FS); ClickHouse OSS has no true
shuffle and fails the query on any node loss; Polars' fault-tolerant distributed engine is
closed SaaS. But the *components* of one are reusable. DataFusion exposes a **typed logical plan**
(the join/group keys we cut stages on), **two-phase aggregation** (`AggregateMode::Partial` →
`FinalPartitioned`, correct distributed `GROUP BY`), reusable **physical operators** (joins,
aggregates, sorts), and **`object_store` GCS/S3 readers** with predicate/stats/bloom pushdown
(`research.md` §4a). So we don't build an engine — we reuse DataFusion's planner and operators and
supply the **one piece no OSS project ships open-source: object-store shuffle with worker-failure
tolerance** (it exists only in the closed-ish Spice AI Ballista fork). That, plus the coordinator
and `IrisDaemon` worker fleet, is what's actually ours.

## Costs / Risks

- **We build and own the coordinator, object-store shuffle, and preemption-recovery.** No
  vendor does this for us; that's real, load-bearing code.
- **A new long-lived service to operate** — a non-preemptible coordinator VM, much like finelog.
- **Hard dependency on a new Iris-core primitive (`IrisDaemon`).** smallquery has no worker fleet
  until that lands, coupling its schedule to Iris-core review/rollout
  (`.agents/projects/iris_daemon/`).
- **Result latency is gated by object storage**, not local disk — every shuffle/result write is
  a GCS/S3 round-trip. We trade latency for preemption-tolerance and disk-independence.
- **Partition sizing is a real constraint**: with no local disk to spill to, each partition's
  working set — and each shuffle reducer's — must fit in worker RAM. A bounded DataFusion memory pool
  makes an oversized partition fail **cleanly** (`ResourcesExhausted`) instead of OOM-killing the
  worker (`OOMKilled`, `lib/iris/src/iris/runtime/docker.py:764-772`); the fix is a larger fan-out N,
  not in-engine spill.
- **Shuffle in v0 puts the hardest path on the critical path from day one** — preemption recovery
  through a multi-stage object-store shuffle, not just stateless scans.
- **The DataFusion arm is Rust worker code** — a custom object-store `ExchangeExec`
  (`ExecutionPlan`); **no `PhysicalExtensionCodec`** is needed since plans aren't serialized across
  the wire — and the kernel bake-off (DataFusion vs DuckDB) is itself a cost: two prototypes before
  locking.
- **Coordinator is Python** (`datafusion-python`, verified 54.0.0), cutting stages from the typed
  *logical* plan; the Rust surface is confined to the workers. This pins coordinator + worker to
  **compatible `datafusion-proto` versions** (we ship logical sub-plans across the boundary).
  Tradeoff: logical-level cutting means the coordinator makes its own shuffle-placement decisions
  rather than reusing DataFusion's physical `EnforceDistribution` — a thin Rust introspection shim is
  the fallback if we ever need DataFusion's
  exact physical shuffle choices.
- **Coordinator keeps an embedded SQLite control DB, snapshotted to GCS** (the Iris-controller
  pattern, `controller/checkpoint.py`) — a small piece of state to operate; `report_done` drives it,
  and object-store `_SUCCESS` manifests are the data backstop, reconciled only on restore.
- **Object-store op cost is real**: a shuffle stage is ≈ M chunks + M sidecars written and ≈ M×N
  ranged reads, plus commit markers — and the project exists partly to avoid egress/op cost, so
  fan-out N and file sizing must be tuned, not left default.

## Design

**Shape: a persistent coordinator + a daemon-deployed worker fleet.** A non-preemptible
**coordinator** runs as a long-lived service (finelog's deployment shape: it
`RegisterEndpoint`s its query API + Vue dashboard and is reached through the Iris controller
proxy at `/proxy/<name>/`, `lib/iris/src/iris/controller/endpoint_service.py:83-132`,
`controller/endpoint_proxy.py`). The workers are deployed by **`IrisDaemon`** — a new Iris
primitive on its own design track (`.agents/projects/iris_daemon/`): "run exactly one worker on
every node of scale group S, for that node's lifetime." A controller-side reconciler pins one
daemon task per matching worker, each registering its endpoint, so newly-autoscaled capacity
**self-joins** the query fleet and preempted nodes drop out — **no pool to size or refill**.
Workers are **CPU-only, BATCH priority band**, requesting **tiny/zero `cpu_millicores`** (so the
scheduler packs them onto idle cores, `scheduler.py:271-276`) and a **bounded, hard `memory_bytes`
reservation** — and **no TPU device** (attaching one floors CPU to 4 cores,
`cluster/types.py:589-597`). Be honest about the memory: it is a **real reservation, not free
overflow** — smallquery's "spare capacity" comes from opportunistic CPU + BATCH preemptibility, not
from unreserved RAM (a worker that assumed leftover RAM would be OOM-killed under load, since memory
is a hard cgroup cap, `runtime/docker.py:684-686`). We size the per-worker RAM budget modestly and
treat it as committed (`IrisDaemon` `reserve_memory=true`). They stay warm and registered as
endpoints; the coordinator load-balances query-shard RPCs to the live set, not a task per query.

smallquery **depends on `IrisDaemon` landing** — the two ship together. The Iris primitive is
specified separately (`.agents/projects/iris_daemon/`) so Iris owners review it independently,
but smallquery has no worker fleet without it (no WorkerPool/per-query-task fallback — that
churn isn't worth carrying for a model we'd immediately replace).

**Coordinator: a thin reconciler.** It does two things and runs **no compute over query *data*** (all
bulk data lives in object storage); its only state is a small control DB (step 2). (1) **Plan once**: it uses **DataFusion's parser + optimizer**
(`SessionContext` → optimized `LogicalPlan`, no execution) to decompose the SQL into a **stage
DAG** — splitting at join / partitioned-aggregate boundaries keyed by the typed join/group keys
(rationale below) — and writes a **plan manifest** (stage DAG + per-partition shard specs) to
object storage. (Planning does a one-time prefix **LIST** + Parquet-**footer read** per table to
resolve schemas and pin generations — metadata I/O; the "no compute over data" / "never LIST"
guarantees below scope to the commit/recovery path.) The manifest **pins every input to an immutable
object generation**: on a **versioned** store (GCS generation / S3 versionId) retries fetch *by*
generation, byte-identical by construction; on an **unversioned** store (ETag only, e.g. plain
CoreWeave S3) it's GET-then-validate-ETag — detect-after-read (→ `INPUT_CHANGED` on mismatch),
best-effort, not byte-identical. (2) **Drive from a control
DB**: query state — per-stage/shard status, each shard's canonical committed attempt,
admission/quota, cancellation, failure counts, ownership — lives in an **embedded SQLite database**
on the coordinator, hot-backed-up to GCS exactly like the Iris controller
(`lib/iris/src/iris/cluster/controller/checkpoint.py`: SQLite backup-API → zstd →
`controller-state/{epoch_ms}/…`). In steady state the loop is driven by worker **`report_done` RPCs**
updating rows — no object-store polling. The durable commit barrier for a shard's *data* is its
**`_SUCCESS` manifest** in object storage, written after the outputs are durable and *before*
`report_done` — so `report_done` implies the data is committed, and downstream consumers read that
attempt's named objects. The coordinator runs **no compute over data** — even the final merge is a
worker shard. On **restart** (VM loss) it restores the latest SQLite snapshot and reconciles only
the gap since: for any shard the DB left uncertain it **probes `_SUCCESS` by key** (it knows the
attempt ids — never a LIST). The object-store manifests are the backstop, so the snapshot's RPO never
threatens *data* correctness — only a little control bookkeeping rolls back and is re-derived. (The
coordinator stays non-preemptible for endpoint stability.) The coordinator can be **Python**:
`datafusion-python` (verified, 54.0.0) plans without executing and exposes a **typed logical plan**
(`to_variant()` → `datafusion.expr.{Join,Aggregate,TableScan,…}` with join/group keys), rich enough
to derive the stage DAG. It ships **logical stage specs** (a stage's datafusion-proto *logical*
sub-plan + shuffle bindings), **not physical plans** — so there's **no `PhysicalExtensionCodec` and
no custom-operator serialization** (shuffle-input leaves are plain `TableScan`s the worker resolves
by name). The logical plan *is* `datafusion-proto`, so coordinator and worker must run **compatible
`datafusion-proto` versions** — a far milder constraint than serializing physical plans. The Rust
surface is confined to the **workers**, which build each stage's physical plan and host the
`ExchangeExec`. (Physical-plan introspection from Python is display-string-only, so
we cut stages from the *logical* plan — fine, since shuffle boundaries are determined by join/group
keys, which are typed there.)

**Shared substrate (both engine arms).** A query is a **DAG of stages** separated by hash-shuffle
boundaries. A stage is N independent **shards** (one per partition); a shard is a **pure
function of (immutable inputs) → (committed Parquet outputs)** in a query-scoped, TTL'd object-store
prefix. The **exchange between stages is object storage, with a defined shuffle format** (not raw
"range-GET a Parquet file," which Parquet doesn't support): each map shard writes **one Parquet
chunk holding one row-group per reducer**, plus a tiny **sidecar index** (reducer →
row-group + byte range) — zephyr's scatter-chunk idea in Parquet row groups. A reducer reads
**only its row groups** across the M upstream chunks: **M ranged reads + M sidecar reads, not M×N
objects**. Each attempt writes to an attempt-unique prefix and **seals with a `_SUCCESS` manifest**
(see Preemption recovery). This is our object-store `ExchangeExec` (Arm A) / scatter writer (Arm B);
**never local disk**. The coordinator
dispatches each shard as a **logical stage spec + input/output URIs + input generations**; the
worker builds the physical plan for its stage (Arm A: DataFusion + `ExchangeExec`; Arm B: DuckDB
SQL). Shipping logical specs — not serialized physical plans — keeps the coordinator engine-light
and avoids *physical*-plan-format skew across a fleet roll (the logical-proto compat requirement is
much milder).

**Engine: a bake-off between two arms over that substrate.**

- _Arm A — DataFusion-native (recommended)._ **Reuses DataFusion wholesale**, split across
  coordinator and worker. The coordinator (Python) decomposes the optimized **logical** plan into a
  stage DAG — splitting each join / partitioned aggregate at its key (a `Partial` aggregate in the
  upstream stage, `FinalPartitioned` after the shuffle) — and ships each stage as a logical spec.
  Each **worker** (Rust) builds *that stage's* **physical** plan: DataFusion does the `object_store`
  Parquet scan with predicate/stats/bloom pushdown, the local `HashJoinExec` / `AggregateExec`, and
  our **`ExchangeExec`** for the cross-worker hash exchange, run via `ExecutionPlan::execute`. The
  **only genuinely new engine code** is that custom object-store `ExchangeExec` (an `ExecutionPlan`
  that writes hash partitions to object storage instead of in-memory/local-disk) and a matching
  **`TableProvider`** that reads a stage's shuffle input back. Everything else — parse, optimize,
  scan, join, aggregate — is reused (`research.md` §4a).
- _Arm B — DuckDB-bolted (alternative)._ DuckDB per partition; the coordinator hand-builds the
  stage DAG, writes the partial-aggregate SQL itself, and drives the same object-store shuffle.
  Trivial warm embed and the richest SQL dialect, but **no serializable plan or built-in
  two-phase aggregation** — partial-aggregate correctness and shuffle planning are entirely the
  coordinator's burden.

**The bake-off is a throwaway spike to pick *one* engine, not two production stacks.** Arm A and
Arm B do not cleanly "swap" — Arm B needs its own planner, partial-aggregate SQL rewriting, and
scatter writer, and its correctness surface (`DISTINCT`, percentiles, UDAFs, windows — none of
which decompose into a trivial two-phase split) is far larger. So we prototype both **only far
enough to measure per-node engine throughput** on a real join + high-card-`GROUP BY` slice, then
**commit to one** and build the distributed layer once. The recommendation leans Arm A: the
coordinator already uses DataFusion as its planner, so Arm A is the straight-through path (Arm B
would re-emit per-shard SQL), and two-phase aggregation comes from the engine rather than
hand-written SQL. We measure first, then delete the loser.

**Query flow (v0, multi-stage with shuffle):**
1. Coordinator plans SQL → stage DAG + manifest (above), resolving tables to Parquet prefixes.
   Leaf stages are partitioned by **footer-only row-group splitting** — group row groups into
   ~RAM-sized partitions without dividing one (DataFusion's Parquet file-scan already assigns row
   groups to splits; a small helper if we need finer control).
2. **Leaf stage** (scan/filter/partial-agg, or a join's build/probe side): each shard reads its
   Parquet partition (predicate/projection/stats/bloom pushdown), computes, and writes its
   **per-reducer row-groups + sidecar** to an attempt-unique prefix, sealing with a `_SUCCESS`
   manifest (below).
3. **Shuffle stage(s)**: each reducer reads its row groups across the upstream files and runs the
   `HashJoinExec` / `FinalPartitioned` aggregate, writing the next stage's partitions the same way.
4. **Final stage** writes the result as **(possibly many) Parquet parts** — final-stage shards are
   att-prefixed and seal like any other stage; a global `ORDER BY` uses a **range-partitioned**
   final stage (sample → range boundaries → per-range sorted part), *not* a single-worker merge, so
   large/sorted results don't bottleneck one worker. The coordinator then **composes the top-level
   result `_SUCCESS`** from the N committed per-shard manifests (naming the real att-prefixed part
   URIs — control-plane metadata, not data compute) and returns `{result_prefix, schema, row_count}`
   + a **capped inline Arrow preview** (finelog caps a response at 64 MiB,
   `lib/finelog/rust/src/server/mod.rs:34`). The user streams/downloads the parts; the
   **query-scoped** TTL reclaims the scratch.

**Preemption recovery = stage-granular lineage materialized through object storage.** Every
shard's output is durable Parquet, so **completed shards are never recomputed**; an
in-flight shard on a preempted worker left **no sealed attempt**, so the reconciler
**re-dispatches it**. We commit with a **per-attempt SUCCESS manifest** — the cloud-native
"manifest committer" pattern, needing **neither atomic rename nor conditional-create**: each
attempt writes its outputs under an **attempt-unique prefix** (`shard-7/att-<id>/…`, where `<id>` is
a **random UUID** — never a snapshot-derived counter, so a post-restore re-dispatch can't reuse a
prefix), then, once those objects are durable, writes a small **`_SUCCESS` manifest in its own
prefix** enumerating its exact output objects (+ row count, plan hash). Because attempts never share
an object key, concurrent attempts can't interleave files and the marker write has **no overwrite
race**. A shard is **done when its attempt is sealed** (its `_SUCCESS` manifest is written);
`report_done` (`ReportShard`) signals this in steady state, and on restart the coordinator probes
`_SUCCESS` **by key** (it knows the attempt ids — never a LIST). It records the **canonical
committed attempt** — **write-once**: the first COMMITTED wins (conditional DB update), later commits
are answered `superseded` and ignored — so every downstream consumer (and its retries) reads the
**same** attempt's **named objects**, never a mix.
Losing/orphan attempts are reclaimed by the query-scoped TTL. A *deterministically* failing shard
is bounded by a tiered retry ceiling in the control DB (deterministic vs infra — the idea from
zephyr). This needs only **read-after-write of named objects** (universal on GCS and
modern S3; the weaker requirement is easy to validate on CoreWeave) — it drops the conditional-create
and strong-LIST dependencies entirely. Correct under arbitrary worker loss — exactly why we **must
not** adopt stock Ballista, whose local-disk shuffle loses intermediates and restarts the whole job
on executor loss (`research.md` §4).

**Skew and memory limits.** Reducers must fit their partition in RAM (no local disk to spill to),
and a single hot key can't be split by any fan-out. We keep this simple — **fail cheaply, don't
mitigate.** The coordinator picks hash fan-out **N** from input-size stats so partitions target the
worker RAM budget (best-effort sizing, also bounding object/row-group count), and each worker runs
its DataFusion plan under a **bounded memory pool** so an oversized partition fails **early and
cleanly** with `ResourcesExhausted` / `partition too large` — caught by memory accounting, not a
thrash-then-OOM (finelog's `GreedyMemoryPool` pattern, `lib/finelog/rust/src/query/mod.rs:95-117`).
The cheap "fix" is **re-running with a larger N** (more, smaller partitions); a genuinely skewed
single key still fails, and that's acceptable. We do **not** build hot-key salting or object-store
spill in v0 — they're real work; revisit only if skew proves common *and* a mitigation turns out to
be easy.

**Concurrency, admission, cancellation.** The fleet is shared and worker RAM is a hard cap, so
**admission control is mandatory**: each query declares a per-worker memory budget; the coordinator
admits it only when the fleet can seat its shards within budget, queuing otherwise. Shards
from different queries co-resident on a worker run under separate budgets summing to ≤ the worker
reservation. **Cancellation** is first-class: the coordinator marks the query cancelled in the
control DB and pushes `CancelQuery` to the fleet; workers **cooperatively abort** the in-flight shard
(cancel the running `ExecutionPlan` stream), retries are suppressed, and the query-scoped prefix is
deleted.

**Security.** Workers hold the cluster's ambient object-store credentials, so **raw `gs://`/`s3://`
globs in user SQL are not allowed** — datasets are named through a **catalog of allow-listed
prefixes** (a finelog-style namespace registry, `lib/finelog/rust/proto/finelog_stats.proto:21-57`),
and the coordinator authorizes the query against the caller before planning. Coordinator↔worker RPC
is authenticated (Iris endpoint auth / mTLS); plan size and operator set are bounded.
**Cross-location queries are rejected by default** — keyed on a table's catalog `location`
(cloud + region), so any two tables in different regions (even two GCS regions, not just GCS↔CoreWeave)
trip it; cross-region egress is a top cost driver per `AGENTS.md`. An explicit `allow_cross_location`
opt-in with a cost warning is the only path.

**Reuse boundary:** new `lib/smallquery` package, with **no code dependency on zephyr** (which
also respects the `iris/haliax → levanter/zephyr → marin` layering — smallquery doesn't sit under
zephyr). It reuses **DataFusion core** for SQL parsing, the logical-plan API (coordinator) and
physical operators + aggregation + Parquet/`object_store` reads (workers), follows **finelog's**
deployment / dashboard / endpoint-proxy patterns, and reuses the **Iris controller's**
SQLite-checkpoint-to-object-storage pattern (`controller/checkpoint.py`) for the control DB. zephyr
is a **reference design** for object-store shuffle and tiered
failure accounting — we borrow the ideas, not the modules (its scatter is Python pickle/Parquet;
ours is a DataFusion `ExchangeExec`). Genuinely new code: the SQL service, the
coordinator/reconciler, and the object-store `ExchangeExec` + `TableProvider`.

## Testing

- **Unit**: stage-DAG planning (partition counts, where shuffle boundaries land) and
  footer-based partition planning. Pure functions; test them as such.
- **Correctness oracle**: for a corpus of queries over fixture Parquet — including **joins and
  high-cardinality `GROUP BY` that force a shuffle** — assert smallquery's result equals
  single-process DuckDB over the same data. This is the primary guard against silent wrong
  answers from partial/`FinalPartitioned` aggregation and shuffle mis-routing, and it runs
  against **both engine arms** during the bake-off.
- **Preemption through shuffle**: an integration test on an Iris dev cluster that **kills workers
  mid-query** (`iris job kick --state preempted`, `lib/iris/src/iris/cli/job.py:1045-1080`),
  specifically during a shuffle stage, and asserts the query still returns the correct result —
  exercising shard re-dispatch, durable-intermediate reuse, and fleet re-convergence.
- **No-local-disk invariant**: assert workers never write shuffle/spill to local disk, and a
  too-large partition fails **cleanly and early** with `ResourcesExhausted` / `partition too large`
  (bounded memory pool), never an OOM-kill or a silent local-disk lean.
- **Commit protocol & consistency**: on **both** GCS and CoreWeave S3, validate **read-after-write
  of named objects**; drive concurrent duplicate attempts of one shard and assert a consumer
  reads **exactly one sealed attempt** (no file interleaving across attempts) and orphan attempts
  are ignored and TTL-reclaimed.
- **Object-store reach**: smoke both backends — regional GCS (workload identity) and CoreWeave
  S3 (`AWS_*` env), per finelog's two-backend credential story (`_k8s.py:70-114`).

## Open Questions

- **Engine bake-off criteria.** Arm A (DataFusion-native) vs Arm B (DuckDB-bolted) — what
  thresholds decide it? Shuffle throughput on a real join/high-card-`GROUP BY` workload, SQL
  coverage, and the Rust-vs-Python maintenance cost. The recommendation leans Arm A (reuse +
  finelog synergy); reviewers should weigh whether the Rust engine work is worth it over a
  DuckDB-bolted coordinator.
- **`IrisDaemon` coupling.** The worker fleet depends on the new `IrisDaemon` primitive
  (`.agents/projects/iris_daemon/`). Do we gate smallquery's first milestone on it landing, or
  develop the coordinator/query path against a local fake fleet and integrate last? (The
  resource-accounting policy is settled on smallquery's side — workers take a real bounded RAM
  reservation, `reserve_memory=true` — but the primitive's general policy is its own open question.)
- **Fan-out N sizing.** The policy for choosing hash fan-out N from input-size stats + fleet size —
  wrong values starve parallelism or explode object/row-group count. (Skew mitigation beyond clean
  failure + larger-N retry is deliberately out of scope for v0; revisit only if real workloads make
  it both necessary and easy.)
- **CoreWeave read-after-write + versioning.** The SUCCESS-manifest committer needs only
  read-after-write of named objects (no conditional-create, no strong LIST) — universal on GCS and
  modern S3, but validate it on CoreWeave (Testing). Also confirm whether CoreWeave is **versioned**
  (pinned-GET immutability) or **ETag-only** (detect-after-read) — the latter weakens the input-pin
  to best-effort `INPUT_CHANGED`.
