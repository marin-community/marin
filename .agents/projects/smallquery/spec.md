# smallquery — spec

Contracts for `lib/smallquery`. Pins the object-store layout, the persisted/wire proto messages,
the RPC services, the Python client API, the SQLite control-DB schema, the catalog, the two new
Rust operator contracts, and the error kinds. Implementation (algorithms, file-by-file plan) is out
of scope — this is the surface reviewers agree to.

Language split (from `design.md`): **coordinator = Python** (`datafusion-python`), **workers =
Rust** (DataFusion + the two custom operators). Protos are the cross-language contract.

## 1. Object-store layout

Everything for one query lives under a **query-scoped scratch prefix**, reclaimed by a **bucket
lifecycle TTL** (no per-object cleanup needed). Results live under a separate, non-TTL'd
`result_prefix`.

```
{scratch_prefix}/{query_id}/
  plan.pb                                   # PlanManifest (§2)
  stage-{s}/shard-{p}/att-{attempt_id}/     # one shard's output, per attempt
      chunk.parquet                         # one row-group per reducer
      chunk.idx                             # ShuffleSidecar (§2), reducer → row-groups + byte range
      _SUCCESS                              # SuccessManifest (§2) — the commit barrier

{result_prefix}/{query_id}/
  shard-{p}/att-{attempt_id}/part-0.parquet # final-stage shard output — att-prefixed like any stage
  _SUCCESS                                  # written by the COORDINATOR: a SuccessManifest enumerating
                                            # the committed att-prefixed result parts across all final shards
```

`chunk.parquet` is the **shuffle chunk** a map shard writes (one row-group per reducer — zephyr's
scatter-chunk, in Parquet); `chunk.idx` maps reducer → row-group + byte range.

Rules:
- An attempt **only ever writes under its own `att-{attempt_id}/` prefix** — attempts never share an
  object key, so concurrent/duplicate attempts cannot interleave files. This holds for the **final
  stage too** (its shards are att-prefixed; the flat result layout of earlier drafts is gone).
- `attempt_id` is a **random UUID**, never derived from control-DB-resident counters — so a
  post-restore re-dispatch (after the RPO gap, §5/design) can never reuse a prior attempt's prefix.
- `_SUCCESS` is written **last**, after `chunk.parquet`+`chunk.idx` are durable. Its presence (read
  by key) is the sole "this attempt is complete and readable" signal. The **result-level** `_SUCCESS`
  is composed by the **coordinator** from the N committed final-shard `SuccessManifest`s (control-plane
  metadata, not data compute), naming the real att-prefixed part URIs.
- A reducer `r` in stage `s+1` reads, for each committed shard of stage `s`, the row groups named by
  that shard's `chunk.idx[r]` from its `chunk.parquet`.
- Commit ordering relies only on **read-after-write of named objects** (no atomic rename, no
  conditional-create, no strong LIST).

## 2. Persisted / wire protos

`lib/smallquery/proto/smallquery.proto`, package `smallquery.v1`, proto3. Persisted manifests are
stored as **binary proto** at the paths in §1.

```proto
// ---- Plan manifest (object storage: plan.pb) ----
message PlanManifest {
  string query_id        = 1;
  string sql             = 2;
  uint64 created_unix_ms = 3;
  string plan_hash       = 4;   // hash of (sql + resolved inputs + stage DAG); stamped into every
                                // SuccessManifest. Guards a stale-snapshot/re-plan mismatch — a
                                // restored coordinator that re-plans rejects chunks whose plan_hash differs.
  repeated InputTable inputs  = 5;  // resolved tables, generation-pinned
  repeated StagePlan  stages  = 6;  // topological order; last stage sinks the result
  string result_prefix   = 7;
}

message InputTable { string name = 1; repeated PinnedObject objects = 2; }
message PinnedObject { string uri = 1; string generation = 2; int64 size_bytes = 3; }
// generation pins immutability. Two regimes (see M7 / design): VERSIONED store (GCS generation,
// S3 versionId) → fetch BY generation, byte-identical by construction. UNVERSIONED store (ETag only,
// e.g. plain CoreWeave S3) → GET then validate ETag; mismatch → INPUT_CHANGED. Detect-after-read,
// best-effort — NOT byte-identical-by-construction. Empty generation ⇒ unversioned/ETag mode.

message StagePlan {
  uint32 stage_id        = 1;
  uint32 num_shards      = 2;             // this stage's parallelism (number of shards). NOT the
                                          // output fan-out — that is ShuffleSink.num_reducers, independent.
  repeated uint32 input_stage_ids = 3;    // upstream stages feeding this one via shuffle
  EnginePlan plan        = 4;
  StageOutput output     = 5;
}

message EnginePlan {
  oneof body {
    bytes  datafusion_logical = 1;        // datafusion-proto *LOGICAL* plan bytes (Arm A). NOT a
                                          // physical plan — no PhysicalExtensionCodec. Shuffle-input
                                          // leaves are plain TableScans resolved by name (§3), so no
                                          // custom logical codec either. Requires coordinator (datafusion-python)
                                          // and worker (datafusion) on compatible datafusion-proto versions.
    string duckdb_sql         = 2;        // per-shard SQL (Arm B)
  }
  AggregateMode aggregate_mode      = 3;  // distributed-aggregate decomposition for this stage
  repeated string shuffle_key_columns = 4;// hash key for THIS stage's output partitioning.
                                          // EMPTY ⇒ single reducer (partition 0); ShuffleSink.num_reducers must be 1.
}
// AGG_FINAL_PARTITIONED == DataFusion FinalPartitioned (post-hash-shuffle), NEVER Final.
// AGG_SINGLE == fused single-stage aggregate (no shuffle).
enum AggregateMode { AGG_NONE = 0; AGG_PARTIAL = 1; AGG_FINAL_PARTITIONED = 2; AGG_SINGLE = 3; }

message StageOutput {
  oneof sink {
    ShuffleSink shuffle = 1;
    ResultSink  result  = 2;
  }
}
message ShuffleSink {
  uint32 num_reducers      = 1;   // == downstream stage's num_shards (1 if shuffle_key_columns empty)
  bytes  shuffle_schema_ipc = 2;  // Arrow schema (IPC) of the chunk payload — the partial-aggregate
                                  // STATE schema for AGG_PARTIAL, else the row schema. Lets a reducer
                                  // synthesize an empty stream and validate FinalPartitioned input
                                  // without a footer read (needed for the empty-partition case, §7).
}
message ResultSink  { string result_prefix = 1; }

// ---- Shuffle sidecar (object storage: chunk.idx) ----
message ShuffleSidecar { repeated ReducerSlice reducers = 1; }  // index == reducer id
message ReducerSlice {
  repeated uint32 row_groups = 1;   // row-group indices in chunk.parquet for this reducer
  uint64 byte_offset = 2;           // start of this reducer's row-group span (ranged GET)
  uint64 byte_len    = 3;
  uint64 num_rows    = 4;
}

// ---- Commit barrier (object storage: _SUCCESS) ----
message SuccessManifest {
  string query_id   = 1;
  uint32 stage_id   = 2;
  uint32 shard_id = 3;          // the map output partition that produced this attempt
  string attempt_id = 4;
  string plan_hash  = 5;            // must match PlanManifest.plan_hash
  repeated OutputObject objects = 6;// exact objects this attempt committed
  uint64 num_rows   = 7;
}
message OutputObject { string uri = 1; int64 size_bytes = 2; ObjectRole role = 3; }
enum ObjectRole { ROLE_SHUFFLE_CHUNK = 0; ROLE_SHUFFLE_SIDECAR = 1; ROLE_RESULT_PART = 2; }
```

## 3. RPC services

Connect-RPC over HTTP (Iris/finelog convention). Three services across two language boundaries.

```proto
// ---- User → coordinator (Python server) ----
service QueryService {
  rpc SubmitQuery(SubmitQueryRequest)             returns (SubmitQueryResponse);
  rpc GetQuery(GetQueryRequest)                   returns (QueryStatus);
  rpc CancelQuery(CancelQueryRequest)             returns (CancelQueryAck);
  rpc GetResultPreview(GetResultPreviewRequest)   returns (ResultPreview);
}

message SubmitQueryRequest { string sql = 1; QueryOptions options = 2; string user = 3; }
message QueryOptions {
  uint64 per_worker_memory_bytes = 1;  // admission budget; 0 = service default
  bool   allow_cross_location    = 2;  // default false → cross-region/cross-cloud queries rejected
  uint32 fanout_override         = 3;  // 0 = auto-size N
}
message SubmitQueryResponse { string query_id = 1; }

message GetQueryRequest { string query_id = 1; }
message QueryStatus {
  string query_id = 1;
  State  state    = 2;
  uint32 stages_total = 3;  uint32 stages_done = 4;
  uint32 shards_total = 5;  uint32 shards_done = 6;
  ResultRef result = 7;     // set iff state == SUCCEEDED
  string error_kind = 8;    // ErrorKind name, set iff state == FAILED
  string error_detail = 9;
}
enum State { QUEUED = 0; PLANNING = 1; RUNNING = 2; SUCCEEDED = 3; FAILED = 4; CANCELLED = 5; }
message ResultRef { string result_prefix = 1; bytes arrow_schema_ipc = 2; uint64 num_rows = 3; }

message CancelQueryRequest { string query_id = 1; }
message CancelQueryAck {}

message GetResultPreviewRequest { string query_id = 1; uint32 max_rows = 2; }
message ResultPreview { bytes arrow_ipc = 1; bool truncated = 2; }  // ≤ 64 MiB (finelog cap)

// ---- Coordinator → worker (Rust server on each IrisDaemon worker) ----
service WorkerService {
  rpc RunShard(RunShardRequest) returns (RunShardAck);  // async; result via ReportShard
  rpc CancelQuery(CancelQueryRequest) returns (CancelQueryAck);
  rpc Health(HealthRequest)           returns (HealthResponse);  // load for placement
}
message RunShardRequest {
  string query_id   = 1;
  uint32 stage_id   = 2;
  uint32 shard_id   = 3;             // this shard's index within the stage; == reducer_id for its inputs
  string attempt_id = 4;             // coordinator-assigned random UUID (never a snapshot-derived counter, §1)
  StagePlan stage   = 5;             // inlined (small); EnginePlan body + output sink
  repeated ShuffleInput inputs = 6;  // resolved upstream committed chunks for THIS shard (reducer = shard_id)
  string output_prefix = 7;          // {scratch}/{query}/stage-{s}/shard-{p}/att-{att}/
  uint64 memory_budget_bytes = 8;    // bounded DataFusion memory pool
}
message ShuffleInput {
  uint32 input_stage_id  = 1;
  string scan_table_name = 2;          // the TableScan leaf name in EnginePlan this input binds to;
                                       // the worker registers ShuffleReader under this name before planning
  bytes  shuffle_schema_ipc = 3;       // copied from the upstream ShuffleSink.shuffle_schema_ipc
  repeated ShuffleSource sources = 4;  // committed upstream chunks to read for reducer = shard_id
}
message ShuffleSource { string chunk_uri = 1; string sidecar_uri = 2; }
message RunShardAck { bool accepted = 1; string reject_reason = 2; }  // accepted=false = LOCAL backpressure
                                                                       // (worker full), NOT a shard failure
message HealthRequest {}
message HealthResponse { uint64 free_memory_bytes = 1; uint32 running_shards = 2; }

// ---- Worker → coordinator (Python server) ----
service CoordinatorInternalService {
  rpc ReportShard(ShardReport) returns (ReportAck);
}
message ShardReport {
  string query_id = 1; uint32 stage_id = 2; uint32 shard_id = 3; string attempt_id = 4;
  ShardStatus status = 5;
  string success_manifest_uri = 6;   // set iff COMMITTED (the _SUCCESS this attempt wrote)
  string error_kind = 7;             // ErrorKind name, set iff FAILED
  string error_detail = 8;
  PartitionStats stats = 9;
}
enum ShardStatus { SHARD_COMMITTED = 0; SHARD_FAILED = 1; }
message PartitionStats { uint64 num_rows = 1; uint64 num_bytes = 2; }
message ReportAck { bool superseded = 1; }  // true → this attempt is stale; worker discards output
```

Contract notes:
- A worker writes outputs → `_SUCCESS` → **then** `ReportShard(COMMITTED)`. So a COMMITTED report
  implies the data is durable and readable by key.
- **Worker discovery**: the coordinator enumerates the live fleet via the Iris `EndpointService`
  lease registry (`IrisDaemon` workers self-register, design §Shape); it pushes `RunShard` to a
  chosen worker. A `RunShardAck.accepted=false` is **local backpressure** — re-place on another
  worker; it does **not** increment `task_failures`/`infra_failures`.
- **Canonical attempt is write-once.** The first `ReportShard(COMMITTED)` sets
  `shards.committed_attempt`/`success_uri` via a conditional `UPDATE … WHERE committed_attempt IS
  NULL`; it is **immutable** for the life of the shard. Every later COMMITTED for an
  already-committed shard is answered `superseded=true` and ignored — so a downstream reducer (and
  its retries) always reads the *same* attempt's chunks, never a mix.
- The worker registers `ShuffleReader(reducer_id = shard_id, sources, schema = shuffle_schema_ipc)`
  under each `ShuffleInput.scan_table_name` **before** deserializing/building the physical plan.
- `ReportShard` is a **latency hint**: the coordinator may already know via restart-probe, and a
  lost report just means the shard is re-dispatched (the new attempt's distinct UUID `att-` prefix
  makes that safe).
- `CancelQuery` to a worker **cooperatively aborts** any in-flight `RunShard` for that `query_id`
  (cancels the running `ExecutionPlan` stream) and suppresses new dispatch — it is a push, not a poll.
- `superseded=true` tells a straggler its attempt is no longer canonical; it stops (orphan reaped
  by TTL).

## 4. Python client API

`lib/smallquery/src/smallquery/client.py`.

```python
class SmallQueryClient:
    """Client for the smallquery coordinator (reached via the Iris endpoint proxy)."""
    def __init__(self, endpoint: str, *, credentials: Credentials | None = None) -> None: ...

    def submit(
        self, sql: str, *,
        per_worker_memory_bytes: int | None = None,
        allow_cross_location: bool = False,
        fanout: int | None = None,
        user: str | None = None,
    ) -> "QueryHandle":
        """Submit a query and return immediately with a handle. Does not block on execution."""

    def query(self, sql: str, **opts) -> "pa.Table":
        """Blocking convenience: submit, wait, and return the full result as an Arrow table.
        Reads all result Parquet parts. Raises QueryError on failure."""

class QueryHandle:
    query_id: str
    def status(self) -> QueryStatus: ...
    def wait(self, timeout: float | None = None) -> QueryStatus:
        """Block until terminal (SUCCEEDED/FAILED/CANCELLED) or timeout. Raises QueryError on FAILED."""
    def preview(self, max_rows: int = 100_000) -> "pa.Table":
        """Capped inline preview (≤ 64 MiB / max_rows). For the full result use to_table/to_reader."""
    def result_parts(self) -> list[str]:
        """URIs of the committed (att-prefixed) result Parquet parts, read from the
        coordinator-composed result `_SUCCESS` manifest (after SUCCEEDED)."""
    def to_table(self) -> "pa.Table": ...
    def to_reader(self) -> "pa.RecordBatchReader":
        """Stream the full result without materializing it in memory."""
    def cancel(self) -> None: ...
```

`QueryError(Exception)` carries `.kind: ErrorKind` and `.detail: str`.

## 5. Control-DB schema (SQLite)

`lib/smallquery/src/smallquery/control_db.py` — embedded SQLite, WAL, single writer (Iris
`controller/db.py` pattern), snapshotted to object storage via the Iris `checkpoint.py` pattern.
This is the **only** authoritative control-plane state.

```sql
CREATE TABLE queries (
  query_id      TEXT PRIMARY KEY,
  sql           TEXT NOT NULL,
  state         TEXT NOT NULL,        -- State enum name
  user          TEXT,
  plan_hash     TEXT,
  manifest_uri  TEXT,                 -- plan.pb
  result_prefix TEXT,
  per_worker_memory_bytes INTEGER,
  allow_cross_location INTEGER NOT NULL DEFAULT 0,
  error_kind    TEXT,
  error_detail  TEXT,
  created_ms    INTEGER NOT NULL,
  updated_ms    INTEGER NOT NULL
);

CREATE TABLE stages (
  query_id       TEXT NOT NULL,
  stage_id       INTEGER NOT NULL,
  num_shards INTEGER NOT NULL,
  state          TEXT NOT NULL,       -- PENDING | RUNNING | COMPLETE
  PRIMARY KEY (query_id, stage_id)
);

CREATE TABLE shards (
  query_id          TEXT NOT NULL,
  stage_id          INTEGER NOT NULL,
  shard_id      INTEGER NOT NULL,
  state             TEXT NOT NULL,    -- PENDING | DISPATCHED | COMMITTED | FAILED
  committed_attempt TEXT,             -- attempt_id of the canonical sealed attempt; WRITE-ONCE
  success_uri       TEXT,             -- its _SUCCESS manifest (set with committed_attempt, immutable)
  task_failures     INTEGER NOT NULL DEFAULT 0,  -- deterministic-error tier
  infra_failures    INTEGER NOT NULL DEFAULT 0,  -- preemption/infra tier
  PRIMARY KEY (query_id, stage_id, shard_id)
);

CREATE TABLE attempts (
  query_id     TEXT NOT NULL,
  stage_id     INTEGER NOT NULL,
  shard_id INTEGER NOT NULL,
  attempt_id   TEXT NOT NULL,
  worker       TEXT,                  -- IrisDaemon worker endpoint
  state        TEXT NOT NULL,         -- DISPATCHED | COMMITTED | FAILED | ORPHANED
  started_ms   INTEGER,
  ended_ms     INTEGER,
  PRIMARY KEY (query_id, stage_id, shard_id, attempt_id)
);
```

Failure ceilings (zephyr-style tiers): a shard fails the query at `task_failures >=
MAX_TASK_FAILURES` (deterministic, default 3); `infra_failures` (preemption) is retried up to
`MAX_INFRA_FAILURES` (default 20) and does **not** count toward the query-fail tier. The
ErrorKind→tier mapping is in §8.

**Stage barrier (v0)**: stage `s+1` shard `p` is **dispatchable iff every shard of every
`input_stage_id` is COMMITTED** (full barrier — no pipelining in v0). `stages.state = COMPLETE` ⇔
all of its shards are COMMITTED; a stage is RUNNING once any shard is DISPATCHED.

## 6. Catalog

Datasets are named, never raw `gs://`/`s3://` globs in SQL. Registry in the control DB:

```sql
CREATE TABLE catalog (
  name          TEXT PRIMARY KEY,     -- table name in SQL, e.g. "datakit.scores"
  uri_prefix    TEXT NOT NULL,        -- gs://… or s3://… prefix/glob of Parquet
  cloud         TEXT NOT NULL,        -- "gcs" | "cw"  (selects credentials/backend)
  location      TEXT NOT NULL,        -- cloud+region, e.g. "gcs:us-central1", "cw:us-east"
  allowed_users TEXT                  -- comma list; NULL = all authenticated users
);
```

A query referencing tables spanning more than one distinct **`location`** is **rejected**
(`CROSS_LOCATION_REJECTED`) unless `allow_cross_location=true`. Keying on `location` (not `cloud`)
is deliberate: two GCS buckets in different regions incur the same cross-region egress the rule
exists to prevent.

## 7. New Rust operator contracts

`lib/smallquery/rust/src/` — the only genuinely new engine code (Arm A).

```rust
/// Physical operator: hash-partitions its input by `hash_keys % num_reducers` and writes ONE
/// Parquet **chunk** (one row-group per reducer) + a ShuffleSidecar + a _SUCCESS manifest under
/// `output_prefix`, on object storage. Never touches local disk.
pub struct ObjectStoreExchangeExec {
    input: Arc<dyn ExecutionPlan>,
    hash_keys: Vec<PhysicalExprRef>,  // EMPTY ⇒ all rows → reducer 0; num_reducers must be 1
    num_reducers: usize,
    output_prefix: ObjectStorePath,   // .../stage-{s}/shard-{p}/att-{att}/
    store: Arc<dyn ObjectStore>,
}
impl ExecutionPlan for ObjectStoreExchangeExec { /* execute() streams input → buffered per-reducer
    row groups → multipart upload → sidecar → _SUCCESS; output stream yields PartitionStats. */ }

/// TableProvider: presents the shuffle input for ONE reducer, reading only that
/// reducer's row groups across the committed upstream chunks (via each sidecar).
pub struct ShuffleReader {
    reducer_id: usize,                // == RunShardRequest.shard_id
    sources: Vec<ShuffleSource>,      // committed (chunk_uri, sidecar_uri) per upstream shard
    schema: SchemaRef,                // from ShuffleInput.shuffle_schema_ipc — NOT read from a footer,
                                      // so an all-empty reducer still yields a correctly-typed empty stream
    store: Arc<dyn ObjectStore>,
}
#[async_trait] impl TableProvider for ShuffleReader {
    async fn scan(&self, _: &dyn Session, projection: Option<&Vec<usize>>,
                  _: &[Expr], _: Option<usize>) -> Result<Arc<dyn ExecutionPlan>>;
}
```

Empty-partition rule (from Spice's lesson): a missing `chunk.parquet` for a reducer whose sidecar
slice says 0 rows is an **empty** input; a missing object for a slice that should have rows is a
hard error (`SHUFFLE_FETCH_FAILED`), never a silent drop.

## 8. Error kinds

`ErrorKind` (string in protos / DB):

| Kind | Trigger | Tier → behavior |
|---|---|---|
| `PLAN_ERROR` | invalid/unsupported SQL at planning | terminal (no retry) |
| `TABLE_NOT_FOUND` | name not in catalog / user not allowed | terminal |
| `CROSS_LOCATION_REJECTED` | tables span >1 `location`, `allow_cross_location=false` | terminal |
| `ADMISSION_REJECTED` | can't seat within budget after queue timeout | terminal |
| `PARTITION_TOO_LARGE` | bounded memory pool `ResourcesExhausted` | **deterministic, ceiling 1** → fail fast (re-running same N fails identically; user retries with larger N) |
| `INPUT_CHANGED` | pinned generation/ETag no longer matches | terminal |
| `SHUFFLE_FETCH_FAILED` | expected committed upstream chunk missing | infra tier → re-dispatch reducer; persistent miss escalates to query-fail |
| `WORKER_FAILED` | preemption / unreachable — **coordinator-synthesized** (a preempted worker sends no `ShardReport`); inferred from `EndpointService` lease loss / RPC timeout | infra tier → re-dispatch |
| `CANCELLED` | user cancel | terminal |
| `INTERNAL` | unexpected | terminal (surfaced) |

Tier semantics: **task tier** (`task_failures`, deterministic errors) fails the query at
`MAX_TASK_FAILURES`; **infra tier** (`infra_failures`, preemption/transient) retries to
`MAX_INFRA_FAILURES` without counting toward query-fail; **terminal** fails the query immediately.

## 9. File map

| Piece | Path |
|---|---|
| Shared proto | `lib/smallquery/proto/smallquery.proto` |
| Coordinator service + reconciler | `lib/smallquery/src/smallquery/coordinator.py` |
| Planner (logical → stage DAG) | `lib/smallquery/src/smallquery/planner.py` |
| Control DB + checkpoint | `lib/smallquery/src/smallquery/control_db.py`, `checkpoint.py` |
| Catalog | `lib/smallquery/src/smallquery/catalog.py` |
| Admission / scheduler | `lib/smallquery/src/smallquery/admission.py` |
| Python client | `lib/smallquery/src/smallquery/client.py` |
| Dashboard (Vue SPA) | `lib/smallquery/web/` (served at `/`, finelog pattern) |
| Rust worker crate | `lib/smallquery/rust/` |
| `ObjectStoreExchangeExec` | `lib/smallquery/rust/src/exchange.rs` |
| `ShuffleReader` TableProvider | `lib/smallquery/rust/src/shuffle_reader.rs` |
| Worker service + shard runner | `lib/smallquery/rust/src/worker.rs` |
| Deploy (coordinator + `IrisDaemon` spec) | `lib/smallquery/src/smallquery/deploy/` |

## 10. Out of scope (v0)

- **Writes**: `INSERT` / `CREATE TABLE AS` / materialized datasets. Read-only `SELECT`.
- **UDFs / Python or SQL extensions** in queries.
- **Skew mitigation** beyond clean failure + larger-N retry: no hot-key salting, no object-store
  spill (`design.md` "Skew and memory limits").
- **Cross-location queries** by default (rejected; opt-in flag only).
- **Flight peer-to-peer shuffle fetch** — all shuffle goes through object storage.
- **Multi-coordinator HA** — single non-preemptible coordinator, restart-from-snapshot only.
- **Streaming result delivery** beyond the capped inline preview + Parquet parts (no Arrow-Flight
  result stream in v0).
- **Engine choice**: Arm A (DataFusion) vs Arm B (DuckDB) is settled by the bake-off, not here;
  the proto carries both `datafusion_logical` and `duckdb_sql` so the contract survives either.
