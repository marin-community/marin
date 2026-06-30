# smallquery — research

Background research for `lib/smallquery`: a BigQuery-like distributed SQL engine that runs
on Iris, bursts onto spare CPU/RAM of **preemptible** TPU VMs, reads Parquet from regional
GCS / CoreWeave object storage, and materializes results back to object storage. SHA pins are
against `main` @ `6a6699b7e`.

## 1. The decisive constraints (these pick the architecture)

From the user framing + Iris research:

- **Workers are preemptible, lost at any instant, with no graceful drain signal.** A query
  must survive arbitrary mid-query worker loss.
- **Local disk is tiny and slow** (~50–90 GB pd-ssd; worse, the default worker scratch is
  `/dev/shm/iris` *tmpfs*, RAM-backed — see §2). Shuffle/spill must **not** depend on fast
  local disk; route intermediates through object storage instead.
- **Compute should be colocated with the data** — i.e. workers in the **same region/cloud** as
  the bucket, to avoid cross-region egress (a top cost driver per `AGENTS.md`). Note: Iris has
  *no node-level data-locality primitive*; "colocation" here is region/zone placement, not
  per-object node affinity (§2.3).
- **Big RAM, lots of CPU** on the TPU VMs — favors in-RAM per-partition execution.
- Build on existing software where possible; persistent **non-preemptible coordinator**
  (user's call) orchestrating preemptible worker bursts.

## 2. Iris primitives (`lib/iris`)

### 2.1 Job submission & resource model
- CLI `iris job run` (`lib/iris/src/iris/cli/job.py:775-1006`); Python SDK
  `IrisClient.remote(...).submit(...)` (`lib/iris/src/iris/client/client.py:486,597-617`).
- `ResourceSpecProto` = `cpu_millicores`, `memory_bytes`, `disk_bytes`, + a `DeviceConfig`
  oneof (Cpu/Gpu/Tpu) (`lib/iris/src/iris/rpc/job.proto:391-426`). Attaching any device floors
  CPU to 4 cores (`MIN_ACCELERATOR_CPU_MILLICORES=4000`, `cluster/types.py:589-597`) — so
  smallquery workers should request **CPU only**, *not* the TPU device, to stay tiny.
- **No K8s-style requests-vs-limits / QoS / overcommit.** Scheduler is request-only: a worker's
  free capacity = `total − committed(requests)` (`scheduling/scheduler.py:61-137,240-299`).

### 2.2 Spare-capacity / burstable recipe (most important)
The idiomatic "use whatever's left, yield to everyone" recipe is **not** overcommit; it's:
1. **BATCH priority band** — never preempts, preempted by both higher bands
   (`scheduling/policy.py:567-585,640-642`; `docs/priority-bands.md`). The safe-to-bulk-launch
   opportunistic class.
2. **Zero/tiny `cpu_millicores`** — `can_fit` only checks CPU when `cpu_need>0`
   (`scheduler.py:271-276`), so a 0-CPU CPU-only task packs onto any constraint-matching worker
   regardless of remaining CPU. Closest thing to a best-effort pod.
3. **Realistic hard `memory_bytes`** — memory is *always* a hard cgroup cap, no bursting, OOM
   surfaced as `OOMKilled` (`runtime/docker.py:684-686,764-772`). Size it as a real ceiling.
- CPU **bursting via soft `--cpu-shares` exists only on `ON_DEMAND` workers**; on preemptible
  TPU VMs CPU is hard-capped (`runtime/docker.py:675-686`). So on our target nodes the lever is
  "request little CPU and let the scheduler pack us," not "burst above request."

### 2.3 Scheduling, colocation, gangs
- Affinity = the **constraint system** (`Constraint{key,op,value,mode}` HARD filter / SOFT rank,
  `job.proto:480-518`), keyed on worker attributes (`region`, `zone`, `device-variant`,
  `preemptible`, custom `IRIS_WORKER_ATTRIBUTES`). **No built-in data-locality or job-to-job
  affinity** — region/zone placement is the only "colocation" available
  (`cluster/types.py:58-72`).
- Gang/coscheduling via `CoschedulingConfig{group_by}` (`job.proto:525-530`,
  `scheduler.py:799-866`).

### 2.4 Preemption = reactive, no drain signal
- **No GCE spot/maintenance-event watcher, no SIGTERM-on-preemption.** Liveness is inferred
  from the per-worker Reconcile RPC going UNREACHABLE; dead workers reaped, their tasks →
  `WORKER_FAILED`, retried under `max_retries_preemption` (default **100** CLI / **1000** SDK).
  Design for abrupt loss + **idempotent restart**; never expect a graceful drain.
- `TASK_STATE_PREEMPTED` (proto 10) specifically = controller *priority* preemption, distinct
  from spot loss.

### 2.5 Services, proxy, dashboard
- **EndpointService**: a leased registry (`controller/endpoint_service.py:83-132`). A task calls
  `RegisterEndpoint{name,address,...,lease_duration}`; client `EndpointClient.register(...)`
  auto-renews at 1/3 lease and unregisters on close (`client/endpoint_client.py:83-154`).
  Endpoints auto-expire when the task goes terminal — crashed service self-cleans.
- **Reverse proxy** on the controller (`controller/endpoint_proxy.py`): users reach a service at
  `https://<controller>/proxy/<encoded_name>/<sub_path>` (`.`→`/` in the name), IAP/GCLB-fronted
  (`docs/iap-gclb.md`). This is exactly how finelog's dashboard is reached
  (`/proxy/system.log-server/`). smallquery's coordinator should `RegisterEndpoint` its query
  API + dashboard and be reached the same way.

### 2.6 Disk surprise
- Default worker scratch `cache_dir = /dev/shm/iris` is **tmpfs (RAM-backed)**
  (`config.py:378`); tmpfs usage counts against memory. `--disk` is **not** a scheduler fit
  dimension (`scheduler.py:140-148`), only accounting. For a memory-hungry SQL engine, either
  request a scale group with a real-disk `cache_dir`, or (better) **avoid local spill entirely**.
- Verify on-cluster: actual TPU-VM boot disk size (no 90 GB constant found in-repo; GCP default
  override is 50 GB pd-ssd, `platforms/gcp/workers.py:101-103`).

## 3. In-repo prior art

### 3.1 finelog (`lib/finelog`) — single-node DataFusion, local-first
- **Surprise: finelog queries LOCAL parquet, not object storage.** Object store (`gs://`/`s3://`
  via the Rust `object_store` crate, `rust/src/store/remote.rs:46-78`) is its **durability /
  archive** tier; the DataFusion `ListingTable` is built over `file://` local sealed segments
  (`rust/src/query/provider.rs:59-76`). So finelog's *read path is not reusable* for
  scan-over-object-store — that's exactly the piece smallquery must build.
- Engine: per-query Rust DataFusion `SessionContext`, tuned to emulate DuckDB result shape +
  `parquet.pushdown_filters=true` (`rust/src/query/mod.rs:147-153,258`). Real predicate pushdown
  incl. a custom trigram substring prune (`provider.rs:114-158`).
- **Single-node**, runs as a **k8s Deployment (1 replica) with a PVC**, *not* an Iris job
  (`src/finelog/deploy/_k8s.py:36`). Reached via the Iris endpoint proxy as
  `/system/log-server` (`OPS.md`).
- **Result fully materialized, capped at 64 MiB Arrow IPC** in one `QueryResponse`
  (`rust/src/server/mod.rs:34`, `stats_service.rs:168-185`) — no streaming, no spooling.
- Query RPC (connect-rpc): `StatsService.Query(QueryRequest{sql}) -> QueryResponse{arrow_ipc,
  row_count}` (`rust/proto/finelog_stats.proto:117-180`). Python `LogClient.query()` decodes IPC
  → `pa.Table` (`src/finelog/client/log_client.py:561`).
- **Reusable patterns:** the DataFusion tuning, the **Vue SPA dashboard served at `/`**
  (`rust/src/server/app.rs:111-139`), the endpoint-proxy deployment shape, the object_store creds
  story (`AWS_*` env / GCS workload identity, `_k8s.py:70-114`).
- Stale labels to *not* inherit: proto/handler comments say "DuckDB"/"rusqlite"; engine is
  DataFusion (`finelog_stats.proto:117-121`).

### 3.2 zephyr (`lib/zephyr`) — the real execution-layer prior art
- **Coordinator-as-Iris-job + pull-based workers.** `ZephyrContext.execute` submits a
  coordinator fray job hosting `ZephyrCoordinator`; workers are a child actor group that
  `pull_task` → execute one shard → `report_result` (`src/zephyr/execution.py:1603,1633-1643,
  634,714`). Coordinator drives stages with a per-stage queue + barrier
  (`execution.py:951-1005,856,891`).
- **Shuffle = scatter to object store, never local disk.** Map stages write `PickleDiskChunk`s;
  scatter writes one `shard-NNNN.shuffle` (zstd-framed) + `.scatter_meta` sidecar per source
  shard; reducers range-GET only their target byte ranges (`src/zephyr/stage_io.py:213-253`,
  `shuffle.py:4-31`, `execution.py:1494-1516`). `chunk_storage_prefix` defaults to a GCS temp
  bucket (`execution.py:1809-1811`). **This is precisely the object-store-shuffle pattern
  smallquery needs**, and it's already preemption-safe + disk-free.
- **No data locality.** grep for `localit|affinit|collocat|node_id` in `lib/zephyr/src` →
  nothing; workers pull whatever shard is next and fetch inputs from GCS. "Colocate compute with
  data" is *not* an existing primitive — would be added (and at region granularity it mostly
  doesn't matter; §2.3).
- **Preemption tolerance = three nested retry tiers:** per-shard requeue (separating
  `MAX_SHARD_FAILURES=3` deterministic vs `MAX_SHARD_INFRA_FAILURES=20` infra,
  `execution.py:568-624`); Iris actor retries (`max_task_retries=10`); whole-pipeline retry
  (`max_execution_retries=100`, `execution.py:1778-1779`). Idempotency from UUID-unique chunk
  paths + monotonic attempt-generation stale-result rejection
  (`stage_io.py:60-90`, `execution.py:726-732`). Non-preemptible coordinator (1g/0.1cpu) +
  preemptible workers (`execution.py:1771-1774`).
- **Pushdown at plan time** (`src/zephyr/plan.py:504-580`): first `FilterOp` + `SelectOp` folded
  into `InputFileSpec.filter_expr`/`columns`. Caveats: only the first filter pushed (no
  AND-merge), lambda filters / Map stop pushdown; realized as a per-row-group **post-decode**
  pyarrow filter (`readers.py:289-314`) — *not* stats/bloom row-group skipping or Hive partition
  pruning. Footer-only row-group splitting for sharding `compute_parquet_splits`
  (`readers.py:183`).
- **No HTTP dashboard** — status is pushed into the Iris task status text
  (`execution.py:449-482`). (finelog is the opposite: self-served SPA.)
- Two **different object-store stacks**: finelog = Rust `object_store` + `AWS_*` env; zephyr =
  Python fsspec/rigging `open_url`. A shared engine must pick one or bridge.

## 4. External build-vs-buy (smallpond / DuckDB / DataFusion+Ballista / ClickHouse / Polars)

**Headline: no off-the-shelf OSS distributed engine satisfies (preemption-tolerant) +
(no-fast-local-disk shuffle) together.** Every distributed engine either needs local-disk
shuffle, dies on mid-query node loss, or hides the good part behind proprietary cloud.

| Criterion | smallpond | DuckDB | DataFusion+Ballista | ClickHouse OSS | chDB/CH-local | Polars OSS | Polars Cloud |
|---|---|---|---|---|---|---|---|
| survives worker preemption mid-query | 🟡 Ray retry+durable files | ❌ single-node | ❌ executor loss→job restart | ❌ node loss→query fails | ❌ n/a | ❌ n/a | 🟡 *claimed*, closed |
| shuffle without fast local disk | ❌ needs shared FS (3FS) | ❌ spill→local disk | ❌ shuffle→executor local disk | ❌ central-merge/local spill | ❌ n/a | ❌ local disk | ❓ undocumented |
| colocate w/ object-store data | 🟡 | ✅ httpfs S3/GCS pushdown | ✅ object_store S3+GCS | ✅ s3()/gcs() | ✅ s3() | ✅ scan_parquet | ✅ |
| Python-embeddable | ✅ | ✅ best Arrow | 🟡 datafusion✅/ballista weak | 🟡 chDB✅/server client | ✅ chDB | ✅ Python-first | ✅ SaaS |
| least ops weight | 🟡 Ray+shared FS | ✅ library | 🟡 scheduler+executors | ❌ heavy stateful+Keeper | ✅ library | ✅ library | 🟡 external+license |
| SQL completeness | ✅ DuckDB | ✅ best | ✅ broad | ✅ quirky | ✅ quirky | 🟡 secondary | 🟡 |
| license / self-host | ✅ MIT (dormant) | ✅ MIT | ✅ Apache-2 | ✅ Apache-2 | ✅ Apache-2 | ✅ MIT | ❌ proprietary |

Per-option biggest risk:
- **smallpond:** abandoned (3 commits, all early 2025); 3FS-shuffle assumption is the wrong
  shape for object-store/preemptible. Don't adopt.
- **DuckDB:** no distribution; spill wants local disk → we build coordination/shuffle and must
  size partitions to fit in RAM. Richest SQL, best Arrow streaming, simplest embed.
- **DataFusion+Ballista:** executor preemption restarts the whole job; object-store/in-memory
  shuffle is fork-only (Spice AI). *DataFusion as a library* (not Ballista) is a strong Rust
  kernel — and finelog already invests here.
- **ClickHouse OSS:** heavy stateful cluster, no true OSS shuffle, query dies on node loss.
- **Polars:** the distributed/fault-tolerant engine is **closed SaaS**, un-self-hostable; OSS
  Polars is single-node with second-class SQL.
- **Spark / Trino (considered, not tabled above):** turnkey distributed SQL — but both assume
  executor **local-disk shuffle** and tolerate node loss only via **whole-query/stage restart**, so
  on preemptible + tmpfs they hit the disqualifying constraints. Viable *if* we relax to on-demand
  workers with real SSD scratch. Worth a benchmark as the **buy-fallback**, not the v0 build.
- **Managed BigQuery / external tables (GCS data only):** zero build for GCS-resident datasets, but
  no CoreWeave coverage, per-query billing, and governance outside our stack. A reasonable escape
  hatch for the GCS half if the build proves too costly.

**Conclusion — build, don't buy (with an honest caveat).** Two disqualifying constraints —
**preemptible nodes** and **tmpfs scratch** — are *our choices*, not laws; they're what eliminate
Spark/Trino/Ballista. If we relaxed to on-demand workers with real SSD, a turnkey engine could
work; that trade is tracked in the design (Open Questions / costs), with the managed/heavyweight
options above as buy-fallbacks. Keeping the preemptible + object-store-shuffle profile, smallquery =
a **per-node engine (DataFusion-native recommended; DuckDB the alternative — decided by a throwaway
bake-off, §4a/§5) under a smallquery coordinator on Iris**. The two things no engine gives for free,
which we own:
1. **Object-store shuffle** — repartition intermediates written as Parquet to GCS/CoreWeave
   (zephyr's scatter *idea*, our own `ExchangeExec`), surviving node death and never touching local
   disk.
2. **Idempotent durable-output retry** — each shard reads immutable inputs and writes outputs
   sealed by a **per-attempt `_SUCCESS` manifest** (needs only read-after-write of named objects,
   no atomic rename / conditional-create / strong LIST); a preempted shard just re-runs (see
   design).

## 4a. Shuffle: DataFusion component reuse (verified 2026)

We want first-class shuffle (joins, high-card `GROUP BY`). **ClickHouse is rejected**: OSS has no
true shuffle (broadcast/`GLOBAL JOIN` + centralized initiator merge), spills to local disk, dies
on mid-query node loss, and is a heavy stateful cluster; the real exchange operators
(`ShuffleExchange`/`Broadcast`/`Gather`) are ClickHouse **Cloud private-preview, not OSS**. chDB
is only a per-node kernel, not a shuffle solution.

**DataFusion is reusable as components, not a turnkey engine.** Verified API surface (DataFusion
53.x, 2026):
- Plan layers: SQL → `LogicalPlan` → physical `Arc<dyn ExecutionPlan>`; executed via
  `ExecutionPlan::execute(partition: usize, ctx: Arc<TaskContext>) -> SendableRecordBatchStream`.
  Partitioning is read via `properties().output_partitioning()` (a `PlanProperties`) — the old
  standalone `output_partitioning()` was removed.
  (https://docs.rs/datafusion/latest/datafusion/physical_plan/trait.ExecutionPlan.html)
- Custom scan: `TableProvider::scan(&self, state, projection: Option<&Vec<usize>>, filters:
  &[Expr], limit) -> Result<Arc<dyn ExecutionPlan>>`; filter pushdown is opt-in via
  `supports_filters_pushdown` (Exact/Inexact/Unsupported).
  (https://datafusion.apache.org/blog/2026/03/31/writing-table-providers/)
- Custom operator: implement `ExecutionPlan` (`name`, `properties`, `children`,
  `with_new_children`, `execute`; override `required_input_distribution`) — this is our
  object-store exchange.
- Shuffle boundary: the `EnforceDistribution` optimizer inserts `RepartitionExec` with
  `Partitioning::Hash(exprs, n)` before `HashJoinExec` / partitioned aggregates — the natural
  stage cut.
- Two-phase aggregation: `AggregateExec` with `AggregateMode::Partial` → `FinalPartitioned`
  (post-shuffle, no single-partition bottleneck) gives correct distributed `GROUP BY` for free.
- Plan serialization: `datafusion-proto` serializes **physical** plans (`physical_plan_to_bytes`
  / `physical_plan_from_bytes`); custom operators need a `PhysicalExtensionCodec`. **Substrait is
  logical-first / incomplete for physical plans** (EPIC #5173) — use `datafusion-proto` to ship
  fragments.
- Object store: `object_store` crate via `ObjectStoreRegistry` — GCS (`GoogleCloudStorageBuilder`),
  S3-compatible (`AmazonS3Builder`); Parquet projection + predicate + row-group-stats + bloom
  pruning.

**Ballista** = DataFusion + scheduler + executors (gRPC + Arrow Flight). Stage-splitter =
`DistributedPlanner`; shuffle operators = `ShuffleWriterExec`/`ShuffleReaderExec`. **Upstream
shuffle is executor LOCAL DISK (Arrow IPC) → executor loss restarts the whole job** — disqualified
for preemptible. Object-store/remote shuffle is an **open upstream request (#1539)**; it exists
only in the **Spice AI fork** (`spiceai/spiceai`): object-store shuffle backends (S3/Azure/GCS),
**failed tasks auto-retried from intermediate shuffle data**, automatic partition reassignment on
executor drop, multi-active scheduler HA coordinated *through object storage*. Production-targeted
(Spice v2.0 RCs, May 2026) but a fork coupled to Spice's stack, not upstreamed.

**Reuse map / decision:**

| Need | Reuse off the shelf | Build |
|---|---|---|
| SQL→logical→physical planning, `EnforceDistribution`, joins, Partial/`FinalPartitioned` agg, Parquet scan+pushdown | DataFusion core | — |
| GCS/S3 reads | `object_store` + `ObjectStoreRegistry` | register bucket builders |
| Custom scan over shuffle files | `TableProvider::scan` | provider impl |
| Object-store exchange operator | `ExecutionPlan` trait | the operator |
| Ship plan fragments | `datafusion-proto` `physical_plan_to_bytes` | `PhysicalExtensionCodec` |
| Stage splitting at shuffle boundaries | Ballista `DistributedPlanner` (borrow logic) | thin reimpl, skip Ballista scheduler |
| **Object-store shuffle + worker-failure tolerance** | **NOT upstream** (Spice fork only) | **build on DataFusion core (chosen)** |

→ **Decision:** build shuffle on DataFusion core + our own object-store `ExchangeExec`, driven by
the coordinator + Iris (skip Ballista's scheduler entirely). Kernel (DataFusion-native vs
DuckDB-bolted) left to a measured bake-off over the shared object-store-shuffle substrate. Shuffle
is **in v0**. Spice's fork is the port-or-adopt fallback if building the exchange proves too
costly.

## 6. Per-VM daemon feasibility (`lib/iris`)

Can smallquery workers be auto-deployed as a daemon on every preemptible VM (not as per-query
tasks)? **Verdict: no native DaemonSet primitive.** Workloads flow only through the scheduler as
job tasks.
- Worker bootstrap is a hardcoded template launching **exactly one** container — the agent — via
  one `docker run`, with no injection point for extra containers
  (`worker/worker_bootstrap.py:259-268,109-294`; substitutions only at `:297-325`).
- `/system/` endpoints are an **endpoint-registry** concept (never-expiring names), not a
  service Iris deploys on nodes (`controller/endpoint_service.py:9-10,62`). The log server /
  controller are single-replica, not per-node (`controller/main.py:49-66`).
- **No anti-affinity / one-per-VM operator.** Constraints are positive-match only (EQ/IN/EXISTS,
  `constraints.py:129,186,609`); coscheduling `group_by` is gang *affinity*, the opposite of
  spread (`scheduler.py:539-548,799-862`). One-per-VM is only weakly approximable via unique
  per-VM attributes + N constrained jobs.
- Scale groups / `WorkerSettings` / `WorkerConfig` carry **no sidecar/extra-container field**
  (`config.py:315-340,365-393`). Worker image is a config value but the launch command is
  hardcoded to `iris.cluster.worker.main serve` (`worker_bootstrap.py:267`).
- Process model: the agent is a host container with the Docker socket mounted; all tasks are
  sibling Docker containers it launches (`worker_bootstrap.py:259-265`, `worker/worker.py:169`).
  A host-process daemon *could* run beside it but would be invisible to scheduler/autoscaler/
  preemption.
- **Closest existing patterns (no new primitive):** `WorkerPool`
  (`client/worker_pool.py:4-9,135-151`) — a coscheduled gang of standing worker tasks receiving
  dispatched callables; and the **actor framework + `ActorPool`** (`actor/server.py:4-9`,
  `actor/pool.py:58`) — long-lived RPC servers registered as endpoints, coordinator
  load-balances/broadcasts RPCs. Both are scheduler-placed tasks, not per-VM daemons — but they
  give "coordinator pushes RPCs to warm standing workers" for free.
- k8s backend already composes a log-shipping **sidecar** per pod
  (`backends/k8s/tasks.py:460-470`) — so a real DaemonSet/sidecar is plausible to *add* on the
  k8s/CoreWeave path, not the GCP/TPU RPC path.

→ **Decision:** smallquery's worker fleet = the new **`IrisDaemon`** primitive
(`.agents/projects/iris_daemon/`), specified separately for independent Iris-owner review. No
`WorkerPool`/per-query-task fallback — that churn isn't worth carrying for a model we'd
immediately replace. The closest existing patterns (`WorkerPool`, `ActorPool`) are reference
designs for `IrisDaemon`'s standing-RPC-worker behavior, not the smallquery substrate.

## 5. Resolved architecture decisions
- **Service shape**: persistent **non-preemptible coordinator** (finelog deployment shape) +
  **`IrisDaemon`** worker fleet (one warm worker per node of a scale group). No per-query Iris
  jobs, no WorkerPool fallback. smallquery depends on the `IrisDaemon` primitive
  (`.agents/projects/iris_daemon/`, issue #6763).
- **Coordinator backbone**: a **fresh `lib/smallquery` coordinator** that is a **thin stateless
  reconciler** — plans once via DataFusion's parser/planner → a manifest, then drives from an
  **embedded SQLite control DB** (query/stage/shard status, admission, cancellation, failure
  counts) hot-snapshotted to GCS like the Iris controller (`controller/checkpoint.py`): `report_done`
  drives steady state, object-store `_SUCCESS` manifests are the data commit barrier reconciled only
  on restore. No compute over data; restartable. Coordinator is **Python** (`datafusion-python`,
  verified 54.0.0):
  it cuts stages from the **typed logical plan** (`to_variant()` → `Join.on()`,
  `Aggregate.group_by_exprs()`, `TableScan.filters()`); physical-plan introspection from Python is
  display-string-only, so we cut at the logical level (a thin Rust shim is the fallback if we ever
  need DataFusion's exact physical shuffle decisions). It ships **logical stage specs**, not
  serialized physical plans — so no `datafusion-proto`/`PhysicalExtensionCodec` and no plan-version
  skew. **No code dependency on zephyr** — DataFusion provides Parquet + `object_store`; our Rust
  worker's `ExchangeExec` replaces scatter; sealed-attempt manifests replace zephyr's retry
  machinery. zephyr is reference-only.
- **Shuffle**: build on **DataFusion core + our own object-store `ExchangeExec`** (skip Ballista's
  scheduler). **In v0** (joins + high-card `GROUP BY`). Spice fork = port/adopt fallback (§4a).
- **Kernel**: **bake-off** — DataFusion-native (Arm A, recommended) vs DuckDB-bolted (Arm B) —
  over the shared object-store-shuffle substrate; Arrow-pinned shard contract; decide on
  measured results.
- **Result contract**: materialize result Parquet to a **TTL'd temp object-store location**
  (`marin_temp_bucket(ttl_days=X)`) + a **capped inline Arrow preview** (finelog 64 MiB cap).
  User streams/downloads from the temp location.
