//! Store orchestration: the seam the RPC handlers sit on.
//!
//! On construct: open the catalog, create `data_dir`, rehydrate the live
//! registry from `catalog.list_all()`, then ensure the privileged `log`
//! namespace is registered (`with_implicit_seq(LOG_REGISTERED_SCHEMA)`).
//!
//! Critical behaviors:
//! - `register_table` returns the EFFECTIVE store-form schema (WITH `seq`); the
//!   RPC handler strips `seq` for the wire.
//! - re-register with an EMPTY policy KEEPS the existing policy.
//! - `log` is privileged and undroppable.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::datatypes::SchemaRef;

use crate::errors::StatsError;
use crate::proto::finelog::stats::ColumnType;
use crate::query::provider::NamespaceProvider;
use crate::query::RegisteredProvider;
use crate::store::catalog::{Catalog, RegisteredNamespace};
use crate::store::namespace::Namespace;
use crate::store::namespace_name::validate_namespace_name;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{
    merge_schemas, resolve_key_column, with_implicit_cluster, with_implicit_seq, AlignedBatch,
    Column, Schema,
};
use crate::store::types::NamespaceStats;

/// The privileged log namespace name.
pub const LOG_NAMESPACE_NAME: &str = "log";
/// Its on-disk subdirectory.
pub const LOG_NAMESPACE_DIR: &str = "log";

/// Bounded budget for stopping + joining a namespace's background tasks during a
/// live lifecycle transition (re-register replacement, drop). Runs inside the
/// RPC's `spawn_blocking` worker, so it must not block long: a task that misses
/// this window is aborted rather than wedging the worker. Distinct from the
/// process-shutdown drain budget passed to [`Store::shutdown`] at SIGTERM.
const NAMESPACE_LIFECYCLE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Registered schema for the privileged `log` namespace; `key_column = "key"`.
///
/// The original five columns (key/source/data/epoch_ms/level) are non-nullable.
/// `cluster` is a later **additive, nullable** column: the writer-supplied origin
/// cluster of each push (trusted — writers are authenticated), which namespaces
/// logs a global finelog collects from many federated clusters. It is nullable so
/// it evolves an already-registered `log` namespace additively — `merge_schemas`
/// requires new columns to be nullable, and segments written before the column
/// existed null-fill it on read.
pub(crate) fn log_registered_schema() -> Schema {
    Schema::new(
        vec![
            Column::new("key", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("source", ColumnType::COLUMN_TYPE_STRING, false),
            // The log message body — substring-searched via contains()/LIKE, so
            // it carries the trigram index.
            Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
            Column::new("epoch_ms", ColumnType::COLUMN_TYPE_INT64, false),
            Column::new("level", ColumnType::COLUMN_TYPE_INT32, false),
            Column::new("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        ],
        "key",
    )
}

/// One consistent view of a namespace's sealed local segments: the arrow schema to
/// read them with, their paths, and the lowest `seq` they hold (`None` when there is
/// no local segment). Captured under a single hold of the engine's insertion lock, so
/// `min_seq` always describes exactly the segments in `paths`.
pub struct NamespaceSnapshot {
    pub schema: SchemaRef,
    pub paths: Vec<String>,
    pub min_seq: Option<i64>,
}

/// Store backed by the Rust catalog plus per-namespace durability engines.
///
/// The catalog owns the persistent registry + segments table; the `engines`
/// map owns one `Namespace` per live namespace (built at boot from the catalog
/// and on `register_table`). The data path (WriteRows / PushLogs) routes through
/// these engines; the metadata RPCs stay on the catalog.
pub struct Store {
    data_dir: Option<PathBuf>,
    remote_log_dir: String,
    catalog: Arc<Catalog>,
    engines: Mutex<HashMap<String, Arc<Namespace>>>,
    /// Process-wide query-visibility lock. A query / FetchLogs holds the READ
    /// side across the full DataFusion scan, because `query_providers` snapshots
    /// segment PATHS and DataFusion opens those parquet files LAZILY during
    /// `collect()`. Structural mutations that unlink segment files — `drop_table`,
    /// compaction/eviction — take the WRITE side so no scan is mid-flight over
    /// paths about to disappear.
    ///
    /// ONE shared instance for the whole process (queries are cross-namespace, so
    /// the drain must be global). Cloned into each `Namespace` so the per-ns
    /// maintenance task takes `.blocking_write()` inside its `spawn_blocking`.
    ///
    /// `tokio::sync::RwLock` is WRITE-preferring (a new reader waits behind a
    /// pending writer). It upholds the safety invariant (a writer never proceeds
    /// while any reader holds the lock, so no scan opens a file mid-unlink), and
    /// write-preference is safer here — it cannot starve compaction/eviction under
    /// a steady query stream.
    query_visibility: Arc<tokio::sync::RwLock<()>>,
}

impl Store {
    /// Construct the store: create `data_dir`, rehydrate the live registry +
    /// per-namespace engines from the catalog, and ensure the privileged `log`
    /// namespace exists.
    ///
    /// `remote_log_dir` configures the per-namespace offload target (empty
    /// disables sync). Pass it through to each `Namespace`.
    pub fn new(data_dir: Option<PathBuf>, remote_log_dir: String) -> Result<Store, StatsError> {
        let startup_started = Instant::now();
        if let Some(dir) = &data_dir {
            std::fs::create_dir_all(dir).map_err(|e| {
                StatsError::Internal(format!("create data_dir {}: {e}", dir.display()))
            })?;
        }
        let catalog_open_started = Instant::now();
        let catalog = Arc::new(Catalog::open(data_dir.as_deref())?);
        let catalog_open_ms = catalog_open_started.elapsed().as_millis() as u64;
        // Rebuild-from-disk catalog adoption. On a fresh boot over a log_dir an
        // earlier server populated, the sqlite sidecar is empty, so the disk
        // parquet layout + footers are the only record of the namespaces +
        // segments. The sentinel-gated, idempotent scan persists the recovered
        // `namespaces` + `segments` rows BEFORE `rehydrate_from_catalog` reads
        // them back to build the engines. No-op in in-memory mode + on the done
        // sentinel (subsequent boots). REMOTE adoption is the engines'
        // `boot_reconcile`, run in the background by each namespace's
        // maintenance task (spawned by `bootstrap_maintenance`), not before bind.
        let catalog_adoption_started = Instant::now();
        crate::store::adopt::ensure_catalog_adopted(data_dir.as_deref(), &catalog)?;
        let catalog_adoption_ms = catalog_adoption_started.elapsed().as_millis() as u64;
        let store = Store {
            data_dir,
            remote_log_dir,
            catalog,
            engines: Mutex::new(HashMap::new()),
            query_visibility: Arc::new(tokio::sync::RwLock::new(())),
        };
        // Register/evolve the privileged `log` schema in the catalog BEFORE
        // rehydrate builds the engines, so the log engine is opened exactly once
        // with the current schema. This is what adopts a newly-added additive
        // column (e.g. `cluster`) on an already-registered `log` namespace:
        // evolving after rehydrate would instead require rebuilding a live engine,
        // whose stop-and-join uses a runtime `block_on` that is illegal here (this
        // runs directly on the async `main` task, not a `spawn_blocking` worker).
        let log_schema_started = Instant::now();
        store.ensure_log_namespace_schema()?;
        let log_schema_ms = log_schema_started.elapsed().as_millis() as u64;
        let rehydrate_started = Instant::now();
        store.rehydrate_from_catalog()?;
        let rehydrate_ms = rehydrate_started.elapsed().as_millis() as u64;
        let namespaces = store.engines.lock().unwrap().len();
        tracing::info!(
            namespaces,
            catalog_open_ms,
            catalog_adoption_ms,
            log_schema_ms,
            rehydrate_ms,
            total_ms = startup_started.elapsed().as_millis() as u64,
            "finelog store startup complete"
        );
        Ok(store)
    }

    /// Start each namespace's maintenance task. Called once after `new`, before
    /// serving.
    ///
    /// Each task runs its boot remote reconcile (adopt unknown remote parquet,
    /// redundancy-drop covered segments) in the BACKGROUND as its first step,
    /// before the periodic loop — so the reconcile's object_store footer reads
    /// never block the listener bind / `/health`, and the first maintenance tick
    /// still can't race adoption (it is sequenced after reconcile within the
    /// task). Rehydrated namespaces are backed by local segments, so `next_seq`
    /// is already recovered locally; deferring the remote reconcile only delays
    /// archived-row catalog visibility + redundancy cleanup, never correct
    /// serving of live (local) rows.
    pub fn bootstrap_maintenance(&self) {
        let engines: Vec<Arc<Namespace>> = self.engines.lock().unwrap().values().cloned().collect();
        for engine in &engines {
            engine.spawn_maintenance(true);
        }
    }

    fn rehydrate_from_catalog(&self) -> Result<(), StatsError> {
        for (name, schema) in self.catalog.list_all()? {
            let policy = self.catalog.get_policy(&name)?;
            // Do NOT spawn the maintenance task here — `bootstrap_maintenance`
            // spawns it for the whole rehydrated set (the task then runs its boot
            // reconcile in the background as its first step).
            self.build_engine(&name, schema.clone(), policy.clone(), false)?;
            self.catalog.insert_live(RegisteredNamespace {
                name,
                schema,
                policy,
            });
        }
        Ok(())
    }

    /// Resolve the on-disk subdir for `name` WITHOUT validating (callers that
    /// already hold a validated/registered name; `log` maps to `{data_dir}/log`).
    fn engine_dir(&self, name: &str) -> Option<PathBuf> {
        self.data_dir.as_ref().map(|dir| {
            if name == LOG_NAMESPACE_NAME {
                dir.join(LOG_NAMESPACE_DIR)
            } else {
                dir.join(name)
            }
        })
    }

    /// Build (or rebuild) the engine for `name` with `stored_schema`, replacing
    /// any prior engine. The engine recovers next_seq + adopts local segments.
    ///
    /// `spawn_maint` starts the per-namespace maintenance task immediately —
    /// `true` for a runtime `register_table` (which reconciles synchronously
    /// first for cold-boot next_seq safety, then spawns a task that skips its own
    /// reconcile), `false` during boot rehydrate (where `bootstrap_maintenance`
    /// spawns the task, which reconciles in the background as its first step).
    fn build_engine(
        &self,
        name: &str,
        stored_schema: Schema,
        policy: StoragePolicy,
        spawn_maint: bool,
    ) -> Result<(), StatsError> {
        let ns_dir = self.engine_dir(name);
        // Re-register over a live engine (additive schema evolution): stop AND
        // JOIN the prior engine's flush + maintenance tasks before opening the
        // replacement over the same directory, so the old tasks can't flush /
        // evict / upsert concurrently with the new engine adopting that dir.
        // Disk-backed only — mem-store namespaces spawn no background tasks, so
        // replacing the Arc is enough. This always runs under a runtime: a
        // disk-backed re-register arrives via register_table's spawn_blocking
        // worker; the boot rehydrate path has no prior, so block_on never fires.
        if ns_dir.is_some() {
            let prior = self.engines.lock().unwrap().get(name).cloned();
            if let Some(prior) = prior {
                tokio::runtime::Handle::current()
                    .block_on(prior.shutdown(NAMESPACE_LIFECYCLE_SHUTDOWN_TIMEOUT));
            }
        }
        let engine = Namespace::open(
            name,
            stored_schema,
            ns_dir,
            Arc::clone(&self.catalog),
            Arc::clone(&self.query_visibility),
            &self.remote_log_dir,
            policy,
        )?;
        if spawn_maint {
            // Runtime register: run the boot remote reconcile SYNCHRONOUSLY (so a
            // re-register over a wiped catalog adopts the bucket's segments before
            // the caller observes the namespace), then start the maintenance
            // task. `register_table` runs inside a `spawn_blocking` worker on the
            // multi-threaded runtime, so `Handle::block_on` of the async reconcile
            // is safe here (it never blocks a reactor thread). No-op without a
            // remote dir.
            if engine.has_remote() {
                let engine_for_reconcile = Arc::clone(&engine);
                tokio::runtime::Handle::current()
                    .block_on(async move { engine_for_reconcile.boot_reconcile().await })?;
            }
            // Reconcile already ran synchronously above (cold-boot next_seq
            // safety), so the task must NOT reconcile again — pass false.
            engine.spawn_maintenance(false);
        }
        self.engines
            .lock()
            .unwrap()
            .insert(name.to_string(), engine);
        Ok(())
    }

    /// The live engine for `name`, or `NamespaceNotFound`.
    fn require_engine(&self, name: &str) -> Result<Arc<Namespace>, StatsError> {
        self.engines
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| {
                StatsError::NamespaceNotFound(format!("namespace {name:?} is not registered"))
            })
    }

    /// Register the privileged `log` namespace's schema in the catalog, or
    /// additively evolve an already-registered one to the current
    /// [`log_registered_schema`] (the union: existing columns plus any new
    /// nullable ones, e.g. `cluster`). Catalog-only — no engine is built here;
    /// `rehydrate_from_catalog` (which runs immediately after) opens the engine
    /// from the resulting catalog schema.
    ///
    /// This runs before rehydrate, so the catalog's live map is still empty and
    /// `register_or_evolve` takes its fresh-registration path. We therefore pass
    /// the *persisted* policy (not [`StoragePolicy::default`]) so a store that
    /// already has a custom `log` retention/offload policy keeps it across boots
    /// rather than having the row reset.
    fn ensure_log_namespace_schema(&self) -> Result<(), StatsError> {
        let schema = log_registered_schema();
        resolve_key_column(&schema)?;
        let stored = with_implicit_seq(schema);
        let policy = self.catalog.get_policy(LOG_NAMESPACE_NAME)?;
        let stored_for_merge = stored.clone();
        self.catalog
            .register_or_evolve(LOG_NAMESPACE_NAME, stored, policy, move |existing| {
                merge_schemas(existing, &stored_for_merge)
            })?;
        Ok(())
    }

    /// Resolve the on-disk subdir for `name`, validating the name. The `log`
    /// namespace maps to `{data_dir}/log`; in-memory mode still enforces the
    /// regex.
    fn namespace_dir(&self, name: &str) -> Result<Option<PathBuf>, StatsError> {
        match &self.data_dir {
            None => {
                validate_namespace_name(name, None)?;
                Ok(None)
            }
            Some(dir) => {
                if name == LOG_NAMESPACE_NAME {
                    return Ok(Some(dir.join(LOG_NAMESPACE_DIR)));
                }
                validate_namespace_name(name, Some(dir))
            }
        }
    }

    /// Register or evolve `name` to `schema`; return the EFFECTIVE store-form
    /// schema (WITH implicit `seq` and `cluster`). On re-register an empty policy
    /// is kept.
    ///
    /// Every registered table gains the implicit `cluster` origin column when it
    /// does not declare one, so a table's rows become attributable to their origin
    /// cluster on a hub finelog uniformly — the forwarder stamps that column, and a
    /// producer need not know the column exists.
    pub fn register_table(
        &self,
        name: &str,
        schema: Schema,
        policy: StoragePolicy,
    ) -> Result<Schema, StatsError> {
        // Validate the name (and fence the `log` dir special-case) first.
        self.namespace_dir(name)?;
        resolve_key_column(&schema)?;
        let stored = with_implicit_seq(with_implicit_cluster(schema));

        // `merge_schemas` (pure) raises SchemaConflict on a non-additive change.
        // The catalog applies the empty-policy-keeps-existing rule and persists
        // under a single lock; we only supply the schema-merge decision.
        let stored_for_merge = stored.clone();
        let had_engine = self.engines.lock().unwrap().contains_key(name);
        let (effective_schema, effective_policy) =
            self.catalog
                .register_or_evolve(name, stored, policy, move |existing_schema| {
                    merge_schemas(existing_schema, &stored_for_merge)
                })?;
        // (Re)build the engine on fresh registration or when the effective schema
        // evolved. The engine re-opens on the same dir, adopting existing
        // segments and recovering next_seq, so an additive evolution keeps the
        // already-flushed data visible. A runtime register spawns the maintenance
        // task immediately (no boot reconcile needed for an existing/fresh dir).
        let needs_engine = !had_engine
            || self
                .engines
                .lock()
                .unwrap()
                .get(name)
                .map(|e| e.schema() != &effective_schema)
                .unwrap_or(true);
        if needs_engine {
            self.build_engine(name, effective_schema.clone(), effective_policy, true)?;
        } else {
            // Engine kept; push the (possibly updated) policy onto it so a
            // policy-only re-register takes effect on the next eviction tick.
            if let Some(engine) = self.engines.lock().unwrap().get(name) {
                engine.update_policy(effective_policy);
            }
        }
        Ok(effective_schema)
    }

    /// Decode + validate + append a WriteRows batch, returning
    /// `(rows_written, last_seq)`. `last_seq` is the durability target the caller
    /// awaits (`-1` for an empty batch). The size/row caps and IPC decode happen
    /// before namespace resolution, then validate/align runs OUTSIDE any lock.
    pub fn write_rows(&self, name: &str, arrow_ipc: &[u8]) -> Result<(i64, i64), StatsError> {
        use crate::store::ipc::decode_one_record_batch;
        use crate::store::schema::{
            validate_and_align_batch, MAX_WRITE_ROWS_BYTES, MAX_WRITE_ROWS_ROWS,
        };

        if arrow_ipc.len() > MAX_WRITE_ROWS_BYTES {
            return Err(StatsError::SchemaValidation(format!(
                "WriteRows body {} bytes exceeds {MAX_WRITE_ROWS_BYTES} limit",
                arrow_ipc.len()
            )));
        }
        let batch = decode_one_record_batch(arrow_ipc)?;
        if batch.num_rows() > MAX_WRITE_ROWS_ROWS {
            return Err(StatsError::SchemaValidation(format!(
                "WriteRows batch {} rows exceeds {MAX_WRITE_ROWS_ROWS} limit",
                batch.num_rows()
            )));
        }
        let engine = self.require_engine(name)?;
        let aligned: AlignedBatch = validate_and_align_batch(&batch, engine.schema())?;
        let n = aligned.num_rows as i64;
        let last_seq = engine.append_aligned_batch(&aligned);
        Ok((n, last_seq))
    }

    /// Append log columns to the reserved `log` namespace, returning the last
    /// seq (or `-1`). `columns` are the six non-seq log columns in registered
    /// order (key/source/data/epoch_ms/level/cluster), prepared by the caller
    /// outside the lock.
    pub fn append_log_columns(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        num_rows: usize,
        added_bytes: i64,
    ) -> Result<i64, StatsError> {
        let engine = self.require_engine(LOG_NAMESPACE_NAME)?;
        Ok(engine.append_log_batch(columns, num_rows, added_bytes))
    }

    /// Block until `target` is durable in `name`, bounded by `timeout`.
    pub async fn await_persisted(
        &self,
        name: &str,
        target: i64,
        timeout: Duration,
    ) -> Result<(), StatsError> {
        let engine = self.require_engine(name)?;
        engine.await_persisted(target, timeout).await
    }

    /// Return the store-form schema for `name`. NamespaceNotFound if missing.
    pub fn get_table_schema(&self, name: &str) -> Result<Schema, StatsError> {
        Ok(self.catalog.require_live(name)?.schema)
    }

    /// The process-wide query-visibility lock. Query/FetchLogs handlers hold the
    /// READ side across the full DataFusion scan; structural mutations that
    /// unlink segments (`drop_table`, compaction/eviction) take the WRITE side.
    /// See the field doc on [`Store`].
    pub fn query_visibility(&self) -> &tokio::sync::RwLock<()> {
        &self.query_visibility
    }

    /// Snapshot every live namespace into a `RegisteredProvider` over its sealed
    /// segments — the registration set for a `Query`.
    ///
    /// Snapshot the live registry, then for each namespace capture its arrow
    /// schema + sealed-segment paths (under the engine's insertion lock).
    /// Visibility = sealed segments ONLY (the RAM buffer is not exposed). Every
    /// live namespace is registered so cross-namespace SQL and the reserved `log`
    /// namespace both resolve.
    pub fn query_providers(&self) -> Result<Vec<RegisteredProvider>, StatsError> {
        let mut out = Vec::new();
        for ns in self.catalog.snapshot_live() {
            let engine = match self.engines.lock().unwrap().get(&ns.name) {
                Some(e) => Arc::clone(e),
                // A registry entry with no engine is a transient state during
                // (re)build; skip it rather than fail the whole query.
                None => continue,
            };
            let arrow_schema = Arc::clone(engine.arrow_schema());
            let paths = engine.query_snapshot().paths;
            let provider = NamespaceProvider::build(arrow_schema, &paths)
                .map_err(|e| StatsError::Internal(format!("build provider {:?}: {e}", ns.name)))?;
            out.push(RegisteredProvider {
                name: ns.name,
                provider,
            });
        }
        Ok(out)
    }

    /// Snapshot `name`'s arrow schema alongside one consistent observation of its sealed
    /// segments: the paths a scan may read, and the lowest `seq` those paths hold. Both
    /// describe the same segment set, so a reader can tell a `seq` it simply has not
    /// reached from one that eviction put out of reach.
    pub fn query_snapshot(&self, name: &str) -> Result<NamespaceSnapshot, StatsError> {
        let engine = self.require_engine(name)?;
        let segments = engine.query_snapshot();
        Ok(NamespaceSnapshot {
            schema: Arc::clone(engine.arrow_schema()),
            paths: segments.paths,
            min_seq: segments.min_seq,
        })
    }

    /// `name`'s durability high-water mark: every row with `seq <= value` has been sealed
    /// into a segment, so it is visible to a scan unless it has since been evicted.
    pub fn namespace_persisted_seq(&self, name: &str) -> Result<i64, StatsError> {
        Ok(*self.require_engine(name)?.watch_persisted_seq().borrow())
    }

    /// The seq in `namespace` below which this store will never send to `target` again.
    pub fn forward_cursor(&self, target: &str, namespace: &str) -> Result<Option<i64>, StatsError> {
        self.catalog.forward_cursor(target, namespace)
    }

    /// Record `cursor` as settled for `(target, namespace)`.
    pub fn set_forward_cursor(
        &self,
        target: &str,
        namespace: &str,
        cursor: i64,
    ) -> Result<(), StatsError> {
        self.catalog.set_forward_cursor(target, namespace, cursor)
    }

    /// Return `(name, schema, stats, policy)` for every live namespace in
    /// registration order. Stats come from the per-namespace engine (sealed
    /// segments + RAM buffer seq-window math), falling back to the catalog
    /// aggregate if an engine is somehow absent.
    pub fn list_namespaces_with_stats(
        &self,
    ) -> Result<Vec<(String, Schema, NamespaceStats, StoragePolicy)>, StatsError> {
        let mut out = Vec::new();
        for ns in self.catalog.snapshot_live() {
            let stats = match self.engines.lock().unwrap().get(&ns.name) {
                Some(engine) => engine.stats(),
                None => self.catalog.aggregate_namespace_stats(&ns.name)?,
            };
            let policy = self.catalog.get_policy(&ns.name)?;
            out.push((ns.name, ns.schema, stats, policy));
        }
        Ok(out)
    }

    /// Return the effective policy now in force for `name`.
    pub fn get_policy(&self, name: &str) -> Result<StoragePolicy, StatsError> {
        self.catalog.get_policy(name)
    }

    /// Run one full maintenance cycle for `name`:
    /// `flush -> compact (planner-drained, or forced L0->L1) -> sync -> evict ->
    /// backfill missing trigram sidecars`.
    ///
    /// This is the body the per-namespace background maintenance task runs on its
    /// tick, and the entry point the `--debug-admin` `POST /debug/maintain` drives
    /// to force the pipeline deterministically. ALL stages are real (compaction +
    /// object_store sync + eviction).
    ///
    /// The query-visibility WRITE lock is taken INSIDE the engine
    /// (`commit_swap` / `evict_segment` via `blocking_write`), drained against
    /// in-flight queries that hold the READ side across their scan — so the caller
    /// MUST NOT hold the write lock (that would deadlock the blocking acquire).
    pub async fn maintain_namespace(
        &self,
        name: &str,
        force_compact_l0: bool,
    ) -> Result<(), StatsError> {
        let engine = self.require_engine(name)?;
        engine.run_maintenance(force_compact_l0).await
    }

    /// Backdate a segment's `created_at_ms` (test-only `/debug/backdate` seam, so
    /// age-eviction tests stay RPC-only with no sleep). `path_basename` is the
    /// segment filename; all matching rows in `name` are updated.
    pub fn backdate_segment(
        &self,
        name: &str,
        path_basename: &str,
        created_at_ms: i64,
    ) -> Result<(), StatsError> {
        let engine = self.require_engine(name)?;
        engine.backdate_segment(path_basename, created_at_ms)
    }

    /// Per-segment catalog rows for `name`, ordered by `min_seq`, for the
    /// `--debug-admin` `GET /debug/segments` observation surface. Exposes
    /// level/location/seq-bounds that `NamespaceInfo` does not.
    pub fn list_segments(
        &self,
        name: &str,
    ) -> Result<Vec<crate::store::types::SegmentRow>, StatsError> {
        self.catalog.list_segments(name)
    }

    /// Remove `name` from the registry and delete its catalog rows + on-disk
    /// subdir. Rejects the privileged `log` namespace.
    pub fn drop_table(&self, name: &str) -> Result<(), StatsError> {
        if name == LOG_NAMESPACE_NAME {
            return Err(StatsError::InvalidNamespace(format!(
                "namespace {name:?} is privileged and cannot be dropped via DropTable"
            )));
        }
        self.catalog.begin_drop(name)?;
        // Drop the engine first so its flush task stops touching the dir/catalog
        // before we delete rows + files.
        let engine = self.engines.lock().unwrap().remove(name);
        let result = (|| {
            if let Some(engine) = engine {
                if self.data_dir.is_some() {
                    // Disk-backed: stop AND JOIN the flush + maintenance tasks
                    // before deleting the dir + catalog rows, so an in-flight
                    // flush can't write parquet / upsert a row into the namespace
                    // we are tearing down (orphaned file, resurrected row).
                    // drop_table runs in a spawn_blocking worker, so block_on of
                    // the async join is safe (never blocks a reactor thread).
                    tokio::runtime::Handle::current()
                        .block_on(engine.stop_and_join(NAMESPACE_LIFECYCLE_SHUTDOWN_TIMEOUT));
                } else {
                    // mem-store: no background tasks and no dir; a sync stop
                    // signal suffices and needs no runtime.
                    engine.request_stop();
                }
            }
            self.catalog.delete(name)?;
            if let Some(dir) = &self.data_dir {
                let sub = dir.join(name);
                if sub.exists() {
                    std::fs::remove_dir_all(&sub).map_err(|e| {
                        StatsError::Internal(format!("remove namespace dir {}: {e}", sub.display()))
                    })?;
                }
            }
            Ok(())
        })();
        self.catalog.finish_drop(name);
        result
    }

    /// Aggregate in-RAM accounting across live namespaces for the periodic
    /// diagnostics line. `namespaces` is the live engine count, `ram_bytes` /
    /// `chunks` sum the per-namespace RAM buffers.
    pub fn memory_summary(&self) -> crate::store::types::MemorySummary {
        let engines: Vec<Arc<Namespace>> = self.engines.lock().unwrap().values().cloned().collect();
        let mut ram_bytes = 0i64;
        let mut chunks = 0usize;
        for engine in &engines {
            let (b, c) = engine.memory_summary();
            ram_bytes += b;
            chunks += c;
        }
        crate::store::types::MemorySummary {
            namespaces: engines.len(),
            ram_bytes,
            chunks,
        }
    }

    /// Cooperatively shut down every namespace's background tasks.
    ///
    /// Called after the server loop returns. Each engine's
    /// [`Namespace::shutdown`] latches its stop flag, wakes its flush +
    /// maintenance tasks, JOINs them bounded by `per_namespace_timeout`, and does
    /// a final `flush_once`. Durability is preserved: an acked write was already
    /// on a sealed L0 segment before the ack, and the final flush drains any
    /// not-yet-acked RAM rows. The bounded join (plus the task-abort fallback on
    /// timeout) guarantees this cannot hang — `main` applies its own outer
    /// timeout around `shutdown` for defense in depth.
    pub async fn shutdown(&self, per_namespace_timeout: Duration) {
        let engines: Vec<Arc<Namespace>> = self.engines.lock().unwrap().values().cloned().collect();
        // Shut namespaces down concurrently so the total drain is bounded by the
        // per-namespace timeout, not its product with the namespace count.
        futures::future::join_all(
            engines
                .iter()
                .map(|engine| engine.shutdown(per_namespace_timeout)),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker_schema() -> Schema {
        Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        )
    }

    fn mem_store() -> Store {
        Store::new(None, String::new()).unwrap()
    }

    #[test]
    fn register_returns_store_form_with_seq_and_cluster() {
        let store = mem_store();
        let effective = store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        // The store form prepends the implicit `seq` and appends the implicit
        // origin `cluster` column, so a producer's plain schema becomes
        // cluster-attributable without declaring the column.
        assert_eq!(
            effective,
            with_implicit_seq(with_implicit_cluster(worker_schema()))
        );
        assert_eq!(effective.columns[0].name, "seq");
        let cluster = effective
            .column("cluster")
            .expect("implicit cluster column");
        assert!(cluster.nullable);
    }

    #[test]
    fn rejects_invalid_names() {
        let store = mem_store();
        for name in [
            "",
            "Iris.Worker",
            ".starts-dot",
            "1starts-digit",
            "has space",
            "has/slash",
            "..",
        ] {
            assert!(
                matches!(
                    store.register_table(name, worker_schema(), StoragePolicy::default()),
                    Err(StatsError::InvalidNamespace(_))
                ),
                "name={name}",
            );
        }
    }

    #[test]
    fn rejects_path_traversal() {
        let store = mem_store();
        assert!(matches!(
            store.register_table("../escape", worker_schema(), StoragePolicy::default()),
            Err(StatsError::InvalidNamespace(_))
        ));
    }

    #[test]
    fn rejects_schema_without_ordering_key() {
        let store = mem_store();
        let schema = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        assert!(matches!(
            store.register_table("iris.worker", schema, StoragePolicy::default()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn explicit_key_missing_rejects() {
        let store = mem_store();
        let schema = Schema::new(
            vec![Column::new(
                "worker_id",
                ColumnType::COLUMN_TYPE_STRING,
                false,
            )],
            "ts",
        );
        assert!(matches!(
            store.register_table("iris.worker", schema, StoragePolicy::default()),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn idempotent_and_subset_return_full() {
        let store = mem_store();
        let full = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("cpu_pct", ColumnType::COLUMN_TYPE_FLOAT64, true),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        let first = store
            .register_table("iris.worker", full.clone(), StoragePolicy::default())
            .unwrap();
        let again = store
            .register_table("iris.worker", full.clone(), StoragePolicy::default())
            .unwrap();
        assert_eq!(first, again);
        let subset = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        let eff = store
            .register_table("iris.worker", subset, StoragePolicy::default())
            .unwrap();
        assert_eq!(eff, with_implicit_seq(with_implicit_cluster(full)));
    }

    #[test]
    fn additive_nullable_merge() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let mut cols = worker_schema().columns;
        cols.push(Column::new("note", ColumnType::COLUMN_TYPE_STRING, true));
        let eff = store
            .register_table(
                "iris.worker",
                Schema::new(cols, ""),
                StoragePolicy::default(),
            )
            .unwrap();
        // `cluster` was added implicitly at the first registration, so it precedes
        // `note`, which this re-register adds as the new additive column.
        assert_eq!(
            eff.column_names(),
            vec![
                "seq",
                "worker_id",
                "mem_bytes",
                "timestamp_ms",
                "cluster",
                "note"
            ]
        );
    }

    #[test]
    fn type_change_and_non_nullable_reject() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let type_change = Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("mem_bytes", ColumnType::COLUMN_TYPE_FLOAT64, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        );
        assert!(matches!(
            store.register_table("iris.worker", type_change, StoragePolicy::default()),
            Err(StatsError::SchemaConflict(_))
        ));
        let mut cols = worker_schema().columns;
        cols.push(Column::new(
            "cpu_pct",
            ColumnType::COLUMN_TYPE_FLOAT64,
            false,
        ));
        assert!(matches!(
            store.register_table(
                "iris.worker",
                Schema::new(cols, ""),
                StoragePolicy::default()
            ),
            Err(StatsError::SchemaConflict(_))
        ));
    }

    #[test]
    fn key_hint_coerced_to_registered() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let req = Schema::new(worker_schema().columns, "timestamp_ms");
        let eff = store
            .register_table("iris.worker", req, StoragePolicy::default())
            .unwrap();
        assert_eq!(eff.key_column, ""); // registered (empty) wins
    }

    #[test]
    fn empty_policy_on_reregister_keeps_existing() {
        let store = mem_store();
        store
            .register_table(
                "iris.worker",
                worker_schema(),
                StoragePolicy {
                    max_segments: Some(9),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            store.get_policy("iris.worker").unwrap().max_segments,
            Some(9)
        );
        // re-register with empty policy -> existing kept.
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        assert_eq!(
            store.get_policy("iris.worker").unwrap().max_segments,
            Some(9)
        );
    }

    #[test]
    fn get_table_schema_unknown_is_not_found() {
        let store = mem_store();
        assert!(matches!(
            store.get_table_schema("nope"),
            Err(StatsError::NamespaceNotFound(_))
        ));
    }

    #[test]
    fn list_includes_log_with_zero_stats() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let entries = store.list_namespaces_with_stats().unwrap();
        let names: Vec<&str> = entries.iter().map(|(n, _, _, _)| n.as_str()).collect();
        assert!(names.contains(&"log"));
        assert!(names.contains(&"iris.worker"));
        for (_, _, stats, _) in &entries {
            assert_eq!(*stats, NamespaceStats::empty());
        }
    }

    #[test]
    fn drop_registered_then_gone() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        store.drop_table("iris.worker").unwrap();
        assert!(matches!(
            store.get_table_schema("iris.worker"),
            Err(StatsError::NamespaceNotFound(_))
        ));
        // re-register starts fresh.
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        assert!(store.get_table_schema("iris.worker").is_ok());
    }

    #[test]
    fn drop_unknown_is_not_found() {
        let store = mem_store();
        assert!(matches!(
            store.drop_table("nope.unknown"),
            Err(StatsError::NamespaceNotFound(_))
        ));
    }

    #[test]
    fn drop_log_rejected() {
        let store = mem_store();
        assert!(matches!(
            store.drop_table("log"),
            Err(StatsError::InvalidNamespace(_))
        ));
        assert!(store.get_table_schema("log").is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn boot_evolves_preexisting_log_schema_and_preserves_policy() {
        // A store booting over a deployment whose persisted `log` schema predates
        // the `cluster` column must additively evolve it (`ensure_log_namespace_schema`
        // merges the column into the catalog BEFORE rehydrate opens the engine, so
        // no live-engine rebuild happens at boot) WITHOUT resetting the namespace's
        // persisted storage policy.
        let dir = std::env::temp_dir().join(format!(
            "finelog_evolve_log_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // Seed the catalog with the frozen pre-cluster (five-column) `log` schema
        // and a non-default policy. The schema is spelled out because it is the
        // historical layout — deliberately different from today's
        // `log_registered_schema`.
        let seeded_policy = StoragePolicy {
            max_segments: Some(7),
            ..Default::default()
        };
        {
            let catalog = Catalog::open(Some(dir.as_path())).unwrap();
            let old = with_implicit_seq(Schema::new(
                vec![
                    Column::new("key", ColumnType::COLUMN_TYPE_STRING, false),
                    Column::new("source", ColumnType::COLUMN_TYPE_STRING, false),
                    Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
                    Column::new("epoch_ms", ColumnType::COLUMN_TYPE_INT64, false),
                    Column::new("level", ColumnType::COLUMN_TYPE_INT32, false),
                ],
                "key",
            ));
            catalog
                .register_or_evolve(LOG_NAMESPACE_NAME, old, seeded_policy.clone(), |existing| {
                    Ok(existing.clone())
                })
                .unwrap();
        }

        // Boot over that catalog: the schema gains the nullable `cluster` column,
        // appended after the original five, and the policy is preserved.
        let store = Store::new(Some(dir.clone()), String::new()).unwrap();
        let schema = store.get_table_schema(LOG_NAMESPACE_NAME).unwrap();
        assert_eq!(
            schema.column_names(),
            vec!["seq", "key", "source", "data", "epoch_ms", "level", "cluster"]
        );
        assert!(
            schema.column("cluster").unwrap().nullable,
            "the evolved cluster column is nullable"
        );
        assert_eq!(
            store.get_policy(LOG_NAMESPACE_NAME).unwrap(),
            seeded_policy,
            "boot evolution must not reset the persisted log policy"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
