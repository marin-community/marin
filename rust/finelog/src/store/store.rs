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
use std::time::Duration;

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
    merge_schemas, resolve_key_column, with_implicit_seq, AlignedBatch, Column, Schema,
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

/// Registered schema for the privileged `log` namespace. All non-nullable;
/// `key_column = "key"`.
fn log_registered_schema() -> Schema {
    Schema::new(
        vec![
            Column::new("key", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("source", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("data", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("epoch_ms", ColumnType::COLUMN_TYPE_INT64, false),
            Column::new("level", ColumnType::COLUMN_TYPE_INT32, false),
        ],
        "key",
    )
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
        if let Some(dir) = &data_dir {
            std::fs::create_dir_all(dir).map_err(|e| {
                StatsError::Internal(format!("create data_dir {}: {e}", dir.display()))
            })?;
        }
        let catalog = Arc::new(Catalog::open(data_dir.as_deref())?);
        // Rebuild-from-disk catalog adoption. On a fresh boot over a log_dir an
        // earlier server populated, the sqlite sidecar is empty, so the disk
        // parquet layout + footers are the only record of the namespaces +
        // segments. The sentinel-gated, idempotent scan persists the recovered
        // `namespaces` + `segments` rows BEFORE `rehydrate_from_catalog` reads
        // them back to build the engines. No-op in in-memory mode + on the done
        // sentinel (subsequent boots). REMOTE adoption is the engines'
        // `boot_reconcile` (run by `bootstrap_maintenance` before bind).
        crate::store::adopt::ensure_catalog_adopted(data_dir.as_deref(), &catalog)?;
        let store = Store {
            data_dir,
            remote_log_dir,
            catalog,
            engines: Mutex::new(HashMap::new()),
            query_visibility: Arc::new(tokio::sync::RwLock::new(())),
        };
        store.rehydrate_from_catalog()?;
        store.ensure_log_namespace_registered()?;
        Ok(store)
    }

    /// Run the boot remote reconcile for every namespace, then start each
    /// namespace's maintenance task. Called once after `new`, before serving.
    ///
    /// Reconcile is async (object_store footer reads); the maintenance task must
    /// not start until reconcile has populated the catalog so the first tick
    /// doesn't race adoption.
    pub async fn bootstrap_maintenance(&self) -> Result<(), StatsError> {
        let engines: Vec<Arc<Namespace>> = self.engines.lock().unwrap().values().cloned().collect();
        for engine in &engines {
            if engine.has_remote() {
                engine.boot_reconcile().await?;
            }
        }
        for engine in &engines {
            engine.spawn_maintenance();
        }
        Ok(())
    }

    fn rehydrate_from_catalog(&self) -> Result<(), StatsError> {
        for (name, schema) in self.catalog.list_all()? {
            let policy = self.catalog.get_policy(&name)?;
            // Do NOT spawn the maintenance task here — `bootstrap_maintenance`
            // runs boot reconcile first, then spawns for the rehydrated set.
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
    /// `true` for a runtime `register_table` (no boot reconcile needed for a
    /// fresh dir), `false` during boot rehydrate (where `bootstrap_maintenance`
    /// reconciles first, then spawns).
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
            engine.spawn_maintenance();
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

    fn ensure_log_namespace_registered(&self) -> Result<(), StatsError> {
        if self.catalog.contains(LOG_NAMESPACE_NAME) {
            // Engine already built by rehydrate.
            return Ok(());
        }
        let schema = log_registered_schema();
        resolve_key_column(&schema)?;
        let stored = with_implicit_seq(schema);
        self.catalog.register_or_evolve(
            LOG_NAMESPACE_NAME,
            stored.clone(),
            StoragePolicy::default(),
            |existing| Ok(existing.clone()),
        )?;
        // No maintenance spawn here — `bootstrap_maintenance` handles the log
        // namespace alongside the rehydrated set (boot reconcile first).
        self.build_engine(LOG_NAMESPACE_NAME, stored, StoragePolicy::default(), false)?;
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
    /// schema (WITH implicit `seq`). On re-register an empty policy is kept.
    pub fn register_table(
        &self,
        name: &str,
        schema: Schema,
        policy: StoragePolicy,
    ) -> Result<Schema, StatsError> {
        // Validate the name (and fence the `log` dir special-case) first.
        self.namespace_dir(name)?;
        resolve_key_column(&schema)?;
        let stored = with_implicit_seq(schema);

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
    /// seq (or `-1`). `columns` are the five non-seq log columns in registered
    /// order, prepared by the caller outside the lock.
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
            let paths = engine.query_snapshot();
            let provider = NamespaceProvider::build(arrow_schema, &paths)
                .map_err(|e| StatsError::Internal(format!("build provider {:?}: {e}", ns.name)))?;
            out.push(RegisteredProvider {
                name: ns.name,
                provider,
            });
        }
        Ok(out)
    }

    /// Snapshot the reserved `log` namespace's arrow schema + sealed-segment
    /// paths for a FetchLogs read.
    pub fn log_query_snapshot(&self) -> Result<(SchemaRef, Vec<String>), StatsError> {
        let engine = self.require_engine(LOG_NAMESPACE_NAME)?;
        Ok((Arc::clone(engine.arrow_schema()), engine.query_snapshot()))
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
    /// `flush -> compact (planner-drained, or forced L0->L1) -> sync -> evict`.
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
    fn register_returns_store_form_with_seq() {
        let store = mem_store();
        let effective = store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        assert_eq!(effective, with_implicit_seq(worker_schema()));
        assert_eq!(effective.columns[0].name, "seq");
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
        assert_eq!(eff, with_implicit_seq(full));
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
        assert_eq!(
            eff.column_names(),
            vec!["seq", "worker_id", "mem_bytes", "timestamp_ms", "note"]
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
}
