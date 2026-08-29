//! The table manager: the only public control surface for tables.
//!
//! [`TableManager`] owns the live registry, routes appends, hands out durable
//! state controllers, publishes read snapshots, leases maintenance work, and
//! drops tables. Everything above it — the RPC services, the forwarder, the
//! debug admin surface — reaches a table through this type. Nothing above it
//! constructs a [`TableController`] or talks to a [`TableStateStore`].
//!
//! Each table has two deliberately different paths:
//!
//! - the ingest fast path takes the RAM buffer's short lock to validate, assign
//!   sequence numbers, and append. It never waits on the controller mailbox and
//!   never waits on durable I/O.
//! - the controller path serializes durable state transitions and republishes
//!   the immutable [`TableSnapshot`] readers observe.

mod controller;
pub mod query_view;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{watch, RwLock};

use crate::errors::StatsError;
use crate::indices::cache::IndexCache;
use crate::indices::IndexRegistry;
use crate::maintenance::MaintenanceLimits;
use crate::store::catalog::state_store::{StoredTableState, TableStateStore};
use crate::store::catalog::Catalog;
use crate::store::namespace::Namespace;
use crate::store::object_store::ObjectStore;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{AlignedBatch, Schema};
use crate::store::store::{ServeMode, LOG_NAMESPACE_DIR, LOG_NAMESPACE_NAME};
use crate::store::table_state::{TableSnapshot, WriterFence};

pub use controller::{
    file_sha256, local_artifacts, object_segment_is_query_visible, MaintenanceLease,
    ObjectPersistence, TableController, WrittenObject,
};

/// Bounded budget for stopping and joining a table's background tasks during a
/// live lifecycle transition (re-register replacement, drop). Runs inside the
/// RPC's `spawn_blocking` worker, so it must not block long: a task that misses
/// this window is aborted rather than wedging the worker. Distinct from the
/// process-shutdown drain budget passed to [`TableManager::shutdown`] at SIGTERM.
pub const TABLE_LIFECYCLE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub struct TableManager {
    data_dir: Option<PathBuf>,
    mode: ServeMode,
    catalog: Arc<Catalog>,
    object_store: Option<Arc<dyn ObjectStore>>,
    legacy_object_store: Option<Arc<dyn ObjectStore>>,
    /// Durable state authority for object-backed tables. Absent when no remote
    /// object store is configured.
    object_state_store: Option<Arc<dyn TableStateStore>>,
    fence: WriterFence,
    runtimes: Mutex<HashMap<String, Arc<Namespace>>>,
    /// One controller per table, retained across engine rebuilds so a re-register
    /// never loses an owed publication or a claimed fence.
    controllers: Mutex<HashMap<String, Arc<TableController>>>,
    /// Serializes the complete catalog-and-engine registration lifecycle per
    /// table. Concurrent first registrations must not build and displace
    /// separate engines.
    registration_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Process-wide query-visibility lock, held for the paths a scan may open
    /// that some other operation may unlink.
    ///
    /// It exists for path-based liveness. An object-backed table no longer
    /// depends on it: its reads plan from an immutable `TableSnapshot`, its data
    /// and index objects are immutable, and retirement never unlinks a
    /// referenced file. The remaining holders that do need it are:
    ///
    /// - legacy tables, whose query view is the set of files currently on disk;
    /// - legacy compaction (`commit_swap`) and eviction, which unlink or rename
    ///   the very files a snapshot named;
    /// - `DropTable`, which deletes a table's directory;
    /// - a table still importing its version-0 history, whose query view is the
    ///   set of files on disk until that import activates. Activation moves it
    ///   onto the snapshot path, and retirement removes the imported files from
    ///   the catalog entirely.
    ///
    /// Object-backed reads still take the READ side because a single query
    /// registers every live table, legacy ones included. The lock is deleted
    /// once no consumer plans from paths and no operation unlinks a referenced
    /// file.
    ///
    /// ONE shared instance for the whole process (queries are cross-table, so
    /// the drain must be global). Cloned into each table runtime so its
    /// maintenance work takes `.blocking_write()` inside its `spawn_blocking`.
    ///
    /// `tokio::sync::RwLock` is WRITE-preferring (a new reader waits behind a
    /// pending writer). It upholds the safety invariant (a writer never proceeds
    /// while any reader holds the lock, so no scan opens a file mid-unlink), and
    /// write-preference is safer here — it cannot starve compaction/eviction
    /// under a steady query stream.
    query_visibility: Arc<RwLock<()>>,
    indices: Arc<IndexRegistry>,
    /// Process-wide maintenance concurrency limits, handed to every runtime.
    limits: Arc<MaintenanceLimits>,
    /// The maintenance scheduler's wake signal. Held here because runtimes are
    /// built before the scheduler starts and must already carry it.
    maintenance_wake: Arc<tokio::sync::Notify>,
}

impl TableManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_dir: Option<PathBuf>,
        mode: ServeMode,
        catalog: Arc<Catalog>,
        object_store: Option<Arc<dyn ObjectStore>>,
        legacy_object_store: Option<Arc<dyn ObjectStore>>,
        object_state_store: Option<Arc<dyn TableStateStore>>,
        fence: WriterFence,
        index_cache_mb: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            data_dir,
            mode,
            catalog,
            object_store,
            legacy_object_store,
            object_state_store,
            fence,
            runtimes: Mutex::new(HashMap::new()),
            controllers: Mutex::new(HashMap::new()),
            registration_locks: Mutex::new(HashMap::new()),
            query_visibility: Arc::new(RwLock::new(())),
            indices: Arc::new(IndexRegistry::new(Arc::new(IndexCache::new(
                index_cache_mb,
            )))),
            limits: MaintenanceLimits::new(),
            maintenance_wake: Arc::new(tokio::sync::Notify::new()),
        })
    }

    pub fn query_visibility(&self) -> &Arc<RwLock<()>> {
        &self.query_visibility
    }

    pub fn indices(&self) -> &Arc<IndexRegistry> {
        &self.indices
    }

    pub fn mode(&self) -> ServeMode {
        self.mode
    }

    pub fn maintenance_limits(&self) -> &Arc<MaintenanceLimits> {
        &self.limits
    }

    pub fn maintenance_wake(&self) -> &Arc<tokio::sync::Notify> {
        &self.maintenance_wake
    }

    /// The on-disk subdirectory for `name`, without validating it. Callers hold
    /// an already-validated or registered name; `log` maps to `{data_dir}/log`.
    pub fn table_dir(&self, name: &str) -> Option<PathBuf> {
        self.data_dir.as_ref().map(|dir| {
            if name == LOG_NAMESPACE_NAME {
                dir.join(LOG_NAMESPACE_DIR)
            } else {
                dir.join(name)
            }
        })
    }

    /// The lock serializing `name`'s whole registration lifecycle.
    pub fn registration_lock(&self, name: &str) -> Arc<Mutex<()>> {
        let mut locks = self.registration_locks.lock().unwrap();
        Arc::clone(
            locks
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    /// The durable-state controller for `name`, created on first use.
    ///
    /// A controller exists whether or not the table has a live runtime, because
    /// registration commits a table's first revision before its engine is built.
    pub fn controller(&self, name: &str) -> Arc<TableController> {
        let mut controllers = self.controllers.lock().unwrap();
        if let Some(controller) = controllers.get(name) {
            return Arc::clone(controller);
        }
        let objects = match (
            self.table_dir(name),
            self.object_store.clone(),
            self.legacy_object_store.clone(),
            self.object_state_store.clone(),
        ) {
            (Some(table_dir), Some(store), Some(legacy_store), Some(state_store)) => {
                Some(ObjectPersistence {
                    table_dir,
                    store,
                    legacy_store,
                    state_store,
                })
            }
            _ => None,
        };
        let controller = TableController::start(
            name.to_string(),
            Arc::clone(&self.catalog),
            objects,
            self.fence,
        );
        controllers.insert(name.to_string(), Arc::clone(&controller));
        controller
    }

    /// Seed `name`'s controller with the state a bootstrap claim selected.
    pub fn adopt_claimed_state(&self, name: &str, claimed: StoredTableState) {
        self.controller(name).adopt_claimed(claimed);
    }

    /// Stop accepting writes for `name` until a restart recovers it.
    pub fn mark_unready(&self, name: &str, reason: &str) {
        self.controller(name).mark_unready(reason);
    }

    pub fn get(&self, name: &str) -> Option<Arc<Namespace>> {
        self.runtimes.lock().unwrap().get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.runtimes.lock().unwrap().contains_key(name)
    }

    /// How many tables have a live runtime.
    pub fn table_count(&self) -> usize {
        self.runtimes.lock().unwrap().len()
    }

    /// Every live table runtime.
    pub fn runtimes(&self) -> Vec<Arc<Namespace>> {
        self.runtimes.lock().unwrap().values().cloned().collect()
    }

    /// The live runtime for `name`, or `NamespaceNotFound`.
    pub fn require(&self, name: &str) -> Result<Arc<Namespace>, StatsError> {
        self.get(name).ok_or_else(|| {
            StatsError::NamespaceNotFound(format!("namespace {name:?} is not registered"))
        })
    }

    /// The live runtime for `name`, rejected when another writer owns the
    /// table's durable state. A fenced table serves reads and takes no writes
    /// until a restart re-claims it.
    pub fn require_writable(&self, name: &str) -> Result<Arc<Namespace>, StatsError> {
        let runtime = self.require(name)?;
        if !runtime.write_ready() {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {name:?} is fenced by another writer and is not accepting writes"
            )));
        }
        Ok(runtime)
    }

    /// Build (or rebuild) the runtime for `name`, replacing any prior one.
    ///
    /// A replacement stops and joins the prior runtime's background tasks before
    /// the new one adopts the same directory, so the old tasks cannot flush,
    /// evict, or upsert concurrently. In-memory tables spawn no background tasks,
    /// so replacing the `Arc` is enough. The maintenance scheduler picks the
    /// replacement up on its next poll.
    pub fn register(
        &self,
        name: &str,
        schema: Schema,
        policy: StoragePolicy,
    ) -> Result<Arc<Namespace>, StatsError> {
        let table_dir = self.table_dir(name);
        // A disk-backed re-register always runs under a runtime: it arrives via
        // the registration RPC's `spawn_blocking` worker. The boot rehydrate path
        // has no prior runtime, so `block_on` never fires there.
        if table_dir.is_some() {
            if let Some(prior) = self.get(name) {
                tokio::runtime::Handle::current()
                    .block_on(prior.shutdown(TABLE_LIFECYCLE_SHUTDOWN_TIMEOUT));
            }
        }
        let runtime = Namespace::open(
            name,
            schema,
            table_dir,
            Arc::clone(&self.catalog),
            Arc::clone(&self.query_visibility),
            Arc::clone(&self.indices),
            Arc::clone(&self.limits),
            Arc::clone(&self.maintenance_wake),
            self.controller(name),
            policy,
        )?;
        self.runtimes
            .lock()
            .unwrap()
            .insert(name.to_string(), Arc::clone(&runtime));
        Ok(runtime)
    }

    /// Append one validated batch to `name`'s RAM buffer and return its
    /// durability target.
    ///
    /// This is the ingest fast path: it takes the buffer's short lock only. A
    /// busy or blocked controller cannot delay it.
    pub fn append(&self, name: &str, aligned: &AlignedBatch) -> Result<i64, StatsError> {
        Ok(self.require_writable(name)?.append_aligned_batch(aligned))
    }

    /// The latest published state for `name`, or `None` before its first durable
    /// transition or when the table is not object-backed.
    pub fn snapshot(&self, name: &str) -> Option<Arc<TableSnapshot>> {
        self.controllers
            .lock()
            .unwrap()
            .get(name)
            .and_then(|controller| controller.snapshot())
    }

    /// Follow `name`'s published state.
    pub fn watch_snapshot(&self, name: &str) -> watch::Receiver<Option<Arc<TableSnapshot>>> {
        self.controller(name).watch_snapshot()
    }

    /// Publish the revision a synchronous caller committed, so direct readers
    /// see it before the RPC returns.
    pub async fn publish(&self, name: &str) -> Result<Arc<TableSnapshot>, StatsError> {
        let controller = self.controller(name);
        if !controller.is_object_backed() {
            return Err(StatsError::SchemaValidation(
                "object-backed table specifications require a configured remote_log_dir"
                    .to_string(),
            ));
        }
        Ok(controller.publish_state().await?)
    }

    /// Take a lease over `inputs` for one compaction of `name`.
    pub fn begin_compaction(
        &self,
        name: &str,
        inputs: Vec<String>,
    ) -> Result<MaintenanceLease, StatsError> {
        self.controller(name).begin_compaction(inputs)
    }

    /// Run one full maintenance cycle for `name`.
    pub async fn maintain(&self, name: &str, force_compact_l0: bool) -> Result<(), StatsError> {
        self.require(name)?.run_maintenance(force_compact_l0).await
    }

    /// Remove `name`'s runtime and controller from the registry.
    ///
    /// Returns them so the caller can stop the runtime and publish the table's
    /// tombstone before the local rows and files disappear.
    pub fn take(&self, name: &str) -> (Option<Arc<Namespace>>, Option<Arc<TableController>>) {
        let runtime = self.runtimes.lock().unwrap().remove(name);
        let controller = self.controllers.lock().unwrap().remove(name);
        (runtime, controller)
    }

    /// Aggregate in-RAM accounting across live tables for the periodic
    /// diagnostics line.
    pub fn memory_summary(&self) -> crate::store::types::MemorySummary {
        let runtimes = self.runtimes();
        let mut ram_bytes = 0i64;
        let mut chunks = 0usize;
        for runtime in &runtimes {
            let (bytes, count) = runtime.memory_summary();
            ram_bytes += bytes;
            chunks += count;
        }
        crate::store::types::MemorySummary {
            namespaces: runtimes.len(),
            ram_bytes,
            chunks,
        }
    }

    /// Cooperatively shut down every table's background tasks.
    ///
    /// Tables shut down concurrently so the total drain is bounded by the
    /// per-table timeout, not its product with the table count.
    pub async fn shutdown(&self, per_table_timeout: Duration) {
        let runtimes = self.runtimes();
        futures::future::join_all(
            runtimes
                .iter()
                .map(|runtime| runtime.shutdown(per_table_timeout)),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Int64Array, StringArray};
    use buffa::MessageField;
    use sha2::{Digest, Sha256};

    use crate::proto::finelog::stats::{
        ColumnType, OperatingPolicy, SourceLayout, TableSpec as ProtoTableSpec,
    };
    use crate::store::catalog::object_state_store::ObjectTableStateStore;
    use crate::store::object_store::{
        build_remote_object_store, CachedObjectStore, LegacyObjectStore,
    };
    use crate::store::schema::{schema_to_proto_owned, with_implicit_seq, AlignedBatch, Column};
    use crate::test_support::{
        FaultAction, FaultGate, FaultInjectingObjectStore, ObjectFault, ObjectOp, ObjectPattern,
    };

    const TABLE: &str = "iris.worker";

    fn worker_schema() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        ))
    }

    fn aligned(rows: i64) -> AlignedBatch {
        AlignedBatch {
            arrays: vec![
                Arc::new(StringArray::from(
                    (0..rows).map(|row| format!("w{row}")).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    (0..rows).map(|row| 1000 + row).collect::<Vec<_>>(),
                )),
            ],
            fields: vec![
                arrow::datatypes::Field::new("worker_id", arrow::datatypes::DataType::Utf8, false),
                arrow::datatypes::Field::new(
                    "timestamp_ms",
                    arrow::datatypes::DataType::Int64,
                    false,
                ),
            ],
            num_rows: rows as usize,
            byte_size: 16 * rows,
        }
    }

    fn register_versioned_spec(catalog: &Catalog) {
        let spec = ProtoTableSpec {
            version: Some(1),
            logical_schema: MessageField::some(schema_to_proto_owned(&worker_schema())),
            source_layout: MessageField::some(SourceLayout::default()),
            operating_policy: MessageField::some(OperatingPolicy::default()),
            ..Default::default()
        };
        let hash: [u8; 32] =
            Sha256::digest(crate::store::table_spec::canonical_json_bytes(&spec).unwrap()).into();
        catalog
            .register_table_spec(TABLE, &spec, &hash, false)
            .unwrap();
    }

    /// Ingest never queues behind a durable state transition: appends keep
    /// assigning sequence numbers while the controller is parked inside a
    /// publication that cannot finish.
    #[tokio::test]
    async fn appends_proceed_while_the_controller_is_blocked_publishing() {
        let root = crate::test_support::unique_dir("manager_append_fast_path");
        let remote_dir = root.join("remote");
        let data_dir = root.join("data");
        std::fs::create_dir_all(&remote_dir).unwrap();
        std::fs::create_dir_all(data_dir.join(TABLE)).unwrap();
        let provider = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let object_store =
            Arc::new(CachedObjectStore::new(Arc::new(provider.clone()), data_dir.clone()).unwrap())
                as Arc<dyn ObjectStore>;
        // The controller parks inside the HEAD swap its publication performs.
        let publication = FaultGate::new();
        let faults = FaultInjectingObjectStore::new(Arc::clone(&object_store));
        faults.arm(ObjectFault::new(
            ObjectOp::CompareAndSwap,
            ObjectPattern::EndsWith("HEAD.json".to_string()),
            FaultAction::Park(Arc::clone(&publication)),
        ));
        let state_store = Arc::new(ObjectTableStateStore::new(
            Arc::clone(&faults) as Arc<dyn ObjectStore>
        ));
        let catalog = Arc::new(Catalog::open(Some(&data_dir)).unwrap());
        register_versioned_spec(&catalog);
        let manager = TableManager::new(
            Some(data_dir),
            ServeMode::Shadow,
            Arc::clone(&catalog),
            Some(object_store),
            Some(Arc::new(LegacyObjectStore::new(&provider))),
            Some(Arc::clone(&state_store) as Arc<dyn TableStateStore>),
            WriterFence::new(11),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
        );
        manager
            .register(TABLE, worker_schema(), StoragePolicy::default())
            .unwrap();

        let publishing = {
            let manager = Arc::clone(&manager);
            tokio::spawn(async move { manager.publish(TABLE).await.map(|_| ()) })
        };
        publication.entered().await;

        // The controller is parked mid-commit. Ingest is a different path, so it
        // completes rather than queueing behind the publication.
        let first = tokio::time::timeout(
            Duration::from_secs(5),
            tokio::task::spawn_blocking({
                let manager = Arc::clone(&manager);
                move || manager.append(TABLE, &aligned(4))
            }),
        )
        .await
        .expect("append must not wait for the blocked controller")
        .unwrap()
        .unwrap();
        let second = manager.append(TABLE, &aligned(2)).unwrap();
        assert_eq!(first, 4);
        assert_eq!(second, 6);

        publication.release();
        publishing.await.unwrap().unwrap();
        assert!(manager.snapshot(TABLE).is_some());
    }
}
