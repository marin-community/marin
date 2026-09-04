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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use buffa::{Message, MessageField, MessageView};
use clap::ValueEnum;

use crate::errors::StatsError;
use crate::indices::IndexRegistry;
use crate::ingestion_policy::IngestionBatchSource;
use crate::levanter_metrics_policy::{
    has_managed_levanter_layout, initial_managed_table_spec, managed_table_spec_for,
    LEVANTER_METRICS_NAMESPACE,
};
use crate::maintenance::scheduler::MaintenanceScheduler;
use crate::policies::{
    physical_partition_policy_for, schema_for_namespace, segment_indexes_enabled_for,
    storage_policy_for, PolicyRegistry,
};
use crate::proto::finelog::stats::{ColumnType, L0Mode, SchemaView, TableSpecView};
use crate::query::provider::NamespaceProvider;
use crate::query::RegisteredProvider;
use crate::store::catalog::{Catalog, PublishedObjectSegment, RegisteredNamespace, SpecLifecycle};
use crate::store::ipc::decode_one_record_batch;
use crate::store::namespace_name::validate_namespace_name;
use crate::store::object_store::{
    build_remote_object_store, CachedObjectStore, LegacyObjectStore, ObjectStore, RemoteObjectStore,
};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{
    merge_managed_schema, merge_schemas, resolve_key_column, schema_from_proto_view,
    schema_to_proto_owned, stamp_cluster_column, stored_form, validate_and_align_batch,
    validate_and_align_forwarded_batch, validate_index_policies, AlignedBatch, Column, Schema,
    IMPLICIT_SEQ_COLUMN, MAX_WRITE_ROWS_BYTES, MAX_WRITE_ROWS_ROWS,
};
use crate::store::state_store::object::ObjectTableStateStore;
use crate::store::table::query_view::SegmentObjectMap;
use crate::store::table::{TableManager, TABLE_LIFECYCLE_SHUTDOWN_TIMEOUT};
use crate::store::table_spec::ValidatedTableSpec;
use crate::store::table_state::{ArtifactReferences, TableRevision, TableSnapshot, WriterFence};
use crate::store::types::NamespaceStats;
use crate::telemetry_policy::{TelemetryRootWriteMode, TELEMETRY_NAMESPACE};

/// The privileged log namespace name, which is also its on-disk subdirectory.
pub const LOG_NAMESPACE_NAME: &str = "log";
const STORE_LOCK_FILENAME: &str = ".finelog-store.lock";

fn writer_epoch() -> Result<u64, StatsError> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            StatsError::Internal(format!("system clock precedes Unix epoch: {error}"))
        })?
        .as_nanos() as u64;
    Ok(nanos ^ u64::from(std::process::id()))
}

/// A decorator applied to each object store the composition root builds — the
/// cached object store and the legacy archive alike — so a caller can observe or
/// interfere with every object operation the store makes. Production composes no
/// decorator.
pub type ObjectStoreInterposer =
    Arc<dyn Fn(Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> + Send + Sync>;

/// Result of appending one schema-compatible federated batch.
pub struct ForwardedWrite {
    pub rows_written: i64,
    pub persisted_targets: Vec<(String, i64)>,
    pub ignored_columns: Vec<String>,
}

pub struct VersionedRegistration {
    pub schema: Schema,
    pub policy: StoragePolicy,
    pub spec_lifecycle: SpecLifecycle,
    pub object_backed: bool,
}

#[derive(Clone, Copy)]
enum SchemaRegistration {
    Additive,
    Managed,
}

#[derive(Clone, Copy)]
enum BatchAlignment {
    Strict,
    ForwardCompatible,
}

impl BatchAlignment {
    fn align(
        self,
        batch: &RecordBatch,
        schema: &Schema,
    ) -> Result<(AlignedBatch, Vec<String>), StatsError> {
        match self {
            Self::Strict => {
                validate_and_align_batch(batch, schema).map(|aligned| (aligned, Vec::new()))
            }
            Self::ForwardCompatible => validate_and_align_forwarded_batch(batch, schema),
        }
    }
}

/// Registered schema for the privileged `log` namespace; `key_column = "key"`.
///
/// The original five columns (key/source/data/epoch_ms/level) are non-nullable.
/// `cluster` is a later **additive, nullable** column: the writer-supplied origin
/// cluster of each push (trusted — writers are authenticated), which namespaces
/// logs a global finelog collects from many federated clusters. It is nullable
/// because segments written before the column existed null-fill it on read,
/// which is also why `merge_schemas` adopts any new column as nullable.
pub fn log_registered_schema() -> Schema {
    Schema::new(
        vec![
            // The job and task sit mid-key, so `key LIKE '%<job>%'` is opaque to
            // the sort's min/max statistics and needs a trigram index of its own.
            Column::new("key", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
            Column::new("source", ColumnType::COLUMN_TYPE_STRING, false),
            // Substring-searched via contains()/LIKE.
            Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
            Column::new("epoch_ms", ColumnType::COLUMN_TYPE_INT64, false),
            Column::new("level", ColumnType::COLUMN_TYPE_INT32, false),
            Column::new("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        ],
        "key",
    )
}

/// One consistent view of a namespace's sealed local segments: the Arrow schema,
/// resolved key and known bounds, paths, and lowest `seq`. Captured under one hold
/// of the insertion lock so all segment metadata describes exactly the same paths.
pub struct NamespaceSnapshot {
    pub schema: SchemaRef,
    pub exact_postings_policy: BTreeMap<String, Vec<String>>,
    pub key_column: String,
    pub paths: Vec<String>,
    pub key_bounds: BTreeMap<String, (i64, i64)>,
    pub seq_bounds: BTreeMap<String, (i64, i64)>,
    pub partitions: BTreeMap<String, crate::partition_policy::SegmentPartition>,
    pub min_seq: Option<i64>,
    pub indices: Arc<IndexRegistry>,
    pub artifacts: crate::indices::SegmentArtifacts,
    /// The immutable objects each path resolves to, and the store that
    /// localizes them. Absent for a legacy table, whose paths are already files.
    pub sources: Option<(Arc<dyn ObjectStore>, SegmentObjectMap)>,
}

/// What a store may do to what it opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ServeMode {
    /// A deployed server: runs per-namespace maintenance.
    Live,
    /// A rehearsal against a copy of a real store: runs no maintenance, so
    /// nothing compacts, evicts, rewrites a segment layout, or redundancy-drops
    /// a covered segment (which deletes its archived object).
    Shadow,
}

/// The application composition root the RPC handlers sit on.
///
/// It constructs the object store, the durable state stores, and the
/// [`TableManager`], then delegates every table operation to that manager. The
/// catalog owns the persistent registry and segment rows; the metadata RPCs stay
/// on it. Registration policy — schema merge rules, the privileged `log`
/// namespace, ingestion routing — stays here.
pub struct Store {
    data_dir: Option<PathBuf>,
    remote_log_dir: String,
    mode: ServeMode,
    catalog: Arc<Catalog>,
    object_store: Option<Arc<dyn ObjectStore>>,
    /// Durable state authority for object-backed tables. Absent when no remote
    /// object store is configured.
    object_state_store: Option<Arc<ObjectTableStateStore>>,
    fence: WriterFence,
    /// The only table control surface. Owns the live registry, the per-table
    /// durable-state controllers, and the query-visibility lock.
    tables: Arc<TableManager>,
    /// The process's only maintenance cadence owner.
    scheduler: Arc<MaintenanceScheduler>,
    policies: PolicyRegistry,
    _store_lock: Option<File>,
}

pub(crate) fn acquire_exclusive_store_lock(data_dir: &Path) -> Result<File, StatsError> {
    let path = data_dir.join(STORE_LOCK_FILENAME);
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|error| {
            StatsError::Internal(format!("open store lock {}: {error}", path.display()))
        })?;
    // SAFETY: `file` owns this valid descriptor for the duration of the call and
    // retains it until the returned lock guard is dropped.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result != 0 {
        return Err(StatsError::Internal(format!(
            "Finelog store is open in another process: {}",
            data_dir.display()
        )));
    }
    Ok(file)
}

fn decode_bounded_write_batch(arrow_ipc: &[u8]) -> Result<RecordBatch, StatsError> {
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
    Ok(batch)
}

impl Store {
    /// Construct the store: create `data_dir`, rehydrate the live registry +
    /// per-namespace engines from the catalog, and ensure the privileged `log`
    /// namespace exists.
    ///
    /// `remote_log_dir` configures the per-table offload target (empty disables
    /// sync). Pass it through to each table runtime.
    pub fn new(
        data_dir: Option<PathBuf>,
        remote_log_dir: String,
        index_cache_mb: usize,
        mode: ServeMode,
    ) -> Result<Store, StatsError> {
        Self::open(
            data_dir,
            remote_log_dir,
            index_cache_mb,
            mode,
            TelemetryRootWriteMode::SemanticOnly,
            None,
            None,
        )
    }

    /// Construct the store with every dial exposed. `interpose` wraps each
    /// object store as it is built; the failure-scenario tests use it to fail,
    /// delay, or count individual object operations against an otherwise
    /// complete composition.
    pub fn open(
        data_dir: Option<PathBuf>,
        remote_log_dir: String,
        index_cache_mb: usize,
        mode: ServeMode,
        telemetry_root_write_mode: TelemetryRootWriteMode,
        object_cache_bytes: Option<u64>,
        interpose: Option<ObjectStoreInterposer>,
    ) -> Result<Store, StatsError> {
        let startup_started = Instant::now();
        if let Some(dir) = &data_dir {
            std::fs::create_dir_all(dir).map_err(|e| {
                StatsError::Internal(format!("create data_dir {}: {e}", dir.display()))
            })?;
        }
        let store_lock = data_dir
            .as_deref()
            .map(acquire_exclusive_store_lock)
            .transpose()?;
        let catalog_open_started = Instant::now();
        let catalog = Arc::new(Catalog::open(data_dir.as_deref())?);
        let provider = if data_dir.is_some() {
            build_remote_object_store(&remote_log_dir)?
        } else {
            None
        };
        let legacy_object_store = provider
            .as_ref()
            .map(LegacyObjectStore::new)
            .map(|store| Arc::new(store) as Arc<dyn ObjectStore>)
            .map(|store| match &interpose {
                Some(interpose) => interpose(store),
                None => store,
            });
        if let Some((base_url, backend)) = provider
            .as_ref()
            .and_then(RemoteObjectStore::scan_registration)
        {
            // Scans read uncached objects straight from the provider while the
            // cache fills behind them.
            crate::query::register_scan_object_store(&base_url, backend)?;
        }
        let query_visibility = Arc::new(tokio::sync::RwLock::new(()));
        let object_store = match (&provider, &data_dir) {
            (Some(provider), Some(root)) => Some(Arc::new(CachedObjectStore::new(
                Arc::new(provider.clone()),
                root.clone(),
                Arc::clone(&query_visibility),
                object_cache_bytes,
            )?) as Arc<dyn ObjectStore>),
            _ => None,
        };
        let object_store = match (object_store, &interpose) {
            (Some(store), Some(interpose)) => Some(interpose(store)),
            (store, _) => store,
        };
        let fence = WriterFence::new(writer_epoch()?);
        let object_state_store = object_store
            .clone()
            .map(|storage| Arc::new(ObjectTableStateStore::new(storage)));
        let catalog_open_ms = catalog_open_started.elapsed().as_millis() as u64;
        // Rebuild-from-disk catalog adoption. On a fresh boot over a log_dir an
        // earlier server populated, the sqlite sidecar is empty, so the disk
        // parquet layout + footers are the only record of the namespaces +
        // segments. The sentinel-gated, idempotent scan persists the recovered
        // `namespaces` + `segments` rows BEFORE `rehydrate_from_catalog` reads
        // them back to build the engines. No-op in in-memory mode + on the done
        // sentinel (subsequent boots). Object-backed recovery loads the published
        // state explicitly in `recover_tables`.
        let catalog_adoption_started = Instant::now();
        crate::store::adopt::ensure_catalog_adopted(data_dir.as_deref(), &catalog)?;
        let catalog_adoption_ms = catalog_adoption_started.elapsed().as_millis() as u64;
        let tables = TableManager::new(
            data_dir.clone(),
            mode,
            Arc::clone(&catalog),
            object_store.clone(),
            legacy_object_store,
            object_state_store.clone(),
            fence,
            index_cache_mb,
            query_visibility,
        );
        let scheduler = MaintenanceScheduler::new(Arc::clone(&tables));
        let store = Store {
            data_dir,
            remote_log_dir,
            mode,
            catalog,
            object_store,
            object_state_store,
            fence,
            tables,
            scheduler,
            policies: PolicyRegistry::new(telemetry_root_write_mode),
            _store_lock: store_lock,
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
        let namespaces = store.tables.table_count();
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

    /// Return the root telemetry sequence fence before this process accepts writes.
    pub fn telemetry_root_max_seq(&self) -> Result<i64, StatsError> {
        match self.tables.get(TELEMETRY_NAMESPACE) {
            Some(table) => Ok(table.stats().max_seq),
            None => Ok(self
                .catalog
                .aggregate_namespace_stats(TELEMETRY_NAMESPACE)?
                .max_seq),
        }
    }

    /// The table control surface, for tests that lease maintenance work
    /// directly instead of going through a maintenance cycle.
    #[cfg(any(test, feature = "test-util"))]
    pub fn tables(&self) -> &Arc<TableManager> {
        &self.tables
    }

    /// Fast-forward a migration's observation window, for tests that drive a
    /// table through retirement without waiting out `rollback_window_ms`.
    #[cfg(any(test, feature = "test-util"))]
    pub fn expire_migration_observation(&self, namespace: &str) -> Result<(), StatsError> {
        self.catalog.expire_migration_observation(namespace)
    }

    /// Start the maintenance scheduler. Called once after `new`, before serving.
    ///
    /// The scheduler is the process's only cadence owner: it polls the registry
    /// and dispatches flush, compaction, publication, backfill, migration, and
    /// GC work for every table.
    pub fn bootstrap_maintenance(&self) {
        self.scheduler.start();
    }

    /// Load and claim every table's durable state before the server accepts
    /// traffic.
    ///
    /// Returns the number of object-backed tables whose local projection was
    /// rebuilt from their durable state.
    pub async fn recover_tables(&self) -> Result<usize, StatsError> {
        let loaded = self.recover_object_tables().await?;
        self.register_managed_table_specs().await?;
        Ok(loaded)
    }

    /// Register server-owned physical layouts after remote recovery and writer
    /// claim, but before the maintenance scheduler can select work under the
    /// superseded definition.
    async fn register_managed_table_specs(&self) -> Result<(), StatsError> {
        if self.mode != ServeMode::Live || !self.catalog.contains(LEVANTER_METRICS_NAMESPACE) {
            return Ok(());
        }
        let controller = self.tables.controller(LEVANTER_METRICS_NAMESPACE);
        if !controller.writes_ready() {
            return Ok(());
        }
        let lifecycle = self.catalog.spec_lifecycle(LEVANTER_METRICS_NAMESPACE)?;
        if let Some(desired) = lifecycle.desired.as_ref() {
            if has_managed_levanter_layout(desired) {
                controller.publish_owed().await?;
                return Ok(());
            }
            return Err(StatsError::SchemaConflict(
                "levanter.metrics has a table-spec migration in progress; finish or abort it before applying the managed physical layout"
                    .to_string(),
            ));
        }
        let Some(active) = lifecycle.active.as_ref() else {
            return Ok(());
        };
        let cache_policy = self.catalog.get_policy(LEVANTER_METRICS_NAMESPACE)?;
        let Some(managed) =
            managed_table_spec_for(LEVANTER_METRICS_NAMESPACE, active, &cache_policy)?
        else {
            return Ok(());
        };
        if managed.l0_mode != L0Mode::L0_MODE_OBJECT_STORE {
            return Ok(());
        }
        let registration = self.register_versioned_table(LEVANTER_METRICS_NAMESPACE, managed)?;
        self.publish_object_catalog(LEVANTER_METRICS_NAMESPACE)
            .await?;
        tracing::info!(
            namespace = LEVANTER_METRICS_NAMESPACE,
            active_version = registration.spec_lifecycle.active_version(),
            desired_version = registration.spec_lifecycle.desired_version(),
            "registered managed table layout"
        );
        Ok(())
    }

    /// Recover object-backed tables from their durable state before the server
    /// accepts traffic.
    ///
    /// Each head is loaded metadata-only and claimed under this process's
    /// fence; no data object is read, downloaded, or localized here. The local
    /// SQLite projection is rebuilt from the loaded state when it is behind
    /// HEAD, and rolls forward when it holds an unpublished revision this
    /// writer now owns. A tombstoned head deletes the local projection; a table
    /// whose claim fails stays unready and rejects writes.
    ///
    /// Returns the number of tables whose projection was rebuilt from HEAD.
    async fn recover_object_tables(&self) -> Result<usize, StatsError> {
        let Some(state_store) = &self.object_state_store else {
            return Ok(0);
        };
        let mut loaded_count = 0;
        for head in state_store.list().await? {
            let namespace = head.table;
            validate_namespace_name(&namespace, self.data_dir.as_deref())?;
            if head.tombstoned {
                self.discard_tombstoned_table(&namespace).await?;
                continue;
            }
            let Some(selected) = state_store.load(&namespace).await? else {
                continue;
            };
            let claimed = match state_store
                .claim_writer(&namespace, self.fence, &selected)
                .await
            {
                Ok(claimed) => claimed,
                Err(error) => {
                    self.mark_table_unready(&namespace, &error.to_string());
                    continue;
                }
            };
            let state = claimed.catalog.clone();
            let high_water = state.persisted_high_water.unwrap_or(0);
            self.tables.adopt_claimed_state(&namespace, claimed);
            if self.recover_claimed_table(&namespace, state).await? {
                loaded_count += 1;
            }
            // The durable state's high-water mark can exceed the max seq in its
            // published segments (a legacy import excludes archive-only rows;
            // retirement deletes legacy rows). Seed the allocator past it so a
            // recovered writer never reissues a sequence number.
            if high_water > 0 {
                if let Some(runtime) = self.tables.get(&namespace) {
                    runtime.raise_next_seq_floor(high_water + 1);
                }
            }
        }
        // A root with catalog history but no HEAD is an external anomaly (the
        // software never deletes HEAD). The table keeps serving from its local
        // projection and keeps taking writes; publication defers until an
        // operator restores HEAD, and the degraded flag says why.
        for namespace in state_store.headless_tables().await? {
            let reason = format!(
                "table {namespace:?} remote root holds catalog history but no HEAD; restore \
                 HEAD.json from the newest catalog document and restart"
            );
            tracing::error!(namespace, reason, "recovered a headless table");
            if self.catalog.contains(&namespace) {
                self.tables.controller(&namespace).mark_degraded(&reason);
            }
        }
        Ok(loaded_count)
    }

    /// The operator-attention reason for `namespace`, when its durable state
    /// is confused — e.g. catalog history without a HEAD. A degraded table
    /// keeps serving reads and accumulating local writes.
    pub fn table_degraded_reason(&self, namespace: &str) -> Option<String> {
        self.tables
            .get(namespace)
            .and_then(|_| self.tables.controller(namespace).degraded_reason())
    }

    /// Drop the local projection of a table whose durable state is tombstoned.
    async fn discard_tombstoned_table(&self, namespace: &str) -> Result<(), StatsError> {
        if !self.catalog.contains(namespace) {
            return Ok(());
        }
        let (runtime, _controller) = self.tables.take(namespace);
        if let Some(runtime) = runtime {
            runtime.shutdown(TABLE_LIFECYCLE_SHUTDOWN_TIMEOUT).await;
        }
        self.catalog.delete(namespace)?;
        tracing::info!(namespace, "discarded the projection of a deleted table");
        Ok(())
    }

    /// Stop accepting writes for `namespace` until a restart recovers it.
    fn mark_table_unready(&self, namespace: &str, reason: &str) {
        self.tables.mark_unready(namespace, reason);
    }

    /// Bring the local projection in line with the claimed durable state.
    ///
    /// Returns whether the projection was rebuilt. A local revision ahead of
    /// HEAD is this writer's own unpublished state under the same claimed
    /// fence, so it rolls forward through the table's next publication instead
    /// of being replaced.
    async fn recover_claimed_table(
        &self,
        namespace: &str,
        state: crate::proto::finelog::stats::NamespaceCatalog,
    ) -> Result<bool, StatsError> {
        let remote_revision = state.catalog_generation.unwrap_or(0);
        let local_revision = self.catalog.spec_lifecycle(namespace)?.catalog_generation;
        if local_revision > remote_revision {
            self.tables.controller(namespace).mark_publication_owed();
            tracing::info!(
                namespace,
                local_revision,
                remote_revision,
                "rolling an unpublished local revision forward under the claimed fence"
            );
            return Ok(false);
        }
        if local_revision == remote_revision {
            return Ok(false);
        }
        if state.active_table_spec_version.unwrap_or(0) == 0
            && state.desired_table_spec_version.unwrap_or(0) == 0
        {
            // An aborted first migration commits a revision whose active and
            // desired versions are both the legacy version 0 and retains no
            // TableSpec. The legacy path is the table's authority; the durable
            // state carries only the claimed fence.
            tracing::info!(
                namespace,
                remote_revision,
                "durable table state selects the legacy version; no object projection to rebuild"
            );
            return Ok(false);
        }
        let schema_spec = state
            .retained_table_specs
            .iter()
            .find(|spec| spec.version == state.active_table_spec_version)
            .or_else(|| {
                state
                    .retained_table_specs
                    .iter()
                    .find(|spec| spec.version == state.desired_table_spec_version)
            })
            .or_else(|| state.retained_table_specs.last())
            .ok_or_else(|| {
                StatsError::Internal(format!(
                    "table state for {namespace:?} has no retained TableSpec"
                ))
            })?;
        let schema_proto = schema_spec.logical_schema.as_option().ok_or_else(|| {
            StatsError::Internal(format!(
                "table state TableSpec for {namespace:?} has no logical schema"
            ))
        })?;
        let schema_bytes = schema_proto.encode_to_vec();
        let schema_view = SchemaView::decode_view(&schema_bytes).map_err(|error| {
            StatsError::Internal(format!(
                "decode recovered logical schema for {namespace:?}: {error}"
            ))
        })?;
        let schema = stored_form(schema_from_proto_view(&schema_view)?);
        let policy = schema_spec
            .operating_policy
            .as_option()
            .and_then(|operating| operating.local_cache.as_option())
            .map(StoragePolicy::from_proto_owned)
            .unwrap_or_default();
        let object_store = self.object_store.as_ref().ok_or_else(|| {
            StatsError::Internal("table state store configured without an object store".to_string())
        })?;
        let mut published_segments = HashMap::<String, PublishedObjectSegment>::new();
        for version in &state.version_segments {
            for segment in version
                .live_segments
                .iter()
                .chain(version.retired_segments.iter())
            {
                let source = segment.source.as_option().ok_or_else(|| {
                    StatsError::Internal(format!(
                        "table state segment for {namespace:?} has no source"
                    ))
                })?;
                let table_spec_version = segment
                    .table_spec_version
                    .unwrap_or(version.table_spec_version.unwrap_or(0));
                let reference = match crate::store::object_store::ObjectReference::try_from(source)
                {
                    Ok(reference) => reference,
                    Err(error) if table_spec_version == 0 => {
                        // A version-0 entry can point into the legacy archive
                        // rather than the object layout (an archive-only row
                        // retained until retirement). Its bytes are not
                        // recoverable from the object store; the entry is
                        // rollback bookkeeping, not a projection row.
                        tracing::debug!(
                            namespace,
                            %error,
                            "skipping a legacy-archive segment in the durable state"
                        );
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                reference.id.table_relative(namespace).ok_or_else(|| {
                    StatsError::Internal(format!(
                        "table state segment for {namespace:?} references another table"
                    ))
                })?;
                // Recovery names the file a later query would localize; it
                // never downloads or opens it.
                let local_path = object_store.planned_local_path(&reference.id)?;
                let partition = segment
                    .partition_json
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(|error| {
                        StatsError::Internal(format!(
                            "table state segment for {namespace:?} has invalid partition metadata: {error}"
                        ))
                    })?;
                let path = local_path.to_string_lossy().into_owned();
                published_segments
                    .entry(path.clone())
                    .or_insert_with(|| PublishedObjectSegment {
                        row: crate::store::types::SegmentRow {
                            namespace: namespace.to_string(),
                            path,
                            level: segment.level.unwrap_or(0),
                            min_seq: segment.min_seq.unwrap_or(0),
                            max_seq: segment.max_seq.unwrap_or(0),
                            row_count: segment.row_count.unwrap_or(0),
                            byte_size: i64::try_from(reference.version.byte_size)
                                .unwrap_or(i64::MAX),
                            created_at_ms: segment.created_at_ms.unwrap_or(0),
                            min_key_value: segment.min_key_value.clone(),
                            max_key_value: segment.max_key_value.clone(),
                            partition,
                            location: crate::store::types::SegmentLocation::Remote,
                        },
                        table_spec_version,
                        source: source.clone(),
                        artifacts: ArtifactReferences::from_catalog_segment(segment),
                        migration_backfill: segment.migration_backfill.unwrap_or(false),
                        migration_source_id: segment.migration_source_id.clone(),
                        migration_source_rows: segment.migration_source_rows,
                    });
            }
        }
        let published_segments: Vec<_> = published_segments.into_values().collect();
        // The claimed state stays authoritative whatever the projection does:
        // it is rebuildable, so a failure leaves the table unready rather than
        // undoing or blocking the durable revision.
        if let Err(error) = self.catalog.replace_with_published_snapshot(
            namespace,
            schema.clone(),
            policy.clone(),
            &state,
            &published_segments,
        ) {
            tracing::error!(
                namespace,
                %error,
                remote_revision,
                "rebuilding the local projection from the claimed state failed"
            );
            self.mark_table_unready(namespace, &error.to_string());
            return Ok(false);
        }
        let (prior, _controller) = self.tables.take(namespace);
        if let Some(prior) = prior {
            prior.shutdown(TABLE_LIFECYCLE_SHUTDOWN_TIMEOUT).await;
        }
        // The rebuilt runtime seeds its sequence high-water mark from the
        // projection of the claimed state.
        self.tables.register(namespace, schema, policy)?;
        Ok(true)
    }

    fn rehydrate_from_catalog(&self) -> Result<(), StatsError> {
        for (name, schema) in self.catalog.list_all()? {
            let policy = self.catalog.get_policy(&name)?;
            // `bootstrap_maintenance` starts all engines after startup loading is
            // complete.
            self.tables
                .register(&name, schema.clone(), policy.clone())?;
            self.catalog.insert_live(RegisteredNamespace {
                name,
                schema,
                policy,
            });
        }
        Ok(())
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
        let stored = stored_form(schema);
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
                    return Ok(Some(dir.join(LOG_NAMESPACE_NAME)));
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
        self.register_table_with(name, schema, policy, SchemaRegistration::Additive, None)
            .map(|(schema, _)| schema)
    }

    /// Register a server-owned schema whose derived layout is authoritative.
    ///
    /// Column evolution remains additive, but index and projection policy may
    /// be withdrawn by a rollout. Callers must supply the canonical server
    /// schema rather than a client declaration.
    pub fn register_managed_table(
        &self,
        name: &str,
        schema: Schema,
        policy: StoragePolicy,
    ) -> Result<Schema, StatsError> {
        self.register_table_with(name, schema, policy, SchemaRegistration::Managed, None)
            .map(|(schema, _)| schema)
    }

    pub fn register_versioned_table(
        &self,
        name: &str,
        validated: ValidatedTableSpec,
    ) -> Result<VersionedRegistration, StatsError> {
        if name == LEVANTER_METRICS_NAMESPACE && !has_managed_levanter_layout(&validated.spec) {
            return Err(StatsError::SchemaValidation(
                "levanter.metrics uses a server-owned run_id-partitioned physical layout"
                    .to_string(),
            ));
        }
        if validated.l0_mode == L0Mode::L0_MODE_OBJECT_STORE
            && self.remote_log_dir.trim().is_empty()
        {
            return Err(StatsError::SchemaValidation(
                "object-backed table specifications require a configured remote_log_dir"
                    .to_string(),
            ));
        }
        let schema = validated.schema.clone();
        let policy = validated.cache_policy.clone();
        let (schema, spec_lifecycle) = self.register_table_with(
            name,
            schema,
            policy.clone(),
            SchemaRegistration::Additive,
            Some(&validated),
        )?;
        let policy = self.get_policy(name)?;
        Ok(VersionedRegistration {
            schema,
            policy,
            spec_lifecycle: spec_lifecycle.expect("versioned registration returns status"),
            object_backed: validated.l0_mode == L0Mode::L0_MODE_OBJECT_STORE,
        })
    }

    /// Publish the revision a synchronous caller committed, so direct readers
    /// see it before the RPC returns.
    ///
    /// Returns the state HEAD selects after publication.
    pub async fn publish_object_catalog(
        &self,
        namespace: &str,
    ) -> Result<TableSnapshot, StatsError> {
        self.tables.require(namespace)?;
        Ok((*self.tables.publish(namespace).await?).clone())
    }

    /// A schema-changing legacy registration on a table governed by a table
    /// spec must evolve the spec, not just the local catalog: recovery rebuilds
    /// schemas from retained specs, so a spec-silent evolution would be dropped
    /// by the next cold boot while flushed objects still carry the column.
    ///
    /// Returns the table's next spec version — the retained spec with the
    /// merged schema — for a settled table, `None` when the table has no spec
    /// or the registration changes nothing, and an error while a spec
    /// transition is in flight (the caller retries after it settles; the
    /// forwarder holds its cursor on this refusal).
    fn spec_for_legacy_schema_evolution(
        &self,
        name: &str,
        stored: &Schema,
        registration: SchemaRegistration,
    ) -> Result<Option<ValidatedTableSpec>, StatsError> {
        let lifecycle = self.catalog.spec_lifecycle(name)?;
        if lifecycle.active.is_none() && lifecycle.desired.is_none() {
            return Ok(None);
        }
        let Some(existing) = self.catalog.get_live(name) else {
            return Ok(None);
        };
        let merged = match registration {
            SchemaRegistration::Additive => merge_schemas(&existing.schema, stored),
            SchemaRegistration::Managed => merge_managed_schema(&existing.schema, stored),
        }?;
        if merged == existing.schema {
            return Ok(None);
        }
        if lifecycle.desired.is_some() {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {name:?} is migrating between table specification versions; \
                 retry the schema change after the transition settles"
            )));
        }
        let next_version = lifecycle.active_version() + 1;
        let active = lifecycle
            .active
            .expect("a lifecycle without a transition has an active spec");
        let active_bytes = active.encode_to_vec();
        let active_view = TableSpecView::decode_view(&active_bytes).map_err(|error| {
            StatsError::Internal(format!("decode retained table spec for {name:?}: {error}"))
        })?;
        let logical_view = active_view.logical_schema.as_option().ok_or_else(|| {
            StatsError::Internal(format!(
                "retained table spec for {name:?} has no logical schema"
            ))
        })?;
        let active_logical = schema_from_proto_view(logical_view)?;
        // The spec declares the pre-implicit schema. `stored_form` prepends the
        // implicit `seq` and appends `cluster` only when absent, so declaring
        // `merged` with the implicit `seq` removed — `cluster` kept explicit —
        // reproduces `merged` exactly under `stored_form`, column order
        // included.
        let mut next_logical = merged.clone();
        if !active_logical
            .columns
            .iter()
            .any(|column| column.name == IMPLICIT_SEQ_COLUMN)
        {
            next_logical
                .columns
                .retain(|column| column.name != IMPLICIT_SEQ_COLUMN);
        }
        let mut next = active.clone();
        next.version = Some(next_version);
        next.logical_schema = MessageField::some(schema_to_proto_owned(&next_logical));
        let next_bytes = next.encode_to_vec();
        let next_view = TableSpecView::decode_view(&next_bytes).map_err(|error| {
            StatsError::Internal(format!("encode evolved table spec for {name:?}: {error}"))
        })?;
        let policy = self.catalog.get_policy(name)?;
        Ok(Some(ValidatedTableSpec::from_view(
            &next_view,
            &next_logical,
            &policy,
        )?))
    }

    fn register_table_with(
        &self,
        name: &str,
        schema: Schema,
        policy: StoragePolicy,
        registration: SchemaRegistration,
        table_spec: Option<&ValidatedTableSpec>,
    ) -> Result<(Schema, Option<SpecLifecycle>), StatsError> {
        // Validate the name (and fence the `log` dir special-case) first.
        self.namespace_dir(name)?;
        validate_index_policies(&schema)?;
        resolve_key_column(&schema)?;
        let initial_managed_spec = if table_spec.is_none()
            && self.mode == ServeMode::Live
            && !self.remote_log_dir.trim().is_empty()
            && !self.catalog.contains(name)
        {
            initial_managed_table_spec(name, &schema, &policy)?
        } else {
            None
        };
        let stored = stored_form(schema);
        let registration_lock = self.tables.registration_lock(name);
        let _registration_guard = registration_lock.lock().unwrap();
        let evolved_spec = match (table_spec, initial_managed_spec.as_ref()) {
            (Some(_), _) | (_, Some(_)) => None,
            (None, None) => self.spec_for_legacy_schema_evolution(name, &stored, registration)?,
        };
        let table_spec = table_spec
            .or(initial_managed_spec.as_ref())
            .or(evolved_spec.as_ref());
        if let Some(table_spec) = table_spec {
            self.catalog.validate_table_spec_registration(
                name,
                &table_spec.spec,
                &table_spec.hash,
                self.catalog.aggregate_namespace_stats(name)?.row_count > 0,
            )?;
            let declared_schema = stored_form(table_spec.schema.clone());
            let prospective_schema = match self.catalog.get_live(name) {
                Some(existing) => match registration {
                    SchemaRegistration::Additive => merge_schemas(&existing.schema, &stored),
                    SchemaRegistration::Managed => merge_managed_schema(&existing.schema, &stored),
                }?,
                None => stored.clone(),
            };
            if prospective_schema != declared_schema {
                return Err(StatsError::SchemaValidation(format!(
                    "table_spec.logical_schema must describe the complete effective schema for namespace {name:?}"
                )));
            }
            let prospective_policy = if self.catalog.contains(name) && policy.is_empty() {
                self.catalog.get_policy(name)?
            } else {
                policy.clone()
            };
            if prospective_policy != table_spec.cache_policy {
                return Err(StatsError::SchemaValidation(format!(
                    "table_spec.operating_policy.local_cache must describe the effective storage policy for namespace {name:?}"
                )));
            }
        }

        // `merge_schemas` (pure) raises SchemaConflict on a column-type change.
        // The catalog applies the empty-policy-keeps-existing rule and persists
        // under a single lock; we only supply the schema-merge decision.
        let stored_for_merge = stored.clone();
        let had_engine = self.tables.contains(name);
        let (effective_schema, effective_policy) =
            self.catalog
                .register_or_evolve(
                    name,
                    stored,
                    policy,
                    move |existing_schema| match registration {
                        SchemaRegistration::Additive => {
                            merge_schemas(existing_schema, &stored_for_merge)
                        }
                        SchemaRegistration::Managed => {
                            merge_managed_schema(existing_schema, &stored_for_merge)
                        }
                    },
                )?;
        // (Re)build the engine on fresh registration or when the effective schema
        // evolved. The engine re-opens on the same dir, adopting existing
        // segments and recovering next_seq, so an additive evolution keeps the
        // already-flushed data visible. A runtime registration starts maintenance
        // immediately after construction.
        let needs_engine = !had_engine
            || self
                .tables
                .get(name)
                .map(|table| table.schema() != &effective_schema)
                .unwrap_or(true);
        if needs_engine {
            self.tables
                .register(name, effective_schema.clone(), effective_policy)?;
        } else {
            // Runtime kept; push the (possibly updated) policy onto it so a
            // policy-only re-register takes effect on the next eviction tick.
            if let Some(table) = self.tables.get(name) {
                table.update_policy(effective_policy);
            }
        }
        // Registration is a synchronous RPC path, so its committed revision is
        // owed to the table's maintenance loop, which publishes it or a later
        // revision containing it.
        let controller = self.tables.controller(name);
        let spec_lifecycle = table_spec
            .map(|table_spec| {
                controller
                    .commit_owing_publication(|| {
                        let has_rows = self.catalog.aggregate_namespace_stats(name)?.row_count > 0;
                        let status = self.catalog.register_table_spec(
                            name,
                            &table_spec.spec,
                            &table_spec.hash,
                            has_rows,
                        )?;
                        Ok((TableRevision::new(status.catalog_generation), status))
                    })
                    .map(|committed| committed.output)
                    .map_err(StatsError::from)
            })
            .transpose()?;
        if let Some(status) = &spec_lifecycle {
            if let Some(table) = self.tables.get(name) {
                table.update_table_spec(status);
            }
        }
        Ok((effective_schema, spec_lifecycle))
    }

    /// Append a pre-routed batch and return its row count and durability
    /// target. Test-only seam: the ingest RPCs route through
    /// [`Store::write_ingestion_rows`] / [`Store::write_forwarded_telemetry_rows`].
    #[cfg(any(test, feature = "test-util"))]
    pub fn write_rows(
        &self,
        name: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
    ) -> Result<(i64, i64), StatsError> {
        let outcome = self.write_physical_rows(name, arrow_ipc, origin_cluster)?;
        debug_assert!(outcome.ignored_columns.is_empty());
        let last_seq = outcome
            .persisted_targets
            .first()
            .map(|(_, seq)| *seq)
            .unwrap_or(-1);
        Ok((outcome.rows_written, last_seq))
    }

    /// Route and append a declared ingestion batch using strict schemas.
    pub fn write_ingestion_rows(
        &self,
        name: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
    ) -> Result<ForwardedWrite, StatsError> {
        let batch = decode_bounded_write_batch(arrow_ipc)?;
        self.write_routed_batch(
            IngestionBatchSource::Declared(name),
            batch,
            origin_cluster,
            BatchAlignment::Strict,
        )
    }

    /// Append telemetry forwarded by another Finelog while preserving the hub's
    /// server-owned schema. Unknown nullable columns are omitted and returned for
    /// observability; all other validation remains strict.
    pub fn write_forwarded_telemetry_rows(
        &self,
        name: &str,
        arrow_ipc: &[u8],
        origin_cluster: &str,
    ) -> Result<ForwardedWrite, StatsError> {
        let batch = decode_bounded_write_batch(arrow_ipc)?;
        let routed = self
            .policies
            .route_ingestion_batch(IngestionBatchSource::Stored(name), &batch)?;
        for partition in &routed {
            let namespace = &partition.destination.logical_namespace;
            match self.tables.require(namespace) {
                Ok(_) => {}
                Err(StatsError::NamespaceNotFound(_)) => {
                    let schema = schema_for_namespace(namespace).ok_or_else(|| {
                        StatsError::SchemaValidation(format!(
                            "no server-owned schema is registered for {namespace:?}"
                        ))
                    })?;
                    self.register_managed_table(namespace, schema, storage_policy_for(namespace)?)?;
                }
                Err(error) => return Err(error),
            }
        }
        self.append_routed_batches(
            batch.num_rows() as i64,
            routed,
            Some(origin_cluster),
            BatchAlignment::ForwardCompatible,
        )
    }

    fn write_routed_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: RecordBatch,
        origin_cluster: Option<&str>,
        alignment: BatchAlignment,
    ) -> Result<ForwardedWrite, StatsError> {
        let routed = self.policies.route_ingestion_batch(source, &batch)?;
        self.append_routed_batches(batch.num_rows() as i64, routed, origin_cluster, alignment)
    }

    pub(crate) fn write_prepared_ingestion_batches(
        &self,
        rows_written: i64,
        routed: Vec<crate::ingestion_policy::RoutedIngestionBatch>,
        origin_cluster: Option<&str>,
    ) -> Result<ForwardedWrite, StatsError> {
        self.append_routed_batches(rows_written, routed, origin_cluster, BatchAlignment::Strict)
    }

    fn append_routed_batches(
        &self,
        rows_written: i64,
        routed: Vec<crate::ingestion_policy::RoutedIngestionBatch>,
        origin_cluster: Option<&str>,
        alignment: BatchAlignment,
    ) -> Result<ForwardedWrite, StatsError> {
        let mut prepared_partitions = Vec::with_capacity(routed.len());
        let mut ignored_columns = BTreeSet::new();
        for partition in routed {
            let destination = partition.destination.logical_namespace;
            let table = self.tables.require_writable(&destination)?;
            let (mut aligned, ignored) = alignment.align(&partition.batch, table.schema())?;
            if let Some(origin) = origin_cluster {
                stamp_cluster_column(&mut aligned, origin);
            }
            ignored_columns.extend(ignored);
            prepared_partitions.push((destination, table, aligned));
        }
        let persisted_targets = prepared_partitions
            .into_iter()
            .map(|(destination, table, aligned)| {
                table
                    .append_aligned_batch(&aligned)
                    .map(|last_seq| (destination, last_seq))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ForwardedWrite {
            rows_written,
            persisted_targets,
            ignored_columns: ignored_columns.into_iter().collect(),
        })
    }

    pub(crate) fn route_ingestion_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<crate::ingestion_policy::RoutedIngestionBatch>, StatsError> {
        self.policies.route_ingestion_batch(source, batch)
    }

    #[cfg(any(test, feature = "test-util"))]
    fn write_physical_rows(
        &self,
        name: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
    ) -> Result<ForwardedWrite, StatsError> {
        let batch = decode_bounded_write_batch(arrow_ipc)?;
        let table = self.tables.require_writable(name)?;
        let mut aligned = validate_and_align_batch(&batch, table.schema())?;
        if let Some(origin) = origin_cluster {
            stamp_cluster_column(&mut aligned, origin);
        }
        let n = aligned.num_rows as i64;
        let last_seq = self.tables.append(name, &aligned)?;
        debug_assert_eq!(table.name(), name);
        Ok(ForwardedWrite {
            rows_written: n,
            persisted_targets: vec![(name.to_string(), last_seq)],
            ignored_columns: Vec::new(),
        })
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
        let table = self.tables.require(LOG_NAMESPACE_NAME)?;
        table.append_log_batch(columns, num_rows, added_bytes)
    }

    /// Block until `target` is durable in `name`, bounded by `timeout`.
    pub async fn await_persisted(
        &self,
        name: &str,
        target: i64,
        timeout: Duration,
    ) -> Result<(), StatsError> {
        self.tables
            .require(name)?
            .await_persisted(target, timeout)
            .await
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
        self.tables.query_visibility()
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
            if self.tables.get(&ns.name).is_none() {
                // A registry entry with no runtime is a transient state during
                // (re)build; skip it rather than fail the whole query.
                continue;
            }
            let snapshot = self.query_snapshot(&ns.name)?;
            let provider = self.namespace_provider(&ns.name, snapshot)?;
            out.push(RegisteredProvider {
                name: ns.name,
                provider,
            });
        }
        Ok(out)
    }

    /// Assemble the scan provider for one namespace from its snapshot.
    ///
    /// Every read surface — `Query`, `FetchLogs`, and the forwarder — plans
    /// through this one assembly so segment-index policy, partition pruning,
    /// and object-source resolution cannot diverge between them.
    pub fn namespace_provider(
        &self,
        name: &str,
        snapshot: NamespaceSnapshot,
    ) -> Result<NamespaceProvider, StatsError> {
        let provider = NamespaceProvider::build(snapshot.schema, &snapshot.paths, snapshot.indices)
            .map_err(|e| StatsError::Internal(format!("build provider {name:?}: {e}")))?
            .with_segment_artifacts(snapshot.artifacts)
            .with_segment_indexes_enabled(segment_indexes_enabled_for(name))
            .with_exact_postings_policy(snapshot.exact_postings_policy)
            .with_segment_key_bounds(snapshot.key_column, snapshot.key_bounds)
            .with_segment_seq_bounds(snapshot.seq_bounds)
            .with_segment_partitions(physical_partition_policy_for(name), snapshot.partitions);
        Ok(match snapshot.sources {
            Some((store, segments)) => provider.with_object_sources(store, segments),
            None => provider,
        })
    }

    /// The tightest maximum query time among the object-backed tables a server
    /// read plans over, or `None` when it plans over none.
    ///
    /// The read's effective deadline may not exceed this: past it the table
    /// stops promising that objects retired by a newer state are still readable.
    pub fn object_query_bound(&self) -> Option<Duration> {
        self.tables
            .runtimes()
            .iter()
            .filter_map(|runtime| runtime.snapshot_query_bound())
            .min()
    }

    /// Snapshot `name`'s arrow schema alongside one consistent observation of its sealed
    /// segments: the paths a scan may read, and the lowest `seq` those paths hold. Both
    /// describe the same segment set, so a reader can tell a `seq` it simply has not
    /// reached from one that eviction put out of reach.
    pub fn query_snapshot(&self, name: &str) -> Result<NamespaceSnapshot, StatsError> {
        let engine = self.tables.require(name)?;
        let segments = engine.query_snapshot()?;
        Ok(NamespaceSnapshot {
            schema: Arc::clone(engine.arrow_schema()),
            exact_postings_policy: engine.schema().exact_postings_policy(),
            key_column: engine.key_column().to_string(),
            paths: segments.paths,
            key_bounds: segments.key_bounds,
            seq_bounds: segments.seq_bounds,
            partitions: segments.partitions,
            min_seq: segments.min_seq,
            indices: Arc::clone(self.tables.indices()),
            artifacts: segments.artifacts,
            sources: self
                .object_store
                .clone()
                .filter(|_| !segments.sources.is_empty())
                .map(|store| (store, segments.sources)),
        })
    }

    pub fn indices(&self) -> &Arc<IndexRegistry> {
        self.tables.indices()
    }

    /// `name`'s durability high-water mark: every row with `seq <= value` has been sealed
    /// into a segment, so it is visible to a scan unless it has since been evicted.
    pub fn namespace_persisted_seq(&self, name: &str) -> Result<i64, StatsError> {
        Ok(*self.tables.require(name)?.watch_persisted_seq().borrow())
    }

    /// The seq in `namespace` below which this store will never send to `target` again.
    pub fn forward_cursor(&self, target: &str, namespace: &str) -> Result<Option<i64>, StatsError> {
        self.catalog.forward_cursor(target, namespace)
    }

    /// Record `cursor` as settled for `(target, namespace)`.
    ///
    /// A cursor advance is a durable table-state transition: it changes both
    /// recovery state and the published direct-query snapshot.
    pub async fn set_forward_cursor(
        &self,
        target: &str,
        namespace: &str,
        cursor: i64,
    ) -> Result<(), StatsError> {
        self.tables
            .controller(namespace)
            .commit(|| {
                let revision = self.catalog.set_forward_cursor(target, namespace, cursor)?;
                Ok((revision, ()))
            })
            .await?;
        Ok(())
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
            let stats = match self.tables.get(&ns.name) {
                Some(table) => table.stats(),
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

    pub fn spec_lifecycle(&self, name: &str) -> Result<SpecLifecycle, StatsError> {
        self.catalog.require_live(name)?;
        self.catalog.spec_lifecycle(name)
    }

    /// The repeated error `name`'s specification transition is stuck on, or
    /// `None` while it is progressing, has none to run, or has no live runtime.
    pub fn blocked_migration_error(&self, name: &str) -> Option<String> {
        self.tables
            .get(name)
            .and_then(|table| table.blocked_migration_error())
    }

    pub async fn abort_table_migration(&self, name: &str) -> Result<SpecLifecycle, StatsError> {
        self.catalog.require_live(name)?;
        let _visibility_guard = self.tables.query_visibility().write().await;
        let table = self.tables.require(name)?;
        // The query view follows the abort only once its revision is published.
        let status = table
            .controller()
            .commit(|| {
                let status = self.catalog.abort_table_migration(name)?;
                Ok((TableRevision::new(status.catalog_generation), status))
            })
            .await?
            .output;
        self.apply_spec_lifecycle(name, &status)?;
        Ok(status)
    }

    fn apply_spec_lifecycle(&self, name: &str, status: &SpecLifecycle) -> Result<(), StatsError> {
        let table = self.tables.require(name)?;
        table.activate_query_version(status.active_version())?;
        table.update_table_spec(status);
        Ok(())
    }

    /// Run one full maintenance cycle for `name`:
    /// `flush -> compact (planner-drained, or forced L0->L1) -> sync -> evict ->
    /// backfill missing segment-index bundles`.
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
        self.tables.maintain(name, force_compact_l0).await
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
        self.tables
            .require(name)?
            .backdate_segment(path_basename, created_at_ms)
    }

    /// Directory holding the catalog and every segment file, or `None` when the
    /// store is RAM-only.
    pub fn data_dir(&self) -> Option<&Path> {
        self.data_dir.as_deref()
    }

    /// Offload target segments sync to; empty when sync is disabled.
    pub fn remote_log_dir(&self) -> &str {
        &self.remote_log_dir
    }

    /// Per-segment catalog rows for `name`, ordered by `min_seq`. Exposes
    /// level/location/seq-bounds that `NamespaceInfo` does not.
    pub fn list_segments(
        &self,
        name: &str,
    ) -> Result<Vec<crate::store::types::SegmentRow>, StatsError> {
        self.catalog.list_segments(name)
    }

    /// Remove `name` from the registry and delete its local table state.
    ///
    /// An object-backed table publishes a tombstone revision under this
    /// writer's fence, so a later load reports the table deleted rather than
    /// never published. Rejects the privileged `log` namespace.
    pub fn drop_table(&self, name: &str) -> Result<(), StatsError> {
        if name == LOG_NAMESPACE_NAME {
            return Err(StatsError::InvalidNamespace(format!(
                "namespace {name:?} is privileged and cannot be dropped via DropTable"
            )));
        }
        self.catalog.begin_drop(name)?;
        // Remove the table from the registry first so its flush task stops
        // touching the dir/catalog before we delete rows + files.
        let (engine, controller) = self.tables.take(name);
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
                        .block_on(engine.stop_and_join(TABLE_LIFECYCLE_SHUTDOWN_TIMEOUT));
                } else {
                    // mem-store: no background tasks and no dir; a sync stop
                    // signal suffices and needs no runtime.
                    engine.request_stop();
                }
            }
            // Only an object-backed table has a HEAD to tombstone. A legacy
            // table's deletion is complete once its catalog rows and files are
            // gone.
            if let Some(controller) = controller.filter(|controller| controller.is_object_backed())
            {
                tokio::runtime::Handle::current().block_on(controller.tombstone())?;
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
        self.tables.memory_summary()
    }

    /// Cooperatively shut down every namespace's background tasks.
    ///
    /// Called after the server loop returns. Each table's
    /// [`TableRuntime::shutdown`](crate::store::table::TableRuntime::shutdown)
    /// latches its stop flag, wakes its dispatched maintenance work, JOINs it
    /// bounded by `per_namespace_timeout`, and does a final flush. Durability is preserved: an acked write was already
    /// on a sealed L0 segment before the ack, and the final flush drains any
    /// not-yet-acked RAM rows. The bounded join (plus the task-abort fallback on
    /// timeout) guarantees this cannot hang — `main` applies its own outer
    /// timeout around `shutdown` for defense in depth.
    pub async fn shutdown(&self, per_namespace_timeout: Duration) {
        // Stop dispatching before draining, so no table is handed new work while
        // it is joining what it already has.
        self.scheduler.shutdown().await;
        self.tables.shutdown(per_namespace_timeout).await;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::array::{Int64Array, StringArray};
    use buffa::{Message, MessageField, MessageView};

    use super::*;
    use crate::levanter_metrics_policy::levanter_metrics_schema;
    use crate::proto::finelog::stats::{
        partition_field, OperatingPolicy, PartitionField, PartitionSpec, RemoteRetentionPolicy,
        SourceLayout, TableSpec, TableSpecView,
    };
    use crate::store::schema::{
        schema_to_arrow, schema_to_proto_owned, with_implicit_cluster, with_implicit_seq,
        CoveringProjection,
    };
    use crate::store::table::maintenance::{self, TableWork};
    use crate::store::types::SegmentLocation;
    use crate::test_support::{
        FaultAction, FaultInjectingObjectStore, ObjectFault, ObjectOp, ObjectPattern,
    };
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

    /// The same columns under an explicit ordering key, so a registration can
    /// attempt the key change online migration refuses.
    fn worker_schema_keyed_on(key: &str) -> Schema {
        Schema::new(worker_schema().columns, key)
    }

    fn mem_store() -> Store {
        Store::new(
            None,
            String::new(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn first_levanter_registration_uses_the_managed_object_layout() {
        let data_dir = crate::test_support::unique_dir("managed_levanter_data");
        let remote_dir = crate::test_support::unique_dir("managed_levanter_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();

        store
            .register_table(
                LEVANTER_METRICS_NAMESPACE,
                levanter_metrics_schema(),
                StoragePolicy::default(),
            )
            .unwrap();

        let lifecycle = store.spec_lifecycle(LEVANTER_METRICS_NAMESPACE).unwrap();
        assert_eq!(lifecycle.active_version(), 1);
        assert!(lifecycle.desired.is_none());
        assert!(has_managed_levanter_layout(
            lifecycle.active.as_ref().unwrap()
        ));
        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn bootstrap_registers_managed_vnext_for_unpartitioned_levanter() {
        let data_dir = crate::test_support::unique_dir("upgrade_levanter_data");
        let remote_dir = crate::test_support::unique_dir("upgrade_levanter_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
        let unpartitioned = ObjectSpec::new(1)
            .schema(levanter_metrics_schema())
            .validated();
        store
            .register_table_with(
                LEVANTER_METRICS_NAMESPACE,
                unpartitioned.schema.clone(),
                unpartitioned.cache_policy.clone(),
                SchemaRegistration::Additive,
                Some(&unpartitioned),
            )
            .unwrap();
        store
            .publish_object_catalog(LEVANTER_METRICS_NAMESPACE)
            .await
            .unwrap();

        store.recover_tables().await.unwrap();

        let lifecycle = store.spec_lifecycle(LEVANTER_METRICS_NAMESPACE).unwrap();
        assert_eq!(lifecycle.active_version(), 2);
        assert!(lifecycle.desired.is_none());
        assert!(has_managed_levanter_layout(
            lifecycle.active.as_ref().unwrap()
        ));
        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// Builds every object-store-backed table spec these tests register. All
    /// such specs run L0 in object-store mode and retain remote data forever;
    /// callers vary the schema, the source layout, and the timing policy.
    struct ObjectSpec {
        version: u64,
        schema: Schema,
        source_layout: SourceLayout,
        max_query_time_ms: u64,
        rollback_window_ms: u64,
        cache_policy: StoragePolicy,
    }

    impl ObjectSpec {
        fn new(version: u64) -> Self {
            Self {
                version,
                schema: worker_schema(),
                source_layout: SourceLayout::default(),
                max_query_time_ms: 0,
                rollback_window_ms: 0,
                cache_policy: StoragePolicy::default(),
            }
        }

        /// The local-cache policy the spec declares, which a registration
        /// requires to match the policy the table is already registered with.
        fn cache_policy(mut self, cache_policy: StoragePolicy) -> Self {
            self.cache_policy = cache_policy;
            self
        }

        fn schema(mut self, schema: Schema) -> Self {
            self.schema = schema;
            self
        }

        fn source_layout(mut self, source_layout: SourceLayout) -> Self {
            self.source_layout = source_layout;
            self
        }

        fn max_query_time_ms(mut self, max_query_time_ms: u64) -> Self {
            self.max_query_time_ms = max_query_time_ms;
            self
        }

        fn rollback_window_ms(mut self, rollback_window_ms: u64) -> Self {
            self.rollback_window_ms = rollback_window_ms;
            self
        }

        fn validated(self) -> ValidatedTableSpec {
            let spec = TableSpec {
                version: Some(self.version),
                logical_schema: MessageField::some(schema_to_proto_owned(&self.schema)),
                source_layout: MessageField::some(self.source_layout),
                operating_policy: MessageField::some(OperatingPolicy {
                    l0_mode: Some(L0Mode::L0_MODE_OBJECT_STORE.into()),
                    remote_retention: MessageField::some(RemoteRetentionPolicy {
                        retain_forever: Some(true),
                        ..Default::default()
                    }),
                    max_query_time_ms: (self.max_query_time_ms > 0)
                        .then_some(self.max_query_time_ms),
                    rollback_window_ms: (self.rollback_window_ms > 0)
                        .then_some(self.rollback_window_ms),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let encoded = spec.encode_to_vec();
            let view = TableSpecView::decode_view(&encoded).unwrap();
            ValidatedTableSpec::from_view(&view, &self.schema, &self.cache_policy).unwrap()
        }
    }

    fn object_backed_spec(version: u64) -> ValidatedTableSpec {
        ObjectSpec::new(version).validated()
    }

    /// The same logical schema under a different physical object size: a
    /// compatible rewrite that changes no schema and so needs no engine rebuild.
    fn retargeted_object_backed_spec(version: u64) -> ValidatedTableSpec {
        ObjectSpec::new(version)
            .source_layout(SourceLayout {
                target_object_bytes: Some(8 * 1024 * 1024),
                ..Default::default()
            })
            .validated()
    }

    fn partitioned_object_backed_spec(version: u64) -> ValidatedTableSpec {
        ObjectSpec::new(version)
            .source_layout(SourceLayout {
                partition: MessageField::some(PartitionSpec {
                    spec_id: Some(1),
                    fields: vec![PartitionField {
                        source_column: Some("worker_id".to_string()),
                        name: Some("worker_id".to_string()),
                        transform: Some(partition_field::Transform::Identity(Box::default())),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .validated()
    }

    fn sorted_object_backed_spec(version: u64) -> ValidatedTableSpec {
        ObjectSpec::new(version)
            .schema(worker_schema().with_sort_columns(["mem_bytes"]))
            .source_layout(SourceLayout {
                sort_columns: vec!["mem_bytes".to_string()],
                ..Default::default()
            })
            .validated()
    }

    struct PublishedObjectTableFixture {
        store: Store,
        data_dir: PathBuf,
        remote_dir: PathBuf,
        batch_schema: SchemaRef,
        paths: Vec<String>,
        last_seq: i64,
        initial_catalog: TableSnapshot,
        current_catalog: TableSnapshot,
    }

    async fn published_object_table(tag: &str) -> PublishedObjectTableFixture {
        let data_dir = crate::test_support::unique_dir(&format!("{tag}_data"));
        let remote_dir = crate::test_support::unique_dir(&format!("{tag}_remote"));
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        let initial_catalog = store.publish_object_catalog("iris.worker").await.unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["w-1", "w-2"])),
                Arc::new(Int64Array::from(vec![128, 256])),
                Arc::new(Int64Array::from(vec![1, 2])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();
        let paths = store.query_snapshot("iris.worker").unwrap().paths;
        let current_catalog = store.publish_object_catalog("iris.worker").await.unwrap();
        PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            batch_schema,
            paths,
            last_seq,
            initial_catalog,
            current_catalog,
        }
    }

    /// Scan `table` the way the Query RPC does, so the provider prunes from the
    /// pinned state and then localizes exactly the objects it selected. Returns
    /// the row count.
    async fn scan_table(store: &Store, table: &str) -> i64 {
        let providers = store.query_providers().unwrap();
        let sql = format!("SELECT * FROM \"{table}\"");
        let result = crate::query::run_query_over(&crate::query::make_ctx(), providers, &sql)
            .await
            .unwrap();
        result.batches.iter().map(|b| b.num_rows() as i64).sum()
    }

    fn assert_local_content_object(data_dir: &Path, path: &str) {
        let path = Path::new(path);
        assert!(path.starts_with(data_dir.join("_finelog/tables/iris.worker/objects")));
        let filename = path.file_name().and_then(|name| name.to_str()).unwrap();
        let hash = filename.strip_suffix(".parquet").unwrap();
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn versioned_registration_publishes_and_recovers_remote_head() {
        let fixture = published_object_table("versioned_registration").await;
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            paths,
            last_seq,
            initial_catalog: first,
            current_catalog: after_flush,
            ..
        } = fixture;
        assert_eq!(
            store
                .spec_lifecycle("iris.worker")
                .unwrap()
                .active_version(),
            1
        );
        assert_eq!(first.state().catalog().active_table_spec_version, Some(1));
        assert_eq!(first.state().catalog().retained_table_specs.len(), 1);
        assert_eq!(paths.len(), 1);
        assert_local_content_object(&data_dir, &paths[0]);
        assert_eq!(after_flush.state().catalog().catalog_generation, Some(2));
        let version = after_flush
            .state()
            .catalog()
            .version_segments
            .iter()
            .find(|segments| segments.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(version.live_segments.len(), 1);
        assert_eq!(version.live_segments[0].row_count, Some(2));
        let object_id = version.live_segments[0]
            .source
            .as_option()
            .unwrap()
            .object_id
            .as_deref()
            .unwrap();
        let object_id = crate::store::object_store::ObjectId::parse(object_id).unwrap();
        assert!(object_id
            .table_relative("iris.worker")
            .unwrap()
            .starts_with("objects/"));

        // An identical registration and publication is a retry, not a new
        // generation. Loading from a fresh RemoteObjectStore recovers the same HEAD.
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        let retry = store.publish_object_catalog("iris.worker").await.unwrap();
        assert_eq!(retry.state().catalog(), after_flush.state().catalog());
        let recovered = ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ))
        .load("iris.worker")
        .await
        .unwrap()
        .unwrap();
        assert_eq!(&recovered.catalog, after_flush.state().catalog());

        store
            .set_forward_cursor("hub", "iris.worker", last_seq)
            .await
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let with_cursor = ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ))
        .load("iris.worker")
        .await
        .unwrap()
        .unwrap();
        assert_eq!(with_cursor.catalog.forward_cursors.len(), 1);
        assert_eq!(
            with_cursor.catalog.forward_cursors[0].cursor,
            Some(last_seq)
        );

        // A scan localizes the objects it selected. Deleting the cached file
        // does not change what the pinned state says is live, and the next scan
        // fetches it back.
        std::fs::remove_file(&paths[0]).unwrap();
        assert_eq!(store.query_snapshot("iris.worker").unwrap().paths, paths);
        assert_eq!(scan_table(&store, "iris.worker").await, 2);
        assert!(Path::new(&paths[0]).exists());

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A revision committed locally but never published belongs to the writer
    /// that claims the table next: recovery rolls it forward instead of
    /// replacing it with the state HEAD selects.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unpublished_local_revision_rolls_forward_under_a_fresh_claim() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            paths,
            ..
        } = published_object_table("unpublished_local_revision").await;
        let unpublished = store
            .register_versioned_table("iris.worker", partitioned_object_backed_spec(2))
            .unwrap();
        assert_eq!(unpublished.spec_lifecycle.desired_version(), 2);

        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        std::fs::remove_file(&paths[0]).unwrap();
        let reopened = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        reopened.bootstrap_maintenance();

        // The local revision is ahead of HEAD, so nothing is rebuilt from it.
        assert_eq!(reopened.recover_tables().await.unwrap(), 0);
        let recovered_status = reopened.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(recovered_status.active_version(), 1);
        assert_eq!(recovered_status.desired_version(), 2);
        // Recovery localized nothing: the query view names the segments the
        // pinned state references, and no object has been fetched for them.
        assert!(!Path::new(&paths[0]).exists());
        assert_eq!(reopened.query_snapshot("iris.worker").unwrap().paths, paths);

        // The claimed fence owns that revision, so publishing it advances HEAD.
        let published = reopened
            .publish_object_catalog("iris.worker")
            .await
            .unwrap();
        assert_eq!(
            published.state().catalog().desired_table_spec_version,
            Some(2)
        );
        assert_eq!(
            published.revision().get(),
            recovered_status.catalog_generation
        );

        assert_eq!(reopened.query_snapshot("iris.worker").unwrap().paths, paths);
        reopened.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A first publication into an empty remote root re-uploads every
    /// referenced data object from the local bytes before HEAD is created:
    /// acknowledgement is local, so a wiped or repointed-to-empty root is
    /// restored from the machine that owns the bytes, exactly like the
    /// crash-before-first-upload recovery path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_publication_into_an_empty_remote_root_reuploads_local_bytes() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            ..
        } = published_object_table("empty_remote_root").await;
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);

        let repointed_dir = crate::test_support::unique_dir("empty_remote_root_repointed");
        let reopened = Store::new(
            Some(data_dir.clone()),
            repointed_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        reopened.bootstrap_maintenance();
        // The repointed root has no HEAD to recover.
        assert_eq!(reopened.recover_tables().await.unwrap(), 0);

        let published = reopened
            .publish_object_catalog("iris.worker")
            .await
            .unwrap();
        // The root now holds HEAD and every data object the state references.
        let restored = ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(repointed_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ))
        .load("iris.worker")
        .await
        .unwrap()
        .unwrap();
        assert_eq!(&restored.catalog, published.state().catalog());
        for segments in &restored.catalog.version_segments {
            for segment in &segments.live_segments {
                let object_id = segment
                    .source
                    .as_option()
                    .unwrap()
                    .object_id
                    .as_deref()
                    .unwrap();
                let relative = crate::store::object_store::ObjectId::parse(object_id)
                    .unwrap()
                    .table_relative("iris.worker")
                    .unwrap()
                    .to_string();
                assert!(
                    repointed_dir
                        .join("_finelog/tables/iris.worker")
                        .join(&relative)
                        .exists(),
                    "publication must re-upload {relative}"
                );
            }
        }

        reopened.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
        std::fs::remove_dir_all(repointed_dir).ok();
    }

    /// A publication whose referenced data objects exist neither remotely nor
    /// locally is refused before HEAD is created: the catalog and the remote
    /// root describe different histories, and a catalog published there would
    /// be unservable.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_publication_with_no_local_or_remote_bytes_is_refused() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            paths,
            ..
        } = published_object_table("unservable_publication").await;
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        for path in &paths {
            std::fs::remove_file(path).unwrap();
        }

        let repointed_dir = crate::test_support::unique_dir("unservable_publication_repointed");
        let reopened = Store::new(
            Some(data_dir.clone()),
            repointed_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        reopened.bootstrap_maintenance();
        assert_eq!(reopened.recover_tables().await.unwrap(), 0);

        let error = reopened
            .publish_object_catalog("iris.worker")
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("different histories"),
            "unexpected error: {error}"
        );
        // The refusal published nothing: the repointed root still has no HEAD.
        assert!(ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(repointed_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ))
        .load("iris.worker")
        .await
        .unwrap()
        .is_none());

        reopened.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
        std::fs::remove_dir_all(repointed_dir).ok();
    }

    /// Object recovery reads durable state only. Files left in the local object
    /// cache are never evidence that a segment is live.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cached_file_creates_no_visibility_during_recovery() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            paths,
            ..
        } = published_object_table("cached_file_recovery").await;
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        assert!(Path::new(&paths[0]).exists());

        let reopened = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        reopened.bootstrap_maintenance();
        assert_eq!(reopened.recover_tables().await.unwrap(), 0);
        // Visibility comes from the pinned state. A cache file the state does
        // not reference is not a segment, and a referenced object is live
        // whether or not its file is present.
        let stray = data_dir
            .join("iris.worker")
            .join("objects")
            .join("stray.parquet");
        std::fs::create_dir_all(stray.parent().unwrap()).unwrap();
        std::fs::copy(&paths[0], &stray).unwrap();
        assert_eq!(reopened.query_snapshot("iris.worker").unwrap().paths, paths);
        std::fs::remove_file(&paths[0]).unwrap();
        assert_eq!(reopened.query_snapshot("iris.worker").unwrap().paths, paths);
        assert_eq!(scan_table(&reopened, "iris.worker").await, 2);
        assert!(Path::new(&paths[0]).exists());
        reopened.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A replacement process claims the table's fence. The original store may
    /// no longer publish, and stops accepting writes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replacement_claim_leaves_the_original_store_unready_for_writes() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            batch_schema,
            ..
        } = published_object_table("replacement_claim").await;
        let replacement_dir = crate::test_support::unique_dir("replacement_claim_target");
        let replacement = Store::new(
            Some(replacement_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        replacement.bootstrap_maintenance();
        assert_eq!(replacement.recover_tables().await.unwrap(), 1);

        let error = store
            .publish_object_catalog("iris.worker")
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));

        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["w-fenced"])),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![9])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let rejected = store.write_rows("iris.worker", &ipc, None).unwrap_err();
        assert!(matches!(rejected, StatsError::SchemaConflict(_)));
        // The replacement still owns the table and keeps writing.
        replacement.write_rows("iris.worker", &ipc, None).unwrap();

        store.shutdown(Duration::from_secs(1)).await;
        replacement.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(replacement_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_head_recovers_an_empty_store_and_preserves_sequence_high_water() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            batch_schema,
            ..
        } = published_object_table("empty_store_recovery").await;
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        let empty_data_dir = crate::test_support::unique_dir("empty_store_recovery_target");
        let recovered_store = Store::new(
            Some(empty_data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        recovered_store.bootstrap_maintenance();
        assert_eq!(recovered_store.recover_tables().await.unwrap(), 1);
        assert_eq!(
            recovered_store
                .spec_lifecycle("iris.worker")
                .unwrap()
                .active_version(),
            1
        );
        assert_eq!(
            recovered_store
                .query_snapshot("iris.worker")
                .unwrap()
                .paths
                .len(),
            1
        );
        let recovery_batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["w-3"])),
                Arc::new(Int64Array::from(vec![512])),
                Arc::new(Int64Array::from(vec![3])),
            ],
        )
        .unwrap();
        let recovery_ipc = crate::store::ipc::encode_ipc(&batch_schema, &[recovery_batch]).unwrap();
        let (_, recovery_seq) = recovered_store
            .write_rows("iris.worker", &recovery_ipc, None)
            .unwrap();
        assert_eq!(recovery_seq, 3);
        recovered_store
            .await_persisted("iris.worker", recovery_seq, Duration::from_secs(10))
            .await
            .unwrap();
        recovered_store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(empty_data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_layout_change_backfills_existing_rows_then_retires_the_source() {
        let data_dir = crate::test_support::unique_dir("layout_upgrade_data");
        let remote_dir = crate::test_support::unique_dir("layout_upgrade_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table(
                "iris.worker",
                ObjectSpec::new(1).max_query_time_ms(100).validated(),
            )
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        for (worker, mem_bytes) in [("existing-a", 128), ("existing-b", 256)] {
            let batch = RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![worker])),
                    Arc::new(Int64Array::from(vec![mem_bytes])),
                    Arc::new(Int64Array::from(vec![mem_bytes])),
                ],
            )
            .unwrap();
            let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
            let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
            store
                .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
                .await
                .unwrap();
        }

        let registration = store
            .register_versioned_table("iris.worker", retargeted_object_backed_spec(2))
            .unwrap();
        assert_eq!(registration.spec_lifecycle.active_version(), 1);
        assert_eq!(registration.spec_lifecycle.desired_version(), 2);
        let transition = store.publish_object_catalog("iris.worker").await.unwrap();
        let active = transition
            .state()
            .catalog()
            .version_segments
            .iter()
            .find(|version| version.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(active.live_segments.len(), 2);
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();

        let status = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(status.active_version(), 2);
        // The backfill coalesces both version-1 sources into one rewritten
        // output object.
        let snapshot = store.query_snapshot("iris.worker").unwrap();
        assert_eq!(snapshot.paths.len(), 1);
        let rewritten = store
            .catalog
            .list_segments("iris.worker")
            .unwrap()
            .into_iter()
            .find(|segment| segment.path == snapshot.paths[0])
            .unwrap();
        assert_eq!(rewritten.level, 1);

        // Observation permits compaction, but the backfill output must remain
        // marked as backfill so a rollback can discard it in favor of version 1.
        store.maintain_namespace("iris.worker", true).await.unwrap();
        let compacted = store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .into_iter()
            .filter(|segment| segment.table_spec_version == 2)
            .collect::<Vec<_>>();
        assert_eq!(compacted.len(), 1);
        assert!(compacted[0].migration_backfill);
        let compacted_snapshot = store.publish_object_catalog("iris.worker").await.unwrap();
        let compacted_active = compacted_snapshot
            .state()
            .catalog()
            .version_segments
            .iter()
            .find(|version| version.table_spec_version == Some(2))
            .unwrap();
        assert_eq!(compacted_active.live_segments.len(), 1);
        assert_eq!(
            compacted_active.live_segments[0].migration_backfill,
            Some(true)
        );
        assert_eq!(compacted_active.live_segments[0].row_count, Some(2));

        store
            .catalog
            .expire_migration_observation("iris.worker")
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let retired = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(
            retired.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_RETIRED
        );
        assert!(store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .iter()
            .all(|segment| segment.table_spec_version == 2));
        assert!(store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .iter()
            .all(|segment| !segment.migration_backfill));
        let retired_catalog = store.publish_object_catalog("iris.worker").await.unwrap();
        assert!(retired_catalog
            .state()
            .catalog()
            .version_segments
            .iter()
            .all(|version| version.table_spec_version == Some(2)));

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A backfill spanning several batches must not revisit sources an earlier
    /// batch checkpointed: every tick re-derives the covered set by splitting
    /// each checkpoint's separator-joined source identities, and a source
    /// migrated twice would duplicate its rows.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_multi_batch_backfill_skips_sources_covered_by_a_joined_checkpoint() {
        let data_dir = crate::test_support::unique_dir("multi_batch_backfill_data");
        let remote_dir = crate::test_support::unique_dir("multi_batch_backfill_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .tables()
            .set_compaction_config(crate::store::compaction::config::CompactionConfig {
                migration_batch_sources: 2,
                ..Default::default()
            });
        store
            .register_versioned_table(
                "iris.worker",
                ObjectSpec::new(1).max_query_time_ms(100).validated(),
            )
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        for (worker, mem_bytes) in [("w-1", 10), ("w-2", 20), ("w-3", 30)] {
            let batch = RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![worker])),
                    Arc::new(Int64Array::from(vec![mem_bytes])),
                    Arc::new(Int64Array::from(vec![mem_bytes])),
                ],
            )
            .unwrap();
            let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
            let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
            store
                .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
                .await
                .unwrap();
        }

        store
            .register_versioned_table("iris.worker", retargeted_object_backed_spec(2))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();

        // Tick 1 checkpoints a two-source batch under one joined identity;
        // tick 2 must skip both and finish the third.
        for _ in 0..6 {
            store
                .maintain_namespace("iris.worker", false)
                .await
                .unwrap();
            if store
                .spec_lifecycle("iris.worker")
                .unwrap()
                .active_version()
                == 2
            {
                break;
            }
        }
        assert_eq!(
            store
                .spec_lifecycle("iris.worker")
                .unwrap()
                .active_version(),
            2
        );

        let backfilled: Vec<_> = store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .into_iter()
            .filter(|segment| segment.table_spec_version == 2 && segment.migration_backfill)
            .collect();
        let total_rows: i64 = backfilled
            .iter()
            .map(|segment| segment.artifacts.binding.row_count)
            .sum();
        assert_eq!(total_rows, 3, "a revisited source would duplicate its rows");
        let mut covered: Vec<_> = backfilled
            .iter()
            .filter_map(|segment| segment.migration_source_id.as_deref())
            .flat_map(|ids| ids.split('\n'))
            .collect();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(
            covered.len(),
            3,
            "each source is covered by exactly one checkpoint"
        );

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn legacy_table_migrates_automatically_and_aborts_without_duplicate_rows() {
        let data_dir = crate::test_support::unique_dir("table_spec_migration_data");
        let remote_dir = crate::test_support::unique_dir("table_spec_migration_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let legacy_batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let legacy_ipc = crate::store::ipc::encode_ipc(&batch_schema, &[legacy_batch]).unwrap();
        let (_, legacy_seq) = store.write_rows("iris.worker", &legacy_ipc, None).unwrap();
        store
            .await_persisted("iris.worker", legacy_seq, Duration::from_secs(10))
            .await
            .unwrap();
        let legacy_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(legacy_paths.len(), 1);
        assert!(!legacy_paths[0].contains("/_finelog/"));

        let registration = store
            .register_versioned_table(
                "iris.worker",
                ObjectSpec::new(1).max_query_time_ms(1).validated(),
            )
            .unwrap();
        assert_eq!(registration.spec_lifecycle.active_version(), 0);
        assert_eq!(registration.spec_lifecycle.desired_version(), 1);
        store.publish_object_catalog("iris.worker").await.unwrap();

        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let activated = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(activated.active_version(), 1);
        assert_eq!(activated.desired_version(), 0);
        assert_eq!(
            activated.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        let active_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(active_paths.len(), 1);
        assert_local_content_object(&data_dir, &active_paths[0]);

        let current_batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["current"])),
                Arc::new(Int64Array::from(vec![256])),
                Arc::new(Int64Array::from(vec![2])),
            ],
        )
        .unwrap();
        let current_ipc = crate::store::ipc::encode_ipc(&batch_schema, &[current_batch]).unwrap();
        let (_, current_seq) = store.write_rows("iris.worker", &current_ipc, None).unwrap();
        store
            .await_persisted("iris.worker", current_seq, Duration::from_secs(10))
            .await
            .unwrap();

        let aborted = store.abort_table_migration("iris.worker").await.unwrap();
        assert_eq!(aborted.active_version(), 0);
        let rollback_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(rollback_paths.len(), 2);
        assert!(rollback_paths.iter().any(|path| path == &legacy_paths[0]));
        assert!(rollback_paths.iter().any(|path| Path::new(path)
            .starts_with(data_dir.join("_finelog/tables/iris.worker/objects"))));

        let snapshot = store.publish_object_catalog("iris.worker").await.unwrap();
        let rollback_version = snapshot
            .state()
            .catalog()
            .version_segments
            .iter()
            .find(|segments| segments.table_spec_version == Some(0))
            .unwrap();
        assert_eq!(
            rollback_version
                .live_segments
                .iter()
                .map(|segment| segment.row_count.unwrap_or(0))
                .sum::<i64>(),
            1
        );

        let cleaned_records = store.catalog.object_segments("iris.worker").unwrap();
        assert_eq!(cleaned_records.len(), 1);
        assert_eq!(cleaned_records[0].table_spec_version, 0);
        assert!(!cleaned_records[0].migration_backfill);
        let cleaned_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(cleaned_paths.len(), 2);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A definition version that changes no physical layout is not a migration:
    /// it activates in the registration's own state commit, and the objects the
    /// prior version wrote answer queries under the new version unchanged.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_metadata_only_version_activates_in_one_commit() {
        let data_dir = crate::test_support::unique_dir("metadata_only_activation_data");
        let remote_dir = crate::test_support::unique_dir("metadata_only_activation_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["existing"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();
        let before = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(before.len(), 1);
        let generation_before = store
            .spec_lifecycle("iris.worker")
            .unwrap()
            .catalog_generation;

        // Same layout, different operating policy.
        let registration = store
            .register_versioned_table(
                "iris.worker",
                ObjectSpec::new(2).max_query_time_ms(30_000).validated(),
            )
            .unwrap();
        assert_eq!(registration.spec_lifecycle.active_version(), 2);
        assert_eq!(registration.spec_lifecycle.desired_version(), 0);
        assert!(registration.spec_lifecycle.migration.is_none());
        assert_eq!(
            registration.spec_lifecycle.catalog_generation,
            generation_before + 1
        );

        // The existing objects moved onto the new version in that same commit,
        // so nothing was rewritten and nothing left the query view.
        assert!(store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .iter()
            .all(|segment| segment.table_spec_version == 2));
        assert_eq!(store.query_snapshot("iris.worker").unwrap().paths, before);
        assert_eq!(scan_table(&store, "iris.worker").await, 1);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A logical change no online migration can express is refused at
    /// registration rather than recorded as a transition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_incompatible_logical_change_is_rejected() {
        let data_dir = crate::test_support::unique_dir("incompatible_change_data");
        let remote_dir = crate::test_support::unique_dir("incompatible_change_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["existing"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();

        let rekeyed = ObjectSpec::new(2)
            .schema(worker_schema_keyed_on("mem_bytes"))
            .validated();
        let error = match store.register_versioned_table("iris.worker", rekeyed) {
            Ok(_) => panic!("an incompatible key change must not register"),
            Err(error) => error,
        };
        assert!(matches!(error, StatsError::SchemaConflict(_)), "{error}");

        // The rejected version left no transition behind.
        let status = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(status.active_version(), 1);
        assert_eq!(status.desired_version(), 0);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A legacy additive registration on a settled object-backed table folds
    /// the new column into the table's next spec version — a spec-silent
    /// evolution would be dropped by the next cold recovery, which rebuilds
    /// schemas from retained specs while the flushed objects still carry the
    /// column.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_legacy_additive_registration_evolves_the_object_spec() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            ..
        } = published_object_table("legacy_evolution").await;

        let mut evolved = worker_schema();
        evolved
            .columns
            .push(Column::new("gpu_id", ColumnType::COLUMN_TYPE_STRING, true));
        tokio::task::block_in_place(|| {
            store.register_table("iris.worker", evolved.clone(), StoragePolicy::default())
        })
        .unwrap();

        // An unchanged layout makes the evolution metadata-only: version 2 is
        // active in the registration commit, and its spec declares the column.
        let lifecycle = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(lifecycle.active_version(), 2);
        assert_eq!(lifecycle.desired_version(), 0);
        let declared: Vec<String> = lifecycle
            .active
            .unwrap()
            .logical_schema
            .columns
            .iter()
            .filter_map(|column| column.name.clone())
            .collect();
        assert!(declared.contains(&"gpu_id".to_string()), "{declared:?}");

        // Re-registering the same schema is a no-op, not another version.
        tokio::task::block_in_place(|| {
            store.register_table("iris.worker", evolved, StoragePolicy::default())
        })
        .unwrap();
        assert_eq!(
            store
                .spec_lifecycle("iris.worker")
                .unwrap()
                .active_version(),
            2
        );

        // A cold store recovering from the published state keeps the column.
        store.publish_object_catalog("iris.worker").await.unwrap();
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        std::fs::remove_dir_all(&data_dir).unwrap();
        let cold = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        cold.recover_tables().await.unwrap();
        let recovered = cold.get_table_schema("iris.worker").unwrap();
        assert!(
            recovered
                .columns
                .iter()
                .any(|column| column.name == "gpu_id"),
            "cold recovery dropped the evolved column"
        );

        cold.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// While a spec transition is in flight, a schema-changing legacy
    /// registration is refused — the forwarder holds its cursor and retries —
    /// while an identical re-registration stays idempotent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_schema_change_is_refused_while_a_spec_transition_is_in_flight() {
        let PublishedObjectTableFixture {
            store,
            data_dir,
            remote_dir,
            ..
        } = published_object_table("evolution_mid_transition").await;
        store
            .register_versioned_table("iris.worker", retargeted_object_backed_spec(2))
            .unwrap();

        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();

        let mut evolved = worker_schema();
        evolved
            .columns
            .push(Column::new("gpu_id", ColumnType::COLUMN_TYPE_STRING, true));
        let error = store
            .register_table("iris.worker", evolved, StoragePolicy::default())
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)), "{error}");
        assert!(error.to_string().contains("transition settles"), "{error}");

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// The rollback window is not the query bound. A definition that promises a
    /// 50 ms query lifetime and a one-hour rollback window keeps the version it
    /// replaced activatable, and its objects retained, long after every query
    /// that could have pinned them has expired.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_long_rollback_window_outlives_a_short_query_bound() {
        let data_dir = crate::test_support::unique_dir("rollback_window_data");
        let remote_dir = crate::test_support::unique_dir("rollback_window_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table(
                "iris.worker",
                ObjectSpec::new(1)
                    .max_query_time_ms(50)
                    .rollback_window_ms(3_600_000)
                    .validated(),
            )
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["existing"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();
        let source_paths = store.query_snapshot("iris.worker").unwrap().paths;

        store
            .register_versioned_table(
                "iris.worker",
                ObjectSpec::new(2)
                    .source_layout(SourceLayout {
                        target_object_bytes: Some(8 * 1024 * 1024),
                        ..Default::default()
                    })
                    .max_query_time_ms(50)
                    .rollback_window_ms(3_600_000)
                    .validated(),
            )
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let activated = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(activated.active_version(), 2);
        assert_eq!(
            activated.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        let deadline = activated
            .migration
            .as_ref()
            .and_then(|migration| migration.observation_deadline_ms)
            .unwrap();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        assert!(deadline > now_ms + 60_000, "deadline {deadline}");

        // Every query the 50 ms bound admits has expired, and the source version
        // is still there.
        tokio::time::sleep(Duration::from_millis(200)).await;
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let observing = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(
            observing.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        assert!(store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .iter()
            .any(|segment| segment.table_spec_version == 1));
        for path in &source_paths {
            assert!(
                Path::new(path).exists(),
                "retired object {path} was collected"
            );
        }

        // Within that window the prior definition is still activatable.
        let rolled_back = store.abort_table_migration("iris.worker").await.unwrap();
        assert_eq!(rolled_back.active_version(), 1);
        assert_eq!(
            store.query_snapshot("iris.worker").unwrap().paths,
            source_paths
        );
        assert_eq!(scan_table(&store, "iris.worker").await, 1);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// Filesystem adoption is a one-time bootstrap input. Once a table's
    /// version-0 history has been imported and retired, a rescan of its
    /// directory contributes nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retired_version_zero_import_disables_filesystem_adoption() {
        let data_dir = crate::test_support::unique_dir("version_zero_import_data");
        let remote_dir = crate::test_support::unique_dir("version_zero_import_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();
        let legacy_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(legacy_paths.len(), 1);

        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        assert!(!store
            .catalog
            .filesystem_adoption_disabled("iris.worker")
            .unwrap());

        store
            .catalog
            .expire_migration_observation("iris.worker")
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let retired = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(
            retired.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_RETIRED
        );
        assert!(store
            .catalog
            .filesystem_adoption_disabled("iris.worker")
            .unwrap());

        // The imported history now lives in objects, and a full rescan of the
        // directory the legacy parquet still sits in adds nothing back.
        let imported = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(imported.len(), 1);
        assert_local_content_object(&data_dir, &imported[0]);
        assert!(Path::new(&legacy_paths[0]).exists());
        crate::store::adopt::adopt_store_from_disk(&data_dir, &store.catalog).unwrap();
        let after_rescan: Vec<String> = store
            .catalog
            .list_segments("iris.worker")
            .unwrap()
            .into_iter()
            .map(|row| row.path)
            .collect();
        assert_eq!(after_rescan, imported);
        assert_eq!(scan_table(&store, "iris.worker").await, 1);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A legacy table lived in long enough that its oldest segments exist only
    /// in the legacy archive. Every object operation its store performs passes
    /// through a recorder, so a test can assert what an import did and did not
    /// read.
    struct ArchivedLegacyTable {
        store: Store,
        data_dir: PathBuf,
        remote_dir: PathBuf,
        objects: Arc<std::sync::Mutex<Vec<Arc<FaultInjectingObjectStore>>>>,
        /// The evicted segment: a `REMOTE` catalog row with no local file,
        /// whose only copy is the archive object named by `archived_segment`.
        archived_path: String,
        archived_segment: String,
        local_paths: Vec<String>,
    }

    impl ArchivedLegacyTable {
        /// Every object key any of the store's object stores has been asked for
        /// since the last [`Self::forget_object_calls`].
        fn object_keys(&self) -> Vec<String> {
            self.objects
                .lock()
                .unwrap()
                .iter()
                .flat_map(|store| store.calls())
                .map(|call| call.key)
                .collect()
        }

        fn forget_object_calls(&self) {
            for store in self.objects.lock().unwrap().iter() {
                store.clear_calls();
            }
        }

        /// Fail any read of the archived segment, so a migration that reaches
        /// for archived bytes fails its tick rather than silently succeeding.
        fn refuse_archive_reads(&self) {
            for store in self.objects.lock().unwrap().iter() {
                store.arm(
                    ObjectFault::new(
                        ObjectOp::Read,
                        ObjectPattern::Contains(self.archived_segment.clone()),
                        FaultAction::Fail(StatsError::Internal(
                            "a migration must not read the legacy archive".to_string(),
                        )),
                    )
                    .forever(),
                );
            }
        }

        fn cleanup(self) {
            std::fs::remove_dir_all(self.data_dir).ok();
            std::fs::remove_dir_all(self.remote_dir).ok();
        }
    }

    /// The retention the archived-legacy fixture registers its table with: one
    /// local segment, and an age bound the fixture leaves slack under.
    fn archived_retention() -> StoragePolicy {
        StoragePolicy {
            max_segments: Some(1),
            max_age_seconds: Some(60),
            ..Default::default()
        }
    }

    /// The object-backed definition a version-0 table imports into.
    fn imported_spec() -> ValidatedTableSpec {
        ObjectSpec::new(1)
            .cache_policy(archived_retention())
            .validated()
    }

    /// Append one row to the legacy table and drive it to an uploaded L1
    /// segment, applying whatever retention the table was registered with.
    async fn write_legacy_row(store: &Store, worker: &str, mem_bytes: i64) {
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![worker])),
                Arc::new(Int64Array::from(vec![mem_bytes])),
                Arc::new(Int64Array::from(vec![mem_bytes])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store.maintain_namespace("iris.worker", true).await.unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
    }

    fn segment_rows(store: &Store) -> Vec<crate::store::types::SegmentRow> {
        store.list_segments("iris.worker").unwrap()
    }

    /// Build a legacy table holding `rows` rows, retained one segment at a time
    /// so that every segment but the newest has been uploaded to the archive and
    /// evicted. Maintenance is driven by hand throughout, so eviction happens
    /// where the test says it does.
    async fn archived_legacy_table(tag: &str, rows: &[&str]) -> ArchivedLegacyTable {
        let data_dir = crate::test_support::unique_dir(&format!("{tag}_data"));
        let remote_dir = crate::test_support::unique_dir(&format!("{tag}_remote"));
        let objects: Arc<std::sync::Mutex<Vec<Arc<FaultInjectingObjectStore>>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = Arc::clone(&objects);
        let store = Store::open(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
            TelemetryRootWriteMode::SemanticOnly,
            None,
            Some(Arc::new(move |inner| {
                let recorder = FaultInjectingObjectStore::new(inner);
                captured.lock().unwrap().push(Arc::clone(&recorder));
                recorder as Arc<dyn ObjectStore>
            })),
        )
        .unwrap();
        store
            .register_table("iris.worker", worker_schema(), archived_retention())
            .unwrap();
        for (index, worker) in rows.iter().enumerate() {
            write_legacy_row(&store, worker, 128 + index as i64).await;
        }

        let catalog_rows = segment_rows(&store);
        let archived: Vec<_> = catalog_rows
            .iter()
            .filter(|row| row.location == SegmentLocation::Remote)
            .collect();
        assert_eq!(
            archived.len(),
            rows.len() - 1,
            "retention leaves one local segment and archives the rest: {catalog_rows:?}"
        );
        let archived_path = archived[0].path.clone();
        assert!(
            !Path::new(&archived_path).exists(),
            "an archived segment has no local file"
        );
        ArchivedLegacyTable {
            archived_segment: crate::store::types::basename(&archived_path),
            archived_path,
            local_paths: catalog_rows
                .iter()
                .filter(|row| row.location != SegmentLocation::Remote)
                .map(|row| row.path.clone())
                .collect(),
            store,
            data_dir,
            remote_dir,
            objects,
        }
    }

    /// The archive files the legacy table left in the bucket.
    fn archived_objects(remote_dir: &Path) -> Vec<String> {
        crate::store::table::test_tables::remote_files(remote_dir, "iris.worker")
    }

    /// A table whose oldest history was evicted to the legacy archive imports
    /// what it still holds locally. The archive is not a migration source: its
    /// objects are never read, its bytes are never restored, and the rows they
    /// hold — already outside the live query view — are not counted in the
    /// import's row total, so the import reaches an activation instead of
    /// waiting forever for rows no source can supply.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_version_zero_import_skips_archive_only_segments() {
        let table = archived_legacy_table("import_skips_archive", &["archived", "live"]).await;
        let store = &table.store;
        // The archived segment's row is already unserved: excluding it from the
        // import changes nothing a query can observe.
        assert_eq!(scan_table(store, "iris.worker").await, 1);

        table.forget_object_calls();
        table.refuse_archive_reads();
        let registration = store
            .register_versioned_table("iris.worker", imported_spec())
            .unwrap();
        let pending = registration.spec_lifecycle.migration.unwrap();
        assert_eq!(
            pending.rows_total,
            Some(1),
            "the import owes only the rows it still holds locally"
        );

        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let activated = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(activated.active_version(), 1);
        assert_eq!(
            activated.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        let progress = activated.migration.clone().unwrap();
        assert_eq!(
            (progress.rows_completed, progress.rows_total),
            (Some(1), Some(1)),
            "the locally held row is migrated and the archived row is not owed"
        );

        let touched: Vec<_> = table
            .object_keys()
            .into_iter()
            .filter(|key| key.contains(&table.archived_segment))
            .collect();
        assert!(
            touched.is_empty(),
            "the import asked the object store for archived bytes: {touched:?}"
        );
        assert!(
            !Path::new(&table.archived_path).exists(),
            "the import restored an archived segment to local disk"
        );
        assert!(
            archived_objects(&table.remote_dir).contains(&table.archived_segment),
            "the archived segment keeps its object"
        );

        // Every locally held row is now served from an imported object.
        assert_eq!(scan_table(store, "iris.worker").await, 1);
        let imported = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(imported.len(), 1);
        assert_local_content_object(&table.data_dir, &imported[0]);

        store.shutdown(Duration::from_secs(1)).await;
        table.cleanup();
    }

    /// A source can leave the import's universe after registration: an eviction
    /// unlinks its local file and collapses its catalog row to `REMOTE`, which
    /// makes it archive-only like any other. The import skips it on the tick
    /// that would have rewritten it and still completes, because what it owes is
    /// decided by the sources that remain rather than by a total frozen at
    /// registration.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_source_evicted_after_registration_is_skipped_and_the_import_completes() {
        let table = archived_legacy_table("import_skips_evicted", &["archived", "live"]).await;
        let store = &table.store;
        let registration = store
            .register_versioned_table("iris.worker", imported_spec())
            .unwrap();
        assert_eq!(
            registration.spec_lifecycle.migration.unwrap().rows_total,
            Some(1)
        );

        // Retention ages the last local source out before the backfill reaches
        // it: its bytes are in the archive and its rows have left the query
        // view, exactly like the segment evicted before registration.
        store
            .backdate_segment(
                "iris.worker",
                &crate::store::types::basename(&table.local_paths[0]),
                1,
            )
            .unwrap();
        let runtime = store.tables.require("iris.worker").unwrap();
        maintenance::run(&runtime, TableWork::Eviction)
            .await
            .unwrap();
        assert!(!Path::new(&table.local_paths[0]).exists());
        assert!(segment_rows(store)
            .iter()
            .all(|row| row.location == SegmentLocation::Remote));
        table.refuse_archive_reads();

        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let activated = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(activated.active_version(), 1);
        assert_eq!(
            activated.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        let progress = activated.migration.unwrap();
        assert_eq!(
            (progress.rows_completed, progress.rows_total),
            (Some(0), Some(0)),
            "a universe emptied by eviction is reachable, not permanently owed"
        );
        assert_eq!(
            archived_objects(&table.remote_dir).len(),
            2,
            "both archived segments keep their objects"
        );

        store.shutdown(Duration::from_secs(1)).await;
        table.cleanup();
    }

    /// A catalog row can claim a local file that exists nowhere — no file on
    /// disk and no archive object (the production example is a nested
    /// partitioned path left behind by an interrupted layout change). Such a
    /// row is unserveable, so a version-0 import drops it and completes
    /// instead of forever retrying a source nothing can supply.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_version_zero_import_drops_a_source_with_no_bytes_anywhere() {
        let data_dir = crate::test_support::unique_dir("import_drops_phantom_data");
        let remote_dir = crate::test_support::unique_dir("import_drops_phantom_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
        let phantom_path = format!(
            "{}/iris.worker/run_id/31/seg_L2_0000000000000000900.parquet",
            data_dir.display()
        );
        store
            .catalog
            .upsert_segments(&[crate::store::types::SegmentRow {
                namespace: "iris.worker".to_string(),
                path: phantom_path.clone(),
                level: 2,
                min_seq: 900,
                max_seq: 909,
                row_count: 10,
                byte_size: 1234,
                created_at_ms: 1,
                min_key_value: None,
                max_key_value: None,
                partition: None,
                location: SegmentLocation::Local,
            }])
            .unwrap();

        let registration = store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        assert_eq!(
            registration.spec_lifecycle.migration.unwrap().rows_total,
            Some(11),
            "the frozen total still counts the unserveable row"
        );
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let activated = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(activated.active_version(), 1);
        assert_eq!(
            activated.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        let progress = activated.migration.unwrap();
        assert_eq!(
            (progress.rows_completed, progress.rows_total),
            (Some(1), Some(1)),
            "the universe restates to the rows a source can supply"
        );
        assert!(
            store
                .list_segments("iris.worker")
                .unwrap()
                .iter()
                .all(|row| row.path != phantom_path),
            "the unserveable row is dropped from the catalog"
        );
        assert_eq!(scan_table(&store, "iris.worker").await, 1);
        assert!(store.blocked_migration_error("iris.worker").is_none());

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// Retirement is a catalog operation. It drops every pre-fence legacy row,
    /// the archive-only ones included, and deletes no file: the archived objects
    /// stay in the bucket where `finelog gcs-query` finds them by listing, just
    /// as the uploaded copies of the retired local segments do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retiring_a_version_zero_import_keeps_the_archive() {
        let table = archived_legacy_table("import_retires_archive", &["archived", "live"]).await;
        let store = &table.store;
        store
            .register_versioned_table("iris.worker", imported_spec())
            .unwrap();
        table.refuse_archive_reads();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let archived_before = archived_objects(&table.remote_dir);

        store
            .catalog
            .expire_migration_observation("iris.worker")
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let retired = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(
            retired.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_RETIRED
        );

        let rows = segment_rows(store);
        assert!(
            rows.iter().all(|row| row.path != table.archived_path),
            "retirement drops the archive-only row with the rest of the legacy rows"
        );
        assert_eq!(
            archived_objects(&table.remote_dir),
            archived_before,
            "retirement deletes no archive object"
        );
        assert_eq!(scan_table(store, "iris.worker").await, 1);

        store.shutdown(Duration::from_secs(1)).await;
        table.cleanup();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn migration_can_be_aborted_before_background_activation() {
        let data_dir = crate::test_support::unique_dir("abort_pending_migration_data");
        let remote_dir = crate::test_support::unique_dir("abort_pending_migration_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();

        let pending = store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        assert_eq!(pending.spec_lifecycle.desired_version(), 1);
        store.publish_object_catalog("iris.worker").await.unwrap();
        let aborted = store.abort_table_migration("iris.worker").await.unwrap();

        assert_eq!(aborted.active_version(), 0);
        assert_eq!(aborted.desired_version(), 0);
        assert!(aborted.migration.is_none());
        assert!(store
            .catalog
            .object_segments("iris.worker")
            .unwrap()
            .is_empty());
        assert_eq!(store.query_snapshot("iris.worker").unwrap().paths.len(), 1);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// An abort reverts the table to legacy authority, but its retained HEAD
    /// keeps the table object-backed for maintenance, so the GC loop still
    /// sweeps the objects the aborted migration wrote: superseded catalogs
    /// after the retention window and unreferenced data after the orphan
    /// grace. HEAD and the current catalog are the intended residue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_aborted_migration_is_garbage_collected() {
        let data_dir = crate::test_support::unique_dir("abort_gc_data");
        let remote_dir = crate::test_support::unique_dir("abort_gc_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        // One maintenance pass migrates the row into the object layout.
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        store.abort_table_migration("iris.worker").await.unwrap();
        let aborted = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(aborted.active_version(), 0);

        let table_prefix = remote_dir.join("_finelog/tables/iris.worker");
        let files_under = |dir: &str| -> Vec<String> {
            std::fs::read_dir(table_prefix.join(dir))
                .map(|entries| {
                    entries
                        .flatten()
                        .filter_map(|entry| entry.file_name().into_string().ok())
                        .collect()
                })
                .unwrap_or_default()
        };
        assert!(
            !files_under("objects").is_empty(),
            "the aborted migration wrote no objects; the test exercises nothing"
        );
        let controller = store.tables.controller("iris.worker");
        assert!(
            controller.is_object_backed(),
            "the retained HEAD must keep the aborted table on the maintenance GC path"
        );
        let removed = controller
            .gc_published(crate::store::table::now_ms(), 0, 0, 0, true)
            .await
            .unwrap();
        assert!(removed > 0, "GC removed nothing after the abort");
        assert!(
            files_under("objects").is_empty(),
            "unreferenced data objects survive GC: {:?}",
            files_under("objects")
        );
        assert!(
            files_under("indices").is_empty(),
            "unreferenced index bundles survive GC: {:?}",
            files_under("indices")
        );
        assert_eq!(
            files_under("catalogs").len(),
            1,
            "only the current catalog revision is retained"
        );
        assert!(table_prefix.join("HEAD.json").exists());

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    /// A legacy import excludes archive-only rows, so the imported segments can
    /// top out below sequence numbers the table already issued. The published
    /// high-water mark must carry those, and a recovered allocator must start
    /// past them: reissuing a sequence number silently breaks every consumer
    /// that checkpoints on seq.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_recovery_seeds_the_allocator_past_archive_only_sequences() {
        let data_dir = crate::test_support::unique_dir("seq_floor_data");
        let remote_dir = crate::test_support::unique_dir("seq_floor_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        assert_eq!(
            store
                .spec_lifecycle("iris.worker")
                .unwrap()
                .active_version(),
            1
        );
        // An archive-only legacy row holds the table's highest sequence
        // numbers. It stays in the catalog until retirement, so the published
        // state carries it as a version-0 legacy-archive entry and its seqs
        // through the high-water mark.
        store
            .catalog
            .upsert_segments(&[crate::store::types::SegmentRow {
                namespace: "iris.worker".to_string(),
                path: format!(
                    "{}/iris.worker/seg_L2_0000000000000999999.parquet",
                    data_dir.display()
                ),
                level: 2,
                min_seq: 900_000,
                max_seq: 999_999,
                row_count: 10,
                byte_size: 1,
                created_at_ms: 1,
                min_key_value: None,
                max_key_value: None,
                partition: None,
                location: SegmentLocation::Remote,
            }])
            .unwrap();
        // The next flush commits a revision whose published state carries the
        // archive-only row and its high-water mark.
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);

        let cold_data_dir = crate::test_support::unique_dir("seq_floor_cold_target");
        let recovered = Store::new(
            Some(cold_data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
        recovered.bootstrap_maintenance();
        assert_eq!(recovered.recover_tables().await.unwrap(), 1);
        let (_, recovered_seq) = recovered.write_rows("iris.worker", &ipc, None).unwrap();
        assert!(
            recovered_seq > 999_999,
            "recovered allocator reissued sequence numbers: {recovered_seq}"
        );

        recovered.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(cold_data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_aborted_migration_head_does_not_block_cold_recovery() {
        let data_dir = crate::test_support::unique_dir("aborted_head_cold_data");
        let remote_dir = crate::test_support::unique_dir("aborted_head_cold_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();
        store.abort_table_migration("iris.worker").await.unwrap();
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);

        // A server with an empty local store must boot from a remote layout
        // that still holds the aborted table's durable state.
        let cold_data_dir = crate::test_support::unique_dir("aborted_head_cold_target");
        let recovered = Store::new(
            Some(cold_data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
        recovered.bootstrap_maintenance();
        assert_eq!(recovered.recover_tables().await.unwrap(), 0);
        // The aborted table's authority is the legacy path, which the object
        // layout does not carry: the cold server neither recovers nor fails it,
        // and the retained head does not block re-creating the table.
        let missing = recovered.spec_lifecycle("iris.worker").unwrap_err();
        assert!(matches!(missing, StatsError::NamespaceNotFound(_)));
        recovered
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        recovered.write_rows("iris.worker", &ipc, None).unwrap();

        recovered.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(cold_data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn object_backed_transition_never_deletes_the_legacy_object_store() {
        let data_dir = crate::test_support::unique_dir("object_archive_data");
        let remote_dir = crate::test_support::unique_dir("object_archive_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();

        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["legacy"])),
                Arc::new(Int64Array::from(vec![128])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();
        store.maintain_namespace("iris.worker", true).await.unwrap();

        let archive_dir = remote_dir.join("iris.worker");
        let archived = std::fs::read_dir(&archive_dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "parquet")
            })
            .unwrap();
        let retained_orphan = archive_dir.join("pre-object-archive.parquet");
        std::fs::copy(&archived, &retained_orphan).unwrap();

        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();
        let table = store.tables.require("iris.worker").unwrap();
        crate::store::table::maintenance::sync_archive(&table)
            .await
            .unwrap();

        assert!(retained_orphan.exists());
        store.shutdown(Duration::from_secs(1)).await;
        assert!(retained_orphan.exists());
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn object_backed_l0_applies_declared_sort_order() {
        let data_dir = crate::test_support::unique_dir("sorted_object_data");
        let remote_dir = crate::test_support::unique_dir("sorted_object_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table("iris.worker", sorted_object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();
        let batch_schema = schema_to_arrow(&worker_schema().with_sort_columns(["mem_bytes"]));
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["w3", "w1", "w2"])),
                Arc::new(Int64Array::from(vec![30, 10, 20])),
                Arc::new(Int64Array::from(vec![3, 1, 2])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();
        let paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(paths.len(), 1);
        let batches =
            crate::store::compaction::executor::read_segment_projected(Path::new(&paths[0]), None)
                .unwrap();
        let mem_bytes = batches
            .iter()
            .flat_map(|batch| {
                let values = batch
                    .column_by_name("mem_bytes")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                (0..values.len())
                    .map(|index| values.value(index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(mem_bytes, vec![10, 20, 30]);
        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn object_backed_compaction_replaces_inputs_in_one_catalog_generation() {
        let data_dir = crate::test_support::unique_dir("object_compaction_data");
        let remote_dir = crate::test_support::unique_dir("object_compaction_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        for (worker, mem_bytes) in [("w1", 10), ("w2", 20)] {
            let batch = RecordBatch::try_new(
                batch_schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![worker])),
                    Arc::new(Int64Array::from(vec![mem_bytes])),
                    Arc::new(Int64Array::from(vec![mem_bytes])),
                ],
            )
            .unwrap();
            let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
            let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
            store
                .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
                .await
                .unwrap();
        }
        assert_eq!(store.query_snapshot("iris.worker").unwrap().paths.len(), 2);
        let before = store.publish_object_catalog("iris.worker").await.unwrap();
        assert!(before.state().catalog().direct_query_segments.is_empty());
        store.maintain_namespace("iris.worker", true).await.unwrap();
        let snapshot = store.query_snapshot("iris.worker").unwrap();
        let paths = snapshot.paths.clone();
        assert_eq!(paths.len(), 1);
        assert_local_content_object(&data_dir, &paths[0]);

        // The compacted segment advertises its derived artifacts as content-addressed
        // objects of their own, not as files named after the output Parquet.
        let artifacts = snapshot
            .artifacts
            .get(&paths[0])
            .expect("the compacted segment carries artifact references");
        let bundle = artifacts
            .bundle
            .as_ref()
            .expect("compaction indexes its L1 output");
        assert!(bundle.exists());
        assert_ne!(
            bundle.as_path(),
            crate::indices::format::bundle_path(Path::new(&paths[0])).as_path()
        );
        assert!(bundle.starts_with(data_dir.join("_finelog/tables/iris.worker/indices")));
        let bundle_name = bundle.file_name().and_then(|name| name.to_str()).unwrap();
        let bundle_hash = bundle_name.strip_suffix(".fidx").unwrap();
        assert_eq!(bundle_hash.len(), 64);
        let after = store.publish_object_catalog("iris.worker").await.unwrap();
        assert_eq!(
            after.state().catalog().catalog_generation,
            before
                .state()
                .catalog()
                .catalog_generation
                .map(|generation| generation + 1)
        );
        let version = after
            .state()
            .catalog()
            .version_segments
            .iter()
            .find(|version| version.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(version.live_segments.len(), 1);
        assert_eq!(version.live_segments[0].row_count, Some(2));
        assert_eq!(version.live_segments[0].level, Some(1));
        assert_eq!(after.state().catalog().direct_query_segments.len(), 1);
        assert_eq!(after.state().catalog().direct_query_high_water, Some(2));

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn migration_rewrites_one_source_into_partitioned_objects() {
        let data_dir = crate::test_support::unique_dir("partitioned_migration_data");
        let remote_dir = crate::test_support::unique_dir("partitioned_migration_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store.bootstrap_maintenance();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["w2", "w1", "w2"])),
                Arc::new(Int64Array::from(vec![30, 20, 10])),
                Arc::new(Int64Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, last_seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", last_seq, Duration::from_secs(10))
            .await
            .unwrap();

        store
            .register_versioned_table("iris.worker", partitioned_object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();

        let status = store.spec_lifecycle("iris.worker").unwrap();
        assert_eq!(status.active_version(), 1);
        let snapshot = store.publish_object_catalog("iris.worker").await.unwrap();
        let version = snapshot
            .state()
            .catalog()
            .version_segments
            .iter()
            .find(|version| version.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(version.live_segments.len(), 2);
        assert_eq!(
            version
                .live_segments
                .iter()
                .map(|segment| segment.row_count.unwrap_or(0))
                .sum::<i64>(),
            3
        );
        let migration_source_ids = version
            .live_segments
            .iter()
            .map(|segment| segment.migration_source_id.as_deref().unwrap_or(""))
            .collect::<HashSet<_>>();
        assert_eq!(migration_source_ids.len(), 1);
        assert!(!migration_source_ids.contains(""));
        assert!(version
            .live_segments
            .iter()
            .all(|segment| segment.migration_source_rows == Some(3)));

        let query = store.query_snapshot("iris.worker").unwrap();
        assert_eq!(query.paths.len(), 2);
        let w2_path = query
            .partitions
            .iter()
            .find(|(_, partition)| partition.value("worker_id") == Some("w2"))
            .map(|(path, _)| path)
            .unwrap();
        let w2_rows =
            crate::store::compaction::executor::read_segment_projected(Path::new(w2_path), None)
                .unwrap()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>();
        assert_eq!(w2_rows, 2);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
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
    fn managed_levanter_registration_can_disable_stale_index_policy() {
        let store = mem_store();
        let mut stale = levanter_metrics_schema();
        stale
            .columns
            .iter_mut()
            .find(|column| column.name == "name")
            .unwrap()
            .index
            .trigram = true;
        stale
            .columns
            .iter_mut()
            .find(|column| column.name == "kind")
            .unwrap()
            .index
            .value_counts = true;
        store
            .catalog
            .register_or_evolve(
                "levanter.metrics",
                stored_form(stale),
                StoragePolicy::default(),
                |_| unreachable!("fresh catalog registration does not merge"),
            )
            .unwrap();

        let effective = store
            .register_managed_table(
                "levanter.metrics",
                levanter_metrics_schema(),
                StoragePolicy::default(),
            )
            .unwrap();
        assert!(effective.columns.iter().all(|column| {
            !column.index.trigram
                && !column.index.value_counts
                && column.index.exact_values.is_empty()
        }));
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
    fn type_change_rejects_and_new_non_nullable_widens() {
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
        let effective = store
            .register_table(
                "iris.worker",
                Schema::new(cols, ""),
                StoragePolicy::default(),
            )
            .unwrap();
        assert!(effective.column("cpu_pct").unwrap().nullable);
    }

    #[test]
    fn redefined_covering_projection_supersedes_instead_of_conflicting() {
        // A binary whose covering projection differs from the catalog's must
        // still register: the catalog rehydrates the registered definition at
        // boot, so a rejection wedges the namespace's ingest on every restart.
        let store = mem_store();
        let projection = |values: &[&str]| {
            CoveringProjection::new("busy-workers", "worker_id", values.to_vec(), ["worker_id"])
        };
        let schema = |values: &[&str]| worker_schema().with_covering_projection(projection(values));

        store
            .register_table("iris.worker", schema(&["w1"]), StoragePolicy::default())
            .unwrap();
        let effective = store
            .register_table("iris.worker", schema(&["w2"]), StoragePolicy::default())
            .unwrap();
        assert_eq!(effective.projections, vec![projection(&["w2"])]);
        assert_eq!(
            store.get_table_schema("iris.worker").unwrap().projections,
            vec![projection(&["w2"])],
        );
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
            assert_eq!(*stats, NamespaceStats::default());
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
    async fn dropped_object_table_is_not_recovered_from_retained_objects() {
        let data_dir = crate::test_support::unique_dir("object_drop_data");
        let recovered_data_dir = crate::test_support::unique_dir("object_drop_recovered");
        let remote_dir = crate::test_support::unique_dir("object_drop_remote");
        let store = Arc::new(
            Store::new(
                Some(data_dir.clone()),
                remote_dir.to_string_lossy().into_owned(),
                crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
                ServeMode::Shadow,
            )
            .unwrap(),
        );
        store.bootstrap_maintenance();
        store
            .register_versioned_table("iris.worker", object_backed_spec(1))
            .unwrap();
        store.publish_object_catalog("iris.worker").await.unwrap();
        let batch_schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["w1"])),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let ipc = crate::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
        let (_, seq) = store.write_rows("iris.worker", &ipc, None).unwrap();
        store
            .await_persisted("iris.worker", seq, Duration::from_secs(10))
            .await
            .unwrap();
        let cache_root = data_dir.join("_finelog/tables/iris.worker");
        assert!(cache_root.exists());

        let dropping = Arc::clone(&store);
        tokio::task::spawn_blocking(move || dropping.drop_table("iris.worker"))
            .await
            .unwrap()
            .unwrap();
        // The drop publishes a tombstone revision. HEAD survives so a later
        // load distinguishes a deleted table from one that never published.
        let deleted = ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ))
        .load("iris.worker")
        .await
        .unwrap()
        .unwrap();
        assert!(deleted.is_tombstoned());
        assert!(remote_dir
            .join("_finelog/tables/iris.worker/catalogs")
            .exists());
        assert!(cache_root.exists(), "cache GC is intentionally a no-op");

        let recovered = Store::new(
            Some(recovered_data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        recovered.bootstrap_maintenance();
        assert_eq!(recovered.recover_tables().await.unwrap(), 0);
        assert!(matches!(
            recovered.get_table_schema("iris.worker"),
            Err(StatsError::NamespaceNotFound(_))
        ));

        // Reopening the original data directory discards the projection the
        // tombstoned table left behind.
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        let reopened = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        reopened.bootstrap_maintenance();
        assert_eq!(reopened.recover_tables().await.unwrap(), 0);
        assert!(matches!(
            reopened.get_table_schema("iris.worker"),
            Err(StatsError::NamespaceNotFound(_))
        ));
        reopened.shutdown(Duration::from_secs(1)).await;

        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(recovered_data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
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
        let store = Store::new(
            Some(dir.clone()),
            String::new(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
        store.bootstrap_maintenance();
        let schema = store.get_table_schema(LOG_NAMESPACE_NAME).unwrap();
        assert_eq!(
            schema.column_names(),
            vec!["seq", "key", "source", "data", "epoch_ms", "level", "cluster"]
        );
        assert!(
            schema.column("cluster").unwrap().nullable,
            "the evolved cluster column is nullable"
        );
        assert!(
            schema.column("key").unwrap().index.trigram,
            "boot enables the key trigram index a pre-existing namespace lacks"
        );
        assert_eq!(
            store.get_policy(LOG_NAMESPACE_NAME).unwrap(),
            seeded_policy,
            "boot evolution must not reset the persisted log policy"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
