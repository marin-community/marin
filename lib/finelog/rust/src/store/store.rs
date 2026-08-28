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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use buffa::{Message, MessageView};
use clap::ValueEnum;

use crate::errors::StatsError;
use crate::ingestion_policy::IngestionBatchSource;
use crate::policies::{
    physical_partition_policy_for, schema_for_namespace, segment_indexes_enabled_for,
    storage_policy_for, PolicyRegistry,
};
use crate::proto::finelog::stats::{ColumnType, L0Mode, SchemaView};
use crate::query::index_cache::IndexCache;
use crate::query::provider::NamespaceProvider;
use crate::query::RegisteredProvider;
use crate::store::catalog::{
    Catalog, RecoveredNativeSegment, RegisteredNamespace, TableSpecStatus,
};
use crate::store::ipc::decode_one_record_batch;
use crate::store::namespace::{native_cache_path, relative_cache_path, Namespace};
use crate::store::namespace_name::validate_namespace_name;
use crate::store::native_catalog::{CatalogSnapshot, NativeCatalog};
use crate::store::policy::StoragePolicy;
use crate::store::remote::build_remote_store;
use crate::store::schema::{
    merge_managed_schema, merge_schemas, resolve_key_column, schema_from_proto_view,
    stamp_cluster_column, stored_form, validate_and_align_batch,
    validate_and_align_forwarded_batch, validate_index_policies, AlignedBatch, Column, Schema,
    MAX_WRITE_ROWS_BYTES, MAX_WRITE_ROWS_ROWS,
};
use crate::store::table_spec::ValidatedTableSpec;
use crate::store::types::NamespaceStats;
use crate::telemetry_policy::{TelemetryRootWriteMode, TELEMETRY_NAMESPACE};

/// The privileged log namespace name.
pub const LOG_NAMESPACE_NAME: &str = "log";
/// Its on-disk subdirectory.
pub const LOG_NAMESPACE_DIR: &str = "log";
const STORE_LOCK_FILENAME: &str = ".finelog-store.lock";

fn writer_epoch() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ u64::from(std::process::id())
}

fn recovered_segment_cache_path(
    namespace_dir: &Path,
    namespace: &str,
    object_key: &str,
    table_spec_version: u64,
) -> Result<PathBuf, StatsError> {
    if table_spec_version > 0 {
        return native_cache_path(namespace_dir, namespace, object_key);
    }
    relative_cache_path(namespace_dir, object_key)
}

/// Bounded budget for stopping + joining a namespace's background tasks during a
/// live lifecycle transition (re-register replacement, drop). Runs inside the
/// RPC's `spawn_blocking` worker, so it must not block long: a task that misses
/// this window is aborted rather than wedging the worker. Distinct from the
/// process-shutdown drain budget passed to [`Store::shutdown`] at SIGTERM.
const NAMESPACE_LIFECYCLE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of appending one schema-compatible federated batch.
pub struct ForwardedWrite {
    pub rows_written: i64,
    pub persisted_targets: Vec<(String, i64)>,
    pub ignored_columns: Vec<String>,
}

pub struct VersionedRegistration {
    pub schema: Schema,
    pub policy: StoragePolicy,
    pub table_spec_status: TableSpecStatus,
    pub object_native: bool,
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
    pub partitions: BTreeMap<String, crate::partition_policy::SegmentPartition>,
    pub min_seq: Option<i64>,
    pub index_cache: Arc<IndexCache>,
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

/// Store backed by the Rust catalog plus per-namespace durability engines.
///
/// The catalog owns the persistent registry + segments table; the `engines`
/// map owns one `Namespace` per live namespace (built at boot from the catalog
/// and on `register_table`). The data path (WriteRows / PushLogs) routes through
/// these engines; the metadata RPCs stay on the catalog.
pub struct Store {
    data_dir: Option<PathBuf>,
    remote_log_dir: String,
    mode: ServeMode,
    catalog: Arc<Catalog>,
    native_catalog: Option<NativeCatalog>,
    writer_epoch: u64,
    engines: Mutex<HashMap<String, Arc<Namespace>>>,
    /// Serializes the complete catalog-and-engine registration lifecycle per namespace.
    /// Concurrent first registrations must not build and displace separate engines.
    namespace_registration_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
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
    index_cache: Arc<IndexCache>,
    index_backfill_slot: Arc<Mutex<()>>,
    physical_layout_migration_slot: Arc<Mutex<()>>,
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
    /// `remote_log_dir` configures the per-namespace offload target (empty
    /// disables sync). Pass it through to each `Namespace`.
    pub fn new(
        data_dir: Option<PathBuf>,
        remote_log_dir: String,
        index_cache_mb: usize,
        mode: ServeMode,
    ) -> Result<Store, StatsError> {
        Self::new_with_telemetry_root_write_mode(
            data_dir,
            remote_log_dir,
            index_cache_mb,
            mode,
            TelemetryRootWriteMode::SemanticOnly,
        )
    }

    pub fn new_with_telemetry_root_write_mode(
        data_dir: Option<PathBuf>,
        remote_log_dir: String,
        index_cache_mb: usize,
        mode: ServeMode,
        telemetry_root_write_mode: TelemetryRootWriteMode,
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
        let native_catalog = if data_dir.is_some() {
            build_remote_store(&remote_log_dir)?.map(NativeCatalog::new)
        } else {
            None
        };
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
            mode,
            catalog,
            native_catalog,
            writer_epoch: writer_epoch(),
            engines: Mutex::new(HashMap::new()),
            namespace_registration_locks: Mutex::new(HashMap::new()),
            query_visibility: Arc::new(tokio::sync::RwLock::new(())),
            index_cache: Arc::new(IndexCache::new(index_cache_mb)),
            index_backfill_slot: Arc::new(Mutex::new(())),
            physical_layout_migration_slot: Arc::new(Mutex::new(())),
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

    /// Return the root telemetry sequence fence before this process accepts writes.
    pub fn telemetry_root_max_seq(&self) -> Result<i64, StatsError> {
        match self.engines.lock().unwrap().get(TELEMETRY_NAMESPACE) {
            Some(engine) => Ok(engine.stats().max_seq),
            None => Ok(self
                .catalog
                .aggregate_namespace_stats(TELEMETRY_NAMESPACE)?
                .max_seq),
        }
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
        if self.mode == ServeMode::Shadow {
            tracing::info!("shadow mode: maintenance not started");
            return;
        }
        let engines: Vec<Arc<Namespace>> = self.engines.lock().unwrap().values().cloned().collect();
        for engine in &engines {
            engine.spawn_maintenance(true);
        }
    }

    /// Rebuild namespaces, TableSpecs, segment pointers, and sequence fences
    /// from remote HEAD/catalog snapshots before the server accepts traffic.
    ///
    /// Returns the number of namespaces whose remote generation was recovered.
    pub async fn recover_native_namespaces(&self) -> Result<usize, StatsError> {
        let Some(native_catalog) = &self.native_catalog else {
            return Ok(0);
        };
        let remote = build_remote_store(&self.remote_log_dir)?.ok_or_else(|| {
            StatsError::Internal("native catalog configured without a remote store".to_string())
        })?;
        let mut recovered_count = 0;
        for namespace in remote.list_native_namespaces().await? {
            validate_namespace_name(&namespace, self.data_dir.as_deref())?;
            let Some(snapshot) = native_catalog.load(&namespace).await? else {
                continue;
            };
            let remote_generation = snapshot.catalog.catalog_generation.unwrap_or(0);
            if self
                .catalog
                .table_spec_status(&namespace)?
                .catalog_generation
                == remote_generation
            {
                continue;
            }
            let schema_spec =
                snapshot
                    .catalog
                    .retained_table_specs
                    .iter()
                    .find(|spec| spec.version == snapshot.catalog.active_table_spec_version)
                    .or_else(|| {
                        snapshot.catalog.retained_table_specs.iter().find(|spec| {
                            spec.version == snapshot.catalog.desired_table_spec_version
                        })
                    })
                    .or_else(|| snapshot.catalog.retained_table_specs.last())
                    .ok_or_else(|| {
                        StatsError::Internal(format!(
                            "native catalog for {namespace:?} has no retained TableSpec"
                        ))
                    })?;
            let schema_proto = schema_spec.logical_schema.as_option().ok_or_else(|| {
                StatsError::Internal(format!(
                    "native catalog TableSpec for {namespace:?} has no logical schema"
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
            let namespace_dir = self.namespace_dir(&namespace)?.ok_or_else(|| {
                StatsError::Internal("native recovery requires a disk-backed store".to_string())
            })?;
            let mut recovered = HashMap::<String, RecoveredNativeSegment>::new();
            for version in &snapshot.catalog.version_segments {
                for segment in version
                    .live_segments
                    .iter()
                    .chain(version.retired_segments.iter())
                {
                    let source = segment.source.as_option().ok_or_else(|| {
                        StatsError::Internal(format!(
                            "native catalog segment for {namespace:?} has no source"
                        ))
                    })?;
                    let object_key = source.uri.as_deref().ok_or_else(|| {
                        StatsError::Internal(format!(
                            "native catalog segment for {namespace:?} has an empty source URI"
                        ))
                    })?;
                    let table_spec_version = segment
                        .schema_revision
                        .unwrap_or(version.table_spec_version.unwrap_or(0));
                    let cache_path = recovered_segment_cache_path(
                        &namespace_dir,
                        &namespace,
                        object_key,
                        table_spec_version,
                    )?;
                    let path = cache_path.to_string_lossy().into_owned();
                    recovered
                        .entry(path.clone())
                        .or_insert_with(|| RecoveredNativeSegment {
                            row: crate::store::types::SegmentRow {
                                namespace: namespace.clone(),
                                path,
                                level: segment.level.unwrap_or(0),
                                min_seq: segment.min_seq.unwrap_or(0),
                                max_seq: segment.max_seq.unwrap_or(0),
                                row_count: segment.row_count.unwrap_or(0),
                                byte_size: i64::try_from(source.byte_size.unwrap_or(0))
                                    .unwrap_or(i64::MAX),
                                created_at_ms: segment.created_at_ms.unwrap_or(0),
                                min_key_value: segment.min_key_value.clone(),
                                max_key_value: segment.max_key_value.clone(),
                                partition: segment
                                    .partition_json
                                    .as_deref()
                                    .and_then(|value| serde_json::from_str(value).ok()),
                                location: crate::store::types::SegmentLocation::Remote,
                            },
                            table_spec_version,
                            source: source.clone(),
                            migration_backfill: segment
                                .migration_backfill
                                .unwrap_or_else(|| object_key.contains("/backfill/")),
                            migration_source_id: segment.migration_source_id.clone(),
                            migration_source_rows: segment.migration_source_rows,
                        });
                }
            }
            let recovered: Vec<_> = recovered.into_values().collect();
            self.catalog.restore_native_snapshot(
                &namespace,
                schema.clone(),
                policy.clone(),
                &snapshot.catalog,
                &recovered,
            )?;
            let prior = self.engines.lock().unwrap().remove(&namespace);
            if let Some(prior) = prior {
                prior.shutdown(NAMESPACE_LIFECYCLE_SHUTDOWN_TIMEOUT).await;
            }
            self.build_engine(&namespace, schema, policy, false)?;
            recovered_count += 1;
        }
        Ok(recovered_count)
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
            Arc::clone(&self.index_cache),
            Arc::clone(&self.index_backfill_slot),
            Arc::clone(&self.physical_layout_migration_slot),
            &self.remote_log_dir,
            policy,
            self.writer_epoch,
        )?;
        if spawn_maint && self.mode == ServeMode::Live {
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
        if validated.l0_mode == L0Mode::L0_MODE_OBJECT_NATIVE
            && self.remote_log_dir.trim().is_empty()
        {
            return Err(StatsError::SchemaValidation(
                "object-native table specifications require a configured remote_log_dir"
                    .to_string(),
            ));
        }
        let schema = validated.schema.clone();
        let policy = validated.cache_policy.clone();
        let (schema, table_spec_status) = self.register_table_with(
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
            table_spec_status: table_spec_status.expect("versioned registration returns status"),
            object_native: validated.l0_mode == L0Mode::L0_MODE_OBJECT_NATIVE,
        })
    }

    /// Make one complete local metadata snapshot visible to direct readers.
    ///
    /// Returns the selected remote snapshot after publication.
    pub async fn publish_native_catalog(
        &self,
        namespace: &str,
    ) -> Result<CatalogSnapshot, StatsError> {
        let native = self.native_catalog.as_ref().ok_or_else(|| {
            StatsError::SchemaValidation(
                "object-native table specifications require a configured remote_log_dir"
                    .to_string(),
            )
        })?;
        let namespace_dir = self.namespace_dir(namespace)?.ok_or_else(|| {
            StatsError::SchemaValidation(
                "object-native catalogs require a disk-backed store".to_string(),
            )
        })?;
        native
            .publish_local(&self.catalog, namespace, &namespace_dir, self.writer_epoch)
            .await
    }

    fn register_table_with(
        &self,
        name: &str,
        schema: Schema,
        policy: StoragePolicy,
        registration: SchemaRegistration,
        table_spec: Option<&ValidatedTableSpec>,
    ) -> Result<(Schema, Option<TableSpecStatus>), StatsError> {
        // Validate the name (and fence the `log` dir special-case) first.
        self.namespace_dir(name)?;
        validate_index_policies(&schema)?;
        resolve_key_column(&schema)?;
        let stored = stored_form(schema);
        let registration_lock = {
            let mut locks = self.namespace_registration_locks.lock().unwrap();
            Arc::clone(
                locks
                    .entry(name.to_string())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        let _registration_guard = registration_lock.lock().unwrap();
        if let Some(table_spec) = table_spec {
            self.catalog.validate_table_spec_registration(
                name,
                table_spec.spec.version.unwrap_or(0),
                &table_spec.hash,
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
        let had_engine = self.engines.lock().unwrap().contains_key(name);
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
        let table_spec_status = table_spec
            .map(|table_spec| {
                let has_rows = self.catalog.aggregate_namespace_stats(name)?.row_count > 0;
                self.catalog
                    .register_table_spec(name, &table_spec.spec, &table_spec.hash, has_rows)
            })
            .transpose()?;
        if let Some(status) = &table_spec_status {
            if let Some(engine) = self.engines.lock().unwrap().get(name) {
                engine.update_table_spec(status);
            }
        }
        Ok((effective_schema, table_spec_status))
    }

    /// Append a routed batch and return its row count and durability target.
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
            match self.require_engine(namespace) {
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
            let engine = self.require_engine(&destination)?;
            let (mut aligned, ignored) = alignment.align(&partition.batch, engine.schema())?;
            if let Some(origin) = origin_cluster {
                stamp_cluster_column(&mut aligned, origin);
            }
            ignored_columns.extend(ignored);
            prepared_partitions.push((destination, engine, aligned));
        }
        let persisted_targets = prepared_partitions
            .into_iter()
            .map(|(destination, engine, aligned)| {
                let last_seq = engine.append_aligned_batch(&aligned);
                (destination, last_seq)
            })
            .collect();
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

    fn write_physical_rows(
        &self,
        name: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
    ) -> Result<ForwardedWrite, StatsError> {
        let batch = decode_bounded_write_batch(arrow_ipc)?;
        let engine = self.require_engine(name)?;
        let mut aligned = validate_and_align_batch(&batch, engine.schema())?;
        if let Some(origin) = origin_cluster {
            stamp_cluster_column(&mut aligned, origin);
        }
        let n = aligned.num_rows as i64;
        let last_seq = engine.append_aligned_batch(&aligned);
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
            let exact_postings_policy = engine.schema().exact_postings_policy();
            let key_column = engine.key_column().to_string();
            let segments = engine.query_snapshot();
            let provider = NamespaceProvider::build(
                arrow_schema,
                &segments.paths,
                Arc::clone(&self.index_cache),
            )
            .map_err(|e| StatsError::Internal(format!("build provider {:?}: {e}", ns.name)))?
            .with_segment_indexes_enabled(segment_indexes_enabled_for(&ns.name))
            .with_exact_postings_policy(exact_postings_policy)
            .with_segment_key_bounds(key_column, segments.key_bounds)
            .with_segment_partitions(physical_partition_policy_for(&ns.name), segments.partitions);
            out.push(RegisteredProvider {
                name: ns.name,
                provider,
            });
        }
        Ok(out)
    }

    /// Restore missing mirrored cache files before a server-directed query.
    pub async fn ensure_native_query_cache(&self) -> Result<(), StatsError> {
        let engines: Vec<_> = self.engines.lock().unwrap().values().cloned().collect();
        for engine in engines {
            engine.ensure_native_query_cache().await?;
        }
        Ok(())
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
            exact_postings_policy: engine.schema().exact_postings_policy(),
            key_column: engine.key_column().to_string(),
            paths: segments.paths,
            key_bounds: segments.key_bounds,
            partitions: segments.partitions,
            min_seq: segments.min_seq,
            index_cache: Arc::clone(&self.index_cache),
        })
    }

    pub fn index_cache(&self) -> &Arc<IndexCache> {
        &self.index_cache
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
        if self.catalog.set_forward_cursor(target, namespace, cursor)? {
            self.require_engine(namespace)?
                .mark_native_publish_pending();
        }
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

    pub fn table_spec_status(&self, name: &str) -> Result<TableSpecStatus, StatsError> {
        self.catalog.require_live(name)?;
        self.catalog.table_spec_status(name)
    }

    pub async fn rollback_table_version(
        &self,
        name: &str,
        retained_version: u64,
    ) -> Result<TableSpecStatus, StatsError> {
        self.catalog.require_live(name)?;
        let _visibility_guard = self.query_visibility.write().await;
        let status = self.catalog.rollback_table_spec(name, retained_version)?;
        self.apply_table_spec_status(name, &status)?;
        if let Err(error) = self.publish_native_catalog(name).await {
            self.require_engine(name)?.mark_native_publish_pending();
            return Err(error);
        }
        Ok(status)
    }

    fn apply_table_spec_status(
        &self,
        name: &str,
        status: &TableSpecStatus,
    ) -> Result<(), StatsError> {
        let engine = self.require_engine(name)?;
        engine.activate_query_version(status.active_version())?;
        engine.update_table_spec(status);
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
        Store::new(
            None,
            String::new(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap()
    }

    fn object_native_spec(version: u64) -> ValidatedTableSpec {
        object_native_spec_with_query_time(version, 0)
    }

    fn object_native_spec_with_query_time(
        version: u64,
        max_query_time_ms: u64,
    ) -> ValidatedTableSpec {
        let schema = worker_schema();
        let spec = TableSpec {
            version: Some(version),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema)),
            operating_policy: MessageField::some(OperatingPolicy {
                l0_mode: Some(L0Mode::L0_MODE_OBJECT_NATIVE.into()),
                remote_retention: MessageField::some(RemoteRetentionPolicy {
                    retain_forever: Some(true),
                    ..Default::default()
                }),
                max_query_time_ms: (max_query_time_ms > 0).then_some(max_query_time_ms),
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        ValidatedTableSpec::from_view(&view, &schema, &StoragePolicy::default()).unwrap()
    }

    fn partitioned_object_native_spec(version: u64) -> ValidatedTableSpec {
        let schema = worker_schema();
        let spec = TableSpec {
            version: Some(version),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema)),
            source_layout: MessageField::some(SourceLayout {
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
            }),
            operating_policy: MessageField::some(OperatingPolicy {
                l0_mode: Some(L0Mode::L0_MODE_OBJECT_NATIVE.into()),
                remote_retention: MessageField::some(RemoteRetentionPolicy {
                    retain_forever: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        ValidatedTableSpec::from_view(&view, &schema, &StoragePolicy::default()).unwrap()
    }

    fn sorted_object_native_spec(version: u64) -> ValidatedTableSpec {
        let schema = worker_schema().with_sort_columns(["mem_bytes"]);
        let spec = TableSpec {
            version: Some(version),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema)),
            source_layout: MessageField::some(SourceLayout {
                sort_columns: vec!["mem_bytes".to_string()],
                ..Default::default()
            }),
            operating_policy: MessageField::some(OperatingPolicy {
                l0_mode: Some(L0Mode::L0_MODE_OBJECT_NATIVE.into()),
                remote_retention: MessageField::some(RemoteRetentionPolicy {
                    retain_forever: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = spec.encode_to_vec();
        let view = TableSpecView::decode_view(&encoded).unwrap();
        ValidatedTableSpec::from_view(&view, &schema, &StoragePolicy::default()).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn versioned_registration_publishes_and_recovers_remote_head() {
        let data_dir = crate::test_support::unique_dir("versioned_registration_data");
        let remote_dir = crate::test_support::unique_dir("versioned_registration_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();

        let registration = store
            .register_versioned_table("iris.worker", object_native_spec(1))
            .unwrap();
        assert!(registration.object_native);
        assert_eq!(registration.table_spec_status.active_version(), 1);
        let first = store.publish_native_catalog("iris.worker").await.unwrap();
        assert_eq!(first.catalog.active_table_spec_version, Some(1));
        assert_eq!(first.catalog.retained_table_specs.len(), 1);

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
        assert_eq!(paths.len(), 1);
        assert!(paths[0].contains("/_native/namespaces/iris.worker/objects/v1/l0/"));
        let after_flush = store.publish_native_catalog("iris.worker").await.unwrap();
        assert_eq!(after_flush.catalog.catalog_generation, Some(2));
        let version = after_flush
            .catalog
            .version_segments
            .iter()
            .find(|segments| segments.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(version.live_segments.len(), 1);
        assert_eq!(version.live_segments[0].row_count, Some(2));
        assert!(version.live_segments[0]
            .source
            .as_option()
            .unwrap()
            .uri
            .as_deref()
            .unwrap()
            .starts_with("objects/v1/l0/"));

        // An identical registration and publication is a retry, not a new
        // generation. Loading from a fresh RemoteStore recovers the same HEAD.
        store
            .register_versioned_table("iris.worker", object_native_spec(1))
            .unwrap();
        let retry = store.publish_native_catalog("iris.worker").await.unwrap();
        assert_eq!(retry.catalog, after_flush.catalog);
        let recovered = NativeCatalog::new(
            build_remote_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        )
        .load("iris.worker")
        .await
        .unwrap()
        .unwrap();
        assert_eq!(recovered.catalog, after_flush.catalog);

        store
            .set_forward_cursor("hub", "iris.worker", last_seq)
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let with_cursor = NativeCatalog::new(
            build_remote_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        )
        .load("iris.worker")
        .await
        .unwrap()
        .unwrap();
        assert_eq!(with_cursor.catalog.forward_cursors.len(), 1);
        assert_eq!(
            with_cursor.catalog.forward_cursors[0].cursor,
            Some(last_seq)
        );

        std::fs::remove_file(&paths[0]).unwrap();
        store.ensure_native_query_cache().await.unwrap();
        assert!(Path::new(&paths[0]).exists());

        // Simulate a crash after SQLite accepted the next version but before
        // its HEAD CAS. Remote HEAD remains the canonical committed state.
        let unpublished = store
            .register_versioned_table("iris.worker", object_native_spec(2))
            .unwrap();
        assert_eq!(unpublished.table_spec_status.desired_version(), 2);

        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
        std::fs::remove_file(&paths[0]).unwrap();
        let reopened = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        assert_eq!(
            reopened
                .table_spec_status("iris.worker")
                .unwrap()
                .desired_version(),
            2
        );
        assert_eq!(reopened.recover_native_namespaces().await.unwrap(), 1);
        let recovered_status = reopened.table_spec_status("iris.worker").unwrap();
        assert_eq!(recovered_status.active_version(), 1);
        assert_eq!(recovered_status.desired_version(), 0);
        assert!(reopened
            .query_snapshot("iris.worker")
            .unwrap()
            .paths
            .is_empty());
        reopened
            .require_engine("iris.worker")
            .unwrap()
            .boot_reconcile()
            .await
            .unwrap();
        let reopened_paths = reopened.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(reopened_paths, paths);
        reopened.shutdown(Duration::from_secs(1)).await;
        drop(reopened);

        // A crash can leave the mirrored file present while the process-local
        // deque is empty. Recovery must adopt that cache file even when remote
        // HEAD and SQLite already name the same generation.
        let cached_reopen = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        cached_reopen
            .require_engine("iris.worker")
            .unwrap()
            .activate_query_version(0)
            .unwrap();
        assert!(cached_reopen
            .query_snapshot("iris.worker")
            .unwrap()
            .paths
            .is_empty());
        assert_eq!(cached_reopen.recover_native_namespaces().await.unwrap(), 0);
        cached_reopen.ensure_native_query_cache().await.unwrap();
        assert_eq!(
            cached_reopen.query_snapshot("iris.worker").unwrap().paths,
            paths
        );
        cached_reopen.shutdown(Duration::from_secs(1)).await;
        drop(cached_reopen);

        let empty_data_dir = crate::test_support::unique_dir("versioned_empty_recovery_data");
        let recovered_store = Store::new(
            Some(empty_data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        assert_eq!(
            recovered_store.recover_native_namespaces().await.unwrap(),
            1
        );
        assert_eq!(
            recovered_store
                .table_spec_status("iris.worker")
                .unwrap()
                .active_version(),
            1
        );
        assert!(recovered_store
            .query_snapshot("iris.worker")
            .unwrap()
            .paths
            .is_empty());
        recovered_store.ensure_native_query_cache().await.unwrap();
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
    async fn next_version_with_the_same_layout_preserves_existing_rows() {
        let data_dir = crate::test_support::unique_dir("same_layout_upgrade_data");
        let remote_dir = crate::test_support::unique_dir("same_layout_upgrade_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store
            .register_versioned_table("iris.worker", object_native_spec_with_query_time(1, 100))
            .unwrap();
        store.publish_native_catalog("iris.worker").await.unwrap();

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
            .register_versioned_table("iris.worker", object_native_spec_with_query_time(2, 100))
            .unwrap();
        assert_eq!(registration.table_spec_status.active_version(), 1);
        assert_eq!(registration.table_spec_status.desired_version(), 2);
        let transition = store.publish_native_catalog("iris.worker").await.unwrap();
        let active = transition
            .catalog
            .version_segments
            .iter()
            .find(|version| version.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(active.live_segments.len(), 2);
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();

        let status = store.table_spec_status("iris.worker").unwrap();
        assert_eq!(status.active_version(), 2);
        let snapshot = store.query_snapshot("iris.worker").unwrap();
        assert_eq!(snapshot.paths.len(), 2);

        // Observation permits compaction, but its output must remain marked as
        // backfill so a rollback can discard it in favor of version 1.
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let compacted = store
            .catalog
            .native_segments("iris.worker")
            .unwrap()
            .into_iter()
            .filter(|segment| segment.table_spec_version == 2)
            .collect::<Vec<_>>();
        assert_eq!(compacted.len(), 1);
        assert!(compacted[0].migration_backfill);
        let compacted_snapshot = store.publish_native_catalog("iris.worker").await.unwrap();
        let compacted_active = compacted_snapshot
            .catalog
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
        let retired = store.table_spec_status("iris.worker").unwrap();
        assert_eq!(
            retired.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_RETIRED
        );
        assert!(store
            .catalog
            .native_segments("iris.worker")
            .unwrap()
            .iter()
            .all(|segment| segment.table_spec_version == 2));
        assert!(store
            .catalog
            .native_segments("iris.worker")
            .unwrap()
            .iter()
            .all(|segment| !segment.migration_backfill));
        let retired_catalog = store.publish_native_catalog("iris.worker").await.unwrap();
        assert!(retired_catalog
            .catalog
            .version_segments
            .iter()
            .all(|version| version.table_spec_version == Some(2)));

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn legacy_table_migrates_automatically_and_rolls_back_without_duplicate_rows() {
        let data_dir = crate::test_support::unique_dir("table_spec_migration_data");
        let remote_dir = crate::test_support::unique_dir("table_spec_migration_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
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
        assert!(!legacy_paths[0].contains("/_native/"));

        let registration = store
            .register_versioned_table("iris.worker", object_native_spec_with_query_time(1, 1))
            .unwrap();
        assert_eq!(registration.table_spec_status.active_version(), 0);
        assert_eq!(registration.table_spec_status.desired_version(), 1);
        store.publish_native_catalog("iris.worker").await.unwrap();

        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let activated = store.table_spec_status("iris.worker").unwrap();
        assert_eq!(activated.active_version(), 1);
        assert_eq!(activated.desired_version(), 0);
        assert_eq!(
            activated.phase,
            crate::proto::finelog::stats::MigrationPhase::MIGRATION_PHASE_OBSERVING
        );
        let active_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(active_paths.len(), 1);
        assert!(active_paths[0].contains("/_native/namespaces/iris.worker/objects/v1/backfill/"));

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

        let rolled_back = store
            .rollback_table_version("iris.worker", 0)
            .await
            .unwrap();
        assert_eq!(rolled_back.active_version(), 0);
        let rollback_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(rollback_paths.len(), 2);
        assert!(rollback_paths.iter().any(|path| path == &legacy_paths[0]));
        assert!(rollback_paths
            .iter()
            .any(|path| path.contains("/_native/namespaces/iris.worker/objects/v1/l0/")));

        let snapshot = store.publish_native_catalog("iris.worker").await.unwrap();
        let rollback_version = snapshot
            .catalog
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

        store
            .catalog
            .expire_migration_observation("iris.worker")
            .unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let cleaned_records = store.catalog.native_segments("iris.worker").unwrap();
        assert_eq!(cleaned_records.len(), 1);
        assert_eq!(cleaned_records[0].table_spec_version, 0);
        assert!(!cleaned_records[0].migration_backfill);
        let cleaned_paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(cleaned_paths.len(), 2);

        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn object_native_transition_never_deletes_the_legacy_archive() {
        let data_dir = crate::test_support::unique_dir("native_archive_data");
        let remote_dir = crate::test_support::unique_dir("native_archive_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
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
        let retained_orphan = archive_dir.join("pre-native-archive.parquet");
        std::fs::copy(&archived, &retained_orphan).unwrap();

        store
            .register_versioned_table("iris.worker", object_native_spec(1))
            .unwrap();
        store.publish_native_catalog("iris.worker").await.unwrap();
        store
            .require_engine("iris.worker")
            .unwrap()
            .sync_step()
            .await
            .unwrap();

        assert!(retained_orphan.exists());
        store.shutdown(Duration::from_secs(1)).await;
        assert!(retained_orphan.exists());
        std::fs::remove_dir_all(data_dir).ok();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn object_native_l0_applies_declared_sort_order() {
        let data_dir = crate::test_support::unique_dir("sorted_native_data");
        let remote_dir = crate::test_support::unique_dir("sorted_native_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store
            .register_versioned_table("iris.worker", sorted_object_native_spec(1))
            .unwrap();
        store.publish_native_catalog("iris.worker").await.unwrap();
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
    async fn object_native_compaction_replaces_inputs_in_one_catalog_generation() {
        let data_dir = crate::test_support::unique_dir("native_compaction_data");
        let remote_dir = crate::test_support::unique_dir("native_compaction_remote");
        let store = Store::new(
            Some(data_dir.clone()),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
        store
            .register_versioned_table("iris.worker", object_native_spec(1))
            .unwrap();
        store.publish_native_catalog("iris.worker").await.unwrap();
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
        let before = store.publish_native_catalog("iris.worker").await.unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();
        let paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(paths.len(), 1);
        assert!(paths[0].contains("/objects/v1/compact/"));
        let after = store.publish_native_catalog("iris.worker").await.unwrap();
        assert_eq!(
            after.catalog.catalog_generation,
            before
                .catalog
                .catalog_generation
                .map(|generation| generation + 1)
        );
        let version = after
            .catalog
            .version_segments
            .iter()
            .find(|version| version.table_spec_version == Some(1))
            .unwrap();
        assert_eq!(version.live_segments.len(), 1);
        assert_eq!(version.live_segments[0].row_count, Some(2));
        assert_eq!(version.live_segments[0].level, Some(1));

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
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap();
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
            .register_versioned_table("iris.worker", partitioned_object_native_spec(1))
            .unwrap();
        store.publish_native_catalog("iris.worker").await.unwrap();
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();

        let status = store.table_spec_status("iris.worker").unwrap();
        assert_eq!(status.active_version(), 1);
        let snapshot = store.publish_native_catalog("iris.worker").await.unwrap();
        let version = snapshot
            .catalog
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

        let paths = store.query_snapshot("iris.worker").unwrap().paths;
        assert_eq!(paths.len(), 2);
        let w2_path = paths
            .iter()
            .find(|path| path.contains("worker_id=w2"))
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
        let store = Store::new(
            Some(dir.clone()),
            String::new(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_shadow_store_starts_no_maintenance_for_a_namespace_registered_at_runtime() {
        // A shadow boot registers its own namespaces after opening the snapshot,
        // and a runtime `register_table` normally starts that namespace's
        // maintenance task itself. Gating only the boot path would leave a
        // rehearsal free to compact, evict, and rewrite the copy it was handed.
        let live_dir = crate::test_support::unique_dir("maintenance_live");
        let shadow_dir = crate::test_support::unique_dir("maintenance_shadow");
        let mut counts = Vec::new();
        for (dir, mode) in [
            (&live_dir, ServeMode::Live),
            (&shadow_dir, ServeMode::Shadow),
        ] {
            let store = Store::new(
                Some(dir.clone()),
                String::new(),
                crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
                mode,
            )
            .unwrap();
            store.bootstrap_maintenance();
            tokio::task::spawn_blocking(move || {
                store
                    .register_table("iris.worker", worker_schema(), StoragePolicy::default())
                    .unwrap();
                store
                    .require_engine("iris.worker")
                    .unwrap()
                    .background_task_count()
            })
            .await
            .map(|count| counts.push(count))
            .unwrap();
        }
        let (live, shadow) = (counts[0], counts[1]);
        assert!(
            shadow < live,
            "a shadow store must run fewer background tasks than a live one \
             (live {live}, shadow {shadow})"
        );
        std::fs::remove_dir_all(&live_dir).ok();
        std::fs::remove_dir_all(&shadow_dir).ok();
    }
}
