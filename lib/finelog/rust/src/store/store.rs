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
use sha2::{Digest, Sha256};

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
use crate::store::telemetry_catalog::{base_record_columns, resource_columns};
use crate::store::types::{NamespaceStats, WriteRowsResult};

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
const MAX_BATCH_ID_BYTES: usize = 256;
const WRITE_ROWS_DIGEST_DOMAIN: &[u8] = b"finelog.write_rows.v1\0";

fn write_rows_payload_sha256(origin_cluster: Option<&str>, arrow_ipc: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(WRITE_ROWS_DIGEST_DOMAIN);
    match origin_cluster {
        Some(origin) => {
            digest.update([1]);
            digest.update((origin.len() as u64).to_be_bytes());
            digest.update(origin.as_bytes());
        }
        None => digest.update([0]),
    }
    digest.update((arrow_ipc.len() as u64).to_be_bytes());
    digest.update(arrow_ipc);
    format!("{:x}", digest.finalize())
}

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
    let mut columns = vec![
        Column::new("key", ColumnType::COLUMN_TYPE_STRING, false),
        Column::new("source", ColumnType::COLUMN_TYPE_STRING, false),
        // The log message body — substring-searched via contains()/LIKE, so
        // it carries the trigram index.
        Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
        Column::new("epoch_ms", ColumnType::COLUMN_TYPE_INT64, false),
        Column::new("level", ColumnType::COLUMN_TYPE_INT32, false),
        Column::new("cluster", ColumnType::COLUMN_TYPE_STRING, true),
    ];
    columns.extend(
        base_record_columns()
            .into_iter()
            .chain(resource_columns())
            .filter(|column| column.name != "cluster")
            .into_iter()
            .map(|mut column| {
                column.nullable = true;
                column
            }),
    );
    columns.extend([
        Column::new("event_name", ColumnType::COLUMN_TYPE_STRING, true),
        Column::new("severity_text", ColumnType::COLUMN_TYPE_STRING, true),
        Column::new("attributes", ColumnType::COLUMN_TYPE_MAP, true),
        Column::new("trace_id", ColumnType::COLUMN_TYPE_STRING, true),
        Column::new("span_id", ColumnType::COLUMN_TYPE_STRING, true),
    ]);
    Schema::new(columns, "key")
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

    /// Decode, validate, deduplicate, and append a non-empty WriteRows batch.
    ///
    /// `batch_id` is required and scoped by namespace. The server computes the
    /// SHA-256 digest over a domain separator, authenticated origin (or the
    /// trusted/local sentinel), and exact Arrow IPC payload. Empty batches are
    /// rejected rather than acknowledged without reconstructible metadata.
    /// `origin_cluster` is the authenticated origin the
    /// rows are attributed to (`Some` for a forwarding JWT; `None` for a
    /// trusted-network writer, which names its own origin — empty for a local
    /// write). When set, it overwrites the implicit `cluster` column after
    /// alignment so origin does not depend on the sender having stamped it.
    pub fn write_rows(
        &self,
        name: &str,
        batch_id: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
    ) -> Result<WriteRowsResult, StatsError> {
        let payload_sha256 = write_rows_payload_sha256(origin_cluster, arrow_ipc);
        self.write_rows_with_payload_digest(
            name,
            batch_id,
            arrow_ipc,
            origin_cluster,
            &payload_sha256,
        )
    }

    /// Append a REST-ingested batch using the digest of the authoritative request.
    ///
    /// The REST boundary computes this digest over the authenticated origin,
    /// endpoint, content type, and exact uncompressed request bytes. That digest
    /// must be shared by every namespace sub-batch so changing any authoritative
    /// request semantic conflicts even when the changed records route elsewhere.
    pub(crate) fn write_rows_with_payload_digest(
        &self,
        name: &str,
        batch_id: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
        payload_sha256: &str,
    ) -> Result<WriteRowsResult, StatsError> {
        let (engine, aligned) = self.validate_rows_for_write(
            name,
            batch_id,
            arrow_ipc,
            origin_cluster,
            payload_sha256,
        )?;
        engine.append_idempotent_batch(&aligned, batch_id, payload_sha256)
    }

    /// Run the exact WriteRows decode/alignment checks without allocating a receipt.
    ///
    /// REST uses this for all namespace children before it durably reserves the
    /// request digest. A validation or encoded-size rejection therefore cannot
    /// leave a visible batch intent or completion record behind.
    pub(crate) fn preflight_rows(
        &self,
        name: &str,
        batch_id: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
        payload_sha256: &str,
    ) -> Result<(), StatsError> {
        self.validate_rows_for_write(name, batch_id, arrow_ipc, origin_cluster, payload_sha256)?;
        Ok(())
    }

    fn validate_rows_for_write(
        &self,
        name: &str,
        batch_id: &str,
        arrow_ipc: &[u8],
        origin_cluster: Option<&str>,
        payload_sha256: &str,
    ) -> Result<(Arc<Namespace>, AlignedBatch), StatsError> {
        use crate::store::ipc::decode_one_record_batch;
        use crate::store::schema::{
            stamp_cluster_column, validate_and_align_batch, MAX_WRITE_ROWS_BYTES,
            MAX_WRITE_ROWS_ROWS,
        };

        if batch_id.is_empty() || batch_id.len() > MAX_BATCH_ID_BYTES {
            return Err(StatsError::SchemaValidation(format!(
                "WriteRows batch_id must contain 1..={MAX_BATCH_ID_BYTES} bytes"
            )));
        }
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
        if batch.num_rows() == 0 {
            return Err(StatsError::SchemaValidation(
                "WriteRows batches must contain at least one row".to_string(),
            ));
        }
        if payload_sha256.len() != 64
            || !payload_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(StatsError::SchemaValidation(
                "WriteRows payload digest must be 64 hexadecimal characters".to_string(),
            ));
        }
        let engine = self.require_engine(name)?;
        let mut aligned: AlignedBatch = validate_and_align_batch(&batch, engine.schema())?;
        if let Some(origin) = origin_cluster {
            stamp_cluster_column(&mut aligned, origin);
        }
        Ok((engine, aligned))
    }

    /// Append log columns to the reserved `log` namespace, returning the last
    /// seq (or `-1`). `columns` are the six non-seq log columns in registered
    /// order (key/source/data/epoch_ms/level/cluster), prepared by the caller
    /// outside the lock.
    pub fn append_log_columns(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        num_rows: usize,
    ) -> Result<i64, StatsError> {
        let engine = self.require_engine(LOG_NAMESPACE_NAME)?;
        Ok(engine.append_log_batch(columns, num_rows))
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

    /// Snapshot providers for referenced bare namespaces over sealed segments.
    ///
    /// Unknown and SQL-qualified references are omitted so DataFusion reports
    /// its normal catalog error. Queries without table references avoid every
    /// namespace snapshot.
    pub fn query_providers_for(
        &self,
        references: &std::collections::BTreeSet<datafusion::common::TableReference>,
    ) -> Result<Vec<RegisteredProvider>, StatsError> {
        let mut out = Vec::new();
        for reference in references {
            let datafusion::common::TableReference::Bare { table } = reference else {
                continue;
            };
            let name = table.as_ref();
            let engine = match self.engines.lock().unwrap().get(name) {
                Some(e) => Arc::clone(e),
                None => continue,
            };
            let arrow_schema = Arc::clone(engine.arrow_schema());
            let paths = engine.query_snapshot().paths;
            let provider = NamespaceProvider::build(arrow_schema, &paths)
                .map_err(|e| StatsError::Internal(format!("build provider {name:?}: {e}")))?;
            out.push(RegisteredProvider {
                name: name.to_string(),
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
    use std::sync::Barrier;

    use arrow::array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;
    use crate::store::catalog::CATALOG_DB_FILENAME;
    use crate::store::ipc::encode_ipc;
    use crate::store::namespace::{
        inject_crash_after_segment_rename, inject_failure_after_receipt_manifest,
    };
    use crate::store::receipts::{receipt_manifest_count, write_legacy_receipt_manifest_fixture};
    use crate::store::types::{BatchReceipt, ReceiptState};
    use crate::test_support::unique_dir;

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
    fn log_append_accounts_for_aligned_nullable_columns() {
        let store = mem_store();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["key"])),
            Arc::new(StringArray::from(vec!["source"])),
            Arc::new(StringArray::from(vec!["message"])),
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(Int32Array::from(vec![2_i32])),
            Arc::new(StringArray::from(vec!["cluster"])),
        ];
        let input_and_seq_bytes = columns
            .iter()
            .map(crate::store::schema::array_buffer_size)
            .sum::<i64>()
            + 8;

        store.append_log_columns(columns, 1).unwrap();

        let accounted = store
            .require_engine(LOG_NAMESPACE_NAME)
            .unwrap()
            .stats()
            .byte_size;
        assert!(
            accounted > input_and_seq_bytes,
            "expanded nullable telemetry columns must contribute to buffered-byte accounting"
        );
    }

    fn worker_batch(worker_ids: &[&str], offset: i64) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("mem_bytes", DataType::Int64, false),
            Field::new("timestamp_ms", DataType::Int64, false),
        ]));
        let len = worker_ids.len() as i64;
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(worker_ids.to_vec())),
                Arc::new(Int64Array::from_iter_values(offset..offset + len)),
                Arc::new(Int64Array::from_iter_values(
                    1_000 + offset..1_000 + offset + len,
                )),
            ],
        )
        .unwrap()
    }

    fn worker_ipc(worker_ids: &[&str], offset: i64) -> Vec<u8> {
        let batch = worker_batch(worker_ids, offset);
        encode_ipc(&batch.schema(), &[batch]).unwrap()
    }

    async fn stop_background_tasks(store: &Store) {
        let engines: Vec<Arc<Namespace>> =
            store.engines.lock().unwrap().values().cloned().collect();
        for engine in engines {
            engine.stop_and_join(Duration::from_secs(5)).await;
        }
    }

    async fn stopped_disk_store(tag: &str) -> (Arc<Store>, PathBuf, Arc<Namespace>) {
        let dir = unique_dir(tag);
        let store = Arc::new(Store::new(Some(dir.clone()), String::new()).unwrap());
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        stop_background_tasks(&store).await;
        let engine = store.require_engine("iris.worker").unwrap();
        (store, dir, engine)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_write_rows_retries_share_pending_receipt_and_conflicts() {
        let (store, dir, engine) = stopped_disk_store("concurrent_receipts").await;
        let ipc = Arc::new(worker_ipc(&["w-1", "w-2"], 0));
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let store = Arc::clone(&store);
            let ipc = Arc::clone(&ipc);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                store.write_rows("iris.worker", "same-pending", &ipc, None)
            }));
        }
        barrier.wait();
        let results: Vec<WriteRowsResult> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap().unwrap())
            .collect();

        assert_eq!(engine.stats().row_count, 2);
        assert_eq!(
            results.iter().filter(|result| result.deduplicated).count(),
            1
        );
        assert!(results
            .iter()
            .all(|result| result.receipt_state == ReceiptState::Pending));
        assert_eq!(results[0].receipt, results[1].receipt);
        assert!(store
            .catalog
            .list_batch_receipts("iris.worker")
            .unwrap()
            .is_empty());

        let payloads = [
            Arc::new(worker_ipc(&["left"], 10)),
            Arc::new(worker_ipc(&["right"], 20)),
        ];
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for payload in payloads {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                store.write_rows("iris.worker", "conflicting-pending", &payload, None)
            }));
        }
        barrier.wait();
        let results: Vec<Result<WriteRowsResult, StatsError>> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(StatsError::IdempotencyConflict(_))))
                .count(),
            1
        );
        assert_eq!(engine.stats().row_count, 3);

        drop(engine);
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn write_rows_digest_includes_authoritative_origin() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let ipc = worker_ipc(&["w-1"], 0);
        let accepted = store
            .write_rows("iris.worker", "origin-bound", &ipc, None)
            .unwrap();
        assert_eq!(accepted.receipt_state, ReceiptState::Durable);

        let conflict = store.write_rows("iris.worker", "origin-bound", &ipc, Some("cluster-a"));
        assert!(matches!(conflict, Err(StatsError::IdempotencyConflict(_))));
    }

    #[test]
    fn empty_write_rows_is_rejected_without_a_receipt() {
        let store = mem_store();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        let ipc = worker_ipc(&[], 0);

        assert!(matches!(
            store.write_rows("iris.worker", "empty", &ipc, None),
            Err(StatsError::SchemaValidation(_))
        ));
        assert!(store
            .catalog
            .list_batch_receipts("iris.worker")
            .unwrap()
            .is_empty());
        assert_eq!(
            store
                .require_engine("iris.worker")
                .unwrap()
                .stats()
                .row_count,
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn crash_after_segment_rename_repairs_receipt_and_deduplicates() {
        let (store, dir, engine) = stopped_disk_store("receipt_crash").await;
        let ipc = worker_ipc(&["w-1", "w-2"], 0);
        let result = store
            .write_rows("iris.worker", "crash-batch", &ipc, None)
            .unwrap();
        assert_eq!(result.receipt_state, ReceiptState::Pending);
        inject_crash_after_segment_rename("iris.worker");

        let crash = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.flush_once().unwrap();
        }));
        assert!(crash.is_err());
        let namespace_dir = dir.join("iris.worker");
        assert_eq!(receipt_manifest_count(&namespace_dir), 0);
        assert!(store
            .catalog
            .list_batch_receipts("iris.worker")
            .unwrap()
            .is_empty());

        drop(engine);
        drop(store);
        let reopened = Arc::new(Store::new(Some(dir.clone()), String::new()).unwrap());
        stop_background_tasks(&reopened).await;
        assert_eq!(receipt_manifest_count(&namespace_dir), 1);
        let duplicate = reopened
            .write_rows("iris.worker", "crash-batch", &ipc, None)
            .unwrap();
        assert!(duplicate.deduplicated);
        assert_eq!(duplicate.receipt_state, ReceiptState::Durable);
        assert_eq!(
            reopened
                .require_engine("iris.worker")
                .unwrap()
                .stats()
                .row_count,
            2
        );

        drop(reopened);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_after_manifest_failure_reuses_receipt_commit_time() {
        let (store, dir, engine) = stopped_disk_store("receipt_manifest_retry").await;
        let ipc = worker_ipc(&["w-1", "w-2"], 0);
        store
            .write_rows("iris.worker", "manifest-retry", &ipc, None)
            .unwrap();
        inject_failure_after_receipt_manifest("iris.worker");

        assert!(matches!(
            engine.flush_once(),
            Err(StatsError::Internal(message))
                if message == "injected failure after durable receipt manifest"
        ));
        let namespace_dir = dir.join("iris.worker");
        assert_eq!(receipt_manifest_count(&namespace_dir), 1);
        assert!(store
            .catalog
            .list_batch_receipts("iris.worker")
            .unwrap()
            .is_empty());

        engine.flush_once().unwrap();
        let receipts = store.catalog.list_batch_receipts("iris.worker").unwrap();
        assert_eq!(receipts.len(), 1);
        assert!(receipts[0].committed_at_ms > 0);
        let duplicate = store
            .write_rows("iris.worker", "manifest-retry", &ipc, None)
            .unwrap();
        assert!(duplicate.deduplicated);
        assert_eq!(duplicate.receipt_state, ReceiptState::Durable);
        assert_eq!(duplicate.receipt, receipts[0]);
        assert_eq!(receipt_manifest_count(&namespace_dir), 1);
        assert_eq!(engine.stats().row_count, 2);

        drop(engine);
        drop(store);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_migrates_old_receipt_index_and_repairs_from_legacy_manifest() {
        let dir = unique_dir("legacy_receipt_index");
        let store = Arc::new(Store::new(Some(dir.clone()), String::new()).unwrap());
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        stop_background_tasks(&store).await;
        drop(store);

        let connection = rusqlite::Connection::open(dir.join(CATALOG_DB_FILENAME)).unwrap();
        connection
            .execute_batch(
                r#"
                ALTER TABLE batch_receipts RENAME TO batch_receipts_with_commit_time;
                CREATE TABLE batch_receipts (
                    namespace      TEXT NOT NULL,
                    batch_id       TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    rows_written   INTEGER NOT NULL,
                    first_seq      INTEGER NOT NULL,
                    last_seq       INTEGER NOT NULL,
                    PRIMARY KEY (namespace, batch_id)
                );
                INSERT INTO batch_receipts
                    (namespace, batch_id, payload_sha256, rows_written, first_seq, last_seq)
                VALUES
                    ('iris.worker', 'legacy-batch', 'legacy-digest', 2, 1, 2);
                DROP TABLE batch_receipts_with_commit_time;
                "#,
            )
            .unwrap();
        drop(connection);
        write_legacy_receipt_manifest_fixture(&dir.join("iris.worker"));

        let reopened = Arc::new(Store::new(Some(dir.clone()), String::new()).unwrap());
        stop_background_tasks(&reopened).await;
        assert_eq!(
            reopened.catalog.list_batch_receipts("iris.worker").unwrap(),
            vec![BatchReceipt {
                batch_id: "legacy-batch".to_string(),
                payload_sha256: "legacy-digest".to_string(),
                rows_written: 2,
                first_seq: 1,
                last_seq: 2,
                committed_at_ms: 0,
            }]
        );

        drop(reopened);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn receipt_manifest_survives_compaction_and_restart() {
        let (store, dir, engine) = stopped_disk_store("receipt_compaction").await;
        let ipc = worker_ipc(&["w-1", "w-2"], 0);
        let result = store
            .write_rows("iris.worker", "compacted-batch", &ipc, None)
            .unwrap();
        engine.flush_once().unwrap();
        assert_eq!(receipt_manifest_count(&dir.join("iris.worker")), 1);

        let compacting = Arc::clone(&engine);
        tokio::task::spawn_blocking(move || compacting.force_compact_l0())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(receipt_manifest_count(&dir.join("iris.worker")), 1);
        assert!(engine
            .query_snapshot()
            .paths
            .iter()
            .all(|path| !path.contains("seg_L0_")));

        drop(engine);
        drop(store);
        let reopened = Arc::new(Store::new(Some(dir.clone()), String::new()).unwrap());
        stop_background_tasks(&reopened).await;
        let duplicate = reopened
            .write_rows("iris.worker", "compacted-batch", &ipc, None)
            .unwrap();
        assert!(duplicate.deduplicated);
        assert_eq!(duplicate.receipt.batch_id, result.receipt.batch_id);
        assert_eq!(
            duplicate.receipt.payload_sha256,
            result.receipt.payload_sha256
        );
        assert_eq!(duplicate.receipt.rows_written, result.receipt.rows_written);
        assert_eq!(duplicate.receipt.first_seq, result.receipt.first_seq);
        assert_eq!(duplicate.receipt.last_seq, result.receipt.last_seq);
        assert!(duplicate.receipt.committed_at_ms > 0);
        assert_eq!(duplicate.receipt_state, ReceiptState::Durable);
        assert_eq!(receipt_manifest_count(&dir.join("iris.worker")), 1);

        drop(reopened);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn durable_duplicate_does_not_depend_on_local_persisted_watermark() {
        let (store, dir, engine) = stopped_disk_store("receipt_remote").await;
        let ipc = worker_ipc(&["w-1"], 0);
        let result = store
            .write_rows("iris.worker", "remote-batch", &ipc, None)
            .unwrap();
        engine.flush_once().unwrap();
        let segment = store
            .catalog
            .list_segments("iris.worker")
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        store
            .catalog
            .set_location(
                "iris.worker",
                &segment.path,
                crate::store::types::SegmentLocation::Remote,
            )
            .unwrap();
        std::fs::remove_file(&segment.path).unwrap();

        drop(engine);
        drop(store);
        let reopened = Arc::new(Store::new(Some(dir.clone()), String::new()).unwrap());
        stop_background_tasks(&reopened).await;
        let reopened_engine = reopened.require_engine("iris.worker").unwrap();
        assert_eq!(*reopened_engine.watch_persisted_seq().borrow(), -1);
        let duplicate = reopened
            .write_rows("iris.worker", "remote-batch", &ipc, None)
            .unwrap();
        assert!(duplicate.deduplicated);
        assert_eq!(duplicate.receipt.batch_id, result.receipt.batch_id);
        assert_eq!(
            duplicate.receipt.payload_sha256,
            result.receipt.payload_sha256
        );
        assert_eq!(duplicate.receipt.rows_written, result.receipt.rows_written);
        assert_eq!(duplicate.receipt.first_seq, result.receipt.first_seq);
        assert_eq!(duplicate.receipt.last_seq, result.receipt.last_seq);
        assert!(duplicate.receipt.committed_at_ms > 0);
        assert_eq!(duplicate.receipt_state, ReceiptState::Durable);
        assert!(matches!(
            reopened
                .await_persisted("iris.worker", duplicate.receipt.last_seq, Duration::ZERO,)
                .await,
            Err(StatsError::DeadlineExceeded(_))
        ));

        drop(reopened_engine);
        drop(reopened);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_fails_closed_on_sqlite_only_receipt() {
        let (store, dir, engine) = stopped_disk_store("receipt_corruption").await;
        store
            .catalog
            .upsert_batch_receipt(
                "iris.worker",
                &BatchReceipt {
                    batch_id: "orphan".to_string(),
                    payload_sha256: "digest".to_string(),
                    rows_written: 1,
                    first_seq: 1,
                    last_seq: 1,
                    committed_at_ms: 1,
                },
            )
            .unwrap();
        drop(engine);
        drop(store);

        let error = match Store::new(Some(dir.clone()), String::new()) {
            Ok(_) => panic!("startup accepted a SQLite-only receipt"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("SQLite receipt has no durable manifest/footer metadata"));
        std::fs::remove_dir_all(dir).ok();
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
        let dir = unique_dir("evolve_log");

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

        // Boot over that catalog: the schema gains every current nullable log
        // identity column after the original five, and the policy is preserved.
        let store = Store::new(Some(dir.clone()), String::new()).unwrap();
        let schema = store.get_table_schema(LOG_NAMESPACE_NAME).unwrap();
        assert_eq!(
            schema.column_names(),
            with_implicit_seq(log_registered_schema()).column_names()
        );
        for column in schema.columns.iter().skip(6) {
            assert!(
                column.nullable,
                "evolved column {:?} is nullable",
                column.name
            );
        }
        assert_eq!(
            store.get_policy(LOG_NAMESPACE_NAME).unwrap(),
            seeded_policy,
            "boot evolution must not reset the persisted log policy"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
