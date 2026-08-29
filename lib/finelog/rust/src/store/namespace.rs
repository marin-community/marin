//! Per-namespace durability engine.
//!
//! Write/flush/durability machinery built on tokio primitives:
//!
//! - The per-namespace insertion lock (`std::sync::Mutex<NsInner>`) guards the
//!   `RamBuffers` (seq counter + chunks + in-flight buffer) and the
//!   `local_segments` deque.
//! - A `tokio::sync::watch::<i64>` `persisted_seq` (init `-1`) is the durability
//!   primitive. The flush task `send`s the new high-water seq **only after** the
//!   parquet file is renamed into place AND the catalog row is committed
//!   (durability-before-ack).
//! - `await_persisted(target)` subscribes to the watch and waits, bounded by a
//!   caller-supplied timeout, after registering flush demand with the scheduler.
//! - Flush cadence belongs to
//!   [`MaintenanceScheduler`](crate::maintenance::scheduler::MaintenanceScheduler).
//!   A namespace records demand; the scheduler seals at most one L0 per
//!   `MIN_FLUSH_INTERVAL` unless the buffer already holds a whole segment.
//!
//! `MemoryNamespace` (no `data_dir`) treats every append as immediately
//! persisted: it stamps into a RAM buffer, advances `persisted_seq` to the
//! freshly allocated seq under the lock, and never writes parquet.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::compute::{cast, lexsort_to_indices, take, SortColumn, SortOptions};
use arrow::datatypes::{DataType, SchemaRef};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use tokio::sync::{watch, Notify, RwLock};

use crate::errors::StatsError;
use crate::indices::exact::{ExactIndexConfig, NAMED_PROJECTION_MARKER};
use crate::indices::projection::{covering_projection_paths, covering_projection_staging_paths};
use crate::indices::{
    legacy_artifact_paths, local_sidecar_artifacts, needs_rebuild as segment_index_needs_rebuild,
    remove_if_exists, IndexBuildRequest, IndexRegistry, SegmentArtifacts, SegmentIndexConfig,
};
use crate::maintenance::scheduler::FlushDemand;
use crate::maintenance::{
    MaintenanceLimits, OBJECT_GC_INTERVAL, OBJECT_ORPHAN_GRACE, REWRITE_LAYOUT_BUDGET,
};
use crate::partition_policy::{
    segment_path, select_rows, PartitionedBatches, PhysicalPartitionPolicy, SegmentPartition,
};
use crate::policies::{physical_partition_policy_for, segment_indexes_enabled_for};
use crate::proto::finelog::stats::{
    partition_field, ColumnType, L0Mode, MigrationPhase, SourceLayout, TableMigrationStatus,
};
use crate::store::catalog::object_state_store::OBJECTS_PREFIX;
use crate::store::catalog::{Catalog, ObjectSegmentRecord, TableSpecStatus};
use crate::store::compaction::config::{CompactionConfig, CompactionJob};
use crate::store::compaction::executor::{
    read_segment_projected, run_job_with_partition_policy, CompactionExecution, CompactionLayout,
    OutputPolicy, PlannedSwap,
};
use crate::store::compaction::planner::{build_job, plan};
use crate::store::object_store::{ObjectId, ObjectPrefix, ObjectStore};
use crate::store::policy::StoragePolicy;
use crate::store::ram_buffer::{stamp_seq_and_build, RamBuffers, SealedBuffer};
use crate::store::schema::{
    resolve_key_column, resolve_sort_columns, schema_to_arrow, AlignedBatch, Schema,
};
#[cfg(test)]
use crate::store::segment::write_segment_to_dir;
use crate::store::segment::{
    discover_files, discover_segments, read_segment_footer, segment_id, segment_layout_is_current,
    stage_rewritten_segment, write_segment_to_dir_with_max_row_group_rows,
    write_segment_with_max_row_group_rows, MAX_ROW_GROUP_ROWS,
};
use crate::store::table::query_view::{plan_visible_segments, SegmentObjectMap};
use crate::store::table::{
    file_sha256, local_artifacts, object_segment_is_query_visible, MaintenanceLease,
    TableController, WrittenObject,
};
use crate::store::table_state::{
    ArtifactReferences, CommitError, LocalArtifacts, SegmentDescriptor, SourceBinding,
    TableRevision, TableSnapshot,
};
use crate::store::types::{
    basename, segment_relative_key, LocalSegment, NamespaceStats, SegmentLocation, SegmentRow,
};

/// Best-effort removal of a segment's derived index files, co-located with every
/// parquet unlink. Missing artifacts are not errors.
fn remove_index_artifacts(parquet_path: &str) {
    let parquet = Path::new(parquet_path);
    let mut artifacts = vec![
        (crate::indices::format::bundle_path(parquet), "index bundle"),
        (
            crate::indices::format::staging_path(parquet),
            "staged index bundle",
        ),
    ];
    artifacts.extend(
        legacy_artifact_paths(parquet)
            .into_iter()
            .map(|path| (path, "legacy index")),
    );
    match covering_projection_paths(parquet) {
        Ok(paths) => artifacts.extend(paths.into_iter().map(|path| (path, "covering projection"))),
        Err(error) => {
            tracing::warn!(path = %parquet.display(), %error, "failed to enumerate segment index artifacts");
        }
    }
    match covering_projection_staging_paths(parquet) {
        Ok(paths) => artifacts.extend(
            paths
                .into_iter()
                .map(|path| (path, "staged covering projection")),
        ),
        Err(error) => {
            tracing::warn!(path = %parquet.display(), %error, "failed to enumerate staged segment index artifacts");
        }
    }
    for (path, kind) in artifacts {
        if let Err(error) = remove_if_exists(&path) {
            tracing::warn!(path = %path.display(), %error, index_artifact = kind, "failed to remove segment index artifact");
        }
    }
}

/// The local files one segment's artifacts resolve to.
///
/// An object-backed segment resolves each path from the object identity its
/// table state names, so an empty reference set means the segment advertises no
/// artifacts. A version-0 segment has no references at all; its sidecars come
/// from the local layout it was written with, and stop being consulted once the
/// table is imported to object storage.
fn segment_artifacts(
    store: Option<&dyn ObjectStore>,
    record: Option<&ObjectSegmentRecord>,
    parquet: &Path,
) -> Result<LocalArtifacts, StatsError> {
    let Some(record) = record else {
        return Ok(local_sidecar_artifacts(parquet));
    };
    let store = store.ok_or_else(|| {
        StatsError::Internal(format!(
            "object segment {} has no object store to resolve artifacts",
            parquet.display()
        ))
    })?;
    local_artifacts(store, &record.artifacts)
}

/// One job promoting every L0 segment in `rows` to L1.
///
/// The leveled policy only promotes a run that meets its byte or fanout target.
/// A caller that needs L0 stabilized now — a forced maintenance cycle — asks for
/// this instead.
fn l0_promotion_job(rows: &[SegmentRow]) -> Option<CompactionJob> {
    let mut inputs: Vec<SegmentRow> = rows.iter().filter(|row| row.level == 0).cloned().collect();
    if inputs.is_empty() {
        return None;
    }
    inputs.sort_by_key(|row| row.min_seq);
    let output_min_seq = inputs
        .iter()
        .map(|row| row.min_seq)
        .min()
        .expect("a non-empty run has a minimum seq");
    Some(CompactionJob {
        inputs,
        output_level: 1,
        output_min_seq,
    })
}

/// Whether a leased commit lost a real conflict rather than hitting a transient
/// failure.
///
/// A conflict means the replacement can never apply: an input was retired, the
/// definition version moved, or another writer owns the table. Its outputs are
/// already immutable objects, so abandoning them leaves them unreferenced and
/// collectible instead of failing the table.
fn is_lease_conflict(error: &CommitError) -> bool {
    matches!(
        error,
        CommitError::Fenced(_) | CommitError::NotCommitted(StatsError::SchemaConflict(_))
    )
}

/// A private directory one compaction stages its outputs and artifacts in.
///
/// The executor writes ordinary local Parquet plus its derived artifacts here;
/// each is uploaded as an immutable object and the directory is removed. Nothing
/// in the query view ever points at a staged file.
struct CompactionStaging {
    path: PathBuf,
}

impl CompactionStaging {
    fn create(table_dir: &Path) -> Result<Self, StatsError> {
        let path = table_dir.join(format!("{COMPACTION_STAGING_DIR}/{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).map_err(|error| {
            StatsError::Internal(format!(
                "create compaction staging directory {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CompactionStaging {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.path) {
            tracing::warn!(path = %self.path.display(), %error, "failed to remove compaction staging directory");
        }
    }
}

/// Why one sealed buffer did not reach a published table revision.
enum SealedCommit {
    /// Nothing was committed. The sealed rows return to the RAM buffer.
    NotCommitted(StatsError),
    /// The rows are committed to durable local state but HEAD does not name
    /// that revision yet. Re-flushing them would duplicate committed data.
    PublicationUnresolved(StatsError),
}

fn fixed_index_artifacts_exist(parquet_path: &Path) -> bool {
    crate::indices::format::bundle_path(parquet_path).exists()
        || crate::indices::format::staging_path(parquet_path).exists()
        || legacy_artifact_paths(parquet_path)
            .into_iter()
            .any(|path| path.exists())
}

fn remove_orphaned_index_artifact(namespace: &str, path: &Path, kind: &str) {
    if let Err(error) = remove_if_exists(path) {
        tracing::warn!(
            namespace,
            path = %path.display(),
            index_artifact = kind,
            %error,
            "failed to remove orphaned segment index artifact"
        );
    }
}

/// Buffered-byte size at which an append forces an early flush, short-circuiting
/// the flush-rate cooldown so a write burst can't buffer unboundedly (and bounds
/// a single L0's size).
pub const SEGMENT_TARGET_BYTES: i64 = 100 * 1024 * 1024;

/// Maximum idle gap before the flush task wakes on its own. With steady writes
/// the per-append nudge drives flushes; this is the ceiling for a quiet namespace.
pub const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// Default durability-await budget when the RPC carries no deadline.
pub const DEFAULT_PERSIST_TIMEOUT: Duration = Duration::from_secs(30);

/// Segment index bundles rebuilt or removed per maintenance tick.
///
/// A single index build over a terminal-level segment is heavy (substantial CPU
/// and RAM), and the backfill is the lowest-priority maintenance work, so this
/// stays small enough never to starve compaction/sync/eviction. It is four rather
/// than one so a namespace whose bundles all need rebuilding converges in tens
/// of minutes instead of hours while queries safely use partial coverage.
pub const INDEX_BUNDLES_PER_TICK: usize = 4;

/// Per-maintenance budget for converging legacy physical placement.
///
/// Each rewrite already in flight may overrun this budget. The bound controls
/// how many additional jobs start before ordinary compaction and remote sync get
/// their turn.
const PHYSICAL_LAYOUT_MIGRATION_BUDGET: Duration = Duration::from_secs(3);
const PHYSICAL_LAYOUT_MIGRATION_CONCURRENCY: usize = 2;
const PHYSICAL_LAYOUT_MIGRATION_WORKER_COMPRESSED_BYTES: i64 = 32 * 1024 * 1024;

/// Source objects copied per maintenance tick while a TableSpec transition is active.
const TABLE_SPEC_MIGRATION_SEGMENTS_PER_TICK: usize = 4;
const NULL_PARTITION_VALUE: &str = "__HIVE_DEFAULT_PARTITION__";
/// Table-relative directory compaction stages its outputs in before upload.
const COMPACTION_STAGING_DIR: &str = "_compaction";

/// Process-wide permit for the layout rewrite, so only one namespace re-encodes
/// at a time.
///
/// Every namespace runs its own maintenance task, so a per-namespace budget
/// alone let a store with dozens of namespaces spend dozens of budgets at once.
/// On the marin hub that saturated the box: a 10 min telemetry query that
/// normally answers in 0.18 s took 15 s, and `count(*)` went from 0.3 s to 17 s,
/// because re-encoding pushes tens of GiB through the page cache the queries are
/// served from and invalidates the parquet metadata cache entry for every
/// segment it touches.
///
/// A namespace that cannot take the permit skips the step for this tick rather
/// than queueing behind it, which would stall the rest of its maintenance.
static REWRITE_SLOT: Mutex<()> = Mutex::new(());

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Insertion-lock-guarded mutable state.
struct NsInner {
    buffers: RamBuffers,
    local_segments: VecDeque<LocalSegment>,
    /// Per-namespace retention overrides; `None` fields inherit the
    /// cluster-wide `CompactionConfig` caps in `eviction_step`. Guarded by the
    /// insertion lock so a concurrent `RegisterTable` re-register (which calls
    /// `update_policy`) and a maintenance-tick read never tear.
    storage_policy: StoragePolicy,
}

#[derive(Debug, Clone)]
struct TableRuntimePolicy {
    l0_mode: L0Mode,
    table_spec_version: u64,
    max_buffer_bytes: i64,
    max_flush_age: Duration,
    max_query_time_ms: u64,
    rollback_window_ms: u64,
    target_object_bytes: i64,
    source_layout: Option<SourceLayout>,
}

impl Default for TableRuntimePolicy {
    fn default() -> Self {
        Self {
            l0_mode: L0Mode::L0_MODE_LEGACY_LOCAL,
            table_spec_version: 0,
            max_buffer_bytes: SEGMENT_TARGET_BYTES,
            max_flush_age: DEFAULT_FLUSH_INTERVAL,
            max_query_time_ms: crate::store::table_spec::DEFAULT_MAX_QUERY_TIME_MS,
            rollback_window_ms: crate::store::table_spec::DEFAULT_ROLLBACK_WINDOW_MS,
            target_object_bytes: crate::store::table_spec::DEFAULT_TARGET_OBJECT_BYTES as i64,
            source_layout: None,
        }
    }
}

impl TableRuntimePolicy {
    fn from_status(status: &TableSpecStatus) -> Self {
        let Some(spec) = status.desired.as_ref().or(status.active.as_ref()) else {
            return Self::default();
        };
        let Some(operating) = spec.operating_policy.as_option() else {
            return Self::default();
        };
        let l0_mode = operating
            .l0_mode
            .and_then(|mode| mode.as_known())
            .filter(|mode| *mode != L0Mode::L0_MODE_UNSPECIFIED)
            .unwrap_or(L0Mode::L0_MODE_LEGACY_LOCAL);
        Self {
            l0_mode,
            table_spec_version: spec.version.unwrap_or(0),
            max_buffer_bytes: i64::try_from(
                operating
                    .max_buffer_bytes
                    .unwrap_or(SEGMENT_TARGET_BYTES as u64),
            )
            .unwrap_or(i64::MAX),
            max_flush_age: Duration::from_millis(
                operating
                    .max_flush_age_ms
                    .unwrap_or(DEFAULT_FLUSH_INTERVAL.as_millis() as u64),
            ),
            max_query_time_ms: crate::store::table_spec::max_query_time_ms(spec),
            rollback_window_ms: crate::store::table_spec::rollback_window_ms(spec),
            target_object_bytes: spec
                .source_layout
                .as_option()
                .and_then(|layout| layout.target_object_bytes)
                .and_then(|bytes| i64::try_from(bytes).ok())
                .unwrap_or(crate::store::table_spec::DEFAULT_TARGET_OBJECT_BYTES as i64),
            source_layout: spec.source_layout.as_option().cloned(),
        }
    }

    fn object_backed(&self) -> bool {
        self.l0_mode == L0Mode::L0_MODE_OBJECT_STORE
    }
}

fn sorted_object_batch(
    batch: &RecordBatch,
    source_layout: Option<&SourceLayout>,
) -> Result<RecordBatch, StatsError> {
    let Some(layout) = source_layout else {
        return Ok(batch.clone());
    };
    let mut names = layout.sort_columns.clone();
    if !names.iter().any(|name| name == "seq") {
        names.push("seq".to_string());
    }
    if names.is_empty() {
        return Ok(batch.clone());
    }
    let columns = names
        .iter()
        .map(|name| {
            let column = batch.column_by_name(name).ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "object source layout sort column {name:?} is missing"
                ))
            })?;
            Ok(SortColumn {
                values: column.clone(),
                options: Some(SortOptions {
                    descending: false,
                    nulls_first: false,
                }),
            })
        })
        .collect::<Result<Vec<_>, StatsError>>()?;
    let indices = lexsort_to_indices(&columns, None)
        .map_err(|error| StatsError::Internal(format!("sort object-backed batch: {error}")))?;
    let arrays = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| StatsError::Internal(format!("apply object-backed sort: {error}")))?;
    RecordBatch::try_new(batch.schema(), arrays)
        .map_err(|error| StatsError::Internal(format!("build sorted object batch: {error}")))
}

fn partition_object_batch(
    batch: &RecordBatch,
    source_layout: Option<&SourceLayout>,
) -> Result<Vec<(Option<SegmentPartition>, RecordBatch)>, StatsError> {
    let Some(partition) = source_layout.and_then(|layout| layout.partition.as_option()) else {
        return Ok(vec![(None, batch.clone())]);
    };
    if partition.fields.is_empty() {
        return Ok(vec![(None, batch.clone())]);
    }
    let spec_id = u32::try_from(partition.spec_id.unwrap_or(0)).map_err(|_| {
        StatsError::SchemaValidation("partition spec_id exceeds the supported range".to_string())
    })?;
    let rendered_columns = partition
        .fields
        .iter()
        .map(|field| {
            let source = field.source_column.as_deref().unwrap_or("");
            let column = batch.column_by_name(source).ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "partition source column {source:?} is missing"
                ))
            })?;
            let rendered = cast(column, &DataType::Utf8).map_err(|error| {
                StatsError::SchemaValidation(format!(
                    "partition source column {source:?} cannot be rendered: {error}"
                ))
            })?;
            Ok(rendered)
        })
        .collect::<Result<Vec<_>, StatsError>>()?;
    let mut indices: BTreeMap<SegmentPartition, Vec<u32>> = BTreeMap::new();
    for row in 0..batch.num_rows() {
        let mut values = BTreeMap::new();
        for (field, rendered) in partition.fields.iter().zip(&rendered_columns) {
            let values_array = rendered
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Arrow UTF-8 cast returns StringArray");
            let value = match field.transform.as_ref() {
                Some(partition_field::Transform::Identity(_)) if values_array.is_null(row) => {
                    NULL_PARTITION_VALUE.to_string()
                }
                Some(partition_field::Transform::Identity(_)) => {
                    values_array.value(row).to_string()
                }
                Some(partition_field::Transform::Bucket(_)) if values_array.is_null(row) => {
                    NULL_PARTITION_VALUE.to_string()
                }
                Some(partition_field::Transform::Bucket(bucket)) => {
                    let buckets = bucket.buckets.unwrap_or(0);
                    if buckets == 0 {
                        return Err(StatsError::SchemaValidation(format!(
                            "partition field {:?} bucket count must be positive",
                            field.name.as_deref().unwrap_or("")
                        )));
                    }
                    let digest = Sha256::digest(values_array.value(row).as_bytes());
                    let hash = u32::from_be_bytes(
                        digest[..4]
                            .try_into()
                            .expect("SHA-256 prefix is four bytes"),
                    );
                    (hash % buckets).to_string()
                }
                None => {
                    return Err(StatsError::SchemaValidation(format!(
                        "partition field {:?} has no transform",
                        field.name.as_deref().unwrap_or("")
                    )))
                }
            };
            values.insert(field.name.as_deref().unwrap_or("").to_string(), value);
        }
        indices
            .entry(SegmentPartition { spec_id, values })
            .or_default()
            .push(row as u32);
    }
    indices
        .into_iter()
        .map(|(partition, indices)| {
            select_rows(batch, indices).map(|batch| (Some(partition), batch))
        })
        .collect()
}

fn batch_seq_bounds(batch: &RecordBatch) -> Result<(i64, i64), StatsError> {
    let seq = batch
        .column_by_name("seq")
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| StatsError::Internal("object-backed batch has no Int64 seq".to_string()))?;
    let min = (0..seq.len())
        .filter(|index| !seq.is_null(*index))
        .map(|index| seq.value(index))
        .min()
        .ok_or_else(|| {
            StatsError::Internal("object-backed batch has no sequence values".to_string())
        })?;
    let max = (0..seq.len())
        .filter(|index| !seq.is_null(*index))
        .map(|index| seq.value(index))
        .max()
        .expect("non-empty sequence iterator");
    Ok((min, max))
}

/// Directory fan-out for a migration rewrite's staged partition outputs. The
/// uploaded objects are content-addressed, so this only keeps one source's
/// staged files from colliding.
const MIGRATION_PARTITION_DIRECTORIES: u32 = 64;

/// Stable identity of one migration source, so a rewrite that is interrupted
/// after its objects upload but before its checkpoint commits is recognized and
/// not applied twice.
///
/// It binds the source's content to the exact rows it covers: an object-backed
/// source is already named by its content SHA-256, and a version-0 file is
/// hashed where it lies.
async fn migration_source_id(
    row: &SegmentRow,
    object_record: Option<&ObjectSegmentRecord>,
    localized: &Path,
) -> Result<String, StatsError> {
    let mut digest = Sha256::new();
    match object_record {
        Some(record) => {
            digest.update(
                record
                    .source
                    .object_id
                    .as_deref()
                    .unwrap_or(&row.path)
                    .as_bytes(),
            );
            digest.update(record.source.sha256.as_deref().unwrap_or_default());
        }
        None => {
            digest.update(row.path.as_bytes());
            let path = localized.to_path_buf();
            let content: [u8; 32] = tokio::task::spawn_blocking(move || file_sha256(&path))
                .await
                .map_err(|error| {
                    StatsError::Internal(format!("migration source hash task panicked: {error}"))
                })??;
            digest.update(content);
        }
    }
    digest.update(row.min_seq.to_be_bytes());
    digest.update(row.max_seq.to_be_bytes());
    digest.update(row.row_count.to_be_bytes());
    Ok(crate::hex::encode(&digest.finalize()))
}

/// The physical partitioning one table specification's source layout declares.
///
/// The compaction executor asks for partitions through this trait, so a
/// migration into a changed partition spec splits its rewritten rows exactly the
/// way the ingest path splits a freshly sealed buffer.
#[derive(Debug)]
struct SourceLayoutPartitions {
    spec_id: u32,
    layout: SourceLayout,
}

impl SourceLayoutPartitions {
    /// `None` when `layout` declares no partitioning, in which case a rewrite
    /// produces one unpartitioned output per source.
    fn new(layout: &SourceLayout) -> Option<Self> {
        let partition = layout.partition.as_option()?;
        if partition.fields.is_empty() {
            return None;
        }
        let spec_id = u32::try_from(partition.spec_id.unwrap_or(0)).ok()?;
        Some(Self {
            spec_id,
            layout: layout.clone(),
        })
    }
}

impl PhysicalPartitionPolicy for SourceLayoutPartitions {
    fn is_current_partition(&self, partition: &SegmentPartition) -> bool {
        partition.spec_id == self.spec_id
    }

    fn partition_batches(
        &self,
        batches: &[RecordBatch],
    ) -> Result<Vec<PartitionedBatches>, StatsError> {
        let mut outputs: BTreeMap<SegmentPartition, Vec<RecordBatch>> = BTreeMap::new();
        for batch in batches {
            for (partition, split) in partition_object_batch(batch, Some(&self.layout))? {
                let partition = partition.ok_or_else(|| {
                    StatsError::Internal(
                        "partitioned source layout produced an unpartitioned split".to_string(),
                    )
                })?;
                outputs.entry(partition).or_default().push(split);
            }
        }
        Ok(outputs
            .into_iter()
            .map(|(partition, batches)| PartitionedBatches { partition, batches })
            .collect())
    }

    fn partitions_for_exact_values(
        &self,
        _exact_values: &HashMap<String, Vec<String>>,
    ) -> Option<BTreeSet<SegmentPartition>> {
        // Query pruning uses the namespace's own registered policy. A migration
        // rewrite never prunes, so this policy claims nothing.
        None
    }

    fn segment_directory(&self, partition: &SegmentPartition) -> PathBuf {
        let mut digest = Sha256::new();
        for (field, value) in &partition.values {
            digest.update(field.as_bytes());
            digest.update([0]);
            digest.update(value.as_bytes());
            digest.update([0]);
        }
        let bucket = u32::from_be_bytes(
            digest.finalize()[..4]
                .try_into()
                .expect("sha256 prefix is four bytes"),
        ) % MIGRATION_PARTITION_DIRECTORIES;
        Path::new("partition").join(format!("{bucket:02}"))
    }
}

/// A single namespace's write engine, disk-backed or in-memory.
pub struct Namespace {
    name: String,
    schema: Schema,
    arrow_schema: SchemaRef,
    key_column: String,
    sort_columns: Vec<String>,
    max_row_group_rows: usize,
    /// `None` => in-memory mode (every append immediately persisted, no parquet).
    data_dir: Option<PathBuf>,
    catalog: Arc<Catalog>,
    /// Leveled-compaction tuning. The maintenance task reads `check_interval`,
    /// the planner reads `level_targets`/`max_segments_per_level`.
    compaction_config: CompactionConfig,
    inner: Mutex<NsInner>,
    table_runtime: Mutex<TableRuntimePolicy>,
    /// Serializes the whole `flush_once` body (seal → write → catalog → commit →
    /// publish). Without it two concurrent flushers race: the second `seal()`
    /// would overwrite the first's
    /// in-flight `flushing` buffer, and `send_replace` could publish a newer
    /// high-water seq before the older segment is durable. Distinct from `inner`
    /// (the short insertion lock) so appends are never blocked by a flush write.
    flush_lock: Mutex<()>,
    object_flush_lock: tokio::sync::Mutex<()>,
    /// Serializes the maintenance cycle (compaction drain + sync + evict)
    /// against direct `maintain` callers. The flush path uses `flush_lock`
    /// instead so flushes and compactions stay concurrent. A
    /// `tokio::sync::Mutex` because the maintenance body awaits (sync_step is
    /// async object_store I/O).
    maint_lock: tokio::sync::Mutex<()>,
    /// Process-wide query-visibility lock (one shared instance for the whole
    /// store). `commit_swap` / `evict_segment` take the WRITE side via
    /// `blocking_write()` inside a `spawn_blocking` so a query that snapshotted
    /// pre-swap paths drains before any rename/unlink. Query/FetchLogs handlers
    /// hold the READ side across `collect()`.
    query_visibility: Arc<RwLock<()>>,
    /// Durable-state controller for this table: the only owner of its
    /// publication, writer claim, and canonical object writes.
    controller: Arc<TableController>,
    last_object_gc: Mutex<Option<Instant>>,
    persisted_seq: watch::Sender<i64>,
    /// Latched by every append (and a durability await): "there may be data to
    /// flush". The scheduler reads and clears it.
    flush_requested: AtomicBool,
    /// Latched only when a buffer crosses the definition's maximum buffer size:
    /// "flush now, don't wait out the coalescing window". Lets a write burst
    /// bypass `MIN_FLUSH_INTERVAL` so RAM and L0 size stay bounded, while
    /// ordinary per-append demand keeps coalescing.
    flush_forced: AtomicBool,
    /// The scheduler's wake signal, shared by every table in the store. Latching
    /// flush demand signals it so flush latency does not wait out the poll
    /// interval.
    maintenance_wake: Arc<Notify>,
    stop: Arc<Notify>,
    /// Latched stop flag the background tasks check at the TOP of each loop
    /// iteration, in addition to selecting on the `stop` Notify. `Notify`
    /// stores no permit for `notify_waiters`, so a task that is mid-flush
    /// (off in `spawn_blocking`) when `stop` fires would otherwise re-subscribe
    /// after the wake and park forever, hanging the join. The latch closes that
    /// race: once set, the next loop iteration sees it and returns even if it
    /// missed the Notify wake. Set by `stop_and_join` / `request_stop`.
    stopped: AtomicBool,
    /// JoinHandles for the maintenance work the scheduler dispatched against
    /// this namespace. Retained so a re-register replacement, a drop, or
    /// `Store::shutdown` can cooperatively cancel (via the `stop` Notify) and
    /// JOIN that work within a bounded timeout instead of busy-waiting. Pushed
    /// to by [`spawn_tracked`](Namespace::spawn_tracked); drained by
    /// [`stop_and_join`](Namespace::stop_and_join).
    task_handles: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    /// Segments the index backfill has already rebuilt without reaching
    /// coverage. See [`IndexBackfillSkips`].
    index_backfill_skips: Mutex<IndexBackfillSkips>,
    /// Segments already confirmed to carry the current physical layout.
    ///
    /// Determining staleness means parsing a segment's whole footer, so without
    /// this the rewrite pass would re-read every footer in the namespace on
    /// every tick — hundreds of MiB of thrift for a large namespace, forever,
    /// long after there is nothing left to rewrite. A path's layout only ever
    /// changes because this pass changed it.
    current_layouts: Mutex<HashSet<String>>,
    indices: Arc<IndexRegistry>,
    /// Process-wide maintenance concurrency limits, shared by every namespace in
    /// this store.
    limits: Arc<MaintenanceLimits>,
}

/// Segments the index backfill cannot bring up to date, and the indexed set
/// that verdict was reached under.
///
/// A trigram index covers only the columns a segment actually has: one written
/// before a column existed indexes nothing for it, and its bundle can never
/// satisfy the rebuild condition. Without this the backfill would re-read and
/// re-serialize that segment on every maintenance tick forever, at one segment
/// per tick starving every segment that can still make progress. Enabling
/// another index resets the verdict, since the new column may well be present.
#[derive(Default)]
struct IndexBackfillSkips {
    indexed: Vec<String>,
    paths: HashSet<String>,
}

impl IndexBackfillSkips {
    /// Drop the recorded verdicts when the indexed set changes, and forget
    /// segments that are no longer local (compacted away or evicted).
    fn reconcile(&mut self, indexed: &[&str], live: &HashSet<&str>) {
        if self.indexed.len() != indexed.len()
            || !self.indexed.iter().zip(indexed).all(|(a, b)| a == b)
        {
            self.indexed = indexed.iter().map(|c| c.to_string()).collect();
            self.paths.clear();
            return;
        }
        self.paths.retain(|p| live.contains(p.as_str()));
    }
}

struct BackfillCandidate {
    path: String,
    expected_rows: i64,
}

#[derive(Clone, Copy, Default)]
struct PhysicalLayoutMigrationPending {
    migration_l0: usize,
    stale_partitions: usize,
    misplaced_local: usize,
}

impl PhysicalLayoutMigrationPending {
    fn any(self) -> bool {
        self.migration_l0 > 0 || self.stale_partitions > 0 || self.misplaced_local > 0
    }
}

fn migration_l0_needs_rebuild(segment: &LocalSegment) -> bool {
    segment.level == 0 && (segment.min_seq < 0 || segment.partition.is_some())
}

fn partition_is_stale(
    segment: &LocalSegment,
    policy: &dyn crate::partition_policy::PhysicalPartitionPolicy,
) -> bool {
    segment.level >= 1
        && segment
            .partition
            .as_ref()
            .is_none_or(|partition| !policy.is_current_partition(partition))
}

fn current_layout_destination(
    dir: &Path,
    path: &str,
    level: i32,
    partition: Option<&SegmentPartition>,
    policy: &dyn crate::partition_policy::PhysicalPartitionPolicy,
) -> Option<PathBuf> {
    let partition = partition?;
    if level < 1 || !policy.is_current_partition(partition) {
        return None;
    }
    let filename = Path::new(path).file_name()?.to_str()?;
    let destination = segment_path(dir, filename, level, Some(partition), Some(policy));
    (Path::new(path) != destination).then_some(destination)
}

/// A namespace's readable segments as one consistent observation: the files a
/// scan may open, their known key bounds and partitions, and the lowest `seq`
/// any of them holds.
///
/// An object-backed table plans this from its pinned `TableSnapshot`, so
/// `sources` names the immutable object behind every path and the paths exist
/// only after the scan localizes the ones it selected. A legacy table plans it
/// from files already on disk and carries no `sources`.
pub struct SegmentSnapshot {
    pub paths: Vec<String>,
    pub key_bounds: BTreeMap<String, (i64, i64)>,
    pub partitions: BTreeMap<String, SegmentPartition>,
    pub min_seq: Option<i64>,
    /// What each snapshotted segment advertises, so a scan opens artifacts by
    /// reference instead of probing for files beside the Parquet.
    pub artifacts: SegmentArtifacts,
    /// The immutable objects each path resolves to, for the segments the scan
    /// selects. Empty for a legacy table.
    pub sources: SegmentObjectMap,
}

impl Namespace {
    /// Build a namespace over `data_dir` (disk-backed when `Some`).
    ///
    /// On a disk namespace the next seq is recovered from segment footers and any
    /// existing local segment files are adopted into the deque (sorted by
    /// min_seq); `persisted_seq` starts at the recovered high-water seq so a
    /// caller awaiting a previously-durable seq returns immediately.
    ///
    /// `query_visibility` is the one process-wide lock (cloned into each
    /// namespace) the maintenance task takes the WRITE side of before any
    /// rename/unlink. Storage implementations are built by `Store` and injected
    /// here. `storage_policy` is the per-namespace retention override. The caller
    /// starts the per-namespace maintenance task once the store is fully built.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        name: &str,
        schema: Schema,
        data_dir: Option<PathBuf>,
        catalog: Arc<Catalog>,
        query_visibility: Arc<RwLock<()>>,
        indices: Arc<IndexRegistry>,
        limits: Arc<MaintenanceLimits>,
        maintenance_wake: Arc<Notify>,
        controller: Arc<TableController>,
        storage_policy: StoragePolicy,
    ) -> Result<Arc<Namespace>, StatsError> {
        let startup_started = Instant::now();
        let arrow_schema = schema_to_arrow(&schema);
        let sort_columns = resolve_sort_columns(&schema)?;
        let max_row_group_rows = if schema.max_row_group_rows == 0 {
            MAX_ROW_GROUP_ROWS
        } else {
            schema.max_row_group_rows as usize
        };
        let key_column = resolve_key_column(&schema)?;

        let local_recovery_started = Instant::now();
        let (next_seq, adopted, init_persisted) = match (&data_dir, controller.is_object_backed()) {
            (None, _) => (1_i64, VecDeque::new(), -1_i64),
            // An object-backed table's contents come from its durable state,
            // never from files present in the local object cache. Recovery
            // seeds sequence allocation from the catalog projection of that
            // state and adopts nothing from disk; a query localizes the objects
            // it selects.
            (Some(dir), true) => {
                std::fs::create_dir_all(dir).map_err(|e| {
                    StatsError::Internal(format!("create namespace dir {}: {e}", dir.display()))
                })?;
                let rows = catalog.list_segments(name)?;
                let max_persisted = rows
                    .iter()
                    .filter(|row| row.row_count > 0)
                    .map(|row| row.max_seq)
                    .max()
                    .unwrap_or(-1);
                (
                    crate::store::adopt::recover_next_seq(&rows),
                    VecDeque::new(),
                    max_persisted,
                )
            }
            (Some(dir), false) => {
                std::fs::create_dir_all(dir).map_err(|e| {
                    StatsError::Internal(format!("create namespace dir {}: {e}", dir.display()))
                })?;
                let adopted = adopt_local_segments(
                    dir,
                    Some(&key_column),
                    &catalog,
                    name,
                    controller.object_store().map(Arc::as_ref),
                )?;
                // Seed next_seq past every segment the catalog knows about, not
                // just on-disk footers. A segment evicted to remote has its local
                // parquet unlinked, so a footer-only scan under-counts and would
                // reuse live seqs (silent overwrite). Union the footer scan with
                // the full catalog (LOCAL, REMOTE, and BOTH rows).
                // `adopt_local_segments` has already read every healthy local
                // footer. Reuse those max_seq values instead of scanning every
                // Parquet footer a second time.
                let local_next_seq = adopted
                    .iter()
                    .map(|segment| segment.max_seq + 1)
                    .max()
                    .unwrap_or(1)
                    .max(1);
                let next_seq = local_next_seq.max(crate::store::adopt::recover_next_seq(
                    &catalog.list_segments(name)?,
                ));
                let max_persisted = adopted
                    .iter()
                    .filter(|s| s.row_count > 0)
                    .map(|s| s.max_seq)
                    .max()
                    .unwrap_or(-1);
                (next_seq, adopted, max_persisted)
            }
        };
        let local_recovery_ms = local_recovery_started.elapsed().as_millis() as u64;

        let table_runtime = TableRuntimePolicy::from_status(&catalog.table_spec_status(name)?);

        let (tx, _rx) = watch::channel(init_persisted);
        let ns = Arc::new(Namespace {
            name: name.to_string(),
            schema,
            arrow_schema: Arc::clone(&arrow_schema),
            key_column,
            sort_columns,
            max_row_group_rows,
            data_dir,
            catalog: Arc::clone(&catalog),
            compaction_config: CompactionConfig::default(),
            inner: Mutex::new(NsInner {
                buffers: RamBuffers::new(arrow_schema, next_seq),
                local_segments: adopted.clone(),
                storage_policy,
            }),
            table_runtime: Mutex::new(table_runtime),
            flush_lock: Mutex::new(()),
            object_flush_lock: tokio::sync::Mutex::new(()),
            maint_lock: tokio::sync::Mutex::new(()),
            query_visibility,
            controller,
            last_object_gc: Mutex::new(None),
            persisted_seq: tx,
            flush_requested: AtomicBool::new(false),
            flush_forced: AtomicBool::new(false),
            maintenance_wake,
            stop: Arc::new(Notify::new()),
            stopped: AtomicBool::new(false),
            task_handles: Mutex::new(Vec::new()),
            index_backfill_skips: Mutex::new(IndexBackfillSkips::default()),
            current_layouts: Mutex::new(HashSet::new()),
            indices,
            limits,
        });

        // An object-backed table publishes its locally durable state so readers
        // have something to pin, then rebuilds its in-memory segment view from
        // that state's catalog rows. Metadata only: nothing is downloaded and
        // the local cache is not consulted.
        if ns.controller.is_object_backed() {
            ns.controller.seed_local_snapshot();
            let active = catalog.table_spec_status(name)?.active_version();
            ns.activate_query_version(active)?;
        }

        // Refresh the catalog from the adopted deque so the segments table
        // reflects on-disk reality after a fresh boot from a wiped catalog.
        let catalog_refresh_started = Instant::now();
        let adopted_rows: Vec<SegmentRow> = adopted
            .iter()
            .map(|segment| segment_to_row(name, segment))
            .collect();
        catalog.upsert_segments(&adopted_rows)?;
        let catalog_refresh_ms = catalog_refresh_started.elapsed().as_millis() as u64;

        tracing::info!(
            namespace = name,
            segments = adopted.len(),
            next_seq,
            local_recovery_ms,
            catalog_refresh_ms,
            total_ms = startup_started.elapsed().as_millis() as u64,
            "finelog namespace startup complete"
        );
        Ok(ns)
    }

    /// Whether this namespace has a remote offload target configured.
    pub fn has_remote(&self) -> bool {
        self.controller.is_object_backed()
    }

    /// Whether this process still owns the table's durable state. A fenced
    /// object-backed table rejects writes until a restart re-claims it.
    pub(crate) fn write_ready(&self) -> bool {
        self.controller.writes_ready()
    }

    /// Swap in a new retention policy (re-register). Picked up next eviction
    /// tick.
    pub fn update_policy(&self, policy: StoragePolicy) {
        self.inner.lock().unwrap().storage_policy = policy;
    }

    pub fn update_table_spec(&self, status: &TableSpecStatus) {
        *self.table_runtime.lock().unwrap() = TableRuntimePolicy::from_status(status);
    }

    /// The durable-state commit owner for this table.
    pub(crate) fn controller(&self) -> &Arc<TableController> {
        &self.controller
    }

    async fn publish_owed_object_catalog(&self) -> Result<(), StatsError> {
        self.controller.publish_owed().await
    }

    fn runtime_policy(&self) -> TableRuntimePolicy {
        self.table_runtime.lock().unwrap().clone()
    }

    async fn publish_object_catalog_state(&self) -> Result<(), StatsError> {
        self.controller.publish_state().await?;
        Ok(())
    }

    pub(crate) fn activate_query_version(&self, version: u64) -> Result<(), StatsError> {
        let status = self.catalog.table_spec_status(&self.name)?;
        let rollback_alias = status.migration.as_ref().and_then(|migration| {
            (status.active_version() == version && migration.from_version == Some(version))
                .then_some(migration.to_version.unwrap_or(0))
        });
        let object_records: HashMap<_, _> = self
            .catalog
            .object_segments(&self.name)?
            .into_iter()
            .map(|record| (record.path.clone(), record))
            .collect();
        let mut segments = VecDeque::new();
        for row in self.catalog.list_segments(&self.name)? {
            let visible = match object_records.get(&row.path) {
                Some(record) => {
                    record.table_spec_version == version
                        || (rollback_alias == Some(record.table_spec_version)
                            && !record.migration_backfill)
                }
                None => version == 0,
            };
            let record = object_records.get(&row.path);
            // A legacy segment is visible because its file is there; an
            // object-backed segment is visible because the table's state
            // references it. Cache contents never create or remove visibility.
            if !visible || (record.is_none() && !Path::new(&row.path).exists()) {
                continue;
            }
            let artifacts = segment_artifacts(
                self.controller.object_store().map(Arc::as_ref),
                record,
                Path::new(&row.path),
            )?;
            segments.push_back(LocalSegment {
                path: row.path,
                size_bytes: row.byte_size,
                level: row.level,
                min_seq: row.min_seq,
                max_seq: row.max_seq,
                row_count: row.row_count,
                created_at_ms: row.created_at_ms,
                min_key_value: row.min_key_value.and_then(|value| value.parse().ok()),
                max_key_value: row.max_key_value.and_then(|value| value.parse().ok()),
                partition: row.partition,
                location: row.location,
                artifacts,
            });
        }
        segments
            .make_contiguous()
            .sort_by_key(|segment| segment.min_seq);
        debug_assert_unique_paths(&segments);
        self.inner.lock().unwrap().local_segments = segments;
        Ok(())
    }

    async fn activate_verified_table_spec(&self) -> Result<TableSpecStatus, StatsError> {
        let _visibility_guard = self.query_visibility.write().await;
        // The in-memory query view swaps only after the activation revision is
        // known to be published, so queries never see a version whose state no
        // reader can recover.
        let status = self
            .controller
            .commit(|| {
                let status = self.catalog.activate_desired_table_spec(&self.name)?;
                Ok((TableRevision::new(status.catalog_generation), status))
            })
            .await?
            .output;
        self.activate_query_version(status.active_version())?;
        self.update_table_spec(&status);
        tracing::info!(
            namespace = %self.name,
            table_spec_version = status.active_version(),
            catalog_generation = status.catalog_generation,
            "activated migrated table specification"
        );
        Ok(status)
    }

    /// Advance an automatic TableSpec transition. Returns true while ordinary
    /// compaction and eviction must stay frozen to preserve the migration source.
    async fn advance_table_spec_migration(&self) -> Result<bool, StatsError> {
        let status = self.catalog.table_spec_status(&self.name)?;
        let Some(migration) = status.migration.clone() else {
            return Ok(false);
        };
        match status.phase {
            MigrationPhase::MIGRATION_PHASE_RETIRED => {
                self.publish_owed_object_catalog().await?;
                return Ok(false);
            }
            MigrationPhase::MIGRATION_PHASE_UNSPECIFIED => return Ok(false),
            MigrationPhase::MIGRATION_PHASE_OBSERVING => {
                // Heal a process failure after the local activation commit but
                // before HEAD publication or the in-memory view swap.
                self.publish_object_catalog_state().await?;
                self.activate_query_version(status.active_version())?;
                self.update_table_spec(&status);
                self.controller
                    .commit(|| {
                        let retired = self.catalog.retire_observed_migration(&self.name)?;
                        Ok((TableRevision::new(retired.catalog_generation), retired))
                    })
                    .await?;
                // The old version remains remotely referenced for rollback.
                // Compaction keeps backfill and post-cutover writes in separate
                // provenance groups, while cache eviction can rehydrate either
                // version from its canonical object.
                return Ok(false);
            }
            MigrationPhase::MIGRATION_PHASE_VERIFY => {
                self.activate_verified_table_spec().await?;
                return Ok(true);
            }
            MigrationPhase::MIGRATION_PHASE_ACTIVATED => {
                return Err(StatsError::Internal(format!(
                    "namespace {:?} persisted unsupported transient ACTIVATED phase",
                    self.name
                )));
            }
            MigrationPhase::MIGRATION_PHASE_DUAL_WRITE
            | MigrationPhase::MIGRATION_PHASE_BACKFILL => {}
        }

        self.backfill_table_spec_migration(&status, &migration)
            .await
    }

    /// Rewrite the source version's remaining segments into the target layout.
    ///
    /// Every source below the durable fence is rewritten exactly once, keyed by
    /// a deterministic `migration_source_id` so an interrupted tick resumes
    /// without duplicating rows. Rows written after the fence are already in the
    /// target layout and are referenced from both query views until activation.
    ///
    /// Returns true while the migration still owns the table's maintenance
    /// cycle.
    async fn backfill_table_spec_migration(
        &self,
        status: &TableSpecStatus,
        migration: &TableMigrationStatus,
    ) -> Result<bool, StatsError> {
        let _migration_guard = self.object_flush_lock.lock().await;
        let table_dir = self.data_dir.clone().ok_or_else(|| {
            StatsError::Internal(format!(
                "migration for {:?} requires a disk-backed cache",
                self.name
            ))
        })?;
        let from_version = migration.from_version.unwrap_or(0);
        let to_version = migration.to_version.unwrap_or(0);
        let fence_seq = migration.fence_seq.unwrap_or(-1);
        let object_records: HashMap<_, _> = self
            .catalog
            .object_segments(&self.name)?
            .into_iter()
            .map(|record| (record.path.clone(), record))
            .collect();
        let rows = self.catalog.list_segments(&self.name)?;
        let mut covered: HashSet<_> = object_records
            .values()
            .filter(|record| record.table_spec_version == to_version && record.migration_backfill)
            .filter_map(|record| record.migration_source_id.clone())
            .collect();
        let mut pending: Vec<_> = rows
            .iter()
            .filter(|row| row.max_seq <= fence_seq)
            .filter(|row| match object_records.get(&row.path) {
                None => from_version == 0,
                Some(record) => record.table_spec_version == from_version,
            })
            .cloned()
            .collect();
        pending.sort_by_key(|row| row.min_seq);
        let target_layout = status
            .desired
            .as_ref()
            .and_then(|spec| spec.source_layout.as_option())
            .cloned();
        let mut processed = 0;
        for row in &pending {
            if processed >= TABLE_SPEC_MIGRATION_SEGMENTS_PER_TICK {
                break;
            }
            let record = object_records.get(&row.path);
            let localized = self.controller.localize_source(row, record).await?;
            let migration_source_id = migration_source_id(row, record, &localized).await?;
            if covered.contains(&migration_source_id) {
                continue;
            }
            let staging = CompactionStaging::create(&table_dir)?;
            let outcome = self
                .rewrite_migration_source(
                    &staging,
                    row,
                    &localized,
                    target_layout.as_ref(),
                    to_version,
                    &migration_source_id,
                )
                .await;
            drop(staging);
            outcome?;
            covered.insert(migration_source_id);
            processed += 1;
        }

        let status = self.catalog.table_spec_status(&self.name)?;
        let progress = status.migration.as_ref().ok_or_else(|| {
            StatsError::Internal(format!("migration for {:?} disappeared", self.name))
        })?;
        if progress.rows_completed != progress.rows_total {
            return Ok(true);
        }
        let verified = self
            .controller
            .commit(|| {
                let verified = self.catalog.update_migration_phase(
                    &self.name,
                    MigrationPhase::MIGRATION_PHASE_BACKFILL,
                    MigrationPhase::MIGRATION_PHASE_VERIFY,
                )?;
                Ok((TableRevision::new(verified.catalog_generation), verified))
            })
            .await?
            .output;
        debug_assert_eq!(verified.phase, MigrationPhase::MIGRATION_PHASE_VERIFY);
        self.activate_verified_table_spec().await?;
        Ok(true)
    }

    /// Rewrite the segment `row`, localized at `localized`, into
    /// `target_layout` at table spec version `to_version`, and checkpoint the
    /// result against `migration_source_id`.
    ///
    /// On success the rewritten objects are committed and the source is
    /// recorded as migrated; the migration will not revisit it. An error leaves
    /// the source unmigrated and any uploaded outputs unreferenced.
    async fn rewrite_migration_source(
        &self,
        staging: &CompactionStaging,
        row: &SegmentRow,
        localized: &Path,
        target_layout: Option<&SourceLayout>,
        to_version: u64,
        migration_source_id: &str,
    ) -> Result<(), StatsError> {
        let job = CompactionJob {
            inputs: vec![SegmentRow {
                path: localized.to_string_lossy().into_owned(),
                ..row.clone()
            }],
            output_level: row.level,
            output_min_seq: row.min_seq,
        };
        let index_config = self.segment_index_config();
        let arrow_schema = Arc::clone(&self.arrow_schema);
        let sort_columns = target_layout
            .map(|layout| layout.sort_columns.clone())
            .filter(|columns| !columns.is_empty())
            .unwrap_or_else(|| self.sort_columns.clone());
        let key_column = self.key_column.clone();
        let max_row_group_rows = target_layout
            .and_then(|layout| layout.max_row_group_rows)
            .map(|rows| rows as usize)
            .unwrap_or(self.max_row_group_rows);
        let max_merge_arrow_bytes = self.compaction_config.max_merge_arrow_bytes;
        let partitions = target_layout.and_then(SourceLayoutPartitions::new);
        let key_bounds = self.input_key_bounds(&row.path);
        let staging_dir = staging.path().to_path_buf();
        let swap = tokio::task::spawn_blocking(move || {
            run_job_with_partition_policy(
                &job,
                &staging_dir,
                &arrow_schema,
                CompactionExecution {
                    layout: CompactionLayout {
                        sort_columns: &sort_columns,
                        key_column: &key_column,
                        max_row_group_rows,
                    },
                    index_config: &index_config,
                    partition_policy: partitions
                        .as_ref()
                        .map(|policy| policy as &dyn PhysicalPartitionPolicy),
                    max_merge_arrow_bytes,
                    output: OutputPolicy::AlwaysRewrite,
                },
                |_| key_bounds,
            )
        })
        .await
        .map_err(|error| {
            StatsError::Internal(format!("migration rewrite task panicked: {error}"))
        })??;

        let rewritten_rows: i64 = swap.added.iter().map(|segment| segment.row_count).sum();
        if rewritten_rows != row.row_count {
            return Err(StatsError::Internal(format!(
                "migration source {} rewrote {} rows as {rewritten_rows}",
                row.path, row.row_count
            )));
        }
        let mut migrated = Vec::with_capacity(swap.added.len());
        for staged in swap.added {
            let staged_path = PathBuf::from(&staged.path);
            let stored = self
                .controller
                .write_staged_object(OBJECTS_PREFIX, "parquet", &staged_path)
                .await?;
            let (references, local) = self
                .publish_segment_artifacts(&staged_path, &stored)
                .await?;
            let segment = LocalSegment {
                path: stored.path.to_string_lossy().into_owned(),
                size_bytes: stored.byte_size,
                location: SegmentLocation::Both,
                created_at_ms: row.created_at_ms,
                artifacts: local,
                ..staged
            };
            migrated.push(SegmentDescriptor {
                row: segment_to_row(&self.name, &segment),
                source: stored.source,
                artifacts: references,
            });
        }
        let source_rows = row.row_count;
        self.controller
            .commit(|| {
                let revision = self.catalog.commit_migration_segments(
                    &migrated,
                    to_version,
                    migration_source_id,
                    source_rows,
                )?;
                Ok((revision, ()))
            })
            .await?;
        Ok(())
    }

    /// Compact one planner-issued run of immutable objects and commit the
    /// replacement under a maintenance lease.
    ///
    /// `force_compact_l0` makes an L0 run eligible regardless of the size
    /// threshold the planner would otherwise apply.
    ///
    /// Returns `true` when a run was replaced and `false` when nothing is
    /// eligible or the commit lost a conflict; a lost commit leaves the
    /// uploaded outputs unreferenced for object GC to collect after the orphan
    /// grace, and the table keeps running.
    async fn object_compaction_step(&self, force_compact_l0: bool) -> Result<bool, StatsError> {
        let status = self.catalog.table_spec_status(&self.name)?;
        let active_version = status.active_version();
        if active_version == 0 {
            return Ok(false);
        }
        let Some(table_dir) = self.data_dir.clone() else {
            return Err(StatsError::Internal(
                "object-backed compaction requires local table state".to_string(),
            ));
        };
        let object_records: HashMap<_, _> = self
            .catalog
            .object_segments(&self.name)?
            .into_iter()
            .filter(|record| record.table_spec_version == active_version)
            .map(|record| (record.path.clone(), record))
            .collect();
        // A backfilled run and an ordinary run are separate compaction streams:
        // a checkpointed migration output must not be merged with a segment the
        // migration has not accounted for.
        let mut rows_by_class: BTreeMap<bool, Vec<SegmentRow>> = BTreeMap::new();
        for row in self.catalog.list_segments(&self.name)? {
            if let Some(record) = object_records.get(&row.path) {
                rows_by_class
                    .entry(record.migration_backfill)
                    .or_default()
                    .push(row);
            }
        }
        let config = self.object_compaction_config();
        let Some((migration_backfill, job)) =
            rows_by_class.into_iter().find_map(|(backfill, rows)| {
                plan(&config, &rows)
                    .or_else(|| force_compact_l0.then(|| l0_promotion_job(&rows)).flatten())
                    .map(|job| (backfill, job))
            })
        else {
            return Ok(false);
        };

        // The lease pins the definition version and the exact inputs. Merging,
        // encoding, and uploading then run outside the controller; only the
        // replacement commit is serialized against concurrent flushes.
        let lease = self
            .controller
            .begin_compaction(job.inputs.iter().map(|row| row.path.clone()).collect())?;
        for input in &job.inputs {
            let record = object_records.get(&input.path).ok_or_else(|| {
                StatsError::Internal(format!("object compaction lost input {}", input.path))
            })?;
            let localized = self.controller.localize(&record.source).await?;
            if localized != Path::new(&input.path) {
                return Err(StatsError::Internal(format!(
                    "object {} localized to {} rather than its catalog path",
                    input.path,
                    localized.display()
                )));
            }
        }

        let staging = CompactionStaging::create(&table_dir)?;
        let outcome = self
            .run_object_compaction(&staging, &job, &lease, active_version, migration_backfill)
            .await;
        drop(staging);
        outcome
    }

    /// The leveled policy for an object-backed table.
    ///
    /// The table specification declares the object size it wants, so every
    /// level promotes at that byte target rather than the process-wide default.
    fn object_compaction_config(&self) -> CompactionConfig {
        let target = self.runtime_policy().target_object_bytes;
        CompactionConfig {
            level_targets: vec![target; self.compaction_config.level_targets.len()],
            ..self.compaction_config.clone()
        }
    }

    /// Execute one object compaction inside `staging` and commit its result.
    async fn run_object_compaction(
        &self,
        staging: &CompactionStaging,
        job: &CompactionJob,
        lease: &MaintenanceLease,
        active_version: u64,
        migration_backfill: bool,
    ) -> Result<bool, StatsError> {
        let index_config = self.segment_index_config();
        let arrow_schema = Arc::clone(&self.arrow_schema);
        let sort_columns = self.sort_columns.clone();
        let key_column = self.key_column.clone();
        let max_row_group_rows = self.max_row_group_rows;
        let max_merge_arrow_bytes = self.compaction_config.max_merge_arrow_bytes;
        let key_bounds: HashMap<String, (Option<i64>, Option<i64>)> = job
            .inputs
            .iter()
            .map(|row| (row.path.clone(), self.input_key_bounds(&row.path)))
            .collect();
        let job_for_run = job.clone();
        let staging_dir = staging.path().to_path_buf();
        let swap = tokio::task::spawn_blocking(move || {
            run_job_with_partition_policy(
                &job_for_run,
                &staging_dir,
                &arrow_schema,
                CompactionExecution {
                    layout: CompactionLayout {
                        sort_columns: &sort_columns,
                        key_column: &key_column,
                        max_row_group_rows,
                    },
                    index_config: &index_config,
                    partition_policy: None,
                    max_merge_arrow_bytes,
                    output: OutputPolicy::PromoteWhenUnchanged,
                },
                |path| key_bounds.get(path).copied().unwrap_or((None, None)),
            )
        })
        .await
        .map_err(|error| {
            StatsError::Internal(format!("object compaction task panicked: {error}"))
        })??;

        // A single-input run is a level promotion. An immutable object is never
        // renamed, so the promotion re-advertises the same source and artifacts
        // at the higher level instead of rewriting anything.
        if swap.bump_rename.is_some() {
            return self
                .commit_object_level_bump(
                    &swap.removed,
                    swap.added,
                    lease,
                    active_version,
                    migration_backfill,
                )
                .await;
        }
        if swap.added.is_empty() {
            tracing::warn!(
                namespace = %self.name,
                dropped = ?swap.removed,
                "object compaction found no readable input; leaving the run for the next tick"
            );
            return Ok(false);
        }

        let mut outputs = Vec::with_capacity(swap.added.len());
        let mut published = Vec::with_capacity(swap.added.len());
        for staged in swap.added {
            let staged_path = PathBuf::from(&staged.path);
            let stored = self
                .controller
                .write_staged_object(OBJECTS_PREFIX, "parquet", &staged_path)
                .await?;
            let (references, local) = self
                .publish_segment_artifacts(&staged_path, &stored)
                .await?;
            let segment = LocalSegment {
                path: stored.path.to_string_lossy().into_owned(),
                size_bytes: stored.byte_size,
                location: SegmentLocation::Both,
                artifacts: local,
                ..staged
            };
            outputs.push(SegmentDescriptor {
                row: segment_to_row(&self.name, &segment),
                source: stored.source,
                artifacts: references,
            });
            published.push(segment);
        }
        self.commit_object_replacement(
            &swap.removed,
            outputs,
            published,
            lease,
            active_version,
            migration_backfill,
        )
        .await
    }

    /// Upload the artifacts the executor built beside `staged` and return both
    /// their durable references and the local files those references resolve to.
    ///
    /// An artifact that fails to upload is omitted: the source segment commits
    /// without it and index backfill supplies it later.
    async fn publish_segment_artifacts(
        &self,
        staged: &Path,
        stored: &WrittenObject,
    ) -> Result<(ArtifactReferences, LocalArtifacts), StatsError> {
        let built = local_sidecar_artifacts(staged);
        if built.is_empty() {
            return Ok((ArtifactReferences::default(), LocalArtifacts::default()));
        }
        let binding = SourceBinding {
            segment_uuid: segment_id(&stored.path).map(|id| id.to_string()),
            row_count: built.binding.row_count,
        };
        let mut references = ArtifactReferences {
            binding: binding.clone(),
            ..Default::default()
        };
        let mut local = LocalArtifacts {
            binding,
            ..Default::default()
        };
        for (name, path) in &built.projections {
            match self
                .controller
                .write_staged_object("projections", "parquet", path)
                .await
            {
                Ok(uploaded) => {
                    references.projections.insert(name.clone(), uploaded.source);
                    local.projections.insert(name.clone(), uploaded.path);
                }
                Err(error) => {
                    tracing::warn!(namespace = %self.name, projection = %name, %error, "covering projection upload failed; committing without it");
                    return Ok((ArtifactReferences::default(), LocalArtifacts::default()));
                }
            }
        }
        let Some(bundle) = built.bundle.as_ref() else {
            return Ok((ArtifactReferences::default(), LocalArtifacts::default()));
        };
        match self
            .controller
            .write_staged_object("indices", "fidx", bundle)
            .await
        {
            Ok(uploaded) => {
                references.bundle = Some(uploaded.source);
                local.bundle = Some(uploaded.path);
                Ok((references, local))
            }
            Err(error) => {
                tracing::warn!(namespace = %self.name, %error, "index bundle upload failed; committing the segment without it");
                Ok((ArtifactReferences::default(), LocalArtifacts::default()))
            }
        }
    }

    /// Promote a single input to the next level without rewriting its object.
    async fn commit_object_level_bump(
        &self,
        removed: &[String],
        added: Vec<LocalSegment>,
        lease: &MaintenanceLease,
        active_version: u64,
        migration_backfill: bool,
    ) -> Result<bool, StatsError> {
        let records: HashMap<_, _> = self
            .catalog
            .object_segments(&self.name)?
            .into_iter()
            .map(|record| (record.path.clone(), record))
            .collect();
        let mut outputs = Vec::with_capacity(added.len());
        let mut published = Vec::with_capacity(added.len());
        for (staged, input) in added.into_iter().zip(removed) {
            let record = records
                .get(input)
                .ok_or_else(|| StatsError::Internal(format!("level bump lost input {input}")))?;
            let existing = self
                .inner
                .lock()
                .unwrap()
                .local_segments
                .iter()
                .find(|segment| &segment.path == input)
                .map(|segment| segment.artifacts.clone())
                .unwrap_or_default();
            let segment = LocalSegment {
                path: input.clone(),
                artifacts: existing,
                ..staged
            };
            outputs.push(SegmentDescriptor {
                row: segment_to_row(&self.name, &segment),
                source: record.source.clone(),
                artifacts: record.artifacts.clone(),
            });
            published.push(segment);
        }
        self.commit_object_replacement(
            removed,
            outputs,
            published,
            lease,
            active_version,
            migration_backfill,
        )
        .await
    }

    /// Commit one leased replacement and swap the local query view.
    ///
    /// The controller rebases the replacement onto whatever state is current.
    /// A conflict — a retired input, a moved definition version, or a fenced
    /// writer — abandons the outputs rather than failing the table.
    async fn commit_object_replacement(
        &self,
        removed: &[String],
        outputs: Vec<SegmentDescriptor>,
        published: Vec<LocalSegment>,
        lease: &MaintenanceLease,
        active_version: u64,
        migration_backfill: bool,
    ) -> Result<bool, StatsError> {
        let removed_paths = removed.to_vec();
        let committed = match self
            .controller
            .commit_maintenance(lease, || {
                let live: HashSet<String> = self
                    .catalog
                    .object_segments(&self.name)?
                    .into_iter()
                    .map(|record| record.path)
                    .collect();
                if let Some(retired) = removed_paths.iter().find(|path| !live.contains(*path)) {
                    return Err(StatsError::SchemaConflict(format!(
                        "compaction input {retired} is no longer live"
                    )));
                }
                let revision = self.catalog.replace_object_segments(
                    &self.name,
                    &removed_paths,
                    &outputs,
                    active_version,
                    migration_backfill,
                )?;
                Ok((revision, ()))
            })
            .await
        {
            Ok(committed) => Some(committed.token.revision()),
            Err(error) if is_lease_conflict(&error) => {
                tracing::info!(
                    namespace = %self.name,
                    inputs = removed_paths.len(),
                    outputs = published.len(),
                    %error,
                    "compaction lease lost a real conflict; abandoning the uploaded outputs"
                );
                return Ok(false);
            }
            Err(error) if !error.is_committed() => return Err(error.into()),
            // Durable locally but not published; the maintenance loop owes HEAD
            // that revision. The local view still follows the committed rows.
            Err(error) => {
                tracing::warn!(namespace = %self.name, %error, "compaction commit awaits publication");
                None
            }
        };
        let _visibility_guard = self.query_visibility.write().await;
        let retired: HashSet<&String> = removed_paths.iter().collect();
        let mut inner = self.inner.lock().unwrap();
        inner
            .local_segments
            .retain(|segment| !retired.contains(&segment.path));
        inner.local_segments.extend(published.iter().cloned());
        inner
            .local_segments
            .make_contiguous()
            .sort_by_key(|segment| segment.min_seq);
        debug_assert_unique_paths(&inner.local_segments);
        drop(inner);
        tracing::info!(
            namespace = %self.name,
            inputs = removed_paths.len(),
            outputs = published.len(),
            rows = published.iter().map(|segment| segment.row_count).sum::<i64>(),
            catalog_generation = committed.map(|revision| revision.get()),
            "object-backed compaction committed"
        );
        Ok(true)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The arrow schema this namespace's segments are written with (store-form,
    /// includes the implicit `seq` column).
    pub fn arrow_schema(&self) -> &SchemaRef {
        &self.arrow_schema
    }

    /// Resolved physical key, including the implicit `timestamp_ms` default.
    pub fn key_column(&self) -> &str {
        &self.key_column
    }

    /// Snapshot sealed local paths, key bounds, and their lowest `seq` under one
    /// hold of the insertion lock so they describe the same segment set.
    /// `min_seq` is `None` when no local segment exists (an empty namespace, or one
    /// whose segments have all been evicted to remote).
    ///
    /// Snapshotting under the lock is the read side of the query-visibility seam —
    /// compaction takes the write side before unlinking a file, so a query that
    /// captured the pre-compaction paths keeps scanning the files it snapshotted.
    ///
    /// Only SEALED segments appear: queries see flushed data, never the in-RAM buffer.
    pub fn query_snapshot(&self) -> Result<SegmentSnapshot, StatsError> {
        // A table is on the snapshot path once it has an activated object-native
        // definition. Legacy tables, and object-backed tables still importing
        // their version-0 history, read the local segment files the shared
        // query-visibility lock guards.
        if let Some(snapshot) = self.controller.snapshot() {
            if snapshot
                .state()
                .catalog()
                .active_table_spec_version
                .unwrap_or(0)
                > 0
            {
                return self.object_query_snapshot(&snapshot);
            }
        }
        Ok(self.local_query_snapshot())
    }

    /// Plan this table's read from the state its controller last published.
    ///
    /// Metadata only: no object is fetched and the local cache is not consulted,
    /// so a cold cache plans exactly what a warm one does. The scan localizes the
    /// objects its pruning selected.
    fn object_query_snapshot(
        &self,
        snapshot: &TableSnapshot,
    ) -> Result<SegmentSnapshot, StatsError> {
        let store = self.controller.object_store().ok_or_else(|| {
            StatsError::Internal(format!("table {:?} has no object store", self.name))
        })?;
        let planned = plan_visible_segments(snapshot, store.as_ref())?;
        let mut view = SegmentSnapshot {
            paths: Vec::with_capacity(planned.len()),
            key_bounds: BTreeMap::new(),
            partitions: BTreeMap::new(),
            min_seq: planned.iter().map(|segment| segment.min_seq).min(),
            artifacts: SegmentArtifacts::new(),
            sources: SegmentObjectMap::new(),
        };
        for segment in planned {
            if let Some(bounds) = segment.key_bounds {
                view.key_bounds.insert(segment.path.clone(), bounds);
            }
            if let Some(partition) = segment.partition {
                view.partitions.insert(segment.path.clone(), partition);
            }
            if !segment.artifacts.is_empty() {
                view.artifacts
                    .insert(segment.path.clone(), segment.artifacts);
            }
            view.sources.insert(segment.path.clone(), segment.objects);
            view.paths.push(segment.path);
        }
        Ok(view)
    }

    /// The maximum query time this table's pinned state promises, for the
    /// tables a server read plans from immutable object references. `None` for a
    /// legacy table, whose readable window is the query-visibility lock rather
    /// than a duration.
    pub fn snapshot_query_bound(&self) -> Option<Duration> {
        let snapshot = self.controller.snapshot()?;
        let catalog = snapshot.state().catalog();
        if catalog.active_table_spec_version.unwrap_or(0) == 0 {
            return None;
        }
        Some(Duration::from_millis(catalog.max_query_time_ms.unwrap_or(
            crate::store::table_spec::DEFAULT_MAX_QUERY_TIME_MS,
        )))
    }

    fn local_query_snapshot(&self) -> SegmentSnapshot {
        let inner = self.inner.lock().unwrap();
        let key_bounds = inner
            .local_segments
            .iter()
            .filter_map(|segment| {
                Some((
                    segment.path.clone(),
                    (segment.min_key_value?, segment.max_key_value?),
                ))
            })
            .collect();
        SegmentSnapshot {
            paths: inner
                .local_segments
                .iter()
                .map(|s| s.path.clone())
                .collect(),
            key_bounds,
            partitions: inner
                .local_segments
                .iter()
                .filter_map(|segment| Some((segment.path.clone(), segment.partition.clone()?)))
                .collect(),
            min_seq: inner.local_segments.iter().map(|s| s.min_seq).min(),
            artifacts: inner
                .local_segments
                .iter()
                .filter(|segment| !segment.artifacts.is_empty())
                .map(|segment| (segment.path.clone(), segment.artifacts.clone()))
                .collect(),
            sources: SegmentObjectMap::new(),
        }
    }

    /// Subscribe to the durability high-water mark. The current value is already
    /// marked seen, so a caller must read `borrow()` before awaiting `changed()`.
    pub fn watch_persisted_seq(&self) -> watch::Receiver<i64> {
        self.persisted_seq.subscribe()
    }

    /// Record flush demand after an append and wake the scheduler. A buffer that
    /// already holds a whole segment also forces the flush, bypassing the
    /// coalescing window so RAM and L0 size stay bounded. No-op in memory mode,
    /// which never flushes. Call after dropping the inner lock.
    fn notify_flush_after_append(&self, buffered_bytes: i64) {
        if self.data_dir.is_none() {
            return;
        }
        self.flush_requested.store(true, Ordering::SeqCst);
        if buffered_bytes >= self.runtime_policy().max_buffer_bytes {
            self.flush_forced.store(true, Ordering::SeqCst);
        }
        self.maintenance_wake.notify_one();
    }

    /// Record flush demand directly. `forced` bypasses the coalescing window
    /// the way a buffer holding a whole segment does.
    #[cfg(test)]
    pub(crate) fn request_flush(&self, forced: bool) {
        self.flush_requested.store(true, Ordering::SeqCst);
        if forced {
            self.flush_forced.store(true, Ordering::SeqCst);
        }
        self.maintenance_wake.notify_one();
    }

    /// What the scheduler needs to time this namespace's next flush.
    pub(crate) fn flush_demand(&self) -> FlushDemand {
        FlushDemand {
            requested: self.flush_requested.load(Ordering::SeqCst),
            forced: self.flush_forced.load(Ordering::SeqCst),
            max_flush_age: self.runtime_policy().max_flush_age,
        }
    }

    /// Clear the demand the scheduler is about to satisfy. Called immediately
    /// before the flush runs, so an append landing during it re-arms.
    pub(crate) fn clear_flush_demand(&self) {
        self.flush_requested.store(false, Ordering::SeqCst);
        self.flush_forced.store(false, Ordering::SeqCst);
    }

    /// Whether this namespace persists to disk. A memory namespace neither
    /// flushes nor maintains.
    pub(crate) fn is_disk_backed(&self) -> bool {
        self.data_dir.is_some()
    }

    pub(crate) fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// How often this namespace owes an ordinary maintenance cycle.
    pub(crate) fn maintenance_interval(&self) -> Duration {
        self.compaction_config.check_interval
    }

    /// Run `work` as a background task this namespace owns.
    ///
    /// Returns false when the namespace has already stopped, in which case
    /// nothing is spawned: `stop_and_join` has drained the handle list and a
    /// task registered after it would outlive the shutdown window.
    pub(crate) fn spawn_tracked(
        self: Arc<Self>,
        work: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> bool {
        let mut handles = self.task_handles.lock().unwrap();
        if self.stopped.load(Ordering::SeqCst) {
            return false;
        }
        handles.retain(|handle| !handle.is_finished());
        handles.push(tokio::spawn(work));
        true
    }

    /// Stamp `seq` onto `aligned` and append it; returns the last seq allocated
    /// (or `-1` if empty). In memory mode the rows are immediately "persisted".
    pub fn append_aligned_batch(&self, aligned: &AlignedBatch) -> i64 {
        if aligned.num_rows == 0 {
            return -1;
        }
        let mut inner = self.inner.lock().unwrap();
        let n = aligned.num_rows as i64;
        let first_seq = inner.buffers.allocate_seq(n);
        let stamped = stamp_seq_and_build(aligned, first_seq, &self.arrow_schema);
        inner
            .buffers
            .append_batch(stamped, aligned.byte_size + 8 * n);
        let last_seq = first_seq + n - 1;
        if self.data_dir.is_none() {
            // Memory mode: no parquet; the rows are durable the instant they
            // land in RAM, so advance the high-water mark under the lock.
            self.persisted_seq.send_replace(last_seq);
        }
        let buffered_bytes = inner.buffers.ram_bytes();
        drop(inner);
        self.notify_flush_after_append(buffered_bytes);
        last_seq
    }

    /// Append already-built log columns (`seq` excluded) and return the last seq.
    ///
    /// `columns` are the six non-seq log columns in registered order
    /// (key/source/data/epoch_ms/level/cluster), prepared by the caller OUTSIDE
    /// the lock. `num_rows` is their common length and `added_bytes` their raw
    /// buffer size.
    pub fn append_log_batch(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        num_rows: usize,
        added_bytes: i64,
    ) -> i64 {
        if num_rows == 0 {
            return -1;
        }
        let mut inner = self.inner.lock().unwrap();
        let n = num_rows as i64;
        let first_seq = inner.buffers.allocate_seq(n);
        let seq_array: Int64Array = (first_seq..first_seq + n).collect();
        let mut all: Vec<arrow::array::ArrayRef> = Vec::with_capacity(columns.len() + 1);
        all.push(Arc::new(seq_array));
        all.extend(columns);
        let batch = RecordBatch::try_new(Arc::clone(&self.arrow_schema), all)
            .expect("log columns match the stored log schema");
        inner.buffers.append_batch(batch, added_bytes + 8 * n);
        let last_seq = first_seq + n - 1;
        if self.data_dir.is_none() {
            self.persisted_seq.send_replace(last_seq);
        }
        let buffered_bytes = inner.buffers.ram_bytes();
        drop(inner);
        self.notify_flush_after_append(buffered_bytes);
        last_seq
    }

    /// Block until `target` is durable, bounded by `timeout`.
    ///
    /// `target < 0` returns immediately. Otherwise subscribe to `persisted_seq`,
    /// nudge the flush task, and wait for the watch to reach `target`, returning
    /// `Err(DeadlineExceeded)` (mapped to a 504) on timeout.
    pub async fn await_persisted(&self, target: i64, timeout: Duration) -> Result<(), StatsError> {
        if target < 0 {
            return Ok(());
        }
        let mut rx = self.persisted_seq.subscribe();
        if *rx.borrow() >= target {
            return Ok(());
        }
        self.flush_requested.store(true, Ordering::SeqCst);
        self.maintenance_wake.notify_one();
        let wait = async {
            loop {
                if *rx.borrow() >= target {
                    return;
                }
                // `changed()` errors only if the sender dropped; the namespace
                // owns the sender for its whole lifetime, so this cannot happen.
                if rx.changed().await.is_err() {
                    return;
                }
            }
        };
        match tokio::time::timeout(timeout, wait).await {
            Ok(()) => {
                if *self.persisted_seq.borrow() >= target {
                    Ok(())
                } else {
                    // Sender dropped before reaching target — should not happen.
                    Err(StatsError::Internal(format!(
                        "namespace {:?} persisted_seq channel closed before seq>={target}",
                        self.name
                    )))
                }
            }
            Err(_elapsed) => Err(StatsError::DeadlineExceeded(format!(
                "timed out waiting for namespace {:?} to persist seq>={target}",
                self.name
            ))),
        }
    }

    /// Aggregate row/byte/seq stats over sealed segments + the RAM buffer.
    ///
    /// The seq-window math: `min_seq = seg_min if seg_min else (next_seq -
    /// ram_rows if ram_rows else 0)`; `max_seq = max(seg_max, next_seq - 1) if
    /// (seg_max or ram_rows) else 0`. `seg_min`/`seg_max` only consider segments
    /// with `row_count > 0`.
    pub fn stats(&self) -> NamespaceStats {
        let inner = self.inner.lock().unwrap();
        let ram_rows = inner.buffers.ram_rows();
        if inner.local_segments.is_empty() && ram_rows == 0 {
            return NamespaceStats::empty();
        }
        let seg_rows: i64 = inner.local_segments.iter().map(|s| s.row_count).sum();
        let seg_bytes: i64 = inner.local_segments.iter().map(|s| s.size_bytes).sum();
        let seg_min = inner
            .local_segments
            .iter()
            .filter(|s| s.row_count > 0)
            .map(|s| s.min_seq)
            .min()
            .unwrap_or(0);
        let seg_max = inner
            .local_segments
            .iter()
            .filter(|s| s.row_count > 0)
            .map(|s| s.max_seq)
            .max()
            .unwrap_or(0);
        let ram_bytes = inner.buffers.ram_bytes();
        let next_seq = inner.buffers.next_seq();
        let segment_count = inner.local_segments.len() as i32;
        drop(inner);

        let min_seq = if seg_min != 0 {
            seg_min
        } else if ram_rows != 0 {
            next_seq - ram_rows
        } else {
            0
        };
        let max_seq = if seg_max != 0 || ram_rows != 0 {
            seg_max.max(next_seq - 1)
        } else {
            0
        };
        NamespaceStats {
            row_count: seg_rows + ram_rows,
            byte_size: seg_bytes + ram_bytes,
            min_seq,
            max_seq,
            segment_count,
        }
    }

    /// Drain the in-RAM buffer to a new L0 segment, synchronously.
    ///
    /// Test/close sync-point and the body the flush task runs. Returns `Ok(())`
    /// when there was nothing to flush. On parquet-write failure the in-flight
    /// buffer is restored and `persisted_seq` is NOT advanced.
    pub fn flush_once(&self) -> Result<(), StatsError> {
        if self.runtime_policy().object_backed() {
            return Err(StatsError::Internal(format!(
                "object-backed namespace {:?} requires flush_once_async",
                self.name
            )));
        }
        let Some(dir) = self.data_dir.clone() else {
            return Ok(());
        };
        // Serialize the whole seal→write→commit→publish against any other
        // flusher (the bg task and a shutdown/`close` flush can both call this).
        // Holding it across `seal()` is what guarantees a single in-flight
        // `flushing` buffer and in-seq-order `send_replace`.
        let _flush_guard = self.flush_lock.lock().unwrap();
        let sealed = {
            let mut inner = self.inner.lock().unwrap();
            inner.buffers.seal()
        };
        let Some(sealed) = sealed else {
            return Ok(());
        };

        match self.write_sealed(&dir, &sealed) {
            Ok(()) => {
                // Durability-before-ack: the file is renamed and the catalog row
                // is committed before we publish the new high-water seq.
                self.persisted_seq.send_replace(sealed.max_seq);
                Ok(())
            }
            Err(e) => {
                let mut inner = self.inner.lock().unwrap();
                inner.buffers.restore_flush();
                tracing::warn!(namespace = %self.name, error = %e, "flush failed; restored RAM buffer");
                Err(e)
            }
        }
    }

    pub async fn flush_once_async(self: &Arc<Self>) -> Result<(), StatsError> {
        if !self.runtime_policy().object_backed() {
            let namespace = Arc::clone(self);
            return tokio::task::spawn_blocking(move || namespace.flush_once())
                .await
                .map_err(|error| StatsError::Internal(format!("flush task panicked: {error}")))?;
        }
        if self.data_dir.is_none() {
            return Ok(());
        }
        let _flush_guard = self.object_flush_lock.lock().await;
        let policy = self.runtime_policy();
        let sealed = {
            let mut inner = self.inner.lock().unwrap();
            inner.buffers.seal()
        };
        let Some(sealed) = sealed else {
            return Ok(());
        };

        match self.write_sealed_object(&sealed, &policy).await {
            Ok(()) => {
                self.persisted_seq.send_replace(sealed.max_seq);
                Ok(())
            }
            Err(SealedCommit::NotCommitted(error)) => {
                self.inner.lock().unwrap().buffers.restore_flush();
                tracing::warn!(namespace = %self.name, %error, "object-backed flush failed; restored RAM buffer");
                Err(error)
            }
            Err(SealedCommit::PublicationUnresolved(error)) => {
                // The rows are durable in the local catalog. Republishing that
                // revision is the only repair; re-flushing would duplicate it,
                // and the high-water mark waits for HEAD to name it.
                tracing::warn!(namespace = %self.name, %error, "object-backed flush committed without a published revision");
                Err(error)
            }
        }
    }

    async fn write_sealed_object(
        &self,
        sealed: &SealedBuffer,
        policy: &TableRuntimePolicy,
    ) -> Result<(), SealedCommit> {
        let batch = sealed.batch.clone();
        let source_layout = policy.source_layout.clone();
        let max_row_group_rows = source_layout
            .as_ref()
            .and_then(|layout| layout.max_row_group_rows)
            .map(|rows| rows as usize)
            .unwrap_or(self.max_row_group_rows);
        let encoded = tokio::task::spawn_blocking(move || {
            let sorted = sorted_object_batch(&batch, source_layout.as_ref())?;
            partition_object_batch(&sorted, source_layout.as_ref())?
                .into_iter()
                .map(|(partition, batch)| {
                    let (min_seq, max_seq) = batch_seq_bounds(&batch)?;
                    let parquet =
                        write_segment_with_max_row_group_rows(&batch, max_row_group_rows)?;
                    Ok((partition, batch, parquet, min_seq, max_seq))
                })
                .collect::<Result<Vec<_>, StatsError>>()
        })
        .await
        .map_err(|error| {
            StatsError::Internal(format!("object-backed parquet task panicked: {error}"))
        })
        .and_then(|encoded| encoded)
        .map_err(SealedCommit::NotCommitted)?;
        let mut segments = Vec::with_capacity(encoded.len());
        let mut descriptors = Vec::with_capacity(encoded.len());
        for (partition, batch, parquet, min_seq, max_seq) in encoded {
            let stored = self
                .controller
                .write_parquet(Bytes::from(parquet))
                .await
                .map_err(SealedCommit::NotCommitted)?;

            let (min_key, max_key) = self.key_bounds(&batch);
            let segment = LocalSegment {
                path: stored.path.to_string_lossy().into_owned(),
                size_bytes: stored.byte_size,
                level: 0,
                min_seq,
                max_seq,
                row_count: batch.num_rows() as i64,
                created_at_ms: now_ms(),
                min_key_value: min_key,
                max_key_value: max_key,
                partition,
                location: SegmentLocation::Both,
                artifacts: LocalArtifacts::default(),
            };
            descriptors.push(SegmentDescriptor {
                row: segment_to_row(&self.name, &segment),
                source: stored.source,
                // L0 is unindexed: a flush advertises no derived artifacts.
                artifacts: ArtifactReferences::default(),
            });
            segments.push(segment);
        }
        let table_spec_version = policy.table_spec_version;
        // A committed revision owns these objects whether or not HEAD names it
        // yet, so the sealed rows are never re-flushed.
        let committed = match self
            .controller
            .commit(|| {
                let revision =
                    self.catalog
                        .commit_object_segments(&descriptors, table_spec_version, false)?;
                Ok((revision, ()))
            })
            .await
        {
            Err(error) if !error.is_committed() => {
                return Err(SealedCommit::NotCommitted(error.into()))
            }
            outcome => outcome,
        };
        let mut inner = self.inner.lock().unwrap();
        inner.local_segments.extend(segments);
        inner
            .local_segments
            .make_contiguous()
            .sort_by_key(|segment| segment.min_seq);
        debug_assert_unique_paths(&inner.local_segments);
        inner.buffers.commit_flush();
        drop(inner);
        committed
            .map(|_| ())
            .map_err(|error| SealedCommit::PublicationUnresolved(error.into()))
    }

    /// Write the sealed buffer to disk + catalog (no `persisted_seq` advance).
    fn write_sealed(&self, dir: &std::path::Path, sealed: &SealedBuffer) -> Result<(), StatsError> {
        let (path, size) = write_segment_to_dir_with_max_row_group_rows(
            dir,
            0,
            sealed.min_seq,
            &sealed.batch,
            self.max_row_group_rows,
        )?;
        // L0 files are small and short-lived. Derived indexes are built after
        // compaction promotes them to L1+, keeping flush acknowledgement fast
        // while query plans merge indexed counts with uncovered L0 data.
        let (min_key, max_key) = self.key_bounds(&sealed.batch);
        let seg = LocalSegment {
            path: path.to_string_lossy().into_owned(),
            size_bytes: size,
            level: 0,
            min_seq: sealed.min_seq,
            max_seq: sealed.max_seq,
            row_count: sealed.batch.num_rows() as i64,
            created_at_ms: now_ms(),
            min_key_value: min_key,
            max_key_value: max_key,
            partition: None,
            location: SegmentLocation::Local,
            artifacts: LocalArtifacts::default(),
        };
        let row = segment_to_row(&self.name, &seg);
        // Persist the catalog row BEFORE committing the in-RAM flush: the file is
        // already renamed into place, so on an upsert error `flushing` is still
        // intact and `flush_once`'s `restore_flush` returns the rows for retry
        // (rather than silently clearing them with the catalog row missing).
        self.catalog.upsert_segment(&row)?;
        {
            let mut inner = self.inner.lock().unwrap();
            inner.local_segments.push_back(seg);
            debug_assert_unique_paths(&inner.local_segments);
            inner.buffers.commit_flush();
        }
        Ok(())
    }

    /// Int64 key-column bounds from the in-memory sealed batch (cheaper than
    /// re-reading the parquet footer we just wrote).
    fn key_bounds(&self, batch: &RecordBatch) -> (Option<i64>, Option<i64>) {
        let Ok(idx) = batch.schema().index_of(&self.key_column) else {
            return (None, None);
        };
        let Some(col) = batch.column(idx).as_any().downcast_ref::<Int64Array>() else {
            return (None, None);
        };
        if col.null_count() == col.len() {
            return (None, None);
        }
        let mut lo: Option<i64> = None;
        let mut hi: Option<i64> = None;
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let v = col.value(i);
            lo = Some(lo.map_or(v, |x: i64| x.min(v)));
            hi = Some(hi.map_or(v, |x: i64| x.max(v)));
        }
        (lo, hi)
    }

    /// Run one planner-issued compaction job, returning `true` if a job ran.
    ///
    /// Snapshot the deque as `SegmentRow`s under the insertion lock, `plan`, and
    /// if a job is due, execute it and commit the swap. The caller (the
    /// maintenance task / debug `maintain`) drains by
    /// looping while this returns `true`. No-op (returns `false`) in memory mode.
    pub fn compaction_step(&self) -> Result<bool, StatsError> {
        let Some(dir) = self.data_dir.clone() else {
            return Ok(false);
        };
        let rows = {
            let inner = self.inner.lock().unwrap();
            inner
                .local_segments
                .iter()
                .map(|s| segment_to_row(&self.name, s))
                .collect::<Vec<_>>()
        };
        let Some(job) = plan(&self.compaction_config, &rows) else {
            return Ok(false);
        };
        self.run_one_job(&dir, &job)?;
        Ok(true)
    }

    /// Select up to `limit` independent legacy L0 rebuild jobs.
    fn physical_layout_migration_l0_jobs(&self, limit: usize) -> Vec<CompactionJob> {
        if physical_partition_policy_for(&self.name).is_none() {
            return Vec::new();
        }
        let inner = self.inner.lock().unwrap();
        let mut jobs = Vec::new();
        let mut inputs = Vec::new();
        let mut compressed_bytes: i64 = 0;
        // Coalesce compressed inputs before repartitioning. One output set per
        // migration source would turn the existing backlog into hundreds of
        // thousands of tiny L1s.
        for segment in inner
            .local_segments
            .iter()
            .filter(|segment| migration_l0_needs_rebuild(segment))
        {
            if !inputs.is_empty()
                && compressed_bytes.saturating_add(segment.size_bytes)
                    > PHYSICAL_LAYOUT_MIGRATION_WORKER_COMPRESSED_BYTES
            {
                let input_refs = inputs.iter().collect();
                jobs.push(build_job(input_refs, 1));
                inputs.clear();
                compressed_bytes = 0;
                if jobs.len() >= limit {
                    break;
                }
            }
            let mut row = segment_to_row(&self.name, segment);
            // An unpartitioned job forces the executor through its sort and
            // partition path. This also repairs the legacy version that stamped
            // an L0 footer before L0 was defined as policy-free.
            row.partition = None;
            compressed_bytes = compressed_bytes.saturating_add(segment.size_bytes);
            inputs.push(row);
        }
        if jobs.len() < limit && !inputs.is_empty() {
            jobs.push(build_job(inputs.iter().collect(), 1));
        }
        jobs
    }

    /// Rebuild one parallel wave of legacy L0s and atomically publish each input.
    fn physical_layout_migration_l0_wave(&self) -> Result<usize, StatsError> {
        let Some(dir) = self.data_dir.as_deref() else {
            return Ok(0);
        };
        let jobs = self.physical_layout_migration_l0_jobs(PHYSICAL_LAYOUT_MIGRATION_CONCURRENCY);
        if jobs.is_empty() {
            return Ok(0);
        }
        let results = std::thread::scope(|scope| {
            jobs.into_iter()
                .map(|job| scope.spawn(move || self.run_one_job(dir, &job)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join())
                .collect::<Vec<_>>()
        });
        let mut completed = 0;
        for result in results {
            result.map_err(|_| {
                StatsError::Internal("physical layout migration worker panicked".to_string())
            })??;
            completed += 1;
        }
        Ok(completed)
    }

    /// Rebuild or relocate one legacy L1+ segment.
    fn physical_layout_migration_non_l0_step(&self) -> Result<bool, StatsError> {
        let Some(dir) = self.data_dir.as_deref() else {
            return Ok(false);
        };
        let Some(policy) = physical_partition_policy_for(&self.name) else {
            return Ok(false);
        };

        let stale_partition = {
            let inner = self.inner.lock().unwrap();
            inner
                .local_segments
                .iter()
                .find(|segment| partition_is_stale(segment, policy))
                .map(|segment| {
                    let mut row = segment_to_row(&self.name, segment);
                    row.partition = None;
                    row
                })
        };
        if let Some(input) = stale_partition {
            let output_level = input.level;
            let output_min_seq = input.min_seq;
            self.run_one_job(
                dir,
                &CompactionJob {
                    inputs: vec![input],
                    output_level,
                    output_min_seq,
                },
            )?;
            return Ok(true);
        }

        let relocation = {
            let inner = self.inner.lock().unwrap();
            inner.local_segments.iter().find_map(|segment| {
                current_layout_destination(
                    dir,
                    &segment.path,
                    segment.level,
                    segment.partition.as_ref(),
                    policy,
                )
                .map(|destination| (segment.clone(), destination))
            })
        };
        let Some((segment, destination)) = relocation else {
            return Ok(false);
        };
        let mut moved = segment.clone();
        moved.path = destination.to_string_lossy().into_owned();
        // The durable copy, if any, still has the old flat object key. Mark the
        // moved row LOCAL so sync uploads the new key before orphan cleanup can
        // delete the old one.
        moved.location = SegmentLocation::Local;
        self.commit_swap(PlannedSwap {
            removed: vec![segment.path.clone()],
            added: vec![moved],
            unlink_removed: false,
            bump_rename: Some((PathBuf::from(segment.path), destination)),
            input_arrow_bytes: 0,
        })?;
        Ok(true)
    }

    fn physical_layout_migration_pending(&self) -> PhysicalLayoutMigrationPending {
        let Some(dir) = self.data_dir.as_deref() else {
            return PhysicalLayoutMigrationPending::default();
        };
        let Some(policy) = physical_partition_policy_for(&self.name) else {
            return PhysicalLayoutMigrationPending::default();
        };
        let inner = self.inner.lock().unwrap();
        let mut pending = PhysicalLayoutMigrationPending::default();
        for segment in &inner.local_segments {
            if migration_l0_needs_rebuild(segment) {
                pending.migration_l0 += 1;
                continue;
            }
            if segment.level == 0 {
                continue;
            };
            if partition_is_stale(segment, policy) {
                pending.stale_partitions += 1;
                continue;
            }
            pending.misplaced_local += usize::from(
                current_layout_destination(
                    dir,
                    &segment.path,
                    segment.level,
                    segment.partition.as_ref(),
                    policy,
                )
                .is_some(),
            );
        }
        pending
    }

    pub(crate) fn physical_layout_migration_is_pending(&self) -> bool {
        self.physical_layout_migration_pending().any()
    }

    /// Relocate one archived segment to its current physical path.
    async fn remote_layout_migration_step(&self) -> Result<bool, StatsError> {
        let (Some(dir), Some(remote)) = (self.data_dir.as_deref(), self.controller.legacy_store())
        else {
            return Ok(false);
        };

        let candidate = self
            .remote_layout_migration_candidates()?
            .into_iter()
            .next();
        let Some((row, destination)) = candidate else {
            return Ok(false);
        };
        let source_key = segment_relative_key(dir, &row.path).ok_or_else(|| {
            StatsError::Internal(format!(
                "remote layout source is outside namespace directory: {}",
                row.path
            ))
        })?;
        let destination_path = destination.to_string_lossy().into_owned();
        let destination_key = segment_relative_key(dir, &destination_path).ok_or_else(|| {
            StatsError::Internal(format!(
                "remote layout destination is outside namespace directory: {destination_path}"
            ))
        })?;

        let source_id = ObjectId::table(&self.name, &source_key)?;
        let destination_id = ObjectId::table(&self.name, &destination_key)?;
        let source = remote.read(&source_id).await?.ok_or_else(|| {
            StatsError::Internal(format!(
                "legacy object {:?} disappeared during layout migration",
                source_id.as_str()
            ))
        })?;
        remote.write(&destination_id, source.bytes).await?;

        let mut moved = row.clone();
        moved.path = destination_path;
        {
            let _write_guard = self.query_visibility.write().await;
            self.catalog
                .replace_segments(&self.name, std::slice::from_ref(&row.path), &[moved])?;
        }
        if let Err(error) = remote.delete(&source_id).await {
            tracing::warn!(namespace = %self.name, key = %source_key, %error, "legacy object delete failed");
        }
        tracing::info!(
            namespace = %self.name,
            from = %source_key,
            to = %destination_key,
            rows = row.row_count,
            "remote physical layout segment relocated"
        );
        Ok(true)
    }

    fn remote_layout_migration_candidates(&self) -> Result<Vec<(SegmentRow, PathBuf)>, StatsError> {
        let (Some(dir), Some(policy)) = (
            self.data_dir.as_deref(),
            physical_partition_policy_for(&self.name),
        ) else {
            return Ok(Vec::new());
        };
        Ok(self
            .catalog
            .list_segments_min_level(&self.name, 1)?
            .into_iter()
            .filter(|row| row.location == SegmentLocation::Remote)
            .filter_map(|row| {
                current_layout_destination(
                    dir,
                    &row.path,
                    row.level,
                    row.partition.as_ref(),
                    policy,
                )
                .map(|destination| (row, destination))
            })
            .collect())
    }

    fn remote_layout_migration_remaining_count(&self) -> Result<usize, StatsError> {
        Ok(self.remote_layout_migration_candidates()?.len())
    }

    /// Synthesize and apply a single L0->L1 merge of every L0 segment that fits
    /// `max_merge_arrow_bytes` (all of them, at test data sizes).
    ///
    /// Tests use this to land L1 state without configuring tiny `level_targets`.
    /// Production never calls it. No-op when there are no L0 segments (or in
    /// memory mode).
    pub fn force_compact_l0(&self) -> Result<(), StatsError> {
        let Some(dir) = self.data_dir.clone() else {
            return Ok(());
        };
        let l0: Vec<SegmentRow> = {
            let inner = self.inner.lock().unwrap();
            let mut rows: Vec<SegmentRow> = inner
                .local_segments
                .iter()
                .filter(|s| s.level == 0)
                .map(|s| segment_to_row(&self.name, s))
                .collect();
            rows.sort_by_key(|r| r.min_seq);
            rows
        };
        if l0.is_empty() {
            return Ok(());
        }
        let output_min_seq = l0.iter().map(|r| r.min_seq).min().expect("non-empty");
        let job = CompactionJob {
            inputs: l0,
            output_level: 1,
            output_min_seq,
        };
        self.run_one_job(&dir, &job)
    }

    /// Execute `job` (read+merge+write or rename) then commit the resulting swap.
    ///
    /// The executor may consume only a prefix of `job.inputs` — as much as
    /// `max_merge_arrow_bytes` admits — so the committed span comes from the
    /// swap, not the job, and both counts are logged.
    fn run_one_job(&self, dir: &std::path::Path, job: &CompactionJob) -> Result<(), StatsError> {
        let index_config = self.segment_index_config();
        let started = Instant::now();
        tracing::info!(
            namespace = %self.name,
            planned_inputs = job.inputs.len(),
            output_level = job.output_level,
            input_bytes = job.inputs.iter().map(|s| s.byte_size).sum::<i64>(),
            input_rows = job.inputs.iter().map(|s| s.row_count).sum::<i64>(),
            "compaction job starting"
        );
        let swap = run_job_with_partition_policy(
            job,
            dir,
            &self.arrow_schema,
            CompactionExecution {
                layout: CompactionLayout {
                    sort_columns: &self.sort_columns,
                    key_column: &self.key_column,
                    max_row_group_rows: self.max_row_group_rows,
                },
                index_config: &index_config,
                partition_policy: physical_partition_policy_for(&self.name),
                max_merge_arrow_bytes: self.compaction_config.max_merge_arrow_bytes,
                output: OutputPolicy::PromoteWhenUnchanged,
            },
            |path| self.input_key_bounds(path),
        )?;
        let merged_inputs = swap.removed.len();
        let input_arrow_bytes = swap.input_arrow_bytes;
        // A missing head input produces no output — the swap only names the stale
        // reference to drop. Route it through `evict_segment`, which is
        // location-aware (a BOTH segment collapses to REMOTE, preserving its
        // durable archive; a LOCAL-only row is removed) and tolerates the already
        // absent file. This unwedges compaction without deleting a segment that
        // still has a remote copy.
        if swap.added.is_empty() {
            for path in &swap.removed {
                self.evict_segment(path);
            }
            tracing::warn!(
                namespace = %self.name,
                dropped = ?swap.removed,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "dropped stale segment reference with no local file; compaction resumed"
            );
            return Ok(());
        }
        let output_bytes: i64 = swap.added.iter().map(|added| added.size_bytes).sum();
        let output_rows: i64 = swap.added.iter().map(|added| added.row_count).sum();
        let output_segments = swap.added.len();
        // A bump is a rename; a merge decodes its inputs into RAM. The
        // distinction is the whole memory story, so name it — along with the
        // decoded size the ceiling actually bounds, and how much of the planned
        // job that ceiling let this tick take.
        let kind = if swap.bump_rename.is_some() {
            "bump"
        } else {
            "merge"
        };
        self.commit_swap(swap)?;
        tracing::info!(
            namespace = %self.name,
            kind,
            output_level = job.output_level,
            output_segments,
            output_bytes,
            output_rows,
            merged_inputs,
            planned_inputs = job.inputs.len(),
            input_arrow_bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "compaction job committed"
        );
        Ok(())
    }

    /// Names of the schema's STRING columns carrying a trigram substring index
    /// (`ColumnIndex::trigram`); one bloom set is built per returned column.
    fn indexed_columns(&self) -> Vec<&str> {
        self.schema
            .columns
            .iter()
            .filter(|c| c.index.trigram && c.r#type == ColumnType::COLUMN_TYPE_STRING)
            .map(|c| c.name.as_str())
            .collect()
    }

    /// Explicit exact row-index and value-count policies for string columns.
    fn exact_indexes(&self) -> Vec<ExactIndexConfig> {
        self.schema
            .columns
            .iter()
            .filter(|column| {
                column.r#type == ColumnType::COLUMN_TYPE_STRING
                    && (column.index.value_counts || !column.index.exact_values.is_empty())
            })
            .map(|column| ExactIndexConfig {
                column: column.name.clone(),
                exact_values: column.index.exact_values.clone(),
                value_counts: column.index.value_counts,
            })
            .collect()
    }

    fn adaptive_count_columns(&self) -> impl Iterator<Item = &str> {
        self.schema
            .columns
            .iter()
            .filter(|column| column.r#type == ColumnType::COLUMN_TYPE_STRING)
            .map(|column| column.name.as_str())
    }

    fn segment_index_config(&self) -> SegmentIndexConfig {
        if !segment_indexes_enabled_for(&self.name) {
            return SegmentIndexConfig::from_policies(Vec::<String>::new(), &[], &[], None);
        }
        SegmentIndexConfig::from_policies(
            self.indexed_columns(),
            &self.exact_indexes(),
            &self.schema.projections,
            Some(self.key_column.clone()),
        )
        .with_adaptive_value_counts(self.adaptive_count_columns())
        .with_adaptive_group_extrema(self.schema.grouped_extrema.clone())
    }

    /// Recover the typed Int64 key bounds for an input segment from the in-memory
    /// deque (the catalog round-trip stringifies them, losing numeric ordering).
    fn input_key_bounds(&self, path: &str) -> (Option<i64>, Option<i64>) {
        let inner = self.inner.lock().unwrap();
        inner
            .local_segments
            .iter()
            .find(|s| s.path == path)
            .map(|s| (s.min_key_value, s.max_key_value))
            .unwrap_or((None, None))
    }

    /// Splice the deque + catalog: replace `swap.removed` paths with `swap.added`.
    ///
    /// Takes the process-wide query-visibility WRITE lock (via `blocking_write`,
    /// so it is only safe from a `spawn_blocking` / synchronous context — the
    /// maintenance task always calls it that way) so
    /// in-flight queries (which snapshot segment paths and open the parquet files
    /// lazily) have drained before any rename/unlink: renaming or unlinking a
    /// file under a stale snapshot path surfaces as "No files found". A level-bump
    /// rename (`swap.bump_rename`) runs FIRST, inside the held write lock, then
    /// the deque + catalog are spliced under the short insertion lock; merge
    /// inputs are unlinked last.
    ///
    /// Lock order: query_visibility(write) -> insertion lock. The flush path
    /// takes flush_lock + insertion lock but NOT query_visibility, so there is no
    /// cycle.
    fn commit_swap(&self, swap: PlannedSwap) -> Result<(), StatsError> {
        let _write_guard = self.query_visibility.blocking_write();
        // 1) Level-bump rename happens before the deque mirrors the new path, so
        //    a drained reader never sees a half-renamed file. A failure here is
        //    propagated BEFORE any deque/catalog mutation, so the swap aborts
        //    with nothing changed.
        if let Some((from, to)) = &swap.bump_rename {
            let destination_dir = to.parent().ok_or_else(|| {
                StatsError::Internal(format!(
                    "segment destination has no parent: {}",
                    to.display()
                ))
            })?;
            std::fs::create_dir_all(destination_dir).map_err(|e| {
                StatsError::Internal(format!(
                    "create segment destination directory {}: {e}",
                    destination_dir.display()
                ))
            })?;
            std::fs::rename(from, to).map_err(|e| {
                StatsError::Internal(format!(
                    "level-bump rename {} -> {} failed: {e}",
                    from.display(),
                    to.display()
                ))
            })?;
            // The query path no longer reads legacy containers. A source rename
            // would orphan them, so delete them instead of carrying them forward.
            for legacy in legacy_artifact_paths(from) {
                remove_orphaned_index_artifact(&self.name, &legacy, "legacy index");
            }
            let (bundle_from, bundle_to) = (
                crate::indices::format::bundle_path(from),
                crate::indices::format::bundle_path(to),
            );
            if bundle_from.exists() {
                if let Err(error) = std::fs::rename(&bundle_from, &bundle_to) {
                    tracing::warn!(namespace = %self.name, from = %bundle_from.display(), %error, "failed to carry index bundle on level bump");
                    remove_orphaned_index_artifact(&self.name, &bundle_from, "index bundle");
                }
            }
            if let (Some(from_name), Some(to_name)) = (from.file_name(), to.file_name()) {
                let prefix = format!("{}{NAMED_PROJECTION_MARKER}", from_name.to_string_lossy());
                match covering_projection_paths(from) {
                    Ok(projections) => {
                        for source in projections {
                            let name = source.file_name().and_then(|name| name.to_str()).unwrap();
                            let suffix = name.strip_prefix(&prefix).unwrap();
                            let destination = destination_dir.join(format!(
                                "{}{NAMED_PROJECTION_MARKER}{suffix}",
                                to_name.to_string_lossy()
                            ));
                            if let Err(error) = std::fs::rename(&source, &destination) {
                                tracing::warn!(namespace = %self.name, from = %source.display(), %error, "failed to carry covering projection on level bump");
                                remove_orphaned_index_artifact(
                                    &self.name,
                                    &source,
                                    "covering projection",
                                );
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(namespace = %self.name, path = %from.display(), %error, "failed to enumerate covering projections on level bump")
                    }
                }
            }
        }
        let removed_set: std::collections::HashSet<&str> =
            swap.removed.iter().map(|s| s.as_str()).collect();
        assert!(
            !swap.added.is_empty(),
            "commit_swap requires output segments; drops are handled by run_one_job"
        );
        let added_rows: Vec<SegmentRow> = swap
            .added
            .iter()
            .map(|segment| segment_to_row(&self.name, segment))
            .collect();
        {
            let mut inner = self.inner.lock().unwrap();
            inner
                .local_segments
                .retain(|segment| !removed_set.contains(segment.path.as_str()));
            inner.local_segments.extend(swap.added.iter().cloned());
            inner
                .local_segments
                .make_contiguous()
                .sort_by(|left, right| {
                    (left.min_seq, &left.path).cmp(&(right.min_seq, &right.path))
                });
            debug_assert_unique_paths(&inner.local_segments);
            // Atomic catalog splice. Propagate on failure: the
            // deque now points at paths that exist on disk (the renamed bump
            // target / the already-written merged output), so a propagated error
            // is a stats/boot-adoption metadata inconsistency that self-heals at
            // next boot adoption — never a mid-scan-unlink hazard — and the merge
            // inputs below are left intact because we return before unlinking.
            self.catalog
                .replace_segments(&self.name, &swap.removed, &added_rows)?;
        }
        // 2) Unlink merged inputs after the swap (level bumps already renamed).
        if swap.unlink_removed {
            for path in &swap.removed {
                if let Err(e) = std::fs::remove_file(path) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(namespace = %self.name, path = %path, error = %e, "failed to unlink merged input");
                    }
                }
                // The merged output carries a fresh bundle; the inputs' derived
                // indexes are stale and unlinked with their Parquet.
                remove_index_artifacts(path);
            }
        }
        Ok(())
    }

    // ----- remote sync --------------------------------------------------

    /// Two-phase remote sync.
    ///
    /// Phase 1: upload every L>=1 `LOCAL` catalog row (or adopt a row whose file
    /// is already remote — crash recovery), flipping it to `BOTH`. If any upload
    /// fails, `all_durable` is `false`.
    ///
    /// Phase 2 (orphan delete): runs ONLY if `all_durable`. Delete remote files
    /// whose relative key has no catalog row — those are compaction inputs whose row
    /// was dropped at commit. The ordering is the data-safety invariant: by the
    /// time phase 2 runs, the merged output subsuming those inputs is durable in
    /// the bucket (uploaded in phase 1), so the durable copy is in place before
    /// any input remote bytes are deleted. Skipping phase 2 on a failed upload
    /// means the only remaining copies of an unmerged seq range (the inputs in
    /// the bucket) are preserved.
    ///
    /// No-op without a remote dir / in memory mode.
    pub async fn sync_step(&self) -> Result<(), StatsError> {
        let Some(remote) = self.controller.legacy_store() else {
            return Ok(());
        };
        // A TableSpec migration temporarily keeps legacy segments and
        // object-backed cache entries in the same SQLite `segments` table.
        // Legacy sync must continue making the former durable while backfill
        // is incomplete, but object objects already live under the canonical
        // `_finelog` prefix and must never be interpreted as legacy upload
        // candidates or orphans.
        let object_paths: HashSet<String> = self
            .catalog
            .object_segments(&self.name)?
            .into_iter()
            .map(|record| record.path)
            .collect();
        let remote_keys: std::collections::HashSet<String> =
            match remote.list(&ObjectPrefix::table(&self.name, "")?).await {
                Ok(objects) => objects
                    .into_iter()
                    .filter_map(|object| object.id.table_relative(&self.name).map(str::to_string))
                    .collect(),
                Err(e) => {
                    tracing::warn!(namespace = %self.name, error = %e, "remote sync list failed");
                    return Ok(());
                }
            };

        let rows = self.catalog.list_segments_min_level(&self.name, 1)?;
        let namespace_dir = self
            .data_dir
            .as_deref()
            .expect("disk-backed namespace with remote store has a data directory");
        let mut all_durable = true;
        for row in &rows {
            if row.location != SegmentLocation::Local || object_paths.contains(&row.path) {
                continue;
            }
            let Some(key) = segment_relative_key(namespace_dir, &row.path) else {
                tracing::warn!(namespace = %self.name, path = %row.path, "catalog segment is outside its namespace directory");
                all_durable = false;
                continue;
            };
            if remote_keys.contains(&key) {
                // Uploaded but the catalog never flipped — adopt, no re-upload.
                self.mark_uploaded(&row.path)?;
                continue;
            }
            let bytes = match tokio::fs::read(&row.path).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    tracing::warn!(namespace = %self.name, path = %row.path, %error, "legacy upload read failed");
                    all_durable = false;
                    continue;
                }
            };
            if let Err(error) = remote
                .write(&ObjectId::table(&self.name, &key)?, Bytes::from(bytes))
                .await
            {
                tracing::warn!(namespace = %self.name, key = %key, %error, "legacy upload failed");
                all_durable = false;
                continue;
            }
            self.mark_uploaded(&row.path)?;
        }

        if !all_durable {
            return Ok(());
        }

        // The legacy archive is outside the object-backed catalog's MVCC
        // lifetime. Object-backed policy therefore retains archive objects
        // when their local migration source is replaced.
        if self.runtime_policy().object_backed() {
            return Ok(());
        }

        // Re-snapshot the L>=1 catalog rows (phase 1 may have added keys) and
        // delete only genuine orphans. min_level=1 is equivalent to scanning all
        // levels here because remote files are exclusively L>=1 (L0 is never
        // uploaded), so an L0 key can never appear in the remote set.
        let catalog_keys: std::collections::HashSet<String> = self
            .catalog
            .list_segments_min_level(&self.name, 1)?
            .iter()
            .filter(|row| !object_paths.contains(&row.path))
            .filter_map(|row| segment_relative_key(namespace_dir, &row.path))
            .collect();
        for key in remote_keys.difference(&catalog_keys) {
            if let Err(error) = remote.delete(&ObjectId::table(&self.name, key)?).await {
                tracing::warn!(namespace = %self.name, key = %key, %error, "legacy object delete failed");
                continue;
            }
            tracing::info!(namespace = %self.name, segment = %key, "deleted orphan remote segment");
        }
        Ok(())
    }

    /// Flip `path`'s location to `BOTH` after a successful upload, in both the
    /// in-memory deque and the catalog under the insertion lock.
    fn mark_uploaded(&self, path: &str) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        for s in inner.local_segments.iter_mut() {
            if s.path == path {
                s.location = SegmentLocation::Both;
                break;
            }
        }
        self.catalog
            .set_location(&self.name, path, SegmentLocation::Both)?;
        Ok(())
    }

    // ----- eviction -----------------------------------------------------

    /// Evict the namespace's oldest L>=1 copied segments until under the
    /// count/byte caps, then age-trim.
    ///
    /// Caps resolve from the per-namespace `StoragePolicy` first; unset fields
    /// fall back to the cluster-wide `CompactionConfig`. Size/count trim is
    /// FIFO-by-`min_seq` through `select_eviction_candidate` (BOTH only, so a
    /// LOCAL-only segment is never destroyed by the offload path). The age trim
    /// (when `max_age_seconds` is set) drops eligible BOTH segments older than
    /// `now - max_age`, ordered by `created_at_ms`.
    fn eviction_step(&self) -> Result<(), StatsError> {
        let config = &self.compaction_config;
        let policy = self.inner.lock().unwrap().storage_policy.clone();
        let max_segments = policy
            .max_segments
            .map(|v| v as usize)
            .unwrap_or(config.max_segments_per_namespace);
        let max_bytes = policy.max_bytes.unwrap_or(config.max_bytes_per_namespace);
        let max_age_ms = policy.max_age_seconds.map(|s| s * 1000);

        // Size + count trim: FIFO-by-min_seq.
        loop {
            let (seg_count, byte_total) = {
                let inner = self.inner.lock().unwrap();
                let count = inner.local_segments.len();
                let bytes: i64 = inner.local_segments.iter().map(|s| s.size_bytes).sum();
                (count, bytes)
            };
            if seg_count <= max_segments && byte_total <= max_bytes {
                break;
            }
            let Some(row) = self.catalog.select_eviction_candidate(&self.name)? else {
                // Over cap but nothing eligible (still L0, or not yet uploaded).
                break;
            };
            self.evict_segment(&row.path);
        }

        // Age trim: independent of size; ordered by created_at_ms.
        let Some(max_age_ms) = max_age_ms else {
            return Ok(());
        };
        let cutoff_ms = now_ms() - max_age_ms;
        while let Some(row) = self
            .catalog
            .select_aged_eviction_candidate(&self.name, cutoff_ms)?
        {
            self.evict_segment(&row.path);
        }
        Ok(())
    }

    /// Drop `path` from the deque and unlink the local file.
    ///
    /// A `BOTH` segment becomes `REMOTE` in the catalog (the bucket copy is the
    /// durable archive) and the local file is unlinked. A `LOCAL`-only segment
    /// has no durable copy, so eviction is destructive — the catalog row is
    /// dropped. Production eviction routes through `select_eviction_candidate`
    /// (BOTH only); the destructive branch is for direct callers (tests).
    ///
    /// Takes the query-visibility WRITE lock (via `blocking_write`) before the
    /// unlink so an in-flight query that snapshotted this path drains first.
    /// Same lock order as `commit_swap` (query_visibility -> insertion lock).
    pub fn evict_segment(&self, path: &str) -> i64 {
        let _write_guard = self.query_visibility.blocking_write();
        let (removed_bytes, removed_location) = {
            let mut inner = self.inner.lock().unwrap();
            let mut new: VecDeque<LocalSegment> = VecDeque::new();
            let mut removed_bytes = 0;
            let mut removed_location: Option<SegmentLocation> = None;
            for s in inner.local_segments.drain(..) {
                if s.path == path {
                    removed_bytes = s.size_bytes;
                    removed_location = Some(s.location);
                } else {
                    new.push_back(s);
                }
            }
            inner.local_segments = new;
            (removed_bytes, removed_location)
        };
        if removed_location == Some(SegmentLocation::Both) {
            let _ = self
                .catalog
                .set_location(&self.name, path, SegmentLocation::Remote);
        } else {
            let _ = self.catalog.remove_segment(&self.name, path);
        }
        if let Err(e) = std::fs::remove_file(path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(namespace = %self.name, path = %path, error = %e, "failed to delete evicted segment");
            }
        }
        // Derived indexes are local-only, so they are unlinked with the local
        // Parquet on eviction.
        remove_index_artifacts(path);
        removed_bytes
    }

    // ----- segment index backfill ---------------------------------------

    /// Rebuild complete segment-index artifacts for up to `max` L>=1 segments.
    ///
    /// All methods share one projected source read. For a local table the
    /// segment records the local files the build produced. For an object-backed
    /// table each artifact becomes an immutable object and its references are
    /// added to the table state in a new revision: adjacency to a filename is
    /// never what makes an artifact live.
    async fn backfill_index_artifacts(&self, max: usize) -> usize {
        if self.data_dir.is_none() || max == 0 {
            return 0;
        }
        let config = self.segment_index_config();
        if config.is_empty() {
            return 0;
        }
        let Ok(_slot) = self.limits.index_backfill().try_lock() else {
            return 0;
        };
        let candidates = self.index_backfill_candidates(&config, max);
        let mut built = 0;
        for candidate in candidates {
            let path = PathBuf::from(&candidate.path);
            let indices = Arc::clone(&self.indices);
            let build_config = config.clone();
            let build_path = path.clone();
            let artifacts = match tokio::task::spawn_blocking(move || {
                let projection = build_config.input_columns();
                let batches = read_segment_projected(&build_path, Some(&projection))?;
                indices
                    .build(IndexBuildRequest {
                        source: &build_path,
                        batches: &batches,
                        config: &build_config,
                    })
                    .map_err(|error| StatsError::Internal(format!("build segment index: {error}")))
            })
            .await
            {
                Ok(Ok(artifacts)) => artifacts,
                Ok(Err(error)) => {
                    tracing::warn!(namespace = %self.name, path = %candidate.path, %error, "index backfill build failed");
                    continue;
                }
                Err(error) => {
                    tracing::warn!(namespace = %self.name, path = %candidate.path, %error, "index backfill task panicked");
                    continue;
                }
            };
            if !artifacts.is_empty() {
                if let Err(error) = self.commit_backfilled_artifacts(&candidate.path).await {
                    tracing::warn!(namespace = %self.name, path = %candidate.path, %error, "index backfill commit failed");
                    continue;
                }
                built += 1;
                tracing::debug!(namespace = %self.name, path = %candidate.path, "backfilled segment index artifacts");
            }
            if segment_index_needs_rebuild(
                &path,
                candidate.expected_rows,
                &local_sidecar_artifacts(&path),
                &config,
            ) {
                tracing::debug!(namespace = %self.name, path = %candidate.path, "segment cannot satisfy the current index policy; not retrying");
                self.index_backfill_skips
                    .lock()
                    .unwrap()
                    .paths
                    .insert(candidate.path);
            }
        }
        built
    }

    /// The oldest-first segments whose artifacts do not satisfy `config`.
    fn index_backfill_candidates(
        &self,
        config: &SegmentIndexConfig,
        max: usize,
    ) -> Vec<BackfillCandidate> {
        let segments: Vec<LocalSegment> = self
            .inner
            .lock()
            .unwrap()
            .local_segments
            .iter()
            .cloned()
            .collect();
        let fingerprint = format!("{:?}", config.policy_fingerprint());
        let mut skips = self.index_backfill_skips.lock().unwrap();
        let live: HashSet<&str> = segments
            .iter()
            .map(|segment| segment.path.as_str())
            .collect();
        skips.reconcile(&[fingerprint.as_str()], &live);
        let mut candidates = segments
            .iter()
            .filter(|segment| !skips.paths.contains(&segment.path))
            .filter(|segment| {
                segment.level >= 1
                    && self.layout_is_current(&segment.path)
                    && segment_index_needs_rebuild(
                        Path::new(&segment.path),
                        segment.row_count,
                        &segment.artifacts,
                        config,
                    )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|segment| Reverse(segment.max_seq));
        candidates
            .into_iter()
            .take(max)
            .map(|segment| BackfillCandidate {
                path: segment.path.clone(),
                expected_rows: segment.row_count,
            })
            .collect()
    }

    /// Publish the artifacts just built beside `path` and make them live.
    ///
    /// A local table records the files on its segment. An object-backed table
    /// uploads them and commits the references as a new table revision, so a
    /// reader on a cold cache finds them from state rather than from the local
    /// directory.
    async fn commit_backfilled_artifacts(&self, path: &str) -> Result<(), StatsError> {
        let staged = PathBuf::from(path);
        if !self.controller.is_object_backed() {
            let local = local_sidecar_artifacts(&staged);
            if let Some(bundle) = local.bundle.as_ref() {
                self.indices.invalidate(bundle);
            }
            self.set_segment_artifacts(path, local);
            return Ok(());
        }
        let record = self
            .catalog
            .object_segments(&self.name)?
            .into_iter()
            .find(|record| record.path == path)
            .ok_or_else(|| {
                StatsError::Internal(format!("index backfill lost object segment {path}"))
            })?;
        let stored = WrittenObject {
            path: staged.clone(),
            source: record.source.clone(),
            byte_size: 0,
        };
        let (references, local) = self.publish_segment_artifacts(&staged, &stored).await?;
        if references.is_empty() {
            return Ok(());
        }
        let owned_path = path.to_string();
        self.controller
            .commit(|| {
                let revision =
                    self.catalog
                        .set_segment_artifacts(&self.name, &owned_path, &references)?;
                Ok((revision, ()))
            })
            .await?;
        if let Some(bundle) = local.bundle.as_ref() {
            self.indices.invalidate(bundle);
        }
        self.set_segment_artifacts(path, local);
        Ok(())
    }

    /// Point `path`'s query view at the artifacts its references resolve to.
    fn set_segment_artifacts(&self, path: &str, artifacts: LocalArtifacts) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(segment) = inner
            .local_segments
            .iter_mut()
            .find(|segment| segment.path == path)
        {
            segment.artifacts = artifacts;
        }
    }

    /// Remove derived index files from a namespace whose policy disables them.
    fn cleanup_disabled_index_bundles(&self, max: usize) -> usize {
        if self.data_dir.is_none() || max == 0 || segment_indexes_enabled_for(&self.name) {
            return 0;
        }
        let Ok(_slot) = self.limits.index_backfill().try_lock() else {
            return 0;
        };
        let segments: Vec<LocalSegment> = self
            .inner
            .lock()
            .unwrap()
            .local_segments
            .iter()
            .cloned()
            .collect();
        let candidates = {
            let mut skips = self.index_backfill_skips.lock().unwrap();
            let live: HashSet<&str> = segments
                .iter()
                .map(|segment| segment.path.as_str())
                .collect();
            skips.reconcile(&["disabled"], &live);
            let mut candidates = Vec::new();
            for segment in segments.iter().rev() {
                if skips.paths.contains(&segment.path) {
                    continue;
                }
                if fixed_index_artifacts_exist(Path::new(&segment.path)) {
                    candidates.push(segment.path.clone());
                    if candidates.len() >= max {
                        break;
                    }
                } else {
                    skips.paths.insert(segment.path.clone());
                }
            }
            candidates
        };
        let mut cleaned = 0;
        for path in candidates {
            remove_index_artifacts(&path);
            if fixed_index_artifacts_exist(Path::new(&path)) {
                continue;
            }
            if let Some(bundle) = local_sidecar_artifacts(Path::new(&path)).bundle.as_ref() {
                self.indices.invalidate(bundle);
            }
            self.set_segment_artifacts(&path, LocalArtifacts::default());
            self.index_backfill_skips.lock().unwrap().paths.insert(path);
            cleaned += 1;
        }
        if cleaned > 0 {
            tracing::info!(
                namespace = %self.name,
                segments = cleaned,
                "removed disabled segment index artifacts"
            );
        }
        cleaned
    }

    /// Maintain one bounded index batch and return the bundle count changed.
    async fn maintain_index_artifacts(&self) -> usize {
        if segment_indexes_enabled_for(&self.name) {
            self.backfill_index_artifacts(INDEX_BUNDLES_PER_TICK).await
        } else {
            self.cleanup_disabled_index_bundles(INDEX_BUNDLES_PER_TICK)
        }
    }

    fn layout_is_current(&self, path: &str) -> bool {
        if self.current_layouts.lock().unwrap().contains(path) {
            return true;
        }
        if !segment_layout_is_current(Path::new(path)) {
            return false;
        }
        self.current_layouts
            .lock()
            .unwrap()
            .insert(path.to_string());
        true
    }

    /// Advance local layout work and report whether another fast retry is due.
    fn advance_physical_layout_migration(&self) -> Result<bool, StatsError> {
        let migration_slot = match self.limits.layout_migration().try_lock() {
            Ok(slot) => slot,
            Err(TryLockError::WouldBlock) => return Ok(self.physical_layout_migration_is_pending()),
            Err(TryLockError::Poisoned(_)) => {
                return Err(StatsError::Internal(
                    "physical layout migration permit is poisoned".to_string(),
                ));
            }
        };
        let started = Instant::now();
        let mut migrated = 0;
        while !self.stopped.load(Ordering::SeqCst)
            && started.elapsed() < PHYSICAL_LAYOUT_MIGRATION_BUDGET
        {
            let rebuilt = self.physical_layout_migration_l0_wave()?;
            if rebuilt > 0 {
                migrated += rebuilt;
                continue;
            }
            if self.physical_layout_migration_non_l0_step()? {
                migrated += 1;
                continue;
            }
            break;
        }
        drop(migration_slot);
        let pending = self.physical_layout_migration_pending();
        if migrated > 0 {
            tracing::info!(
                namespace = %self.name,
                jobs = migrated,
                remaining_migration_l0 = pending.migration_l0,
                remaining_stale_partition = pending.stale_partitions,
                remaining_misplaced = pending.misplaced_local,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "physical layout migration advanced"
            );
        }
        Ok(pending.any())
    }

    async fn advance_remote_layout_migration(&self) -> Result<(), StatsError> {
        let started = Instant::now();
        let mut migrated = 0;
        while !self.stopped.load(Ordering::SeqCst)
            && started.elapsed() < PHYSICAL_LAYOUT_MIGRATION_BUDGET
            && self.remote_layout_migration_step().await?
        {
            migrated += 1;
        }
        if migrated > 0 {
            tracing::info!(
                namespace = %self.name,
                segments = migrated,
                remaining = self.remote_layout_migration_remaining_count()?,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "remote physical layout migration advanced"
            );
        }
        Ok(())
    }

    // ----- maintenance orchestration ------------------------------------

    async fn gc_published_catalog(&self) -> Result<(), StatsError> {
        if !self.controller.is_object_backed() {
            return Ok(());
        }
        let should_run = {
            let mut last = self.last_object_gc.lock().unwrap();
            let due = last.is_none_or(|instant| instant.elapsed() >= OBJECT_GC_INTERVAL);
            if due {
                *last = Some(Instant::now());
            }
            due
        };
        if !should_run {
            return Ok(());
        }
        // Retired objects answer two independent readers: a query holding a
        // pinned snapshot, and a rollback to the definition they belong to.
        // Collection waits for whichever window is longer.
        let policy = self.runtime_policy();
        let catalog_retention_ms = crate::store::table_spec::retired_object_retention_ms(
            policy.max_query_time_ms,
            policy.rollback_window_ms,
        );
        let orphan_grace_ms = u64::try_from(OBJECT_ORPHAN_GRACE.as_millis()).unwrap_or(u64::MAX);
        let removed = self
            .controller
            .gc_published(now_ms(), catalog_retention_ms, orphan_grace_ms)
            .await?;
        if removed > 0 {
            tracing::info!(namespace = %self.name, removed, "removed obsolete table objects");
        }
        Ok(())
    }

    /// Run one maintenance cycle, serialized against other callers.
    ///
    /// Object-backed tables publish pending catalogs, compact immutable objects,
    /// invoke object-store GC, collect expired objects, and maintain indexes. Legacy
    /// tables compact local segments, synchronize the archive, evict, and perform
    /// physical-layout and index maintenance. No-op in memory mode.
    pub async fn run_maintenance(
        self: &Arc<Self>,
        force_compact_l0: bool,
    ) -> Result<(), StatsError> {
        if self.data_dir.is_none() {
            return Ok(());
        }
        let _maint_guard = self.maint_lock.lock().await;
        self.flush_once_async().await?;
        if self.advance_table_spec_migration().await? {
            return Ok(());
        }
        if self.runtime_policy().object_backed() {
            self.publish_owed_object_catalog().await?;
            self.object_compaction_step(force_compact_l0).await?;
            self.controller.gc_objects().await?;
            self.gc_published_catalog().await?;
            self.maintain_index_artifacts().await;
            return Ok(());
        }

        // Compact (blocking parquet + commit_swap under blocking_write).
        let ns = Arc::clone(self);
        tokio::task::spawn_blocking(move || -> Result<(), StatsError> {
            let migration_pending = ns.advance_physical_layout_migration()?;
            // An optional forced L0->L1 merge, then the planner-drain loop runs so
            // a forced compaction that leaves >= 32 L1 segments still promotes
            // L1->L2 in the same maintenance call. The drain checks the stop latch
            // between jobs: a stop signalled mid-backlog (a re-register replacing
            // this engine, or shutdown) then ends the drain promptly so
            // `stop_and_join` JOINS this task inside its timeout. Otherwise a long
            // drain outlives the timeout, the task is aborted, and its detached
            // blocking compaction keeps unlinking inputs while the replacement
            // engine adopts the same dir — the race that plants a phantom segment
            // (#7361).
            if force_compact_l0 {
                ns.force_compact_l0()?;
            }
            while !ns.stopped.load(Ordering::SeqCst) {
                if !ns.compaction_step()? {
                    break;
                }
                // While the rebuild is active, one ordinary job is enough to
                // keep live L0 bounded. Spend the remaining CPU on releasing
                // legacy inputs; partition-local L1 consolidation can catch up
                // after the source backlog is gone.
                if migration_pending {
                    break;
                }
            }
            Ok(())
        })
        .await
        .map_err(|e| StatsError::Internal(format!("maintenance compact task panicked: {e}")))??;

        // Sync (async object_store).
        self.sync_step().await?;
        self.gc_published_catalog().await?;

        // Relocate evicted objects after local outputs are durable. Each copy is
        // server-side and crash-safe; the time budget prevents a cold archive
        // backlog from monopolizing the maintenance cycle.
        self.advance_remote_layout_migration().await?;

        // Evict (blocking; evict_segment takes blocking_write per segment).
        let ns = Arc::clone(self);
        tokio::task::spawn_blocking(move || ns.eviction_step())
            .await
            .map_err(|e| StatsError::Internal(format!("maintenance evict task panicked: {e}")))??;

        // Maintain derived indexes last and in bounded batches. Namespaces with
        // an active policy backfill missing bundles; namespaces whose managed
        // policy disables indexes remove stale bundles left by older binaries.
        self.maintain_index_artifacts().await;

        // Re-encode segments still on an older physical layout (blocking parquet
        // read + write). Also lowest-priority and bounded: the terminal level is
        // never re-compacted, so without this a namespace carries whatever layout
        // it was written with until eviction ages it out.
        let ns = Arc::clone(self);
        tokio::task::spawn_blocking(move || ns.rewrite_stale_layouts(REWRITE_LAYOUT_BUDGET))
            .await
            .map_err(|e| StatsError::Internal(format!("maintenance rewrite task panicked: {e}")))?;
        Ok(())
    }

    /// Re-encode segments whose physical layout predates the current writer
    /// policy, oldest first, for up to `budget` of wall clock. Returns how many
    /// were rewritten.
    ///
    /// Oldest first because the deque is age-ordered and a leveled store keeps
    /// nearly all of its bytes in the oldest, terminal-level segments — going the
    /// other way spends the first hour rewriting small recent segments while the
    /// footer this exists to shrink stays untouched.
    ///
    /// Costs no remote bandwidth: the rewrite keeps the filename, and the sync
    /// step only uploads segments the catalog still marks `Local`, so a segment
    /// already flipped to `Both` is never re-uploaded. Its remote copy keeps the
    /// old layout while holding identical rows, and ages out normally.
    ///
    /// Bundles on UUID-stamped segments remain valid because the rewrite
    /// preserves segment ID, rows, and row order. Rewriting an older unstamped
    /// segment replaces its local generation identity with a UUID, so its bundle
    /// safely falls back until the next index-backfill pass rebuilds it.
    fn rewrite_stale_layouts(&self, budget: Duration) -> usize {
        if self.data_dir.is_none() {
            return 0;
        }
        let Ok(_slot) = REWRITE_SLOT.try_lock() else {
            return 0;
        };
        let deadline = Instant::now() + budget;
        let mut rewritten = 0;
        // A segment that fails to stage or commit stays stale, so it would be
        // picked again immediately; skipping it for the rest of the pass is what
        // keeps one unreadable file from starving every other segment. The next
        // tick retries it, because the set is per-pass.
        let mut failed: HashSet<String> = HashSet::new();
        while Instant::now() < deadline {
            let Some((path, was)) = self.next_stale_layout(&failed) else {
                break;
            };
            let started = Instant::now();
            let (staging, size) = match stage_rewritten_segment(
                Path::new(&path),
                self.max_row_group_rows,
            ) {
                Ok(staged) => staged,
                Err(e) => {
                    tracing::warn!(namespace = %self.name, segment = %basename(&path), error = %e,
                        "layout rewrite failed; leaving the segment as it was");
                    failed.insert(path);
                    continue;
                }
            };
            match self.commit_rewritten_segment(&path, &staging, size) {
                Ok(true) => {}
                Ok(false) => {
                    // Evicted mid-rewrite: it is gone from the deque, so it will
                    // not come back around.
                    tracing::debug!(namespace = %self.name, segment = %basename(&path),
                        "segment went away mid-rewrite; discarded the replacement");
                    continue;
                }
                Err(e) => {
                    tracing::warn!(namespace = %self.name, segment = %basename(&path), error = %e,
                        "layout rewrite commit failed");
                    failed.insert(path);
                    continue;
                }
            }
            self.current_layouts.lock().unwrap().insert(path.clone());
            tracing::info!(
                namespace = %self.name,
                segment = %basename(&path),
                was_bytes = was,
                now_bytes = size,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "rewrote segment layout"
            );
            rewritten += 1;
        }
        rewritten
    }

    /// The oldest local segment not yet known to carry the current layout, as
    /// `(path, size)`, ignoring anything in `skip`. `None` once every segment is
    /// current.
    ///
    /// Reads footers OUTSIDE the insertion lock — the lock guards only a snapshot
    /// of candidate paths — so a slow filesystem cannot stall writers behind this.
    fn next_stale_layout(&self, skip: &HashSet<String>) -> Option<(String, i64)> {
        let candidates: Vec<(String, i64)> = {
            let inner = self.inner.lock().unwrap();
            let live: HashSet<&str> = inner
                .local_segments
                .iter()
                .map(|s| s.path.as_str())
                .collect();
            let mut known = self.current_layouts.lock().unwrap();
            known.retain(|p| live.contains(p.as_str()));
            inner
                .local_segments
                .iter()
                .filter(|s| s.level >= 1 && !known.contains(&s.path) && !skip.contains(&s.path))
                .map(|s| (s.path.clone(), s.size_bytes))
                .collect()
        };
        for (path, size) in candidates {
            if self.layout_is_current(&path) {
                continue;
            }
            return Some((path, size));
        }
        None
    }

    /// Swap a staged rewrite over its segment and record the new size, under the
    /// insertion lock. Returns `false` (discarding the staged file) when the
    /// segment is no longer live — eviction can drop it while the rewrite runs,
    /// and renaming over that path would resurrect a file nothing references.
    fn commit_rewritten_segment(
        &self,
        path: &str,
        staging: &Path,
        byte_size: i64,
    ) -> Result<bool, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let Some(seg) = inner.local_segments.iter_mut().find(|s| s.path == path) else {
            let _ = std::fs::remove_file(staging);
            return Ok(false);
        };
        std::fs::rename(staging, path).map_err(|e| {
            StatsError::Internal(format!("rename {} -> {path}: {e}", staging.display()))
        })?;
        seg.size_bytes = byte_size;
        drop(inner);
        self.catalog.set_byte_size(&self.name, path, byte_size)?;
        Ok(true)
    }

    /// Backdate one segment's `created_at_ms` in the catalog. Test-only seam
    /// (`/debug/backdate`) so age-eviction tests stay RPC-only (no sleep).
    pub fn backdate_segment(
        &self,
        path_basename: &str,
        created_at_ms: i64,
    ) -> Result<(), StatsError> {
        // Resolve the catalog row whose basename matches; the catalog stores the
        // absolute path, while tests pass the basename.
        let rows = self.catalog.list_segments(&self.name)?;
        for row in &rows {
            if basename(&row.path) == path_basename {
                self.catalog
                    .set_created_at_ms(&self.name, &row.path, created_at_ms)?;
            }
        }
        Ok(())
    }

    /// How many background tasks the scheduler currently has in flight against
    /// this namespace.
    pub fn background_task_count(&self) -> usize {
        self.task_handles.lock().unwrap().len()
    }

    /// Aggregate in-RAM accounting for the diagnostics line:
    /// `(ram_bytes, chunk_count)` under the insertion lock.
    pub fn memory_summary(&self) -> (i64, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.buffers.ram_bytes(), inner.buffers.chunk_count())
    }

    /// Latch the stop flag, wake any dispatched maintenance work, and JOIN it
    /// bounded by `timeout` (a wedged task that misses the window is aborted, so
    /// this can never hang). Does NOT flush — callers sequence durability
    /// (`shutdown`) or pre-delete teardown (`drop_table` re-register replacement)
    /// themselves. Safe to drive via `block_on` from a `spawn_blocking` worker.
    pub async fn stop_and_join(&self, timeout: Duration) {
        // Latch the stop flag FIRST so a task that is mid-flush when the Notify
        // fires still sees the stop on its next loop iteration (the Notify alone
        // stores no permit for notify_waiters), then wake any parked waiters.
        self.stopped.store(true, Ordering::SeqCst);
        self.stop.notify_waiters();
        let handles: Vec<tokio::task::JoinHandle<()>> =
            std::mem::take(&mut *self.task_handles.lock().unwrap());
        // Keep an abort handle for each task so a wedged task that misses the
        // bounded join window can still be cancelled (never busy-wait, never
        // hang). `JoinHandle::abort` is idempotent on an already-finished task.
        let abort_handles: Vec<tokio::task::AbortHandle> =
            handles.iter().map(|h| h.abort_handle()).collect();
        let joined = tokio::time::timeout(timeout, futures::future::join_all(handles)).await;
        match joined {
            Ok(results) => {
                for r in results {
                    if let Err(e) = r {
                        if !e.is_cancelled() {
                            tracing::warn!(namespace = %self.name, error = %e, "shutdown: bg task join error");
                        }
                    }
                }
            }
            Err(_elapsed) => {
                tracing::warn!(
                    namespace = %self.name,
                    "shutdown: bg tasks did not join within timeout; aborting them"
                );
                for h in &abort_handles {
                    h.abort();
                }
            }
        }
    }

    /// Cooperatively shut the namespace down.
    ///
    /// Stops + JOINs any dispatched maintenance work (bounded by `timeout`), then
    /// does a final `flush_once` (no RAM-only rows survive; durability is already
    /// preserved — an acked write was on a sealed segment) and, for a
    /// remote-configured namespace, a final bounded `sync_step` so the bucket
    /// matches the catalog at shutdown.
    pub async fn shutdown(self: &Arc<Self>, timeout: Duration) {
        self.stop_and_join(timeout).await;
        // Final drain so no acked-but-still-RAM rows are lost.
        if let Err(error) = self.flush_once_async().await {
            tracing::warn!(namespace = %self.name, %error, "shutdown: final flush failed");
        }
        // Legacy tables get one final bounded archive sync. Object-backed tables
        // already publish through their write path, so this is a no-op for them.
        if self.has_remote() {
            match tokio::time::timeout(timeout, self.sync_step()).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    tracing::warn!(namespace = %self.name, %error, "shutdown: final remote sync failed");
                }
                Err(_) => {
                    tracing::warn!(namespace = %self.name, "shutdown: final remote sync timed out");
                }
            }
        }
    }

    /// Signal dispatched maintenance work to stop without awaiting it. Safe
    /// to call from a synchronous context with no tokio runtime — used by
    /// `drop_table` for mem-store namespaces, which spawn no background tasks (so
    /// there is nothing to join) before deleting their catalog rows.
    pub fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.stop.notify_waiters();
    }
}

/// Debug-only deque invariant: no two entries share a path.
///
/// A same-path duplicate is a phantom reference — two entries for one seq range,
/// one of whose file a prior compaction already unlinked (#7361). It surfaces
/// duplicate rows in a query and wedges compaction when the planner picks the
/// dead entry. Compiled out of release builds; a cheap guard that trips tests
/// the instant any deque mutation reintroduces a duplicate.
fn debug_assert_unique_paths(segments: &VecDeque<LocalSegment>) {
    if !cfg!(debug_assertions) {
        return;
    }
    let mut seen = std::collections::HashSet::with_capacity(segments.len());
    for s in segments {
        debug_assert!(
            seen.insert(s.path.as_str()),
            "duplicate local-segment deque path: {}",
            s.path
        );
    }
}

/// Build the catalog `SegmentRow` mirroring `seg` (key bounds stringified at the
/// catalog boundary).
fn segment_to_row(namespace: &str, seg: &LocalSegment) -> SegmentRow {
    SegmentRow {
        namespace: namespace.to_string(),
        path: seg.path.clone(),
        level: seg.level,
        min_seq: seg.min_seq,
        max_seq: seg.max_seq,
        row_count: seg.row_count,
        byte_size: seg.size_bytes,
        created_at_ms: seg.created_at_ms,
        min_key_value: seg.min_key_value.map(|v| v.to_string()),
        max_key_value: seg.max_key_value.map(|v| v.to_string()),
        partition: seg.partition.clone(),
        location: seg.location,
    }
}

/// Delete `*.parquet.tmp` left behind by a segment write or layout rewrite that
/// died before its rename. Nothing references them: the catalog only ever names
/// the final path, and `discover_segments` ignores the extension, so a survivor
/// is disk the namespace's own byte accounting cannot see.
fn discard_staging_files(dir: &std::path::Path, namespace: &str) {
    for path in discover_files(dir) {
        if path.extension().and_then(|extension| extension.to_str()) != Some("tmp") {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(namespace = %namespace, file = %path.display(),
                "discarded an abandoned staging file"),
            Err(e) => {
                tracing::warn!(namespace = %namespace, file = %path.display(), error = %e,
                "could not discard an abandoned staging file")
            }
        }
    }
}

/// Adopt segments at boot, reconciling catalog rows against local files.
///
/// Two-pass reconcile:
/// - **Pass 1** walks existing catalog rows. A catalog row with a local file
///   present enters the deque (a `REMOTE` row whose file reappeared collapses to
///   `BOTH`). A `LOCAL` row whose file vanished is dropped (data lost). A `BOTH`
///   row whose file vanished collapses to `REMOTE` (durable archive survives).
///   A `REMOTE`-only row stays in the catalog but NEVER enters the deque (queries
///   don't see archived data; stats exclude it).
/// - **Pass 2** walks local files not seen in pass 1 — files with no catalog row
///   — and adopts them as `LOCAL`, EXCEPT one whose seq range the catalog already
///   covers. A file with no catalog row is either genuinely-new flushed data
///   whose catalog upsert had not yet run (adopt it: crash recovery) or a
///   compaction input the catalog has already superseded — its row replaced by
///   the merge output — but whose unlink has not yet run. Adopting the latter
///   resurrects a phantom segment whose file is about to vanish: a dangling
///   deque reference that wedges compaction (#7361). Monotonic seq allocation
///   separates the two — a genuine flush orphan always sits strictly ABOVE the
///   cataloged high-water seq, a superseded input at or below it — so pass 2
///   skips any file whose `min_seq` is not past every catalog row's `max_seq`.
///
/// The deque is sorted by `min_seq` so iteration matches the planner's
/// oldest-first expectation. Catalog REMOTE rows are left untouched.
fn adopt_local_segments(
    dir: &std::path::Path,
    key_column: Option<&str>,
    catalog: &Catalog,
    namespace: &str,
    objects: Option<&dyn ObjectStore>,
) -> Result<VecDeque<LocalSegment>, StatsError> {
    let started = Instant::now();
    let mut segs: Vec<LocalSegment> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    discard_staging_files(dir, namespace);

    let status = catalog.table_spec_status(namespace)?;
    let object_records: HashMap<_, _> = catalog
        .object_segments(namespace)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();

    let discover_started = Instant::now();
    let mut local_files: std::collections::HashMap<String, PathBuf> = discover_segments(dir)
        .into_iter()
        .map(|p| (p.to_string_lossy().into_owned(), p))
        .collect();
    for record in object_records.values() {
        let path = PathBuf::from(&record.path);
        if path.exists() {
            local_files.insert(record.path.clone(), path);
        }
    }
    let discover_ms = discover_started.elapsed().as_millis() as u64;

    // Pass 1: catalog rows.
    let catalog_started = Instant::now();
    let catalog_rows = catalog.list_segments(namespace)?;
    let catalog_read_ms = catalog_started.elapsed().as_millis() as u64;
    let footer_reconcile_started = Instant::now();
    // Highest seq the catalog still ACCOUNTS FOR after reconciliation: a local
    // file past it is genuinely new (an uncataloged flush); a file at or below it
    // whose row is gone is a compaction input the catalog already superseded (the
    // pass-2 skip). A `LOCAL` row whose file vanished is dropped by pass 1 below —
    // its data is lost — so it must not extend the cutoff, or a lower-seq on-disk
    // file it does not actually cover would be misread as superseded and skipped.
    // Every other row's range stays covered: adopted to the deque, or kept as a
    // `REMOTE` / `BOTH` durable archive.
    let max_catalog_seq = catalog_rows
        .iter()
        .filter(|r| r.location != SegmentLocation::Local || local_files.contains_key(&r.path))
        .map(|r| r.max_seq)
        .max();
    for row in &catalog_rows {
        seen.insert(row.path.clone());
        let Some(local_path) = local_files.get(&row.path) else {
            // Local file gone.
            match row.location {
                SegmentLocation::Local => {
                    // No durable copy — drop the row.
                    catalog.remove_segment(namespace, &row.path)?;
                }
                SegmentLocation::Both => {
                    // Bucket copy is durable; collapse to REMOTE.
                    catalog.set_location(namespace, &row.path, SegmentLocation::Remote)?;
                }
                SegmentLocation::Remote => {}
            }
            continue;
        };
        let Some(meta) = read_segment_footer(local_path, key_column) else {
            continue;
        };
        let location = if row.location == SegmentLocation::Remote {
            SegmentLocation::Both
        } else {
            row.location
        };
        let query_visible = match object_records.get(&row.path) {
            Some(record) => object_segment_is_query_visible(&status, record),
            None => status.active_version() == 0,
        };
        if !query_visible {
            continue;
        }
        let size = std::fs::metadata(local_path)
            .map(|m| m.len() as i64)
            .unwrap_or(0);
        segs.push(LocalSegment {
            path: row.path.clone(),
            size_bytes: size,
            level: meta.level,
            min_seq: meta.min_seq,
            max_seq: meta.max_seq,
            row_count: meta.row_count,
            created_at_ms: row.created_at_ms,
            min_key_value: meta.min_key_value,
            max_key_value: meta.max_key_value,
            partition: meta.partition,
            location,
            artifacts: segment_artifacts(objects, object_records.get(&row.path), local_path)?,
        });
    }

    // Pass 2: local files with no catalog row -> fresh LOCAL segments.
    for (path_str, path) in &local_files {
        if seen.contains(path_str) {
            continue;
        }
        if status.active_version() > 0 {
            continue;
        }
        let Some(meta) = read_segment_footer(path, key_column) else {
            continue;
        };
        // A file with no catalog row whose seq range the catalog already covers
        // is a compaction input whose row the merge output replaced but whose
        // unlink has not yet run (a concurrent adopt caught the post-splice /
        // pre-unlink window). Adopting it would resurrect a phantom segment whose
        // file is about to vanish, wedging compaction (#7361). A genuine
        // uncataloged flush always sits strictly above the cataloged high-water
        // seq, so this only skips the superseded case.
        if let Some(max_seq) = max_catalog_seq {
            if meta.min_seq <= max_seq {
                tracing::warn!(
                    namespace,
                    path = %path_str,
                    file_min_seq = meta.min_seq,
                    file_max_seq = meta.max_seq,
                    max_catalog_seq = max_seq,
                    "skipping orphan segment file already covered by the catalog (superseded compaction input mid-unlink)"
                );
                continue;
            }
        }
        let size = std::fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
        let created_at_ms = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or_else(now_ms);
        segs.push(LocalSegment {
            path: path_str.clone(),
            size_bytes: size,
            level: meta.level,
            min_seq: meta.min_seq,
            max_seq: meta.max_seq,
            row_count: meta.row_count,
            created_at_ms,
            min_key_value: meta.min_key_value,
            max_key_value: meta.max_key_value,
            partition: meta.partition,
            location: SegmentLocation::Local,
            // A file with no catalog row is not object-backed, so its sidecars
            // come from the local layout.
            artifacts: local_sidecar_artifacts(path),
        });
    }

    segs.sort_by_key(|s| s.min_seq);
    let deque: VecDeque<LocalSegment> = segs.into();
    debug_assert_unique_paths(&deque);
    tracing::info!(
        namespace,
        local_files = local_files.len(),
        catalog_rows = catalog_rows.len(),
        adopted_segments = deque.len(),
        discover_ms,
        catalog_read_ms,
        footer_reconcile_ms = footer_reconcile_started.elapsed().as_millis() as u64,
        total_ms = started.elapsed().as_millis() as u64,
        "finelog local segment adoption complete"
    );
    Ok(deque)
}

/// Spawn the per-namespace flush task.
///
/// It wakes on a `Notify` (set by writers), a flush-interval tick, or when the
/// RAM buffer crosses the segment-target byte threshold, and drains the buffer
/// to a new L0 segment via the synchronous `flush_once` (which encodes parquet
/// under the tokio blocking pool implicitly — the encode is fast for the small
/// batches the durability path produces; large batches are bounded by the
/// 16MiB/1Mi write caps).
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;
    use crate::levanter_metrics_policy::levanter_metrics_schema;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::schema::{stored_form, with_implicit_seq, Column, Schema};
    use crate::store::types::seg_filename;

    fn worker_schema() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        ))
    }

    fn aligned(n: i64) -> AlignedBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
        let ts: Vec<i64> = (0..n).map(|i| 1000 + i).collect();
        AlignedBatch {
            arrays: vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(Int64Array::from(ts)),
            ],
            fields: vec![
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            num_rows: n as usize,
            byte_size: 16 * n,
        }
    }

    fn metrics_aligned(run_ids: &[&str]) -> AlignedBatch {
        let schema = schema_to_arrow(&levanter_metrics_schema());
        let rows = run_ids.len();
        let column_names = ["timestamp_ms", "run_id", "step", "name", "kind", "value"];
        AlignedBatch {
            arrays: vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..rows).map(|row| 1_700_000_000_000 + row as i64),
                )),
                Arc::new(StringArray::from(run_ids.to_vec())),
                Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                Arc::new(StringArray::from(vec!["training_loss"; rows])),
                Arc::new(StringArray::from(vec!["scalar"; rows])),
                Arc::new(Float64Array::from_iter_values(
                    (0..rows).map(|row| row as f64 / 10.0),
                )),
            ],
            fields: column_names
                .iter()
                .map(|name| schema.field_with_name(name).unwrap().clone())
                .collect(),
            num_rows: rows,
            byte_size: 128 * rows as i64,
        }
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_namespace_test_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Open a namespace with default wiring (a fresh shared
    /// query-visibility lock, no remote, empty policy) for the unit tests.
    /// Seal the buffer the way the maintenance scheduler would, then wait for
    /// the durability high-water mark. These tests construct a namespace
    /// directly, so no scheduler is polling it.
    async fn flush_and_await(ns: &Arc<Namespace>, target: i64) {
        ns.flush_once_async().await.unwrap();
        ns.await_persisted(target, Duration::from_secs(10))
            .await
            .unwrap();
    }

    fn open_ns(
        name: &str,
        schema: Schema,
        data_dir: Option<PathBuf>,
        catalog: Arc<Catalog>,
    ) -> Arc<Namespace> {
        open_ns_with_policy(name, schema, data_dir, catalog, StoragePolicy::default())
    }

    fn open_ns_with_policy(
        name: &str,
        schema: Schema,
        data_dir: Option<PathBuf>,
        catalog: Arc<Catalog>,
        policy: StoragePolicy,
    ) -> Arc<Namespace> {
        Namespace::open(
            name,
            schema,
            data_dir,
            Arc::clone(&catalog),
            Arc::new(RwLock::new(())),
            crate::indices::test_index_registry(),
            crate::maintenance::MaintenanceLimits::new(),
            Arc::new(Notify::new()),
            TableController::start(
                name.to_string(),
                catalog,
                None,
                crate::store::table_state::WriterFence::UNCLAIMED,
            ),
            policy,
        )
        .unwrap()
    }

    /// Open a namespace with a configured remote dir + per-namespace policy.
    fn open_ns_remote(
        name: &str,
        schema: Schema,
        data_dir: Option<PathBuf>,
        catalog: Arc<Catalog>,
        remote_log_dir: &str,
        policy: StoragePolicy,
    ) -> Arc<Namespace> {
        let provider = crate::store::object_store::build_remote_object_store(remote_log_dir)
            .unwrap()
            .unwrap();
        let cache_root = data_dir
            .as_ref()
            .and_then(|path| path.parent())
            .unwrap()
            .to_path_buf();
        let object_store = Arc::new(
            crate::store::object_store::CachedObjectStore::new(
                Arc::new(provider.clone()),
                cache_root.clone(),
            )
            .unwrap(),
        ) as Arc<dyn crate::store::object_store::ObjectStore>;
        let legacy_object_store = Arc::new(crate::store::object_store::LegacyObjectStore::new(
            &provider,
        ));
        let state_store = Arc::new(
            crate::store::catalog::object_state_store::ObjectTableStateStore::new(
                object_store.clone(),
            ),
        ) as Arc<dyn crate::store::catalog::state_store::TableStateStore>;
        let controller = TableController::start(
            name.to_string(),
            Arc::clone(&catalog),
            Some(crate::store::table::ObjectPersistence {
                table_dir: data_dir.clone().unwrap(),
                store: object_store,
                legacy_store: legacy_object_store,
                state_store,
            }),
            crate::store::table_state::WriterFence::new(1),
        );
        Namespace::open(
            name,
            schema,
            data_dir,
            catalog,
            Arc::new(RwLock::new(())),
            crate::indices::test_index_registry(),
            crate::maintenance::MaintenanceLimits::new(),
            Arc::new(Notify::new()),
            controller,
            policy,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn shutdown_aborts_wedged_task_within_timeout() {
        // The riskiest shutdown path: a bg task stuck in a long compaction/upload
        // that never observes the stop latch. shutdown() must JOIN bounded and
        // ABORT the laggard rather than hang. Inject a never-completing task into
        // the handle set and assert shutdown returns far inside the join timeout.
        let dir = tempdir();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "iris.worker",
            worker_schema(),
            Some(dir.join("iris.worker")),
            catalog,
        );
        let wedged = tokio::spawn(async { std::future::pending::<()>().await });
        ns.task_handles.lock().unwrap().push(wedged);

        let start = std::time::Instant::now();
        ns.shutdown(Duration::from_millis(50)).await;
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "shutdown hung on a wedged task instead of aborting it: {:?}",
            start.elapsed()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn append_then_await_persisted_writes_a_segment() {
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
        );

        let last = ns.append_aligned_batch(&aligned(3));
        assert_eq!(last, 3);
        flush_and_await(&ns, last).await;

        // A segment file exists and stats reflect it.
        let segs = discover_segments(&ns_dir);
        assert_eq!(segs.len(), 1);
        let stats = ns.stats();
        assert_eq!(stats.row_count, 3);
        assert_eq!(stats.min_seq, 1);
        assert_eq!(stats.max_seq, 3);
        assert_eq!(stats.segment_count, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_l0_stays_flat_and_compaction_writes_bucketed_l1() {
        let dir = tempdir();
        let ns_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            catalog,
        );
        ns.append_aligned_batch(&metrics_aligned(&["run-a", "run-b"]));
        ns.flush_once().unwrap();

        let l0 = discover_segments(&ns_dir);
        assert_eq!(l0.len(), 1);
        assert_eq!(l0[0].parent(), Some(ns_dir.as_path()));
        assert!(read_segment_footer(&l0[0], Some("timestamp_ms"))
            .unwrap()
            .partition
            .is_none());

        ns.run_maintenance(true).await.unwrap();
        let l1 = discover_segments(&ns_dir);
        assert_eq!(l1.len(), 2);
        for path in &l1 {
            assert_eq!(
                path.parent().unwrap().parent(),
                Some(ns_dir.join("run_id").as_path())
            );
            let bucket: u32 = path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .parse()
                .unwrap();
            assert!(bucket < 32);
            assert!(read_segment_footer(path, Some("timestamp_ms"))
                .unwrap()
                .partition
                .is_some());
        }
        assert_eq!(ns.stats().row_count, 2);
        ns.shutdown(Duration::from_secs(10)).await;
        let reopened = open_ns(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        assert_eq!(reopened.stats().row_count, 2);
        assert_eq!(reopened.query_snapshot().unwrap().paths.len(), 2);
        reopened.shutdown(Duration::from_secs(10)).await;

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_index_cleanup_converges_across_restart() {
        let dir = tempdir();
        let ns_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            catalog,
        );
        ns.append_aligned_batch(&metrics_aligned(&["run-a", "run-b"]));
        ns.flush_once().unwrap();
        ns.run_maintenance(true).await.unwrap();

        let segments = discover_segments(&ns_dir);
        assert_eq!(segments.len(), 2);
        for segment in &segments {
            std::fs::write(crate::indices::format::bundle_path(segment), b"stale").unwrap();
        }
        let legacy = legacy_artifact_paths(&segments[0])[0].clone();
        std::fs::write(&legacy, b"stale").unwrap();
        let projection = crate::indices::exact::named_projection_path(&segments[0], "legacy");
        std::fs::write(&projection, b"stale").unwrap();
        ns.shutdown(Duration::from_secs(10)).await;

        let cleanup = open_ns(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        assert_eq!(cleanup.cleanup_disabled_index_bundles(1), 1);
        assert_eq!(
            segments
                .iter()
                .filter(|segment| crate::indices::format::bundle_path(segment).exists())
                .count(),
            1
        );
        cleanup.shutdown(Duration::from_secs(10)).await;

        let reopened = open_ns(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        reopened.run_maintenance(false).await.unwrap();
        for segment in &segments {
            assert!(!crate::indices::format::bundle_path(segment).exists());
        }
        assert!(!legacy.exists());
        assert!(!projection.exists());
        assert_eq!(reopened.stats().row_count, 2);
        reopened.shutdown(Duration::from_secs(10)).await;

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_runtime_layout_migration_repairs_legacy_l0_and_flat_l1() {
        let dir = tempdir();
        let ns_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            catalog,
        );
        ns.append_aligned_batch(&metrics_aligned(&["run-a", "run-b"]));
        ns.flush_once().unwrap();
        ns.run_maintenance(true).await.unwrap();

        let first = ns.inner.lock().unwrap().local_segments[0].clone();
        let legacy_l0_path = ns_dir.join(seg_filename(0, first.min_seq));
        let mut legacy_l0 = first.clone();
        legacy_l0.path = legacy_l0_path.to_string_lossy().into_owned();
        legacy_l0.level = 0;
        legacy_l0.location = SegmentLocation::Local;
        let swap = PlannedSwap {
            removed: vec![first.path.clone()],
            added: vec![legacy_l0],
            unlink_removed: false,
            bump_rename: Some((PathBuf::from(first.path), legacy_l0_path)),
            input_arrow_bytes: 0,
        };
        let commit_ns = Arc::clone(&ns);
        tokio::task::spawn_blocking(move || commit_ns.commit_swap(swap))
            .await
            .unwrap()
            .unwrap();
        ns.run_maintenance(false).await.unwrap();

        let second = ns.inner.lock().unwrap().local_segments[1].clone();
        let flat_l1_path = ns_dir.join(seg_filename(1, second.min_seq));
        let mut flat_l1 = second.clone();
        flat_l1.path = flat_l1_path.to_string_lossy().into_owned();
        flat_l1.location = SegmentLocation::Local;
        let swap = PlannedSwap {
            removed: vec![second.path.clone()],
            added: vec![flat_l1],
            unlink_removed: false,
            bump_rename: Some((PathBuf::from(second.path), flat_l1_path.clone())),
            input_arrow_bytes: 0,
        };
        let commit_ns = Arc::clone(&ns);
        tokio::task::spawn_blocking(move || commit_ns.commit_swap(swap))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(flat_l1_path.parent(), Some(ns_dir.as_path()));
        ns.run_maintenance(false).await.unwrap();
        assert!(!ns.physical_layout_migration_is_pending());

        let segments = ns.inner.lock().unwrap().local_segments.clone();
        assert_eq!(segments.len(), 2);
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.row_count)
                .sum::<i64>(),
            2
        );
        for segment in segments {
            assert!(segment.level >= 1);
            assert!(segment.partition.is_some());
            assert_eq!(
                Path::new(&segment.path).parent().unwrap().parent(),
                Some(ns_dir.join("run_id").as_path())
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_streaming_rebuild_publishes_unindexed_outputs() {
        let dir = tempdir();
        let ns_dir = dir.join("levanter.metrics");
        std::fs::create_dir_all(&ns_dir).unwrap();
        let schema = stored_form(levanter_metrics_schema());
        let arrow_schema = schema_to_arrow(&schema);
        for input in 0..6_i64 {
            let first_seq = -1_000_000 + input * 10;
            let batch = stamp_seq_and_build(
                &metrics_aligned(&["run-a", "run-b"]),
                first_seq,
                &arrow_schema,
            );
            write_segment_to_dir(&ns_dir, 0, first_seq, &batch).unwrap();
        }
        let ns = open_ns(
            "levanter.metrics",
            schema,
            Some(ns_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        assert_eq!(ns.physical_layout_migration_pending().migration_l0, 6);

        let migration_ns = Arc::clone(&ns);
        let rebuilt =
            tokio::task::spawn_blocking(move || migration_ns.physical_layout_migration_l0_wave())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(rebuilt, 1);
        assert_eq!(ns.physical_layout_migration_pending().migration_l0, 0);
        assert_eq!(ns.stats().row_count, 12);

        for path in discover_segments(&ns_dir) {
            if read_segment_footer(&path, Some("timestamp_ms"))
                .unwrap()
                .level
                >= 1
            {
                assert!(path.starts_with(ns_dir.join("run_id")));
                assert!(!crate::indices::format::bundle_path(&path).exists());
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn runtime_layout_migration_ignores_namespaces_without_a_partition_policy() {
        let dir = tempdir();
        let ns_dir = dir.join("telemetry_v1.vllm");
        std::fs::create_dir_all(&ns_dir).unwrap();
        let schema = worker_schema();
        let batch = stamp_seq_and_build(&aligned(2), -10, &schema_to_arrow(&schema));
        write_segment_to_dir(&ns_dir, 0, -10, &batch).unwrap();
        let ns = open_ns(
            "telemetry_v1.vllm",
            schema,
            Some(ns_dir),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );

        assert!(!ns.physical_layout_migration_pending().any());
        let migration_ns = Arc::clone(&ns);
        assert_eq!(
            tokio::task::spawn_blocking(move || migration_ns.physical_layout_migration_l0_wave())
                .await
                .unwrap()
                .unwrap(),
            0
        );
        assert_eq!(ns.stats().row_count, 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_runtime_layout_migration_relocates_remote_only_l1() {
        let dir = tempdir();
        let remote_dir = dir.join("remote");
        let ns_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns_remote(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(ns_dir.clone()),
            catalog,
            remote_dir.to_str().unwrap(),
            StoragePolicy::default(),
        );
        ns.append_aligned_batch(&metrics_aligned(&["run-a"]));
        ns.flush_once().unwrap();
        ns.run_maintenance(true).await.unwrap();

        let current = ns
            .catalog
            .list_segments("levanter.metrics")
            .unwrap()
            .remove(0);
        assert_eq!(current.location, SegmentLocation::Both);
        let current_key = segment_relative_key(&ns_dir, &current.path).unwrap();
        let current_remote_path = remote_dir.join("levanter.metrics").join(&current_key);
        assert!(current_remote_path.exists());

        let evict_ns = Arc::clone(&ns);
        let evict_path = current.path.clone();
        tokio::task::spawn_blocking(move || evict_ns.evict_segment(&evict_path))
            .await
            .unwrap();
        let mut legacy = ns
            .catalog
            .list_segments("levanter.metrics")
            .unwrap()
            .remove(0);
        assert_eq!(legacy.location, SegmentLocation::Remote);

        let filename = Path::new(&legacy.path).file_name().unwrap();
        let legacy_remote_path = remote_dir.join("levanter.metrics").join(filename);
        std::fs::rename(&current_remote_path, &legacy_remote_path).unwrap();
        let old_path = legacy.path.clone();
        legacy.path = ns_dir.join(filename).to_string_lossy().into_owned();
        ns.catalog
            .replace_segments("levanter.metrics", &[old_path], &[legacy.clone()])
            .unwrap();

        assert_eq!(ns.remote_layout_migration_remaining_count().unwrap(), 1);
        assert!(ns.remote_layout_migration_step().await.unwrap());
        assert_eq!(ns.remote_layout_migration_remaining_count().unwrap(), 0);
        assert!(!ns.remote_layout_migration_step().await.unwrap());

        let relocated = ns
            .catalog
            .list_segments("levanter.metrics")
            .unwrap()
            .remove(0);
        assert_eq!(relocated.location, SegmentLocation::Remote);
        assert_eq!(
            Path::new(&relocated.path).parent().unwrap().parent(),
            Some(ns_dir.join("run_id").as_path())
        );
        let relocated_key = segment_relative_key(&ns_dir, &relocated.path).unwrap();
        assert!(remote_dir
            .join("levanter.metrics")
            .join(relocated_key)
            .exists());
        assert!(!legacy_remote_path.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn implicit_timestamp_key_captures_segment_bounds() {
        let dir = tempdir();
        let mut schema = worker_schema();
        schema.key_column.clear();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "iris.worker",
            schema,
            Some(dir.join("iris.worker")),
            catalog,
        );
        let last = ns.append_aligned_batch(&aligned(3));
        flush_and_await(&ns, last).await;

        let snapshot = ns.query_snapshot().unwrap();
        assert_eq!(
            snapshot.key_bounds.values().copied().collect::<Vec<_>>(),
            vec![(1000, 1002)]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn await_persisted_negative_returns_immediately() {
        let dir = tempdir();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "iris.worker",
            worker_schema(),
            Some(dir.join("iris.worker")),
            catalog,
        );
        ns.await_persisted(-1, Duration::from_millis(1))
            .await
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn stats_ram_only_seq_window() {
        // Memory mode: no flush; stats come from RAM via the seq window.
        let catalog = Arc::new(Catalog::open(None).unwrap());
        let ns = open_ns("iris.worker", worker_schema(), None, catalog);
        ns.append_aligned_batch(&aligned(3));
        ns.append_aligned_batch(&aligned(2));
        let stats = ns.stats();
        assert_eq!(stats.row_count, 5);
        assert_eq!(stats.min_seq, 1);
        assert_eq!(stats.max_seq, 5);
        assert!(stats.byte_size > 0);
        assert_eq!(stats.segment_count, 0);
    }

    #[tokio::test]
    async fn restart_recovers_next_seq_past_persisted_max() {
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        {
            let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
            let ns = open_ns(
                "iris.worker",
                worker_schema(),
                Some(ns_dir.clone()),
                catalog,
            );
            let last = ns.append_aligned_batch(&aligned(4));
            flush_and_await(&ns, last).await;
        }
        // Second namespace over the same dir: next seq is past the persisted max,
        // and a previously-durable seq is already satisfied.
        let catalog2 = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns2 = open_ns("iris.worker", worker_schema(), Some(ns_dir), catalog2);
        let stats = ns2.stats();
        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.max_seq, 4);
        // A new append continues monotonically from seq 5.
        let last = ns2.append_aligned_batch(&aligned(1));
        assert_eq!(last, 5);
        ns2.await_persisted(4, Duration::from_secs(1))
            .await
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Write a `seg_L{level}_{first_seq}.parquet` of `n` worker rows to `dir` and
    /// return its path. Used to stage the on-disk state adoption reconciles.
    fn write_seg(dir: &Path, level: i32, first_seq: i64, n: i64) -> PathBuf {
        let arrow = schema_to_arrow(&worker_schema());
        let batch = stamp_seq_and_build(&aligned(n), first_seq, &arrow);
        write_segment_to_dir(dir, level, first_seq, &batch)
            .unwrap()
            .0
    }

    /// A `LOCAL` catalog `SegmentRow` for `path`. Key bounds are re-read from the
    /// file footer during adoption, so they are left `None` here.
    fn seg_row(path: &Path, level: i32, min_seq: i64, max_seq: i64) -> SegmentRow {
        SegmentRow {
            namespace: "iris.task".to_string(),
            path: path.to_string_lossy().into_owned(),
            level,
            min_seq,
            max_seq,
            row_count: max_seq - min_seq + 1,
            byte_size: 1,
            created_at_ms: 1,
            min_key_value: None,
            max_key_value: None,
            partition: None,
            location: SegmentLocation::Local,
        }
    }

    #[test]
    fn adopt_skips_superseded_compaction_input_still_on_disk() {
        // Regression for #7361. A compaction had committed its catalog splice —
        // `replace_segments` swapped the L0 input rows for the merged L1 output —
        // but had not yet unlinked the input files when adoption ran (a
        // re-register replacing the engine, or a crash, caught the post-splice /
        // pre-unlink window). Pass 2 must not resurrect those inputs as phantom
        // segments whose files are about to vanish, while still adopting a genuine
        // uncataloged flush orphan.
        let dir = tempdir();
        let ns_dir = dir.join("iris.task");
        std::fs::create_dir_all(&ns_dir).unwrap();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());

        // The committed merge output L1 [1..4]: on disk AND in the catalog.
        let l1 = write_seg(&ns_dir, 1, 1, 4);
        catalog.upsert_segment(&seg_row(&l1, 1, 1, 4)).unwrap();

        // A superseded L0 input [1..2]: still on disk, its catalog row already
        // gone. Its seq range is covered by the L1 output — the phantom.
        let l0_input = write_seg(&ns_dir, 0, 1, 2);

        // A genuine fresh flush orphan [5..6]: file written, catalog upsert not
        // yet run. It sits above the cataloged high-water seq (4) — adopt it.
        let l0_new = write_seg(&ns_dir, 0, 5, 2);

        let deque =
            adopt_local_segments(&ns_dir, Some("timestamp_ms"), &catalog, "iris.task", None)
                .unwrap();
        let has = |p: &Path| deque.iter().any(|s| s.path == p.to_string_lossy());

        assert!(has(&l1), "the merge output is adopted");
        assert!(
            has(&l0_new),
            "a genuine flush orphan above the cataloged high-water seq is adopted"
        );
        assert!(
            !has(&l0_input),
            "the superseded compaction input must NOT be resurrected as a phantom"
        );
        assert_eq!(deque.len(), 2, "only the output and the genuine orphan");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn adopt_recovers_low_orphan_under_a_stale_high_catalog_row() {
        // The coverage cutoff must come only from rows whose data survives pass 1.
        // A stale LOCAL catalog row whose file is gone is dropped (its data lost),
        // so it must not extend the cutoff and mask a lower-seq on-disk file that
        // it never actually covered — that file is a recoverable orphan.
        let dir = tempdir();
        let ns_dir = dir.join("iris.task");
        std::fs::create_dir_all(&ns_dir).unwrap();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());

        // A stale LOCAL row [100..200] whose parquet was never written to disk.
        let gone = ns_dir.join("seg_L0_00000000000000000100.parquet");
        catalog
            .upsert_segment(&seg_row(&gone, 0, 100, 200))
            .unwrap();

        // A real on-disk orphan [1..2], lower seq than the stale row, no catalog
        // row. It must be recovered, not skipped as covered.
        let orphan = write_seg(&ns_dir, 0, 1, 2);

        let deque =
            adopt_local_segments(&ns_dir, Some("timestamp_ms"), &catalog, "iris.task", None)
                .unwrap();
        let has = |p: &Path| deque.iter().any(|s| s.path == p.to_string_lossy());
        assert!(
            has(&orphan),
            "a low-seq orphan is recovered even under a stale high catalog row"
        );
        assert_eq!(deque.len(), 1, "only the recovered orphan");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn flush_coalesces_multiple_appends_into_few_segments() {
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
        );
        // Many small appends; flush via the direct sync-point once.
        let mut last = -1;
        for _ in 0..5 {
            last = ns.append_aligned_batch(&aligned(2));
        }
        ns.flush_once().unwrap();
        ns.await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        let segs = discover_segments(&ns_dir);
        assert_eq!(segs.len(), 1, "one flush coalesces buffered appends");
        assert_eq!(ns.stats().row_count, 10);
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- segment index backfill -----------------------------------------

    /// Log-form schema carrying the trigram-indexed `data` string column.
    fn data_schema() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        ))
    }

    fn exact_data_schema() -> Schema {
        with_implicit_seq(
            Schema::new(
                vec![
                    Column::new("data", ColumnType::COLUMN_TYPE_STRING, false)
                        .with_exact_values(["log line 0 searchable text"])
                        .with_value_counts(),
                    Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                ],
                "timestamp_ms",
            )
            .with_covering_projection(crate::store::schema::CoveringProjection::new(
                "matching-lines",
                "data",
                ["log line 0 searchable text"],
                ["seq", "data", "timestamp_ms"],
            )),
        )
    }

    #[tokio::test]
    async fn low_cardinality_counts_are_automatic_for_string_columns() {
        let dir = tempdir();
        let ns = open_ns(
            "logs",
            data_schema(),
            Some(dir.join("logs")),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );

        assert!(ns.segment_index_config().indexes.iter().any(|index| {
            matches!(
                index,
                crate::indices::IndexSpec::AdaptiveValueCounts { column }
                    if column == "data"
            )
        }));
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn grouped_extrema_are_declared_instead_of_column_inferred() {
        let dir = tempdir();
        let schema = Schema::new(
            vec![
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("service", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new(
                    "resource_attributes_json",
                    ColumnType::COLUMN_TYPE_STRING,
                    false,
                ),
            ],
            "timestamp_ms",
        );
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let undeclared = open_ns(
            "same_columns_without_policy",
            with_implicit_seq(schema.clone()),
            Some(dir.join("same_columns_without_policy")),
            Arc::clone(&catalog),
        );
        assert!(!undeclared
            .segment_index_config()
            .indexes
            .iter()
            .any(|index| {
                matches!(
                    index,
                    crate::indices::IndexSpec::AdaptiveGroupExtrema { .. }
                )
            }));

        let config = crate::indices::group_extrema::GroupExtremaConfig::new(
            "service",
            "resource_attributes_json",
            "job_id",
            "timestamp_ms",
        );
        let declared = open_ns(
            "declared_policy",
            with_implicit_seq(schema.with_grouped_extrema(config)),
            Some(dir.join("declared_policy")),
            catalog,
        );
        assert!(declared.segment_index_config().indexes.iter().any(|index| {
            matches!(
                index,
                crate::indices::IndexSpec::AdaptiveGroupExtrema { config }
                    if config.filter_column == "service"
                        && config.json_column == "resource_attributes_json"
                        && config.json_key == "job_id"
                        && config.extrema_column == "timestamp_ms"
            )
        }));
        std::fs::remove_dir_all(dir).ok();
    }

    /// `n` rows of searchable `data` + monotonic `timestamp_ms` (non-seq columns
    /// in registered order, as `append_aligned_batch` expects).
    fn data_aligned(n: i64, first: i64) -> AlignedBatch {
        let data: Vec<String> = (0..n)
            .map(|i| format!("log line {} searchable text", first + i))
            .collect();
        let ts: Vec<i64> = (0..n).map(|i| 1000 + first + i).collect();
        AlignedBatch {
            arrays: vec![
                Arc::new(StringArray::from(data)),
                Arc::new(Int64Array::from(ts)),
            ],
            fields: vec![
                Field::new("data", DataType::Utf8, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            num_rows: n as usize,
            byte_size: 48 * n,
        }
    }

    #[tokio::test]
    async fn backfill_rebuilds_missing_index_bundle() {
        let dir = tempdir();
        let ns_dir = dir.join("log.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns("log.test", data_schema(), Some(ns_dir.clone()), catalog);

        // Two L0 flushes merged to one L1 — the merge builds the bundle.
        ns.append_aligned_batch(&data_aligned(5, 0));
        ns.flush_once().unwrap();
        let last = ns.append_aligned_batch(&data_aligned(5, 5));
        ns.flush_once().unwrap();
        ns.await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        // run_maintenance wraps the merge in spawn_blocking (commit_swap takes the
        // blocking query-visibility lock); a multi-input merge builds the bundle.
        ns.run_maintenance(true).await.unwrap();

        let segs = discover_segments(&ns_dir);
        assert_eq!(segs.len(), 1, "two L0 merged into one L1");
        let bundle = crate::indices::format::bundle_path(&segs[0]);
        assert!(bundle.exists(), "the merge wrote an index bundle");

        // Simulate a segment compaction never indexed (single-input bump, or one
        // written before indexes existed): drop the bundle.
        std::fs::remove_file(&bundle).unwrap();
        assert!(!bundle.exists());

        // The backfill rebuilds exactly the one missing bundle, then idles.
        assert_eq!(ns.backfill_index_artifacts(10).await, 1);
        assert!(bundle.exists(), "backfill rebuilt the bundle");
        assert_eq!(
            ns.backfill_index_artifacts(10).await,
            0,
            "nothing left to do"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn exact_backfill_rebuilds_a_missing_filtered_projection() {
        let dir = tempdir();
        let ns_dir = dir.join("telemetry.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "telemetry.test",
            exact_data_schema(),
            Some(ns_dir.clone()),
            catalog,
        );
        ns.append_aligned_batch(&data_aligned(5, 0));
        ns.flush_once().unwrap();
        ns.run_maintenance(true).await.unwrap();

        let segments = discover_segments(&ns_dir);
        assert_eq!(segments.len(), 1);
        assert!(segments[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("seg_L1_"));
        let bundle = crate::indices::format::bundle_path(&segments[0]);
        let projection =
            crate::indices::exact::named_projection_path(&segments[0], "matching-lines");
        assert!(bundle.exists());
        assert!(projection.exists());
        std::fs::remove_file(&projection).unwrap();

        assert_eq!(ns.backfill_index_artifacts(10).await, 1);
        assert!(projection.exists());
        assert_eq!(ns.backfill_index_artifacts(10).await, 0);

        assert!(crate::indices::format::bundle_path(&segments[0]).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn l0_does_not_write_derived_index_artifacts() {
        let dir = tempdir();
        let ns_dir = dir.join("telemetry.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "telemetry.test",
            exact_data_schema(),
            Some(ns_dir.clone()),
            catalog,
        );
        ns.append_aligned_batch(&data_aligned(5, 0));
        ns.flush_once().unwrap();

        let segments = discover_segments(&ns_dir);
        assert_eq!(segments.len(), 1);
        assert!(segments[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("seg_L0_"));
        assert!(!crate::indices::format::bundle_path(&segments[0]).exists());
        assert!(
            !crate::indices::exact::named_projection_path(&segments[0], "matching-lines").exists()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn backfill_is_a_noop_without_the_indexed_column() {
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        // worker_schema has no `data` column, so there is nothing to index.
        let ns = open_ns(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
        );
        ns.append_aligned_batch(&aligned(3));
        ns.flush_once().unwrap();
        let last = ns.append_aligned_batch(&aligned(3));
        ns.flush_once().unwrap();
        ns.await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        ns.run_maintenance(true).await.unwrap();

        assert_eq!(ns.backfill_index_artifacts(10).await, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A column indexed after a segment was written is not in that segment to
    /// index, so its bundle can never satisfy the rebuild condition. The
    /// backfill must try once and drop it, or it re-reads that segment on every
    /// tick and never reaches the segments that can still be indexed.
    #[tokio::test]
    async fn backfill_gives_up_on_a_segment_predating_an_indexed_column() {
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());

        // Segments written under a schema with no `data` column at all.
        let before = open_ns(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog.clone(),
        );
        before.append_aligned_batch(&aligned(3));
        before.flush_once().unwrap();
        let last = before.append_aligned_batch(&aligned(3));
        before.flush_once().unwrap();
        before
            .await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        before.run_maintenance(true).await.unwrap();
        let segs = discover_segments(&ns_dir);
        assert_eq!(segs.len(), 1, "two L0 merged into one L1");
        assert!(
            crate::indices::format::bundle_path(&segs[0]).exists(),
            "the original string columns receive adaptive counts"
        );
        before.shutdown(Duration::from_secs(10)).await;

        // Reopen with `data` added and indexed, as `merge_schemas` would leave it.
        let mut columns = worker_schema().columns;
        columns
            .push(Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index());
        let after = open_ns(
            "iris.worker",
            Schema::new(columns, "timestamp_ms"),
            Some(ns_dir.clone()),
            catalog,
        );

        assert_eq!(
            after.backfill_index_artifacts(10).await,
            1,
            "available adaptive sections are rebuilt once"
        );
        let path = segs[0].to_string_lossy().to_string();
        assert!(
            after
                .index_backfill_skips
                .lock()
                .unwrap()
                .paths
                .contains(&path),
            "the segment is dropped from future ticks rather than retried",
        );

        // Indexing another column is a new question, so the verdict is dropped.
        after
            .index_backfill_skips
            .lock()
            .unwrap()
            .reconcile(&["data", "worker_id"], &HashSet::from([path.as_str()]));
        assert!(after.index_backfill_skips.lock().unwrap().paths.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn maintenance_drops_dangling_segment_reference_instead_of_wedging() {
        // Regression for the `iris.task` compaction wedge. A merge that consumed
        // and unlinked a segment can leave its deque/catalog reference behind (a
        // duplicate entry the splice missed). The planner then hands back a job
        // whose head input file is gone; the old recovery tried to promote it by
        // rename, which failed on the absent source every `check_interval` and
        // wedged the namespace's compaction for good (14k L0 files and growing in
        // production). Maintenance must instead DROP the dangling reference and
        // keep compacting.
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
        );

        // Three L0 segments (seq 1, 2, 3), each its own flush.
        write_one(&ns).await;
        write_one(&ns).await;
        write_one(&ns).await;
        let before = discover_segments(&ns_dir);
        assert_eq!(before.len(), 3, "three L0 segments on disk");

        // Delete the lowest-min_seq file while its deque + catalog rows survive —
        // the dangling reference an already-consumed-and-unlinked input leaves,
        // and the head of the next planned run.
        let head = before.iter().min().unwrap().clone();
        std::fs::remove_file(&head).unwrap();

        // Before the fix this returned Err (rename of the absent head failed) and
        // every later tick replanned the identical doomed job.
        ns.run_maintenance(true)
            .await
            .expect("a dangling reference must not wedge maintenance");

        // The stale reference is gone from the catalog, and the two intact rows
        // survive (compacted forward, none lost).
        let rows = ns.catalog.list_segments("iris.worker").unwrap();
        let head_str = head.to_string_lossy().to_string();
        assert!(
            rows.iter().all(|r| r.path != head_str),
            "the dangling reference was dropped from the catalog"
        );
        let total_rows: i64 = rows.iter().map(|r| r.row_count).sum();
        assert_eq!(total_rows, 2, "the two intact segments' rows survive");

        // Compaction is live again: a further tick runs without error.
        ns.run_maintenance(true)
            .await
            .expect("compaction stays live after the drop");
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- maintenance + remote sync + eviction ---------------------------

    fn remote_files(remote: &std::path::Path, namespace: &str) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(remote.join(namespace))
            .map(|rd| {
                rd.flatten()
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| n.ends_with(".parquet"))
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }

    /// Append one batch and force it durable on a sealed L0 segment.
    async fn write_one(ns: &Arc<Namespace>) {
        let last = ns.append_aligned_batch(&aligned(1));
        ns.flush_once().unwrap();
        ns.await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn maintain_uploads_compacted_segment_and_flips_both() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir),
            catalog,
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        write_one(&ns).await;
        // L0 promoted to L1, then sync uploads it -> BOTH; remote file present.
        ns.run_maintenance(true).await.unwrap();
        let files = remote_files(&remote, "iris.worker");
        assert_eq!(files.len(), 1, "one compacted L1 segment uploaded");
        let segs = ns.catalog.list_segments("iris.worker").unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].level, 1);
        assert_eq!(segs[0].location, SegmentLocation::Both);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn layout_rewrite_updates_the_local_segment_without_re_uploading() {
        use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
        use parquet::file::properties::WriterProperties;

        let dir = tempdir();
        let remote = dir.join("remote");
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir),
            catalog,
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap();

        let seg = ns.catalog.list_segments("iris.worker").unwrap().remove(0);
        assert_eq!(seg.location, SegmentLocation::Both);
        let remote_name = remote_files(&remote, "iris.worker").remove(0);
        let remote_path = remote.join("iris.worker").join(&remote_name);
        let remote_before = std::fs::read(&remote_path).unwrap();

        // Put the local file back onto an older layout (no stamp), as a
        // pre-existing segment would be.
        let path = std::path::PathBuf::from(&seg.path);
        let batches: Vec<RecordBatch> = {
            let f = std::fs::File::open(&path).unwrap();
            ParquetRecordBatchReaderBuilder::try_new(f)
                .unwrap()
                .build()
                .unwrap()
                .map(|b| b.unwrap())
                .collect()
        };
        let out = std::fs::File::create(&path).unwrap();
        let opts = ArrowWriterOptions::new().with_properties(WriterProperties::builder().build());
        let mut w = ArrowWriter::try_new_with_options(out, batches[0].schema(), opts).unwrap();
        for b in &batches {
            w.write(b).unwrap();
        }
        w.close().unwrap();
        // The layout cache is per-process and starts empty, so a segment written
        // by an older build is always first seen after a restart. Clear it to
        // model that: nothing in production edits a segment behind the cache.
        ns.current_layouts.lock().unwrap().clear();
        assert_eq!(ns.rewrite_stale_layouts(REWRITE_LAYOUT_BUDGET), 1);

        // Local file adopted the current layout and the catalog followed it.
        assert!(crate::store::segment::segment_layout_is_current(&path));
        let after = ns.catalog.list_segments("iris.worker").unwrap().remove(0);
        assert_eq!(
            after.byte_size,
            std::fs::metadata(&path).unwrap().len() as i64
        );
        assert_eq!(after.location, SegmentLocation::Both);

        // The archive is untouched: same object, same bytes, no re-upload.
        assert_eq!(remote_files(&remote, "iris.worker"), vec![remote_name]);
        assert_eq!(std::fs::read(&remote_path).unwrap(), remote_before);

        // A second pass finds nothing left to do.
        assert_eq!(ns.rewrite_stale_layouts(REWRITE_LAYOUT_BUDGET), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn eviction_drops_oldest_both_preserving_remote_archive() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        // cap = 1 segment: after two compaction+upload cycles the oldest is
        // evicted (BOTH -> REMOTE + local unlink), remote archive survives.
        let ns = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
            remote.to_str().unwrap(),
            StoragePolicy {
                max_segments: Some(1),
                ..Default::default()
            },
        );

        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap(); // L1 #1, uploaded, BOTH
        let first_l1: Vec<_> = discover_segments(&ns_dir)
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("seg_L1_")
            })
            .collect();
        assert_eq!(first_l1.len(), 1);

        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap(); // L1 #2; cap=1 evicts oldest

        // Local L1 files: exactly one remains, and it is NOT the first one.
        let local_l1: Vec<_> = discover_segments(&ns_dir)
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("seg_L1_")
            })
            .collect();
        assert_eq!(local_l1.len(), 1, "evicted oldest local L1");
        assert!(!first_l1[0].exists(), "oldest local file unlinked");

        // Remote keeps BOTH segments (durable archive preserved).
        assert_eq!(remote_files(&remote, "iris.worker").len(), 2);

        // Catalog: the evicted segment is REMOTE; stats exclude it.
        let segs = ns.catalog.list_segments("iris.worker").unwrap();
        let remote_rows = segs
            .iter()
            .filter(|s| s.location == SegmentLocation::Remote)
            .count();
        assert_eq!(remote_rows, 1);
        let stats = ns.stats();
        assert_eq!(stats.segment_count, 1, "REMOTE excluded from stats");
        assert_eq!(stats.row_count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn eviction_skips_local_only_when_no_remote() {
        let dir = tempdir();
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        // cap = 1, NO remote: nothing is BOTH, so nothing is evictable — two L1
        // segments must survive (eviction must never destroy LOCAL-only data).
        let ns = open_ns_with_policy(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
            StoragePolicy {
                max_segments: Some(1),
                ..Default::default()
            },
        );
        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap();
        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap();
        let local_l1 = discover_segments(&ns_dir)
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("seg_L1_")
            })
            .count();
        assert_eq!(local_l1, 2, "LOCAL-only segments are never evicted");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn age_eviction_drops_backdated_both_segment() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let ns_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir),
            catalog,
            remote.to_str().unwrap(),
            StoragePolicy {
                max_age_seconds: Some(60),
                ..Default::default()
            },
        );
        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap(); // L1, BOTH
        let segs = ns.catalog.list_segments("iris.worker").unwrap();
        assert_eq!(segs.len(), 1);
        let base = basename(&segs[0].path);

        // Within window: a fresh maintain keeps it.
        ns.run_maintenance(false).await.unwrap();
        assert_eq!(ns.stats().segment_count, 1);

        // Backdate past the cutoff (now - 60s); maintain age-evicts it.
        ns.backdate_segment(&base, 1).unwrap();
        ns.run_maintenance(false).await.unwrap();
        assert_eq!(ns.stats().segment_count, 0, "aged-out segment dropped");
        // Remote archive preserved.
        assert_eq!(remote_files(&remote, "iris.worker").len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
