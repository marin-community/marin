//! Automatic transition between two versions of a table specification.
//!
//! A registration that changes a table's physical layout records a pending
//! transition; maintenance drives it to completion here. The arms are the phases
//! the durable migration state names:
//!
//! - BACKFILL rewrites every source below the durable fence into the target
//!   layout, exactly once per source, checkpointing each by a deterministic
//!   source identity so an interrupted tick resumes without duplicating rows;
//! - VERIFY activates the new version in one state commit;
//! - OBSERVING heals a process failure between that commit and its publication,
//!   then retires the transition;
//! - RETIRED simply publishes whatever revision is still owed.
//!
//! Rows written after the fence are already in the target layout and are
//! referenced from both query views until activation. There is no separate
//! migration service and no activation RPC.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::RecordBatch;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use crate::errors::StatsError;
use crate::partition_policy::{PartitionedBatches, PhysicalPartitionPolicy, SegmentPartition};
use crate::proto::finelog::stats::{MigrationPhase, SourceLayout, TableMigrationStatus};
use crate::store::catalog::{
    Catalog, ObjectSegmentRecord, SpecLifecycle, MIGRATION_SOURCE_ID_SEPARATOR,
};
use crate::store::compaction::config::CompactionJob;
use crate::store::compaction::executor::{
    run_job_with_partition_policy, run_merge_thread, CompactionExecution, CompactionLayout,
    OutputPolicy,
};
use crate::store::compaction::staging::StagingDir;
use crate::store::object_store::OBJECTS_PREFIX;
use crate::store::table::controller::TableController;
use crate::store::table::flush::partition_object_batch;
use crate::store::table::index_artifacts::publish_segment_artifacts;
use crate::store::table::segment_format::SegmentFormat;
use crate::store::table::segment_view::SegmentView;
use crate::store::table_state::{SegmentDescriptor, TableRevision};
use crate::store::types::{segment_to_row, LocalSegment, SegmentLocation, SegmentRow};

/// Floor for the batch's compressed-input byte cap; the working cap is the
/// larger of this and a quarter of the merge's decoded-bytes budget, so batch
/// assembly does not localize far more than one job can consume.
const MIN_BATCH_SOURCE_BYTES: i64 = 64 << 20;

/// Directory fan-out for a rewrite's staged partition outputs. The uploaded
/// objects use opaque names, so this keeps one source's staged files
/// from colliding.
const PARTITION_DIRECTORIES: u32 = 64;

/// A single source can contain unbounded partition cardinality. Refuse a
/// rewrite before uploading anything when one atomic checkpoint would create
/// an impractically large output fanout.
const MAX_MIGRATION_OUTPUT_PARTITIONS: usize = 4_096;

/// Migration rewrites enter the first indexed level. They must not inherit a
/// source's depth: a batch can span old levels, and publishing it directly at
/// L2 or terminal L3 bypasses the normal leveling policy.
const MIGRATION_OUTPUT_LEVEL: i32 = 1;

/// Consecutive identical failures after which a transition is reported blocked.
///
/// A few repeats are ordinary — a contended commit, a transient object read —
/// and the maintenance cadence retries them within a minute or two. Beyond this
/// the same error is not going to clear itself.
const BLOCKED_FAILURE_THRESHOLD: u32 = 5;

/// How one table's specification transition is failing.
///
/// A transition that fails the same way every tick makes no progress while its
/// status keeps reporting an ordinary BACKFILL phase, so without this an
/// operator sees only a repeating warning. Once the same error has repeated
/// [`BLOCKED_FAILURE_THRESHOLD`] times the table reports itself blocked and the
/// failure is logged at ERROR; any tick that gets through clears it.
#[derive(Default)]
pub struct MigrationBlock {
    error: Option<String>,
    consecutive_failures: u32,
}

impl MigrationBlock {
    /// The repeated error this transition is stuck on, or `None` while it is
    /// still making progress or failing in varied ways.
    pub fn blocked_error(&self) -> Option<&str> {
        if self.consecutive_failures < BLOCKED_FAILURE_THRESHOLD {
            return None;
        }
        self.error.as_deref()
    }

    /// Count one failed tick and report how many times this exact error has now
    /// repeated. A different error starts the count over.
    fn record_failure(&mut self, error: &str) -> u32 {
        if self.error.as_deref() == Some(error) {
            self.consecutive_failures += 1;
        } else {
            self.error = Some(error.to_string());
            self.consecutive_failures = 1;
        }
        self.consecutive_failures
    }

    fn clear(&mut self) {
        self.error = None;
        self.consecutive_failures = 0;
    }
}

/// Everything a migration step reads and commits through.
pub struct SpecMigration<'a> {
    pub table: &'a str,
    pub table_dir: &'a Path,
    pub format: &'a SegmentFormat,
    pub index_config: crate::indices::SegmentIndexConfig,
    pub catalog: &'a Catalog,
    pub controller: &'a TableController,
    pub segments: &'a SegmentView,
    pub query_visibility: &'a Arc<RwLock<()>>,
    /// The table's object-flush gate. A backfill tick holds it only while
    /// assembling its batch from one consistent catalog snapshot; the rewrite
    /// and commit run outside it so concurrent writes keep acking.
    pub flush_gate: &'a tokio::sync::Mutex<()>,
    pub max_merge_arrow_bytes: i64,
    /// Most sources one backfill batch coalesces
    /// (`CompactionConfig::migration_batch_sources`).
    pub migration_batch_sources: usize,
    /// How this table's transition is failing, carried across ticks so a
    /// permanently stuck transition becomes visible.
    pub blocked: &'a std::sync::Mutex<MigrationBlock>,
    /// Applied when an activation moves the table to a new version, so the
    /// runtime's cached policy and query view follow the commit.
    pub on_activated: &'a (dyn Fn(&SpecLifecycle) -> Result<(), StatsError> + Send + Sync),
}

/// Advance an automatic transition by one tick, recording whether it is stuck.
///
/// Returns true while ordinary compaction and eviction must stay frozen to
/// preserve the migration source.
pub async fn advance(migration: &SpecMigration<'_>) -> Result<bool, StatsError> {
    match advance_phase(migration).await {
        Ok(owns_cycle) => {
            migration.blocked.lock().unwrap().clear();
            Ok(owns_cycle)
        }
        Err(error) => {
            report_failure(migration, &error);
            Err(error)
        }
    }
}

/// Escalate a transition that keeps failing the same way.
///
/// The failure itself is already reported by the maintenance scheduler at WARN.
/// This adds the one line an operator needs to tell a retry from a wedge: the
/// phase it is stuck in and how far its backfill got.
fn report_failure(migration: &SpecMigration<'_>, error: &StatsError) {
    let message = error.to_string();
    let failures = migration.blocked.lock().unwrap().record_failure(&message);
    if failures < BLOCKED_FAILURE_THRESHOLD {
        return;
    }
    let status = migration.catalog.spec_lifecycle(migration.table).ok();
    let progress = status.as_ref().and_then(|status| status.migration.as_ref());
    tracing::error!(
        namespace = %migration.table,
        phase = ?status.as_ref().map(|status| status.phase),
        rows_completed = progress.and_then(|progress| progress.rows_completed).unwrap_or(0),
        rows_total = progress.and_then(|progress| progress.rows_total).unwrap_or(0),
        consecutive_failures = failures,
        error = %message,
        "spec migration blocked"
    );
}

async fn advance_phase(migration: &SpecMigration<'_>) -> Result<bool, StatsError> {
    let status = migration.catalog.spec_lifecycle(migration.table)?;
    let Some(pending) = status.migration.clone() else {
        return Ok(false);
    };
    match status.phase {
        MigrationPhase::MIGRATION_PHASE_RETIRED => {
            migration.controller.publish_owed().await?;
            return Ok(false);
        }
        MigrationPhase::MIGRATION_PHASE_UNSPECIFIED => return Ok(false),
        MigrationPhase::MIGRATION_PHASE_OBSERVING => {
            // Heal a process failure after the local activation commit but
            // before HEAD publication or the query-view swap.
            migration.controller.publish_state().await?;
            (migration.on_activated)(&status)?;
            migration
                .controller
                .commit(|| {
                    let retired = migration
                        .catalog
                        .retire_observed_migration(migration.table)?;
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
            activate(migration).await?;
            return Ok(true);
        }
        MigrationPhase::MIGRATION_PHASE_ACTIVATED => {
            return Err(StatsError::Internal(format!(
                "table {:?} persisted unsupported transient ACTIVATED phase",
                migration.table
            )));
        }
        MigrationPhase::MIGRATION_PHASE_DUAL_WRITE | MigrationPhase::MIGRATION_PHASE_BACKFILL => {}
    }
    backfill(migration, &status, &pending).await
}

/// Publish the verified target version in one state commit and swap the query
/// view onto it.
async fn activate(migration: &SpecMigration<'_>) -> Result<SpecLifecycle, StatsError> {
    // The in-memory query view swaps only after the activation revision is known
    // to be published, so queries never see a version whose state no reader can
    // recover. The commit and its publication run before the visibility lock is
    // taken: publication is a network round trip, and queries keep planning
    // against the old view until the swap.
    let status = migration
        .controller
        .commit(|| {
            let status = migration
                .catalog
                .activate_desired_table_spec(migration.table)?;
            Ok((TableRevision::new(status.catalog_generation), status))
        })
        .await?
        .output;
    let _visibility_guard = migration.query_visibility.write().await;
    (migration.on_activated)(&status)?;
    tracing::info!(
        namespace = %migration.table,
        table_spec_version = status.active_version(),
        catalog_generation = status.catalog_generation,
        "activated migrated table specification"
    );
    Ok(status)
}

/// Rewrite the source version's remaining segments into the target layout.
///
/// The universe is the pre-fence segments of the source version whose bytes the
/// table can still read: object-backed sources by reference, legacy sources
/// while a local copy exists. A legacy segment the catalog reports as `REMOTE`
/// lives only in the legacy archive, which a migration never reads, rewrites, or
/// moves, so it is not a source — including one already evicted at registration
/// and one evicted between registration and the tick that would have reached it.
/// Skipping it costs the migrated table nothing that was queryable: an archived
/// segment is outside the live query view either way, and its bytes stay in the
/// archive for history queries.
///
/// The backfill is finished when a tick has examined every source and found
/// each one already rewritten, which is decided by the sources themselves
/// rather than by a row count: the universe shrinks under it, so a total frozen
/// at registration would name rows no source can supply and never be reached.
/// Checkpoints stay durable and exact: a source is rewritten once, identified
/// by content, so an interrupted tick resumes without re-rewriting or
/// double-counting, and each tick restates the reported row total against the
/// universe that is actually left.
async fn backfill(
    migration: &SpecMigration<'_>,
    status: &SpecLifecycle,
    pending: &TableMigrationStatus,
) -> Result<bool, StatsError> {
    let table = migration.table;
    let from_version = pending.from_version.unwrap_or(0);
    let to_version = pending.to_version.unwrap_or(0);
    let fence_seq = pending.fence_seq.unwrap_or(-1);
    let target_layout = status
        .desired
        .as_ref()
        .and_then(|spec| spec.source_layout.as_option())
        .cloned();
    let batch_byte_cap = MIN_BATCH_SOURCE_BYTES.max(migration.max_merge_arrow_bytes / 4);
    // Only the catalog snapshot holds the flush gate, so the tick reads one
    // consistent view of segments and checkpoints. Everything after — identity
    // assembly, the rewrite, the commit — runs outside it: every candidate
    // source is fence-frozen and immutable, a concurrent flush commits only
    // post-fence rows at the target version, and both commits serialize in the
    // controller. A WriteRows ack waits on this very gate, so any expensive
    // work held under it stalls every write to the table.
    let snapshot_started = Instant::now();
    let (object_records, sources, covered) = {
        let _flush_guard = migration.flush_gate.lock().await;
        let object_records: HashMap<_, _> = migration
            .catalog
            .object_segments(table)?
            .into_iter()
            .map(|record| (record.path.clone(), record))
            .collect();
        let rows = migration.catalog.list_segments(table)?;
        let covered: HashSet<_> = object_records
            .values()
            .filter(|record| record.table_spec_version == to_version && record.migration_backfill)
            .filter_map(|record| record.migration_source_id.as_deref())
            .flat_map(|ids| ids.split(MIGRATION_SOURCE_ID_SEPARATOR).map(str::to_string))
            .collect();
        let sources: Vec<_> = rows
            .iter()
            .filter(|row| row.max_seq <= fence_seq)
            .filter(|row| match object_records.get(&row.path) {
                None => from_version == 0 && row.location != SegmentLocation::Remote,
                Some(record) => record.table_spec_version == from_version,
            })
            .cloned()
            .collect();
        (object_records, sources, covered)
    };
    // TODO(#8909): Remove after 2026-09-18 once every deployed table is
    // RETIRED and no older server can publish the pre-metadata checkpoint
    // format. Without it, a mixed-version migration could rewrite an already
    // checkpointed source.
    if covered.iter().any(|identity| {
        identity.len() == 64 && identity.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        return Err(StatsError::SchemaConflict(format!(
            "migration for {table:?} has SHA-based checkpoints; complete or abort it with the prior server version before upgrading"
        )));
    }
    let partitioned_target = target_layout
        .as_ref()
        .and_then(|layout| layout.partition.as_option())
        .is_some_and(|partition| !partition.fields.is_empty());
    let source_groups = migration_source_groups(sources, partitioned_target);
    let snapshot_ms = snapshot_started.elapsed().as_millis() as u64;
    let select_started = Instant::now();
    let mut batch: Vec<BatchSource> = Vec::new();
    let mut batch_bytes: i64 = 0;
    let mut unexamined = false;
    'groups: for (group_index, group) in source_groups.iter().enumerate() {
        for row in group {
            if !batch.is_empty()
                && (batch.len() >= migration.migration_batch_sources
                    || batch_bytes >= batch_byte_cap)
            {
                unexamined = true;
                break 'groups;
            }
            let record = object_records.get(&row.path);
            let source_id = source_identity(row, record)?;
            if covered.contains(&source_id) {
                continue;
            }
            let Some(localized) = migration.controller.localize_source(row, record).await? else {
                // The row's bytes exist neither locally nor in the archive: no
                // reader can serve them and no source can supply them. Drop the
                // row so the restated universe stops owing rows nothing holds.
                tracing::warn!(
                    namespace = %table,
                    path = %row.path,
                    rows = row.row_count,
                    "dropping a legacy migration source whose bytes are unrecoverable"
                );
                migration.catalog.remove_segment(table, &row.path)?;
                continue;
            };
            batch_bytes += row.byte_size;
            batch.push(BatchSource {
                row: row.clone(),
                localized,
                source_id,
            });
        }
        if !batch.is_empty() {
            // An unpartitioned rewrite never crosses a real sequence gap. If
            // this component is only partially consumed or older components
            // remain, the next tick still owns backfill work.
            unexamined |= group_index + 1 < source_groups.len();
            break;
        }
    }
    if !batch.is_empty() {
        tracing::info!(
            namespace = %table,
            batch_sources = batch.len(),
            batch_bytes,
            universe = source_groups.iter().map(Vec::len).sum::<usize>(),
            snapshot_ms,
            select_ms = select_started.elapsed().as_millis() as u64,
            "migration backfill batch selected"
        );
        let staging = StagingDir::create(migration.table_dir)?;
        let outcome = rewrite_batch(
            migration,
            &staging,
            &batch,
            target_layout.as_ref(),
            to_version,
        )
        .await;
        drop(staging);
        let consumed = outcome?;
        if consumed < batch.len() {
            // The merge's decoded-bytes budget ended the job early; the
            // remainder is still unmigrated.
            unexamined = true;
        }
    }

    migration.catalog.refresh_migration_rows_total(table)?;
    if unexamined {
        return Ok(true);
    }
    // A transition whose universe held no source at all never checkpointed, so
    // it is still in the phase registration left it in.
    let backfilled = migration.catalog.spec_lifecycle(table)?.phase;
    migration
        .controller
        .commit(|| {
            let verified = migration.catalog.update_migration_phase(
                table,
                backfilled,
                MigrationPhase::MIGRATION_PHASE_VERIFY,
            )?;
            Ok((TableRevision::new(verified.catalog_generation), verified))
        })
        .await?;
    activate(migration).await?;
    Ok(true)
}

/// One localized, metadata-identified, uncovered source awaiting rewrite.
struct BatchSource {
    row: SegmentRow,
    localized: PathBuf,
    source_id: String,
}

/// Deterministic source groups for one migration target.
///
/// A partitioned target defines sparse stream membership from row values, so
/// global sequence gaps do not constrain a batch. An unpartitioned target has
/// no such identity: keep each overlap-or-adjacency component separate so one
/// output footer never invents coverage across a real gap. Components are
/// processed newest first, while rows inside one component remain in canonical
/// sequence/path order.
fn migration_source_groups(
    mut sources: Vec<SegmentRow>,
    partitioned_target: bool,
) -> Vec<Vec<SegmentRow>> {
    if partitioned_target {
        sources.sort_by(|left, right| {
            (right.min_seq, right.max_seq, &right.path).cmp(&(
                left.min_seq,
                left.max_seq,
                &left.path,
            ))
        });
        return (!sources.is_empty())
            .then_some(sources)
            .into_iter()
            .collect();
    }

    sources.sort_by(|left, right| {
        (left.min_seq, left.max_seq, &left.path).cmp(&(right.min_seq, right.max_seq, &right.path))
    });
    let mut groups: Vec<Vec<SegmentRow>> = Vec::new();
    let mut component_max = i64::MIN;
    for source in sources {
        let starts_new_component = groups
            .last()
            .is_some_and(|_| source.min_seq > component_max.saturating_add(1));
        if starts_new_component || groups.is_empty() {
            groups.push(Vec::new());
            component_max = source.max_seq;
        } else {
            component_max = component_max.max(source.max_seq);
        }
        groups
            .last_mut()
            .expect("migration source group was just created")
            .push(source);
    }
    groups.reverse();
    groups
}

/// Rewrite `batch` into `target_layout` at version `to_version` as one
/// multi-input job, and checkpoint every consumed source in one commit.
///
/// Returns how many of the batch's sources the job consumed — the executor
/// takes the longest input prefix fitting its decoded-bytes budget and leaves
/// the rest unmigrated. On success the rewritten objects are committed and the
/// consumed sources are recorded as migrated; an error leaves every source
/// unmigrated and any uploaded outputs unreferenced.
async fn rewrite_batch(
    migration: &SpecMigration<'_>,
    staging: &StagingDir,
    batch: &[BatchSource],
    target_layout: Option<&SourceLayout>,
    to_version: u64,
) -> Result<usize, StatsError> {
    let by_localized: HashMap<String, &BatchSource> = batch
        .iter()
        .map(|source| (source.localized.to_string_lossy().into_owned(), source))
        .collect();
    let job = CompactionJob {
        inputs: batch
            .iter()
            .map(|source| SegmentRow {
                path: source.localized.to_string_lossy().into_owned(),
                ..source.row.clone()
            })
            .collect(),
        output_level: MIGRATION_OUTPUT_LEVEL,
        output_min_seq: batch
            .iter()
            .map(|source| source.row.min_seq)
            .min()
            .unwrap_or(0),
    };
    let index_config = migration.index_config.clone();
    let arrow_schema = Arc::clone(migration.format.arrow_schema());
    let sort_columns = target_layout
        .map(|layout| layout.sort_columns.clone())
        .filter(|columns| !columns.is_empty())
        .unwrap_or_else(|| migration.format.sort_columns().to_vec());
    let key_column = migration.format.key_column().to_string();
    let max_row_group_rows = target_layout
        .and_then(|layout| layout.max_row_group_rows)
        .map(|rows| rows as usize)
        .unwrap_or(migration.format.max_row_group_rows());
    let max_merge_arrow_bytes = migration.max_merge_arrow_bytes;
    let partitions = target_layout.and_then(TargetPartitions::new);
    let bounds_by_path: HashMap<String, (Option<String>, Option<String>)> = batch
        .iter()
        .map(|source| {
            (
                source.localized.to_string_lossy().into_owned(),
                migration.segments.key_bounds(&source.row.path),
            )
        })
        .collect();
    let staging_dir = staging.path().to_path_buf();
    // One dedicated thread at normal priority: the single thread caps the
    // rewrite at one core, which is the pacing, while a fair share of that
    // core keeps a many-hundred-batch backfill finishing in hours even when
    // queries saturate the box (a deprioritized thread would starve outright).
    let merge_started = Instant::now();
    let swap = run_merge_thread("finelog-rewrite", 0, move || {
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
            move |path| bounds_by_path.get(path).cloned().unwrap_or((None, None)),
        )
    })
    .await??;
    let merge_ms = merge_started.elapsed().as_millis() as u64;

    let consumed: Vec<&BatchSource> = swap
        .removed
        .iter()
        .map(|path| {
            by_localized.get(path.as_str()).copied().ok_or_else(|| {
                StatsError::Internal(format!("migration job consumed unknown input {path:?}"))
            })
        })
        .collect::<Result<_, _>>()?;
    let consumed_rows: i64 = consumed.iter().map(|source| source.row.row_count).sum();
    let rewritten_rows: i64 = swap.added.iter().map(|segment| segment.row_count).sum();
    if rewritten_rows != consumed_rows {
        return Err(StatsError::Internal(format!(
            "migration batch of {} sources rewrote {consumed_rows} rows as {rewritten_rows}",
            consumed.len()
        )));
    }
    validate_output_partition_count(&swap.added)?;
    let created_at_ms = consumed
        .iter()
        .map(|source| source.row.created_at_ms)
        .max()
        .unwrap_or(0);
    let upload_started = Instant::now();
    let mut migrated = Vec::with_capacity(swap.added.len());
    for staged in swap.added {
        let staged_path = PathBuf::from(&staged.path);
        let stored = migration
            .controller
            .write_staged_object(OBJECTS_PREFIX, "parquet", &staged_path)
            .await?;
        let (references, local) =
            publish_segment_artifacts(migration.controller, migration.table, &staged_path, &stored)
                .await?;
        let segment = LocalSegment {
            path: stored.path.to_string_lossy().into_owned(),
            size_bytes: stored.byte_size,
            location: SegmentLocation::Both,
            created_at_ms,
            artifacts: local,
            ..staged
        };
        migrated.push(SegmentDescriptor {
            row: segment_to_row(migration.table, &segment),
            source: stored.source,
            artifacts: references,
        });
    }
    let upload_ms = upload_started.elapsed().as_millis() as u64;
    let source_ids: Vec<String> = consumed
        .iter()
        .map(|source| source.source_id.clone())
        .collect();
    let joined_ids = source_ids.join(&MIGRATION_SOURCE_ID_SEPARATOR.to_string());
    let commit_started = Instant::now();
    let outputs = migrated.len();
    migration
        .controller
        .commit(|| {
            let revision = migration.catalog.commit_migration_segments(
                &migrated,
                to_version,
                &joined_ids,
                consumed_rows,
            )?;
            Ok((revision, ()))
        })
        .await?;
    tracing::info!(
        namespace = %migration.table,
        sources = consumed.len(),
        rows = consumed_rows,
        outputs,
        merge_ms,
        upload_ms,
        commit_ms = commit_started.elapsed().as_millis() as u64,
        "migration backfill batch committed"
    );
    Ok(consumed.len())
}

fn validate_output_partition_count(outputs: &[LocalSegment]) -> Result<(), StatsError> {
    let partitions = outputs
        .iter()
        .map(|segment| segment.partition.as_ref())
        .collect::<BTreeSet<_>>()
        .len();
    if partitions > MAX_MIGRATION_OUTPUT_PARTITIONS {
        return Err(StatsError::ResourceExhausted(format!(
            "migration batch would emit {partitions} partitions; limit is {MAX_MIGRATION_OUTPUT_PARTITIONS}"
        )));
    }
    Ok(())
}

/// Stable metadata identity of one immutable migration source. This lets a
/// restart recognize a committed rewrite without reading the source bytes.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MigrationSourceIdentity<'a> {
    format: &'static str,
    object_id: &'a str,
    provider_version: Option<&'a str>,
    etag: Option<&'a str>,
    byte_size: i64,
    min_seq: i64,
    max_seq: i64,
    row_count: i64,
}

fn source_identity(
    row: &SegmentRow,
    object_record: Option<&ObjectSegmentRecord>,
) -> Result<String, StatsError> {
    let source = object_record.map(|record| &record.source);
    serde_json::to_string(&MigrationSourceIdentity {
        format: "metadata-v1",
        object_id: source
            .and_then(|source| source.object_id.as_deref())
            .unwrap_or(&row.path),
        provider_version: source.and_then(|source| source.provider_version.as_deref()),
        etag: source.and_then(|source| source.etag.as_deref()),
        byte_size: row.byte_size,
        min_seq: row.min_seq,
        max_seq: row.max_seq,
        row_count: row.row_count,
    })
    .map_err(|error| StatsError::Internal(format!("encode migration source identity: {error}")))
}

/// The physical partitioning a target source layout declares.
///
/// The compaction executor asks for partitions through this trait, so a
/// migration into a changed partition spec splits its rewritten rows exactly the
/// way the ingest path splits a freshly sealed buffer.
#[derive(Debug)]
struct TargetPartitions {
    spec_id: u32,
    layout: SourceLayout,
}

impl TargetPartitions {
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

impl PhysicalPartitionPolicy for TargetPartitions {
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
        // Query pruning uses the table's own registered policy. A migration
        // rewrite never prunes, so this policy claims nothing.
        None
    }

    fn segment_directory(&self, partition: &SegmentPartition) -> PathBuf {
        let mut digest = Sha256::new();
        digest.update(partition.spec_id.to_be_bytes());
        for (field, value) in &partition.values {
            digest.update((field.len() as u64).to_be_bytes());
            digest.update(field.as_bytes());
            digest.update((value.len() as u64).to_be_bytes());
            digest.update(value.as_bytes());
        }
        let digest = digest.finalize();
        let bucket =
            u32::from_be_bytes(digest[..4].try_into().expect("sha256 prefix is four bytes"))
                % PARTITION_DIRECTORIES;
        Path::new("partition")
            .join(format!("{bucket:02}"))
            .join(crate::hex::encode(&digest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(path: &str, min_seq: i64, max_seq: i64) -> SegmentRow {
        SegmentRow {
            namespace: "events".to_string(),
            path: path.to_string(),
            level: 2,
            min_seq,
            max_seq,
            row_count: 1,
            byte_size: 1,
            created_at_ms: 0,
            min_key_value: None,
            max_key_value: None,
            partition: None,
            location: SegmentLocation::Both,
        }
    }

    /// The point of the counter is to separate a transition that retries from
    /// one that cannot proceed, so only an unchanging error escalates and any
    /// progress clears the verdict.
    #[test]
    fn only_an_unchanging_error_reports_a_transition_blocked() {
        let mut block = MigrationBlock::default();
        for _ in 0..BLOCKED_FAILURE_THRESHOLD - 1 {
            block.record_failure("object read timed out");
            assert_eq!(block.blocked_error(), None);
        }
        block.record_failure("object read timed out");
        assert_eq!(block.blocked_error(), Some("object read timed out"));

        block.record_failure("a different failure");
        assert_eq!(
            block.blocked_error(),
            None,
            "a new error is a fresh problem, not a continuation"
        );

        for _ in 1..BLOCKED_FAILURE_THRESHOLD {
            block.record_failure("a different failure");
        }
        assert_eq!(block.blocked_error(), Some("a different failure"));

        block.clear();
        assert_eq!(block.blocked_error(), None);
    }

    #[test]
    fn unpartitioned_migration_never_batches_across_a_real_sequence_gap() {
        let groups = migration_source_groups(
            vec![
                source("nested", 4, 6),
                source("new", 20, 25),
                source("bridge", 1, 10),
                source("adjacent", 11, 12),
                source("new-overlap", 24, 30),
            ],
            false,
        );

        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups[0]
                .iter()
                .map(|row| row.path.as_str())
                .collect::<Vec<_>>(),
            vec!["new", "new-overlap"]
        );
        assert_eq!(
            groups[1]
                .iter()
                .map(|row| row.path.as_str())
                .collect::<Vec<_>>(),
            vec!["bridge", "nested", "adjacent"]
        );
    }

    #[test]
    fn unpartitioned_migration_handles_the_maximum_sequence_without_overflow() {
        let groups = migration_source_groups(
            vec![
                source("last", i64::MAX, i64::MAX),
                source("penultimate", i64::MAX - 1, i64::MAX - 1),
            ],
            false,
        );
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn partitioned_migration_keeps_sparse_sources_in_one_newest_first_group() {
        let groups = migration_source_groups(
            vec![source("old", -1_000, -900), source("new", 1_000, 1_100)],
            true,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0][0].path, "new");
        assert_eq!(groups[0][1].path, "old");
    }

    #[test]
    fn partitioned_staging_names_distinguish_equal_minimum_sequences() {
        let policy = TargetPartitions {
            spec_id: 1,
            layout: SourceLayout::default(),
        };
        let mut by_bucket: BTreeMap<PathBuf, (SegmentPartition, PathBuf)> = BTreeMap::new();
        for index in 0..=PARTITION_DIRECTORIES {
            let partition = SegmentPartition {
                spec_id: 1,
                values: BTreeMap::from([("run_id".to_string(), format!("run-{index}"))]),
            };
            let directory = policy.segment_directory(&partition);
            let bucket = directory
                .parent()
                .expect("partition path has a placement bucket")
                .to_path_buf();
            if let Some((other_partition, other_directory)) = by_bucket.get(&bucket) {
                assert_ne!(partition, *other_partition);
                assert_ne!(directory, *other_directory);
                assert_eq!(directory.parent(), other_directory.parent());
                return;
            }
            by_bucket.insert(bucket, (partition, directory));
        }
        panic!("65 partitions must contain a collision across 64 placement buckets");
    }

    #[test]
    fn migration_rejects_excessive_partition_fanout() {
        let outputs = (0..=MAX_MIGRATION_OUTPUT_PARTITIONS)
            .map(|index| LocalSegment {
                path: format!("/{index}.parquet"),
                size_bytes: 1,
                level: MIGRATION_OUTPUT_LEVEL,
                min_seq: index as i64,
                max_seq: index as i64,
                row_count: 1,
                created_at_ms: 0,
                min_key_value: None,
                max_key_value: None,
                partition: Some(SegmentPartition {
                    spec_id: 1,
                    values: BTreeMap::from([("run_id".to_string(), index.to_string())]),
                }),
                location: SegmentLocation::Local,
                artifacts: Default::default(),
            })
            .collect::<Vec<_>>();

        assert!(matches!(
            validate_output_partition_count(&outputs),
            Err(StatsError::ResourceExhausted(_))
        ));
        assert!(
            validate_output_partition_count(&outputs[..MAX_MIGRATION_OUTPUT_PARTITIONS]).is_ok()
        );
    }
}
