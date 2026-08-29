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

use arrow::array::RecordBatch;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use crate::errors::StatsError;
use crate::partition_policy::{PartitionedBatches, PhysicalPartitionPolicy, SegmentPartition};
use crate::proto::finelog::stats::{MigrationPhase, SourceLayout, TableMigrationStatus};
use crate::store::catalog::object_state_store::OBJECTS_PREFIX;
use crate::store::catalog::{Catalog, ObjectSegmentRecord, TableSpecStatus};
use crate::store::compaction::config::CompactionJob;
use crate::store::compaction::executor::{
    run_job_with_partition_policy, CompactionExecution, CompactionLayout, OutputPolicy,
};
use crate::store::compaction::staging::StagingDir;
use crate::store::table::controller::{file_sha256, TableController};
use crate::store::table::flush::partition_object_batch;
use crate::store::table::index_artifacts::publish_segment_artifacts;
use crate::store::table::segment_format::SegmentFormat;
use crate::store::table::segment_view::SegmentView;
use crate::store::table_state::{SegmentDescriptor, TableRevision};
use crate::store::types::{segment_to_row, LocalSegment, SegmentLocation, SegmentRow};

/// Source objects rewritten per maintenance tick while a transition is active.
const SEGMENTS_PER_TICK: usize = 4;

/// Directory fan-out for a rewrite's staged partition outputs. The uploaded
/// objects are content-addressed, so this only keeps one source's staged files
/// from colliding.
const PARTITION_DIRECTORIES: u32 = 64;

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
    /// The table's object-flush gate. A backfill rewrites the very sources a
    /// concurrent flush would commit against, so the two never overlap.
    pub flush_gate: &'a tokio::sync::Mutex<()>,
    pub max_merge_arrow_bytes: i64,
    /// How this table's transition is failing, carried across ticks so a
    /// permanently stuck transition becomes visible.
    pub blocked: &'a std::sync::Mutex<MigrationBlock>,
    /// Applied when an activation moves the table to a new version, so the
    /// runtime's cached policy and query view follow the commit.
    pub on_activated: &'a (dyn Fn(&TableSpecStatus) -> Result<(), StatsError> + Send + Sync),
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
    let status = migration.catalog.table_spec_status(migration.table).ok();
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
    let status = migration.catalog.table_spec_status(migration.table)?;
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
async fn activate(migration: &SpecMigration<'_>) -> Result<TableSpecStatus, StatsError> {
    let _visibility_guard = migration.query_visibility.write().await;
    // The in-memory query view swaps only after the activation revision is known
    // to be published, so queries never see a version whose state no reader can
    // recover.
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
    status: &TableSpecStatus,
    pending: &TableMigrationStatus,
) -> Result<bool, StatsError> {
    let _flush_guard = migration.flush_gate.lock().await;
    let table = migration.table;
    let from_version = pending.from_version.unwrap_or(0);
    let to_version = pending.to_version.unwrap_or(0);
    let fence_seq = pending.fence_seq.unwrap_or(-1);
    let object_records: HashMap<_, _> = migration
        .catalog
        .object_segments(table)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();
    let rows = migration.catalog.list_segments(table)?;
    let mut covered: HashSet<_> = object_records
        .values()
        .filter(|record| record.table_spec_version == to_version && record.migration_backfill)
        .filter_map(|record| record.migration_source_id.clone())
        .collect();
    let mut sources: Vec<_> = rows
        .iter()
        .filter(|row| row.max_seq <= fence_seq)
        .filter(|row| match object_records.get(&row.path) {
            None => from_version == 0 && row.location != SegmentLocation::Remote,
            Some(record) => record.table_spec_version == from_version,
        })
        .cloned()
        .collect();
    sources.sort_by_key(|row| row.min_seq);
    let target_layout = status
        .desired
        .as_ref()
        .and_then(|spec| spec.source_layout.as_option())
        .cloned();
    let mut processed = 0;
    let mut unexamined = false;
    for row in &sources {
        if processed >= SEGMENTS_PER_TICK {
            unexamined = true;
            break;
        }
        let record = object_records.get(&row.path);
        let localized = migration.controller.localize_source(row, record).await?;
        let source_id = source_identity(row, record, &localized).await?;
        if covered.contains(&source_id) {
            continue;
        }
        let staging = StagingDir::create(migration.table_dir)?;
        let outcome = rewrite_source(
            migration,
            &staging,
            row,
            &localized,
            target_layout.as_ref(),
            to_version,
            &source_id,
        )
        .await;
        drop(staging);
        outcome?;
        covered.insert(source_id);
        processed += 1;
    }

    migration.catalog.refresh_migration_rows_total(table)?;
    if unexamined {
        return Ok(true);
    }
    // A transition whose universe held no source at all never checkpointed, so
    // it is still in the phase registration left it in.
    let backfilled = migration.catalog.table_spec_status(table)?.phase;
    let verified = migration
        .controller
        .commit(|| {
            let verified = migration.catalog.update_migration_phase(
                table,
                backfilled,
                MigrationPhase::MIGRATION_PHASE_VERIFY,
            )?;
            Ok((TableRevision::new(verified.catalog_generation), verified))
        })
        .await?
        .output;
    debug_assert_eq!(verified.phase, MigrationPhase::MIGRATION_PHASE_VERIFY);
    activate(migration).await?;
    Ok(true)
}

/// Rewrite the segment `row`, localized at `localized`, into `target_layout` at
/// version `to_version`, and checkpoint the result against `source_id`.
///
/// On success the rewritten objects are committed and the source is recorded as
/// migrated; the migration will not revisit it. An error leaves the source
/// unmigrated and any uploaded outputs unreferenced.
async fn rewrite_source(
    migration: &SpecMigration<'_>,
    staging: &StagingDir,
    row: &SegmentRow,
    localized: &Path,
    target_layout: Option<&SourceLayout>,
    to_version: u64,
    source_id: &str,
) -> Result<(), StatsError> {
    let job = CompactionJob {
        inputs: vec![SegmentRow {
            path: localized.to_string_lossy().into_owned(),
            ..row.clone()
        }],
        output_level: row.level,
        output_min_seq: row.min_seq,
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
    let key_bounds = migration.segments.key_bounds(&row.path);
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
    .map_err(|error| StatsError::Internal(format!("migration rewrite task panicked: {error}")))??;

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
            created_at_ms: row.created_at_ms,
            artifacts: local,
            ..staged
        };
        migrated.push(SegmentDescriptor {
            row: segment_to_row(migration.table, &segment),
            source: stored.source,
            artifacts: references,
        });
    }
    let source_rows = row.row_count;
    migration
        .controller
        .commit(|| {
            let revision = migration.catalog.commit_migration_segments(
                &migrated,
                to_version,
                source_id,
                source_rows,
            )?;
            Ok((revision, ()))
        })
        .await?;
    Ok(())
}

/// Stable identity of one migration source, so a rewrite interrupted after its
/// objects upload but before its checkpoint commits is recognized and not
/// applied twice.
///
/// It binds the source's content to the exact rows it covers: an object-backed
/// source is already named by its content SHA-256, and a version-0 file is hashed
/// where it lies.
async fn source_identity(
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
        ) % PARTITION_DIRECTORIES;
        Path::new("partition").join(format!("{bucket:02}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
