//! Converging legacy segments onto the current physical layout.
//!
//! Three separate pieces of work share this module because they all exist to
//! retire a placement or encoding an older writer produced:
//!
//! - [`advance_partitioning`] rebuilds L0 segments written before L0 was defined
//!   as policy-free and repartitions or relocates L1+ segments whose partition
//!   spec has moved;
//! - [`advance_archive_placement`] copies an archived segment to the key its
//!   current partition implies;
//! - [`rewrite_stale_encodings`] re-encodes segments whose row-group layout
//!   predates the current writer policy.
//!
//! All three are bounded and lowest-priority: they must never starve compaction,
//! sync, or eviction. They retire with legacy tables, which are the only tables
//! whose placement is a local directory rather than an object identity.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use crate::errors::StatsError;
use crate::maintenance::MaintenanceLimits;
use crate::partition_policy::{segment_path, PhysicalPartitionPolicy, SegmentPartition};
use crate::store::compaction::config::CompactionJob;
use crate::store::compaction::executor::PlannedSwap;
use crate::store::compaction::local_driver::{commit_swap, run_job, LocalCompaction};
use crate::store::compaction::planner::build_job;
use crate::store::object_store::{ObjectId, ObjectStore};
use crate::store::segment::{segment_layout_is_current, stage_rewritten_segment};
use crate::store::types::{
    basename, segment_relative_key, segment_to_row, LocalSegment, SegmentLocation, SegmentRow,
};

/// Per-maintenance budget for converging legacy physical placement.
///
/// Each rewrite already in flight may overrun this budget. The bound controls
/// how many additional jobs start before ordinary compaction and remote sync get
/// their turn.
const PARTITIONING_BUDGET: Duration = Duration::from_secs(3);
const PARTITIONING_CONCURRENCY: usize = 2;
const PARTITIONING_WORKER_COMPRESSED_BYTES: i64 = 32 * 1024 * 1024;

/// Process-wide permit for the encoding rewrite, so only one table re-encodes at
/// a time.
///
/// A per-table budget alone let a store with dozens of tables spend dozens of
/// budgets at once. On the marin hub that saturated the box: a 10 min telemetry
/// query that normally answers in 0.18 s took 15 s, and `count(*)` went from
/// 0.3 s to 17 s, because re-encoding pushes tens of GiB through the page cache
/// the queries are served from and invalidates the parquet metadata cache entry
/// for every segment it touches.
///
/// A table that cannot take the permit skips the step for this tick rather than
/// queueing behind it, which would stall the rest of its maintenance.
static REWRITE_SLOT: Mutex<()> = Mutex::new(());

/// Everything the layout work reads and commits through.
pub struct LocalLayout<'a> {
    pub compaction: LocalCompaction<'a>,
    pub limits: &'a MaintenanceLimits,
    pub tracker: &'a LayoutTracker,
    /// The table's stop latch, so a long budget ends promptly at shutdown.
    pub stopped: &'a AtomicBool,
    /// The legacy archive, when one is configured.
    pub remote: Option<Arc<dyn ObjectStore>>,
}

impl LocalLayout<'_> {
    fn table(&self) -> &str {
        self.compaction.table
    }

    fn dir(&self) -> &Path {
        self.compaction.table_dir
    }

    fn policy(&self) -> Option<&'static dyn PhysicalPartitionPolicy> {
        self.compaction.partition_policy
    }

    fn running(&self) -> bool {
        !self.stopped.load(Ordering::SeqCst)
    }
}

/// Segments already confirmed to carry the current physical layout.
///
/// Determining staleness means parsing a segment's whole footer, so without this
/// the rewrite pass would re-read every footer in the table on every tick —
/// hundreds of MiB of thrift for a large table, forever, long after there is
/// nothing left to rewrite. A path's layout only ever changes because this pass
/// changed it.
#[derive(Default)]
pub struct LayoutTracker {
    current: Mutex<HashSet<String>>,
}

impl LayoutTracker {
    pub fn is_current(&self, path: &str) -> bool {
        if self.current.lock().unwrap().contains(path) {
            return true;
        }
        if !segment_layout_is_current(Path::new(path)) {
            return false;
        }
        self.current.lock().unwrap().insert(path.to_string());
        true
    }

    fn record(&self, path: &str) {
        self.current.lock().unwrap().insert(path.to_string());
    }

    fn retain_live(&self, live: &HashSet<&str>) {
        self.current
            .lock()
            .unwrap()
            .retain(|path| live.contains(path.as_str()));
    }
}

/// How much placement work one table still owes.
#[derive(Clone, Copy, Default)]
pub struct PendingPlacement {
    migration_l0: usize,
    stale_partitions: usize,
    misplaced_local: usize,
}

impl PendingPlacement {
    pub fn any(self) -> bool {
        self.migration_l0 > 0 || self.stale_partitions > 0 || self.misplaced_local > 0
    }
}

/// Advance local placement work and report whether the table still owes some.
pub fn advance_partitioning(layout: &LocalLayout<'_>) -> Result<bool, StatsError> {
    let permit = match layout.limits.layout_migration().try_lock() {
        Ok(permit) => permit,
        Err(TryLockError::WouldBlock) => return Ok(pending_placement(layout).any()),
        Err(TryLockError::Poisoned(_)) => {
            return Err(StatsError::Internal(
                "physical layout migration permit is poisoned".to_string(),
            ));
        }
    };
    let started = Instant::now();
    let mut migrated = 0;
    while layout.running() && started.elapsed() < PARTITIONING_BUDGET {
        let rebuilt = rebuild_l0_wave(layout)?;
        if rebuilt > 0 {
            migrated += rebuilt;
            continue;
        }
        if repartition_one_segment(layout)? {
            migrated += 1;
            continue;
        }
        break;
    }
    drop(permit);
    let pending = pending_placement(layout);
    if migrated > 0 {
        tracing::info!(
            namespace = %layout.table(),
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

/// Whether the table still owes placement work, without doing any.
pub fn partitioning_is_pending(layout: &LocalLayout<'_>) -> bool {
    pending_placement(layout).any()
}

/// Select up to `limit` independent legacy L0 rebuild jobs.
fn l0_rebuild_jobs(layout: &LocalLayout<'_>, limit: usize) -> Vec<CompactionJob> {
    if layout.policy().is_none() {
        return Vec::new();
    }
    let mut jobs = Vec::new();
    let mut inputs: Vec<SegmentRow> = Vec::new();
    let mut compressed_bytes: i64 = 0;
    // Coalesce compressed inputs before repartitioning. One output set per
    // migration source would turn the existing backlog into hundreds of
    // thousands of tiny L1s.
    for segment in layout
        .compaction
        .segments
        .segments()
        .iter()
        .filter(|segment| l0_needs_rebuild(segment))
    {
        if !inputs.is_empty()
            && compressed_bytes.saturating_add(segment.size_bytes)
                > PARTITIONING_WORKER_COMPRESSED_BYTES
        {
            jobs.push(build_job(inputs.iter().collect(), 1));
            inputs.clear();
            compressed_bytes = 0;
            if jobs.len() >= limit {
                break;
            }
        }
        let mut row = segment_to_row(layout.table(), segment);
        // An unpartitioned job forces the executor through its sort and
        // partition path. This also repairs the legacy version that stamped an
        // L0 footer before L0 was defined as policy-free.
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
fn rebuild_l0_wave(layout: &LocalLayout<'_>) -> Result<usize, StatsError> {
    let jobs = l0_rebuild_jobs(layout, PARTITIONING_CONCURRENCY);
    if jobs.is_empty() {
        return Ok(0);
    }
    let compaction = &layout.compaction;
    let results = std::thread::scope(|scope| {
        jobs.into_iter()
            .map(|job| scope.spawn(move || run_job(compaction, &job)))
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
fn repartition_one_segment(layout: &LocalLayout<'_>) -> Result<bool, StatsError> {
    let Some(policy) = layout.policy() else {
        return Ok(false);
    };
    let stale = layout
        .compaction
        .segments
        .find(|segment| partition_is_stale(segment, policy))
        .map(|segment| {
            let mut row = segment_to_row(layout.table(), &segment);
            row.partition = None;
            row
        });
    if let Some(input) = stale {
        let output_level = input.level;
        let output_min_seq = input.min_seq;
        run_job(
            &layout.compaction,
            &CompactionJob {
                inputs: vec![input],
                output_level,
                output_min_seq,
            },
        )?;
        return Ok(true);
    }

    let relocation = layout.compaction.segments.find_map(|segment| {
        current_layout_destination(
            layout.dir(),
            &segment.path,
            segment.level,
            segment.partition.as_ref(),
            policy,
        )
        .map(|destination| (segment.clone(), destination))
    });
    let Some((segment, destination)) = relocation else {
        return Ok(false);
    };
    let mut moved = segment.clone();
    moved.path = destination.to_string_lossy().into_owned();
    // The durable copy, if any, still has the old flat object key. Mark the moved
    // row LOCAL so sync uploads the new key before orphan cleanup can delete the
    // old one.
    moved.location = SegmentLocation::Local;
    commit_swap(
        &layout.compaction,
        PlannedSwap {
            removed: vec![segment.path.clone()],
            added: vec![moved],
            unlink_removed: false,
            bump_rename: Some((PathBuf::from(segment.path), destination)),
            input_arrow_bytes: 0,
        },
    )?;
    Ok(true)
}

fn pending_placement(layout: &LocalLayout<'_>) -> PendingPlacement {
    let Some(policy) = layout.policy() else {
        return PendingPlacement::default();
    };
    let mut pending = PendingPlacement::default();
    for segment in layout.compaction.segments.segments() {
        if l0_needs_rebuild(&segment) {
            pending.migration_l0 += 1;
            continue;
        }
        if segment.level == 0 {
            continue;
        }
        if partition_is_stale(&segment, policy) {
            pending.stale_partitions += 1;
            continue;
        }
        pending.misplaced_local += usize::from(
            current_layout_destination(
                layout.dir(),
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

fn l0_needs_rebuild(segment: &LocalSegment) -> bool {
    segment.level == 0 && (segment.min_seq < 0 || segment.partition.is_some())
}

fn partition_is_stale(segment: &LocalSegment, policy: &dyn PhysicalPartitionPolicy) -> bool {
    segment.level >= 1
        && segment
            .partition
            .as_ref()
            .is_none_or(|partition| !policy.is_current_partition(partition))
}

/// Where a segment belongs under the current policy, or `None` when it is
/// already there.
fn current_layout_destination(
    dir: &Path,
    path: &str,
    level: i32,
    partition: Option<&SegmentPartition>,
    policy: &dyn PhysicalPartitionPolicy,
) -> Option<PathBuf> {
    let partition = partition?;
    if level < 1 || !policy.is_current_partition(partition) {
        return None;
    }
    let filename = Path::new(path).file_name()?.to_str()?;
    let destination = segment_path(dir, filename, level, Some(partition), Some(policy));
    (Path::new(path) != destination).then_some(destination)
}

/// Relocate archived segments to their current physical key, bounded by the same
/// budget as local placement.
///
/// Each copy is server-side and crash-safe; the budget prevents a cold archive
/// backlog from monopolizing the maintenance cycle.
pub async fn advance_archive_placement(layout: &LocalLayout<'_>) -> Result<(), StatsError> {
    let started = Instant::now();
    let mut migrated = 0;
    while layout.running()
        && started.elapsed() < PARTITIONING_BUDGET
        && relocate_one_archived_segment(layout).await?
    {
        migrated += 1;
    }
    if migrated > 0 {
        tracing::info!(
            namespace = %layout.table(),
            segments = migrated,
            remaining = archive_placement_candidates(layout)?.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "remote physical layout migration advanced"
        );
    }
    Ok(())
}

async fn relocate_one_archived_segment(layout: &LocalLayout<'_>) -> Result<bool, StatsError> {
    let Some(remote) = layout.remote.as_ref() else {
        return Ok(false);
    };
    let table = layout.table();
    let Some((row, destination)) = archive_placement_candidates(layout)?.into_iter().next() else {
        return Ok(false);
    };
    let source_key = segment_relative_key(layout.dir(), &row.path).ok_or_else(|| {
        StatsError::Internal(format!(
            "remote layout source is outside table directory: {}",
            row.path
        ))
    })?;
    let destination_path = destination.to_string_lossy().into_owned();
    let destination_key =
        segment_relative_key(layout.dir(), &destination_path).ok_or_else(|| {
            StatsError::Internal(format!(
                "remote layout destination is outside table directory: {destination_path}"
            ))
        })?;

    let source_id = ObjectId::table(table, &source_key)?;
    let destination_id = ObjectId::table(table, &destination_key)?;
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
        let _write_guard = layout.compaction.query_visibility.write().await;
        layout.compaction.catalog.replace_segments(
            table,
            std::slice::from_ref(&row.path),
            &[moved],
        )?;
    }
    if let Err(error) = remote.delete(&source_id).await {
        tracing::warn!(namespace = %table, key = %source_key, %error, "legacy object delete failed");
    }
    tracing::info!(
        namespace = %table,
        from = %source_key,
        to = %destination_key,
        rows = row.row_count,
        "remote physical layout segment relocated"
    );
    Ok(true)
}

fn archive_placement_candidates(
    layout: &LocalLayout<'_>,
) -> Result<Vec<(SegmentRow, PathBuf)>, StatsError> {
    let Some(policy) = layout.policy() else {
        return Ok(Vec::new());
    };
    Ok(layout
        .compaction
        .catalog
        .list_segments_min_level(layout.table(), 1)?
        .into_iter()
        .filter(|row| row.location == SegmentLocation::Remote)
        .filter_map(|row| {
            current_layout_destination(
                layout.dir(),
                &row.path,
                row.level,
                row.partition.as_ref(),
                policy,
            )
            .map(|destination| (row, destination))
        })
        .collect())
}

/// Re-encode segments whose physical layout predates the current writer policy,
/// oldest first, for up to `budget` of wall clock. Returns how many were
/// rewritten.
///
/// Oldest first because the view is age-ordered and a leveled store keeps nearly
/// all of its bytes in the oldest, terminal-level segments — going the other way
/// spends the first hour rewriting small recent segments while the footer this
/// exists to shrink stays untouched.
///
/// Costs no remote bandwidth: the rewrite keeps the filename, and the sync step
/// only uploads segments the catalog still marks `Local`, so a segment already
/// flipped to `Both` is never re-uploaded. Its remote copy keeps the old layout
/// while holding identical rows, and ages out normally.
///
/// Bundles on UUID-stamped segments remain valid because the rewrite preserves
/// segment ID, rows, and row order. Rewriting an older unstamped segment
/// replaces its local generation identity with a UUID, so its bundle safely falls
/// back until the next index-backfill pass rebuilds it.
pub fn rewrite_stale_encodings(layout: &LocalLayout<'_>, budget: Duration) -> usize {
    let Ok(_permit) = REWRITE_SLOT.try_lock() else {
        return 0;
    };
    let deadline = Instant::now() + budget;
    let mut rewritten = 0;
    // A segment that fails to stage or commit stays stale, so it would be picked
    // again immediately; skipping it for the rest of the pass is what keeps one
    // unreadable file from starving every other segment. The next tick retries
    // it, because the set is per-pass.
    let mut failed: HashSet<String> = HashSet::new();
    while Instant::now() < deadline {
        let Some((path, was)) = next_stale_encoding(layout, &failed) else {
            break;
        };
        let started = Instant::now();
        let (staging, size) = match stage_rewritten_segment(
            Path::new(&path),
            layout.compaction.format.max_row_group_rows(),
        ) {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!(namespace = %layout.table(), segment = %basename(&path), %error,
                    "layout rewrite failed; leaving the segment as it was");
                failed.insert(path);
                continue;
            }
        };
        match commit_rewritten_segment(layout, &path, &staging, size) {
            Ok(true) => {}
            Ok(false) => {
                // Evicted mid-rewrite: it is gone from the view, so it will not
                // come back around.
                tracing::debug!(namespace = %layout.table(), segment = %basename(&path),
                    "segment went away mid-rewrite; discarded the replacement");
                continue;
            }
            Err(error) => {
                tracing::warn!(namespace = %layout.table(), segment = %basename(&path), %error,
                    "layout rewrite commit failed");
                failed.insert(path);
                continue;
            }
        }
        layout.tracker.record(&path);
        tracing::info!(
            namespace = %layout.table(),
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

/// The oldest live segment not yet known to carry the current encoding, as
/// `(path, size)`, ignoring anything in `skip`.
///
/// Reads footers OUTSIDE the view lock — the lock guards only a snapshot of
/// candidate paths — so a slow filesystem cannot stall writers behind this.
fn next_stale_encoding(layout: &LocalLayout<'_>, skip: &HashSet<String>) -> Option<(String, i64)> {
    let segments = layout.compaction.segments.segments();
    let live: HashSet<&str> = segments
        .iter()
        .map(|segment| segment.path.as_str())
        .collect();
    layout.tracker.retain_live(&live);
    let candidates: Vec<(String, i64)> = segments
        .iter()
        .filter(|segment| segment.level >= 1 && !skip.contains(&segment.path))
        .map(|segment| (segment.path.clone(), segment.size_bytes))
        .collect();
    candidates
        .into_iter()
        .find(|(path, _)| !layout.tracker.is_current(path))
}

/// Swap a staged rewrite over its segment and record the new size, under the
/// view lock. Returns `false` (discarding the staged file) when the segment is no
/// longer live — eviction can drop it while the rewrite runs, and renaming over
/// that path would resurrect a file nothing references.
fn commit_rewritten_segment(
    layout: &LocalLayout<'_>,
    path: &str,
    staging: &Path,
    byte_size: i64,
) -> Result<bool, StatsError> {
    let renamed = layout.compaction.segments.update(path, |segment| {
        std::fs::rename(staging, path).map_err(|error| {
            StatsError::Internal(format!("rename {} -> {path}: {error}", staging.display()))
        })?;
        segment.size_bytes = byte_size;
        Ok::<(), StatsError>(())
    });
    let Some(renamed) = renamed else {
        let _ = std::fs::remove_file(staging);
        return Ok(false);
    };
    renamed?;
    layout
        .compaction
        .catalog
        .set_byte_size(layout.table(), path, byte_size)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use crate::levanter_metrics_policy::levanter_metrics_schema;
    use crate::maintenance::REWRITE_LAYOUT_BUDGET;
    use crate::store::catalog::Catalog;
    use crate::store::legacy::archive::evict_segment;
    use crate::store::policy::StoragePolicy;
    use crate::store::ram_buffer::stamp_seq_and_build;
    use crate::store::schema::{schema_to_arrow, stored_form};
    use crate::store::segment::{discover_segments, read_segment_footer, write_segment_to_dir};
    use crate::store::table::maintenance::{self, TableWork};
    use crate::store::table::test_tables::*;
    use crate::store::table::TableRuntime;
    use crate::store::types::seg_filename;

    async fn maintain(table: &Arc<TableRuntime>, force_compact_l0: bool) {
        maintenance::run(table, TableWork::Cycle { force_compact_l0 })
            .await
            .unwrap();
    }

    /// Apply `swap` through the production commit path, which takes the
    /// query-visibility write lock and so must run off the reactor.
    async fn commit(table: &Arc<TableRuntime>, swap: PlannedSwap) {
        let table = Arc::clone(table);
        tokio::task::spawn_blocking(move || {
            commit_swap(&maintenance::local_compaction(&table), swap)
        })
        .await
        .unwrap()
        .unwrap();
    }

    fn live_segments(table: &Arc<TableRuntime>) -> Vec<LocalSegment> {
        maintenance::local_compaction(table).segments.segments()
    }

    #[tokio::test]
    async fn levanter_runtime_layout_migration_repairs_legacy_l0_and_flat_l1() {
        let dir = tempdir();
        let table_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            catalog,
        );
        table.append_aligned_batch(&metrics_aligned(&["run-a", "run-b"]));
        table.flush().await.unwrap();
        maintain(&table, true).await;

        let first = live_segments(&table)[0].clone();
        let legacy_l0_path = table_dir.join(seg_filename(0, first.min_seq));
        let mut legacy_l0 = first.clone();
        legacy_l0.path = legacy_l0_path.to_string_lossy().into_owned();
        legacy_l0.level = 0;
        legacy_l0.location = SegmentLocation::Local;
        commit(
            &table,
            PlannedSwap {
                removed: vec![first.path.clone()],
                added: vec![legacy_l0],
                unlink_removed: false,
                bump_rename: Some((PathBuf::from(first.path), legacy_l0_path)),
                input_arrow_bytes: 0,
            },
        )
        .await;
        maintain(&table, false).await;

        let second = live_segments(&table)[1].clone();
        let flat_l1_path = table_dir.join(seg_filename(1, second.min_seq));
        let mut flat_l1 = second.clone();
        flat_l1.path = flat_l1_path.to_string_lossy().into_owned();
        flat_l1.location = SegmentLocation::Local;
        commit(
            &table,
            PlannedSwap {
                removed: vec![second.path.clone()],
                added: vec![flat_l1],
                unlink_removed: false,
                bump_rename: Some((PathBuf::from(second.path), flat_l1_path.clone())),
                input_arrow_bytes: 0,
            },
        )
        .await;
        assert_eq!(flat_l1_path.parent(), Some(table_dir.as_path()));
        maintain(&table, false).await;
        assert!(!maintenance::placement_is_pending(&table));

        let segments = live_segments(&table);
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
                Some(table_dir.join("run_id").as_path())
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_streaming_rebuild_publishes_unindexed_outputs() {
        let dir = tempdir();
        let table_dir = dir.join("levanter.metrics");
        std::fs::create_dir_all(&table_dir).unwrap();
        let schema = stored_form(levanter_metrics_schema());
        let arrow_schema = schema_to_arrow(&schema);
        for input in 0..6_i64 {
            let first_seq = -1_000_000 + input * 10;
            let batch = stamp_seq_and_build(
                &metrics_aligned(&["run-a", "run-b"]),
                first_seq,
                &arrow_schema,
            );
            write_segment_to_dir(&table_dir, 0, first_seq, &batch).unwrap();
        }
        let table = open_table(
            "levanter.metrics",
            schema,
            Some(table_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        assert_eq!(
            pending_placement(&maintenance::local_layout(&table)).migration_l0,
            6
        );

        let rebuild_table = Arc::clone(&table);
        let rebuilt = tokio::task::spawn_blocking(move || {
            rebuild_l0_wave(&maintenance::local_layout(&rebuild_table))
        })
        .await
        .unwrap()
        .unwrap();
        assert_eq!(rebuilt, 1);
        assert_eq!(
            pending_placement(&maintenance::local_layout(&table)).migration_l0,
            0
        );
        assert_eq!(table.stats().row_count, 12);

        for path in discover_segments(&table_dir) {
            if read_segment_footer(&path, Some("timestamp_ms"))
                .unwrap()
                .level
                >= 1
            {
                assert!(path.starts_with(table_dir.join("run_id")));
                assert!(!crate::indices::format::bundle_path(&path).exists());
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn runtime_layout_migration_ignores_namespaces_without_a_partition_policy() {
        let dir = tempdir();
        let table_dir = dir.join("telemetry_v1.vllm");
        std::fs::create_dir_all(&table_dir).unwrap();
        let schema = worker_schema();
        let batch = stamp_seq_and_build(&aligned(2), -10, &schema_to_arrow(&schema));
        write_segment_to_dir(&table_dir, 0, -10, &batch).unwrap();
        let table = open_table(
            "telemetry_v1.vllm",
            schema,
            Some(table_dir),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );

        assert!(!pending_placement(&maintenance::local_layout(&table)).any());
        let rebuild_table = Arc::clone(&table);
        assert_eq!(
            tokio::task::spawn_blocking(move || rebuild_l0_wave(&maintenance::local_layout(
                &rebuild_table
            )))
            .await
            .unwrap()
            .unwrap(),
            0
        );
        assert_eq!(table.stats().row_count, 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn levanter_runtime_layout_migration_relocates_remote_only_l1() {
        let dir = tempdir();
        let remote_dir = dir.join("remote");
        let table_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table_remote(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            Arc::clone(&catalog),
            remote_dir.to_str().unwrap(),
            StoragePolicy::default(),
        );
        table.append_aligned_batch(&metrics_aligned(&["run-a"]));
        table.flush().await.unwrap();
        maintain(&table, true).await;

        let current = catalog.list_segments("levanter.metrics").unwrap().remove(0);
        assert_eq!(current.location, SegmentLocation::Both);
        let current_key = segment_relative_key(&table_dir, &current.path).unwrap();
        let current_remote_path = remote_dir.join("levanter.metrics").join(&current_key);
        assert!(current_remote_path.exists());

        let evict_table = Arc::clone(&table);
        let evict_path = current.path.clone();
        tokio::task::spawn_blocking(move || {
            let compaction = maintenance::local_compaction(&evict_table);
            evict_segment(
                compaction.table,
                compaction.catalog,
                compaction.segments,
                compaction.query_visibility,
                &evict_path,
            )
        })
        .await
        .unwrap();
        let mut legacy = catalog.list_segments("levanter.metrics").unwrap().remove(0);
        assert_eq!(legacy.location, SegmentLocation::Remote);

        let filename = Path::new(&legacy.path).file_name().unwrap();
        let legacy_remote_path = remote_dir.join("levanter.metrics").join(filename);
        std::fs::rename(&current_remote_path, &legacy_remote_path).unwrap();
        let old_path = legacy.path.clone();
        legacy.path = table_dir.join(filename).to_string_lossy().into_owned();
        catalog
            .replace_segments("levanter.metrics", &[old_path], &[legacy.clone()])
            .unwrap();

        let layout = maintenance::local_layout(&table);
        assert_eq!(archive_placement_candidates(&layout).unwrap().len(), 1);
        assert!(relocate_one_archived_segment(&layout).await.unwrap());
        assert_eq!(archive_placement_candidates(&layout).unwrap().len(), 0);
        assert!(!relocate_one_archived_segment(&layout).await.unwrap());

        let relocated = catalog.list_segments("levanter.metrics").unwrap().remove(0);
        assert_eq!(relocated.location, SegmentLocation::Remote);
        assert_eq!(
            Path::new(&relocated.path).parent().unwrap().parent(),
            Some(table_dir.join("run_id").as_path())
        );
        let relocated_key = segment_relative_key(&table_dir, &relocated.path).unwrap();
        assert!(remote_dir
            .join("levanter.metrics")
            .join(relocated_key)
            .exists());
        assert!(!legacy_remote_path.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn layout_rewrite_updates_the_local_segment_without_re_uploading() {
        use arrow::array::RecordBatch;
        use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
        use parquet::file::properties::WriterProperties;

        let dir = tempdir();
        let remote = dir.join("remote");
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table_remote(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            Arc::clone(&catalog),
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        write_one(&table).await;
        maintain(&table, true).await;

        let segment = catalog.list_segments("iris.worker").unwrap().remove(0);
        assert_eq!(segment.location, SegmentLocation::Both);
        let remote_name = remote_files(&remote, "iris.worker").remove(0);
        let remote_path = remote.join("iris.worker").join(&remote_name);
        let remote_before = std::fs::read(&remote_path).unwrap();

        // Put the local file back onto an older layout (no stamp), as a
        // pre-existing segment would be.
        let path = PathBuf::from(&segment.path);
        let batches: Vec<RecordBatch> = {
            let file = std::fs::File::open(&path).unwrap();
            ParquetRecordBatchReaderBuilder::try_new(file)
                .unwrap()
                .build()
                .unwrap()
                .map(|batch| batch.unwrap())
                .collect()
        };
        let out = std::fs::File::create(&path).unwrap();
        let options =
            ArrowWriterOptions::new().with_properties(WriterProperties::builder().build());
        let mut writer =
            ArrowWriter::try_new_with_options(out, batches[0].schema(), options).unwrap();
        for batch in &batches {
            writer.write(batch).unwrap();
        }
        writer.close().unwrap();
        // The layout tracker is per-runtime and starts empty, so a segment written
        // by an older build is always first seen after a restart. Reopen the table
        // to model that: nothing in production edits a segment behind the tracker.
        table.shutdown(Duration::from_secs(10)).await;
        let restarted = open_table_remote(
            "iris.worker",
            worker_schema(),
            Some(table_dir),
            Arc::clone(&catalog),
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        assert_eq!(
            rewrite_stale_encodings(
                &maintenance::local_layout(&restarted),
                REWRITE_LAYOUT_BUDGET
            ),
            1
        );

        // Local file adopted the current layout and the catalog followed it.
        assert!(crate::store::segment::segment_layout_is_current(&path));
        let after = catalog.list_segments("iris.worker").unwrap().remove(0);
        assert_eq!(
            after.byte_size,
            std::fs::metadata(&path).unwrap().len() as i64
        );
        assert_eq!(after.location, SegmentLocation::Both);

        // The archive is untouched: same object, same bytes, no re-upload.
        assert_eq!(remote_files(&remote, "iris.worker"), vec![remote_name]);
        assert_eq!(std::fs::read(&remote_path).unwrap(), remote_before);

        // A second pass finds nothing left to do.
        assert_eq!(
            rewrite_stale_encodings(
                &maintenance::local_layout(&restarted),
                REWRITE_LAYOUT_BUDGET
            ),
            0
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
