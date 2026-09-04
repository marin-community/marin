//! One table's live runtime: the parts, and nothing that operates on them.
//!
//! A [`TableRuntime`] owns the ingest buffer, the live segment view, the durable
//! state controller, and the table's resolved policy, and it hands those parts
//! to the modules that do the work — flush, compaction, index artifacts,
//! specification migration, and the legacy archive and layout paths. It
//! implements none of that itself.
//!
//! The public surface is deliberately small: accept rows, wait for durability,
//! read a snapshot, report accounting, and shut down. Maintenance reaches the
//! work modules through
//! [`TableManager::run_work`](crate::store::table::TableManager::run_work), not
//! through methods here.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use tokio::sync::{watch, Notify, RwLock};

use crate::errors::StatsError;
use crate::indices::IndexRegistry;
use crate::maintenance::MaintenanceLimits;
use crate::store::adopt::adopt_local_segments;
use crate::store::catalog::{Catalog, SpecLifecycle};
use crate::store::compaction::config::CompactionConfig;
use crate::store::legacy::layout::LayoutTracker;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{AlignedBatch, Schema};
use crate::store::table::controller::TableController;
use crate::store::table::flush;
use crate::store::table::index_artifacts::BackfillSkips;
use crate::store::table::ingest::{FlushDemand, IngestBuffer};
use crate::store::table::query_view::{plan_visible_segments, SegmentObjectMap};
use crate::store::table::segment_format::SegmentFormat;
use crate::store::table::segment_view::{visible_segments, SegmentSnapshot, SegmentView};
use crate::store::table::spec_migration::MigrationBlock;
use crate::store::table_spec::TablePolicy;
use crate::store::table_state::TableSnapshot;
use crate::store::types::{segment_to_row, NamespaceStats, SegmentRow};

/// A single table's live runtime, disk-backed or in-memory.
///
/// Owns the table's concurrency envelope — the append fast path into the
/// ingest buffer, the flush/observation locks that serialize seal-and-commit
/// against each other, the resolved policy cells, and tracked background
/// tasks. Durable transitions and heavy work stay with their owners: the
/// controller publishes state, and flush/compaction/migration modules borrow
/// this runtime's pieces through narrow views like [`flush::FlushTarget`].
pub struct TableRuntime {
    pub(super) name: String,
    pub(super) format: SegmentFormat,
    /// `None` => in-memory mode: every append is immediately persisted and no
    /// Parquet is ever written.
    pub(super) data_dir: Option<PathBuf>,
    pub(super) catalog: Arc<Catalog>,
    /// Durable-state controller for this table: the only owner of its
    /// publication, writer claim, and canonical object writes.
    pub(super) controller: Arc<TableController>,
    pub(super) buffer: IngestBuffer,
    pub(super) segments: SegmentView,
    /// The operating policy the table's specification resolves to. Replaced on
    /// re-registration and on migration activation.
    pub(super) policy: Mutex<TablePolicy>,
    /// Per-table retention overrides; `None` fields inherit the cluster-wide
    /// [`CompactionConfig`] caps.
    pub(super) storage_policy: Mutex<StoragePolicy>,
    /// Leveled-compaction tuning: the scheduler reads `check_interval`, the
    /// planner reads `level_targets`/`max_segments_per_level`.
    pub(super) compaction_config: CompactionConfig,
    /// Serializes the whole local flush (seal → write → catalog → commit).
    /// Without it two concurrent flushers race: the second seal would overwrite
    /// the first's in-flight buffer, and the high-water mark could advance before
    /// the older segment is durable. Distinct from the buffer's own short lock,
    /// so appends are never blocked by a flush write.
    pub(super) flush_lock: Mutex<()>,
    /// The same serialization for the object-backed flush, which awaits object
    /// I/O. Also held by a migration backfill, which rewrites the very sources a
    /// flush would commit against.
    pub(super) object_flush_lock: tokio::sync::Mutex<()>,
    /// Serializes the maintenance cycle against direct callers. The flush path
    /// uses its own lock instead, so flushes and compactions stay concurrent.
    pub(super) maint_lock: tokio::sync::Mutex<()>,
    /// Process-wide query-visibility lock (one shared instance for the whole
    /// store), taken on the WRITE side before any rename or unlink of a file a
    /// query may have snapshotted.
    pub(super) query_visibility: Arc<RwLock<()>>,
    pub(super) indices: Arc<IndexRegistry>,
    /// Process-wide maintenance concurrency limits, shared by every table.
    pub(super) limits: Arc<MaintenanceLimits>,
    pub(super) layout_tracker: LayoutTracker,
    pub(super) index_skips: Mutex<BackfillSkips>,
    /// How this table's specification transition is failing, carried across
    /// maintenance ticks.
    pub(super) migration_block: Mutex<MigrationBlock>,
    pub(super) last_object_gc: Mutex<Option<Instant>>,
    pub(super) last_orphan_sweep: Mutex<Option<Instant>>,
    /// Latched stop flag the dispatched work checks at the top of each loop
    /// iteration.
    pub(super) stopped: AtomicBool,
    /// Handles for the maintenance work the scheduler dispatched against this
    /// table. Retained so a re-register replacement, a drop, or store shutdown
    /// can cooperatively cancel and JOIN that work within a bounded timeout.
    pub(super) task_handles: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

impl TableRuntime {
    /// Build a runtime over `data_dir` (disk-backed when `Some`).
    ///
    /// An object-backed table's contents come from its durable state: recovery
    /// seeds sequence allocation from the catalog projection of that state and
    /// adopts nothing from disk. A legacy table recovers its next seq from
    /// segment footers, adopts the local segment files it finds, and starts its
    /// high-water mark at the recovered seq so a caller awaiting a
    /// previously-durable seq returns immediately.
    ///
    /// Storage implementations are built by `Store` and injected here. The
    /// caller starts the maintenance scheduler once the store is fully built.
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
        compaction_config: CompactionConfig,
    ) -> Result<Arc<TableRuntime>, StatsError> {
        let startup_started = Instant::now();
        let format = SegmentFormat::resolve(schema)?;

        let local_recovery_started = Instant::now();
        let (next_seq, adopted, initial_persisted) =
            match (&data_dir, controller.is_object_backed()) {
                (None, _) => (1_i64, VecDeque::new(), -1_i64),
                (Some(dir), true) => {
                    create_table_dir(dir)?;
                    let rows = catalog.list_segments(name)?;
                    let max_persisted = rows
                        .iter()
                        .filter(|row| row.row_count > 0)
                        .map(|row| row.max_seq)
                        .max()
                        .unwrap_or(-1);
                    // The claimed state's high-water mark can exceed the max
                    // seq in its published segments (a legacy import excludes
                    // archive-only rows; retirement deletes legacy rows), and
                    // reissuing a sequence number breaks seq checkpointing.
                    let next_seq = crate::store::adopt::recover_next_seq(&rows)
                        .max(controller.claimed_high_water() + 1);
                    (next_seq, VecDeque::new(), max_persisted)
                }
                (Some(dir), false) => {
                    create_table_dir(dir)?;
                    let adopted = adopt_local_segments(
                        dir,
                        Some(format.key_column()),
                        &catalog,
                        name,
                        controller.object_store().map(Arc::as_ref),
                    )?;
                    // Seed next_seq past every segment the catalog knows about,
                    // not just on-disk footers. A segment evicted to remote has
                    // its local Parquet unlinked, so a footer-only scan
                    // under-counts and would reuse live seqs (silent overwrite).
                    // Adoption has already read every healthy local footer, so
                    // reuse those max_seq values rather than scanning again.
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
                        .filter(|segment| segment.row_count > 0)
                        .map(|segment| segment.max_seq)
                        .max()
                        .unwrap_or(-1);
                    (next_seq, adopted, max_persisted)
                }
            };
        let local_recovery_ms = local_recovery_started.elapsed().as_millis() as u64;

        let policy = TablePolicy::resolve(catalog.spec_lifecycle(name)?.operative());
        let runtime = Arc::new(TableRuntime {
            name: name.to_string(),
            buffer: IngestBuffer::new(
                name,
                Arc::clone(format.arrow_schema()),
                next_seq,
                initial_persisted,
                data_dir.is_none(),
                maintenance_wake,
            ),
            segments: SegmentView::new(name, adopted.clone()),
            format,
            data_dir,
            catalog: Arc::clone(&catalog),
            controller,
            policy: Mutex::new(policy),
            storage_policy: Mutex::new(storage_policy),
            compaction_config,
            flush_lock: Mutex::new(()),
            object_flush_lock: tokio::sync::Mutex::new(()),
            maint_lock: tokio::sync::Mutex::new(()),
            query_visibility,
            indices,
            limits,
            layout_tracker: LayoutTracker::default(),
            index_skips: Mutex::new(BackfillSkips::default()),
            migration_block: Mutex::new(MigrationBlock::default()),
            last_object_gc: Mutex::new(None),
            last_orphan_sweep: Mutex::new(None),
            stopped: AtomicBool::new(false),
            task_handles: Mutex::new(Vec::new()),
        });

        // An object-backed table publishes its locally durable state so readers
        // have something to pin, then rebuilds its query view from that state's
        // catalog rows. Metadata only: nothing is downloaded and the local cache
        // is not consulted.
        if runtime.controller.is_object_backed() {
            runtime.controller.seed_local_snapshot();
            let active = catalog.spec_lifecycle(name)?.active_version();
            runtime.activate_query_version(active)?;
        }

        // Refresh the catalog from the adopted view so the segments table
        // reflects on-disk reality after a fresh boot from a wiped catalog.
        let catalog_refresh_started = Instant::now();
        let adopted_rows: Vec<SegmentRow> = adopted
            .iter()
            .map(|segment| segment_to_row(name, segment))
            .collect();
        catalog.upsert_segments(&adopted_rows)?;

        tracing::info!(
            namespace = name,
            segments = adopted.len(),
            next_seq,
            local_recovery_ms,
            catalog_refresh_ms = catalog_refresh_started.elapsed().as_millis() as u64,
            total_ms = startup_started.elapsed().as_millis() as u64,
            "finelog table startup complete"
        );
        Ok(runtime)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Schema {
        self.format.schema()
    }

    /// The schema this table's segments are written with (store form: includes
    /// the implicit `seq` column).
    pub fn arrow_schema(&self) -> &SchemaRef {
        self.format.arrow_schema()
    }

    /// Resolved physical key, including the implicit `timestamp_ms` default.
    pub fn key_column(&self) -> &str {
        self.format.key_column()
    }

    /// Whether this process still owns the table's durable state. A fenced
    /// object-backed table rejects writes until a restart re-claims it.
    pub fn write_ready(&self) -> bool {
        self.controller.writes_ready()
    }

    /// The durable-state commit owner for this table.
    pub fn controller(&self) -> &Arc<TableController> {
        &self.controller
    }

    /// The repeated error this table's specification transition is stuck on.
    /// `None` while the transition is progressing or has none to run.
    pub fn blocked_migration_error(&self) -> Option<String> {
        self.migration_block
            .lock()
            .unwrap()
            .blocked_error()
            .map(str::to_string)
    }

    /// Swap in a new retention policy (re-register). Picked up next eviction.
    pub fn update_policy(&self, policy: StoragePolicy) {
        *self.storage_policy.lock().unwrap() = policy;
    }

    /// Swap in the operating policy a new specification resolves to.
    pub fn update_table_spec(&self, status: &SpecLifecycle) {
        *self.policy.lock().unwrap() = TablePolicy::resolve(status.operative());
    }

    /// Rebuild the query view from the segments visible at definition version
    /// `version`.
    pub fn activate_query_version(&self, version: u64) -> Result<(), StatsError> {
        let segments = visible_segments(
            &self.catalog,
            &self.name,
            version,
            self.controller.object_store().map(Arc::as_ref),
        )?;
        self.segments.replace_all(segments);
        Ok(())
    }

    /// Stamp `seq` onto `aligned` and append it, returning the last allocated
    /// seq (or `-1` if empty). Rejects a write that would exceed the RAM limit.
    pub fn append_aligned_batch(&self, aligned: &AlignedBatch) -> Result<i64, StatsError> {
        self.buffer
            .append_aligned(aligned, self.policy().max_buffer_bytes)
    }

    /// Append already-built log columns (`seq` excluded), returning the last
    /// seq. Rejects a write that would exceed the RAM limit.
    pub fn append_log_batch(
        &self,
        columns: Vec<ArrayRef>,
        num_rows: usize,
        added_bytes: i64,
    ) -> Result<i64, StatsError> {
        self.buffer.append_columns(
            columns,
            num_rows,
            added_bytes,
            self.policy().max_buffer_bytes,
        )
    }

    /// Raise the sequence allocator to at least `next_seq`; never lowers it.
    pub fn raise_next_seq_floor(&self, next_seq: i64) {
        self.buffer.raise_next_seq_floor(next_seq);
    }

    /// Block until `target` is durable, bounded by `timeout`.
    pub async fn await_persisted(&self, target: i64, timeout: Duration) -> Result<(), StatsError> {
        self.buffer.await_persisted(target, timeout).await
    }

    /// Subscribe to the durability high-water mark.
    pub fn watch_persisted_seq(&self) -> watch::Receiver<i64> {
        self.buffer.watch_persisted()
    }

    /// The table's readable segments as one consistent observation.
    ///
    /// A table is on the snapshot path once it has an activated object-native
    /// definition. Legacy tables, and object-backed tables still importing their
    /// version-0 history, read the local segment files the shared
    /// query-visibility lock guards.
    ///
    /// Only sealed segments appear: queries see flushed data, never the in-RAM
    /// buffer.
    pub fn query_snapshot(&self) -> Result<SegmentSnapshot, StatsError> {
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
        Ok(self.segments.snapshot())
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
            key_bounds: Default::default(),
            seq_bounds: Default::default(),
            row_counts: Default::default(),
            partitions: Default::default(),
            min_seq: planned.iter().map(|segment| segment.min_seq).min(),
            artifacts: Default::default(),
            sources: SegmentObjectMap::new(),
        };
        for segment in planned {
            if let Some(bounds) = segment.key_bounds {
                view.key_bounds.insert(segment.path.clone(), bounds);
            }
            view.seq_bounds
                .insert(segment.path.clone(), (segment.min_seq, segment.max_seq));
            if let Some(row_count) = segment.row_count {
                view.row_counts.insert(segment.path.clone(), row_count);
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

    /// The maximum query time this table's pinned state promises. `None` for a
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

    /// Aggregate row/byte/seq stats over sealed segments plus the RAM buffer.
    ///
    /// The seq-window math: `min_seq = seg_min if seg_min else (next_seq -
    /// ram_rows if ram_rows else 0)`; `max_seq = max(seg_max, next_seq - 1) if
    /// (seg_max or ram_rows) else 0`. Segment bounds only consider segments with
    /// rows.
    pub fn stats(&self) -> NamespaceStats {
        let segments = self.segments.totals();
        let buffered = self.buffer.buffered();
        if segments.count == 0 && buffered.rows == 0 {
            return NamespaceStats::default();
        }
        let segment_min = segments.min_seq.unwrap_or(0);
        let segment_max = segments.max_seq.unwrap_or(0);
        let min_seq = if segment_min != 0 {
            segment_min
        } else if buffered.rows != 0 {
            buffered.next_seq - buffered.rows
        } else {
            0
        };
        let max_seq = if segment_max != 0 || buffered.rows != 0 {
            segment_max.max(buffered.next_seq - 1)
        } else {
            0
        };
        NamespaceStats {
            row_count: segments.rows + buffered.rows,
            byte_size: segments.bytes + buffered.bytes,
            min_seq,
            max_seq,
            segment_count: segments.count as i32,
        }
    }

    /// Aggregate in-RAM accounting for the diagnostics line:
    /// `(ram_bytes, chunk_count)`.
    pub fn memory_summary(&self) -> (i64, usize) {
        let buffered = self.buffer.buffered();
        (buffered.bytes, buffered.chunks)
    }

    /// Drain the in-RAM buffer to one new segment.
    ///
    /// Returns `Ok(())` when there was nothing to flush. On failure the in-flight
    /// buffer is restored and the durability mark is not advanced.
    pub async fn flush(self: &Arc<Self>) -> Result<(), StatsError> {
        let Some(dir) = self.data_dir.clone() else {
            return Ok(());
        };
        let policy = self.policy();
        if !policy.object_backed() {
            let runtime = Arc::clone(self);
            return tokio::task::spawn_blocking(move || {
                let _flush_guard = runtime.flush_lock.lock().unwrap();
                flush::flush_local(runtime.flush_target(), &dir)
            })
            .await
            .map_err(|error| StatsError::Internal(format!("flush task panicked: {error}")))?;
        }
        {
            let _flush_guard = self.object_flush_lock.lock().await;
            flush::flush_to_objects(self.flush_target(), &policy).await?;
        }
        // Publication leaves the flush entirely: a flush occupies one of the
        // process's few flush permits, every object-backed table flushes on
        // the same cadence, and a network publication inside that window
        // oversubscribes the pool until every table's acks queue behind it.
        // The committed revision is owed from the moment it is locally
        // durable; publications serialize in the per-table controller mailbox
        // where bursts coalesce, and a failure stays owed to maintenance.
        let runtime = Arc::clone(self);
        Arc::clone(self).spawn_tracked(async move {
            if let Err(error) = runtime.controller.publish_owed().await {
                tracing::warn!(
                    namespace = %runtime.name,
                    %error,
                    "flush publication deferred; revision stays owed to maintenance"
                );
            }
        });
        Ok(())
    }

    pub(super) fn flush_target(&self) -> flush::FlushTarget<'_> {
        flush::FlushTarget {
            table: &self.name,
            format: &self.format,
            buffer: &self.buffer,
            segments: &self.segments,
            catalog: &self.catalog,
            controller: &self.controller,
        }
    }

    /// Backdate one segment's `created_at_ms` in the catalog. Serves the
    /// flag-gated `/debug/backdate` admin route so age-eviction tests stay
    /// RPC-only (no sleep).
    pub fn backdate_segment(
        &self,
        path_basename: &str,
        created_at_ms: i64,
    ) -> Result<(), StatsError> {
        // The catalog stores the absolute path, while callers pass the basename.
        for row in self.catalog.list_segments(&self.name)? {
            if crate::store::types::basename(&row.path) == path_basename {
                self.catalog
                    .set_created_at_ms(&self.name, &row.path, created_at_ms)?;
            }
        }
        Ok(())
    }

    /// The table's resolved operating policy.
    pub(super) fn policy(&self) -> TablePolicy {
        self.policy.lock().unwrap().clone()
    }

    /// Whether this table persists to disk. A memory table neither flushes nor
    /// maintains.
    pub fn is_disk_backed(&self) -> bool {
        self.data_dir.is_some()
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// How often this table owes an ordinary maintenance cycle.
    pub fn maintenance_interval(&self) -> Duration {
        self.compaction_config.check_interval
    }

    /// What the scheduler needs to time this table's next flush.
    pub fn flush_demand(&self) -> FlushDemand {
        self.buffer.demand(self.policy().max_flush_age)
    }

    /// Clear the demand the scheduler is about to satisfy.
    pub fn clear_flush_demand(&self) {
        self.buffer.clear_demand();
    }

    /// Record flush demand directly. `forced` bypasses the coalescing window the
    /// way a buffer holding a whole segment does.
    #[cfg(test)]
    pub fn request_flush(&self, forced: bool) {
        self.buffer.request_flush(forced);
    }

    /// Run `work` as a background task this table owns.
    ///
    /// Returns false when the table has already stopped, in which case nothing is
    /// spawned: `stop_and_join` has drained the handle list and a task registered
    /// after it would outlive the shutdown window.
    pub fn spawn_tracked(
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

    /// How many background tasks the scheduler currently has in flight against
    /// this table.
    #[cfg(test)]
    pub fn background_task_count(&self) -> usize {
        self.task_handles.lock().unwrap().len()
    }

    /// Latch the stop flag, wake any dispatched maintenance work, and JOIN it
    /// bounded by `timeout` (a wedged task that misses the window is aborted, so
    /// this can never hang). Does NOT flush — callers sequence durability
    /// (`shutdown`) or pre-delete teardown themselves.
    pub async fn stop_and_join(&self, timeout: Duration) {
        self.stopped.store(true, Ordering::SeqCst);
        let handles: Vec<tokio::task::JoinHandle<()>> =
            std::mem::take(&mut *self.task_handles.lock().unwrap());
        // Keep an abort handle for each task so a wedged task that misses the
        // bounded join window can still be cancelled (never busy-wait, never
        // hang). `JoinHandle::abort` is idempotent on an already-finished task.
        let abort_handles: Vec<tokio::task::AbortHandle> =
            handles.iter().map(|handle| handle.abort_handle()).collect();
        match tokio::time::timeout(timeout, futures::future::join_all(handles)).await {
            Ok(results) => {
                for result in results {
                    if let Err(error) = result {
                        if !error.is_cancelled() {
                            tracing::warn!(namespace = %self.name, %error, "shutdown: bg task join error");
                        }
                    }
                }
            }
            Err(_elapsed) => {
                tracing::warn!(
                    namespace = %self.name,
                    "shutdown: bg tasks did not join within timeout; aborting them"
                );
                for handle in &abort_handles {
                    handle.abort();
                }
            }
        }
    }

    /// Signal dispatched maintenance work to stop without awaiting it. Safe to
    /// call from a synchronous context with no tokio runtime — used by
    /// `drop_table` for in-memory tables, which spawn no background tasks.
    pub fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
    }

    /// Cooperatively shut the table down.
    ///
    /// Stops and JOINs any dispatched maintenance work (bounded by `timeout`),
    /// then does a final flush (no RAM-only rows survive; durability is already
    /// preserved — an acked write was on a sealed segment), publishes any owed
    /// object-backed revision, and gives a legacy archive one final bounded sync.
    pub async fn shutdown(self: &Arc<Self>, timeout: Duration) {
        self.stop_and_join(timeout).await;
        if let Err(error) = self.flush().await {
            tracing::warn!(namespace = %self.name, %error, "shutdown: final flush failed");
        }
        if self.controller.publication_owed() {
            match tokio::time::timeout(timeout, self.controller.publish_state()).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    tracing::warn!(namespace = %self.name, %error, "shutdown: final publication failed");
                }
                Err(_elapsed) => {
                    tracing::warn!(namespace = %self.name, "shutdown: final publication timed out");
                }
            }
        }
        // Legacy tables get one final bounded archive sync. This is a no-op for
        // object-backed tables.
        if self.controller.object_persistence_configured() {
            match tokio::time::timeout(timeout, super::maintenance::sync_archive(self)).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    tracing::warn!(namespace = %self.name, %error, "shutdown: final remote sync failed");
                }
                Err(_elapsed) => {
                    tracing::warn!(namespace = %self.name, "shutdown: final remote sync timed out");
                }
            }
        }
    }
}

fn create_table_dir(dir: &std::path::Path) -> Result<(), StatsError> {
    std::fs::create_dir_all(dir).map_err(|error| {
        StatsError::Internal(format!("create table dir {}: {error}", dir.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::store::segment::discover_segments;
    use crate::store::table::test_tables::*;

    /// Seal the buffer the way the maintenance scheduler would, then wait for the
    /// durability high-water mark. These tests open a runtime directly, so no
    /// scheduler is polling it.
    async fn flush_and_await(table: &Arc<TableRuntime>, target: i64) {
        table.flush().await.unwrap();
        table
            .await_persisted(target, Duration::from_secs(10))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn shutdown_aborts_wedged_task_within_timeout() {
        // The riskiest shutdown path: a bg task stuck in a long compaction/upload
        // that never observes the stop latch. shutdown() must JOIN bounded and
        // ABORT the laggard rather than hang. Inject a never-completing task into
        // the handle set and assert shutdown returns far inside the join timeout.
        let dir = tempdir();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(dir.join("iris.worker")),
            catalog,
        );
        let wedged = tokio::spawn(async { std::future::pending::<()>().await });
        table.task_handles.lock().unwrap().push(wedged);

        let start = std::time::Instant::now();
        table.shutdown(Duration::from_millis(50)).await;
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "shutdown hung on a wedged task instead of aborting it: {:?}",
            start.elapsed()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn a_disk_backed_append_rejects_a_full_buffer() {
        let dir = tempdir();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(dir.join("iris.worker")),
            catalog,
        );
        let mut batch = aligned(1);
        batch.byte_size = crate::store::table::ingest::MAX_TABLE_RAM_BYTES - 8;
        table.append_aligned_batch(&batch).unwrap();
        let before = table.memory_summary();

        let error = table.append_aligned_batch(&aligned(1)).unwrap_err();

        assert!(matches!(error, StatsError::ResourceExhausted(_)));
        assert_eq!(table.memory_summary(), before);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn append_then_await_persisted_writes_a_segment() {
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            catalog,
        );

        let last = table.append_aligned_batch(&aligned(3)).unwrap();
        assert_eq!(last, 3);
        flush_and_await(&table, last).await;

        // A segment file exists and stats reflect it.
        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1);
        let stats = table.stats();
        assert_eq!(stats.row_count, 3);
        assert_eq!(stats.min_seq, 1);
        assert_eq!(stats.max_seq, 3);
        assert_eq!(stats.segment_count, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn implicit_timestamp_key_captures_segment_bounds() {
        let dir = tempdir();
        let mut schema = worker_schema();
        schema.key_column.clear();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            schema,
            Some(dir.join("iris.worker")),
            catalog,
        );
        let last = table.append_aligned_batch(&aligned(3)).unwrap();
        flush_and_await(&table, last).await;

        let snapshot = table.query_snapshot().unwrap();
        assert_eq!(
            snapshot.key_bounds.values().cloned().collect::<Vec<_>>(),
            vec![("1000".to_string(), "1002".to_string())]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn await_persisted_negative_returns_immediately() {
        let dir = tempdir();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(dir.join("iris.worker")),
            catalog,
        );
        table
            .await_persisted(-1, Duration::from_millis(1))
            .await
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn stats_ram_only_seq_window() {
        // Memory mode: no flush; stats come from RAM via the seq window.
        let catalog = Arc::new(Catalog::open(None).unwrap());
        let table = open_table("iris.worker", worker_schema(), None, catalog);
        table.append_aligned_batch(&aligned(3)).unwrap();
        table.append_aligned_batch(&aligned(2)).unwrap();
        let stats = table.stats();
        assert_eq!(stats.row_count, 5);
        assert_eq!(stats.min_seq, 1);
        assert_eq!(stats.max_seq, 5);
        assert!(stats.byte_size > 0);
        assert_eq!(stats.segment_count, 0);
    }

    #[tokio::test]
    async fn restart_recovers_next_seq_past_persisted_max() {
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        {
            let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
            let table = open_table(
                "iris.worker",
                worker_schema(),
                Some(table_dir.clone()),
                catalog,
            );
            let last = table.append_aligned_batch(&aligned(4)).unwrap();
            flush_and_await(&table, last).await;
        }
        // Second runtime over the same dir: next seq is past the persisted max,
        // and a previously-durable seq is already satisfied.
        let catalog2 = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let restarted = open_table("iris.worker", worker_schema(), Some(table_dir), catalog2);
        let stats = restarted.stats();
        assert_eq!(stats.row_count, 4);
        assert_eq!(stats.max_seq, 4);
        // A new append continues monotonically from seq 5.
        let last = restarted.append_aligned_batch(&aligned(1)).unwrap();
        assert_eq!(last, 5);
        restarted
            .await_persisted(4, Duration::from_secs(1))
            .await
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn flush_coalesces_multiple_appends_into_few_segments() {
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            catalog,
        );
        // Many small appends; flush via the direct sync-point once.
        let mut last = -1;
        for _ in 0..5 {
            last = table.append_aligned_batch(&aligned(2)).unwrap();
        }
        table.flush().await.unwrap();
        table
            .await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1, "one flush coalesces buffered appends");
        assert_eq!(table.stats().row_count, 10);
        std::fs::remove_dir_all(&dir).ok();
    }
}
