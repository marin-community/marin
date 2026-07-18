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
//!   caller-supplied timeout, nudging the flush task via a `Notify`.
//! - The flush task seals one L0 per wake but then holds off for
//!   `MIN_FLUSH_INTERVAL`, so the appends in that window coalesce into a single
//!   L0 instead of one tiny segment per nudge; a full buffer
//!   (`SEGMENT_TARGET_BYTES`) bypasses the cooldown via `force_flush`.
//!
//! `MemoryNamespace` (no `data_dir`) treats every append as immediately
//! persisted: it stamps into a RAM buffer, advances `persisted_seq` to the
//! freshly allocated seq under the lock, and never writes parquet.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::SchemaRef;
use tokio::sync::{watch, Notify, RwLock};

use crate::errors::StatsError;
use crate::proto::finelog::stats::ColumnType;
use crate::store::catalog::Catalog;
use crate::store::compaction::config::{CompactionConfig, CompactionJob};
use crate::store::compaction::executor::{read_segment_batches, run_job, PlannedSwap};
use crate::store::compaction::planner::plan;
use crate::store::policy::StoragePolicy;
use crate::store::ram_buffer::{stamp_seq_and_build, RamBuffers, SealedBuffer};
use crate::store::reconcile::reconcile_remote_segments;
use crate::store::remote::{build_remote_store, RemoteStore};
use crate::store::schema::{schema_to_arrow, AlignedBatch, Schema};
use crate::store::segment::{
    discover_segments, read_segment_footer, recover_next_seq, write_segment_to_dir,
};
use crate::store::trigram::{sidecar_path, write_sidecar};
use crate::store::types::{LocalSegment, NamespaceStats, SegmentLocation, SegmentRow};

/// Best-effort removal of a segment's trigram sidecar (`<path>.tgm`), co-located
/// with every parquet unlink. A missing sidecar (an L0 / unindexed-namespace
/// segment never had one) is not an error.
fn remove_sidecar(parquet_path: &str) {
    let s = sidecar_path(Path::new(parquet_path));
    if let Err(e) = std::fs::remove_file(&s) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(path = %s.display(), error = %e, "failed to remove trigram sidecar");
        }
    }
}

/// Buffered-byte size at which an append forces an early flush, short-circuiting
/// the flush-rate cooldown so a write burst can't buffer unboundedly (and bounds
/// a single L0's size).
pub const SEGMENT_TARGET_BYTES: i64 = 100 * 1024 * 1024;

/// Maximum idle gap before the flush task wakes on its own. With steady writes
/// the per-append nudge drives flushes; this is the ceiling for a quiet namespace.
pub const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// Minimum spacing between consecutive L0 flushes. Every append nudges the flush
/// task, so without this floor a steadily-written namespace seals a fresh tiny L0
/// on each wake (many per second). Holding off coalesces all appends in the
/// window into ONE L0, capping L0 creation at one segment per interval.
pub const MIN_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Default durability-await budget when the RPC carries no deadline.
pub const DEFAULT_PERSIST_TIMEOUT: Duration = Duration::from_secs(30);

/// Trigram sidecars rebuilt per maintenance tick by the background backfill.
/// Kept at one because a single index build over a terminal-level segment is
/// itself heavy (the builder currently uses substantial CPU + RAM); rebuilding
/// one per tick keeps the backfill the lowest-priority maintenance work and
/// never starves compaction/sync/eviction. Raise once the builder is cheaper.
pub const BACKFILL_SIDECARS_PER_TICK: usize = 1;

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

/// A single namespace's write engine, disk-backed or in-memory.
pub struct Namespace {
    name: String,
    schema: Schema,
    arrow_schema: SchemaRef,
    key_column: Option<String>,
    /// `None` => in-memory mode (every append immediately persisted, no parquet).
    data_dir: Option<PathBuf>,
    catalog: Arc<Catalog>,
    /// Leveled-compaction tuning. The maintenance task reads `check_interval`,
    /// the planner reads `level_targets`/`max_segments_per_level`.
    compaction_config: CompactionConfig,
    inner: Mutex<NsInner>,
    /// Serializes the whole `flush_once` body (seal → write → catalog → commit →
    /// publish). Without it two concurrent flushers race: the second `seal()`
    /// would overwrite the first's
    /// in-flight `flushing` buffer, and `send_replace` could publish a newer
    /// high-water seq before the older segment is durable. Distinct from `inner`
    /// (the short insertion lock) so appends are never blocked by a flush write.
    flush_lock: Mutex<()>,
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
    /// Configured remote store (`None` disables sync). The maintenance task's
    /// `sync_step` uploads L>=1 LOCAL segments here; eviction flips BOTH->REMOTE.
    remote: Option<RemoteStore>,
    persisted_seq: watch::Sender<i64>,
    /// Nudged by every append (and a durability await): "there may be data to
    /// flush". Drives the flush task's normal wake.
    flush_notify: Arc<Notify>,
    /// Nudged only when a buffer crosses `SEGMENT_TARGET_BYTES`: "flush now,
    /// don't wait out the rate cooldown". Lets a write burst bypass
    /// `MIN_FLUSH_INTERVAL` so RAM and L0 size stay bounded, while normal
    /// per-append nudges (which spam `flush_notify`) keep coalescing.
    force_flush: Arc<Notify>,
    stop: Arc<Notify>,
    /// Latched stop flag the background tasks check at the TOP of each loop
    /// iteration, in addition to selecting on the `stop` Notify. `Notify`
    /// stores no permit for `notify_waiters`, so a task that is mid-flush
    /// (off in `spawn_blocking`) when `stop` fires would otherwise re-subscribe
    /// after the wake and park forever, hanging the join. The latch closes that
    /// race: once set, the next loop iteration sees it and returns even if it
    /// missed the Notify wake. Set by `stop_and_join` / `request_stop`.
    stopped: AtomicBool,
    /// JoinHandles for the spawned per-namespace background tasks (flush +
    /// maintenance). Retained so `Store::shutdown` can cooperatively
    /// cancel (via the `stop` Notify) and JOIN them within a bounded timeout
    /// instead of busy-waiting. Pushed to by `spawn_flush_task` /
    /// `spawn_maintenance_task`; drained by [`shutdown`](Namespace::shutdown).
    task_handles: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

/// A namespace's sealed local segments as one consistent observation: the files a
/// scan may read, and the lowest `seq` any of them holds.
pub struct SegmentSnapshot {
    pub paths: Vec<String>,
    pub min_seq: Option<i64>,
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
    /// rename/unlink. `remote_log_dir` configures the offload target (empty
    /// disables sync). `storage_policy` is the per-namespace retention override.
    /// The per-namespace maintenance task and the boot remote reconcile are NOT
    /// started here — the caller calls [`spawn_maintenance`] once the store is
    /// fully built, and the task runs [`boot_reconcile`] in the background as its
    /// first step (or the caller reconciles synchronously for the runtime path).
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        name: &str,
        schema: Schema,
        data_dir: Option<PathBuf>,
        catalog: Arc<Catalog>,
        query_visibility: Arc<RwLock<()>>,
        remote_log_dir: &str,
        storage_policy: StoragePolicy,
    ) -> Result<Arc<Namespace>, StatsError> {
        let arrow_schema = schema_to_arrow(&schema);
        let key_column = if schema.key_column.is_empty() {
            None
        } else {
            Some(schema.key_column.clone())
        };

        let (next_seq, adopted, init_persisted) = match &data_dir {
            None => (1_i64, VecDeque::new(), -1_i64),
            Some(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| {
                    StatsError::Internal(format!("create namespace dir {}: {e}", dir.display()))
                })?;
                let adopted = adopt_local_segments(dir, key_column.as_deref(), &catalog, name);
                // Seed next_seq past every segment the catalog knows about, not
                // just on-disk footers. A segment evicted to remote has its local
                // parquet unlinked, so a footer-only scan under-counts and would
                // reuse live seqs (silent overwrite). Union the footer scan with
                // the full catalog (LOCAL, REMOTE, and BOTH rows).
                let next_seq = recover_next_seq(dir).max(crate::store::adopt::recover_next_seq(
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

        // The RemoteStore is rooted at the remote dir and composes the
        // namespace prefix internally, so we only need the dir to be configured.
        let remote = if data_dir.is_some() {
            build_remote_store(remote_log_dir)?
        } else {
            None
        };

        let (tx, _rx) = watch::channel(init_persisted);
        let ns = Arc::new(Namespace {
            name: name.to_string(),
            schema,
            arrow_schema: Arc::clone(&arrow_schema),
            key_column,
            data_dir,
            catalog: Arc::clone(&catalog),
            compaction_config: CompactionConfig::default(),
            inner: Mutex::new(NsInner {
                buffers: RamBuffers::new(arrow_schema, next_seq),
                local_segments: adopted.clone(),
                storage_policy,
            }),
            flush_lock: Mutex::new(()),
            maint_lock: tokio::sync::Mutex::new(()),
            query_visibility,
            remote,
            persisted_seq: tx,
            flush_notify: Arc::new(Notify::new()),
            force_flush: Arc::new(Notify::new()),
            stop: Arc::new(Notify::new()),
            stopped: AtomicBool::new(false),
            task_handles: Mutex::new(Vec::new()),
        });

        // Refresh the catalog from the adopted deque so the segments table
        // reflects on-disk reality after a fresh boot from a wiped catalog.
        for seg in &adopted {
            catalog.upsert_segment(&segment_to_row(name, seg))?;
        }

        if ns.data_dir.is_some() {
            let handle = spawn_flush_task(Arc::clone(&ns));
            ns.task_handles.lock().unwrap().push(handle);
        }
        Ok(ns)
    }

    /// Whether this namespace has a remote offload target configured.
    pub fn has_remote(&self) -> bool {
        self.remote.is_some()
    }

    /// Run the boot-time remote reconcile (adopt unknown remote parquet as
    /// REMOTE, redundancy-drop covered segments). No-op without a remote dir or
    /// in memory mode. Called once after construction, before serving.
    pub async fn boot_reconcile(&self) -> Result<(), StatsError> {
        let (Some(remote), Some(dir)) = (&self.remote, &self.data_dir) else {
            return Ok(());
        };
        reconcile_remote_segments(
            &self.catalog,
            remote,
            &self.name,
            dir,
            self.key_column.as_deref(),
        )
        .await?;
        // Reconcile may have adopted REMOTE-only segments the catalog did not
        // know at open() time (cold boot from a wiped catalog). Reseed next_seq
        // past them so freshly allocated seqs never collide with offloaded data.
        let target =
            crate::store::adopt::recover_next_seq(&self.catalog.list_segments(&self.name)?);
        self.inner
            .lock()
            .unwrap()
            .buffers
            .ensure_next_seq_at_least(target);
        Ok(())
    }

    /// Swap in a new retention policy (re-register). Picked up next eviction
    /// tick.
    pub fn update_policy(&self, policy: StoragePolicy) {
        self.inner.lock().unwrap().storage_policy = policy;
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

    /// Snapshot the sealed local segment paths and the lowest `seq` they hold, under
    /// one hold of the insertion lock so the two describe the same segment set.
    /// `min_seq` is `None` when no local segment exists (an empty namespace, or one
    /// whose segments have all been evicted to remote).
    ///
    /// Snapshotting under the lock is the read side of the query-visibility seam —
    /// compaction takes the write side before unlinking a file, so a query that
    /// captured the pre-compaction paths keeps scanning the files it snapshotted.
    ///
    /// Only SEALED segments appear: queries see flushed data, never the in-RAM buffer.
    pub fn query_snapshot(&self) -> SegmentSnapshot {
        let inner = self.inner.lock().unwrap();
        SegmentSnapshot {
            paths: inner
                .local_segments
                .iter()
                .map(|s| s.path.clone())
                .collect(),
            min_seq: inner.local_segments.iter().map(|s| s.min_seq).min(),
        }
    }

    /// Subscribe to the durability high-water mark. The current value is already
    /// marked seen, so a caller must read `borrow()` before awaiting `changed()`.
    pub fn watch_persisted_seq(&self) -> watch::Receiver<i64> {
        self.persisted_seq.subscribe()
    }

    /// Wake the flush task after an append. Nudges the rate-limited flush loop;
    /// a buffer that already holds a full segment (`>= SEGMENT_TARGET_BYTES`)
    /// also trips `force_flush` to bypass the cooldown and bound RAM / L0 size.
    /// No-op in memory mode, which has no flush task. Call after dropping the
    /// inner lock.
    fn notify_flush_after_append(&self, buffered_bytes: i64) {
        if self.data_dir.is_none() {
            return;
        }
        self.flush_notify.notify_one();
        if buffered_bytes >= SEGMENT_TARGET_BYTES {
            self.force_flush.notify_one();
        }
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
        self.flush_notify.notify_one();
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

    /// Write the sealed buffer to disk + catalog (no `persisted_seq` advance).
    fn write_sealed(&self, dir: &std::path::Path, sealed: &SealedBuffer) -> Result<(), StatsError> {
        let (path, size) = write_segment_to_dir(dir, 0, sealed.min_seq, &sealed.batch)?;
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
            location: SegmentLocation::Local,
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
        let Some(key) = &self.key_column else {
            return (None, None);
        };
        let Ok(idx) = batch.schema().index_of(key) else {
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
        let indexed = self.indexed_columns();
        let started = Instant::now();
        tracing::info!(
            namespace = %self.name,
            planned_inputs = job.inputs.len(),
            output_level = job.output_level,
            input_bytes = job.inputs.iter().map(|s| s.byte_size).sum::<i64>(),
            input_rows = job.inputs.iter().map(|s| s.row_count).sum::<i64>(),
            "compaction job starting"
        );
        let swap = run_job(
            job,
            dir,
            &self.arrow_schema,
            self.key_column.as_deref(),
            &indexed,
            self.compaction_config.max_merge_arrow_bytes,
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
        let Some(added) = swap.added.clone() else {
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
        };
        let output_path = added.path.clone();
        let output_bytes = added.size_bytes;
        let output_rows = added.row_count;
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
            output_path = %output_path,
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
            std::fs::rename(from, to).map_err(|e| {
                StatsError::Internal(format!(
                    "level-bump rename {} -> {} failed: {e}",
                    from.display(),
                    to.display()
                ))
            })?;
            // Carry the trigram sidecar with the segment it indexes (a bump is a
            // pure rename, no rewrite, so the index stays valid). Best-effort: a
            // missing sidecar (the bumped segment was never indexed) is fine.
            let (sidecar_from, sidecar_to) = (sidecar_path(from), sidecar_path(to));
            if sidecar_from.exists() {
                if let Err(e) = std::fs::rename(&sidecar_from, &sidecar_to) {
                    tracing::warn!(namespace = %self.name, from = %sidecar_from.display(), error = %e, "failed to carry trigram sidecar on level bump");
                    // The parquet at `from` is gone (renamed to `to`), so no later
                    // unlink or eviction will ever reach this sidecar — it would
                    // linger as a stale orphan forever. Drop it; the bumped segment
                    // then scans unpruned, which is correct.
                    let _ = std::fs::remove_file(&sidecar_from);
                }
            }
        }
        let removed_set: std::collections::HashSet<&str> =
            swap.removed.iter().map(|s| s.as_str()).collect();
        // A drop (missing head input) never reaches `commit_swap` — `run_one_job`
        // routes it through `evict_segment`. Every committed swap therefore
        // replaces its inputs with a real output segment.
        let added = swap
            .added
            .as_ref()
            .expect("commit_swap requires an output segment; drops are handled by run_one_job");
        let added_row = segment_to_row(&self.name, added);
        {
            let mut inner = self.inner.lock().unwrap();
            let mut new_segments: VecDeque<LocalSegment> =
                VecDeque::with_capacity(inner.local_segments.len());
            let mut inserted = false;
            for s in inner.local_segments.drain(..) {
                if removed_set.contains(s.path.as_str()) {
                    if !inserted {
                        new_segments.push_back(added.clone());
                        inserted = true;
                    }
                } else {
                    new_segments.push_back(s);
                }
            }
            if !inserted {
                new_segments.push_back(added.clone());
            }
            inner.local_segments = new_segments;
            debug_assert_unique_paths(&inner.local_segments);
            // Atomic catalog splice. Propagate on failure: the
            // deque now points at paths that exist on disk (the renamed bump
            // target / the already-written merged output), so a propagated error
            // is a stats/boot-adoption metadata inconsistency that self-heals at
            // next boot adoption — never a mid-scan-unlink hazard — and the merge
            // inputs below are left intact because we return before unlinking.
            self.catalog
                .replace_segments(&self.name, &swap.removed, &[added_row])?;
        }
        // 2) Unlink merged inputs after the swap (level bumps already renamed).
        if swap.unlink_removed {
            for path in &swap.removed {
                if let Err(e) = std::fs::remove_file(path) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::warn!(namespace = %self.name, path = %path, error = %e, "failed to unlink merged input");
                    }
                }
                // The merged output carries a freshly-built sidecar; the inputs'
                // sidecars are now stale and unlinked with their parquet.
                remove_sidecar(path);
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
    /// whose basename has no catalog row — those are compaction inputs whose row
    /// was dropped at commit. The ordering is the data-safety invariant: by the
    /// time phase 2 runs, the merged output subsuming those inputs is durable in
    /// the bucket (uploaded in phase 1), so the durable copy is in place before
    /// any input remote bytes are deleted. Skipping phase 2 on a failed upload
    /// means the only remaining copies of an unmerged seq range (the inputs in
    /// the bucket) are preserved.
    ///
    /// No-op without a remote dir / in memory mode.
    pub async fn sync_step(&self) -> Result<(), StatsError> {
        let Some(remote) = &self.remote else {
            return Ok(());
        };
        let remote_basenames: std::collections::HashSet<String> =
            match remote.list_basenames(&self.name).await {
                Ok(names) => names.into_iter().collect(),
                Err(e) => {
                    tracing::warn!(namespace = %self.name, error = %e, "remote sync list failed");
                    return Ok(());
                }
            };

        let rows = self.catalog.list_segments_min_level(&self.name, 1)?;
        let mut all_durable = true;
        for row in &rows {
            if row.location != SegmentLocation::Local {
                continue;
            }
            let base = basename(&row.path);
            if remote_basenames.contains(&base) {
                // Uploaded but the catalog never flipped — adopt, no re-upload.
                self.mark_uploaded(&row.path)?;
                continue;
            }
            if remote
                .upload(&self.name, std::path::Path::new(&row.path))
                .await
            {
                self.mark_uploaded(&row.path)?;
            } else {
                all_durable = false;
            }
        }

        if !all_durable {
            return Ok(());
        }

        // Re-snapshot the L>=1 catalog rows (phase 1 may have added basenames) and
        // delete only genuine orphans. min_level=1 is equivalent to scanning all
        // levels here because remote files are exclusively L>=1 (L0 is never
        // uploaded), so an L0 basename can never appear in `remote_basenames`.
        let catalog_basenames: std::collections::HashSet<String> = self
            .catalog
            .list_segments_min_level(&self.name, 1)?
            .iter()
            .map(|r| basename(&r.path))
            .collect();
        for base in remote_basenames.difference(&catalog_basenames) {
            remote.delete(&self.name, base).await;
            tracing::info!(namespace = %self.name, segment = %base, "deleted orphan remote segment");
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
        // The local trigram sidecar is local-only (never uploaded in v1), so it
        // is unlinked with the local parquet on eviction.
        remove_sidecar(path);
        removed_bytes
    }

    // ----- trigram sidecar backfill -------------------------------------

    /// Rebuild trigram sidecars for up to `max` local L>=1 segments missing one.
    ///
    /// Compaction only builds a sidecar for the segments it merges (see
    /// `executor::run_job`); L0 is intentionally unindexed, and a terminal-level
    /// segment that never re-merges — or any segment written before sidecars
    /// existed — stays without a `.tgm` and so scans `contains()`/`LIKE`
    /// unpruned. This background sweep closes that gap for already-durable data.
    ///
    /// Bounded per call and run as the lowest-priority maintenance step so the
    /// parquet reads + index build can't delay compaction/sync/eviction. A no-op
    /// for namespaces without the indexed string column. Best-effort per segment
    /// (a read/build failure only leaves that segment unpruned, never wrong),
    /// mirroring the compaction-time sidecar write. Returns the number rebuilt.
    ///
    /// Writes are atomic (`write_sidecar` renames into place), so a query that
    /// races the backfill sees either the old (absent) or the complete sidecar,
    /// never a partial one.
    fn backfill_missing_sidecars(&self, max: usize) -> usize {
        if self.data_dir.is_none() || max == 0 {
            return 0;
        }
        // Only namespaces with at least one indexed column benefit; skip the
        // parquet reads entirely otherwise.
        let indexed = self.indexed_columns();
        if indexed.is_empty() {
            return 0;
        }
        let candidates: Vec<String> = {
            let inner = self.inner.lock().unwrap();
            inner
                .local_segments
                .iter()
                .filter(|s| s.level >= 1)
                .map(|s| s.path.clone())
                .filter(|p| !sidecar_path(Path::new(p)).exists())
                .take(max)
                .collect()
        };
        let mut built = 0;
        for path in candidates {
            let p = Path::new(&path);
            let batches = match read_segment_batches(p) {
                Ok(batches) => batches,
                Err(e) => {
                    tracing::warn!(namespace = %self.name, path = %path, error = %e, "sidecar backfill: read failed");
                    continue;
                }
            };
            match write_sidecar(p, &batches, &indexed, self.key_column.as_deref()) {
                Ok(true) => {
                    built += 1;
                    tracing::debug!(namespace = %self.name, path = %path, "backfilled trigram sidecar");
                }
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!(namespace = %self.name, path = %path, error = %e, "sidecar backfill: write failed")
                }
            }
        }
        built
    }

    // ----- maintenance orchestration ------------------------------------

    /// Run one full maintenance cycle: `flush -> compact -> sync -> evict ->
    /// backfill sidecars`, serialized against other maintenance callers via
    /// `maint_lock`.
    ///
    /// Supports an optional forced L0->L1 (the debug `force_compact_l0` flag).
    /// The blocking compaction (read/merge/write +
    /// `commit_swap`, which takes `blocking_write`) runs under `spawn_blocking`;
    /// the async `sync_step` runs on the reactor; the blocking `eviction_step`
    /// (which takes `blocking_write` per evict) runs under `spawn_blocking`. No-op
    /// in memory mode.
    pub async fn run_maintenance(
        self: &Arc<Self>,
        force_compact_l0: bool,
    ) -> Result<(), StatsError> {
        if self.data_dir.is_none() {
            return Ok(());
        }
        let _maint_guard = self.maint_lock.lock().await;

        // Flush + compact (blocking parquet + commit_swap under blocking_write).
        let ns = Arc::clone(self);
        tokio::task::spawn_blocking(move || -> Result<(), StatsError> {
            ns.flush_once()?;
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
            while !ns.stopped.load(Ordering::SeqCst) && ns.compaction_step()? {}
            Ok(())
        })
        .await
        .map_err(|e| StatsError::Internal(format!("maintenance compact task panicked: {e}")))??;

        // Sync (async object_store).
        self.sync_step().await?;

        // Evict (blocking; evict_segment takes blocking_write per segment).
        let ns = Arc::clone(self);
        tokio::task::spawn_blocking(move || ns.eviction_step())
            .await
            .map_err(|e| StatsError::Internal(format!("maintenance evict task panicked: {e}")))??;

        // Backfill (blocking parquet reads + index build). Last + bounded so it is
        // the lowest-priority work: older/terminal segments compaction never
        // indexed get their trigram sidecars rebuilt a few per tick.
        let ns = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            ns.backfill_missing_sidecars(BACKFILL_SIDECARS_PER_TICK)
        })
        .await
        .map_err(|e| StatsError::Internal(format!("maintenance backfill task panicked: {e}")))?;
        Ok(())
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

    /// Spawn the per-namespace maintenance task.
    /// No-op in memory mode. Called once by the store after construction.
    ///
    /// When `reconcile_first` is set, the task runs the boot remote reconcile
    /// (adopt unknown remote parquet, redundancy-drop covered segments) as its
    /// FIRST step, in the background, before the periodic loop — so a large
    /// first-time reconcile never blocks the listener bind / `/health`. The boot
    /// path (rehydrated namespaces, already backed by local segments → next_seq
    /// recovered locally) passes `true`. The runtime-register path reconciles
    /// synchronously beforehand (cold-boot next_seq safety) and passes `false` so
    /// the task does not reconcile twice.
    pub fn spawn_maintenance(self: &Arc<Self>, reconcile_first: bool) {
        if self.data_dir.is_none() {
            return;
        }
        let handle = spawn_maintenance_task(Arc::clone(self), reconcile_first);
        self.task_handles.lock().unwrap().push(handle);
    }

    /// Aggregate in-RAM accounting for the diagnostics line:
    /// `(ram_bytes, chunk_count)` under the insertion lock.
    pub fn memory_summary(&self) -> (i64, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.buffers.ram_bytes(), inner.buffers.chunk_count())
    }

    /// Latch the stop flag, wake the flush + maintenance tasks, and JOIN them
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
    /// Stops + JOINs the flush + maintenance tasks (bounded by `timeout`), then
    /// does a final `flush_once` (no RAM-only rows survive; durability is already
    /// preserved — an acked write was on a sealed segment) and, for a
    /// remote-configured namespace, a final bounded `sync_step` so the bucket
    /// matches the catalog at shutdown.
    pub async fn shutdown(&self, timeout: Duration) {
        self.stop_and_join(timeout).await;
        // Final drain so no acked-but-still-RAM rows are lost (best-effort;
        // failures are already logged inside flush_once).
        let _ = self.flush_once();
        // Final reconcile so the bucket matches the catalog at shutdown.
        // Best-effort + bounded by the same per-namespace `timeout`; if it
        // doesn't finish, `boot_reconcile` re-syncs on the next start. No-op
        // (early return) without a remote dir.
        if self.has_remote() {
            let _ = tokio::time::timeout(timeout, self.sync_step()).await;
        }
    }

    /// Signal the flush + maintenance tasks to stop without awaiting them. Safe
    /// to call from a synchronous context with no tokio runtime — used by
    /// `drop_table` for mem-store namespaces, which spawn no background tasks (so
    /// there is nothing to join) before deleting their catalog rows.
    pub fn request_stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.stop.notify_waiters();
    }
}

fn basename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
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
        location: seg.location,
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
) -> VecDeque<LocalSegment> {
    let mut segs: Vec<LocalSegment> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let local_files: std::collections::HashMap<String, PathBuf> = discover_segments(dir)
        .into_iter()
        .map(|p| (p.to_string_lossy().into_owned(), p))
        .collect();

    // Pass 1: catalog rows.
    let catalog_rows = catalog.list_segments(namespace).unwrap_or_default();
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
        .filter(|r| {
            r.location != SegmentLocation::Local || local_files.contains_key(&r.path)
        })
        .map(|r| r.max_seq)
        .max();
    for row in &catalog_rows {
        seen.insert(row.path.clone());
        let Some(local_path) = local_files.get(&row.path) else {
            // Local file gone.
            match row.location {
                SegmentLocation::Local => {
                    // No durable copy — drop the row.
                    let _ = catalog.remove_segment(namespace, &row.path);
                }
                SegmentLocation::Both => {
                    // Bucket copy is durable; collapse to REMOTE.
                    let _ = catalog.set_location(namespace, &row.path, SegmentLocation::Remote);
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
            location,
        });
    }

    // Pass 2: local files with no catalog row -> fresh LOCAL segments.
    for (path_str, path) in &local_files {
        if seen.contains(path_str) {
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
            location: SegmentLocation::Local,
        });
    }

    segs.sort_by_key(|s| s.min_seq);
    let deque: VecDeque<LocalSegment> = segs.into();
    debug_assert_unique_paths(&deque);
    deque
}

/// Spawn the per-namespace flush task.
///
/// It wakes on a `Notify` (set by writers), a flush-interval tick, or when the
/// RAM buffer crosses the segment-target byte threshold, and drains the buffer
/// to a new L0 segment via the synchronous `flush_once` (which encodes parquet
/// under the tokio blocking pool implicitly — the encode is fast for the small
/// batches the durability path produces; large batches are bounded by the
/// 16MiB/1Mi write caps).
fn spawn_flush_task(ns: Arc<Namespace>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            // Stop latch checked before parking so a stop signalled while we
            // were mid-flush (and thus missed the Notify wake) still exits.
            if ns.stopped.load(Ordering::SeqCst) {
                let _ = ns.flush_once();
                return;
            }
            let notified = ns.flush_notify.notified();
            let stopped = ns.stop.notified();
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep(DEFAULT_FLUSH_INTERVAL) => {}
                _ = stopped => {
                    let _ = ns.flush_once();
                    return;
                }
            }
            if ns.stopped.load(Ordering::SeqCst) {
                let _ = ns.flush_once();
                return;
            }
            // Run the (blocking) parquet encode off the reactor.
            let ns2 = Arc::clone(&ns);
            let res = tokio::task::spawn_blocking(move || ns2.flush_once()).await;
            if let Ok(Err(e)) = res {
                tracing::warn!(namespace = %ns.name, error = %e, "flush task: flush_once failed");
            }
            // Flush-rate cooldown: coalesce the appends that arrive during the
            // window into the next single L0 (cap = one segment per
            // MIN_FLUSH_INTERVAL) instead of sealing a tiny L0 per nudge. A burst
            // that fills a whole segment cuts the wait short via `force_flush`.
            tokio::select! {
                _ = tokio::time::sleep(MIN_FLUSH_INTERVAL) => {}
                _ = ns.force_flush.notified() => {}
                _ = ns.stop.notified() => {
                    let _ = ns.flush_once();
                    return;
                }
            }
        }
    })
}

/// Spawn the per-namespace maintenance task.
///
/// When `reconcile_first` is set, the task FIRST runs the boot remote reconcile
/// (adopt unknown remote parquet, redundancy-drop covered segments), then enters
/// the periodic loop. Running it here — on the spawned task rather than before
/// the listener binds — keeps the reconcile's object_store footer reads off the
/// startup / `/health` path, while still sequencing it before the first
/// maintenance tick so the tick can never race adoption. A stop signalled during
/// the reconcile is honoured (the latch is checked before it runs, and the
/// in-flight reconcile future is cancelled when the task is aborted on shutdown).
///
/// Every `check_interval` the loop runs one `run_maintenance` cycle (compaction
/// drain, then sync, then evict). The cycle's heavy work is dispatched onto
/// `spawn_blocking` inside `run_maintenance`, so the reactor is never stalled. On
/// the stop notify the task exits immediately WITHOUT a final maintenance cycle;
/// the final drain-to-disk and reconcile on shutdown are handled by
/// [`Namespace::shutdown`].
fn spawn_maintenance_task(
    ns: Arc<Namespace>,
    reconcile_first: bool,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if reconcile_first && ns.has_remote() && !ns.stopped.load(Ordering::SeqCst) {
            if let Err(e) = ns.boot_reconcile().await {
                tracing::warn!(namespace = %ns.name, error = %e, "boot reconcile failed");
            }
        }
        let interval = ns.compaction_config.check_interval;
        loop {
            // Stop latch: a stop signalled while a maintenance cycle was running
            // (and thus missed the Notify wake) still exits here.
            if ns.stopped.load(Ordering::SeqCst) {
                return;
            }
            let stopped = ns.stop.notified();
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = stopped => {
                    return;
                }
            }
            if ns.stopped.load(Ordering::SeqCst) {
                return;
            }
            if let Err(e) = ns.run_maintenance(false).await {
                tracing::warn!(namespace = %ns.name, error = %e, "maintenance task: run_maintenance failed");
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::schema::{with_implicit_seq, Column, Schema};

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
    fn open_ns(
        name: &str,
        schema: Schema,
        data_dir: Option<PathBuf>,
        catalog: Arc<Catalog>,
    ) -> Arc<Namespace> {
        Namespace::open(
            name,
            schema,
            data_dir,
            catalog,
            Arc::new(RwLock::new(())),
            "",
            StoragePolicy::default(),
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
        Namespace::open(
            name,
            schema,
            data_dir,
            catalog,
            Arc::new(RwLock::new(())),
            remote_log_dir,
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
        ns.await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();

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
            ns.await_persisted(last, Duration::from_secs(10))
                .await
                .unwrap();
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
        write_segment_to_dir(dir, level, first_seq, &batch).unwrap().0
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

        let deque = adopt_local_segments(&ns_dir, Some("timestamp_ms"), &catalog, "iris.task");
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
        catalog.upsert_segment(&seg_row(&gone, 0, 100, 200)).unwrap();

        // A real on-disk orphan [1..2], lower seq than the stale row, no catalog
        // row. It must be recovered, not skipped as covered.
        let orphan = write_seg(&ns_dir, 0, 1, 2);

        let deque = adopt_local_segments(&ns_dir, Some("timestamp_ms"), &catalog, "iris.task");
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

    // --- trigram sidecar backfill ---------------------------------------

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
    async fn backfill_rebuilds_missing_trigram_sidecar() {
        let dir = tempdir();
        let ns_dir = dir.join("log.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns = open_ns("log.test", data_schema(), Some(ns_dir.clone()), catalog);

        // Two L0 flushes merged to one L1 — the merge builds the sidecar.
        ns.append_aligned_batch(&data_aligned(5, 0));
        ns.flush_once().unwrap();
        let last = ns.append_aligned_batch(&data_aligned(5, 5));
        ns.flush_once().unwrap();
        ns.await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        // run_maintenance wraps the merge in spawn_blocking (commit_swap takes the
        // blocking query-visibility lock); a multi-input merge builds the sidecar.
        ns.run_maintenance(true).await.unwrap();

        let segs = discover_segments(&ns_dir);
        assert_eq!(segs.len(), 1, "two L0 merged into one L1");
        let sidecar = sidecar_path(&segs[0]);
        assert!(sidecar.exists(), "the merge wrote a sidecar");

        // Simulate a segment compaction never indexed (single-input bump, or one
        // written before sidecars existed): drop the sidecar.
        std::fs::remove_file(&sidecar).unwrap();
        assert!(!sidecar.exists());

        // The backfill rebuilds exactly the one missing sidecar, then idles.
        assert_eq!(ns.backfill_missing_sidecars(10), 1);
        assert!(sidecar.exists(), "backfill rebuilt the sidecar");
        assert_eq!(ns.backfill_missing_sidecars(10), 0, "nothing left to do");

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

        assert_eq!(ns.backfill_missing_sidecars(10), 0);
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
        let first_l1: Vec<_> = std::fs::read_dir(&ns_dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("seg_L1_"))
            .map(|e| e.path())
            .collect();
        assert_eq!(first_l1.len(), 1);

        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap(); // L1 #2; cap=1 evicts oldest

        // Local L1 files: exactly one remains, and it is NOT the first one.
        let local_l1: Vec<_> = std::fs::read_dir(&ns_dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("seg_L1_"))
            .map(|e| e.path())
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
        let ns = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir.clone()),
            catalog,
            "",
            StoragePolicy {
                max_segments: Some(1),
                ..Default::default()
            },
        );
        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap();
        write_one(&ns).await;
        ns.run_maintenance(true).await.unwrap();
        let local_l1 = std::fs::read_dir(&ns_dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("seg_L1_"))
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

    #[tokio::test]
    async fn boot_reconcile_adopts_remote_as_remote_rows() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let ns_dir = dir.join("iris.worker");
        // First process: write, compact, upload (BOTH).
        {
            let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
            let ns = open_ns_remote(
                "iris.worker",
                worker_schema(),
                Some(ns_dir.clone()),
                catalog,
                remote.to_str().unwrap(),
                StoragePolicy::default(),
            );
            write_one(&ns).await;
            ns.run_maintenance(true).await.unwrap();
            assert_eq!(remote_files(&remote, "iris.worker").len(), 1);
        }
        // Wipe local catalog + parquet, keep the remote bucket.
        std::fs::remove_file(dir.join(crate::store::catalog::CATALOG_DB_FILENAME)).ok();
        std::fs::remove_dir_all(&ns_dir).ok();

        // Second process: fresh catalog, boot reconcile adopts the remote file.
        let catalog2 = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns2 = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir),
            catalog2,
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        ns2.boot_reconcile().await.unwrap();
        let segs = ns2.catalog.list_segments("iris.worker").unwrap();
        assert_eq!(segs.len(), 1, "remote file adopted as a catalog row");
        assert_eq!(segs[0].location, SegmentLocation::Remote);
        assert_eq!(segs[0].level, 1);
        // Remote file is NOT deleted by adoption.
        assert_eq!(remote_files(&remote, "iris.worker").len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn spawn_maintenance_reconciles_remote_in_background() {
        // The boot path defers remote reconcile onto the maintenance task
        // (reconcile_first=true) so it never blocks startup. Prove the task
        // actually performs the adoption: set up a remote-only segment (catalog +
        // local parquet wiped), `spawn_maintenance(true)` WITHOUT an explicit
        // boot_reconcile await, and assert the background task adopts it.
        let dir = tempdir();
        let remote = dir.join("remote");
        let ns_dir = dir.join("iris.worker");
        {
            let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
            let ns = open_ns_remote(
                "iris.worker",
                worker_schema(),
                Some(ns_dir.clone()),
                catalog,
                remote.to_str().unwrap(),
                StoragePolicy::default(),
            );
            write_one(&ns).await;
            ns.run_maintenance(true).await.unwrap();
            assert_eq!(remote_files(&remote, "iris.worker").len(), 1);
        }
        std::fs::remove_file(dir.join(crate::store::catalog::CATALOG_DB_FILENAME)).ok();
        std::fs::remove_dir_all(&ns_dir).ok();

        let catalog2 = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let ns2 = open_ns_remote(
            "iris.worker",
            worker_schema(),
            Some(ns_dir),
            catalog2,
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        // No explicit boot_reconcile: the background maintenance task must run it.
        assert!(
            ns2.catalog.list_segments("iris.worker").unwrap().is_empty(),
            "fresh catalog starts with no segment rows",
        );
        ns2.spawn_maintenance(true);
        // Poll (bounded) for the background reconcile to adopt the remote row.
        // The first periodic tick is check_interval (30s) away, so only the
        // reconcile can mutate the catalog within this window.
        let mut segs = Vec::new();
        for _ in 0..200 {
            segs = ns2.catalog.list_segments("iris.worker").unwrap();
            if !segs.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            segs.len(),
            1,
            "background reconcile adopted the remote segment"
        );
        assert_eq!(segs[0].location, SegmentLocation::Remote);
        ns2.shutdown(Duration::from_secs(2)).await;
        std::fs::remove_dir_all(&dir).ok();
    }
}
