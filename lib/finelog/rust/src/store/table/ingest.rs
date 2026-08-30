//! The short-lock ingest buffer and sequence allocator.
//!
//! This is the write fast path. An append takes one short lock to allocate
//! sequence numbers and push a batch into RAM; it never waits on durable I/O or
//! on the table controller. Durability is published separately through a
//! `watch` channel: the flush pipeline seals a buffer, makes it durable, and
//! only then advances the high-water mark this exposes.
//!
//! A memory-mode table has no flush destination, so an append is durable the
//! instant it lands and the high-water mark advances under the append lock.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::datatypes::SchemaRef;
use tokio::sync::{watch, Notify};

use crate::errors::StatsError;
use crate::store::ram_buffer::{stamp_seq_and_build, RamBuffers, SealedBuffer};
use crate::store::schema::AlignedBatch;

/// Default durability-await budget when the RPC carries no deadline.
pub const DEFAULT_PERSIST_TIMEOUT: Duration = Duration::from_secs(30);

/// Hard cap on one table's raw Arrow buffer bytes. When persistence stalls the
/// force-flush threshold stops draining the buffer, so writes beyond this cap
/// are rejected instead of accumulating until an Arrow offset overflows.
pub const MAX_TABLE_RAM_BYTES: i64 = 2 * crate::store::table::runtime_policy::SEGMENT_TARGET_BYTES;

/// What the maintenance scheduler needs to know about a buffer to time its next
/// flush.
pub struct FlushDemand {
    /// An append has arrived since the last flush.
    pub requested: bool,
    /// The buffer already holds a whole segment, so the coalescing window is
    /// bypassed.
    pub forced: bool,
    /// The definition's maximum buffer age.
    pub max_flush_age: Duration,
}

/// Aggregate RAM accounting read under one hold of the append lock.
pub struct BufferedRows {
    pub rows: i64,
    pub bytes: i64,
    pub chunks: usize,
    pub next_seq: i64,
}

/// One table's RAM buffer, sequence allocator, and durability high-water mark.
pub struct IngestBuffer {
    table: String,
    arrow_schema: SchemaRef,
    buffers: Mutex<RamBuffers>,
    persisted_seq: watch::Sender<i64>,
    /// Latched by every append (and a durability await): "there may be data to
    /// flush". The scheduler reads and clears it.
    flush_requested: AtomicBool,
    /// Latched only when a buffer crosses the definition's maximum buffer size:
    /// "flush now, don't wait out the coalescing window". Lets a write burst
    /// bypass the flush-rate cap so RAM and L0 size stay bounded, while ordinary
    /// per-append demand keeps coalescing.
    flush_forced: AtomicBool,
    /// The scheduler's wake signal, shared by every table in the store. Latching
    /// flush demand signals it so flush latency does not wait out the poll
    /// interval.
    wake: Arc<Notify>,
    /// A memory-mode table never flushes: its rows are durable when they land.
    durable_on_append: bool,
}

impl IngestBuffer {
    pub fn new(
        table: &str,
        arrow_schema: SchemaRef,
        next_seq: i64,
        initial_persisted_seq: i64,
        durable_on_append: bool,
        wake: Arc<Notify>,
    ) -> Self {
        let (persisted_seq, _receiver) = watch::channel(initial_persisted_seq);
        Self {
            table: table.to_string(),
            arrow_schema: Arc::clone(&arrow_schema),
            buffers: Mutex::new(RamBuffers::new(arrow_schema, next_seq)),
            persisted_seq,
            flush_requested: AtomicBool::new(false),
            flush_forced: AtomicBool::new(false),
            wake,
            durable_on_append,
        }
    }

    /// Stamp `seq` onto `aligned` and append it, returning the last allocated
    /// seq (or `-1` if empty). Rejects a flushed table's write that would exceed
    /// the RAM limit.
    pub fn append_aligned(
        &self,
        aligned: &AlignedBatch,
        max_buffer_bytes: i64,
    ) -> Result<i64, StatsError> {
        if aligned.num_rows == 0 {
            return Ok(-1);
        }
        let rows = aligned.num_rows as i64;
        let added_bytes = aligned.byte_size + 8 * rows;
        let (last_seq, buffered_bytes) = {
            let mut buffers = self.buffers.lock().unwrap();
            self.ensure_append_capacity(&buffers, added_bytes)?;
            let first_seq = buffers.allocate_seq(rows);
            let stamped = stamp_seq_and_build(aligned, first_seq, &self.arrow_schema);
            buffers.append_batch(stamped, added_bytes);
            let last_seq = first_seq + rows - 1;
            if self.durable_on_append {
                // No parquet: the rows are durable the instant they land in RAM,
                // so advance the high-water mark under the append lock.
                self.persisted_seq.send_replace(last_seq);
            }
            (last_seq, buffers.ram_bytes())
        };
        self.request_flush(buffered_bytes >= max_buffer_bytes);
        Ok(last_seq)
    }

    /// Append already-built log columns (`seq` excluded), returning the last
    /// seq. Rejects a flushed table's write that would exceed the RAM limit.
    ///
    /// `columns` are the non-seq log columns in registered order, prepared by the
    /// caller OUTSIDE the lock. `num_rows` is their common length and
    /// `added_bytes` their raw buffer size.
    pub fn append_columns(
        &self,
        columns: Vec<ArrayRef>,
        num_rows: usize,
        added_bytes: i64,
        max_buffer_bytes: i64,
    ) -> Result<i64, StatsError> {
        if num_rows == 0 {
            return Ok(-1);
        }
        let rows = num_rows as i64;
        let added_bytes = added_bytes + 8 * rows;
        let (last_seq, buffered_bytes) = {
            let mut buffers = self.buffers.lock().unwrap();
            self.ensure_append_capacity(&buffers, added_bytes)?;
            let first_seq = buffers.allocate_seq(rows);
            let seq_array: Int64Array = (first_seq..first_seq + rows).collect();
            let mut all: Vec<ArrayRef> = Vec::with_capacity(columns.len() + 1);
            all.push(Arc::new(seq_array));
            all.extend(columns);
            let batch = RecordBatch::try_new(Arc::clone(&self.arrow_schema), all)
                .expect("log columns match the stored log schema");
            buffers.append_batch(batch, added_bytes);
            let last_seq = first_seq + rows - 1;
            if self.durable_on_append {
                self.persisted_seq.send_replace(last_seq);
            }
            (last_seq, buffers.ram_bytes())
        };
        self.request_flush(buffered_bytes >= max_buffer_bytes);
        Ok(last_seq)
    }

    /// A memory-mode table's rows never wait on a flush, so its buffer is the
    /// table itself and stays exempt from the cap.
    fn ensure_append_capacity(
        &self,
        buffers: &RamBuffers,
        added_bytes: i64,
    ) -> Result<(), StatsError> {
        if self.durable_on_append
            || buffers.ram_bytes().saturating_add(added_bytes) <= MAX_TABLE_RAM_BYTES
        {
            return Ok(());
        }
        Err(StatsError::ResourceExhausted(format!(
            "table {:?} has reached its {MAX_TABLE_RAM_BYTES}-byte ingest limit",
            self.table
        )))
    }

    /// Move the buffered chunks into the in-flight slot for one flush, or `None`
    /// when there is nothing buffered. The caller serializes seals.
    pub fn seal(&self) -> Option<SealedBuffer> {
        self.buffers.lock().unwrap().seal()
    }

    /// Return a failed flush's rows to the buffer.
    pub fn restore_sealed(&self) {
        self.buffers.lock().unwrap().restore_flush();
    }

    /// Discard the in-flight slot after its rows reached durable state.
    pub fn commit_sealed(&self) {
        self.buffers.lock().unwrap().commit_flush();
    }

    /// Publish a new durability high-water mark.
    pub fn publish_persisted(&self, seq: i64) {
        self.persisted_seq.send_replace(seq);
    }

    /// Raise the sequence allocator to at least `next_seq`; never lowers it.
    ///
    /// Recovery calls this with the claimed durable state's high-water mark so
    /// a projection that carries fewer rows than the table ever allocated (for
    /// example after a legacy import that leaves archive-only rows behind)
    /// cannot cause sequence reuse.
    pub fn raise_next_seq_floor(&self, next_seq: i64) {
        self.buffers
            .lock()
            .unwrap()
            .ensure_next_seq_at_least(next_seq);
    }

    /// Subscribe to the durability high-water mark. The current value is already
    /// marked seen, so a caller must read `borrow()` before awaiting `changed()`.
    pub fn watch_persisted(&self) -> watch::Receiver<i64> {
        self.persisted_seq.subscribe()
    }

    pub fn buffered(&self) -> BufferedRows {
        let buffers = self.buffers.lock().unwrap();
        BufferedRows {
            rows: buffers.ram_rows(),
            bytes: buffers.ram_bytes(),
            chunks: buffers.chunk_count(),
            next_seq: buffers.next_seq(),
        }
    }

    /// Record flush demand and wake the scheduler. `forced` bypasses the
    /// coalescing window the way a buffer holding a whole segment does. A
    /// memory-mode table never flushes, so its demand is not recorded.
    pub fn request_flush(&self, forced: bool) {
        if self.durable_on_append {
            return;
        }
        self.flush_requested.store(true, Ordering::SeqCst);
        if forced {
            self.flush_forced.store(true, Ordering::SeqCst);
        }
        self.wake.notify_one();
    }

    pub fn demand(&self, max_flush_age: Duration) -> FlushDemand {
        FlushDemand {
            requested: self.flush_requested.load(Ordering::SeqCst),
            forced: self.flush_forced.load(Ordering::SeqCst),
            max_flush_age,
        }
    }

    /// Clear the demand the scheduler is about to satisfy. Called immediately
    /// before the flush runs, so an append landing during it re-arms.
    pub fn clear_demand(&self) {
        self.flush_requested.store(false, Ordering::SeqCst);
        self.flush_forced.store(false, Ordering::SeqCst);
    }

    /// Block until `target` is durable, bounded by `timeout`.
    ///
    /// `target < 0` returns immediately. Otherwise subscribe to the high-water
    /// mark, nudge the scheduler, and wait, returning `DeadlineExceeded` (mapped
    /// to a 504) on timeout.
    pub async fn await_persisted(&self, target: i64, timeout: Duration) -> Result<(), StatsError> {
        if target < 0 {
            return Ok(());
        }
        let mut receiver = self.persisted_seq.subscribe();
        if *receiver.borrow() >= target {
            return Ok(());
        }
        self.request_flush(false);
        let wait = async {
            loop {
                if *receiver.borrow() >= target {
                    return;
                }
                // `changed()` errors only if the sender dropped; the buffer owns
                // the sender for its whole lifetime, so this cannot happen.
                if receiver.changed().await.is_err() {
                    return;
                }
            }
        };
        match tokio::time::timeout(timeout, wait).await {
            Ok(()) if *self.persisted_seq.borrow() >= target => Ok(()),
            Ok(()) => Err(StatsError::Internal(format!(
                "table {:?} persisted_seq channel closed before seq>={target}",
                self.table
            ))),
            Err(_elapsed) => Err(StatsError::DeadlineExceeded(format!(
                "timed out waiting for table {:?} to persist seq>={target}",
                self.table
            ))),
        }
    }
}
