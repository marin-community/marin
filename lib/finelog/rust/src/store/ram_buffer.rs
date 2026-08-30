//! In-RAM write state for a single namespace (LSM-style chunk list).
//!
//! Not thread-safe — the enclosing `DiskNamespace` serializes calls under its
//! insertion lock.
//!
//! The chunk list maintains the LSM invariant `chunks[i-1].rows > chunks[i].rows`
//! while adjacent chunks fit within one segment. Tail merges and seals never
//! concatenate more than `SEGMENT_TARGET_BYTES`, which stays well below Arrow's
//! 32-bit variable-width offset limit. Byte/row accounting is O(1): the caller
//! supplies `added_bytes` (the `AlignedBatch.byte_size` plus 8 bytes per row for
//! the stamped `seq` column) so the hot path never walks the batch buffers.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;

use crate::store::schema::{AlignedBatch, IMPLICIT_SEQ_COLUMN};

/// Buffered-byte target for one L0 segment and one Arrow concatenation.
pub const SEGMENT_TARGET_BYTES: i64 = 100 * 1024 * 1024;

#[derive(Debug)]
struct BufferedChunk {
    batch: RecordBatch,
    nbytes: i64,
}

/// An immutable, in-flight flush buffer.
///
/// `nbytes`/`num_rows` are carried over from `RamBuffers` accounting so seal-time
/// does not walk the batch buffers. `min_seq`/`max_seq` are the seq-column bounds.
#[derive(Debug, Clone)]
pub struct SealedBuffer {
    pub batch: RecordBatch,
    pub nbytes: i64,
    pub num_rows: i64,
    pub min_seq: i64,
    pub max_seq: i64,
}

/// Owns the in-RAM write state for a single namespace.
pub struct RamBuffers {
    arrow_schema: SchemaRef,
    chunks: Vec<BufferedChunk>,
    flushing: Option<SealedBuffer>,
    next_seq: i64,
    ram_bytes: i64,
    ram_rows: i64,
}

impl RamBuffers {
    pub fn new(arrow_schema: SchemaRef, next_seq: i64) -> RamBuffers {
        RamBuffers {
            arrow_schema,
            chunks: Vec::new(),
            flushing: None,
            next_seq,
            ram_bytes: 0,
            ram_rows: 0,
        }
    }

    pub fn next_seq(&self) -> i64 {
        self.next_seq
    }

    /// Reserve `count` seqs, returning the first; advances `next_seq`.
    pub fn allocate_seq(&mut self, count: i64) -> i64 {
        let first = self.next_seq;
        self.next_seq += count;
        first
    }

    /// Raise `next_seq` to at least `seq` (never lowers it). Used at cold boot
    /// to seed the counter past segments adopted from the catalog — including
    /// REMOTE-only rows that have no on-disk parquet to scan.
    pub fn ensure_next_seq_at_least(&mut self, seq: i64) {
        self.next_seq = self.next_seq.max(seq);
    }

    /// Append `batch` (already seq-stamped) to the chunk list; the caller
    /// supplies `added_bytes` (so the hot path never walks the batch buffers).
    pub fn append_batch(&mut self, batch: RecordBatch, added_bytes: i64) {
        let rows = batch.num_rows() as i64;
        self.chunks.push(BufferedChunk {
            batch,
            nbytes: added_bytes,
        });
        maintain_chunk_invariant(&mut self.chunks, &self.arrow_schema);
        self.ram_bytes += added_bytes;
        self.ram_rows += rows;
    }

    pub fn ram_bytes(&self) -> i64 {
        self.ram_bytes + self.flushing.as_ref().map_or(0, |f| f.nbytes)
    }

    pub fn ram_rows(&self) -> i64 {
        self.ram_rows + self.flushing.as_ref().map_or(0, |f| f.num_rows)
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn has_chunks(&self) -> bool {
        !self.chunks.is_empty()
    }

    /// Move up to one segment target of accumulated chunks into a sealed buffer.
    ///
    /// A single incoming chunk may exceed the target and is sealed alone. The
    /// sealed buffer is also stored on `self.flushing` so `ram_bytes`/`ram_rows`
    /// keep counting it until the flush commits.
    pub fn seal(&mut self) -> Option<SealedBuffer> {
        if self.chunks.is_empty() {
            return None;
        }

        let mut selected_count = 1;
        let mut sealed_bytes = self.chunks[0].nbytes;
        let mut sealed_rows = self.chunks[0].batch.num_rows() as i64;
        while selected_count < self.chunks.len() {
            let next = &self.chunks[selected_count];
            let Some(combined_bytes) = sealed_bytes.checked_add(next.nbytes) else {
                break;
            };
            if combined_bytes > SEGMENT_TARGET_BYTES {
                break;
            }
            sealed_bytes = combined_bytes;
            sealed_rows += next.batch.num_rows() as i64;
            selected_count += 1;
        }

        let visible = if selected_count == 1 {
            self.chunks[0].batch.clone()
        } else {
            let tables: Vec<RecordBatch> = self.chunks[..selected_count]
                .iter()
                .map(|chunk| chunk.batch.clone())
                .collect();
            match concat_batches(&self.arrow_schema, &tables) {
                Ok(batch) => batch,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        chunks = selected_count,
                        bytes = sealed_bytes,
                        "failed to concatenate sealed RAM chunks; sealing oldest chunk"
                    );
                    selected_count = 1;
                    sealed_bytes = self.chunks[0].nbytes;
                    sealed_rows = self.chunks[0].batch.num_rows() as i64;
                    self.chunks[0].batch.clone()
                }
            }
        };
        self.chunks.drain(..selected_count);
        self.ram_bytes -= sealed_bytes;
        self.ram_rows -= sealed_rows;
        let seq_idx = self
            .arrow_schema
            .index_of(IMPLICIT_SEQ_COLUMN)
            .expect("stored schema carries the implicit seq column");
        let seq_col = visible
            .column(seq_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("seq column is Int64");
        let (min_seq, max_seq) = seq_minmax(seq_col);
        let sealed = SealedBuffer {
            batch: visible,
            nbytes: sealed_bytes,
            num_rows: sealed_rows,
            min_seq,
            max_seq,
        };
        self.flushing = Some(sealed.clone());
        Some(sealed)
    }

    /// Drop the in-flight flushing buffer (parquet write succeeded).
    pub fn commit_flush(&mut self) {
        self.flushing = None;
    }

    /// Push the in-flight buffer back to the head of chunks (write failed).
    pub fn restore_flush(&mut self) {
        if let Some(f) = self.flushing.take() {
            self.ram_bytes += f.nbytes;
            self.ram_rows += f.num_rows;
            self.chunks.insert(
                0,
                BufferedChunk {
                    batch: f.batch,
                    nbytes: f.nbytes,
                },
            );
        }
    }
}

fn seq_minmax(seq: &Int64Array) -> (i64, i64) {
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for i in 0..seq.len() {
        let v = seq.value(i);
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    (min, max)
}

/// Restore the LSM row-count invariant while merged chunks fit one segment.
///
/// Called after each append. Only the tail can violate the invariant, so we
/// cascade-merge from the tail until the previous chunk is strictly larger than
/// the last or merging would cross the segment target. A failed optional merge
/// leaves both original chunks intact for bounded sealing.
fn maintain_chunk_invariant(chunks: &mut Vec<BufferedChunk>, arrow_schema: &SchemaRef) {
    while chunks.len() >= 2
        && chunks[chunks.len() - 2].batch.num_rows() <= chunks[chunks.len() - 1].batch.num_rows()
    {
        let len = chunks.len();
        let Some(merged_bytes) = chunks[len - 2].nbytes.checked_add(chunks[len - 1].nbytes) else {
            break;
        };
        if merged_bytes > SEGMENT_TARGET_BYTES {
            break;
        }
        let tables = [chunks[len - 2].batch.clone(), chunks[len - 1].batch.clone()];
        let merged = match concat_batches(arrow_schema, &tables) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    %error,
                    bytes = merged_bytes,
                    "failed to coalesce RAM chunks; preserving separate chunks"
                );
                break;
            }
        };
        chunks.pop();
        chunks.pop();
        chunks.push(BufferedChunk {
            batch: merged,
            nbytes: merged_bytes,
        });
    }
}

/// Build the seq-stamped `RecordBatch` from `aligned` in registered column order.
///
/// The `seq` column is `Int64 [first..first+n)`; other columns come from
/// `aligned` in registered order. Any column declared by
/// `arrow_schema` but absent from `aligned` (the benign additive-evolution race —
/// the writer validated against schema v, an additive evolution landed before the
/// namespace took its lock) is NULL-filled.
pub fn stamp_seq_and_build(
    aligned: &AlignedBatch,
    first_seq: i64,
    arrow_schema: &SchemaRef,
) -> RecordBatch {
    let n = aligned.num_rows;
    let seq_array: Int64Array = (first_seq..first_seq + n as i64).collect();
    let seq_ref = Arc::new(seq_array) as arrow::array::ArrayRef;

    // Fast path: aligned columns line up 1:1 with the non-seq schema columns.
    if aligned.fields.len() + 1 == arrow_schema.fields().len() {
        let mut out: Vec<arrow::array::ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
        let mut ai = 0usize;
        let mut matched = true;
        for field in arrow_schema.fields() {
            if field.name() == IMPLICIT_SEQ_COLUMN {
                out.push(Arc::clone(&seq_ref));
            } else if ai < aligned.fields.len() && aligned.fields[ai].name() == field.name() {
                out.push(Arc::clone(&aligned.arrays[ai]));
                ai += 1;
            } else {
                matched = false;
                break;
            }
        }
        if matched && ai == aligned.fields.len() {
            return RecordBatch::try_new(Arc::clone(arrow_schema), out)
                .expect("stamped batch matches the stored schema");
        }
    }

    // Slow path: NULL-fill any column present in the schema but absent from aligned.
    let mut out: Vec<arrow::array::ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    for field in arrow_schema.fields() {
        if field.name() == IMPLICIT_SEQ_COLUMN {
            out.push(Arc::clone(&seq_ref));
            continue;
        }
        match aligned.fields.iter().position(|f| f.name() == field.name()) {
            Some(idx) => out.push(Arc::clone(&aligned.arrays[idx])),
            None => out.push(arrow::array::new_null_array(field.data_type(), n)),
        }
    }
    RecordBatch::try_new(Arc::clone(arrow_schema), out)
        .expect("stamped batch matches the stored schema")
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::schema::{schema_to_arrow, with_implicit_seq, Column, Schema};

    fn worker_stored_schema() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ))
    }

    fn worker_arrow() -> SchemaRef {
        schema_to_arrow(&worker_stored_schema())
    }

    /// Build an AlignedBatch of `n` rows for the worker schema (non-seq cols).
    fn worker_aligned(n: i64) -> AlignedBatch {
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

    fn stamp_n(buffers: &mut RamBuffers, schema: &SchemaRef, n: i64) -> i64 {
        let aligned = worker_aligned(n);
        let first = buffers.allocate_seq(n);
        let stamped = stamp_seq_and_build(&aligned, first, schema);
        buffers.append_batch(stamped, aligned.byte_size + 8 * n);
        first + n - 1
    }

    #[test]
    fn stamp_seq_is_contiguous_int64_in_registered_order() {
        let schema = worker_arrow();
        let aligned = worker_aligned(3);
        let stamped = stamp_seq_and_build(&aligned, 5, &schema);
        // Registered order: seq, worker_id, timestamp_ms.
        let stamped_schema = stamped.schema();
        let names: Vec<&str> = stamped_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["seq", "worker_id", "timestamp_ms"]);
        let seq = stamped
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(seq.values(), &[5_i64, 6, 7]);
    }

    #[test]
    fn stamp_null_fills_an_evolved_column() {
        // Stored schema gains a nullable `note`; aligned predates it.
        let evolved = with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("note", ColumnType::COLUMN_TYPE_STRING, true),
            ],
            "",
        ));
        let arrow_schema = schema_to_arrow(&evolved);
        let aligned = worker_aligned(2); // no `note`
        let stamped = stamp_seq_and_build(&aligned, 1, &arrow_schema);
        assert_eq!(stamped.num_columns(), 4);
        let note_idx = stamped.schema().index_of("note").unwrap();
        assert_eq!(stamped.column(note_idx).null_count(), 2);
    }

    #[test]
    fn append_maintains_strict_decreasing_chunk_invariant() {
        let schema = worker_arrow();
        let mut buffers = RamBuffers::new(Arc::clone(&schema), 1);
        // Append a sequence of equal-and-varying sizes; after each append the
        // invariant chunks[i-1].rows > chunks[i].rows must hold.
        for n in [3_i64, 3, 1, 5, 2] {
            stamp_n(&mut buffers, &schema, n);
            assert_invariant(&buffers);
        }
    }

    fn assert_invariant(buffers: &RamBuffers) {
        for w in buffers.chunks.windows(2) {
            assert!(
                w[0].batch.num_rows() > w[1].batch.num_rows(),
                "chunk invariant violated: {} <= {}",
                w[0].batch.num_rows(),
                w[1].batch.num_rows()
            );
        }
    }

    #[test]
    fn accounting_carries_through_seal_and_restore() {
        let schema = worker_arrow();
        let mut buffers = RamBuffers::new(Arc::clone(&schema), 1);
        stamp_n(&mut buffers, &schema, 3);
        stamp_n(&mut buffers, &schema, 2);
        // 3 rows then 2 rows -> 5 rows; bytes = (16*3 + 24) + (16*2 + 16).
        assert_eq!(buffers.ram_rows(), 5);
        let bytes_before = buffers.ram_bytes();
        assert_eq!(bytes_before, (16 * 3 + 24) + (16 * 2 + 16));

        let sealed = buffers.seal().unwrap();
        assert_eq!(sealed.num_rows, 5);
        assert_eq!(sealed.min_seq, 1);
        assert_eq!(sealed.max_seq, 5);
        // While flushing, ram_rows/ram_bytes still count the sealed buffer.
        assert_eq!(buffers.ram_rows(), 5);
        assert_eq!(buffers.ram_bytes(), bytes_before);

        buffers.restore_flush();
        assert_eq!(buffers.ram_rows(), 5);
        assert_eq!(buffers.ram_bytes(), bytes_before);
        assert!(buffers.has_chunks());

        // Re-seal and commit -> flushing cleared, buffer empty.
        buffers.seal().unwrap();
        buffers.commit_flush();
        assert_eq!(buffers.ram_rows(), 0);
        assert_eq!(buffers.ram_bytes(), 0);
        assert!(!buffers.has_chunks());
    }

    #[test]
    fn seal_splits_chunks_that_exceed_the_segment_target() {
        let schema = worker_arrow();
        let mut buffers = RamBuffers::new(Arc::clone(&schema), 1);
        let chunk_bytes = SEGMENT_TARGET_BYTES / 2 + 1;

        for _ in 0..2 {
            let aligned = worker_aligned(1);
            let first = buffers.allocate_seq(1);
            let stamped = stamp_seq_and_build(&aligned, first, &schema);
            buffers.append_batch(stamped, chunk_bytes);
        }

        let first = buffers.seal().unwrap();
        assert_eq!(first.num_rows, 1);
        assert_eq!((first.min_seq, first.max_seq), (1, 1));
        assert_eq!(first.nbytes, chunk_bytes);
        assert_eq!(buffers.ram_rows(), 2);
        assert_eq!(buffers.ram_bytes(), 2 * chunk_bytes);

        buffers.restore_flush();
        let retried = buffers.seal().unwrap();
        assert_eq!((retried.min_seq, retried.max_seq), (1, 1));
        buffers.commit_flush();

        let second = buffers.seal().unwrap();
        assert_eq!(second.num_rows, 1);
        assert_eq!((second.min_seq, second.max_seq), (2, 2));
        assert_eq!(second.nbytes, chunk_bytes);
        buffers.commit_flush();

        assert_eq!(buffers.ram_rows(), 0);
        assert_eq!(buffers.ram_bytes(), 0);
    }

    #[test]
    fn seal_empty_is_none() {
        let schema = worker_arrow();
        let mut buffers = RamBuffers::new(schema, 1);
        assert!(buffers.seal().is_none());
    }

    #[test]
    fn allocate_seq_advances_monotonically() {
        let schema = worker_arrow();
        let mut buffers = RamBuffers::new(schema, 10);
        assert_eq!(buffers.allocate_seq(3), 10);
        assert_eq!(buffers.allocate_seq(2), 13);
        assert_eq!(buffers.next_seq(), 15);
    }
}
