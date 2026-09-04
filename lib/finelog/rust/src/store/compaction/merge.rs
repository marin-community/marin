//! Native arrow k-way merge of N segments by configured sort columns plus `seq`.
//!
//! Each input batch is already internally sorted on the sort keys (every segment
//! is written sorted by L0->L1 compaction, and L0 inputs to a force-compact are
//! seq-monotonic), so we MERGE rather than re-sort: encode the sort-key columns of
//! each batch into the comparable `arrow::row` byte form, keep a per-batch cursor,
//! repeatedly pop the globally-min current row from a `BinaryHeap`, and gather all
//! output columns via `arrow::compute::interleave` in 16384-row chunks (row-group
//! aligned).
//!
//! `seq` is the unique monotonic tiebreaker, so the merge is stable and
//! order-independent regardless of input file order.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, ArrayRef, RecordBatch};
use arrow::compute::{cast, interleave, SortOptions};
use arrow::datatypes::{DataType, Schema as ArrowSchema, SchemaRef};
use arrow::error::ArrowError;
use arrow::row::{Row, RowConverter, Rows, SortField};

/// Rows per output batch from the k-way merge. This is a batching decision only:
/// the writer accumulates across batches and cuts row groups at its own
/// byte-derived stride (see
/// [`crate::store::segment::segment_writer_properties_with_max_rows`]).
const MERGE_CHUNK_ROWS: usize = 16_384;

/// Project `batch` onto `target_schema`, additive-null-filling any target column
/// absent from the batch.
///
/// A segment written before an additive schema evolution lacks the new (nullable)
/// columns, so they are materialized as null arrays of the target type. Columns
/// are reordered to match `target_schema`. A column whose physical type differs
/// from the target only by timestamp unit is cast to the target (a legacy
/// microsecond segment already equals the microsecond storage canonical; a stray
/// millisecond segment from an older build is up-cast, so the merge output is
/// uniformly canonical). Any other type difference is a non-additive schema
/// conflict the register path already rejects and surfaces as an `ArrowError`.
pub fn project_to_schema(
    batch: &RecordBatch,
    target_schema: &SchemaRef,
) -> Result<RecordBatch, ArrowError> {
    let n = batch.num_rows();
    let src_schema = batch.schema();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        match src_schema.index_of(field.name()) {
            Ok(idx) => {
                let col = batch.column(idx);
                if col.data_type() == field.data_type() {
                    columns.push(Arc::clone(col));
                } else if matches!(col.data_type(), DataType::Timestamp(_, _))
                    && matches!(field.data_type(), DataType::Timestamp(_, _))
                {
                    // The only legitimate physical-type difference is a timestamp
                    // unit (a legacy microsecond segment is already equal to the
                    // microsecond storage canonical; a stray millisecond segment
                    // from an older build is up-cast). Reconcile by casting.
                    columns.push(cast(col, field.data_type())?);
                } else {
                    return Err(ArrowError::SchemaError(format!(
                        "column {:?}: type mismatch projecting to merge schema: \
                         input={:?} target={:?}",
                        field.name(),
                        col.data_type(),
                        field.data_type()
                    )));
                }
            }
            Err(_) => columns.push(new_null_array(field.data_type(), n)),
        }
    }
    RecordBatch::try_new(Arc::clone(target_schema), columns)
}

/// The merge sort-key collation: ascending, NULLS LAST, shared by the run
/// encoding and the merge so a key column with NULLs lands null-key rows at the
/// END in both paths, giving a consistent physical segment layout.
const MERGE_SORT_OPTIONS: SortOptions = SortOptions {
    descending: false,
    nulls_first: false,
};

/// Build the row converter every run and merge over `sort_cols` shares.
///
/// One conversion of each batch's sort keys serves both the per-batch sort and
/// the k-way merge, and using one converter guarantees the two compare rows
/// identically.
pub fn merge_row_converter(
    schema: &ArrowSchema,
    sort_cols: &[usize],
) -> Result<RowConverter, ArrowError> {
    let fields: Vec<SortField> = sort_cols
        .iter()
        .map(|&i| {
            SortField::new_with_options(schema.field(i).data_type().clone(), MERGE_SORT_OPTIONS)
        })
        .collect();
    RowConverter::new(fields)
}

/// One internally-sorted merge input: a batch, the row encoding of its sort
/// keys, and — for a batch that arrived unsorted — the logical-to-physical
/// order that sorts it.
///
/// Sort order is ascending, NULLS LAST ([`MERGE_SORT_OPTIONS`]). L0 segments
/// are written UNSORTED (seq-monotonic only), so an L0 input is not key-sorted;
/// higher-level inputs and migration sources the legacy compactor wrote already
/// are, and their check costs one linear scan of the encoding. An unsorted
/// input keeps its rows in place and carries the sorted order as indirection —
/// no gather copy, no re-encoding — because the merge's interleave gathers
/// physical row indices anyway.
pub struct SortedRun {
    batch: RecordBatch,
    rows: Rows,
    order: Option<Vec<u32>>,
}

impl SortedRun {
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    fn physical(&self, logical: usize) -> usize {
        match &self.order {
            Some(order) => order[logical] as usize,
            None => logical,
        }
    }

    fn row(&self, logical: usize) -> Row<'_> {
        self.rows.row(self.physical(logical))
    }
}

/// Encode `batch`'s sort keys once and wrap it as a [`SortedRun`].
pub fn sort_batch_to_run(
    batch: &RecordBatch,
    converter: &RowConverter,
    sort_cols: &[usize],
) -> Result<SortedRun, ArrowError> {
    let columns: Vec<ArrayRef> = sort_cols
        .iter()
        .map(|&i| Arc::clone(batch.column(i)))
        .collect();
    let rows = converter.convert_columns(&columns)?;
    let sorted = (1..rows.num_rows()).all(|i| rows.row(i - 1) <= rows.row(i));
    let order = if sorted {
        None
    } else {
        let mut indices: Vec<u32> = (0..batch.num_rows() as u32).collect();
        indices.sort_unstable_by(|&a, &b| rows.row(a as usize).cmp(&rows.row(b as usize)));
        Some(indices)
    };
    Ok(SortedRun {
        batch: batch.clone(),
        rows,
        order,
    })
}

/// A heap entry: the borrowed sort-key `Row` of one batch's current cursor.
/// Borrowing from the per-batch `Rows` buffers avoids an allocation per merged
/// row; the buffers outlive the heap and are never mutated while it runs.
///
/// `Ord` is reversed (min-heap via `BinaryHeap`, which is a max-heap): the
/// "greatest" entry is the row that should be popped LAST, so we compare so the
/// globally-smallest row is the heap's max. The `(Reverse-style)` comparison is
/// implemented directly to keep the smallest sort-key row at the top.
struct HeapEntry<'a> {
    row: Row<'a>,
    batch_idx: usize,
}

impl PartialEq for HeapEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}
impl Eq for HeapEntry<'_> {}
impl PartialOrd for HeapEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) yields the smallest row first.
        other.row.cmp(&self.row)
    }
}

/// Merge `runs` into a globally sorted sequence of `RecordBatch` chunks of at
/// most 16384 rows each.
///
/// All runs must share the same schema (project via `project_to_schema` first)
/// and the same converter ([`merge_row_converter`]). Returns one batch per
/// 16384-row chunk so a large merge doesn't materialize a single giant batch
/// and the chunks line up with the parquet row-group size.
///
/// The implicit `seq` sort key makes every row's key unique, so the merge is
/// order-independent regardless of input order.
pub fn merge_runs(runs: &[SortedRun]) -> Result<Vec<RecordBatch>, ArrowError> {
    let schema = runs
        .first()
        .map(|run| run.batch.schema())
        .ok_or_else(|| ArrowError::ComputeError("merge_runs: no input runs".into()))?;
    let non_empty: Vec<&SortedRun> = runs.iter().filter(|run| run.batch.num_rows() > 0).collect();
    if non_empty.is_empty() {
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }

    let total: usize = non_empty.iter().map(|run| run.batch.num_rows()).sum();
    let mut cursors = vec![0usize; non_empty.len()];

    // Seed the heap with each run's first row.
    let mut heap: BinaryHeap<HeapEntry<'_>> = BinaryHeap::with_capacity(non_empty.len());
    for (bi, run) in non_empty.iter().enumerate() {
        heap.push(HeapEntry {
            row: run.row(0),
            batch_idx: bi,
        });
    }

    // (batch_idx, physical_row_idx) pairs in global sort order, partitioned
    // into chunks. An unsorted run's indirection resolves here, so interleave
    // performs its reorder as part of the ordinary gather.
    let mut chunks: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current: Vec<(usize, usize)> = Vec::with_capacity(MERGE_CHUNK_ROWS.min(total));
    while let Some(entry) = heap.pop() {
        let bi = entry.batch_idx;
        current.push((bi, non_empty[bi].physical(cursors[bi])));
        cursors[bi] += 1;
        if cursors[bi] < non_empty[bi].batch.num_rows() {
            heap.push(HeapEntry {
                row: non_empty[bi].row(cursors[bi]),
                batch_idx: bi,
            });
        }
        if current.len() >= MERGE_CHUNK_ROWS {
            chunks.push(std::mem::take(&mut current));
            current = Vec::with_capacity(MERGE_CHUNK_ROWS);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    // Gather each output column for each chunk via interleave.
    let num_cols = schema.fields().len();
    let mut out: Vec<RecordBatch> = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(num_cols);
        for c in 0..num_cols {
            let arrays: Vec<&dyn Array> = non_empty
                .iter()
                .map(|run| run.batch.column(c).as_ref())
                .collect();
            cols.push(interleave(&arrays, chunk)?);
        }
        out.push(RecordBatch::try_new(Arc::clone(&schema), cols)?);
    }
    Ok(out)
}

/// Resolve configured sort columns followed by the implicit `seq` tie-breaker.
pub fn sort_col_indices(arrow_schema: &ArrowSchema, sort_columns: &[String]) -> Vec<usize> {
    let seq_idx = arrow_schema
        .index_of(crate::store::schema::IMPLICIT_SEQ_COLUMN)
        .expect("stored segment schema always carries the implicit seq column");
    let mut indices: Vec<usize> = sort_columns
        .iter()
        .map(|column| {
            arrow_schema.index_of(column).unwrap_or_else(|_| {
                panic!("validated sort column {column:?} is missing from the stored schema")
            })
        })
        .filter(|&index| index != seq_idx)
        .collect();
    if !indices.contains(&seq_idx) {
        indices.push(seq_idx);
    }
    indices
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

    /// Store-form schema: seq, key (int64), worker_id (utf8).
    fn schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
        ]))
    }

    /// A batch already sorted by (key, seq).
    fn batch(rows: Vec<(i64, i64, &str)>) -> RecordBatch {
        let seqs: Int64Array = rows.iter().map(|(s, _, _)| *s).collect();
        let keys: Int64Array = rows.iter().map(|(_, k, _)| *k).collect();
        let ids: Vec<&str> = rows.iter().map(|(_, _, w)| *w).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(seqs),
                Arc::new(keys),
                Arc::new(StringArray::from(ids)),
            ],
        )
        .unwrap()
    }

    /// Merge batches through the run API: one shared converter, one encoding.
    fn merge(batches: &[RecordBatch], sort_cols: &[usize]) -> Vec<RecordBatch> {
        let converter = merge_row_converter(&batches[0].schema(), sort_cols).unwrap();
        let runs: Vec<SortedRun> = batches
            .iter()
            .map(|b| sort_batch_to_run(b, &converter, sort_cols).unwrap())
            .collect();
        merge_runs(&runs).unwrap()
    }

    fn collect_rows(batches: &[RecordBatch]) -> Vec<(i64, i64, String)> {
        let mut out = Vec::new();
        for b in batches {
            let seqs = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let keys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            let ids = b.column(2).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..b.num_rows() {
                out.push((seqs.value(i), keys.value(i), ids.value(i).to_string()));
            }
        }
        out
    }

    #[test]
    fn merge_three_overlapping_inputs_sorts_by_key_then_seq_no_loss_no_dup() {
        // Three internally-sorted (key,seq) batches with INTERLEAVING key ranges
        // so a naive concat would be wrong.
        let a = batch(vec![(1, 10, "a1"), (4, 30, "a4")]);
        let b = batch(vec![(2, 10, "b2"), (5, 20, "b5")]);
        let c = batch(vec![(3, 20, "c3"), (6, 30, "c6")]);
        let merged = merge(&[a, b, c], &[1, 0]);

        let rows = collect_rows(&merged);
        assert_eq!(rows.len(), 6, "no row loss / no duplication");
        // Globally sorted by (key, seq):
        let keyed: Vec<(i64, i64)> = rows.iter().map(|(s, k, _)| (*k, *s)).collect();
        assert_eq!(
            keyed,
            vec![(10, 1), (10, 2), (20, 3), (20, 5), (30, 4), (30, 6)]
        );
        // key tie (10) is resolved by seq (1 < 2); (20) by seq (3 < 5).
        let mut sorted = keyed.clone();
        sorted.sort();
        assert_eq!(keyed, sorted);
        // every original worker id survives exactly once.
        let mut ids: Vec<String> = rows.iter().map(|(_, _, w)| w.clone()).collect();
        ids.sort();
        assert_eq!(ids, vec!["a1", "a4", "b2", "b5", "c3", "c6"]);
    }

    #[test]
    fn an_unsorted_run_merges_in_sorted_order() {
        // an UNSORTED L0-style batch: keys 30,10,20 with seqs 1,2,3. The run
        // keeps the batch in place and carries the order as indirection; the
        // merge output is what must come back sorted.
        let b = batch(vec![(1, 30, "a"), (2, 10, "b"), (3, 20, "c")]);
        let rows = collect_rows(&merge(&[b], &[1, 0]));
        let keyed: Vec<(i64, i64)> = rows.iter().map(|(s, k, _)| (*k, *s)).collect();
        assert_eq!(keyed, vec![(10, 2), (20, 3), (30, 1)]);
    }

    #[test]
    fn merge_seq_only_sort_key() {
        // No key column: sort by seq alone. Disjoint seq ranges, reversed input.
        let a = batch(vec![(3, 0, "a3"), (4, 0, "a4")]);
        let b = batch(vec![(1, 0, "b1"), (2, 0, "b2")]);
        let merged = merge(&[a, b], &[0]);
        let seqs: Vec<i64> = collect_rows(&merged).iter().map(|(s, _, _)| *s).collect();
        assert_eq!(seqs, vec![1, 2, 3, 4]);
    }

    #[test]
    fn merge_emits_fixed_size_chunks() {
        // A merge that exceeds MERGE_CHUNK_ROWS must cut its output into chunks of
        // that size; the writer's row groups are sized separately, by bytes.
        let n = MERGE_CHUNK_ROWS as i64 + 100;
        let a = batch((0..n).step_by(2).map(|s| (s, s, "a")).collect());
        let b = batch((1..n).step_by(2).map(|s| (s, s, "b")).collect());
        let merged = merge(&[a, b], &[1, 0]);
        assert!(merged.len() >= 2, "large merge splits into chunks");
        assert_eq!(merged[0].num_rows(), MERGE_CHUNK_ROWS);
        let total: usize = merged.iter().map(|m| m.num_rows()).sum();
        assert_eq!(total as i64, n);
        // strictly increasing seq across the whole stream.
        let seqs: Vec<i64> = collect_rows(&merged).iter().map(|(s, _, _)| *s).collect();
        for w in seqs.windows(2) {
            assert!(w[0] < w[1], "seq strictly increasing across chunks");
        }
    }

    #[test]
    fn project_to_schema_null_fills_additive_column() {
        // older input lacks the additive nullable `note` column.
        let target: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("note", DataType::Utf8, true),
        ]));
        let old = batch(vec![(1, 10, "a1")]);
        let projected = project_to_schema(&old, &target).unwrap();
        assert_eq!(projected.num_columns(), 4);
        let note = projected.column(3);
        assert_eq!(note.data_type(), &DataType::Utf8);
        assert_eq!(note.len(), 1);
        assert_eq!(note.null_count(), 1);
    }

    #[test]
    fn project_to_schema_handles_native_map_column() {
        // The full-DataType-equality gate must accept a canonical Map<Utf8,Utf8>
        // column unchanged (fast path) and null-fill it when an older segment
        // predates the column — a naming/nullability/sorted drift here would hard
        // reject at compaction.
        use crate::store::schema::map_utf8_utf8_type;
        use arrow::array::{MapBuilder, MapFieldNames, StringBuilder};

        let target: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("labels", map_utf8_utf8_type(), true),
        ]));

        // Older segment without `labels` → null-filled as a typed empty MapArray.
        let old = batch(vec![(1, 10, "a1")]);
        let filled = project_to_schema(&old, &target).unwrap();
        assert_eq!(filled.column(3).data_type(), &map_utf8_utf8_type());
        assert_eq!(filled.column(3).null_count(), 1);

        // A segment that already carries the canonical Map passes through as-is.
        let names = MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        };
        let mut mb = MapBuilder::new(Some(names), StringBuilder::new(), StringBuilder::new());
        mb.keys().append_value("scope");
        mb.values().append_value("fleet");
        mb.append(true).unwrap();
        let labels = mb.finish();
        let with_labels = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("seq", DataType::Int64, false),
                Field::new("key", DataType::Int64, false),
                Field::new("worker_id", DataType::Utf8, false),
                Field::new("labels", map_utf8_utf8_type(), true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1_i64])),
                Arc::new(Int64Array::from(vec![10_i64])),
                Arc::new(StringArray::from(vec!["a1"])),
                Arc::new(labels),
            ],
        )
        .unwrap();
        let projected = project_to_schema(&with_labels, &target).unwrap();
        assert_eq!(projected.column(3).data_type(), &map_utf8_utf8_type());
        assert_eq!(projected.column(3).null_count(), 0);
    }

    #[test]
    fn project_to_schema_type_mismatch_errors() {
        // target says `key` is Utf8 but batch has Int64 — a non-additive conflict
        // (NOT a timestamp-unit difference), so it must still error.
        let target: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("worker_id", DataType::Utf8, false),
        ]));
        let b = batch(vec![(1, 10, "a1")]);
        assert!(project_to_schema(&b, &target).is_err());
    }

    #[test]
    fn project_to_schema_upcasts_millisecond_timestamp_to_microsecond() {
        use arrow::array::{TimestampMicrosecondArray, TimestampMillisecondArray};
        use arrow::datatypes::TimeUnit;

        // A stray millisecond segment merged under the microsecond storage target.
        let target: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]));
        let ms = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                Field::new("seq", DataType::Int64, false),
                Field::new(
                    "ts",
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    false,
                ),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1_i64])),
                Arc::new(TimestampMillisecondArray::from(vec![1_700_000_000_000_i64])),
            ],
        )
        .unwrap();

        let projected = project_to_schema(&ms, &target).unwrap();
        assert_eq!(
            projected.column(1).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let ts = projected
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(ts.values(), &[1_700_000_000_000_000_i64]);
    }

    #[test]
    fn sort_col_indices_with_and_without_key() {
        let s = schema();
        assert_eq!(sort_col_indices(&s, &["key".to_string()]), vec![1, 0]);
        assert_eq!(sort_col_indices(&s, &[]), vec![0]);
    }

    #[test]
    #[should_panic(expected = "validated sort column \"nope\" is missing")]
    fn missing_configured_sort_column_is_a_programming_error() {
        sort_col_indices(&schema(), &["nope".to_string()]);
    }
}
