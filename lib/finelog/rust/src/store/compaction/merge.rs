//! Native arrow k-way merge of N already-sorted segments by `(key_column, seq)`.
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
use arrow::compute::{
    cast, interleave, lexsort_to_indices, take_record_batch, SortColumn, SortOptions,
};
use arrow::datatypes::{DataType, Schema as ArrowSchema, SchemaRef};
use arrow::error::ArrowError;
use arrow::row::{OwnedRow, RowConverter, Rows, SortField};

use crate::store::segment::ROW_GROUP_SIZE;

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

/// The merge sort-key collation: ascending, NULLS LAST. The pre-sort
/// (`sort_batch_by`) and the merge (`kway_merge` via `RowConverter`) MUST share
/// this so a key column with NULLs lands null-key rows at the END in both paths,
/// giving a consistent physical segment layout.
const MERGE_SORT_OPTIONS: SortOptions = SortOptions {
    descending: false,
    nulls_first: false,
};

/// Stably sort `batch` by `sort_cols` (ascending, NULLS LAST), returning the
/// reordered batch.
///
/// L0 segments are written UNSORTED (seq-monotonic only), so an L0->L1 merge's
/// inputs are not key-sorted; `kway_merge` requires each input to be internally
/// sorted on the merge keys. The executor runs each projected input through this
/// before merging, so the merge sees pre-sorted inputs and the global output is
/// `(key, seq)`-ordered. A segment already sorted (L>=1 inputs) is unchanged by a
/// stable sort.
pub fn sort_batch_by(batch: &RecordBatch, sort_cols: &[usize]) -> Result<RecordBatch, ArrowError> {
    if batch.num_rows() <= 1 {
        return Ok(batch.clone());
    }
    let columns: Vec<SortColumn> = sort_cols
        .iter()
        .map(|&i| SortColumn {
            values: Arc::clone(batch.column(i)),
            options: Some(MERGE_SORT_OPTIONS),
        })
        .collect();
    let indices = lexsort_to_indices(&columns, None)?;
    take_record_batch(batch, &indices)
}

/// A heap entry: the owned sort-key `Row` of one batch's current cursor.
///
/// `Ord` is reversed (min-heap via `BinaryHeap`, which is a max-heap): the
/// "greatest" entry is the row that should be popped LAST, so we compare so the
/// globally-smallest row is the heap's max. The `(Reverse-style)` comparison is
/// implemented directly to keep the smallest `(key, seq)` at the top.
struct HeapEntry {
    row: OwnedRow,
    batch_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) yields the smallest row first.
        other.row.cmp(&self.row)
    }
}

/// Merge already-sorted `batches` into a globally `(sort_cols)`-sorted sequence
/// of `RecordBatch` chunks of at most 16384 rows each.
///
/// `sort_cols` are column indices into the (uniform) batch schema, e.g.
/// `[key_idx, seq_idx]`. All batches must share the same schema (project via
/// `project_to_schema` first). Returns one batch per 16384-row chunk so a large
/// merge doesn't materialize a single giant batch and the chunks line up with
/// the parquet row-group size.
pub fn kway_merge(
    batches: &[RecordBatch],
    sort_cols: &[usize],
) -> Result<Vec<RecordBatch>, ArrowError> {
    let non_empty: Vec<&RecordBatch> = batches.iter().filter(|b| b.num_rows() > 0).collect();
    let schema = batches
        .first()
        .map(|b| b.schema())
        .ok_or_else(|| ArrowError::ComputeError("kway_merge: no input batches".into()))?;
    if non_empty.is_empty() {
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }

    let fields: Vec<SortField> = sort_cols
        .iter()
        .map(|&i| {
            SortField::new_with_options(schema.field(i).data_type().clone(), MERGE_SORT_OPTIONS)
        })
        .collect();
    let converter = RowConverter::new(fields)?;

    // Per-batch Row encodings of the sort-key columns.
    let rows_per_batch: Vec<Rows> = non_empty
        .iter()
        .map(|b| {
            let cols: Vec<ArrayRef> = sort_cols.iter().map(|&i| Arc::clone(b.column(i))).collect();
            converter.convert_columns(&cols)
        })
        .collect::<Result<_, _>>()?;

    let total: usize = non_empty.iter().map(|b| b.num_rows()).sum();
    let mut cursors = vec![0usize; non_empty.len()];

    // Seed the heap with each batch's first row. We store the `OwnedRow`
    // (detached from its `Rows` buffer) so the heap can outlive a single borrow.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(non_empty.len());
    for (bi, rows) in rows_per_batch.iter().enumerate() {
        heap.push(HeapEntry {
            row: rows.row(0).owned(),
            batch_idx: bi,
        });
    }

    // (batch_idx, row_idx) pairs in global sort order, partitioned into chunks.
    let mut chunks: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current: Vec<(usize, usize)> = Vec::with_capacity(ROW_GROUP_SIZE.min(total));
    while let Some(entry) = heap.pop() {
        let bi = entry.batch_idx;
        let ri = cursors[bi];
        current.push((bi, ri));
        cursors[bi] += 1;
        if cursors[bi] < rows_per_batch[bi].num_rows() {
            heap.push(HeapEntry {
                row: rows_per_batch[bi].row(cursors[bi]).owned(),
                batch_idx: bi,
            });
        }
        if current.len() >= ROW_GROUP_SIZE {
            chunks.push(std::mem::take(&mut current));
            current = Vec::with_capacity(ROW_GROUP_SIZE);
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
            let arrays: Vec<&dyn Array> = non_empty.iter().map(|b| b.column(c).as_ref()).collect();
            cols.push(interleave(&arrays, chunk)?);
        }
        out.push(RecordBatch::try_new(Arc::clone(&schema), cols)?);
    }
    Ok(out)
}

/// Resolve sort-key column indices `[key_column?, seq]` for `arrow_schema`.
///
/// Returns the `seq` index alone when `key_column` is `None` or absent — the
/// `compaction_sort_keys` contract. `seq` is required (every stored segment
/// carries it); its absence is a programming error.
pub fn sort_col_indices(arrow_schema: &ArrowSchema, key_column: Option<&str>) -> Vec<usize> {
    let seq_idx = arrow_schema
        .index_of(crate::store::schema::IMPLICIT_SEQ_COLUMN)
        .expect("stored segment schema always carries the implicit seq column");
    match key_column.and_then(|k| arrow_schema.index_of(k).ok()) {
        Some(key_idx) if key_idx != seq_idx => vec![key_idx, seq_idx],
        _ => vec![seq_idx],
    }
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
        let merged = kway_merge(&[a, b, c], &[1, 0]).unwrap();

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
    fn sort_batch_by_orders_unsorted_input() {
        // an UNSORTED L0-style batch: keys 30,10,20 with seqs 1,2,3.
        let b = batch(vec![(1, 30, "a"), (2, 10, "b"), (3, 20, "c")]);
        let sorted = sort_batch_by(&b, &[1, 0]).unwrap();
        let rows = collect_rows(&[sorted]);
        let keyed: Vec<(i64, i64)> = rows.iter().map(|(s, k, _)| (*k, *s)).collect();
        assert_eq!(keyed, vec![(10, 2), (20, 3), (30, 1)]);
    }

    #[test]
    fn merge_seq_only_sort_key() {
        // No key column: sort by seq alone. Disjoint seq ranges, reversed input.
        let a = batch(vec![(3, 0, "a3"), (4, 0, "a4")]);
        let b = batch(vec![(1, 0, "b1"), (2, 0, "b2")]);
        let merged = kway_merge(&[a, b], &[0]).unwrap();
        let seqs: Vec<i64> = collect_rows(&merged).iter().map(|(s, _, _)| *s).collect();
        assert_eq!(seqs, vec![1, 2, 3, 4]);
    }

    #[test]
    fn merge_emits_row_group_aligned_chunks() {
        // A merge that exceeds 16384 rows must produce 16384-row chunks.
        let n = ROW_GROUP_SIZE as i64 + 100;
        let a = batch((0..n).step_by(2).map(|s| (s, s, "a")).collect());
        let b = batch((1..n).step_by(2).map(|s| (s, s, "b")).collect());
        let merged = kway_merge(&[a, b], &[1, 0]).unwrap();
        assert!(merged.len() >= 2, "large merge splits into chunks");
        assert_eq!(merged[0].num_rows(), ROW_GROUP_SIZE);
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
        assert_eq!(sort_col_indices(&s, Some("key")), vec![1, 0]);
        assert_eq!(sort_col_indices(&s, None), vec![0]);
        // a key column that doesn't exist falls back to seq alone.
        assert_eq!(sort_col_indices(&s, Some("nope")), vec![0]);
    }
}
