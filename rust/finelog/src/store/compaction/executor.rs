//! Apply a `CompactionJob`: produce the merged/bumped segment on disk and a
//! `PlannedSwap` for the caller to commit.
//!
//! Port of `_run_job` / `_apply_level_bump` / `_apply_merge` from
//! `log_namespace.py`, with the DuckDB COPY replaced by the native arrow k-way
//! merge (`merge.rs`).
//!
//! The executor performs the heavy, lock-free work — parquet read, merge, write,
//! and (for a multi-input merge) the staging-file rename to the distinctly-named
//! output. It returns a [`PlannedSwap`] describing the deque/catalog mutation;
//! the *commit* of that swap (deque splice + catalog `replace_segments` + the
//! single-input bump rename + input unlink) is done by the caller under the
//! query-visibility write lock (4d's `commit_swap`). This keeps the destructive
//! visibility-affecting step on the locked path while the CPU/IO runs free.
//!
//! Single-input job  => `apply_level_bump`: NO rewrite. The output file does not
//! exist yet; the rename `seg_L{n}_{min}` -> `seg_L{n+1}_{min}` is deferred to
//! the commit (`PlannedSwap::bump_rename`), preserving `created_at_ms` + bounds.
//!
//! Multi-input job   => `apply_merge`: read each input's batches via
//! `ParquetRecordBatchReaderBuilder` (sync) under `spawn_blocking`, project each
//! onto the namespace schema (additive null-fill), k-way merge by
//! `(key_column, seq)`, write via `ArrowWriter` (rg=16384, zstd) to a
//! `.parquet.tmp`, then rename to the final distinctly-named output. The inputs
//! stay on disk until the commit unlinks them.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};

use crate::errors::StatsError;
use crate::store::compaction::config::CompactionJob;
use crate::store::compaction::merge::{
    kway_merge, project_to_schema, sort_batch_by, sort_col_indices,
};
use crate::store::compaction::planner::aggregate_key_bounds;
use crate::store::segment::{segment_bounds, segment_writer_properties};
use crate::store::types::{seg_filename, LocalSegment, SegmentLocation};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The deque/catalog mutation a `CompactionJob` resolves to, ready for the
/// caller to commit under the query-visibility write lock.
///
/// `removed` are the input segment paths to splice out. `added` is the single
/// output segment (its file already exists for a merge; for a bump the file
/// appears only after `bump_rename` runs in the commit). `unlink_removed` is
/// `false` for a level bump (the input file was renamed, so its old path is
/// already gone after `bump_rename`) and `true` for a merge (the inputs are
/// still on disk). `bump_rename`, when `Some((from, to))`, is the in-place
/// promotion rename the commit performs first.
#[derive(Debug, Clone)]
pub struct PlannedSwap {
    pub removed: Vec<String>,
    pub added: LocalSegment,
    pub unlink_removed: bool,
    pub bump_rename: Option<(PathBuf, PathBuf)>,
}

/// Resolve `job` into a `PlannedSwap`, performing the heavy read/merge/write for
/// a multi-input job. `dir` is the namespace directory; `arrow_schema` is the
/// store-form schema (with `seq`); `key_column` is the namespace's ordering key.
///
/// `inputs_by_path` lets the caller supply the typed in-memory key bounds for
/// each input (the catalog round-trip stringifies them, losing numeric
/// ordering): a closure mapping an input path to its `(min_key, max_key)`. For a
/// bump that is the single input's bounds; for a merge it folds them via
/// `aggregate_key_bounds`.
pub fn run_job(
    job: &CompactionJob,
    dir: &Path,
    arrow_schema: &SchemaRef,
    key_column: Option<&str>,
    input_key_bounds: impl Fn(&str) -> (Option<i64>, Option<i64>),
) -> Result<PlannedSwap, StatsError> {
    if job.inputs.len() == 1 {
        apply_level_bump(job, dir, &input_key_bounds)
    } else {
        apply_merge(job, dir, arrow_schema, key_column, &input_key_bounds)
    }
}

/// Single-input promotion: a rename, no rewrite. The output `LocalSegment`
/// carries the new level + path but PRESERVES the input's `created_at_ms`,
/// row_count, seq window, and typed key bounds. The rename itself is deferred to
/// the commit via `PlannedSwap::bump_rename`.
fn apply_level_bump(
    job: &CompactionJob,
    dir: &Path,
    input_key_bounds: &impl Fn(&str) -> (Option<i64>, Option<i64>),
) -> Result<PlannedSwap, StatsError> {
    let old = &job.inputs[0];
    let new_filename = seg_filename(job.output_level, old.min_seq);
    let new_path = dir.join(&new_filename);
    let (min_key, max_key) = input_key_bounds(&old.path);
    let bumped = LocalSegment {
        path: new_path.to_string_lossy().into_owned(),
        size_bytes: old.byte_size,
        level: job.output_level,
        min_seq: old.min_seq,
        max_seq: old.max_seq,
        row_count: old.row_count,
        created_at_ms: old.created_at_ms,
        min_key_value: min_key,
        max_key_value: max_key,
        location: SegmentLocation::Local,
    };
    Ok(PlannedSwap {
        removed: vec![old.path.clone()],
        added: bumped,
        unlink_removed: false,
        bump_rename: Some((PathBuf::from(&old.path), new_path)),
    })
}

/// Multi-input merge: read inputs, project, k-way merge, write the output file,
/// rename `.tmp` -> final. Returns the swap with `unlink_removed = true`.
fn apply_merge(
    job: &CompactionJob,
    dir: &Path,
    arrow_schema: &SchemaRef,
    key_column: Option<&str>,
    input_key_bounds: &impl Fn(&str) -> (Option<i64>, Option<i64>),
) -> Result<PlannedSwap, StatsError> {
    let merged_filename = seg_filename(job.output_level, job.output_min_seq);
    let merged_path = dir.join(&merged_filename);
    let staging_path = dir.join(format!("{merged_filename}.tmp"));

    let sort_cols = sort_col_indices(arrow_schema, key_column);

    // Read each input, project onto the namespace schema (additive null-fill),
    // then SORT it on the merge keys. L0 segments are written UNSORTED, so this
    // sort is what lets the k-way merge produce globally `(key, seq)`-ordered
    // output (the DuckDB COPY's `ORDER BY (key, seq)` analog). One sorted batch
    // per input keeps the merge a true N-way merge.
    let mut projected: Vec<RecordBatch> = Vec::new();
    for inp in &job.inputs {
        let batches = read_segment_batches(Path::new(&inp.path))?;
        let mut cols: Vec<RecordBatch> = Vec::with_capacity(batches.len());
        for b in batches {
            cols.push(
                project_to_schema(&b, arrow_schema)
                    .map_err(|e| StatsError::Internal(format!("project merge input: {e}")))?,
            );
        }
        // Concatenate the input's own batches into one, then sort it as a unit
        // (a single segment can span multiple row groups / batches).
        let combined = arrow::compute::concat_batches(arrow_schema, &cols)
            .map_err(|e| StatsError::Internal(format!("concat merge input: {e}")))?;
        let sorted = sort_batch_by(&combined, &sort_cols)
            .map_err(|e| StatsError::Internal(format!("sort merge input: {e}")))?;
        projected.push(sorted);
    }

    let merged = kway_merge(&projected, &sort_cols)
        .map_err(|e| StatsError::Internal(format!("k-way merge: {e}")))?;
    write_merged_segment(&staging_path, arrow_schema, &merged)?;
    std::fs::rename(&staging_path, &merged_path).map_err(|e| {
        StatsError::Internal(format!(
            "rename merge output {} -> {}: {e}",
            staging_path.display(),
            merged_path.display()
        ))
    })?;

    let size = std::fs::metadata(&merged_path)
        .map_err(|e| StatsError::Internal(format!("stat {}: {e}", merged_path.display())))?
        .len() as i64;
    // row_count = sum of inputs (matches Python `_apply_merge`).
    let row_count: i64 = job.inputs.iter().map(|s| s.row_count).sum();
    let (merged_min_key, merged_max_key) =
        aggregate_key_bounds(job.inputs.iter().map(|s| input_key_bounds(&s.path)));
    let merged_seg = LocalSegment {
        path: merged_path.to_string_lossy().into_owned(),
        size_bytes: size,
        level: job.output_level,
        min_seq: job.output_min_seq,
        max_seq: job.output_max_seq,
        row_count,
        created_at_ms: now_ms(),
        min_key_value: merged_min_key,
        max_key_value: merged_max_key,
        location: SegmentLocation::Local,
    };
    Ok(PlannedSwap {
        removed: job.inputs.iter().map(|s| s.path.clone()).collect(),
        added: merged_seg,
        unlink_removed: true,
        bump_rename: None,
    })
}

/// Read all `RecordBatch`es from the parquet file at `path` (sync reader).
/// Wrapped in `spawn_blocking` by the maintenance task; the body is sync so
/// `run_job` can also be exercised directly in unit tests.
pub fn read_segment_batches(path: &Path) -> Result<Vec<RecordBatch>, StatsError> {
    let file = std::fs::File::open(path)
        .map_err(|e| StatsError::Internal(format!("open merge input {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StatsError::Internal(format!("parquet reader {}: {e}", path.display())))?;
    let reader = builder
        .build()
        .map_err(|e| StatsError::Internal(format!("parquet reader build: {e}")))?;
    let mut out = Vec::new();
    for b in reader {
        out.push(b.map_err(|e| StatsError::Internal(format!("parquet read batch: {e}")))?);
    }
    Ok(out)
}

/// Write `batches` to `path` via `ArrowWriter` (rg=16384, zstd-1, bloom — the
/// shared `segment_writer_properties`, identical to the L0 flush writer).
fn write_merged_segment(
    path: &Path,
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<(), StatsError> {
    let props = segment_writer_properties()?;
    let file = std::fs::File::create(path)
        .map_err(|e| StatsError::Internal(format!("create {}: {e}", path.display())))?;
    let opts = ArrowWriterOptions::new().with_properties(props);
    let mut writer = ArrowWriter::try_new_with_options(file, Arc::clone(schema), opts)
        .map_err(|e| StatsError::Internal(format!("arrow writer init: {e}")))?;
    for b in batches {
        writer
            .write(b)
            .map_err(|e| StatsError::Internal(format!("arrow write: {e}")))?;
    }
    writer
        .close()
        .map_err(|e| StatsError::Internal(format!("arrow writer close: {e}")))?;
    Ok(())
}

/// Footer-only `row_count` for a written segment (verification helper for the
/// caller / tests). Returns `None` on an unreadable footer.
pub fn segment_row_count(path: &Path) -> Option<i64> {
    segment_bounds(path, None).map(|(n, _, _)| n)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;
    use crate::store::segment::{read_segment_footer, write_segment_to_dir};
    use crate::store::types::{seg_filename, SegmentRow};

    fn tempdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_executor_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
        ]))
    }

    /// rows: (seq, key, worker_id).
    fn batch(rows: &[(i64, i64, &str)]) -> RecordBatch {
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

    fn row_for(path: &str, level: i32, min_seq: i64, max_seq: i64, byte_size: i64) -> SegmentRow {
        SegmentRow {
            namespace: "ns".to_string(),
            path: path.to_string(),
            level,
            min_seq,
            max_seq,
            row_count: max_seq - min_seq + 1,
            byte_size,
            created_at_ms: 111,
            min_key_value: None,
            max_key_value: None,
            location: SegmentLocation::Local,
        }
    }

    #[test]
    fn merge_three_inputs_writes_one_sorted_segment() {
        let dir = tempdir("merge");
        // three L0 segments, seq-disjoint, interleaving keys.
        let (p1, _) =
            write_segment_to_dir(&dir, 0, 1, &batch(&[(1, 30, "a"), (2, 10, "b")])).unwrap();
        let (p2, _) =
            write_segment_to_dir(&dir, 0, 3, &batch(&[(3, 20, "c"), (4, 40, "d")])).unwrap();
        let (p3, _) =
            write_segment_to_dir(&dir, 0, 5, &batch(&[(5, 5, "e"), (6, 25, "f")])).unwrap();

        let job = CompactionJob {
            inputs: vec![
                row_for(&p1.to_string_lossy(), 0, 1, 2, 100),
                row_for(&p2.to_string_lossy(), 0, 3, 4, 100),
                row_for(&p3.to_string_lossy(), 0, 5, 6, 100),
            ],
            output_level: 1,
            output_min_seq: 1,
            output_max_seq: 6,
        };
        // typed key bounds per input.
        let bounds = |path: &str| -> (Option<i64>, Option<i64>) {
            match path {
                p if p == p1.to_string_lossy() => (Some(10), Some(30)),
                p if p == p2.to_string_lossy() => (Some(20), Some(40)),
                p if p == p3.to_string_lossy() => (Some(5), Some(25)),
                _ => (None, None),
            }
        };
        let swap = run_job(&job, &dir, &schema(), Some("key"), bounds).unwrap();
        assert!(swap.bump_rename.is_none());
        assert!(swap.unlink_removed);
        assert_eq!(swap.removed.len(), 3);
        assert_eq!(swap.added.level, 1);
        assert_eq!(swap.added.row_count, 6);
        assert_eq!(swap.added.min_seq, 1);
        assert_eq!(swap.added.max_seq, 6);
        // folded key bounds preserve numeric ordering.
        assert_eq!(swap.added.min_key_value, Some(5));
        assert_eq!(swap.added.max_key_value, Some(40));

        // the output file exists with the expected name and is (key,seq)-sorted.
        let out = PathBuf::from(&swap.added.path);
        assert_eq!(
            out.file_name().unwrap().to_str().unwrap(),
            seg_filename(1, 1)
        );
        let batches = read_segment_batches(&out).unwrap();
        let mut keyed: Vec<(i64, i64)> = Vec::new();
        for b in &batches {
            let seqs = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let keys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                keyed.push((keys.value(i), seqs.value(i)));
            }
        }
        assert_eq!(keyed.len(), 6, "no row loss / no duplication");
        let mut sorted = keyed.clone();
        sorted.sort();
        assert_eq!(keyed, sorted, "globally sorted by (key, seq)");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn level_bump_renames_preserving_metadata_no_rewrite() {
        let dir = tempdir("bump");
        let (p, size) =
            write_segment_to_dir(&dir, 2, 1, &batch(&[(1, 10, "a"), (2, 20, "b")])).unwrap();
        let mut input = row_for(&p.to_string_lossy(), 2, 1, 2, size);
        input.created_at_ms = 9999;
        let job = CompactionJob {
            inputs: vec![input],
            output_level: 3,
            output_min_seq: 1,
            output_max_seq: 2,
        };
        let bounds = |_: &str| (Some(10), Some(20));
        let swap = run_job(&job, &dir, &schema(), Some("key"), bounds).unwrap();

        // It's a bump: a deferred rename, not a rewrite.
        let (from, to) = swap.bump_rename.clone().unwrap();
        assert_eq!(from, p);
        assert_eq!(
            to.file_name().unwrap().to_str().unwrap(),
            seg_filename(3, 1)
        );
        assert!(!swap.unlink_removed);
        assert_eq!(swap.added.level, 3);
        assert_eq!(swap.added.created_at_ms, 9999, "birth time preserved");
        assert_eq!(swap.added.size_bytes, size, "no rewrite -> same bytes");
        assert_eq!(swap.added.min_key_value, Some(10));
        assert_eq!(swap.added.max_key_value, Some(20));

        // The executor itself does NOT rename (deferred to commit); the old file
        // is still present and the new one absent.
        assert!(p.exists());
        assert!(!to.exists());
        // Performing the deferred rename yields a footer-readable L3 segment.
        std::fs::rename(&from, &to).unwrap();
        let meta = read_segment_footer(&to, Some("key")).unwrap();
        assert_eq!(meta.level, 3);
        assert_eq!(meta.row_count, 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_null_fills_additive_column() {
        // newer input has an extra nullable `note` column; older lacks it. Merge
        // under the wider schema must null-fill the older rows.
        let dir = tempdir("nullfill");
        let wide: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("note", DataType::Utf8, true),
        ]));
        // old segment: narrow schema (no note).
        let (p_old, _) = write_segment_to_dir(&dir, 0, 1, &batch(&[(1, 10, "a")])).unwrap();
        // new segment: wide schema with note.
        let wide_batch = RecordBatch::try_new(
            Arc::clone(&wide),
            vec![
                Arc::new(Int64Array::from(vec![2_i64])),
                Arc::new(Int64Array::from(vec![20_i64])),
                Arc::new(StringArray::from(vec!["b"])),
                Arc::new(StringArray::from(vec![Some("hi")])),
            ],
        )
        .unwrap();
        let (p_new, _) = write_segment_to_dir(&dir, 0, 2, &wide_batch).unwrap();

        let job = CompactionJob {
            inputs: vec![
                row_for(&p_old.to_string_lossy(), 0, 1, 1, 50),
                row_for(&p_new.to_string_lossy(), 0, 2, 2, 50),
            ],
            output_level: 1,
            output_min_seq: 1,
            output_max_seq: 2,
        };
        let swap = run_job(&job, &dir, &wide, Some("key"), |_| (None, None)).unwrap();
        let batches = read_segment_batches(Path::new(&swap.added.path)).unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
        // note column exists and the first (old) row is null.
        let note_idx = batches[0].schema().index_of("note").unwrap();
        let note = batches[0].column(note_idx);
        assert_eq!(note.data_type(), &DataType::Utf8);
        assert!(note.null_count() >= 1, "older input's note null-filled");
        std::fs::remove_dir_all(&dir).ok();
    }
}
