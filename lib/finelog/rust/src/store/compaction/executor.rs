//! Apply a `CompactionJob`: produce the merged/bumped segment on disk and a
//! `PlannedSwap` for the caller to commit.
//!
//! The executor performs the heavy, lock-free work — parquet read, merge, write,
//! and (for a multi-input merge) the staging-file rename to the distinctly-named
//! output. It returns a [`PlannedSwap`] describing the deque/catalog mutation;
//! the *commit* of that swap (deque splice + catalog `replace_segments` + the
//! single-input bump rename + input unlink) is done by the caller under the
//! query-visibility write lock (`commit_swap`). This keeps the destructive
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
use crate::store::types::{seg_filename, LocalSegment, SegmentLocation, SegmentRow};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The deque/catalog mutation a `CompactionJob` resolves to, ready for the
/// caller to commit under the query-visibility write lock.
///
/// `removed` are the input segment paths to splice out — the prefix of the job's
/// inputs that fit the merge memory ceiling, which may be shorter than the job.
/// `added` is the single output segment (its file already exists for a merge;
/// for a bump the file appears only after `bump_rename` runs in the commit).
/// `unlink_removed` is `false` for a level bump (the input file was renamed, so
/// its old path is already gone after `bump_rename`) and `true` for a merge (the
/// inputs are still on disk). `bump_rename`, when `Some((from, to))`, is the
/// in-place promotion rename the commit performs first. `input_arrow_bytes` is
/// the measured decoded size of the consumed inputs (0 for a bump, which decodes
/// nothing) — the quantity the ceiling bounds, logged so merge memory is visible
/// in production.
#[derive(Debug, Clone)]
pub struct PlannedSwap {
    pub removed: Vec<String>,
    pub added: LocalSegment,
    pub unlink_removed: bool,
    pub bump_rename: Option<(PathBuf, PathBuf)>,
    pub input_arrow_bytes: i64,
}

/// Resolve `job` into a `PlannedSwap`, performing the heavy read/merge/write for
/// a multi-input job. `dir` is the namespace directory; `arrow_schema` is the
/// store-form schema (with `seq`); `key_column` is the namespace's ordering key.
///
/// `max_merge_arrow_bytes` caps the decoded size the merge holds: inputs are
/// read in order and the merge takes the longest prefix that fits, leaving the
/// rest of the run for the next tick.
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
    indexed_columns: &[&str],
    max_merge_arrow_bytes: i64,
    input_key_bounds: impl Fn(&str) -> (Option<i64>, Option<i64>),
) -> Result<PlannedSwap, StatsError> {
    if job.inputs.len() == 1 {
        apply_level_bump(&job.inputs[0], job.output_level, dir, &input_key_bounds)
    } else {
        apply_merge(
            job,
            dir,
            arrow_schema,
            key_column,
            indexed_columns,
            max_merge_arrow_bytes,
            &input_key_bounds,
        )
    }
}

/// Single-input promotion: a rename, no rewrite. The output `LocalSegment`
/// carries the new level + path but PRESERVES the input's `created_at_ms`,
/// row_count, seq window, and typed key bounds. The rename itself is deferred to
/// the commit via `PlannedSwap::bump_rename`.
fn apply_level_bump(
    old: &SegmentRow,
    output_level: i32,
    dir: &Path,
    input_key_bounds: &impl Fn(&str) -> (Option<i64>, Option<i64>),
) -> Result<PlannedSwap, StatsError> {
    let new_filename = seg_filename(output_level, old.min_seq);
    let new_path = dir.join(&new_filename);
    let (min_key, max_key) = input_key_bounds(&old.path);
    let bumped = LocalSegment {
        path: new_path.to_string_lossy().into_owned(),
        size_bytes: old.byte_size,
        level: output_level,
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
        input_arrow_bytes: 0,
    })
}

/// Multi-input merge: read inputs, project, k-way merge, write the output file,
/// rename `.tmp` -> final. Returns the swap with `unlink_removed = true`.
///
/// Merges the longest prefix of `job.inputs` whose measured Arrow size fits
/// `max_merge_arrow_bytes`; the remaining inputs stay at their level for the
/// next tick, which replans them as a shorter run. A prefix of a contiguous run
/// is itself contiguous, so the output's seq window stays gap-free and the
/// commit's deque splice is unaffected. Truncating to a single input degenerates
/// to a level bump — a rename, no rewrite.
fn apply_merge(
    job: &CompactionJob,
    dir: &Path,
    arrow_schema: &SchemaRef,
    key_column: Option<&str>,
    indexed_columns: &[&str],
    max_merge_arrow_bytes: i64,
    input_key_bounds: &impl Fn(&str) -> (Option<i64>, Option<i64>),
) -> Result<PlannedSwap, StatsError> {
    let merged_filename = seg_filename(job.output_level, job.output_min_seq);
    let merged_path = dir.join(&merged_filename);
    let staging_path = dir.join(format!("{merged_filename}.tmp"));

    let sort_cols = sort_col_indices(arrow_schema, key_column);

    // Read each input's row-group batches, project onto the namespace schema
    // (additive null-fill), then SORT each batch and feed it to the k-way merge
    // as its own sorted run. L0 segments are written UNSORTED, so this per-batch
    // sort is what lets the merge produce globally `(key, seq)`-ordered output.
    // An N-way merge is partition-independent — splitting one segment into its
    // row-group batches yields identical output to merging the segment whole.
    //
    // We deliberately do NOT concat an input's batches into a single RecordBatch
    // first. A segment's decompressed `data` column can exceed Arrow's 2^31
    // 32-bit-offset `Utf8` ceiling (high-ratio log-text compression inflates a
    // ~256 MiB compressed `log` segment past 2 GiB), and that concat overflowed
    // and wedged the `log` namespace's compaction indefinitely. Each reader batch
    // is row-group-bounded, so sorting it in isolation never overflows.
    //
    // Inputs are taken ONE AT A TIME and measured against the memory ceiling
    // before the merge commits to them, because a segment's decoded size is only
    // knowable by decoding it: every size in its footer counts encoded bytes,
    // where a repeated log line costs one dictionary index rather than its text.
    // An input that does not fit is therefore read, measured, and dropped again —
    // one input of transient RAM over the ceiling, the price of measuring instead
    // of guessing.
    let mut projected: Vec<RecordBatch> = Vec::new();
    let mut consumed: Vec<&SegmentRow> = Vec::new();
    let mut input_arrow_bytes: i64 = 0;
    for inp in &job.inputs {
        // An input we cannot read is one we can never merge, and failing the tick
        // would replan the identical job every check_interval and wedge the level
        // for good. Route around it instead: as the run's head it is promoted by
        // rename (the branch below), and otherwise it ends the prefix and becomes
        // the next tick's head. Either way it moves and compaction stays live.
        // Only the READ is forgiven — a projection or sort failure is a schema bug
        // and still propagates.
        let raw = match read_segment_batches(Path::new(&inp.path)) {
            Ok(raw) => raw,
            Err(e) => {
                tracing::warn!(
                    path = %inp.path,
                    error = %e,
                    "unreadable merge input; promoting it past the merge"
                );
                if consumed.is_empty() {
                    return apply_level_bump(inp, job.output_level, dir, input_key_bounds);
                }
                break;
            }
        };
        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut batch_bytes: i64 = 0;
        for b in raw {
            let projected_batch = project_to_schema(&b, arrow_schema)
                .map_err(|e| StatsError::Internal(format!("project merge input: {e}")))?;
            let sorted = sort_batch_by(&projected_batch, &sort_cols)
                .map_err(|e| StatsError::Internal(format!("sort merge input: {e}")))?;
            batch_bytes = batch_bytes.saturating_add(sorted.get_array_memory_size() as i64);
            batches.push(sorted);
        }
        if !consumed.is_empty()
            && input_arrow_bytes.saturating_add(batch_bytes) > max_merge_arrow_bytes
        {
            break;
        }
        input_arrow_bytes = input_arrow_bytes.saturating_add(batch_bytes);
        projected.extend(batches);
        consumed.push(inp);
        if input_arrow_bytes > max_merge_arrow_bytes {
            break;
        }
    }

    // One input has nothing to merge with — whether it busted the ceiling alone
    // or merely left no room for the next input. Promote it by rename instead,
    // so it costs no merge memory and its level still advances.
    if consumed.len() == 1 {
        drop(projected);
        return apply_level_bump(consumed[0], job.output_level, dir, input_key_bounds);
    }

    let merged = kway_merge(&projected, &sort_cols)
        .map_err(|e| StatsError::Internal(format!("k-way merge: {e}")))?;
    // `kway_merge` copied the rows it needs into `merged`; free the sorted inputs
    // now so the segment isn't held in RAM twice through the parquet + sidecar
    // writes below (each input plus the output is a fully materialized,
    // uncompressed copy of the segment).
    drop(projected);
    write_merged_segment(&staging_path, arrow_schema, &merged)?;
    std::fs::rename(&staging_path, &merged_path).map_err(|e| {
        StatsError::Internal(format!(
            "rename merge output {} -> {}: {e}",
            staging_path.display(),
            merged_path.display()
        ))
    })?;

    // Build the trigram substring-index sidecar next to the merged output, one
    // bloom set per `indexed_columns` entry (a no-op for namespaces with no
    // indexed columns). Best-effort: the index is optional, so a missing sidecar
    // only disables row-group pruning for this segment, never correctness.
    // Sidecars are built here, at the L0->L1+ merge (where the bulk of queryable
    // data lands), and carried forward verbatim by single-input level bumps; L0
    // is intentionally left unindexed.
    //
    // The parquet rename above already committed the segment, so a crash in the
    // gap before this write leaves the segment without a sidecar. That is the
    // same correct-but-unpruned state as any missing sidecar; a later compaction
    // consuming this segment rebuilds it. A terminal-level segment that is never
    // re-merged (or one written before sidecars existed) stays unindexed until
    // the maintenance backfill (`Namespace::backfill_missing_sidecars`) rebuilds
    // it a few segments per tick.
    if let Err(e) =
        crate::store::trigram::write_sidecar(&merged_path, &merged, indexed_columns, key_column)
    {
        tracing::warn!(path = %merged_path.display(), error = %e, "trigram sidecar write failed");
    }

    let size = std::fs::metadata(&merged_path)
        .map_err(|e| StatsError::Internal(format!("stat {}: {e}", merged_path.display())))?
        .len() as i64;
    // Bounds and counts fold over the CONSUMED prefix, not the planned job: the
    // inputs left behind are still live segments at their own level.
    let row_count: i64 = consumed.iter().map(|s| s.row_count).sum();
    let (merged_min_key, merged_max_key) =
        aggregate_key_bounds(consumed.iter().map(|s| input_key_bounds(&s.path)));
    let merged_seg = LocalSegment {
        path: merged_path.to_string_lossy().into_owned(),
        size_bytes: size,
        level: job.output_level,
        min_seq: consumed.iter().map(|s| s.min_seq).min().expect("non-empty"),
        max_seq: consumed.iter().map(|s| s.max_seq).max().expect("non-empty"),
        row_count,
        created_at_ms: now_ms(),
        min_key_value: merged_min_key,
        max_key_value: merged_max_key,
        location: SegmentLocation::Local,
    };
    Ok(PlannedSwap {
        removed: consumed.iter().map(|s| s.path.clone()).collect(),
        added: merged_seg,
        unlink_removed: true,
        bump_rename: None,
        input_arrow_bytes,
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
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], i64::MAX, bounds).unwrap();
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

    /// A segment of `n` rows all repeating the same wide log line — the shape
    /// that makes parquet sizes useless as a memory proxy. Both the compressed
    /// and the footer's `total_byte_size` collapse to near nothing (one
    /// dictionary entry plus run-length-encoded indices) while the decode
    /// materializes every row's text in full.
    fn repeated_line_segment(dir: &Path, first_seq: i64, n: i64) -> (PathBuf, i64, i64) {
        let rows: Vec<(i64, i64, &str)> = (0..n)
            .map(|i| {
                (
                    first_seq + i,
                    first_seq + i,
                    "a-log-line-repeated-verbatim-across-every-row",
                )
            })
            .collect();
        let (path, _) = write_segment_to_dir(dir, 0, first_seq, &batch(&rows)).unwrap();
        (path, first_seq, first_seq + n - 1)
    }

    fn decoded_bytes(path: &Path) -> i64 {
        read_segment_batches(path)
            .unwrap()
            .iter()
            .map(|b| b.get_array_memory_size() as i64)
            .sum()
    }

    /// A ceiling that admits only part of a planned job merges the prefix that
    /// fits and leaves the rest at their level for the next tick, so a backlog
    /// drains in bounded chunks instead of one unbounded merge.
    ///
    /// The ceiling here sits ABOVE the inputs' combined parquet size and below
    /// their combined decode, so it also pins the denomination: a budget counting
    /// stored bytes would wave this whole run through. That is precisely how the
    /// `log` namespace OOM-killed its container — a 4 GiB budget admitted a job
    /// whose footer read 1.9 GiB and whose decode reached ~15 GiB.
    #[test]
    fn merge_takes_the_prefix_that_fits_and_leaves_the_rest() {
        let dir = tempdir("prefix");
        let (p0, min0, max0) = repeated_line_segment(&dir, 1, 20_000);
        let (p1, min1, max1) = repeated_line_segment(&dir, 20_001, 20_000);
        let (p2, min2, max2) = repeated_line_segment(&dir, 40_001, 20_000);
        let one = decoded_bytes(&p0);
        let stored: i64 = [&p0, &p1, &p2]
            .iter()
            .map(|p| std::fs::metadata(p).unwrap().len() as i64)
            .sum();
        assert!(
            one * 2 + 1 > stored,
            "the ceiling must exceed every input's combined STORED size ({stored}), \
             else this cannot distinguish a decoded budget from a parquet-sized one"
        );

        let job = CompactionJob {
            inputs: vec![
                row_for(&p0.to_string_lossy(), 0, min0, max0, 100),
                row_for(&p1.to_string_lossy(), 0, min1, max1, 100),
                row_for(&p2.to_string_lossy(), 0, min2, max2, 100),
            ],
            output_level: 1,
            output_min_seq: min0,
        };

        // A ceiling holding two of the three segments.
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], one * 2 + 1, |_| {
            (None, None)
        })
        .unwrap();
        assert_eq!(swap.removed.len(), 2, "only the fitting prefix is consumed");
        assert_eq!(
            swap.added.max_seq, max1,
            "the output spans the consumed prefix, not the planned job"
        );
        assert_eq!(swap.added.row_count, 40_000);
        assert!(
            swap.input_arrow_bytes <= one * 2 + 1,
            "the measured decode must respect the ceiling"
        );
        assert!(
            Path::new(&p2).exists(),
            "the untaken input stays live for the next tick"
        );

        // A ceiling over the whole job merges it whole, as if uncapped.
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], i64::MAX, |_| {
            (None, None)
        })
        .unwrap();
        assert_eq!(swap.removed.len(), 3);
        assert_eq!(swap.added.max_seq, max2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An unreadable input must not fail the tick. The planner would hand back
    /// the identical job every `check_interval`, so propagating the read error
    /// would wedge the level permanently on one corrupt file — the very failure
    /// the memory ceiling exists to prevent. It is promoted past the merge
    /// instead, and its readable neighbours still compact.
    #[test]
    fn unreadable_input_is_promoted_past_the_merge_not_propagated() {
        let dir = tempdir("unreadable");
        let (p_bad, min_bad, max_bad) = repeated_line_segment(&dir, 1, 100);
        let (p_good, min_good, max_good) = repeated_line_segment(&dir, 101, 100);
        std::fs::write(&p_bad, b"this is not a parquet file").unwrap();

        let job = CompactionJob {
            inputs: vec![
                row_for(&p_bad.to_string_lossy(), 0, min_bad, max_bad, 100),
                row_for(&p_good.to_string_lossy(), 0, min_good, max_good, 100),
            ],
            output_level: 1,
            output_min_seq: min_bad,
        };
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], i64::MAX, |_| {
            (None, None)
        })
        .expect("an unreadable input must not fail the tick");
        assert!(
            swap.bump_rename.is_some(),
            "the unreadable head is renamed past, not merged"
        );
        assert_eq!(swap.removed, vec![p_bad.to_string_lossy().to_string()]);
        assert!(
            Path::new(&p_good).exists(),
            "its readable neighbour stays live to compact next tick"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A lone input over the ceiling has nothing to merge with, so it is
    /// promoted by rename — no rewrite, no memory — rather than wedging its
    /// level forever behind a merge that can never fit.
    #[test]
    fn input_alone_over_ceiling_is_promoted_by_rename() {
        let dir = tempdir("lone_over");
        let (p0, min0, max0) = repeated_line_segment(&dir, 1, 20_000);
        let (p1, min1, max1) = repeated_line_segment(&dir, 20_001, 20_000);
        let job = CompactionJob {
            inputs: vec![
                row_for(&p0.to_string_lossy(), 0, min0, max0, 100),
                row_for(&p1.to_string_lossy(), 0, min1, max1, 100),
            ],
            output_level: 1,
            output_min_seq: min0,
        };
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], 1, |_| (None, None)).unwrap();
        assert!(swap.bump_rename.is_some(), "must degenerate to a rename");
        assert!(!swap.unlink_removed, "a rename leaves nothing to unlink");
        assert_eq!(swap.removed, vec![p0.to_string_lossy().to_string()]);
        assert_eq!(swap.added.max_seq, max0, "the bump carries its own span");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression for the `log`-namespace compaction wedge: an input segment
    /// spanning multiple parquet row groups must merge WITHOUT concatenating its
    /// batches into one array. The executor used to `concat_batches` each input
    /// whole before sorting, which overflowed Arrow's 2^31 `Utf8` offset ceiling
    /// once a segment's decompressed `data` column crossed 2 GiB — every
    /// subsequent `run_maintenance` failed on the same poison segment and remote
    /// uploads froze. This exercises the per-row-group merge path and asserts no
    /// row loss and global (key, seq) order across row-group boundaries.
    #[test]
    fn merge_multi_row_group_input_no_concat() {
        use crate::store::segment::ROW_GROUP_SIZE;
        let dir = tempdir("multirg");

        // One large L0 segment: >2 row groups, written UNSORTED (descending key)
        // so the per-batch sort is load-bearing. seq is unique and monotonic.
        let n = (ROW_GROUP_SIZE as i64) * 2 + 500;
        let big: Vec<(i64, i64, &str)> = (1..=n).map(|s| (s, n - s + 1, "big")).collect();
        let (p_big, _) = write_segment_to_dir(&dir, 0, 1, &batch(&big)).unwrap();
        let (p_small, _) =
            write_segment_to_dir(&dir, 0, n + 1, &batch(&[(n + 1, 7, "s")])).unwrap();

        // Reading `big` back yields many row-group-bounded batches, not one array
        // — the condition under which the old concat path overflowed.
        assert!(
            read_segment_batches(&p_big).unwrap().len() > 1,
            "large input must span multiple reader batches"
        );

        let job = CompactionJob {
            inputs: vec![
                row_for(&p_big.to_string_lossy(), 0, 1, n, 100),
                row_for(&p_small.to_string_lossy(), 0, n + 1, n + 1, 100),
            ],
            output_level: 1,
            output_min_seq: 1,
        };
        let bounds = |_: &str| (Some(1), Some(n));
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], i64::MAX, bounds).unwrap();

        assert_eq!(swap.added.row_count, n + 1, "no row loss");
        let out = PathBuf::from(&swap.added.path);
        let mut keyed: Vec<(i64, i64)> = Vec::new();
        for b in &read_segment_batches(&out).unwrap() {
            let seqs = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let keys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                keyed.push((keys.value(i), seqs.value(i)));
            }
        }
        assert_eq!(keyed.len() as i64, n + 1);
        let mut sorted = keyed.clone();
        sorted.sort();
        assert_eq!(
            keyed, sorted,
            "globally (key, seq)-sorted across row groups"
        );
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
        };
        let bounds = |_: &str| (Some(10), Some(20));
        let swap = run_job(&job, &dir, &schema(), Some("key"), &[], i64::MAX, bounds).unwrap();

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
    fn merge_writes_trigram_sidecar_for_data_column() {
        use crate::store::trigram::{read_column_from_bytes, sidecar_path};
        let dir = tempdir("tgm_sidecar");
        // Log-form schema with a `data` column (the indexed column).
        let log: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let mk = |first_seq: i64, lines: &[&str]| {
            let n = lines.len() as i64;
            RecordBatch::try_new(
                Arc::clone(&log),
                vec![
                    Arc::new(Int64Array::from_iter_values(first_seq..first_seq + n)),
                    Arc::new(Int64Array::from(vec![1_i64; lines.len()])),
                    Arc::new(StringArray::from(lines.to_vec())),
                ],
            )
            .unwrap()
        };
        let (p1, _) =
            write_segment_to_dir(&dir, 0, 1, &mk(1, &["Bootstrap completed for TPU"])).unwrap();
        let (p2, _) = write_segment_to_dir(&dir, 0, 2, &mk(2, &["unrelated heartbeat"])).unwrap();
        // L0 inputs have no sidecars (intentionally unindexed).
        assert!(!sidecar_path(&p1).exists());

        let job = CompactionJob {
            inputs: vec![
                row_for(&p1.to_string_lossy(), 0, 1, 1, 50),
                row_for(&p2.to_string_lossy(), 0, 2, 2, 50),
            ],
            output_level: 1,
            output_min_seq: 1,
        };
        let swap = run_job(&job, &dir, &log, Some("key"), &["data"], i64::MAX, |_| {
            (None, None)
        })
        .unwrap();

        // The merged output carries a sidecar whose mask prunes correctly.
        let out = PathBuf::from(&swap.added.path);
        let sc = sidecar_path(&out);
        assert!(sc.exists(), "merge output must have a trigram sidecar");
        let index = read_column_from_bytes(&std::fs::read(&sc).unwrap(), "data").unwrap();
        assert_eq!(index.len(), 1, "one row group");
        assert_eq!(
            index.keep_mask("Bootstrap completed for TPU").unwrap(),
            vec![true]
        );
        assert_eq!(
            index.keep_mask("string definitely absent zzz").unwrap(),
            vec![false]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sidecar_row_groups_match_parquet_across_batch_boundaries() {
        // The prune contract depends on the index having exactly one Bloom per
        // parquet row group. The index chunks at ROW_GROUP_SIZE; `ArrowWriter`
        // (via `segment_writer_properties`) flushes at the same stride REGARDLESS
        // of how the written batches are split. Lock that with input batches whose
        // boundaries straddle a row-group boundary (10k|10k|10005 over a 16384
        // stride), so the writer must re-chunk across `write()` calls.
        use crate::store::segment::{segment_row_group_count, ROW_GROUP_SIZE};
        use crate::store::trigram::TrigramIndex;

        let dir = tempdir("tgm_align");
        let log: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let mk = |first_seq: i64, n: usize| {
            let lines: Vec<String> = (0..n).map(|i| format!("log line number {i}")).collect();
            RecordBatch::try_new(
                Arc::clone(&log),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        first_seq..first_seq + n as i64,
                    )),
                    Arc::new(Int64Array::from(vec![1_i64; n])),
                    Arc::new(StringArray::from(lines)),
                ],
            )
            .unwrap()
        };
        let batches = vec![mk(1, 10_000), mk(10_001, 10_000), mk(20_001, 10_005)];
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        let expected_groups = total.div_ceil(ROW_GROUP_SIZE);
        assert_eq!(
            expected_groups, 2,
            "30005 rows over a 16384 stride is 2 groups"
        );

        let path = dir.join("seg_L1_00000000000000000001.parquet");
        write_merged_segment(&path, &log, &batches).unwrap();

        let parquet_groups =
            segment_row_group_count(&path).expect("readable footer for the written segment");
        let index = TrigramIndex::build(&batches, "data").unwrap();
        assert_eq!(
            parquet_groups, expected_groups,
            "ArrowWriter must flush a row group every ROW_GROUP_SIZE rows"
        );
        assert_eq!(
            index.len(),
            parquet_groups,
            "sidecar must carry exactly one Bloom per parquet row group"
        );
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
        };
        let swap = run_job(&job, &dir, &wide, Some("key"), &[], i64::MAX, |_| {
            (None, None)
        })
        .unwrap();
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
