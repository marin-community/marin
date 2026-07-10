//! L0 parquet segment writer + footer recovery.
//!
//! CRITICAL: L0 is written **UNSORTED**. Rows already arrive seq-monotonic (seq
//! is allocated under the insertion lock at append time); the explicit
//! `ORDER BY (key, seq)` sort happens only at L0->L1 compaction, so a single
//! write's sort cost lands once in the bg compactor, not on every flush.
//! `write_segment` therefore writes the batch verbatim.

use std::io::Write;
use std::path::{Path, PathBuf};

use arrow::array::RecordBatch;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;

use crate::errors::StatsError;
use crate::store::types::{parse_seg_filename, seg_filename};

/// Parquet row-group size.
pub const ROW_GROUP_SIZE: usize = 16_384;

/// Parquet `WriterProperties` shared by every finelog segment writer — the L0
/// flush (`write_segment`) and the compaction output (`write_merged_segment`).
///
/// Encoding contract: row-group 16384, zstd level 1 (not the library default 3),
/// and bloom filters enabled (so EXACT-key FetchLogs / equality predicates get
/// row-group pruning). Centralizing it keeps L0 and compacted segments using one
/// consistent on-disk layout.
pub fn segment_writer_properties() -> Result<WriterProperties, StatsError> {
    let zstd =
        ZstdLevel::try_new(1).map_err(|e| StatsError::Internal(format!("zstd level 1: {e}")))?;
    Ok(WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .set_compression(Compression::ZSTD(zstd))
        .set_bloom_filter_enabled(true)
        .build())
}

/// Per-segment metadata recovered from filename + parquet footer.
///
/// `min_seq` comes from the FILENAME (`seg_L{level}_{min_seq}`); `max_seq` is
/// `min_seq + row_count - 1`. `min_key_value`/`max_key_value` are the parquet
/// column statistics for the key column when it is an Int64 column carrying
/// statistics, else `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMetadata {
    pub level: i32,
    pub min_seq: i64,
    pub max_seq: i64,
    pub row_count: i64,
    pub min_key_value: Option<i64>,
    pub max_key_value: Option<i64>,
}

/// Encode `batch` to parquet bytes (UNSORTED L0, row-group 16384, zstd-1, bloom).
pub fn write_segment(batch: &RecordBatch) -> Result<Vec<u8>, StatsError> {
    let props = segment_writer_properties()?;
    let mut buf: Vec<u8> = Vec::new();
    let opts = ArrowWriterOptions::new().with_properties(props);
    let mut writer = ArrowWriter::try_new_with_options(&mut buf, batch.schema(), opts)
        .map_err(|e| StatsError::Internal(format!("parquet writer init: {e}")))?;
    writer
        .write(batch)
        .map_err(|e| StatsError::Internal(format!("parquet write: {e}")))?;
    writer
        .close()
        .map_err(|e| StatsError::Internal(format!("parquet close: {e}")))?;
    Ok(buf)
}

/// Write `batch` to `{dir}/seg_L{level}_{min_seq}.parquet` via a staging
/// `.parquet.tmp` file + atomic rename. Returns the final path and the file's
/// byte size on disk.
pub fn write_segment_to_dir(
    dir: &Path,
    level: i32,
    min_seq: i64,
    batch: &RecordBatch,
) -> Result<(PathBuf, i64), StatsError> {
    let bytes = write_segment(batch)?;
    let filename = seg_filename(level, min_seq);
    let final_path = dir.join(&filename);
    let staging_path = dir.join(format!("{filename}.tmp"));
    {
        let mut f = std::fs::File::create(&staging_path).map_err(|e| {
            StatsError::Internal(format!("create staging {}: {e}", staging_path.display()))
        })?;
        f.write_all(&bytes).map_err(|e| {
            StatsError::Internal(format!("write staging {}: {e}", staging_path.display()))
        })?;
        f.sync_all().map_err(|e| {
            StatsError::Internal(format!("fsync staging {}: {e}", staging_path.display()))
        })?;
    }
    std::fs::rename(&staging_path, &final_path).map_err(|e| {
        StatsError::Internal(format!(
            "rename {} -> {}: {e}",
            staging_path.display(),
            final_path.display()
        ))
    })?;
    let size = std::fs::metadata(&final_path)
        .map_err(|e| StatsError::Internal(format!("stat {}: {e}", final_path.display())))?
        .len() as i64;
    Ok((final_path, size))
}

/// Read a segment's footer metadata: row count from the footer, `min_seq` from
/// the FILENAME, `max_seq = min_seq + row_count - 1`, and the Int64 key-column
/// min/max from row-group statistics.
///
/// Returns `None` for an unparseable filename or footer-read failure (the caller
/// treats that as an empty/discardable segment).
pub fn read_segment_footer(path: &Path, key_column: Option<&str>) -> Option<SegmentMetadata> {
    let name = path.file_name()?.to_str()?;
    let (level, min_seq) = parse_seg_filename(name)?;
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    let md = reader.metadata();
    let num_rows = md.file_metadata().num_rows();
    if num_rows <= 0 {
        return Some(SegmentMetadata {
            level,
            min_seq,
            max_seq: min_seq,
            row_count: 0,
            min_key_value: None,
            max_key_value: None,
        });
    }
    let (min_key, max_key) = key_column
        .and_then(|kc| key_int64_bounds(&reader, kc))
        .unwrap_or((None, None));
    Some(SegmentMetadata {
        level,
        min_seq,
        max_seq: min_seq + num_rows - 1,
        row_count: num_rows,
        min_key_value: min_key,
        max_key_value: max_key,
    })
}

/// Aggregate Int64 (min, max) for `key_column` across all row groups, or `None`
/// if the column is absent or carries no Int64 statistics.
fn key_int64_bounds(
    reader: &SerializedFileReader<std::fs::File>,
    key_column: &str,
) -> Option<(Option<i64>, Option<i64>)> {
    let md = reader.metadata();
    let schema = md.file_metadata().schema_descr();
    let col_idx = (0..schema.num_columns()).find(|&i| schema.column(i).name() == key_column)?;
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    for rg in md.row_groups() {
        if let Some(Statistics::Int64(s)) = rg.column(col_idx).statistics() {
            if let Some(&m) = s.min_opt() {
                lo = Some(lo.map_or(m, |x: i64| x.min(m)));
            }
            if let Some(&m) = s.max_opt() {
                hi = Some(hi.map_or(m, |x: i64| x.max(m)));
            }
        }
    }
    Some((lo, hi))
}

/// Footer-only `(row_count, min_key, max_key)` for `key_column` in the parquet
/// file at `path`.
///
/// Reads only the footer (no column page scan). `min_key`/`max_key` are the
/// aggregated Int64 statistics for `key_column` across row groups, or `None`
/// when the column is absent / key-less / carries no Int64 statistics. Used by
/// the executor to recover a merged segment's row_count cheaply and by boot
/// adoption. Returns `None` only on an unreadable footer.
pub fn segment_bounds(
    path: &Path,
    key_column: Option<&str>,
) -> Option<(i64, Option<i64>, Option<i64>)> {
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    let num_rows = reader.metadata().file_metadata().num_rows();
    let (lo, hi) = key_column
        .and_then(|kc| key_int64_bounds(&reader, kc))
        .unwrap_or((None, None));
    Some((num_rows, lo, hi))
}

/// Footer-only sum of each row group's uncompressed byte size, or `None` on an
/// unreadable footer.
///
/// This is the size of the segment's decoded column data — the scale of the
/// Arrow arrays a merge materializes in RAM. The on-disk (compressed) size
/// understates it by the compression ratio, which reaches 10-20x on log text,
/// so the compaction planner budgets merge memory against this instead.
pub fn segment_uncompressed_bytes(path: &Path) -> Option<i64> {
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    Some(
        reader
            .metadata()
            .row_groups()
            .iter()
            .map(|rg| rg.total_byte_size())
            .sum(),
    )
}

/// Footer-only row-group count for the parquet file at `path`, or `None` on an
/// unreadable footer. Used by the trigram prune to confirm a sidecar's
/// per-row-group entries align with the segment before attaching an access plan.
pub fn segment_row_group_count(path: &Path) -> Option<usize> {
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    Some(reader.metadata().num_row_groups())
}

/// All `seg_L*_*.parquet` files in `dir`, sorted by filename (== by min_seq for
/// a fixed level width). Returns an empty list if the dir does not exist.
pub fn discover_segments(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if parse_seg_filename(name).is_some() {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Recover the next seq to allocate by scanning `dir`'s segment footers.
///
/// Returns `max(max_seq over all segments) + 1`, or `1` when no segments exist.
pub fn recover_next_seq(dir: &Path) -> i64 {
    let mut next_seq = 1_i64;
    for p in discover_segments(dir) {
        if let Some(meta) = read_segment_footer(&p, None) {
            if meta.max_seq + 1 > next_seq {
                next_seq = meta.max_seq + 1;
            }
        }
    }
    next_seq
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_segment_test_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Build a seq-stamped batch with a `key` Int64 column (non-monotonic to
    /// prove UNSORTED writes preserve row order).
    fn batch_with_keys(first_seq: i64, keys: Vec<i64>) -> RecordBatch {
        let n = keys.len() as i64;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
        ]));
        let seqs: Int64Array = (first_seq..first_seq + n).collect();
        let ids: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(seqs),
                Arc::new(Int64Array::from(keys)),
                Arc::new(StringArray::from(ids)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn write_and_read_footer_round_trips_seq_window_and_key_bounds() {
        let dir = tempdir();
        // non-monotonic keys: 30, 10, 20.
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        let (path, size) = write_segment_to_dir(&dir, 0, 1, &batch).unwrap();
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "seg_L0_0000000000000000001.parquet"
        );
        assert!(size > 0);

        let meta = read_segment_footer(&path, Some("key")).unwrap();
        assert_eq!(meta.level, 0);
        assert_eq!(meta.min_seq, 1);
        assert_eq!(meta.max_seq, 3);
        assert_eq!(meta.row_count, 3);
        assert_eq!(meta.min_key_value, Some(10));
        assert_eq!(meta.max_key_value, Some(30));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn l0_write_is_unsorted_preserving_row_order() {
        let dir = tempdir();
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        let (path, _) = write_segment_to_dir(&dir, 0, 1, &batch).unwrap();
        // Read the rows back; their key order must be the on-write order.
        let file = std::fs::File::open(&path).unwrap();
        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();
        let read = reader.next().unwrap().unwrap();
        let key_idx = read.schema().index_of("key").unwrap();
        let keys = read
            .column(key_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(keys.values(), &[30_i64, 10, 20], "L0 must be UNSORTED");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recover_next_seq_over_two_segments() {
        let dir = tempdir();
        // segment 1: seqs 1..3 (min_seq 1); segment 2: seqs 4..5 (min_seq 4).
        write_segment_to_dir(&dir, 0, 1, &batch_with_keys(1, vec![1, 2, 3])).unwrap();
        write_segment_to_dir(&dir, 0, 4, &batch_with_keys(4, vec![4, 5])).unwrap();
        assert_eq!(recover_next_seq(&dir), 6);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recover_next_seq_empty_dir_is_one() {
        let dir = tempdir();
        assert_eq!(recover_next_seq(&dir), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn seg_filename_round_trips() {
        let name = seg_filename(0, 42);
        assert_eq!(name, "seg_L0_0000000000000000042.parquet");
        assert_eq!(parse_seg_filename(&name), Some((0, 42)));
    }
}
