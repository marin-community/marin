//! L0 parquet segment writer + footer recovery.
//!
//! CRITICAL: L0 is written **UNSORTED**. Rows already arrive seq-monotonic (seq
//! is allocated under the insertion lock at append time); the explicit
//! `ORDER BY (key, seq)` sort happens only at L0->L1 compaction, so a single
//! write's sort cost lands once in the bg compactor, not on every flush.
//! `write_segment` therefore writes the batch verbatim.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use arrow::array::RecordBatch;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use parquet::schema::types::ColumnPath;

use crate::errors::StatsError;
use crate::store::types::{parse_seg_filename, seg_filename};

/// Bytes of in-memory row data a parquet row group should hold.
///
/// Row groups are the unit of footer metadata: every one costs a record per
/// column, with offsets, encodings, sizes, and min/max statistics. Sizing them by
/// rows alone makes that cost track a namespace's row *width* — telemetry rows
/// are ~20x narrower than log lines, so a fixed row count gave `telemetry_v1`
/// 108K row groups and 206 MiB of footer for 15 GiB of data, most of a query's
/// latency before any column was read. Sizing by bytes keeps the footer
/// proportional to the data instead.
const TARGET_ROW_GROUP_BYTES: usize = 16 * 1024 * 1024;

/// Row-group bounds. The floor keeps small segments from collapsing to a single
/// row group (which prunes nothing); the ceiling bounds the writer's buffered
/// row group and keeps key statistics from covering too wide a band.
const MIN_ROW_GROUP_ROWS: usize = 16_384;
const MAX_ROW_GROUP_ROWS: usize = 1_048_576;

/// Rows per row group for a segment holding `batches`, from their mean in-memory
/// width. Empty input takes the floor.
pub fn row_group_rows(batches: &[RecordBatch]) -> usize {
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    if rows == 0 {
        return MIN_ROW_GROUP_ROWS;
    }
    let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    let bytes_per_row = (bytes / rows).max(1);
    (TARGET_ROW_GROUP_BYTES / bytes_per_row).clamp(MIN_ROW_GROUP_ROWS, MAX_ROW_GROUP_ROWS)
}

/// Parquet `WriterProperties` shared by every finelog segment writer — the L0
/// flush (`write_segment`) and the compaction output (`write_merged_segment`).
///
/// Sets `row_group_rows` (see [`row_group_rows`]), zstd level 1 (not the library
/// default 3), and bloom filters: `Some(col)` writes one for exactly that column,
/// `None` writes none. Centralizing this keeps L0 and compacted segments on one
/// consistent on-disk layout.
///
/// Callers pass the key column for L0 and `None` for compacted output. L0 is
/// written unsorted, so its key statistics span the namespace and a bloom is the
/// only thing that prunes an exact-key lookup; L1+ is sorted by `(key, seq)`, so
/// min/max statistics already prune the key band.
pub fn segment_writer_properties(
    bloom_column: Option<&str>,
    row_group_rows: usize,
) -> Result<WriterProperties, StatsError> {
    let zstd =
        ZstdLevel::try_new(1).map_err(|e| StatsError::Internal(format!("zstd level 1: {e}")))?;
    let mut builder = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_rows))
        .set_compression(Compression::ZSTD(zstd))
        .set_bloom_filter_enabled(false);
    if let Some(column) = bloom_column {
        builder = builder.set_column_bloom_filter_enabled(ColumnPath::from(column), true);
    }
    Ok(builder.build())
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

/// Encode `batch` to parquet bytes (UNSORTED L0, zstd-1, a bloom filter on
/// `key_column`).
pub fn write_segment(batch: &RecordBatch, key_column: Option<&str>) -> Result<Vec<u8>, StatsError> {
    let props = segment_writer_properties(key_column, row_group_rows(std::slice::from_ref(batch)))?;
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
    key_column: Option<&str>,
) -> Result<(PathBuf, i64), StatsError> {
    let bytes = write_segment(batch, key_column)?;
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

/// Cached row-group layouts, keyed by path and the file identity (length and
/// modified time) the layout was read from. Cleared wholesale past
/// [`ROW_GROUP_CACHE_ENTRIES`] rather than evicted one at a time: a segment
/// is written once and read many times, so entries only go stale when a segment
/// is compacted away, and re-reading a few footers after a clear is far cheaper
/// than tracking liveness.
static ROW_GROUP_LAYOUTS: OnceLock<Mutex<HashMap<PathBuf, CachedRowGroups>>> = OnceLock::new();

/// A segment's row-group row counts and the file identity they were read from.
struct CachedRowGroups {
    len: u64,
    modified: SystemTime,
    rows: Arc<[usize]>,
}

/// Entries held before the row-group cache is cleared. A hub holds low thousands
/// of segments across every namespace, so this is headroom, not a working limit.
const ROW_GROUP_CACHE_ENTRIES: usize = 8192;

/// Footer-only row counts, one per row group, for the parquet file at `path`, or
/// `None` on an unreadable footer. The trigram prune uses them to map its span
/// mask onto row groups.
///
/// Reading them means parsing the whole footer — every row group's metadata —
/// which over a namespace's segments costs more than the scan the prune is there
/// to avoid. Segments are immutable once written, so the layout is cached against
/// the file's length and modified time; a path written again is read again rather
/// than answered from the old entry.
pub fn segment_row_group_rows(path: &Path) -> Option<Arc<[usize]>> {
    let file = std::fs::File::open(path).ok()?;
    let meta = file.metadata().ok()?;
    let (len, modified) = (meta.len(), meta.modified().ok()?);

    let cache = ROW_GROUP_LAYOUTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(entry) = cache.lock().unwrap().get(path) {
        if entry.len == len && entry.modified == modified {
            return Some(Arc::clone(&entry.rows));
        }
    }

    let reader = SerializedFileReader::new(file).ok()?;
    let rows: Arc<[usize]> = reader
        .metadata()
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as usize)
        .collect();
    let mut cache = cache.lock().unwrap();
    if cache.len() >= ROW_GROUP_CACHE_ENTRIES {
        cache.clear();
    }
    cache.insert(
        path.to_path_buf(),
        CachedRowGroups {
            len,
            modified,
            rows: Arc::clone(&rows),
        },
    );
    Some(rows)
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

    /// The layout is cached against the file's identity, not just its path, so a
    /// path written again is re-read.
    #[test]
    fn row_group_layout_is_reread_when_the_file_changes() {
        let dir = tempdir();
        let path = dir.join("seg_L1_0000000000000000001.parquet");

        let one_group = batch_with_keys(1, (0..10).collect());
        std::fs::write(&path, write_segment(&one_group, None).unwrap()).unwrap();
        assert_eq!(segment_row_group_rows(&path).as_deref(), Some(&[10][..]));
        assert_eq!(
            segment_row_group_rows(&path).as_deref(),
            Some(&[10][..]),
            "second call cached"
        );

        // A different layout at the same path must not be answered from the old
        // entry: a stale layout would make the trigram prune reject a valid
        // sidecar, or attach a plan the parquet opener rejects.
        let more = batch_with_keys(1, (0..25).collect());
        std::fs::write(&path, write_segment(&more, None).unwrap()).unwrap();
        assert_eq!(segment_row_group_rows(&path).as_deref(), Some(&[25][..]));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_and_read_footer_round_trips_seq_window_and_key_bounds() {
        let dir = tempdir();
        // non-monotonic keys: 30, 10, 20.
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        let (path, size) = write_segment_to_dir(&dir, 0, 1, &batch, Some("key")).unwrap();
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

    /// Column names carrying a parquet bloom filter in `path`'s first row group.
    fn bloom_columns(path: &Path) -> Vec<String> {
        let file = std::fs::File::open(path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let rg = reader.metadata().row_group(0);
        (0..rg.num_columns())
            .filter(|&i| rg.column(i).bloom_filter_offset().is_some())
            .map(|i| rg.column(i).column_path().string())
            .collect()
    }

    #[test]
    fn only_the_named_column_gets_a_bloom_filter() {
        let dir = tempdir();
        let batch = batch_with_keys(1, vec![30, 10, 20]);

        // L0 carries one for its key, which is the only prune available on an
        // unsorted segment.
        let (keyed, _) = write_segment_to_dir(&dir, 0, 1, &batch, Some("key")).unwrap();
        assert_eq!(bloom_columns(&keyed), vec!["key".to_string()]);

        // Compacted output names no column, so the segment carries none.
        let (unkeyed, _) = write_segment_to_dir(&dir, 0, 9, &batch, None).unwrap();
        assert!(bloom_columns(&unkeyed).is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn l0_write_is_unsorted_preserving_row_order() {
        let dir = tempdir();
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        let (path, _) = write_segment_to_dir(&dir, 0, 1, &batch, Some("key")).unwrap();
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
    fn seg_filename_round_trips() {
        let name = seg_filename(0, 42);
        assert_eq!(name, "seg_L0_0000000000000000042.parquet");
        assert_eq!(parse_seg_filename(&name), Some((0, 42)));
    }
}
