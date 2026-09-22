//! L0 parquet segment writer + footer recovery.
//!
//! CRITICAL: L0 is written **UNSORTED**. Rows already arrive seq-monotonic (seq
//! is allocated under the insertion lock at append time); the explicit
//! configured sort plus `seq` happens only at L0->L1 compaction, so a single
//! write's sort cost lands once in the bg compactor, not on every flush.
//! `write_segment` therefore writes the batch verbatim.

use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use arrow::array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use parquet::basic::{Compression, Type as PhysicalType, ZstdLevel};
use parquet::file::metadata::{KeyValue, ParquetMetaData};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::errors::StatsError;
use crate::partition_policy::SegmentPartition;
use crate::store::types::{parse_seg_filename, seg_filename};

/// Encoded size at which the parquet writer closes a row group.
///
/// Row groups are the unit of footer metadata: every one costs a thrift record
/// per column, carrying offsets, encodings, sizes, and min/max statistics. Sizing
/// them by rows alone makes that cost track a namespace's row *width* — a
/// telemetry row compresses to ~8 bytes where a log line takes hundreds, so a
/// fixed 16K-row group gave `telemetry_v1` 108K row groups and 201 MiB of footer
/// for 15 GiB of data, most of a query's latency before any column was read.
///
/// The target is denominated in ENCODED bytes, which is what footer weight
/// actually tracks. An in-memory (Arrow) target would miss by the compression
/// ratio, and that ratio is itself width-dependent: telemetry compresses ~66x
/// against a log line's ~4x, so the narrow namespace this is meant to fix is
/// exactly where an Arrow-denominated target under-sizes worst.
pub const TARGET_ROW_GROUP_BYTES: usize = 16 * 1024 * 1024;

/// Ceiling on rows per row group, applied alongside the byte target (parquet
/// closes the group at whichever binds first).
///
/// Row-group min/max statistics are what prune a key range, so an extremely
/// compressible namespace should not collapse into a few groups each spanning a
/// large share of the key space. Substring pruning is unaffected either way: a
/// sidecar span is 16,384 rows regardless, and a partly-covered row group is
/// pruned with a row selection rather than skipped whole.
pub const MAX_ROW_GROUP_ROWS: usize = 1_048_576;

/// Physical layout revision stamped into every segment's parquet footer.
///
/// Bump this when the writer's row-group or encoding policy changes. It
/// describes the file's physical shape ONLY — two segments at different
/// versions hold identical rows in identical order — which is what lets
/// maintenance re-encode a stale one in place without touching the catalog's
/// view of its contents or its remote copy.
pub const LAYOUT_VERSION: u32 = 1;
const LAYOUT_VERSION_KEY: &str = "finelog.layout_version";
const SEGMENT_ID_KEY: &str = "finelog.segment_id";
const PARTITION_KEY: &str = "finelog.partition";

/// Rows per batch when streaming a segment through a re-encode. Bounds the
/// rewrite's memory to one batch rather than the whole segment, which for a
/// terminal-level file is hundreds of MiB of Arrow.
const REWRITE_BATCH_ROWS: usize = 8_192;

/// Parquet `WriterProperties` shared by every finelog segment writer — the L0
/// flush (`write_segment`) and the compaction output (`write_merged_segment`).
///
/// Sets the row-group bounds ([`TARGET_ROW_GROUP_BYTES`] and the caller's row
/// ceiling) and zstd level 1 (not the library default 3).
/// Centralizing this keeps L0 and compacted segments on one consistent on-disk
/// layout.
///
/// No segment carries a parquet bloom filter. Writing them for every column cost
/// 15% of each segment and pruned nothing measurable; the key-column bloom that
/// outlived that only served exact-key lookups against unsorted L0, which is a
/// few hundred KiB that compaction consumes within a tick or two, while its write
/// cost fell on every flush. Multi-input compaction uses the schema's configured
/// sort order plus `seq`; single-input promotions retain their input order.
/// Substring queries prune from the trigram sidecar.
pub fn segment_writer_properties_with_max_rows(
    max_row_group_rows: usize,
) -> Result<WriterProperties, StatsError> {
    segment_writer_properties_with_partition(max_row_group_rows, None)
}

pub fn segment_writer_properties_with_partition(
    max_row_group_rows: usize,
    partition: Option<&SegmentPartition>,
) -> Result<WriterProperties, StatsError> {
    parquet_writer_properties_with_id(
        TARGET_ROW_GROUP_BYTES,
        max_row_group_rows,
        Uuid::new_v4(),
        partition,
    )
}

pub(crate) fn parquet_writer_properties(
    target_row_group_bytes: usize,
    max_row_group_rows: usize,
) -> Result<WriterProperties, StatsError> {
    parquet_writer_properties_with_id(
        target_row_group_bytes,
        max_row_group_rows,
        Uuid::new_v4(),
        None,
    )
}

fn parquet_writer_properties_with_id(
    target_row_group_bytes: usize,
    max_row_group_rows: usize,
    segment_id: Uuid,
    partition: Option<&SegmentPartition>,
) -> Result<WriterProperties, StatsError> {
    let zstd =
        ZstdLevel::try_new(1).map_err(|e| StatsError::Internal(format!("zstd level 1: {e}")))?;
    let mut metadata = vec![
        KeyValue::new(LAYOUT_VERSION_KEY.to_string(), LAYOUT_VERSION.to_string()),
        KeyValue::new(SEGMENT_ID_KEY.to_string(), segment_id.to_string()),
    ];
    if let Some(partition) = partition {
        metadata.push(KeyValue::new(
            PARTITION_KEY.to_string(),
            serde_json::to_string(partition).map_err(|error| {
                StatsError::Internal(format!("serialize segment partition: {error}"))
            })?,
        ));
    }
    Ok(WriterProperties::builder()
        .set_max_row_group_bytes(Some(target_row_group_bytes))
        .set_max_row_group_row_count(Some(max_row_group_rows))
        .set_compression(Compression::ZSTD(zstd))
        .set_bloom_filter_enabled(false)
        .set_key_value_metadata(Some(metadata))
        .build())
}

/// Per-segment metadata recovered from filename + parquet footer.
///
/// `min_seq`/`max_seq` come from Parquet statistics because partitioned files
/// contain sparse namespace-wide sequence ranges. Files written before `seq`
/// statistics were available fall back to the filename and row count.
/// `min_key_value`/`max_key_value` are the Parquet statistics for an Int64 or
/// UTF-8 key, encoded in the key's logical string representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMetadata {
    pub level: i32,
    pub min_seq: i64,
    pub max_seq: i64,
    pub row_count: i64,
    pub min_key_value: Option<String>,
    pub max_key_value: Option<String>,
    pub partition: Option<SegmentPartition>,
}

/// Encode `batch` to parquet bytes (UNSORTED L0, zstd-1).
pub fn write_segment(batch: &RecordBatch) -> Result<Vec<u8>, StatsError> {
    write_segment_with_max_row_group_rows(batch, MAX_ROW_GROUP_ROWS)
}

pub fn write_segment_with_max_row_group_rows(
    batch: &RecordBatch,
    max_row_group_rows: usize,
) -> Result<Vec<u8>, StatsError> {
    let props = segment_writer_properties_with_max_rows(max_row_group_rows)?;
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
    write_segment_to_dir_with_max_row_group_rows(dir, level, min_seq, batch, MAX_ROW_GROUP_ROWS)
}

pub fn write_segment_to_dir_with_max_row_group_rows(
    dir: &Path,
    level: i32,
    min_seq: i64,
    batch: &RecordBatch,
    max_row_group_rows: usize,
) -> Result<(PathBuf, i64), StatsError> {
    let bytes = write_segment_with_max_row_group_rows(batch, max_row_group_rows)?;
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

/// The physical shape of one segment file, read from its parquet footer.
///
/// Every field is footer-resident, so filling this costs a tail read rather
/// than a scan — but it is still a read per segment, which is why the
/// introspection route asks for it explicitly instead of always.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentPhysical {
    /// Layout revision stamped at write time. `None` for a segment written
    /// before the stamp existed, which reads the same as "stale".
    pub layout_version: Option<u32>,
    pub row_groups: usize,
    pub rows: i64,
    /// Bytes the footer occupies on disk, including its length prefix and
    /// trailing magic. This is the per-segment cost a query pays before it
    /// reads a single column.
    pub footer_bytes: i64,
    /// Summed uncompressed column-chunk size. Against the file size on disk
    /// this gives the segment's compression ratio.
    pub uncompressed_bytes: i64,
    /// Writer that produced the file, e.g. `parquet-rs version 58.3.0`.
    pub created_by: Option<String>,
}

/// Read `path`'s footer and report its physical shape.
pub fn segment_physical(path: &Path) -> Result<SegmentPhysical, StatsError> {
    let file = std::fs::File::open(path)
        .map_err(|e| StatsError::Internal(format!("open {}: {e}", path.display())))?;
    let footer_bytes = parquet_footer_bytes(&file)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| StatsError::Internal(format!("footer {}: {e}", path.display())))?;
    let meta = reader.metadata();
    let uncompressed_bytes = meta
        .row_groups()
        .iter()
        .fold(0_i64, |total, rg| total + rg.total_byte_size());
    Ok(SegmentPhysical {
        layout_version: layout_version_of(meta),
        row_groups: meta.num_row_groups(),
        rows: meta.file_metadata().num_rows(),
        footer_bytes,
        uncompressed_bytes,
        created_by: meta.file_metadata().created_by().map(str::to_string),
    })
}

/// Size of the parquet footer at the tail of `file`: the serialized metadata
/// plus its 4-byte length and the 4-byte `PAR1` magic.
fn parquet_footer_bytes(file: &std::fs::File) -> Result<i64, StatsError> {
    let len = file
        .metadata()
        .map_err(|e| StatsError::Internal(format!("stat segment: {e}")))?
        .len();
    if len < FOOTER_TAIL_BYTES as u64 {
        return Err(StatsError::Internal(format!(
            "segment is {len} bytes, too short to hold a parquet footer"
        )));
    }
    let mut tail = [0_u8; FOOTER_TAIL_BYTES];
    file.read_exact_at(&mut tail, len - FOOTER_TAIL_BYTES as u64)
        .map_err(|e| StatsError::Internal(format!("read footer tail: {e}")))?;
    if &tail[4..] != PARQUET_MAGIC {
        return Err(StatsError::Internal("not a parquet file".to_string()));
    }
    let metadata_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as i64;
    Ok(metadata_len + FOOTER_TAIL_BYTES as i64)
}

/// Trailing `[metadata_len: u32][PAR1]` every parquet file ends with.
const FOOTER_TAIL_BYTES: usize = 8;
const PARQUET_MAGIC: &[u8] = b"PAR1";

/// The layout revision stamped in `meta`, or `None` when it carries no stamp.
fn layout_version_of(meta: &ParquetMetaData) -> Option<u32> {
    meta.file_metadata()
        .key_value_metadata()
        .and_then(|kvs| kvs.iter().find(|kv| kv.key == LAYOUT_VERSION_KEY))
        .and_then(|kv| kv.value.as_deref())
        .and_then(|v| v.parse::<u32>().ok())
}

fn segment_id_of(meta: &ParquetMetaData) -> Option<Uuid> {
    meta.file_metadata()
        .key_value_metadata()
        .and_then(|kvs| kvs.iter().find(|kv| kv.key == SEGMENT_ID_KEY))
        .and_then(|kv| kv.value.as_deref())
        .and_then(|value| Uuid::parse_str(value).ok())
}

fn partition_of(meta: &ParquetMetaData) -> Option<SegmentPartition> {
    meta.file_metadata()
        .key_value_metadata()
        .and_then(|kvs| kvs.iter().find(|kv| kv.key == PARTITION_KEY))
        .and_then(|kv| kv.value.as_deref())
        .and_then(|value| serde_json::from_str(value).ok())
}

/// Immutable logical identity stamped into a segment's Parquet metadata.
pub fn segment_id(path: &Path) -> Option<Uuid> {
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    segment_id_of(reader.metadata())
}

/// Stable local generation identity for an immutable segment.
///
/// New segments carry a UUID in their Parquet metadata. Segments written before
/// that stamp was introduced derive an identity from their Unix file generation
/// instead, so they can acquire indexes without forcing a fleet-wide rewrite.
/// Finelog replaces immutable files with rename, which changes this identity;
/// ordinary level-bump renames preserve it.
pub fn segment_identity(path: &Path) -> Option<Uuid> {
    let file = std::fs::File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    Some(segment_id_of(reader.metadata()).unwrap_or_else(|| legacy_segment_identity(&metadata)))
}

fn legacy_segment_identity(metadata: &std::fs::Metadata) -> Uuid {
    let mut hasher = Sha256::new();
    hasher.update(b"finelog.local-segment-generation.v1\0");
    hasher.update(metadata.dev().to_le_bytes());
    hasher.update(metadata.ino().to_le_bytes());
    hasher.update(metadata.len().to_le_bytes());
    hasher.update(metadata.mtime().to_le_bytes());
    hasher.update(metadata.mtime_nsec().to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes)
}

/// Whether `path` was written by the current [`LAYOUT_VERSION`]. A segment
/// written before the stamp existed, or by an older policy, reads as stale.
pub fn segment_layout_is_current(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return true; // unreadable: leave it alone, not a rewrite candidate
    };
    let Ok(reader) = SerializedFileReader::new(file) else {
        return true;
    };
    layout_version_of(reader.metadata()).is_some_and(|v| v == LAYOUT_VERSION)
}

/// Re-encode `path` under the current writer properties into a sibling
/// `.parquet.tmp`, preserving its rows and their order exactly. Returns the
/// staging path and its size; the caller renames it over `path` to commit.
///
/// Staging and committing are separate because the rewrite is slow and takes no
/// lock: eviction may unlink the segment while this runs, and renaming over a
/// path the catalog has since dropped would resurrect an untracked file. The
/// caller commits under the insertion lock, after re-checking the segment.
///
/// The FILENAME is unchanged, which is what makes this cheap: the remote archive
/// keys objects by basename and only uploads segments the catalog still marks
/// `Local`, so a rewritten segment is never re-uploaded. Its remote copy keeps
/// the old physical layout while holding the same rows, and ages out normally.
///
/// Streams a batch at a time rather than materializing the segment, which for a
/// terminal-level file would be hundreds of MiB of Arrow.
pub fn stage_rewritten_segment(
    path: &Path,
    max_row_group_rows: usize,
) -> Result<(PathBuf, i64), StatsError> {
    let file = std::fs::File::open(path)
        .map_err(|e| StatsError::Internal(format!("open {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StatsError::Internal(format!("read {}: {e}", path.display())))?;
    let segment_id = segment_id_of(builder.metadata()).unwrap_or_else(Uuid::new_v4);
    let partition = partition_of(builder.metadata());
    let schema = Arc::clone(builder.schema());
    let reader = builder
        .with_batch_size(REWRITE_BATCH_ROWS)
        .build()
        .map_err(|e| StatsError::Internal(format!("reader {}: {e}", path.display())))?;

    let staging = PathBuf::from(format!("{}.tmp", path.display()));
    let out = std::fs::File::create(&staging)
        .map_err(|e| StatsError::Internal(format!("create {}: {e}", staging.display())))?;
    let opts = ArrowWriterOptions::new().with_properties(parquet_writer_properties_with_id(
        TARGET_ROW_GROUP_BYTES,
        max_row_group_rows,
        segment_id,
        partition.as_ref(),
    )?);
    let mut writer = ArrowWriter::try_new_with_options(out, schema, opts)
        .map_err(|e| StatsError::Internal(format!("parquet writer init: {e}")))?;
    for batch in reader {
        let batch =
            batch.map_err(|e| StatsError::Internal(format!("decode {}: {e}", path.display())))?;
        writer
            .write(&batch)
            .map_err(|e| StatsError::Internal(format!("parquet write: {e}")))?;
    }
    let out = writer
        .into_inner()
        .map_err(|e| StatsError::Internal(format!("parquet close: {e}")))?;
    out.sync_all()
        .map_err(|e| StatsError::Internal(format!("fsync {}: {e}", staging.display())))?;
    let size = out
        .metadata()
        .map_err(|e| StatsError::Internal(format!("stat {}: {e}", staging.display())))?
        .len() as i64;
    Ok((staging, size))
}

/// Read a segment's footer metadata, including actual seq statistics and the
/// optional hidden partition stamp.
///
/// Returns `None` for an unparseable filename or footer-read failure (the caller
/// treats that as an empty/discardable segment).
pub fn read_segment_footer(path: &Path, key_column: Option<&str>) -> Option<SegmentMetadata> {
    let name = path.file_name()?.to_str()?;
    let (level, filename_min_seq) = parse_seg_filename(name)?;
    read_segment_footer_at(path, level, filename_min_seq, key_column)
}

/// Read footer metadata for an opaque object whose filename does not
/// encode its level or sequence origin.
pub fn read_segment_footer_at(
    path: &Path,
    level: i32,
    filename_min_seq: i64,
    key_column: Option<&str>,
) -> Option<SegmentMetadata> {
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    segment_metadata_from_parquet(reader.metadata(), level, filename_min_seq, key_column)
}

pub(crate) fn segment_metadata_from_parquet(
    metadata: &ParquetMetaData,
    level: i32,
    filename_min_seq: i64,
    key_column: Option<&str>,
) -> Option<SegmentMetadata> {
    let partition = partition_of(metadata);
    let num_rows = metadata.file_metadata().num_rows();
    if num_rows <= 0 {
        return Some(SegmentMetadata {
            level,
            min_seq: filename_min_seq,
            max_seq: filename_min_seq,
            row_count: 0,
            min_key_value: None,
            max_key_value: None,
            partition,
        });
    }
    let (seq_min, seq_max) = metadata_int64_bounds(metadata, "seq")?;
    let min_seq = seq_min.unwrap_or(filename_min_seq);
    let max_seq = seq_max.unwrap_or(filename_min_seq + num_rows - 1);
    let (min_key, max_key) = key_column
        .and_then(|kc| metadata_key_bounds(metadata, kc))
        .unwrap_or((None, None));
    Some(SegmentMetadata {
        level,
        min_seq,
        max_seq,
        row_count: num_rows,
        min_key_value: min_key,
        max_key_value: max_key,
        partition,
    })
}

fn metadata_int64_bounds(
    md: &ParquetMetaData,
    key_column: &str,
) -> Option<(Option<i64>, Option<i64>)> {
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

fn metadata_key_bounds(
    metadata: &ParquetMetaData,
    key_column: &str,
) -> Option<(Option<String>, Option<String>)> {
    let schema = metadata.file_metadata().schema_descr();
    let column =
        (0..schema.num_columns()).find(|&index| schema.column(index).name() == key_column)?;
    match schema.column(column).physical_type() {
        PhysicalType::INT64 => {
            metadata_int64_bounds(metadata, key_column).map(|(minimum, maximum)| {
                (
                    minimum.map(|value| value.to_string()),
                    maximum.map(|value| value.to_string()),
                )
            })
        }
        PhysicalType::BYTE_ARRAY => {
            let mut minimum: Option<Vec<u8>> = None;
            let mut maximum: Option<Vec<u8>> = None;
            for row_group in metadata.row_groups() {
                let Some(Statistics::ByteArray(statistics)) = row_group.column(column).statistics()
                else {
                    continue;
                };
                if let Some(value) = statistics.min_opt() {
                    let value = value.data();
                    if minimum.as_deref().is_none_or(|current| value < current) {
                        minimum = Some(value.to_vec());
                    }
                }
                if let Some(value) = statistics.max_opt() {
                    let value = value.data();
                    if maximum.as_deref().is_none_or(|current| value > current) {
                        maximum = Some(value.to_vec());
                    }
                }
            }
            Some((
                minimum.and_then(|value| String::from_utf8(value).ok()),
                maximum.and_then(|value| String::from_utf8(value).ok()),
            ))
        }
        _ => Some((None, None)),
    }
}

/// Footer-only `(row_count, min_key, max_key)` for `key_column` in the parquet
/// file at `path`.
///
/// Reads only the footer (no column page scan). `min_key`/`max_key` are the
/// aggregated Int64 or UTF-8 statistics for `key_column` across row groups, or
/// `None` when the column is absent / key-less / carries no supported statistics. Used by
/// the executor to recover a merged segment's row_count cheaply and by boot
/// adoption. Returns `None` only on an unreadable footer.
pub fn segment_bounds(
    path: &Path,
    key_column: Option<&str>,
) -> Option<(i64, Option<String>, Option<String>)> {
    let file = std::fs::File::open(path).ok()?;
    let reader = SerializedFileReader::new(file).ok()?;
    let num_rows = reader.metadata().file_metadata().num_rows();
    let (lo, hi) = key_column
        .and_then(|kc| metadata_key_bounds(reader.metadata(), kc))
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
    dev: u64,
    ino: u64,
    len: u64,
    modified: SystemTime,
    segment_identity: Uuid,
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
    cached_segment_identity_and_row_group_rows(path).map(|(_, rows)| rows)
}

/// Immutable segment identity plus row-group layout from one cached footer read.
pub fn segment_id_and_row_group_rows(path: &Path) -> Option<(Uuid, Arc<[usize]>)> {
    cached_segment_identity_and_row_group_rows(path)
}

fn cached_segment_identity_and_row_group_rows(path: &Path) -> Option<(Uuid, Arc<[usize]>)> {
    let file = std::fs::File::open(path).ok()?;
    let meta = file.metadata().ok()?;
    let (dev, ino, len, modified) = (meta.dev(), meta.ino(), meta.len(), meta.modified().ok()?);

    let cache = ROW_GROUP_LAYOUTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(entry) = cache.lock().unwrap().get(path) {
        if entry.dev == dev && entry.ino == ino && entry.len == len && entry.modified == modified {
            return Some((entry.segment_identity, Arc::clone(&entry.rows)));
        }
    }

    let reader = SerializedFileReader::new(file).ok()?;
    let segment_identity =
        segment_id_of(reader.metadata()).unwrap_or_else(|| legacy_segment_identity(&meta));
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
            dev,
            ino,
            len,
            modified,
            segment_identity,
            rows: Arc::clone(&rows),
        },
    );
    Some((segment_identity, rows))
}

pub(crate) fn discover_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!(
                "file discovery could not read directory {}: {error}",
                directory.display()
            ),
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => panic!(
                    "file discovery could not read an entry in {}: {error}",
                    directory.display()
                ),
            };
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => panic!(
                    "file discovery could not read the type of {}: {error}",
                    entry.path().display()
                ),
            };
            let path = entry.path();
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if file_type.is_file() {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// All `seg_L*_*.parquet` files under `dir`, sorted by path.
///
/// L0 lives directly in the namespace directory. Physical policies may place
/// L1+ segments in bounded subdirectories, so discovery is recursive. Symlinked
/// directories are not followed. Returns an empty list if `dir` does not exist.
pub fn discover_segments(dir: &Path) -> Vec<PathBuf> {
    discover_files(dir)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| parse_seg_filename(name).is_some())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

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
        let dir = crate::test_support::unique_dir("segment_test");
        let path = dir.join("seg_L1_0000000000000000001.parquet");

        let one_group = batch_with_keys(1, (0..10).collect());
        std::fs::write(&path, write_segment(&one_group).unwrap()).unwrap();
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
        std::fs::write(&path, write_segment(&more).unwrap()).unwrap();
        assert_eq!(segment_row_group_rows(&path).as_deref(), Some(&[25][..]));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_and_read_footer_round_trips_seq_window_and_key_bounds() {
        let dir = crate::test_support::unique_dir("segment_test");
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
        assert_eq!(meta.min_key_value.as_deref(), Some("10"));
        assert_eq!(meta.max_key_value.as_deref(), Some("30"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn string_key_bounds_round_trip_through_footer() {
        let dir = crate::test_support::unique_dir("segment_string_key_test");
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["/task/z", "/task/a", "/task/m"])),
            ],
        )
        .unwrap();
        let (path, _) = write_segment_to_dir(&dir, 0, 1, &batch).unwrap();

        let metadata = read_segment_footer(&path, Some("key")).unwrap();
        assert_eq!(metadata.min_key_value.as_deref(), Some("/task/a"));
        assert_eq!(metadata.max_key_value.as_deref(), Some("/task/z"));

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

    /// Write `batch` the way an older policy did: a small fixed row-group count
    /// and no layout stamp in the footer.
    fn write_legacy_layout(path: &Path, batch: &RecordBatch, row_group_rows: usize) {
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group_rows))
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(1).unwrap()))
            .build();
        let file = std::fs::File::create(path).unwrap();
        let opts = ArrowWriterOptions::new().with_properties(props);
        let mut w = ArrowWriter::try_new_with_options(file, batch.schema(), opts).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
    }

    fn read_all(path: &Path) -> Vec<RecordBatch> {
        let file = std::fs::File::open(path).unwrap();
        ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .map(|b| b.unwrap())
            .collect()
    }

    #[test]
    fn physical_stats_report_the_footer_a_query_would_read() {
        let dir = crate::test_support::unique_dir("segment_test");
        let batch = batch_with_keys(1, (0..500).collect());
        let (path, size) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();

        let physical = segment_physical(&path).unwrap();
        assert_eq!(physical.layout_version, Some(LAYOUT_VERSION));
        assert_eq!(physical.rows, 500);
        assert_eq!(physical.row_groups, 1);
        assert!(physical.created_by.is_some());
        // The footer is a real slice of the file, not an in-memory estimate, so
        // it has to be smaller than the file and big enough to hold a row group's
        // column metadata.
        assert!(physical.footer_bytes > 0 && physical.footer_bytes < size);
        // Compression is the point of the encoded-byte row-group target; a
        // segment whose columns did not compress would not be worth re-encoding.
        assert!(physical.uncompressed_bytes > size);

        let legacy = dir.join("seg_L1_0000000000000000002.parquet");
        write_legacy_layout(&legacy, &batch, 8);
        let stale = segment_physical(&legacy).unwrap();
        assert_eq!(stale.layout_version, None);
        assert!(stale.row_groups > physical.row_groups);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn layout_stamp_distinguishes_current_from_legacy_segments() {
        let dir = crate::test_support::unique_dir("segment_test");
        let batch = batch_with_keys(1, (0..100).collect());

        let legacy = dir.join("seg_L1_0000000000000000001.parquet");
        write_legacy_layout(&legacy, &batch, 8);
        assert!(!segment_layout_is_current(&legacy));

        let (current, _) = write_segment_to_dir(&dir, 1, 200, &batch).unwrap();
        assert!(segment_layout_is_current(&current));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn segment_identity_is_unique_and_survives_layout_rewrite() {
        let dir = crate::test_support::unique_dir("segment_test");
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        let (first, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let (second, _) = write_segment_to_dir(&dir, 1, 10, &batch).unwrap();
        let first_id = segment_id(&first).unwrap();
        assert_ne!(first_id, segment_id(&second).unwrap());

        let (staging, _) = stage_rewritten_segment(&first, MAX_ROW_GROUP_ROWS).unwrap();
        std::fs::rename(staging, &first).unwrap();

        assert_eq!(segment_id(&first), Some(first_id));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_segment_identity_survives_rename_and_changes_on_replacement() {
        let dir = crate::test_support::unique_dir("segment_test");
        let first_path = dir.join("seg_L1_0000000000000000001.parquet");
        let renamed_path = dir.join("seg_L2_0000000000000000001.parquet");
        let replacement = dir.join("replacement.parquet");
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        write_legacy_layout(&first_path, &batch, 8);

        assert_eq!(segment_id(&first_path), None);
        let first_identity = segment_identity(&first_path).unwrap();
        assert_eq!(
            segment_id_and_row_group_rows(&first_path).unwrap().0,
            first_identity
        );

        std::fs::rename(&first_path, &renamed_path).unwrap();
        assert_eq!(segment_identity(&renamed_path), Some(first_identity));

        write_legacy_layout(&replacement, &batch, 8);
        std::fs::rename(&replacement, &renamed_path).unwrap();
        let replacement_identity = segment_identity(&renamed_path).unwrap();
        assert_ne!(replacement_identity, first_identity);
        assert_eq!(
            segment_id_and_row_group_rows(&renamed_path).unwrap().0,
            replacement_identity,
            "the row-group cache must reject a reused path with a new inode"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rewrite_reencodes_in_place_keeping_rows_and_filename() {
        let dir = crate::test_support::unique_dir("segment_test");
        // Non-monotonic keys, so a reordering rewrite would be visible.
        let batch = batch_with_keys(1, (0..400).map(|i| (i * 7919) % 400).collect());
        let path = dir.join("seg_L1_0000000000000000001.parquet");
        write_legacy_layout(&path, &batch, 8);

        let before_groups = segment_row_group_rows(&path).unwrap().len();
        let before_rows = read_all(&path);
        assert!(before_groups > 1, "fixture must have several row groups");

        let max_row_group_rows = 32;
        let (staging, size) = stage_rewritten_segment(&path, max_row_group_rows).unwrap();
        std::fs::rename(&staging, &path).unwrap();

        // Same file, same rows in the same order, now on the current layout.
        assert!(path.exists());
        assert_eq!(size, std::fs::metadata(&path).unwrap().len() as i64);
        assert!(segment_layout_is_current(&path));
        assert_eq!(read_all(&path), before_rows);
        assert!(
            segment_row_group_rows(&path).unwrap().len() < before_groups,
            "the byte target should coalesce the legacy row groups"
        );
        assert!(segment_row_group_rows(&path)
            .unwrap()
            .iter()
            .all(|&rows| rows <= max_row_group_rows));
        // No stray staging file survives.
        assert!(!dir.join("seg_L1_0000000000000000001.parquet.tmp").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn segments_carry_no_bloom_filters() {
        let dir = crate::test_support::unique_dir("segment_test");
        let batch = batch_with_keys(1, vec![30, 10, 20]);
        let (path, _) = write_segment_to_dir(&dir, 0, 1, &batch).unwrap();
        assert!(bloom_columns(&path).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn l0_write_is_unsorted_preserving_row_order() {
        let dir = crate::test_support::unique_dir("segment_test");
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
    fn seg_filename_round_trips() {
        let name = seg_filename(0, 42);
        assert_eq!(name, "seg_L0_0000000000000000042.parquet");
        assert_eq!(parse_seg_filename(&name), Some((0, 42)));
    }

    #[test]
    fn discover_segments_includes_nested_physical_directories() {
        let dir = crate::test_support::unique_dir("segment_test");
        let nested = dir.join("run_id/07");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(dir.join(seg_filename(0, 1)), b"l0").unwrap();
        std::fs::write(nested.join(seg_filename(1, 2)), b"l1").unwrap();
        std::fs::write(nested.join("ignored.parquet.tmp"), b"tmp").unwrap();

        let discovered = discover_segments(&dir);
        assert_eq!(discovered.len(), 2);
        assert!(discovered.iter().any(|path| path.parent() == Some(&dir)));
        assert!(discovered
            .iter()
            .any(|path| path.parent() == Some(nested.as_path())));

        std::fs::remove_dir_all(dir).ok();
    }
}
