//! Exact string-value index payloads and covering-projection writers.
//!
//! The active writer stores two independently checksummed `.fidx` section types
//! per column:
//!
//! - row runs for explicitly configured values, used to turn `=` and `IN`
//!   predicates into parquet row selections when the projection is absent;
//! - exact counts for every distinct value (including null), used by the
//!   `GROUP BY column, COUNT(...)` summary path.
//!
//! Independently named covering projections remain narrow Parquet files beside
//! the source segment and are referenced by `.fidx`. Their small row groups
//! preserve range pruning while avoiding scattered reads from the full segment.
//! Historical `.eqi`/`.eqp` paths remain only so lifecycle cleanup can remove
//! artifacts written by older binaries.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BooleanArray, RecordBatch};
use arrow::compute::filter_record_batch;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use serde::{Deserialize, Serialize};

use crate::store::schema::CoveringProjection;
use crate::store::segment::parquet_writer_properties;
use crate::store::string_column::StringColumn;
use crate::store::trigram::ByteReader;

const MAGIC: &[u8; 4] = b"FLEQ";
const VERSION: u8 = 2;
const MAX_COUNT_VALUES: usize = 4_096;
const MAX_COUNT_KEY_BYTES: usize = 1024 * 1024;
const PROJECTION_ROW_GROUP_BYTES: usize = 1024 * 1024;
const PROJECTION_ROW_GROUP_ROWS: usize = 16_384;
const TEMP_SUFFIX: &str = ".tmp";
pub const NAMED_PROJECTION_MARKER: &str = ".fidx.";

/// One string column's exact indexing policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactIndexConfig {
    pub column: String,
    pub exact_values: Vec<String>,
    pub value_counts: bool,
}

/// A half-open global row interval `[start, start + len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRun {
    pub start: u64,
    pub len: u64,
}

/// Parsed exact index for one column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactColumn {
    /// Present only when the sidecar contains a complete value-count summary.
    pub counts: Option<BTreeMap<Option<String>, u64>>,
    /// Configured exact values, including values with no matching rows.
    pub rows: BTreeMap<String, Vec<RowRun>>,
}

/// Decoded exact-postings or value-count section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactSection {
    pub total_rows: u64,
    pub columns: BTreeMap<String, ExactColumn>,
}

/// A published covering projection referenced by the segment index bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionDescriptor {
    pub name: String,
    pub row_count: u64,
    pub columns: Vec<String>,
    pub predicate_column: String,
    pub predicate_values: Vec<String>,
}

impl ExactSection {
    /// Approximate decoded heap retained by the query cache.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .columns
                .iter()
                .map(|(name, column)| name.capacity() + column.heap_bytes())
                .sum::<usize>()
    }
}

impl ExactColumn {
    fn heap_bytes(&self) -> usize {
        let counts = self.counts.as_ref().map_or(0, |counts| {
            counts
                .keys()
                .map(|value| {
                    std::mem::size_of::<(Option<String>, u64)>()
                        + value.as_ref().map_or(0, String::capacity)
                })
                .sum()
        });
        let rows = self
            .rows
            .iter()
            .map(|(value, runs)| {
                std::mem::size_of::<(String, Vec<RowRun>)>()
                    + value.capacity()
                    + runs.capacity() * std::mem::size_of::<RowRun>()
            })
            .sum::<usize>();
        std::mem::size_of::<Self>() + counts + rows
    }
}

fn build_column(batches: &[RecordBatch], config: &ExactIndexConfig) -> Option<(ExactColumn, u64)> {
    let mut rows: BTreeMap<String, Vec<RowRun>> = config
        .exact_values
        .iter()
        .cloned()
        .map(|value| (value, Vec::new()))
        .collect();
    let mut string_counts: Option<BTreeMap<String, u64>> = config.value_counts.then(BTreeMap::new);
    let mut count_key_bytes = 0_usize;
    let mut null_count = 0_u64;
    let mut offset = 0_u64;
    for batch in batches {
        let index = batch.schema().index_of(&config.column).ok()?;
        let values = StringColumn::new(batch.column(index).as_ref())?;
        for row in 0..batch.num_rows() {
            let value = values.value(row);
            if let Some(counts) = string_counts.as_mut() {
                match value {
                    Some(value) => {
                        if let Some(count) = counts.get_mut(value) {
                            *count += 1;
                        } else if counts.len() + usize::from(null_count > 0) < MAX_COUNT_VALUES
                            && count_key_bytes.saturating_add(value.len()) <= MAX_COUNT_KEY_BYTES
                        {
                            count_key_bytes += value.len();
                            counts.insert(value.to_string(), 1);
                        } else {
                            // High-cardinality columns retain configured row
                            // indexes but omit the all-values summary.
                            string_counts = None;
                        }
                    }
                    None => {
                        if null_count > 0 || counts.len() < MAX_COUNT_VALUES {
                            null_count += 1;
                        } else {
                            string_counts = None;
                        }
                    }
                }
            }
            if let Some(runs) = value.and_then(|value| rows.get_mut(value)) {
                let position = offset + row as u64;
                match runs.last_mut() {
                    Some(last) if last.start + last.len == position => last.len += 1,
                    _ => runs.push(RowRun {
                        start: position,
                        len: 1,
                    }),
                }
            }
        }
        offset += batch.num_rows() as u64;
    }
    let counts = string_counts.map(|counts| {
        let mut result: BTreeMap<Option<String>, u64> = counts
            .into_iter()
            .map(|(value, count)| (Some(value), count))
            .collect();
        if null_count > 0 {
            result.insert(None, null_count);
        }
        result
    });
    Some((ExactColumn { counts, rows }, offset))
}

/// Build exact postings and value summaries without publishing any files.
pub fn build_sidecar(
    batches: &[RecordBatch],
    configs: &[ExactIndexConfig],
) -> Option<ExactSection> {
    let mut columns = BTreeMap::new();
    let mut total_rows = None;
    for config in configs {
        let Some((column, rows)) = build_column(batches, config) else {
            continue;
        };
        if let Some(expected) = total_rows {
            debug_assert_eq!(expected, rows);
        } else {
            total_rows = Some(rows);
        }
        columns.insert(config.column.clone(), column);
    }
    (!columns.is_empty()).then_some(ExactSection {
        total_rows: total_rows.unwrap_or(0),
        columns,
    })
}

/// The exact sidecar path for a parquet segment.
pub fn sidecar_path(parquet_path: &Path) -> PathBuf {
    let mut path = parquet_path.as_os_str().to_os_string();
    path.push(".eqi");
    PathBuf::from(path)
}

/// The filtered Parquet projection stored beside an exact sidecar.
pub fn projection_path(parquet_path: &Path) -> PathBuf {
    let mut path = parquet_path.as_os_str().to_os_string();
    path.push(".eqp");
    PathBuf::from(path)
}

/// Path of a named covering projection referenced from `<segment>.fidx`.
pub fn named_projection_path(parquet_path: &Path, name: &str) -> PathBuf {
    let mut path = parquet_path.as_os_str().to_os_string();
    path.push(format!("{NAMED_PROJECTION_MARKER}{name}.parquet"));
    PathBuf::from(path)
}

pub(crate) fn serialize(sidecar: &ExactSection) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&sidecar.total_rows.to_le_bytes());
    // Version 2 reserved this byte for the removed monolithic `.eqp` writer.
    out.push(0);
    put_u16(&mut out, sidecar.columns.len());
    for (name, column) in &sidecar.columns {
        put_string_u16(&mut out, name);
        out.push(u8::from(column.counts.is_some()));
        if let Some(counts) = &column.counts {
            put_u32(&mut out, counts.len());
            for (value, count) in counts {
                match value {
                    None => out.push(0),
                    Some(value) => {
                        out.push(1);
                        put_string_u32(&mut out, value);
                    }
                }
                out.extend_from_slice(&count.to_le_bytes());
            }
        }
        put_u32(&mut out, column.rows.len());
        for (value, runs) in &column.rows {
            put_string_u32(&mut out, value);
            put_u32(&mut out, runs.len());
            let mut previous_end = 0_u64;
            for run in runs {
                put_varint(&mut out, run.start - previous_end);
                put_varint(&mut out, run.len);
                previous_end = run.start + run.len;
            }
        }
    }
    out
}

/// Write one named, narrow covering projection and return its descriptor.
pub fn write_covering_projection(
    parquet_path: &Path,
    batches: &[RecordBatch],
    sidecar: &ExactSection,
    projection: &CoveringProjection,
) -> std::io::Result<ProjectionDescriptor> {
    let Some(column) = sidecar.columns.get(&projection.predicate_column) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "projection {:?} predicate column {:?} has no exact postings",
                projection.name, projection.predicate_column
            ),
        ));
    };
    let runs = coalesce_runs(
        projection
            .predicate_values
            .iter()
            .flat_map(|value| column.rows.get(value).into_iter().flatten().copied())
            .collect(),
    );
    if !projection
        .predicate_values
        .iter()
        .all(|value| column.rows.contains_key(value))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "projection {:?} is not covered by postings",
                projection.name
            ),
        ));
    }
    let expected_rows = runs.iter().map(|run| run.len).sum::<u64>();
    let first = batches.first().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "projection has no input batches",
        )
    })?;
    let source_schema = first.schema();
    let projection_indices: Vec<usize> = projection
        .columns
        .iter()
        .map(|name| {
            source_schema.index_of(name).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "projection {:?} input is missing column {name:?}",
                        projection.name
                    ),
                )
            })
        })
        .collect::<std::io::Result<_>>()?;
    let output_schema = Arc::new(
        source_schema
            .project(&projection_indices)
            .map_err(io_other)?,
    );
    let final_path = named_projection_path(parquet_path, &projection.name);
    let tmp_path = temporary_path(&final_path);
    let output = File::create(&tmp_path)?;
    let options = ArrowWriterOptions::new().with_properties(
        parquet_writer_properties(PROJECTION_ROW_GROUP_BYTES, PROJECTION_ROW_GROUP_ROWS)
            .map_err(io_other)?,
    );
    let mut writer =
        ArrowWriter::try_new_with_options(output, output_schema, options).map_err(io_other)?;
    let mut written = 0_u64;
    let mut offset = 0_u64;
    for batch in batches {
        let projected = batch.project(&projection_indices).map_err(io_other)?;
        let mask = run_mask(&runs, offset, batch.num_rows());
        let filtered = filter_record_batch(&projected, &mask).map_err(io_other)?;
        if filtered.num_rows() > 0 {
            written += filtered.num_rows() as u64;
            writer.write(&filtered).map_err(io_other)?;
        }
        offset += batch.num_rows() as u64;
    }
    let output = writer.into_inner().map_err(io_other)?;
    output.sync_all()?;
    if written != expected_rows || offset != sidecar.total_rows {
        let mismatch = format!(
            "projection {:?} wrote {written}/{expected_rows} rows from {offset}/{} inputs",
            projection.name, sidecar.total_rows
        );
        if let Err(cleanup) = std::fs::remove_file(&tmp_path) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{mismatch}; could not remove {}: {cleanup}",
                    tmp_path.display()
                ),
            ));
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            mismatch,
        ));
    }
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(ProjectionDescriptor {
        name: projection.name.clone(),
        row_count: written,
        columns: projection.columns.clone(),
        predicate_column: projection.predicate_column.clone(),
        predicate_values: projection.predicate_values.clone(),
    })
}

fn temporary_path(final_path: &Path) -> PathBuf {
    let mut path = final_path.as_os_str().to_os_string();
    path.push(TEMP_SUFFIX);
    PathBuf::from(path)
}

/// Sort and coalesce overlapping or adjacent row runs.
pub(crate) fn coalesce_runs(mut runs: Vec<RowRun>) -> Vec<RowRun> {
    runs.sort_by_key(|run| run.start);
    let mut merged: Vec<RowRun> = Vec::with_capacity(runs.len());
    for run in runs {
        match merged.last_mut() {
            Some(previous) if run.start <= previous.start + previous.len => {
                let end = (run.start + run.len).max(previous.start + previous.len);
                previous.len = end - previous.start;
            }
            _ => merged.push(run),
        }
    }
    merged
}

fn run_mask(runs: &[RowRun], offset: u64, rows: usize) -> BooleanArray {
    let end = offset + rows as u64;
    let mut selected = vec![false; rows];
    let first = runs.partition_point(|run| run.start + run.len <= offset);
    for run in &runs[first..] {
        if run.start >= end {
            break;
        }
        let start = run.start.max(offset);
        let run_end = (run.start + run.len).min(end);
        if start < run_end {
            selected[(start - offset) as usize..(run_end - offset) as usize].fill(true);
        }
    }
    BooleanArray::from(selected)
}

fn io_other(error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

pub(crate) fn parse(bytes: &[u8]) -> Option<ExactSection> {
    let mut input = ByteReader::new(bytes);
    if input.take(4)? != MAGIC || input.u8()? != VERSION {
        return None;
    }
    let total_rows = input.u64()?;
    match input.u8()? {
        0 => None,
        1 => {
            let rows = input.u64()?;
            Some((rows <= total_rows).then_some(rows)?)
        }
        _ => return None,
    };
    let column_count = input.u16()? as usize;
    let mut columns = BTreeMap::new();
    for _ in 0..column_count {
        let name = take_string_u16(&mut input)?;
        let counts = match input.u8()? {
            0 => None,
            1 => {
                let count = input.u32()? as usize;
                if count > MAX_COUNT_VALUES + 1 {
                    return None;
                }
                let mut values = BTreeMap::new();
                for _ in 0..count {
                    let value = match input.u8()? {
                        0 => None,
                        1 => Some(take_string_u32(&mut input)?),
                        _ => return None,
                    };
                    values.insert(value, input.u64()?);
                }
                Some(values)
            }
            _ => return None,
        };
        let value_count = input.u32()? as usize;
        if value_count > input.remaining() / 4 + 1 {
            return None;
        }
        let mut rows = BTreeMap::new();
        for _ in 0..value_count {
            let value = take_string_u32(&mut input)?;
            let run_count = input.u32()? as usize;
            if run_count as u64 > total_rows || run_count > input.remaining() / 2 {
                return None;
            }
            let mut runs = Vec::new();
            runs.try_reserve(run_count).ok()?;
            let mut previous_end = 0_u64;
            for _ in 0..run_count {
                let start = previous_end.checked_add(take_varint(&mut input)?)?;
                let len = take_varint(&mut input)?;
                let end = start.checked_add(len)?;
                if len == 0 || end > total_rows {
                    return None;
                }
                runs.push(RowRun { start, len });
                previous_end = end;
            }
            rows.insert(value, runs);
        }
        columns.insert(name, ExactColumn { counts, rows });
    }
    input.is_empty().then_some(ExactSection {
        total_rows,
        columns,
    })
}

fn put_u16(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u16).to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

fn put_string_u16(out: &mut Vec<u8>, value: &str) {
    put_u16(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn put_string_u32(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_string_u16(input: &mut ByteReader<'_>) -> Option<String> {
    let len = input.u16()? as usize;
    String::from_utf8(input.take(len)?.to_vec()).ok()
}

fn take_string_u32(input: &mut ByteReader<'_>) -> Option<String> {
    let len = input.u32()? as usize;
    String::from_utf8(input.take(len)?.to_vec()).ok()
}

fn take_varint(input: &mut ByteReader<'_>) -> Option<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = input.u8()?;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn batch(values: Vec<Option<&str>>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "service",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from(values))],
        )
        .unwrap()
    }

    fn config() -> ExactIndexConfig {
        ExactIndexConfig {
            column: "service".to_string(),
            exact_values: vec!["api".to_string(), "worker".to_string()],
            value_counts: true,
        }
    }

    #[test]
    fn round_trip_preserves_counts_nulls_and_row_runs() {
        let (column, rows) = build_column(
            &[batch(vec![
                Some("api"),
                Some("api"),
                Some("other"),
                None,
                Some("worker"),
                Some("worker"),
            ])],
            &config(),
        )
        .unwrap();
        let sidecar = ExactSection {
            total_rows: rows,
            columns: BTreeMap::from([("service".to_string(), column)]),
        };
        let decoded = parse(&serialize(&sidecar)).unwrap();
        assert_eq!(decoded, sidecar);
        let service = &decoded.columns["service"];
        assert_eq!(
            service.counts.as_ref().unwrap()[&Some("api".to_string())],
            2
        );
        assert_eq!(service.counts.as_ref().unwrap()[&None], 1);
        assert_eq!(service.rows["worker"], vec![RowRun { start: 4, len: 2 }]);
    }

    #[test]
    fn malformed_or_trailing_bytes_are_rejected() {
        assert!(parse(b"not an index").is_none());
        let sidecar = ExactSection {
            total_rows: 0,
            columns: BTreeMap::new(),
        };
        let mut bytes = serialize(&sidecar);
        bytes.push(0);
        assert!(parse(&bytes).is_none());

        let mut oversized = Vec::new();
        oversized.extend_from_slice(MAGIC);
        oversized.push(VERSION);
        oversized.extend_from_slice(&u64::MAX.to_le_bytes());
        oversized.push(0);
        put_u16(&mut oversized, 1);
        put_string_u16(&mut oversized, "service");
        oversized.push(0);
        put_u32(&mut oversized, 1);
        put_string_u32(&mut oversized, "api");
        put_u32(&mut oversized, u32::MAX as usize);
        assert!(parse(&oversized).is_none());
    }

    #[test]
    fn high_cardinality_omits_counts_but_keeps_selected_rows() {
        let mut values: Vec<Option<String>> = (0..=MAX_COUNT_VALUES)
            .map(|index| Some(format!("value-{index}")))
            .collect();
        values.push(None);
        values.push(Some("api".to_string()));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "service",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from(values))],
        )
        .unwrap();

        let (column, _) = build_column(&[batch], &config()).unwrap();

        assert!(column.counts.is_none());
        assert_eq!(
            column.rows["api"],
            vec![RowRun {
                start: (MAX_COUNT_VALUES + 2) as u64,
                len: 1,
            }]
        );
    }

    #[test]
    fn large_distinct_values_decline_before_the_count_index_grows_unbounded() {
        let value = "x".repeat(MAX_COUNT_KEY_BYTES / 2 + 1);
        let values = vec![Some(value.clone()), Some(format!("y{value}"))];
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "service",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from(values))],
        )
        .unwrap();

        let (column, _) = build_column(&[batch], &config()).unwrap();

        assert!(column.counts.is_none());
    }
}
