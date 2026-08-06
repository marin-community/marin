//! Exact string-value indexes stored beside a parquet segment.
//!
//! The `.eqi` metadata sidecar has two independent, optional payloads per
//! column:
//!
//! - row runs for explicitly configured values, used to turn `=` and `IN`
//!   predicates into parquet row selections when the projection is absent;
//! - exact counts for every distinct value (including null), used by the
//!   `GROUP BY column, COUNT(...)` summary path.
//!
//! When any values are configured, a companion `.eqp` Parquet file contains
//! their complete rows. Its small row groups preserve range pruning while
//! avoiding scattered reads from the full segment. Both files are derived
//! state: writers publish by atomic rename, queries fall back on missing or
//! malformed artifacts, and fast paths activate only when every visible segment
//! is covered.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{
    Array, BooleanArray, LargeStringArray, RecordBatch, StringArray, StringViewArray,
};
use arrow::compute::filter_record_batch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};

use crate::store::segment::parquet_writer_properties;

const MAGIC: &[u8; 4] = b"FLEQ";
const VERSION: u8 = 2;
const MAX_COUNT_VALUES: usize = 4_096;
const PROJECTION_ROW_GROUP_BYTES: usize = 1024 * 1024;
const PROJECTION_ROW_GROUP_ROWS: usize = 16_384;

/// One string column's exact indexing policy.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Parsed `.eqi` contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactSidecar {
    pub total_rows: u64,
    /// Rows in the complete filtered Parquet projection, when present.
    pub projection_rows: Option<u64>,
    pub columns: BTreeMap<String, ExactColumn>,
}

impl ExactSidecar {
    /// Whether this artifact contains every payload requested by `configs`.
    pub fn covers(&self, configs: &[ExactIndexConfig]) -> bool {
        (!configs.iter().any(|config| !config.exact_values.is_empty())
            || self.projection_rows.is_some())
            && configs.iter().all(|config| {
                let Some(column) = self.columns.get(&config.column) else {
                    return false;
                };
                (!config.value_counts || column.counts.is_some())
                    && config
                        .exact_values
                        .iter()
                        .all(|value| column.rows.contains_key(value))
            })
    }
}

enum Strings<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
    View(&'a StringViewArray),
}

impl<'a> Strings<'a> {
    fn new(array: &'a dyn Array) -> Option<Self> {
        if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
            return Some(Self::Utf8(values));
        }
        if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
            return Some(Self::Large(values));
        }
        array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .map(Self::View)
    }

    fn value(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(values) => (!values.is_null(row)).then(|| values.value(row)),
            Self::Large(values) => (!values.is_null(row)).then(|| values.value(row)),
            Self::View(values) => (!values.is_null(row)).then(|| values.value(row)),
        }
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
    let mut null_count = 0_u64;
    let mut offset = 0_u64;
    for batch in batches {
        let index = batch.schema().index_of(&config.column).ok()?;
        let values = Strings::new(batch.column(index).as_ref())?;
        for row in 0..batch.num_rows() {
            let value = values.value(row);
            if let Some(counts) = string_counts.as_mut() {
                match value {
                    Some(value) => {
                        if let Some(count) = counts.get_mut(value) {
                            *count += 1;
                        } else if counts.len() + usize::from(null_count > 0) < MAX_COUNT_VALUES {
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

/// Build and atomically publish the exact sidecar for `parquet_path`.
///
/// Returns `Ok(false)` when no configured column exists in the batches.
pub fn write_sidecar(
    parquet_path: &Path,
    batches: &[RecordBatch],
    configs: &[ExactIndexConfig],
) -> std::io::Result<bool> {
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
    if columns.is_empty() {
        return Ok(false);
    }
    let mut sidecar = ExactSidecar {
        total_rows: total_rows.unwrap_or(0),
        projection_rows: None,
        columns,
    };
    if sidecar
        .columns
        .values()
        .any(|column| !column.rows.is_empty())
    {
        sidecar.projection_rows = Some(write_projection(parquet_path, batches, &sidecar)?);
    }
    let final_path = sidecar_path(parquet_path);
    let mut tmp = final_path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    std::fs::write(&tmp_path, serialize(&sidecar))?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(true)
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

/// Read a complete exact sidecar. Malformed and unknown versions are absent.
pub fn read_sidecar(path: &Path) -> Option<ExactSidecar> {
    parse(&std::fs::read(path).ok()?)
}

fn serialize(sidecar: &ExactSidecar) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&sidecar.total_rows.to_le_bytes());
    match sidecar.projection_rows {
        Some(rows) => {
            out.push(1);
            out.extend_from_slice(&rows.to_le_bytes());
        }
        None => out.push(0),
    }
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

fn write_projection(
    parquet_path: &Path,
    batches: &[RecordBatch],
    sidecar: &ExactSidecar,
) -> std::io::Result<u64> {
    let runs = projection_runs(sidecar);
    let expected_rows = runs.iter().map(|run| run.len).sum::<u64>();
    let file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(io_other)?;
    let parquet_schema = builder.schema().clone();
    let input_is_complete = batches
        .first()
        .is_some_and(|batch| batch.schema().fields() == parquet_schema.fields());

    let final_path = projection_path(parquet_path);
    let mut tmp = final_path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp_path = PathBuf::from(tmp);
    let output = File::create(&tmp_path)?;
    let options = ArrowWriterOptions::new().with_properties(
        parquet_writer_properties(PROJECTION_ROW_GROUP_BYTES, PROJECTION_ROW_GROUP_ROWS)
            .map_err(io_other)?,
    );
    let mut writer = ArrowWriter::try_new_with_options(output, parquet_schema.clone(), options)
        .map_err(io_other)?;
    let mut written = 0_u64;

    if input_is_complete {
        let mut offset = 0_u64;
        for batch in batches {
            let mask = run_mask(&runs, offset, batch.num_rows());
            let filtered = filter_record_batch(batch, &mask).map_err(io_other)?;
            if filtered.num_rows() > 0 {
                writer.write(&filtered).map_err(io_other)?;
                written += filtered.num_rows() as u64;
            }
            offset += batch.num_rows() as u64;
        }
    } else {
        let selection = runs_row_selection(&runs, sidecar.total_rows);
        let reader = builder
            .with_row_selection(selection)
            .build()
            .map_err(io_other)?;
        for batch in reader {
            let batch = batch.map_err(io_other)?;
            written += batch.num_rows() as u64;
            writer.write(&batch).map_err(io_other)?;
        }
    }
    writer.close().map_err(io_other)?;
    if written != expected_rows {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("exact projection wrote {written} rows; expected {expected_rows}"),
        ));
    }
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(written)
}

fn projection_runs(sidecar: &ExactSidecar) -> Vec<RowRun> {
    let mut runs: Vec<RowRun> = sidecar
        .columns
        .values()
        .flat_map(|column| column.rows.values())
        .flatten()
        .copied()
        .collect();
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

fn runs_row_selection(runs: &[RowRun], total_rows: u64) -> RowSelection {
    let mut selectors = Vec::with_capacity(runs.len() * 2 + 1);
    let mut cursor = 0_u64;
    for run in runs {
        if cursor < run.start {
            selectors.push(RowSelector::skip((run.start - cursor) as usize));
        }
        selectors.push(RowSelector::select(run.len as usize));
        cursor = run.start + run.len;
    }
    if cursor < total_rows {
        selectors.push(RowSelector::skip((total_rows - cursor) as usize));
    }
    RowSelection::from(selectors)
}

fn io_other(error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

fn parse(bytes: &[u8]) -> Option<ExactSidecar> {
    let mut input = bytes;
    if take(&mut input, 4)? != MAGIC || take_u8(&mut input)? != VERSION {
        return None;
    }
    let total_rows = take_u64(&mut input)?;
    let projection_rows = match take_u8(&mut input)? {
        0 => None,
        1 => {
            let rows = take_u64(&mut input)?;
            Some((rows <= total_rows).then_some(rows)?)
        }
        _ => return None,
    };
    let column_count = take_u16(&mut input)? as usize;
    let mut columns = BTreeMap::new();
    for _ in 0..column_count {
        let name = take_string_u16(&mut input)?;
        let counts = match take_u8(&mut input)? {
            0 => None,
            1 => {
                let count = take_u32(&mut input)? as usize;
                if count > MAX_COUNT_VALUES + 1 {
                    return None;
                }
                let mut values = BTreeMap::new();
                for _ in 0..count {
                    let value = match take_u8(&mut input)? {
                        0 => None,
                        1 => Some(take_string_u32(&mut input)?),
                        _ => return None,
                    };
                    values.insert(value, take_u64(&mut input)?);
                }
                Some(values)
            }
            _ => return None,
        };
        let value_count = take_u32(&mut input)? as usize;
        if value_count > input.len() / 4 + 1 {
            return None;
        }
        let mut rows = BTreeMap::new();
        for _ in 0..value_count {
            let value = take_string_u32(&mut input)?;
            let run_count = take_u32(&mut input)? as usize;
            if run_count as u64 > total_rows {
                return None;
            }
            let mut runs = Vec::with_capacity(run_count);
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
    input.is_empty().then_some(ExactSidecar {
        total_rows,
        projection_rows,
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

fn take<'a>(input: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
    let (head, tail) = input.split_at_checked(len)?;
    *input = tail;
    Some(head)
}

fn take_u8(input: &mut &[u8]) -> Option<u8> {
    Some(take(input, 1)?[0])
}

fn take_u16(input: &mut &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(take(input, 2)?.try_into().ok()?))
}

fn take_u32(input: &mut &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(take(input, 4)?.try_into().ok()?))
}

fn take_u64(input: &mut &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(take(input, 8)?.try_into().ok()?))
}

fn take_string_u16(input: &mut &[u8]) -> Option<String> {
    let len = take_u16(input)? as usize;
    String::from_utf8(take(input, len)?.to_vec()).ok()
}

fn take_string_u32(input: &mut &[u8]) -> Option<String> {
    let len = take_u32(input)? as usize;
    String::from_utf8(take(input, len)?.to_vec()).ok()
}

fn take_varint(input: &mut &[u8]) -> Option<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let byte = take_u8(input)?;
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

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::store::compaction::executor::read_segment_projected;
    use crate::store::segment::write_segment_to_dir;

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

    fn full_batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("seq", DataType::Int64, false),
                Field::new("service", DataType::Utf8, true),
                Field::new("payload", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    Some("api"),
                    Some("other"),
                    Some("worker"),
                    None,
                ])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            ],
        )
        .unwrap()
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
        let sidecar = ExactSidecar {
            total_rows: rows,
            projection_rows: Some(4),
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
        let sidecar = ExactSidecar {
            total_rows: 0,
            projection_rows: None,
            columns: BTreeMap::new(),
        };
        let mut bytes = serialize(&sidecar);
        bytes.push(0);
        assert!(parse(&bytes).is_none());
    }

    #[test]
    fn coverage_requires_every_configured_payload() {
        let sidecar = ExactSidecar {
            total_rows: 1,
            projection_rows: Some(1),
            columns: BTreeMap::from([(
                "service".to_string(),
                ExactColumn {
                    counts: Some(BTreeMap::from([(Some("api".to_string()), 1)])),
                    rows: BTreeMap::from([
                        ("api".to_string(), vec![RowRun { start: 0, len: 1 }]),
                        ("worker".to_string(), vec![]),
                    ]),
                },
            )]),
        };
        assert!(sidecar.covers(&[config()]));
        let mut missing = config();
        missing.exact_values.push("scheduler".to_string());
        assert!(!sidecar.covers(&[missing]));
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
    fn projection_contains_complete_rows_for_selected_values() {
        let dir = tempfile_dir("full_projection");
        let batch = full_batch();
        let (parquet, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();

        assert!(write_sidecar(&parquet, &[batch], &[config()]).unwrap());

        let projected = read_segment_projected(&projection_path(&parquet), None).unwrap();
        assert_eq!(
            projected.iter().map(RecordBatch::num_rows).sum::<usize>(),
            2
        );
        assert_eq!(projected[0].schema().fields().len(), 3);
        let payload = projected[0]
            .column_by_name("payload")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            payload.iter().collect::<Vec<_>>(),
            vec![Some("a"), Some("c")]
        );
        assert_eq!(
            read_sidecar(&sidecar_path(&parquet))
                .unwrap()
                .projection_rows,
            Some(2)
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn backfill_projection_recovers_columns_not_loaded_for_indexing() {
        let dir = tempfile_dir("projected_backfill");
        let batch = full_batch();
        let (parquet, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let indexed_only = read_segment_projected(&parquet, Some(&["service"])).unwrap();

        assert!(write_sidecar(&parquet, &indexed_only, &[config()]).unwrap());

        let projected = read_segment_projected(&projection_path(&parquet), None).unwrap();
        assert!(projected[0].column_by_name("payload").is_some());
        assert_eq!(
            projected.iter().map(RecordBatch::num_rows).sum::<usize>(),
            2
        );
        std::fs::remove_dir_all(dir).ok();
    }

    fn tempfile_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "finelog_exact_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
