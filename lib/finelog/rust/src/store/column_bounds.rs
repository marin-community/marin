// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-segment min/max of the columns a table flags with `catalog_bounds`.
//!
//! Query planning drops a segment whose bounds miss a filter's range before it
//! opens the file. The bounds are optional catalog metadata: a segment without
//! them for a filtered column is always scanned, so absent or partial bounds
//! never change a result.
//!
//! An object segment's bounds are stored beside the `ObjectRef` keys in its
//! `object_segments.source_json`, and in its checkpoint `CatalogSegment`. An
//! older binary ignores both, so they need no catalog migration; a checkpoint
//! it writes drops them, and the backfill restores them after the next upgrade.

use std::collections::BTreeMap;

use arrow::array::{Array, Int64Array, RecordBatch};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics;
use serde::{Deserialize, Serialize};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogColumnBounds, ObjectRef};

/// Column name to the segment's inclusive `(min, max)` over its non-null values.
pub type ColumnBounds = BTreeMap<String, (i64, i64)>;

/// The bounds of a column with no non-null value in the segment: all null, or
/// absent from a file written before the column existed. Planning keeps a
/// segment whose min exceeds its max, so such a segment is scanned as before;
/// recording it still ends the segment's backfill.
pub const NO_VALUES: (i64, i64) = (i64::MAX, i64::MIN);

/// The `source_json` key that carries the bounds beside the `ObjectRef` fields.
const SOURCE_JSON_KEY: &str = "columnBounds";

#[derive(Serialize, Deserialize)]
struct StoredColumnBounds {
    column: String,
    min: i64,
    max: i64,
}

#[derive(Deserialize)]
struct StoredSource {
    #[serde(default, rename = "columnBounds")]
    column_bounds: Vec<StoredColumnBounds>,
}

fn widen(bounds: &mut Option<(i64, i64)>, low: i64, high: i64) {
    *bounds = Some(match *bounds {
        None => (low, high),
        Some((minimum, maximum)) => (minimum.min(low), maximum.max(high)),
    });
}

/// The bounds of `columns` over `batches`, computed while the rows are still in
/// memory (flush and merge outputs). A column the batches do not hold has no
/// values; a column that is not Int64 gets no bounds and is always scanned.
pub fn from_batches(batches: &[RecordBatch], columns: &[String]) -> ColumnBounds {
    let mut bounds = ColumnBounds::new();
    for column in columns {
        let mut range = None;
        let mut typed = true;
        for batch in batches {
            let Some(array) = batch.column_by_name(column) else {
                continue;
            };
            let Some(values) = array.as_any().downcast_ref::<Int64Array>() else {
                typed = false;
                break;
            };
            for index in 0..values.len() {
                if values.is_valid(index) {
                    let value = values.value(index);
                    widen(&mut range, value, value);
                }
            }
        }
        if typed {
            bounds.insert(column.clone(), range.unwrap_or(NO_VALUES));
        }
    }
    bounds
}

/// The bounds of `columns` from a Parquet footer's row-group statistics, for
/// segments written before their table flagged a column. A column whose
/// statistics are missing in any non-empty row group gets no bounds.
pub fn from_parquet(metadata: &ParquetMetaData, columns: &[String]) -> ColumnBounds {
    let schema = metadata.file_metadata().schema_descr();
    let mut bounds = ColumnBounds::new();
    for column in columns {
        let Some(index) = (0..schema.num_columns())
            .find(|&i| schema.column(i).path().parts() == [column.as_str()])
        else {
            bounds.insert(column.clone(), NO_VALUES);
            continue;
        };
        if let Some(range) = footer_int64_bounds(metadata, index) {
            bounds.insert(column.clone(), range);
        }
    }
    bounds
}

fn footer_int64_bounds(metadata: &ParquetMetaData, column: usize) -> Option<(i64, i64)> {
    let mut range = None;
    for row_group in metadata.row_groups() {
        if row_group.num_rows() == 0 {
            continue;
        }
        let Some(Statistics::Int64(statistics)) = row_group.column(column).statistics() else {
            return None;
        };
        if statistics.null_count_opt() == Some(row_group.num_rows() as u64) {
            continue;
        }
        let (Some(&minimum), Some(&maximum)) = (statistics.min_opt(), statistics.max_opt()) else {
            return None;
        };
        widen(&mut range, minimum, maximum);
    }
    Some(range.unwrap_or(NO_VALUES))
}

/// Whether `bounds` lacks any of the flagged `columns`, so the backfill still
/// owes the segment a footer read.
pub fn missing(bounds: &ColumnBounds, columns: &[String]) -> bool {
    columns.iter().any(|column| !bounds.contains_key(column))
}

/// Encode `source` for `object_segments.source_json`, with `bounds` under an
/// extra key that `ObjectRef` decoding ignores. Without bounds this is the plain
/// `ObjectRef` encoding, byte for byte.
pub fn source_json(source: &ObjectRef, bounds: &ColumnBounds) -> Result<String, StatsError> {
    let encode_error = |error: serde_json::Error| {
        StatsError::Internal(format!("serialize object segment source: {error}"))
    };
    if bounds.is_empty() {
        return serde_json::to_string(source).map_err(encode_error);
    }
    let mut value = serde_json::to_value(source).map_err(encode_error)?;
    let object = value.as_object_mut().ok_or_else(|| {
        StatsError::Internal("object segment source is not a JSON object".to_string())
    })?;
    let stored: Vec<_> = bounds
        .iter()
        .map(|(column, &(min, max))| StoredColumnBounds {
            column: column.clone(),
            min,
            max,
        })
        .collect();
    object.insert(
        SOURCE_JSON_KEY.to_string(),
        serde_json::to_value(stored).map_err(encode_error)?,
    );
    serde_json::to_string(&value).map_err(encode_error)
}

/// The bounds stored in a `source_json`; empty for one written without them.
pub fn from_source_json(json: &str) -> Result<ColumnBounds, StatsError> {
    let stored: StoredSource = serde_json::from_str(json).map_err(|error| {
        StatsError::Internal(format!("decode object segment column bounds: {error}"))
    })?;
    Ok(stored
        .column_bounds
        .into_iter()
        .map(|entry| (entry.column, (entry.min, entry.max)))
        .collect())
}

/// The checkpoint form of `bounds`.
pub fn to_proto(bounds: &ColumnBounds) -> Vec<CatalogColumnBounds> {
    bounds
        .iter()
        .map(|(column, &(min, max))| CatalogColumnBounds {
            column: Some(column.clone()),
            min: Some(min),
            max: Some(max),
            ..Default::default()
        })
        .collect()
}

/// The bounds a checkpoint records; an entry missing a field is dropped, so its
/// column is scanned and backfilled again.
pub fn from_proto(entries: &[CatalogColumnBounds]) -> ColumnBounds {
    entries
        .iter()
        .filter_map(|entry| Some((entry.column.clone()?, (entry.min?, entry.max?))))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn batch(values: Vec<Option<i64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("epoch_ms", DataType::Int64, true),
            Field::new("key", DataType::Utf8, false),
        ]));
        let keys = StringArray::from(vec!["k"; values.len()]);
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(values)), Arc::new(keys)],
        )
        .unwrap()
    }

    fn columns(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn batches_give_the_range_of_non_null_values() {
        let bounds = from_batches(
            &[batch(vec![Some(5), None, Some(9)]), batch(vec![Some(2)])],
            &columns(&["epoch_ms"]),
        );
        assert_eq!(bounds.get("epoch_ms"), Some(&(2, 9)));
    }

    #[test]
    fn an_all_null_or_absent_column_has_no_values_and_a_string_column_no_bounds() {
        let bounds = from_batches(
            &[batch(vec![None, None])],
            &columns(&["epoch_ms", "added_later", "key"]),
        );
        assert_eq!(bounds.get("epoch_ms"), Some(&NO_VALUES));
        assert_eq!(bounds.get("added_later"), Some(&NO_VALUES));
        assert_eq!(bounds.get("key"), None);
    }

    #[test]
    fn source_json_keeps_the_object_ref_decodable_with_and_without_bounds() {
        let source = ObjectRef {
            object_id: Some("objects/a.parquet".to_string()),
            byte_size: Some(42),
            ..Default::default()
        };
        let bounds = ColumnBounds::from([("epoch_ms".to_string(), (10, 20))]);
        let json = source_json(&source, &bounds).unwrap();
        let decoded: ObjectRef = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded, source,
            "an older binary decodes the same ObjectRef"
        );
        assert_eq!(from_source_json(&json).unwrap(), bounds);

        let without = serde_json::to_string(&source).unwrap();
        assert_eq!(source_json(&source, &ColumnBounds::new()).unwrap(), without);
        assert!(from_source_json(&without).unwrap().is_empty());
    }

    #[test]
    fn checkpoint_form_round_trips_and_drops_incomplete_entries() {
        let bounds = ColumnBounds::from([
            ("epoch_ms".to_string(), (10, 20)),
            ("empty".to_string(), NO_VALUES),
        ]);
        assert_eq!(from_proto(&to_proto(&bounds)), bounds);
        let incomplete = CatalogColumnBounds {
            column: Some("epoch_ms".to_string()),
            min: Some(1),
            ..Default::default()
        };
        assert!(from_proto(&[incomplete]).is_empty());
    }

    #[test]
    fn missing_names_a_flagged_column_without_bounds() {
        let bounds = ColumnBounds::from([("epoch_ms".to_string(), (1, 2))]);
        assert!(!missing(&bounds, &columns(&["epoch_ms"])));
        assert!(missing(&bounds, &columns(&["epoch_ms", "ts"])));
        assert!(!missing(&ColumnBounds::new(), &[]));
    }
}
