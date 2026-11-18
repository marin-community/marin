// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Per-segment min/max of the columns a table flags with `catalog_bounds`.
//!
//! Flush, merge and spec migration record them; maintenance backfills them
//! from Parquet footers for segments written before the column was flagged.
//!
//! An object segment's bounds are stored in its `object_segments.column_bounds`
//! (NULL when none are recorded) and in its checkpoint `CatalogSegment`. A row
//! or checkpoint written without them reads back as no bounds, and the backfill
//! fills them in.

use std::collections::BTreeMap;

use arrow::array::{Array, Int64Array, RecordBatch};
use parquet::file::metadata::ParquetMetaData;

use crate::errors::StatsError;
use crate::proto::finelog::stats::CatalogColumnBounds;
use crate::store::segment::metadata_int64_bounds;

/// One flagged column's recorded range over a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRange {
    /// The inclusive range of the segment's non-null values; `min <= max`.
    Range { min: i64, max: i64 },
    /// No usable range: the column is all null, absent from a file written
    /// before it existed, or without footer statistics. Planning
    /// always scans the segment. Recording it still ends the segment's
    /// backfill, so it never holds a backfill slot again.
    Unknown,
}

impl ColumnRange {
    /// The range `min..=max`, or `Unknown` when `min > max`, so a `Range`
    /// always holds a non-empty range.
    pub fn new(min: i64, max: i64) -> Self {
        if min > max {
            Self::Unknown
        } else {
            Self::Range { min, max }
        }
    }

    /// The stored form: `Some` for both ends of a range, `None` for both when
    /// unknown.
    fn ends(self) -> (Option<i64>, Option<i64>) {
        match self {
            Self::Range { min, max } => (Some(min), Some(max)),
            Self::Unknown => (None, None),
        }
    }

    /// The inverse of `ends`; `None` for an entry with only one end, which is
    /// dropped so its column is backfilled again.
    fn from_ends(min: Option<i64>, max: Option<i64>) -> Option<Self> {
        match (min, max) {
            (Some(min), Some(max)) => Some(Self::new(min, max)),
            (None, None) => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// Column name to the segment's recorded range for that column.
pub type ColumnBounds = BTreeMap<String, ColumnRange>;

/// The inclusive range of the non-null values in `arrays`; `None` when every
/// value is null.
pub fn int64_range<'a>(arrays: impl IntoIterator<Item = &'a Int64Array>) -> Option<(i64, i64)> {
    arrays.into_iter().fold(None, |range, values| {
        let (Some(low), Some(high)) = (arrow::compute::min(values), arrow::compute::max(values))
        else {
            return range;
        };
        Some(range.map_or((low, high), |(minimum, maximum): (i64, i64)| {
            (minimum.min(low), maximum.max(high))
        }))
    })
}

/// The bounds of `columns` over `batches`, computed while the rows are still in
/// memory (flush and merge outputs). A column the batches do not hold is
/// `Unknown`; one that is not Int64 is an error, since schema validation allows
/// `catalog_bounds` only on INT64 columns.
pub fn from_batches(
    batches: &[RecordBatch],
    columns: &[String],
) -> Result<ColumnBounds, StatsError> {
    let mut bounds = ColumnBounds::new();
    for column in columns {
        let mut arrays = Vec::with_capacity(batches.len());
        for batch in batches {
            let Some(array) = batch.column_by_name(column) else {
                continue;
            };
            let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                StatsError::Internal(format!("catalog_bounds column {column:?} is not Int64"))
            })?;
            arrays.push(values);
        }
        let range = int64_range(arrays).map_or(ColumnRange::Unknown, |(min, max)| {
            ColumnRange::new(min, max)
        });
        bounds.insert(column.clone(), range);
    }
    Ok(bounds)
}

/// The bounds of `columns` from a Parquet footer's row-group statistics, for
/// segments written before their table flagged a column. A column whose
/// statistics are missing in any non-empty row group is `Unknown`.
pub fn from_parquet(metadata: &ParquetMetaData, columns: &[String]) -> ColumnBounds {
    let mut bounds = ColumnBounds::new();
    for column in columns {
        let range = metadata_int64_bounds(metadata, column)
            .and_then(|(minimum, maximum)| minimum.zip(maximum))
            .map_or(ColumnRange::Unknown, |(minimum, maximum)| {
                ColumnRange::new(minimum, maximum)
            });
        bounds.insert(column.clone(), range);
    }
    bounds
}

/// The flagged `columns` that `bounds` has no entry for, which the backfill
/// still owes the segment a footer read.
pub fn missing_columns(bounds: &ColumnBounds, columns: &[String]) -> Vec<String> {
    columns
        .iter()
        .filter(|column| !bounds.contains_key(*column))
        .cloned()
        .collect()
}

/// Encode `bounds` for `object_segments.column_bounds` as the JSON of its
/// checkpoint entries; `None` (NULL) when the segment records none.
pub fn to_column(bounds: &ColumnBounds) -> Result<Option<String>, StatsError> {
    if bounds.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(&to_proto(bounds))
        .map(Some)
        .map_err(|error| StatsError::Internal(format!("serialize column bounds: {error}")))
}

/// The bounds stored in `object_segments.column_bounds`; empty for NULL. Entries
/// are dropped as in `from_proto`.
pub fn from_column(stored: Option<&str>) -> Result<ColumnBounds, StatsError> {
    let Some(json) = stored else {
        return Ok(ColumnBounds::new());
    };
    let entries: Vec<CatalogColumnBounds> = serde_json::from_str(json).map_err(|error| {
        StatsError::Internal(format!("decode object segment column bounds: {error}"))
    })?;
    Ok(from_proto(&entries))
}

/// The checkpoint form of `bounds`.
pub fn to_proto(bounds: &ColumnBounds) -> Vec<CatalogColumnBounds> {
    bounds
        .iter()
        .map(|(column, range)| {
            let (min, max) = range.ends();
            CatalogColumnBounds {
                column: Some(column.clone()),
                min,
                max,
                ..Default::default()
            }
        })
        .collect()
}

/// The bounds a checkpoint records. An entry without a column or with only one
/// end is dropped, so its column is scanned and backfilled again.
pub fn from_proto(entries: &[CatalogColumnBounds]) -> ColumnBounds {
    entries
        .iter()
        .filter_map(|entry| {
            let range = ColumnRange::from_ends(entry.min, entry.max)?;
            Some((entry.column.clone()?, range))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};
    use parquet::file::reader::{FileReader, SerializedFileReader};

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
        )
        .unwrap();
        assert_eq!(
            bounds.get("epoch_ms"),
            Some(&ColumnRange::Range { min: 2, max: 9 })
        );
    }

    #[test]
    fn an_all_null_or_absent_column_is_unknown() {
        let bounds = from_batches(
            &[batch(vec![None, None])],
            &columns(&["epoch_ms", "added_later"]),
        )
        .unwrap();
        assert_eq!(bounds.get("epoch_ms"), Some(&ColumnRange::Unknown));
        assert_eq!(bounds.get("added_later"), Some(&ColumnRange::Unknown));
    }

    #[test]
    fn a_footer_without_statistics_is_unknown() {
        let input = batch(vec![Some(5), Some(9)]);
        let properties = WriterProperties::builder()
            .set_statistics_enabled(EnabledStatistics::None)
            .build();
        let mut encoded = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut encoded, input.schema(), Some(properties)).unwrap();
        writer.write(&input).unwrap();
        writer.close().unwrap();
        let reader = SerializedFileReader::new(bytes::Bytes::from(encoded)).unwrap();
        let bounds = from_parquet(reader.metadata(), &columns(&["epoch_ms", "added_later"]));
        assert_eq!(bounds.get("epoch_ms"), Some(&ColumnRange::Unknown));
        assert_eq!(bounds.get("added_later"), Some(&ColumnRange::Unknown));
        assert!(missing_columns(&bounds, &columns(&["epoch_ms", "added_later"])).is_empty());
    }

    #[test]
    fn unknown_round_trips_without_ends_in_the_column_and_checkpoint() {
        let bounds = ColumnBounds::from([
            ("epoch_ms".to_string(), ColumnRange::new(10, 20)),
            ("empty".to_string(), ColumnRange::Unknown),
        ]);
        let json = to_column(&bounds).unwrap().unwrap();
        assert_eq!(
            json,
            r#"[{"column":"empty"},{"column":"epoch_ms","min":"10","max":"20"}]"#
        );
        assert_eq!(from_column(Some(&json)).unwrap(), bounds);

        let checkpoint = to_proto(&bounds);
        let empty = checkpoint
            .iter()
            .find(|entry| entry.column.as_deref() == Some("empty"))
            .unwrap();
        assert_eq!((empty.min, empty.max), (None, None));
        assert_eq!(from_proto(&checkpoint), bounds);
    }

    #[test]
    fn an_inverted_range_is_never_a_range() {
        assert_eq!(ColumnRange::new(i64::MAX, i64::MIN), ColumnRange::Unknown);
        assert_eq!(
            ColumnRange::new(7, 7),
            ColumnRange::Range { min: 7, max: 7 }
        );

        let inverted =
            r#"[{"column":"epoch_ms","min":"9223372036854775807","max":"-9223372036854775808"}]"#;
        let unknown = ColumnBounds::from([("epoch_ms".to_string(), ColumnRange::Unknown)]);
        assert_eq!(from_column(Some(inverted)).unwrap(), unknown);
        let entry = CatalogColumnBounds {
            column: Some("epoch_ms".to_string()),
            min: Some(i64::MAX),
            max: Some(i64::MIN),
            ..Default::default()
        };
        assert_eq!(from_proto(&[entry]), unknown);
    }

    #[test]
    fn an_entry_with_one_end_is_dropped_for_backfill() {
        let entry = CatalogColumnBounds {
            column: Some("epoch_ms".to_string()),
            min: Some(1),
            ..Default::default()
        };
        assert!(from_proto(&[entry]).is_empty());
        let json = r#"[{"column":"epoch_ms","max":"5"}]"#;
        assert!(from_column(Some(json)).unwrap().is_empty());
    }
}
