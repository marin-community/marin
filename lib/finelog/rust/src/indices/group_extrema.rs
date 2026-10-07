//! Bounded per-segment summaries for grouped `MIN`/`MAX` query families.
//!
//! The first family derives a top-level JSON value, groups it together with a
//! low-cardinality string discriminator, and records the extrema of an Int64
//! column. Compaction already owns decoded Arrow batches, so this adds no source
//! read. A segment simply omits the adaptive section when its key set exceeds
//! the fixed cardinality or byte budget.

use std::collections::BTreeMap;

use arrow::array::{Array, Int64Array, RecordBatch};
use serde::{Deserialize, Serialize};

use crate::store::string_column::StringColumn;

const MAX_GROUPS: usize = 4_096;
const MAX_KEY_BYTES: usize = 1024 * 1024;
const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;

/// One planner-facing adaptive grouped-extrema method.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GroupExtremaConfig {
    pub filter_column: String,
    pub json_column: String,
    pub json_key: String,
    pub extrema_column: String,
}

impl GroupExtremaConfig {
    pub fn new(
        filter_column: impl Into<String>,
        json_column: impl Into<String>,
        json_key: impl Into<String>,
        extrema_column: impl Into<String>,
    ) -> Self {
        Self {
            filter_column: filter_column.into(),
            json_column: json_column.into(),
            json_key: json_key.into(),
            extrema_column: extrema_column.into(),
        }
    }
}

/// One exact group within an immutable segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupExtremaEntry {
    pub filter_value: String,
    pub group_value: String,
    pub min: i64,
    pub max: i64,
}

/// Decoded grouped-extrema section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupExtremaSection {
    pub entries: Vec<GroupExtremaEntry>,
}

impl GroupExtremaSection {
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.entries.capacity() * std::mem::size_of::<GroupExtremaEntry>()
            + self
                .entries
                .iter()
                .map(|entry| entry.filter_value.capacity() + entry.group_value.capacity())
                .sum::<usize>()
    }
}

/// Build one exact section, declining when the distinct key budget is exceeded.
pub fn build(batches: &[RecordBatch], config: &GroupExtremaConfig) -> Option<GroupExtremaSection> {
    let mut groups: BTreeMap<String, BTreeMap<String, (i64, i64)>> = BTreeMap::new();
    let mut group_count = 0_usize;
    let mut key_bytes = 0_usize;
    let mut previous_document: Option<String> = None;
    let mut previous_group: Option<Option<String>> = None;
    for batch in batches {
        let filter_index = batch.schema().index_of(&config.filter_column).ok()?;
        let json_index = batch.schema().index_of(&config.json_column).ok()?;
        let extrema_index = batch.schema().index_of(&config.extrema_column).ok()?;
        let filter = StringColumn::new(batch.column(filter_index).as_ref())?;
        let documents = StringColumn::new(batch.column(json_index).as_ref())?;
        let extrema = batch
            .column(extrema_index)
            .as_any()
            .downcast_ref::<Int64Array>()?;
        for row in 0..batch.num_rows() {
            let (Some(filter_value), Some(document)) = (filter.value(row), documents.value(row))
            else {
                continue;
            };
            if extrema.is_null(row) {
                continue;
            }
            if previous_document.as_deref() != Some(document) {
                previous_document = Some(document.to_string());
                previous_group = Some(crate::json::get_text(document, &config.json_key));
            }
            let Some(group_value) = previous_group
                .as_ref()
                .expect("JSON lookup is cached")
                .as_deref()
            else {
                continue;
            };
            let extrema_value = extrema.value(row);
            if let Some(filter_groups) = groups.get_mut(filter_value) {
                if let Some((min, max)) = filter_groups.get_mut(group_value) {
                    *min = (*min).min(extrema_value);
                    *max = (*max).max(extrema_value);
                    continue;
                }
            }
            let added_bytes = filter_value.len().checked_add(group_value.len())?;
            if group_count >= MAX_GROUPS || key_bytes.checked_add(added_bytes)? > MAX_KEY_BYTES {
                return None;
            }
            key_bytes += added_bytes;
            group_count += 1;
            groups
                .entry(filter_value.to_string())
                .or_default()
                .insert(group_value.to_string(), (extrema_value, extrema_value));
        }
    }
    Some(GroupExtremaSection {
        entries: groups
            .into_iter()
            .flat_map(|(filter_value, groups)| {
                groups
                    .into_iter()
                    .map(move |(group_value, (min, max))| GroupExtremaEntry {
                        filter_value: filter_value.clone(),
                        group_value,
                        min,
                        max,
                    })
            })
            .collect(),
    })
}

pub fn serialize(section: &GroupExtremaSection) -> Option<Vec<u8>> {
    let payload = serde_json::to_vec(&section.entries).ok()?;
    (payload.len() <= MAX_PAYLOAD_BYTES).then_some(payload)
}

pub fn parse(payload: &[u8]) -> Option<GroupExtremaSection> {
    if payload.len() > MAX_PAYLOAD_BYTES {
        return None;
    }
    let entries: Vec<GroupExtremaEntry> = serde_json::from_slice(payload).ok()?;
    if entries.len() > MAX_GROUPS {
        return None;
    }
    let key_bytes = entries.iter().try_fold(0_usize, |total, entry| {
        total
            .checked_add(entry.filter_value.len())?
            .checked_add(entry.group_value.len())
    })?;
    if key_bytes > MAX_KEY_BYTES || entries.iter().any(|entry| entry.min > entry.max) {
        return None;
    }
    Some(GroupExtremaSection { entries })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn config() -> GroupExtremaConfig {
        GroupExtremaConfig {
            filter_column: "service".to_string(),
            json_column: "resource_attributes_json".to_string(),
            json_key: "job_id".to_string(),
            extrema_column: "timestamp_ms".to_string(),
        }
    }

    #[test]
    fn builds_exact_json_groups_and_extrema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("service", DataType::Utf8, false),
            Field::new("resource_attributes_json", DataType::Utf8, true),
            Field::new("timestamp_ms", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    "levanter", "levanter", "vllm", "vllm",
                ])),
                Arc::new(StringArray::from(vec![
                    Some(r#"{"job_id":"a"}"#),
                    Some(r#"{"job_id":"a"}"#),
                    Some(r#"{"job_id":"b"}"#),
                    Some(r#"{"other":"x"}"#),
                ])),
                Arc::new(Int64Array::from(vec![10, 30, 20, 40])),
            ],
        )
        .unwrap();
        let section = build(&[batch], &config()).unwrap();
        assert_eq!(
            section.entries,
            vec![
                GroupExtremaEntry {
                    filter_value: "levanter".to_string(),
                    group_value: "a".to_string(),
                    min: 10,
                    max: 30,
                },
                GroupExtremaEntry {
                    filter_value: "vllm".to_string(),
                    group_value: "b".to_string(),
                    min: 20,
                    max: 20,
                },
            ]
        );
        assert_eq!(parse(&serialize(&section).unwrap()), Some(section));
    }

    #[test]
    fn high_cardinality_declines_the_adaptive_section() {
        let rows = MAX_GROUPS + 1;
        let schema = Arc::new(Schema::new(vec![
            Field::new("service", DataType::Utf8, false),
            Field::new("resource_attributes_json", DataType::Utf8, true),
            Field::new("timestamp_ms", DataType::Int64, false),
        ]));
        let documents = (0..rows)
            .map(|row| format!(r#"{{"job_id":"job-{row}"}}"#))
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "levanter", rows,
                ))),
                Arc::new(StringArray::from(documents)),
                Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            ],
        )
        .unwrap();
        assert!(build(&[batch], &config()).is_none());
    }
}
