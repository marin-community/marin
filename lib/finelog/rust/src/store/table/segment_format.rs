//! How one table's segments are physically written.
//!
//! Everything that encodes Parquet for a table — the flush pipeline, both
//! compaction drivers, the migration rewrite, and the index backfill — needs the
//! same five facts: the store-form Arrow schema, the resolved key column, the
//! sort order, the row-group ceiling, and the derived-index policy. They take
//! this value rather than the table runtime.

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::SchemaRef;

use crate::errors::StatsError;
use crate::indices::exact::ExactIndexConfig;
use crate::indices::SegmentIndexConfig;
use crate::policies::segment_indexes_enabled_for;
use crate::proto::finelog::stats::ColumnType;
use crate::store::compaction::executor::CompactionLayout;
use crate::store::schema::{resolve_key_column, resolve_sort_columns, schema_to_arrow, Schema};
use crate::store::segment::MAX_ROW_GROUP_ROWS;

/// The physical write format of one table's segments.
pub struct SegmentFormat {
    schema: Schema,
    arrow_schema: SchemaRef,
    key_column: String,
    sort_columns: Vec<String>,
    max_row_group_rows: usize,
}

impl SegmentFormat {
    /// Resolve the format `schema` declares, including its implicit key column
    /// and row-group default.
    pub fn resolve(schema: Schema) -> Result<Self, StatsError> {
        let arrow_schema = schema_to_arrow(&schema);
        let sort_columns = resolve_sort_columns(&schema)?;
        let key_column = resolve_key_column(&schema)?;
        let max_row_group_rows = if schema.max_row_group_rows == 0 {
            MAX_ROW_GROUP_ROWS
        } else {
            schema.max_row_group_rows as usize
        };
        Ok(Self {
            schema,
            arrow_schema,
            key_column,
            sort_columns,
            max_row_group_rows,
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The schema segments are written with (store form: includes the implicit
    /// `seq` column).
    pub fn arrow_schema(&self) -> &SchemaRef {
        &self.arrow_schema
    }

    /// Resolved physical key, including the implicit `timestamp_ms` default.
    pub fn key_column(&self) -> &str {
        &self.key_column
    }

    pub fn sort_columns(&self) -> &[String] {
        &self.sort_columns
    }

    pub fn max_row_group_rows(&self) -> usize {
        self.max_row_group_rows
    }

    /// The layout the compaction executor writes its outputs with.
    pub fn compaction_layout(&self) -> CompactionLayout<'_> {
        CompactionLayout {
            sort_columns: &self.sort_columns,
            key_column: &self.key_column,
            max_row_group_rows: self.max_row_group_rows,
        }
    }

    /// The derived-index policy for `table`, empty when the deployment disables
    /// segment indexes for it.
    pub fn index_config(&self, table: &str) -> SegmentIndexConfig {
        if !segment_indexes_enabled_for(table) {
            return SegmentIndexConfig::from_policies(Vec::<String>::new(), &[], &[], None);
        }
        SegmentIndexConfig::from_policies(
            self.trigram_columns(),
            &self.exact_indexes(),
            &self.schema.projections,
            Some(self.key_column.clone()),
        )
        .with_adaptive_value_counts(self.string_columns())
        .with_adaptive_group_extrema(self.schema.grouped_extrema.clone())
    }

    /// Int64 key-column bounds from an in-memory batch (cheaper than re-reading
    /// the Parquet footer just written).
    pub fn key_bounds(&self, batch: &RecordBatch) -> (Option<i64>, Option<i64>) {
        let Ok(index) = batch.schema().index_of(&self.key_column) else {
            return (None, None);
        };
        let Some(column) = batch.column(index).as_any().downcast_ref::<Int64Array>() else {
            return (None, None);
        };
        if column.null_count() == column.len() {
            return (None, None);
        }
        let mut low: Option<i64> = None;
        let mut high: Option<i64> = None;
        for row in 0..column.len() {
            if column.is_null(row) {
                continue;
            }
            let value = column.value(row);
            low = Some(low.map_or(value, |current: i64| current.min(value)));
            high = Some(high.map_or(value, |current: i64| current.max(value)));
        }
        (low, high)
    }

    /// Names of the STRING columns carrying a trigram substring index; one bloom
    /// set is built per returned column.
    fn trigram_columns(&self) -> Vec<&str> {
        self.schema
            .columns
            .iter()
            .filter(|column| {
                column.index.trigram && column.r#type == ColumnType::COLUMN_TYPE_STRING
            })
            .map(|column| column.name.as_str())
            .collect()
    }

    /// Explicit exact row-index and value-count policies for string columns.
    fn exact_indexes(&self) -> Vec<ExactIndexConfig> {
        self.schema
            .columns
            .iter()
            .filter(|column| {
                column.r#type == ColumnType::COLUMN_TYPE_STRING
                    && (column.index.value_counts || !column.index.exact_values.is_empty())
            })
            .map(|column| ExactIndexConfig {
                column: column.name.clone(),
                exact_values: column.index.exact_values.clone(),
                value_counts: column.index.value_counts,
            })
            .collect()
    }

    fn string_columns(&self) -> impl Iterator<Item = &str> {
        self.schema
            .columns
            .iter()
            .filter(|column| column.r#type == ColumnType::COLUMN_TYPE_STRING)
            .map(|column| column.name.as_str())
    }
}
