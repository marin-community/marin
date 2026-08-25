//! Typed Levanter metric events and their run-partitioned storage policy.

use arrow::array::{Array, Float64Array, Int64Array, ListArray, StringArray};
use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;
use crate::ingestion_policy::{
    IngestionBatchSource, IngestionDestination, IngestionPolicy, IngestionState,
    RoutedIngestionBatch,
};
use crate::partition_policy::{PhysicalPartitionPolicy, StringIdentityPartitionPolicy};
use crate::proto::finelog::stats::ColumnType;
use crate::storage_policy::NamespaceStoragePolicy;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{Column, Schema};

pub(crate) const LEVANTER_METRICS_NAMESPACE: &str = "levanter.metrics";
const LEVANTER_METRICS_ALIAS_PREFIX: &str = "levanter.metrics.";
const GIBIBYTE: i64 = 1024 * 1024 * 1024;
const LEVANTER_METRICS_MAX_BYTES: i64 = 32 * GIBIBYTE;
const METRICS_ROW_GROUP_ROWS: u32 = 131_072;

pub(crate) const LEVANTER_RUN_PARTITION_POLICY: StringIdentityPartitionPolicy =
    StringIdentityPartitionPolicy {
        spec_id: 1,
        column: "run_id",
        partition_field: "run_id",
    };

#[derive(Clone, Copy, Debug)]
pub(crate) struct LevanterMetricsPolicy;

pub(crate) const LEVANTER_METRICS_POLICY: LevanterMetricsPolicy = LevanterMetricsPolicy;

pub(crate) fn matches_levanter_metrics_namespace(namespace: &str) -> bool {
    namespace == LEVANTER_METRICS_NAMESPACE || namespace.starts_with(LEVANTER_METRICS_ALIAS_PREFIX)
}

impl LevanterMetricsPolicy {
    pub(crate) fn registration_namespace(&self, namespace: &str) -> Result<String, StatsError> {
        declared_run_id(namespace)?;
        Ok(LEVANTER_METRICS_NAMESPACE.to_string())
    }

    pub(crate) fn physical_partition_policy(
        &self,
        namespace: &str,
    ) -> Option<&'static dyn PhysicalPartitionPolicy> {
        (namespace == LEVANTER_METRICS_NAMESPACE)
            .then_some(&LEVANTER_RUN_PARTITION_POLICY as &dyn PhysicalPartitionPolicy)
    }
}

impl IngestionPolicy for LevanterMetricsPolicy {
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
        _state: &mut IngestionState,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        let requested_run_id = declared_run_id(source.namespace())?;
        validate_metric_batch(batch, requested_run_id)?;
        Ok(vec![RoutedIngestionBatch {
            destination: IngestionDestination {
                logical_namespace: LEVANTER_METRICS_NAMESPACE.to_string(),
            },
            batch: batch.clone(),
        }])
    }
}

impl NamespaceStoragePolicy for LevanterMetricsPolicy {
    fn storage_policy(&self, namespace: &str) -> Result<StoragePolicy, StatsError> {
        declared_run_id(namespace)?;
        Ok(StoragePolicy {
            max_bytes: Some(LEVANTER_METRICS_MAX_BYTES),
            ..StoragePolicy::default()
        })
    }

    fn eager_namespaces(&self) -> Vec<&str> {
        vec![LEVANTER_METRICS_NAMESPACE]
    }
}

fn declared_run_id(namespace: &str) -> Result<Option<&str>, StatsError> {
    if namespace == LEVANTER_METRICS_NAMESPACE {
        return Ok(None);
    }
    let run_id = namespace
        .strip_prefix(LEVANTER_METRICS_ALIAS_PREFIX)
        .filter(|run_id| !run_id.is_empty())
        .ok_or_else(|| {
            StatsError::InvalidNamespace(format!(
                "Levanter metric namespace {namespace:?} must be {LEVANTER_METRICS_NAMESPACE:?} or carry a nonempty run id"
            ))
        })?;
    Ok(Some(run_id))
}

fn validate_metric_batch(
    batch: &RecordBatch,
    requested_run_id: Option<&str>,
) -> Result<(), StatsError> {
    let run_ids = string_column(batch, "run_id")?;
    let names = string_column(batch, "name")?;
    let kinds = string_column(batch, "kind")?;
    let timestamps = int64_column(batch, "timestamp_ms")?;
    let values = float64_column(batch, "value")?;
    let minima = float64_column(batch, "min")?;
    let maxima = float64_column(batch, "max")?;
    let counts = int64_column(batch, "count")?;
    let nonzero_counts = int64_column(batch, "nonzero_count")?;
    let sums = float64_column(batch, "sum")?;
    let sum_squares = float64_column(batch, "sum_squares")?;
    let means = float64_column(batch, "mean")?;
    let variances = float64_column(batch, "variance")?;
    let root_mean_squares = float64_column(batch, "rms")?;
    let bucket_limits = list_column(batch, "bucket_limits")?;
    let bucket_counts = list_column(batch, "bucket_counts")?;

    for row in 0..batch.num_rows() {
        require_non_null(timestamps, "timestamp_ms", row)?;
        require_non_null(run_ids, "run_id", row)?;
        require_non_null(names, "name", row)?;
        require_non_null(kinds, "kind", row)?;
        let run_id = run_ids.value(row);
        if requested_run_id.is_some_and(|requested| requested != run_id) {
            return Err(StatsError::SchemaValidation(format!(
                "Levanter metric namespace run id {requested_run_id:?} does not match row run_id {run_id:?}"
            )));
        }

        let summary_values: [&dyn Array; 9] = [
            minima,
            maxima,
            counts,
            nonzero_counts,
            sums,
            sum_squares,
            means,
            variances,
            root_mean_squares,
        ];
        match kinds.value(row) {
            "scalar" => {
                require_non_null(values, "value", row)?;
                require_all_null(&summary_values, "summary", row)?;
                require_null(bucket_limits, "bucket_limits", row)?;
                require_null(bucket_counts, "bucket_counts", row)?;
            }
            "summary" | "histogram" => {
                require_null(values, "value", row)?;
                require_all_non_null(&summary_values, "summary", row)?;
                if kinds.value(row) == "summary" {
                    require_null(bucket_limits, "bucket_limits", row)?;
                    require_null(bucket_counts, "bucket_counts", row)?;
                } else {
                    require_non_null(bucket_limits, "bucket_limits", row)?;
                    require_non_null(bucket_counts, "bucket_counts", row)?;
                    if bucket_limits.value_length(row) != bucket_counts.value_length(row) + 1 {
                        return Err(StatsError::SchemaValidation(format!(
                            "histogram row {row} requires one more bucket limit than bucket counts"
                        )));
                    }
                }
            }
            kind => {
                return Err(StatsError::SchemaValidation(format!(
                    "Levanter metric row {row} has unsupported kind {kind:?}"
                )))
            }
        }
    }
    Ok(())
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, StatsError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| required_column_error(name, "UTF-8"))
}

fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, StatsError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| required_column_error(name, "int64"))
}

fn float64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array, StatsError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Float64Array>())
        .ok_or_else(|| required_column_error(name, "float64"))
}

fn list_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ListArray, StatsError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<ListArray>())
        .ok_or_else(|| required_column_error(name, "list"))
}

fn required_column_error(name: &str, data_type: &str) -> StatsError {
    StatsError::SchemaValidation(format!(
        "Levanter metrics policy requires a {data_type} {name:?} column"
    ))
}

fn require_non_null(values: &dyn Array, name: &str, row: usize) -> Result<(), StatsError> {
    if values.is_null(row) {
        return Err(StatsError::SchemaValidation(format!(
            "Levanter metric row {row} requires a non-null {name:?} value"
        )));
    }
    Ok(())
}

fn require_null(values: &dyn Array, name: &str, row: usize) -> Result<(), StatsError> {
    if values.is_valid(row) {
        return Err(StatsError::SchemaValidation(format!(
            "Levanter metric row {row} requires a null {name:?} value"
        )));
    }
    Ok(())
}

fn require_all_non_null(values: &[&dyn Array], name: &str, row: usize) -> Result<(), StatsError> {
    if values.iter().any(|value| value.is_null(row)) {
        return Err(StatsError::SchemaValidation(format!(
            "Levanter metric row {row} requires complete {name} values"
        )));
    }
    Ok(())
}

fn require_all_null(values: &[&dyn Array], name: &str, row: usize) -> Result<(), StatsError> {
    if values.iter().any(|value| value.is_valid(row)) {
        return Err(StatsError::SchemaValidation(format!(
            "Levanter metric row {row} requires null {name} values"
        )));
    }
    Ok(())
}

pub(crate) fn levanter_metrics_schema() -> Schema {
    Schema::new(
        vec![
            nullable_column("timestamp_ms", ColumnType::COLUMN_TYPE_INT64),
            nullable_column("run_id", ColumnType::COLUMN_TYPE_STRING),
            nullable_column("execution_uid", ColumnType::COLUMN_TYPE_STRING),
            nullable_column("job_id", ColumnType::COLUMN_TYPE_STRING),
            nullable_column("node_name", ColumnType::COLUMN_TYPE_STRING),
            nullable_column("process_index", ColumnType::COLUMN_TYPE_INT64),
            nullable_column("step", ColumnType::COLUMN_TYPE_INT64),
            nullable_column("name", ColumnType::COLUMN_TYPE_STRING).with_trigram_index(),
            nullable_column("kind", ColumnType::COLUMN_TYPE_STRING)
                .with_exact_values(["scalar", "summary", "histogram"])
                .with_value_counts(),
            nullable_column("value", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("min", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("max", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("count", ColumnType::COLUMN_TYPE_INT64),
            nullable_column("nonzero_count", ColumnType::COLUMN_TYPE_INT64),
            nullable_column("sum", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("sum_squares", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("mean", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("variance", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("rms", ColumnType::COLUMN_TYPE_FLOAT64),
            nullable_column("bucket_limits", ColumnType::COLUMN_TYPE_FLOAT64_LIST),
            nullable_column("bucket_counts", ColumnType::COLUMN_TYPE_INT64_LIST),
            nullable_column("unit", ColumnType::COLUMN_TYPE_STRING),
            nullable_column("attributes", ColumnType::COLUMN_TYPE_MAP),
            nullable_column("batch_id", ColumnType::COLUMN_TYPE_STRING),
            nullable_column("record_index", ColumnType::COLUMN_TYPE_INT64),
        ],
        "timestamp_ms",
    )
    .with_sort_columns(["run_id", "name", "step", "timestamp_ms"])
    .with_max_row_group_rows(METRICS_ROW_GROUP_ROWS)
}

fn nullable_column(name: &str, column_type: ColumnType) -> Column {
    Column::new(name, column_type, true)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{new_null_array, Float64Array, Int64Array, StringArray};

    use super::*;
    use crate::store::schema::schema_to_arrow;

    fn scalar_batch(run_id: &str) -> RecordBatch {
        let schema = levanter_metrics_schema();
        let arrow_schema = schema_to_arrow(&schema);
        RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![Some(10)])),
                Arc::new(StringArray::from(vec![Some(run_id)])),
                Arc::new(StringArray::from(vec![Some("attempt-1")])),
                Arc::new(StringArray::from(vec![Some("/job")])),
                Arc::new(StringArray::from(vec![Some("node")])),
                Arc::new(Int64Array::from(vec![Some(0)])),
                Arc::new(Int64Array::from(vec![Some(7)])),
                Arc::new(StringArray::from(vec![Some("train_loss")])),
                Arc::new(StringArray::from(vec![Some("scalar")])),
                Arc::new(Float64Array::from(vec![Some(0.5)])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(Int64Array::from(vec![None])),
                Arc::new(Int64Array::from(vec![None])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(Float64Array::from(vec![None])),
                new_null_array(arrow_schema.field(19).data_type(), 1),
                new_null_array(arrow_schema.field(20).data_type(), 1),
                Arc::new(StringArray::from(vec![None::<&str>])),
                new_null_array(arrow_schema.field(22).data_type(), 1),
                Arc::new(StringArray::from(vec![None::<&str>])),
                Arc::new(Int64Array::from(vec![None])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn per_run_alias_routes_to_one_canonical_table() {
        let batch = scalar_batch("run/+long/name");
        let routed = LEVANTER_METRICS_POLICY
            .route_batch(
                IngestionBatchSource::Declared("levanter.metrics.run/+long/name"),
                &batch,
                &mut IngestionState::default(),
            )
            .unwrap();
        assert_eq!(routed.len(), 1);
        assert_eq!(
            routed[0].destination.logical_namespace,
            LEVANTER_METRICS_NAMESPACE
        );
        assert_eq!(
            LEVANTER_METRICS_POLICY
                .registration_namespace("levanter.metrics.run/+long/name")
                .unwrap(),
            LEVANTER_METRICS_NAMESPACE
        );
    }

    #[test]
    fn alias_and_record_run_id_must_agree() {
        let error = LEVANTER_METRICS_POLICY
            .route_batch(
                IngestionBatchSource::Declared("levanter.metrics.other"),
                &scalar_batch("run"),
                &mut IngestionState::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("does not match"));
    }
}
