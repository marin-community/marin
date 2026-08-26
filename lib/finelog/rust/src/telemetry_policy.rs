//! Stable telemetry stream names and their logical and physical policies.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use arrow::array::{new_null_array, Array, ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;
use crate::ingestion_policy::{
    IngestionBatchSource, IngestionDestination, IngestionPolicy, IngestionState,
    RoutedIngestionBatch, StepCursor,
};
use crate::levanter_metrics_policy::{levanter_metrics_schema, LEVANTER_METRICS_NAMESPACE};
use crate::partition_policy::{select_rows, PhysicalPartitionPolicy};
use crate::storage_policy::NamespaceStoragePolicy;
use crate::store::namespace_name::MAX_NAMESPACE_NAME_BYTES;
use crate::store::policy::StoragePolicy;
use crate::store::schema::schema_to_arrow;

pub(crate) const TELEMETRY_NAMESPACE: &str = "telemetry_v1";
const SEMANTIC_NAMESPACE_PREFIX: &str = "telemetry_v1.";
const GIBIBYTE: i64 = 1024 * 1024 * 1024;
const DEFAULT_STREAM_MAX_BYTES: i64 = 2 * GIBIBYTE;
const IRIS_CONTROLLER_SERVICE: &str = "iris-controller";
const LEVANTER_SERVICE: &str = "levanter";
const PROCESS_INDEX_COLUMN: &str = "process_index";
pub(crate) const LEVANTER_NAMESPACE: &str = "telemetry_v1.levanter";
pub(crate) const NODE_AGENT_NAMESPACE: &str = "telemetry_v1.node_agent";
pub(crate) const IRIS_RPC_NAMESPACE: &str = "telemetry_v1.iris.rpc";
pub(crate) const IRIS_NAMESPACE: &str = "telemetry_v1.iris";
pub(crate) const VLLM_NAMESPACE: &str = "telemetry_v1.vllm";
pub(crate) const ZEPHYR_NAMESPACE: &str = "telemetry_v1.zephyr";
const LEGACY_STEP_METRIC_NAMES: [&str; 2] = ["step", "global_step"];

#[derive(Clone, Copy, Debug)]
pub(crate) struct TelemetryPolicy {
    logical_inference_rules: &'static [LogicalInferenceRule],
}

#[derive(Debug)]
struct LogicalInferenceRule {
    service: &'static str,
    name_prefixes: &'static [&'static str],
    logical_namespace: &'static str,
}

const LOGICAL_INFERENCE_RULES: [LogicalInferenceRule; 6] = [
    LogicalInferenceRule {
        service: IRIS_CONTROLLER_SERVICE,
        name_prefixes: &["rpc_", "proxy_"],
        logical_namespace: IRIS_RPC_NAMESPACE,
    },
    LogicalInferenceRule {
        service: IRIS_CONTROLLER_SERVICE,
        name_prefixes: &[],
        logical_namespace: IRIS_NAMESPACE,
    },
    LogicalInferenceRule {
        service: "iris-node-agent",
        name_prefixes: &[],
        logical_namespace: NODE_AGENT_NAMESPACE,
    },
    LogicalInferenceRule {
        service: LEVANTER_SERVICE,
        name_prefixes: &[],
        logical_namespace: LEVANTER_NAMESPACE,
    },
    LogicalInferenceRule {
        service: "vllm",
        name_prefixes: &[],
        logical_namespace: VLLM_NAMESPACE,
    },
    LogicalInferenceRule {
        service: "zephyr",
        name_prefixes: &[],
        logical_namespace: ZEPHYR_NAMESPACE,
    },
];

pub(crate) const TELEMETRY_POLICY: TelemetryPolicy = TelemetryPolicy {
    logical_inference_rules: &LOGICAL_INFERENCE_RULES,
};

impl IngestionPolicy for TelemetryPolicy {
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
        state: &mut IngestionState,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        if source.namespace() != TELEMETRY_NAMESPACE {
            if !is_semantic_namespace(source.namespace()) {
                return Err(namespace_error(source.namespace()));
            }
            return Ok(vec![RoutedIngestionBatch {
                destination: IngestionDestination {
                    logical_namespace: source.namespace().to_string(),
                },
                batch: batch.clone(),
            }]);
        }

        let mut partitions: BTreeMap<IngestionDestination, Vec<u32>> = BTreeMap::new();
        let mut legacy_levanter_rows = Vec::new();
        for row_index in 0..batch.num_rows() {
            let record = TelemetryRecord { batch, row_index };
            if record.is_legacy_levanter_metric()? {
                legacy_levanter_rows.push(row_index as u32);
                continue;
            }
            let logical_namespace = self.infer_logical_namespace(&record)?;
            partitions
                .entry(IngestionDestination { logical_namespace })
                .or_default()
                .push(row_index as u32);
        }
        let mut routed = partitions
            .into_iter()
            .map(|(destination, row_indices)| {
                let batch = select_rows(batch, row_indices)?;
                Ok(RoutedIngestionBatch { destination, batch })
            })
            .collect::<Result<Vec<_>, StatsError>>()?;
        if let Some(batch) = transform_legacy_levanter(batch, &legacy_levanter_rows, state)? {
            routed.push(RoutedIngestionBatch {
                destination: IngestionDestination {
                    logical_namespace: LEVANTER_METRICS_NAMESPACE.to_string(),
                },
                batch,
            });
        }
        Ok(routed)
    }

    fn index_migration_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
        state: &mut IngestionState,
    ) -> Result<(), StatsError> {
        if source.namespace() != TELEMETRY_NAMESPACE {
            return Ok(());
        }
        let mut resource_cache = HashMap::new();
        for row_index in 0..batch.num_rows() {
            let record = TelemetryRecord { batch, row_index };
            if !record.is_legacy_levanter_metric()? {
                continue;
            }
            let Some((execution_uid, process_index, cursor)) =
                legacy_step_cursor(&record, &mut resource_cache)?
            else {
                continue;
            };
            state.index_levanter_step(execution_uid, process_index, cursor);
        }
        Ok(())
    }
}

impl TelemetryPolicy {
    /// Infer the semantic namespace for an old writer using the complete row.
    fn infer_logical_namespace(&self, record: &TelemetryRecord<'_>) -> Result<String, StatsError> {
        let service = record.required_string("service")?;
        let name = record.required_string("name")?;
        if let Some(namespace) = self.logical_inference_rules.iter().find_map(|rule| {
            (rule.service == service
                && (rule.name_prefixes.is_empty()
                    || rule
                        .name_prefixes
                        .iter()
                        .any(|prefix| name.starts_with(prefix))))
            .then_some(rule.logical_namespace)
        }) {
            return Ok(namespace.to_string());
        }
        Ok(format!(
            "{SEMANTIC_NAMESPACE_PREFIX}{}",
            normalized_service_component(service)
        ))
    }

    pub(crate) fn physical_partition_policy(
        &self,
        _namespace: &str,
    ) -> Option<&'static dyn PhysicalPartitionPolicy> {
        None
    }
}

impl NamespaceStoragePolicy for TelemetryPolicy {
    fn storage_policy(&self, namespace: &str) -> Result<StoragePolicy, StatsError> {
        let max_bytes = match namespace {
            TELEMETRY_NAMESPACE => 50 * GIBIBYTE,
            LEVANTER_NAMESPACE => 32 * GIBIBYTE,
            NODE_AGENT_NAMESPACE => 15 * GIBIBYTE,
            IRIS_RPC_NAMESPACE => GIBIBYTE,
            VLLM_NAMESPACE => 2 * GIBIBYTE,
            _ if is_semantic_namespace(namespace) => DEFAULT_STREAM_MAX_BYTES,
            _ => return Err(namespace_error(namespace)),
        };
        Ok(StoragePolicy {
            max_bytes: Some(max_bytes),
            ..StoragePolicy::default()
        })
    }

    fn eager_namespaces(&self) -> Vec<&str> {
        vec![
            LEVANTER_METRICS_NAMESPACE,
            LEVANTER_NAMESPACE,
            NODE_AGENT_NAMESPACE,
            IRIS_RPC_NAMESPACE,
            IRIS_NAMESPACE,
            VLLM_NAMESPACE,
            ZEPHYR_NAMESPACE,
        ]
    }
}

struct TelemetryRecord<'a> {
    batch: &'a RecordBatch,
    row_index: usize,
}

impl TelemetryRecord<'_> {
    fn required_string(&self, column_name: &str) -> Result<&str, StatsError> {
        let values = self
            .batch
            .column_by_name(column_name)
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "telemetry policy requires a UTF-8 {column_name:?} column"
                ))
            })?;
        if values.is_null(self.row_index) {
            return Err(StatsError::SchemaValidation(format!(
                "telemetry policy requires non-null {column_name:?} values"
            )));
        }
        Ok(values.value(self.row_index))
    }

    fn optional_string(&self, column_name: &str) -> Result<Option<&str>, StatsError> {
        let Some(column) = self.batch.column_by_name(column_name) else {
            return Ok(None);
        };
        let values = column
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "telemetry policy requires a UTF-8 {column_name:?} column"
                ))
            })?;
        Ok((!values.is_null(self.row_index)).then(|| values.value(self.row_index)))
    }

    fn required_int64(&self, column_name: &str) -> Result<i64, StatsError> {
        let values = self
            .batch
            .column_by_name(column_name)
            .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "telemetry policy requires an int64 {column_name:?} column"
                ))
            })?;
        if values.is_null(self.row_index) {
            return Err(StatsError::SchemaValidation(format!(
                "telemetry policy requires non-null {column_name:?} values"
            )));
        }
        Ok(values.value(self.row_index))
    }

    fn optional_int64(&self, column_name: &str) -> Result<Option<i64>, StatsError> {
        let Some(column) = self.batch.column_by_name(column_name) else {
            return Ok(None);
        };
        let values = column
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "telemetry policy requires an int64 {column_name:?} column"
                ))
            })?;
        Ok((!values.is_null(self.row_index)).then(|| values.value(self.row_index)))
    }

    fn required_float64(&self, column_name: &str) -> Result<f64, StatsError> {
        let values = self
            .batch
            .column_by_name(column_name)
            .and_then(|column| column.as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "telemetry policy requires a float64 {column_name:?} column"
                ))
            })?;
        if values.is_null(self.row_index) {
            return Err(StatsError::SchemaValidation(format!(
                "telemetry policy requires non-null {column_name:?} values"
            )));
        }
        Ok(values.value(self.row_index))
    }

    fn is_legacy_levanter_metric(&self) -> Result<bool, StatsError> {
        if self.required_string("service")? != LEVANTER_SERVICE {
            return Ok(false);
        }
        let attributes: serde_json::Value =
            serde_json::from_str(self.required_string("attributes_json")?).map_err(|error| {
                StatsError::SchemaValidation(format!(
                    "Levanter telemetry attributes are not valid JSON: {error}"
                ))
            })?;
        Ok(attributes
            .get("source_kind")
            .and_then(serde_json::Value::as_str)
            == Some("gauge"))
    }
}

#[derive(Clone, Debug, Default)]
struct LegacyResource {
    run_id: Option<String>,
    execution_uid: Option<String>,
    job_id: Option<String>,
    node_name: Option<String>,
    process_index: Option<String>,
}

#[derive(Clone, Debug)]
struct LegacyMetricRow {
    timestamp_ms: i64,
    order: i64,
    run_id: String,
    execution_uid: String,
    job_id: Option<String>,
    node_name: Option<String>,
    process_index: i64,
    name: String,
    value: f64,
    unit: Option<String>,
    batch_id: String,
    record_index: i64,
}

#[derive(Debug)]
struct LegacyTypedMetric {
    row: LegacyMetricRow,
    step: Option<i64>,
    value: LegacyMetricValue,
}

#[derive(Debug)]
enum LegacyMetricValue {
    Scalar(f64),
    Summary {
        min: f64,
        max: f64,
        count: i64,
        sum: f64,
        sum_squares: f64,
        mean: f64,
        variance: f64,
        rms: f64,
    },
}

const LEGACY_SUMMARY_SUFFIXES: [&str; 7] = [
    "_mean",
    "_min",
    "_max",
    "_variance",
    "_rms",
    "_count",
    "_sum",
];

fn transform_legacy_levanter(
    batch: &RecordBatch,
    row_indices: &[u32],
    state: &mut IngestionState,
) -> Result<Option<RecordBatch>, StatsError> {
    if row_indices.is_empty() {
        return Ok(None);
    }
    let mut resource_cache = HashMap::new();
    let mut rows = row_indices
        .iter()
        .map(|row_index| legacy_metric_row(batch, *row_index as usize, &mut resource_cache))
        .collect::<Result<Vec<_>, StatsError>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        (&left.execution_uid, left.timestamp_ms, left.order).cmp(&(
            &right.execution_uid,
            right.timestamp_ms,
            right.order,
        ))
    });

    let mut scalar_rows = Vec::new();
    for row in rows {
        if LEGACY_STEP_METRIC_NAMES.contains(&row.name.as_str()) {
            let step = exact_step(row.value)?;
            let candidate = StepCursor {
                timestamp_ms: row.timestamp_ms,
                order: row.order,
                step,
            };
            state.update_levanter_step(row.execution_uid, row.process_index, candidate);
            continue;
        }
        let step = state.levanter_step_at(
            &row.execution_uid,
            row.process_index,
            row.timestamp_ms,
            row.order,
        );
        scalar_rows.push((row, step));
    }
    if scalar_rows.is_empty() {
        return Ok(None);
    }
    Ok(Some(legacy_metric_batch(collapse_legacy_summaries(
        scalar_rows,
    )?)?))
}

fn legacy_step_cursor(
    record: &TelemetryRecord<'_>,
    resource_cache: &mut HashMap<String, LegacyResource>,
) -> Result<Option<(String, i64, StepCursor)>, StatsError> {
    let name = record.required_string("name")?;
    if !LEGACY_STEP_METRIC_NAMES.contains(&name) {
        return Ok(None);
    }
    let resource_json = record.required_string("resource_attributes_json")?;
    if !resource_cache.contains_key(resource_json) {
        resource_cache.insert(
            resource_json.to_string(),
            parse_legacy_resource(resource_json)?,
        );
    }
    let resource = resource_cache
        .get(resource_json)
        .expect("legacy resource was inserted");
    let process_index = record
        .optional_string(PROCESS_INDEX_COLUMN)?
        .or(resource.process_index.as_deref())
        .ok_or_else(|| {
            StatsError::SchemaValidation("legacy Levanter metric has no process_index".to_string())
        })?;
    if process_index != "0" {
        return Ok(None);
    }
    let execution_uid = record
        .optional_string("execution_uid")?
        .map(str::to_string)
        .or_else(|| resource.execution_uid.clone())
        .ok_or_else(|| {
            StatsError::SchemaValidation("legacy Levanter metric has no execution_uid".to_string())
        })?;
    let cursor = StepCursor {
        timestamp_ms: record.required_int64("timestamp_ms")?,
        order: record
            .optional_int64("seq")?
            .unwrap_or(record.required_int64("record_index")?),
        step: exact_step(record.required_float64("value")?)?,
    };
    Ok(Some((execution_uid, 0, cursor)))
}

fn collapse_legacy_summaries(
    rows: Vec<(LegacyMetricRow, Option<i64>)>,
) -> Result<Vec<LegacyTypedMetric>, StatsError> {
    let mut output = Vec::with_capacity(rows.len());
    let mut pending = VecDeque::from(rows);
    while !pending.is_empty() {
        let candidate = pending
            .iter()
            .take(LEGACY_SUMMARY_SUFFIXES.len())
            .collect::<Vec<_>>();
        if let Some(summary) = legacy_summary(&candidate)? {
            output.push(summary);
            for _ in 0..LEGACY_SUMMARY_SUFFIXES.len() {
                pending.pop_front();
            }
            continue;
        }
        let current = pending.pop_front().expect("pending is nonempty");
        output.push(LegacyTypedMetric {
            value: LegacyMetricValue::Scalar(current.0.value),
            row: current.0,
            step: current.1,
        });
    }
    Ok(output)
}

fn legacy_summary(
    rows: &[&(LegacyMetricRow, Option<i64>)],
) -> Result<Option<LegacyTypedMetric>, StatsError> {
    if rows.len() < LEGACY_SUMMARY_SUFFIXES.len() {
        return Ok(None);
    }
    let candidate = &rows[..LEGACY_SUMMARY_SUFFIXES.len()];
    let Some(base_name) = candidate[0].0.name.strip_suffix(LEGACY_SUMMARY_SUFFIXES[0]) else {
        return Ok(None);
    };
    if base_name.is_empty() {
        return Ok(None);
    }
    for (index, ((row, step), suffix)) in candidate.iter().zip(LEGACY_SUMMARY_SUFFIXES).enumerate()
    {
        if row.name.strip_suffix(suffix) != Some(base_name)
            || row.batch_id != candidate[0].0.batch_id
            || row.timestamp_ms != candidate[0].0.timestamp_ms
            || row.execution_uid != candidate[0].0.execution_uid
            || row.run_id != candidate[0].0.run_id
            || *step != candidate[0].1
            || row.record_index != candidate[0].0.record_index + index as i64
        {
            return Ok(None);
        }
    }

    let count = exact_nonnegative_count(candidate[5].0.value)?;
    let mean = candidate[0].0.value;
    let rms = candidate[4].0.value;
    let row = LegacyMetricRow {
        name: base_name.to_string(),
        ..candidate[0].0.clone()
    };
    Ok(Some(LegacyTypedMetric {
        row,
        step: candidate[0].1,
        value: LegacyMetricValue::Summary {
            min: candidate[1].0.value,
            max: candidate[2].0.value,
            variance: candidate[3].0.value,
            rms,
            count,
            sum: candidate[6].0.value,
            sum_squares: rms * rms * count as f64,
            mean,
        },
    }))
}

fn exact_nonnegative_count(value: f64) -> Result<i64, StatsError> {
    let count = exact_step(value)?;
    if count < 0 {
        return Err(StatsError::SchemaValidation(format!(
            "legacy Levanter summary count {value:?} is negative"
        )));
    }
    Ok(count)
}

fn legacy_metric_row(
    batch: &RecordBatch,
    row: usize,
    resource_cache: &mut HashMap<String, LegacyResource>,
) -> Result<Option<LegacyMetricRow>, StatsError> {
    let record = TelemetryRecord {
        batch,
        row_index: row,
    };
    let resource_json = record.required_string("resource_attributes_json")?;
    if !resource_cache.contains_key(resource_json) {
        resource_cache.insert(
            resource_json.to_string(),
            parse_legacy_resource(resource_json)?,
        );
    }
    let resource = resource_cache
        .get(resource_json)
        .expect("legacy resource was inserted");
    let process_index = record
        .optional_string(PROCESS_INDEX_COLUMN)?
        .or(resource.process_index.as_deref())
        .ok_or_else(|| {
            StatsError::SchemaValidation("legacy Levanter metric has no process_index".to_string())
        })?;
    if process_index != "0" {
        return Ok(None);
    }
    let run_id = record
        .optional_string("run_id")?
        .map(str::to_string)
        .or_else(|| resource.run_id.clone())
        .ok_or_else(|| {
            StatsError::SchemaValidation("legacy Levanter metric has no run id".to_string())
        })?;
    let execution_uid = record
        .optional_string("execution_uid")?
        .map(str::to_string)
        .or_else(|| resource.execution_uid.clone())
        .ok_or_else(|| {
            StatsError::SchemaValidation("legacy Levanter metric has no execution_uid".to_string())
        })?;
    let value = record.required_float64("value")?;
    Ok(Some(LegacyMetricRow {
        timestamp_ms: record.required_int64("timestamp_ms")?,
        order: record
            .optional_int64("seq")?
            .unwrap_or(record.required_int64("record_index")?),
        run_id,
        execution_uid,
        job_id: record
            .optional_string("job_id")?
            .map(str::to_string)
            .or_else(|| resource.job_id.clone()),
        node_name: record
            .optional_string("node_name")?
            .map(str::to_string)
            .or_else(|| resource.node_name.clone()),
        process_index: 0,
        name: record.required_string("name")?.to_string(),
        value,
        unit: record.optional_string("unit")?.map(str::to_string),
        batch_id: record.required_string("batch_id")?.to_string(),
        record_index: record.required_int64("record_index")?,
    }))
}

fn parse_legacy_resource(json: &str) -> Result<LegacyResource, StatsError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|error| {
        StatsError::SchemaValidation(format!(
            "legacy Levanter resource attributes are not valid JSON: {error}"
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        StatsError::SchemaValidation(
            "legacy Levanter resource attributes must be a JSON object".to_string(),
        )
    })?;
    let string = |key: &str| {
        object
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };
    Ok(LegacyResource {
        run_id: string("run_id")
            .or_else(|| string("root_run_uid"))
            .or_else(|| string("run")),
        execution_uid: string("execution_uid"),
        job_id: string("job_id"),
        node_name: string("node_name"),
        process_index: string(PROCESS_INDEX_COLUMN),
    })
}

fn exact_step(value: f64) -> Result<i64, StatsError> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i64::MIN as f64
        || value > i64::MAX as f64
    {
        return Err(StatsError::SchemaValidation(format!(
            "legacy Levanter step {value:?} is not an exact int64"
        )));
    }
    Ok(value as i64)
}

fn legacy_metric_batch(rows: Vec<LegacyTypedMetric>) -> Result<RecordBatch, StatsError> {
    let schema = schema_to_arrow(&levanter_metrics_schema());
    let row_count = rows.len();
    let timestamp_ms =
        Int64Array::from_iter_values(rows.iter().map(|metric| metric.row.timestamp_ms));
    let run_ids =
        StringArray::from_iter_values(rows.iter().map(|metric| metric.row.run_id.as_str()));
    let execution_uids =
        StringArray::from_iter_values(rows.iter().map(|metric| metric.row.execution_uid.as_str()));
    let job_ids = StringArray::from_iter(rows.iter().map(|metric| metric.row.job_id.as_deref()));
    let node_names =
        StringArray::from_iter(rows.iter().map(|metric| metric.row.node_name.as_deref()));
    let process_indices =
        Int64Array::from_iter_values(rows.iter().map(|metric| metric.row.process_index));
    let steps = Int64Array::from_iter(rows.iter().map(|metric| metric.step));
    let names = StringArray::from_iter_values(rows.iter().map(|metric| metric.row.name.as_str()));
    let kinds = StringArray::from_iter_values(rows.iter().map(|metric| match &metric.value {
        LegacyMetricValue::Scalar(_) => "scalar",
        LegacyMetricValue::Summary { .. } => "summary",
    }));
    let values = Float64Array::from_iter(rows.iter().map(|metric| match &metric.value {
        LegacyMetricValue::Scalar(value) => Some(*value),
        LegacyMetricValue::Summary { .. } => None,
    }));
    let summary_float = |value: fn(&LegacyMetricValue) -> Option<f64>| {
        Float64Array::from_iter(rows.iter().map(|metric| value(&metric.value)))
    };
    let minima = summary_float(|value| match value {
        LegacyMetricValue::Summary { min, .. } => Some(*min),
        _ => None,
    });
    let maxima = summary_float(|value| match value {
        LegacyMetricValue::Summary { max, .. } => Some(*max),
        _ => None,
    });
    let counts = Int64Array::from_iter(rows.iter().map(|metric| match &metric.value {
        LegacyMetricValue::Summary { count, .. } => Some(*count),
        _ => None,
    }));
    let sums = summary_float(|value| match value {
        LegacyMetricValue::Summary { sum, .. } => Some(*sum),
        _ => None,
    });
    let sum_squares = summary_float(|value| match value {
        LegacyMetricValue::Summary { sum_squares, .. } => Some(*sum_squares),
        _ => None,
    });
    let means = summary_float(|value| match value {
        LegacyMetricValue::Summary { mean, .. } => Some(*mean),
        _ => None,
    });
    let variances = summary_float(|value| match value {
        LegacyMetricValue::Summary { variance, .. } => Some(*variance),
        _ => None,
    });
    let root_mean_squares = summary_float(|value| match value {
        LegacyMetricValue::Summary { rms, .. } => Some(*rms),
        _ => None,
    });
    let units = StringArray::from_iter(rows.iter().map(|metric| metric.row.unit.as_deref()));
    let batch_ids =
        StringArray::from_iter_values(rows.iter().map(|metric| metric.row.batch_id.as_str()));
    let record_indices =
        Int64Array::from_iter_values(rows.iter().map(|metric| metric.row.record_index));
    let null_column = |index: usize| new_null_array(schema.field(index).data_type(), row_count);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(timestamp_ms),
        Arc::new(run_ids),
        Arc::new(execution_uids),
        Arc::new(job_ids),
        Arc::new(node_names),
        Arc::new(process_indices),
        Arc::new(steps),
        Arc::new(names),
        Arc::new(kinds),
        Arc::new(values),
        Arc::new(minima),
        Arc::new(maxima),
        Arc::new(counts),
        null_column(13),
        Arc::new(sums),
        Arc::new(sum_squares),
        Arc::new(means),
        Arc::new(variances),
        Arc::new(root_mean_squares),
        null_column(19),
        null_column(20),
        Arc::new(units),
        null_column(22),
        Arc::new(batch_ids),
        Arc::new(record_indices),
    ];
    RecordBatch::try_new(schema, columns).map_err(|error| {
        StatsError::Internal(format!("build typed legacy Levanter metric batch: {error}"))
    })
}

fn namespace_error(namespace: &str) -> StatsError {
    StatsError::SchemaValidation(format!(
        "namespace {namespace:?} is not present in the telemetry stream policy"
    ))
}

fn normalized_service_component(service: &str) -> String {
    const MAX_SERVICE_BYTES: usize = MAX_NAMESPACE_NAME_BYTES - SEMANTIC_NAMESPACE_PREFIX.len();

    let mut normalized = String::with_capacity(service.len().min(MAX_SERVICE_BYTES));
    let mut previous_was_separator = false;
    for character in service.chars().flat_map(char::to_lowercase) {
        let character = if character.is_ascii_lowercase() || character.is_ascii_digit() {
            character
        } else {
            '_'
        };
        if character == '_' && (normalized.is_empty() || previous_was_separator) {
            continue;
        }
        if normalized.len() + character.len_utf8() > MAX_SERVICE_BYTES {
            break;
        }
        normalized.push(character);
        previous_was_separator = character == '_';
    }
    while normalized.ends_with('_') {
        normalized.pop();
    }
    if normalized.is_empty() {
        return "unknown".to_string();
    }
    if !normalized.starts_with(|character: char| character.is_ascii_lowercase()) {
        normalized.insert_str(0, "service_");
        normalized.truncate(MAX_SERVICE_BYTES);
    }
    normalized
}

pub(crate) fn is_forwarded_telemetry_namespace(namespace: &str) -> bool {
    namespace == LEVANTER_METRICS_NAMESPACE
        || namespace == TELEMETRY_NAMESPACE
        || is_semantic_namespace(namespace)
}

pub(crate) fn matches_telemetry_namespace(namespace: &str) -> bool {
    namespace == TELEMETRY_NAMESPACE || namespace.starts_with(SEMANTIC_NAMESPACE_PREFIX)
}

fn is_semantic_namespace(namespace: &str) -> bool {
    let Some(suffix) = namespace.strip_prefix(SEMANTIC_NAMESPACE_PREFIX) else {
        return false;
    };
    if namespace.len() > MAX_NAMESPACE_NAME_BYTES {
        return false;
    }
    suffix.split('.').all(|component| {
        let mut chars = component.chars();
        chars.next().is_some_and(|first| first.is_ascii_lowercase())
            && chars.all(|character| {
                character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
            })
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{new_null_array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::server::telemetry::telemetry_schema;

    fn telemetry_batch(services: &[&str], kinds: &[&str], names: &[&str]) -> RecordBatch {
        assert_eq!(services.len(), kinds.len());
        assert_eq!(services.len(), names.len());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("service", DataType::Utf8, false),
                Field::new("kind", DataType::Utf8, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("attributes_json", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(services.to_vec())),
                Arc::new(StringArray::from(kinds.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
                Arc::new(StringArray::from(vec![
                    "{\"source_kind\":\"nccl_ras\"}";
                    services.len()
                ])),
            ],
        )
        .unwrap()
    }

    fn destinations(source: IngestionBatchSource<'_>, batch: &RecordBatch) -> Vec<(String, usize)> {
        TELEMETRY_POLICY
            .route_batch(source, batch, &mut IngestionState::default())
            .unwrap()
            .into_iter()
            .map(|partition| {
                (
                    partition.destination.logical_namespace,
                    partition.batch.num_rows(),
                )
            })
            .collect()
    }

    fn legacy_batch(
        names: &[&str],
        values: &[f64],
        processes: &[&str],
        source_kinds: &[&str],
    ) -> RecordBatch {
        let row_count = names.len();
        assert_eq!(values.len(), row_count);
        assert_eq!(processes.len(), row_count);
        assert_eq!(source_kinds.len(), row_count);
        let schema = schema_to_arrow(&telemetry_schema());
        let resources = processes
            .iter()
            .map(|process| {
                format!(
                    "{{\"execution_uid\":\"attempt-1\",\"root_run_uid\":\"run/+long\",\"job_id\":\"/job\",\"process_index\":\"{process}\"}}"
                )
            })
            .collect::<Vec<_>>();
        let attributes = source_kinds
            .iter()
            .map(|kind| format!("{{\"source_kind\":\"{kind}\"}}"))
            .collect::<Vec<_>>();
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1; row_count])),
                Arc::new(Int64Array::from(vec![10; row_count])),
                Arc::new(StringArray::from(vec!["batch"; row_count])),
                Arc::new(Int64Array::from_iter_values(0..row_count as i64)),
                Arc::new(StringArray::from(vec!["levanter"; row_count])),
                new_null_array(schema.field(5).data_type(), row_count),
                new_null_array(schema.field(6).data_type(), row_count),
                new_null_array(schema.field(7).data_type(), row_count),
                new_null_array(schema.field(8).data_type(), row_count),
                new_null_array(schema.field(9).data_type(), row_count),
                Arc::new(StringArray::from(processes.to_vec())),
                Arc::new(StringArray::from(vec!["gauge"; row_count])),
                Arc::new(StringArray::from(names.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
                new_null_array(schema.field(14).data_type(), row_count),
                new_null_array(schema.field(15).data_type(), row_count),
                Arc::new(StringArray::from(resources)),
                Arc::new(StringArray::from(attributes)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn root_batch_infers_semantic_destinations_from_complete_rows() {
        let batch = telemetry_batch(
            &["levanter", "iris-controller", "rigging", "zephyr"],
            &["gauge"; 4],
            &[
                "train_loss",
                "rpc_latency_ms",
                "queue_depth",
                "progress_time_seconds",
            ],
        );
        let routed = destinations(IngestionBatchSource::Declared(TELEMETRY_NAMESPACE), &batch);
        assert!(routed.contains(&(LEVANTER_NAMESPACE.to_string(), 1)));
        assert!(routed.contains(&(IRIS_RPC_NAMESPACE.to_string(), 1)));
        assert!(routed.contains(&("telemetry_v1.rigging".to_string(), 1)));
        assert!(routed.contains(&(ZEPHYR_NAMESPACE.to_string(), 1)));
    }

    #[test]
    fn explicit_semantic_namespace_is_authoritative() {
        let batch = telemetry_batch(&["levanter"], &["gauge"], &["train_loss"]);
        assert_eq!(
            destinations(
                IngestionBatchSource::Declared("telemetry_v1.rigging.scheduler"),
                &batch,
            ),
            vec![("telemetry_v1.rigging.scheduler".to_string(), 1)]
        );
    }

    #[test]
    fn root_batch_derives_a_valid_namespace_from_an_unmapped_service() {
        let batch = telemetry_batch(&["123/Custom Service"], &["gauge"], &["queue_depth"]);
        assert_eq!(
            destinations(IngestionBatchSource::Declared(TELEMETRY_NAMESPACE), &batch),
            vec![("telemetry_v1.service_123_custom_service".to_string(), 1)]
        );
    }

    #[test]
    fn legacy_levanter_metrics_become_typed_process_zero_rows() {
        let batch = legacy_batch(
            &["step", "train_loss", "train_loss", "communicator_ranks"],
            &[7.0, 0.5, 0.6, 8.0],
            &["0", "0", "1", "0"],
            &["gauge", "gauge", "gauge", "nccl_ras"],
        );
        let mut state = IngestionState::default();
        let routed = TELEMETRY_POLICY
            .route_batch(
                IngestionBatchSource::Stored(TELEMETRY_NAMESPACE),
                &batch,
                &mut state,
            )
            .unwrap();

        let metrics = routed
            .iter()
            .find(|partition| partition.destination.logical_namespace == LEVANTER_METRICS_NAMESPACE)
            .unwrap();
        assert_eq!(metrics.batch.num_rows(), 1);
        let run_ids = metrics
            .batch
            .column_by_name("run_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let steps = metrics
            .batch
            .column_by_name("step")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(run_ids.value(0), "run/+long");
        assert_eq!(steps.value(0), 7);

        let ras = routed
            .iter()
            .find(|partition| partition.destination.logical_namespace == LEVANTER_NAMESPACE)
            .unwrap();
        assert_eq!(ras.batch.num_rows(), 1);
    }

    #[test]
    fn legacy_step_state_is_owned_by_the_registry_across_batches() {
        let mut state = IngestionState::default();
        let step = legacy_batch(&["step"], &[12.0], &["0"], &["gauge"]);
        assert!(TELEMETRY_POLICY
            .route_batch(
                IngestionBatchSource::Declared(TELEMETRY_NAMESPACE),
                &step,
                &mut state,
            )
            .unwrap()
            .is_empty());

        let loss = legacy_batch(&["train_loss"], &[0.25], &["0"], &["gauge"]);
        let routed = TELEMETRY_POLICY
            .route_batch(
                IngestionBatchSource::Declared(TELEMETRY_NAMESPACE),
                &loss,
                &mut state,
            )
            .unwrap();
        let steps = routed[0]
            .batch
            .column_by_name("step")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(steps.value(0), 12);
    }

    #[test]
    fn legacy_global_step_is_index_state_not_a_metric_row() {
        let mut state = IngestionState::default();
        let batch = legacy_batch(
            &["global_step", "train_loss"],
            &[14.0, 0.2],
            &["0", "0"],
            &["gauge", "gauge"],
        );
        let routed = TELEMETRY_POLICY
            .route_batch(
                IngestionBatchSource::Stored(TELEMETRY_NAMESPACE),
                &batch,
                &mut state,
            )
            .unwrap();
        assert_eq!(routed.len(), 1);
        assert_eq!(routed[0].batch.num_rows(), 1);
        let names = routed[0]
            .batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let steps = routed[0]
            .batch
            .column_by_name("step")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(names.value(0), "train_loss");
        assert_eq!(steps.value(0), 14);
    }

    #[test]
    fn legacy_seven_stat_summary_becomes_one_typed_row() {
        let batch = legacy_batch(
            &[
                "step",
                "grad_mean",
                "grad_min",
                "grad_max",
                "grad_variance",
                "grad_rms",
                "grad_count",
                "grad_sum",
            ],
            &[3.0, 2.0, -1.0, 5.0, 4.0, 3.0, 8.0, 16.0],
            &["0"; 8],
            &["gauge"; 8],
        );
        let routed = TELEMETRY_POLICY
            .route_batch(
                IngestionBatchSource::Stored(TELEMETRY_NAMESPACE),
                &batch,
                &mut IngestionState::default(),
            )
            .unwrap();
        assert_eq!(routed.len(), 1);
        let metrics = &routed[0].batch;
        assert_eq!(metrics.num_rows(), 1);
        assert_eq!(
            metrics
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "grad"
        );
        assert_eq!(
            metrics
                .column_by_name("kind")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0),
            "summary"
        );
        assert_eq!(
            metrics
                .column_by_name("count")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            8
        );
        assert_eq!(
            metrics
                .column_by_name("sum_squares")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(0),
            72.0
        );
    }
}
