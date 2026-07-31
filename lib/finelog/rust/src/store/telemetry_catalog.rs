//! Checked-in telemetry descriptor catalog and logical namespace schemas.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::proto::finelog::stats::ColumnType;
use crate::store::schema::{Column, Schema};

pub const CATALOG_VERSION: &str = "telemetry-catalog.v1";
pub const WORKLOAD_METRIC_NAMESPACE: &str = "telemetry.workload_metric.v1";
pub const SERVICE_METRIC_NAMESPACE: &str = "telemetry.service_metric.v1";
pub const HARDWARE_METRIC_NAMESPACE: &str = "telemetry.hardware_metric.v1";
pub const EVENT_NAMESPACE: &str = "telemetry.event.v1";
pub const ARTIFACT_NAMESPACE: &str = "telemetry.artifact.v1";
pub const RUN_NAMESPACE: &str = "telemetry.run.v1";
pub const ENTITY_LINK_NAMESPACE: &str = "telemetry.entity_link.v1";
pub const METRIC_ROLLUP_NAMESPACE: &str = "telemetry.metric_rollup.v1";
pub const ENTITY_STATE_NAMESPACE: &str = "telemetry.entity_state.v1";
pub const ALERT_STATE_NAMESPACE: &str = "telemetry.alert_state.v1";
pub const BATCH_INTENT_NAMESPACE: &str = "telemetry.batch_intent.v1";
pub const BATCH_NAMESPACE: &str = "telemetry.batch.v1";

const CATALOG_JSON: &str = include_str!("../../telemetry_catalog.v1.json");

#[derive(Debug, Clone, Deserialize)]
pub struct MetricCatalogEntry {
    pub scope: String,
    pub name: String,
    pub description: String,
    pub unit: String,
    pub instrument_kind: String,
    pub temporality: String,
    pub delivery_class: String,
    pub namespace: String,
    pub attributes: BTreeMap<String, Vec<String>>,
    pub buckets: Vec<f64>,
    pub owner: String,
    pub cadence_seconds: u64,
    pub cardinality_limit: usize,
    pub maturity: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventCatalogEntry {
    pub event_name: String,
    pub namespace: String,
    pub delivery_class: String,
    pub owner: String,
    pub attributes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CatalogDocument {
    catalog_version: String,
    metrics: Vec<MetricCatalogEntry>,
    events: Vec<EventCatalogEntry>,
    artifacts: Vec<String>,
}

#[derive(Debug)]
pub struct TelemetryCatalog {
    metrics: BTreeMap<(String, String), MetricCatalogEntry>,
    events: BTreeMap<String, EventCatalogEntry>,
    artifacts: BTreeSet<String>,
}

impl TelemetryCatalog {
    pub fn metric(&self, scope: &str, name: &str) -> Option<&MetricCatalogEntry> {
        self.metrics.get(&(scope.to_string(), name.to_string()))
    }

    pub fn event(&self, name: &str) -> Option<&EventCatalogEntry> {
        self.events.get(name)
    }

    pub fn artifact(&self, capture_type: &str) -> bool {
        self.artifacts.contains(capture_type)
    }
}

pub fn catalog() -> &'static TelemetryCatalog {
    static CATALOG: OnceLock<TelemetryCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let document: CatalogDocument =
            serde_json::from_str(CATALOG_JSON).expect("checked-in telemetry catalog is valid JSON");
        assert_eq!(
            document.catalog_version, CATALOG_VERSION,
            "telemetry catalog version drift"
        );
        let approved_namespaces: BTreeSet<&str> = approved_namespace_schemas()
            .iter()
            .map(|(name, _)| *name)
            .collect();
        let mut metrics = BTreeMap::new();
        for metric in document.metrics {
            assert!(
                approved_namespaces.contains(metric.namespace.as_str()),
                "metric {}.{} routes to unknown namespace {}",
                metric.scope,
                metric.name,
                metric.namespace
            );
            let key = (metric.scope.clone(), metric.name.clone());
            assert!(
                metrics.insert(key.clone(), metric).is_none(),
                "duplicate telemetry metric descriptor {key:?}"
            );
        }
        let mut events = BTreeMap::new();
        for event in document.events {
            assert!(
                approved_namespaces.contains(event.namespace.as_str()),
                "event {} routes to unknown namespace {}",
                event.event_name,
                event.namespace
            );
            assert!(
                events.insert(event.event_name.clone(), event).is_none(),
                "duplicate telemetry event descriptor"
            );
        }
        TelemetryCatalog {
            metrics,
            events,
            artifacts: document.artifacts.into_iter().collect(),
        }
    })
}

fn column(name: &str, r#type: ColumnType, nullable: bool) -> Column {
    Column::new(name, r#type, nullable)
}

pub(crate) fn base_record_columns() -> Vec<Column> {
    vec![
        column("schema_version", ColumnType::COLUMN_TYPE_INT32, false),
        column("catalog_version", ColumnType::COLUMN_TYPE_STRING, false),
        column("batch_id", ColumnType::COLUMN_TYPE_STRING, false),
        column("record_index", ColumnType::COLUMN_TYPE_INT32, false),
        column("point_id", ColumnType::COLUMN_TYPE_STRING, false),
        column("event_ts_unix_nano", ColumnType::COLUMN_TYPE_INT64, false),
        column(
            "observed_ts_unix_nano",
            ColumnType::COLUMN_TYPE_INT64,
            false,
        ),
    ]
}

pub fn resource_columns() -> Vec<Column> {
    vec![
        column("service_name", ColumnType::COLUMN_TYPE_STRING, false),
        column("service_instance_id", ColumnType::COLUMN_TYPE_STRING, false),
        column("role", ColumnType::COLUMN_TYPE_STRING, true),
        column("root_run_uid", ColumnType::COLUMN_TYPE_STRING, true),
        column("service_version", ColumnType::COLUMN_TYPE_STRING, true),
        column("run_id_alias", ColumnType::COLUMN_TYPE_STRING, true),
        column("iris_job_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("iris_task_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("task_index", ColumnType::COLUMN_TYPE_INT32, true),
        column("attempt_id", ColumnType::COLUMN_TYPE_INT32, true),
        column("attempt_uid", ColumnType::COLUMN_TYPE_STRING, true),
        column("worker_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("node_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("pod_uid", ColumnType::COLUMN_TYPE_STRING, true),
        column("container_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("rank", ColumnType::COLUMN_TYPE_INT32, true),
        column("process_index", ColumnType::COLUMN_TYPE_INT32, true),
        column("actor_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("engine_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("repository", ColumnType::COLUMN_TYPE_STRING, true),
        column("git_revision", ColumnType::COLUMN_TYPE_STRING, true),
        column("image_digest", ColumnType::COLUMN_TYPE_STRING, true),
        column("model_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("model_revision", ColumnType::COLUMN_TYPE_STRING, true),
        column("policy_step", ColumnType::COLUMN_TYPE_INT64, true),
        column("owner", ColumnType::COLUMN_TYPE_STRING, true),
        column("experiment_issue", ColumnType::COLUMN_TYPE_INT32, true),
        column("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        column("entity_authority", ColumnType::COLUMN_TYPE_STRING, true),
        column("entity_type", ColumnType::COLUMN_TYPE_STRING, true),
        column("entity_uid", ColumnType::COLUMN_TYPE_STRING, true),
    ]
}

fn metric_schema() -> Schema {
    let mut columns = base_record_columns();
    columns.extend(resource_columns());
    columns.extend([
        column("scope", ColumnType::COLUMN_TYPE_STRING, false),
        column("scope_version", ColumnType::COLUMN_TYPE_STRING, true),
        column("name", ColumnType::COLUMN_TYPE_STRING, false),
        column("description", ColumnType::COLUMN_TYPE_STRING, false),
        column("unit", ColumnType::COLUMN_TYPE_STRING, false),
        column("instrument_kind", ColumnType::COLUMN_TYPE_STRING, false),
        column("temporality", ColumnType::COLUMN_TYPE_STRING, false),
        column("start_ts_unix_nano", ColumnType::COLUMN_TYPE_INT64, true),
        column("reset_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("series_id", ColumnType::COLUMN_TYPE_STRING, false),
        column("value", ColumnType::COLUMN_TYPE_FLOAT64, true),
        column("count", ColumnType::COLUMN_TYPE_INT64, true),
        column("sum", ColumnType::COLUMN_TYPE_FLOAT64, true),
        column(
            "explicit_bounds",
            ColumnType::COLUMN_TYPE_LIST_FLOAT64,
            true,
        ),
        column("bucket_counts", ColumnType::COLUMN_TYPE_LIST_INT64, true),
        column("attributes", ColumnType::COLUMN_TYPE_MAP, false),
        column("delivery_class", ColumnType::COLUMN_TYPE_STRING, false),
        column("device_uid", ColumnType::COLUMN_TYPE_STRING, true),
        column("device_type", ColumnType::COLUMN_TYPE_STRING, true),
    ]);
    Schema::new(columns, "event_ts_unix_nano")
}

fn event_schema() -> Schema {
    let mut columns = base_record_columns();
    columns.extend(resource_columns());
    columns.extend([
        column("event_name", ColumnType::COLUMN_TYPE_STRING, false),
        column("severity_number", ColumnType::COLUMN_TYPE_INT32, false),
        column("severity_text", ColumnType::COLUMN_TYPE_STRING, false),
        column("outcome", ColumnType::COLUMN_TYPE_STRING, true),
        column("phase", ColumnType::COLUMN_TYPE_STRING, true),
        column("error_type", ColumnType::COLUMN_TYPE_STRING, true),
        column("body", ColumnType::COLUMN_TYPE_STRING, true).with_trigram_index(),
        column("attributes", ColumnType::COLUMN_TYPE_MAP, false),
        column("trace_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("span_id", ColumnType::COLUMN_TYPE_STRING, true),
        column("evidence_uri", ColumnType::COLUMN_TYPE_STRING, true),
        column("result_uri", ColumnType::COLUMN_TYPE_STRING, true),
        column("delivery_class", ColumnType::COLUMN_TYPE_STRING, false),
        column("probe_status", ColumnType::COLUMN_TYPE_STRING, true),
    ]);
    Schema::new(columns, "event_ts_unix_nano")
}

fn artifact_schema() -> Schema {
    let mut columns = base_record_columns();
    columns.extend(resource_columns());
    columns.extend([
        column("capture_type", ColumnType::COLUMN_TYPE_STRING, false),
        column("trigger", ColumnType::COLUMN_TYPE_STRING, false),
        column(
            "capture_start_ts_unix_nano",
            ColumnType::COLUMN_TYPE_INT64,
            true,
        ),
        column(
            "capture_end_ts_unix_nano",
            ColumnType::COLUMN_TYPE_INT64,
            true,
        ),
        column("outcome", ColumnType::COLUMN_TYPE_STRING, false),
        column("size_bytes", ColumnType::COLUMN_TYPE_INT64, true),
        column("sha256", ColumnType::COLUMN_TYPE_STRING, true),
        column("uri", ColumnType::COLUMN_TYPE_STRING, true),
        column("summary", ColumnType::COLUMN_TYPE_STRING, true).with_trigram_index(),
        column("attributes", ColumnType::COLUMN_TYPE_MAP, false),
    ]);
    Schema::new(columns, "event_ts_unix_nano")
}

fn batch_schema() -> Schema {
    Schema::new(
        vec![
            column("schema_version", ColumnType::COLUMN_TYPE_INT32, false),
            column("catalog_version", ColumnType::COLUMN_TYPE_STRING, false),
            column("batch_id", ColumnType::COLUMN_TYPE_STRING, false),
            column("payload_sha256", ColumnType::COLUMN_TYPE_STRING, false),
            column("record_count", ColumnType::COLUMN_TYPE_INT32, false),
            column(
                "received_ts_unix_nano",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            ),
            column("content_type", ColumnType::COLUMN_TYPE_STRING, false),
            column("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        ],
        "received_ts_unix_nano",
    )
}

fn run_schema() -> Schema {
    Schema::new(
        vec![
            column("schema_version", ColumnType::COLUMN_TYPE_INT32, false),
            column("root_run_uid", ColumnType::COLUMN_TYPE_STRING, false),
            column("event_ts_unix_nano", ColumnType::COLUMN_TYPE_INT64, false),
            column("status", ColumnType::COLUMN_TYPE_STRING, false),
            column("owner", ColumnType::COLUMN_TYPE_STRING, true),
            column("repository", ColumnType::COLUMN_TYPE_STRING, true),
            column("git_revision", ColumnType::COLUMN_TYPE_STRING, true),
            column("model_id", ColumnType::COLUMN_TYPE_STRING, true),
            column("model_revision", ColumnType::COLUMN_TYPE_STRING, true),
            column("aliases", ColumnType::COLUMN_TYPE_MAP, false),
            column("links", ColumnType::COLUMN_TYPE_MAP, false),
            column("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        ],
        "event_ts_unix_nano",
    )
}

fn entity_link_schema() -> Schema {
    Schema::new(
        vec![
            column("schema_version", ColumnType::COLUMN_TYPE_INT32, false),
            column("link_id", ColumnType::COLUMN_TYPE_STRING, false),
            column("root_run_uid", ColumnType::COLUMN_TYPE_STRING, true),
            column("source_authority", ColumnType::COLUMN_TYPE_STRING, false),
            column("source_type", ColumnType::COLUMN_TYPE_STRING, false),
            column("source_id", ColumnType::COLUMN_TYPE_STRING, false),
            column("target_authority", ColumnType::COLUMN_TYPE_STRING, false),
            column("target_type", ColumnType::COLUMN_TYPE_STRING, false),
            column("target_id", ColumnType::COLUMN_TYPE_STRING, false),
            column(
                "valid_from_ts_unix_nano",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            ),
            column("valid_to_ts_unix_nano", ColumnType::COLUMN_TYPE_INT64, true),
            column("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        ],
        "valid_from_ts_unix_nano",
    )
}

fn derived_schema(kind_column: &str) -> Schema {
    Schema::new(
        vec![
            column("schema_version", ColumnType::COLUMN_TYPE_INT32, false),
            column("record_id", ColumnType::COLUMN_TYPE_STRING, false),
            column("event_ts_unix_nano", ColumnType::COLUMN_TYPE_INT64, false),
            column(kind_column, ColumnType::COLUMN_TYPE_STRING, false),
            column("root_run_uid", ColumnType::COLUMN_TYPE_STRING, true),
            column("entity_uid", ColumnType::COLUMN_TYPE_STRING, true),
            column("name", ColumnType::COLUMN_TYPE_STRING, true),
            column("value", ColumnType::COLUMN_TYPE_FLOAT64, true),
            column("status", ColumnType::COLUMN_TYPE_STRING, true),
            column("complete", ColumnType::COLUMN_TYPE_BOOL, false),
            column("input_watermark", ColumnType::COLUMN_TYPE_INT64, true),
            column("attributes", ColumnType::COLUMN_TYPE_MAP, false),
            column("cluster", ColumnType::COLUMN_TYPE_STRING, true),
        ],
        "event_ts_unix_nano",
    )
}

pub fn approved_namespace_schemas() -> Vec<(&'static str, Schema)> {
    vec![
        (WORKLOAD_METRIC_NAMESPACE, metric_schema()),
        (SERVICE_METRIC_NAMESPACE, metric_schema()),
        (HARDWARE_METRIC_NAMESPACE, metric_schema()),
        (EVENT_NAMESPACE, event_schema()),
        (ARTIFACT_NAMESPACE, artifact_schema()),
        (RUN_NAMESPACE, run_schema()),
        (ENTITY_LINK_NAMESPACE, entity_link_schema()),
        (METRIC_ROLLUP_NAMESPACE, derived_schema("window")),
        (ENTITY_STATE_NAMESPACE, derived_schema("state_kind")),
        (ALERT_STATE_NAMESPACE, derived_schema("detector")),
        (BATCH_INTENT_NAMESPACE, batch_schema()),
        (BATCH_NAMESPACE, batch_schema()),
    ]
}

pub fn schema_for_namespace(namespace: &str) -> Option<Schema> {
    approved_namespace_schemas()
        .into_iter()
        .find_map(|(name, schema)| (name == namespace).then_some(schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::schema::resolve_key_column;

    #[test]
    fn catalog_routes_only_to_approved_schemas() {
        let catalog = catalog();
        assert_eq!(
            catalog
                .metric("telemetry.runtime", "losses")
                .unwrap()
                .namespace,
            SERVICE_METRIC_NAMESPACE
        );
        assert_eq!(
            catalog.event("telemetry.runtime.gap").unwrap().namespace,
            EVENT_NAMESPACE
        );
    }

    #[test]
    fn approved_schemas_have_unique_columns_and_valid_keys() {
        for (namespace, schema) in approved_namespace_schemas() {
            let names: BTreeSet<&str> = schema.column_names().into_iter().collect();
            assert_eq!(
                names.len(),
                schema.columns.len(),
                "{namespace} has duplicate columns"
            );
            resolve_key_column(&schema).unwrap();
        }
    }

    #[test]
    fn schemas_keep_authority_device_and_probe_identity_typed() {
        let hardware = schema_for_namespace(HARDWARE_METRIC_NAMESPACE).unwrap();
        let events = schema_for_namespace(EVENT_NAMESPACE).unwrap();
        for name in ["entity_authority", "entity_type", "entity_uid"] {
            assert!(
                hardware.column(name).is_some(),
                "missing resource field {name}"
            );
        }
        for name in ["device_uid", "device_type"] {
            assert!(
                hardware.column(name).is_some(),
                "missing hardware field {name}"
            );
        }
        assert!(events.column("probe_status").is_some());
    }
}
