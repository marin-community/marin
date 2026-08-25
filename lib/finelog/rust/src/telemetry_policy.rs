//! Stable telemetry stream names and their server-owned storage layout.

use std::collections::BTreeMap;

use arrow::array::{Array, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;
use crate::ingestion_policy::{
    IngestionBatchSource, IngestionDestination, IngestionPolicy, RoutedIngestionBatch,
};
use crate::storage_policy::NamespaceStoragePolicy;
use crate::store::namespace_name::MAX_NAMESPACE_NAME_BYTES;
use crate::store::policy::StoragePolicy;

pub(crate) const TELEMETRY_NAMESPACE: &str = "telemetry_v1";
const SEMANTIC_NAMESPACE_PREFIX: &str = "telemetry_v1.";
const STORAGE_NAMESPACE_PREFIX: &str = "telemetry_storage_v1.";
const GIBIBYTE: i64 = 1024 * 1024 * 1024;
const DEFAULT_STREAM_MAX_BYTES: i64 = 2 * GIBIBYTE;
const IRIS_CONTROLLER_SERVICE: &str = "iris-controller";
pub(crate) const LEVANTER_NAMESPACE: &str = "telemetry_v1.levanter";
pub(crate) const NODE_AGENT_NAMESPACE: &str = "telemetry_v1.node_agent";
pub(crate) const IRIS_RPC_NAMESPACE: &str = "telemetry_v1.iris.rpc";
pub(crate) const IRIS_NAMESPACE: &str = "telemetry_v1.iris";
pub(crate) const VLLM_NAMESPACE: &str = "telemetry_v1.vllm";
pub(crate) const ZEPHYR_NAMESPACE: &str = "telemetry_v1.zephyr";
pub(crate) const LEVANTER_STATUS_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.levanter.status";
pub(crate) const LEVANTER_DETAIL_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.levanter.detail";
pub(crate) const NODE_AGENT_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.node_agent";
pub(crate) const IRIS_RPC_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.iris_rpc";
pub(crate) const VLLM_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.vllm";

#[derive(Clone, Copy, Debug)]
pub(crate) struct TelemetryPolicy {
    logical_inference_rules: &'static [LogicalInferenceRule],
    storage_shards: &'static [TelemetryStorageShard],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TelemetryStorageShard {
    logical_namespace: &'static str,
    storage_namespace: &'static str,
    max_bytes: i64,
    predicate: StoragePredicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StoragePredicate {
    All,
    LevanterStatus,
    LevanterDetail,
}

const TELEMETRY_STORAGE_SHARDS: [TelemetryStorageShard; 5] = [
    TelemetryStorageShard {
        logical_namespace: LEVANTER_NAMESPACE,
        storage_namespace: LEVANTER_STATUS_STORAGE_NAMESPACE,
        max_bytes: 22 * GIBIBYTE,
        predicate: StoragePredicate::LevanterStatus,
    },
    TelemetryStorageShard {
        logical_namespace: LEVANTER_NAMESPACE,
        storage_namespace: LEVANTER_DETAIL_STORAGE_NAMESPACE,
        max_bytes: 10 * GIBIBYTE,
        predicate: StoragePredicate::LevanterDetail,
    },
    TelemetryStorageShard {
        logical_namespace: NODE_AGENT_NAMESPACE,
        storage_namespace: NODE_AGENT_STORAGE_NAMESPACE,
        max_bytes: 15 * GIBIBYTE,
        predicate: StoragePredicate::All,
    },
    TelemetryStorageShard {
        logical_namespace: IRIS_RPC_NAMESPACE,
        storage_namespace: IRIS_RPC_STORAGE_NAMESPACE,
        max_bytes: GIBIBYTE,
        predicate: StoragePredicate::All,
    },
    TelemetryStorageShard {
        logical_namespace: VLLM_NAMESPACE,
        storage_namespace: VLLM_STORAGE_NAMESPACE,
        max_bytes: 2 * GIBIBYTE,
        predicate: StoragePredicate::All,
    },
];

const LEVANTER_STATUS_NAMES: [&str; 5] = [
    "global_step",
    "phase",
    "progress_time_seconds",
    "step",
    "train_loss",
];

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
        service: "levanter",
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
    storage_shards: &TELEMETRY_STORAGE_SHARDS,
};

impl IngestionPolicy for TelemetryPolicy {
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        let mut partitions: BTreeMap<IngestionDestination, Vec<u32>> = BTreeMap::new();
        for row_index in 0..batch.num_rows() {
            let record = TelemetryRecord { batch, row_index };
            let destination = self.destination(source, &record)?;
            partitions
                .entry(destination)
                .or_default()
                .push(row_index as u32);
        }

        partitions
            .into_iter()
            .map(|(destination, row_indices)| {
                let indices = UInt32Array::from(row_indices);
                let columns = batch
                    .columns()
                    .iter()
                    .map(|column| take(column.as_ref(), &indices, None))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| {
                        StatsError::Internal(format!("partition telemetry batch: {error}"))
                    })?;
                let batch = RecordBatch::try_new(batch.schema(), columns).map_err(|error| {
                    StatsError::Internal(format!("build telemetry partition: {error}"))
                })?;
                Ok(RoutedIngestionBatch { destination, batch })
            })
            .collect()
    }
}

impl TelemetryPolicy {
    fn destination(
        &self,
        source: IngestionBatchSource<'_>,
        record: &TelemetryRecord<'_>,
    ) -> Result<IngestionDestination, StatsError> {
        let requested_namespace = source.namespace();
        if matches!(source, IngestionBatchSource::Stored(_))
            && requested_namespace.starts_with(STORAGE_NAMESPACE_PREFIX)
        {
            let logical_namespace = logical_namespace_for_storage(requested_namespace)
                .ok_or_else(|| namespace_error(requested_namespace))?;
            return Ok(IngestionDestination {
                logical_namespace: logical_namespace.to_string(),
                physical_namespace: requested_namespace.to_string(),
            });
        }

        let logical_namespace = if requested_namespace == TELEMETRY_NAMESPACE {
            self.infer_logical_namespace(record)?
        } else if is_semantic_namespace(requested_namespace) {
            requested_namespace.to_string()
        } else {
            return Err(namespace_error(requested_namespace));
        };
        if !is_semantic_namespace(&logical_namespace) {
            return Err(namespace_error(requested_namespace));
        }
        let physical_namespace = self.storage_namespace(&logical_namespace, record)?;
        Ok(IngestionDestination {
            logical_namespace,
            physical_namespace,
        })
    }

    /// Classify a row written through the unscoped root namespace.
    ///
    /// The root carries no client-selected scope, so the owning service is the
    /// primary semantic boundary. Iris controller rows are narrower: only the
    /// native RPC and proxy metric families belong to `iris.rpc`.
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
            normalized_scope_component(service)
        ))
    }

    fn storage_namespace(
        &self,
        logical_namespace: &str,
        record: &TelemetryRecord<'_>,
    ) -> Result<String, StatsError> {
        for shard in self
            .storage_shards
            .iter()
            .filter(|shard| shard.logical_namespace == logical_namespace)
        {
            if shard.predicate.matches(record)? {
                return Ok(shard.storage_namespace.to_string());
            }
        }
        is_semantic_namespace(logical_namespace)
            .then(|| logical_namespace.to_string())
            .ok_or_else(|| namespace_error(logical_namespace))
    }
}

impl NamespaceStoragePolicy for TelemetryPolicy {
    fn storage_policy(&self, namespace: &str) -> Result<StoragePolicy, StatsError> {
        let max_bytes = if namespace == TELEMETRY_NAMESPACE {
            50 * GIBIBYTE
        } else if let Some(max_bytes) = self
            .storage_shards
            .iter()
            .find_map(|shard| (shard.storage_namespace == namespace).then_some(shard.max_bytes))
        {
            max_bytes
        } else if is_semantic_namespace(namespace) {
            DEFAULT_STREAM_MAX_BYTES
        } else {
            return Err(namespace_error(namespace));
        };
        Ok(StoragePolicy {
            max_bytes: Some(max_bytes),
            ..StoragePolicy::default()
        })
    }

    fn eager_namespaces(&self) -> Vec<&str> {
        self.storage_shards
            .iter()
            .map(|shard| shard.storage_namespace)
            .collect()
    }
}

impl StoragePredicate {
    fn matches(self, record: &TelemetryRecord<'_>) -> Result<bool, StatsError> {
        Ok(match self {
            Self::All => true,
            Self::LevanterStatus | Self::LevanterDetail => {
                let name = record.required_string("name")?;
                let is_status = record.required_string("kind")? != "histogram"
                    && LEVANTER_STATUS_NAMES.contains(&name);
                matches!(self, Self::LevanterStatus) == is_status
            }
        })
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
}

fn namespace_error(namespace: &str) -> StatsError {
    StatsError::SchemaValidation(format!(
        "namespace {namespace:?} is not present in the telemetry stream policy"
    ))
}

fn normalized_scope_component(service: &str) -> String {
    const MAX_SCOPE_BYTES: usize = MAX_NAMESPACE_NAME_BYTES - SEMANTIC_NAMESPACE_PREFIX.len();

    let mut normalized = String::with_capacity(service.len().min(MAX_SCOPE_BYTES));
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
        if normalized.len() + character.len_utf8() > MAX_SCOPE_BYTES {
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
        normalized.truncate(MAX_SCOPE_BYTES);
    }
    normalized
}

pub(crate) fn logical_namespace_for_storage(storage_namespace: &str) -> Option<&'static str> {
    TELEMETRY_STORAGE_SHARDS.iter().find_map(|shard| {
        (shard.storage_namespace == storage_namespace).then_some(shard.logical_namespace)
    })
}

pub(crate) fn is_forwarded_telemetry_namespace(namespace: &str) -> bool {
    namespace == TELEMETRY_NAMESPACE
        || is_semantic_namespace(namespace)
        || logical_namespace_for_storage(namespace).is_some()
}

pub(crate) fn matches_telemetry_namespace(namespace: &str) -> bool {
    namespace == TELEMETRY_NAMESPACE
        || namespace.starts_with(SEMANTIC_NAMESPACE_PREFIX)
        || namespace.starts_with(STORAGE_NAMESPACE_PREFIX)
}

fn is_semantic_namespace(namespace: &str) -> bool {
    let Some(scope) = namespace.strip_prefix(SEMANTIC_NAMESPACE_PREFIX) else {
        return false;
    };
    if namespace.len() > MAX_NAMESPACE_NAME_BYTES {
        return false;
    }
    scope.split('.').all(|component| {
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

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn telemetry_batch(services: &[&str], kinds: &[&str], names: &[&str]) -> RecordBatch {
        assert_eq!(services.len(), kinds.len());
        assert_eq!(services.len(), names.len());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("service", DataType::Utf8, false),
                Field::new("kind", DataType::Utf8, false),
                Field::new("name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(services.to_vec())),
                Arc::new(StringArray::from(kinds.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap()
    }

    fn destinations(
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Vec<(IngestionDestination, usize)> {
        TELEMETRY_POLICY
            .route_batch(source, batch)
            .unwrap()
            .into_iter()
            .map(|partition| (partition.destination, partition.batch.num_rows()))
            .collect()
    }

    #[test]
    fn root_batch_infers_logical_and_physical_destinations_from_rows() {
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
        assert!(routed.contains(&(
            IngestionDestination {
                logical_namespace: LEVANTER_NAMESPACE.to_string(),
                physical_namespace: LEVANTER_STATUS_STORAGE_NAMESPACE.to_string(),
            },
            1,
        )));
        assert!(routed.contains(&(
            IngestionDestination {
                logical_namespace: IRIS_RPC_NAMESPACE.to_string(),
                physical_namespace: IRIS_RPC_STORAGE_NAMESPACE.to_string(),
            },
            1,
        )));
        assert!(routed.contains(&(
            IngestionDestination {
                logical_namespace: "telemetry_v1.rigging".to_string(),
                physical_namespace: "telemetry_v1.rigging".to_string(),
            },
            1,
        )));
        assert!(routed.contains(&(
            IngestionDestination {
                logical_namespace: ZEPHYR_NAMESPACE.to_string(),
                physical_namespace: ZEPHYR_NAMESPACE.to_string(),
            },
            1,
        )));
    }

    #[test]
    fn explicit_semantic_namespaces_are_authoritative() {
        let batch = telemetry_batch(&["levanter"], &["gauge"], &["train_loss"]);
        assert_eq!(
            destinations(
                IngestionBatchSource::Declared("telemetry_v1.rigging.scheduler"),
                &batch,
            )[0]
            .0,
            IngestionDestination {
                logical_namespace: "telemetry_v1.rigging.scheduler".to_string(),
                physical_namespace: "telemetry_v1.rigging.scheduler".to_string(),
            }
        );
        assert_eq!(
            destinations(IngestionBatchSource::Declared(LEVANTER_NAMESPACE), &batch)[0].0,
            IngestionDestination {
                logical_namespace: LEVANTER_NAMESPACE.to_string(),
                physical_namespace: LEVANTER_STATUS_STORAGE_NAMESPACE.to_string(),
            }
        );
    }

    #[test]
    fn physical_cleavage_can_use_multiple_record_columns() {
        let batch = telemetry_batch(
            &["levanter", "levanter"],
            &["gauge", "histogram"],
            &["train_loss", "train_loss"],
        );
        let routed = destinations(IngestionBatchSource::Declared(LEVANTER_NAMESPACE), &batch);
        assert_eq!(routed.len(), 2);
        assert!(routed.iter().any(|(destination, rows)| {
            destination.physical_namespace == LEVANTER_STATUS_STORAGE_NAMESPACE && *rows == 1
        }));
        assert!(routed.iter().any(|(destination, rows)| {
            destination.physical_namespace == LEVANTER_DETAIL_STORAGE_NAMESPACE && *rows == 1
        }));
    }

    #[test]
    fn stored_batches_preserve_physical_layout() {
        let batch = telemetry_batch(&["levanter"], &["gauge"], &["train_loss"]);
        assert_eq!(
            destinations(
                IngestionBatchSource::Stored(LEVANTER_DETAIL_STORAGE_NAMESPACE),
                &batch,
            )[0]
            .0,
            IngestionDestination {
                logical_namespace: LEVANTER_NAMESPACE.to_string(),
                physical_namespace: LEVANTER_DETAIL_STORAGE_NAMESPACE.to_string(),
            }
        );
    }

    #[test]
    fn root_batch_derives_a_valid_scope_from_an_unmapped_service() {
        let batch = telemetry_batch(&["123/Custom Service"], &["gauge"], &["queue_depth"]);
        let destination = destinations(IngestionBatchSource::Declared(TELEMETRY_NAMESPACE), &batch)
            [0]
        .0
        .clone();
        assert_eq!(
            destination.logical_namespace,
            "telemetry_v1.service_123_custom_service"
        );
        assert_eq!(
            destination.physical_namespace,
            destination.logical_namespace
        );
    }
}
