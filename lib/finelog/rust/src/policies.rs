//! Ordered registration of programmable schema policies.

use arrow::record_batch::RecordBatch;
use std::sync::Mutex;

use crate::errors::StatsError;
use crate::ingestion_policy::{
    IngestionBatchSource, IngestionPolicy, IngestionState, RoutedIngestionBatch,
    IDENTITY_INGESTION_POLICY,
};
use crate::levanter_metrics_policy::{
    levanter_metrics_schema, matches_levanter_metrics_namespace, LevanterMetricsPolicy,
    LEVANTER_METRICS_POLICY,
};
use crate::partition_policy::PhysicalPartitionPolicy;
use crate::storage_policy::{NamespaceStoragePolicy, DEFAULT_NAMESPACE_STORAGE_POLICY};
use crate::store::policy::StoragePolicy;
use crate::telemetry_policy::{
    matches_telemetry_namespace, TelemetryPolicy, TelemetryRootWriteMode, TELEMETRY_POLICY,
};

struct PolicyRule<P: ?Sized + 'static> {
    matches: fn(&str) -> bool,
    policy: &'static P,
}

impl<P: ?Sized + 'static> PolicyRule<P> {
    const fn new(matches: fn(&str) -> bool, policy: &'static P) -> Self {
        Self { matches, policy }
    }
}

trait SchemaPolicy: IngestionPolicy + NamespaceStoragePolicy {
    fn registration_namespace(&self, namespace: &str) -> Result<String, StatsError> {
        Ok(namespace.to_string())
    }

    fn physical_partition_policy(
        &self,
        namespace: &str,
    ) -> Option<&'static dyn PhysicalPartitionPolicy>;
}

// Rules are evaluated in declaration order. Each rule owns the complete policy
// for a namespace family: logical routing, retention, and hidden partitioning.
static SCHEMA_POLICIES: [PolicyRule<dyn SchemaPolicy>; 2] = [
    PolicyRule::new(matches_levanter_metrics_namespace, &LEVANTER_METRICS_POLICY),
    PolicyRule::new(matches_telemetry_namespace, &TELEMETRY_POLICY),
];

fn matching_policy<P: ?Sized + 'static>(
    rules: &'static [PolicyRule<P>],
    namespace: &str,
) -> Option<&'static P> {
    rules
        .iter()
        .find(|rule| (rule.matches)(namespace))
        .map(|rule| rule.policy)
}

#[derive(Debug)]
pub(crate) struct PolicyRegistry {
    ingestion_state: Mutex<IngestionState>,
    telemetry_policy: TelemetryPolicy,
}

impl PolicyRegistry {
    pub(crate) fn new(telemetry_root_write_mode: TelemetryRootWriteMode) -> Self {
        Self {
            ingestion_state: Mutex::new(IngestionState::default()),
            telemetry_policy: TelemetryPolicy::with_root_write_mode(telemetry_root_write_mode),
        }
    }

    pub(crate) fn route_ingestion_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        let mut state = self.ingestion_state.lock().unwrap();
        if matches_telemetry_namespace(source.namespace()) {
            return self.telemetry_policy.route_batch(source, batch, &mut state);
        }
        match matching_policy(&SCHEMA_POLICIES, source.namespace()) {
            Some(policy) => policy.route_batch(source, batch, &mut state),
            None => IDENTITY_INGESTION_POLICY.route_batch(source, batch, &mut state),
        }
    }

    pub(crate) fn index_migration_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<(), StatsError> {
        let mut state = self.ingestion_state.lock().unwrap();
        if matches_telemetry_namespace(source.namespace()) {
            return self
                .telemetry_policy
                .index_migration_batch(source, batch, &mut state);
        }
        match matching_policy(&SCHEMA_POLICIES, source.namespace()) {
            Some(policy) => policy.index_migration_batch(source, batch, &mut state),
            None => IDENTITY_INGESTION_POLICY.index_migration_batch(source, batch, &mut state),
        }
    }

    pub(crate) fn finish_migration_index(&self) {
        self.ingestion_state
            .lock()
            .unwrap()
            .finish_migration_index();
    }
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::new(TelemetryRootWriteMode::SemanticOnly)
    }
}

pub(crate) fn storage_policy_for(namespace: &str) -> Result<StoragePolicy, StatsError> {
    match matching_policy(&SCHEMA_POLICIES, namespace) {
        Some(policy) => policy.storage_policy(namespace),
        None => DEFAULT_NAMESPACE_STORAGE_POLICY.storage_policy(namespace),
    }
}

pub(crate) fn managed_storage_policy_for(
    namespace: &str,
) -> Result<Option<StoragePolicy>, StatsError> {
    matching_policy(&SCHEMA_POLICIES, namespace)
        .map(|policy| policy.storage_policy(namespace))
        .transpose()
}

pub(crate) fn registration_namespace_for(namespace: &str) -> Result<String, StatsError> {
    match matching_policy(&SCHEMA_POLICIES, namespace) {
        Some(policy) => policy.registration_namespace(namespace),
        None => Ok(namespace.to_string()),
    }
}

pub(crate) fn schema_for_namespace(namespace: &str) -> Option<crate::store::schema::Schema> {
    if matches_levanter_metrics_namespace(namespace) {
        return Some(levanter_metrics_schema());
    }
    if matches_telemetry_namespace(namespace) {
        return Some(crate::server::telemetry::telemetry_schema());
    }
    None
}

pub(crate) fn physical_partition_policy_for(
    namespace: &str,
) -> Option<&'static dyn PhysicalPartitionPolicy> {
    matching_policy(&SCHEMA_POLICIES, namespace)
        .and_then(|policy| policy.physical_partition_policy(namespace))
}

pub(crate) fn eager_storage_namespaces_for(namespace: &str) -> Vec<&'static str> {
    match matching_policy(&SCHEMA_POLICIES, namespace) {
        Some(policy) => policy.eager_namespaces(),
        None => DEFAULT_NAMESPACE_STORAGE_POLICY.eager_namespaces(),
    }
}

impl SchemaPolicy for crate::telemetry_policy::TelemetryPolicy {
    fn physical_partition_policy(
        &self,
        namespace: &str,
    ) -> Option<&'static dyn PhysicalPartitionPolicy> {
        self.physical_partition_policy(namespace)
    }
}

impl SchemaPolicy for LevanterMetricsPolicy {
    fn registration_namespace(&self, namespace: &str) -> Result<String, StatsError> {
        self.registration_namespace(namespace)
    }

    fn physical_partition_policy(
        &self,
        namespace: &str,
    ) -> Option<&'static dyn PhysicalPartitionPolicy> {
        self.physical_partition_policy(namespace)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn unregistered_schema_uses_identity_ingestion_and_default_storage() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![1, 2]))],
        )
        .unwrap();

        let routed = PolicyRegistry::default()
            .route_ingestion_batch(IngestionBatchSource::Declared("experiments"), &batch)
            .unwrap();
        assert_eq!(routed.len(), 1);
        assert_eq!(routed[0].destination.logical_namespace, "experiments");
        assert_eq!(routed[0].batch, batch);
        assert_eq!(
            storage_policy_for("experiments").unwrap(),
            StoragePolicy::default()
        );
        assert!(eager_storage_namespaces_for("experiments").is_empty());
    }
}
