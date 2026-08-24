//! Ordered registration of programmable ingestion and storage policies.

use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;
use crate::ingestion_policy::{
    IngestionBatchSource, IngestionPolicy, RoutedIngestionBatch, IDENTITY_INGESTION_POLICY,
};
use crate::storage_policy::{NamespaceStoragePolicy, DEFAULT_NAMESPACE_STORAGE_POLICY};
use crate::store::policy::StoragePolicy;
use crate::telemetry_policy::{matches_telemetry_namespace, TELEMETRY_POLICY};

struct PolicyRule<P: ?Sized + 'static> {
    matches: fn(&str) -> bool,
    policy: &'static P,
}

impl<P: ?Sized + 'static> PolicyRule<P> {
    const fn new(matches: fn(&str) -> bool, policy: &'static P) -> Self {
        Self { matches, policy }
    }
}

// Rules are evaluated in declaration order. Keep the identity/default policies
// as explicit fallbacks rather than catch-all matchers in these lists.
static INGESTION_POLICIES: [PolicyRule<dyn IngestionPolicy>; 1] = [PolicyRule::new(
    matches_telemetry_namespace,
    &TELEMETRY_POLICY,
)];

static STORAGE_POLICIES: [PolicyRule<dyn NamespaceStoragePolicy>; 1] = [PolicyRule::new(
    matches_telemetry_namespace,
    &TELEMETRY_POLICY,
)];

fn matching_policy<P: ?Sized + 'static>(
    rules: &'static [PolicyRule<P>],
    namespace: &str,
) -> Option<&'static P> {
    rules
        .iter()
        .find(|rule| (rule.matches)(namespace))
        .map(|rule| rule.policy)
}

pub(crate) fn route_ingestion_batch(
    source: IngestionBatchSource<'_>,
    batch: &RecordBatch,
) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
    matching_policy(&INGESTION_POLICIES, source.namespace())
        .unwrap_or(&IDENTITY_INGESTION_POLICY)
        .route_batch(source, batch)
}

pub(crate) fn storage_policy_for(namespace: &str) -> Result<StoragePolicy, StatsError> {
    matching_policy(&STORAGE_POLICIES, namespace)
        .unwrap_or(&DEFAULT_NAMESPACE_STORAGE_POLICY)
        .storage_policy(namespace)
}

pub(crate) fn eager_storage_namespaces_for(namespace: &str) -> Vec<&'static str> {
    matching_policy(&STORAGE_POLICIES, namespace)
        .unwrap_or(&DEFAULT_NAMESPACE_STORAGE_POLICY)
        .eager_namespaces()
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

        let routed =
            route_ingestion_batch(IngestionBatchSource::Declared("experiments"), &batch).unwrap();
        assert_eq!(routed.len(), 1);
        assert_eq!(routed[0].destination.logical_namespace, "experiments");
        assert_eq!(routed[0].destination.physical_namespace, "experiments");
        assert_eq!(routed[0].batch, batch);
        assert_eq!(
            storage_policy_for("experiments").unwrap(),
            StoragePolicy::default()
        );
        assert!(eager_storage_namespaces_for("experiments").is_empty());
    }
}
