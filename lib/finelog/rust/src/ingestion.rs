//! Registered ingestion layout policies.

use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;
use crate::ingestion_policy::{
    IngestionBatchSource, IngestionPolicyRegistration, IngestionPolicyRegistry,
    RoutedIngestionBatch, IDENTITY_LAYOUT_POLICY,
};
use crate::telemetry_policy::{matches_telemetry_namespace, TELEMETRY_LAYOUT_POLICY};

const POLICY_REGISTRATIONS: [IngestionPolicyRegistration; 1] = [IngestionPolicyRegistration::new(
    matches_telemetry_namespace,
    &TELEMETRY_LAYOUT_POLICY,
)];

pub(crate) const INGESTION_POLICY_REGISTRY: IngestionPolicyRegistry =
    IngestionPolicyRegistry::new(&POLICY_REGISTRATIONS, &IDENTITY_LAYOUT_POLICY);

pub(crate) fn route_ingestion_batch(
    source: IngestionBatchSource<'_>,
    batch: &RecordBatch,
) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
    INGESTION_POLICY_REGISTRY.route_batch(source, batch)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    #[test]
    fn unregistered_schema_uses_identity_layout() {
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
    }
}
