//! Logical and physical layout policies for complete ingestion batches.

use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IngestionBatchSource<'a> {
    /// A namespace selected by the writer at the ingestion boundary.
    Declared(&'a str),
    /// A namespace already attached to persisted or forwarded rows.
    Stored(&'a str),
}

impl<'a> IngestionBatchSource<'a> {
    pub(crate) fn namespace(self) -> &'a str {
        match self {
            Self::Declared(namespace) | Self::Stored(namespace) => namespace,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct IngestionDestination {
    pub logical_namespace: String,
    pub physical_namespace: String,
}

#[derive(Debug)]
pub(crate) struct RoutedIngestionBatch {
    pub destination: IngestionDestination,
    pub batch: RecordBatch,
}

pub(crate) trait IngestionPolicy: Sync {
    /// Partition a complete batch into logical and physical destinations.
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError>;
}

#[derive(Debug)]
pub(crate) struct IdentityIngestionPolicy;

pub(crate) const IDENTITY_INGESTION_POLICY: IdentityIngestionPolicy = IdentityIngestionPolicy;

impl IngestionPolicy for IdentityIngestionPolicy {
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        let namespace = source.namespace().to_string();
        Ok(vec![RoutedIngestionBatch {
            destination: IngestionDestination {
                logical_namespace: namespace.clone(),
                physical_namespace: namespace,
            },
            batch: batch.clone(),
        }])
    }
}
