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

pub(crate) trait IngestionLayoutPolicy: Sync {
    /// Partition a complete batch into logical and physical destinations.
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError>;
}

#[derive(Debug)]
pub(crate) struct IdentityLayoutPolicy;

pub(crate) const IDENTITY_LAYOUT_POLICY: IdentityLayoutPolicy = IdentityLayoutPolicy;

impl IngestionLayoutPolicy for IdentityLayoutPolicy {
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

pub(crate) struct IngestionPolicyRegistration {
    namespace_matches: fn(&str) -> bool,
    policy: &'static dyn IngestionLayoutPolicy,
}

impl IngestionPolicyRegistration {
    pub(crate) const fn new(
        namespace_matches: fn(&str) -> bool,
        policy: &'static dyn IngestionLayoutPolicy,
    ) -> Self {
        Self {
            namespace_matches,
            policy,
        }
    }
}

pub(crate) struct IngestionPolicyRegistry {
    registrations: &'static [IngestionPolicyRegistration],
    default_policy: &'static dyn IngestionLayoutPolicy,
}

impl IngestionPolicyRegistry {
    pub(crate) const fn new(
        registrations: &'static [IngestionPolicyRegistration],
        default_policy: &'static dyn IngestionLayoutPolicy,
    ) -> Self {
        Self {
            registrations,
            default_policy,
        }
    }

    pub(crate) fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        let policy = self
            .registrations
            .iter()
            .find_map(|registration| {
                (registration.namespace_matches)(source.namespace()).then_some(registration.policy)
            })
            .unwrap_or(self.default_policy);
        policy.route_batch(source, batch)
    }
}
