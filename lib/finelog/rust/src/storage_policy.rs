//! Programmable storage policy definitions selected by namespace.

use crate::errors::StatsError;
use crate::store::policy::StoragePolicy;

pub(crate) trait NamespaceStoragePolicy: Sync {
    /// Resolve the persisted retention policy for one physical namespace.
    fn storage_policy(&self, namespace: &str) -> Result<StoragePolicy, StatsError>;

    /// Physical namespaces that the server must register before first use.
    fn eager_namespaces(&self) -> Vec<&str> {
        Vec::new()
    }
}

#[derive(Debug)]
pub(crate) struct DefaultNamespaceStoragePolicy;

pub(crate) const DEFAULT_NAMESPACE_STORAGE_POLICY: DefaultNamespaceStoragePolicy =
    DefaultNamespaceStoragePolicy;

impl NamespaceStoragePolicy for DefaultNamespaceStoragePolicy {
    fn storage_policy(&self, _namespace: &str) -> Result<StoragePolicy, StatsError> {
        Ok(StoragePolicy::default())
    }
}
