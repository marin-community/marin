//! Interface for publishing and loading table catalogs.

use async_trait::async_trait;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogHead, NamespaceCatalog};
use crate::store::object_store::ObjectVersion;

#[derive(Debug, Clone)]
pub struct CatalogSnapshot {
    pub head: CatalogHead,
    pub catalog: NamespaceCatalog,
    pub(crate) head_version: ObjectVersion,
}

/// Published-catalog boundary used by store orchestration and table tasks.
#[async_trait]
pub trait PublishedCatalog: Send + Sync {
    async fn load(&self, table: &str) -> Result<Option<CatalogSnapshot>, StatsError>;

    async fn claim_writer(
        &self,
        table: &str,
        writer_epoch: u64,
        snapshot: &CatalogSnapshot,
    ) -> Result<CatalogSnapshot, StatsError>;

    async fn publish(
        &self,
        table: &str,
        catalog: NamespaceCatalog,
        writer_epoch: u64,
    ) -> Result<CatalogSnapshot, StatsError>;

    async fn delete_head(&self, table: &str) -> Result<(), StatsError>;

    async fn gc_obsolete_catalogs(
        &self,
        table: &str,
        now_ms: i64,
        catalog_retention_ms: u64,
        orphan_grace_ms: u64,
        writer_epoch: u64,
    ) -> Result<usize, StatsError>;
}
