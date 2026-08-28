//! Transition scaffolding: the durable state boundary for legacy tables.
//!
//! A legacy (non-object-backed) table's single authority is the local SQLite
//! catalog. It has one logical revision — `table_heads.catalog_generation` —
//! advanced by the same transaction that writes its row diffs, so this store
//! reports and verifies revisions rather than rewriting rows a caller already
//! committed. "No deltas" constrains the object snapshot format, not SQLite's
//! internal update plan.
//!
//! Its writer fence is the process-local writer identity backed by the
//! exclusive `.finelog-store.lock` flock on the data directory: one process at
//! a time may open the catalog, so there is no durable cross-process fence to
//! compare against. That is weaker than the object store's HEAD fence and is
//! the reason this type is temporary — it disappears with the last legacy
//! table.

use std::sync::Arc;

use async_trait::async_trait;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogHead, NamespaceCatalog};
use crate::store::catalog::object_state_store::TABLE_STATE_FORMAT_VERSION;
use crate::store::catalog::state_store::{
    fenced_error, BackendToken, StoredTableState, TableHead, TableStateStore,
};
use crate::store::catalog::Catalog;
use crate::store::table_state::{TableRevision, WriterFence};

pub struct SqliteTableStateStore {
    catalog: Arc<Catalog>,
    fence: WriterFence,
}

impl SqliteTableStateStore {
    pub fn new(catalog: Arc<Catalog>, fence: WriterFence) -> Self {
        Self { catalog, fence }
    }

    fn state(&self, table: &str) -> Result<StoredTableState, StatsError> {
        let status = self.catalog.table_spec_status(table)?;
        let stats = self.catalog.aggregate_namespace_stats(table)?;
        let catalog = NamespaceCatalog {
            format_version: Some(TABLE_STATE_FORMAT_VERSION),
            namespace: Some(table.to_string()),
            catalog_generation: Some(status.catalog_generation),
            active_table_spec_version: Some(status.active_version()),
            desired_table_spec_version: Some(status.desired_version()),
            persisted_high_water: Some(stats.max_seq),
            ..Default::default()
        };
        Ok(StoredTableState {
            head: CatalogHead {
                format_version: Some(TABLE_STATE_FORMAT_VERSION),
                namespace: Some(table.to_string()),
                writer_epoch: Some(self.fence.get()),
                catalog_generation: Some(status.catalog_generation),
                active_table_spec_version: Some(status.active_version()),
                tombstoned: Some(false),
                ..Default::default()
            },
            catalog,
            token: BackendToken::Local(TableRevision::new(status.catalog_generation)),
        })
    }
}

#[async_trait]
impl TableStateStore for SqliteTableStateStore {
    async fn list(&self) -> Result<Vec<TableHead>, StatsError> {
        self.catalog
            .list_all()?
            .into_iter()
            .map(|(table, _)| {
                let revision = self.catalog.table_spec_status(&table)?.catalog_generation;
                Ok(TableHead {
                    table,
                    revision: TableRevision::new(revision),
                    fence: self.fence,
                    // A legacy drop removes the rows outright, so no head
                    // survives its table.
                    tombstoned: false,
                })
            })
            .collect()
    }

    async fn load(&self, table: &str) -> Result<Option<StoredTableState>, StatsError> {
        if !self.catalog.contains(table) {
            return Ok(None);
        }
        Ok(Some(self.state(table)?))
    }

    /// Record this process as the writer. The data-dir flock already granted
    /// exclusivity, so the claim reads back the selected state unchanged.
    async fn claim_writer(
        &self,
        table: &str,
        fence: WriterFence,
        selected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError> {
        if fence != self.fence {
            return Err(fenced_error(table, fence, self.fence));
        }
        Ok(selected.clone())
    }

    /// Verify that `next` is the revision the caller's SQLite transaction
    /// allocated. The rows are already durable when this runs.
    async fn commit(
        &self,
        table: &str,
        fence: WriterFence,
        _expected: Option<&StoredTableState>,
        next: NamespaceCatalog,
    ) -> Result<StoredTableState, StatsError> {
        if fence != self.fence {
            return Err(fenced_error(table, fence, self.fence));
        }
        let selected = self.state(table)?;
        if selected.revision() != TableRevision::new(next.catalog_generation.unwrap_or(0)) {
            return Err(StatsError::SchemaConflict(format!(
                "table {table:?} committed revision {} but SQLite holds {}",
                next.catalog_generation.unwrap_or(0),
                selected.revision()
            )));
        }
        Ok(selected)
    }

    /// Delete the table's rows and report the deleted state.
    ///
    /// Unlike the object store, SQLite retains no durable history: the
    /// tombstone exists only for the caller that performed the drop.
    async fn tombstone(
        &self,
        table: &str,
        fence: WriterFence,
        _expected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError> {
        if fence != self.fence {
            return Err(fenced_error(table, fence, self.fence));
        }
        let mut deleted = self.state(table)?;
        self.catalog.delete(table)?;
        deleted.head.tombstoned = Some(true);
        deleted.catalog.tombstoned = Some(true);
        Ok(deleted)
    }

    async fn gc_obsolete_states(
        &self,
        _table: &str,
        _now_ms: i64,
        _state_retention_ms: u64,
        _orphan_grace_ms: u64,
        _fence: WriterFence,
    ) -> Result<usize, StatsError> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::policy::StoragePolicy;
    use crate::store::schema::{with_implicit_seq, Column, Schema};

    const TABLE: &str = "legacy.table";

    fn registered() -> (SqliteTableStateStore, Arc<Catalog>) {
        let catalog = Arc::new(Catalog::open(None).unwrap());
        let schema = with_implicit_seq(Schema::new(
            vec![Column::new(
                "timestamp_ms",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            )],
            "",
        ));
        catalog
            .register_or_evolve(TABLE, schema, StoragePolicy::default(), |existing| {
                Ok(existing.clone())
            })
            .unwrap();
        (
            SqliteTableStateStore::new(Arc::clone(&catalog), WriterFence::new(7)),
            catalog,
        )
    }

    #[tokio::test]
    async fn an_unregistered_table_has_no_state() {
        let (states, _catalog) = registered();
        assert!(states.load("absent.table").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_legacy_table_reports_one_revision_under_the_process_fence() {
        let (states, _catalog) = registered();
        let loaded = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(loaded.fence(), WriterFence::new(7));
        assert!(!loaded.is_tombstoned());
        let listed = states.list().await.unwrap();
        assert!(listed.iter().any(|head| head.table == TABLE));

        let claimed = states
            .claim_writer(TABLE, WriterFence::new(7), &loaded)
            .await
            .unwrap();
        assert_eq!(claimed.revision(), loaded.revision());
        let error = states
            .claim_writer(TABLE, WriterFence::new(8), &loaded)
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
    }

    #[tokio::test]
    async fn a_commit_must_name_the_revision_sqlite_holds() {
        let (states, _catalog) = registered();
        let loaded = states.load(TABLE).await.unwrap().unwrap();
        let mut ahead = loaded.catalog.clone();
        ahead.catalog_generation = Some(loaded.revision().get() + 1);
        let error = states
            .commit(TABLE, WriterFence::new(7), Some(&loaded), ahead)
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
        states
            .commit(
                TABLE,
                WriterFence::new(7),
                Some(&loaded),
                loaded.catalog.clone(),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_tombstone_deletes_the_rows() {
        let (states, catalog) = registered();
        let loaded = states.load(TABLE).await.unwrap().unwrap();
        // A drop unregisters the table before deleting its durable state.
        catalog.begin_drop(TABLE).unwrap();

        let deleted = states
            .tombstone(TABLE, WriterFence::new(7), &loaded)
            .await
            .unwrap();

        assert!(deleted.is_tombstoned());
        assert!(states.load(TABLE).await.unwrap().is_none());
        assert!(catalog.list_segments(TABLE).unwrap().is_empty());
    }
}
