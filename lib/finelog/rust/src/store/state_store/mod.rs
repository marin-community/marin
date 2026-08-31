//! Durable table-state boundary.
//!
//! A [`TableStateStore`] is the only authority for one table's durable state.
//! It enumerates heads, loads the selected state without touching data objects,
//! hands a writer its fence, commits a complete next state under that fence,
//! and publishes a tombstone revision when the table is deleted.
//!
//! Every mutating operation is fenced. A commit presents the backend token it
//! observed plus its [`WriterFence`]; a store whose HEAD records a different
//! fence rejects it, so a stale process cannot advance a table another process
//! has claimed.

use async_trait::async_trait;

pub mod object;
pub mod sqlite;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogHead, NamespaceCatalog};
use crate::store::object_store::ObjectVersion;
use crate::store::table_state::{TableRevision, WriterFence};

/// The backend's proof that a loaded state is still current.
///
/// Opaque to callers: each store defines what its conditional write compares.
#[derive(Debug, Clone)]
pub enum BackendToken {
    /// CAS token of the HEAD object naming this state.
    Head(ObjectVersion),
    /// Revision recorded in the local catalog, whose writer holds the data-dir
    /// lock for the whole process lifetime.
    Local(TableRevision),
}

/// One table's durable state as the store selected it, plus its commit token.
#[derive(Debug, Clone)]
pub struct StoredTableState {
    pub head: CatalogHead,
    pub catalog: NamespaceCatalog,
    pub(crate) token: BackendToken,
}

impl StoredTableState {
    pub fn revision(&self) -> TableRevision {
        TableRevision::new(self.head.catalog_generation.unwrap_or(0))
    }

    pub fn fence(&self) -> WriterFence {
        WriterFence::new(self.head.writer_epoch.unwrap_or(0))
    }

    /// Whether the selected revision deleted the table.
    pub fn is_tombstoned(&self) -> bool {
        self.head.tombstoned.unwrap_or(false)
    }

    pub fn token(&self) -> &BackendToken {
        &self.token
    }

    pub(crate) fn head_version(&self) -> Option<&ObjectVersion> {
        match &self.token {
            BackendToken::Head(version) => Some(version),
            BackendToken::Local(_) => None,
        }
    }
}

/// One table head as enumeration sees it, live or tombstoned.
#[derive(Debug, Clone)]
pub struct TableHead {
    pub table: String,
    pub revision: TableRevision,
    pub fence: WriterFence,
    pub tombstoned: bool,
}

#[async_trait]
pub trait TableStateStore: Send + Sync {
    /// Every table this store has ever published, including tombstoned ones.
    async fn list(&self) -> Result<Vec<TableHead>, StatsError>;

    /// The selected state for `table`, or `None` when it was never published.
    ///
    /// Metadata only: loading never reads the table's data objects.
    async fn load(&self, table: &str) -> Result<Option<StoredTableState>, StatsError>;

    /// Take ownership of durable writes while retaining the selected state.
    async fn claim_writer(
        &self,
        table: &str,
        fence: WriterFence,
        selected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError>;

    /// Durably select `next` as the table's state.
    ///
    /// `expected` is the token this writer last observed; `None` claims the
    /// table's first revision. The commit fails when HEAD records a different
    /// fence, when `expected` is no longer current, or when `next` does not
    /// advance the selected revision.
    async fn commit(
        &self,
        table: &str,
        fence: WriterFence,
        expected: Option<&StoredTableState>,
        next: NamespaceCatalog,
    ) -> Result<StoredTableState, StatsError>;

    /// Publish a new revision marking the table deleted.
    ///
    /// Deletion is durable state, not the absence of it: a missing head means
    /// the table was never published.
    async fn tombstone(
        &self,
        table: &str,
        fence: WriterFence,
        expected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError>;

    /// Remove superseded state documents and unreferenced objects.
    async fn gc_obsolete_states(
        &self,
        table: &str,
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
        fence: WriterFence,
    ) -> Result<usize, StatsError>;
}

/// The error a store raises when HEAD belongs to another writer.
pub fn fenced_error(table: &str, fence: WriterFence, owner: WriterFence) -> StatsError {
    StatsError::SchemaConflict(format!(
        "table {table:?} is owned by writer {owner}, fencing writer {fence}"
    ))
}
