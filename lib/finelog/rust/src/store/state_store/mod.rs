//! Durable table-state boundary.
//!
//! [`object::ObjectTableStateStore`] is the only authority for an
//! object-backed table's durable state. It enumerates heads, loads the
//! selected state without touching data objects, hands a writer its fence,
//! commits a complete next state under that fence, and publishes a tombstone
//! revision when the table is deleted.
//!
//! Every mutating operation is fenced. A commit presents the HEAD version it
//! observed plus its [`WriterFence`]; a HEAD that records a different fence
//! rejects it, so a stale process cannot advance a table another process has
//! claimed. Legacy tables have no durable state store: their authority is the
//! local SQLite catalog, whose writer holds the data-dir flock for the whole
//! process lifetime.

pub mod object;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogHead, NamespaceCatalog};
use crate::store::object_store::ObjectVersion;
use crate::store::table_state::{TableRevision, WriterFence};

/// One table's durable state as the store selected it, plus the HEAD CAS
/// token proving the selection is still current.
#[derive(Debug, Clone)]
pub struct StoredTableState {
    pub head: CatalogHead,
    pub catalog: NamespaceCatalog,
    pub(crate) head_version: ObjectVersion,
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
}

/// One table head as enumeration sees it, live or tombstoned.
#[derive(Debug, Clone)]
pub struct TableHead {
    pub table: String,
    pub revision: TableRevision,
    pub fence: WriterFence,
    pub tombstoned: bool,
}

/// The error a store raises when HEAD belongs to another writer.
pub fn fenced_error(table: &str, fence: WriterFence, owner: WriterFence) -> StatsError {
    StatsError::SchemaConflict(format!(
        "table {table:?} is owned by writer {owner}, fencing writer {fence}"
    ))
}
