// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! The local metadata catalog: one SQLite sidecar behind a typed interface.
//!
//! [`Catalog`] is the only live-system code that speaks SQL. Everything else
//! sees typed records and methods, split by facet:
//!
//! - [`namespaces`]: the live namespace registry (which tables exist) and the
//!   drop fence.
//! - [`table_specs`]: durable table specifications and the migration
//!   lifecycle; [`TableSpecStatus`] is what the runtime resolves policy from.
//! - [`segments`]: legacy segment rows, storage policy, and eviction — the
//!   authority for local-L0 tables.
//! - [`object_segments`]: the derived projection of an object-backed table's
//!   published remote state; the remote `TableState` is authoritative and
//!   these rows are rebuilt from it whenever the projection trails HEAD.
//! - [`cursors`]: durable forwarding cursors.
//! - [`migrations`]: the ordered schema migrations the sidecar itself runs
//!   through at open.
//!
//! Durable *table state* is a separate boundary: [`state_store`] defines the
//! `TableStateStore` trait, implemented by [`sqlite_state_store`] (legacy
//! tables — this catalog is the backing authority) and
//! [`object_state_store`] (object tables — HEAD and immutable state documents
//! in the object store, with this catalog holding only the projection that
//! [`projection`] rebuilds).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::Connection;

use crate::errors::StatsError;

mod cursors;
pub(crate) mod migrations;
mod namespaces;
mod object_segments;
pub mod object_state_store;
pub(crate) mod projection;
mod segments;
pub(crate) mod sqlite_state_store;
pub(crate) mod state_store;
mod table_specs;
#[cfg(test)]
mod tests;

pub use namespaces::RegisteredNamespace;
pub use object_segments::{ObjectSegmentRecord, PublishedObjectSegment};
pub use table_specs::TableSpecStatus;

/// Sidecar filename.
pub const CATALOG_DB_FILENAME: &str = "_finelog_catalog.sqlite";

/// Every table keyed by `namespace`, ordered so the `namespaces` row that
/// defines the namespace is removed last.
const NAMESPACE_OWNED_TABLES: [&str; 8] = [
    "segments",
    "storage_policies",
    "table_specs",
    "object_segments",
    "table_migrations",
    "table_heads",
    "forward_state",
    "namespaces",
];

struct CatalogInner {
    conn: Connection,
    live: BTreeMap<String, RegisteredNamespace>,
    /// Monotonic insertion ordinal per name; renders `list_namespaces` in
    /// registration order.
    registered_at: HashMap<String, u64>,
    next_ordinal: u64,
    dropping: HashSet<String>,
}

/// Single source of truth for namespace state, persistent and live.
pub struct Catalog {
    inner: Mutex<CatalogInner>,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn sqlite_err(e: rusqlite::Error) -> StatsError {
    StatsError::Internal(format!("catalog sqlite error: {e}"))
}

impl Catalog {
    /// Open the catalog. `data_dir = None` uses memory; otherwise the sidecar
    /// lives at `{data_dir}/_finelog_catalog.sqlite`. Initializes its schema
    /// idempotently.
    pub fn open(data_dir: Option<&Path>) -> Result<Catalog, StatsError> {
        let mut conn = match data_dir {
            None => Connection::open_in_memory().map_err(sqlite_err)?,
            Some(dir) => Connection::open(dir.join(CATALOG_DB_FILENAME)).map_err(sqlite_err)?,
        };
        let journal_mode: String = if data_dir.is_some() {
            // WAL requires shared memory and is not safe on the network-backed
            // filesystems used by Kubernetes finelog. PERSIST keeps rollback
            // journaling while avoiding a journal-file delete on every commit.
            conn.query_row("PRAGMA journal_mode = PERSIST", [], |row| row.get(0))
                .map_err(sqlite_err)?
        } else {
            conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))
                .map_err(sqlite_err)?
        };
        conn.execute_batch("PRAGMA synchronous = FULL;")
            .map_err(sqlite_err)?;
        let synchronous: i64 = conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .map_err(sqlite_err)?;
        migrations::migrate(&mut conn)?;
        tracing::info!(
            persistent = data_dir.is_some(),
            journal_mode,
            synchronous,
            "finelog catalog sqlite ready"
        );
        Ok(Catalog {
            inner: Mutex::new(CatalogInner {
                conn,
                live: BTreeMap::new(),
                registered_at: HashMap::new(),
                next_ordinal: 0,
                dropping: HashSet::new(),
            }),
        })
    }
}
