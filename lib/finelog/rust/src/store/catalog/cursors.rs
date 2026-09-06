// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Durable forwarding cursors: the last sequence number each downstream
//! target acknowledged per namespace, so a restart resumes forwarding instead
//! of replaying or reseeding.

use rusqlite::OptionalExtension;

use super::*;
use crate::errors::StatsError;
use crate::proto::finelog::stats::ForwardCursor;
use crate::store::table_state::TableRevision;

impl Catalog {
    /// The seq in `namespace` below which nothing will be sent to `target` again, or
    /// `None` if this store has never forwarded that namespace there. A high-water mark
    /// of what is settled, which is not the same as what `target` holds: a sender may
    /// give a row up rather than deliver it.
    ///
    /// Keyed by `(target, namespace)` so each table advances independently and repointing
    /// a forwarder reseeds instead of replaying one store's seq space into another's.
    pub fn forward_cursor(&self, target: &str, namespace: &str) -> Result<Option<i64>, StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .query_row(
                "SELECT cursor FROM forward_state WHERE target = ?1 AND namespace = ?2",
                [target, namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)
    }
    pub fn forward_cursors(&self, namespace: &str) -> Result<Vec<ForwardCursor>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut statement = inner
            .conn
            .prepare(
                "SELECT target, cursor FROM forward_state
                 WHERE namespace = ?1 ORDER BY target",
            )
            .map_err(sqlite_err)?;
        let rows = statement
            .query_map([namespace], |row| {
                Ok(ForwardCursor {
                    target: Some(row.get(0)?),
                    cursor: Some(row.get(1)?),
                    ..Default::default()
                })
            })
            .map_err(sqlite_err)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_err)
    }
    /// Record `cursor` as settled for `(target, namespace)`. Callers write it only once
    /// the rows below it can never be sent again, so a crash mid-batch re-forwards that
    /// batch rather than losing it.
    ///
    /// Returns the table's revision after the write. A namespace without a
    /// versioned head keeps revision zero and publishes nothing.
    pub fn set_forward_cursor(
        &self,
        target: &str,
        namespace: &str,
        cursor: i64,
    ) -> Result<TableRevision, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        transaction
            .execute(
                "INSERT INTO forward_state (target, namespace, cursor) VALUES (?1, ?2, ?3)
                 ON CONFLICT(target, namespace) DO UPDATE SET cursor = excluded.cursor",
                rusqlite::params![target, namespace, cursor],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        let generation: Option<i64> = transaction
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(TableRevision::new(generation.unwrap_or(0) as u64))
    }
}
