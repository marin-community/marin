// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Durable forwarding cursors: the last sequence number each downstream
//! target acknowledged per namespace, so a restart resumes forwarding instead
//! of replaying or reseeding.

use rusqlite::OptionalExtension;

use super::object_segments::advance_generation_in;
use super::segments::remove_segments_in;
use super::*;
use crate::errors::StatsError;
use crate::proto::finelog::stats::ForwardCursor;
use crate::store::table_state::TableRevision;

/// The logical relay spool released by one durable forwarding settlement.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ForwardingSettlement {
    pub cursor: i64,
    pub removed_paths: Vec<String>,
    pub removed_rows: i64,
    pub removed_bytes: i64,
}

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

    /// Advance one target cursor and retire every object segment it fully covers.
    ///
    /// Both effects share one SQLite transaction so recovery can never observe a
    /// cursor that promises not to resend rows while those rows remain in the live
    /// relay spool, or the inverse. The cursor is monotonic across retries.
    pub fn settle_forwarding(
        &self,
        target: &str,
        namespace: &str,
        cursor: i64,
    ) -> Result<(TableRevision, ForwardingSettlement), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let has_versioned_head = transaction
            .query_row(
                "SELECT 1 FROM table_heads WHERE namespace = ?1",
                [namespace],
                |_| Ok(()),
            )
            .optional()
            .map_err(sqlite_err)?
            .is_some();
        let previous_cursor = transaction
            .query_row(
                "SELECT cursor FROM forward_state WHERE target = ?1 AND namespace = ?2",
                rusqlite::params![target, namespace],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        let settled_cursor = previous_cursor.map_or(cursor, |previous| previous.max(cursor));
        let other_cursor = transaction
            .query_row(
                "SELECT MIN(cursor) FROM forward_state WHERE namespace = ?1 AND target != ?2",
                rusqlite::params![namespace, target],
                |row| row.get::<_, Option<i64>>(0),
            )
            .map_err(sqlite_err)?;
        let retirement_cursor = other_cursor
            .map(|other| other.min(settled_cursor))
            .unwrap_or(settled_cursor);

        let removed = {
            let mut statement = transaction
                .prepare(
                    "SELECT s.path, s.row_count, s.byte_size
                     FROM segments s
                     INNER JOIN object_segments o
                       ON o.namespace = s.namespace AND o.path = s.path
                     WHERE s.namespace = ?1 AND s.max_seq <= ?2
                     ORDER BY s.max_seq, s.path",
                )
                .map_err(sqlite_err)?;
            let rows = statement
                .query_map(rusqlite::params![namespace, retirement_cursor], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .map_err(sqlite_err)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(sqlite_err)?;
            rows
        };
        let cursor_changed = previous_cursor != Some(settled_cursor);
        if !cursor_changed && removed.is_empty() {
            let generation = transaction
                .query_row(
                    "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                    [namespace],
                    |row| row.get::<_, i64>(0),
                )
                .optional()
                .map_err(sqlite_err)?
                .unwrap_or(0);
            transaction.commit().map_err(sqlite_err)?;
            return Ok((
                TableRevision::new(generation as u64),
                ForwardingSettlement {
                    cursor: settled_cursor,
                    ..Default::default()
                },
            ));
        }

        transaction
            .execute(
                "INSERT INTO forward_state (target, namespace, cursor) VALUES (?1, ?2, ?3)
                 ON CONFLICT(target, namespace) DO UPDATE SET cursor = excluded.cursor",
                rusqlite::params![target, namespace, settled_cursor],
            )
            .map_err(sqlite_err)?;
        let removed_paths = removed
            .iter()
            .map(|(path, _, _)| path.clone())
            .collect::<Vec<_>>();
        remove_segments_in(&transaction, namespace, &removed_paths)?;
        let revision = if has_versioned_head {
            advance_generation_in(&transaction, namespace)?
        } else {
            TableRevision::new(0)
        };
        transaction.commit().map_err(sqlite_err)?;
        Ok((
            revision,
            ForwardingSettlement {
                cursor: settled_cursor,
                removed_paths,
                removed_rows: removed.iter().map(|(_, rows, _)| rows).sum(),
                removed_bytes: removed.iter().map(|(_, _, bytes)| bytes).sum(),
            },
        ))
    }
}
