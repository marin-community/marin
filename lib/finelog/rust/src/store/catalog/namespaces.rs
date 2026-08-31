// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! The live namespace registry: which tables exist right now.
//!
//! An in-memory map for lock-free-ish reads on the hot path, backed by
//! `namespaces` rows for restarts, plus the `dropping` reservation set that
//! fences a concurrent re-register during a drop.

use super::*;
use crate::errors::StatsError;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_from_json, schema_to_json, Schema};

/// A live namespace value.
#[derive(Debug, Clone)]
pub struct RegisteredNamespace {
    pub name: String,
    pub schema: Schema,
    pub policy: StoragePolicy,
}

impl Catalog {
    pub fn contains(&self, name: &str) -> bool {
        self.inner.lock().unwrap().live.contains_key(name)
    }
    pub fn get_live(&self, name: &str) -> Option<RegisteredNamespace> {
        self.inner.lock().unwrap().live.get(name).cloned()
    }
    pub fn require_live(&self, name: &str) -> Result<RegisteredNamespace, StatsError> {
        self.get_live(name).ok_or_else(|| {
            StatsError::NamespaceNotFound(format!("namespace {name:?} is not registered"))
        })
    }
    #[cfg(test)]
    pub fn is_dropping(&self, name: &str) -> bool {
        self.inner.lock().unwrap().dropping.contains(name)
    }
    /// Live namespaces in registration order.
    pub fn snapshot_live(&self) -> Vec<RegisteredNamespace> {
        let inner = self.inner.lock().unwrap();
        let mut entries: Vec<&RegisteredNamespace> = inner.live.values().collect();
        entries.sort_by_key(|ns| inner.registered_at.get(&ns.name).copied().unwrap_or(0));
        entries.into_iter().cloned().collect()
    }
    /// Publish a freshly-built namespace (rehydrate path).
    pub fn insert_live(&self, ns: RegisteredNamespace) {
        let mut inner = self.inner.lock().unwrap();
        inner.publish_locked(ns);
    }
    /// Atomically register `name` or evolve the existing namespace.
    ///
    /// The whole decision-and-publish runs under a SINGLE lock so it cannot
    /// interleave with `begin_drop`/`finish_drop`. Releasing the lock between
    /// the drop-fence check and publish is unsafe: because RPC handlers dispatch blocking
    /// `Store` calls onto a multi-threaded `spawn_blocking` pool sharing one
    /// `Arc<Store>`, a concurrent register+drop of the same name could resurrect
    /// a dropped namespace with no persisted row.
    ///
    /// On a fresh registration, persists `stored_schema` + `policy` and
    /// publishes, returning `(stored_schema, policy)`. On an existing namespace,
    /// `merge` computes the effective schema from the existing one (a PURE
    /// function — it must not call back into the catalog, since the lock is
    /// held); the effective schema is persisted only if it changed, and an
    /// empty `policy` preserves the existing policy.
    ///
    /// Raises `InvalidNamespace` if a drop is in flight.
    pub fn register_or_evolve(
        &self,
        name: &str,
        stored_schema: Schema,
        policy: StoragePolicy,
        merge: impl FnOnce(&Schema) -> Result<Schema, StatsError>,
    ) -> Result<(Schema, StoragePolicy), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.dropping.contains(name) {
            return Err(StatsError::InvalidNamespace(format!(
                "namespace {name:?} is currently being dropped; retry once drop_table completes"
            )));
        }

        if let Some(existing) = inner.live.get(name).cloned() {
            // `merge` raises SchemaConflict on a column-type change.
            let effective = merge(&existing.schema)?;
            if effective != existing.schema {
                inner.upsert_locked(name, &effective)?;
            }
            let effective_policy = if policy.is_empty() {
                existing.policy.clone()
            } else {
                inner.upsert_policy_locked(name, &policy)?;
                policy
            };
            inner.publish_locked(RegisteredNamespace {
                name: name.to_string(),
                schema: effective.clone(),
                policy: effective_policy.clone(),
            });
            return Ok((effective, effective_policy));
        }

        inner.upsert_locked(name, &stored_schema)?;
        inner.upsert_policy_locked(name, &policy)?;
        inner.publish_locked(RegisteredNamespace {
            name: name.to_string(),
            schema: stored_schema.clone(),
            policy: policy.clone(),
        });
        Ok((stored_schema, policy))
    }
    /// Pop `name` from the registry and reserve it in `dropping`.
    pub fn begin_drop(&self, name: &str) -> Result<RegisteredNamespace, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let ns = inner.live.remove(name).ok_or_else(|| {
            StatsError::NamespaceNotFound(format!("namespace {name:?} is not registered"))
        })?;
        inner.registered_at.remove(name);
        inner.dropping.insert(name.to_string());
        Ok(ns)
    }
    pub fn finish_drop(&self, name: &str) {
        self.inner.lock().unwrap().dropping.remove(name);
    }

    // ----- namespaces table ---------------------------------------------
    /// All persisted `(name, schema)` rows (used by rehydrate).
    pub fn list_all(&self) -> Result<Vec<(String, Schema)>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut stmt = inner
            .conn
            .prepare("SELECT namespace, schema_json FROM namespaces")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for r in rows {
            let (name, json) = r.map_err(sqlite_err)?;
            out.push((name, schema_from_json(&json)?));
        }
        Ok(out)
    }
    /// Remove every row `name` owns in one transaction, so a failure part way
    /// through leaves the namespace whole rather than half deleted. Idempotent.
    pub fn delete(&self, name: &str) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        for table in NAMESPACE_OWNED_TABLES {
            transaction
                .execute(&format!("DELETE FROM {table} WHERE namespace = ?1"), [name])
                .map_err(sqlite_err)?;
        }
        transaction.commit().map_err(sqlite_err)?;
        Ok(())
    }
    /// Insert or evolve the row for `name`, keeping the live value in sync.
    /// `registered_at_ms` is preserved on update; `last_modified_ms` is bumped.
    pub fn upsert(&self, name: &str, schema: &Schema) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        inner.upsert_locked(name, schema)?;
        if let Some(ns) = inner.live.get_mut(name) {
            ns.schema = schema.clone();
        }
        Ok(())
    }
}

impl CatalogInner {
    /// Persist the `namespaces` row for `name` (no live-registry update — the
    /// caller publishes). `registered_at_ms` is preserved on update;
    /// `last_modified_ms` is bumped. Operates on the held guard so it composes
    /// inside a single `register_or_evolve` critical section.
    pub(super) fn upsert_locked(&mut self, name: &str, schema: &Schema) -> Result<(), StatsError> {
        let now = now_ms();
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT registered_at_ms FROM namespaces WHERE namespace = ?1",
                [name],
                |row| row.get(0),
            )
            .ok();
        let registered_at = existing.unwrap_or(now);
        self.conn
            .execute(
                r#"
                INSERT INTO namespaces (namespace, schema_json, registered_at_ms, last_modified_ms)
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT (namespace) DO UPDATE
                  SET schema_json = excluded.schema_json,
                      last_modified_ms = excluded.last_modified_ms
                "#,
                rusqlite::params![name, schema_to_json(schema), registered_at, now],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }

    /// Persist `policy` for `name`, or delete the row when every field is
    /// `None`. No live-registry update (the caller publishes).
    pub(super) fn upsert_policy_locked(
        &mut self,
        name: &str,
        policy: &StoragePolicy,
    ) -> Result<(), StatsError> {
        if policy.is_empty() {
            self.conn
                .execute("DELETE FROM storage_policies WHERE namespace = ?1", [name])
                .map_err(sqlite_err)?;
        } else {
            self.conn
                .execute(
                    r#"
                    INSERT INTO storage_policies (namespace, max_segments, max_bytes, max_age_seconds)
                    VALUES (?1, ?2, ?3, ?4)
                    ON CONFLICT (namespace) DO UPDATE
                      SET max_segments    = excluded.max_segments,
                          max_bytes       = excluded.max_bytes,
                          max_age_seconds = excluded.max_age_seconds
                    "#,
                    rusqlite::params![name, policy.max_segments, policy.max_bytes, policy.max_age_seconds],
                )
                .map_err(sqlite_err)?;
        }
        Ok(())
    }

    pub(super) fn publish_locked(&mut self, ns: RegisteredNamespace) {
        let name = ns.name.clone();
        self.live.insert(name.clone(), ns);
        self.registered_at.entry(name).or_insert_with(|| {
            let o = self.next_ordinal;
            self.next_ordinal += 1;
            o
        });
    }
}
