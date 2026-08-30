//! Object-store implementation of the durable table-state boundary.
//!
//! One table is an immutable state document per revision plus a mutable
//! `HEAD.json` pointer swapped by compare-and-swap. HEAD records the revision
//! and the [`WriterFence`] that owns it, so every commit is checked against
//! both the backend token and the fence.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogHead, NamespaceCatalog, ObjectRef};
use crate::store::catalog::state_store::{
    fenced_error, BackendToken, StoredTableState, TableHead, TableStateStore,
};
use crate::store::object_store::{ObjectId, ObjectMetadata, ObjectPrefix, ObjectStore};
use crate::store::table_state::{TableRevision, WriterFence};

pub const TABLE_STATE_FORMAT_VERSION: u64 = 1;
const HEAD_KEY: &str = "HEAD.json";
const STATES_PREFIX: &str = "catalogs";
/// Key prefix under a table for the data objects its states reference.
pub(crate) const OBJECTS_PREFIX: &str = "objects";
/// Key prefix under a table for segment index bundles.
pub(crate) const INDICES_PREFIX: &str = "indices";
/// Key prefix under a table for covering-projection artifacts.
pub(crate) const PROJECTIONS_PREFIX: &str = "projections";

#[derive(Clone)]
pub struct ObjectTableStateStore {
    storage: Arc<dyn ObjectStore>,
}

impl ObjectTableStateStore {
    pub fn new(storage: Arc<dyn ObjectStore>) -> Self {
        Self { storage }
    }

    /// Read and validate HEAD without reading the state document it names.
    async fn load_head(
        &self,
        table: &str,
    ) -> Result<Option<(CatalogHead, BackendToken)>, StatsError> {
        let head_id = ObjectId::table(table, HEAD_KEY)?;
        let Some(head_object) = self.storage.read(&head_id).await? else {
            return Ok(None);
        };
        let head: CatalogHead = serde_json::from_slice(&head_object.bytes).map_err(|error| {
            StatsError::Internal(format!("decode object HEAD for {table:?}: {error}"))
        })?;
        validate_head(table, &head)?;
        Ok(Some((head, BackendToken::Head(head_object.version))))
    }

    pub async fn load(&self, table: &str) -> Result<Option<StoredTableState>, StatsError> {
        let Some((head, token)) = self.load_head(table).await? else {
            return Ok(None);
        };
        let state_ref = head.catalog.as_option().ok_or_else(|| {
            StatsError::Internal(format!("object HEAD for {table:?} has no state reference"))
        })?;
        let state_object_id = state_ref.object_id.as_deref().ok_or_else(|| {
            StatsError::Internal(format!(
                "table HEAD for {table:?} has an empty state object ID"
            ))
        })?;
        let state_id = ObjectId::parse(state_object_id)?;
        if state_id.table_relative(table).is_none() {
            return Err(StatsError::Internal(format!(
                "table HEAD for {table:?} references an object from another table"
            )));
        }
        let state_object = self.storage.read(&state_id).await?.ok_or_else(|| {
            StatsError::Internal(format!(
                "table HEAD for {table:?} references missing state {state_object_id:?}"
            ))
        })?;
        if state_ref.sha256.as_deref() != Some(state_object.version.content_sha256.as_slice()) {
            return Err(StatsError::Internal(format!(
                "table state {state_object_id:?} for {table:?} failed SHA-256 validation"
            )));
        }
        let catalog: NamespaceCatalog =
            serde_json::from_slice(&state_object.bytes).map_err(|error| {
                StatsError::Internal(format!(
                    "decode table state {state_object_id:?} for {table:?}: {error}"
                ))
            })?;
        validate_state(table, &head, &catalog)?;
        Ok(Some(StoredTableState {
            head,
            catalog,
            token,
        }))
    }

    pub async fn list(&self) -> Result<Vec<TableHead>, StatsError> {
        let mut heads = Vec::new();
        for table in self.storage.list_tables().await? {
            let Some((head, _)) = self.load_head(&table).await? else {
                continue;
            };
            heads.push(TableHead {
                revision: TableRevision::new(head.catalog_generation.unwrap_or(0)),
                fence: WriterFence::new(head.writer_epoch.unwrap_or(0)),
                tombstoned: head.tombstoned.unwrap_or(false),
                table,
            });
        }
        Ok(heads)
    }

    pub async fn claim_writer(
        &self,
        table: &str,
        fence: WriterFence,
        selected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError> {
        if selected.fence() == fence {
            return Ok(selected.clone());
        }
        let mut head = selected.head.clone();
        head.writer_epoch = Some(fence.get());
        let head_bytes = serde_json::to_vec(&head).map_err(|error| {
            StatsError::Internal(format!("encode table HEAD for {table:?}: {error}"))
        })?;
        let head_version = self
            .storage
            .compare_and_swap(
                &ObjectId::table(table, HEAD_KEY)?,
                selected.head_version(),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(StoredTableState {
            head,
            catalog: selected.catalog.clone(),
            token: BackendToken::Head(head_version),
        })
    }

    /// Write the immutable state document and swap HEAD onto it.
    async fn publish(
        &self,
        table: &str,
        fence: WriterFence,
        catalog: NamespaceCatalog,
        expected: Option<&StoredTableState>,
    ) -> Result<StoredTableState, StatsError> {
        let revision = catalog.catalog_generation.unwrap_or(0);
        let previous = expected.map(|state| state.revision().get());
        if revision == 0 || previous.is_some_and(|previous| revision <= previous) {
            return Err(StatsError::SchemaConflict(format!(
                "table state revision {revision} does not advance {previous:?} for {table:?}"
            )));
        }
        if catalog.format_version.unwrap_or(0) != TABLE_STATE_FORMAT_VERSION
            || catalog.namespace.as_deref() != Some(table)
        {
            return Err(StatsError::SchemaValidation(format!(
                "table state identity does not match table {table:?}"
            )));
        }

        let state_bytes = serde_json::to_vec(&catalog).map_err(|error| {
            StatsError::Internal(format!("encode table state for {table:?}: {error}"))
        })?;
        let state_sha256: [u8; 32] = Sha256::digest(&state_bytes).into();
        let state_key = format!(
            "{STATES_PREFIX}/{revision:020}-{}.json",
            short_hex(&state_sha256)
        );
        let state_id = ObjectId::table(table, &state_key)?;
        let state_version = self
            .storage
            .write(&state_id, Bytes::from(state_bytes.clone()))
            .await?;
        let head = CatalogHead {
            format_version: Some(TABLE_STATE_FORMAT_VERSION),
            namespace: Some(table.to_string()),
            writer_epoch: Some(fence.get()),
            catalog_generation: Some(revision),
            active_table_spec_version: catalog.active_table_spec_version,
            tombstoned: catalog.tombstoned,
            catalog: buffa::MessageField::some(ObjectRef {
                object_id: Some(state_id.as_str().to_string()),
                provider_version: state_version.provider_version.clone(),
                etag: state_version.e_tag.clone(),
                byte_size: Some(state_bytes.len() as u64),
                sha256: Some(state_sha256.to_vec()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let head_bytes = serde_json::to_vec(&head).map_err(|error| {
            StatsError::Internal(format!("encode object HEAD for {table:?}: {error}"))
        })?;
        let head_version = self
            .storage
            .compare_and_swap(
                &ObjectId::table(table, HEAD_KEY)?,
                expected.and_then(StoredTableState::head_version),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(StoredTableState {
            head,
            catalog,
            token: BackendToken::Head(head_version),
        })
    }

    /// The selected state a fenced mutation may build on.
    ///
    /// Fails when another writer owns HEAD, when the caller's token is stale,
    /// or when the table is already tombstoned.
    async fn fenced_selection(
        &self,
        table: &str,
        fence: WriterFence,
        expected: Option<&StoredTableState>,
    ) -> Result<Option<StoredTableState>, StatsError> {
        let Some(current) = self.load(table).await? else {
            if expected.is_some() {
                return Err(StatsError::SchemaConflict(format!(
                    "table {table:?} has no HEAD for the presented commit token"
                )));
            }
            return Ok(None);
        };
        if current.fence() != fence {
            return Err(fenced_error(table, fence, current.fence()));
        }
        if current.is_tombstoned() {
            return Err(StatsError::SchemaConflict(format!(
                "table {table:?} was deleted at revision {}",
                current.revision()
            )));
        }
        if let Some(expected) = expected {
            if expected.revision() != current.revision() {
                return Err(StatsError::SchemaConflict(format!(
                    "table {table:?} moved from revision {} to {} under this writer's token",
                    expected.revision(),
                    current.revision()
                )));
            }
        }
        Ok(Some(current))
    }

    pub async fn commit(
        &self,
        table: &str,
        fence: WriterFence,
        expected: Option<&StoredTableState>,
        next: NamespaceCatalog,
    ) -> Result<StoredTableState, StatsError> {
        let current = self.fenced_selection(table, fence, expected).await?;
        if let Some(current) = &current {
            let selected = current.revision().get();
            let attempted = next.catalog_generation.unwrap_or(0);
            if selected == attempted {
                if current.catalog != next {
                    return Err(StatsError::SchemaConflict(format!(
                        "table {table:?} publishes a different state at revision {attempted}"
                    )));
                }
                return Ok(current.clone());
            }
            if selected > attempted {
                return Err(StatsError::SchemaConflict(format!(
                    "table state revision {attempted} for {table:?} does not advance the selected revision {selected}"
                )));
            }
        }
        self.publish(table, fence, next, current.as_ref()).await
    }

    pub async fn tombstone(
        &self,
        table: &str,
        fence: WriterFence,
        expected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError> {
        let current = self
            .fenced_selection(table, fence, Some(expected))
            .await?
            .ok_or_else(|| {
                StatsError::SchemaConflict(format!("table {table:?} has no HEAD to tombstone"))
            })?;
        let mut catalog = current.catalog.clone();
        catalog.catalog_generation = Some(current.revision().get() + 1);
        catalog.tombstoned = Some(true);
        self.publish(table, fence, catalog, Some(&current)).await
    }

    #[cfg(test)]
    async fn state_keys(&self, table: &str) -> Result<Vec<String>, StatsError> {
        Ok(self
            .table_objects(table, STATES_PREFIX)
            .await?
            .into_iter()
            .map(|(key, _)| key)
            .collect())
    }

    /// Remove superseded state documents after the maximum query lifetime.
    pub async fn gc_obsolete_states(
        &self,
        table: &str,
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
        fence: WriterFence,
    ) -> Result<usize, StatsError> {
        let Some(selected) = self.load(table).await? else {
            return Ok(0);
        };
        if selected.fence() != fence {
            tracing::warn!(
                table,
                expected_writer_fence = %fence,
                head_writer_fence = %selected.fence(),
                "skipping object GC from a fenced writer"
            );
            return Ok(0);
        }
        let current_id = ObjectId::parse(
            selected
                .head
                .catalog
                .as_option()
                .and_then(|reference| reference.object_id.as_deref())
                .ok_or_else(|| {
                    StatsError::Internal(format!("table HEAD for {table:?} has no state object ID"))
                })?,
        )?;
        let current = current_id.table_relative(table).ok_or_else(|| {
            StatsError::Internal(format!("table HEAD for {table:?} points outside its table"))
        })?;
        let state_cutoff =
            now_ms.saturating_sub(i64::try_from(state_retention_ms).unwrap_or(i64::MAX));
        let orphan_cutoff =
            now_ms.saturating_sub(i64::try_from(orphan_grace_ms).unwrap_or(i64::MAX));
        let current_revision = selected.revision().get();
        let state_objects = self.table_objects(table, STATES_PREFIX).await?;
        let mut removed = 0;
        for (key, meta) in &state_objects {
            if key == current {
                continue;
            }
            let Some(revision) = state_revision_from_key(key) else {
                tracing::warn!(table, key, "retaining unrecognized table state key");
                continue;
            };
            let obsolete_at_ms = if revision < current_revision {
                state_objects
                    .iter()
                    .filter_map(|(candidate_key, candidate_meta)| {
                        let candidate_revision = state_revision_from_key(candidate_key)?;
                        (candidate_revision > revision && candidate_revision <= current_revision)
                            .then_some(candidate_meta.modified_at_ms)
                    })
                    .min()
            } else {
                // Same-revision and future-revision objects never won HEAD.
                Some(meta.modified_at_ms)
            };
            if obsolete_at_ms.is_none_or(|obsolete_at_ms| obsolete_at_ms > state_cutoff) {
                continue;
            }
            self.storage.delete(&ObjectId::table(table, key)?).await?;
            removed += 1;
        }
        let mut referenced = referenced_object_keys(&selected.catalog);
        for (key, _) in self.table_objects(table, STATES_PREFIX).await? {
            if key == current {
                continue;
            }
            let Some(object) = self.storage.read(&ObjectId::table(table, &key)?).await? else {
                continue;
            };
            let catalog: NamespaceCatalog =
                serde_json::from_slice(&object.bytes).map_err(|error| {
                    StatsError::Internal(format!(
                        "decode retained table state {key:?} for {table:?}: {error}"
                    ))
                })?;
            referenced.extend(referenced_object_keys(&catalog));
        }
        for prefix in [OBJECTS_PREFIX, INDICES_PREFIX, PROJECTIONS_PREFIX] {
            for (key, meta) in self.table_objects(table, prefix).await? {
                let id = ObjectId::table(table, &key)?;
                if referenced.contains(id.as_str()) || meta.modified_at_ms > orphan_cutoff {
                    continue;
                }
                self.storage.delete(&id).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    async fn table_objects(
        &self,
        table: &str,
        relative_prefix: &str,
    ) -> Result<Vec<(String, ObjectMetadata)>, StatsError> {
        let objects = self
            .storage
            .list(&ObjectPrefix::table(table, relative_prefix)?)
            .await?;
        objects
            .into_iter()
            .map(|metadata| {
                let key = metadata
                    .id
                    .table_relative(table)
                    .ok_or_else(|| {
                        StatsError::Internal(format!(
                            "object {:?} escaped table {table:?}",
                            metadata.id.as_str()
                        ))
                    })?
                    .to_string();
                Ok((key, metadata))
            })
            .collect()
    }
}

#[async_trait]
impl TableStateStore for ObjectTableStateStore {
    async fn list(&self) -> Result<Vec<TableHead>, StatsError> {
        ObjectTableStateStore::list(self).await
    }

    async fn load(&self, table: &str) -> Result<Option<StoredTableState>, StatsError> {
        ObjectTableStateStore::load(self, table).await
    }

    async fn claim_writer(
        &self,
        table: &str,
        fence: WriterFence,
        selected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError> {
        ObjectTableStateStore::claim_writer(self, table, fence, selected).await
    }

    async fn commit(
        &self,
        table: &str,
        fence: WriterFence,
        expected: Option<&StoredTableState>,
        next: NamespaceCatalog,
    ) -> Result<StoredTableState, StatsError> {
        ObjectTableStateStore::commit(self, table, fence, expected, next).await
    }

    async fn tombstone(
        &self,
        table: &str,
        fence: WriterFence,
        expected: &StoredTableState,
    ) -> Result<StoredTableState, StatsError> {
        ObjectTableStateStore::tombstone(self, table, fence, expected).await
    }

    async fn gc_obsolete_states(
        &self,
        table: &str,
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
        fence: WriterFence,
    ) -> Result<usize, StatsError> {
        ObjectTableStateStore::gc_obsolete_states(
            self,
            table,
            now_ms,
            state_retention_ms,
            orphan_grace_ms,
            fence,
        )
        .await
    }
}

/// Every object ID a segment keeps alive: its data source, its index bundle,
/// and each covering-projection artifact.
fn segment_object_keys(
    segment: &crate::proto::finelog::stats::CatalogSegment,
) -> impl Iterator<Item = String> + '_ {
    segment
        .source
        .as_option()
        .and_then(|source| source.object_id.clone())
        .into_iter()
        .chain(
            segment
                .index_bundle
                .as_option()
                .and_then(|bundle| bundle.object_id.clone()),
        )
        .chain(segment.projections.iter().filter_map(|projection| {
            projection
                .object
                .as_option()
                .and_then(|object| object.object_id.clone())
        }))
}

fn referenced_object_keys(catalog: &NamespaceCatalog) -> std::collections::HashSet<String> {
    catalog
        .version_segments
        .iter()
        .flat_map(|version| {
            version
                .live_segments
                .iter()
                .chain(version.retired_segments.iter())
        })
        .chain(catalog.direct_query_segments.iter())
        .flat_map(segment_object_keys)
        .collect()
}

fn validate_head(table: &str, head: &CatalogHead) -> Result<(), StatsError> {
    if head.format_version.unwrap_or(0) != TABLE_STATE_FORMAT_VERSION
        || head.namespace.as_deref() != Some(table)
        || head.catalog_generation.unwrap_or(0) == 0
    {
        return Err(StatsError::Internal(format!(
            "invalid object HEAD for table {table:?}"
        )));
    }
    Ok(())
}

fn validate_state(
    table: &str,
    head: &CatalogHead,
    catalog: &NamespaceCatalog,
) -> Result<(), StatsError> {
    if catalog.format_version.unwrap_or(0) != TABLE_STATE_FORMAT_VERSION
        || catalog.namespace.as_deref() != Some(table)
        || catalog.catalog_generation != head.catalog_generation
        || catalog.active_table_spec_version != head.active_table_spec_version
        || catalog.tombstoned.unwrap_or(false) != head.tombstoned.unwrap_or(false)
    {
        return Err(StatsError::Internal(format!(
            "table state does not match HEAD for table {table:?}"
        )));
    }
    Ok(())
}

fn short_hex(bytes: &[u8; 32]) -> String {
    crate::hex::encode(&bytes[..8])
}

fn state_revision_from_key(key: &str) -> Option<u64> {
    key.strip_prefix(STATES_PREFIX)?
        .strip_prefix('/')?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::object_store::build_remote_object_store;

    const TABLE: &str = "iris.worker";

    fn state(table: &str, revision: u64, active_version: u64) -> NamespaceCatalog {
        NamespaceCatalog {
            format_version: Some(TABLE_STATE_FORMAT_VERSION),
            namespace: Some(table.to_string()),
            catalog_generation: Some(revision),
            active_table_spec_version: Some(active_version),
            max_query_time_ms: Some(600_000),
            ..Default::default()
        }
    }

    fn store(tag: &str) -> (ObjectTableStateStore, std::path::PathBuf) {
        let remote_dir = crate::test_support::unique_dir(tag);
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        (ObjectTableStateStore::new(Arc::new(remote)), remote_dir)
    }

    #[tokio::test]
    async fn head_cas_selects_one_complete_revision() {
        let (states, remote_dir) = store("object_state_store_cas");
        let fence = WriterFence::new(11);

        let first = states
            .commit(TABLE, fence, None, state(TABLE, 1, 1))
            .await
            .unwrap();
        let loaded = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(loaded.fence(), fence);
        assert_eq!(loaded.revision(), TableRevision::new(1));

        let second = states
            .commit(TABLE, fence, Some(&first), state(TABLE, 2, 2))
            .await
            .unwrap();
        // The stale token names a revision HEAD has already moved past.
        let stale = states
            .commit(TABLE, fence, Some(&first), state(TABLE, 2, 3))
            .await
            .unwrap_err();
        assert!(matches!(stale, StatsError::SchemaConflict(_)));

        let loaded = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(loaded.catalog.active_table_spec_version, Some(2));
        assert_eq!(second.catalog, loaded.catalog);
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn a_commit_from_a_stale_fence_is_rejected() {
        let (states, remote_dir) = store("object_state_store_fence");
        let stale_fence = WriterFence::new(11);
        let selected = states
            .commit(TABLE, stale_fence, None, state(TABLE, 1, 1))
            .await
            .unwrap();

        let claimed = states
            .claim_writer(TABLE, WriterFence::new(12), &selected)
            .await
            .unwrap();
        assert_eq!(claimed.fence(), WriterFence::new(12));
        assert_eq!(claimed.revision(), TableRevision::new(1));

        // The stale writer still holds a token that named HEAD, and a fresh
        // load would hand it the current one. The recorded fence rejects it.
        let error = states
            .commit(TABLE, stale_fence, Some(&selected), state(TABLE, 2, 1))
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
        let error = states
            .commit(TABLE, stale_fence, None, state(TABLE, 2, 1))
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision(),
            TableRevision::new(1)
        );

        states
            .commit(
                TABLE,
                WriterFence::new(12),
                Some(&claimed),
                state(TABLE, 2, 1),
            )
            .await
            .unwrap();
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn a_tombstone_publishes_a_deleted_revision_that_list_and_load_report() {
        let (states, remote_dir) = store("object_state_store_tombstone");
        let fence = WriterFence::new(11);
        let selected = states
            .commit(TABLE, fence, None, state(TABLE, 1, 1))
            .await
            .unwrap();

        let deleted = states.tombstone(TABLE, fence, &selected).await.unwrap();
        assert!(deleted.is_tombstoned());
        assert_eq!(deleted.revision(), TableRevision::new(2));

        let loaded = states.load(TABLE).await.unwrap().unwrap();
        assert!(loaded.is_tombstoned());
        let listed = states.list().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].tombstoned);
        assert_eq!(listed[0].table, TABLE);

        // A deleted table accepts no further state.
        let error = states
            .commit(TABLE, fence, Some(&deleted), state(TABLE, 3, 1))
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn immutable_state_documents_reject_differing_retries() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_immutable");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let published = states
            .commit(TABLE, WriterFence::new(1), None, state(TABLE, 1, 1))
            .await
            .unwrap();
        let key = published
            .head
            .catalog
            .as_option()
            .unwrap()
            .object_id
            .as_deref()
            .unwrap();
        let object_id = ObjectId::parse(key).unwrap();
        let existing = remote.read(&object_id).await.unwrap().unwrap();
        remote.write(&object_id, existing.bytes).await.unwrap();
        let error = remote
            .write(&object_id, Bytes::from_static(b"different"))
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn garbage_collection_retains_states_within_the_query_grace() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_gc");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let fence = WriterFence::new(11);
        let first = states
            .commit(TABLE, fence, None, state(TABLE, 1, 1))
            .await
            .unwrap();
        let second = states
            .commit(TABLE, fence, Some(&first), state(TABLE, 2, 2))
            .await
            .unwrap();

        let current_key = second
            .head
            .catalog
            .as_option()
            .unwrap()
            .object_id
            .as_deref()
            .unwrap();
        let current_modified_ms = remote
            .list(&ObjectPrefix::table(TABLE, STATES_PREFIX).unwrap())
            .await
            .unwrap()
            .into_iter()
            .find(|metadata| metadata.id.as_str() == current_key)
            .unwrap()
            .modified_at_ms;
        assert_eq!(
            states
                .gc_obsolete_states(TABLE, current_modified_ms + 5, 10, 10, fence)
                .await
                .unwrap(),
            0
        );
        let future = i64::MAX;
        // A fenced writer collects nothing.
        assert_eq!(
            states
                .gc_obsolete_states(TABLE, future, 0, 0, WriterFence::new(12))
                .await
                .unwrap(),
            0
        );
        assert_eq!(states.state_keys(TABLE).await.unwrap().len(), 2);
        assert_eq!(
            states
                .gc_obsolete_states(TABLE, future, 600_000, 600_000, fence)
                .await
                .unwrap(),
            1
        );
        assert_eq!(states.state_keys(TABLE).await.unwrap().len(), 1);
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn garbage_collection_removes_unreferenced_objects_after_query_grace() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_orphan_gc");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        states
            .commit(TABLE, WriterFence::new(1), None, state(TABLE, 1, 1))
            .await
            .unwrap();
        let orphan_id = ObjectId::table(TABLE, "objects/v1/l0/orphan/source.parquet").unwrap();
        remote
            .write(&orphan_id, Bytes::from_static(b"orphan"))
            .await
            .unwrap();

        assert_eq!(
            states
                .gc_obsolete_states(TABLE, i64::MAX, 600_000, 600_000, WriterFence::new(1))
                .await
                .unwrap(),
            1
        );
        assert!(remote.read(&orphan_id).await.unwrap().is_none());
        std::fs::remove_dir_all(remote_dir).ok();
    }
}
