//! Object-store implementation of the durable table-state boundary.
//!
//! One table is an immutable parent-linked catalog tree plus a mutable
//! `HEAD.json` pointer swapped by compare-and-swap. Most publications append a
//! typed delta; bounded checkpoints cap recovery work and cut old ancestry.
//! HEAD records the selected tip, revision, and [`WriterFence`], so every
//! commit is checked against both the backend token and the fence.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use uuid::Uuid;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    CatalogHead, CatalogNode, NamespaceCatalog, ObjectRef, ReleasedObject,
};
use crate::store::object_store::{
    ObjectId, ObjectMetadata, ObjectPrefix, ObjectStore, ObjectVersion, StoredObject,
    INDICES_PREFIX, OBJECTS_PREFIX, PROJECTIONS_PREFIX,
};
use crate::store::state_store::tree::{
    apply as apply_delta, canonical_catalog, catalogs_equal, delta as catalog_delta,
    logical_reference, referenced_objects, released_objects,
};
use crate::store::state_store::{fenced_error, StoredTableState, TableHead};
use crate::store::table::now_ms;
use crate::store::table_state::{TableRevision, WriterFence};

pub const TABLE_STATE_FORMAT_VERSION: u64 = 2;
pub(crate) const LEGACY_TABLE_STATE_FORMAT_VERSION: u64 = 1;
const HEAD_KEY: &str = "HEAD.json";
const STATES_PREFIX: &str = "catalogs";
const MAX_DELTA_DEPTH: u32 = 64;
const MAX_DELTA_BYTES: u64 = 1024 * 1024;
const MAX_RECOVERY_NODES: usize = MAX_DELTA_DEPTH as usize + 1;

#[derive(Clone)]
pub struct ObjectTableStateStore {
    storage: Arc<dyn ObjectStore>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StateGcPolicy {
    pub pin_retention_ms: u64,
    pub state_retention_ms: u64,
    pub orphan_grace_ms: u64,
    pub sweep_orphans: bool,
}

#[derive(Debug, Default)]
pub(crate) struct StateGcResult {
    pub removed: usize,
    pub deleted_releases: Vec<String>,
}

struct LoadedCatalogTree {
    catalog: NamespaceCatalog,
    chain: Vec<String>,
    delta_depth: u32,
    delta_bytes_since_checkpoint: u64,
    pending_releases: Vec<ReleasedObject>,
    release_summary_complete: bool,
    legacy_history_safe_after_ms: i64,
}

impl ObjectTableStateStore {
    pub fn new(storage: Arc<dyn ObjectStore>) -> Self {
        Self { storage }
    }

    /// Read and validate HEAD without reading the state document it names.
    async fn load_head(
        &self,
        table: &str,
    ) -> Result<Option<(CatalogHead, ObjectVersion)>, StatsError> {
        let head_id = ObjectId::table(table, HEAD_KEY)?;
        let Some(head_object) = self.storage.read(&head_id).await? else {
            return Ok(None);
        };
        let head: CatalogHead = serde_json::from_slice(&head_object.bytes).map_err(|error| {
            StatsError::Internal(format!("decode object HEAD for {table:?}: {error}"))
        })?;
        validate_head(table, &head)?;
        Ok(Some((head, head_object.version)))
    }

    pub async fn load(&self, table: &str) -> Result<Option<StoredTableState>, StatsError> {
        let Some((head, head_version)) = self.load_head(table).await? else {
            return Ok(None);
        };
        let state_reference = head.catalog.as_option().ok_or_else(|| {
            StatsError::Internal(format!("object HEAD for {table:?} has no state reference"))
        })?;
        let state_object_id = state_object_id(table, &head)?;
        let format = head.format_version.unwrap_or(0);
        let loaded = match format {
            LEGACY_TABLE_STATE_FORMAT_VERSION => {
                let state_object = self
                    .read_table_object(table, &state_object_id, state_reference.byte_size)
                    .await?;
                let catalog: NamespaceCatalog = serde_json::from_slice(&state_object.bytes)
                    .map_err(|error| {
                        StatsError::Internal(format!(
                            "decode legacy table state {state_object_id:?} for {table:?}: {error}"
                        ))
                    })?;
                validate_state(table, &head, &catalog)?;
                LoadedCatalogTree {
                    catalog,
                    chain: vec![state_object_id],
                    delta_depth: 0,
                    delta_bytes_since_checkpoint: 0,
                    pending_releases: Vec::new(),
                    release_summary_complete: false,
                    legacy_history_safe_after_ms: 0,
                }
            }
            TABLE_STATE_FORMAT_VERSION => {
                let byte_size = catalog_reference_size(table, state_reference)?;
                self.load_catalog_tree(table, &head, state_object_id, Some(byte_size))
                    .await?
            }
            _ => {
                return Err(StatsError::Internal(format!(
                    "unsupported object HEAD format {format} for table {table:?}"
                )))
            }
        };
        Ok(Some(StoredTableState {
            head,
            catalog: loaded.catalog,
            head_version,
            catalog_chain: loaded.chain,
            delta_depth: loaded.delta_depth,
            delta_bytes_since_checkpoint: loaded.delta_bytes_since_checkpoint,
            pending_releases: loaded.pending_releases,
            release_summary_complete: loaded.release_summary_complete,
            legacy_history_safe_after_ms: loaded.legacy_history_safe_after_ms,
        }))
    }

    async fn read_table_object(
        &self,
        table: &str,
        object_id: &str,
        expected_size: Option<u64>,
    ) -> Result<StoredObject, StatsError> {
        let id = ObjectId::parse(object_id)?;
        if id.table_relative(table).is_none() {
            return Err(StatsError::Internal(format!(
                "table HEAD for {table:?} references an object from another table"
            )));
        }
        let object = self.storage.read(&id).await?.ok_or_else(|| {
            StatsError::Internal(format!(
                "table HEAD for {table:?} references missing state {object_id:?}"
            ))
        })?;
        if expected_size.is_some_and(|expected| expected != object.bytes.len() as u64) {
            return Err(StatsError::Internal(format!(
                "catalog object {object_id:?} for table {table:?} has the wrong byte size"
            )));
        }
        Ok(object)
    }

    async fn load_catalog_tree(
        &self,
        table: &str,
        head: &CatalogHead,
        mut object_id: String,
        mut expected_size: Option<u64>,
    ) -> Result<LoadedCatalogTree, StatsError> {
        let mut chain = Vec::new();
        let mut deltas = Vec::new();
        let mut recovered_delta_bytes = 0_u64;
        let mut tip_shape = None;
        let mut pending_releases = Vec::new();
        let mut release_summary_complete = true;
        let mut legacy_history_safe_after_ms = 0;
        let checkpoint = loop {
            if chain.len() >= MAX_RECOVERY_NODES {
                return Err(StatsError::Internal(format!(
                    "catalog chain for table {table:?} exceeds {MAX_DELTA_DEPTH} deltas"
                )));
            }
            if chain.iter().any(|seen| seen == &object_id) {
                return Err(StatsError::Internal(format!(
                    "catalog chain for table {table:?} contains a cycle at {object_id:?}"
                )));
            }
            let object = self
                .read_table_object(table, &object_id, expected_size)
                .await?;
            let node: CatalogNode = serde_json::from_slice(&object.bytes).map_err(|error| {
                StatsError::Internal(format!(
                    "decode catalog node {object_id:?} for {table:?}: {error}"
                ))
            })?;
            validate_node(table, &node)?;
            pending_releases.extend(node.released_objects.iter().cloned());
            release_summary_complete &= node.release_summary.unwrap_or(false);
            legacy_history_safe_after_ms =
                legacy_history_safe_after_ms.max(node.legacy_history_safe_after_ms.unwrap_or(0));
            if chain.is_empty() && node.catalog_generation != head.catalog_generation {
                return Err(StatsError::Internal(format!(
                    "catalog tip does not match HEAD for table {table:?}"
                )));
            }
            let tip_depth = node.delta_depth.unwrap_or(0);
            let tip_bytes = node.delta_bytes_since_checkpoint.unwrap_or(0);
            tip_shape.get_or_insert((tip_depth, tip_bytes));
            chain.push(object_id.clone());
            if let Some(checkpoint) = node.checkpoint.as_option() {
                if node.parent.as_option().is_some()
                    || node.delta.as_option().is_some()
                    || tip_depth != 0
                    || tip_bytes != 0
                {
                    return Err(StatsError::Internal(format!(
                        "invalid checkpoint node {object_id:?} for table {table:?}"
                    )));
                }
                break checkpoint.clone();
            }
            let delta = node.delta.as_option().ok_or_else(|| {
                StatsError::Internal(format!(
                    "catalog node {object_id:?} for table {table:?} has no payload"
                ))
            })?;
            recovered_delta_bytes = recovered_delta_bytes.saturating_add(
                serde_json::to_vec(delta)
                    .map_err(|error| {
                        StatsError::Internal(format!(
                            "encode recovered catalog delta for {table:?}: {error}"
                        ))
                    })?
                    .len() as u64,
            );
            deltas.push((node.catalog_generation.unwrap_or(0), delta.clone()));
            let parent = node.parent.as_option().ok_or_else(|| {
                StatsError::Internal(format!(
                    "delta node {object_id:?} for table {table:?} has no parent"
                ))
            })?;
            expected_size = Some(catalog_reference_size(table, parent)?);
            object_id = parent.object_id.clone().ok_or_else(|| {
                StatsError::Internal(format!(
                    "delta node {object_id:?} for table {table:?} has no parent"
                ))
            })?;
        };
        let (tip_depth, tip_bytes) = tip_shape.unwrap_or_default();
        if usize::try_from(tip_depth).ok() != Some(deltas.len())
            || tip_bytes != recovered_delta_bytes
        {
            return Err(StatsError::Internal(format!(
                "catalog tip shape does not match its chain for table {table:?}"
            )));
        }
        let mut catalog = canonical_catalog(checkpoint);
        for (generation, delta) in deltas.into_iter().rev() {
            if generation <= catalog.catalog_generation.unwrap_or(0) {
                return Err(StatsError::Internal(format!(
                    "catalog generations do not increase along the chain for table {table:?}"
                )));
            }
            catalog = apply_delta(&catalog, &delta)?;
        }
        validate_state(table, head, &catalog)?;
        Ok(LoadedCatalogTree {
            catalog,
            chain,
            delta_depth: tip_depth,
            delta_bytes_since_checkpoint: tip_bytes,
            pending_releases: merge_released_objects(pending_releases),
            release_summary_complete,
            legacy_history_safe_after_ms,
        })
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

    /// Whether `table`'s root holds any catalog revision documents.
    pub async fn catalog_history_exists(&self, table: &str) -> Result<bool, StatsError> {
        Ok(!self.table_objects(table, STATES_PREFIX).await?.is_empty())
    }

    /// Tables whose root holds catalog history but no HEAD.
    ///
    /// The software never deletes HEAD, so this state is always external —
    /// human error or a bucket lifecycle rule — and each such table needs an
    /// operator to restore the known selected HEAD from deployment records or
    /// backup. Listing "the newest" node is unsafe because failed CAS attempts
    /// deliberately leave unselected siblings.
    pub async fn headless_tables(&self) -> Result<Vec<String>, StatsError> {
        let mut headless = Vec::new();
        for table in self.storage.list_tables().await? {
            if self.load_head(&table).await?.is_none()
                && self.catalog_history_exists(&table).await?
            {
                headless.push(table);
            }
        }
        Ok(headless)
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
                Some(&selected.head_version),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(StoredTableState {
            head,
            catalog: selected.catalog.clone(),
            head_version,
            catalog_chain: selected.catalog_chain.clone(),
            delta_depth: selected.delta_depth,
            delta_bytes_since_checkpoint: selected.delta_bytes_since_checkpoint,
            pending_releases: selected.pending_releases.clone(),
            release_summary_complete: selected.release_summary_complete,
            legacy_history_safe_after_ms: selected.legacy_history_safe_after_ms,
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
        let catalog = canonical_catalog(catalog);
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

        let delta = expected
            .map(|previous| catalog_delta(&previous.catalog, &catalog))
            .transpose()?;
        let delta_bytes = delta
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| {
                StatsError::Internal(format!("encode catalog delta for {table:?}: {error}"))
            })?
            .map(|bytes| bytes.len() as u64)
            .unwrap_or(0);
        let write_checkpoint = expected.is_none()
            || expected.is_some_and(|state| {
                state.head.format_version.unwrap_or(0) != TABLE_STATE_FORMAT_VERSION
                    || state.delta_depth.saturating_add(1) > MAX_DELTA_DEPTH
                    || state
                        .delta_bytes_since_checkpoint
                        .saturating_add(delta_bytes)
                        > MAX_DELTA_BYTES
            });
        // A definition change may shorten either retention policy while reads
        // or migration rollback still pin the parent. Release against the
        // longest policy on either side.
        let release_grace_ms = expected
            .map(|previous| catalog_retention_ms(&previous.catalog))
            .unwrap_or(0)
            .max(catalog_retention_ms(&catalog));
        let now_ms = now_ms();
        let delete_after_ms =
            now_ms.saturating_add(i64::try_from(release_grace_ms).unwrap_or(i64::MAX));
        let newly_released = expected
            .map(|previous| released_objects(&previous.catalog, &catalog, delete_after_ms))
            .unwrap_or_default();
        let mut pending_releases = expected
            .map(|previous| previous.pending_releases.clone())
            .unwrap_or_default();
        let referenced = referenced_object_keys(&catalog);
        if let Some(reentered) = pending_releases.iter().find_map(|released| {
            released
                .object
                .as_option()
                .and_then(|object| object.object_id.as_deref())
                .filter(|object_id| referenced.contains(*object_id))
        }) {
            return Err(StatsError::SchemaConflict(format!(
                "released object {reentered:?} cannot re-enter table {table:?}"
            )));
        }
        pending_releases.extend(newly_released.iter().cloned());
        pending_releases = merge_released_objects(pending_releases);
        let previous_summary_complete = expected
            .map(|previous| previous.release_summary_complete)
            .unwrap_or(true);
        let mut legacy_history_safe_after_ms = expected
            .map(|previous| previous.legacy_history_safe_after_ms)
            .unwrap_or(0);
        if !previous_summary_complete {
            if legacy_history_safe_after_ms <= now_ms {
                legacy_history_safe_after_ms = delete_after_ms;
            }
            for released in &mut pending_releases {
                released.delete_after_ms = Some(
                    released
                        .delete_after_ms
                        .unwrap_or(0)
                        .max(legacy_history_safe_after_ms),
                );
            }
        }
        // A checkpoint owns the complete pending set. A marked delta normally
        // carries only its mutation's releases, but the first marked node over
        // an old span also persists the conservatively extended inherited set.
        let node_releases = if write_checkpoint || !previous_summary_complete {
            pending_releases.clone()
        } else {
            newly_released
        };
        let (parent, checkpoint, delta, delta_depth, delta_bytes_since_checkpoint) =
            if write_checkpoint {
                (
                    buffa::MessageField::none(),
                    buffa::MessageField::some(catalog.clone()),
                    buffa::MessageField::none(),
                    0,
                    0,
                )
            } else {
                let previous = expected.expect("delta publication has a selected parent");
                let parent = previous.head.catalog.as_option().ok_or_else(|| {
                    StatsError::Internal(format!(
                        "selected HEAD for table {table:?} has no catalog reference"
                    ))
                })?;
                (
                    buffa::MessageField::some(logical_reference(parent)),
                    buffa::MessageField::none(),
                    buffa::MessageField::some(delta.expect("delta was built from the parent")),
                    previous.delta_depth + 1,
                    previous.delta_bytes_since_checkpoint + delta_bytes,
                )
            };
        let node = CatalogNode {
            format_version: Some(TABLE_STATE_FORMAT_VERSION),
            namespace: Some(table.to_string()),
            catalog_generation: Some(revision),
            parent,
            delta_depth: Some(delta_depth),
            delta_bytes_since_checkpoint: Some(delta_bytes_since_checkpoint),
            checkpoint,
            delta,
            released_objects: node_releases,
            release_summary: Some(true),
            legacy_history_safe_after_ms: Some(legacy_history_safe_after_ms),
            ..Default::default()
        };
        let state_bytes = serde_json::to_vec(&node).map_err(|error| {
            StatsError::Internal(format!("encode catalog node for {table:?}: {error}"))
        })?;
        let state_key = format!("{STATES_PREFIX}/{revision:020}-{}.json", Uuid::new_v4());
        let state_id = ObjectId::table(table, &state_key)?;
        self.storage
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
                byte_size: Some(state_bytes.len() as u64),
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
                expected.map(|state| &state.head_version),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(StoredTableState {
            head,
            catalog,
            head_version,
            catalog_chain: if write_checkpoint {
                vec![state_id.as_str().to_string()]
            } else {
                std::iter::once(state_id.as_str().to_string())
                    .chain(
                        expected
                            .expect("delta publication has a selected parent")
                            .catalog_chain
                            .iter()
                            .cloned(),
                    )
                    .collect()
            },
            delta_depth,
            delta_bytes_since_checkpoint,
            pending_releases,
            release_summary_complete: write_checkpoint || previous_summary_complete,
            legacy_history_safe_after_ms,
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
        let Some((head, head_version)) = self.load_head(table).await? else {
            if expected.is_some() {
                return Err(StatsError::SchemaConflict(format!(
                    "table {table:?} has no HEAD for the presented commit token"
                )));
            }
            return Ok(None);
        };
        let owner = WriterFence::new(head.writer_epoch.unwrap_or(0));
        if owner != fence {
            return Err(fenced_error(table, fence, owner));
        }
        if head.tombstoned.unwrap_or(false) {
            return Err(StatsError::SchemaConflict(format!(
                "table {table:?} was deleted at revision {}",
                head.catalog_generation.unwrap_or(0)
            )));
        }
        let Some(expected) = expected else {
            return Err(StatsError::SchemaConflict(format!(
                "table {table:?} already has HEAD at revision {}",
                head.catalog_generation.unwrap_or(0)
            )));
        };
        if expected.head != head {
            return Err(StatsError::SchemaConflict(format!(
                "table {table:?} changed HEAD value at revision {}",
                expected.revision(),
            )));
        }
        if !same_pointer_version(&expected.head_version, &head_version) {
            return Err(StatsError::SchemaConflict(format!(
                "table {table:?} changed HEAD version at revision {}: expected {:?}, found {:?}",
                expected.revision(),
                expected.head_version,
                head_version,
            )));
        }
        Ok(Some(expected.clone()))
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
                if !catalogs_equal(&current.catalog, &next) {
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

    /// Test entry point that reloads the selected state before collection.
    #[cfg(test)]
    pub(crate) async fn gc_obsolete_states(
        &self,
        table: &str,
        now_ms: i64,
        policy: StateGcPolicy,
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
        self.gc_selected(table, now_ms, policy, &selected)
            .await
            .map(|result| result.removed)
    }

    pub(crate) async fn gc_selected(
        &self,
        table: &str,
        now_ms: i64,
        policy: StateGcPolicy,
        selected: &StoredTableState,
    ) -> Result<StateGcResult, StatsError> {
        let started = std::time::Instant::now();
        let Some((head, head_version)) = self.load_head(table).await? else {
            return Ok(StateGcResult::default());
        };
        if selected.head != head || !same_pointer_version(&selected.head_version, &head_version) {
            tracing::debug!(table, "skipping object GC after HEAD changed");
            return Ok(StateGcResult::default());
        }
        if selected.fence() != WriterFence::new(head.writer_epoch.unwrap_or(0)) {
            tracing::warn!(table, "skipping object GC from a fenced writer");
            return Ok(StateGcResult::default());
        }
        if !selected.release_summary_complete {
            tracing::debug!(
                table,
                "skipping object GC until a checkpoint consolidates release history"
            );
            return Ok(StateGcResult::default());
        }
        let pin_cutoff =
            now_ms.saturating_sub(i64::try_from(policy.pin_retention_ms).unwrap_or(i64::MAX));
        let state_cutoff =
            now_ms.saturating_sub(i64::try_from(policy.state_retention_ms).unwrap_or(i64::MAX));
        let orphan_cutoff =
            now_ms.saturating_sub(i64::try_from(policy.orphan_grace_ms).unwrap_or(i64::MAX));
        let deletion_cutoff = pin_cutoff.min(state_cutoff);
        let current_chain = selected
            .catalog_chain
            .iter()
            .map(|object_id| {
                ObjectId::parse(object_id).and_then(|id| {
                    id.table_relative(table).map(str::to_string).ok_or_else(|| {
                        StatsError::Internal(format!(
                            "catalog node {object_id:?} escaped table {table:?}"
                        ))
                    })
                })
            })
            .collect::<Result<HashSet<_>, _>>()?;

        let list_started = std::time::Instant::now();
        let state_objects = self.table_objects(table, STATES_PREFIX).await?;
        let catalog_list_ms = list_started.elapsed().as_millis() as u64;
        let catalog_objects = state_objects.len();
        let mut referenced = referenced_object_keys(&selected.catalog);
        let expired_releases = selected
            .pending_releases
            .iter()
            .filter(|released| released.delete_after_ms.unwrap_or(i64::MAX) <= now_ms)
            .filter_map(|released| {
                released
                    .object
                    .as_option()
                    .and_then(|object| object.object_id.clone())
            })
            .filter(|object_id| !referenced.contains(object_id))
            .map(|object_id| ObjectId::parse(&object_id))
            .collect::<Result<Vec<_>, _>>()?;
        let release_delete_started = std::time::Instant::now();
        let released = self.storage.delete_many(expired_releases).await;
        let release_delete_ms = release_delete_started.elapsed().as_millis() as u64;
        if let Some(error) = released.error {
            tracing::warn!(table, %error, "released-object batch deletion was partial");
        }
        let deleted_releases = released
            .deleted
            .iter()
            .map(|id| id.as_str().to_string())
            .collect::<Vec<_>>();
        let deleted_release_set = deleted_releases.iter().cloned().collect::<HashSet<_>>();
        for release in &selected.pending_releases {
            let Some(object_id) = release
                .object
                .as_option()
                .and_then(|object| object.object_id.as_deref())
            else {
                continue;
            };
            if !deleted_release_set.contains(object_id) {
                referenced.insert(object_id.to_string());
            }
        }

        let legacy_history_protected = selected.legacy_history_safe_after_ms > now_ms;
        let obsolete_states = if legacy_history_protected {
            Vec::new()
        } else {
            state_objects
                .into_iter()
                .filter(|(key, metadata)| {
                    !current_chain.contains(key) && metadata.modified_at_ms <= deletion_cutoff
                })
                .map(|(key, _)| ObjectId::table(table, &key))
                .collect::<Result<Vec<_>, _>>()?
        };
        let catalog_delete_started = std::time::Instant::now();
        let deleted_states = self.storage.delete_many(obsolete_states).await;
        let catalog_delete_ms = catalog_delete_started.elapsed().as_millis() as u64;
        if let Some(error) = deleted_states.error {
            tracing::warn!(table, %error, "catalog-node batch deletion was partial");
        }

        let released_deleted = released.deleted.len();
        let catalogs_deleted = deleted_states.deleted.len();
        let mut removed = released_deleted + catalogs_deleted;
        let orphan_started = std::time::Instant::now();
        let mut orphans_examined = 0;
        let mut orphans_deleted = 0;
        if policy.sweep_orphans && !legacy_history_protected {
            let mut orphans = Vec::new();
            for prefix in [OBJECTS_PREFIX, INDICES_PREFIX, PROJECTIONS_PREFIX] {
                for (key, metadata) in self.table_objects(table, prefix).await? {
                    orphans_examined += 1;
                    let id = ObjectId::table(table, &key)?;
                    if referenced.contains(id.as_str()) || metadata.modified_at_ms > orphan_cutoff {
                        continue;
                    }
                    orphans.push(id);
                }
            }
            let deleted_orphans = self.storage.delete_many(orphans).await;
            if let Some(error) = deleted_orphans.error {
                tracing::warn!(table, %error, "orphan-object batch deletion was partial");
            }
            orphans_deleted = deleted_orphans.deleted.len();
            removed += orphans_deleted;
        }
        let orphan_ms = orphan_started.elapsed().as_millis() as u64;
        let total_ms = started.elapsed().as_millis() as u64;
        if removed > 0 || total_ms >= 1_000 {
            tracing::info!(
                table,
                catalog_objects,
                selected_chain_nodes = current_chain.len(),
                historical_nodes_opened = 0,
                pending_releases = selected.pending_releases.len(),
                released_deleted,
                catalogs_deleted,
                orphans_examined,
                orphans_deleted,
                catalog_list_ms,
                release_delete_ms,
                catalog_delete_ms,
                orphan_ms,
                total_ms,
                "collected object table state"
            );
        }
        Ok(StateGcResult {
            removed,
            deleted_releases,
        })
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

fn same_pointer_version(left: &ObjectVersion, right: &ObjectVersion) -> bool {
    match (&left.local_value, &right.local_value) {
        (Some(left), Some(right)) => left == right,
        _ => {
            left.e_tag == right.e_tag
                && left.provider_version == right.provider_version
                && left.byte_size == right.byte_size
        }
    }
}

fn referenced_object_keys(catalog: &NamespaceCatalog) -> std::collections::HashSet<String> {
    referenced_objects(catalog)
        .into_iter()
        .filter_map(|reference| reference.object_id)
        .collect()
}

fn catalog_retention_ms(catalog: &NamespaceCatalog) -> u64 {
    catalog
        .max_query_time_ms
        .unwrap_or(0)
        .max(catalog.rollback_window_ms.unwrap_or(0))
}

fn merge_released_objects(
    releases: impl IntoIterator<Item = ReleasedObject>,
) -> Vec<ReleasedObject> {
    let mut by_id = HashMap::new();
    for release in releases {
        let object_id = release
            .object
            .as_option()
            .and_then(|object| object.object_id.clone())
            .expect("validated released objects have an ID");
        match by_id.entry(object_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(release);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if release.delete_after_ms.unwrap_or(0) > entry.get().delete_after_ms.unwrap_or(0) {
                    entry.insert(release);
                }
            }
        }
    }
    let mut releases = by_id.into_values().collect::<Vec<_>>();
    releases.sort_by(|left, right| {
        left.object
            .as_option()
            .and_then(|object| object.object_id.as_deref())
            .cmp(
                &right
                    .object
                    .as_option()
                    .and_then(|object| object.object_id.as_deref()),
            )
    });
    releases
}

fn validate_head(table: &str, head: &CatalogHead) -> Result<(), StatsError> {
    if !matches!(
        head.format_version.unwrap_or(0),
        LEGACY_TABLE_STATE_FORMAT_VERSION | TABLE_STATE_FORMAT_VERSION
    ) || head.namespace.as_deref() != Some(table)
        || head.catalog_generation.unwrap_or(0) == 0
        || head
            .catalog
            .as_option()
            .and_then(|reference| reference.object_id.as_deref())
            .is_none()
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
    if catalog.format_version != head.format_version
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

fn validate_node(table: &str, node: &CatalogNode) -> Result<(), StatsError> {
    if node.format_version.unwrap_or(0) != TABLE_STATE_FORMAT_VERSION
        || node.namespace.as_deref() != Some(table)
        || node.catalog_generation.unwrap_or(0) == 0
        || node.delta_depth.unwrap_or(0) > MAX_DELTA_DEPTH
        || node.delta_bytes_since_checkpoint.unwrap_or(0) > MAX_DELTA_BYTES
    {
        return Err(StatsError::Internal(format!(
            "invalid catalog node for table {table:?}"
        )));
    }
    if node.released_objects.iter().any(|released| {
        released.delete_after_ms.unwrap_or(0) <= 0
            || released
                .object
                .as_option()
                .and_then(|object| object.object_id.as_deref())
                .is_none()
    }) {
        return Err(StatsError::Internal(format!(
            "invalid released object in catalog node for table {table:?}"
        )));
    }
    Ok(())
}

fn state_object_id(table: &str, head: &CatalogHead) -> Result<String, StatsError> {
    head.catalog
        .as_option()
        .and_then(|reference| reference.object_id.clone())
        .filter(|object_id| !object_id.is_empty())
        .ok_or_else(|| {
            StatsError::Internal(format!("object HEAD for {table:?} has no state reference"))
        })
}

fn catalog_reference_size(table: &str, reference: &ObjectRef) -> Result<u64, StatsError> {
    reference.byte_size.filter(|size| *size > 0).ok_or_else(|| {
        StatsError::Internal(format!(
            "catalog reference for table {table:?} has no byte size"
        ))
    })
}

#[cfg(test)]
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
    use crate::proto::finelog::stats::{CatalogSegment, TableVersionSegments};
    use crate::store::object_store::build_remote_object_store;
    use crate::test_support::{FaultInjectingObjectStore, ObjectOp};

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

    fn segment(number: usize) -> CatalogSegment {
        let id = format!("segment-{number:04}.parquet");
        CatalogSegment {
            segment_id: Some(id.clone()),
            source: buffa::MessageField::some(ObjectRef {
                object_id: Some(format!("_finelog/tables/{TABLE}/objects/v1/l0/{id}")),
                byte_size: Some(100),
                ..Default::default()
            }),
            table_spec_version: Some(1),
            ..Default::default()
        }
    }

    fn state_with_segments(revision: u64, segments: Vec<CatalogSegment>) -> NamespaceCatalog {
        let mut catalog = state(TABLE, revision, 1);
        catalog.version_segments = vec![TableVersionSegments {
            table_spec_version: Some(1),
            live_segments: segments.clone(),
            ..Default::default()
        }];
        catalog.direct_query_segments = segments;
        catalog
    }

    fn store(tag: &str) -> (ObjectTableStateStore, std::path::PathBuf) {
        let remote_dir = crate::test_support::unique_dir(tag);
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        (ObjectTableStateStore::new(Arc::new(remote)), remote_dir)
    }

    #[test]
    fn state_decoder_ignores_removed_sha_field() {
        let catalog: NamespaceCatalog =
            serde_json::from_str(r#"{"directQuerySegments":[{"source":{"sha256":"bGVnYWN5"}}]}"#)
                .unwrap();

        assert_eq!(catalog.direct_query_segments.len(), 1);
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
    async fn ordinary_commit_writes_one_small_typed_delta_with_the_exact_parent() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_delta_shape");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let original_segments = (0..100).map(segment).collect::<Vec<_>>();
        let first = states
            .commit(
                TABLE,
                WriterFence::new(4),
                None,
                state_with_segments(1, original_segments.clone()),
            )
            .await
            .unwrap();
        let mut next_segments = original_segments;
        next_segments.push(segment(100));
        let second = states
            .commit(
                TABLE,
                WriterFence::new(4),
                Some(&first),
                state_with_segments(2, next_segments),
            )
            .await
            .unwrap();

        let first_object = remote
            .read(&ObjectId::parse(&first.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let second_object = remote
            .read(&ObjectId::parse(&second.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let node: CatalogNode = serde_json::from_slice(&second_object.bytes).unwrap();
        assert!(node.checkpoint.as_option().is_none());
        assert_eq!(node.delta.as_option().unwrap().segment_additions.len(), 1);
        assert_eq!(
            node.delta.as_option().unwrap().direct_query_additions.len(),
            1
        );
        assert_eq!(
            node.parent
                .as_option()
                .and_then(|parent| parent.object_id.as_deref()),
            Some(first.catalog_chain[0].as_str())
        );
        assert!(second_object.bytes.len() * 4 < first_object.bytes.len());
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().catalog,
            second.catalog
        );
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn a_large_delta_becomes_a_checkpoint() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_byte_checkpoint");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let first = states
            .commit(TABLE, WriterFence::new(4), None, state(TABLE, 1, 1))
            .await
            .unwrap();
        let mut large = segment(1);
        large.min_key_value = Some("x".repeat(MAX_DELTA_BYTES as usize));
        let second = states
            .commit(
                TABLE,
                WriterFence::new(4),
                Some(&first),
                state_with_segments(7, vec![large]),
            )
            .await
            .unwrap();

        assert_eq!(
            second.revision().get(),
            7,
            "coalesced generations are valid"
        );
        assert_eq!(second.catalog_chain.len(), 1);
        let object = remote
            .read(&ObjectId::parse(&second.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let node: CatalogNode = serde_json::from_slice(&object.bytes).unwrap();
        assert!(node.checkpoint.as_option().is_some());
        assert!(node.delta.as_option().is_none());
        assert_eq!(node.delta_depth, Some(0));
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            7
        );
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn release_grace_uses_the_longer_parent_query_lifetime() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_parent_query_grace");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let first = states
            .commit(
                TABLE,
                WriterFence::new(4),
                None,
                state_with_segments(1, vec![segment(1)]),
            )
            .await
            .unwrap();
        let mut next = state_with_segments(2, vec![]);
        next.max_query_time_ms = Some(1);
        let started = now_ms();
        let second = states
            .commit(TABLE, WriterFence::new(4), Some(&first), next)
            .await
            .unwrap();
        let object = remote
            .read(&ObjectId::parse(&second.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let node: CatalogNode = serde_json::from_slice(&object.bytes).unwrap();
        assert_eq!(node.released_objects.len(), 1);
        assert!(node.released_objects[0].delete_after_ms.unwrap() >= started + 600_000);
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn released_object_survives_its_query_grace_then_becomes_collectible() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_release_grace");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let source = segment(1)
            .source
            .as_option()
            .and_then(|source| source.object_id.as_deref())
            .unwrap()
            .to_string();
        let source_id = ObjectId::parse(&source).unwrap();
        remote
            .write(&source_id, Bytes::from_static(b"parquet"))
            .await
            .unwrap();
        let first = states
            .commit(
                TABLE,
                WriterFence::new(4),
                None,
                state_with_segments(1, vec![segment(1)]),
            )
            .await
            .unwrap();
        states
            .commit(
                TABLE,
                WriterFence::new(4),
                Some(&first),
                state_with_segments(2, vec![]),
            )
            .await
            .unwrap();
        let policy = StateGcPolicy {
            pin_retention_ms: 0,
            state_retention_ms: u64::MAX,
            orphan_grace_ms: 0,
            sweep_orphans: true,
        };

        states
            .gc_obsolete_states(TABLE, now_ms() + 1, policy, WriterFence::new(4))
            .await
            .unwrap();
        assert!(remote.read(&source_id).await.unwrap().is_some());
        states
            .gc_obsolete_states(TABLE, i64::MAX, policy, WriterFence::new(4))
            .await
            .unwrap();
        assert!(remote.read(&source_id).await.unwrap().is_none());
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn first_v2_commit_upgrades_a_v1_head_with_a_checkpoint() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_v1_upgrade");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let legacy_catalog = NamespaceCatalog {
            format_version: Some(LEGACY_TABLE_STATE_FORMAT_VERSION),
            namespace: Some(TABLE.to_string()),
            catalog_generation: Some(1),
            active_table_spec_version: Some(1),
            ..Default::default()
        };
        let legacy_id = ObjectId::table(TABLE, "catalogs/00000000000000000001-v1.json").unwrap();
        let legacy_bytes = serde_json::to_vec(&legacy_catalog).unwrap();
        remote
            .write(&legacy_id, Bytes::from(legacy_bytes.clone()))
            .await
            .unwrap();
        let legacy_head = CatalogHead {
            format_version: Some(LEGACY_TABLE_STATE_FORMAT_VERSION),
            namespace: Some(TABLE.to_string()),
            writer_epoch: Some(9),
            catalog_generation: Some(1),
            active_table_spec_version: Some(1),
            catalog: buffa::MessageField::some(ObjectRef {
                object_id: Some(legacy_id.as_str().to_string()),
                byte_size: Some(legacy_bytes.len() as u64),
                ..Default::default()
            }),
            ..Default::default()
        };
        remote
            .compare_and_swap(
                &ObjectId::table(TABLE, HEAD_KEY).unwrap(),
                None,
                Bytes::from(serde_json::to_vec(&legacy_head).unwrap()),
            )
            .await
            .unwrap();

        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.head.format_version, Some(1));
        let upgraded = states
            .commit(
                TABLE,
                WriterFence::new(9),
                Some(&selected),
                state(TABLE, 2, 1),
            )
            .await
            .unwrap();

        assert_eq!(
            upgraded.head.format_version,
            Some(TABLE_STATE_FORMAT_VERSION)
        );
        assert_eq!(upgraded.catalog_chain.len(), 1);
        let tip = remote
            .read(&ObjectId::parse(&upgraded.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let node: CatalogNode = serde_json::from_slice(&tip.bytes).unwrap();
        assert!(node.checkpoint.as_option().is_some());
        assert!(node.parent.as_option().is_none());
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn publication_after_an_unmarked_delta_rearms_legacy_history_protection() {
        let remote_dir = crate::test_support::unique_dir("object_state_old_writer_roll_forward");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let fence = WriterFence::new(9);
        let first = states
            .commit(TABLE, fence, None, state(TABLE, 1, 1))
            .await
            .unwrap();

        let second_catalog = state(TABLE, 2, 1);
        let delta = catalog_delta(&first.catalog, &second_catalog).unwrap();
        let delta_bytes = serde_json::to_vec(&delta).unwrap().len() as u64;
        let node = CatalogNode {
            format_version: Some(TABLE_STATE_FORMAT_VERSION),
            namespace: Some(TABLE.to_string()),
            catalog_generation: Some(2),
            parent: buffa::MessageField::some(logical_reference(
                first.head.catalog.as_option().unwrap(),
            )),
            delta_depth: Some(1),
            delta_bytes_since_checkpoint: Some(delta_bytes),
            delta: buffa::MessageField::some(delta),
            ..Default::default()
        };
        let node_bytes = serde_json::to_vec(&node).unwrap();
        let node_id = ObjectId::table(TABLE, "catalogs/00000000000000000002-old.json").unwrap();
        remote
            .write(&node_id, Bytes::from(node_bytes.clone()))
            .await
            .unwrap();
        let mut head = first.head.clone();
        head.catalog_generation = Some(2);
        head.catalog = buffa::MessageField::some(ObjectRef {
            object_id: Some(node_id.as_str().to_string()),
            byte_size: Some(node_bytes.len() as u64),
            ..Default::default()
        });
        remote
            .compare_and_swap(
                &ObjectId::table(TABLE, HEAD_KEY).unwrap(),
                Some(&first.head_version),
                Bytes::from(serde_json::to_vec(&head).unwrap()),
            )
            .await
            .unwrap();

        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert!(!selected.release_summary_complete);
        let started = now_ms();
        let rolled_forward = states
            .commit(TABLE, fence, Some(&selected), state(TABLE, 3, 1))
            .await
            .unwrap();
        assert!(!rolled_forward.release_summary_complete);
        assert!(rolled_forward.legacy_history_safe_after_ms >= started + 600_000);
        let tip = remote
            .read(&ObjectId::parse(&rolled_forward.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let tip: CatalogNode = serde_json::from_slice(&tip.bytes).unwrap();
        assert!(tip.release_summary.unwrap());
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn recovery_rejects_a_tip_whose_exact_parent_is_missing() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_missing_parent");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let first = states
            .commit(TABLE, WriterFence::new(3), None, state(TABLE, 1, 1))
            .await
            .unwrap();
        let second = states
            .commit(TABLE, WriterFence::new(3), Some(&first), state(TABLE, 2, 1))
            .await
            .unwrap();
        assert_eq!(second.catalog_chain.len(), 2);
        remote
            .delete(&ObjectId::parse(&second.catalog_chain[1]).unwrap())
            .await
            .unwrap();

        states
            .load(TABLE)
            .await
            .expect_err("missing parent must fail recovery");
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
                .gc_obsolete_states(
                    TABLE,
                    current_modified_ms + 5,
                    StateGcPolicy {
                        pin_retention_ms: 10,
                        state_retention_ms: 10,
                        orphan_grace_ms: 10,
                        sweep_orphans: true,
                    },
                    fence,
                )
                .await
                .unwrap(),
            0
        );
        let future = i64::MAX;
        // A fenced writer collects nothing.
        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    future,
                    StateGcPolicy {
                        pin_retention_ms: 0,
                        state_retention_ms: 0,
                        orphan_grace_ms: 0,
                        sweep_orphans: true,
                    },
                    WriterFence::new(12),
                )
                .await
                .unwrap(),
            0
        );
        assert_eq!(states.state_keys(TABLE).await.unwrap().len(), 2);
        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    future,
                    StateGcPolicy {
                        pin_retention_ms: 600_000,
                        state_retention_ms: 600_000,
                        orphan_grace_ms: 600_000,
                        sweep_orphans: true,
                    },
                    fence,
                )
                .await
                .unwrap(),
            0
        );
        // Revision two is a delta over revision one, so both nodes remain
        // reachable from HEAD until an automatic checkpoint cuts the chain.
        assert_eq!(states.state_keys(TABLE).await.unwrap().len(), 2);
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

        // Without the orphan sweep the unreferenced object survives.
        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    i64::MAX,
                    StateGcPolicy {
                        pin_retention_ms: 600_000,
                        state_retention_ms: 600_000,
                        orphan_grace_ms: 600_000,
                        sweep_orphans: false,
                    },
                    WriterFence::new(1),
                )
                .await
                .unwrap(),
            0
        );
        assert!(remote.read(&orphan_id).await.unwrap().is_some());
        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    i64::MAX,
                    StateGcPolicy {
                        pin_retention_ms: 600_000,
                        state_retention_ms: 600_000,
                        orphan_grace_ms: 600_000,
                        sweep_orphans: true,
                    },
                    WriterFence::new(1),
                )
                .await
                .unwrap(),
            1
        );
        assert!(remote.read(&orphan_id).await.unwrap().is_none());
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn garbage_collection_deletes_a_known_release_only_after_rollback_retention() {
        let remote_dir = crate::test_support::unique_dir("object_state_known_release_gc");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let fence = WriterFence::new(1);
        let released = segment(1);
        let object_id = ObjectId::parse(
            released
                .source
                .as_option()
                .unwrap()
                .object_id
                .as_deref()
                .unwrap(),
        )
        .unwrap();
        remote
            .write(&object_id, Bytes::from_static(b"released"))
            .await
            .unwrap();
        let mut first_state = state_with_segments(1, vec![released]);
        first_state.rollback_window_ms = Some(3_600_000);
        let first = states
            .commit(TABLE, fence, None, first_state)
            .await
            .unwrap();
        let mut second_state = state_with_segments(2, vec![]);
        second_state.rollback_window_ms = Some(3_600_000);
        let second = states
            .commit(TABLE, fence, Some(&first), second_state)
            .await
            .unwrap();
        let release_deadline = second.pending_releases[0].delete_after_ms.unwrap();

        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    release_deadline - 1,
                    StateGcPolicy {
                        pin_retention_ms: 0,
                        state_retention_ms: 0,
                        orphan_grace_ms: 0,
                        sweep_orphans: false,
                    },
                    fence,
                )
                .await
                .unwrap(),
            0
        );
        assert!(remote.read(&object_id).await.unwrap().is_some());

        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    release_deadline + 1,
                    StateGcPolicy {
                        pin_retention_ms: 0,
                        state_retention_ms: 0,
                        orphan_grace_ms: u64::MAX,
                        sweep_orphans: false,
                    },
                    fence,
                )
                .await
                .unwrap(),
            1
        );
        assert!(remote.read(&object_id).await.unwrap().is_none());
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn garbage_collection_reclaims_history_behind_a_checkpoint() {
        let remote_dir = crate::test_support::unique_dir("object_state_store_gc_thin");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let fence = WriterFence::new(7);
        let mut previous = None;
        for revision in 1..=66 {
            let committed = states
                .commit(TABLE, fence, previous.as_ref(), state(TABLE, revision, 1))
                .await
                .unwrap();
            previous = Some(committed);
        }
        // Pin the state files to known modification times so each revision's
        // obsolete-at instant (the next revision's write) is deterministic.
        let base_ms: i64 = 1_000_000_000_000;
        for metadata in remote
            .list(&ObjectPrefix::table(TABLE, STATES_PREFIX).unwrap())
            .await
            .unwrap()
        {
            let revision =
                state_revision_from_key(metadata.id.table_relative(TABLE).unwrap()).unwrap();
            let modified = std::time::UNIX_EPOCH
                + std::time::Duration::from_millis((base_ms + revision as i64 * 1_000) as u64);
            let file = std::fs::OpenOptions::new()
                .append(true)
                .open(remote_dir.join(metadata.id.as_str()))
                .unwrap();
            file.set_modified(modified).unwrap();
        }
        // Revision 66 is the automatic checkpoint after 64 deltas. Its parent
        // chain is no longer needed for recovery and becomes collectible once
        // both the pin and rollback windows expire.
        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    i64::MAX,
                    StateGcPolicy {
                        pin_retention_ms: 0,
                        state_retention_ms: 0,
                        orphan_grace_ms: 0,
                        sweep_orphans: false,
                    },
                    fence,
                )
                .await
                .unwrap(),
            65
        );
        let keys = states.state_keys(TABLE).await.unwrap();
        assert_eq!(keys.len(), 1);
        assert!(keys
            .iter()
            .any(|key| state_revision_from_key(key) == Some(66)));
        // The selected checkpoint survives a repeated pass.
        assert_eq!(
            states
                .gc_obsolete_states(
                    TABLE,
                    i64::MAX,
                    StateGcPolicy {
                        pin_retention_ms: 0,
                        state_retention_ms: 0,
                        orphan_grace_ms: 0,
                        sweep_orphans: false,
                    },
                    fence,
                )
                .await
                .unwrap(),
            0
        );
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn checkpoint_carries_releases_from_the_previous_span() {
        let remote_dir = crate::test_support::unique_dir("object_state_release_checkpoint");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote.clone()));
        let fence = WriterFence::new(2);
        let first = states
            .commit(TABLE, fence, None, state_with_segments(1, vec![segment(1)]))
            .await
            .unwrap();
        let mut selected = states
            .commit(TABLE, fence, Some(&first), state_with_segments(2, vec![]))
            .await
            .unwrap();
        for revision in 3..=66 {
            selected = states
                .commit(
                    TABLE,
                    fence,
                    Some(&selected),
                    state_with_segments(revision, vec![]),
                )
                .await
                .unwrap();
        }

        assert_eq!(selected.catalog_chain.len(), 1);
        assert_eq!(selected.pending_releases.len(), 1);
        let checkpoint = remote
            .read(&ObjectId::parse(&selected.catalog_chain[0]).unwrap())
            .await
            .unwrap()
            .unwrap();
        let checkpoint: CatalogNode = serde_json::from_slice(&checkpoint.bytes).unwrap();
        assert_eq!(checkpoint.released_objects.len(), 1);
        assert!(checkpoint.release_summary.unwrap());
        assert_eq!(
            states
                .load(TABLE)
                .await
                .unwrap()
                .unwrap()
                .pending_releases
                .len(),
            1
        );
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn checkpoint_persists_locally_pruned_releases() {
        let remote_dir = crate::test_support::unique_dir("object_state_pruned_checkpoint");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let states = ObjectTableStateStore::new(Arc::new(remote));
        let fence = WriterFence::new(3);
        let first = states
            .commit(TABLE, fence, None, state_with_segments(1, vec![segment(1)]))
            .await
            .unwrap();
        let mut selected = states
            .commit(TABLE, fence, Some(&first), state_with_segments(2, vec![]))
            .await
            .unwrap();
        assert_eq!(selected.pending_releases.len(), 1);
        selected.pending_releases.clear();
        for revision in 3..=66 {
            selected = states
                .commit(
                    TABLE,
                    fence,
                    Some(&selected),
                    state_with_segments(revision, vec![]),
                )
                .await
                .unwrap();
        }

        assert_eq!(selected.catalog_chain.len(), 1);
        assert!(selected.pending_releases.is_empty());
        assert!(states
            .load(TABLE)
            .await
            .unwrap()
            .unwrap()
            .pending_releases
            .is_empty());
        std::fs::remove_dir_all(remote_dir).ok();
    }

    #[tokio::test]
    async fn collection_never_reads_catalog_nodes_outside_the_selected_chain() {
        let remote_dir = crate::test_support::unique_dir("object_state_metadata_only_gc");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let faults = FaultInjectingObjectStore::new(remote);
        let states = ObjectTableStateStore::new(Arc::clone(&faults) as Arc<dyn ObjectStore>);
        let fence = WriterFence::new(4);
        let selected = states
            .commit(TABLE, fence, None, state(TABLE, 1, 1))
            .await
            .unwrap();
        for revision in 2..=315 {
            let stale =
                ObjectId::table(TABLE, &format!("catalogs/{revision:020}-stale.json")).unwrap();
            faults
                .write(&stale, Bytes::from_static(b"not decoded"))
                .await
                .unwrap();
        }
        faults.clear_calls();

        let result = states
            .gc_selected(
                TABLE,
                i64::MAX,
                StateGcPolicy {
                    pin_retention_ms: 0,
                    state_retention_ms: 0,
                    orphan_grace_ms: 0,
                    sweep_orphans: false,
                },
                &selected,
            )
            .await
            .unwrap();

        assert_eq!(result.removed, 314);
        assert!(faults
            .keys_for(ObjectOp::Read)
            .iter()
            .all(|key| !key.contains("/catalogs/")));
        std::fs::remove_dir_all(remote_dir).ok();
    }
}
