//! Object-backed table controller.
//!
//! `Namespace` owns buffering and query state. This controller owns canonical
//! object writes, local materialization, and the durable state commits of one
//! table, which it makes under this process's writer fence.
//! Transitional migration code receives the legacy layout only as an opaque
//! [`ObjectStore`] implementation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::ObjectRef;
use crate::store::catalog::projection::namespace_catalog;
use crate::store::catalog::state_store::{StoredTableState, TableStateStore};
use crate::store::catalog::{Catalog, ObjectSegmentRecord, TableSpecStatus};
use crate::store::object_store::{ObjectId, ObjectReference, ObjectStore};
use crate::store::segment::segment_bounds;
use crate::store::table_state::{
    resolve_publication, CommitError, TableSnapshot, TableState, WriterFence,
};
use crate::store::types::{segment_relative_key, LocalSegment, SegmentLocation, SegmentRow};

pub struct WrittenObject {
    pub path: PathBuf,
    pub source: ObjectRef,
    pub byte_size: i64,
}

pub struct ObjectTable {
    table: String,
    table_dir: PathBuf,
    catalog: Arc<Catalog>,
    store: Arc<dyn ObjectStore>,
    legacy_store: Arc<dyn ObjectStore>,
    state_store: Arc<dyn TableStateStore>,
    fence: WriterFence,
    /// The state this writer last observed selected, and the commit token it
    /// presents. Absent until this process claims the table.
    selected: std::sync::Mutex<Option<StoredTableState>>,
    /// Set once this process owns durable writes for the table.
    claimed: AtomicBool,
    /// Cleared when another writer fences this one. A fenced table stops
    /// accepting writes; only a restart can re-claim it.
    writes_ready: AtomicBool,
    /// Set while a locally committed revision is not known to be published.
    publication_owed: AtomicBool,
}

impl ObjectTable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table: String,
        table_dir: PathBuf,
        catalog: Arc<Catalog>,
        store: Arc<dyn ObjectStore>,
        legacy_store: Arc<dyn ObjectStore>,
        state_store: Arc<dyn TableStateStore>,
        fence: WriterFence,
    ) -> Self {
        Self {
            table,
            table_dir,
            catalog,
            store,
            legacy_store,
            state_store,
            fence,
            selected: std::sync::Mutex::new(None),
            claimed: AtomicBool::new(false),
            writes_ready: AtomicBool::new(true),
            publication_owed: AtomicBool::new(false),
        }
    }

    pub fn legacy_store(&self) -> Arc<dyn ObjectStore> {
        Arc::clone(&self.legacy_store)
    }

    pub fn writer_fence(&self) -> WriterFence {
        self.fence
    }

    /// Whether this writer still owns the table's durable state.
    pub fn writes_ready(&self) -> bool {
        self.writes_ready.load(Ordering::SeqCst)
    }

    /// Stop accepting writes for this table until a restart recovers it.
    pub fn mark_unready(&self, reason: &str) {
        self.writes_ready.store(false, Ordering::SeqCst);
        tracing::error!(
            table = %self.table,
            fence = %self.fence,
            reason,
            "table stops accepting writes"
        );
    }

    /// Take ownership of durable writes before the first commit.
    ///
    /// A table with a published HEAD is claimed by swapping its recorded fence.
    /// A table that was never published is claimed by its first commit, whose
    /// HEAD creation fails if another writer got there first.
    pub async fn claim_writer(&self) -> Result<(), StatsError> {
        if self.claimed.load(Ordering::SeqCst) {
            return Ok(());
        }
        if let Some(published) = self.state_store.load(&self.table).await? {
            let claimed = self
                .state_store
                .claim_writer(&self.table, self.fence, &published)
                .await?;
            *self.selected.lock().unwrap() = Some(claimed);
        }
        self.claimed.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Record that a committed revision must still reach HEAD.
    pub fn mark_publication_owed(&self) {
        self.publication_owed.store(true, Ordering::SeqCst);
    }

    pub fn publication_owed(&self) -> bool {
        self.publication_owed.load(Ordering::SeqCst)
    }

    /// Publish an owed revision. A table with nothing owed is already current.
    pub async fn publish_owed(&self) -> Result<(), StatsError> {
        if !self.publication_owed() {
            return Ok(());
        }
        self.publish_state().await?;
        Ok(())
    }

    /// Publish the current local table state and settle the outcome.
    ///
    /// Returns the state HEAD selects, whose revision is at least the caller's
    /// committed revision. Publication stays owed until HEAD is known
    /// to name that state, so the same revision is published again rather than
    /// undone. A fenced table stops owing publication: this writer must not
    /// overwrite the state another writer selected.
    pub async fn publish_state(&self) -> Result<TableSnapshot, CommitError> {
        self.mark_publication_owed();
        self.claim_writer()
            .await
            .map_err(CommitError::PublicationDeferred)?;
        let state = TableState::new(
            namespace_catalog(&self.catalog, &self.table, &self.table_dir)
                .map_err(CommitError::PublicationDeferred)?,
        );
        let expected = self.selected.lock().unwrap().clone();
        let outcome = self
            .state_store
            .commit(
                &self.table,
                self.fence,
                expected.as_ref(),
                state.catalog().clone(),
            )
            .await;
        let published = match outcome {
            Ok(committed) => {
                let snapshot = TableSnapshot::from_stored(&committed);
                *self.selected.lock().unwrap() = Some(committed);
                snapshot
            }
            Err(error) => match self.resolve_lost_publication(&state, error).await {
                Ok(published) => published,
                Err(error) => {
                    if matches!(error, CommitError::Fenced(_)) {
                        self.publication_owed.store(false, Ordering::SeqCst);
                        self.mark_unready(&error.to_string());
                    }
                    return Err(error);
                }
            },
        };
        self.publication_owed.store(false, Ordering::SeqCst);
        Ok(published)
    }

    /// Settle a commit whose outcome the state store did not report against the
    /// state HEAD now selects.
    async fn resolve_lost_publication(
        &self,
        attempted: &TableState,
        error: StatsError,
    ) -> Result<TableSnapshot, CommitError> {
        let published = self
            .state_store
            .load(&self.table)
            .await
            .map_err(|head_error| {
                CommitError::PublicationDeferred(StatsError::AmbiguousCommit(format!(
                    "committing {:?} failed with {error}; reading HEAD to resolve it failed with {head_error}",
                    self.table
                )))
            })?;
        let snapshot = published.as_ref().map(TableSnapshot::from_stored);
        let resolved =
            resolve_publication(&self.table, attempted, self.fence, snapshot.as_ref(), error)?;
        *self.selected.lock().unwrap() = published;
        Ok(resolved)
    }

    pub async fn write_parquet(&self, bytes: Bytes) -> Result<WrittenObject, StatsError> {
        let sha256: [u8; 32] = Sha256::digest(&bytes).into();
        let id = ObjectId::table(
            &self.table,
            &format!("objects/{}.parquet", full_hex(&sha256)),
        )?;
        let version = self.store.write(&id, bytes).await?;
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        let path = self.store.local_path(&reference).await?;
        Ok(WrittenObject {
            path,
            source: ObjectRef {
                object_id: Some(id.as_str().to_string()),
                provider_version: version.provider_version,
                etag: version.e_tag,
                byte_size: Some(version.byte_size),
                sha256: Some(version.content_sha256.to_vec()),
                ..Default::default()
            },
            byte_size: i64::try_from(version.byte_size).unwrap_or(i64::MAX),
        })
    }

    pub async fn source_bytes(
        &self,
        row: &SegmentRow,
        object_record: Option<&ObjectSegmentRecord>,
    ) -> Result<Bytes, StatsError> {
        if let Some(record) = object_record {
            let reference = ObjectReference::try_from(&record.source)?;
            reference.id.table_relative(&self.table).ok_or_else(|| {
                StatsError::Internal(format!(
                    "object migration source {:?} belongs to another table",
                    reference.id.as_str()
                ))
            })?;
            let object = self.store.read(&reference.id).await?.ok_or_else(|| {
                StatsError::Internal(format!(
                    "object migration source {:?} is missing for {:?}",
                    reference.id.as_str(),
                    self.table
                ))
            })?;
            if object.version.content_sha256 != reference.version.content_sha256 {
                return Err(StatsError::Internal(format!(
                    "object migration source {:?} failed SHA-256 validation",
                    reference.id.as_str()
                )));
            }
            return Ok(object.bytes);
        }
        if Path::new(&row.path).exists() {
            return tokio::fs::read(&row.path)
                .await
                .map(Bytes::from)
                .map_err(|error| {
                    StatsError::Internal(format!("read migration source {}: {error}", row.path))
                });
        }
        let key = segment_relative_key(&self.table_dir, &row.path).ok_or_else(|| {
            StatsError::Internal(format!(
                "legacy migration source {} is outside {}",
                row.path,
                self.table_dir.display()
            ))
        })?;
        self.legacy_store
            .read(&ObjectId::table(&self.table, &key)?)
            .await?
            .map(|object| object.bytes)
            .ok_or_else(|| {
                StatsError::Internal(format!(
                    "legacy migration source {key:?} is missing for {:?}",
                    self.table
                ))
            })
    }

    pub async fn local_query_segments(
        &self,
        key_column: &str,
    ) -> Result<Vec<LocalSegment>, StatsError> {
        let status = self.catalog.table_spec_status(&self.table)?;
        let rows: HashMap<_, _> = self
            .catalog
            .list_segments(&self.table)?
            .into_iter()
            .map(|row| (row.path.clone(), row))
            .collect();
        let mut restored = Vec::new();
        for record in self.catalog.object_segments(&self.table)? {
            if !object_segment_is_query_visible(&status, &record) {
                continue;
            }
            let row = rows.get(&record.path).ok_or_else(|| {
                StatsError::Internal(format!(
                    "object segment {:?} has no local catalog row",
                    record.path
                ))
            })?;
            let reference = ObjectReference::try_from(&record.source)?;
            reference.id.table_relative(&self.table).ok_or_else(|| {
                StatsError::Internal(format!(
                    "object segment {:?} references another table",
                    record.path
                ))
            })?;
            let path = self.store.local_path(&reference).await?;
            if path != Path::new(&record.path) {
                return Err(StatsError::Internal(format!(
                    "materialized object path {} differs from catalog path {}",
                    path.display(),
                    record.path
                )));
            }
            let (row_count, min_key_value, max_key_value) = segment_bounds(&path, Some(key_column))
                .ok_or_else(|| {
                    StatsError::Internal(format!(
                        "materialized object {} has an unreadable Parquet footer",
                        path.display()
                    ))
                })?;
            if row_count != row.row_count {
                return Err(StatsError::Internal(format!(
                    "materialized object {} has {row_count} rows, expected {}",
                    path.display(),
                    row.row_count
                )));
            }
            restored.push(LocalSegment {
                path: record.path.clone(),
                size_bytes: row.byte_size,
                level: row.level,
                min_seq: row.min_seq,
                max_seq: row.max_seq,
                row_count,
                created_at_ms: row.created_at_ms,
                min_key_value,
                max_key_value,
                partition: row.partition.clone(),
                location: SegmentLocation::Both,
            });
            self.catalog
                .set_location(&self.table, &record.path, SegmentLocation::Both)?;
        }
        Ok(restored)
    }

    pub async fn gc_published(
        &self,
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
    ) -> Result<usize, StatsError> {
        self.state_store
            .gc_obsolete_states(
                &self.table,
                now_ms,
                state_retention_ms,
                orphan_grace_ms,
                self.fence,
            )
            .await
    }

    pub async fn gc(&self) -> Result<(), StatsError> {
        self.store.gc().await?;
        Ok(())
    }
}

pub fn object_segment_is_query_visible(
    status: &TableSpecStatus,
    record: &ObjectSegmentRecord,
) -> bool {
    record.table_spec_version == status.active_version()
        || (status.desired_version() == record.table_spec_version && !record.migration_backfill)
        || (status.migration.as_ref().is_some_and(|migration| {
            migration.from_version == Some(status.active_version())
                && migration.to_version == Some(record.table_spec_version)
                && !record.migration_backfill
        }))
}

fn full_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use buffa::MessageField;

    use crate::proto::finelog::stats::{
        ColumnType, NamespaceCatalog, OperatingPolicy, SourceLayout, TableSpec as ProtoTableSpec,
    };
    use crate::store::catalog::object_state_store::ObjectTableStateStore;
    use crate::store::catalog::state_store::{StoredTableState, TableHead};
    use crate::store::object_store::build_remote_object_store;
    use crate::store::schema::{schema_to_proto_owned, with_implicit_seq, Column, Schema};
    use crate::store::table_spec::canonical_json_bytes;

    const TABLE: &str = "iris.worker";

    /// How a commit is lost between this writer and the object store.
    #[derive(Clone, Copy)]
    enum LostCommit {
        /// HEAD swapped, but the caller only sees an ambiguous failure.
        AppliedButUnreported,
        /// HEAD never swapped and the store said so definitively.
        NeverApplied,
    }

    /// A state store whose commits always fail, either after or without
    /// applying to the object store underneath.
    struct LosingStateStore {
        inner: ObjectTableStateStore,
        loss: LostCommit,
    }

    #[async_trait]
    impl TableStateStore for LosingStateStore {
        async fn list(&self) -> Result<Vec<TableHead>, StatsError> {
            self.inner.list().await
        }

        async fn load(&self, table: &str) -> Result<Option<StoredTableState>, StatsError> {
            self.inner.load(table).await
        }

        async fn claim_writer(
            &self,
            table: &str,
            fence: WriterFence,
            selected: &StoredTableState,
        ) -> Result<StoredTableState, StatsError> {
            self.inner.claim_writer(table, fence, selected).await
        }

        async fn commit(
            &self,
            table: &str,
            fence: WriterFence,
            expected: Option<&StoredTableState>,
            next: NamespaceCatalog,
        ) -> Result<StoredTableState, StatsError> {
            match self.loss {
                LostCommit::AppliedButUnreported => {
                    self.inner.commit(table, fence, expected, next).await?;
                    Err(StatsError::AmbiguousCommit(
                        "HEAD swap response was lost".to_string(),
                    ))
                }
                LostCommit::NeverApplied => Err(StatsError::SchemaConflict(
                    "object pointer changed concurrently".to_string(),
                )),
            }
        }

        async fn tombstone(
            &self,
            table: &str,
            fence: WriterFence,
            expected: &StoredTableState,
        ) -> Result<StoredTableState, StatsError> {
            self.inner.tombstone(table, fence, expected).await
        }

        async fn gc_obsolete_states(
            &self,
            table: &str,
            now_ms: i64,
            state_retention_ms: u64,
            orphan_grace_ms: u64,
            fence: WriterFence,
        ) -> Result<usize, StatsError> {
            self.inner
                .gc_obsolete_states(table, now_ms, state_retention_ms, orphan_grace_ms, fence)
                .await
        }
    }

    fn registered_catalog() -> Arc<Catalog> {
        let catalog = Catalog::open(None).unwrap();
        let schema = with_implicit_seq(Schema::new(
            vec![Column::new(
                "timestamp_ms",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            )],
            "",
        ));
        let spec = ProtoTableSpec {
            version: Some(1),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema)),
            source_layout: MessageField::some(SourceLayout::default()),
            operating_policy: MessageField::some(OperatingPolicy::default()),
            ..Default::default()
        };
        let hash: [u8; 32] = Sha256::digest(canonical_json_bytes(&spec).unwrap()).into();
        catalog
            .register_table_spec(TABLE, &spec, &hash, false)
            .unwrap();
        Arc::new(catalog)
    }

    /// An object table whose commits are always lost, plus direct access to
    /// the state store underneath it.
    fn losing_object_table(
        tag: &str,
        fence: u64,
        loss: LostCommit,
    ) -> (ObjectTable, ObjectTableStateStore) {
        let remote_dir = crate::test_support::unique_dir(tag);
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = ObjectTableStateStore::new(remote.clone());
        let table = ObjectTable::new(
            TABLE.to_string(),
            remote_dir,
            registered_catalog(),
            remote.clone(),
            remote,
            Arc::new(LosingStateStore {
                inner: states.clone(),
                loss,
            }),
            WriterFence::new(fence),
        );
        (table, states)
    }

    #[tokio::test]
    async fn a_commit_that_lost_its_response_is_durable_when_head_names_it() {
        let (table, _states) = losing_object_table(
            "object_table_lost_response",
            11,
            LostCommit::AppliedButUnreported,
        );

        let published = table.publish_state().await.unwrap();

        assert_eq!(published.revision().get(), 1);
        assert_eq!(published.fence(), WriterFence::new(11));
        assert!(!table.publication_owed());
        assert!(table.writes_ready());
    }

    #[tokio::test]
    async fn a_commit_the_store_never_applied_stays_owed_at_the_same_revision() {
        let (table, _states) =
            losing_object_table("object_table_unapplied", 11, LostCommit::NeverApplied);

        let error = table.publish_state().await.unwrap_err();

        assert!(matches!(error, CommitError::PublicationDeferred(_)));
        assert!(error.is_committed());
        assert!(table.publication_owed());
        assert!(table.writes_ready());
        assert_eq!(
            table
                .catalog
                .table_spec_status(TABLE)
                .unwrap()
                .catalog_generation,
            1
        );
    }

    /// Claiming a table another writer published takes ownership of exactly the
    /// state HEAD selects.
    #[tokio::test]
    async fn claiming_an_existing_head_retains_the_selected_state() {
        let (table, states) =
            losing_object_table("object_table_claim", 11, LostCommit::NeverApplied);
        let state = namespace_catalog(&table.catalog, TABLE, &table.table_dir).unwrap();
        states
            .commit(TABLE, WriterFence::new(12), None, state.clone())
            .await
            .unwrap();

        table.claim_writer().await.unwrap();

        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.fence(), WriterFence::new(11));
        assert_eq!(selected.catalog, state);
    }

    /// A second process claims a table this writer already published, then this
    /// writer commits again from a state it loaded after the claim.
    #[tokio::test]
    async fn a_replacement_claim_fences_every_later_commit_from_the_stale_writer() {
        let remote_dir = crate::test_support::unique_dir("object_table_replacement_claim");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let stale = ObjectTable::new(
            TABLE.to_string(),
            remote_dir.clone(),
            Arc::clone(&catalog),
            remote.clone(),
            remote.clone(),
            states.clone(),
            WriterFence::new(11),
        );
        stale.publish_state().await.unwrap();

        let replacement = ObjectTable::new(
            TABLE.to_string(),
            remote_dir,
            Arc::clone(&catalog),
            remote.clone(),
            remote,
            states.clone(),
            WriterFence::new(12),
        );
        replacement.claim_writer().await.unwrap();

        // The stale writer advances its local revision and republishes.
        catalog.set_forward_cursor("hub", TABLE, 7).unwrap();
        let error = stale.publish_state().await.unwrap_err();

        assert!(matches!(error, CommitError::Fenced(_)));
        assert!(!stale.writes_ready());
        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.fence(), WriterFence::new(12));
        assert_eq!(selected.revision().get(), 1);
    }

    /// The local projection is rebuildable; losing it never disturbs the state
    /// HEAD selects.
    #[tokio::test]
    async fn a_lost_local_projection_leaves_the_committed_state_selected() {
        let remote_dir = crate::test_support::unique_dir("object_table_lost_projection");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let table = ObjectTable::new(
            TABLE.to_string(),
            remote_dir,
            Arc::clone(&catalog),
            remote.clone(),
            remote,
            states.clone(),
            WriterFence::new(11),
        );
        let committed = table.publish_state().await.unwrap();
        assert_eq!(committed.revision().get(), 1);

        catalog.delete(TABLE).unwrap();

        // The projection is gone, so no next state can be built, and the
        // committed revision remains the selected one.
        let error = table.publish_state().await.unwrap_err();
        assert!(matches!(error, CommitError::PublicationDeferred(_)));
        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.revision().get(), 1);
        assert_eq!(selected.fence(), WriterFence::new(11));
    }
}
