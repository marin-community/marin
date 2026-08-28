//! Object-backed table controller.
//!
//! `Namespace` owns buffering and query state. This controller owns canonical
//! object writes, local materialization, and published catalogs for one table.
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
use crate::store::catalog::published::PublishedCatalog;
use crate::store::catalog::{Catalog, ObjectSegmentRecord, TableSpecStatus};
use crate::store::object_store::{ObjectId, ObjectReference, ObjectStore};
use crate::store::segment::segment_bounds;
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
    published_catalog: Arc<dyn PublishedCatalog>,
    writer_epoch: u64,
    publish_pending: AtomicBool,
}

impl ObjectTable {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table: String,
        table_dir: PathBuf,
        catalog: Arc<Catalog>,
        store: Arc<dyn ObjectStore>,
        legacy_store: Arc<dyn ObjectStore>,
        published_catalog: Arc<dyn PublishedCatalog>,
        writer_epoch: u64,
    ) -> Self {
        Self {
            table,
            table_dir,
            catalog,
            store,
            legacy_store,
            published_catalog,
            writer_epoch,
            publish_pending: AtomicBool::new(false),
        }
    }

    pub fn legacy_store(&self) -> Arc<dyn ObjectStore> {
        Arc::clone(&self.legacy_store)
    }

    pub fn mark_publish_pending(&self) {
        self.publish_pending.store(true, Ordering::SeqCst);
    }

    pub async fn publish_pending(&self) -> Result<(), StatsError> {
        if !self.publish_pending.swap(false, Ordering::SeqCst) {
            return Ok(());
        }
        if let Err(error) = self.publish().await {
            self.publish_pending.store(true, Ordering::SeqCst);
            return Err(error);
        }
        Ok(())
    }

    pub async fn publish(&self) -> Result<(), StatsError> {
        let catalog = namespace_catalog(&self.catalog, &self.table, &self.table_dir)?;
        self.published_catalog
            .publish(&self.table, catalog, self.writer_epoch)
            .await?;
        Ok(())
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
        catalog_retention_ms: u64,
        orphan_grace_ms: u64,
    ) -> Result<usize, StatsError> {
        self.published_catalog
            .gc_obsolete_catalogs(
                &self.table,
                now_ms,
                catalog_retention_ms,
                orphan_grace_ms,
                self.writer_epoch,
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
