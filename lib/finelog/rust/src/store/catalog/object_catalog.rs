//! Object-store implementation of the published-catalog interface.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{CatalogHead, NamespaceCatalog, ObjectRef};
use crate::store::catalog::published::{CatalogSnapshot, PublishedCatalog};
use crate::store::object_store::{ObjectId, ObjectMetadata, ObjectPrefix, ObjectStore};

pub const OBJECT_CATALOG_FORMAT_VERSION: u64 = 1;
const HEAD_KEY: &str = "HEAD.json";
const CATALOGS_PREFIX: &str = "catalogs";

#[derive(Clone)]
pub struct ObjectCatalog {
    storage: Arc<dyn ObjectStore>,
}

impl ObjectCatalog {
    pub fn new(storage: Arc<dyn ObjectStore>) -> Self {
        Self { storage }
    }

    pub async fn load(&self, namespace: &str) -> Result<Option<CatalogSnapshot>, StatsError> {
        let head_id = ObjectId::table(namespace, HEAD_KEY)?;
        let Some(head_object) = self.storage.read(&head_id).await? else {
            return Ok(None);
        };
        let head: CatalogHead = serde_json::from_slice(&head_object.bytes).map_err(|error| {
            StatsError::Internal(format!("decode object HEAD for {namespace:?}: {error}"))
        })?;
        validate_head(namespace, &head)?;
        let catalog_ref = head.catalog.as_option().ok_or_else(|| {
            StatsError::Internal(format!(
                "object HEAD for {namespace:?} has no catalog reference"
            ))
        })?;
        let catalog_object_id = catalog_ref.object_id.as_deref().ok_or_else(|| {
            StatsError::Internal(format!(
                "table HEAD for {namespace:?} has an empty catalog object ID"
            ))
        })?;
        let catalog_id = ObjectId::parse(catalog_object_id)?;
        if catalog_id.table_relative(namespace).is_none() {
            return Err(StatsError::Internal(format!(
                "table HEAD for {namespace:?} references an object from another table"
            )));
        }
        let catalog_object = self.storage.read(&catalog_id).await?.ok_or_else(|| {
            StatsError::Internal(format!(
                "table HEAD for {namespace:?} references missing catalog {catalog_object_id:?}"
            ))
        })?;
        if catalog_ref.sha256.as_deref() != Some(catalog_object.version.content_sha256.as_slice()) {
            return Err(StatsError::Internal(format!(
                "table catalog {catalog_object_id:?} for {namespace:?} failed SHA-256 validation"
            )));
        }
        let catalog: NamespaceCatalog =
            serde_json::from_slice(&catalog_object.bytes).map_err(|error| {
                StatsError::Internal(format!(
                    "decode table catalog {catalog_object_id:?} for {namespace:?}: {error}"
                ))
            })?;
        validate_catalog(namespace, &head, &catalog)?;
        Ok(Some(CatalogSnapshot {
            head,
            catalog,
            head_version: head_object.version,
        }))
    }

    /// Remove the query and recovery pointer while retaining immutable history.
    pub async fn delete_head(&self, namespace: &str) -> Result<(), StatsError> {
        self.storage
            .delete(&ObjectId::table(namespace, HEAD_KEY)?)
            .await
    }

    pub async fn claim_writer(
        &self,
        namespace: &str,
        writer_epoch: u64,
        snapshot: &CatalogSnapshot,
    ) -> Result<CatalogSnapshot, StatsError> {
        if snapshot.head.writer_epoch == Some(writer_epoch) {
            return Ok(snapshot.clone());
        }
        let mut head = snapshot.head.clone();
        head.writer_epoch = Some(writer_epoch);
        let head_bytes = serde_json::to_vec(&head).map_err(|error| {
            StatsError::Internal(format!("encode table HEAD for {namespace:?}: {error}"))
        })?;
        let head_version = self
            .storage
            .compare_and_swap(
                &ObjectId::table(namespace, HEAD_KEY)?,
                Some(&snapshot.head_version),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(CatalogSnapshot {
            head,
            catalog: snapshot.catalog.clone(),
            head_version,
        })
    }

    pub async fn publish(
        &self,
        namespace: &str,
        writer_epoch: u64,
        catalog: NamespaceCatalog,
        expected: Option<&CatalogSnapshot>,
    ) -> Result<CatalogSnapshot, StatsError> {
        let generation = catalog.catalog_generation.unwrap_or(0);
        let previous_generation =
            expected.map(|snapshot| snapshot.head.catalog_generation.unwrap_or(0));
        if generation == 0 || previous_generation.is_some_and(|previous| generation <= previous) {
            return Err(StatsError::SchemaConflict(format!(
                "object catalog generation {generation} does not advance {previous_generation:?} for {namespace:?}"
            )));
        }
        if catalog.format_version.unwrap_or(0) != OBJECT_CATALOG_FORMAT_VERSION
            || catalog.namespace.as_deref() != Some(namespace)
        {
            return Err(StatsError::SchemaValidation(format!(
                "object catalog identity does not match namespace {namespace:?}"
            )));
        }

        let catalog_bytes = serde_json::to_vec(&catalog).map_err(|error| {
            StatsError::Internal(format!("encode object catalog for {namespace:?}: {error}"))
        })?;
        let catalog_sha256: [u8; 32] = Sha256::digest(&catalog_bytes).into();
        let catalog_key = format!(
            "{CATALOGS_PREFIX}/{generation:020}-{}.json",
            short_hex(&catalog_sha256)
        );
        let catalog_id = ObjectId::table(namespace, &catalog_key)?;
        let catalog_version = self
            .storage
            .write(&catalog_id, Bytes::from(catalog_bytes.clone()))
            .await?;
        let head = CatalogHead {
            format_version: Some(OBJECT_CATALOG_FORMAT_VERSION),
            namespace: Some(namespace.to_string()),
            writer_epoch: Some(writer_epoch),
            catalog_generation: Some(generation),
            active_table_spec_version: catalog.active_table_spec_version,
            catalog: buffa::MessageField::some(ObjectRef {
                object_id: Some(catalog_id.as_str().to_string()),
                provider_version: catalog_version.provider_version.clone(),
                etag: catalog_version.e_tag.clone(),
                byte_size: Some(catalog_bytes.len() as u64),
                sha256: Some(catalog_sha256.to_vec()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let head_bytes = serde_json::to_vec(&head).map_err(|error| {
            StatsError::Internal(format!("encode object HEAD for {namespace:?}: {error}"))
        })?;
        let head_version = self
            .storage
            .compare_and_swap(
                &ObjectId::table(namespace, HEAD_KEY)?,
                expected.map(|snapshot| &snapshot.head_version),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(CatalogSnapshot {
            head,
            catalog,
            head_version,
        })
    }

    #[cfg(test)]
    async fn catalog_keys(&self, namespace: &str) -> Result<Vec<String>, StatsError> {
        Ok(self
            .table_objects(namespace, CATALOGS_PREFIX)
            .await?
            .into_iter()
            .map(|(key, _)| key)
            .collect())
    }

    /// Remove superseded catalog documents after the maximum query lifetime.
    pub async fn gc_obsolete_catalogs(
        &self,
        namespace: &str,
        now_ms: i64,
        catalog_retention_ms: u64,
        orphan_grace_ms: u64,
        writer_epoch: u64,
    ) -> Result<usize, StatsError> {
        let Some(snapshot) = self.load(namespace).await? else {
            return Ok(0);
        };
        if snapshot.head.writer_epoch != Some(writer_epoch) {
            tracing::warn!(
                namespace,
                expected_writer_epoch = writer_epoch,
                head_writer_epoch = snapshot.head.writer_epoch.unwrap_or(0),
                "skipping object GC from a fenced writer"
            );
            return Ok(0);
        }
        let current_id = ObjectId::parse(
            snapshot
                .head
                .catalog
                .as_option()
                .and_then(|reference| reference.object_id.as_deref())
                .ok_or_else(|| {
                    StatsError::Internal(format!(
                        "table HEAD for {namespace:?} has no catalog object ID"
                    ))
                })?,
        )?;
        let current = current_id.table_relative(namespace).ok_or_else(|| {
            StatsError::Internal(format!(
                "table HEAD for {namespace:?} points outside its table"
            ))
        })?;
        let catalog_cutoff =
            now_ms.saturating_sub(i64::try_from(catalog_retention_ms).unwrap_or(i64::MAX));
        let orphan_cutoff =
            now_ms.saturating_sub(i64::try_from(orphan_grace_ms).unwrap_or(i64::MAX));
        let current_generation = snapshot.head.catalog_generation.unwrap_or(0);
        let catalog_objects = self.table_objects(namespace, CATALOGS_PREFIX).await?;
        let mut removed = 0;
        for (key, meta) in &catalog_objects {
            if key == current {
                continue;
            }
            let Some(generation) = catalog_generation_from_key(key) else {
                tracing::warn!(namespace, key, "retaining unrecognized object catalog key");
                continue;
            };
            let obsolete_at_ms = if generation < current_generation {
                catalog_objects
                    .iter()
                    .filter_map(|(candidate_key, candidate_meta)| {
                        let candidate_generation = catalog_generation_from_key(candidate_key)?;
                        (candidate_generation > generation
                            && candidate_generation <= current_generation)
                            .then_some(candidate_meta.modified_at_ms)
                    })
                    .min()
            } else {
                // Same-generation and future-generation objects never won HEAD.
                Some(meta.modified_at_ms)
            };
            if obsolete_at_ms.is_none_or(|obsolete_at_ms| obsolete_at_ms > catalog_cutoff) {
                continue;
            }
            self.storage
                .delete(&ObjectId::table(namespace, key)?)
                .await?;
            removed += 1;
        }
        let mut referenced = referenced_object_keys(&snapshot.catalog);
        for (key, _) in self.table_objects(namespace, CATALOGS_PREFIX).await? {
            if key == current {
                continue;
            }
            let Some(object) = self
                .storage
                .read(&ObjectId::table(namespace, &key)?)
                .await?
            else {
                continue;
            };
            let catalog: NamespaceCatalog =
                serde_json::from_slice(&object.bytes).map_err(|error| {
                    StatsError::Internal(format!(
                        "decode retained object catalog {key:?} for {namespace:?}: {error}"
                    ))
                })?;
            referenced.extend(referenced_object_keys(&catalog));
        }
        for (key, meta) in self.table_objects(namespace, "objects").await? {
            let id = ObjectId::table(namespace, &key)?;
            if referenced.contains(id.as_str()) || meta.modified_at_ms > orphan_cutoff {
                continue;
            }
            self.storage.delete(&id).await?;
            removed += 1;
        }
        Ok(removed)
    }

    async fn table_objects(
        &self,
        namespace: &str,
        relative_prefix: &str,
    ) -> Result<Vec<(String, ObjectMetadata)>, StatsError> {
        let objects = self
            .storage
            .list(&ObjectPrefix::table(namespace, relative_prefix)?)
            .await?;
        objects
            .into_iter()
            .map(|metadata| {
                let key = metadata
                    .id
                    .table_relative(namespace)
                    .ok_or_else(|| {
                        StatsError::Internal(format!(
                            "object {:?} escaped table {namespace:?}",
                            metadata.id.as_str()
                        ))
                    })?
                    .to_string();
                Ok((key, metadata))
            })
            .collect()
    }

    /// Publish a complete catalog value unless HEAD already selects it.
    pub async fn publish_selected(
        &self,
        namespace: &str,
        contents: NamespaceCatalog,
        writer_epoch: u64,
    ) -> Result<CatalogSnapshot, StatsError> {
        let remote = self.load(namespace).await?;
        if let Some(remote) = &remote {
            let remote_generation = remote.head.catalog_generation.unwrap_or(0);
            let local_generation = contents.catalog_generation.unwrap_or(0);
            if remote_generation == local_generation {
                if remote.catalog != contents {
                    return Err(StatsError::SchemaConflict(format!(
                        "local catalog generation {local_generation} for {namespace:?} differs from the published generation"
                    )));
                }
                return Ok(remote.clone());
            }
            if remote_generation >= local_generation {
                return Err(StatsError::SchemaConflict(format!(
                    "local catalog generation {local_generation} for {namespace:?} does not advance remote generation {remote_generation}"
                )));
            }
        }
        self.publish(namespace, writer_epoch, contents, remote.as_ref())
            .await
    }
}

#[async_trait]
impl PublishedCatalog for ObjectCatalog {
    async fn load(&self, table: &str) -> Result<Option<CatalogSnapshot>, StatsError> {
        ObjectCatalog::load(self, table).await
    }

    async fn claim_writer(
        &self,
        table: &str,
        writer_epoch: u64,
        snapshot: &CatalogSnapshot,
    ) -> Result<CatalogSnapshot, StatsError> {
        ObjectCatalog::claim_writer(self, table, writer_epoch, snapshot).await
    }

    async fn publish(
        &self,
        table: &str,
        catalog: NamespaceCatalog,
        writer_epoch: u64,
    ) -> Result<CatalogSnapshot, StatsError> {
        ObjectCatalog::publish_selected(self, table, catalog, writer_epoch).await
    }

    async fn delete_head(&self, table: &str) -> Result<(), StatsError> {
        ObjectCatalog::delete_head(self, table).await
    }

    async fn gc_obsolete_catalogs(
        &self,
        table: &str,
        now_ms: i64,
        catalog_retention_ms: u64,
        orphan_grace_ms: u64,
        writer_epoch: u64,
    ) -> Result<usize, StatsError> {
        ObjectCatalog::gc_obsolete_catalogs(
            self,
            table,
            now_ms,
            catalog_retention_ms,
            orphan_grace_ms,
            writer_epoch,
        )
        .await
    }
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
        .filter_map(|segment| {
            segment
                .source
                .as_option()
                .and_then(|source| source.object_id.clone())
        })
        .chain(catalog.direct_query_segments.iter().filter_map(|segment| {
            segment
                .source
                .as_option()
                .and_then(|source| source.object_id.clone())
        }))
        .collect()
}

fn validate_head(namespace: &str, head: &CatalogHead) -> Result<(), StatsError> {
    if head.format_version.unwrap_or(0) != OBJECT_CATALOG_FORMAT_VERSION
        || head.namespace.as_deref() != Some(namespace)
        || head.catalog_generation.unwrap_or(0) == 0
    {
        return Err(StatsError::Internal(format!(
            "invalid object HEAD for namespace {namespace:?}"
        )));
    }
    Ok(())
}

fn validate_catalog(
    namespace: &str,
    head: &CatalogHead,
    catalog: &NamespaceCatalog,
) -> Result<(), StatsError> {
    if catalog.format_version.unwrap_or(0) != OBJECT_CATALOG_FORMAT_VERSION
        || catalog.namespace.as_deref() != Some(namespace)
        || catalog.catalog_generation != head.catalog_generation
        || catalog.active_table_spec_version != head.active_table_spec_version
    {
        return Err(StatsError::Internal(format!(
            "object catalog does not match HEAD for namespace {namespace:?}"
        )));
    }
    Ok(())
}

fn short_hex(bytes: &[u8; 32]) -> String {
    bytes[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn catalog_generation_from_key(key: &str) -> Option<u64> {
    key.strip_prefix(CATALOGS_PREFIX)?
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

    fn catalog(namespace: &str, generation: u64, active_version: u64) -> NamespaceCatalog {
        NamespaceCatalog {
            format_version: Some(OBJECT_CATALOG_FORMAT_VERSION),
            namespace: Some(namespace.to_string()),
            catalog_generation: Some(generation),
            active_table_spec_version: Some(active_version),
            max_query_time_ms: Some(600_000),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn local_head_cas_publishes_one_complete_generation() {
        let remote_dir = crate::test_support::unique_dir("object_catalog_cas");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let object = ObjectCatalog::new(Arc::new(remote.clone()));

        let first = object
            .publish("iris.worker", 11, catalog("iris.worker", 1, 1), None)
            .await
            .unwrap();
        let loaded = object.load("iris.worker").await.unwrap().unwrap();
        assert_eq!(loaded.head.writer_epoch, Some(11));
        assert_eq!(loaded.catalog.catalog_generation, Some(1));

        let second = object
            .publish(
                "iris.worker",
                11,
                catalog("iris.worker", 2, 2),
                Some(&first),
            )
            .await
            .unwrap();
        let stale = object
            .publish(
                "iris.worker",
                12,
                catalog("iris.worker", 2, 3),
                Some(&first),
            )
            .await
            .unwrap_err();
        assert!(matches!(stale, StatsError::SchemaConflict(_)));

        let loaded = object.load("iris.worker").await.unwrap().unwrap();
        assert_eq!(loaded.catalog.active_table_spec_version, Some(2));
        assert_eq!(loaded.head.writer_epoch, Some(11));
        assert_eq!(second.catalog, loaded.catalog);

        // A losing writer may leave one immutable, unreachable catalog. HEAD is
        // still the sole visibility boundary, and later GC can remove the orphan.
        let keys = object.catalog_keys("iris.worker").await.unwrap();
        assert_eq!(keys.len(), 3);
        let current_key = second
            .head
            .catalog
            .as_option()
            .unwrap()
            .object_id
            .as_deref()
            .unwrap();
        let current_modified_ms = remote
            .list(&ObjectPrefix::table("iris.worker", CATALOGS_PREFIX).unwrap())
            .await
            .unwrap()
            .into_iter()
            .find(|metadata| metadata.id.as_str() == current_key)
            .unwrap()
            .modified_at_ms;
        assert_eq!(
            object
                .gc_obsolete_catalogs("iris.worker", current_modified_ms + 5, 10, 10, 11)
                .await
                .unwrap(),
            0
        );
        let future = i64::MAX;
        assert_eq!(
            object
                .gc_obsolete_catalogs("iris.worker", future, 0, 0, 12)
                .await
                .unwrap(),
            0
        );
        assert_eq!(object.catalog_keys("iris.worker").await.unwrap().len(), 3);
        assert_eq!(
            object
                .gc_obsolete_catalogs("iris.worker", future, 600_000, 600_000, 11)
                .await
                .unwrap(),
            2
        );
        assert_eq!(object.catalog_keys("iris.worker").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn immutable_catalog_retries_require_identical_bytes() {
        let remote_dir = crate::test_support::unique_dir("object_catalog_immutable");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let object = ObjectCatalog::new(Arc::new(remote.clone()));
        let published = object
            .publish("iris.worker", 1, catalog("iris.worker", 1, 1), None)
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
    }

    #[tokio::test]
    async fn garbage_collection_removes_unreferenced_objects_after_query_grace() {
        let remote_dir = crate::test_support::unique_dir("object_catalog_orphan_gc");
        let remote = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let object = ObjectCatalog::new(Arc::new(remote.clone()));
        object
            .publish("iris.worker", 1, catalog("iris.worker", 1, 1), None)
            .await
            .unwrap();
        let orphan_id =
            ObjectId::table("iris.worker", "objects/v1/l0/orphan/source.parquet").unwrap();
        remote
            .write(&orphan_id, Bytes::from_static(b"orphan"))
            .await
            .unwrap();

        let future = i64::MAX;
        assert_eq!(
            object
                .gc_obsolete_catalogs("iris.worker", future, 600_000, 600_000, 1)
                .await
                .unwrap(),
            1
        );
        assert!(remote.read(&orphan_id).await.unwrap().is_none());
    }
}
