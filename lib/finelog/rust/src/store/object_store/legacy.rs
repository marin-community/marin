//! Object-store implementation for the historical `{table}/...` layout.
//!
//! Logical IDs stay canonical and typed. This implementation alone translates
//! them to the pre-catalog physical layout while legacy tables migrate.

use std::path::PathBuf;

use async_trait::async_trait;
use futures::StreamExt;
use object_store::path::Path as OsPath;
use object_store::{ObjectMeta, ObjectStoreExt, PutMode, PutOptions, UpdateVersion};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::store::object_store::{
    ObjectId, ObjectMetadata, ObjectPrefix, ObjectStore, ObjectVersion, StoredObject,
};

use super::local_file::compare_and_swap as local_compare_and_swap;
use super::provider::Provider;
use super::remote::RemoteObjectStore;

#[derive(Clone)]
pub struct LegacyObjectStore {
    provider: Provider,
}

impl LegacyObjectStore {
    pub fn new(remote: &RemoteObjectStore) -> Self {
        Self {
            provider: remote.provider(),
        }
    }

    fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.provider.prefix_parts()
    }

    fn path(&self, id: &ObjectId) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain([id.table_name()])
            .chain(
                id.relative_key()
                    .split('/')
                    .filter(|component| !component.is_empty()),
            )
            .collect();
        OsPath::from_iter(parts)
    }

    fn prefix(&self, prefix: &ObjectPrefix) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain([prefix.table_name()])
            .chain(
                prefix
                    .relative_prefix()
                    .split('/')
                    .filter(|component| !component.is_empty()),
            )
            .collect();
        OsPath::from_iter(parts)
    }

    fn local_pointer_path(&self, id: &ObjectId) -> Option<PathBuf> {
        let root = self.provider.local_root()?;
        let mut path = root.join(id.table_name());
        for component in id
            .relative_key()
            .split('/')
            .filter(|component| !component.is_empty())
        {
            path.push(component);
        }
        Some(path)
    }

    async fn get_path(&self, path: OsPath) -> Result<Option<StoredObject>, StatsError> {
        let result = match self.provider.backend().get(&path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "read legacy object {path}: {error}"
                )))
            }
        };
        let e_tag = result.meta.e_tag.clone();
        let provider_version = result.meta.version.clone();
        let bytes = result.bytes().await.map_err(|error| {
            StatsError::Internal(format!("read legacy object body {path}: {error}"))
        })?;
        Ok(Some(StoredObject {
            version: ObjectVersion {
                e_tag,
                provider_version,
                content_sha256: Sha256::digest(&bytes).into(),
                byte_size: bytes.len() as u64,
            },
            bytes,
        }))
    }

    async fn list_objects(
        &self,
        prefix: &ObjectPrefix,
    ) -> Result<Vec<(String, ObjectMeta)>, StatsError> {
        let physical_prefix = self.prefix(prefix);
        let table_root = self.prefix(&ObjectPrefix::table(prefix.table_name(), "")?);
        let mut stream = self.provider.backend().list(Some(&physical_prefix));
        let mut objects = Vec::new();
        while let Some(item) = stream.next().await {
            let metadata = item.map_err(|error| {
                StatsError::Internal(format!(
                    "list legacy objects for {:?}: {error}",
                    prefix.table_name()
                ))
            })?;
            let Some(parts) = metadata.location.prefix_match(&table_root) else {
                continue;
            };
            let key = parts
                .map(|part| part.as_ref().to_string())
                .collect::<Vec<_>>()
                .join("/");
            objects.push((key, metadata));
        }
        Ok(objects)
    }
}

#[async_trait]
impl ObjectStore for LegacyObjectStore {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        let path = self.path(id);
        let content_sha256 = Sha256::digest(&bytes).into();
        let byte_size = bytes.len() as u64;
        let result = self
            .provider
            .backend()
            .put(&path, bytes.into())
            .await
            .map_err(|error| {
                StatsError::Internal(format!("write legacy object {path}: {error}"))
            })?;
        Ok(ObjectVersion {
            e_tag: result.e_tag,
            provider_version: result.version,
            content_sha256,
            byte_size,
        })
    }

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.get_path(self.path(id)).await
    }

    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        if let Some(path) = self.local_pointer_path(id) {
            let expected_hash = expected.map(|version| version.content_sha256);
            return tokio::task::spawn_blocking(move || {
                local_compare_and_swap(&path, expected_hash, &bytes)
            })
            .await
            .map_err(|error| StatsError::Internal(format!("legacy object CAS task: {error}")))?;
        }
        let path = self.path(id);
        let mode = match expected {
            None => PutMode::Create,
            Some(version) => PutMode::Update(UpdateVersion {
                e_tag: version.e_tag.clone(),
                version: version.provider_version.clone(),
            }),
        };
        let content_sha256 = Sha256::digest(&bytes).into();
        let byte_size = bytes.len() as u64;
        let result = self
            .provider
            .backend()
            .put_opts(
                &path,
                bytes.into(),
                PutOptions {
                    mode,
                    ..Default::default()
                },
            )
            .await;
        match result {
            Ok(result) => Ok(ObjectVersion {
                e_tag: result.e_tag,
                provider_version: result.version,
                content_sha256,
                byte_size,
            }),
            Err(object_store::Error::AlreadyExists { .. })
            | Err(object_store::Error::Precondition { .. }) => Err(StatsError::SchemaConflict(
                format!("legacy object pointer {path} changed concurrently"),
            )),
            Err(error) => Err(StatsError::Internal(format!(
                "update legacy object pointer {path}: {error}"
            ))),
        }
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError> {
        let path = self.path(id);
        match self.provider.backend().delete(&path).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(StatsError::Internal(format!(
                "delete legacy object {path}: {error}"
            ))),
        }
    }

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        self.list_objects(prefix)
            .await?
            .into_iter()
            .map(|(key, metadata)| {
                Ok(ObjectMetadata {
                    id: ObjectId::table(prefix.table_name(), &key)?,
                    modified_at_ms: metadata.last_modified.timestamp_millis(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::store::object_store::build_remote_object_store;

    fn tempdir() -> PathBuf {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("finelog_legacy_object_store_{nonce}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[tokio::test]
    async fn logical_object_id_uses_the_legacy_physical_layout() {
        let root = tempdir();
        let canonical = build_remote_object_store(root.to_str().unwrap())
            .unwrap()
            .unwrap();
        let legacy = LegacyObjectStore::new(&canonical);
        let id = ObjectId::table("iris.worker", "l1/segment.parquet").unwrap();

        legacy
            .write(&id, bytes::Bytes::from_static(b"rows"))
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(root.join("iris.worker/l1/segment.parquet")).unwrap(),
            b"rows"
        );
        assert!(!root.join(id.as_str()).exists());
        assert_eq!(
            legacy.read(&id).await.unwrap().unwrap().bytes,
            b"rows".as_slice()
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn trait_operations_share_one_physical_mapping() {
        let root = tempdir();
        let canonical = build_remote_object_store(root.to_str().unwrap())
            .unwrap()
            .unwrap();
        let legacy = LegacyObjectStore::new(&canonical);
        let first =
            ObjectId::table("ns.a", "run_id/07/seg_L1_0000000000000000001.parquet").unwrap();
        let second =
            ObjectId::table("ns.a", "run_id/08/seg_L1_0000000000000000001.parquet").unwrap();

        legacy
            .write(&first, bytes::Bytes::from_static(b"hello-parquet"))
            .await
            .unwrap();
        legacy
            .write(&second, bytes::Bytes::from_static(b"hello-parquet"))
            .await
            .unwrap();
        let mut ids: Vec<_> = legacy
            .list(&ObjectPrefix::table("ns.a", "").unwrap())
            .await
            .unwrap()
            .into_iter()
            .map(|metadata| metadata.id)
            .collect();
        ids.sort();
        assert_eq!(ids, vec![first.clone(), second.clone(),]);

        legacy.delete(&first).await.unwrap();
        legacy.delete(&second).await.unwrap();
        assert!(legacy
            .list(&ObjectPrefix::table("ns.a", "").unwrap())
            .await
            .unwrap()
            .is_empty());
        std::fs::remove_dir_all(root).ok();
    }
}
