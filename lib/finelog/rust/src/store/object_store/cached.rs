//! Transparent local-file decorator for an object store.
//!
//! Callers use the [`ObjectStore`] interface. Cache membership and filesystem
//! layout stay private to these wrappers; `local_path` is the only operation
//! that promises a verified local file suitable for DataFusion.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::store::object_store::{
    ObjectByteStream, ObjectId, ObjectMetadata, ObjectPrefix, ObjectReference, ObjectStore,
    ObjectVersion, StoredObject,
};

use super::local_file::atomic_write;

#[derive(Clone)]
struct FileCache {
    root: PathBuf,
}

impl FileCache {
    fn new(root: PathBuf) -> Result<Self, StatsError> {
        std::fs::create_dir_all(&root).map_err(|error| {
            StatsError::Internal(format!("create object cache {}: {error}", root.display()))
        })?;
        Ok(Self { root })
    }

    fn path(&self, id: &ObjectId) -> Result<PathBuf, StatsError> {
        relative_path(&self.root, id.as_str())
    }

    fn verified_path(&self, reference: &ObjectReference) -> Result<Option<PathBuf>, StatsError> {
        let path = self.path(&reference.id)?;
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "read object cache {}: {error}",
                    path.display()
                )))
            }
        };
        let valid_size = bytes.len() as u64 == reference.version.byte_size;
        let valid_hash =
            Sha256::digest(&bytes).as_slice() == reference.version.content_sha256.as_slice();
        if valid_size && valid_hash {
            return Ok(Some(path));
        }
        std::fs::remove_file(&path).map_err(|error| {
            StatsError::Internal(format!(
                "remove invalid object cache {}: {error}",
                path.display()
            ))
        })?;
        Ok(None)
    }

    fn write(&self, reference: &ObjectReference, bytes: &[u8]) -> Result<PathBuf, StatsError> {
        if bytes.len() as u64 != reference.version.byte_size {
            return Err(StatsError::Internal(format!(
                "object cache source {:?} has {} bytes, expected {}",
                reference.id.as_str(),
                bytes.len(),
                reference.version.byte_size
            )));
        }
        if Sha256::digest(bytes).as_slice() != reference.version.content_sha256.as_slice() {
            return Err(StatsError::Internal(format!(
                "object cache source {:?} failed SHA-256 validation",
                reference.id.as_str()
            )));
        }
        let path = self.path(&reference.id)?;
        atomic_write(&path, bytes)?;
        Ok(path)
    }

    fn remove(&self, id: &ObjectId) -> Result<(), StatsError> {
        let path = self.path(id)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(StatsError::Internal(format!(
                "remove object cache {}: {error}",
                path.display()
            ))),
        }
    }
}

#[derive(Clone)]
pub struct CachedObjectStore {
    source: Arc<dyn ObjectStore>,
    cache: FileCache,
}

impl CachedObjectStore {
    pub fn new(source: Arc<dyn ObjectStore>, root: PathBuf) -> Result<Self, StatsError> {
        Ok(Self {
            source,
            cache: FileCache::new(root)?,
        })
    }

    async fn materialize(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        let cache = self.cache.clone();
        let reference_for_lookup = reference.clone();
        if let Some(path) =
            tokio::task::spawn_blocking(move || cache.verified_path(&reference_for_lookup))
                .await
                .map_err(|error| {
                    StatsError::Internal(format!("object cache lookup task: {error}"))
                })??
        {
            return Ok(path);
        }
        let object = self.source.read(&reference.id).await?.ok_or_else(|| {
            StatsError::Internal(format!(
                "object cache source {:?} is missing",
                reference.id.as_str()
            ))
        })?;
        if object.version.content_sha256 != reference.version.content_sha256 {
            return Err(StatsError::Internal(format!(
                "object cache source {:?} failed SHA-256 validation",
                reference.id.as_str()
            )));
        }
        let cache = self.cache.clone();
        let reference = reference.clone();
        tokio::task::spawn_blocking(move || cache.write(&reference, &object.bytes))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache write task: {error}")))?
    }
}

#[async_trait]
impl ObjectStore for CachedObjectStore {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        self.source.write(id, bytes).await
    }

    async fn write_stream(
        &self,
        id: &ObjectId,
        stream: ObjectByteStream,
    ) -> Result<ObjectVersion, StatsError> {
        self.source.write_stream(id, stream).await
    }

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.source.read(id).await
    }

    async fn local_path(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        self.materialize(reference).await
    }

    fn planned_local_path(&self, id: &ObjectId) -> Result<PathBuf, StatsError> {
        self.cache.path(id)
    }

    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        let version = self.source.compare_and_swap(id, expected, bytes).await?;
        let cache = self.cache.clone();
        let id = id.clone();
        tokio::task::spawn_blocking(move || cache.remove(&id))
            .await
            .map_err(|error| {
                StatsError::Internal(format!("object cache CAS cleanup task: {error}"))
            })??;
        Ok(version)
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError> {
        self.source.delete(id).await?;
        let cache = self.cache.clone();
        let id = id.clone();
        tokio::task::spawn_blocking(move || cache.remove(&id))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache delete task: {error}")))?
    }

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        self.source.list(prefix).await
    }

    async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        self.source.list_tables().await
    }

    async fn gc(&self) -> Result<(), StatsError> {
        Ok(())
    }
}

fn relative_path(root: &Path, relative_key: &str) -> Result<PathBuf, StatsError> {
    let mut path = root.to_path_buf();
    for component in relative_key
        .split('/')
        .filter(|component| !component.is_empty())
    {
        if matches!(component, "." | "..") || component.contains('\\') {
            return Err(StatsError::Internal(format!(
                "object key {relative_key:?} is not a safe relative path"
            )));
        }
        path.push(component);
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::object_store::build_remote_object_store;
    use crate::store::object_store::ObjectVersion;
    use crate::test_support::unique_dir;

    fn reference(table: &str, key: &str, bytes: &[u8]) -> ObjectReference {
        ObjectReference {
            id: ObjectId::table(table, key).unwrap(),
            version: ObjectVersion {
                e_tag: None,
                provider_version: None,
                content_sha256: Sha256::digest(bytes).into(),
                byte_size: bytes.len() as u64,
            },
        }
    }

    #[tokio::test]
    async fn local_path_materializes_validates_and_deletes_with_the_object() {
        let remote_root = unique_dir("object_cache_remote");
        let cache_root = unique_dir("object_cache_local");
        let source = Arc::new(
            build_remote_object_store(remote_root.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let store = CachedObjectStore::new(source, cache_root.clone()).unwrap();
        let reference = reference("iris.worker", "objects/kept.parquet", b"kept");
        store
            .write(&reference.id, bytes::Bytes::from_static(b"kept"))
            .await
            .unwrap();

        let path = store.local_path(&reference).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"kept");
        std::fs::write(&path, b"corrupt").unwrap();
        assert_eq!(
            std::fs::read(store.local_path(&reference).await.unwrap()).unwrap(),
            b"kept"
        );

        store.gc().await.unwrap();
        assert!(path.exists());

        store.delete(&reference.id).await.unwrap();
        assert!(!path.exists());
        std::fs::remove_dir_all(remote_root).ok();
        std::fs::remove_dir_all(cache_root).ok();
    }
}
