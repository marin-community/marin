//! Transparent local-file decorator for an object store.
//!
//! Callers use the [`ObjectStore`] interface. Cache membership and filesystem
//! layout stay private to these wrappers; `local_path` is the only operation
//! that promises a verified local file suitable for DataFusion.
//!
//! Writes are dual-ported: the bytes an upload already holds also land in the
//! cache, so the flush → query path never re-downloads its own output. Cache
//! misses download under one store-wide concurrency bound. When a capacity is
//! configured, `gc` evicts least-recently-used files below it; every hit
//! refreshes the file's modification time, and eviction unlinks only behind
//! the store-wide query-visibility write lock so no pinned scan loses a file
//! mid-read.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use tokio::sync::{RwLock, Semaphore};

use crate::errors::StatsError;
use crate::store::object_store::{
    ObjectByteStream, ObjectId, ObjectMetadata, ObjectPrefix, ObjectReference, ObjectStore,
    ObjectVersion, StoredObject, FINELOG_ROOT_COMPONENT,
};

use super::local_file::atomic_write;

/// Concurrent cache-miss downloads across every caller of this store.
const MAX_PARALLEL_FETCHES: usize = 8;

/// Minimum gap between eviction sweeps. `gc` runs once per table per
/// maintenance cycle; the sweep is store-wide, so most calls return
/// immediately.
const CACHE_GC_INTERVAL: Duration = Duration::from_secs(300);

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
            touch(&path)?;
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
    /// Bounds concurrent cache-miss downloads.
    fetches: Arc<Semaphore>,
    /// Total cache bytes to retain; `None` retains everything.
    capacity_bytes: Option<u64>,
    /// Store-wide scan lock. Eviction unlinks only behind its write side, so a
    /// query that localized a file keeps it until the scan drops the read side.
    query_visibility: Arc<RwLock<()>>,
    last_gc: Arc<Mutex<Option<Instant>>>,
}

impl CachedObjectStore {
    pub fn new(
        source: Arc<dyn ObjectStore>,
        root: PathBuf,
        query_visibility: Arc<RwLock<()>>,
        capacity_bytes: Option<u64>,
    ) -> Result<Self, StatsError> {
        Ok(Self {
            source,
            cache: FileCache::new(root)?,
            fetches: Arc::new(Semaphore::new(MAX_PARALLEL_FETCHES)),
            capacity_bytes,
            query_visibility,
            last_gc: Arc::new(Mutex::new(None)),
        })
    }

    async fn lookup(&self, reference: &ObjectReference) -> Result<Option<PathBuf>, StatsError> {
        let cache = self.cache.clone();
        let reference = reference.clone();
        tokio::task::spawn_blocking(move || cache.verified_path(&reference))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache lookup task: {error}")))?
    }

    async fn materialize(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        if let Some(path) = self.lookup(reference).await? {
            return Ok(path);
        }
        let _permit = self.fetches.acquire().await.map_err(|error| {
            StatsError::Internal(format!("object cache fetch semaphore: {error}"))
        })?;
        // A concurrent fetch of the same object may have won while this one
        // waited for a download slot.
        if let Some(path) = self.lookup(reference).await? {
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

    /// Evict least-recently-used cache files until the cache fits its capacity.
    async fn evict_over_capacity(&self, capacity_bytes: u64) -> Result<(), StatsError> {
        let root = self.cache.root.join(FINELOG_ROOT_COMPONENT);
        let mut entries = tokio::task::spawn_blocking(move || scan_cache_files(&root))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache scan task: {error}")))??;
        let total: u64 = entries.iter().map(|entry| entry.bytes).sum();
        if total <= capacity_bytes {
            return Ok(());
        }
        entries.sort_by_key(|entry| entry.modified);
        let mut excess = total - capacity_bytes;
        let mut victims = Vec::new();
        for entry in entries {
            if excess == 0 {
                break;
            }
            excess = excess.saturating_sub(entry.bytes);
            victims.push(entry.path);
        }
        let evicted = victims.len();
        // No scan may hold any of these paths: eviction waits out every pinned
        // read before the first unlink.
        let _visibility = self.query_visibility.write().await;
        tokio::task::spawn_blocking(move || {
            for path in victims {
                match std::fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(StatsError::Internal(format!(
                            "evict object cache {}: {error}",
                            path.display()
                        )))
                    }
                }
            }
            Ok(())
        })
        .await
        .map_err(|error| StatsError::Internal(format!("object cache evict task: {error}")))??;
        tracing::info!(
            evicted,
            cache_bytes = total,
            capacity_bytes,
            "object cache evicted least-recently-used files"
        );
        Ok(())
    }
}

#[async_trait]
impl ObjectStore for CachedObjectStore {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        let version = self.source.write(id, bytes.clone()).await?;
        let cache = self.cache.clone();
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        tokio::task::spawn_blocking(move || cache.write(&reference, &bytes))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache write task: {error}")))??;
        Ok(version)
    }

    async fn write_stream(
        &self,
        id: &ObjectId,
        stream: ObjectByteStream,
    ) -> Result<ObjectVersion, StatsError> {
        // Spool the chunks to a cache staging file while they upload, then
        // publish the spool as the cache entry once the source accepts the
        // object — the streamed bytes never sit in RAM twice and are never
        // downloaded back.
        let path = self.cache.path(id)?;
        let spool = spool_writer(&path)?;
        let hasher = Arc::new(Mutex::new(Sha256::new()));
        let spool_file = Arc::clone(&spool.file);
        let spool_hasher = Arc::clone(&hasher);
        let tee = stream.map(move |chunk| {
            let chunk = chunk?;
            spool_hasher.lock().unwrap().update(&chunk);
            spool_file
                .lock()
                .unwrap()
                .write_all(&chunk)
                .map_err(|error| {
                    StatsError::Internal(format!("write object cache spool: {error}"))
                })?;
            Ok(chunk)
        });
        let version = match self.source.write_stream(id, Box::pin(tee)).await {
            Ok(version) => version,
            Err(error) => {
                spool.discard();
                return Err(error);
            }
        };
        let spooled_sha256: [u8; 32] = Arc::try_unwrap(hasher)
            .map_err(|_| StatsError::Internal("object cache spool hasher is shared".to_string()))?
            .into_inner()
            .unwrap()
            .finalize()
            .into();
        if spooled_sha256 != version.content_sha256 {
            spool.discard();
            return Err(StatsError::Internal(format!(
                "object cache spool for {:?} does not match the uploaded object",
                id.as_str()
            )));
        }
        spool.publish(&path)?;
        Ok(version)
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
        let Some(capacity_bytes) = self.capacity_bytes else {
            return Ok(());
        };
        {
            let mut last = self.last_gc.lock().unwrap();
            if last.is_some_and(|at| at.elapsed() < CACHE_GC_INTERVAL) {
                return Ok(());
            }
            *last = Some(Instant::now());
        }
        self.evict_over_capacity(capacity_bytes).await
    }
}

/// Refresh `path`'s modification time so eviction treats it as recently used.
fn touch(path: &Path) -> Result<(), StatsError> {
    std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .and_then(|file| file.set_modified(SystemTime::now()))
        .map_err(|error| {
            StatsError::Internal(format!("touch object cache {}: {error}", path.display()))
        })
}

struct CacheFile {
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

fn scan_cache_files(root: &Path) -> Result<Vec<CacheFile>, StatsError> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "scan object cache {}: {error}",
                    dir.display()
                )))
            }
        };
        for entry in entries {
            let entry = entry.map_err(|error| {
                StatsError::Internal(format!("scan object cache {}: {error}", dir.display()))
            })?;
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                // Racing a concurrent eviction or invalidation is benign.
                continue;
            };
            if metadata.is_dir() {
                pending.push(path);
                continue;
            }
            // A staging spool is invisible until its atomic rename publishes it.
            if path
                .extension()
                .is_some_and(|extension| extension.to_string_lossy().starts_with("tmp-"))
            {
                continue;
            }
            files.push(CacheFile {
                bytes: metadata.len(),
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                path,
            });
        }
    }
    Ok(files)
}

/// A staging file for one streamed cache entry, published by atomic rename.
struct SpoolWriter {
    file: Arc<Mutex<std::fs::File>>,
    staging: PathBuf,
}

fn spool_writer(path: &Path) -> Result<SpoolWriter, StatsError> {
    let parent = path.parent().ok_or_else(|| {
        StatsError::Internal(format!("object cache {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        StatsError::Internal(format!(
            "create object cache parent {}: {error}",
            parent.display()
        ))
    })?;
    static SPOOL_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let staging = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        SPOOL_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staging)
        .map_err(|error| {
            StatsError::Internal(format!(
                "create object cache spool {}: {error}",
                staging.display()
            ))
        })?;
    Ok(SpoolWriter {
        file: Arc::new(Mutex::new(file)),
        staging,
    })
}

impl SpoolWriter {
    fn publish(self, path: &Path) -> Result<(), StatsError> {
        let file = Arc::try_unwrap(self.file)
            .map_err(|_| StatsError::Internal("object cache spool is still shared".to_string()))?
            .into_inner()
            .unwrap();
        file.sync_all().map_err(|error| {
            StatsError::Internal(format!(
                "fsync object cache spool {}: {error}",
                self.staging.display()
            ))
        })?;
        std::fs::rename(&self.staging, path).map_err(|error| {
            StatsError::Internal(format!(
                "publish object cache {} -> {}: {error}",
                self.staging.display(),
                path.display()
            ))
        })
    }

    fn discard(self) {
        std::fs::remove_file(&self.staging).ok();
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
        let store =
            CachedObjectStore::new(source, cache_root.clone(), Arc::new(RwLock::new(())), None)
                .unwrap();
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

    fn cached_store(
        prefix: &str,
        capacity_bytes: Option<u64>,
    ) -> (CachedObjectStore, [PathBuf; 2]) {
        let remote_root = unique_dir(&format!("{prefix}_remote"));
        let cache_root = unique_dir(&format!("{prefix}_local"));
        let source = Arc::new(
            build_remote_object_store(remote_root.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let store = CachedObjectStore::new(
            source,
            cache_root.clone(),
            Arc::new(RwLock::new(())),
            capacity_bytes,
        )
        .unwrap();
        (store, [remote_root, cache_root])
    }

    /// The remote copy's local twin under the provider's own layout, distinct
    /// from the cache file.
    fn remote_copy(remote_root: &Path, reference: &ObjectReference) -> PathBuf {
        remote_root.join(reference.id.as_str())
    }

    #[tokio::test]
    async fn a_write_populates_the_cache_without_a_second_read() {
        let (store, roots) = cached_store("object_cache_dual_port", None);
        let reference = reference("iris.worker", "objects/dual.parquet", b"dual-ported");
        store
            .write(&reference.id, bytes::Bytes::from_static(b"dual-ported"))
            .await
            .unwrap();

        // Localization must not need the remote copy: the upload already
        // seeded the cache.
        std::fs::remove_file(remote_copy(&roots[0], &reference)).unwrap();
        let path = store.local_path(&reference).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"dual-ported");
        roots.iter().for_each(|root| {
            std::fs::remove_dir_all(root).ok();
        });
    }

    #[tokio::test]
    async fn a_streamed_write_populates_the_cache() {
        let (store, roots) = cached_store("object_cache_spool", None);
        let reference = reference("iris.worker", "indices/spooled.fidx", b"spooled-bytes");
        let chunks: Vec<Result<bytes::Bytes, StatsError>> = vec![
            Ok(bytes::Bytes::from_static(b"spooled-")),
            Ok(bytes::Bytes::from_static(b"bytes")),
        ];
        let version = store
            .write_stream(&reference.id, Box::pin(futures::stream::iter(chunks)))
            .await
            .unwrap();
        assert_eq!(version.content_sha256, reference.version.content_sha256);

        std::fs::remove_file(remote_copy(&roots[0], &reference)).unwrap();
        let path = store.local_path(&reference).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"spooled-bytes");
        roots.iter().for_each(|root| {
            std::fs::remove_dir_all(root).ok();
        });
    }

    #[tokio::test]
    async fn gc_evicts_the_least_recently_used_files_beyond_capacity() {
        let (store, roots) = cached_store("object_cache_lru", Some(8));
        let old = reference("iris.worker", "objects/old.parquet", b"old-old");
        let new = reference("iris.worker", "objects/new.parquet", b"new-new");
        for (reference, bytes) in [(&old, b"old-old"), (&new, b"new-new")] {
            store
                .write(&reference.id, bytes::Bytes::from_static(bytes))
                .await
                .unwrap();
        }
        let old_path = store.planned_local_path(&old.id).unwrap();
        let new_path = store.planned_local_path(&new.id).unwrap();
        std::fs::File::options()
            .append(true)
            .open(&old_path)
            .unwrap()
            .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(1))
            .unwrap();

        // 14 cached bytes against a capacity of 8: only the older file goes.
        store.gc().await.unwrap();
        assert!(!old_path.exists());
        assert!(new_path.exists());

        // Eviction is not loss: the object re-materializes from the remote.
        let path = store.local_path(&old).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"old-old");
        roots.iter().for_each(|root| {
            std::fs::remove_dir_all(root).ok();
        });
    }
}
