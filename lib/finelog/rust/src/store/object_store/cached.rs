//! Transparent local-file decorator for an object store.
//!
//! Callers use the [`ObjectStore`] interface. Cache membership and filesystem
//! layout stay private to these wrappers; `local_path` is the only operation
//! that promises a local file suitable for DataFusion.
//!
//! Writes are dual-ported: the bytes an upload already holds also land in the
//! cache, so the flush → query path never re-downloads its own output. Cache
//! misses download under one store-wide concurrency bound. When a capacity is
//! configured, `gc` evicts least-recently-used files below it; every hit
//! refreshes the file's modification time, and eviction unlinks only behind
//! the store-wide query-visibility write lock so no pinned scan loses a file
//! mid-read.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use tokio::sync::{RwLock, Semaphore};

use crate::errors::StatsError;
use crate::store::object_store::{
    DeleteManyOutcome, ObjectId, ObjectMetadata, ObjectPrefix, ObjectReference, ObjectStore,
    ObjectVersion, StoredObject, FINELOG_ROOT_COMPONENT,
};

use super::admission::{QueryCacheAdmission, RequestHistory};
use super::local_file::atomic_write;

/// Concurrent cache-miss downloads across every caller of this store.
const MAX_PARALLEL_FETCHES: usize = 8;

const MAX_PARALLEL_UPLOADS: usize = 8;

/// Let queued readers run between batches of filesystem removals.
const CACHE_DELETE_BATCH_FILES: usize = 16;

/// Minimum gap between eviction sweeps. `gc` runs once per table per
/// maintenance cycle; the sweep is store-wide, so most calls return
/// immediately.
const CACHE_GC_INTERVAL: Duration = Duration::from_secs(300);

/// Age past which a `tmp-*` staging file is treated as abandoned and removed.
/// An atomic write stages, fsyncs, and renames within seconds; an hour covers
/// any slow disk with a wide margin.
const STALE_STAGING_GRACE: Duration = Duration::from_secs(60 * 60);

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

    fn path(&self, id: &ObjectId) -> PathBuf {
        let mut path = self.root.clone();
        path.extend(id.as_str().split('/'));
        path
    }

    fn cached_path(&self, reference: &ObjectReference) -> Result<Option<PathBuf>, StatsError> {
        let path = self.path(&reference.id);
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "stat object cache {}: {error}",
                    path.display()
                )))
            }
        };
        if metadata.len() == reference.version.byte_size {
            touch(&path)?;
            return Ok(Some(path));
        }
        std::fs::remove_file(&path).map_err(|error| {
            StatsError::Internal(format!(
                "remove truncated object cache {}: {error}",
                path.display()
            ))
        })?;
        Ok(None)
    }

    /// Land `bytes` as the cache file for `reference`.
    fn write(&self, reference: &ObjectReference, bytes: &[u8]) -> Result<PathBuf, StatsError> {
        let path = self.path(&reference.id);
        atomic_write(&path, bytes)?;
        Ok(path)
    }

    fn remove(&self, id: &ObjectId) -> Result<(), StatsError> {
        let path = self.path(id);
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
    /// Bounds write-through uploads independently of reads and flush encoding.
    uploads: Arc<Semaphore>,
    /// Total cache bytes to retain; `None` retains everything.
    capacity_bytes: Option<u64>,
    /// Store-wide scan lock. Eviction unlinks only behind its write side, so a
    /// query that localized a file keeps it until the scan drops the read side.
    query_visibility: Arc<RwLock<()>>,
    last_gc: Arc<Mutex<Option<Instant>>>,
    /// Objects staged locally whose upload has not completed. Their cache files
    /// are the only copy of the bytes, so eviction never selects them.
    staged: Arc<Mutex<HashSet<PathBuf>>>,
    request_history: Arc<Mutex<RequestHistory>>,
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
            uploads: Arc::new(Semaphore::new(MAX_PARALLEL_UPLOADS)),
            capacity_bytes,
            query_visibility,
            last_gc: Arc::new(Mutex::new(None)),
            staged: Arc::new(Mutex::new(HashSet::new())),
            request_history: Arc::new(Mutex::new(RequestHistory::default())),
        })
    }

    async fn lookup(&self, reference: &ObjectReference) -> Result<Option<PathBuf>, StatsError> {
        let cache = self.cache.clone();
        let reference = reference.clone();
        tokio::task::spawn_blocking(move || cache.cached_path(&reference))
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
        self.materialize_reserved(reference).await
    }

    /// Materialize with a download slot already held by the caller.
    async fn materialize_reserved(
        &self,
        reference: &ObjectReference,
    ) -> Result<PathBuf, StatsError> {
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
        if object.version.byte_size != reference.version.byte_size {
            return Err(StatsError::Internal(format!(
                "object cache source {:?} has unexpected byte size",
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
        let staged = self.staged.lock().unwrap().clone();
        let mut excess = total - capacity_bytes;
        let mut victims = Vec::new();
        for entry in entries {
            if excess == 0 {
                break;
            }
            // A staged file is the only copy of its bytes until its upload
            // completes; evicting it would lose acknowledged rows.
            if staged.contains(&entry.path) {
                continue;
            }
            excess = excess.saturating_sub(entry.bytes);
            victims.push(entry.path);
        }
        let evicted = victims.len();
        self.remove_cached_paths(victims).await?;
        tracing::info!(
            evicted,
            cache_bytes = total,
            capacity_bytes,
            "object cache evicted least-recently-used files"
        );
        Ok(())
    }

    async fn remove_cached_paths(&self, paths: Vec<PathBuf>) -> Result<(), StatsError> {
        for batch in paths.chunks(CACHE_DELETE_BATCH_FILES) {
            let batch = batch.to_vec();
            // Readers pin cache paths until their scan completes. Move the
            // guard into the blocking task so cancellation cannot release it
            // while that task is still unlinking files.
            let visibility = self.query_visibility.clone().write_owned().await;
            tokio::task::spawn_blocking(move || {
                let _visibility = visibility;
                for path in batch {
                    match std::fs::remove_file(&path) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => {
                            return Err(StatsError::Internal(format!(
                                "remove object cache {}: {error}",
                                path.display()
                            )))
                        }
                    }
                }
                Ok(())
            })
            .await
            .map_err(|error| {
                StatsError::Internal(format!("object cache removal task: {error}"))
            })??;
            tokio::task::yield_now().await;
        }
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

    /// Land the bytes in the cache only, pinned against eviction until
    /// [`Self::upload_staged`] makes them remotely durable. The version is
    /// computed from the buffer.
    async fn stage(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        let version = ObjectVersion {
            e_tag: None,
            provider_version: None,
            byte_size: bytes.len() as u64,
            local_value: None,
        };
        let cache = self.cache.clone();
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        let path = tokio::task::spawn_blocking(move || cache.write(&reference, &bytes))
            .await
            .map_err(|error| StatsError::Internal(format!("object stage task: {error}")))??;
        self.staged.lock().unwrap().insert(path);
        Ok(version)
    }

    async fn upload_staged(&self, reference: &ObjectReference) -> Result<(), StatsError> {
        let _permit = self.uploads.acquire().await.map_err(|error| {
            StatsError::Internal(format!("object cache upload semaphore: {error}"))
        })?;
        let Some(path) = self.lookup(reference).await? else {
            return Err(StatsError::Internal(format!(
                "staged object {:?} has no local bytes to upload",
                reference.id.as_str()
            )));
        };
        let bytes = tokio::fs::read(&path).await.map_err(|error| {
            StatsError::Internal(format!("read staged object {}: {error}", path.display()))
        })?;
        self.source
            .write(&reference.id, bytes::Bytes::from(bytes))
            .await?;
        self.staged.lock().unwrap().remove(&path);
        Ok(())
    }

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.source.read(id).await
    }

    async fn exists(&self, id: &ObjectId) -> Result<bool, StatsError> {
        self.source.exists(id).await
    }

    async fn local_path(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        self.materialize(reference).await
    }

    fn planned_local_path(&self, id: &ObjectId) -> Result<PathBuf, StatsError> {
        Ok(self.cache.path(id))
    }

    fn remote_scan_url(&self, id: &ObjectId) -> Option<String> {
        self.source.remote_scan_url(id)
    }

    async fn cached_path(
        &self,
        reference: &ObjectReference,
    ) -> Result<Option<PathBuf>, StatsError> {
        self.lookup(reference).await
    }

    fn warm(&self, reference: &ObjectReference) {
        // Warming is optional: scans can read the remote object directly.
        // Reserving before spawning prevents a broad or cancelled query from
        // leaving thousands of queued downloads churning the cache afterward.
        let Ok(permit) = self.fetches.clone().try_acquire_owned() else {
            return;
        };
        let store = self.clone();
        let reference = reference.clone();
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(error) = store.materialize_reserved(&reference).await {
                tracing::debug!(%error, object = ?reference.id.as_str(),
                    "background cache fill failed; the next scan reads remote again");
            }
        });
    }

    fn warm_query(&self, reference: &ObjectReference, query: &QueryCacheAdmission) {
        if query.admit(reference, &self.request_history) {
            self.warm(reference);
        }
    }

    // Mutable pointers are only ever written through CAS and never cached, so
    // there is no cache entry to invalidate here.
    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        self.source.compare_and_swap(id, expected, bytes).await
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError> {
        // A scan may already hold the planned cache path. Wait for every pinned
        // read before removing the remote object or its cache copy.
        let _visibility = self.query_visibility.write().await;
        self.source.delete(id).await?;
        let cache = self.cache.clone();
        let id = id.clone();
        tokio::task::spawn_blocking(move || cache.remove(&id))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache delete task: {error}")))?
    }

    async fn delete_many(&self, ids: Vec<ObjectId>) -> DeleteManyOutcome {
        let outcome = self.source.delete_many(ids).await;
        if outcome.deleted.is_empty() {
            return outcome;
        }
        let paths = outcome
            .deleted
            .iter()
            .map(|id| self.cache.path(id))
            .collect();
        let cache_error = self.remove_cached_paths(paths).await.err();
        DeleteManyOutcome {
            deleted: outcome.deleted,
            error: outcome.error.or(cache_error),
        }
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
            // A staging file is invisible until its atomic rename publishes it.
            // A live write stages for seconds; one this old was abandoned by a
            // failed write or a crashed process and would otherwise sit in the
            // cache forever (staging files are exempt from eviction).
            if path
                .extension()
                .is_some_and(|extension| extension.to_string_lossy().starts_with("tmp-"))
            {
                let age = metadata
                    .modified()
                    .ok()
                    .and_then(|modified| modified.elapsed().ok());
                if age.is_some_and(|age| age >= STALE_STAGING_GRACE) {
                    match std::fs::remove_file(&path) {
                        Ok(()) => {
                            tracing::info!(
                                path = %path.display(),
                                "removed abandoned object staging file"
                            );
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => {
                            tracing::warn!(
                                path = %path.display(),
                                %error,
                                "failed to remove abandoned object staging file"
                            );
                        }
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::provider::NamespaceProvider;
    use crate::query::{make_ctx, run_query_over, RegisteredProvider};
    use crate::store::object_store::build_remote_object_store;
    use crate::store::object_store::ObjectVersion;
    use crate::store::table::query_view::SegmentObjects;
    use crate::store::table_state::ArtifactReferences;
    use crate::test_support::{
        unique_dir, FaultAction, FaultGate, FaultInjectingObjectStore, ObjectFault, ObjectOp,
        ObjectPattern,
    };

    fn reference(table: &str, key: &str, bytes: &[u8]) -> ObjectReference {
        ObjectReference {
            id: ObjectId::table(table, key).unwrap(),
            version: ObjectVersion {
                e_tag: None,
                provider_version: None,
                byte_size: bytes.len() as u64,
                local_value: None,
            },
        }
    }

    #[tokio::test]
    async fn local_path_materializes_and_deletes_with_the_object() {
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
        store.gc().await.unwrap();
        assert!(path.exists());

        store.delete(&reference.id).await.unwrap();
        assert!(!path.exists());
        std::fs::remove_dir_all(remote_root).ok();
        std::fs::remove_dir_all(cache_root).ok();
    }

    #[test]
    fn cache_hits_do_not_read_file_contents() {
        let root = unique_dir("object_cache_hit");
        let cache = FileCache::new(root.clone()).unwrap();
        let reference = reference("iris.worker", "objects/cached.parquet", b"real");
        let path = cache.write(&reference, b"real").unwrap();

        // Cache lookup intentionally performs metadata I/O only. A same-sized
        // change remains a hit instead of triggering a full-file read.
        std::fs::write(&path, b"fake").unwrap();
        assert_eq!(cache.cached_path(&reference).unwrap(), Some(path));
        std::fs::remove_dir_all(root).ok();
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

    #[tokio::test]
    async fn cache_cleanup_admits_readers_before_removing_every_file() {
        let (store, roots) = cached_store("object_cache_cleanup_readers", None);
        let paths: Vec<_> = (0..CACHE_DELETE_BATCH_FILES * 3)
            .map(|index| roots[1].join(format!("victim-{index}")))
            .collect();
        for path in &paths {
            std::fs::write(path, b"cached").unwrap();
        }

        let original_reader = store.query_visibility.read().await;
        let mut cleanup = Box::pin(store.remove_cached_paths(paths.clone()));
        assert!(futures::poll!(&mut cleanup).is_pending());
        let mut next_reader = Box::pin(store.query_visibility.read());
        assert!(futures::poll!(&mut next_reader).is_pending());
        drop(original_reader);

        let (removed, ()) = tokio::join!(cleanup, async {
            let _reader = next_reader.await;
            let remaining = paths.iter().filter(|path| path.exists()).count();
            assert!(
                remaining > 0,
                "cleanup monopolized the lock until every unlink finished"
            );
            assert!(
                remaining < paths.len(),
                "reader did not follow the first cleanup batch"
            );
        });
        removed.unwrap();
        assert!(paths.iter().all(|path| !path.exists()));
        roots.iter().for_each(|root| {
            std::fs::remove_dir_all(root).unwrap();
        });
    }

    #[tokio::test]
    async fn optional_warming_does_not_queue_ahead_of_required_downloads() {
        let remote_root = unique_dir("object_cache_warm_bound_remote");
        let cache_root = unique_dir("object_cache_warm_bound_local");
        let remote = Arc::new(
            build_remote_object_store(remote_root.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let objects: Vec<_> = (0..MAX_PARALLEL_FETCHES + 3)
            .map(|index| reference("log", &format!("objects/{index}.parquet"), b"rows"))
            .collect();
        for object in &objects {
            remote
                .write(&object.id, bytes::Bytes::from_static(b"rows"))
                .await
                .unwrap();
        }
        let source = FaultInjectingObjectStore::new(remote);
        let gate = FaultGate::new();
        source.arm(
            ObjectFault::new(
                ObjectOp::Read,
                ObjectPattern::Contains("/objects/".to_string()),
                FaultAction::Park(Arc::clone(&gate)),
            )
            .forever(),
        );
        let store = CachedObjectStore::new(
            source.clone(),
            cache_root.clone(),
            Arc::new(RwLock::new(())),
            None,
        )
        .unwrap();
        for object in &objects[..MAX_PARALLEL_FETCHES] {
            store.warm(object);
        }
        for _ in 0..MAX_PARALLEL_FETCHES {
            gate.entered().await;
        }
        for object in &objects[MAX_PARALLEL_FETCHES..objects.len() - 1] {
            store.warm(object);
        }
        let required = objects.last().unwrap();
        let download_store = store.clone();
        let required_object = required.clone();
        let download =
            tokio::spawn(async move { download_store.local_path(&required_object).await });

        gate.release();
        gate.entered().await;
        assert_eq!(
            source.keys_for(ObjectOp::Read).last().unwrap(),
            required.id.as_str(),
            "optional cache fills kept running ahead of the required download"
        );
        source.clear_faults();
        for _ in 0..MAX_PARALLEL_FETCHES {
            gate.release();
        }
        let path = download.await.unwrap().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"rows");
        // Drain background writes before removing their directories.
        let _drained = store
            .fetches
            .acquire_many(MAX_PARALLEL_FETCHES as u32)
            .await
            .unwrap();
        for skipped in &objects[MAX_PARALLEL_FETCHES..objects.len() - 1] {
            assert!(store.cached_path(skipped).await.unwrap().is_none());
        }
        for root in [remote_root, cache_root] {
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[tokio::test]
    async fn query_warming_requires_reuse_and_shares_byte_and_object_limits() {
        // Each case tests observable cache contents with real remote files.
        // The first limit binds bytes, the second binds object count.
        for (bytes, objects) in [(8, 10), (100, 2)] {
            let (store, roots) = cached_store("object_cache_query_admission", None);
            let references: Vec<_> = (0..4)
                .map(|index| reference("log", &format!("objects/{index}.parquet"), b"rows"))
                .collect();
            for object in &references {
                store
                    .source
                    .write(&object.id, bytes::Bytes::from_static(b"rows"))
                    .await
                    .unwrap();
            }

            let first_query = QueryCacheAdmission::new(bytes, objects);
            for object in &references {
                // Multiple scans in one query do not qualify as reuse.
                store.warm_query(object, &first_query);
                store.warm_query(object, &first_query);
            }
            {
                let _drained = store
                    .fetches
                    .acquire_many(MAX_PARALLEL_FETCHES as u32)
                    .await
                    .unwrap();
                for object in &references {
                    assert!(store.cached_path(object).await.unwrap().is_none());
                }
            }

            // Only two four-byte fills fit either case.
            let second_query = QueryCacheAdmission::new(bytes, objects);
            for object in &references {
                store.warm_query(object, &second_query);
            }
            {
                let _drained = store
                    .fetches
                    .acquire_many(MAX_PARALLEL_FETCHES as u32)
                    .await
                    .unwrap();
                for (index, object) in references.iter().enumerate() {
                    let path = store.cached_path(object).await.unwrap();
                    assert_eq!(path.is_some(), index < 2);
                    if let Some(path) = path {
                        assert_eq!(std::fs::read(path).unwrap(), b"rows");
                    }
                }
            }

            // Exhausting one query's allowance cannot block required reads or
            // prevent a later query from warming previously skipped objects.
            let path = store.local_path(&references[2]).await.unwrap();
            assert_eq!(std::fs::read(path).unwrap(), b"rows");
            store.warm_query(&references[3], &QueryCacheAdmission::new(bytes, objects));
            let _drained = store
                .fetches
                .acquire_many(MAX_PARALLEL_FETCHES as u32)
                .await
                .unwrap();
            assert!(store.cached_path(&references[3]).await.unwrap().is_some());
            for root in roots {
                std::fs::remove_dir_all(root).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn oversized_query_fill_leaves_room_for_small_objects() {
        let (store, roots) = cached_store("object_cache_query_oversize", None);
        let large = reference("log", "objects/large.parquet", b"large object");
        let small = reference("log", "indices/small.fidx", b"tiny");
        for (object, bytes) in [(&large, &b"large object"[..]), (&small, &b"tiny"[..])] {
            store
                .source
                .write(&object.id, bytes::Bytes::copy_from_slice(bytes))
                .await
                .unwrap();
        }
        for _ in 0..2 {
            let query = QueryCacheAdmission::new(4, 1);
            store.warm_query(&large, &query);
            store.warm_query(&small, &query);
        }
        let _drained = store
            .fetches
            .acquire_many(MAX_PARALLEL_FETCHES as u32)
            .await
            .unwrap();
        assert!(store.cached_path(&large).await.unwrap().is_none());
        assert!(store.cached_path(&small).await.unwrap().is_some());
        for root in roots {
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[tokio::test]
    async fn sql_query_shares_admission_across_namespaces_and_repeated_scans() {
        let (store, roots) = cached_store("object_cache_sql_budget", None);
        let store = Arc::new(store);
        let batch = arrow::record_batch::RecordBatch::try_from_iter(vec![(
            "seq",
            Arc::new(arrow::array::Int64Array::from(vec![1])) as arrow::array::ArrayRef,
        )])
        .unwrap();
        let bytes = bytes::Bytes::from(crate::store::segment::write_segment(&batch).unwrap());
        let mut sources = Vec::new();
        for name in ["left_table", "right_table"] {
            let id = ObjectId::table(name, "objects/one.parquet").unwrap();
            let version = store.source.write(&id, bytes.clone()).await.unwrap();
            sources.push((name, ObjectReference { id, version }));
        }
        for query_number in 0..2 {
            let config = make_ctx()
                .copied_config()
                .with_extension(Arc::new(QueryCacheAdmission::new(u64::MAX, 1)));
            let ctx = datafusion::prelude::SessionContext::new_with_config(config);
            let providers = sources
                .iter()
                .map(|(name, source)| {
                    let path = store
                        .planned_local_path(&source.id)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    RegisteredProvider {
                        name: name.to_string(),
                        provider: NamespaceProvider::build_with_local_artifacts(
                            batch.schema(),
                            std::slice::from_ref(&path),
                        )
                        .unwrap()
                        .with_object_sources(
                            store.clone(),
                            std::collections::BTreeMap::from([(
                                path,
                                SegmentObjects {
                                    source: source.clone(),
                                    artifacts: ArtifactReferences::default(),
                                },
                            )]),
                        ),
                    }
                })
                .collect();
            let result = run_query_over(&ctx, providers,
                "SELECT seq FROM left_table UNION ALL SELECT seq FROM right_table UNION ALL SELECT seq FROM left_table")
                .await.unwrap();
            assert_eq!(
                result
                    .batches
                    .iter()
                    .map(|batch| batch.num_rows())
                    .sum::<usize>(),
                3
            );
            let _drained = store
                .fetches
                .acquire_many(MAX_PARALLEL_FETCHES as u32)
                .await
                .unwrap();
            let mut cached = 0;
            for (_, source) in &sources {
                cached += usize::from(store.cached_path(source).await.unwrap().is_some());
            }
            assert_eq!(
                cached, query_number,
                "one SQL request must share one fill budget"
            );
        }
        for root in roots {
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
