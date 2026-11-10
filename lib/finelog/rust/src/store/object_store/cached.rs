//! Transparent local-file decorator for an object store.
//!
//! Callers use the [`ObjectStore`] interface. Cache membership and filesystem
//! layout stay private to these wrappers. `local_path` materializes a file;
//! Range readers pin cached files while they are in use and read misses remotely.
//!
//! Writes are dual-ported: the bytes an upload already holds also land in the
//! cache, so the flush → query path never re-downloads its own output. Cache
//! misses download under one store-wide concurrency bound. When a capacity is
//! configured, `gc` evicts least-recently-used files below it; every hit
//! refreshes the file's modification time. Queries and maintenance pin the
//! individual cached files they use; eviction skips these files without
//! queuing a writer on the store-wide visibility lock.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::errors::StatsError;
use crate::read_work::ReadWorkers;
use crate::store::object_store::{
    DeleteManyOutcome, ObjectId, ObjectMetadata, ObjectPrefix, ObjectReference, ObjectStore,
    ObjectVersion, StoredObject, FINELOG_ROOT_COMPONENT,
};

use super::admission::{AdmissionDecision, QueryCacheAdmission, RequestHistory};
use super::diagnostics::{CacheDiagnostics, FillReason};
use super::local_file::atomic_write;

/// Concurrent cache-miss downloads across every caller of this store.
pub(crate) const MAX_PARALLEL_FETCHES: usize = 8;
const MAX_PARALLEL_CACHE_READS: usize = 8;

/// Files kept resident for a read, independent of query warming policy.
#[derive(Default)]
pub struct CachePins {
    files: Mutex<HashMap<PathBuf, CacheFilePin>>,
}

const MAX_PARALLEL_UPLOADS: usize = 8;

/// Yield between batches of filesystem removals.
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
    pins: Arc<Mutex<HashMap<PathBuf, usize>>>,
}

pub(super) struct CacheFilePin {
    path: PathBuf,
    pins: Arc<Mutex<HashMap<PathBuf, usize>>>,
}

impl Drop for CacheFilePin {
    fn drop(&mut self) {
        let mut pins = self.pins.lock().unwrap();
        let count = pins.get_mut(&self.path).expect("cached file is pinned");
        *count -= 1;
        if *count == 0 {
            pins.remove(&self.path);
        }
    }
}

impl FileCache {
    fn new(root: PathBuf) -> Result<Self, StatsError> {
        std::fs::create_dir_all(&root).map_err(|error| {
            StatsError::Internal(format!("create object cache {}: {error}", root.display()))
        })?;
        Ok(Self {
            root,
            pins: Arc::new(Mutex::new(HashMap::new())),
        })
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

    fn pin(&self, reference: &ObjectReference) -> Result<Option<CacheFilePin>, StatsError> {
        let mut pins = self.pins.lock().unwrap();
        let Some(path) = self.cached_path(reference)? else {
            return Ok(None);
        };
        *pins.entry(path.clone()).or_default() += 1;
        Ok(Some(CacheFilePin {
            path,
            pins: Arc::clone(&self.pins),
        }))
    }

    fn remove(&self, path: &Path) -> Result<bool, StatsError> {
        let pins = self.pins.lock().unwrap();
        if pins.contains_key(path) {
            return Ok(false);
        }
        match std::fs::remove_file(path) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
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
    read_workers: Arc<ReadWorkers>,
    fill_workers: Arc<ReadWorkers>,
    /// Bounds write-through uploads independently of reads and flush encoding.
    uploads: Arc<Semaphore>,
    /// Total cache bytes to retain; `None` retains everything.
    capacity_bytes: Option<u64>,
    last_gc: Arc<Mutex<Option<Instant>>>,
    gc_running: Arc<tokio::sync::Mutex<()>>,
    /// Objects staged locally whose upload has not completed. Their cache files
    /// are the only copy of the bytes, so eviction never selects them.
    staged: Arc<Mutex<HashSet<PathBuf>>>,
    request_history: Arc<Mutex<RequestHistory>>,
    diagnostics: Arc<CacheDiagnostics>,
}

impl CachedObjectStore {
    pub fn new(
        source: Arc<dyn ObjectStore>,
        root: PathBuf,
        capacity_bytes: Option<u64>,
    ) -> Result<Self, StatsError> {
        Ok(Self {
            source,
            cache: FileCache::new(root)?,
            fetches: Arc::new(Semaphore::new(MAX_PARALLEL_FETCHES)),
            read_workers: Arc::new(ReadWorkers::new(MAX_PARALLEL_CACHE_READS)),
            fill_workers: Arc::new(ReadWorkers::new(MAX_PARALLEL_FETCHES)),
            uploads: Arc::new(Semaphore::new(MAX_PARALLEL_UPLOADS)),
            capacity_bytes,
            last_gc: Arc::new(Mutex::new(None)),
            gc_running: Arc::new(tokio::sync::Mutex::new(())),
            staged: Arc::new(Mutex::new(HashSet::new())),
            request_history: Arc::new(Mutex::new(RequestHistory::default())),
            diagnostics: Arc::new(CacheDiagnostics::default()),
        })
    }

    async fn lookup(&self, reference: &ObjectReference) -> Result<Option<PathBuf>, StatsError> {
        let cache = self.cache.clone();
        let reference = reference.clone();
        let path = self
            .read_workers
            .run(move || cache.cached_path(&reference))
            .await?;
        let counter = if path.is_some() {
            &self.diagnostics.hits
        } else {
            &self.diagnostics.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(path)
    }

    async fn materialize(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        if let Some(path) = self.lookup(reference).await? {
            return Ok(path);
        }
        let permit = Arc::clone(&self.fetches)
            .acquire_owned()
            .await
            .map_err(|error| {
                StatsError::Internal(format!("object cache fetch semaphore: {error}"))
            })?;
        self.materialize_reserved(reference, FillReason::Required, permit)
            .await
    }

    /// Materialize with a download slot already held by the caller.
    async fn materialize_reserved(
        &self,
        reference: &ObjectReference,
        reason: FillReason,
        permit: OwnedSemaphorePermit,
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
        let diagnostics = self.diagnostics.clone();
        self.fill_workers
            .run(move || {
                let _permit = permit;
                let path = cache.write(&reference, &object.bytes)?;
                diagnostics.fills(reason).record(object.bytes.len() as u64);
                Ok(path)
            })
            .await
    }

    /// Evict least-recently-used cache files until the cache fits its capacity.
    async fn evict_over_capacity(&self, capacity_bytes: u64) -> Result<(), StatsError> {
        let started = Instant::now();
        let root = self.cache.root.join(FINELOG_ROOT_COMPONENT);
        let mut entries = tokio::task::spawn_blocking(move || scan_cache_files(&root))
            .await
            .map_err(|error| StatsError::Internal(format!("object cache scan task: {error}")))??;
        let total: u64 = entries.iter().map(|entry| entry.bytes).sum();
        let files = entries.len();
        let scan_ms = started.elapsed().as_millis() as u64;
        if total > capacity_bytes {
            entries.sort_by_key(|entry| entry.modified);
        }
        let staged = self.staged.lock().unwrap().clone();
        let pinned = self
            .cache
            .pins
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        let mut excess = total.saturating_sub(capacity_bytes);
        let mut victims = Vec::new();
        let mut selected_bytes = 0;
        for entry in entries {
            if excess == 0 {
                break;
            }
            // A staged file is the only copy of its bytes until its upload
            // completes; evicting it would lose acknowledged rows.
            if staged.contains(&entry.path) || pinned.contains(&entry.path) {
                continue;
            }
            excess = excess.saturating_sub(entry.bytes);
            selected_bytes += entry.bytes;
            victims.push(entry.path);
        }
        let selected_files = victims.len();
        let removal = self.remove_cached_paths(victims).await?;
        tracing::info!(
            evicted = removal.removed,
            selected_files,
            selected_bytes,
            cache_bytes = total,
            capacity_bytes,
            files,
            remaining_excess_bytes = excess,
            scan_ms,
            sweep_ms = started.elapsed().as_millis() as u64,
            cleanup_ms = removal.elapsed.as_millis() as u64,
            max_batch_us = removal.max_batch.as_micros() as u64,
            "object cache sweep"
        );
        Ok(())
    }

    async fn remove_cached_paths(&self, paths: Vec<PathBuf>) -> Result<RemovalStats, StatsError> {
        let mut stats = RemovalStats::default();
        for batch in paths.chunks(CACHE_DELETE_BATCH_FILES) {
            let batch = batch.to_vec();
            let holding = Instant::now();
            let diagnostics = self.diagnostics.clone();
            let cache = self.cache.clone();
            let (removed, elapsed) = tokio::task::spawn_blocking(move || {
                let mut removed = 0;
                let result: Result<(), StatsError> = batch.into_iter().try_for_each(|path| {
                    if cache.remove(&path)? {
                        removed += 1;
                    }
                    Ok(())
                });
                let elapsed = holding.elapsed();
                diagnostics.cleanup_batches.fetch_add(1, Ordering::Relaxed);
                diagnostics
                    .cleanup_time_us
                    .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
                diagnostics
                    .cleanup_max_batch_us
                    .fetch_max(elapsed.as_micros() as u64, Ordering::Relaxed);
                result.map(|()| (removed, elapsed))
            })
            .await
            .map_err(|error| {
                StatsError::Internal(format!("object cache removal task: {error}"))
            })??;
            stats.removed += removed;
            stats.elapsed += elapsed;
            stats.max_batch = stats.max_batch.max(elapsed);
            tokio::task::yield_now().await;
        }
        Ok(stats)
    }

    fn start_warm(&self, reference: &ObjectReference, query: &QueryCacheAdmission) {
        // Reserve before spawning so cancelled queries cannot leave queued fills.
        let Ok(permit) = self.fetches.clone().try_acquire_owned() else {
            self.diagnostics.warm_busy.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let store = self.clone();
        let reference = reference.clone();
        let deadline = query.fill_deadline();
        let task = tokio::spawn(async move {
            let fill = async {
                let result = store
                    .materialize_reserved(&reference, FillReason::Query, permit)
                    .await;
                if let Err(error) = &result {
                    store
                        .diagnostics
                        .warm_errors
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(%error, object = ?reference.id.as_str(),
                    "background cache fill failed; the next scan reads remote again");
                }
            };
            match deadline {
                Some(deadline) => {
                    let _ = tokio::time::timeout_at(deadline, fill).await;
                }
                None => fill.await,
            }
        });
        query.track_fill(task.abort_handle());
    }
}

#[derive(Default)]
struct RemovalStats {
    removed: usize,
    elapsed: Duration,
    max_batch: Duration,
}

#[async_trait]
impl ObjectStore for CachedObjectStore {
    async fn read_range(
        &self,
        reference: &ObjectReference,
        range: std::ops::Range<u64>,
    ) -> Result<Option<bytes::Bytes>, StatsError> {
        Ok(self
            .read_ranges(reference, &[range])
            .await?
            .and_then(|mut bytes| bytes.pop()))
    }

    async fn read_ranges(
        &self,
        reference: &ObjectReference,
        ranges: &[std::ops::Range<u64>],
    ) -> Result<Option<Vec<bytes::Bytes>>, StatsError> {
        let cache = self.cache.clone();
        let reference_for_cache = reference.clone();
        let local_ranges = ranges.to_vec();
        let local = self
            .read_workers
            .run(move || {
                let Some(pin) = cache.pin(&reference_for_cache)? else {
                    return Ok(None);
                };
                let read = || -> std::io::Result<Vec<bytes::Bytes>> {
                    let file = std::fs::File::open(&pin.path)?;
                    local_ranges
                        .into_iter()
                        .map(|range| {
                            let len = usize::try_from(
                                range.end.checked_sub(range.start).ok_or_else(|| {
                                    std::io::Error::new(
                                        std::io::ErrorKind::InvalidInput,
                                        "reversed object range",
                                    )
                                })?,
                            )
                            .map_err(std::io::Error::other)?;
                            let mut bytes = vec![0; len];
                            file.read_exact_at(&mut bytes, range.start)?;
                            Ok(bytes::Bytes::from(bytes))
                        })
                        .collect()
                };
                read().map(Some).map_err(|error| {
                    StatsError::Internal(format!(
                        "read cached ranges {}: {error}",
                        pin.path.display()
                    ))
                })
            })
            .await?;
        if local.is_some() {
            return Ok(local);
        }
        self.source.read_ranges(reference, ranges).await
    }

    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        let version = self.source.write(id, bytes.clone()).await?;
        let cache = self.cache.clone();
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        let diagnostics = self.diagnostics.clone();
        tokio::task::spawn_blocking(move || {
            cache.write(&reference, &bytes)?;
            diagnostics.writes.record(bytes.len() as u64);
            Ok::<_, StatsError>(())
        })
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
        let diagnostics = self.diagnostics.clone();
        let path = tokio::task::spawn_blocking(move || {
            let path = cache.write(&reference, &bytes)?;
            diagnostics.writes.record(bytes.len() as u64);
            Ok::<_, StatsError>(path)
        })
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

    async fn pin_cached_path(
        &self,
        reference: &ObjectReference,
        pins: &CachePins,
    ) -> Result<Option<PathBuf>, StatsError> {
        let cache = self.cache.clone();
        let reference = reference.clone();
        let pin = self.read_workers.run(move || cache.pin(&reference)).await?;
        let counter = if pin.is_some() {
            &self.diagnostics.hits
        } else {
            &self.diagnostics.misses
        };
        counter.fetch_add(1, Ordering::Relaxed);
        let Some(pin) = pin else {
            return Ok(None);
        };
        let path = pin.path.clone();
        pins.files
            .lock()
            .unwrap()
            .entry(path.clone())
            .or_insert(pin);
        Ok(Some(path))
    }

    fn warm_query(&self, reference: &ObjectReference, query: &QueryCacheAdmission) {
        let counter = match query.admit(reference, &self.request_history) {
            AdmissionDecision::Admitted => {
                self.start_warm(reference, query);
                &self.diagnostics.query_admitted
            }
            AdmissionDecision::Cold => &self.diagnostics.query_cold,
            AdmissionDecision::Duplicate => &self.diagnostics.query_duplicate,
            AdmissionDecision::ObjectLimit => &self.diagnostics.query_object_limit,
            AdmissionDecision::ByteLimit => &self.diagnostics.query_byte_limit,
        };
        counter.fetch_add(1, Ordering::Relaxed);
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
        // Remote collection already honors the table's snapshot retention.
        self.source.delete(id).await?;
        self.remove_cached_paths(vec![self.cache.path(id)]).await?;
        Ok(())
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
        let staged_objects = self.staged.lock().unwrap().len();
        self.diagnostics.log_if_due(
            &self.request_history,
            MAX_PARALLEL_FETCHES - self.fetches.available_permits(),
            staged_objects,
        );
        let Some(capacity_bytes) = self.capacity_bytes else {
            return Ok(());
        };
        let Ok(_sweep) = self.gc_running.try_lock() else {
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
        let store = CachedObjectStore::new(source, cache_root.clone(), None).unwrap();
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
        let store = CachedObjectStore::new(source, cache_root.clone(), capacity_bytes).unwrap();
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
        assert_eq!(
            store.read_range(&reference, 5..11).await.unwrap().unwrap(),
            &b"ported"[..]
        );
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
    async fn cache_cleanup_skips_pinned_files_without_blocking_other_reads() {
        let (store, roots) = cached_store("object_cache_cleanup_readers", None);
        let pinned = reference("log", "objects/pinned.parquet", b"pinned");
        let unpinned = reference("log", "objects/unpinned.parquet", b"unrelated");
        for (object, bytes) in [(&pinned, &b"pinned"[..]), (&unpinned, &b"unrelated"[..])] {
            store
                .write(&object.id, bytes::Bytes::copy_from_slice(bytes))
                .await
                .unwrap();
        }
        let pins = CachePins::default();
        let path = store
            .pin_cached_path(&pinned, &pins)
            .await
            .unwrap()
            .unwrap();
        store
            .remove_cached_paths(vec![path.clone(), store.cache.path(&unpinned.id)])
            .await
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"pinned");
        assert!(store.cached_path(&unpinned).await.unwrap().is_none());
        assert_eq!(
            store
                .read_range(&unpinned, 0..9)
                .await
                .unwrap()
                .unwrap()
                .as_ref(),
            b"unrelated"
        );
        drop(pins);
        store.remove_cached_paths(vec![path.clone()]).await.unwrap();
        assert!(!path.exists());
        for root in roots {
            std::fs::remove_dir_all(root).unwrap();
        }
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
        let store = CachedObjectStore::new(source.clone(), cache_root.clone(), None).unwrap();
        let first_query = QueryCacheAdmission::default();
        for object in &objects {
            store.warm_query(object, &first_query);
        }
        let repeated_query = QueryCacheAdmission::default();
        for object in &objects[..MAX_PARALLEL_FETCHES] {
            store.warm_query(object, &repeated_query);
        }
        for _ in 0..MAX_PARALLEL_FETCHES {
            gate.entered().await;
        }
        for object in &objects[MAX_PARALLEL_FETCHES..objects.len() - 1] {
            store.warm_query(object, &repeated_query);
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
    async fn cancelled_and_successful_requests_bound_optional_downloads() {
        for request_succeeded in [false, true] {
            let (base, roots) = cached_store("object_cache_cancel_query", None);
            let object = reference("log", "objects/cancel.parquet", b"rows");
            base.source
                .write(&object.id, bytes::Bytes::from_static(b"rows"))
                .await
                .unwrap();
            let source = FaultInjectingObjectStore::new(base.source);
            let gate = FaultGate::new();
            source.arm(
                ObjectFault::new(
                    ObjectOp::Read,
                    ObjectPattern::Contains("cancel.parquet".to_string()),
                    FaultAction::Park(Arc::clone(&gate)),
                )
                .forever(),
            );
            let store = CachedObjectStore::new(source, roots[1].clone(), None).unwrap();
            let first_query = QueryCacheAdmission::default();
            store.warm_query(&object, &first_query);
            let query = Arc::new(QueryCacheAdmission::default());
            let scope = query.cancel_on_drop(Some(Duration::from_secs(10)));
            store.warm_query(&object, &query);
            gate.entered().await;
            if request_succeeded {
                scope.complete();
                // Pause only after the download reaches the gate; filesystem work
                // must not let automatic clock advancement expire it before entry.
                tokio::time::pause();
                tokio::time::advance(Duration::from_secs(10)).await;
            } else {
                drop(scope);
            }
            let _drained = store
                .fetches
                .acquire_many(MAX_PARALLEL_FETCHES as u32)
                .await
                .unwrap();
            assert!(store.cached_path(&object).await.unwrap().is_none());
            for root in roots {
                std::fs::remove_dir_all(root).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn query_warming_requires_reuse_and_shares_byte_and_object_limits() {
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
            let later_query = QueryCacheAdmission::new(bytes, objects);
            store.warm_query(&references[3], &later_query);
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
    async fn query_warming_never_exceeds_the_byte_budget() {
        for warm_small_first in [false, true] {
            let (store, roots) = cached_store("object_cache_query_oversize", None);
            let small = reference("log", "indices/small.fidx", b"tiny");
            let large = reference("log", "objects/large.parquet", b"large object");
            let tail = reference("log", "objects/tail.parquet", b"tail");
            for (object, bytes) in [
                (&small, &b"tiny"[..]),
                (&large, &b"large object"[..]),
                (&tail, &b"tail"[..]),
            ] {
                store
                    .source
                    .write(&object.id, bytes::Bytes::copy_from_slice(bytes))
                    .await
                    .unwrap();
            }
            let queries = [
                QueryCacheAdmission::new(8, 10),
                QueryCacheAdmission::new(8, 10),
            ];
            for query in &queries {
                if warm_small_first {
                    store.warm_query(&small, query);
                }
                store.warm_query(&large, query);
                store.warm_query(&tail, query);
            }
            let _drained = store
                .fetches
                .acquire_many(MAX_PARALLEL_FETCHES as u32)
                .await
                .unwrap();
            assert!(store.cached_path(&large).await.unwrap().is_none());
            assert_eq!(
                store.cached_path(&small).await.unwrap().is_some(),
                warm_small_first
            );
            assert!(store.cached_path(&tail).await.unwrap().is_some());
            for root in roots {
                std::fs::remove_dir_all(root).unwrap();
            }
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
