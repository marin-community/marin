//! object_store remote sync surface (LOCAL -> BOTH -> REMOTE).
//!
//! `build_remote_store` dispatches on the configured `remote_log_dir`:
//! `gs://bucket/prefix` -> `GoogleCloudStorageBuilder` (GCP prod);
//! `s3://bucket/prefix` -> `AmazonS3Builder` for any S3-compatible store
//! (Cloudflare R2 / CoreWeave Object Storage on CoreWeave clusters); any other
//! non-empty value -> `LocalFileSystem` rooted at that directory (tests pass a
//! plain tmp path). An empty `remote_log_dir` disables sync (returns `None`).
//!
//! The on-disk layout is `{remote_log_dir}/{namespace}/{relative segment key}`; the
//! `RemoteStore` carries an optional bucket-relative `prefix` (the `gs://`
//! path component) and every per-namespace op composes `{prefix}/{namespace}`.
//! object_store 0.13 moved `put`/`get`/`head`/`delete` onto the `ObjectStoreExt`
//! blanket trait, which must be in scope.

use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutOptions, UpdateVersion};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;

/// A configured remote object store plus the bucket-relative prefix the store
/// is rooted under (empty for a `LocalFileSystem` rooted at the remote dir).
#[derive(Clone)]
pub struct RemoteStore {
    store: Arc<dyn ObjectStore>,
    prefix: String,
    local_root: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct RemoteObject {
    pub bytes: bytes::Bytes,
    pub version: RemoteVersion,
}

#[derive(Debug, Clone)]
pub struct RemoteVersion {
    pub e_tag: Option<String>,
    pub provider_version: Option<String>,
    pub content_sha256: [u8; 32],
}

const GCS_SCHEME: &str = "gs://";
const S3_SCHEME: &str = "s3://";

/// Whether `remote_log_dir` names an object store rather than a local directory.
pub fn is_object_store(remote_log_dir: &str) -> bool {
    let dir = remote_log_dir.trim();
    dir.starts_with(GCS_SCHEME) || dir.starts_with(S3_SCHEME)
}

/// Build the remote store from `remote_log_dir`, or `None` when sync is
/// disabled (empty string).
///
/// `gs://bucket/sub/dir` -> a GCS store on `bucket` with prefix `sub/dir`.
/// `s3://bucket/sub/dir` -> an S3-compatible store on `bucket` with prefix
/// `sub/dir`. Everything else about the connection comes from the standard
/// `AWS_*` env the deploy environment injects (on CoreWeave: the `iris-task-env`
/// Secret's R2 creds). `AmazonS3Builder::from_env` reads credentials, region,
/// the custom `AWS_ENDPOINT_URL`, and `AWS_VIRTUAL_HOSTED_STYLE_REQUEST` — so
/// iris owns the addressing-style decision (path-style for R2, virtual-hosted
/// for CoreWeave Object Storage) and this server stays endpoint-agnostic.
/// Any other value -> a `LocalFileSystem` rooted at that (created) directory,
/// with an empty prefix, writing into `{remote_log_dir}/{namespace}/{relative segment key}`.
pub fn build_remote_store(remote_log_dir: &str) -> Result<Option<RemoteStore>, StatsError> {
    let dir = remote_log_dir.trim_end_matches('/');
    if dir.is_empty() {
        return Ok(None);
    }
    if let Some(rest) = dir.strip_prefix(GCS_SCHEME) {
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p),
            None => (rest, ""),
        };
        let store = object_store::gcp::GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| StatsError::Internal(format!("build gcs store {bucket:?}: {e}")))?;
        return Ok(Some(RemoteStore {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
            local_root: None,
        }));
    }
    if let Some(rest) = dir.strip_prefix(S3_SCHEME) {
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p),
            None => (rest, ""),
        };
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| StatsError::Internal(format!("build s3 store {bucket:?}: {e}")))?;
        return Ok(Some(RemoteStore {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
            local_root: None,
        }));
    }
    // Local filesystem remote (tests). Root the store at the remote dir so
    // object paths are `{namespace}/{relative segment key}`.
    std::fs::create_dir_all(dir)
        .map_err(|e| StatsError::Internal(format!("create remote dir {dir}: {e}")))?;
    let store = LocalFileSystem::new_with_prefix(dir)
        .map_err(|e| StatsError::Internal(format!("local remote store {dir}: {e}")))?;
    Ok(Some(RemoteStore {
        store: Arc::new(store),
        prefix: String::new(),
        local_root: Some(PathBuf::from(dir)),
    }))
}

impl RemoteStore {
    /// Split the configured prefix on `/` into individual path components.
    /// `OsPath::from_iter` escapes `/` *within* a single part, so a multi-segment
    /// prefix like `logs/sub` must be pushed component-by-component.
    fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.prefix.split('/').filter(|s| !s.is_empty())
    }

    /// The object path for `{prefix}/{namespace}/{relative segment key}`.
    fn object_path(&self, namespace: &str, relative_key: &str) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain([namespace])
            .chain(relative_key.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    /// The directory prefix for one namespace, `{prefix}/{namespace}`.
    fn namespace_prefix(&self, namespace: &str) -> OsPath {
        let parts: Vec<&str> = self.prefix_parts().chain([namespace]).collect();
        OsPath::from_iter(parts)
    }

    fn native_path(&self, namespace: &str, relative_key: &str) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain(["_native", "namespaces", namespace])
            .chain(relative_key.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    fn native_prefix(&self, namespace: &str, relative_prefix: &str) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain(["_native", "namespaces", namespace])
            .chain(relative_prefix.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    pub async fn list_native_namespaces(&self) -> Result<Vec<String>, StatsError> {
        let root = OsPath::from_iter(self.prefix_parts().chain(["_native", "namespaces"]));
        let mut stream = self.store.list(Some(&root));
        let mut namespaces = std::collections::BTreeSet::new();
        while let Some(result) = stream.next().await {
            let meta = result.map_err(|error| {
                StatsError::Internal(format!("list native namespaces {root}: {error}"))
            })?;
            let Some(mut parts) = meta.location.prefix_match(&root) else {
                continue;
            };
            if let Some(namespace) = parts.next() {
                namespaces.insert(namespace.as_ref().to_string());
            }
        }
        Ok(namespaces.into_iter().collect())
    }

    pub async fn get_native(
        &self,
        namespace: &str,
        relative_key: &str,
    ) -> Result<Option<RemoteObject>, StatsError> {
        let path = self.native_path(namespace, relative_key);
        let result = match self.store.get(&path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "read native object {path}: {error}"
                )))
            }
        };
        let e_tag = result.meta.e_tag.clone();
        let provider_version = result.meta.version.clone();
        let bytes = result.bytes().await.map_err(|error| {
            StatsError::Internal(format!("read native object body {path}: {error}"))
        })?;
        Ok(Some(RemoteObject {
            version: RemoteVersion {
                e_tag,
                provider_version,
                content_sha256: Sha256::digest(&bytes).into(),
            },
            bytes,
        }))
    }

    /// Read one object from the legacy namespace prefix.
    pub async fn get(
        &self,
        namespace: &str,
        relative_key: &str,
    ) -> Result<Option<RemoteObject>, StatsError> {
        let path = self.object_path(namespace, relative_key);
        let result = match self.store.get(&path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "read remote object {path}: {error}"
                )))
            }
        };
        let e_tag = result.meta.e_tag.clone();
        let provider_version = result.meta.version.clone();
        let bytes = result.bytes().await.map_err(|error| {
            StatsError::Internal(format!("read remote object body {path}: {error}"))
        })?;
        Ok(Some(RemoteObject {
            version: RemoteVersion {
                e_tag,
                provider_version,
                content_sha256: Sha256::digest(&bytes).into(),
            },
            bytes,
        }))
    }

    /// Create an immutable object, accepting an identical retry.
    pub async fn put_native_immutable(
        &self,
        namespace: &str,
        relative_key: &str,
        bytes: bytes::Bytes,
    ) -> Result<RemoteVersion, StatsError> {
        let path = self.native_path(namespace, relative_key);
        let content_sha256 = Sha256::digest(&bytes).into();
        let result = self
            .store
            .put_opts(
                &path,
                bytes.clone().into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await;
        match result {
            Ok(result) => Ok(RemoteVersion {
                e_tag: result.e_tag,
                provider_version: result.version,
                content_sha256,
            }),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let existing =
                    self.get_native(namespace, relative_key)
                        .await?
                        .ok_or_else(|| {
                            StatsError::Internal(format!(
                                "native object {path} disappeared after create conflict"
                            ))
                        })?;
                if existing.bytes == bytes {
                    Ok(existing.version)
                } else {
                    Err(StatsError::SchemaConflict(format!(
                        "immutable native object {path} already exists with different contents"
                    )))
                }
            }
            Err(error) => Err(StatsError::Internal(format!(
                "create native object {path}: {error}"
            ))),
        }
    }

    /// Atomically create or replace a mutable native pointer.
    pub async fn compare_and_swap_native(
        &self,
        namespace: &str,
        relative_key: &str,
        expected: Option<&RemoteVersion>,
        bytes: bytes::Bytes,
    ) -> Result<RemoteVersion, StatsError> {
        if let Some(local_root) = &self.local_root {
            let path = local_native_path(local_root, namespace, relative_key);
            let expected_hash = expected.map(|version| version.content_sha256);
            return tokio::task::spawn_blocking(move || {
                local_compare_and_swap(&path, expected_hash, &bytes)
            })
            .await
            .map_err(|error| StatsError::Internal(format!("local native CAS task: {error}")))?;
        }

        let path = self.native_path(namespace, relative_key);
        let mode = match expected {
            None => PutMode::Create,
            Some(version) => PutMode::Update(UpdateVersion {
                e_tag: version.e_tag.clone(),
                version: version.provider_version.clone(),
            }),
        };
        let content_sha256 = Sha256::digest(&bytes).into();
        match self
            .store
            .put_opts(
                &path,
                bytes.into(),
                PutOptions {
                    mode,
                    ..Default::default()
                },
            )
            .await
        {
            Ok(result) => Ok(RemoteVersion {
                e_tag: result.e_tag,
                provider_version: result.version,
                content_sha256,
            }),
            Err(object_store::Error::AlreadyExists { .. })
            | Err(object_store::Error::Precondition { .. }) => Err(StatsError::SchemaConflict(
                format!("native pointer {path} changed concurrently"),
            )),
            Err(error) => Err(StatsError::Internal(format!(
                "update native pointer {path}: {error}"
            ))),
        }
    }

    pub async fn list_native_keys(
        &self,
        namespace: &str,
        relative_prefix: &str,
    ) -> Result<Vec<String>, StatsError> {
        let prefix = self.native_prefix(namespace, relative_prefix);
        let namespace_prefix = self.native_prefix(namespace, "");
        let mut stream = self.store.list(Some(&prefix));
        let mut keys = Vec::new();
        while let Some(result) = stream.next().await {
            let meta = result.map_err(|error| {
                StatsError::Internal(format!("list native objects {prefix}: {error}"))
            })?;
            let Some(parts) = meta.location.prefix_match(&namespace_prefix) else {
                continue;
            };
            keys.push(
                parts
                    .map(|part| part.as_ref().to_string())
                    .collect::<Vec<_>>()
                    .join("/"),
            );
        }
        Ok(keys)
    }

    pub async fn list_native_objects(
        &self,
        namespace: &str,
        relative_prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StatsError> {
        let prefix = self.native_prefix(namespace, relative_prefix);
        let namespace_prefix = self.native_prefix(namespace, "");
        let mut stream = self.store.list(Some(&prefix));
        let mut objects = Vec::new();
        while let Some(result) = stream.next().await {
            let meta = result.map_err(|error| {
                StatsError::Internal(format!("list native objects {prefix}: {error}"))
            })?;
            let Some(parts) = meta.location.prefix_match(&namespace_prefix) else {
                continue;
            };
            let key = parts
                .map(|part| part.as_ref().to_string())
                .collect::<Vec<_>>()
                .join("/");
            objects.push((key, meta));
        }
        Ok(objects)
    }

    pub async fn delete_native(
        &self,
        namespace: &str,
        relative_key: &str,
    ) -> Result<(), StatsError> {
        let path = self.native_path(namespace, relative_key);
        self.store
            .delete(&path)
            .await
            .map_err(|error| StatsError::Internal(format!("delete native object {path}: {error}")))
    }

    /// Upload `local_path` to `{namespace}/{relative_key}`. Returns `true` on
    /// success; the next sync retries on failure. The byte read + put run as
    /// async object_store calls (no spawn_blocking).
    pub async fn upload(
        &self,
        namespace: &str,
        relative_key: &str,
        local_path: &std::path::Path,
    ) -> bool {
        let bytes = match tokio::fs::read(local_path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(path = %local_path.display(), error = %e, "remote upload: read local failed");
                return false;
            }
        };
        let remote = self.object_path(namespace, relative_key);
        match self
            .store
            .put(&remote, bytes::Bytes::from(bytes).into())
            .await
        {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(remote = %remote, error = %e, "remote upload failed");
                false
            }
        }
    }

    /// Copy an object between two keys in the same namespace.
    pub async fn copy(
        &self,
        namespace: &str,
        from_key: &str,
        to_key: &str,
    ) -> Result<(), StatsError> {
        let from = self.object_path(namespace, from_key);
        let to = self.object_path(namespace, to_key);
        self.store.copy(&from, &to).await.map_err(|error| {
            StatsError::Internal(format!("remote copy {from} -> {to} failed: {error}"))
        })
    }

    async fn list_objects(&self, namespace: &str) -> Result<Vec<(String, ObjectMeta)>, StatsError> {
        let prefix = self.namespace_prefix(namespace);
        let mut stream = self.store.list(Some(&prefix));
        let mut objects = Vec::new();
        while let Some(item) = stream.next().await {
            let meta = item.map_err(|error| {
                StatsError::Internal(format!("remote list {namespace:?}: {error}"))
            })?;
            let Some(parts) = meta.location.prefix_match(&prefix) else {
                continue;
            };
            let key = parts
                .map(|part| part.as_ref().to_string())
                .collect::<Vec<_>>()
                .join("/");
            objects.push((key, meta));
        }
        Ok(objects)
    }

    /// List the relative keys of every object under `{namespace}/`.
    pub async fn list_keys(&self, namespace: &str) -> Result<Vec<String>, StatsError> {
        Ok(self
            .list_objects(namespace)
            .await?
            .into_iter()
            .map(|(key, _meta)| key)
            .collect())
    }

    /// List `(relative_key, byte_size)` for every parquet object under
    /// `{namespace}/` whose filename parses as a segment filename. Used by boot
    /// reconcile to enumerate adoption candidates.
    pub async fn list_segment_objects(
        &self,
        namespace: &str,
    ) -> Result<Vec<(String, u64)>, StatsError> {
        Ok(self
            .list_objects(namespace)
            .await?
            .into_iter()
            .filter_map(|(key, meta)| {
                let filename = meta.location.filename()?;
                crate::store::types::parse_seg_filename(filename)?;
                Some((key, meta.size))
            })
            .collect())
    }

    /// Delete `{namespace}/{relative_key}` from the remote store. Best-effort; logs
    /// and swallows on error (warn-and-continue).
    pub async fn delete(&self, namespace: &str, relative_key: &str) {
        let remote = self.object_path(namespace, relative_key);
        if let Err(e) = self.store.delete(&remote).await {
            tracing::warn!(remote = %remote, error = %e, "remote delete failed");
        }
    }

    /// Async footer read of `{namespace}/{relative_key}`, including actual seq
    /// bounds and hidden partition metadata.
    /// Returns `None` on an unreadable footer.
    ///
    /// `file_size` is the object size already known from `list_segment_objects`,
    /// passed in so this is a single ranged GET of the file tail with NO preceding
    /// `head` round-trip — halving the cross-region RPCs per segment on reconcile.
    pub async fn read_footer(
        &self,
        namespace: &str,
        relative_key: &str,
        file_size: u64,
        key_column: Option<&str>,
    ) -> Option<crate::store::segment::SegmentMetadata> {
        use parquet::arrow::async_reader::ParquetObjectReader;
        use parquet::file::metadata::ParquetMetaDataReader;

        let filename = std::path::Path::new(relative_key).file_name()?.to_str()?;
        let (level, filename_min_seq) = crate::store::types::parse_seg_filename(filename)?;

        let remote = self.object_path(namespace, relative_key);
        let mut reader =
            ParquetObjectReader::new(Arc::clone(&self.store), remote).with_file_size(file_size);
        let md = ParquetMetaDataReader::new()
            .with_prefetch_hint(Some(64 * 1024))
            .load_via_suffix_and_finish(&mut reader)
            .await
            .ok()?;
        crate::store::segment::segment_metadata_from_parquet(
            &md,
            level,
            filename_min_seq,
            key_column,
        )
    }
}

fn local_native_path(root: &Path, namespace: &str, relative_key: &str) -> PathBuf {
    let mut path = root.join("_native").join("namespaces").join(namespace);
    for part in relative_key.split('/').filter(|part| !part.is_empty()) {
        path.push(part);
    }
    path
}

fn local_compare_and_swap(
    path: &Path,
    expected_hash: Option<[u8; 32]>,
    bytes: &[u8],
) -> Result<RemoteVersion, StatsError> {
    let parent = path.parent().ok_or_else(|| {
        StatsError::Internal(format!("native pointer {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        StatsError::Internal(format!(
            "create native pointer parent {}: {error}",
            parent.display()
        ))
    })?;
    let lock_path = path.with_extension("lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|error| {
            StatsError::Internal(format!(
                "open native pointer lock {}: {error}",
                lock_path.display()
            ))
        })?;
    // SAFETY: lock owns this file descriptor until the function returns.
    let lock_result = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) };
    if lock_result != 0 {
        return Err(StatsError::Internal(format!(
            "lock native pointer {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        )));
    }

    let current = match std::fs::read(path) {
        Ok(current) => Some(current),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(StatsError::Internal(format!(
                "read native pointer {}: {error}",
                path.display()
            )))
        }
    };
    let current_hash = current.as_deref().map(|value| Sha256::digest(value).into());
    if current_hash != expected_hash {
        return Err(StatsError::SchemaConflict(format!(
            "native pointer {} changed concurrently",
            path.display()
        )));
    }

    let suffix = format!("tmp-{}-{}", std::process::id(), monotonic_nonce());
    let staging = path.with_extension(suffix);
    let mut staging_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staging)
        .map_err(|error| {
            StatsError::Internal(format!(
                "create native pointer staging {}: {error}",
                staging.display()
            ))
        })?;
    staging_file.write_all(bytes).map_err(|error| {
        StatsError::Internal(format!(
            "write native pointer staging {}: {error}",
            staging.display()
        ))
    })?;
    staging_file.sync_all().map_err(|error| {
        StatsError::Internal(format!(
            "fsync native pointer staging {}: {error}",
            staging.display()
        ))
    })?;
    std::fs::rename(&staging, path).map_err(|error| {
        StatsError::Internal(format!(
            "publish native pointer {} -> {}: {error}",
            staging.display(),
            path.display()
        ))
    })?;
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            StatsError::Internal(format!(
                "fsync native pointer parent {}: {error}",
                parent.display()
            ))
        })?;
    Ok(RemoteVersion {
        e_tag: None,
        provider_version: None,
        content_sha256: Sha256::digest(bytes).into(),
    })
}

fn monotonic_nonce() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_remote_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn empty_remote_dir_disables_sync() {
        assert!(build_remote_store("").unwrap().is_none());
    }

    #[test]
    fn gs_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here).
        let store = build_remote_store("gs://my-bucket/logs/sub")
            .unwrap()
            .unwrap();
        assert_eq!(store.prefix, "logs/sub");
        let p = store.object_path("ns.a", "seg_L1_0001.parquet");
        assert_eq!(p.to_string(), "logs/sub/ns.a/seg_L1_0001.parquet");
    }

    #[test]
    fn s3_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here). No
        // AWS_ENDPOINT_URL is set, so the builder keeps its default endpoint.
        let store = build_remote_store("s3://my-bucket/finelog/cw-us-east-02a")
            .unwrap()
            .unwrap();
        assert_eq!(store.prefix, "finelog/cw-us-east-02a");
        let p = store.object_path("iris.worker", "seg_L1_0001.parquet");
        assert_eq!(
            p.to_string(),
            "finelog/cw-us-east-02a/iris.worker/seg_L1_0001.parquet"
        );
    }

    #[tokio::test]
    async fn local_remote_upload_list_delete_round_trip() {
        let remote_dir = tempdir("rt");
        let local_dir = tempdir("local");
        let store = build_remote_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();

        let local_file = local_dir.join("seg_L1_0000000000000000001.parquet");
        std::fs::write(&local_file, b"hello-parquet").unwrap();
        assert!(
            store
                .upload(
                    "ns.a",
                    "run_id/07/seg_L1_0000000000000000001.parquet",
                    &local_file
                )
                .await
        );

        let on_disk = remote_dir
            .join("ns.a")
            .join("run_id/07")
            .join(local_file.file_name().unwrap());
        assert!(on_disk.exists());

        store
            .copy(
                "ns.a",
                "run_id/07/seg_L1_0000000000000000001.parquet",
                "run_id/08/seg_L1_0000000000000000001.parquet",
            )
            .await
            .unwrap();
        let copied = remote_dir
            .join("ns.a")
            .join("run_id/08")
            .join(local_file.file_name().unwrap());
        assert_eq!(std::fs::read(&copied).unwrap(), b"hello-parquet");

        let mut names = store.list_keys("ns.a").await.unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                "run_id/07/seg_L1_0000000000000000001.parquet".to_string(),
                "run_id/08/seg_L1_0000000000000000001.parquet".to_string(),
            ]
        );

        store
            .delete("ns.a", "run_id/07/seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!on_disk.exists());
        store
            .delete("ns.a", "run_id/08/seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!copied.exists());
        assert!(store.list_keys("ns.a").await.unwrap().is_empty());

        std::fs::remove_dir_all(&remote_dir).ok();
        std::fs::remove_dir_all(&local_dir).ok();
    }
}
