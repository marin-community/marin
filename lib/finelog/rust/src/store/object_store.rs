//! Canonical object persistence and the temporary legacy-archive backend seam.
//!
//! `build_object_storage` dispatches on the configured `remote_log_dir`:
//! `gs://bucket/prefix` -> `GoogleCloudStorageBuilder` (GCP prod);
//! `s3://bucket/prefix` -> `AmazonS3Builder` for any S3-compatible store
//! (Cloudflare R2 / CoreWeave Object Storage on CoreWeave clusters); any other
//! non-empty value -> `LocalFileSystem` rooted at that directory (tests pass a
//! plain tmp path). An empty `remote_log_dir` disables sync (returns `None`).
//!
//! Canonical objects use root-relative [`ObjectId`] values under
//! `_finelog/tables`. Legacy archive helpers remain on the concrete adapter only
//! until table migration is complete; query and published-catalog code use the
//! [`ObjectStore`] trait.
//! object_store 0.13 moved `put`/`get`/`head`/`delete` onto the `ObjectStoreExt`
//! blanket trait, which must be in scope.

use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;
use object_store::{
    ObjectMeta, ObjectStore as BackendObjectStore, ObjectStoreExt, PutMode, PutOptions,
    UpdateVersion,
};
use sha2::{Digest, Sha256};
use url::Url;

use crate::errors::StatsError;

/// A configured remote object store plus the bucket-relative prefix the store
/// is rooted under (empty for a `LocalFileSystem` rooted at the remote dir).
#[derive(Clone)]
pub struct ObjectStorage {
    store: Arc<dyn BackendObjectStore>,
    prefix: String,
    local_root: Option<PathBuf>,
    root_url: Url,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredObject {
    pub bytes: bytes::Bytes,
    pub version: ObjectVersion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectVersion {
    pub e_tag: Option<String>,
    pub provider_version: Option<String>,
    pub content_sha256: [u8; 32],
    pub byte_size: u64,
}

/// Complete, validated key relative to the configured object-store root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(String);

impl ObjectId {
    pub fn parse(value: &str) -> Result<Self, StatsError> {
        let value = validate_relative_key(value)?;
        let mut components = value.split('/');
        if components.next() != Some(FINELOG_ROOT_COMPONENT)
            || components.next() != Some(TABLES_COMPONENT)
        {
            return Err(StatsError::Internal(format!(
                "object ID {value:?} is outside {FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}"
            )));
        }
        let table = components
            .next()
            .ok_or_else(|| StatsError::Internal(format!("object ID {value:?} has no table")))?;
        validate_component(table, "table")?;
        if components.next().is_none() {
            return Err(StatsError::Internal(format!(
                "object ID {value:?} has no table-relative key"
            )));
        }
        Ok(Self(value.to_string()))
    }

    pub fn table(table: &str, relative_key: &str) -> Result<Self, StatsError> {
        validate_component(table, "table")?;
        let relative_key = validate_relative_key(relative_key)?;
        Ok(Self(format!(
            "{FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}/{table}/{relative_key}"
        )))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn table_relative<'a>(&'a self, table: &str) -> Option<&'a str> {
        self.0.strip_prefix(&format!(
            "{FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}/{table}/"
        ))
    }
}

/// Validated key prefix used only for listings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectPrefix(String);

impl ObjectPrefix {
    pub fn table(table: &str, relative_prefix: &str) -> Result<Self, StatsError> {
        validate_component(table, "table")?;
        let suffix = validate_relative_prefix(relative_prefix)?;
        let root = format!("{FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}/{table}");
        Ok(Self(if suffix.is_empty() {
            root
        } else {
            format!("{root}/{suffix}")
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectLocation(Url);

impl ObjectLocation {
    pub fn as_url(&self) -> &Url {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub id: ObjectId,
    pub e_tag: Option<String>,
    pub provider_version: Option<String>,
    pub byte_size: u64,
    pub modified_at_ms: i64,
}

/// Canonical object persistence independent of provider-specific locations.
///
/// Catalog code carries [`ObjectId`] values through this boundary. Only the
/// implementation resolves those IDs to GCS, S3, or local object locations.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    fn location_for(&self, id: &ObjectId) -> Result<ObjectLocation, StatsError>;

    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError>;
    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError>;
    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError>;
    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError>;
    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError>;
}

const GCS_SCHEME: &str = "gs://";
const S3_SCHEME: &str = "s3://";
pub const FINELOG_ROOT_COMPONENT: &str = "_finelog";
pub const TABLES_COMPONENT: &str = "tables";

fn validate_component(value: &str, description: &str) -> Result<(), StatsError> {
    if value.is_empty()
        || matches!(value, "." | "..")
        || value.contains('/')
        || value.contains('\\')
    {
        return Err(StatsError::InvalidNamespace(format!(
            "invalid object {description} {value:?}"
        )));
    }
    Ok(())
}

fn validate_relative_key(value: &str) -> Result<&str, StatsError> {
    if value.is_empty() {
        return Err(StatsError::Internal(
            "object key must not be empty".to_string(),
        ));
    }
    validate_relative_prefix(value)?;
    Ok(value.trim_matches('/'))
}

fn validate_relative_prefix(value: &str) -> Result<&str, StatsError> {
    if value != value.trim_matches('/')
        || (!value.is_empty() && value.split('/').any(str::is_empty))
    {
        return Err(StatsError::Internal(format!(
            "object key {value:?} is not canonical"
        )));
    }
    for component in value.split('/').filter(|component| !component.is_empty()) {
        validate_component(component, "key component")?;
    }
    Ok(value)
}

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
pub fn build_object_storage(remote_log_dir: &str) -> Result<Option<ObjectStorage>, StatsError> {
    let dir = remote_log_dir;
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
        return Ok(Some(ObjectStorage {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
            local_root: None,
            root_url: object_store_root_url(dir)?,
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
        return Ok(Some(ObjectStorage {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
            local_root: None,
            root_url: object_store_root_url(dir)?,
        }));
    }
    // Local filesystem remote (tests). Root the store at the remote dir so
    // object paths are `{namespace}/{relative segment key}`.
    std::fs::create_dir_all(dir)
        .map_err(|e| StatsError::Internal(format!("create remote dir {dir}: {e}")))?;
    let store = LocalFileSystem::new_with_prefix(dir)
        .map_err(|e| StatsError::Internal(format!("local remote store {dir}: {e}")))?;
    Ok(Some(ObjectStorage {
        store: Arc::new(store),
        prefix: String::new(),
        local_root: Some(PathBuf::from(dir)),
        root_url: Url::from_directory_path(
            std::fs::canonicalize(dir)
                .map_err(|error| StatsError::Internal(format!("canonicalize {dir}: {error}")))?,
        )
        .map_err(|_| StatsError::Internal(format!("local object root {dir:?} is not absolute")))?,
    }))
}

fn object_store_root_url(value: &str) -> Result<Url, StatsError> {
    let mut url = Url::parse(value).map_err(|error| {
        StatsError::Internal(format!("invalid object-store root {value:?}: {error}"))
    })?;
    if !url.path().ends_with('/') {
        let directory_path = format!("{}/", url.path());
        url.set_path(&directory_path);
    }
    Ok(url)
}

impl ObjectStorage {
    /// Split the configured prefix on `/` into individual path components.
    /// `OsPath::from_iter` escapes `/` *within* a single part, so a multi-segment
    /// prefix like `logs/sub` must be pushed component-by-component.
    fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.prefix.split('/').filter(|s| !s.is_empty())
    }

    /// Temporary archive path for `{prefix}/{namespace}/{relative segment key}`.
    fn legacy_path(&self, namespace: &str, relative_key: &str) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain([namespace])
            .chain(relative_key.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    /// The directory prefix for one namespace, `{prefix}/{namespace}`.
    fn legacy_prefix(&self, namespace: &str) -> OsPath {
        let parts: Vec<&str> = self.prefix_parts().chain([namespace]).collect();
        OsPath::from_iter(parts)
    }

    fn canonical_path(&self, id: &ObjectId) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain(id.as_str().split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    fn canonical_prefix(&self, prefix: &ObjectPrefix) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain(prefix.0.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    pub async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        let root = OsPath::from_iter(
            self.prefix_parts()
                .chain([FINELOG_ROOT_COMPONENT, TABLES_COMPONENT]),
        );
        let mut stream = self.store.list(Some(&root));
        let mut namespaces = std::collections::BTreeSet::new();
        while let Some(result) = stream.next().await {
            let meta = result.map_err(|error| {
                StatsError::Internal(format!("list object tables {root}: {error}"))
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

    /// Read one object from the legacy namespace prefix.
    pub(super) async fn read_legacy(
        &self,
        namespace: &str,
        relative_key: &str,
    ) -> Result<Option<StoredObject>, StatsError> {
        self.get_path(
            self.legacy_path(namespace, relative_key),
            "legacy archive object",
        )
        .await
    }

    async fn read_object(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.get_path(self.canonical_path(id), "object").await
    }

    async fn get_path(
        &self,
        path: OsPath,
        description: &str,
    ) -> Result<Option<StoredObject>, StatsError> {
        let result = match self.store.get(&path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => {
                return Err(StatsError::Internal(format!(
                    "read {description} {path}: {error}"
                )))
            }
        };
        let e_tag = result.meta.e_tag.clone();
        let provider_version = result.meta.version.clone();
        let bytes = result.bytes().await.map_err(|error| {
            StatsError::Internal(format!("read {description} body {path}: {error}"))
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

    /// Create an immutable object, accepting an identical retry.
    async fn write_immutable(
        &self,
        id: &ObjectId,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        let path = self.canonical_path(id);
        let content_sha256 = Sha256::digest(&bytes).into();
        let byte_size = bytes.len() as u64;
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
            Ok(result) => Ok(ObjectVersion {
                e_tag: result.e_tag,
                provider_version: result.version,
                content_sha256,
                byte_size,
            }),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let existing = self.read_object(id).await?.ok_or_else(|| {
                    StatsError::Internal(format!("object {path} disappeared after create conflict"))
                })?;
                if existing.bytes == bytes {
                    Ok(existing.version)
                } else {
                    Err(StatsError::SchemaConflict(format!(
                        "immutable object {path} already exists with different contents"
                    )))
                }
            }
            Err(error) => Err(StatsError::Internal(format!(
                "create object {path}: {error}"
            ))),
        }
    }

    async fn compare_and_swap_object(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        if let Some(local_root) = &self.local_root {
            let path = local_object_path(local_root, id);
            let expected_hash = expected.map(|version| version.content_sha256);
            return tokio::task::spawn_blocking(move || {
                local_compare_and_swap(&path, expected_hash, &bytes)
            })
            .await
            .map_err(|error| StatsError::Internal(format!("local object CAS task: {error}")))?;
        }

        let path = self.canonical_path(id);
        let mode = match expected {
            None => PutMode::Create,
            Some(version) => PutMode::Update(UpdateVersion {
                e_tag: version.e_tag.clone(),
                version: version.provider_version.clone(),
            }),
        };
        let content_sha256 = Sha256::digest(&bytes).into();
        let byte_size = bytes.len() as u64;
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
            Ok(result) => Ok(ObjectVersion {
                e_tag: result.e_tag,
                provider_version: result.version,
                content_sha256,
                byte_size,
            }),
            Err(object_store::Error::AlreadyExists { .. })
            | Err(object_store::Error::Precondition { .. }) => Err(StatsError::SchemaConflict(
                format!("object pointer {path} changed concurrently"),
            )),
            Err(error) => Err(StatsError::Internal(format!(
                "update object pointer {path}: {error}"
            ))),
        }
    }

    async fn list_objects(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        let path = self.canonical_prefix(prefix);
        let mut stream = self.store.list(Some(&path));
        let mut objects = Vec::new();
        while let Some(result) = stream.next().await {
            let meta = result
                .map_err(|error| StatsError::Internal(format!("list objects {path}: {error}")))?;
            let root = OsPath::from_iter(self.prefix_parts());
            let Some(parts) = meta.location.prefix_match(&root) else {
                continue;
            };
            let id = parts
                .map(|part| part.as_ref().to_string())
                .collect::<Vec<_>>()
                .join("/");
            objects.push(ObjectMetadata {
                id: ObjectId::parse(&id)?,
                e_tag: meta.e_tag.clone(),
                provider_version: meta.version.clone(),
                byte_size: meta.size,
                modified_at_ms: meta.last_modified.timestamp_millis(),
            });
        }
        Ok(objects)
    }

    async fn delete_object(&self, id: &ObjectId) -> Result<(), StatsError> {
        let path = self.canonical_path(id);
        match self.store.delete(&path).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(StatsError::Internal(format!(
                "delete object {path}: {error}"
            ))),
        }
    }

    /// Upload `local_path` to `{namespace}/{relative_key}`. Returns `true` on
    /// success; the next sync retries on failure. The byte read + put run as
    /// async object_store calls (no spawn_blocking).
    pub(super) async fn upload_legacy(
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
        let remote = self.legacy_path(namespace, relative_key);
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
    pub(super) async fn copy_legacy(
        &self,
        namespace: &str,
        from_key: &str,
        to_key: &str,
    ) -> Result<(), StatsError> {
        let from = self.legacy_path(namespace, from_key);
        let to = self.legacy_path(namespace, to_key);
        self.store.copy(&from, &to).await.map_err(|error| {
            StatsError::Internal(format!("remote copy {from} -> {to} failed: {error}"))
        })
    }

    async fn list_legacy_objects(
        &self,
        namespace: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StatsError> {
        let prefix = self.legacy_prefix(namespace);
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
    pub(super) async fn list_legacy_keys(
        &self,
        namespace: &str,
    ) -> Result<Vec<String>, StatsError> {
        Ok(self
            .list_legacy_objects(namespace)
            .await?
            .into_iter()
            .map(|(key, _meta)| key)
            .collect())
    }

    /// List `(relative_key, byte_size)` for every parquet object under
    /// `{namespace}/` whose filename parses as a segment filename. Used by boot
    /// reconcile to enumerate adoption candidates.
    pub(super) async fn list_legacy_segment_objects(
        &self,
        namespace: &str,
    ) -> Result<Vec<(String, u64)>, StatsError> {
        Ok(self
            .list_legacy_objects(namespace)
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
    pub(super) async fn delete_legacy(&self, namespace: &str, relative_key: &str) {
        let remote = self.legacy_path(namespace, relative_key);
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
    pub(super) async fn read_legacy_footer(
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

        let remote = self.legacy_path(namespace, relative_key);
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

#[async_trait]
impl ObjectStore for ObjectStorage {
    fn location_for(&self, id: &ObjectId) -> Result<ObjectLocation, StatsError> {
        let location = self.root_url.join(id.as_str()).map_err(|error| {
            StatsError::Internal(format!(
                "resolve object location for {:?}: {error}",
                id.as_str()
            ))
        })?;
        Ok(ObjectLocation(location))
    }

    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        self.write_immutable(id, bytes).await
    }

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.read_object(id).await
    }

    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        self.compare_and_swap_object(id, expected, bytes).await
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError> {
        self.delete_object(id).await
    }

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        self.list_objects(prefix).await
    }
}

fn local_object_path(root: &Path, id: &ObjectId) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in id.as_str().split('/').filter(|part| !part.is_empty()) {
        path.push(part);
    }
    path
}

pub(super) fn atomic_write_file(path: &Path, bytes: &[u8]) -> Result<(), StatsError> {
    let parent = path.parent().ok_or_else(|| {
        StatsError::Internal(format!("local object {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        StatsError::Internal(format!(
            "create local object parent {}: {error}",
            parent.display()
        ))
    })?;
    let staging = path.with_extension(format!("tmp-{}-{}", std::process::id(), monotonic_nonce()));
    let mut staging_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staging)
        .map_err(|error| {
            StatsError::Internal(format!(
                "create local object staging {}: {error}",
                staging.display()
            ))
        })?;
    staging_file.write_all(bytes).map_err(|error| {
        StatsError::Internal(format!(
            "write local object staging {}: {error}",
            staging.display()
        ))
    })?;
    staging_file.sync_all().map_err(|error| {
        StatsError::Internal(format!(
            "fsync local object staging {}: {error}",
            staging.display()
        ))
    })?;
    std::fs::rename(&staging, path).map_err(|error| {
        StatsError::Internal(format!(
            "publish local object {} -> {}: {error}",
            staging.display(),
            path.display()
        ))
    })?;
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            StatsError::Internal(format!(
                "fsync local object parent {}: {error}",
                parent.display()
            ))
        })
}

fn local_compare_and_swap(
    path: &Path,
    expected_hash: Option<[u8; 32]>,
    bytes: &[u8],
) -> Result<ObjectVersion, StatsError> {
    let parent = path.parent().ok_or_else(|| {
        StatsError::Internal(format!("object pointer {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        StatsError::Internal(format!(
            "create object pointer parent {}: {error}",
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
                "open object pointer lock {}: {error}",
                lock_path.display()
            ))
        })?;
    // SAFETY: lock owns this file descriptor until the function returns.
    let lock_result = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) };
    if lock_result != 0 {
        return Err(StatsError::Internal(format!(
            "lock object pointer {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        )));
    }

    let current = match std::fs::read(path) {
        Ok(current) => Some(current),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(StatsError::Internal(format!(
                "read object pointer {}: {error}",
                path.display()
            )))
        }
    };
    let current_hash = current.as_deref().map(|value| Sha256::digest(value).into());
    if current_hash != expected_hash {
        return Err(StatsError::SchemaConflict(format!(
            "object pointer {} changed concurrently",
            path.display()
        )));
    }

    atomic_write_file(path, bytes)?;
    Ok(ObjectVersion {
        e_tag: None,
        provider_version: None,
        content_sha256: Sha256::digest(bytes).into(),
        byte_size: bytes.len() as u64,
    })
}

fn monotonic_nonce() -> u64 {
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
        assert!(build_object_storage("").unwrap().is_none());
    }

    #[test]
    fn gs_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here).
        let store = build_object_storage("gs://my-bucket/logs/sub")
            .unwrap()
            .unwrap();
        assert_eq!(store.prefix, "logs/sub");
        let p = store.legacy_path("ns.a", "seg_L1_0001.parquet");
        assert_eq!(p.to_string(), "logs/sub/ns.a/seg_L1_0001.parquet");
    }

    #[test]
    fn s3_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here). No
        // AWS_ENDPOINT_URL is set, so the builder keeps its default endpoint.
        let store = build_object_storage("s3://my-bucket/finelog/cw-us-east-02a")
            .unwrap()
            .unwrap();
        assert_eq!(store.prefix, "finelog/cw-us-east-02a");
        let p = store.legacy_path("iris.worker", "seg_L1_0001.parquet");
        assert_eq!(
            p.to_string(),
            "finelog/cw-us-east-02a/iris.worker/seg_L1_0001.parquet"
        );
    }

    #[tokio::test]
    async fn local_remote_upload_list_delete_round_trip() {
        let remote_dir = tempdir("rt");
        let local_dir = tempdir("local");
        let store = build_object_storage(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();

        let local_file = local_dir.join("seg_L1_0000000000000000001.parquet");
        std::fs::write(&local_file, b"hello-parquet").unwrap();
        assert!(
            store
                .upload_legacy(
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
            .copy_legacy(
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

        let mut names = store.list_legacy_keys("ns.a").await.unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                "run_id/07/seg_L1_0000000000000000001.parquet".to_string(),
                "run_id/08/seg_L1_0000000000000000001.parquet".to_string(),
            ]
        );

        store
            .delete_legacy("ns.a", "run_id/07/seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!on_disk.exists());
        store
            .delete_legacy("ns.a", "run_id/08/seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!copied.exists());
        assert!(store.list_legacy_keys("ns.a").await.unwrap().is_empty());

        std::fs::remove_dir_all(&remote_dir).ok();
        std::fs::remove_dir_all(&local_dir).ok();
    }

    #[tokio::test]
    async fn canonical_objects_round_trip_by_typed_id() {
        let remote_dir = tempdir("objects");
        let store = build_object_storage(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let id = ObjectId::table("iris.worker", "objects/v1/l1/hash/segment.parquet").unwrap();

        let version = store
            .write(&id, bytes::Bytes::from_static(b"parquet"))
            .await
            .unwrap();
        assert_eq!(version.byte_size, 7);
        assert_eq!(
            store.read(&id).await.unwrap().unwrap().bytes,
            b"parquet"[..]
        );
        assert_eq!(
            store
                .location_for(&id)
                .unwrap()
                .as_url()
                .to_file_path()
                .unwrap(),
            remote_dir.join(id.as_str())
        );
        let listed = store
            .list(&ObjectPrefix::table("iris.worker", "objects/v1").unwrap())
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);

        store.delete(&id).await.unwrap();
        assert!(store.read(&id).await.unwrap().is_none());
        std::fs::remove_dir_all(&remote_dir).ok();
    }
}
