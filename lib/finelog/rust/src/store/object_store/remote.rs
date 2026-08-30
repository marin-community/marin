//! Canonical-layout implementation of the Finelog object-store contract.
//!
//! `build_remote_object_store` dispatches on the configured `remote_log_dir`:
//! `gs://bucket/prefix` -> `GoogleCloudStorageBuilder` (GCP prod);
//! `s3://bucket/prefix` -> `AmazonS3Builder` for any S3-compatible store
//! (Cloudflare R2 / CoreWeave Object Storage on CoreWeave clusters); any other
//! non-empty value -> `LocalFileSystem` rooted at that directory (tests pass a
//! plain tmp path). An empty `remote_log_dir` disables sync (returns `None`).
//!
//! Canonical objects use root-relative [`ObjectId`] values under
//! `_finelog/tables`. Historical layout translation lives in `legacy`; both
//! implementations satisfy [`ObjectStore`].
//! object_store 0.13 moved `put`/`get`/`head`/`delete` onto the `ObjectStoreExt`
//! blanket trait, which must be in scope.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use object_store::path::Path as OsPath;
use object_store::{ObjectStoreExt, PutMode, PutOptions};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::store::object_store::{
    ObjectId, ObjectMetadata, ObjectPrefix, ObjectStore, ObjectVersion, StoredObject,
    FINELOG_ROOT_COMPONENT, TABLES_COMPONENT,
};

use super::provider::{is_remote_url, Provider};

/// A configured remote object store plus the bucket-relative prefix the store
/// is rooted under (empty for a `LocalFileSystem` rooted at the remote dir).
#[derive(Clone)]
pub struct RemoteObjectStore {
    provider: Provider,
}

/// Whether `remote_log_dir` names an object store rather than a local directory.
pub fn is_object_store(remote_log_dir: &str) -> bool {
    is_remote_url(remote_log_dir)
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
pub fn build_remote_object_store(
    remote_log_dir: &str,
) -> Result<Option<RemoteObjectStore>, StatsError> {
    Ok(Provider::from_remote_log_dir(remote_log_dir)?
        .map(|provider| RemoteObjectStore { provider }))
}

impl RemoteObjectStore {
    pub(super) fn provider(&self) -> Provider {
        self.provider.clone()
    }

    /// The backend store and base URL a query engine registers to scan this
    /// provider's objects directly. `None` for a local-directory provider,
    /// whose objects scan through the engine's default file store.
    pub fn scan_registration(&self) -> Option<(String, Arc<dyn object_store::ObjectStore>)> {
        self.provider
            .base_url()
            .map(|base| (base.to_string(), Arc::clone(self.provider.backend())))
    }

    /// The URL a registered scan reads `id` from.
    pub(super) fn scan_url(&self, id: &ObjectId) -> String {
        let key: Vec<&str> = self
            .prefix_parts()
            .chain(id.as_str().split('/').filter(|part| !part.is_empty()))
            .collect();
        match self.provider.base_url() {
            Some(base) => format!("{base}/{}", key.join("/")),
            None => {
                let mut path = self
                    .provider
                    .local_root()
                    .expect("a provider without a base URL has a local root")
                    .to_path_buf();
                for part in key {
                    path.push(part);
                }
                path.to_string_lossy().into_owned()
            }
        }
    }

    /// Split the configured prefix on `/` into individual path components.
    /// `OsPath::from_iter` escapes `/` *within* a single part, so a multi-segment
    /// prefix like `logs/sub` must be pushed component-by-component.
    fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.provider.prefix_parts()
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
            .chain(prefix.as_str().split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    pub async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        let root = OsPath::from_iter(
            self.prefix_parts()
                .chain([FINELOG_ROOT_COMPONENT, TABLES_COMPONENT]),
        );
        let mut stream = self.provider.backend().list(Some(&root));
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

    async fn read_object(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.provider
            .get_path(self.canonical_path(id), "object")
            .await
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
            .provider
            .backend()
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
        let local_path = self
            .provider
            .local_root()
            .map(|root| local_object_path(root, id));
        self.provider
            .compare_and_swap_path(
                self.canonical_path(id),
                local_path,
                expected,
                bytes,
                "object",
            )
            .await
    }

    async fn list_objects(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        let path = self.canonical_prefix(prefix);
        let mut stream = self.provider.backend().list(Some(&path));
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
                modified_at_ms: meta.last_modified.timestamp_millis(),
            });
        }
        Ok(objects)
    }

    async fn delete_object(&self, id: &ObjectId) -> Result<(), StatsError> {
        let path = self.canonical_path(id);
        match self.provider.backend().delete(&path).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(StatsError::Internal(format!(
                "delete object {path}: {error}"
            ))),
        }
    }
}

#[async_trait]
impl ObjectStore for RemoteObjectStore {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        self.write_immutable(id, bytes).await
    }

    fn remote_scan_url(&self, id: &ObjectId) -> Option<String> {
        Some(self.scan_url(id))
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

    async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        RemoteObjectStore::list_tables(self).await
    }
}

fn local_object_path(root: &Path, id: &ObjectId) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in id.as_str().split('/').filter(|part| !part.is_empty()) {
        path.push(part);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_dir;

    #[test]
    fn empty_remote_dir_disables_sync() {
        assert!(build_remote_object_store("").unwrap().is_none());
    }

    #[test]
    fn gs_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here).
        let store = build_remote_object_store("gs://my-bucket/logs/sub")
            .unwrap()
            .unwrap();
        let id = ObjectId::table("ns.a", "seg_L1_0001.parquet").unwrap();
        assert_eq!(
            store.canonical_path(&id).to_string(),
            "logs/sub/_finelog/tables/ns.a/seg_L1_0001.parquet"
        );
    }

    #[test]
    fn s3_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here). No
        // AWS_ENDPOINT_URL is set, so the builder keeps its default endpoint.
        let store = build_remote_object_store("s3://my-bucket/finelog/cw-us-east-02a")
            .unwrap()
            .unwrap();
        let id = ObjectId::table("iris.worker", "seg_L1_0001.parquet").unwrap();
        let p = store.canonical_path(&id);
        assert_eq!(
            p.to_string(),
            "finelog/cw-us-east-02a/_finelog/tables/iris.worker/seg_L1_0001.parquet"
        );
    }

    #[tokio::test]
    async fn canonical_objects_round_trip_by_typed_id() {
        let remote_dir = unique_dir("remote_objects");
        let store = build_remote_object_store(remote_dir.to_str().unwrap())
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

    #[tokio::test]
    async fn streamed_write_uses_the_same_object_contract() {
        use futures::stream;

        let remote_dir = unique_dir("remote_stream");
        let store = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let id = ObjectId::table("iris.worker", "objects/stream.parquet").unwrap();
        let chunks = stream::iter([
            Ok(bytes::Bytes::from_static(b"par")),
            Ok(bytes::Bytes::from_static(b"quet")),
        ])
        .boxed();

        let version = store.write_stream(&id, chunks).await.unwrap();

        assert_eq!(version.byte_size, 7);
        assert_eq!(
            store.read(&id).await.unwrap().unwrap().bytes,
            b"parquet"[..]
        );
        std::fs::remove_dir_all(remote_dir).ok();
    }
}
