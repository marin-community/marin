//! Private provider configuration and physical-path operations shared by the
//! canonical and legacy layouts.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;
use object_store::{
    ObjectStore as BackendObjectStore, ObjectStoreExt, PutMode, PutOptions, UpdateVersion,
};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::store::object_store::{ObjectId, ObjectVersion, StoredObject};

use super::local_file::compare_and_swap as local_compare_and_swap;

const GCS_SCHEME: &str = "gs://";
const S3_SCHEME: &str = "s3://";

#[derive(Clone)]
pub(super) struct Provider {
    backend: Arc<dyn BackendObjectStore>,
    prefix: String,
    local_root: Option<PathBuf>,
    /// `gs://bucket` / `s3://bucket` for a bucket-backed provider; `None` for
    /// a local directory, whose objects scan through the default file store.
    base_url: Option<String>,
}

impl Provider {
    pub(super) fn from_remote_log_dir(value: &str) -> Result<Option<Self>, StatsError> {
        if value.is_empty() {
            return Ok(None);
        }
        if let Some(rest) = value.strip_prefix(GCS_SCHEME) {
            let (bucket, prefix) = split_bucket_and_prefix(rest);
            let backend = object_store::gcp::GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket)
                .build()
                .map_err(|error| {
                    StatsError::Internal(format!("build gcs store {bucket:?}: {error}"))
                })?;
            return Ok(Some(Self {
                backend: Arc::new(backend),
                prefix: prefix.to_string(),
                local_root: None,
                base_url: Some(format!("{GCS_SCHEME}{bucket}")),
            }));
        }
        if let Some(rest) = value.strip_prefix(S3_SCHEME) {
            let (bucket, prefix) = split_bucket_and_prefix(rest);
            let backend = object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()
                .map_err(|error| {
                    StatsError::Internal(format!("build s3 store {bucket:?}: {error}"))
                })?;
            return Ok(Some(Self {
                backend: Arc::new(backend),
                prefix: prefix.to_string(),
                local_root: None,
                base_url: Some(format!("{S3_SCHEME}{bucket}")),
            }));
        }

        std::fs::create_dir_all(value)
            .map_err(|error| StatsError::Internal(format!("create remote dir {value}: {error}")))?;
        let backend = LocalFileSystem::new_with_prefix(value).map_err(|error| {
            StatsError::Internal(format!("local remote store {value}: {error}"))
        })?;
        Ok(Some(Self {
            backend: Arc::new(backend),
            prefix: String::new(),
            local_root: Some(PathBuf::from(value)),
            base_url: None,
        }))
    }

    pub(super) fn backend(&self) -> &Arc<dyn BackendObjectStore> {
        &self.backend
    }

    pub(super) fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    pub(super) fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.prefix
            .split('/')
            .filter(|component| !component.is_empty())
    }

    pub(super) fn local_root(&self) -> Option<&Path> {
        self.local_root.as_deref()
    }

    /// Provider-relative physical path for one validated logical object ID.
    pub(super) fn object_path(&self, id: &ObjectId) -> OsPath {
        OsPath::from_iter(
            self.prefix_parts()
                .chain(id.as_str().split('/').filter(|part| !part.is_empty())),
        )
    }

    /// URL or filesystem path used to scan the same physical object.
    pub(super) fn scan_url(&self, id: &ObjectId) -> String {
        let path = self.object_path(id);
        match self.base_url() {
            Some(base) => format!("{base}/{path}"),
            None => self
                .local_root()
                .expect("a provider without a base URL has a local root")
                .join(path.as_ref())
                .to_string_lossy()
                .into_owned(),
        }
    }

    /// Read the object at `path`, or `None` when it does not exist.
    /// `description` names the object in error messages.
    pub(super) async fn get_path(
        &self,
        path: OsPath,
        description: &str,
    ) -> Result<Option<StoredObject>, StatsError> {
        let result = match self.backend.get(&path).await {
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

    /// Swap the pointer at `path` from `expected` to `bytes`.
    ///
    /// `local_path` is the filesystem path backing `path` when the provider is a
    /// local directory, in which case the swap goes through the content-hash
    /// comparison in [`local_compare_and_swap`].
    ///
    /// A precondition failure is the one outcome the backend states
    /// definitively: the swap did not apply, reported as `SchemaConflict`. Every
    /// other failure leaves the pointer's state unknown, so it is reported as
    /// `AmbiguousCommit` and the caller resolves it by re-reading the pointer.
    pub(super) async fn compare_and_swap_path(
        &self,
        path: OsPath,
        local_path: Option<PathBuf>,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
        description: &str,
    ) -> Result<ObjectVersion, StatsError> {
        if let Some(local_path) = local_path {
            let expected_hash = expected.map(|version| version.content_sha256);
            return tokio::task::spawn_blocking(move || {
                local_compare_and_swap(&local_path, expected_hash, &bytes)
            })
            .await
            .map_err(|error| StatsError::Internal(format!("{description} CAS task: {error}")))?;
        }
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
            .backend
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
                format!("{description} pointer {path} changed concurrently"),
            )),
            Err(error) => Err(StatsError::AmbiguousCommit(format!(
                "update {description} pointer {path}: {error}"
            ))),
        }
    }
}

/// Whether `value` names a remote bucket URL rather than a local directory.
pub fn is_remote_object_store(value: &str) -> bool {
    let value = value.trim();
    value.starts_with(GCS_SCHEME) || value.starts_with(S3_SCHEME)
}

fn split_bucket_and_prefix(value: &str) -> (&str, &str) {
    match value.split_once('/') {
        Some((bucket, prefix)) => (bucket, prefix.trim_matches('/')),
        None => (value, ""),
    }
}
