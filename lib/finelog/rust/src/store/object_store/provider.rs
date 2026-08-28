//! Private provider configuration shared by canonical and legacy layouts.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore as BackendObjectStore;

use crate::errors::StatsError;

const GCS_SCHEME: &str = "gs://";
const S3_SCHEME: &str = "s3://";

#[derive(Clone)]
pub(super) struct Provider {
    backend: Arc<dyn BackendObjectStore>,
    prefix: String,
    local_root: Option<PathBuf>,
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
        }))
    }

    pub(super) fn backend(&self) -> &Arc<dyn BackendObjectStore> {
        &self.backend
    }

    pub(super) fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.prefix
            .split('/')
            .filter(|component| !component.is_empty())
    }

    pub(super) fn local_root(&self) -> Option<&Path> {
        self.local_root.as_deref()
    }
}

pub(super) fn is_remote_url(value: &str) -> bool {
    let value = value.trim();
    value.starts_with(GCS_SCHEME) || value.starts_with(S3_SCHEME)
}

fn split_bucket_and_prefix(value: &str) -> (&str, &str) {
    match value.split_once('/') {
        Some((bucket, prefix)) => (bucket, prefix.trim_matches('/')),
        None => (value, ""),
    }
}
