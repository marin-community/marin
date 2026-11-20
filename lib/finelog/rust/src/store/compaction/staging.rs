//! Private scratch space for one compaction or migration rewrite.
//!
//! The executor writes ordinary local Parquet plus its derived artifacts here;
//! each is uploaded as an immutable object and the directory is removed. Nothing
//! in the query view ever points at a staged file.

use std::path::{Path, PathBuf};

use crate::errors::StatsError;

/// Table-relative directory outputs are staged in before upload.
const STAGING_DIR: &str = "_compaction";

/// A directory that removes itself when the work that created it is done.
pub struct StagingDir {
    path: PathBuf,
}

impl StagingDir {
    pub fn create(table_dir: &Path) -> Result<Self, StatsError> {
        let path = table_dir.join(format!("{STAGING_DIR}/{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).map_err(|error| {
            StatsError::Internal(format!(
                "create compaction staging directory {}: {error}",
                path.display()
            ))
        })?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.path) {
            tracing::warn!(path = %self.path.display(), %error, "failed to remove compaction staging directory");
        }
    }
}
