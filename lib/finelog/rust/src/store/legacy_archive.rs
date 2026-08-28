//! Transitional access to the pre-object-catalog segment archive.
//!
//! Legacy segments live directly under `{table}/...`. New objects use typed
//! [`ObjectId`](crate::store::object_store::ObjectId) values under
//! `_finelog/tables/{table}/...`; keeping this adapter separate prevents new
//! query and catalog code from depending on the historical path layout.

use std::path::Path;

use crate::errors::StatsError;
use crate::store::object_store::{ObjectStorage, StoredObject};
use crate::store::segment::SegmentMetadata;

#[derive(Clone)]
pub struct LegacyArchive {
    storage: ObjectStorage,
}

impl LegacyArchive {
    pub fn new(storage: ObjectStorage) -> Self {
        Self { storage }
    }

    pub async fn read(
        &self,
        table: &str,
        relative_key: &str,
    ) -> Result<Option<StoredObject>, StatsError> {
        self.storage.read_legacy(table, relative_key).await
    }

    /// Upload one legacy segment.
    ///
    /// Returns `true` on success. A `false` result has already been logged and
    /// leaves the segment eligible for the next maintenance retry.
    pub async fn upload(&self, table: &str, relative_key: &str, local_path: &Path) -> bool {
        self.storage
            .upload_legacy(table, relative_key, local_path)
            .await
    }

    pub async fn copy(&self, table: &str, from_key: &str, to_key: &str) -> Result<(), StatsError> {
        self.storage.copy_legacy(table, from_key, to_key).await
    }

    pub async fn list_keys(&self, table: &str) -> Result<Vec<String>, StatsError> {
        self.storage.list_legacy_keys(table).await
    }

    pub async fn list_segments(&self, table: &str) -> Result<Vec<(String, u64)>, StatsError> {
        self.storage.list_legacy_segment_objects(table).await
    }

    pub async fn delete(&self, table: &str, relative_key: &str) {
        self.storage.delete_legacy(table, relative_key).await;
    }

    pub async fn read_footer(
        &self,
        table: &str,
        relative_key: &str,
        file_size: u64,
        key_column: Option<&str>,
    ) -> Option<SegmentMetadata> {
        self.storage
            .read_legacy_footer(table, relative_key, file_size, key_column)
            .await
    }
}
