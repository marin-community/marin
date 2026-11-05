//! Lazy bundle inputs. Object-backed reads use the same cache-aware boundary
//! as source objects; legacy sidecars retain their recorded filesystem paths.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use parquet::arrow::async_reader::AsyncFileReader;
use uuid::Uuid;

use crate::indices::format::{self, BundleHeader};
use crate::store::object_store::{ObjectReference, ObjectStore};
use BundleReadError::{Corrupt, Unavailable};

type SourceLayout = (Uuid, Arc<[usize]>);

/// Shared across all optional index reads in one provider snapshot. Expiration
/// declines further I/O while already parsed cache entries remain usable.
pub(crate) struct ObjectInputs {
    pub store: Arc<dyn ObjectStore>,
    pub references: BTreeMap<PathBuf, ObjectReference>,
    deadline: OnceLock<tokio::time::Instant>,
    footers: Mutex<BTreeMap<PathBuf, SourceLayout>>,
}

const INDEX_READ_BUDGET: Duration = Duration::from_millis(500);

impl ObjectInputs {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        references: BTreeMap<PathBuf, ObjectReference>,
    ) -> Self {
        Self {
            store,
            references,
            deadline: OnceLock::new(),
            footers: Mutex::new(BTreeMap::new()),
        }
    }

    fn deadline(&self) -> tokio::time::Instant {
        *self
            .deadline
            .get_or_init(|| tokio::time::Instant::now() + INDEX_READ_BUDGET)
    }

    pub fn source_layout(&self, path: &Path) -> Option<(Uuid, Arc<[usize]>)> {
        if let Some(layout) = self.footers.lock().unwrap().get(path) {
            return Some(layout.clone());
        }
        let reference = self.references.get(path)?;
        let deadline = self.deadline();
        if deadline <= tokio::time::Instant::now() {
            return None;
        }
        let metadata = tokio::runtime::Handle::current().block_on(async {
            tokio::time::timeout_at(
                deadline,
                crate::query::object_reader::ObjectRangeReader::new(
                    Arc::clone(&self.store),
                    reference.clone(),
                )
                .get_metadata(None),
            )
            .await
            .ok()?
            .ok()
        })?;
        let identity = crate::store::segment::segment_id_of(&metadata)?;
        let rows = metadata
            .row_groups()
            .iter()
            .map(|group| usize::try_from(group.num_rows()).ok())
            .collect::<Option<Vec<_>>>()?;
        let layout = (identity, Arc::<[usize]>::from(rows));
        self.footers
            .lock()
            .unwrap()
            .insert(path.to_path_buf(), layout.clone());
        Some(layout)
    }

    fn read(
        &self,
        reference: &ObjectReference,
        range: std::ops::Range<u64>,
    ) -> Option<bytes::Bytes> {
        let deadline = self.deadline();
        if deadline <= tokio::time::Instant::now() {
            return None;
        }
        // All registry I/O runs on spawn_blocking workers. Keeping the timeout
        // inside the blocking task also bounds work after its caller cancels.
        tokio::runtime::Handle::current().block_on(async {
            match tokio::time::timeout_at(deadline, self.store.read_range(reference, range)).await {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(error)) => {
                    tracing::debug!(%error, object = reference.id.as_str(), "index range unavailable; scan source");
                    None
                }
                Err(_) => None,
            }
        })
    }
}

pub struct BundleInput<'a> {
    pub path: &'a Path,
    pub(crate) objects: Option<&'a ObjectInputs>,
}

pub enum BundleReadError {
    Unavailable,
    Corrupt,
}

impl<'a> BundleInput<'a> {
    pub fn local(path: &'a Path) -> Self {
        Self {
            path,
            objects: None,
        }
    }

    pub fn read_header(&self) -> Result<BundleHeader, BundleReadError> {
        let Some(objects) = self.objects else {
            return format::read_header(self.path).ok_or_else(|| {
                if self.path.exists() {
                    Corrupt
                } else {
                    Unavailable
                }
            });
        };
        let reference = objects.references.get(self.path).ok_or(Unavailable)?;
        let prefix = objects
            .read(reference, 0..format::FIXED_PREFIX_LEN as u64)
            .ok_or(Unavailable)?;
        let header_len = format::peek_header_len(&prefix).ok_or(Corrupt)?;
        if header_len > format::MAX_HEADER_LEN || header_len as u64 > reference.version.byte_size {
            return Err(Corrupt);
        }
        let bytes = objects
            .read(reference, 0..header_len as u64)
            .ok_or(Unavailable)?;
        let header = format::parse_header(&bytes).ok_or(Corrupt)?;
        if header.bundle_len != reference.version.byte_size {
            return Err(Corrupt);
        }
        Ok(header)
    }

    pub fn read_section(
        &self,
        header: &BundleHeader,
        id: &str,
    ) -> Result<Vec<u8>, BundleReadError> {
        let Some(objects) = self.objects else {
            return format::read_section(self.path, header, id).ok_or_else(|| {
                if self.path.exists() {
                    Corrupt
                } else {
                    Unavailable
                }
            });
        };
        let reference = objects.references.get(self.path).ok_or(Unavailable)?;
        let section = header.section(id).ok_or(Corrupt)?;
        let bytes = objects
            .read(
                reference,
                section.offset..section.offset.checked_add(section.len).ok_or(Corrupt)?,
            )
            .ok_or(Unavailable)?;
        if !format::valid_section(section, &bytes) {
            return Err(Corrupt);
        }
        Ok(bytes.to_vec())
    }
}
