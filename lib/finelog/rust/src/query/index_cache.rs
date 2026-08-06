//! Memory-bounded cache for parsed `.fidx` bundle sections.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::store::exact::ExactSection;
use crate::store::index_bundle::{self, BundleHeader, SectionKind};
use crate::store::segment::segment_id_and_row_group_rows;
use crate::store::segment_index::{
    parse_trigram_coverage, read_exact_section, trigram_section_id, TrigramCoverage,
};
use crate::store::trigram::{self, ColumnIndex};

pub const DEFAULT_INDEX_CACHE_MB: usize = 256;

pub struct IndexCache {
    cache: Mutex<Lru>,
    corrupt_bundles: AtomicU64,
    corrupt_sections: AtomicU64,
}

impl fmt::Debug for IndexCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexCache")
            .field("corruption_counts", &self.corruption_counts())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorruptionCounts {
    pub bundles: u64,
    pub sections: u64,
}

pub struct IndexedSegment {
    pub header: Arc<BundleHeader>,
    pub row_group_rows: Arc<[usize]>,
}

impl IndexCache {
    pub fn new(budget_mb: usize) -> Self {
        Self::with_budget_bytes(budget_mb.saturating_mul(1024 * 1024))
    }

    fn with_budget_bytes(budget_bytes: usize) -> Self {
        Self {
            cache: Mutex::new(Lru::new(budget_bytes)),
            corrupt_bundles: AtomicU64::new(0),
            corrupt_sections: AtomicU64::new(0),
        }
    }

    /// Resolve the source footer and its matching bundle header as one lookup.
    pub fn indexed_segment(&self, parquet_path: &Path) -> Option<IndexedSegment> {
        let (source_id, row_group_rows) = segment_id_and_row_group_rows(parquet_path)?;
        let source_rows = row_group_rows.iter().sum::<usize>() as u64;
        let header = self.get_header(parquet_path, source_id, source_rows)?;
        Some(IndexedSegment {
            header,
            row_group_rows,
        })
    }

    /// Load a bundle only when it is bound to the current source segment.
    pub fn get_header(
        &self,
        parquet_path: &Path,
        source_id: Uuid,
        row_count: u64,
    ) -> Option<Arc<BundleHeader>> {
        let bundle_path = index_bundle::bundle_path(parquet_path);
        let key = Key::Header(bundle_path.clone(), source_id);
        if let Some(Cached::Header(header)) = self.lookup(&key) {
            return header.matches(source_id, row_count).then_some(header);
        }
        let Some(header) = index_bundle::read_header(&bundle_path) else {
            if bundle_path.exists() {
                self.corrupt_bundles.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        };
        if !header.matches(source_id, row_count) {
            tracing::debug!(
                path = %bundle_path.display(),
                expected_segment_identity = %source_id,
                bundle_segment_identity = %header.binding.segment_id,
                expected_rows = row_count,
                bundle_rows = header.binding.row_count,
                "stale index bundle does not match source segment"
            );
            return None;
        }
        let bytes = header
            .sections
            .iter()
            .fold(std::mem::size_of::<BundleHeader>(), |total, section| {
                total + section.id.len() + section.coverage.len() + 96
            });
        let header = Arc::new(header);
        Some(self.insert(key, Cached::Header(Arc::clone(&header)), bytes, || header))
    }

    pub fn get_trigram(
        &self,
        parquet_path: &Path,
        header: &BundleHeader,
        column: &str,
    ) -> Option<(TrigramCoverage, Arc<ColumnIndex>)> {
        let bundle_path = index_bundle::bundle_path(parquet_path);
        let id = trigram_section_id(column);
        let section = header.section(&id)?;
        if section.kind != SectionKind::TrigramBloom {
            return None;
        }
        let coverage = parse_trigram_coverage(&section.coverage)?;
        let key = Key::Section(bundle_path.clone(), header.binding.segment_id, id.clone());
        if let Some(Cached::Trigram(index)) = self.lookup(&key) {
            return Some((coverage, index));
        }
        let Some(payload) = index_bundle::read_section(&bundle_path, header, &id) else {
            self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let Some(index) = trigram::parse_column(&payload, coverage.span_count) else {
            self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let bytes = index.heap_bytes();
        let index = Arc::new(index);
        let index = self.insert(key, Cached::Trigram(Arc::clone(&index)), bytes, || index);
        Some((coverage, index))
    }

    pub fn get_exact(
        &self,
        parquet_path: &Path,
        header: &BundleHeader,
        kind: SectionKind,
    ) -> Option<Arc<ExactSection>> {
        let bundle_path = index_bundle::bundle_path(parquet_path);
        let section = header
            .sections
            .iter()
            .find(|section| section.kind == kind)?;
        let key = Key::Section(
            bundle_path.clone(),
            header.binding.segment_id,
            section.id.clone(),
        );
        if let Some(Cached::Exact(index)) = self.lookup(&key) {
            return Some(index);
        }
        let Some(index) = read_exact_section(&bundle_path, header, kind) else {
            self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let bytes = index.heap_bytes();
        let index = Arc::new(index);
        Some(self.insert(key, Cached::Exact(Arc::clone(&index)), bytes, || index))
    }

    pub fn invalidate(&self, bundle_path: &Path) {
        self.cache.lock().unwrap().remove_path(bundle_path);
    }

    pub fn corruption_counts(&self) -> CorruptionCounts {
        CorruptionCounts {
            bundles: self.corrupt_bundles.load(Ordering::Relaxed),
            sections: self.corrupt_sections.load(Ordering::Relaxed),
        }
    }

    fn lookup(&self, key: &Key) -> Option<Cached> {
        self.cache.lock().unwrap().get(key)
    }

    fn insert<T>(
        &self,
        key: Key,
        cached: Cached,
        bytes: usize,
        value: impl FnOnce() -> Arc<T>,
    ) -> Arc<T>
    where
        Cached: CachedValue<T>,
    {
        let mut cache = self.cache.lock().unwrap();
        if let Some(existing) = cache.get(&key).and_then(CachedValue::value) {
            return existing;
        }
        let value = value();
        cache.insert(key, cached, bytes);
        value
    }
}

#[cfg(test)]
pub fn test_index_cache() -> Arc<IndexCache> {
    Arc::new(IndexCache::new(16))
}

trait CachedValue<T> {
    fn value(self) -> Option<Arc<T>>;
}

impl CachedValue<BundleHeader> for Cached {
    fn value(self) -> Option<Arc<BundleHeader>> {
        match self {
            Self::Header(value) => Some(value),
            _ => None,
        }
    }
}

impl CachedValue<ColumnIndex> for Cached {
    fn value(self) -> Option<Arc<ColumnIndex>> {
        match self {
            Self::Trigram(value) => Some(value),
            _ => None,
        }
    }
}

impl CachedValue<ExactSection> for Cached {
    fn value(self) -> Option<Arc<ExactSection>> {
        match self {
            Self::Exact(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone)]
enum Cached {
    Header(Arc<BundleHeader>),
    Trigram(Arc<ColumnIndex>),
    Exact(Arc<ExactSection>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Header(PathBuf, Uuid),
    Section(PathBuf, Uuid, String),
}

impl Key {
    fn path(&self) -> &Path {
        match self {
            Self::Header(path, _) | Self::Section(path, _, _) => path,
        }
    }
}

struct Entry {
    value: Cached,
    bytes: usize,
    last_used: u64,
}

struct Lru {
    budget_bytes: usize,
    used_bytes: usize,
    tick: u64,
    map: HashMap<Key, Entry>,
}

impl Lru {
    fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            map: HashMap::new(),
        }
    }

    fn get(&mut self, key: &Key) -> Option<Cached> {
        self.tick += 1;
        let entry = self.map.get_mut(key)?;
        entry.last_used = self.tick;
        Some(entry.value.clone())
    }

    fn remove_path(&mut self, path: &Path) {
        let mut freed = 0;
        self.map.retain(|key, entry| {
            if key.path() == path {
                freed += entry.bytes;
                false
            } else {
                true
            }
        });
        self.used_bytes -= freed;
    }

    fn insert(&mut self, key: Key, value: Cached, bytes: usize) {
        self.tick += 1;
        while self.used_bytes.saturating_add(bytes) > self.budget_bytes {
            let Some(victim) = self
                .map
                .iter()
                .filter(|(candidate, _)| **candidate != key)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(candidate, _)| candidate.clone())
            else {
                break;
            };
            if let Some(entry) = self.map.remove(&victim) {
                self.used_bytes -= entry.bytes;
            }
        }
        if let Some(old) = self.map.insert(
            key,
            Entry {
                value,
                bytes,
                last_used: self.tick,
            },
        ) {
            self.used_bytes -= old.bytes;
        }
        self.used_bytes += bytes;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    use crate::store::index_bundle::{Exactness, SectionInput, SegmentBinding};

    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "finelog_index_cache_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn binding(segment_id: Uuid) -> SegmentBinding {
        SegmentBinding {
            segment_id,
            row_count: 7,
            schema_fingerprint: index_bundle::fingerprint(b"schema"),
            policy_fingerprint: index_bundle::fingerprint(b"policy"),
        }
    }

    fn exact_section() -> SectionInput {
        SectionInput {
            id: "exact-postings".to_string(),
            kind: SectionKind::ExactPostings,
            method_version: 1,
            exactness: Exactness::ExactRows,
            coverage: b"name".to_vec(),
            payload: crate::store::exact::serialize(&ExactSection {
                total_rows: 7,
                columns: BTreeMap::new(),
            }),
        }
    }

    #[test]
    fn cache_key_tracks_segment_identity_at_a_reused_path() {
        let parquet = temp_path("identity.parquet");
        let first_id = Uuid::from_u128(1);
        let second_id = Uuid::from_u128(2);
        index_bundle::write_bundle(&parquet, &binding(first_id), &[exact_section()]).unwrap();
        let cache = IndexCache::with_budget_bytes(1024 * 1024);
        assert!(cache.get_header(&parquet, first_id, 7).is_some());

        index_bundle::write_bundle(&parquet, &binding(second_id), &[exact_section()]).unwrap();
        assert!(cache.get_header(&parquet, second_id, 7).is_some());
        assert_eq!(
            cache.corruption_counts(),
            CorruptionCounts {
                bundles: 0,
                sections: 0,
            }
        );
        std::fs::remove_file(index_bundle::bundle_path(&parquet)).ok();
    }

    #[test]
    fn corruption_counters_distinguish_bundle_and_section_failures() {
        let parquet = temp_path("section_corruption.parquet");
        let segment_id = Uuid::from_u128(3);
        let path =
            index_bundle::write_bundle(&parquet, &binding(segment_id), &[exact_section()]).unwrap();
        let cache = IndexCache::with_budget_bytes(1024 * 1024);
        let header = cache.get_header(&parquet, segment_id, 7).unwrap();
        let section = header.section("exact-postings").unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&[0xff], section.offset)
            .unwrap();
        assert!(cache
            .get_exact(&parquet, &header, SectionKind::ExactPostings)
            .is_none());
        assert_eq!(
            cache.corruption_counts(),
            CorruptionCounts {
                bundles: 0,
                sections: 1,
            }
        );
        std::fs::remove_file(path).ok();

        let parquet = temp_path("bundle_corruption.parquet");
        let path =
            index_bundle::write_bundle(&parquet, &binding(segment_id), &[exact_section()]).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&[0xff], 0)
            .unwrap();
        assert!(cache.get_header(&parquet, segment_id, 7).is_none());
        assert_eq!(
            cache.corruption_counts(),
            CorruptionCounts {
                bundles: 1,
                sections: 1,
            }
        );
        std::fs::remove_file(path).ok();
    }
}
