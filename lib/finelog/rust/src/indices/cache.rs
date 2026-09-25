//! Memory-bounded cache for parsed `.fidx` bundle sections.
//!
//! The cache is addressed by the bundle file a caller already resolved from an
//! artifact reference. It never derives a bundle filename from a source
//! filename; [`IndexRegistry`](crate::indices::IndexRegistry) owns that
//! resolution.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use uuid::Uuid;

use crate::indices::exact::ExactSection;
use crate::indices::format::{self, BundleHeader, SectionKind};
use crate::indices::group_extrema::{GroupExtremaConfig, GroupExtremaSection};
use crate::indices::trigram::{self, ColumnIndex};
use crate::indices::{
    parse_group_extrema_config, parse_trigram_coverage, read_exact_section,
    read_group_extrema_section, trigram_section_id, TrigramCoverage,
};

pub const DEFAULT_INDEX_CACHE_MB: usize = 256;

pub struct IndexCache {
    state: Mutex<CacheState>,
    loaded: Condvar,
    corrupt_bundles: AtomicU64,
    corrupt_sections: AtomicU64,
    load_attempts: AtomicU64,
    header_load_attempts: AtomicU64,
    section_load_attempts: AtomicU64,
    coalesced_waits: AtomicU64,
    evictions: AtomicU64,
    aggregate_full: AtomicU64,
    aggregate_partial: AtomicU64,
    aggregate_declined: AtomicU64,
    aggregate_fallbacks: AtomicU64,
}

struct CacheState {
    cache: Lru,
    loading: HashSet<Key>,
}

struct Loaded<T> {
    cached: Cached,
    bytes: usize,
    value: Arc<T>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AggregateOutcome {
    Full,
    Partial,
    Declined,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AggregateStats {
    pub full: u64,
    pub partial: u64,
    pub declined: u64,
    pub fallbacks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LoadStats {
    pub attempts: u64,
    pub header_attempts: u64,
    pub section_attempts: u64,
    pub coalesced_waits: u64,
    pub entries: usize,
    pub used_bytes: usize,
    pub budget_bytes: usize,
    pub evictions: u64,
}

impl IndexCache {
    pub fn new(budget_mb: usize) -> Self {
        Self::with_budget_bytes(budget_mb.saturating_mul(1024 * 1024))
    }

    fn with_budget_bytes(budget_bytes: usize) -> Self {
        Self {
            state: Mutex::new(CacheState {
                cache: Lru::new(budget_bytes),
                loading: HashSet::new(),
            }),
            loaded: Condvar::new(),
            corrupt_bundles: AtomicU64::new(0),
            corrupt_sections: AtomicU64::new(0),
            load_attempts: AtomicU64::new(0),
            header_load_attempts: AtomicU64::new(0),
            section_load_attempts: AtomicU64::new(0),
            coalesced_waits: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            aggregate_full: AtomicU64::new(0),
            aggregate_partial: AtomicU64::new(0),
            aggregate_declined: AtomicU64::new(0),
            aggregate_fallbacks: AtomicU64::new(0),
        }
    }

    /// Load `bundle_path` only when it is bound to the current source segment.
    pub fn get_header(
        &self,
        bundle_path: &Path,
        source_id: Uuid,
        row_count: u64,
    ) -> Option<Arc<BundleHeader>> {
        let key = Key::Header(bundle_path.to_path_buf(), source_id);
        let header = self.get_or_load(key, || {
            let Some(header) = format::read_header(bundle_path) else {
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
            Some(Loaded {
                cached: Cached::Header(Arc::clone(&header)),
                bytes,
                value: header,
            })
        })?;
        header.matches(source_id, row_count).then_some(header)
    }

    pub fn get_trigram(
        &self,
        bundle_path: &Path,
        header: &BundleHeader,
        column: &str,
    ) -> Option<(TrigramCoverage, Arc<ColumnIndex>)> {
        let id = trigram_section_id(column);
        let section = header.section(&id)?;
        if section.kind != SectionKind::TrigramBloom {
            return None;
        }
        let coverage = parse_trigram_coverage(&section.coverage)?;
        let key = Key::Section(
            bundle_path.to_path_buf(),
            header.binding.segment_id,
            id.clone(),
        );
        let index = self.get_or_load(key, || {
            let Some(payload) = format::read_section(bundle_path, header, &id) else {
                self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let Some(index) = trigram::parse_column(&payload, coverage.span_count) else {
                self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let bytes = index.heap_bytes();
            let index = Arc::new(index);
            Some(Loaded {
                cached: Cached::Trigram(Arc::clone(&index)),
                bytes,
                value: index,
            })
        })?;
        Some((coverage, index))
    }

    pub fn get_exact(
        &self,
        bundle_path: &Path,
        header: &BundleHeader,
        kind: SectionKind,
    ) -> Option<Arc<ExactSection>> {
        let section = header
            .sections
            .iter()
            .find(|section| section.kind == kind)?;
        let key = Key::Section(
            bundle_path.to_path_buf(),
            header.binding.segment_id,
            section.id.clone(),
        );
        self.get_or_load(key, || {
            let Some(index) = read_exact_section(bundle_path, header, kind) else {
                self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let bytes = index.heap_bytes();
            let index = Arc::new(index);
            Some(Loaded {
                cached: Cached::Exact(Arc::clone(&index)),
                bytes,
                value: index,
            })
        })
    }

    pub fn get_group_extrema(
        &self,
        bundle_path: &Path,
        header: &BundleHeader,
        config: &GroupExtremaConfig,
    ) -> Option<Arc<GroupExtremaSection>> {
        let section = header.sections.iter().find(|section| {
            section.kind == SectionKind::GroupExtrema
                && parse_group_extrema_config(&section.coverage).as_ref() == Some(config)
        })?;
        let key = Key::Section(
            bundle_path.to_path_buf(),
            header.binding.segment_id,
            section.id.clone(),
        );
        self.get_or_load(key, || {
            let Some(index) = read_group_extrema_section(bundle_path, header, config) else {
                self.corrupt_sections.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let bytes = index.heap_bytes();
            let index = Arc::new(index);
            Some(Loaded {
                cached: Cached::GroupExtrema(Arc::clone(&index)),
                bytes,
                value: index,
            })
        })
    }

    pub fn invalidate(&self, bundle_path: &Path) {
        self.state.lock().unwrap().cache.remove_path(bundle_path);
    }

    pub fn corruption_counts(&self) -> CorruptionCounts {
        CorruptionCounts {
            bundles: self.corrupt_bundles.load(Ordering::Relaxed),
            sections: self.corrupt_sections.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn record_aggregate(&self, outcome: AggregateOutcome) {
        let counter = match outcome {
            AggregateOutcome::Full => &self.aggregate_full,
            AggregateOutcome::Partial => &self.aggregate_partial,
            AggregateOutcome::Declined => &self.aggregate_declined,
            AggregateOutcome::Fallback => &self.aggregate_fallbacks,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn aggregate_stats(&self) -> AggregateStats {
        AggregateStats {
            full: self.aggregate_full.load(Ordering::Relaxed),
            partial: self.aggregate_partial.load(Ordering::Relaxed),
            declined: self.aggregate_declined.load(Ordering::Relaxed),
            fallbacks: self.aggregate_fallbacks.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn load_stats(&self) -> LoadStats {
        let state = self.state.lock().unwrap();
        LoadStats {
            attempts: self.load_attempts.load(Ordering::Relaxed),
            header_attempts: self.header_load_attempts.load(Ordering::Relaxed),
            section_attempts: self.section_load_attempts.load(Ordering::Relaxed),
            coalesced_waits: self.coalesced_waits.load(Ordering::Relaxed),
            entries: state.cache.map.len(),
            used_bytes: state.cache.used_bytes,
            budget_bytes: state.cache.budget_bytes,
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    fn get_or_load<T>(&self, key: Key, load: impl FnOnce() -> Option<Loaded<T>>) -> Option<Arc<T>>
    where
        Cached: CachedValue<T>,
    {
        let mut waited = false;
        {
            let mut state = self.state.lock().unwrap();
            loop {
                if let Some(existing) = state.cache.get(&key).and_then(CachedValue::value) {
                    return Some(existing);
                }
                if state.loading.insert(key.clone()) {
                    break;
                }
                if !waited {
                    self.coalesced_waits.fetch_add(1, Ordering::Relaxed);
                    waited = true;
                }
                state = self.loaded.wait(state).unwrap();
            }
        }
        self.load_attempts.fetch_add(1, Ordering::Relaxed);
        match &key {
            Key::Header(..) => self.header_load_attempts.fetch_add(1, Ordering::Relaxed),
            Key::Section(..) => self.section_load_attempts.fetch_add(1, Ordering::Relaxed),
        };
        let loaded = load();

        let mut state = self.state.lock().unwrap();
        state.loading.remove(&key);
        let value = loaded.map(|loaded| {
            if let Some(existing) = state.cache.get(&key).and_then(CachedValue::value) {
                existing
            } else {
                let evictions = state.cache.insert(key, loaded.cached, loaded.bytes);
                self.evictions
                    .fetch_add(evictions as u64, Ordering::Relaxed);
                loaded.value
            }
        });
        drop(state);
        self.loaded.notify_all();
        value
    }
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

impl CachedValue<GroupExtremaSection> for Cached {
    fn value(self) -> Option<Arc<GroupExtremaSection>> {
        match self {
            Self::GroupExtrema(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone)]
enum Cached {
    Header(Arc<BundleHeader>),
    Trigram(Arc<ColumnIndex>),
    Exact(Arc<ExactSection>),
    GroupExtrema(Arc<GroupExtremaSection>),
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

    fn insert(&mut self, key: Key, value: Cached, bytes: usize) -> usize {
        self.tick += 1;
        let mut evictions = 0;
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
                evictions += 1;
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
        evictions
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Barrier;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::indices::format::{Exactness, SectionInput, SegmentBinding};

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
            schema_fingerprint: format::fingerprint(b"schema"),
            policy_fingerprint: format::fingerprint(b"policy"),
        }
    }

    fn exact_section() -> SectionInput {
        SectionInput {
            id: "exact-postings".to_string(),
            kind: SectionKind::ExactPostings,
            method_version: 1,
            exactness: Exactness::ExactRows,
            coverage: b"name".to_vec(),
            payload: crate::indices::exact::serialize(&ExactSection {
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
        let bundle =
            format::write_bundle(&parquet, &binding(first_id), &[exact_section()]).unwrap();
        let cache = IndexCache::with_budget_bytes(1024 * 1024);
        assert!(cache.get_header(&bundle, first_id, 7).is_some());

        format::write_bundle(&parquet, &binding(second_id), &[exact_section()]).unwrap();
        assert!(cache.get_header(&bundle, second_id, 7).is_some());
        assert_eq!(
            cache.corruption_counts(),
            CorruptionCounts {
                bundles: 0,
                sections: 0,
            }
        );
        std::fs::remove_file(bundle).ok();
    }

    #[test]
    fn corruption_counters_distinguish_bundle_and_section_failures() {
        let parquet = temp_path("section_corruption.parquet");
        let segment_id = Uuid::from_u128(3);
        let path =
            format::write_bundle(&parquet, &binding(segment_id), &[exact_section()]).unwrap();
        let cache = IndexCache::with_budget_bytes(1024 * 1024);
        let header = cache.get_header(&path, segment_id, 7).unwrap();
        let section = header.section("exact-postings").unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&[0xff], section.offset)
            .unwrap();
        assert!(cache
            .get_exact(&path, &header, SectionKind::ExactPostings)
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
            format::write_bundle(&parquet, &binding(segment_id), &[exact_section()]).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&[0xff], 0)
            .unwrap();
        assert!(cache.get_header(&path, segment_id, 7).is_none());
        assert_eq!(
            cache.corruption_counts(),
            CorruptionCounts {
                bundles: 1,
                sections: 1,
            }
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn aggregate_counters_are_isolated_per_store_cache() {
        let first = IndexCache::with_budget_bytes(1024);
        let second = IndexCache::with_budget_bytes(1024);
        first.record_aggregate(AggregateOutcome::Full);
        first.record_aggregate(AggregateOutcome::Declined);
        second.record_aggregate(AggregateOutcome::Fallback);

        assert_eq!(
            first.aggregate_stats(),
            AggregateStats {
                full: 1,
                partial: 0,
                declined: 1,
                fallbacks: 0,
            }
        );
        assert_eq!(
            second.aggregate_stats(),
            AggregateStats {
                full: 0,
                partial: 0,
                declined: 0,
                fallbacks: 1,
            }
        );
    }

    #[test]
    fn concurrent_misses_share_one_load() {
        const READERS: usize = 8;

        let cache = Arc::new(IndexCache::with_budget_bytes(1024));
        let key = Key::Section(
            PathBuf::from("shared.fidx"),
            Uuid::from_u128(4),
            "exact-postings".to_string(),
        );
        let start = Arc::new(Barrier::new(READERS + 1));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let loads = Arc::new(AtomicUsize::new(0));
        let handles = (0..READERS)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                let start = Arc::clone(&start);
                let release = Arc::clone(&release);
                let loads = Arc::clone(&loads);
                thread::spawn(move || {
                    start.wait();
                    cache
                        .get_or_load(key, || {
                            loads.fetch_add(1, Ordering::Relaxed);
                            let (lock, loaded) = &*release;
                            let mut can_finish = lock.lock().unwrap();
                            while !*can_finish {
                                can_finish = loaded.wait(can_finish).unwrap();
                            }
                            let section = Arc::new(ExactSection {
                                total_rows: 7,
                                columns: BTreeMap::new(),
                            });
                            Some(Loaded {
                                cached: Cached::Exact(Arc::clone(&section)),
                                bytes: 1,
                                value: section,
                            })
                        })
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        let deadline = Instant::now() + Duration::from_secs(5);
        while cache.load_stats().coalesced_waits != (READERS - 1) as u64 {
            assert!(
                Instant::now() < deadline,
                "readers did not coalesce in time"
            );
            thread::yield_now();
        }
        let (lock, loaded) = &*release;
        *lock.lock().unwrap() = true;
        loaded.notify_one();

        let sections = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert!(sections
            .iter()
            .all(|section| Arc::ptr_eq(section, &sections[0])));
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(
            cache.load_stats(),
            LoadStats {
                attempts: 1,
                header_attempts: 0,
                section_attempts: 1,
                coalesced_waits: (READERS - 1) as u64,
                entries: 1,
                used_bytes: 1,
                budget_bytes: 1024,
                evictions: 0,
            }
        );
    }
}
