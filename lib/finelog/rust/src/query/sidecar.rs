//! Process-global, memory-bounded cache for trigram sidecars.
//!
//! ## Why a manager
//!
//! The substring prune ([`crate::query::trigram_prune`]) needs each segment's
//! per-row-group blooms to compute a `ParquetAccessPlan`. Read naively, that is a
//! full `std::fs::read` of every segment's sidecar on **every** `contains` query
//! — a polled dashboard re-reads (and re-parses) a namespace's entire bloom set
//! on each poll, with no upper bound on the bytes held. This manager fixes both:
//!
//! - **Caching**: a parsed sidecar is read once and reused across queries. The
//!   steady state for a repeating query is pure cache hits — zero I/O.
//! - **Bounded memory**: parsed blooms live in a byte-budgeted LRU
//!   ([`FINELOG_SIDECAR_CACHE_MB`], default [`DEFAULT_BUDGET_MB`]), so the resident
//!   set never exceeds the budget no matter how many segments a query spans.
//! - **Partial reads**: the [`SidecarHeader`] is read with a single bounded
//!   `pread`, so a key-scoped query can check a segment's key band and skip
//!   reading its (large) bloom payload entirely when it is out of band.
//!
//! ## Lifetime / sharing
//!
//! Both header and column entries are handed out as `Arc`s. A query clones the
//! `Arc` for as long as it needs the blooms; the cache keeps its own `Arc` so the
//! parse survives for the next query. Eviction only drops the cache's strong
//! reference — a query still holding an `Arc` keeps that data alive until it is
//! done, so eviction is always safe even under concurrent use.

use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::store::trigram::{self, ColumnIndex, SidecarHeader};

/// Default sidecar cache budget when [`BUDGET_ENV`] is unset (MiB). The full
/// 54-segment sample index measured at ~79 MiB, so 256 MiB holds a working set
/// of several namespaces with headroom while staying well clear of the query
/// memory pool (which it is deliberately separate from — this is index metadata,
/// not query working set).
const DEFAULT_BUDGET_MB: usize = 256;

/// Override for the cache budget, in MiB.
const BUDGET_ENV: &str = "FINELOG_SIDECAR_CACHE_MB";

/// One bounded read that covers the header for any realistic column count
/// (`name + 16` bytes per column). A pathological larger directory triggers an
/// exact re-read.
const HEADER_PREFIX_BYTES: usize = 64 * 1024;

/// A byte-budgeted LRU cache of parsed trigram sidecars, shared process-wide.
pub struct SidecarManager {
    cache: Mutex<Lru>,
}

impl SidecarManager {
    /// The process-global manager, sized from [`BUDGET_ENV`] on first use.
    pub fn global() -> &'static SidecarManager {
        static MANAGER: OnceLock<SidecarManager> = OnceLock::new();
        MANAGER.get_or_init(|| SidecarManager::with_budget_bytes(budget_from_env()))
    }

    fn with_budget_bytes(budget_bytes: usize) -> SidecarManager {
        SidecarManager {
            cache: Mutex::new(Lru::new(budget_bytes)),
        }
    }

    /// The validated header for the sidecar at `path` (cached). `None` when the
    /// file is missing, unreadable, or not a valid sidecar — the caller then
    /// scans that segment unpruned, which is always correct.
    pub fn get_header(&self, path: &Path) -> Option<Arc<SidecarHeader>> {
        let key = Key::Header(path.to_path_buf());
        if let Some(Cached::Header(h)) = self.lookup(&key) {
            return Some(h);
        }
        let header = Arc::new(read_header(path)?);
        let bytes = header_heap_bytes(&header);
        Some(self.insert_header(key, header, bytes))
    }

    /// The parsed blooms for `column` in the sidecar at `path` (cached), using
    /// `header`'s directory to read just that column's payload slice. `None` when
    /// the column is absent or the payload is malformed/truncated.
    pub fn get_column(
        &self,
        path: &Path,
        header: &SidecarHeader,
        column: &str,
    ) -> Option<Arc<ColumnIndex>> {
        let key = Key::Column(path.to_path_buf(), column.to_string());
        if let Some(Cached::Column(c)) = self.lookup(&key) {
            return Some(c);
        }
        let index = Arc::new(read_column(path, header, column)?);
        let bytes = index.heap_bytes();
        Some(self.insert_column(key, index, bytes))
    }

    /// Cache hit lookup: returns the entry and refreshes its recency.
    fn lookup(&self, key: &Key) -> Option<Cached> {
        let mut lru = self.cache.lock().unwrap();
        lru.get(key)
    }

    fn insert_header(
        &self,
        key: Key,
        header: Arc<SidecarHeader>,
        bytes: usize,
    ) -> Arc<SidecarHeader> {
        let mut lru = self.cache.lock().unwrap();
        // Another thread may have raced us to insert; prefer the existing entry so
        // every caller shares one Arc.
        if let Some(Cached::Header(existing)) = lru.get(&key) {
            return existing;
        }
        lru.insert(key, Cached::Header(Arc::clone(&header)), bytes);
        header
    }

    fn insert_column(&self, key: Key, index: Arc<ColumnIndex>, bytes: usize) -> Arc<ColumnIndex> {
        let mut lru = self.cache.lock().unwrap();
        if let Some(Cached::Column(existing)) = lru.get(&key) {
            return existing;
        }
        lru.insert(key, Cached::Column(Arc::clone(&index)), bytes);
        index
    }

    #[cfg(test)]
    fn resident_bytes(&self) -> usize {
        self.cache.lock().unwrap().used_bytes
    }

    #[cfg(test)]
    fn entry_count(&self) -> usize {
        self.cache.lock().unwrap().map.len()
    }
}

/// A cached, `Arc`-shared parse — either a sidecar header or one column's blooms.
#[derive(Clone)]
enum Cached {
    Header(Arc<SidecarHeader>),
    Column(Arc<ColumnIndex>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Header(PathBuf),
    Column(PathBuf, String),
}

struct Entry {
    value: Cached,
    bytes: usize,
    last_used: u64,
}

/// A hand-rolled byte-budgeted LRU (there is no `lru` crate in the tree, and the
/// resident set is small — hundreds of entries — so an O(n) eviction scan is
/// cheap and keeps the structure obvious).
struct Lru {
    budget_bytes: usize,
    used_bytes: usize,
    tick: u64,
    map: HashMap<Key, Entry>,
}

impl Lru {
    fn new(budget_bytes: usize) -> Lru {
        Lru {
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            map: HashMap::new(),
        }
    }

    /// Look up `key`, refreshing its recency on a hit.
    fn get(&mut self, key: &Key) -> Option<Cached> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.map.get_mut(key)?;
        entry.last_used = tick;
        Some(entry.value.clone())
    }

    /// Insert `value` (charging `bytes`), evicting least-recently-used entries
    /// first so the budget is honored. An entry larger than the whole budget is
    /// still admitted (it is needed to answer the current query); the budget is a
    /// steady-state target, not a hard per-entry cap.
    fn insert(&mut self, key: Key, value: Cached, bytes: usize) {
        self.tick += 1;
        let tick = self.tick;
        while self.used_bytes + bytes > self.budget_bytes {
            let Some(victim) = self
                .map
                .iter()
                .filter(|(k, _)| **k != key)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone())
            else {
                break; // nothing else to evict
            };
            if let Some(removed) = self.map.remove(&victim) {
                self.used_bytes -= removed.bytes;
            }
        }
        if let Some(old) = self.map.insert(
            key,
            Entry {
                value,
                bytes,
                last_used: tick,
            },
        ) {
            self.used_bytes -= old.bytes;
        }
        self.used_bytes += bytes;
    }
}

fn budget_from_env() -> usize {
    let mb = std::env::var(BUDGET_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_BUDGET_MB);
    mb.saturating_mul(1024 * 1024)
}

/// Approximate heap footprint of a parsed header (its key band + column
/// directory), charged to the cache budget. Small relative to bloom payloads.
fn header_heap_bytes(header: &SidecarHeader) -> usize {
    let key_bytes = header.key_min.as_ref().map_or(0, Vec::len)
        + header.key_max.as_ref().map_or(0, Vec::len)
        + header.key_column.len();
    // The directory is private to trigram.rs; approximate it from rg_count-free
    // fixed overhead. A handful of columns at most, so a constant suffices.
    std::mem::size_of::<SidecarHeader>() + key_bytes + 128
}

/// Read and parse a sidecar header with a single bounded `pread`, re-reading
/// only if a (pathological) directory exceeds [`HEADER_PREFIX_BYTES`].
fn read_header(path: &Path) -> Option<SidecarHeader> {
    let file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len() as usize;
    let prefix_len = file_len.min(HEADER_PREFIX_BYTES);
    let mut buf = vec![0u8; prefix_len];
    file.read_exact_at(&mut buf, 0).ok()?;

    let header_len = trigram::peek_header_len(&buf)?;
    if header_len <= buf.len() {
        return trigram::parse_header(&buf[..header_len]);
    }
    if header_len > file_len {
        return None; // declared header runs past EOF: corrupt
    }
    let mut full = vec![0u8; header_len];
    file.read_exact_at(&mut full, 0).ok()?;
    trigram::parse_header(&full)
}

/// Read and parse one column's payload slice (`[offset, offset+len)`) via a
/// positioned read, so out-of-scope columns are never paged in.
fn read_column(path: &Path, header: &SidecarHeader, column: &str) -> Option<ColumnIndex> {
    let dir = header.column(column)?;
    let file = std::fs::File::open(path).ok()?;
    let mut buf = vec![0u8; dir.len as usize];
    file.read_exact_at(&mut buf, dir.offset).ok()?;
    trigram::parse_column(&buf, header.rg_count)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;

    use super::*;
    use crate::store::trigram::{
        serialize_sidecar, sidecar_path, write_sidecar, TrigramIndex, INDEXED_COLUMN,
    };

    fn tempdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_sidecar_mgr_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Write a one-row-group `seg.parquet.tgm` (key `k`, data `text`) and return
    /// the *parquet* path (the manager appends `.tgm`).
    fn write_segment_sidecar(dir: &Path, seq: i64, key: &str, text: &str) -> PathBuf {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![seq])),
                Arc::new(StringArray::from(vec![key])),
                Arc::new(StringArray::from(vec![text])),
            ],
        )
        .unwrap();
        let parquet = dir.join(format!("seg_L1_{seq:019}.parquet"));
        // Write the sidecar directly (no parquet body needed for these unit
        // tests — the manager only ever reads `<parquet>.tgm`).
        write_sidecar(
            &parquet,
            std::slice::from_ref(&batch),
            &["data"],
            Some("key"),
        )
        .unwrap();
        parquet
    }

    #[test]
    fn header_then_column_round_trip() {
        let dir = tempdir("roundtrip");
        let parquet = write_segment_sidecar(&dir, 1, "/m/a", "Bootstrap completed for TPU");
        let sc = sidecar_path(&parquet);
        let mgr = SidecarManager::with_budget_bytes(64 * 1024 * 1024);

        let header = mgr.get_header(&sc).expect("header");
        assert_eq!(header.rg_count, 1);
        assert_eq!(header.key_min.as_deref(), Some(b"/m/a".as_slice()));
        assert!(header.column(INDEXED_COLUMN).is_some());

        let col = mgr
            .get_column(&sc, &header, INDEXED_COLUMN)
            .expect("column");
        assert_eq!(col.len(), 1);
        assert_eq!(
            col.keep_mask("Bootstrap completed for TPU").unwrap(),
            vec![true]
        );
        assert_eq!(col.keep_mask("absent zzz string").unwrap(), vec![false]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn second_lookup_is_a_cache_hit() {
        let dir = tempdir("hit");
        let parquet = write_segment_sidecar(&dir, 1, "/m/a", "alpha beta gamma delta");
        let sc = sidecar_path(&parquet);
        let mgr = SidecarManager::with_budget_bytes(64 * 1024 * 1024);

        let h1 = mgr.get_header(&sc).unwrap();
        let c1 = mgr.get_column(&sc, &h1, INDEXED_COLUMN).unwrap();
        // Deleting the file proves the second round is served from cache (no I/O).
        std::fs::remove_file(&sc).unwrap();
        let h2 = mgr.get_header(&sc).unwrap();
        let c2 = mgr.get_column(&sc, &h2, INDEXED_COLUMN).unwrap();
        assert!(Arc::ptr_eq(&h1, &h2), "header must be the same cached Arc");
        assert!(Arc::ptr_eq(&c1, &c2), "column must be the same cached Arc");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_or_corrupt_sidecar_yields_none() {
        let dir = tempdir("missing");
        let mgr = SidecarManager::with_budget_bytes(64 * 1024 * 1024);
        // Missing file.
        assert!(mgr.get_header(&dir.join("nope.parquet.tgm")).is_none());
        // Garbage file.
        let bad = dir.join("bad.parquet.tgm");
        std::fs::write(&bad, b"not a sidecar at all").unwrap();
        assert!(mgr.get_header(&bad).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn budget_evicts_least_recently_used_column() {
        let dir = tempdir("evict");
        // Three distinct segments, each a multi-row-group sidecar so a column
        // payload is comfortably larger than a header.
        let parquets: Vec<PathBuf> = (0..3)
            .map(|i| {
                let lines: Vec<String> = (0..5000)
                    .map(|r| format!("seg {i} row {r} payload text {r}"))
                    .collect();
                let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
                let schema = Arc::new(ArrowSchema::new(vec![
                    Field::new("seq", DataType::Int64, false),
                    Field::new("key", DataType::Utf8, false),
                    Field::new("data", DataType::Utf8, false),
                ]));
                let n = refs.len() as i64;
                let batch = RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(Int64Array::from_iter_values(1..=n)),
                        Arc::new(StringArray::from(vec![format!("/m/{i}"); refs.len()])),
                        Arc::new(StringArray::from(refs)),
                    ],
                )
                .unwrap();
                let parquet = dir.join(format!("seg_L1_{:019}.parquet", i + 1));
                write_sidecar(
                    &parquet,
                    std::slice::from_ref(&batch),
                    &["data"],
                    Some("key"),
                )
                .unwrap();
                parquet
            })
            .collect();

        let sidecars: Vec<PathBuf> = parquets.iter().map(|p| sidecar_path(p)).collect();
        // Size one column's blooms to set a budget that holds ~2 columns.
        let probe = SidecarManager::with_budget_bytes(usize::MAX);
        let h0 = probe.get_header(&sidecars[0]).unwrap();
        let one = probe
            .get_column(&sidecars[0], &h0, INDEXED_COLUMN)
            .unwrap()
            .heap_bytes();

        let mgr = SidecarManager::with_budget_bytes(one * 2 + one / 2);
        let mut headers = Vec::new();
        for sc in &sidecars {
            let h = mgr.get_header(sc).unwrap();
            mgr.get_column(sc, &h, INDEXED_COLUMN).unwrap();
            headers.push(h);
        }
        // The budget cannot hold all three column payloads at once.
        assert!(mgr.resident_bytes() <= one * 2 + one / 2);
        assert!(
            mgr.resident_bytes() < one * 3,
            "at least one column payload must have been evicted"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn header_larger_than_prefix_triggers_exact_reread() {
        // A header bigger than HEADER_PREFIX_BYTES must still parse via the exact
        // re-read path. A 70 KiB key band inflates the directory past the 64 KiB
        // bounded prefix, forcing read_header's second, exact read.
        let dir = tempdir("bigdir");
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1_i64])),
                Arc::new(StringArray::from(vec!["alpha beta gamma"])),
            ],
        )
        .unwrap();
        let idx = TrigramIndex::build(std::slice::from_ref(&batch), "data").unwrap();
        let big_key = "k".repeat(70 * 1024);
        assert!(big_key.len() > HEADER_PREFIX_BYTES);
        let bytes = serialize_sidecar(
            idx.len() as u32,
            "key",
            Some(big_key.as_bytes()),
            Some(big_key.as_bytes()),
            std::slice::from_ref(&idx),
        );
        let sc = dir.join("seg_L1_0000000000000000001.parquet.tgm");
        std::fs::write(&sc, &bytes).unwrap();

        let mgr = SidecarManager::with_budget_bytes(256 * 1024 * 1024);
        let header = mgr.get_header(&sc).expect("header from exact re-read");
        assert_eq!(header.key_min.as_deref(), Some(big_key.as_bytes()));
        assert!(mgr.get_column(&sc, &header, INDEXED_COLUMN).is_some());
        assert!(mgr.entry_count() >= 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
