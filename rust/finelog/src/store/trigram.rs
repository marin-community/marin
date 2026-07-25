//! Per-row-group trigram (3-gram) presence index over a string column, stored as
//! a sidecar file next to its parquet segment (`seg_L*_*.parquet.tgm`).
//!
//! ## What it is
//!
//! `contains(col, needle)` / `LIKE '%needle%'` is opaque to parquet statistics —
//! min/max and bloom filters key on whole values, so a substring match forces a
//! decode of the column for every row in the scanned band. This index makes the
//! match sub-linear: for each parquet row group it records the set of byte
//! 3-grams present in `col`, as a Bloom filter. A query decomposes the needle
//! into its 3-grams; a row group is **skipped unless it contains ALL of them**.
//!
//! The contract is **conservative — never a false negative**. A Bloom filter can
//! only report "definitely absent" or "maybe present", so a kept row group might
//! not actually match (the scan re-checks `contains()` exactly), but a row group
//! that truly contains the needle is never skipped. Needles shorter than 3 bytes
//! have no trigrams and fall back to a full scan.
//!
//! ## File layout (v1)
//!
//! The sidecar is **directory-first and multi-column**, so a reader can map the
//! cheap header (a few hundred bytes) without paging in the per-row-group blooms,
//! and load only the column(s) a query needs:
//!
//! ```text
//! magic "FLTG" | version u8 | flags u8 | header_len u32 | rg_count u32
//! key_column   : u16 len + bytes        (the segment's ordering-key column)
//! key_min      : opt(u32 len + bytes)   (segment key band, UTF-8 bytes; absent
//! key_max      : opt(u32 len + bytes)    for non-string keys)
//! col_count u16
//!   per column: name u16 len + bytes | payload_offset u64 | payload_len u64
//! -- header_len marks the end of the directory; payloads follow --
//! per column @ payload_offset:
//!   per row group: k u8 | m_words u32 | words (m_words × u64)
//! ```
//!
//! `header_len` lets the [`crate::query::sidecar::SidecarManager`] read the
//! header with a single bounded `pread`, check the key band, and only then read a
//! column's payload slice. `key_min`/`key_max` carry the segment's key range so a
//! `key`-constrained `contains` query can skip out-of-band segments without
//! touching their blooms at all. New columns or header fields bump `version`;
//! readers treat any unrecognized version as absent (scan unpruned).
//!
//! ## Why a sidecar (not the parquet footer)
//!
//! The footer is read in full on every file open. A per-row-group structure
//! embedded there would tax every query — including the hot `key = … ORDER BY
//! seq DESC LIMIT n` tail that never calls `contains()`. The sidecar is loaded
//! lazily, only for substring queries. It is a pure, optional, derivable function
//! of the column: a missing or stale sidecar is never *wrong*, only unpruned.
//!
//! ## Row-group alignment
//!
//! `ArrowWriter` flushes a row group strictly every `ROW_GROUP_SIZE` rows (no
//! byte cap is set), so the index — built by chunking the written `data` values
//! at the same stride — aligns 1:1 with the parquet row groups by global row
//! index. The prune path re-checks `index.len() == parquet.num_row_groups` and
//! falls back to scan-all on any mismatch.

use std::path::{Path, PathBuf};

use arrow::array::{Array, LargeStringArray, RecordBatch, StringArray};

use crate::store::segment::ROW_GROUP_SIZE;

/// Sidecar file magic + version.
const TGM_MAGIC: &[u8; 4] = b"FLTG";
const TGM_VERSION: u8 = 1;

/// Minimum needle length that decomposes into at least one trigram. Shorter
/// needles index nothing and must fall back to a scan.
pub const MIN_TRIGRAM_LEN: usize = 3;

/// The single string column indexed for substring (`contains`) pruning in v1.
/// `key` is already range-prunable, so it is not indexed; revisit if a
/// `contains(key, …)` workload appears. The format itself is multi-column, so
/// adding a second indexed column is an additive change.
pub const INDEXED_COLUMN: &str = "data";

/// Target Bloom false-positive rate per row group. A false positive only keeps a
/// row group that doesn't match (a wasted decode the scan filters out); it never
/// drops a match. 1% trades a small over-scan for a ~2x smaller sidecar than an
/// exact trigram set.
const DEFAULT_FPR: f64 = 0.01;

const LN2: f64 = std::f64::consts::LN_2;

/// A row group's trigram-presence Bloom filter.
///
/// Sized per row group from its distinct-trigram count: a uniform size would
/// waste space on sparse groups and overflow dense ones (measured distinct
/// trigrams/row-group on real log text span ~1.3k–11k).
#[derive(Debug, Clone, PartialEq, Eq)]
struct RowGroupBloom {
    /// Bit storage; `m_bits == words.len() * 64`.
    words: Vec<u64>,
    /// Number of hash probes per element.
    k: u8,
}

impl RowGroupBloom {
    /// Allocate a Bloom filter sized for `n` distinct elements at `fpr`.
    ///
    /// `n == 0` yields a one-word, never-set filter so an empty row group is
    /// pruned for every needle (it can contain no substring).
    fn with_capacity(n: usize, fpr: f64) -> RowGroupBloom {
        if n == 0 {
            return RowGroupBloom {
                words: vec![0u64],
                k: 1,
            };
        }
        let m = (-(n as f64) * fpr.ln() / (LN2 * LN2)).ceil();
        // Round up to whole 64-bit words; clamp to a sane band.
        let words = ((m / 64.0).ceil() as usize).clamp(1, 1 << 16);
        let m_bits = (words * 64) as f64;
        let k = ((m_bits / n as f64) * LN2).round().clamp(1.0, 16.0) as u8;
        RowGroupBloom {
            words: vec![0u64; words],
            k,
        }
    }

    /// Heap bytes backing this filter's bit storage (for cache budgeting).
    fn heap_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }

    /// The `k` bit indices probed for `trigram` (double hashing: `h1 + i*h2`),
    /// written into `out` (length `k`) to avoid an allocation per probe.
    #[inline]
    fn probes(m_bits: u64, k: u8, trigram: [u8; 3], out: &mut [usize; 16]) {
        let base = (trigram[0] as u64) | ((trigram[1] as u64) << 8) | ((trigram[2] as u64) << 16);
        let h1 = splitmix64(base);
        // Force `h2` odd so the step doesn't collapse the k probes onto one bit.
        let h2 = splitmix64(base ^ 0xD6E8_FEB8_6659_FD93) | 1;
        for (i, slot) in out.iter_mut().enumerate().take(k as usize) {
            *slot = (h1.wrapping_add((i as u64).wrapping_mul(h2)) % m_bits) as usize;
        }
    }

    #[inline]
    fn insert(&mut self, trigram: [u8; 3]) {
        let m_bits = (self.words.len() * 64) as u64;
        let mut bits = [0usize; 16];
        Self::probes(m_bits, self.k, trigram, &mut bits);
        for &bit in bits.iter().take(self.k as usize) {
            self.words[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    #[inline]
    fn contains(&self, trigram: [u8; 3]) -> bool {
        let m_bits = (self.words.len() * 64) as u64;
        let mut bits = [0usize; 16];
        Self::probes(m_bits, self.k, trigram, &mut bits);
        bits.iter()
            .take(self.k as usize)
            .all(|&bit| self.words[bit / 64] & (1u64 << (bit % 64)) != 0)
    }
}

/// SplitMix64 finalizer — a fast, well-distributed deterministic mix. Stable
/// across builds (the sidecar is a persisted on-disk format).
#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-row-group keep mask for already-tokenized `trigrams`: `keep[i]` is `true`
/// unless row group `i`'s Bloom proves it lacks at least one trigram.
fn keep_mask_over(groups: &[RowGroupBloom], trigrams: &[[u8; 3]]) -> Vec<bool> {
    groups
        .iter()
        .map(|bloom| trigrams.iter().all(|&t| bloom.contains(t)))
        .collect()
}

/// A built trigram index for one column: its name plus one Bloom filter per
/// parquet row group, in row-group order. The write-side representation produced
/// by [`TrigramIndex::build`] and handed to [`serialize_sidecar`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrigramIndex {
    column: String,
    groups: Vec<RowGroupBloom>,
}

impl TrigramIndex {
    /// Number of indexed row groups.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn column(&self) -> &str {
        &self.column
    }

    /// Build an index over `column` across `batches` in row order, chunking at
    /// `ROW_GROUP_SIZE` to mirror the parquet writer's row-group boundaries.
    ///
    /// Returns `None` if `column` is absent or is not a UTF-8 string column (no
    /// index ⇒ no prune, which is safe).
    pub fn build(batches: &[RecordBatch], column: &str) -> Option<TrigramIndex> {
        if batches.is_empty() {
            return Some(TrigramIndex {
                column: column.to_string(),
                groups: Vec::new(),
            });
        }
        let mut groups: Vec<RowGroupBloom> = Vec::new();
        let mut current: TrigramSet = TrigramSet::default();
        let mut rows_in_group = 0usize;

        for batch in batches {
            let idx = batch.schema().index_of(column).ok()?;
            let col = batch.column(idx);
            let values = StringColumn::new(col.as_ref())?;
            for row in 0..batch.num_rows() {
                if let Some(v) = values.value(row) {
                    for t in trigram_windows(v.as_bytes()) {
                        current.insert(t);
                    }
                }
                rows_in_group += 1;
                if rows_in_group == ROW_GROUP_SIZE {
                    groups.push(current.into_bloom(DEFAULT_FPR));
                    current = TrigramSet::default();
                    rows_in_group = 0;
                }
            }
        }
        if rows_in_group > 0 {
            groups.push(current.into_bloom(DEFAULT_FPR));
        }
        Some(TrigramIndex {
            column: column.to_string(),
            groups,
        })
    }
}

/// One column directory entry in a parsed [`SidecarHeader`]: where that column's
/// per-row-group bloom payload lives in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDir {
    pub name: String,
    /// Absolute byte offset of the column's payload within the sidecar file.
    pub offset: u64,
    /// Byte length of the column's payload.
    pub len: u64,
}

/// The parsed sidecar header: everything before the per-column bloom payloads.
///
/// Cheap to read (a bounded prefix of the file) and self-describing — it carries
/// the indexed columns' offsets and the segment's key band, so a caller can scope
/// and slice without paging in any blooms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarHeader {
    /// Parquet row groups covered (shared by every column).
    pub rg_count: u32,
    /// The segment's ordering-key column name (empty if the namespace is keyless).
    pub key_column: String,
    /// Segment key band as raw UTF-8 bytes, present only for string key columns.
    /// `None` ⇒ unknown (non-string key, or missing column) ⇒ no key scoping.
    pub key_min: Option<Vec<u8>>,
    pub key_max: Option<Vec<u8>>,
    columns: Vec<ColumnDir>,
}

impl SidecarHeader {
    /// Directory entry for `name`, or `None` if the column is not indexed.
    pub fn column(&self, name: &str) -> Option<&ColumnDir> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Whether the segment's key band can satisfy the inclusive query range
    /// `[lo, hi]`. Returns `false` only on a **provable** non-overlap — meaning
    /// the segment is out of band and its blooms need not be read. Unknown bounds
    /// or an open range return `true` (must load).
    ///
    /// The caller passes conservatively-widened *inclusive* bounds (a strict `<`
    /// upper is treated as `<=`), so a borderline segment is loaded rather than
    /// wrongly skipped — and skipping is only ever an I/O optimization anyway,
    /// since an out-of-band segment is independently pruned by the parquet key
    /// statistics at scan time.
    pub fn key_band_overlaps(&self, lo: Option<&[u8]>, hi: Option<&[u8]>) -> bool {
        let (Some(kmin), Some(kmax)) = (self.key_min.as_deref(), self.key_max.as_deref()) else {
            return true;
        };
        if let Some(hi) = hi {
            if hi < kmin {
                return false;
            }
        }
        if let Some(lo) = lo {
            if lo > kmax {
                return false;
            }
        }
        true
    }
}

/// One column's parsed per-row-group blooms — the read-side counterpart of a
/// [`TrigramIndex`], shared via `Arc` and cached by the
/// [`crate::query::sidecar::SidecarManager`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnIndex {
    groups: Vec<RowGroupBloom>,
}

impl ColumnIndex {
    /// Number of indexed row groups.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Per-row-group keep mask for `needle`: `keep[i]` is `true` unless row group
    /// `i`'s Bloom proves it cannot contain the needle. `None` for a needle with
    /// no trigrams (`< MIN_TRIGRAM_LEN`), meaning "cannot prune — scan all".
    pub fn keep_mask(&self, needle: &str) -> Option<Vec<bool>> {
        Some(self.keep_mask_for(&needle_trigrams(needle)?))
    }

    /// Per-row-group keep mask for a needle's already-tokenized `trigrams` (see
    /// [`needle_trigrams`]). Splitting tokenization from masking lets a caller
    /// decompose the needle once and reuse it across many segments.
    pub fn keep_mask_for(&self, trigrams: &[[u8; 3]]) -> Vec<bool> {
        keep_mask_over(&self.groups, trigrams)
    }

    /// Heap bytes backing the blooms — what the sidecar cache charges to its
    /// byte budget for this entry.
    pub fn heap_bytes(&self) -> usize {
        self.groups
            .iter()
            .map(RowGroupBloom::heap_bytes)
            .sum::<usize>()
            + self.groups.len() * std::mem::size_of::<RowGroupBloom>()
    }
}

/// Serialize a segment's trigram index to the on-disk sidecar byte format.
///
/// `rg_count` is the shared parquet row-group count; `key_min`/`key_max` are the
/// segment's key band as raw bytes (string keys only); `columns` are the per-
/// column blooms (one entry today, `INDEXED_COLUMN`).
pub fn serialize_sidecar(
    rg_count: u32,
    key_column: &str,
    key_min: Option<&[u8]>,
    key_max: Option<&[u8]>,
    columns: &[TrigramIndex],
) -> Vec<u8> {
    let payloads: Vec<Vec<u8>> = columns
        .iter()
        .map(|c| serialize_column(&c.groups))
        .collect();

    // Directory size is fully determined by the column names and the optional
    // key band, so we can compute payload offsets before writing a byte.
    let mut header_len = 4 + 1 + 1 + 4 + 4; // magic + version + flags + header_len + rg_count
    header_len += 2 + key_column.len(); // key_column (u16 len + bytes)
    header_len += opt_bytes_len(key_min);
    header_len += opt_bytes_len(key_max);
    header_len += 2; // col_count u16
    for c in columns {
        header_len += 2 + c.column.len() + 8 + 8; // name + offset + len
    }

    let mut offset = header_len as u64;
    let mut offsets = Vec::with_capacity(columns.len());
    for p in &payloads {
        offsets.push(offset);
        offset += p.len() as u64;
    }

    let mut out = Vec::with_capacity(offset as usize);
    out.extend_from_slice(TGM_MAGIC);
    out.push(TGM_VERSION);
    out.push(0); // flags (reserved)
    out.extend_from_slice(&(header_len as u32).to_le_bytes());
    out.extend_from_slice(&rg_count.to_le_bytes());
    put_varbytes_u16(&mut out, key_column.as_bytes());
    put_opt_bytes_u32(&mut out, key_min);
    put_opt_bytes_u32(&mut out, key_max);
    out.extend_from_slice(&(columns.len() as u16).to_le_bytes());
    for (c, (off, p)) in columns.iter().zip(offsets.iter().zip(&payloads)) {
        put_varbytes_u16(&mut out, c.column.as_bytes());
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&(p.len() as u64).to_le_bytes());
    }
    debug_assert_eq!(out.len(), header_len, "header_len must match bytes written");
    for p in &payloads {
        out.extend_from_slice(p);
    }
    out
}

/// Serialize one column's per-row-group blooms (the payload body).
fn serialize_column(groups: &[RowGroupBloom]) -> Vec<u8> {
    let mut out = Vec::new();
    for g in groups {
        out.push(g.k);
        out.extend_from_slice(&(g.words.len() as u32).to_le_bytes());
        for w in &g.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

/// The declared header length of a sidecar prefix (bytes `[0, header_len)` hold
/// the directory), or `None` if the magic/version don't match. Lets a reader
/// decide whether a bounded prefix read already covered the whole header.
pub fn peek_header_len(bytes: &[u8]) -> Option<usize> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != TGM_MAGIC {
        return None;
    }
    if r.u8()? != TGM_VERSION {
        return None;
    }
    let _flags = r.u8()?;
    Some(r.u32()? as usize)
}

/// Parse the sidecar header from a buffer holding at least its first
/// `header_len` bytes. Returns `None` on bad magic/version or truncation — a
/// corrupt sidecar is treated as absent (scan unpruned).
pub fn parse_header(bytes: &[u8]) -> Option<SidecarHeader> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != TGM_MAGIC {
        return None;
    }
    if r.u8()? != TGM_VERSION {
        return None;
    }
    let _flags = r.u8()?;
    let header_len = r.u32()? as usize;
    if bytes.len() < header_len {
        return None;
    }
    let rg_count = r.u32()?;
    let key_column = r.string_u16()?;
    let key_min = r.opt_bytes_u32()?;
    let key_max = r.opt_bytes_u32()?;
    let col_count = r.u16()? as usize;
    let mut columns = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        let name = r.string_u16()?;
        let offset = r.u64()?;
        let len = r.u64()?;
        columns.push(ColumnDir { name, offset, len });
    }
    Some(SidecarHeader {
        rg_count,
        key_column,
        key_min,
        key_max,
        columns,
    })
}

/// Parse one column's payload (`rg_count` per-row-group blooms) from its slice.
/// Returns `None` on truncation or a row-group-count mismatch.
pub fn parse_column(bytes: &[u8], rg_count: u32) -> Option<ColumnIndex> {
    let mut r = ByteReader::new(bytes);
    let mut groups = Vec::with_capacity(rg_count as usize);
    for _ in 0..rg_count {
        let k = r.u8()?;
        let words_len = r.u32()? as usize;
        // A well-formed bloom always has `k >= 1` and at least one word (see
        // `RowGroupBloom::with_capacity`). Reject a corrupt zero here: an empty
        // word array makes `m_bits == 0`, so a later `probes`/`contains` would
        // panic on `% 0` instead of degrading to "absent" as the format promises.
        if k == 0 || words_len == 0 {
            return None;
        }
        // Guard against a corrupt length forcing a huge allocation before the
        // bounds check in `take` would fire word-by-word.
        if words_len.checked_mul(8)? > r.remaining() {
            return None;
        }
        let mut words = Vec::with_capacity(words_len);
        for _ in 0..words_len {
            words.push(r.u64()?);
        }
        groups.push(RowGroupBloom { words, k });
    }
    Some(ColumnIndex { groups })
}

/// Read a single column's index from a full in-memory sidecar buffer (header +
/// payloads). Convenience over [`parse_header`] + [`parse_column`] for callers
/// that already hold the whole file; the [`crate::query::sidecar::SidecarManager`]
/// uses positioned slice reads instead. `None` if the column is absent or the
/// buffer is malformed/truncated.
pub fn read_column_from_bytes(bytes: &[u8], column: &str) -> Option<ColumnIndex> {
    let header = parse_header(bytes)?;
    let dir = header.column(column)?;
    let start = dir.offset as usize;
    let end = start.checked_add(dir.len as usize)?;
    parse_column(bytes.get(start..end)?, header.rg_count)
}

/// The sidecar path for a parquet segment: `<segment>.tgm`.
pub fn sidecar_path(parquet_path: &Path) -> PathBuf {
    let mut s = parquet_path.as_os_str().to_os_string();
    s.push(".tgm");
    PathBuf::from(s)
}

/// Build a trigram index over each of `data_columns` across `batches` and write
/// them as the single multi-column sidecar for `parquet_path`. `key_column`
/// (when string-typed in `batches`) supplies the segment's key band, stored in
/// the header for query-time scoping.
///
/// Columns that are absent or not UTF-8 string-typed are skipped (each yields no
/// index, which is safe). Returns `Ok(true)` when a non-empty sidecar was
/// written, `Ok(false)` when nothing was indexable (no such columns, or zero row
/// groups). All surviving column indexes share the same row-group count — they
/// chunk the same `batches` at the same `ROW_GROUP_SIZE` stride.
///
/// The index is optional — a write failure is non-fatal to the caller (the
/// segment just scans unpruned), so callers log rather than propagate.
pub fn write_sidecar(
    parquet_path: &Path,
    batches: &[RecordBatch],
    data_columns: &[&str],
    key_column: Option<&str>,
) -> std::io::Result<bool> {
    let indexes: Vec<TrigramIndex> = data_columns
        .iter()
        .filter_map(|col| {
            let index = TrigramIndex::build(batches, col)?;
            (!index.is_empty()).then_some(index)
        })
        .collect();
    if indexes.is_empty() {
        return Ok(false);
    }
    debug_assert!(
        indexes.iter().all(|i| i.len() == indexes[0].len()),
        "all indexed columns chunk the same batches into the same row-group count"
    );
    let rg_count = indexes[0].len() as u32;
    let (key_min, key_max) = key_column
        .map(|kc| string_key_bounds(batches, kc))
        .unwrap_or((None, None));
    let bytes = serialize_sidecar(
        rg_count,
        key_column.unwrap_or(""),
        key_min.as_deref(),
        key_max.as_deref(),
        &indexes,
    );
    // Write to a temp file and rename into place so a reader never observes a
    // half-written sidecar. The compaction path writes before the segment is
    // query-visible, but the background backfill rebuilds sidecars for segments
    // that are ALREADY visible; a partial read there would parse as absent (scan
    // unpruned) — safe, but the rename makes the transition atomic regardless.
    let final_path = sidecar_path(parquet_path);
    let mut tmp_os = final_path.as_os_str().to_os_string();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);
    std::fs::write(&tmp_path, &bytes)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(true)
}

/// The min/max of `key_column` across `batches` as raw UTF-8 bytes, or
/// `(None, None)` if the column is absent or is not a string column. Byte order
/// matches both the parquet string statistics and the lexicographic order used by
/// the query-time key-range bounds, so the comparison in
/// [`SidecarHeader::key_band_overlaps`] is sound.
fn string_key_bounds(
    batches: &[RecordBatch],
    key_column: &str,
) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut lo: Option<Vec<u8>> = None;
    let mut hi: Option<Vec<u8>> = None;
    for batch in batches {
        let Ok(idx) = batch.schema().index_of(key_column) else {
            return (None, None);
        };
        let Some(values) = StringColumn::new(batch.column(idx).as_ref()) else {
            return (None, None);
        };
        for row in 0..batch.num_rows() {
            if let Some(v) = values.value(row) {
                let vb = v.as_bytes();
                if lo.as_deref().is_none_or(|x| vb < x) {
                    lo = Some(vb.to_vec());
                }
                if hi.as_deref().is_none_or(|x| vb > x) {
                    hi = Some(vb.to_vec());
                }
            }
        }
    }
    (lo, hi)
}

/// The distinct byte 3-grams of `needle`, or `None` when it is shorter than
/// [`MIN_TRIGRAM_LEN`] (no trigrams ⇒ cannot prune, scan all).
///
/// Decompose the needle once per query with this, then feed the result to
/// [`ColumnIndex::keep_mask_for`] for each segment — re-tokenizing per segment
/// is wasted work when a query spans many of them.
pub fn needle_trigrams(needle: &str) -> Option<Vec<[u8; 3]>> {
    if needle.len() < MIN_TRIGRAM_LEN {
        return None;
    }
    Some(distinct_trigrams(needle.as_bytes()))
}

/// Distinct 3-byte windows of `bytes`.
fn distinct_trigrams(bytes: &[u8]) -> Vec<[u8; 3]> {
    let mut set = TrigramSet::default();
    for t in trigram_windows(bytes) {
        set.insert(t);
    }
    set.into_vec()
}

/// Iterator over the sliding 3-byte windows of `bytes`.
fn trigram_windows(bytes: &[u8]) -> impl Iterator<Item = [u8; 3]> + '_ {
    bytes.windows(3).map(|w| [w[0], w[1], w[2]])
}

/// Number of `u64` words in the trigram bitset: one bit per possible 24-bit
/// trigram (256^3 = 16,777,216 bits = exactly 2 MiB).
const TRIGRAM_BITSET_WORDS: usize = (1 << 24) / 64;

/// Dedup set over the 24-bit trigram universe, backed by a fixed 2 MiB bitset.
///
/// A log row group yields millions of trigram windows that are mostly
/// duplicates; a `HashSet<u32>` pays a hash + probe (and periodic rehash growth)
/// on every one. The bitset makes `insert` a single shift-and-or against a
/// direct bit index, with no hashing and no allocation after construction, and
/// turns the distinct count into a `popcount`. Memory is bounded at 2 MiB
/// regardless of cardinality (vs. a `HashSet` that grows toward the 16.7M-entry
/// universe for high-entropy text).
struct TrigramSet {
    // Boxed so the 2 MiB array lives on the heap, not the stack.
    words: Box<[u64]>,
}

impl Default for TrigramSet {
    fn default() -> Self {
        TrigramSet {
            words: vec![0u64; TRIGRAM_BITSET_WORDS].into_boxed_slice(),
        }
    }
}

impl TrigramSet {
    #[inline]
    fn insert(&mut self, t: [u8; 3]) {
        let packed = (t[0] as usize) | ((t[1] as usize) << 8) | ((t[2] as usize) << 16);
        self.words[packed >> 6] |= 1u64 << (packed & 63);
    }

    /// Number of distinct trigrams inserted.
    fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Visit each set trigram once, unpacking its bit index back to bytes.
    fn for_each(&self, mut f: impl FnMut([u8; 3])) {
        for (word_index, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let packed = (word_index << 6) | bits.trailing_zeros() as usize;
                bits &= bits - 1; // clear the lowest set bit
                f([
                    (packed & 0xFF) as u8,
                    ((packed >> 8) & 0xFF) as u8,
                    ((packed >> 16) & 0xFF) as u8,
                ]);
            }
        }
    }

    fn into_vec(self) -> Vec<[u8; 3]> {
        let mut out = Vec::with_capacity(self.len());
        self.for_each(|t| out.push(t));
        out
    }

    fn into_bloom(self, fpr: f64) -> RowGroupBloom {
        let mut bloom = RowGroupBloom::with_capacity(self.len(), fpr);
        self.for_each(|t| bloom.insert(t));
        bloom
    }
}

/// A read-only view over a string column as either `Utf8` or `LargeUtf8`.
enum StringColumn<'a> {
    Utf8(&'a StringArray),
    Large(&'a LargeStringArray),
}

impl<'a> StringColumn<'a> {
    fn new(col: &'a dyn Array) -> Option<StringColumn<'a>> {
        if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
            Some(StringColumn::Utf8(a))
        } else {
            col.as_any()
                .downcast_ref::<LargeStringArray>()
                .map(StringColumn::Large)
        }
    }

    #[inline]
    fn value(&self, row: usize) -> Option<&str> {
        match self {
            StringColumn::Utf8(a) => (!a.is_null(row)).then(|| a.value(row)),
            StringColumn::Large(a) => (!a.is_null(row)).then(|| a.value(row)),
        }
    }
}

/// Append `bytes` framed by a `u16` little-endian length prefix.
fn put_varbytes_u16(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Append an optional byte string: a `u8` present flag, then (if present) a
/// `u32` length prefix and the bytes.
fn put_opt_bytes_u32(out: &mut Vec<u8>, bytes: Option<&[u8]>) {
    match bytes {
        Some(b) => {
            out.push(1);
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
        }
        None => out.push(0),
    }
}

/// Serialized size of an optional byte string written by [`put_opt_bytes_u32`].
fn opt_bytes_len(bytes: Option<&[u8]>) -> usize {
    1 + bytes.map_or(0, |b| 4 + b.len())
}

/// Minimal little-endian byte reader for sidecar parsing.
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> ByteReader<'a> {
        ByteReader { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    /// A `u16`-length-framed UTF-8 string.
    fn string_u16(&mut self) -> Option<String> {
        let n = self.u16()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }

    /// An optional `u32`-length-framed byte string (`u8` present flag first).
    /// `Some(None)` = absent, `Some(Some(_))` = present, `None` = corrupt.
    fn opt_bytes_u32(&mut self) -> Option<Option<Vec<u8>>> {
        match self.u8()? {
            0 => Some(None),
            1 => {
                let n = self.u32()? as usize;
                Some(Some(self.take(n)?.to_vec()))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

    /// A `seq`/`key`/`data` log batch with a string `key` so the sidecar carries
    /// a key band. `keys` and `values` must be the same length.
    fn log_batch(keys: Vec<&str>, values: Vec<Option<&str>>) -> RecordBatch {
        let n = values.len() as i64;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(1..=n)),
                Arc::new(StringArray::from(keys)),
                Arc::new(StringArray::from(values)),
            ],
        )
        .unwrap()
    }

    /// A `seq`/`data` batch (no key column) for tests that don't exercise the
    /// key band.
    fn data_batch(values: Vec<Option<&str>>) -> RecordBatch {
        let n = values.len() as i64;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("data", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(1..=n)),
                Arc::new(StringArray::from(values)),
            ],
        )
        .unwrap()
    }

    /// Serialize a single `data`-column index the way `write_sidecar` would, with
    /// the given key band, and return the bytes.
    fn serialize_data(idx: &TrigramIndex, key_min: Option<&str>, key_max: Option<&str>) -> Vec<u8> {
        serialize_sidecar(
            idx.len() as u32,
            "key",
            key_min.map(str::as_bytes),
            key_max.map(str::as_bytes),
            std::slice::from_ref(idx),
        )
    }

    #[test]
    fn trigram_windows_basic() {
        let t: Vec<[u8; 3]> = trigram_windows(b"abcd").collect();
        assert_eq!(t, vec![*b"abc", *b"bcd"]);
        // Shorter than 3 bytes yields nothing.
        assert!(trigram_windows(b"ab").next().is_none());
    }

    #[test]
    fn bloom_has_no_false_negatives() {
        // Every inserted trigram must report present (Bloom can only false-POSITIVE).
        let mut set = TrigramSet::default();
        let text = b"the quick brown fox jumps over the lazy dog 0123456789";
        for t in trigram_windows(text) {
            set.insert(t);
        }
        let tris: Vec<[u8; 3]> = set.into_vec();
        let mut bloom = RowGroupBloom::with_capacity(tris.len(), 0.01);
        for &t in &tris {
            bloom.insert(t);
        }
        for &t in &tris {
            assert!(bloom.contains(t), "false negative for {t:?}");
        }
    }

    #[test]
    fn trigram_set_dedups_and_round_trips_bit_positions() {
        // Values chosen to land in distinct bitset words and exercise the
        // low/high byte unpacking: packed indices 0, 63 (word boundary), 64
        // (next word), the maximum 2^24-1, and an arbitrary mid value.
        let inputs = [
            [0u8, 0, 0],
            [63, 0, 0],
            [64, 0, 0],
            [255, 255, 255],
            [1, 2, 3],
        ];
        let mut set = TrigramSet::default();
        for &t in &inputs {
            set.insert(t);
            set.insert(t); // a repeated trigram must not change the distinct set
        }
        assert_eq!(set.len(), inputs.len());
        let mut got = set.into_vec();
        got.sort();
        let mut want = inputs.to_vec();
        want.sort();
        assert_eq!(got, want, "every inserted trigram must round-trip exactly");
    }

    #[test]
    fn keep_mask_no_false_negative_per_row_group() {
        // Row group 0 contains the needle; row group 1 does not. The needle's
        // group must be kept; short needles return None (scan all).
        let rg0: Vec<Option<&str>> = (0..ROW_GROUP_SIZE)
            .map(|i| {
                if i == 7 {
                    Some("Bootstrap completed for TPU v6e-4")
                } else {
                    Some("idle heartbeat ok")
                }
            })
            .collect();
        let rg1: Vec<Option<&str>> = (0..100).map(|_| Some("unrelated chatter line")).collect();
        let b0 = data_batch(rg0);
        let b1 = data_batch(rg1);
        let idx = TrigramIndex::build(&[b0, b1], "data").unwrap();
        assert_eq!(idx.len(), 2, "two row groups (16384 + 100 rows)");

        // Round-trip through the on-disk format to a read-side ColumnIndex.
        let bytes = serialize_data(&idx, None, None);
        let col = read_column_from_bytes(&bytes, "data").unwrap();

        let mask = col.keep_mask("Bootstrap completed for TPU").unwrap();
        assert_eq!(mask.len(), 2);
        assert!(mask[0], "row group with the needle must be kept");
        assert!(!mask[1], "row group without the needle is pruned");

        // Sub-trigram needle cannot prune.
        assert!(col.keep_mask("ab").is_none());
    }

    #[test]
    fn build_aligns_row_group_boundaries() {
        // 16384*2 + 5 rows -> 3 row groups, matching ArrowWriter's stride.
        let n = ROW_GROUP_SIZE * 2 + 5;
        let vals: Vec<Option<&str>> = (0..n).map(|_| Some("payload line content")).collect();
        let idx = TrigramIndex::build(&[data_batch(vals)], "data").unwrap();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn serde_round_trips() {
        let vals: Vec<Option<&str>> = vec![Some("alpha beta gamma"), None, Some("delta epsilon")];
        let idx = TrigramIndex::build(&[data_batch(vals)], "data").unwrap();
        let bytes = serialize_data(&idx, Some("/a/b"), Some("/a/c"));

        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.rg_count, 1);
        assert_eq!(header.key_column, "key");
        assert_eq!(header.key_min.as_deref(), Some(b"/a/b".as_slice()));
        assert_eq!(header.key_max.as_deref(), Some(b"/a/c".as_slice()));

        let col = read_column_from_bytes(&bytes, "data").unwrap();
        assert_eq!(col.len(), 1);
        // A real needle present in the data keeps the (single) row group.
        assert_eq!(col.keep_mask("beta gamma").unwrap(), vec![true]);
        // An unindexed column is absent from the directory.
        assert!(read_column_from_bytes(&bytes, "nope").is_none());
    }

    #[test]
    fn write_sidecar_records_string_key_band() {
        // The key band is the min/max of the `key` column, regardless of row order.
        let batch = log_batch(
            vec!["/m/z", "/m/a", "/m/q"],
            vec![
                Some("one two three"),
                Some("four five six"),
                Some("seven eight"),
            ],
        );
        let idx = TrigramIndex::build(std::slice::from_ref(&batch), "data").unwrap();
        let (lo, hi) = string_key_bounds(std::slice::from_ref(&batch), "key");
        assert_eq!(lo.as_deref(), Some(b"/m/a".as_slice()));
        assert_eq!(hi.as_deref(), Some(b"/m/z".as_slice()));
        let bytes = serialize_data(
            &idx,
            lo.as_deref().map(|_| "/m/a"),
            hi.as_deref().map(|_| "/m/z"),
        );
        let header = parse_header(&bytes).unwrap();
        // Out-of-band query ranges are provably non-overlapping; in-band overlaps.
        assert!(!header.key_band_overlaps(Some(b"/n/a"), Some(b"/n/z")));
        assert!(!header.key_band_overlaps(Some(b"/a"), Some(b"/m/")));
        assert!(header.key_band_overlaps(Some(b"/m/a"), Some(b"/m/a")));
        assert!(header.key_band_overlaps(Some(b"/m/b"), None));
    }

    #[test]
    fn non_string_key_has_no_band() {
        // An Int64 key yields no string band -> no scoping, always "overlaps".
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(1..=2)),
                Arc::new(Int64Array::from(vec![10_i64, 20])),
                Arc::new(StringArray::from(vec!["alpha line", "beta line"])),
            ],
        )
        .unwrap();
        let (lo, hi) = string_key_bounds(std::slice::from_ref(&batch), "key");
        assert!(lo.is_none() && hi.is_none());
        let idx = TrigramIndex::build(std::slice::from_ref(&batch), "data").unwrap();
        let bytes = serialize_sidecar(
            idx.len() as u32,
            "key",
            None,
            None,
            std::slice::from_ref(&idx),
        );
        let header = parse_header(&bytes).unwrap();
        assert!(header.key_min.is_none() && header.key_max.is_none());
        assert!(header.key_band_overlaps(Some(b"anything"), Some(b"zzz")));
    }

    #[test]
    fn header_len_lets_a_short_prefix_be_detected() {
        let idx =
            TrigramIndex::build(&[data_batch(vec![Some("alpha beta gamma")])], "data").unwrap();
        let bytes = serialize_data(&idx, Some("/k"), Some("/k"));
        let hlen = peek_header_len(&bytes).unwrap();
        assert!(hlen < bytes.len(), "payload follows the header");
        // The header parses from exactly its prefix...
        assert!(parse_header(&bytes[..hlen]).is_some());
        // ...but not from a prefix shorter than header_len.
        assert!(parse_header(&bytes[..hlen - 1]).is_none());
    }

    #[test]
    fn parse_rejects_garbage() {
        assert!(peek_header_len(b"nope").is_none());
        assert!(parse_header(b"nope").is_none());
        assert!(parse_header(&[]).is_none());
        // Right magic/version, truncated body.
        let mut b = TGM_MAGIC.to_vec();
        b.push(TGM_VERSION);
        assert!(parse_header(&b).is_none());
        // A wrong version is rejected (forward-compat: unknown versions = absent).
        let idx = TrigramIndex::build(&[data_batch(vec![Some("alpha beta")])], "data").unwrap();
        let mut bytes = serialize_data(&idx, None, None);
        bytes[4] = TGM_VERSION + 1;
        assert!(peek_header_len(&bytes).is_none());
        assert!(parse_header(&bytes).is_none());
    }

    #[test]
    fn parse_column_rejects_degenerate_blooms() {
        // A row group is `k u8 | words_len u32 | words[words_len] u64`. A corrupt
        // `words_len == 0` would make `m_bits == 0` and panic on `% 0` at query
        // time; it must be rejected (treated as absent), never panic. Likewise
        // `k == 0` (a real bloom always probes at least one bit).
        assert!(
            parse_column(&[1u8, 0, 0, 0, 0], 1).is_none(),
            "words_len == 0"
        );
        assert!(
            parse_column(&[0u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1).is_none(),
            "k == 0"
        );
        // A valid single-word group parses.
        assert!(
            parse_column(&[1u8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1).is_some(),
            "k=1, one zero word is valid"
        );
    }

    #[test]
    fn missing_or_non_string_column_yields_none() {
        let b = data_batch(vec![Some("x")]);
        assert!(TrigramIndex::build(std::slice::from_ref(&b), "nonexistent").is_none());
        // `seq` is Int64, not a string column.
        assert!(TrigramIndex::build(&[b], "seq").is_none());
    }

    #[test]
    fn sidecar_path_appends_tgm() {
        let p = sidecar_path(Path::new("/x/seg_L1_0000000000000000001.parquet"));
        assert_eq!(
            p.to_str().unwrap(),
            "/x/seg_L1_0000000000000000001.parquet.tgm"
        );
    }
}
