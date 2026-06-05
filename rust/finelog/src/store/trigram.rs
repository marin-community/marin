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
//! ## Why a sidecar (not the parquet footer)
//!
//! The footer is read in full on every file open. A per-row-group structure
//! embedded there would tax every query — including the hot `key = … ORDER BY
//! seq DESC LIMIT n` tail that never calls `contains()`. The sidecar is loaded
//! lazily, only for substring queries. It is a pure, optional, derivable function
//! of the column: a missing or stale sidecar is never *wrong*, only unpruned. See
//! `.agents/projects/2026-06-05_finelog_trigram_index.md`.
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
/// `contains(key, …)` workload appears.
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

/// A segment's trigram index: the indexed column name plus one Bloom filter per
/// parquet row group, in row-group order.
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

    /// Per-row-group keep mask for `needle`: `keep[i]` is `true` unless row group
    /// `i`'s Bloom proves it cannot contain the needle.
    ///
    /// Returns `None` for a needle with no trigrams (`< MIN_TRIGRAM_LEN`), meaning
    /// "cannot prune — scan all". A kept group may still not match (Bloom false
    /// positive or trigrams present in different rows); the caller re-checks
    /// `contains()` exactly. A truly-matching group is never dropped.
    pub fn keep_mask(&self, needle: &str) -> Option<Vec<bool>> {
        let trigrams = distinct_trigrams(needle.as_bytes());
        if trigrams.is_empty() {
            return None;
        }
        Some(
            self.groups
                .iter()
                .map(|bloom| trigrams.iter().all(|&t| bloom.contains(t)))
                .collect(),
        )
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

    /// Serialize to the on-disk sidecar byte format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(TGM_MAGIC);
        out.push(TGM_VERSION);
        let name = self.column.as_bytes();
        out.push(name.len() as u8);
        out.extend_from_slice(name);
        out.extend_from_slice(&(self.groups.len() as u32).to_le_bytes());
        for g in &self.groups {
            out.push(g.k);
            out.extend_from_slice(&(g.words.len() as u32).to_le_bytes());
            for w in &g.words {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    /// Parse a sidecar produced by [`TrigramIndex::to_bytes`]. Returns `None` on
    /// any malformed input (bad magic/version/truncation) — a corrupt sidecar is
    /// treated as absent, i.e. unpruned.
    pub fn from_bytes(bytes: &[u8]) -> Option<TrigramIndex> {
        let mut r = ByteReader::new(bytes);
        if r.take(4)? != TGM_MAGIC {
            return None;
        }
        if r.u8()? != TGM_VERSION {
            return None;
        }
        let name_len = r.u8()? as usize;
        let column = String::from_utf8(r.take(name_len)?.to_vec()).ok()?;
        let rg_count = r.u32()? as usize;
        let mut groups = Vec::with_capacity(rg_count);
        for _ in 0..rg_count {
            let k = r.u8()?;
            let words_len = r.u32()? as usize;
            let mut words = Vec::with_capacity(words_len);
            for _ in 0..words_len {
                words.push(r.u64()?);
            }
            groups.push(RowGroupBloom { words, k });
        }
        Some(TrigramIndex { column, groups })
    }
}

/// The sidecar path for a parquet segment: `<segment>.tgm`.
pub fn sidecar_path(parquet_path: &Path) -> PathBuf {
    let mut s = parquet_path.as_os_str().to_os_string();
    s.push(".tgm");
    PathBuf::from(s)
}

/// Build the trigram index over `column` across `batches` and write it as the
/// sidecar for `parquet_path`. Returns `Ok(true)` when a non-empty sidecar was
/// written, `Ok(false)` when there was nothing to index (no such string column,
/// or zero row groups).
///
/// The index is optional — a write failure is non-fatal to the caller (the
/// segment just scans unpruned), so callers log rather than propagate.
pub fn write_sidecar(
    parquet_path: &Path,
    batches: &[RecordBatch],
    column: &str,
) -> std::io::Result<bool> {
    let Some(index) = TrigramIndex::build(batches, column) else {
        return Ok(false);
    };
    if index.is_empty() {
        return Ok(false);
    }
    std::fs::write(sidecar_path(parquet_path), index.to_bytes())?;
    Ok(true)
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

/// A small dedup set over trigrams. Backed by a `HashSet` keyed on the packed
/// 24-bit trigram value.
#[derive(Default)]
struct TrigramSet {
    seen: std::collections::HashSet<u32>,
}

impl TrigramSet {
    #[inline]
    fn insert(&mut self, t: [u8; 3]) {
        let packed = (t[0] as u32) | ((t[1] as u32) << 8) | ((t[2] as u32) << 16);
        self.seen.insert(packed);
    }

    fn into_vec(self) -> Vec<[u8; 3]> {
        self.seen
            .into_iter()
            .map(|p| {
                [
                    (p & 0xFF) as u8,
                    ((p >> 8) & 0xFF) as u8,
                    ((p >> 16) & 0xFF) as u8,
                ]
            })
            .collect()
    }

    fn into_bloom(self, fpr: f64) -> RowGroupBloom {
        let n = self.seen.len();
        let mut bloom = RowGroupBloom::with_capacity(n, fpr);
        for p in &self.seen {
            let t = [
                (*p & 0xFF) as u8,
                ((*p >> 8) & 0xFF) as u8,
                ((*p >> 16) & 0xFF) as u8,
            ];
            bloom.insert(t);
        }
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

/// Minimal little-endian byte reader for sidecar parsing.
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> ByteReader<'a> {
        ByteReader { bytes, pos: 0 }
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

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

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

    #[test]
    fn trigram_windows_basic() {
        let t: Vec<[u8; 3]> = trigram_windows(b"abcd").collect();
        assert_eq!(t, vec![[b'a', b'b', b'c'], [b'b', b'c', b'd']]);
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

        let mask = idx.keep_mask("Bootstrap completed for TPU").unwrap();
        assert_eq!(mask.len(), 2);
        assert!(mask[0], "row group with the needle must be kept");
        assert!(!mask[1], "row group without the needle is pruned");

        // Sub-trigram needle cannot prune.
        assert!(idx.keep_mask("ab").is_none());
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
        let bytes = idx.to_bytes();
        let back = TrigramIndex::from_bytes(&bytes).unwrap();
        assert_eq!(idx, back);
        assert_eq!(back.column(), "data");
        // A real needle present in the data keeps the (single) row group.
        assert_eq!(back.keep_mask("beta gamma").unwrap(), vec![true]);
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(TrigramIndex::from_bytes(b"nope").is_none());
        assert!(TrigramIndex::from_bytes(&[]).is_none());
        // Right magic, truncated body.
        let mut b = TGM_MAGIC.to_vec();
        b.push(TGM_VERSION);
        assert!(TrigramIndex::from_bytes(&b).is_none());
    }

    #[test]
    fn missing_or_non_string_column_yields_none() {
        let b = data_batch(vec![Some("x")]);
        assert!(TrigramIndex::build(&[b.clone()], "nonexistent").is_none());
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
