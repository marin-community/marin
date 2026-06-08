//! Leaf vocabulary for the store layer.
//!
//! Metadata types (`SegmentRow` / `SegmentLocation` / `LocalSegment`) plus the
//! segment-filename helpers (`seg_filename` / `parse_seg_filename`) used by the
//! flush path and the rebuild-from-disk catalog scan.

use std::sync::OnceLock;

use regex::Regex;

/// Where a segment's bytes currently live. Wire/catalog strings: LOCAL/REMOTE/BOTH.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentLocation {
    Local,
    Remote,
    Both,
}

impl SegmentLocation {
    pub fn as_str(self) -> &'static str {
        match self {
            SegmentLocation::Local => "LOCAL",
            SegmentLocation::Remote => "REMOTE",
            SegmentLocation::Both => "BOTH",
        }
    }

    pub fn parse_str(s: &str) -> Option<SegmentLocation> {
        match s {
            "LOCAL" => Some(SegmentLocation::Local),
            "REMOTE" => Some(SegmentLocation::Remote),
            "BOTH" => Some(SegmentLocation::Both),
            _ => None,
        }
    }
}

/// One persisted row in the `segments` catalog table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentRow {
    pub namespace: String,
    pub path: String,
    pub level: i32,
    pub min_seq: i64,
    pub max_seq: i64,
    pub row_count: i64,
    pub byte_size: i64,
    pub created_at_ms: i64,
    pub min_key_value: Option<String>,
    pub max_key_value: Option<String>,
    pub location: SegmentLocation,
}

/// An in-memory pointer to a segment file the namespace owns.
///
/// The deque only ever holds `Local` or `Both` entries; eviction flips to
/// `Remote` and removes the entry. Segments are created `Local` at flush time.
/// `min_key_value` / `max_key_value` are the Int64 key bounds when the
/// namespace's key column is an Int64/timestamp column carrying parquet
/// statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSegment {
    pub path: String,
    pub size_bytes: i64,
    pub level: i32,
    pub min_seq: i64,
    pub max_seq: i64,
    pub row_count: i64,
    pub created_at_ms: i64,
    pub min_key_value: Option<i64>,
    pub max_key_value: Option<i64>,
    pub location: SegmentLocation,
}

/// Aggregate counters for one namespace's persisted segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamespaceStats {
    pub row_count: i64,
    pub byte_size: i64,
    pub min_seq: i64,
    pub max_seq: i64,
    pub segment_count: i32,
}

impl NamespaceStats {
    pub fn empty() -> NamespaceStats {
        NamespaceStats {
            row_count: 0,
            byte_size: 0,
            min_seq: 0,
            max_seq: 0,
            segment_count: 0,
        }
    }
}

/// Aggregate in-RAM accounting across live namespaces for the diagnostics line:
/// `namespaces` / `ram_bytes` / `chunks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemorySummary {
    pub namespaces: usize,
    pub ram_bytes: i64,
    pub chunks: usize,
}

// ---------------------------------------------------------------------------
// Segment filename grammar.
// ---------------------------------------------------------------------------

/// Filename for a segment at `level` whose smallest seq is `min_seq`.
pub fn seg_filename(level: i32, min_seq: i64) -> String {
    format!("seg_L{level}_{min_seq:019}.parquet")
}

fn seg_filename_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^seg_L(?P<level>\d+)_(?P<seq>\d+)\.parquet$")
            .expect("seg filename regex compiles")
    })
}

/// Recover `(level, min_seq)` from a segment filename, or `None`.
pub fn parse_seg_filename(name: &str) -> Option<(i32, i64)> {
    let caps = seg_filename_re().captures(name)?;
    let level = caps.name("level")?.as_str().parse().ok()?;
    let seq = caps.name("seq")?.as_str().parse().ok()?;
    Some((level, seq))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_round_trip() {
        for loc in [
            SegmentLocation::Local,
            SegmentLocation::Remote,
            SegmentLocation::Both,
        ] {
            assert_eq!(SegmentLocation::parse_str(loc.as_str()), Some(loc));
        }
        assert_eq!(SegmentLocation::parse_str("bogus"), None);
    }

    #[test]
    fn seg_filename_round_trip() {
        let name = seg_filename(0, 42);
        assert_eq!(name, "seg_L0_0000000000000000042.parquet");
        assert_eq!(parse_seg_filename(&name), Some((0, 42)));
        assert_eq!(parse_seg_filename("not_a_segment.parquet"), None);
    }
}
