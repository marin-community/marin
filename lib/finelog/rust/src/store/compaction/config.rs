//! Compaction tuning knobs + the pending-merge descriptor.
//!
//! Pure data: no arrow / parquet / object_store. The planner (`planner.rs`) reads these to
//! decide *which* segments merge into *what* file; the executor (`executor.rs`)
//! carries out the resulting `CompactionJob`.

use std::time::Duration;

use crate::store::types::SegmentRow;

const MIB: i64 = 1024 * 1024;
const GIB: i64 = 1024 * MIB;

/// Tuning knobs for the leveled compaction policy.
///
/// `level_targets[n]` is the summed byte size at which the longest contiguous
/// run of L_n segments is promoted to L_{n+1}. The terminal level is
/// `level_targets.len()`; segments at that tier are never re-compacted (and are
/// the only tier eligible for eviction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionConfig {
    /// L0->L1 at 64 MiB, L1->L2 at 256 MiB, L2->L3 (terminal) at 256 MiB.
    pub level_targets: Vec<i64>,
    /// Per-level fanout cap. Promotes a non-terminal level once its contiguous
    /// run reaches this many segments, even if the byte target isn't met.
    pub max_segments_per_level: usize,
    /// Ceiling on the summed UNCOMPRESSED size of a merge job's inputs.
    ///
    /// `level_targets` bound a job's COMPRESSED input size, but the merge decodes
    /// every input into Arrow arrays and the k-way merge builds the whole merged
    /// copy before either is freed. Peak RSS is therefore about twice this
    /// ceiling, plus the merge's sort-key row encodings — not the compressed
    /// size, which log text undershoots by 10-20x. This is the knob that bounds
    /// merge memory.
    ///
    /// A run whose first segment alone exceeds the ceiling yields a single-input
    /// job, which the executor promotes by rename instead of merging, so an
    /// oversized segment never wedges its level.
    pub max_merge_uncompressed_bytes: i64,
    /// Whole-namespace segment cap (eviction trigger).
    pub max_segments_per_namespace: usize,
    /// Whole-namespace byte cap on locally-retained segments (eviction trigger).
    /// Once a namespace's local L>=1 segments exceed this, the oldest already-
    /// offloaded (BOTH) segments have their local copies unlinked; the remote
    /// archive is kept. At current log volume 15 GiB is ~10 days of backwards
    /// search.
    pub max_bytes_per_namespace: i64,
    /// Maintenance-loop cadence.
    pub check_interval: Duration,
}

impl Default for CompactionConfig {
    fn default() -> CompactionConfig {
        CompactionConfig {
            level_targets: vec![64 * MIB, 256 * MIB, 256 * MIB],
            max_segments_per_level: 32,
            max_merge_uncompressed_bytes: 4 * GIB,
            max_segments_per_namespace: 1000,
            max_bytes_per_namespace: 15 * 1024 * 1024 * 1024,
            check_interval: Duration::from_secs(30),
        }
    }
}

impl CompactionConfig {
    /// Segments at this level are never re-compacted.
    pub fn terminal_level(&self) -> i32 {
        self.level_targets.len() as i32
    }
}

/// One pending merge: `inputs.len()` segments -> one `output_level` segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    pub inputs: Vec<SegmentRow>,
    pub output_level: i32,
    pub output_min_seq: i64,
    pub output_max_seq: i64,
}
