//! Pure leveled-compaction policy.
//!
//! No I/O — this module decides *which* segments to merge into *what* output
//! level; the executor reads/writes the actual parquet (`executor.rs`).
//!
//! Levels are time-ordered: every flush emits an L0 segment. A level is promoted
//! L_n -> L_{n+1} when the longest contiguous run of L_n segments hits the byte
//! target for that tier OR when its length hits `max_segments_per_level`. The
//! count trigger bounds per-read fanout for slow / bursty namespaces whose L0
//! flushes are small. The terminal level (`level_targets.len()`) never
//! re-compacts.

use crate::store::compaction::config::{CompactionConfig, CompactionJob};
use crate::store::schema::{Schema, IMPLICIT_SEQ_COLUMN};
use crate::store::types::SegmentRow;

/// Sort keys for both compaction's merge order: `key_column` first (so range
/// scans on it prune row groups), then the implicit `seq`.
pub fn compaction_sort_keys(schema: &Schema) -> Vec<String> {
    if !schema.key_column.is_empty() {
        vec![schema.key_column.clone(), IMPLICIT_SEQ_COLUMN.to_string()]
    } else {
        vec![IMPLICIT_SEQ_COLUMN.to_string()]
    }
}

/// Return the next merge job, or `None` if nothing is due.
///
/// Walks tiers from L0 upward and returns the first promotable run, capping the
/// selected prefix at the level's compressed byte target.
///
/// The job's MEMORY cost is not decided here: a segment's decoded Arrow size
/// cannot be read off its footer (see `max_merge_arrow_bytes`), so the executor
/// measures it while reading and merges only the prefix of `inputs` that fits.
/// A job this planner returns is therefore an upper bound on what one tick
/// merges, not a promise.
pub fn plan(config: &CompactionConfig, segments: &[SegmentRow]) -> Option<CompactionJob> {
    for (n, &target) in config.level_targets.iter().enumerate() {
        let level = n as i32;
        let mut at_level: Vec<&SegmentRow> = segments.iter().filter(|s| s.level == level).collect();
        if at_level.is_empty() {
            continue;
        }
        at_level.sort_by_key(|s| s.min_seq);
        for run in contiguous_runs(&at_level) {
            if run_bytes(&run) >= target || run.len() >= config.max_segments_per_level {
                return Some(build_job(take_until_target(&run, target), level + 1));
            }
        }
    }
    None
}

/// Group `segments` (sorted by `min_seq`) into adjacency runs. Adjacency means
/// `prev.max_seq + 1 == next.min_seq`.
fn contiguous_runs<'a>(segments: &[&'a SegmentRow]) -> Vec<Vec<&'a SegmentRow>> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut runs: Vec<Vec<&SegmentRow>> = vec![vec![segments[0]]];
    for &seg in &segments[1..] {
        let last_run = runs.last_mut().expect("runs is non-empty");
        if last_run.last().expect("run is non-empty").max_seq + 1 == seg.min_seq {
            last_run.push(seg);
        } else {
            runs.push(vec![seg]);
        }
    }
    runs
}

fn run_bytes(run: &[&SegmentRow]) -> i64 {
    run.iter().map(|s| s.byte_size).sum()
}

/// Take the shortest prefix of `run` whose compressed byte sum hits `target`.
/// Always returns at least one segment.
fn take_until_target<'a>(run: &[&'a SegmentRow], target: i64) -> Vec<&'a SegmentRow> {
    let mut out: Vec<&SegmentRow> = Vec::new();
    let mut compressed: i64 = 0;
    for &seg in run {
        out.push(seg);
        compressed += seg.byte_size;
        if compressed >= target {
            break;
        }
    }
    out
}

fn build_job(run: Vec<&SegmentRow>, output_level: i32) -> CompactionJob {
    let output_min_seq = run.iter().map(|s| s.min_seq).min().expect("run non-empty");
    CompactionJob {
        inputs: run.into_iter().cloned().collect(),
        output_level,
        output_min_seq,
    }
}

/// Fold per-input `(min, max)` Int64 key bounds into one `(min, max)`.
///
/// Operates on the typed `i64` values (the in-memory `LocalSegment` bounds) so
/// numeric keys keep native ordering — the catalog's TEXT round-trip would flip
/// `"10" < "2"`. Inputs whose value is `None` (empty segment / no stats) are
/// skipped; returns `(None, None)` if every input was skipped.
pub fn aggregate_key_bounds<I>(bounds: I) -> (Option<i64>, Option<i64>)
where
    I: IntoIterator<Item = (Option<i64>, Option<i64>)>,
{
    let mut overall_min: Option<i64> = None;
    let mut overall_max: Option<i64> = None;
    for (lo, hi) in bounds {
        if let Some(lo) = lo {
            overall_min = Some(overall_min.map_or(lo, |x| x.min(lo)));
        }
        if let Some(hi) = hi {
            overall_max = Some(overall_max.map_or(hi, |x| x.max(hi)));
        }
    }
    (overall_min, overall_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::types::SegmentLocation;

    /// Build a `SegmentRow` for planner tests.
    fn row(level: i32, min_seq: i64, max_seq: i64, byte_size: i64) -> SegmentRow {
        SegmentRow {
            namespace: "ns".to_string(),
            path: format!("/x/seg_L{level}_{min_seq:019}.parquet"),
            level,
            min_seq,
            max_seq,
            row_count: max_seq - min_seq + 1,
            byte_size,
            created_at_ms: 0,
            min_key_value: None,
            max_key_value: None,
            location: SegmentLocation::Local,
        }
    }

    fn config(level_targets: Vec<i64>, max_segments_per_level: usize) -> CompactionConfig {
        CompactionConfig {
            level_targets,
            max_segments_per_level,
            ..Default::default()
        }
    }

    // --- the 6 planner cases ---------------------

    #[test]
    fn plan_returns_none_when_under_target() {
        let cfg = config(vec![1024], 1024);
        let rows = vec![row(0, 1, 1, 128)];
        assert_eq!(plan(&cfg, &rows), None);
    }

    #[test]
    fn plan_promotes_when_byte_target_reached() {
        let cfg = config(vec![1024], 1024);
        let rows = vec![row(0, 1, 1, 512), row(0, 2, 2, 512)];
        let job = plan(&cfg, &rows).unwrap();
        assert_eq!(job.output_level, 1);
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![1, 2]);
    }

    #[test]
    fn plan_promotes_at_segment_count_below_byte_target() {
        let cfg = config(vec![1 << 30], 3);
        let rows = vec![row(0, 1, 1, 128), row(0, 2, 2, 128), row(0, 3, 3, 128)];
        let job = plan(&cfg, &rows).unwrap();
        assert_eq!(job.output_level, 1);
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![1, 2, 3]);
    }

    #[test]
    fn plan_does_not_count_promote_terminal_level() {
        // terminal level == len(level_targets) == 1; L1 is terminal here.
        let cfg = config(vec![1024], 2);
        let rows = vec![row(1, 1, 1, 128), row(1, 2, 2, 128), row(1, 3, 3, 128)];
        assert_eq!(plan(&cfg, &rows), None);
    }

    #[test]
    fn plan_count_promotes_non_terminal_l1_below_byte_target() {
        // L2 is non-terminal (len == 2), so L1 count-promotes.
        let cfg = config(vec![64, 1 << 30], 2);
        let rows = vec![row(1, 1, 1, 8), row(1, 2, 2, 8)];
        let job = plan(&cfg, &rows).unwrap();
        assert_eq!(job.output_level, 2);
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![1, 2]);
    }

    #[test]
    fn plan_single_l2_segment_at_l3_target_emits_single_input_job() {
        let cfg = config(vec![64, 256, 256], 32);
        let rows = vec![row(2, 1, 100, 256)];
        let job = plan(&cfg, &rows).unwrap();
        assert_eq!(job.output_level, 3);
        assert_eq!(job.inputs.len(), 1);
        assert_eq!(job.inputs[0].min_seq, 1);
    }

    // --- take_until_target / contiguous_runs ----------------------------

    #[test]
    fn take_until_target_shortest_prefix_at_least_one() {
        let r0 = row(0, 1, 1, 30);
        let r1 = row(0, 2, 2, 40);
        let r2 = row(0, 3, 3, 50);
        let run = vec![&r0, &r1, &r2];
        // 30 + 40 >= 64 stops at 2.
        let taken = take_until_target(&run, 64);
        let mins: Vec<i64> = taken.iter().map(|s| s.min_seq).collect();
        assert_eq!(mins, vec![1, 2]);
        // sub-target run still makes forward progress with >=1 input.
        assert_eq!(take_until_target(&run, 1_000_000).len(), 3);
        let single = vec![&r0];
        assert_eq!(take_until_target(&single, 1_000_000).len(), 1);
    }

    #[test]
    fn contiguous_runs_splits_on_gap_and_single_for_suffix() {
        // contiguous suffix: one run.
        let r0 = row(0, 1, 2, 10);
        let r1 = row(0, 3, 4, 10);
        let r2 = row(0, 5, 6, 10);
        let segs = vec![&r0, &r1, &r2];
        let runs = contiguous_runs(&segs);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len(), 3);

        // a seq gap (max_seq+1 != next.min_seq) splits.
        let g0 = row(0, 1, 2, 10);
        let g1 = row(0, 10, 11, 10);
        let gapped = vec![&g0, &g1];
        let runs = contiguous_runs(&gapped);
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn plan_skips_gap_run_then_selects_promotable_run() {
        // Two runs at L0: [1..1] (tiny) and [10..10],[11..11] (count-promotes).
        let cfg = config(vec![1 << 30], 2);
        let rows = vec![row(0, 1, 1, 8), row(0, 10, 10, 8), row(0, 11, 11, 8)];
        let job = plan(&cfg, &rows).unwrap();
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![10, 11]);
    }

    // --- aggregate_key_bounds -------------------------------------------

    #[test]
    fn aggregate_key_bounds_preserves_numeric_ordering() {
        // 2..10: stringified "10" < "2" would flip; native i64 must not.
        let (lo, hi) = aggregate_key_bounds([(Some(2), Some(10)), (Some(5), Some(7))]);
        assert_eq!(lo, Some(2));
        assert_eq!(hi, Some(10));
    }

    #[test]
    fn aggregate_key_bounds_skips_none_inputs() {
        let (lo, hi) = aggregate_key_bounds([(None, None), (Some(3), Some(9))]);
        assert_eq!(lo, Some(3));
        assert_eq!(hi, Some(9));
    }

    #[test]
    fn aggregate_key_bounds_all_none() {
        assert_eq!(
            aggregate_key_bounds([(None, None), (None, None)]),
            (None, None)
        );
    }

    // --- compaction_sort_keys -------------------------------------------

    #[test]
    fn compaction_sort_keys_with_and_without_key_column() {
        use crate::proto::finelog::stats::ColumnType;
        use crate::store::schema::Column;
        let with_key = Schema::new(
            vec![Column::new("ts", ColumnType::COLUMN_TYPE_INT64, false)],
            "ts",
        );
        assert_eq!(compaction_sort_keys(&with_key), vec!["ts", "seq"]);
        let no_key = Schema::new(
            vec![Column::new("x", ColumnType::COLUMN_TYPE_INT64, false)],
            "",
        );
        assert_eq!(compaction_sort_keys(&no_key), vec!["seq"]);
    }
}
