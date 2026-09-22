//! Pure leveled-compaction policy.
//!
//! No I/O — this module decides *which* segments to merge into *what* output
//! level; the executor reads/writes the actual parquet (`executor.rs`).
//!
//! Levels are time-ordered: every flush emits an L0 segment. A level is promoted
//! L_n -> L_{n+1} when one of its streams hits the byte target for that tier OR
//! when its length hits `max_segments_per_level`. The count trigger bounds
//! per-read fanout for slow / bursty namespaces whose L0 flushes are small. The
//! terminal level (`level_targets.len()`) never re-compacts.

use std::collections::BTreeMap;

use crate::partition_policy::SegmentPartition;
use crate::store::compaction::config::{CompactionConfig, CompactionJob};
use crate::store::schema::{Schema, IMPLICIT_SEQ_COLUMN};
use crate::store::types::SegmentRow;

/// How unpartitioned segments form compaction streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnpartitionedRunPolicy {
    /// Split segments unless consecutive sequence intervals are exactly adjacent.
    StrictAdjacency,
    /// Treat every unpartitioned segment at one level as one sparse stream.
    SparseStream,
}

/// Sort keys for compaction, including the implicit `seq` tie-breaker.
pub fn compaction_sort_keys(schema: &Schema) -> Vec<String> {
    let mut columns = crate::store::schema::resolve_sort_columns(schema)
        .expect("registered schema has valid sort columns");
    columns.push(IMPLICIT_SEQ_COLUMN.to_string());
    columns
}

/// Return the next merge job, or `None` if nothing is due.
///
/// Walks tiers from L0 upward and returns the first promotable stream, capping
/// the selected prefix at the level's compressed byte or input-count limit.
///
/// The job's MEMORY cost is not decided here: a segment's decoded Arrow size
/// cannot be read off its footer (see `max_merge_arrow_bytes`), so the executor
/// measures it while reading and merges only the prefix of `inputs` that fits.
/// A job this planner returns is therefore an upper bound on what one tick
/// merges, not a promise.
pub fn plan(
    config: &CompactionConfig,
    segments: &[SegmentRow],
    unpartitioned: UnpartitionedRunPolicy,
) -> Option<CompactionJob> {
    for (n, &target) in config.level_targets.iter().enumerate() {
        let level = n as i32;
        let streams = streams_at_level(segments, level);
        for (partition, mut stream) in streams {
            sort_stream(&mut stream);
            let runs = if partition.is_some() {
                // A partition stream is sparse in the namespace-wide seq space;
                // its own files still form one compaction stream.
                vec![stream]
            } else {
                match unpartitioned {
                    UnpartitionedRunPolicy::StrictAdjacency => contiguous_runs(&stream),
                    UnpartitionedRunPolicy::SparseStream => vec![stream],
                }
            };
            for run in runs {
                if run_bytes(&run) >= target || run.len() >= config.max_segments_per_level {
                    return Some(build_job(
                        take_until_limit(&run, target, config.max_segments_per_level),
                        level + 1,
                    ));
                }
            }
        }
    }
    None
}

/// Promote every L0 input from the first canonical partition stream.
///
/// Forced maintenance bypasses eligibility thresholds, but it still obeys the
/// same exact-partition isolation and deterministic ordering as ordinary
/// planning.
pub fn plan_forced_l0(segments: &[SegmentRow]) -> Option<CompactionJob> {
    let (_, mut stream) = streams_at_level(segments, 0).into_iter().next()?;
    sort_stream(&mut stream);
    Some(build_job(stream, 1))
}

fn streams_at_level(
    segments: &[SegmentRow],
    level: i32,
) -> BTreeMap<Option<SegmentPartition>, Vec<&SegmentRow>> {
    let mut streams: BTreeMap<Option<SegmentPartition>, Vec<&SegmentRow>> = BTreeMap::new();
    for segment in segments.iter().filter(|segment| segment.level == level) {
        streams
            .entry(segment.partition.clone())
            .or_default()
            .push(segment);
    }
    streams
}

fn sort_stream(stream: &mut [&SegmentRow]) {
    stream.sort_by(|left, right| {
        left.min_seq
            .cmp(&right.min_seq)
            .then_with(|| left.max_seq.cmp(&right.max_seq))
            .then_with(|| left.path.cmp(&right.path))
    });
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
        let previous = last_run.last().expect("run is non-empty");
        if previous.max_seq.checked_add(1) == Some(seg.min_seq) {
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

/// Take the shortest prefix of `run` that hits either configured limit.
fn take_until_limit<'a>(
    run: &[&'a SegmentRow],
    target: i64,
    max_segments: usize,
) -> Vec<&'a SegmentRow> {
    let mut out: Vec<&SegmentRow> = Vec::new();
    let mut compressed: i64 = 0;
    for &seg in run {
        out.push(seg);
        compressed += seg.byte_size;
        if compressed >= target || out.len() >= max_segments {
            break;
        }
    }
    out
}

pub(crate) fn build_job(run: Vec<&SegmentRow>, output_level: i32) -> CompactionJob {
    let output_min_seq = run.iter().map(|s| s.min_seq).min().expect("run non-empty");
    CompactionJob {
        inputs: run.into_iter().cloned().collect(),
        output_level,
        output_min_seq,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::partition_policy::SegmentPartition;
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
            partition: None,
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
        assert_eq!(
            plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency),
            None
        );
    }

    #[test]
    fn plan_promotes_when_byte_target_reached() {
        let cfg = config(vec![1024], 1024);
        let rows = vec![row(0, 1, 1, 512), row(0, 2, 2, 512)];
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency).unwrap();
        assert_eq!(job.output_level, 1);
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![1, 2]);
    }

    #[test]
    fn plan_promotes_at_segment_count_below_byte_target() {
        let cfg = config(vec![1 << 30], 3);
        let rows = vec![row(0, 1, 1, 128), row(0, 2, 2, 128), row(0, 3, 3, 128)];
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency).unwrap();
        assert_eq!(job.output_level, 1);
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![1, 2, 3]);
    }

    #[test]
    fn plan_does_not_count_promote_terminal_level() {
        // terminal level == len(level_targets) == 1; L1 is terminal here.
        let cfg = config(vec![1024], 2);
        let rows = vec![row(1, 1, 1, 128), row(1, 2, 2, 128), row(1, 3, 3, 128)];
        assert_eq!(
            plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency),
            None
        );
    }

    #[test]
    fn plan_count_promotes_non_terminal_l1_below_byte_target() {
        // L2 is non-terminal (len == 2), so L1 count-promotes.
        let cfg = config(vec![64, 1 << 30], 2);
        let rows = vec![row(1, 1, 1, 8), row(1, 2, 2, 8)];
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency).unwrap();
        assert_eq!(job.output_level, 2);
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![1, 2]);
    }

    #[test]
    fn plan_single_l2_segment_at_l3_target_emits_single_input_job() {
        let cfg = config(vec![64, 256, 256], 32);
        let rows = vec![row(2, 1, 100, 256)];
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency).unwrap();
        assert_eq!(job.output_level, 3);
        assert_eq!(job.inputs.len(), 1);
        assert_eq!(job.inputs[0].min_seq, 1);
    }

    #[test]
    fn partition_stream_compacts_across_namespace_seq_gaps() {
        let partition = SegmentPartition {
            spec_id: 1,
            values: BTreeMap::from([("name_bucket".to_string(), "6".to_string())]),
        };
        let mut first = row(1, 1, 10, 60);
        first.partition = Some(partition.clone());
        let mut second = row(1, 100, 110, 60);
        second.partition = Some(partition);
        let mut other = row(1, 11, 99, 1_000);
        other.partition = Some(SegmentPartition {
            spec_id: 1,
            values: BTreeMap::from([("name_bucket".to_string(), "7".to_string())]),
        });

        let job = plan(
            &config(vec![64, 100], 32),
            &[first, other, second],
            UnpartitionedRunPolicy::StrictAdjacency,
        )
        .unwrap();
        assert_eq!(
            job.inputs
                .iter()
                .map(|segment| segment.min_seq)
                .collect::<Vec<_>>(),
            vec![1, 100]
        );
    }

    // --- take_until_limit / contiguous_runs -----------------------------

    #[test]
    fn take_until_limit_returns_shortest_bounded_prefix() {
        let r0 = row(0, 1, 1, 30);
        let r1 = row(0, 2, 2, 40);
        let r2 = row(0, 3, 3, 50);
        let run = vec![&r0, &r1, &r2];
        // 30 + 40 >= 64 stops at 2.
        let taken = take_until_limit(&run, 64, 10);
        let mins: Vec<i64> = taken.iter().map(|s| s.min_seq).collect();
        assert_eq!(mins, vec![1, 2]);
        assert_eq!(take_until_limit(&run, 1_000_000, 2).len(), 2);
        let single = vec![&r0];
        assert_eq!(take_until_limit(&single, 1_000_000, 10).len(), 1);
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
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency).unwrap();
        let mins: Vec<i64> = job.inputs.iter().map(|r| r.min_seq).collect();
        assert_eq!(mins, vec![10, 11]);
    }

    #[test]
    fn sparse_stream_compacts_overlaps_nested_ranges_and_true_gaps() {
        let cfg = config(vec![1 << 30], 4);
        let rows = vec![
            row(0, 1_000, 1_010, 8),
            row(0, 20, 30, 8),
            row(0, 90, 110, 8),
            row(0, 0, 100, 8),
        ];

        assert_eq!(
            plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency),
            None
        );
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::SparseStream).unwrap();
        assert_eq!(
            job.inputs
                .iter()
                .map(|segment| (segment.min_seq, segment.max_seq))
                .collect::<Vec<_>>(),
            vec![(0, 100), (20, 30), (90, 110), (1_000, 1_010)]
        );
    }

    #[test]
    fn strict_adjacency_handles_i64_max_without_overflow() {
        let cfg = config(vec![1 << 30], 2);
        let rows = vec![
            row(0, i64::MAX - 1, i64::MAX, 8),
            row(0, i64::MAX, i64::MAX, 8),
        ];

        assert_eq!(
            plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency),
            None
        );
    }

    #[test]
    fn sparse_stream_selects_deterministic_bounded_prefix() {
        let cfg = config(vec![1 << 30], 3);
        let mut first_path = row(0, 1, 3, 8);
        first_path.path = "/x/a.parquet".to_string();
        let mut second_path = row(0, 1, 3, 8);
        second_path.path = "/x/z.parquet".to_string();
        let rows = vec![
            row(0, 10, 20, 8),
            second_path,
            row(0, 1, 8, 8),
            row(0, 100, 200, 8),
            first_path,
        ];

        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::SparseStream).unwrap();
        assert_eq!(
            job.inputs
                .iter()
                .map(|segment| segment.path.as_str())
                .collect::<Vec<_>>(),
            vec![
                "/x/a.parquet",
                "/x/z.parquet",
                "/x/seg_L0_0000000000000000001.parquet",
            ]
        );
    }

    #[test]
    fn sparse_stream_makes_the_retired_levanter_l2_shape_eligible() {
        const MIB: i64 = 1024 * 1024;
        let cfg = config(vec![64 * MIB, 256 * MIB, 256 * MIB], 32);
        let rows = (0..150)
            .map(|index| row(2, index * 10, index * 10, 90 * MIB))
            .collect::<Vec<_>>();

        assert_eq!(
            plan(&cfg, &rows, UnpartitionedRunPolicy::StrictAdjacency),
            None
        );
        let job = plan(&cfg, &rows, UnpartitionedRunPolicy::SparseStream).unwrap();
        assert_eq!(job.output_level, 3);
        assert_eq!(job.inputs.len(), 3);
        assert_eq!(
            job.inputs.iter().map(|row| row.min_seq).collect::<Vec<_>>(),
            vec![0, 10, 20]
        );
    }

    #[test]
    fn forced_l0_promotion_never_mixes_partitions() {
        let mut first = row(0, 1, 1, 1);
        first.partition = Some(SegmentPartition {
            spec_id: 1,
            values: BTreeMap::from([("run_id".to_string(), "a".to_string())]),
        });
        let mut second = row(0, 2, 2, 1);
        second.partition = Some(SegmentPartition {
            spec_id: 1,
            values: BTreeMap::from([("run_id".to_string(), "b".to_string())]),
        });

        let job = plan_forced_l0(&[second, first.clone()]).unwrap();

        assert_eq!(job.inputs, vec![first]);
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
        let secondary = Schema::new(
            vec![
                Column::new("ts", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("worker", ColumnType::COLUMN_TYPE_STRING, false),
            ],
            "ts",
        )
        .with_sort_columns(["worker", "ts"]);
        assert_eq!(
            compaction_sort_keys(&secondary),
            vec!["worker", "ts", "seq"]
        );
        let no_key = Schema::new(
            vec![Column::new("x", ColumnType::COLUMN_TYPE_INT64, false)],
            "",
        );
        assert_eq!(compaction_sort_keys(&no_key), vec!["seq"]);
    }
}
