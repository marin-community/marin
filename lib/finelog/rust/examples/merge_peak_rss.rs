//! Measure peak RSS of a compaction merge against its decoded input size.
//!
//! Builds log-shaped segments (a wide, low-cardinality `data` string column),
//! merges them via the real `run_job`, and reports peak RSS / decoded bytes.
//! Validates the memory model behind `max_merge_uncompressed_bytes`.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use finelog::store::compaction::config::CompactionJob;
use finelog::store::compaction::executor::run_job;
use finelog::store::segment::{segment_uncompressed_bytes, write_segment_to_dir};
use finelog::store::types::{SegmentLocation, SegmentRow};

fn peak_rss_bytes() -> i64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: i64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap();
            return kb * 1024;
        }
    }
    0
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("key", DataType::Int64, true),
        Field::new("data", DataType::Utf8, true),
    ]))
}

/// One segment's worth of rows: unsorted keys, ~`row_bytes` of log-ish text.
fn batch(start_seq: i64, rows: usize, row_bytes: usize) -> RecordBatch {
    let seqs: Vec<i64> = (0..rows as i64).map(|i| start_seq + i).collect();
    let keys: Vec<i64> = (0..rows as i64).map(|i| (rows as i64) - i).collect();
    let data: Vec<String> = (0..rows)
        .map(|i| {
            format!(
                "{:0width$} ts=2026-07-09 lvl=INFO msg=request served",
                i,
                width = row_bytes
            )
        })
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(seqs)),
            Arc::new(Int64Array::from(keys)),
            Arc::new(StringArray::from(data)),
        ],
    )
    .unwrap()
}

fn row_for(path: &str, min_seq: i64, max_seq: i64, byte_size: i64) -> SegmentRow {
    SegmentRow {
        namespace: "log".into(),
        path: path.into(),
        level: 0,
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

fn main() {
    let n_segments: usize = std::env::args().nth(1).map_or(4, |v| v.parse().unwrap());
    let rows_per_segment: usize = std::env::args()
        .nth(2)
        .map_or(400_000, |v| v.parse().unwrap());
    let row_bytes: usize = 200;

    let dir =
        PathBuf::from(std::env::var("TMPDIR").unwrap_or("/tmp".into())).join("merge_peak_rss");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();

    let mut inputs = Vec::new();
    let mut decoded_total: i64 = 0;
    let mut compressed_total: i64 = 0;
    for s in 0..n_segments {
        let start = (s * rows_per_segment) as i64 + 1;
        let b = batch(start, rows_per_segment, row_bytes);
        let (path, _) = write_segment_to_dir(&dir, 0, start, &b).unwrap();
        let decoded = segment_uncompressed_bytes(&path).unwrap();
        let compressed = std::fs::metadata(&path).unwrap().len() as i64;
        decoded_total += decoded;
        compressed_total += compressed;
        inputs.push(row_for(
            &path.to_string_lossy(),
            start,
            start + rows_per_segment as i64 - 1,
            compressed,
        ));
    }

    let rss_before = peak_rss_bytes();
    let mib = |b: i64| b as f64 / (1024.0 * 1024.0);
    println!(
        "inputs={n_segments} rows={} compressed={:.0} MiB decoded={:.0} MiB ratio={:.1}x",
        n_segments * rows_per_segment,
        mib(compressed_total),
        mib(decoded_total),
        decoded_total as f64 / compressed_total as f64,
    );

    let max_seq = inputs.last().unwrap().max_seq;
    let job = CompactionJob {
        inputs,
        output_level: 1,
        output_min_seq: 1,
        output_max_seq: max_seq,
    };
    // Reset the high-water mark so the merge's peak is measured in isolation.
    std::fs::write("/proc/self/clear_refs", "5").ok();
    let swap = run_job(&job, &dir, &schema(), Some("key"), &[], |_| (None, None)).unwrap();
    let rss_after = peak_rss_bytes();

    println!(
        "peak_rss={:.0} MiB (before={:.0} MiB)  peak/decoded={:.2}x  output_rows={}",
        mib(rss_after),
        mib(rss_before),
        rss_after as f64 / decoded_total as f64,
        swap.added.row_count,
    );
    std::fs::remove_dir_all(&dir).ok();
}
