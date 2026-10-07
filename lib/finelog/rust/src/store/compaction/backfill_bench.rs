//! Stage-timed reproduction of one migration backfill batch, sized to the
//! levanter.metrics production profile (~8M rows over ~180 run-partitioned
//! legacy segments, 26 columns). Run with:
//!
//! ```text
//! cargo test --release backfill_bench -- --ignored --nocapture
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;

use crate::indices::{write_segment_index, SegmentIndexConfig};
use crate::levanter_metrics_policy::levanter_metrics_schema;
use crate::store::compaction::executor::read_segment_projected;
use crate::store::compaction::merge::{
    merge_row_converter, merge_runs, project_to_schema, sort_batch_to_run, sort_col_indices,
    SortedRun,
};
use crate::store::schema::{schema_to_arrow, stored_form};
use crate::store::segment::segment_writer_properties_with_max_rows;
use crate::test_support::unique_dir;

const SEGMENTS: usize = 180;
const ROWS_PER_SEGMENT: usize = 46_000;
const NAMES_PER_RUN: i64 = 400;
const ROW_GROUP_ROWS: usize = 131_072;

fn arrow_schema() -> SchemaRef {
    schema_to_arrow(&stored_form(levanter_metrics_schema()))
}

/// One legacy segment: a single run's slice, seq strictly increasing.
///
/// `sorted` mimics an L>=1 legacy segment, emitted name-major — the
/// [run_id, name, step, timestamp_ms] order legacy compaction wrote, which the
/// migration's sorted-run check accepts as-is. Unsorted mimics an L0 segment,
/// emitted step-major (each step's metrics together), which the check re-orders.
fn synth_segment(schema: &SchemaRef, segment: usize, seq_base: i64, sorted: bool) -> RecordBatch {
    let rows = ROWS_PER_SEGMENT;
    let run = format!("run-{:04}", segment % 50);
    let mut seq = Vec::with_capacity(rows);
    let mut cluster = Vec::with_capacity(rows);
    let mut timestamp = Vec::with_capacity(rows);
    let mut run_id = Vec::with_capacity(rows);
    let mut execution_uid = Vec::with_capacity(rows);
    let mut job_id = Vec::with_capacity(rows);
    let mut node_name = Vec::with_capacity(rows);
    let mut process_index = Vec::with_capacity(rows);
    let mut step = Vec::with_capacity(rows);
    let mut name = Vec::with_capacity(rows);
    let mut kind = Vec::with_capacity(rows);
    let mut value = Vec::with_capacity(rows);
    let mut batch_id = Vec::with_capacity(rows);
    let mut record_index = Vec::with_capacity(rows);
    let steps = rows as i64 / NAMES_PER_RUN;
    for row in 0..rows {
        let (metric, at_step) = if sorted {
            ((row as i64) / steps, (row as i64) % steps)
        } else {
            ((row as i64) % NAMES_PER_RUN, (row as i64) / NAMES_PER_RUN)
        };
        seq.push(seq_base + row as i64);
        cluster.push(None::<&str>);
        timestamp.push(Some(1_756_000_000_000 + at_step * 30_000));
        run_id.push(Some(run.clone()));
        execution_uid.push(Some(format!("{run}-attempt-2")));
        job_id.push(Some(format!("/marin/train/{run}")));
        node_name.push(Some(format!("tpu-{:03}", segment % 64)));
        process_index.push(Some((segment % 16) as i64));
        step.push(Some(at_step * 10));
        name.push(Some(format!(
            "train/layer_{:02}/grad_norm_{metric:03}",
            metric % 40
        )));
        kind.push(Some("scalar"));
        value.push(Some((metric as f64) * 0.001 + at_step as f64));
        batch_id.push(Some(format!("{run}-batch-{:06}", row / 512)));
        record_index.push(Some((row % 512) as i64));
    }
    let float_nulls: ArrayRef = Arc::new(Float64Array::from(vec![None::<f64>; rows]));
    let int_nulls: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>; rows]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(seq)),
        Arc::new(Int64Array::from(timestamp)),
        Arc::new(StringArray::from(run_id)),
        Arc::new(StringArray::from(execution_uid)),
        Arc::new(StringArray::from(job_id)),
        Arc::new(StringArray::from(node_name)),
        Arc::new(Int64Array::from(process_index)),
        Arc::new(Int64Array::from(step)),
        Arc::new(StringArray::from(name)),
        Arc::new(StringArray::from(kind)),
        Arc::new(Float64Array::from(value)),
        Arc::clone(&float_nulls), // min
        Arc::clone(&float_nulls), // max
        Arc::clone(&int_nulls),   // count
        Arc::clone(&int_nulls),   // nonzero_count
        Arc::clone(&float_nulls), // sum
        Arc::clone(&float_nulls), // sum_squares
        Arc::clone(&float_nulls), // mean
        Arc::clone(&float_nulls), // variance
        Arc::clone(&float_nulls), // rms
        arrow::array::new_null_array(schema.field(20).data_type(), rows), // bucket_limits
        arrow::array::new_null_array(schema.field(21).data_type(), rows), // bucket_counts
        Arc::new(StringArray::from(vec![None::<&str>; rows])), // unit
        arrow::array::new_null_array(schema.field(23).data_type(), rows), // attributes
        Arc::new(StringArray::from(batch_id)),
        Arc::new(Int64Array::from(record_index)),
        Arc::new(StringArray::from(cluster)), // origin_cluster
    ];
    RecordBatch::try_new(Arc::clone(schema), columns).expect("bench batch construction")
}

fn write_parquet(path: &Path, schema: &SchemaRef, batch: &RecordBatch) {
    let props = segment_writer_properties_with_max_rows(ROW_GROUP_ROWS).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(schema), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn metrics_index_config() -> SegmentIndexConfig {
    SegmentIndexConfig::from_policies(
        Vec::<String>::new(),
        &[],
        &[],
        Some("timestamp_ms".to_string()),
    )
    .with_adaptive_value_counts([
        "run_id",
        "execution_uid",
        "job_id",
        "node_name",
        "name",
        "kind",
        "unit",
        "batch_id",
    ])
}

#[test]
#[ignore = "stage-timing benchmark; run explicitly with --release --ignored --nocapture"]
fn bench_one_backfill_batch_sorted_inputs() {
    bench_one_backfill_batch(true);
}

#[test]
#[ignore = "stage-timing benchmark; run explicitly with --release --ignored --nocapture"]
fn bench_one_backfill_batch_unsorted_inputs() {
    bench_one_backfill_batch(false);
}

fn bench_one_backfill_batch(sorted: bool) {
    let schema = arrow_schema();
    let dir = unique_dir("backfill_bench");

    let generated = Instant::now();
    let mut inputs = Vec::with_capacity(SEGMENTS);
    for segment in 0..SEGMENTS {
        let batch = synth_segment(
            &schema,
            segment,
            (segment * ROWS_PER_SEGMENT) as i64,
            sorted,
        );
        let path = dir.join(format!("seg-{segment:05}.parquet"));
        write_parquet(&path, &schema, &batch);
        inputs.push(path);
    }
    let input_bytes: u64 = inputs
        .iter()
        .map(|path| std::fs::metadata(path).unwrap().len())
        .sum();
    println!(
        "generated {} {} segments, {} rows, {:.1} MiB parquet in {:.1?}",
        SEGMENTS,
        if sorted { "sorted" } else { "unsorted" },
        SEGMENTS * ROWS_PER_SEGMENT,
        input_bytes as f64 / (1024.0 * 1024.0),
        generated.elapsed()
    );

    let sort_cols = sort_col_indices(
        &schema,
        &[
            "run_id".to_string(),
            "name".to_string(),
            "step".to_string(),
            "timestamp_ms".to_string(),
        ],
    );

    let read = Instant::now();
    let raw: Vec<Vec<RecordBatch>> = inputs
        .iter()
        .map(|path| read_segment_projected(path, None).unwrap())
        .collect();
    println!("stage read+decode: {:.1?}", read.elapsed());

    let sort = Instant::now();
    let converter = merge_row_converter(&schema, &sort_cols).unwrap();
    let mut projected: Vec<SortedRun> = Vec::new();
    for batches in &raw {
        for batch in batches {
            let batch = project_to_schema(batch, &schema).unwrap();
            projected.push(sort_batch_to_run(&batch, &converter, &sort_cols).unwrap());
        }
    }
    println!(
        "stage project+sort: {:.1?} ({} sorted runs)",
        sort.elapsed(),
        projected.len()
    );

    let merge = Instant::now();
    let merged = merge_runs(&projected).unwrap();
    let merged_rows: usize = merged.iter().map(RecordBatch::num_rows).sum();
    println!(
        "stage merge_runs: {:.1?} ({merged_rows} rows, {} output chunks)",
        merge.elapsed(),
        merged.len()
    );
    drop(projected);

    let write = Instant::now();
    let out_path = dir.join("merged.parquet");
    let props = segment_writer_properties_with_max_rows(ROW_GROUP_ROWS).unwrap();
    let file = std::fs::File::create(&out_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    for batch in &merged {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    println!(
        "stage parquet write: {:.1?} ({:.1} MiB)",
        write.elapsed(),
        std::fs::metadata(&out_path).unwrap().len() as f64 / (1024.0 * 1024.0)
    );

    let index = Instant::now();
    write_segment_index(&out_path, &merged, &metrics_index_config()).unwrap();
    println!("stage index bundle: {:.1?}", index.elapsed());
}
