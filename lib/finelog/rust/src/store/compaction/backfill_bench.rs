//! Stage-timed reproductions of segment backfill work. Run with:
//!
//! ```text
//! cargo test --release --lib backfill_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The migration batch is sized to the levanter.metrics production profile
//! (~8M rows over ~180 run-partitioned legacy segments, 26 columns). The index
//! backfill builds one telemetry_v1.node_agent segment's bundle with the
//! production index policy, shaped from one minute of hub data: ~450k rows per
//! segment, 630 nodes, 42 metric names, and attribute JSON too varied for value
//! counts.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;

use crate::indices::{exact, format, write_segment_index, SegmentIndexConfig};
use crate::levanter_metrics_policy::levanter_metrics_schema;
use crate::server::telemetry::telemetry_schema;
use crate::store::compaction::executor::read_segment_projected;
use crate::store::compaction::merge::{
    merge_row_converter, merge_runs, project_to_schema, sort_batch_to_run, sort_col_indices,
    SortedRun,
};
use crate::store::schema::{schema_to_arrow, stored_form};
use crate::store::segment::segment_writer_properties_with_max_rows;
use crate::store::table::segment_format::SegmentFormat;
use crate::telemetry_policy::NODE_AGENT_NAMESPACE;
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

const NODE_AGENT_NODES: usize = 630;
const NODE_AGENT_SCRAPES: usize = 54;
const NODE_AGENT_NAMES_PER_SCRAPE: usize = 13;
const NODE_AGENT_SCRAPE_INTERVAL_MS: i64 = 10_000;
const NODE_AGENT_UNITS: [&str; 14] = [
    "bytes",
    "percent",
    "celsius",
    "watts",
    "ratio",
    "count",
    "seconds",
    "bytes_per_second",
    "errors",
    "code",
    "hertz",
    "requests",
    "pages",
    "1",
];

/// Metric names a node agent emits: every host and accelerator name the
/// telemetry schema indexes, plus names that only get value counts.
fn node_agent_names() -> Vec<String> {
    let indexed = [
        "node_cpu_utilization_percent",
        "node_disk_total_bytes",
        "node_disk_used_bytes",
        "node_memory_total_bytes",
        "node_memory_used_bytes",
        "node_network_receive_bytes",
        "node_network_transmit_bytes",
        "gpu_memory_temperature_celsius",
        "gpu_memory_total_bytes",
        "gpu_memory_used_bytes",
        "gpu_nvlink_receive_bytes_per_second",
        "gpu_nvlink_transmit_bytes_per_second",
        "gpu_pcie_replay_errors",
        "gpu_pcie_receive_bytes_per_second",
        "gpu_pcie_transmit_bytes_per_second",
        "gpu_power_watts",
        "gpu_row_remap_failures",
        "gpu_sm_active_ratio",
        "gpu_temperature_celsius",
        "gpu_tensor_active_ratio",
        "gpu_utilization_percent",
        "gpu_xid_error_code",
        "hardware_inventory",
    ];
    let mut names: Vec<String> = indexed.iter().map(|name| name.to_string()).collect();
    names.extend((names.len()..42).map(|index| format!("node_filesystem_metric_{index:02}")));
    names
}

/// One flushed node-agent segment in arrival order: each scrape writes every
/// node's sampled metrics together, so each metric name's rows interleave.
fn synth_node_agent_segment(schema: &SchemaRef) -> RecordBatch {
    let names = node_agent_names();
    let rows = NODE_AGENT_NODES * NODE_AGENT_SCRAPES * NODE_AGENT_NAMES_PER_SCRAPE;
    let mut timestamp = Vec::with_capacity(rows);
    let mut batch_id = Vec::with_capacity(rows);
    let mut record_index = Vec::with_capacity(rows);
    let mut node_name = Vec::with_capacity(rows);
    let mut name = Vec::with_capacity(rows);
    let mut value = Vec::with_capacity(rows);
    let mut unit = Vec::with_capacity(rows);
    let mut resource_attributes = Vec::with_capacity(rows);
    let mut attributes = Vec::with_capacity(rows);
    let mut cluster = Vec::with_capacity(rows);
    for scrape in 0..NODE_AGENT_SCRAPES {
        for node in 0..NODE_AGENT_NODES {
            for sample in 0..NODE_AGENT_NAMES_PER_SCRAPE {
                let metric = (node + sample * 3) % names.len();
                timestamp.push(1_790_000_000_000 + scrape as i64 * NODE_AGENT_SCRAPE_INTERVAL_MS);
                batch_id.push(format!("batch-{scrape:03}-{:03}", node / 4));
                record_index.push(((node % 4) * NODE_AGENT_NAMES_PER_SCRAPE + sample) as i64);
                node_name.push(format!("node-{node:04}"));
                name.push(names[metric].clone());
                value.push((scrape * node + sample) as f64 * 0.5);
                unit.push(Some(NODE_AGENT_UNITS[metric % NODE_AGENT_UNITS.len()]));
                resource_attributes.push(format!(
                    r#"{{"host.name":"node-{node:04}","node.uid":"{node:08x}-4c1e-9a7b","cloud.zone":"us-east-{}","service.version":"2026.11.0"}}"#,
                    node % 4
                ));
                attributes.push(format!(
                    r#"{{"device":"dev{sample}","node_uid":"{node:08x}","window":"{}"}}"#,
                    scrape % 12
                ));
                cluster.push(Some(
                    [
                        "cw-us-east-08a",
                        "cw-us-east-04b",
                        "marin",
                        "gcp-us-central",
                    ][node % 4],
                ));
            }
        }
    }
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|field| -> ArrayRef {
            match field.name().as_str() {
                "seq" => Arc::new(Int64Array::from_iter_values(1..=rows as i64)),
                "schema_version" => Arc::new(Int32Array::from(vec![1; rows])),
                "timestamp_ms" => Arc::new(Int64Array::from(timestamp.clone())),
                "batch_id" => Arc::new(StringArray::from(batch_id.clone())),
                "record_index" => Arc::new(Int64Array::from(record_index.clone())),
                "service" => Arc::new(StringArray::from(vec!["iris-node-agent"; rows])),
                "kind" => Arc::new(StringArray::from(vec!["gauge"; rows])),
                "node_name" => Arc::new(StringArray::from(node_name.clone())),
                "name" => Arc::new(StringArray::from(name.clone())),
                "value" => Arc::new(Float64Array::from(value.clone())),
                "unit" => Arc::new(StringArray::from(unit.clone())),
                "resource_attributes_json" => {
                    Arc::new(StringArray::from(resource_attributes.clone()))
                }
                "attributes_json" => Arc::new(StringArray::from(attributes.clone())),
                "cluster" => Arc::new(StringArray::from(cluster.clone())),
                _ => arrow::array::new_null_array(field.data_type(), rows),
            }
        })
        .collect();
    RecordBatch::try_new(Arc::clone(schema), columns).expect("node-agent batch construction")
}

#[test]
#[ignore = "stage-timing benchmark; run explicitly with --release --ignored --nocapture"]
fn bench_node_agent_index_backfill() {
    let segment_format = SegmentFormat::resolve(stored_form(telemetry_schema())).unwrap();
    let config = segment_format.index_config(NODE_AGENT_NAMESPACE);
    let schema = Arc::clone(segment_format.arrow_schema());
    let dir = unique_dir("node_agent_index_bench");
    let path = dir.join("seg-node-agent.parquet");

    let generated = Instant::now();
    let batch = synth_node_agent_segment(&schema);
    let props =
        segment_writer_properties_with_max_rows(segment_format.max_row_group_rows()).unwrap();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path).unwrap(),
        Arc::clone(&schema),
        Some(props),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    println!(
        "generated node-agent segment: {} rows, {:.1} MiB parquet in {:.1?}",
        batch.num_rows(),
        std::fs::metadata(&path).unwrap().len() as f64 / (1024.0 * 1024.0),
        generated.elapsed()
    );
    drop(batch);

    let read = Instant::now();
    let projection = config.input_columns();
    let batches = read_segment_projected(&path, Some(&projection)).unwrap();
    println!("stage read+decode: {:.1?}", read.elapsed());

    let index = Instant::now();
    let built = write_segment_index(&path, &batches, &config).unwrap();
    println!("stage index bundle: {:.1?}", index.elapsed());

    // The bundle build's exact phases, timed alone so a change to either is
    // visible against the whole build.
    let sidecar_started = Instant::now();
    let sidecar = exact::build_sidecar(&batches, &config.exact_configs())
        .expect("node-agent policy configures exact indexes");
    println!("  exact sidecar build: {:.1?}", sidecar_started.elapsed());
    let sections_started = Instant::now();
    let section_bytes =
        exact::serialize_postings(&sidecar).len() + exact::serialize_counts(&sidecar).len();
    println!(
        "  exact section encode: {:.1?} ({section_bytes} bytes)",
        sections_started.elapsed()
    );

    let header = format::read_header(built.bundle.as_ref().unwrap()).unwrap();
    for section in &header.sections {
        println!("  section {:<34} {:>10} bytes", section.id, section.len);
    }
    let projection_bytes: u64 = built
        .projections
        .iter()
        .map(|path| std::fs::metadata(path).unwrap().len())
        .sum();
    println!(
        "  {} covering projections, {projection_bytes} bytes",
        built.projections.len()
    );
}
