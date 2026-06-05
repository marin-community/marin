// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Benchmark the trigram substring prune against a real prod log slice.
//!
//! Targets the motivating FetchLogs shape from issue #6195: `contains(data, …)`
//! over the log namespace. For each needle it runs the same query twice on the
//! same segments — once through a plain `ListingTable` (today's behavior, no
//! trigram prune) and once through `NamespaceProvider` (which injects the
//! per-row-group access plan) — and reports the wall-clock delta. It also builds
//! any missing `.tgm` sidecars first, reporting build time and on-disk size.
//!
//! Run (not part of `cargo test`):
//!   cargo run --release --example bench_trigram -- \
//!     /home/power/finelog-bench/data/log [max_segments]
//!
//! Results match by construction (the prune never drops a match); the assert
//! guards that. The point is the latency / bytes-scanned ratio.

use std::sync::Arc;
use std::time::Instant;

use arrow::datatypes::SchemaRef;
use datafusion::common::TableReference;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use finelog::query::make_ctx;
use finelog::query::provider::NamespaceProvider;
use finelog::store::segment::{discover_segments, segment_row_group_count};
use finelog::store::trigram::{sidecar_path, write_sidecar, INDEXED_COLUMN};

/// Needles spanning the rarity spectrum measured in the design doc.
const NEEDLES: &[&str] = &[
    "Bootstrap completed for TPU",
    "Lifecycle Leak!",
    "disconnected unexpectedly",
];

fn arrow_schema_of(path: &std::path::Path) -> SchemaRef {
    let file = std::fs::File::open(path).expect("open segment");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
    Arc::clone(builder.schema())
}

fn read_all_batches(path: &std::path::Path) -> Vec<arrow::array::RecordBatch> {
    let file = std::fs::File::open(path).expect("open segment");
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader")
        .build()
        .expect("build reader");
    reader.map(|b| b.expect("read batch")).collect()
}

async fn count_contains(ctx: &SessionContext, table: &str, needle: &str) -> usize {
    let escaped = needle.replace('\'', "''");
    let sql = format!("SELECT count(*) AS n FROM \"{table}\" WHERE contains(data, '{escaped}')");
    let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    col.value(0) as usize
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "/home/power/finelog-bench/data/log".to_string());
    let max_segments: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let dir = std::path::PathBuf::from(dir);

    let mut segments = discover_segments(&dir);
    segments.truncate(max_segments);
    assert!(!segments.is_empty(), "no segments under {}", dir.display());
    let schema = arrow_schema_of(&segments[0]);
    println!(
        "segments: {}  schema cols: {:?}",
        segments.len(),
        schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
    );

    // ---- Phase 1: build any missing sidecars; report cost + size. ----
    let mut built = 0usize;
    let mut total_sidecar_bytes = 0u64;
    let mut total_parquet_bytes = 0u64;
    let mut total_rg = 0usize;
    let build_start = Instant::now();
    for seg in &segments {
        let sc = sidecar_path(seg);
        total_parquet_bytes += std::fs::metadata(seg).map(|m| m.len()).unwrap_or(0);
        if !sc.exists() {
            let batches = read_all_batches(seg);
            write_sidecar(seg, &batches, INDEXED_COLUMN, None).expect("write sidecar");
            built += 1;
        }
        if let Ok(m) = std::fs::metadata(&sc) {
            total_sidecar_bytes += m.len();
        }
        total_rg += segment_row_group_count(seg).unwrap_or(0);
    }
    println!(
        "sidecars: built {built}/{} in {:.1}s  row_groups={total_rg}  \
         index_size={:.1}MB ({:.2}% of {:.0}MB parquet)",
        segments.len(),
        build_start.elapsed().as_secs_f64(),
        total_sidecar_bytes as f64 / 1e6,
        100.0 * total_sidecar_bytes as f64 / total_parquet_bytes.max(1) as f64,
        total_parquet_bytes as f64 / 1e6,
    );

    // ---- Phase 2: time pruned vs unpruned for each needle. ----
    let paths: Vec<String> = segments
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let urls: Vec<ListingTableUrl> = paths
        .iter()
        .map(|p| ListingTableUrl::parse(format!("file://{p}")).unwrap())
        .collect();

    println!(
        "\n{:<32} {:>10} {:>12} {:>12} {:>8}",
        "needle", "matches", "unpruned", "pruned", "speedup"
    );
    for needle in NEEDLES {
        // Baseline: a plain ListingTable (no access-plan injection) — today's path.
        let base_ctx = make_ctx();
        let opts =
            ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
        let cfg = ListingTableConfig::new_with_multi_paths(urls.clone())
            .with_listing_options(opts)
            .with_schema(Arc::clone(&schema));
        let listing = ListingTable::try_new(cfg).unwrap();
        base_ctx
            .register_table(TableReference::bare("log"), Arc::new(listing))
            .unwrap();
        let t0 = Instant::now();
        let base_n = count_contains(&base_ctx, "log", needle).await;
        let base_ms = t0.elapsed().as_millis();

        // Pruned: NamespaceProvider injects the trigram access plan.
        let prov_ctx = make_ctx();
        let provider = NamespaceProvider::build(Arc::clone(&schema), &paths).unwrap();
        prov_ctx
            .register_table(TableReference::bare("log"), Arc::new(provider))
            .unwrap();
        let t1 = Instant::now();
        let prov_n = count_contains(&prov_ctx, "log", needle).await;
        let prov_ms = t1.elapsed().as_millis();

        assert_eq!(base_n, prov_n, "prune changed the result for {needle:?}");
        let speedup = base_ms as f64 / prov_ms.max(1) as f64;
        println!(
            "{:<32} {:>10} {:>10}ms {:>10}ms {:>7.1}x",
            needle, prov_n, base_ms, prov_ms, speedup
        );
    }
}
