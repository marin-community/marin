//! Per-namespace DataFusion `TableProvider` over sealed parquet segments.
//!
//! A namespace's queryable data is the snapshot of its **sealed** local segment
//! files (NOT the in-RAM buffer). The durability contract (WriteRows/PushLogs
//! ack only after L0 seal+persist) makes this complete for RPC clients without
//! unioning the RAM buffer.
//!
//! An empty segment list yields a typed-empty table carrying the registered
//! arrow schema (incl. the implicit `seq` column).
//!
//! ## Read-visibility seam
//!
//! The provider holds a *snapshot* of segment paths captured under the
//! namespace insertion lock before scanning (see `Namespace::query_snapshot`).
//! Compaction takes the query-visibility write side before unlinking a file, so
//! a query that snapshotted the pre-compaction paths keeps scanning the files it
//! captured; the snapshot here is the read side of that seam.

use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Result as DFResult;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;

/// A live namespace as one DataFusion table.
///
/// Backed by a `ListingTable` over the snapshotted sealed parquet files, or —
/// when the namespace has no sealed segments — an empty `MemTable` carrying the
/// registered schema (the typed-empty case).
#[derive(Debug)]
pub struct NamespaceProvider {
    schema: SchemaRef,
    inner: Inner,
    /// The snapshotted sealed segment paths, retained so a `contains()` scan can
    /// locate each segment's trigram sidecar (`<segment>.tgm`) for row-group
    /// pruning. Empty for the typed-empty (no-segments) case.
    segment_paths: Vec<String>,
}

#[derive(Debug)]
enum Inner {
    Listing(Arc<ListingTable>),
    Empty(Arc<MemTable>),
}

impl NamespaceProvider {
    /// Build a provider from the registered arrow `schema` and a snapshot of
    /// sealed segment file paths.
    ///
    /// `segment_paths` are absolute local filesystem paths
    /// (`{ns_dir}/seg_L*_*.parquet`). Each is registered individually (rather
    /// than listing a directory) so the scan sees exactly the snapshotted set —
    /// no re-listing, and compaction can't slip a new file in.
    pub fn build(schema: SchemaRef, segment_paths: &[String]) -> DFResult<NamespaceProvider> {
        if segment_paths.is_empty() {
            let mem = MemTable::try_new(Arc::clone(&schema), vec![vec![]])?;
            return Ok(NamespaceProvider {
                schema,
                inner: Inner::Empty(Arc::new(mem)),
                segment_paths: Vec::new(),
            });
        }

        let urls: Vec<ListingTableUrl> = segment_paths
            .iter()
            .map(|p| ListingTableUrl::parse(format!("file://{p}")))
            .collect::<DFResult<Vec<_>>>()?;
        let opts =
            ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
        let cfg = ListingTableConfig::new_with_multi_paths(urls)
            .with_listing_options(opts)
            .with_schema(Arc::clone(&schema));
        let listing = ListingTable::try_new(cfg)?;
        Ok(NamespaceProvider {
            schema,
            inner: Inner::Listing(Arc::new(listing)),
            segment_paths: segment_paths.to_vec(),
        })
    }
}

#[async_trait]
impl TableProvider for NamespaceProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Inexact: DataFusion re-checks the filters, but the parquet scan can
        // still prune row groups from them.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        match &self.inner {
            Inner::Listing(t) => {
                // Delegate to DataFusion's parquet scan (which keeps the existing
                // range / min-max / bloom row-group pruning), then layer the
                // trigram prune on top by injecting per-file access plans.
                let plan = t.scan(state, projection, filters, limit).await?;
                // Hot path: a query with no `contains(col, …)`/`LIKE` filter on any
                // column does only this cheap expr inspection (no I/O) and returns
                // untouched.
                let needles = crate::query::trigram_prune::substring_needles_by_column(filters);
                if needles.is_empty() {
                    return Ok(plan);
                }
                // Key ranges (incl. the analyzer's synthesized prefix bounds) scope
                // which segments' sidecars the prune reads — cheap expr inspection,
                // done here before the blocking work.
                let key_ranges = crate::query::trigram_prune::string_column_ranges(filters);
                // Substring query: the sidecar + footer reads are blocking, so run
                // the prune off the async worker.
                let segment_paths = self.segment_paths.clone();
                tokio::task::spawn_blocking(move || {
                    crate::query::trigram_prune::apply_with_needles(
                        plan,
                        &segment_paths,
                        &needles,
                        &key_ranges,
                    )
                })
                .await
                .map_err(|e| {
                    datafusion::error::DataFusionError::Execution(format!(
                        "trigram prune task join: {e}"
                    ))
                })
            }
            Inner::Empty(t) => t.scan(state, projection, filters, limit).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::store::segment::write_segment_to_dir;

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_provider_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Store-form worker arrow schema: seq, worker_id, mem_bytes.
    fn worker_arrow() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("mem_bytes", DataType::Int64, false),
        ]))
    }

    fn worker_batch(first_seq: i64, ids: Vec<&str>, mem: Vec<i64>) -> RecordBatch {
        let n = ids.len() as i64;
        RecordBatch::try_new(
            worker_arrow(),
            vec![
                Arc::new(Int64Array::from_iter_values(first_seq..first_seq + n)),
                Arc::new(StringArray::from(ids)),
                Arc::new(Int64Array::from(mem)),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn empty_namespace_scans_zero_rows_typed() {
        let schema = worker_arrow();
        let provider = NamespaceProvider::build(Arc::clone(&schema), &[]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.worker"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql("SELECT * FROM \"iris.worker\"")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
        // Typed empty: the registered column set incl. `seq` survives.
        let result_schema = ctx
            .sql("SELECT * FROM \"iris.worker\"")
            .await
            .unwrap()
            .schema()
            .as_arrow()
            .clone();
        let names: Vec<&str> = result_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["seq", "worker_id", "mem_bytes"]);
    }

    #[tokio::test]
    async fn sealed_segments_scan_with_projection_and_order() {
        let dir = tempdir("scan");
        // Two segments out of seq order to prove the listing reads both.
        write_segment_to_dir(
            &dir,
            0,
            1,
            &worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]),
        )
        .unwrap();
        write_segment_to_dir(&dir, 0, 3, &worker_batch(3, vec!["w-3"], vec![300])).unwrap();
        let paths: Vec<String> = crate::store::segment::discover_segments(&dir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert_eq!(paths.len(), 2);

        let provider = NamespaceProvider::build(worker_arrow(), &paths).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.worker"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql("SELECT worker_id, mem_bytes FROM \"iris.worker\" ORDER BY worker_id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ids: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let c = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(ids, vec!["w-1", "w-2", "w-3"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn native_map_column_scans_and_json_get_reads_it() {
        // A sealed segment carrying a native Map<Utf8,Utf8> column reads back
        // through the ListingTable provider as a MapArray, and the duck-typed
        // json_get UDF filters/groups it — the whole storage-flip point.
        use crate::query::udf::register_scalar_udfs;
        use crate::store::schema::map_utf8_utf8_type;
        use arrow::array::{MapBuilder, MapFieldNames, StringBuilder};

        let dir = tempdir("map_scan");
        let schema: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("labels", map_utf8_utf8_type(), true),
        ]));

        let names = MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        };
        let mut mb = MapBuilder::new(Some(names), StringBuilder::new(), StringBuilder::new());
        for scope in ["fleet", "fleet", "local"] {
            mb.keys().append_value("scope");
            mb.values().append_value(scope);
            mb.append(true).unwrap();
        }
        let labels = mb.finish();
        assert_eq!(labels.data_type(), &map_utf8_utf8_type());

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(1..=3)),
                Arc::new(labels),
            ],
        )
        .unwrap();
        write_segment_to_dir(&dir, 0, 1, &batch).unwrap();
        let paths: Vec<String> = crate::store::segment::discover_segments(&dir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let provider = NamespaceProvider::build(schema, &paths).unwrap();
        let ctx = SessionContext::new();
        register_scalar_udfs(&ctx);
        ctx.register_table(
            datafusion::common::TableReference::bare("probes"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(
                "SELECT count(*) AS n FROM probes \
                 WHERE json_get(labels, 'scope') = 'fleet'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(n.value(0), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn two_providers_join() {
        let wdir = tempdir("join_w");
        let tdir = tempdir("join_t");
        write_segment_to_dir(
            &wdir,
            0,
            1,
            &worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]),
        )
        .unwrap();

        // task table: seq, worker_id, task_count
        let task_arrow: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("task_count", DataType::Int64, false),
        ]));
        let task_batch = RecordBatch::try_new(
            Arc::clone(&task_arrow),
            vec![
                Arc::new(Int64Array::from_iter_values(1..3)),
                Arc::new(StringArray::from(vec!["w-1", "w-2"])),
                Arc::new(Int64Array::from(vec![10_i64, 20])),
            ],
        )
        .unwrap();
        write_segment_to_dir(&tdir, 0, 1, &task_batch).unwrap();

        let wpaths: Vec<String> = crate::store::segment::discover_segments(&wdir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let tpaths: Vec<String> = crate::store::segment::discover_segments(&tdir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let ctx = SessionContext::new();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.worker"),
            Arc::new(NamespaceProvider::build(worker_arrow(), &wpaths).unwrap()),
        )
        .unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.task"),
            Arc::new(NamespaceProvider::build(task_arrow, &tpaths).unwrap()),
        )
        .unwrap();

        let batches = ctx
            .sql(
                "SELECT w.mem_bytes, t.task_count FROM \"iris.worker\" w \
                 JOIN \"iris.task\" t USING (worker_id) ORDER BY w.mem_bytes",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
        std::fs::remove_dir_all(&wdir).ok();
        std::fs::remove_dir_all(&tdir).ok();
    }

    /// Log-form schema: seq, key, data (the columns the trigram prune touches).
    fn log_arrow() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
        ]))
    }

    /// Write one segment whose `data` column is `rg0` rows of `filler` followed
    /// by `rg1`, then build its trigram sidecar — so row group 0 lacks the needle
    /// and row group 1 carries it. Returns the segment path.
    fn write_two_rg_log_segment(dir: &std::path::Path, filler: &str, rg1: &[&str]) -> String {
        use crate::store::segment::ROW_GROUP_SIZE;
        let n0 = ROW_GROUP_SIZE;
        let mut data: Vec<String> = (0..n0).map(|_| filler.to_string()).collect();
        data.extend(rg1.iter().map(|s| s.to_string()));
        let n = data.len() as i64;
        let batch = RecordBatch::try_new(
            log_arrow(),
            vec![
                Arc::new(Int64Array::from_iter_values(1..=n)),
                Arc::new(StringArray::from(vec!["/system/controller"; data.len()])),
                Arc::new(StringArray::from(data)),
            ],
        )
        .unwrap();
        let (path, _) = write_segment_to_dir(dir, 1, 1, &batch).unwrap();
        // Build the sidecar the way the compactor would.
        assert!(
            crate::store::trigram::write_sidecar(&path, &[batch], &["data"], Some("key")).unwrap(),
            "sidecar should be written for a data column"
        );
        path.to_string_lossy().into_owned()
    }

    #[tokio::test]
    async fn contains_query_returns_matches_and_prunes_row_groups() {
        use datafusion::datasource::physical_plan::FileScanConfig;
        use datafusion::datasource::source::DataSourceExec;
        use datafusion::logical_expr::{col, lit};
        use datafusion::logical_expr::{expr::ScalarFunction, Expr};
        use datafusion_datasource_parquet::ParquetAccessPlan;

        let dir = tempdir("contains_prune");
        // The needle lives only in row group 1 (rows 2 and 4 of the tail).
        let needle = "Bootstrap completed for TPU-xyz";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "idle heartbeat ok",
            "another Bootstrap completed for TPU-xyz here",
        ];
        let path = write_two_rg_log_segment(&dir, "idle heartbeat ok", &rg1);

        // 1) End-to-end correctness: the contains() query returns exactly the two
        //    matching rows (and prunes row group 0 along the way).
        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build(log_arrow(), std::slice::from_ref(&path)).unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(&format!(
                "SELECT data FROM \"log\" WHERE contains(data, '{needle}') ORDER BY seq"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let got: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let c = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            got,
            vec![
                "E0601 Bootstrap completed for TPU-xyz started".to_string(),
                "another Bootstrap completed for TPU-xyz here".to_string(),
            ],
            "contains() must return exactly the matching rows"
        );

        // 2) Evidence of pruning: the injected access plan skips row group 0 and
        //    keeps row group 1.
        let state = ctx.state();
        let udf = {
            use datafusion::execution::FunctionRegistry;
            ctx.udf("contains").unwrap()
        };
        let filter =
            Expr::ScalarFunction(ScalarFunction::new_udf(udf, vec![col("data"), lit(needle)]));
        let probe = NamespaceProvider::build(log_arrow(), &[path]).unwrap();
        let plan = probe.scan(&state, None, &[filter], None).await.unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let cfg = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let mut checked = 0;
        for group in &cfg.file_groups {
            for pf in group.files() {
                let ap = pf
                    .extensions
                    .as_ref()
                    .and_then(|e| e.downcast_ref::<ParquetAccessPlan>())
                    .expect("trigram access plan attached to the partitioned file");
                assert!(
                    !ap.should_scan(0),
                    "row group 0 (no needle) must be skipped"
                );
                assert!(
                    ap.should_scan(1),
                    "row group 1 (has needle) must be scanned"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 1,
            "exactly one partitioned file with an access plan"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn like_substring_query_prunes_row_groups() {
        // `data LIKE '%needle%'` must prune like `contains(data, 'needle')`: the
        // expression survives the simplifier as `Expr::Like` and the prune
        // extracts the framed substring. Asserts both the matching rows and the
        // injected skip of the needle-free row group 0.
        use datafusion::datasource::physical_plan::FileScanConfig;
        use datafusion::datasource::source::DataSourceExec;
        use datafusion_datasource_parquet::ParquetAccessPlan;

        let dir = tempdir("like_prune");
        let needle = "Bootstrap completed for TPU-xyz";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "idle heartbeat ok",
        ];
        let path = write_two_rg_log_segment(&dir, "idle heartbeat ok", &rg1);

        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build(log_arrow(), std::slice::from_ref(&path)).unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(&format!(
                "SELECT data FROM \"log\" WHERE data LIKE '%{needle}%' ORDER BY seq"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let got: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let c = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            got,
            vec!["E0601 Bootstrap completed for TPU-xyz started".to_string()],
            "LIKE must return exactly the matching row"
        );

        // The injected access plan skips the needle-free row group 0.
        let plan = NamespaceProvider::build(log_arrow(), &[path])
            .unwrap()
            .scan(
                &ctx.state(),
                None,
                std::slice::from_ref(
                    &datafusion::prelude::col("data")
                        .like(datafusion::prelude::lit(format!("%{needle}%"))),
                ),
                None,
            )
            .await
            .unwrap();
        let cfg = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("a parquet DataSourceExec")
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let mut checked = 0;
        for group in &cfg.file_groups {
            for pf in group.files() {
                let ap = pf
                    .extensions
                    .as_ref()
                    .and_then(|e| e.downcast_ref::<ParquetAccessPlan>())
                    .expect("trigram access plan attached for the LIKE query");
                assert!(
                    !ap.should_scan(0),
                    "row group 0 (no needle) must be skipped"
                );
                assert!(
                    ap.should_scan(1),
                    "row group 1 (has needle) must be scanned"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn non_contains_query_leaves_plan_unchanged() {
        // A query with no contains() filter must not be rewritten — the hot path
        // pays nothing. The returned plan is the untouched ListingTable scan.
        use datafusion::datasource::source::DataSourceExec;
        use datafusion::logical_expr::{col, lit};

        let dir = tempdir("no_contains");
        let path = write_two_rg_log_segment(&dir, "idle heartbeat ok", &["one match here"]);
        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build(log_arrow(), &[path]).unwrap();
        let state = ctx.state();
        let plan = provider
            .scan(&state, None, &[col("seq").gt(lit(0_i64))], None)
            .await
            .unwrap();
        // No access-plan extension is attached when there is no contains() filter.
        let exec = plan.as_any().downcast_ref::<DataSourceExec>().unwrap();
        let cfg = exec
            .data_source()
            .as_any()
            .downcast_ref::<datafusion::datasource::physical_plan::FileScanConfig>()
            .unwrap();
        for group in &cfg.file_groups {
            for pf in group.files() {
                assert!(
                    pf.extensions.is_none(),
                    "no prune extension on the hot path"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
