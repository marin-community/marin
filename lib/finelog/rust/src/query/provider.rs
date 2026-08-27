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
use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
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

use crate::query::index_cache::IndexCache;

/// A live namespace as one DataFusion table.
///
/// Backed by a `ListingTable` over the snapshotted sealed parquet files, or —
/// when the namespace has no sealed segments — an empty `MemTable` carrying the
/// registered schema (the typed-empty case).
#[derive(Debug)]
pub struct NamespaceProvider {
    schema: SchemaRef,
    inner: Inner,
    /// The snapshotted sealed segment paths, retained so scans can locate each
    /// segment's typed index bundle. Empty for the typed-empty
    /// (no-segments) case.
    segment_paths: Vec<String>,
    index_cache: Arc<IndexCache>,
}

#[derive(Debug)]
enum Inner {
    Listing(Arc<ListingTable>),
    Empty(Arc<MemTable>),
}

/// Retype `schema`'s top-level `Utf8` columns as `Utf8View`.
///
/// Parquet stores strings the same way either way, so this only chooses the
/// in-memory layout the scan produces. `Utf8View` lets the reader emit 16-byte
/// views over the decompressed pages rather than copying every value into one
/// contiguous buffer, and lets grouping and comparison settle most rows on the
/// inline 4-byte prefix. On a `GROUP BY name` over a low-cardinality column that
/// is worth roughly 1.8x.
///
/// Nested string types (a `Map`'s keys and values) are left alone: the win comes
/// from the wide top-level columns, and retyping map children would churn the
/// duck-typed `json_*` path for no measured gain.
///
/// [`crate::query::normalize_result`] converts back at the response boundary, so
/// the layout never reaches a client.
fn view_typed_schema(schema: &SchemaRef) -> SchemaRef {
    if !schema
        .fields()
        .iter()
        .any(|f| f.data_type() == &DataType::Utf8)
    {
        return Arc::clone(schema);
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Utf8 => f.as_ref().clone().with_data_type(DataType::Utf8View),
            _ => f.as_ref().clone(),
        })
        .collect();
    Arc::new(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ))
}

impl NamespaceProvider {
    pub fn segment_paths(&self) -> &[String] {
        &self.segment_paths
    }

    pub fn index_cache(&self) -> &Arc<IndexCache> {
        &self.index_cache
    }

    /// Build a provider from the registered arrow `schema` and a snapshot of
    /// sealed segment file paths.
    ///
    /// `segment_paths` are absolute local filesystem paths
    /// (`{ns_dir}/seg_L*_*.parquet`). Each is registered individually (rather
    /// than listing a directory) so the scan sees exactly the snapshotted set —
    /// no re-listing, and compaction can't slip a new file in.
    pub fn build(
        schema: SchemaRef,
        segment_paths: &[String],
        index_cache: Arc<IndexCache>,
    ) -> DFResult<NamespaceProvider> {
        let schema = view_typed_schema(&schema);
        if segment_paths.is_empty() {
            let mem = MemTable::try_new(Arc::clone(&schema), vec![vec![]])?;
            return Ok(NamespaceProvider {
                schema,
                inner: Inner::Empty(Arc::new(mem)),
                segment_paths: Vec::new(),
                index_cache,
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
            index_cache,
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
                // range / min-max row-group pruning), then layer bundle-backed
                // filtered projections or access plans onto its files.
                let plan = t.scan(state, projection, filters, limit).await?;
                let needles = crate::query::trigram_prune::substring_needles_by_column(filters);
                let exact = crate::query::exact_prune::values_by_column(filters);
                if needles.is_empty() && exact.is_empty() {
                    return Ok(plan);
                }
                // Key ranges (incl. the analyzer's synthesized prefix bounds) scope
                // which segments' sections the prune reads — cheap expr inspection,
                // done here before the blocking work.
                let key_ranges = crate::query::trigram_prune::string_column_ranges(filters);
                let mut required_columns: BTreeSet<String> = projection
                    .map(|indices| {
                        indices
                            .iter()
                            .map(|&index| self.schema.field(index).name().clone())
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        self.schema
                            .fields()
                            .iter()
                            .map(|field| field.name().clone())
                            .collect()
                    });
                required_columns.extend(
                    filters
                        .iter()
                        .flat_map(Expr::column_refs)
                        .map(|column| column.name.clone()),
                );
                // Bundle + footer reads are blocking, so run pruning off the
                // async worker.
                let segment_paths = self.segment_paths.clone();
                let index_cache = Arc::clone(&self.index_cache);
                tokio::task::spawn_blocking(move || {
                    let plan = crate::query::trigram_prune::apply_with_needles(
                        plan,
                        &segment_paths,
                        &needles,
                        &key_ranges,
                        &index_cache,
                    );
                    crate::query::exact_prune::apply(
                        plan,
                        &segment_paths,
                        &exact,
                        &required_columns,
                        &index_cache,
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

    use crate::query::string_values::StringValues;
    use crate::store::trigram::SIDECAR_SPAN_ROWS;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::logical_expr::{col, lit};
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
        let provider = NamespaceProvider::build(
            Arc::clone(&schema),
            &[],
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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

        let provider = NamespaceProvider::build(
            worker_arrow(),
            &paths,
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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
        let ids = first_column_strings(&batches);
        assert_eq!(ids, vec!["w-1", "w-2", "w-3"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn exact_predicate_uses_complete_filtered_projections() {
        let dir = tempdir("exact_projection");
        let batch = worker_batch(1, vec!["w-1", "w-2", "w-3"], vec![100, 200, 300]);
        let (path, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let config = crate::store::exact::ExactIndexConfig {
            column: "worker_id".to_string(),
            exact_values: vec!["w-2".to_string()],
            value_counts: false,
        };
        crate::store::segment_index::write_segment_index(
            &path,
            &[batch],
            &crate::store::segment_index::SegmentIndexConfig::from_policies(
                Vec::<String>::new(),
                &[config],
                &[crate::store::schema::CoveringProjection::new(
                    "workers",
                    "worker_id",
                    ["w-2"],
                    ["seq", "worker_id", "mem_bytes"],
                )],
                None,
            ),
        )
        .unwrap();

        let provider = NamespaceProvider::build(
            worker_arrow(),
            &[path.to_string_lossy().into_owned()],
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();
        let plan = provider
            .scan(&state, None, &[col("worker_id").eq(lit("w-2"))], None)
            .await
            .unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let files: Vec<_> = config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .collect();
        assert_eq!(files.len(), 1);
        assert!(files[0]
            .object_meta
            .location
            .as_ref()
            .ends_with(".fidx.workers.parquet"));
        assert!(files[0].extensions.is_none());

        ctx.register_table("workers", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql("SELECT worker_id FROM workers WHERE worker_id = 'w-2'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(first_column_strings(&batches), vec!["w-2"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn missing_projection_falls_back_only_for_that_segment() {
        let dir = tempdir("exact_projection_fallback");
        let first = worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]);
        let second = worker_batch(3, vec!["w-2", "w-3"], vec![300, 400]);
        let (first_path, _) = write_segment_to_dir(&dir, 1, 1, &first).unwrap();
        let (second_path, _) = write_segment_to_dir(&dir, 1, 3, &second).unwrap();
        let config = crate::store::exact::ExactIndexConfig {
            column: "worker_id".to_string(),
            exact_values: vec!["w-2".to_string()],
            value_counts: false,
        };
        let index_config = crate::store::segment_index::SegmentIndexConfig::from_policies(
            Vec::<String>::new(),
            &[config],
            &[crate::store::schema::CoveringProjection::new(
                "workers",
                "worker_id",
                ["w-2"],
                ["seq", "worker_id", "mem_bytes"],
            )],
            None,
        );
        crate::store::segment_index::write_segment_index(&first_path, &[first], &index_config)
            .unwrap();
        crate::store::segment_index::write_segment_index(&second_path, &[second], &index_config)
            .unwrap();
        std::fs::remove_file(crate::store::exact::named_projection_path(
            &second_path,
            "workers",
        ))
        .unwrap();
        let paths = vec![
            first_path.to_string_lossy().into_owned(),
            second_path.to_string_lossy().into_owned(),
        ];
        let provider = NamespaceProvider::build(
            worker_arrow(),
            &paths,
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
        let ctx = SessionContext::new();
        let plan = provider
            .scan(&ctx.state(), None, &[col("worker_id").eq(lit("w-2"))], None)
            .await
            .unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let locations: Vec<&str> = config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .map(|file| file.object_meta.location.as_ref())
            .collect();
        assert_eq!(
            locations
                .iter()
                .filter(|location| location.ends_with(".fidx.workers.parquet"))
                .count(),
            1
        );
        assert_eq!(
            locations
                .iter()
                .filter(|location| location.ends_with(".parquet"))
                .count(),
            2
        );

        ctx.register_table("workers", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql("SELECT worker_id FROM workers WHERE worker_id = 'w-2' ORDER BY seq")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(first_column_strings(&batches), vec!["w-2", "w-2"]);
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

        let provider = NamespaceProvider::build(
            schema,
            &paths,
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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
            Arc::new(
                NamespaceProvider::build(
                    worker_arrow(),
                    &wpaths,
                    crate::query::index_cache::test_index_cache(),
                )
                .unwrap(),
            ),
        )
        .unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.task"),
            Arc::new(
                NamespaceProvider::build(
                    task_arrow,
                    &tpaths,
                    crate::query::index_cache::test_index_cache(),
                )
                .unwrap(),
            ),
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

    /// The first column of `batches` as strings, in row order — what a
    /// single-column projection assertion compares against.
    fn first_column_strings(batches: &[RecordBatch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                let column = b.column(0);
                let c = StringValues::new(column).expect("string column");
                (0..column.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Log-form schema: seq, key, data (the columns the trigram prune touches).
    fn log_arrow() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
        ]))
    }

    /// Write one segment whose `data` column is a full span of `filler` followed
    /// by `rg1`, then build its trigram section — so source span 0 lacks the
    /// needle and span 1 carries it. Returns the segment path.
    fn write_two_span_log_segment(dir: &std::path::Path, filler: &str, rg1: &[&str]) -> String {
        let n0 = SIDECAR_SPAN_ROWS;
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
        crate::store::segment_index::write_segment_index(
            &path,
            &[batch],
            &crate::store::segment_index::SegmentIndexConfig::from_policies(
                ["data"],
                &[],
                &[],
                Some("key".to_string()),
            ),
        )
        .unwrap();
        path.to_string_lossy().into_owned()
    }

    /// Whether any expression in `plan` casts the `data` column. Walks the tree
    /// rather than the rendered text, which qualifies column names inconsistently
    /// across plan stages and would make a substring check pass vacuously.
    fn casts_the_data_column(plan: &datafusion::logical_expr::LogicalPlan) -> bool {
        use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
        use datafusion::logical_expr::Expr;

        let mut found = false;
        plan.apply(|node| {
            for e in node.expressions() {
                e.apply(|sub| {
                    if let Expr::Cast(cast) = sub {
                        if matches!(cast.expr.as_ref(), Expr::Column(c) if c.name == "data") {
                            found = true;
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .unwrap();
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    #[tokio::test]
    async fn string_predicates_read_the_scan_layout_without_casting() {
        // The scan reads strings as Utf8View, and the finelog UDFs accept that
        // layout. A UDF with an exact-Utf8 signature would instead make the
        // planner cast the whole column ahead of the predicate — materializing
        // every value precisely to throw most of them away.
        let dir = tempdir("no_cast");
        let path = write_two_span_log_segment(&dir, "idle heartbeat ok", &["needle here"; 4]);
        let provider = NamespaceProvider::build(
            log_arrow(),
            std::slice::from_ref(&path),
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
        assert_eq!(
            provider
                .schema()
                .field_with_name("data")
                .unwrap()
                .data_type(),
            &DataType::Utf8View,
        );

        let ctx = crate::query::make_ctx();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        for predicate in [
            "contains(data, 'needle')",
            "prefix(data, 'needle')",
            "regexp_matches(data, 'needle')",
            "json_get(data, 'k') IS NOT NULL",
        ] {
            let plan = ctx
                .sql(&format!("SELECT seq FROM \"log\" WHERE {predicate}"))
                .await
                .unwrap()
                .into_optimized_plan()
                .unwrap();
            assert!(
                !casts_the_data_column(&plan),
                "{predicate} planned a whole-column cast:\n{}",
                plan.display_indent()
            );
        }

        // The detector must actually fire on a cast, or the assertions above pass
        // for the wrong reason.
        let cast_plan = ctx
            .sql("SELECT seq FROM \"log\" WHERE CAST(data AS int) = 1")
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap();
        assert!(casts_the_data_column(&cast_plan));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn contains_query_returns_matches_and_prunes_row_groups() {
        use datafusion::datasource::physical_plan::FileScanConfig;
        use datafusion::datasource::source::DataSourceExec;
        use datafusion::logical_expr::{col, lit};
        use datafusion::logical_expr::{expr::ScalarFunction, Expr};
        use datafusion_datasource_parquet::{ParquetAccessPlan, RowGroupAccess};

        let dir = tempdir("contains_prune");
        // The needle lives only in row group 1 (rows 2 and 4 of the tail).
        let needle = "Bootstrap completed for TPU-xyz";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "idle heartbeat ok",
            "another Bootstrap completed for TPU-xyz here",
        ];
        let rg1_rows = rg1.len();
        let path = write_two_span_log_segment(&dir, "idle heartbeat ok", &rg1);

        // 1) End-to-end correctness: the contains() query returns exactly the two
        //    matching rows (and prunes row group 0 along the way).
        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build(
            log_arrow(),
            std::slice::from_ref(&path),
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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
        let got = first_column_strings(&batches);
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
        let probe = NamespaceProvider::build(
            log_arrow(),
            &[path],
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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
                // The needle is confined to the second span. These narrow rows
                // fit one byte-sized row group, so the prune expresses that as a
                // row selection inside it rather than a skipped row group.
                let [RowGroupAccess::Selection(selection)] = ap.inner() else {
                    panic!("expected one row group carrying a row selection: {ap:?}");
                };
                assert_eq!(
                    selection.row_count(),
                    rg1_rows,
                    "only the needle's span may be selected"
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
        use datafusion_datasource_parquet::{ParquetAccessPlan, RowGroupAccess};

        let dir = tempdir("like_prune");
        let needle = "Bootstrap completed for TPU-xyz";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "idle heartbeat ok",
        ];
        let rg1_rows = rg1.len();
        let path = write_two_span_log_segment(&dir, "idle heartbeat ok", &rg1);

        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build(
            log_arrow(),
            std::slice::from_ref(&path),
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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
        let got = first_column_strings(&batches);
        assert_eq!(
            got,
            vec!["E0601 Bootstrap completed for TPU-xyz started".to_string()],
            "LIKE must return exactly the matching row"
        );

        // The injected access plan skips the needle-free row group 0.
        let plan = NamespaceProvider::build(
            log_arrow(),
            &[path],
            crate::query::index_cache::test_index_cache(),
        )
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
                // The needle is confined to the second span. These narrow rows
                // fit one byte-sized row group, so the prune expresses that as a
                // row selection inside it rather than a skipped row group.
                let [RowGroupAccess::Selection(selection)] = ap.inner() else {
                    panic!("expected one row group carrying a row selection: {ap:?}");
                };
                assert_eq!(
                    selection.row_count(),
                    rg1_rows,
                    "only the needle's span may be selected"
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
        let path = write_two_span_log_segment(&dir, "idle heartbeat ok", &["one match here"]);
        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build(
            log_arrow(),
            &[path],
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
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
