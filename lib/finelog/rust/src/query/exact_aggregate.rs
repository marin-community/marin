//! Exact `GROUP BY string_column, COUNT(...)` from segment value-count sections.
//!
//! The fast path is deliberately narrow. It recognizes one unfiltered table
//! scan, one string grouping column, and one ordinary `COUNT(*)` or
//! `COUNT(grouping_column)`. Any extra relational operation falls back to
//! DataFusion. Likewise, every visible segment must have a complete count
//! summary; partial backfill never contributes a partial answer.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::common::tree_node::Transformed;
use datafusion::common::DFSchemaRef;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::SessionState;
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_datasource::memory::MemorySourceConfig;

use crate::query::index_cache::IndexCache;
use crate::query::QueryResult;
use crate::store::index_bundle::SectionKind;
use crate::store::segment::segment_id_and_row_group_rows;

const MAX_COMBINED_COUNT_VALUES: usize = 16_384;

static NEXT_RESULT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CountMode {
    AllRows,
    GroupingColumn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputColumn {
    Group,
    Count,
}

#[derive(Debug, Clone)]
pub struct CountRequest {
    pub table: String,
    column: String,
    mode: CountMode,
    output: [OutputColumn; 2],
    schema: SchemaRef,
}

/// Logical node emitted when every visible segment has an exact count section.
#[derive(Clone)]
struct ExactAggregateNode {
    result_id: u64,
    result: QueryResult,
    schema: DFSchemaRef,
    table: String,
    column: String,
    segments: usize,
}

impl fmt::Debug for ExactAggregateNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExactAggregateNode")
            .field("result_id", &self.result_id)
            .field("table", &self.table)
            .field("column", &self.column)
            .field("segments", &self.segments)
            .finish()
    }
}

impl PartialEq for ExactAggregateNode {
    fn eq(&self, other: &Self) -> bool {
        self.result_id == other.result_id
    }
}

impl Eq for ExactAggregateNode {}

impl PartialOrd for ExactAggregateNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.result_id.partial_cmp(&other.result_id)
    }
}

impl Hash for ExactAggregateNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.result_id.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for ExactAggregateNode {
    fn name(&self) -> &str {
        "FinelogIndexAggregate"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        Vec::new()
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FinelogIndexAggregate: table={}, column={}, segments={}, method=value_counts",
            self.table, self.column, self.segments
        )
    }

    fn with_exprs_and_inputs(
        &self,
        expressions: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self> {
        if !expressions.is_empty() || !inputs.is_empty() {
            return Err(DataFusionError::Plan(
                "FinelogIndexAggregate has no expressions or inputs".to_string(),
            ));
        }
        Ok(self.clone())
    }
}

/// Replaces a supported grouped count with a planner-visible exact-index node.
#[derive(Debug)]
pub struct ExactAggregateRewrite {
    segment_paths: HashMap<String, Vec<String>>,
}

impl ExactAggregateRewrite {
    pub fn new(segment_paths: HashMap<String, Vec<String>>) -> Self {
        Self { segment_paths }
    }
}

impl OptimizerRule for ExactAggregateRewrite {
    fn name(&self) -> &str {
        "finelog_exact_aggregate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        let Some(request) = count_request(&plan) else {
            return Ok(Transformed::no(plan));
        };
        let Some(paths) = self.segment_paths.get(&request.table) else {
            return Ok(Transformed::no(plan));
        };
        let schema = Arc::clone(plan.schema());
        let Some(result) = execute(&request, paths)? else {
            return Ok(Transformed::no(plan));
        };
        let result_id = NEXT_RESULT_ID.fetch_add(1, Ordering::Relaxed);
        let node = ExactAggregateNode {
            result_id,
            result,
            schema,
            table: request.table,
            column: request.column,
            segments: paths.len(),
        };
        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })))
    }
}

#[derive(Debug)]
struct ExactAggregatePlanner;

#[async_trait]
impl ExtensionPlanner for ExactAggregatePlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<ExactAggregateNode>() else {
            return Ok(None);
        };
        let plan = MemorySourceConfig::try_new_exec(
            std::slice::from_ref(&node.result.batches),
            Arc::clone(&node.result.schema),
            None,
        )?;
        Ok(Some(plan))
    }
}

/// Query planner that teaches DataFusion how to execute Finelog extension nodes.
#[derive(Debug)]
pub struct FinelogQueryPlanner;

#[async_trait]
impl QueryPlanner for FinelogQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(ExactAggregatePlanner)])
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// Recognize the exact aggregate shape without accepting semantically richer
/// plans such as filters, sorts, limits, distinct aggregates, or joins.
pub fn count_request(plan: &LogicalPlan) -> Option<CountRequest> {
    let (aggregate, output) = match plan {
        LogicalPlan::Projection(projection) => {
            let LogicalPlan::Aggregate(aggregate) = projection.input.as_ref() else {
                return None;
            };
            if projection.expr.len() != 2
                || aggregate.group_expr.len() != 1
                || aggregate.aggr_expr.len() != 1
            {
                return None;
            }
            let group_name = aggregate.schema.field(0).name();
            let count_name = aggregate.schema.field(1).name();
            let output: Vec<OutputColumn> = projection
                .expr
                .iter()
                .map(|expr| output_column(expr, group_name, count_name))
                .collect::<Option<_>>()?;
            (aggregate, [output[0], output[1]])
        }
        LogicalPlan::Aggregate(aggregate) => {
            (aggregate, [OutputColumn::Group, OutputColumn::Count])
        }
        _ => return None,
    };
    if aggregate.group_expr.len() != 1 || aggregate.aggr_expr.len() != 1 {
        return None;
    }
    let Expr::Column(group) = &aggregate.group_expr[0] else {
        return None;
    };
    let Expr::AggregateFunction(count) = &aggregate.aggr_expr[0] else {
        return None;
    };
    if !count.func.name().eq_ignore_ascii_case("count")
        || count.params.distinct
        || count.params.filter.is_some()
        || !count.params.order_by.is_empty()
        || count.params.args.len() != 1
    {
        return None;
    }
    let mode = match &count.params.args[0] {
        Expr::Column(column) if column.name == group.name => CountMode::GroupingColumn,
        Expr::Literal(value, _) if !value.is_null() => CountMode::AllRows,
        _ => return None,
    };
    let LogicalPlan::TableScan(scan) = aggregate.input.as_ref() else {
        return None;
    };
    if !scan.filters.is_empty() || scan.fetch.is_some() {
        return None;
    }
    let schema = Arc::new(plan.schema().as_arrow().clone());
    if schema.fields().len() != 2
        || !matches!(
            schema.field(0).data_type(),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 | DataType::Int64
        )
    {
        return None;
    }
    Some(CountRequest {
        table: scan.table_name.table().to_string(),
        column: group.name.clone(),
        mode,
        output,
        schema,
    })
}

fn output_column(expr: &Expr, group_name: &str, count_name: &str) -> Option<OutputColumn> {
    let expr = match expr {
        Expr::Alias(alias) => alias.expr.as_ref(),
        expr => expr,
    };
    let Expr::Column(column) = expr else {
        return None;
    };
    if column.name == group_name {
        Some(OutputColumn::Group)
    } else if column.name == count_name {
        Some(OutputColumn::Count)
    } else {
        None
    }
}

/// Load and combine complete segment summaries, then form the query result.
///
/// `Ok(None)` means at least one segment is absent, stale, malformed, or lacks
/// the requested column summary; the caller must execute the original query.
pub fn execute(request: &CountRequest, segment_paths: &[String]) -> DFResult<Option<QueryResult>> {
    let mut combined: BTreeMap<Option<String>, u64> = BTreeMap::new();
    for path in segment_paths {
        let parquet = Path::new(path);
        let Some((source_id, row_groups)) = segment_id_and_row_group_rows(parquet) else {
            return Ok(None);
        };
        let parquet_rows = row_groups.iter().sum::<usize>() as u64;
        let Some(header) = IndexCache::global().get_header(parquet, source_id, parquet_rows) else {
            return Ok(None);
        };
        let Some(index) =
            IndexCache::global().get_exact(parquet, &header, SectionKind::ValueCounts)
        else {
            return Ok(None);
        };
        let Some(counts) = index
            .columns
            .get(&request.column)
            .and_then(|column| column.counts.as_ref())
        else {
            return Ok(None);
        };
        for (value, count) in counts {
            let Some(total) = combined
                .get(value)
                .copied()
                .unwrap_or_default()
                .checked_add(*count)
            else {
                return Ok(None);
            };
            *combined.entry(value.clone()).or_default() = total;
            if combined.len() > MAX_COMBINED_COUNT_VALUES {
                return Ok(None);
            }
        }
    }
    let groups: Vec<Option<String>> = combined.keys().cloned().collect();
    let Some(counts): Option<Vec<i64>> = combined
        .iter()
        .map(|(value, count)| match (request.mode, value) {
            (CountMode::GroupingColumn, None) => Some(0),
            _ => i64::try_from(*count).ok(),
        })
        .collect()
    else {
        return Ok(None);
    };
    let group: ArrayRef = Arc::new(StringArray::from_iter(
        groups.iter().map(|value| value.as_deref()),
    ));
    let count: ArrayRef = Arc::new(Int64Array::from(counts));
    let arrays = request
        .output
        .map(|column| match column {
            OutputColumn::Group => Arc::clone(&group),
            OutputColumn::Count => Arc::clone(&count),
        })
        .into_iter()
        .zip(request.schema.fields())
        .map(|(array, field)| {
            if array.data_type() == field.data_type() {
                Ok(array)
            } else {
                cast(&array, field.data_type())
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| DataFusionError::ArrowError(Box::new(error), None))?;
    let fields: Vec<Field> = request
        .schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    let schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        request.schema.metadata().clone(),
    ));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)
        .map_err(|error| DataFusionError::ArrowError(Box::new(error), None))?;
    tracing::debug!(
        table = request.table,
        column = request.column,
        groups = batch.num_rows(),
        segments = segment_paths.len(),
        "answered value-count aggregate from index sections"
    );
    Ok(Some(QueryResult {
        schema,
        batches: vec![batch],
    }))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, StringArray};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::store::exact::ExactIndexConfig;
    use crate::store::segment::write_segment_to_dir;
    use crate::store::segment_index::{write_segment_index, SegmentIndexConfig};

    async fn plan(sql: &str) -> LogicalPlan {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("service", DataType::Utf8, true),
            Field::new("other", DataType::Int64, true),
        ]));
        let table = MemTable::try_new(schema, vec![vec![]]).unwrap();
        let context = SessionContext::new();
        context
            .register_table("telemetry_v1", Arc::new(table))
            .unwrap();
        context.sql(sql).await.unwrap().into_unoptimized_plan()
    }

    #[tokio::test]
    async fn recognizes_only_plain_grouped_counts() {
        let request = count_request(
            &plan("SELECT service, count(service) FROM telemetry_v1 GROUP BY service").await,
        )
        .unwrap();
        assert_eq!(request.table, "telemetry_v1");
        assert_eq!(request.column, "service");
        assert_eq!(request.mode, CountMode::GroupingColumn);
        assert_eq!(
            count_request(
                &plan("SELECT service, count(*) FROM telemetry_v1 GROUP BY service").await
            )
            .unwrap()
            .mode,
            CountMode::AllRows
        );

        assert!(count_request(
            &plan(
                "SELECT service, count(service) FROM telemetry_v1 WHERE other > 1 GROUP BY service"
            )
            .await
        )
        .is_none());
        assert!(count_request(
            &plan("SELECT service, count(DISTINCT service) FROM telemetry_v1 GROUP BY service")
                .await
        )
        .is_none());
        assert!(count_request(
            &plan("SELECT service, count(NULL) FROM telemetry_v1 GROUP BY service").await
        )
        .is_none());
        assert!(count_request(
            &plan(
                "SELECT service, service AS service_again \
                 FROM telemetry_v1 GROUP BY service"
            )
            .await
        )
        .is_none());
    }

    #[test]
    fn executes_only_when_every_segment_has_a_complete_summary() {
        let dir = crate::test_support::unique_dir("exact_aggregate");
        let source_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "service",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            source_schema,
            vec![Arc::new(StringArray::from(vec![
                Some("worker"),
                Some("worker"),
                Some("api"),
                None,
            ]))],
        )
        .unwrap();
        let (parquet, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let config = ExactIndexConfig {
            column: "service".to_string(),
            exact_values: Vec::new(),
            value_counts: true,
        };
        write_segment_index(
            &parquet,
            &[batch],
            &SegmentIndexConfig::from_policies(Vec::<String>::new(), &[config], &[], None),
        )
        .unwrap();
        let request = CountRequest {
            table: "telemetry_v1".to_string(),
            column: "service".to_string(),
            mode: CountMode::GroupingColumn,
            output: [OutputColumn::Group, OutputColumn::Count],
            schema: Arc::new(ArrowSchema::new(vec![
                Field::new("service", DataType::Utf8, true),
                Field::new("count(service)", DataType::Int64, true),
            ])),
        };
        let paths = vec![parquet.to_string_lossy().into_owned()];
        let result = execute(&request, &paths).unwrap().unwrap();
        let values = result.batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let counts = result.batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let rows: Vec<(Option<&str>, i64)> = (0..values.len())
            .map(|row| {
                (
                    (!values.is_null(row)).then(|| values.value(row)),
                    counts.value(row),
                )
            })
            .collect();
        assert_eq!(rows, vec![(None, 0), (Some("api"), 1), (Some("worker"), 2)]);

        std::fs::remove_file(crate::store::index_bundle::bundle_path(&parquet)).unwrap();
        IndexCache::global().invalidate(&crate::store::index_bundle::bundle_path(&parquet));
        assert!(execute(&request, &paths).unwrap().is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn exact_aggregate_is_visible_in_explain() {
        let dir = crate::test_support::unique_dir("exact_aggregate_explain");
        let source_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "service",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(StringArray::from(vec!["worker", "api"]))],
        )
        .unwrap();
        let (parquet, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        write_segment_index(
            &parquet,
            &[batch],
            &SegmentIndexConfig::from_policies(
                Vec::<String>::new(),
                &[ExactIndexConfig {
                    column: "service".to_string(),
                    exact_values: Vec::new(),
                    value_counts: true,
                }],
                &[],
                None,
            ),
        )
        .unwrap();
        let path = parquet.to_string_lossy().into_owned();
        let provider = crate::query::provider::NamespaceProvider::build(
            source_schema,
            std::slice::from_ref(&path),
        )
        .unwrap();
        let ctx = crate::query::make_ctx();
        ctx.register_table("telemetry_v1", Arc::new(provider))
            .unwrap();
        ctx.add_optimizer_rule(Arc::new(ExactAggregateRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            vec![path],
        )]))));
        let batches = ctx
            .sql("EXPLAIN SELECT service, count(service) FROM telemetry_v1 GROUP BY service")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let explain = batches
            .iter()
            .flat_map(|batch| batch.columns())
            .filter_map(|column| column.as_any().downcast_ref::<StringArray>())
            .flat_map(|column| (0..column.len()).map(|row| column.value(row)))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(explain.contains("FinelogIndexAggregate"), "{explain}");
        std::fs::remove_dir_all(dir).ok();
    }
}
