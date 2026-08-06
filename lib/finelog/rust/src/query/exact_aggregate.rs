//! Exact `GROUP BY string_column, COUNT(...)` from segment value-count sections.
//!
//! The fast path is deliberately narrow. It recognizes one table scan, an
//! optional half-open range on a non-null Int64 column, one string grouping
//! column, and one ordinary `COUNT(*)` or `COUNT(grouping_column)`. Because the
//! optimizer rewrites bottom-up, outer projections, sorts, and limits continue
//! to compose around the indexed aggregate. Stable segments wholly contained
//! by the range use count summaries; boundary and fresh L0 segments use an
//! ordinary aggregate, preserving exact results without derived-index work on
//! the write acknowledgement path.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::common::tree_node::Transformed;
use datafusion::common::DFSchemaRef;
use datafusion::datasource::{provider_as_source, MemTable};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SessionState, TaskContext};
use datafusion::functions_aggregate::expr_fn::{count, sum};
use datafusion::logical_expr::{
    col, lit, Cast, Expr, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::TryStreamExt;

use crate::query::index_cache::IndexCache;
use crate::query::predicate::{half_open_int_range, HalfOpenIntRange};
use crate::query::provider::NamespaceProvider;
use crate::store::index_bundle::SectionKind;
use crate::store::segment::segment_bounds;

const MAX_COMBINED_COUNT_VALUES: usize = 16_384;
const INDEX_GROUP_COLUMN: &str = "__finelog_index_group";
const INDEX_COUNT_COLUMN: &str = "__finelog_index_count";
const INDEX_TOTAL_COLUMN: &str = "__finelog_index_total";

static FULL_INDEX_AGGREGATES: AtomicU64 = AtomicU64::new(0);
static PARTIAL_INDEX_AGGREGATES: AtomicU64 = AtomicU64::new(0);
static DECLINED_INDEX_AGGREGATES: AtomicU64 = AtomicU64::new(0);
static FALLBACK_INDEX_AGGREGATES: AtomicU64 = AtomicU64::new(0);

/// Process-lifetime decisions made by the exact aggregate execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactAggregateStats {
    pub full: u64,
    pub partial: u64,
    pub declined: u64,
    pub fallbacks: u64,
}

/// Snapshot the exact aggregate execution counters.
pub fn stats() -> ExactAggregateStats {
    ExactAggregateStats {
        full: FULL_INDEX_AGGREGATES.load(Ordering::Relaxed),
        partial: PARTIAL_INDEX_AGGREGATES.load(Ordering::Relaxed),
        declined: DECLINED_INDEX_AGGREGATES.load(Ordering::Relaxed),
        fallbacks: FALLBACK_INDEX_AGGREGATES.load(Ordering::Relaxed),
    }
}

pub(crate) fn record_full() {
    FULL_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_partial() {
    PARTIAL_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_declined() {
    DECLINED_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_fallback() {
    FALLBACK_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CountMode {
    AllRows,
    GroupingColumn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum OutputColumn {
    Group,
    Count,
}

#[derive(Debug, Clone)]
pub struct CountRequest {
    pub table: String,
    column: String,
    mode: CountMode,
    range: Option<CountRange>,
    output: [OutputColumn; 2],
    schema: SchemaRef,
}

#[derive(Debug, Clone)]
struct CountRange {
    bounds: HalfOpenIntRange,
    filter_expr: Expr,
}

fn range_signature(request: &CountRequest) -> Option<(&str, i64, i64)> {
    request.range.as_ref().map(|range| {
        (
            range.bounds.column.as_str(),
            range.bounds.lower,
            range.bounds.upper,
        )
    })
}

/// Logical node emitted for a grouped count that can use segment summaries.
#[derive(Clone)]
struct ExactAggregateNode {
    request: CountRequest,
    source: AggregateSource,
    fallback: LogicalPlan,
    schema: DFSchemaRef,
}

impl fmt::Debug for ExactAggregateNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExactAggregateNode")
            .field("table", &self.request.table)
            .field("column", &self.request.column)
            .field("segments", &self.source.segment_paths.len())
            .finish()
    }
}

impl PartialEq for ExactAggregateNode {
    fn eq(&self, other: &Self) -> bool {
        self.request.table == other.request.table
            && self.request.column == other.request.column
            && self.request.mode == other.request.mode
            && range_signature(&self.request) == range_signature(&other.request)
            && self.request.output == other.request.output
            && self.source.segment_paths == other.source.segment_paths
            && Arc::ptr_eq(&self.source.index_cache, &other.source.index_cache)
            && self.fallback == other.fallback
    }
}

impl Eq for ExactAggregateNode {}

impl PartialOrd for ExactAggregateNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (
            &self.request.table,
            &self.request.column,
            self.request.mode,
            range_signature(&self.request),
            self.request.output,
            &self.source.segment_paths,
            Arc::as_ptr(&self.source.index_cache) as usize,
            &self.fallback,
        )
            .partial_cmp(&(
                &other.request.table,
                &other.request.column,
                other.request.mode,
                range_signature(&other.request),
                other.request.output,
                &other.source.segment_paths,
                Arc::as_ptr(&other.source.index_cache) as usize,
                &other.fallback,
            ))
    }
}

impl Hash for ExactAggregateNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.request.table.hash(state);
        self.request.column.hash(state);
        self.request.mode.hash(state);
        range_signature(&self.request).hash(state);
        self.request.output.hash(state);
        self.source.segment_paths.hash(state);
        (Arc::as_ptr(&self.source.index_cache) as usize).hash(state);
        self.fallback.hash(state);
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
            "FinelogIndexAggregate: table={}, column={}, segments={}, method=value_counts, coverage=runtime",
            self.request.table,
            self.request.column,
            self.source.segment_paths.len()
        )
    }

    fn with_exprs_and_inputs(
        &self,
        expressions: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DFResult<Self> {
        if !expressions.is_empty() || !inputs.is_empty() {
            return Err(DataFusionError::Plan(
                "FinelogIndexAggregate has no optimizer-visible inputs".to_string(),
            ));
        }
        Ok(self.clone())
    }
}

/// Replaces a supported grouped count with a planner-visible exact-index node.
#[derive(Debug)]
pub struct ExactAggregateRewrite {
    sources: HashMap<String, AggregateSource>,
}

#[derive(Debug, Clone)]
pub struct AggregateSource {
    pub segment_paths: Vec<String>,
    pub index_cache: Arc<IndexCache>,
    pub schema: SchemaRef,
}

impl ExactAggregateRewrite {
    pub fn new(sources: HashMap<String, AggregateSource>) -> Self {
        Self { sources }
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
        let Some(source) = self.sources.get(&request.table) else {
            return Ok(Transformed::no(plan));
        };
        if request.range.as_ref().is_some_and(|range| {
            source
                .schema
                .field_with_name(&range.bounds.column)
                .map_or(true, |field| {
                    field.is_nullable() || field.data_type() != &DataType::Int64
                })
        }) {
            return Ok(Transformed::no(plan));
        }
        let schema = Arc::clone(plan.schema());
        let node = ExactAggregateNode {
            request,
            source: source.clone(),
            fallback: plan,
            schema,
        };
        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })))
    }
}

#[derive(Debug)]
pub(crate) struct ExactAggregatePlanner;

#[async_trait]
impl ExtensionPlanner for ExactAggregatePlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<ExactAggregateNode>() else {
            return Ok(None);
        };
        Ok(Some(Arc::new(AdaptiveAggregateExec::new(
            node.request.clone(),
            node.source.clone(),
            node.fallback.clone(),
            session_state.clone(),
        ))))
    }
}

#[derive(Debug)]
struct AdaptiveAggregateExec {
    request: CountRequest,
    source: AggregateSource,
    fallback: LogicalPlan,
    session_state: SessionState,
    properties: Arc<PlanProperties>,
}

impl AdaptiveAggregateExec {
    fn new(
        request: CountRequest,
        source: AggregateSource,
        fallback: LogicalPlan,
        session_state: SessionState,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&request.schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            request,
            source,
            fallback,
            session_state,
            properties,
        }
    }
}

impl DisplayAs for AdaptiveAggregateExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "AdaptiveAggregateExec: table={}, column={}, segments={}, method=value_counts",
            self.request.table,
            self.request.column,
            self.source.segment_paths.len()
        )
    }
}

impl ExecutionPlan for AdaptiveAggregateExec {
    fn name(&self) -> &str {
        "AdaptiveAggregateExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::Plan(
                "AdaptiveAggregateExec has no physical children".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            self.request.clone(),
            self.source.clone(),
            self.fallback.clone(),
            self.session_state.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Execution(format!(
                "AdaptiveAggregateExec has no partition {partition}"
            )));
        }
        let request = self.request.clone();
        let source = self.source.clone();
        let fallback = self.fallback.clone();
        let session_state = self.session_state.clone();
        let future = async move {
            let plan = adaptive_physical_plan(request, source, fallback, &session_state).await?;
            let plan = if plan.output_partitioning().partition_count() == 1 {
                plan
            } else {
                Arc::new(CoalescePartitionsExec::new(plan))
            };
            plan.execute(0, context)
        };
        let stream = futures::stream::once(future).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

async fn adaptive_physical_plan(
    request: CountRequest,
    source: AggregateSource,
    fallback: LogicalPlan,
    session_state: &SessionState,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let classify_request = request.clone();
    let classify_source = source.clone();
    let coverage =
        tokio::task::spawn_blocking(move || classify_coverage(&classify_request, &classify_source))
            .await;
    let coverage = match coverage {
        Ok(Some(coverage)) if coverage.covered_segments > 0 => coverage,
        Ok(_) => {
            DECLINED_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
            return DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await;
        }
        Err(error) => {
            FALLBACK_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(%error, "value-count coverage task failed; using source aggregate");
            return DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await;
        }
    };
    let partial_coverage = !coverage.uncovered_paths.is_empty();
    let logical_plan = match indexed_aggregate_plan(&request, &source, coverage) {
        Ok(plan) => plan,
        Err(error) => {
            FALLBACK_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(%error, "could not plan value-count aggregate; using source aggregate");
            return DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await;
        }
    };
    match DefaultPhysicalPlanner::default()
        .create_physical_plan(&logical_plan, session_state)
        .await
    {
        Ok(plan) => {
            if partial_coverage {
                PARTIAL_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
            } else {
                FULL_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
            }
            Ok(plan)
        }
        Err(error) => {
            FALLBACK_INDEX_AGGREGATES.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(%error, "could not build value-count aggregate; using source aggregate");
            DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await
        }
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
        DefaultPhysicalPlanner::with_extension_planners(vec![
            Arc::new(ExactAggregatePlanner),
            Arc::new(crate::query::group_extrema::GroupExtremaPlanner),
        ])
        .create_physical_plan(logical_plan, session_state)
        .await
    }
}

/// Recognize the exact aggregate shape with either no predicate or one half-open
/// range on a non-null Int64 column. Other filters, distinct aggregates, joins,
/// and additional operations stay on the ordinary DataFusion path.
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
    let mut input = aggregate.input.as_ref();
    while let LogicalPlan::Projection(projection) = input {
        input = projection.input.as_ref();
    }
    let (scan, range) = match input {
        LogicalPlan::TableScan(scan) => (scan, None),
        LogicalPlan::Filter(filter) => {
            let mut scan_input = filter.input.as_ref();
            while let LogicalPlan::Projection(projection) = scan_input {
                scan_input = projection.input.as_ref();
            }
            let LogicalPlan::TableScan(scan) = scan_input else {
                return None;
            };
            let bounds = half_open_int_range(&filter.predicate)?;
            (
                scan,
                Some(CountRange {
                    bounds,
                    filter_expr: filter.predicate.clone(),
                }),
            )
        }
        _ => return None,
    };
    if scan.fetch.is_some() {
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
        range,
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

#[derive(Debug)]
struct AggregateCoverage {
    counts: BTreeMap<Option<String>, u64>,
    uncovered_paths: Vec<String>,
    covered_segments: usize,
}

/// Partition the snapshot into summarized and source-scanned segments.
///
/// Missing, stale, or malformed derived data is an uncovered segment, never a
/// query error. `None` declines the optimization when the combined summary is
/// too large or a count cannot be represented safely.
fn classify_coverage(
    request: &CountRequest,
    source: &AggregateSource,
) -> Option<AggregateCoverage> {
    let mut combined: BTreeMap<Option<String>, u64> = BTreeMap::new();
    let mut uncovered_paths = Vec::new();
    let mut covered_segments = 0;
    for path in &source.segment_paths {
        let parquet = Path::new(path);
        if let Some(range) = &request.range {
            let Some((_, Some(min), Some(max))) =
                segment_bounds(parquet, Some(&range.bounds.column))
            else {
                uncovered_paths.push(path.clone());
                continue;
            };
            let disjoint = max < range.bounds.lower || min >= range.bounds.upper;
            if disjoint {
                covered_segments += 1;
                continue;
            }
            let contained = min >= range.bounds.lower && max < range.bounds.upper;
            if !contained {
                uncovered_paths.push(path.clone());
                continue;
            }
        }
        let Some(segment) = source.index_cache.indexed_segment(parquet) else {
            uncovered_paths.push(path.clone());
            continue;
        };
        let Some(index) =
            source
                .index_cache
                .get_exact(parquet, &segment.header, SectionKind::ValueCounts)
        else {
            uncovered_paths.push(path.clone());
            continue;
        };
        let Some(counts) = index
            .columns
            .get(&request.column)
            .and_then(|column| column.counts.as_ref())
        else {
            uncovered_paths.push(path.clone());
            continue;
        };
        covered_segments += 1;
        for (value, count) in counts {
            let Some(total) = combined
                .get(value)
                .copied()
                .unwrap_or_default()
                .checked_add(*count)
            else {
                return None;
            };
            *combined.entry(value.clone()).or_default() = total;
            if combined.len() > MAX_COMBINED_COUNT_VALUES {
                return None;
            }
        }
    }
    Some(AggregateCoverage {
        counts: combined,
        uncovered_paths,
        covered_segments,
    })
}

fn indexed_count_batch(
    request: &CountRequest,
    combined: &BTreeMap<Option<String>, u64>,
) -> DFResult<RecordBatch> {
    let groups: Vec<Option<&str>> = combined.keys().map(|value| value.as_deref()).collect();
    let Some(counts): Option<Vec<i64>> = combined
        .iter()
        .map(|(value, count)| match (request.mode, value) {
            (CountMode::GroupingColumn, None) => Some(0),
            _ => i64::try_from(*count).ok(),
        })
        .collect()
    else {
        return Err(DataFusionError::Execution(
            "value-count summary exceeds SQL BIGINT".to_string(),
        ));
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        arrow::datatypes::Field::new(INDEX_GROUP_COLUMN, DataType::Utf8, true),
        arrow::datatypes::Field::new(INDEX_COUNT_COLUMN, DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(groups)) as ArrayRef,
            Arc::new(Int64Array::from(counts)) as ArrayRef,
        ],
    )
    .map_err(|error| DataFusionError::ArrowError(Box::new(error), None))
}

/// Build a standard DataFusion plan that merges indexed counts with an
/// aggregate over uncovered segments. The final aggregate also restores exact
/// SQL semantics when the same group appears in both inputs.
fn indexed_aggregate_plan(
    request: &CountRequest,
    source: &AggregateSource,
    coverage: AggregateCoverage,
) -> DFResult<LogicalPlan> {
    let covered_segments = coverage.covered_segments;
    let uncovered_segments = coverage.uncovered_paths.len();
    let batch = indexed_count_batch(request, &coverage.counts)?;
    let summary = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    let summary_plan = LogicalPlanBuilder::scan(
        "__finelog_index_summary",
        provider_as_source(Arc::new(summary)),
        None,
    )?
    .build()?;

    let union = if coverage.uncovered_paths.is_empty() {
        LogicalPlanBuilder::from(summary_plan)
    } else {
        let provider = NamespaceProvider::build(
            Arc::clone(&source.schema),
            &coverage.uncovered_paths,
            Arc::clone(&source.index_cache),
        )?;
        let count_expr = match request.mode {
            CountMode::AllRows => count(lit(1_i64)),
            CountMode::GroupingColumn => count(col(&request.column)),
        }
        .alias(INDEX_COUNT_COLUMN);
        let projection = fallback_projection(request, source)?;
        let mut uncovered = LogicalPlanBuilder::scan(
            "__finelog_uncovered_segments",
            provider_as_source(Arc::new(provider)),
            Some(projection),
        )?;
        if let Some(range) = &request.range {
            uncovered = uncovered.filter(range.filter_expr.clone())?;
        }
        let uncovered = uncovered
            .aggregate([col(&request.column)], [count_expr])?
            .project([
                Expr::Cast(Cast::new(Box::new(col(&request.column)), DataType::Utf8))
                    .alias(INDEX_GROUP_COLUMN),
                col(INDEX_COUNT_COLUMN),
            ])?
            .build()?;
        LogicalPlanBuilder::from(summary_plan).union(uncovered)?
    };

    let merged = union
        .aggregate(
            [col(INDEX_GROUP_COLUMN)],
            [sum(col(INDEX_COUNT_COLUMN)).alias(INDEX_TOTAL_COLUMN)],
        )?
        .build()?;
    let output = request.output.iter().enumerate().map(|(position, column)| {
        let source_column = match column {
            OutputColumn::Group => INDEX_GROUP_COLUMN,
            OutputColumn::Count => INDEX_TOTAL_COLUMN,
        };
        Expr::Cast(Cast::new(
            Box::new(col(source_column)),
            request.schema.field(position).data_type().clone(),
        ))
        .alias(request.schema.field(position).name())
    });
    let plan = LogicalPlanBuilder::from(merged).project(output)?.build()?;
    tracing::debug!(
        table = request.table,
        column = request.column,
        covered_segments,
        uncovered_segments,
        "planned value-count aggregate"
    );
    Ok(plan)
}

fn fallback_projection(request: &CountRequest, source: &AggregateSource) -> DFResult<Vec<usize>> {
    let mut projection = vec![source.schema.index_of(&request.column).map_err(|error| {
        DataFusionError::Plan(format!(
            "value-count fallback column {:?} is unavailable: {error}",
            request.column
        ))
    })?];
    if let Some(range) = &request.range {
        projection.push(
            source
                .schema
                .index_of(&range.bounds.column)
                .map_err(|error| {
                    DataFusionError::Plan(format!(
                        "value-count range column {:?} is unavailable: {error}",
                        range.bounds.column
                    ))
                })?,
        );
    }
    projection.sort_unstable();
    projection.dedup();
    Ok(projection)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, StringArray, StringViewArray};
    use arrow::datatypes::Field;
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

    #[tokio::test]
    async fn merges_indexed_counts_with_uncovered_segments() {
        let dir = crate::test_support::unique_dir("exact_aggregate");
        let source_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "service",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(StringArray::from(vec![
                Some("worker"),
                Some("worker"),
                Some("api"),
                None,
            ]))],
        )
        .unwrap();
        let (indexed, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        write_segment_index(
            &indexed,
            std::slice::from_ref(&batch),
            &SegmentIndexConfig::from_policies(Vec::<String>::new(), &[], &[], None)
                .with_adaptive_value_counts(["service"]),
        )
        .unwrap();
        let fresh_batch = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![Arc::new(StringArray::from(vec![
                Some("api"),
                Some("fresh"),
                None,
            ]))],
        )
        .unwrap();
        let (fresh, _) = write_segment_to_dir(&dir, 0, 10, &fresh_batch).unwrap();
        let paths = vec![
            indexed.to_string_lossy().into_owned(),
            fresh.to_string_lossy().into_owned(),
        ];
        let index_cache = Arc::new(IndexCache::new(16));
        let provider =
            NamespaceProvider::build(Arc::clone(&source_schema), &paths, Arc::clone(&index_cache))
                .unwrap();
        let ctx = crate::query::make_ctx();
        ctx.register_table("telemetry_v1", Arc::new(provider))
            .unwrap();
        ctx.add_optimizer_rule(Arc::new(ExactAggregateRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            AggregateSource {
                segment_paths: paths,
                index_cache,
                schema: source_schema,
            },
        )]))));
        let batches = ctx
            .sql("SELECT service, count(service) FROM telemetry_v1 GROUP BY service ORDER BY service NULLS FIRST")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let values = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let counts = batches[0]
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
        assert_eq!(
            rows,
            vec![
                (None, 0),
                (Some("api"), 2),
                (Some("fresh"), 1),
                (Some("worker"), 2),
            ]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn time_bounded_counts_scan_only_segments_that_cannot_use_whole_segment_counts() {
        let dir = crate::test_support::unique_dir("exact_aggregate_range");
        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("service", DataType::Utf8, false),
        ]));
        let make_batch = |timestamps: Vec<i64>, services: Vec<&str>| {
            RecordBatch::try_new(
                Arc::clone(&source_schema),
                vec![
                    Arc::new(Int64Array::from(timestamps)),
                    Arc::new(StringArray::from(services)),
                ],
            )
            .unwrap()
        };
        let index_config = SegmentIndexConfig::from_policies(Vec::<String>::new(), &[], &[], None)
            .with_adaptive_value_counts(["service"]);

        let contained_batch = make_batch(vec![10, 20, 25], vec!["worker", "worker", "api"]);
        let (contained, _) = write_segment_to_dir(&dir, 1, 1, &contained_batch).unwrap();
        write_segment_index(
            &contained,
            std::slice::from_ref(&contained_batch),
            &index_config,
        )
        .unwrap();

        let disjoint_batch = make_batch(vec![100], vec!["outside"]);
        let (disjoint, _) = write_segment_to_dir(&dir, 1, 10, &disjoint_batch).unwrap();

        let boundary_batch = make_batch(vec![5, 15], vec!["boundary-outside", "boundary-inside"]);
        let (boundary, _) = write_segment_to_dir(&dir, 1, 20, &boundary_batch).unwrap();
        write_segment_index(
            &boundary,
            std::slice::from_ref(&boundary_batch),
            &index_config,
        )
        .unwrap();

        let fresh_batch = make_batch(vec![18], vec!["fresh"]);
        let (fresh, _) = write_segment_to_dir(&dir, 0, 30, &fresh_batch).unwrap();

        let paths = [contained, disjoint, boundary, fresh]
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let index_cache = Arc::new(IndexCache::new(16));
        let provider =
            NamespaceProvider::build(Arc::clone(&source_schema), &paths, Arc::clone(&index_cache))
                .unwrap();
        let context = crate::query::make_ctx();
        context
            .register_table("telemetry_v1", Arc::new(provider))
            .unwrap();
        context.add_optimizer_rule(Arc::new(ExactAggregateRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            AggregateSource {
                segment_paths: paths,
                index_cache,
                schema: source_schema,
            },
        )]))));
        let batches = context
            .sql(
                "SELECT service, count(service) FROM telemetry_v1 \
                 WHERE timestamp_ms >= 10 AND timestamp_ms < 30 \
                 GROUP BY service ORDER BY service",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let services = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let counts = batches[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let rows = (0..services.len())
            .map(|row| (services.value(row), counts.value(row)))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            vec![
                ("api", 1),
                ("boundary-inside", 1),
                ("fresh", 1),
                ("worker", 2),
            ]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn exact_aggregate_is_visible_in_explain() {
        let dir = crate::test_support::unique_dir("exact_aggregate_explain");
        let source_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("service", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&source_schema),
            vec![
                Arc::new(Int64Array::from(vec![10, 20])),
                Arc::new(StringArray::from(vec!["worker", "api"])),
            ],
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
            Arc::clone(&source_schema),
            std::slice::from_ref(&path),
            crate::query::index_cache::test_index_cache(),
        )
        .unwrap();
        let ctx = crate::query::make_ctx();
        ctx.register_table("telemetry_v1", Arc::new(provider))
            .unwrap();
        ctx.add_optimizer_rule(Arc::new(ExactAggregateRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            AggregateSource {
                segment_paths: vec![path],
                index_cache: crate::query::index_cache::test_index_cache(),
                schema: source_schema,
            },
        )]))));
        let batches = ctx
            .sql(
                "EXPLAIN SELECT service, count(service) FROM telemetry_v1 \
                 WHERE timestamp_ms >= 10 AND timestamp_ms < 30 GROUP BY service",
            )
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
        assert_eq!(
            explain.matches("FinelogIndexAggregate").count(),
            1,
            "the optimizer must not wrap its hidden fallback repeatedly: {explain}"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
