//! Exact grouped `MAX(Int64)` from bounded per-segment JSON summaries.
//!
//! This recognizes a schema-declared query shape and substitutes summary rows
//! only for segments whose group extrema range is wholly inside or outside the
//! requested half-open window. Boundary, L0, remote-only, missing, or malformed
//! segments keep the ordinary DataFusion aggregate, and a final aggregate
//! merges both inputs exactly.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{DFSchemaRef, ScalarValue};
use datafusion::datasource::{provider_as_source, MemTable};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::execution::{SessionState, TaskContext};
use datafusion::functions_aggregate::expr_fn::max;
use datafusion::logical_expr::{
    col, Cast, Expr, Extension, LogicalPlan, LogicalPlanBuilder, Operator, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::TryStreamExt;

use crate::query::exact_aggregate::{
    record_declined, record_fallback, record_full, record_partial, AggregateSource,
};
use crate::query::provider::NamespaceProvider;
use crate::store::group_extrema::GroupExtremaConfig;

const MAX_COMBINED_GROUPS: usize = 16_384;
const INDEX_GROUP_COLUMN: &str = "__finelog_extrema_group";
const INDEX_MAX_COLUMN: &str = "__finelog_extrema_max";
const INDEX_TOTAL_COLUMN: &str = "__finelog_extrema_total";

#[derive(Debug, Clone)]
pub struct GroupExtremaRequest {
    pub table: String,
    filter_column: String,
    filter_value: String,
    json_column: String,
    json_key: String,
    extrema_column: String,
    lower: i64,
    upper: i64,
    filter_expr: Expr,
    group_expr: Expr,
    schema: SchemaRef,
}

impl GroupExtremaRequest {
    fn config(&self) -> GroupExtremaConfig {
        GroupExtremaConfig {
            filter_column: self.filter_column.clone(),
            json_column: self.json_column.clone(),
            json_key: self.json_key.clone(),
            extrema_column: self.extrema_column.clone(),
        }
    }
}

#[derive(Clone)]
struct GroupExtremaNode {
    request: GroupExtremaRequest,
    source: AggregateSource,
    fallback: LogicalPlan,
    schema: DFSchemaRef,
}

impl fmt::Debug for GroupExtremaNode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GroupExtremaNode")
            .field("table", &self.request.table)
            .field("json_key", &self.request.json_key)
            .field("segments", &self.source.segment_paths.len())
            .finish()
    }
}

impl PartialEq for GroupExtremaNode {
    fn eq(&self, other: &Self) -> bool {
        self.request.table == other.request.table
            && self.request.filter_column == other.request.filter_column
            && self.request.filter_value == other.request.filter_value
            && self.request.json_column == other.request.json_column
            && self.request.json_key == other.request.json_key
            && self.request.extrema_column == other.request.extrema_column
            && self.request.lower == other.request.lower
            && self.request.upper == other.request.upper
            && self.source.segment_paths == other.source.segment_paths
            && Arc::ptr_eq(&self.source.index_cache, &other.source.index_cache)
            && self.fallback == other.fallback
    }
}

impl Eq for GroupExtremaNode {}

impl PartialOrd for GroupExtremaNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (
            &self.request.table,
            &self.request.filter_column,
            &self.request.filter_value,
            &self.request.json_column,
            &self.request.json_key,
            &self.request.extrema_column,
            self.request.lower,
            self.request.upper,
            &self.source.segment_paths,
            Arc::as_ptr(&self.source.index_cache) as usize,
            &self.fallback,
        )
            .partial_cmp(&(
                &other.request.table,
                &other.request.filter_column,
                &other.request.filter_value,
                &other.request.json_column,
                &other.request.json_key,
                &other.request.extrema_column,
                other.request.lower,
                other.request.upper,
                &other.source.segment_paths,
                Arc::as_ptr(&other.source.index_cache) as usize,
                &other.fallback,
            ))
    }
}

impl Hash for GroupExtremaNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.request.table.hash(state);
        self.request.filter_column.hash(state);
        self.request.filter_value.hash(state);
        self.request.json_column.hash(state);
        self.request.json_key.hash(state);
        self.request.extrema_column.hash(state);
        self.request.lower.hash(state);
        self.request.upper.hash(state);
        self.source.segment_paths.hash(state);
        (Arc::as_ptr(&self.source.index_cache) as usize).hash(state);
        self.fallback.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for GroupExtremaNode {
    fn name(&self) -> &str {
        "FinelogGroupExtrema"
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

    fn fmt_for_explain(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "FinelogGroupExtrema: table={}, filter={}={}, group=json_get({}, {}), extrema={}, segments={}, coverage=runtime",
            self.request.table,
            self.request.filter_column,
            self.request.filter_value,
            self.request.json_column,
            self.request.json_key,
            self.request.extrema_column,
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
                "FinelogGroupExtrema has no optimizer-visible inputs".to_string(),
            ));
        }
        Ok(self.clone())
    }
}

#[derive(Debug)]
pub struct GroupExtremaRewrite {
    sources: HashMap<String, AggregateSource>,
}

impl GroupExtremaRewrite {
    pub fn new(sources: HashMap<String, AggregateSource>) -> Self {
        Self { sources }
    }
}

impl OptimizerRule for GroupExtremaRewrite {
    fn name(&self) -> &str {
        "finelog_group_extrema"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        let Some(request) = group_extrema_request(&plan) else {
            return Ok(Transformed::no(plan));
        };
        let Some(source) = self.sources.get(&request.table) else {
            return Ok(Transformed::no(plan));
        };
        let schema = Arc::clone(plan.schema());
        let node = GroupExtremaNode {
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
pub(crate) struct GroupExtremaPlanner;

#[async_trait]
impl ExtensionPlanner for GroupExtremaPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> DFResult<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<GroupExtremaNode>() else {
            return Ok(None);
        };
        Ok(Some(Arc::new(AdaptiveGroupExtremaExec::new(
            node.request.clone(),
            node.source.clone(),
            node.fallback.clone(),
            session_state.clone(),
        ))))
    }
}

#[derive(Debug)]
struct AdaptiveGroupExtremaExec {
    request: GroupExtremaRequest,
    source: AggregateSource,
    fallback: LogicalPlan,
    session_state: SessionState,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    covered_segments: Count,
    uncovered_segments: Count,
}

impl AdaptiveGroupExtremaExec {
    fn new(
        request: GroupExtremaRequest,
        source: AggregateSource,
        fallback: LogicalPlan,
        session_state: SessionState,
    ) -> Self {
        let metrics = ExecutionPlanMetricsSet::new();
        let covered_segments = MetricBuilder::new(&metrics).counter("covered_segments", 0);
        let uncovered_segments = MetricBuilder::new(&metrics).counter("uncovered_segments", 0);
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
            metrics,
            covered_segments,
            uncovered_segments,
        }
    }
}

impl DisplayAs for AdaptiveGroupExtremaExec {
    fn fmt_as(&self, _: DisplayFormatType, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "AdaptiveGroupExtremaExec: table={}, filter={}={}, group=json_get({}, {}), extrema={}, segments={}",
            self.request.table,
            self.request.filter_column,
            self.request.filter_value,
            self.request.json_column,
            self.request.json_key,
            self.request.extrema_column,
            self.source.segment_paths.len()
        )
    }
}

impl ExecutionPlan for AdaptiveGroupExtremaExec {
    fn name(&self) -> &str {
        "AdaptiveGroupExtremaExec"
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
                "AdaptiveGroupExtremaExec has no physical children".to_string(),
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
                "AdaptiveGroupExtremaExec has no partition {partition}"
            )));
        }
        let request = self.request.clone();
        let source = self.source.clone();
        let fallback = self.fallback.clone();
        let session_state = self.session_state.clone();
        let covered_segments = self.covered_segments.clone();
        let uncovered_segments = self.uncovered_segments.clone();
        let future = async move {
            let plan = adaptive_physical_plan(
                request,
                source,
                fallback,
                &session_state,
                covered_segments,
                uncovered_segments,
            )
            .await?;
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

async fn adaptive_physical_plan(
    request: GroupExtremaRequest,
    source: AggregateSource,
    fallback: LogicalPlan,
    session_state: &SessionState,
    covered_metric: Count,
    uncovered_metric: Count,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let classify_request = request.clone();
    let classify_source = source.clone();
    let coverage =
        tokio::task::spawn_blocking(move || classify_coverage(&classify_request, &classify_source))
            .await;
    let coverage = match coverage {
        Ok(Some(coverage)) => {
            covered_metric.add(coverage.covered_segments);
            uncovered_metric.add(coverage.uncovered_paths.len());
            if coverage.covered_segments > 0 {
                coverage
            } else {
                record_declined();
                return DefaultPhysicalPlanner::default()
                    .create_physical_plan(&fallback, session_state)
                    .await;
            }
        }
        Ok(None) => {
            uncovered_metric.add(source.segment_paths.len());
            record_declined();
            return DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await;
        }
        Err(error) => {
            record_fallback();
            tracing::warn!(%error, "grouped-extrema coverage task failed; using source aggregate");
            return DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await;
        }
    };
    let partial_coverage = !coverage.uncovered_paths.is_empty();
    let logical_plan = match indexed_extrema_plan(&request, &source, coverage) {
        Ok(plan) => plan,
        Err(error) => {
            record_fallback();
            tracing::warn!(%error, "could not plan grouped-extrema aggregate; using source aggregate");
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
                record_partial();
            } else {
                record_full();
            }
            Ok(plan)
        }
        Err(error) => {
            record_fallback();
            tracing::warn!(%error, "could not build grouped-extrema aggregate; using source aggregate");
            DefaultPhysicalPlanner::default()
                .create_physical_plan(&fallback, session_state)
                .await
        }
    }
}

#[derive(Debug)]
struct GroupExtremaCoverage {
    maxima: BTreeMap<String, i64>,
    uncovered_paths: Vec<String>,
    covered_segments: usize,
}

fn classify_coverage(
    request: &GroupExtremaRequest,
    source: &AggregateSource,
) -> Option<GroupExtremaCoverage> {
    let config = request.config();
    let mut maxima: BTreeMap<String, i64> = BTreeMap::new();
    let mut uncovered_paths = Vec::new();
    let mut covered_segments = 0;
    for path in &source.segment_paths {
        let parquet = Path::new(path);
        let Some(segment) = source.index_cache.indexed_segment(parquet) else {
            uncovered_paths.push(path.clone());
            continue;
        };
        let Some(index) = source
            .index_cache
            .get_group_extrema(parquet, &segment.header, &config)
        else {
            uncovered_paths.push(path.clone());
            continue;
        };
        let relevant = index
            .entries
            .iter()
            .filter(|entry| entry.filter_value == request.filter_value)
            .collect::<Vec<_>>();
        let boundary = relevant.iter().any(|entry| {
            let disjoint = entry.max < request.lower || entry.min >= request.upper;
            let contained = entry.min >= request.lower && entry.max < request.upper;
            !disjoint && !contained
        });
        if boundary {
            uncovered_paths.push(path.clone());
            continue;
        }
        covered_segments += 1;
        for entry in relevant {
            if entry.min < request.lower || entry.max >= request.upper {
                continue;
            }
            maxima
                .entry(entry.group_value.clone())
                .and_modify(|value| *value = (*value).max(entry.max))
                .or_insert(entry.max);
            if maxima.len() > MAX_COMBINED_GROUPS {
                return None;
            }
        }
    }
    Some(GroupExtremaCoverage {
        maxima,
        uncovered_paths,
        covered_segments,
    })
}

fn indexed_extrema_batch(maxima: &BTreeMap<String, i64>) -> DFResult<RecordBatch> {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new(INDEX_GROUP_COLUMN, DataType::Utf8, false),
        Field::new(INDEX_MAX_COLUMN, DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(maxima.keys())) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(maxima.values().copied())) as ArrayRef,
        ],
    )
    .map_err(|error| DataFusionError::ArrowError(Box::new(error), None))
}

fn indexed_extrema_plan(
    request: &GroupExtremaRequest,
    source: &AggregateSource,
    coverage: GroupExtremaCoverage,
) -> DFResult<LogicalPlan> {
    let covered_segments = coverage.covered_segments;
    let uncovered_segments = coverage.uncovered_paths.len();
    let batch = indexed_extrema_batch(&coverage.maxima)?;
    let summary = MemTable::try_new(batch.schema(), vec![vec![batch]])?;
    let summary_plan = LogicalPlanBuilder::scan(
        "__finelog_extrema_summary",
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
        let projection = fallback_projection(request, source)?;
        let uncovered = LogicalPlanBuilder::scan(
            request.table.clone(),
            provider_as_source(Arc::new(provider)),
            Some(projection),
        )?
        .filter(request.filter_expr.clone())?
        .aggregate(
            [request.group_expr.clone()],
            [max(col(&request.extrema_column)).alias(INDEX_MAX_COLUMN)],
        )?
        .project([
            Expr::Cast(Cast::new(
                Box::new(col(request.schema.field(0).name())),
                DataType::Utf8,
            ))
            .alias(INDEX_GROUP_COLUMN),
            col(INDEX_MAX_COLUMN),
        ])?
        .build()?;
        LogicalPlanBuilder::from(summary_plan).union(uncovered)?
    };
    let merged = union
        .aggregate(
            [col(INDEX_GROUP_COLUMN)],
            [max(col(INDEX_MAX_COLUMN)).alias(INDEX_TOTAL_COLUMN)],
        )?
        .project([
            Expr::Cast(Cast::new(
                Box::new(col(INDEX_GROUP_COLUMN)),
                request.schema.field(0).data_type().clone(),
            ))
            .alias(request.schema.field(0).name()),
            Expr::Cast(Cast::new(
                Box::new(col(INDEX_TOTAL_COLUMN)),
                request.schema.field(1).data_type().clone(),
            ))
            .alias(request.schema.field(1).name()),
        ])?
        .build()?;
    tracing::debug!(
        table = request.table,
        json_key = request.json_key,
        covered_segments,
        uncovered_segments,
        "planned grouped-extrema aggregate"
    );
    Ok(merged)
}

fn fallback_projection(
    request: &GroupExtremaRequest,
    source: &AggregateSource,
) -> DFResult<Vec<usize>> {
    let mut projection = [
        request.filter_column.as_str(),
        request.json_column.as_str(),
        request.extrema_column.as_str(),
    ]
    .into_iter()
    .map(|column| {
        source.schema.index_of(column).map_err(|error| {
            DataFusionError::Plan(format!(
                "grouped-extrema fallback column {column:?} is unavailable: {error}"
            ))
        })
    })
    .collect::<DFResult<Vec<_>>>()?;
    projection.sort_unstable();
    projection.dedup();
    Ok(projection)
}

pub fn group_extrema_request(plan: &LogicalPlan) -> Option<GroupExtremaRequest> {
    let LogicalPlan::Aggregate(aggregate) = plan else {
        return None;
    };
    if aggregate.group_expr.len() != 1 || aggregate.aggr_expr.len() != 1 {
        return None;
    }
    let (json_column, json_key) = json_get(&aggregate.group_expr[0])?;
    let Expr::AggregateFunction(function) = &aggregate.aggr_expr[0] else {
        return None;
    };
    if !function.func.name().eq_ignore_ascii_case("max")
        || function.params.distinct
        || function.params.filter.is_some()
        || !function.params.order_by.is_empty()
        || function.params.args.len() != 1
    {
        return None;
    }
    let Expr::Column(extrema) = &function.params.args[0] else {
        return None;
    };
    let mut input = aggregate.input.as_ref();
    while let LogicalPlan::Projection(projection) = input {
        input = projection.input.as_ref();
    }
    let LogicalPlan::Filter(filter) = input else {
        return None;
    };
    let mut scan_input = filter.input.as_ref();
    while let LogicalPlan::Projection(projection) = scan_input {
        scan_input = projection.input.as_ref();
    }
    let LogicalPlan::TableScan(scan) = scan_input else {
        return None;
    };
    if scan.fetch.is_some() {
        return None;
    }
    let mut terms = Vec::new();
    conjuncts(&filter.predicate, &mut terms);
    let mut filter_pair = None;
    let mut lower = None;
    let mut upper = None;
    let mut has_not_null = false;
    for term in terms {
        if is_json_not_null(term, &aggregate.group_expr[0]) {
            if has_not_null {
                return None;
            }
            has_not_null = true;
            continue;
        }
        if let Some((column, value)) = string_equality(term) {
            if filter_pair.replace((column, value)).is_some() {
                return None;
            }
            continue;
        }
        if let Some((column, operator, value)) = int_comparison(term) {
            if column != extrema.name {
                return None;
            }
            match operator {
                Operator::GtEq if lower.replace(value).is_none() => {}
                Operator::Lt if upper.replace(value).is_none() => {}
                _ => return None,
            }
            continue;
        }
        return None;
    }
    let (filter_column, filter_value) = filter_pair?;
    let (lower, upper) = (lower?, upper?);
    if !has_not_null || lower >= upper {
        return None;
    }
    let schema = Arc::new(plan.schema().as_arrow().clone());
    if schema.fields().len() != 2 {
        return None;
    }
    Some(GroupExtremaRequest {
        table: scan.table_name.table().to_string(),
        filter_column,
        filter_value,
        json_column,
        json_key,
        extrema_column: extrema.name.clone(),
        lower,
        upper,
        filter_expr: filter.predicate.clone(),
        group_expr: aggregate.group_expr[0].clone(),
        schema,
    })
}

fn conjuncts<'a>(expr: &'a Expr, output: &mut Vec<&'a Expr>) {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::And => {
            conjuncts(&binary.left, output);
            conjuncts(&binary.right, output);
        }
        expr => output.push(expr),
    }
}

fn json_get(expr: &Expr) -> Option<(String, String)> {
    let Expr::ScalarFunction(function) = expr else {
        return None;
    };
    if function.func.name() != "json_get" || function.args.len() != 2 {
        return None;
    }
    let Expr::Column(column) = &function.args[0] else {
        return None;
    };
    Some((column.name.clone(), string_literal(&function.args[1])?))
}

fn is_json_not_null(expr: &Expr, group: &Expr) -> bool {
    matches!(expr, Expr::IsNotNull(inner) if inner.as_ref() == group)
}

fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(value)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(value)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(value)), _) => Some(value.clone()),
        Expr::Cast(cast) => string_literal(&cast.expr),
        _ => None,
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(value)), _) => Some(*value),
        Expr::Cast(cast) => int_literal(&cast.expr),
        _ => None,
    }
}

fn string_equality(expr: &Expr) -> Option<(String, String)> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };
    if binary.op != Operator::Eq {
        return None;
    }
    match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Column(column), literal) => Some((column.name.clone(), string_literal(literal)?)),
        (literal, Expr::Column(column)) => Some((column.name.clone(), string_literal(literal)?)),
        _ => None,
    }
}

fn int_comparison(expr: &Expr) -> Option<(String, Operator, i64)> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };
    match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Column(column), literal) => {
            Some((column.name.clone(), binary.op, int_literal(literal)?))
        }
        (literal, Expr::Column(column)) => {
            let operator = match binary.op {
                Operator::Lt => Operator::Gt,
                Operator::LtEq => Operator::GtEq,
                Operator::Gt => Operator::Lt,
                Operator::GtEq => Operator::LtEq,
                _ => return None,
            };
            Some((column.name.clone(), operator, int_literal(literal)?))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::index_cache::IndexCache;
    use crate::query::string_values::StringValues;
    use crate::store::segment::write_segment_to_dir;
    use crate::store::segment_index::{write_segment_index, SegmentIndexConfig};
    use arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;

    fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
        let local = plan
            .metrics()
            .and_then(|metrics| metrics.sum_by_name(name))
            .map(|value| value.as_usize())
            .unwrap_or_default();
        local
            + plan
                .children()
                .into_iter()
                .map(|child| metric(child.as_ref(), name))
                .sum::<usize>()
    }

    fn source_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("service", DataType::Utf8, false),
            Field::new("resource_attributes_json", DataType::Utf8, true),
        ]))
    }

    fn batch(rows: &[(i64, &str, Option<&str>)]) -> RecordBatch {
        RecordBatch::try_new(
            source_schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
                Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
                Arc::new(StringArray::from(
                    rows.iter().map(|row| row.2).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    fn config() -> GroupExtremaConfig {
        GroupExtremaConfig {
            filter_column: "service".to_string(),
            json_column: "resource_attributes_json".to_string(),
            json_key: "job_id".to_string(),
            extrema_column: "timestamp_ms".to_string(),
        }
    }

    const SQL: &str = "SELECT json_get(resource_attributes_json, 'job_id') AS job_id \
        FROM telemetry_v1 \
        WHERE service = 'levanter' \
          AND timestamp_ms >= 0 AND timestamp_ms < 40 \
          AND json_get(resource_attributes_json, 'job_id') IS NOT NULL \
        GROUP BY 1 ORDER BY max(timestamp_ms) DESC";

    #[tokio::test]
    async fn rewrite_is_visible_in_explain() {
        let schema = source_schema();
        let table = MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap();
        let context = crate::query::make_ctx();
        context
            .register_table("telemetry_v1", Arc::new(table))
            .unwrap();
        context.add_optimizer_rule(Arc::new(GroupExtremaRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            AggregateSource {
                segment_paths: Vec::new(),
                index_cache: Arc::new(IndexCache::new(16)),
                schema,
            },
        )]))));
        let plan = context
            .sql(&format!("EXPLAIN {SQL}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let rendered = datafusion::arrow::util::pretty::pretty_format_batches(&plan)
            .unwrap()
            .to_string();
        assert!(rendered.contains("FinelogGroupExtrema"), "{rendered}");
    }

    #[tokio::test]
    async fn merges_indexed_extrema_with_uncovered_segments() {
        let directory = crate::test_support::unique_dir("group_extrema_query");
        let stable = batch(&[
            (10, "levanter", Some(r#"{"job_id":"a"}"#)),
            (30, "levanter", Some(r#"{"job_id":"a"}"#)),
            (20, "levanter", Some(r#"{"job_id":"b"}"#)),
            (25, "vllm", Some(r#"{"job_id":"ignored"}"#)),
        ]);
        let (indexed, _) = write_segment_to_dir(&directory, 1, 1, &stable).unwrap();
        write_segment_index(
            &indexed,
            std::slice::from_ref(&stable),
            &SegmentIndexConfig::from_policies(Vec::<String>::new(), &[], &[], None)
                .with_adaptive_group_extrema([config()]),
        )
        .unwrap();
        let fresh = batch(&[
            (35, "levanter", Some(r#"{"job_id":"a"}"#)),
            (25, "levanter", Some(r#"{"job_id":"c"}"#)),
        ]);
        let (uncovered, _) = write_segment_to_dir(&directory, 0, 10, &fresh).unwrap();
        let paths = vec![
            indexed.to_string_lossy().into_owned(),
            uncovered.to_string_lossy().into_owned(),
        ];
        let schema = source_schema();
        let index_cache = Arc::new(IndexCache::new(16));
        let provider =
            NamespaceProvider::build(Arc::clone(&schema), &paths, Arc::clone(&index_cache))
                .unwrap();
        let context = crate::query::make_ctx();
        context
            .register_table("telemetry_v1", Arc::new(provider))
            .unwrap();
        context.add_optimizer_rule(Arc::new(GroupExtremaRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            AggregateSource {
                segment_paths: paths,
                index_cache,
                schema,
            },
        )]))));
        let dataframe = context.sql(SQL).await.unwrap();
        let plan = dataframe.create_physical_plan().await.unwrap();
        let batches = datafusion::physical_plan::collect(Arc::clone(&plan), context.task_ctx())
            .await
            .unwrap();
        let groups = StringValues::new(batches[0].column(0)).unwrap();
        let rows = (0..batches[0].num_rows())
            .map(|row| groups.value(row))
            .collect::<Vec<_>>();
        assert_eq!(rows, vec!["a", "c", "b"]);
        assert_eq!(metric(plan.as_ref(), "covered_segments"), 1);
        assert_eq!(metric(plan.as_ref(), "uncovered_segments"), 1);
        std::fs::remove_dir_all(directory).ok();
    }

    #[tokio::test]
    async fn boundary_segment_uses_the_source_aggregate() {
        let directory = crate::test_support::unique_dir("group_extrema_boundary");
        let stable = batch(&[
            (10, "levanter", Some(r#"{"job_id":"a"}"#)),
            (30, "levanter", Some(r#"{"job_id":"a"}"#)),
            (20, "levanter", Some(r#"{"job_id":"b"}"#)),
        ]);
        let (indexed, _) = write_segment_to_dir(&directory, 1, 1, &stable).unwrap();
        write_segment_index(
            &indexed,
            std::slice::from_ref(&stable),
            &SegmentIndexConfig::from_policies(Vec::<String>::new(), &[], &[], None)
                .with_adaptive_group_extrema([config()]),
        )
        .unwrap();
        let paths = vec![indexed.to_string_lossy().into_owned()];
        let schema = source_schema();
        let index_cache = Arc::new(IndexCache::new(16));
        let provider =
            NamespaceProvider::build(Arc::clone(&schema), &paths, Arc::clone(&index_cache))
                .unwrap();
        let context = crate::query::make_ctx();
        context
            .register_table("telemetry_v1", Arc::new(provider))
            .unwrap();
        context.add_optimizer_rule(Arc::new(GroupExtremaRewrite::new(HashMap::from([(
            "telemetry_v1".to_string(),
            AggregateSource {
                segment_paths: paths,
                index_cache,
                schema,
            },
        )]))));
        let sql = SQL
            .replace("timestamp_ms >= 0", "timestamp_ms >= 15")
            .replace("timestamp_ms < 40", "timestamp_ms < 25");
        let dataframe = context.sql(&sql).await.unwrap();
        let plan = dataframe.create_physical_plan().await.unwrap();
        let batches = datafusion::physical_plan::collect(Arc::clone(&plan), context.task_ctx())
            .await
            .unwrap();
        let groups = StringValues::new(batches[0].column(0)).unwrap();
        let rows = (0..batches[0].num_rows())
            .map(|row| groups.value(row))
            .collect::<Vec<_>>();
        assert_eq!(rows, vec!["b"]);
        assert_eq!(metric(plan.as_ref(), "covered_segments"), 0);
        assert_eq!(metric(plan.as_ref(), "uncovered_segments"), 1);
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn fallback_reads_only_query_columns() {
        let request = GroupExtremaRequest {
            table: "telemetry_v1".to_string(),
            filter_column: "service".to_string(),
            filter_value: "levanter".to_string(),
            json_column: "resource_attributes_json".to_string(),
            json_key: "job_id".to_string(),
            extrema_column: "timestamp_ms".to_string(),
            lower: 0,
            upper: 1,
            filter_expr: col("service").eq(Expr::Literal(
                ScalarValue::Utf8(Some("levanter".to_string())),
                None,
            )),
            group_expr: col("resource_attributes_json"),
            schema: Arc::new(ArrowSchema::empty()),
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("unused", DataType::Utf8, true),
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("resource_attributes_json", DataType::Utf8, true),
            Field::new("service", DataType::Utf8, false),
        ]));
        let source = AggregateSource {
            segment_paths: Vec::new(),
            index_cache: Arc::new(IndexCache::new(16)),
            schema,
        };
        assert_eq!(
            fallback_projection(&request, &source).unwrap(),
            vec![1, 2, 3]
        );
    }
}
