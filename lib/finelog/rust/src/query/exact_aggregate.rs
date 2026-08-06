//! Exact `GROUP BY string_column, COUNT(...)` from segment sidecars.
//!
//! The fast path is deliberately narrow. It recognizes one unfiltered table
//! scan, one string grouping column, and one ordinary `COUNT(*)` or
//! `COUNT(grouping_column)`. Any extra relational operation falls back to
//! DataFusion. Likewise, every visible segment must have a complete count
//! summary; partial backfill never contributes a partial answer.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{Expr, LogicalPlan};

use crate::query::QueryResult;
use crate::store::exact::{read_sidecar, sidecar_path};
use crate::store::segment::segment_row_group_rows;

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

/// Recognize the exact aggregate shape without accepting semantically richer
/// plans such as filters, sorts, limits, distinct aggregates, or joins.
pub fn count_request(plan: &LogicalPlan) -> Option<CountRequest> {
    let (aggregate, output) = match plan {
        LogicalPlan::Projection(projection) => {
            let LogicalPlan::Aggregate(aggregate) = projection.input.as_ref() else {
                return None;
            };
            if projection.expr.len() != 2 {
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
        Expr::Literal(_, _) => CountMode::AllRows,
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
        let Some(sidecar) = read_sidecar(&sidecar_path(parquet)) else {
            return Ok(None);
        };
        let Some(row_groups) = segment_row_group_rows(parquet) else {
            return Ok(None);
        };
        let parquet_rows = row_groups.iter().sum::<usize>() as u64;
        if sidecar.total_rows != parquet_rows {
            return Ok(None);
        }
        let Some(counts) = sidecar
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
        .to_vec();
    let fields: Vec<Field> = request
        .schema
        .fields()
        .iter()
        .map(|field| {
            let field = field.as_ref().clone().with_nullable(true);
            match field.data_type() {
                DataType::Utf8View | DataType::LargeUtf8 => field.with_data_type(DataType::Utf8),
                _ => field,
            }
        })
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
        "answered value-count aggregate from exact sidecars"
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
    use crate::store::exact::{write_sidecar, ExactIndexConfig};
    use crate::store::segment::write_segment_to_dir;

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
        write_sidecar(&parquet, &[batch], &[config]).unwrap();
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

        std::fs::remove_file(sidecar_path(&parquet)).unwrap();
        assert!(execute(&request, &paths).unwrap().is_none());
        std::fs::remove_dir_all(dir).ok();
    }
}
