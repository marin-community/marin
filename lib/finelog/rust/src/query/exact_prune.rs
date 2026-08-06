//! Filtered projections and exact row pruning for string equality/IN predicates.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::{Expr, Operator};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_datasource_parquet::{ParquetAccessPlan, RowGroupAccess};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

use crate::store::exact::{projection_path, read_sidecar, sidecar_path, RowRun};
use crate::store::segment::segment_row_group_rows;

/// Exact string values implied for each column by top-level `=` and `IN`
/// conjuncts. Multiple constraints on one column are intersected.
pub fn values_by_column(filters: &[Expr]) -> HashMap<String, Vec<String>> {
    let mut constraints: HashMap<String, BTreeSet<String>> = HashMap::new();
    for filter in filters {
        visit_conjunct(filter, &mut |column, values| {
            constraints
                .entry(column)
                .and_modify(|current| current.retain(|value| values.contains(value)))
                .or_insert(values);
        });
    }
    constraints
        .into_iter()
        .map(|(column, values)| (column, values.into_iter().collect()))
        .collect()
}

fn visit_conjunct(expr: &Expr, visit: &mut impl FnMut(String, BTreeSet<String>)) {
    if let Some((column, values)) = exact_disjunction(expr) {
        visit(column, values);
        return;
    }
    if let Expr::BinaryExpr(binary) = expr {
        if binary.op == Operator::And {
            visit_conjunct(&binary.left, visit);
            visit_conjunct(&binary.right, visit);
            return;
        }
    }
    if let Expr::InList(list) = expr {
        if list.negated {
            return;
        }
        let Some(column) = column_name(&list.expr) else {
            return;
        };
        let values: Option<BTreeSet<String>> = list.list.iter().map(string_literal).collect();
        if let Some(values) = values {
            visit(column, values);
        }
    }
}

fn exact_disjunction(expr: &Expr) -> Option<(String, BTreeSet<String>)> {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::Eq => {
            let (column, value) = column_literal(&binary.left, &binary.right)
                .or_else(|| column_literal(&binary.right, &binary.left))?;
            Some((column, BTreeSet::from([value])))
        }
        Expr::BinaryExpr(binary) if binary.op == Operator::Or => {
            let (left_column, mut left_values) = exact_disjunction(&binary.left)?;
            let (right_column, right_values) = exact_disjunction(&binary.right)?;
            if left_column != right_column {
                return None;
            }
            left_values.extend(right_values);
            Some((left_column, left_values))
        }
        _ => None,
    }
}

fn column_literal(column: &Expr, literal: &Expr) -> Option<(String, String)> {
    Some((column_name(column)?, string_literal(literal)?))
}

fn column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name.clone()),
        Expr::Cast(cast) => column_name(&cast.expr),
        _ => None,
    }
}

fn string_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(value)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(value)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(value)), _) => Some(value.clone()),
        Expr::Cast(cast) => string_literal(&cast.expr),
        _ => None,
    }
}

/// Substitute complete filtered projections when available for every segment.
/// Otherwise attach exact row selections to individual source files; existing
/// trigram access plans are intersected rather than replaced.
pub fn apply(
    plan: Arc<dyn ExecutionPlan>,
    segment_paths: &[String],
    constraints: &HashMap<String, Vec<String>>,
) -> Arc<dyn ExecutionPlan> {
    if constraints.is_empty() {
        return plan;
    }
    if let Some(projections) = build_projection_files(segment_paths, constraints) {
        return rewrite_projection_files(plan, &projections);
    }
    let access_plans = build_access_plans(segment_paths, constraints);
    if access_plans.is_empty() {
        return plan;
    }
    rewrite_file_groups(plan, &access_plans)
}

#[derive(Debug)]
struct ProjectionFile {
    location: object_store::path::Path,
    size: u64,
    modified: std::time::SystemTime,
}

fn build_projection_files(
    segment_paths: &[String],
    constraints: &HashMap<String, Vec<String>>,
) -> Option<HashMap<String, ProjectionFile>> {
    let mut projections = HashMap::new();
    let mut projected_rows = 0_u64;
    let mut total_rows = 0_u64;
    for segment in segment_paths {
        let parquet = Path::new(segment);
        let basename = parquet.file_name()?.to_str()?;
        let sidecar = read_sidecar(&sidecar_path(parquet))?;
        let rows = segment_row_group_rows(parquet)?;
        if rows.iter().sum::<usize>() as u64 != sidecar.total_rows {
            return None;
        }
        let projection_rows = sidecar.projection_rows?;
        let supported = constraints.iter().any(|(column_name, values)| {
            sidecar
                .columns
                .get(column_name)
                .is_some_and(|column| values.iter().all(|value| column.rows.contains_key(value)))
        });
        if !supported {
            return None;
        }
        let projection = projection_path(parquet);
        let projection_groups = segment_row_group_rows(&projection)?;
        if projection_groups.iter().sum::<usize>() as u64 != projection_rows {
            return None;
        }
        let metadata = std::fs::metadata(&projection).ok()?;
        projections.insert(
            basename.to_string(),
            ProjectionFile {
                location: object_store::path::Path::from_filesystem_path(&projection).ok()?,
                size: metadata.len(),
                modified: metadata.modified().ok()?,
            },
        );
        projected_rows += projection_rows;
        total_rows += sidecar.total_rows;
    }
    if !projections.is_empty() {
        tracing::debug!(
            segments_projected = projections.len(),
            projected_rows,
            total_rows,
            "exact-value filtered projection"
        );
    }
    Some(projections)
}

fn build_access_plans(
    segment_paths: &[String],
    constraints: &HashMap<String, Vec<String>>,
) -> HashMap<String, ParquetAccessPlan> {
    let mut plans = HashMap::new();
    let mut selected_rows = 0_u64;
    let mut total_rows = 0_u64;
    for segment in segment_paths {
        let parquet = Path::new(segment);
        let Some(basename) = parquet.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(sidecar) = read_sidecar(&sidecar_path(parquet)) else {
            continue;
        };
        let Some(row_group_rows) = segment_row_group_rows(parquet) else {
            continue;
        };
        if row_group_rows.iter().sum::<usize>() as u64 != sidecar.total_rows {
            tracing::warn!(
                segment = basename,
                sidecar_rows = sidecar.total_rows,
                "stale exact sidecar; scanning unpruned"
            );
            continue;
        }
        let mut selected: Option<Vec<RowRun>> = None;
        for (column_name, values) in constraints {
            let Some(column) = sidecar.columns.get(column_name) else {
                continue;
            };
            if !values.iter().all(|value| column.rows.contains_key(value)) {
                continue;
            }
            let union = merge_runs(
                values
                    .iter()
                    .flat_map(|value| column.rows[value].iter().copied())
                    .collect(),
            );
            selected = Some(match selected {
                None => union,
                Some(current) => intersect_runs(&current, &union),
            });
        }
        let Some(selected) = selected else {
            continue;
        };
        selected_rows += selected.iter().map(|run| run.len).sum::<u64>();
        total_rows += sidecar.total_rows;
        plans.insert(
            basename.to_string(),
            runs_access_plan(&selected, &row_group_rows),
        );
    }
    if !plans.is_empty() {
        tracing::debug!(
            segments_pruned = plans.len(),
            selected_rows,
            total_rows,
            "exact-value prune"
        );
    }
    plans
}

fn merge_runs(mut runs: Vec<RowRun>) -> Vec<RowRun> {
    runs.sort_by_key(|run| run.start);
    let mut merged: Vec<RowRun> = Vec::with_capacity(runs.len());
    for run in runs {
        match merged.last_mut() {
            Some(previous) if run.start <= previous.start + previous.len => {
                let end = (run.start + run.len).max(previous.start + previous.len);
                previous.len = end - previous.start;
            }
            _ => merged.push(run),
        }
    }
    merged
}

fn intersect_runs(left: &[RowRun], right: &[RowRun]) -> Vec<RowRun> {
    let mut out = Vec::new();
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        let left_run = left[left_index];
        let right_run = right[right_index];
        let start = left_run.start.max(right_run.start);
        let end = (left_run.start + left_run.len).min(right_run.start + right_run.len);
        if start < end {
            out.push(RowRun {
                start,
                len: end - start,
            });
        }
        if left_run.start + left_run.len < right_run.start + right_run.len {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    out
}

fn runs_access_plan(runs: &[RowRun], row_group_rows: &[usize]) -> ParquetAccessPlan {
    let mut plan = ParquetAccessPlan::new_all(row_group_rows.len());
    let mut group_start = 0_u64;
    let mut run_index = 0;
    for (group_index, &rows) in row_group_rows.iter().enumerate() {
        let group_end = group_start + rows as u64;
        while run_index < runs.len() && runs[run_index].start + runs[run_index].len <= group_start {
            run_index += 1;
        }
        let mut selectors = Vec::new();
        let mut cursor = group_start;
        let mut index = run_index;
        while index < runs.len() && runs[index].start < group_end {
            let start = runs[index].start.max(group_start);
            let end = (runs[index].start + runs[index].len).min(group_end);
            if cursor < start {
                selectors.push(RowSelector::skip((start - cursor) as usize));
            }
            if start < end {
                selectors.push(RowSelector::select((end - start) as usize));
                cursor = end;
            }
            index += 1;
        }
        if cursor < group_end {
            selectors.push(RowSelector::skip((group_end - cursor) as usize));
        }
        if selectors.iter().all(|selector| selector.skip) {
            plan.skip(group_index);
        } else if selectors.len() != 1 || selectors[0].skip {
            plan.scan_selection(group_index, RowSelection::from(selectors));
        }
        group_start = group_end;
    }
    plan
}

fn intersect_access_plans(
    existing: &ParquetAccessPlan,
    additional: &ParquetAccessPlan,
) -> Option<ParquetAccessPlan> {
    if existing.inner().len() != additional.inner().len() {
        return None;
    }
    let mut combined = existing.clone();
    for (index, access) in additional.inner().iter().enumerate() {
        match access {
            RowGroupAccess::Skip => combined.skip(index),
            RowGroupAccess::Scan => {}
            RowGroupAccess::Selection(selection) => {
                combined.scan_selection(index, selection.clone());
            }
        }
    }
    Some(combined)
}

fn rewrite_file_groups(
    plan: Arc<dyn ExecutionPlan>,
    access_plans: &HashMap<String, ParquetAccessPlan>,
) -> Arc<dyn ExecutionPlan> {
    let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return plan;
    };
    let Some(config) = exec.data_source().as_any().downcast_ref::<FileScanConfig>() else {
        return plan;
    };
    let groups = config
        .file_groups
        .iter()
        .map(|group| {
            let files = group
                .files()
                .iter()
                .map(|file| {
                    let Some(additional) = file
                        .object_meta
                        .location
                        .filename()
                        .and_then(|name| access_plans.get(name))
                    else {
                        return file.clone();
                    };
                    let access = file
                        .extensions
                        .as_ref()
                        .and_then(|extension| extension.downcast_ref::<ParquetAccessPlan>())
                        .and_then(|existing| intersect_access_plans(existing, additional))
                        .unwrap_or_else(|| additional.clone());
                    file.clone().with_extensions(Arc::new(access))
                })
                .collect();
            FileGroup::new(files)
        })
        .collect();
    let config = FileScanConfigBuilder::from(config.clone())
        .with_file_groups(groups)
        .build();
    DataSourceExec::from_data_source(config)
}

fn rewrite_projection_files(
    plan: Arc<dyn ExecutionPlan>,
    projections: &HashMap<String, ProjectionFile>,
) -> Arc<dyn ExecutionPlan> {
    let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return plan;
    };
    let Some(config) = exec.data_source().as_any().downcast_ref::<FileScanConfig>() else {
        return plan;
    };
    let groups = config
        .file_groups
        .iter()
        .map(|group| {
            let files = group
                .files()
                .iter()
                .map(|file| {
                    let Some(projection) = file
                        .object_meta
                        .location
                        .filename()
                        .and_then(|name| projections.get(name))
                    else {
                        return file.clone();
                    };
                    let mut projected = file.clone();
                    projected.object_meta.location = projection.location.clone();
                    projected.object_meta.size = projection.size;
                    projected.object_meta.last_modified = projection.modified.into();
                    projected.object_meta.e_tag = None;
                    projected.object_meta.version = None;
                    projected.range = None;
                    projected.statistics = None;
                    projected.ordering = None;
                    projected.extensions = None;
                    projected.metadata_size_hint = None;
                    projected
                })
                .collect();
            FileGroup::new(files)
        })
        .collect();
    let config = FileScanConfigBuilder::from(config.clone())
        .with_file_groups(groups)
        .build();
    DataSourceExec::from_data_source(config)
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::{col, lit};

    use super::*;

    #[test]
    fn extracts_and_intersects_only_top_level_exact_constraints() {
        let filters = vec![
            col("name").in_list(vec![lit("phase"), lit("step")], false),
            col("name").eq(lit("step")),
            col("other").eq(lit("x")).or(col("other").eq(lit("y"))),
        ];
        assert_eq!(
            values_by_column(&filters),
            HashMap::from([
                ("name".to_string(), vec!["step".to_string()]),
                ("other".to_string(), vec!["x".to_string(), "y".to_string()],),
            ])
        );
    }

    #[test]
    fn extracts_same_column_disjunctions() {
        let filters = vec![col("name")
            .eq(lit("phase"))
            .or(lit("step").eq(col("name")))
            .or(col("name").eq(lit("progress_time_seconds")))];
        assert_eq!(
            values_by_column(&filters),
            HashMap::from([(
                "name".to_string(),
                vec![
                    "phase".to_string(),
                    "progress_time_seconds".to_string(),
                    "step".to_string(),
                ],
            )])
        );
    }

    #[test]
    fn ignores_mixed_column_disjunctions() {
        let filters = vec![col("name")
            .eq(lit("phase"))
            .or(col("service").eq(lit("levanter")))];

        assert!(values_by_column(&filters).is_empty());
    }

    #[test]
    fn row_runs_map_to_partial_and_skipped_row_groups() {
        let plan = runs_access_plan(
            &[RowRun { start: 2, len: 2 }, RowRun { start: 8, len: 2 }],
            &[5, 5],
        );
        assert!(matches!(plan.inner()[0], RowGroupAccess::Selection(_)));
        assert!(matches!(plan.inner()[1], RowGroupAccess::Selection(_)));
        assert_eq!(plan.row_group_indexes(), vec![0, 1]);
    }

    #[test]
    fn run_intersection_is_exact() {
        assert_eq!(
            intersect_runs(
                &[RowRun { start: 1, len: 5 }, RowRun { start: 10, len: 2 }],
                &[RowRun { start: 3, len: 5 }, RowRun { start: 11, len: 3 }],
            ),
            vec![RowRun { start: 3, len: 3 }, RowRun { start: 11, len: 1 }]
        );
    }
}
