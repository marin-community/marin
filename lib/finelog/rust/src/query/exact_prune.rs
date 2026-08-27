//! Filtered projections and exact row pruning for string equality/IN predicates.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use crate::query::index_cache::IndexCache;
use crate::store::exact::{coalesce_runs, RowRun};
use crate::store::index_bundle::SectionKind;
use crate::store::segment::{segment_id, segment_row_group_rows};
use crate::store::segment_index::{
    parse_projection_reference, projection_path, ProjectionReference, SOURCE_ROW_OFFSET_IDENTITY,
};
use datafusion::logical_expr::{Expr, Operator};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_datasource_parquet::{ParquetAccessPlan, RowGroupAccess};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

/// Above this retained-row fraction, Parquet's contiguous scan is generally
/// cheaper than materializing and applying a fragmented row selection.
const MAX_POSTINGS_SELECTED_NUMERATOR: u64 = 1;
const MAX_POSTINGS_SELECTED_DENOMINATOR: u64 = 4;

/// Exact string values implied by top-level `=`, `IN`, and same-column `OR`
/// expressions. Multiple conjunctive constraints on one column are intersected.
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

/// Substitute complete filtered projections for each covered segment. Attach
/// exact row selections to uncovered source files; existing trigram access
/// plans are intersected rather than replaced.
pub fn apply(
    plan: Arc<dyn ExecutionPlan>,
    segment_paths: &[String],
    constraints: &HashMap<String, Vec<String>>,
    required_columns: &BTreeSet<String>,
    index_cache: &IndexCache,
) -> Arc<dyn ExecutionPlan> {
    if constraints.is_empty() {
        return plan;
    }
    let projections =
        build_projection_files(segment_paths, constraints, required_columns, index_cache);
    let projected_segments = projections.keys().cloned().collect();
    let plan = rewrite_projection_files(plan, &projections);
    let access_plans =
        build_access_plans(segment_paths, constraints, &projected_segments, index_cache);
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
    required_columns: &BTreeSet<String>,
    index_cache: &IndexCache,
) -> HashMap<String, ProjectionFile> {
    let mut projections = HashMap::new();
    let mut projected_rows = 0_u64;
    let mut total_rows = 0_u64;
    for segment in segment_paths {
        let parquet = Path::new(segment);
        let Some(basename) = parquet.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(segment) = index_cache.indexed_segment(parquet) else {
            continue;
        };
        let source_rows = segment.row_group_rows.iter().sum::<usize>() as u64;
        let Some(reference) = segment
            .header
            .sections
            .iter()
            .filter(|section| section.kind == SectionKind::CoveringProjection)
            .filter_map(|section| parse_projection_reference(&section.coverage))
            .find(|reference| projection_covers(reference, constraints, required_columns))
        else {
            continue;
        };
        let projection = projection_path(parquet, &reference);
        let Some(projection_groups) = segment_row_group_rows(&projection) else {
            continue;
        };
        if projection_groups.iter().sum::<usize>() as u64 != reference.descriptor.row_count
            || segment_id(&projection).map(|id| id.to_string())
                != Some(reference.file_segment_id.clone())
        {
            continue;
        }
        let Some(metadata) = std::fs::metadata(&projection).ok() else {
            continue;
        };
        if metadata.len() != reference.file_bytes {
            continue;
        }
        let Some(location) = object_store::path::Path::from_filesystem_path(&projection).ok()
        else {
            continue;
        };
        let Some(modified) = metadata.modified().ok() else {
            continue;
        };
        projections.insert(
            basename.to_string(),
            ProjectionFile {
                location,
                size: metadata.len(),
                modified,
            },
        );
        projected_rows += reference.descriptor.row_count;
        total_rows += source_rows;
    }
    if !projections.is_empty() {
        tracing::debug!(
            segments_projected = projections.len(),
            projected_rows,
            total_rows,
            "exact-value filtered projection"
        );
    }
    projections
}

fn projection_covers(
    reference: &ProjectionReference,
    constraints: &HashMap<String, Vec<String>>,
    required_columns: &BTreeSet<String>,
) -> bool {
    reference.row_identity == SOURCE_ROW_OFFSET_IDENTITY
        && required_columns
            .iter()
            .all(|column| reference.descriptor.columns.contains(column))
        && constraints
            .get(&reference.descriptor.predicate_column)
            .is_some_and(|values| {
                values
                    .iter()
                    .all(|value| reference.descriptor.predicate_values.contains(value))
            })
}

fn build_access_plans(
    segment_paths: &[String],
    constraints: &HashMap<String, Vec<String>>,
    projected_segments: &BTreeSet<String>,
    index_cache: &IndexCache,
) -> HashMap<String, ParquetAccessPlan> {
    let mut plans = HashMap::new();
    let mut selected_rows = 0_u64;
    let mut total_rows = 0_u64;
    let mut nonselective_segments = 0_usize;
    for segment in segment_paths {
        let parquet = Path::new(segment);
        let Some(basename) = parquet.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if projected_segments.contains(basename) {
            continue;
        }
        let Some(segment) = index_cache.indexed_segment(parquet) else {
            continue;
        };
        let total_segment_rows = segment.row_group_rows.iter().sum::<usize>() as u64;
        let Some(index) =
            index_cache.get_exact(parquet, &segment.header, SectionKind::ExactPostings)
        else {
            continue;
        };
        let mut selected: Option<Vec<RowRun>> = None;
        for (column_name, values) in constraints {
            let Some(column) = index.columns.get(column_name) else {
                continue;
            };
            if !values.iter().all(|value| column.rows.contains_key(value)) {
                continue;
            }
            let union = coalesce_runs(
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
        let segment_selected_rows = selected.iter().map(|run| run.len).sum::<u64>();
        if !postings_are_selective(segment_selected_rows, total_segment_rows) {
            nonselective_segments += 1;
            continue;
        }
        selected_rows += segment_selected_rows;
        total_rows += total_segment_rows;
        plans.insert(
            basename.to_string(),
            runs_access_plan(&selected, &segment.row_group_rows),
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
    if nonselective_segments > 0 {
        tracing::debug!(
            nonselective_segments,
            max_selected_numerator = MAX_POSTINGS_SELECTED_NUMERATOR,
            max_selected_denominator = MAX_POSTINGS_SELECTED_DENOMINATOR,
            "exact postings retained too many rows; scanning those segments contiguously"
        );
    }
    plans
}

fn postings_are_selective(selected_rows: u64, total_rows: u64) -> bool {
    total_rows > 0
        && selected_rows.saturating_mul(MAX_POSTINGS_SELECTED_DENOMINATOR)
            <= total_rows.saturating_mul(MAX_POSTINGS_SELECTED_NUMERATOR)
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
    let mut combined = ParquetAccessPlan::new_all(existing.inner().len());
    for (index, (left, right)) in existing.inner().iter().zip(additional.inner()).enumerate() {
        match (left, right) {
            (RowGroupAccess::Skip, _) | (_, RowGroupAccess::Skip) => combined.skip(index),
            (RowGroupAccess::Scan, RowGroupAccess::Scan) => {}
            (RowGroupAccess::Selection(selection), RowGroupAccess::Scan)
            | (RowGroupAccess::Scan, RowGroupAccess::Selection(selection)) => {
                combined.scan_selection(index, selection.clone())
            }
            (RowGroupAccess::Selection(left), RowGroupAccess::Selection(right)) => {
                combined.scan_selection(index, left.intersection(right))
            }
        }
    }
    Some(combined)
}

fn rewrite_file_groups(
    plan: Arc<dyn ExecutionPlan>,
    access_plans: &HashMap<String, ParquetAccessPlan>,
) -> Arc<dyn ExecutionPlan> {
    crate::query::file_scan::rewrite_parquet_files(plan, |file| {
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
}

fn rewrite_projection_files(
    plan: Arc<dyn ExecutionPlan>,
    projections: &HashMap<String, ProjectionFile>,
) -> Arc<dyn ExecutionPlan> {
    crate::query::file_scan::rewrite_parquet_files(plan, |file| {
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
    fn access_plan_intersection_preserves_both_pruners() {
        let mut left = ParquetAccessPlan::new_all(2);
        left.scan_selection(
            0,
            RowSelection::from(vec![RowSelector::select(5), RowSelector::skip(5)]),
        );
        left.skip(1);
        let mut right = ParquetAccessPlan::new_all(2);
        right.scan_selection(
            0,
            RowSelection::from(vec![
                RowSelector::skip(3),
                RowSelector::select(4),
                RowSelector::skip(3),
            ]),
        );

        let combined = intersect_access_plans(&left, &right).unwrap();
        let RowGroupAccess::Selection(selection) = &combined.inner()[0] else {
            panic!("the overlapping row group should retain a selection");
        };
        assert_eq!(
            Vec::<RowSelector>::from(selection.clone()),
            vec![
                RowSelector::skip(3),
                RowSelector::select(2),
                RowSelector::skip(5),
            ]
        );
        assert!(matches!(combined.inner()[1], RowGroupAccess::Skip));
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

    #[test]
    fn postings_fall_back_to_contiguous_scan_above_quarter_coverage() {
        assert!(postings_are_selective(25, 100));
        assert!(!postings_are_selective(26, 100));
        assert!(!postings_are_selective(1, 0));
    }
}
