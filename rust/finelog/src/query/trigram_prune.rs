//! Row-group pruning for `contains(data, needle)` via per-segment trigram
//! sidecars.
//!
//! The provider delegates the scan to DataFusion as usual, then this module
//! *injects* a `ParquetAccessPlan` into each `PartitionedFile`: the parquet
//! opener composes our per-row-group skips with its existing range / min-max /
//! bloom pruning (`datafusion-datasource-parquet/src/opener.rs`). Our skips are
//! applied first, so the controller `key =` band prune still happens — the
//! trigram prune only removes *more* row groups, never fewer.
//!
//! Safety: we prune only on `contains(col, <literal>)` that appears as a
//! **top-level conjunct**. A `contains()` under an `OR` could drop rows that
//! match the other branch, so those are ignored. The pushdown stays `Inexact`,
//! so DataFusion keeps a `FilterExec` that re-checks `contains()` exactly —
//! a kept row group that doesn't actually match (Bloom false positive, or
//! trigrams split across rows) is filtered there, not returned.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_datasource_parquet::ParquetAccessPlan;

use crate::store::segment::segment_row_group_count;
use crate::store::trigram::{sidecar_path, TrigramIndex, INDEXED_COLUMN};

/// Inject trigram row-group access plans into a parquet scan `plan` for any
/// `contains(INDEXED_COLUMN, needle)` conjunct in `filters`.
///
/// Returns `plan` unchanged when there is nothing to prune (no contains filter,
/// no usable sidecar, or the plan isn't a parquet `DataSourceExec`) — so the
/// hot non-substring path pays nothing.
pub fn apply(
    plan: Arc<dyn ExecutionPlan>,
    segment_paths: &[String],
    filters: &[Expr],
) -> Arc<dyn ExecutionPlan> {
    let needles = contains_needles(filters, INDEXED_COLUMN);
    if needles.is_empty() {
        return plan;
    }
    let access_plans = build_access_plans(segment_paths, &needles);
    if access_plans.is_empty() {
        return plan;
    }
    rewrite_file_groups(plan, &access_plans)
}

/// Needles of every top-level `contains(column, <utf8 literal>)` conjunct.
fn contains_needles(filters: &[Expr], column: &str) -> Vec<String> {
    filters
        .iter()
        .filter_map(|f| contains_literal(f, column))
        .collect()
}

/// `Some(needle)` if `expr` is exactly `contains(<column>, <utf8 literal>)`.
fn contains_literal(expr: &Expr, column: &str) -> Option<String> {
    let Expr::ScalarFunction(sf) = expr else {
        return None;
    };
    if sf.func.name() != "contains" || sf.args.len() != 2 {
        return None;
    }
    let Expr::Column(col) = &sf.args[0] else {
        return None;
    };
    if col.name != column {
        return None;
    }
    match &sf.args[1] {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => Some(s.clone()),
        _ => None,
    }
}

/// Per-segment access plans keyed by file basename (unique within a namespace).
///
/// A segment contributes an entry only when its sidecar loads, aligns with the
/// parquet's row-group count, and the needles actually prune at least one row
/// group. Everything else (missing/stale/corrupt sidecar, short needle, nothing
/// pruned) is skipped — the file then scans unpruned, which is correct.
fn build_access_plans(
    segment_paths: &[String],
    needles: &[String],
) -> HashMap<String, ParquetAccessPlan> {
    let mut out = HashMap::new();
    for path in segment_paths {
        let p = Path::new(path);
        let Some(basename) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Ok(bytes) = std::fs::read(sidecar_path(p)) else {
            continue;
        };
        let Some(index) = TrigramIndex::from_bytes(&bytes) else {
            continue;
        };
        // A row group survives only if it survives EVERY needle's trigram test.
        let mut keep: Option<Vec<bool>> = None;
        for needle in needles {
            // A short needle (< 3 bytes) can't prune; it constrains nothing.
            if let Some(mask) = index.keep_mask(needle) {
                keep = Some(match keep {
                    None => mask,
                    Some(prev) => prev.iter().zip(&mask).map(|(a, b)| *a && *b).collect(),
                });
            }
        }
        let Some(keep) = keep else {
            continue;
        };
        // Only attach when the sidecar aligns with the real parquet (a stale
        // sidecar would otherwise hard-error the query in the opener).
        let Some(rg_count) = segment_row_group_count(p) else {
            continue;
        };
        if rg_count != keep.len() || keep.iter().all(|&k| k) {
            continue;
        }
        let mut access = ParquetAccessPlan::new_all(rg_count);
        for (i, &k) in keep.iter().enumerate() {
            if !k {
                access.skip(i);
            }
        }
        out.insert(basename.to_string(), access);
    }
    out
}

/// Rebuild the scan's file groups, attaching each file's access plan as a
/// `PartitionedFile` extension. Non-parquet plans are returned unchanged.
fn rewrite_file_groups(
    plan: Arc<dyn ExecutionPlan>,
    access_plans: &HashMap<String, ParquetAccessPlan>,
) -> Arc<dyn ExecutionPlan> {
    let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return plan;
    };
    let Some(cfg) = exec.data_source().as_any().downcast_ref::<FileScanConfig>() else {
        return plan;
    };
    let new_groups: Vec<FileGroup> = cfg
        .file_groups
        .iter()
        .map(|group| {
            let files = group
                .files()
                .iter()
                .map(|pf| {
                    match pf
                        .object_meta
                        .location
                        .filename()
                        .and_then(|b| access_plans.get(b))
                    {
                        Some(access) => pf.clone().with_extensions(Arc::new(access.clone())),
                        None => pf.clone(),
                    }
                })
                .collect::<Vec<_>>();
            FileGroup::new(files)
        })
        .collect();
    let new_cfg = FileScanConfigBuilder::from(cfg.clone())
        .with_file_groups(new_groups)
        .build();
    DataSourceExec::from_data_source(new_cfg)
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::{col, lit};

    use super::*;

    /// `contains(data, 'x')` built as a logical expr, mirroring how the planner
    /// represents the UDF call.
    fn contains_expr(column: &str, needle: &str) -> Expr {
        use datafusion::execution::FunctionRegistry;
        use datafusion::logical_expr::expr::ScalarFunction;
        use datafusion::prelude::SessionContext;
        let ctx = SessionContext::new();
        crate::query::udf::register_compat_udfs(&ctx);
        let udf = ctx.udf("contains").unwrap();
        Expr::ScalarFunction(ScalarFunction::new_udf(udf, vec![col(column), lit(needle)]))
    }

    #[test]
    fn extracts_only_top_level_contains_on_indexed_column() {
        let filters = vec![
            contains_expr("data", "Bootstrap completed"),
            contains_expr("source", "stderr"), // wrong column: ignored
            col("seq").gt(lit(5_i64)),         // not a contains: ignored
        ];
        assert_eq!(
            contains_needles(&filters, "data"),
            vec!["Bootstrap completed".to_string()]
        );
    }

    #[test]
    fn contains_under_or_is_not_extracted() {
        // A contains() buried in an OR is unsafe to prune on; only top-level
        // conjuncts (the elements of `filters`) are inspected.
        let buried = contains_expr("data", "x").or(col("seq").gt(lit(1_i64)));
        assert!(contains_needles(&[buried], "data").is_empty());
    }
}
