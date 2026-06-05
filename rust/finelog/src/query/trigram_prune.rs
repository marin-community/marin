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
//! Safety: we prune only on a substring predicate that appears as a **top-level
//! conjunct** — either `contains(col, <literal>)` or `col LIKE '%<literal>%'`. A
//! predicate under an `OR` could drop rows that match the other branch, so those
//! are ignored. The pushdown stays `Inexact`, so DataFusion keeps a `FilterExec`
//! that re-checks the predicate exactly — a kept row group that doesn't actually
//! match (Bloom false positive, or trigrams split across rows) is filtered there,
//! not returned.
//!
//! `LIKE` extraction is deliberately narrow: only a single substring framed by
//! `%` wildcards (`%lit%`, `lit%`, `%lit`) where `lit` has no `_`, `%`, or `\`.
//! Anything else (`NOT LIKE`, `ILIKE`, an `_` single-char wildcard, an escape, or
//! multiple `%`-separated fragments) is left unpruned rather than risk a needle
//! that the match does not actually imply.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::{Expr, Like};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_datasource_parquet::ParquetAccessPlan;

use crate::store::segment::segment_row_group_count;
use crate::store::trigram::{
    needle_trigrams, sidecar_path, TrigramIndex, INDEXED_COLUMN, MIN_TRIGRAM_LEN,
};

/// Prunable substring needles from every top-level `contains(INDEXED_COLUMN, …)`
/// or `INDEXED_COLUMN LIKE '%…%'` conjunct, filtered to those long enough to
/// decompose into at least one trigram (`>= MIN_TRIGRAM_LEN`).
///
/// Pure expr inspection, no I/O — the provider calls this on the hot path to
/// decide (cheaply) whether the substring prune applies at all before touching
/// any sidecar. A too-short needle constrains no trigram, so dropping it here
/// keeps the prune off the blocking path entirely for `contains(col, 'ab')`.
pub fn indexed_column_needles(filters: &[Expr]) -> Vec<String> {
    substring_needles(filters, INDEXED_COLUMN)
        .into_iter()
        .filter(|n| n.len() >= MIN_TRIGRAM_LEN)
        .collect()
}

/// Inject access plans for already-extracted `needles`. Does the blocking
/// sidecar + footer reads, so the provider runs it under `spawn_blocking`.
/// Returns `plan` unchanged when `needles` is empty or nothing prunes.
pub fn apply_with_needles(
    plan: Arc<dyn ExecutionPlan>,
    segment_paths: &[String],
    needles: &[String],
) -> Arc<dyn ExecutionPlan> {
    if needles.is_empty() {
        return plan;
    }
    let access_plans = build_access_plans(segment_paths, needles);
    if access_plans.is_empty() {
        return plan;
    }
    rewrite_file_groups(plan, &access_plans)
}

/// Substring needles from every top-level conjunct that constrains `column` to
/// contain a literal — `contains(column, lit)` or `column LIKE '%lit%'`.
fn substring_needles(filters: &[Expr], column: &str) -> Vec<String> {
    filters
        .iter()
        .filter_map(|f| substring_needle(f, column))
        .collect()
}

/// `Some(needle)` if `expr` constrains `column` to contain a literal substring:
/// `contains(<column>, <utf8 literal>)`, or a `<column> LIKE` whose pattern is a
/// single wildcard-framed substring (see [`like_substring`]).
fn substring_needle(expr: &Expr, column: &str) -> Option<String> {
    match expr {
        Expr::ScalarFunction(sf) => contains_literal(sf, column),
        Expr::Like(like) => like_substring(like, column),
        _ => None,
    }
}

/// `Some(needle)` if `sf` is exactly `contains(<column>, <utf8 literal>)`.
fn contains_literal(
    sf: &datafusion::logical_expr::expr::ScalarFunction,
    column: &str,
) -> Option<String> {
    if sf.func.name() != "contains" || sf.args.len() != 2 {
        return None;
    }
    let Expr::Column(col) = &sf.args[0] else {
        return None;
    };
    if col.name != column {
        return None;
    }
    utf8_literal(&sf.args[1])
}

/// `Some(needle)` if `like` is `<column> LIKE '<pattern>'` where the pattern is a
/// single literal substring framed by `%` wildcards and free of the `_`
/// single-char wildcard and `\` escape — so a match provably contains `needle`.
///
/// Conservative by construction: `NOT LIKE`, `ILIKE` (case-insensitive), an
/// explicit escape char, or a pattern with `_`, `\`, or more than one
/// `%`-separated fragment all return `None` (no prune), because none of those
/// guarantee `needle` appears verbatim in a matching value.
fn like_substring(like: &Like, column: &str) -> Option<String> {
    if like.negated || like.case_insensitive || like.escape_char.is_some() {
        return None;
    }
    let Expr::Column(col) = like.expr.as_ref() else {
        return None;
    };
    if col.name != column {
        return None;
    }
    let pattern = utf8_literal(&like.pattern)?;
    // `\` is LIKE's implicit escape even with no explicit escape char; `_` is the
    // single-char wildcard. A pattern carrying either has subtler semantics than
    // "contains this literal", so refuse it.
    if pattern.contains(['_', '\\']) {
        return None;
    }
    // The literal text lives between the `%` wildcards. Exactly one non-empty
    // fragment ⇒ a single required substring (`%lit%`, `lit%`, `%lit`). Zero
    // fragments (all `%`) matches everything; two or more (`%a%b%`) is an AND of
    // substrings we don't model in v1 — both yield no prunable needle.
    let mut fragments = pattern.split('%').filter(|f| !f.is_empty());
    let needle = fragments.next()?;
    if fragments.next().is_some() {
        return None;
    }
    Some(needle.to_string())
}

/// The string value of a Utf8 / LargeUtf8 / Utf8View literal, else `None`.
fn utf8_literal(expr: &Expr) -> Option<String> {
    match expr {
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
    // Decompose each needle into trigrams ONCE, not once per segment — a single
    // query commonly spans dozens of segments. Needles arrive pre-filtered to
    // `>= MIN_TRIGRAM_LEN`, so each yields a non-empty trigram set.
    let needle_trigrams: Vec<Vec<[u8; 3]>> =
        needles.iter().filter_map(|n| needle_trigrams(n)).collect();
    if needle_trigrams.is_empty() {
        return HashMap::new();
    }

    let mut out = HashMap::new();
    let mut total_row_groups = 0usize;
    let mut skipped_row_groups = 0usize;
    for path in segment_paths {
        let p = Path::new(path);
        let Some(basename) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Ok(bytes) = std::fs::read(sidecar_path(p)) else {
            // No sidecar: expected for L0 / unindexed-namespace segments. The
            // file just scans unpruned — correct, never a false negative.
            tracing::debug!(segment = basename, "no trigram sidecar; scanning unpruned");
            continue;
        };
        let Some(index) = TrigramIndex::from_bytes(&bytes) else {
            tracing::warn!(
                segment = basename,
                "corrupt trigram sidecar; scanning unpruned"
            );
            continue;
        };
        // A row group survives only if it survives EVERY needle's trigram test.
        let mut keep = vec![true; index.len()];
        for trigrams in &needle_trigrams {
            for (k, m) in keep.iter_mut().zip(index.keep_mask_for(trigrams)) {
                *k &= m;
            }
        }
        // Only attach when the sidecar aligns with the real parquet (a stale
        // sidecar would otherwise hard-error the query in the opener).
        let Some(rg_count) = segment_row_group_count(p) else {
            continue;
        };
        if rg_count != keep.len() {
            tracing::warn!(
                segment = basename,
                sidecar_row_groups = keep.len(),
                parquet_row_groups = rg_count,
                "stale trigram sidecar (row-group count mismatch); scanning unpruned"
            );
            continue;
        }
        if keep.iter().all(|&k| k) {
            continue;
        }
        let mut access = ParquetAccessPlan::new_all(rg_count);
        let mut skipped = 0usize;
        for (i, &k) in keep.iter().enumerate() {
            if !k {
                access.skip(i);
                skipped += 1;
            }
        }
        total_row_groups += rg_count;
        skipped_row_groups += skipped;
        out.insert(basename.to_string(), access);
    }
    if !out.is_empty() {
        tracing::debug!(
            needles = needle_trigrams.len(),
            segments_pruned = out.len(),
            row_groups_skipped = skipped_row_groups,
            row_groups_total = total_row_groups,
            "trigram prune"
        );
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

    /// `<column> LIKE '<pattern>'` (or `NOT LIKE` / `ILIKE` via the flags).
    fn like_expr(column: &str, pattern: &str, negated: bool, case_insensitive: bool) -> Expr {
        Expr::Like(Like {
            negated,
            expr: Box::new(col(column)),
            pattern: Box::new(lit(pattern)),
            escape_char: None,
            case_insensitive,
        })
    }

    #[test]
    fn extracts_only_top_level_contains_on_indexed_column() {
        let filters = vec![
            contains_expr("data", "Bootstrap completed"),
            contains_expr("source", "stderr"), // wrong column: ignored
            col("seq").gt(lit(5_i64)),         // not a contains: ignored
        ];
        assert_eq!(
            substring_needles(&filters, "data"),
            vec!["Bootstrap completed".to_string()]
        );
    }

    #[test]
    fn contains_under_or_is_not_extracted() {
        // A contains() buried in an OR is unsafe to prune on; only top-level
        // conjuncts (the elements of `filters`) are inspected.
        let buried = contains_expr("data", "x").or(col("seq").gt(lit(1_i64)));
        assert!(substring_needles(&[buried], "data").is_empty());
    }

    #[test]
    fn like_extracts_wildcard_framed_substring() {
        // `%lit%`, `lit%`, and `%lit` all imply the value contains `lit`.
        for pattern in [
            "%Bootstrap completed%",
            "Bootstrap completed%",
            "%Bootstrap completed",
        ] {
            assert_eq!(
                substring_needles(&[like_expr("data", pattern, false, false)], "data"),
                vec!["Bootstrap completed".to_string()],
                "pattern {pattern:?}"
            );
        }
    }

    #[test]
    fn like_unsafe_patterns_are_not_extracted() {
        // Each of these would risk a needle the match does not imply, so the
        // prune must decline (scan unpruned) rather than over-prune.
        let unsafe_cases = [
            ("%a_c%", false, false),     // `_` single-char wildcard
            ("%a\\c%", false, false),    // `\` escape
            ("%abc%def%", false, false), // two required fragments (AND, not one substring)
            ("%%", false, false),        // matches everything: no needle
            ("%abc%", true, false),      // NOT LIKE
            ("%abc%", false, true),      // ILIKE (case-insensitive)
        ];
        for (pattern, negated, case_insensitive) in unsafe_cases {
            assert!(
                substring_needles(
                    &[like_expr("data", pattern, negated, case_insensitive)],
                    "data"
                )
                .is_empty(),
                "pattern {pattern:?} negated={negated} ci={case_insensitive} must not extract"
            );
        }
        // Wrong column is ignored too.
        assert!(
            substring_needles(&[like_expr("source", "%abc%", false, false)], "data").is_empty()
        );
    }

    #[test]
    fn short_needles_are_dropped_before_the_blocking_path() {
        // `indexed_column_needles` filters needles too short to form a trigram, so
        // the provider returns on the hot path without touching a sidecar.
        let filters = vec![
            contains_expr("data", "ab"),             // 2 bytes: no trigram
            like_expr("data", "%xy%", false, false), // 2 bytes: no trigram
        ];
        assert!(indexed_column_needles(&filters).is_empty());
    }
}
