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
//! A `LIKE` pattern contributes every literal run between its wildcards, since
//! `%` and `_` only insert characters *between* those runs: `%a%b%` requires
//! both `a` and `b`, and `%CUDA\_ERROR%` requires the single run `CUDA_ERROR`.
//! `NOT LIKE`, `ILIKE`, and an explicit `ESCAPE` char are left unpruned — none of
//! them imply the runs appear verbatim under the parse used here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::{BinaryExpr, Expr, Like, Operator};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_datasource_parquet::ParquetAccessPlan;

use crate::query::sidecar::SidecarManager;
use crate::store::segment::segment_row_group_count;
use crate::store::trigram::{needle_trigrams, sidecar_path, MIN_TRIGRAM_LEN};

/// An inclusive key range constraining a single column, distilled from a query's
/// top-level conjuncts. Used to scope which segments' sidecars are read: a
/// segment whose key band can't overlap this range is pruned by the parquet key
/// statistics anyway, so its blooms are never loaded.
///
/// Bounds are conservatively widened to *inclusive* (a strict `<` is treated as
/// `<=`): widening can only keep a borderline segment in scope, never wrongly
/// drop one — and skipping is a pure I/O optimization, so the safe direction is
/// to over-include.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StringRange {
    pub lo: Option<Vec<u8>>,
    pub hi: Option<Vec<u8>>,
}

/// Inject access plans for already-extracted per-column `needles` (from
/// [`substring_needles_by_column`]). Does the blocking sidecar + footer reads
/// (routed through the [`SidecarManager`] cache), so the provider runs it under
/// `spawn_blocking`. `key_ranges` (from [`string_column_ranges`]) scopes which
/// segments are consulted by key band. Returns `plan` unchanged when `needles`
/// is empty or nothing prunes.
pub fn apply_with_needles(
    plan: Arc<dyn ExecutionPlan>,
    segment_paths: &[String],
    needles: &HashMap<String, Vec<String>>,
    key_ranges: &HashMap<String, StringRange>,
) -> Arc<dyn ExecutionPlan> {
    if needles.is_empty() {
        return plan;
    }
    let access_plans = build_access_plans(segment_paths, needles, key_ranges);
    if access_plans.is_empty() {
        return plan;
    }
    rewrite_file_groups(plan, &access_plans)
}

/// Inclusive per-column key ranges implied by a query's top-level conjuncts.
///
/// Walks `filters` (descending through top-level `AND`s) for `column <cmp>
/// <utf8 literal>` comparisons — including the `key >= P AND key < succ(P)`
/// bounds the [`crate::query::optimizer::PrefixRangeRewrite`] synthesizes from a
/// `prefix`/`LIKE`/anchored-regex predicate — and folds them into one inclusive
/// `[lo, hi]` per column (lo = greatest lower bound, hi = least upper bound).
/// Pure expr inspection, no I/O.
pub fn string_column_ranges(filters: &[Expr]) -> HashMap<String, StringRange> {
    let mut out: HashMap<String, StringRange> = HashMap::new();
    for f in filters {
        collect_ranges(f, &mut out);
    }
    out
}

/// Accumulate `column <cmp> literal` bounds from `expr`, descending through
/// top-level conjunctions so a single `AND`-chained predicate contributes each
/// of its comparisons.
fn collect_ranges(expr: &Expr, out: &mut HashMap<String, StringRange>) {
    match expr {
        Expr::BinaryExpr(be) if be.op == Operator::And => {
            collect_ranges(&be.left, out);
            collect_ranges(&be.right, out);
        }
        Expr::BinaryExpr(be) => {
            if let Some((column, op, value)) = col_literal_comparison(be) {
                apply_bound(out.entry(column).or_default(), op, value);
            }
        }
        _ => {}
    }
}

/// Normalize a `column <cmp> utf8-literal` (or the mirrored `literal <cmp>
/// column`) comparison to `(column, op, literal_bytes)` with `op` oriented as
/// `column <op> literal`. `None` for anything else.
fn col_literal_comparison(be: &BinaryExpr) -> Option<(String, Operator, Vec<u8>)> {
    if let Expr::Column(c) = be.left.as_ref() {
        if let Some(v) = utf8_literal(&be.right) {
            return Some((c.name.clone(), be.op, v.into_bytes()));
        }
    }
    if let Expr::Column(c) = be.right.as_ref() {
        if let Some(v) = utf8_literal(&be.left) {
            return Some((c.name.clone(), flip_comparison(be.op)?, v.into_bytes()));
        }
    }
    None
}

/// Mirror a comparison operator for `literal <op> column` ⇒ `column <flipped>
/// literal`. `None` for non-orderings (so they don't constrain the range).
fn flip_comparison(op: Operator) -> Option<Operator> {
    match op {
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        Operator::Eq => Some(Operator::Eq),
        _ => None,
    }
}

/// Fold one `column <op> value` bound into `range`, tightening to the
/// intersection (greatest lower / least upper). Strictness is dropped — the
/// bounds stay inclusive (see [`StringRange`]).
fn apply_bound(range: &mut StringRange, op: Operator, value: Vec<u8>) {
    let tighten_lo = |lo: &mut Option<Vec<u8>>, v: Vec<u8>| {
        if lo.as_deref().is_none_or(|cur| v.as_slice() > cur) {
            *lo = Some(v);
        }
    };
    let tighten_hi = |hi: &mut Option<Vec<u8>>, v: Vec<u8>| {
        if hi.as_deref().is_none_or(|cur| v.as_slice() < cur) {
            *hi = Some(v);
        }
    };
    match op {
        Operator::Eq => {
            tighten_lo(&mut range.lo, value.clone());
            tighten_hi(&mut range.hi, value);
        }
        Operator::Gt | Operator::GtEq => tighten_lo(&mut range.lo, value),
        Operator::Lt | Operator::LtEq => tighten_hi(&mut range.hi, value),
        _ => {}
    }
}

/// The needles [`substring_needles_by_column`] would apply to `column`.
#[cfg(test)]
fn substring_needles(filters: &[Expr], column: &str) -> Vec<String> {
    substring_needles_by_column(filters)
        .remove(column)
        .unwrap_or_default()
}

/// Substring needles grouped by the column each constrains, from every top-level
/// `contains(col, lit)` / `col LIKE '%lit%'` conjunct, keeping the literals long
/// enough to decompose into at least one trigram (`>= MIN_TRIGRAM_LEN`).
///
/// A column's needles are required together, so dropping a too-short one only
/// loosens the constraint.
///
/// Pure expr inspection (no I/O) — the provider calls this on the hot path to
/// decide (cheaply) whether the substring prune applies at all before touching
/// any sidecar. A column the query constrains but a given segment's sidecar does
/// not index is simply ignored when that segment is pruned.
pub fn substring_needles_by_column(filters: &[Expr]) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for f in filters {
        let Some((column, needles)) = substring_column_needles(f) else {
            continue;
        };
        let usable: Vec<String> = needles
            .into_iter()
            .filter(|n| n.len() >= MIN_TRIGRAM_LEN)
            .collect();
        if usable.is_empty() {
            continue;
        }
        out.entry(column).or_default().extend(usable);
    }
    out
}

/// `Some((column, needles))` if `expr` constrains some column to contain literal
/// substrings — all of them, since they are ANDed: `contains(<col>, <utf8
/// literal>)`, or the literal runs of a `<col> LIKE` pattern (see
/// [`like_column_needles`]).
fn substring_column_needles(expr: &Expr) -> Option<(String, Vec<String>)> {
    match expr {
        Expr::ScalarFunction(sf) => {
            contains_column_literal(sf).map(|(col, needle)| (col, vec![needle]))
        }
        Expr::Like(like) => like_column_needles(like),
        _ => None,
    }
}

/// `Some((column, needle))` if `sf` is exactly `contains(<column>, <utf8 literal>)`.
fn contains_column_literal(
    sf: &datafusion::logical_expr::expr::ScalarFunction,
) -> Option<(String, String)> {
    if sf.func.name() != "contains" || sf.args.len() != 2 {
        return None;
    }
    let Expr::Column(col) = &sf.args[0] else {
        return None;
    };
    let needle = utf8_literal(&sf.args[1])?;
    Some((col.name.clone(), needle))
}

/// `Some((column, needles))` if `like` is `<column> LIKE '<pattern>'`, where
/// `needles` are the pattern's literal runs (see [`like_literal_runs`]). Every
/// run is a substring of any matching value, so requiring all of them is sound.
///
/// `NOT LIKE` and `ILIKE` return `None`: a negated match implies nothing about
/// the runs, and a case-insensitive one can match bytes the trigrams never saw.
/// An explicit `ESCAPE` char also returns `None`, because it redefines the
/// escape that [`like_literal_runs`] resolves.
fn like_column_needles(like: &Like) -> Option<(String, Vec<String>)> {
    if like.negated || like.case_insensitive || like.escape_char.is_some() {
        return None;
    }
    let Expr::Column(col) = like.expr.as_ref() else {
        return None;
    };
    let pattern = utf8_literal(&like.pattern)?;
    Some((col.name.clone(), like_literal_runs(&pattern)))
}

/// The literal runs of a `LIKE` pattern, in order.
///
/// `%` (any sequence) and `_` (any single character) end a run, and `\` escapes
/// the next character into the current one — so `%CUDA\_ERROR%` is the single run
/// `CUDA_ERROR`, while the unescaped `%CUDA_ERROR%` is the two runs `CUDA` and
/// `ERROR`. A trailing `\` is a literal backslash. This mirrors how the arrow
/// `LIKE` kernel compiles a pattern with no explicit escape character; the
/// [`crate::query`] tests pin the two against each other.
fn like_literal_runs(pattern: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut run = String::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '%' | '_' if !run.is_empty() => runs.push(std::mem::take(&mut run)),
            '%' | '_' => {}
            '\\' => run.push(chars.next().unwrap_or('\\')),
            c => run.push(c),
        }
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs
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
/// group. Everything else (missing/stale/corrupt sidecar, short needle, key band
/// out of scope, nothing pruned) is skipped — the file then scans unpruned,
/// which is correct.
///
/// Sidecar reads go through the process-global [`SidecarManager`], so a repeated
/// query (the dashboard's poll loop) reuses parsed blooms instead of re-reading
/// them, and the resident bytes stay within the cache budget.
fn build_access_plans(
    segment_paths: &[String],
    needles: &HashMap<String, Vec<String>>,
    key_ranges: &HashMap<String, StringRange>,
) -> HashMap<String, ParquetAccessPlan> {
    // Decompose each constrained column's needles into trigram sets ONCE, not
    // once per segment — a single query commonly spans dozens of segments.
    // Needles arrive pre-filtered to `>= MIN_TRIGRAM_LEN`, so each yields a
    // non-empty set; a column whose needles all degrade is dropped here.
    let trigrams_by_column: HashMap<&str, Vec<Vec<[u8; 3]>>> = needles
        .iter()
        .filter_map(|(col, ns)| {
            let tg: Vec<Vec<[u8; 3]>> = ns.iter().filter_map(|n| needle_trigrams(n)).collect();
            (!tg.is_empty()).then_some((col.as_str(), tg))
        })
        .collect();
    if trigrams_by_column.is_empty() {
        return HashMap::new();
    }

    let manager = SidecarManager::global();
    let mut out = HashMap::new();
    let mut total_row_groups = 0usize;
    let mut skipped_row_groups = 0usize;
    let mut scoped_out = 0usize;
    for path in segment_paths {
        let p = Path::new(path);
        let Some(basename) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let sidecar = sidecar_path(p);
        let Some(header) = manager.get_header(&sidecar) else {
            // No / invalid sidecar: expected for L0 / unindexed-namespace
            // segments. The file just scans unpruned — correct, never a false
            // negative.
            tracing::debug!(
                segment = basename,
                "no usable trigram sidecar; scanning unpruned"
            );
            continue;
        };
        // Key-band scoping: when the query constrains the segment's key column
        // and this segment's band provably can't overlap it, the parquet key
        // statistics will already prune every row group, so skip the bloom read.
        if !header.key_column.is_empty() {
            if let Some(range) = key_ranges.get(&header.key_column) {
                if !header.key_band_overlaps(range.lo.as_deref(), range.hi.as_deref()) {
                    scoped_out += 1;
                    continue;
                }
            }
        }
        // A row group survives only if it survives EVERY constrained column's
        // needles. A column this segment's sidecar does not index can't prune, so
        // it simply contributes no constraint here. The mask is sized from the
        // sidecar's own row-group count; the parquet is consulted below, only for
        // a segment whose blooms actually pruned something.
        let mut keep = vec![true; header.rg_count as usize];
        let mut applied_any = false;
        for (&col, needle_trigrams) in &trigrams_by_column {
            let Some(index) = manager.get_column(&sidecar, &header, col) else {
                continue;
            };
            applied_any = true;
            for trigrams in needle_trigrams {
                for (k, m) in keep.iter_mut().zip(index.keep_mask_for(trigrams)) {
                    *k &= m;
                }
            }
        }
        if !applied_any || keep.iter().all(|&k| k) {
            continue;
        }
        // Confirm the sidecar aligns with the segment before attaching a plan: a
        // row-group mismatch would hard-error the opener. This parses the whole
        // footer, so it runs last — after the cheap header and key-band checks,
        // and only for a segment an access plan would be attached to.
        let Some(rg_count) = segment_row_group_count(p) else {
            continue;
        };
        if rg_count != keep.len() {
            tracing::warn!(
                segment = basename,
                sidecar_row_groups = header.rg_count,
                parquet_row_groups = rg_count,
                "stale trigram sidecar (row-group count mismatch); scanning unpruned"
            );
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
    if !out.is_empty() || scoped_out > 0 {
        tracing::debug!(
            indexed_columns = trigrams_by_column.len(),
            segments_pruned = out.len(),
            segments_scoped_out = scoped_out,
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
        crate::query::udf::register_scalar_udfs(&ctx);
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
    fn like_extracts_every_literal_run() {
        // Wildcards only insert characters between the runs, so each run is
        // required and they are ANDed. An escaped `_` is part of its run; a bare
        // one splits it.
        let cases = [
            ("%abc%def%", vec!["abc", "def"]),
            (r"%CUDA\_ERROR\_OUT%", vec!["CUDA_ERROR_OUT"]),
            ("%CUDA_ERROR%", vec!["CUDA", "ERROR"]),
            ("foo%bar", vec!["foo", "bar"]),
        ];
        for (pattern, expected) in cases {
            assert_eq!(
                substring_needles(&[like_expr("data", pattern, false, false)], "data"),
                expected,
                "pattern {pattern:?}"
            );
        }
    }

    #[test]
    fn like_unsafe_patterns_are_not_extracted() {
        // Each of these would risk a needle the match does not imply, so the
        // prune must decline (scan unpruned) rather than over-prune.
        let unsafe_cases = [
            ("%a_c%", false, false),  // runs too short for a trigram
            ("%a\\c%", false, false), // escape collapses to the 2-byte run `ac`
            ("%%", false, false),     // matches everything: no needle
            ("%abc%", true, false),   // NOT LIKE
            ("%abc%", false, true),   // ILIKE (case-insensitive)
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
        // An explicit ESCAPE redefines what `\` means, so the run parse no longer
        // describes the pattern.
        let escaped = Expr::Like(Like {
            negated: false,
            expr: Box::new(col("data")),
            pattern: Box::new(lit("%abc!%def%")),
            escape_char: Some('!'),
            case_insensitive: false,
        });
        assert!(substring_needles(&[escaped], "data").is_empty());
    }

    #[test]
    fn short_needles_are_dropped_before_the_blocking_path() {
        // `substring_needles_by_column` filters needles too short to form a
        // trigram, so the provider returns on the hot path without touching a
        // sidecar.
        let filters = vec![
            contains_expr("data", "ab"),             // 2 bytes: no trigram
            like_expr("data", "%xy%", false, false), // 2 bytes: no trigram
        ];
        assert!(substring_needles_by_column(&filters).is_empty());
    }

    #[test]
    fn string_column_ranges_folds_conjunct_bounds() {
        // `key >= 'a' AND key < 'b'` is the analyzer's synthesized prefix range
        // shape; the contains() conjunct contributes no range.
        let filters = vec![
            col("key").gt_eq(lit("a")),
            col("key").lt(lit("b")),
            contains_expr("data", "needle here"),
        ];
        let r = string_column_ranges(&filters);
        let band = r.get("key").expect("key range extracted");
        assert_eq!(band.lo.as_deref(), Some(b"a".as_slice()));
        assert_eq!(band.hi.as_deref(), Some(b"b".as_slice()));
        assert!(!r.contains_key("data"));

        // A single AND-chained predicate is descended into.
        let anded = col("key").gt_eq(lit("a")).and(col("key").lt(lit("b")));
        let r2 = string_column_ranges(std::slice::from_ref(&anded));
        assert_eq!(r2.get("key").unwrap().lo.as_deref(), Some(b"a".as_slice()));
        assert_eq!(r2.get("key").unwrap().hi.as_deref(), Some(b"b".as_slice()));

        // `key = 'x'` pins both ends; the mirrored `'m' <= key` orients correctly.
        let eq = string_column_ranges(&[col("key").eq(lit("x"))]);
        assert_eq!(eq.get("key").unwrap().lo.as_deref(), Some(b"x".as_slice()));
        assert_eq!(eq.get("key").unwrap().hi.as_deref(), Some(b"x".as_slice()));
        let mirrored = string_column_ranges(&[lit("m").lt_eq(col("key"))]);
        assert_eq!(
            mirrored.get("key").unwrap().lo.as_deref(),
            Some(b"m".as_slice())
        );
        assert!(mirrored.get("key").unwrap().hi.is_none());
    }

    /// Write a real 2-row-group log segment (all rows under `key`, the needle in
    /// row group 1 only) plus its trigram sidecar; return the segment path.
    fn write_scoping_segment(dir: &std::path::Path, key: &str, needle: &str) -> String {
        use crate::store::segment::{write_segment_to_dir, ROW_GROUP_SIZE};
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let mut data: Vec<String> = (0..ROW_GROUP_SIZE)
            .map(|_| "idle heartbeat ok".to_string())
            .collect();
        data.push(needle.to_string()); // row group 1
        let n = data.len() as i64;
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(1..=n)),
                Arc::new(StringArray::from(vec![key; data.len()])),
                Arc::new(StringArray::from(data)),
            ],
        )
        .unwrap();
        let (path, _) = write_segment_to_dir(dir, 1, 1, &batch, Some("key")).unwrap();
        crate::store::trigram::write_sidecar(
            &path,
            std::slice::from_ref(&batch),
            &["data"],
            Some("key"),
        )
        .unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn key_band_scopes_out_of_band_segments() {
        let dir = std::env::temp_dir().join(format!(
            "finelog_prune_scope_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_scoping_segment(
            &dir,
            "/system/controller",
            "Bootstrap completed for TPU here",
        );
        let paths = vec![path];
        let needles = HashMap::from([(
            "data".to_string(),
            vec!["Bootstrap completed for TPU".to_string()],
        )]);

        // No key constraint: the needle prunes row group 0, so a plan is produced.
        let unscoped = build_access_plans(&paths, &needles, &HashMap::new());
        assert_eq!(
            unscoped.len(),
            1,
            "needle alone must prune the empty row group"
        );

        // In-band key range: still pruned.
        let inband = HashMap::from([(
            "key".to_string(),
            StringRange {
                lo: Some(b"/system/".to_vec()),
                hi: Some(b"/system/z".to_vec()),
            },
        )]);
        assert_eq!(build_access_plans(&paths, &needles, &inband).len(), 1);

        // Out-of-band key range: the segment is scoped out before its blooms load,
        // so no access plan is emitted (the key statistics prune it at scan time).
        let out_of_band = HashMap::from([(
            "key".to_string(),
            StringRange {
                lo: Some(b"/zzz".to_vec()),
                hi: Some(b"/zzz9".to_vec()),
            },
        )]);
        assert!(build_access_plans(&paths, &needles, &out_of_band).is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn like_runs_are_implied_by_the_engines_own_match() {
        // The prune is only sound if every row the engine's LIKE accepts really
        // does contain each extracted run — an escape parse that disagreed with
        // the kernel's would silently drop matching rows. Run the patterns
        // through the real planner and check that against the runs.
        use datafusion::arrow::array::{Array, RecordBatch, StringArray};
        use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

        let values = [
            "CUDA_ERROR_OUT_OF_MEMORY",
            "CUDAxERRORx",
            "cuda_error_out_of_memory",
            "xxabcxxdefxx",
            "foo123bar",
            "foo123barbaz",
            "at 100% capacity",
            r"a back\slash here",
        ];
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Utf8,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values.to_vec()))])
                .unwrap();
        let ctx = crate::query::make_ctx();
        ctx.register_batch("t", batch).unwrap();

        for pattern in [
            r"%CUDA\_ERROR%",
            "%CUDA_ERROR%",
            "%abc%def%",
            "foo%bar",
            r"%100\%%",
            r"%back\\slash%",
        ] {
            let runs = like_literal_runs(pattern);
            let batches = ctx
                .sql(&format!("SELECT v FROM t WHERE v LIKE '{pattern}'"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let matched: Vec<String> = batches
                .iter()
                .flat_map(|b| {
                    let col = b
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .clone();
                    (0..col.len()).map(move |i| col.value(i).to_string())
                })
                .collect();
            assert!(!matched.is_empty(), "pattern {pattern:?} matched nothing");
            for value in &matched {
                for run in &runs {
                    assert!(
                        value.contains(run.as_str()),
                        "pattern {pattern:?} matched {value:?}, which lacks the run {run:?}"
                    );
                }
            }
        }
    }
}
