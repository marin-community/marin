//! Predicate-shaping `AnalyzerRule`s for the finelog read engine.
//!
//! finelog owns layout knowledge DataFusion can't infer — segments are sorted
//! and clustered `[key, seq]`, so a key range prunes parquet row groups. This
//! module encodes that as a planner rule rather than hand-built SQL strings, so
//! it applies to BOTH FetchLogs and the generic Query API and is visible to
//! filter pushdown + parquet stats pruning (see `make_ctx`).
//!
//! ## `PrefixRangeRewrite`
//!
//! A "starts-with" predicate is opaque to statistics — `prefix(key, P)`,
//! `key LIKE 'P%'`, and `regexp_matches(key, '^P…')` all force a scan because
//! min/max stats key on whole values. But each one *implies* the half-open range
//! `[P, succ(P))`. The rule ANDs that implied range onto the predicate:
//!
//! ```text
//! prefix(col, P)  ->  prefix(col, P) AND col >= P AND col < succ(P)
//! ```
//!
//! This is a **tautology-preserving** rewrite: a value satisfying `prefix(col,P)`
//! already lies in `[P, succ(P))`, so the added conjuncts never change the result
//! set — they only expose a range the parquet reader can prune on, while the
//! original predicate stays as the exact residual.
//!
//! Two properties fall out:
//! - **No false negatives**: we only AND in predicates the original already
//!   guarantees.
//! - **Sound anywhere in the expr tree** (even under `OR`): because the range is
//!   attached to the predicate node itself (`X -> X AND implied(X)`), not spliced
//!   as a sibling conjunct. Pruning only *pays off* at a top-level `Filter`
//!   conjunct, but the rewrite is never *wrong* elsewhere.
//!
//! `succ(P)` increments the last byte `< 0xFF` and drops the rest; `None` when no
//! finite UTF-8 successor exists (empty / all-`0xFF` / a non-UTF-8 edge), in which
//! case only the lower bound is added.

use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DFSchema, ScalarValue};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::{Expr, Like, LogicalPlan};
use datafusion::optimizer::AnalyzerRule;

use crate::store::log_read::regex_literal_prefix;

/// Rewrites starts-with predicates (`prefix` / `LIKE 'P%'` / `regexp_matches`
/// with a `^literal` anchor) to additionally carry the implied half-open key
/// range, so DataFusion can prune row groups. See the module docs.
#[derive(Debug, Default)]
pub struct PrefixRangeRewrite;

impl AnalyzerRule for PrefixRangeRewrite {
    fn name(&self) -> &str {
        "finelog_prefix_range_rewrite"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> DFResult<LogicalPlan> {
        plan.transform_down(rewrite_node).map(|t| t.data)
    }
}

/// Rewrite the predicate of a `Filter` node; pass everything else through.
///
/// A `Filter`'s output schema equals its input schema (it drops no columns), so
/// it carries the column type needed to type the synthesized bound literals
/// (Utf8 / LargeUtf8 / Utf8View) — the rule runs after type coercion, so the new
/// comparison must already be well-typed. `map_expressions` rebuilds the node
/// from the rewritten predicate, preserving its structure.
fn rewrite_node(node: LogicalPlan) -> DFResult<Transformed<LogicalPlan>> {
    if !matches!(node, LogicalPlan::Filter(_)) {
        return Ok(Transformed::no(node));
    }
    let schema = Arc::clone(node.schema());
    node.map_expressions(|expr| expr.transform_down(|e| rewrite_expr(e, &schema)))
}

/// AND the implied key range onto a starts-with predicate; leave others as-is.
///
/// Returns `TreeNodeRecursion::Jump` on a rewrite so the traversal does not
/// descend into the replacement (which contains the original predicate node) and
/// re-rewrite it forever.
fn rewrite_expr(expr: Expr, schema: &DFSchema) -> DFResult<Transformed<Expr>> {
    let Some((col, prefix)) = guaranteed_prefix(&expr) else {
        return Ok(Transformed::no(expr));
    };
    let Some(dt) = column_type(schema, &col) else {
        return Ok(Transformed::no(expr));
    };
    let Some(lower) = string_literal(prefix.clone(), &dt) else {
        return Ok(Transformed::no(expr));
    };
    let col_expr = Expr::Column(col);
    let mut conjunction = expr.and(col_expr.clone().gt_eq(lower));
    if let Some(upper) = key_prefix_upper_bound(&prefix).and_then(|hi| string_literal(hi, &dt)) {
        conjunction = conjunction.and(col_expr.lt(upper));
    }
    Ok(Transformed::new(conjunction, true, TreeNodeRecursion::Jump))
}

/// `Some((column, literal_prefix))` when `expr` guarantees that `column` starts
/// with `literal_prefix`:
/// - `prefix(col, 'P')` — P is the guaranteed prefix.
/// - `col LIKE 'P%'` — P is the literal run before the first `%`, free of the
///   `_` wildcard and `\` escape (see [`like_prefix_literal`]).
/// - `regexp_matches(col, '^P…')` — only when `^`-anchored; P is the literal run
///   after `^` (DuckDB `regexp_matches` is otherwise unanchored, so a literal
///   would not pin the start).
fn guaranteed_prefix(expr: &Expr) -> Option<(Column, String)> {
    match expr {
        Expr::ScalarFunction(sf) if sf.args.len() == 2 => {
            let col = column_arg(&sf.args[0])?;
            let arg = utf8_literal(&sf.args[1])?;
            let prefix = match sf.func.name() {
                "prefix" => non_empty(arg),
                "regexp_matches" => anchored_regex_literal(&arg),
                _ => None,
            }?;
            Some((col, prefix))
        }
        Expr::Like(like) if is_plain_like(like) => {
            let col = column_arg(&like.expr)?;
            let pattern = utf8_literal(&like.pattern)?;
            Some((col, like_prefix_literal(&pattern)?))
        }
        _ => None,
    }
}

/// A `LIKE` with no negation, case-insensitivity, or explicit escape char — the
/// only form whose pattern semantics we model.
fn is_plain_like(like: &Like) -> bool {
    !like.negated && !like.case_insensitive && like.escape_char.is_none()
}

/// The guaranteed literal prefix of a `LIKE` pattern: the run before the first
/// `%`, when it is non-empty and free of `_` (single-char wildcard) and `\`
/// (escape). `Some("P")` for `'P%'` / `'P%...'`; `None` for `'%...'`, `'a_b%'`,
/// or an escaped pattern.
fn like_prefix_literal(pattern: &str) -> Option<String> {
    let head = pattern.split('%').next().unwrap_or("");
    if head.is_empty() || head.contains(['_', '\\']) {
        return None;
    }
    Some(head.to_string())
}

/// The guaranteed literal prefix of a `^`-anchored regex: the literal run after
/// the `^`, per [`regex_literal_prefix`] (which drops a trailing char governed
/// by a zero-allowing quantifier). `None` when the pattern is not `^`-anchored
/// or has no guaranteed leading literal.
fn anchored_regex_literal(pattern: &str) -> Option<String> {
    let rest = pattern.strip_prefix('^')?;
    non_empty(regex_literal_prefix(rest).to_string())
}

/// The exclusive upper bound of `[prefix, succ)`: increment the last byte
/// `< 0xFF` and drop the rest. `None` when no finite UTF-8 successor exists
/// (empty, all-`0xFF`, or the increment leaves invalid UTF-8).
fn key_prefix_upper_bound(prefix: &str) -> Option<String> {
    let mut bytes = prefix.as_bytes().to_vec();
    while let Some(&last) = bytes.last() {
        if last < 0xFF {
            let n = bytes.len();
            bytes[n - 1] = last + 1;
            return String::from_utf8(bytes).ok();
        }
        bytes.pop();
    }
    None
}

fn column_arg(expr: &Expr) -> Option<Column> {
    match expr {
        Expr::Column(c) => Some(c.clone()),
        _ => None,
    }
}

fn utf8_literal(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => Some(s.clone()),
        _ => None,
    }
}

fn non_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}

/// The column's data type from the filter input schema, or `None` if unresolved.
fn column_type(schema: &DFSchema, col: &Column) -> Option<DataType> {
    schema
        .index_of_column(col)
        .ok()
        .map(|i| schema.field(i).data_type().clone())
}

/// A string literal `Expr` typed to match the column (`dt`), so the synthesized
/// comparison is well-typed without a second coercion pass. `None` for a
/// non-string column type (the rewrite is then skipped).
fn string_literal(value: String, dt: &DataType) -> Option<Expr> {
    let scalar = match dt {
        DataType::Utf8 => ScalarValue::Utf8(Some(value)),
        DataType::LargeUtf8 => ScalarValue::LargeUtf8(Some(value)),
        DataType::Utf8View => ScalarValue::Utf8View(Some(value)),
        _ => return None,
    };
    Some(Expr::Literal(scalar, None))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema};
    use datafusion::execution::FunctionRegistry;
    use datafusion::logical_expr::expr::ScalarFunction;
    use datafusion::logical_expr::utils::split_conjunction;
    use datafusion::logical_expr::{col, lit, table_scan, Operator};
    use datafusion::prelude::SessionContext;

    use super::*;

    // ----- pure-helper tests -------------------------------------------------

    #[test]
    fn upper_bound_increments_last_byte_or_yields_none() {
        assert_eq!(key_prefix_upper_bound("/a/").as_deref(), Some("/a0")); // '/' -> '0'
        assert_eq!(key_prefix_upper_bound("abc").as_deref(), Some("abd"));
        assert_eq!(key_prefix_upper_bound(""), None);
        // 0x7F -> 0x80 is a lone continuation byte: not valid UTF-8.
        assert_eq!(key_prefix_upper_bound("a\u{7f}"), None);
    }

    #[test]
    fn like_prefix_literal_only_for_clean_leading_literal() {
        assert_eq!(like_prefix_literal("/job/%").as_deref(), Some("/job/"));
        assert_eq!(
            like_prefix_literal("/job/%/tail%").as_deref(),
            Some("/job/")
        );
        assert_eq!(like_prefix_literal("abc").as_deref(), Some("abc")); // no wildcard ⇒ equality
        assert_eq!(like_prefix_literal("%abc%"), None); // leading wildcard
        assert_eq!(like_prefix_literal("a_b%"), None); // `_` single-char wildcard
        assert_eq!(like_prefix_literal("a\\%b%"), None); // escape
    }

    #[test]
    fn anchored_regex_literal_requires_caret() {
        assert_eq!(
            anchored_regex_literal("^/job/test/.*").as_deref(),
            Some("/job/test/")
        );
        assert_eq!(
            anchored_regex_literal("^/job/test/").as_deref(),
            Some("/job/test/")
        );
        assert_eq!(anchored_regex_literal("/job/test/.*"), None); // unanchored
        assert_eq!(anchored_regex_literal("^(a|b)"), None); // metachar right after ^
        assert_eq!(anchored_regex_literal("^"), None); // nothing after anchor
                                                       // A zero-allowing quantifier on the sole literal char leaves no
                                                       // guaranteed prefix: `^a?b` / `^a*c` match keys not starting with `a`.
        assert_eq!(anchored_regex_literal("^a?b"), None);
        assert_eq!(anchored_regex_literal("^a*c"), None);
        assert_eq!(anchored_regex_literal("^ab?c").as_deref(), Some("a")); // `b` optional
    }

    // ----- rule tests --------------------------------------------------------

    fn scalar_call(name: &str, args: Vec<Expr>) -> Expr {
        let ctx = SessionContext::new();
        crate::query::udf::register_scalar_udfs(&ctx);
        Expr::ScalarFunction(ScalarFunction::new_udf(ctx.udf(name).unwrap(), args))
    }

    fn key_log_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("seq", DataType::Int64, false),
        ])
    }

    /// Analyze a `Filter(predicate)` over a `key`/`seq` scan and return the
    /// rewritten predicate's top-level conjuncts.
    fn analyze_filter_conjuncts(predicate: Expr) -> Vec<Expr> {
        let plan = table_scan(Some("t"), &key_log_schema(), None)
            .unwrap()
            .filter(predicate)
            .unwrap()
            .build()
            .unwrap();
        let out = PrefixRangeRewrite
            .analyze(plan, &ConfigOptions::default())
            .unwrap();
        let LogicalPlan::Filter(f) = out else {
            panic!("expected a Filter at the plan root");
        };
        split_conjunction(&f.predicate)
            .into_iter()
            .cloned()
            .collect()
    }

    /// True if `expr` is `key <op> '<value>'` against a Utf8 literal.
    fn is_key_cmp(expr: &Expr, op: Operator, value: &str) -> bool {
        let Expr::BinaryExpr(b) = expr else {
            return false;
        };
        let lhs_is_key = matches!(b.left.as_ref(), Expr::Column(c) if c.name == "key");
        let rhs = matches!(
            b.right.as_ref(),
            Expr::Literal(ScalarValue::Utf8(Some(v)), _) if v == value
        );
        lhs_is_key && b.op == op && rhs
    }

    #[test]
    fn prefix_gains_half_open_range_and_keeps_residual() {
        let conjuncts =
            analyze_filter_conjuncts(scalar_call("prefix", vec![col("key"), lit("/a/")]));
        assert!(
            conjuncts
                .iter()
                .any(|e| is_key_cmp(e, Operator::GtEq, "/a/")),
            "lower bound key >= '/a/' must be added"
        );
        assert!(
            conjuncts.iter().any(|e| is_key_cmp(e, Operator::Lt, "/a0")),
            "upper bound key < succ('/a/') must be added"
        );
        assert!(
            conjuncts
                .iter()
                .any(|e| matches!(e, Expr::ScalarFunction(sf) if sf.func.name() == "prefix")),
            "the prefix() residual must remain for exactness"
        );
    }

    #[test]
    fn like_prefix_and_anchored_regex_gain_range() {
        // LIKE 'P%'
        let like = analyze_filter_conjuncts(col("key").like(lit("/a/%")));
        assert!(like.iter().any(|e| is_key_cmp(e, Operator::GtEq, "/a/")));
        assert!(like.iter().any(|e| is_key_cmp(e, Operator::Lt, "/a0")));
        // regexp_matches(key, '^/a/...')
        let re = analyze_filter_conjuncts(scalar_call(
            "regexp_matches",
            vec![col("key"), lit("^/a/.*")],
        ));
        assert!(re.iter().any(|e| is_key_cmp(e, Operator::GtEq, "/a/")));
        assert!(re.iter().any(|e| is_key_cmp(e, Operator::Lt, "/a0")));
    }

    #[test]
    fn non_starts_with_predicates_are_untouched() {
        // Unanchored regex and substring contains() do NOT pin the start.
        let unanchored = analyze_filter_conjuncts(scalar_call(
            "regexp_matches",
            vec![col("key"), lit("/a/.*")],
        ));
        assert!(!unanchored
            .iter()
            .any(|e| is_key_cmp(e, Operator::Lt, "/a0")));
        let contains =
            analyze_filter_conjuncts(scalar_call("contains", vec![col("key"), lit("/a/")]));
        assert!(!contains
            .iter()
            .any(|e| is_key_cmp(e, Operator::GtEq, "/a/")));
    }

    #[test]
    fn regex_with_zero_allowing_quantifier_synthesizes_no_range() {
        // `^a?b` matches "bzzz" (optional `a` skipped) and `^a*c` matches "czz";
        // a `key >= 'a' AND key < 'b'`-style range would silently drop those
        // rows. The optional leading char yields no guaranteed prefix, so the
        // rule must add no `key` bound at all.
        let is_key_bound = |e: &Expr| {
            matches!(e, Expr::BinaryExpr(b)
                if matches!(b.left.as_ref(), Expr::Column(c) if c.name == "key")
                    && matches!(b.op, Operator::Lt | Operator::GtEq))
        };
        for pat in ["^a?b", "^a*c"] {
            let conjuncts =
                analyze_filter_conjuncts(scalar_call("regexp_matches", vec![col("key"), lit(pat)]));
            assert!(
                !conjuncts.iter().any(is_key_bound),
                "{pat}: no key range may be synthesized (it would drop matching rows)"
            );
        }
    }

    #[tokio::test]
    async fn rule_is_registered_in_make_ctx() {
        // End-to-end: a generic query through make_ctx must gain the synthesized
        // upper bound in its optimized plan — proof the rule is wired into the
        // real pipeline, not just unit-tested in isolation. `/a0` (succ of `/a/`)
        // appears nowhere in the input SQL, so finding it is unambiguous.
        use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
        use datafusion::arrow::datatypes::Schema as ArrowSchema;

        let ctx = crate::query::make_ctx();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("seq", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["/a/1", "/b/1"])),
                Arc::new(Int64Array::from(vec![1_i64, 2])),
            ],
        )
        .unwrap();
        ctx.register_batch("t", batch).unwrap();

        let plan = ctx
            .sql("SELECT key FROM t WHERE prefix(key, '/a/')")
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap();

        let mut found_upper = false;
        plan.apply(|node| {
            for e in node.expressions() {
                if split_conjunction(&e)
                    .iter()
                    .any(|c| is_key_cmp(c, Operator::Lt, "/a0"))
                {
                    found_upper = true;
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        assert!(
            found_upper,
            "make_ctx must register PrefixRangeRewrite so the key range reaches the optimized plan"
        );
    }
}
