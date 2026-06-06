//! DuckDB-compatibility scalar UDFs.
//!
//! DataFusion lacks three functions the finelog query corpus + FetchLogs rely
//! on; register Rust equivalents that match DuckDB semantics:
//!
//! - `prefix(text, prefix) -> bool` — DuckDB's literal-prefix predicate
//!   (`text` starts with `prefix`).
//! - `regexp_matches(text, pattern) -> bool` — DuckDB's partial-match regex
//!   (the pattern matches *somewhere* in `text`, not anchored).
//! - `contains(text, sub) -> bool` — literal substring containment (DuckDB's
//!   `contains` treats `sub` literally; `%`/`_` are NOT wildcards).
//!
//! Each returns NULL when any argument is NULL (matching DuckDB's scalar NULL
//! propagation). A regex that fails to compile yields a DataFusion execution
//! error (surfaced to the client as `invalid_argument`, mirroring DuckDB's
//! parse-error path).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, StringArray};
use arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{create_udf, ColumnarValue, ScalarUDF, Volatility};
use regex::Regex;

/// Register `prefix`, `regexp_matches`, and `contains` on `ctx`.
pub fn register_compat_udfs(ctx: &datafusion::prelude::SessionContext) {
    ctx.register_udf(prefix_udf());
    ctx.register_udf(regexp_matches_udf());
    ctx.register_udf(contains_udf());
}

/// Coerce a `ColumnarValue` to a string array of length `n`, returning a
/// borrowed `StringArray`. Scalars are broadcast.
fn to_string_array(value: &ColumnarValue, n: usize) -> DFResult<ArrayRef> {
    let arr = value.clone().into_array(n)?;
    if arr.data_type() == &DataType::Utf8 {
        Ok(arr)
    } else {
        arrow::compute::cast(&arr, &DataType::Utf8)
            .map_err(|e| DataFusionError::Execution(format!("expected string argument: {e}")))
    }
}

fn binary_string_bool(
    args: &[ColumnarValue],
    name: &str,
    op: impl Fn(&str, &str) -> DFResult<bool>,
) -> DFResult<ColumnarValue> {
    if args.len() != 2 {
        return Err(DataFusionError::Execution(format!(
            "{name} expects 2 arguments, got {}",
            args.len()
        )));
    }
    // Determine the row count from the first array arg (scalars broadcast).
    let n = args
        .iter()
        .find_map(|a| match a {
            ColumnarValue::Array(arr) => Some(arr.len()),
            ColumnarValue::Scalar(_) => None,
        })
        .unwrap_or(1);
    let lhs = to_string_array(&args[0], n)?;
    let rhs = to_string_array(&args[1], n)?;
    let lhs = lhs
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("cast to Utf8 yields StringArray");
    let rhs = rhs
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("cast to Utf8 yields StringArray");
    let mut out = BooleanArray::builder(n);
    for i in 0..n {
        if lhs.is_null(i) || rhs.is_null(i) {
            out.append_null();
        } else {
            out.append_value(op(lhs.value(i), rhs.value(i))?);
        }
    }
    Ok(ColumnarValue::Array(Arc::new(out.finish())))
}

fn prefix_udf() -> ScalarUDF {
    create_udf(
        "prefix",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            binary_string_bool(args, "prefix", |text, p| Ok(text.starts_with(p)))
        }),
    )
}

fn regexp_matches_udf() -> ScalarUDF {
    create_udf(
        "regexp_matches",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            binary_string_bool(args, "regexp_matches", |text, pattern| {
                // DuckDB `regexp_matches` is a partial (unanchored) match.
                let re = Regex::new(pattern).map_err(|e| {
                    DataFusionError::Execution(format!("invalid regex {pattern:?}: {e}"))
                })?;
                Ok(re.is_match(text))
            })
        }),
    )
}

fn contains_udf() -> ScalarUDF {
    create_udf(
        "contains",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            // Literal substring containment — `%`/`_` are NOT wildcards.
            binary_string_bool(args, "contains", |text, sub| Ok(text.contains(sub)))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use datafusion::prelude::SessionContext;

    /// Evaluate a 2-arg string→bool UDF over column inputs through a real
    /// `SessionContext`, returning the boolean results (NULLs as `None`).
    ///
    /// Going through SQL exercises the registered UDF exactly as the query path
    /// does (the raw `invoke_with_args` API is verbose and version-fragile).
    async fn eval(name: &str, lhs: Vec<Option<&str>>, rhs: Vec<Option<&str>>) -> Vec<Option<bool>> {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        let n = lhs.len();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("l", DataType::Utf8, true),
            Field::new("r", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..n as i64)) as ArrayRef,
                Arc::new(StringArray::from(lhs)) as ArrayRef,
                Arc::new(StringArray::from(rhs)) as ArrayRef,
            ],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_compat_udfs(&ctx);
        ctx.register_batch("t", batch).unwrap();
        // Explicit `id` column preserves input order for the assertion.
        let out = ctx
            .sql(&format!("SELECT {name}(l, r) AS m FROM t ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut got = Vec::with_capacity(n);
        for b in &out {
            let col = b.column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
            for i in 0..col.len() {
                got.push(if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i))
                });
            }
        }
        got
    }

    #[tokio::test]
    async fn prefix_udf_matches_literal_prefix() {
        assert_eq!(
            eval(
                "prefix",
                vec![Some("/a/b"), Some("/a/b"), Some("/x")],
                vec![Some("/a"), Some("/x"), Some("/a")],
            )
            .await,
            vec![Some(true), Some(false), Some(false)]
        );
        // NULL propagation.
        assert_eq!(
            eval("prefix", vec![None, Some("/a")], vec![Some("/a"), None]).await,
            vec![None, None]
        );
    }

    #[tokio::test]
    async fn prefix_udf_treats_metachars_literally() {
        // `+` and `.` are literal, not regex.
        assert_eq!(
            eval(
                "prefix",
                vec![Some("/job/curation-9e+20"), Some("/job/literal.value")],
                vec![Some("/job/curation-9e+"), Some("/job/literal.")],
            )
            .await,
            vec![Some(true), Some(true)]
        );
    }

    #[tokio::test]
    async fn regexp_matches_udf_partial_match() {
        assert_eq!(
            eval(
                "regexp_matches",
                vec![Some("/job/test/0"), Some("/job/other/0")],
                vec![Some("/job/test/.*"), Some("/job/test/.*")],
            )
            .await,
            vec![Some(true), Some(false)]
        );
        // Unanchored partial match: a bare literal matches anywhere.
        assert_eq!(
            eval("regexp_matches", vec![Some("abc")], vec![Some("b")]).await,
            vec![Some(true)]
        );
    }

    #[tokio::test]
    async fn contains_udf_treats_wildcards_literally() {
        // `%` and `_` are literal, not LIKE wildcards.
        assert_eq!(
            eval(
                "contains",
                vec![Some("100% done"), Some("a_b_c"), Some("plain")],
                vec![Some("100%"), Some("a_b"), Some("100%")],
            )
            .await,
            vec![Some(true), Some(true), Some(false)]
        );
    }
}
