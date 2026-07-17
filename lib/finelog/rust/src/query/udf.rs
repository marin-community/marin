//! DuckDB-compatibility scalar UDFs.
//!
//! DataFusion lacks functions the finelog query corpus + FetchLogs rely on;
//! register Rust equivalents that match DuckDB semantics:
//!
//! - `prefix(text, prefix) -> bool` — DuckDB's literal-prefix predicate
//!   (`text` starts with `prefix`).
//! - `regexp_matches(text, pattern) -> bool` — DuckDB's partial-match regex
//!   (the pattern matches *somewhere* in `text`, not anchored).
//! - `contains(text, sub) -> bool` — literal substring containment (DuckDB's
//!   `contains` treats `sub` literally; `%`/`_` are NOT wildcards).
//! - `json_get(text, key) -> text` — extract a top-level object key from a
//!   JSON-string column. This makes JSON-encoded label columns (e.g.
//!   `infra/probes`' `labels`) first-class in SQL, so callers can
//!   `WHERE json_get(labels, 'scope') = 'fleet'` instead of flattening the
//!   string in Python or matching it with `regexp_matches`.
//!
//! Each returns NULL when any argument is NULL (matching DuckDB's scalar NULL
//! propagation). A regex that fails to compile yields a DataFusion execution
//! error (surfaced to the client as `invalid_argument`, mirroring DuckDB's
//! parse-error path). `json_get` returns NULL (never errors) when the input is
//! not a JSON object or the key is absent, so a malformed row cannot fail the
//! whole scan.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, StringArray, StringBuilder};
use arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{create_udf, ColumnarValue, ScalarUDF, Volatility};
use regex::Regex;

/// Register `prefix`, `regexp_matches`, `contains`, and `json_get` on `ctx`.
pub fn register_compat_udfs(ctx: &datafusion::prelude::SessionContext) {
    ctx.register_udf(prefix_udf());
    ctx.register_udf(regexp_matches_udf());
    ctx.register_udf(contains_udf());
    ctx.register_udf(json_get_udf());
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

/// Evaluate a 2-arg string→optional-string UDF. Rows where either input is NULL
/// or `op` returns `None` are emitted as NULL.
fn binary_string_opt_string(
    args: &[ColumnarValue],
    name: &str,
    op: impl Fn(&str, &str) -> Option<String>,
) -> DFResult<ColumnarValue> {
    if args.len() != 2 {
        return Err(DataFusionError::Execution(format!(
            "{name} expects 2 arguments, got {}",
            args.len()
        )));
    }
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
    let mut out = StringBuilder::with_capacity(n, 0);
    for i in 0..n {
        if lhs.is_null(i) || rhs.is_null(i) {
            out.append_null();
        } else {
            match op(lhs.value(i), rhs.value(i)) {
                Some(s) => out.append_value(s),
                None => out.append_null(),
            }
        }
    }
    Ok(ColumnarValue::Array(Arc::new(out.finish())))
}

/// Extract top-level `key` from the JSON object encoded in `text`.
///
/// String values are returned unquoted (so `json_get(labels, 'scope')` compares
/// directly against `'fleet'`); other scalars and nested arrays/objects are
/// returned as their compact JSON text. Returns `None` (→ SQL NULL) when `text`
/// is not a JSON object, the key is absent, or its value is JSON `null`.
fn json_get_value(text: &str, key: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    match value.as_object()?.get(key)? {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

fn json_get_udf() -> ScalarUDF {
    create_udf(
        "json_get",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Utf8,
        Volatility::Immutable,
        Arc::new(|args: &[ColumnarValue]| {
            binary_string_opt_string(args, "json_get", json_get_value)
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

    /// Like `eval`, but for a string-returning UDF (NULLs as `None`).
    async fn eval_str(
        name: &str,
        lhs: Vec<Option<&str>>,
        rhs: Vec<Option<&str>>,
    ) -> Vec<Option<String>> {
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
        let out = ctx
            .sql(&format!("SELECT {name}(l, r) AS m FROM t ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut got = Vec::with_capacity(n);
        for b in &out {
            let col = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            for i in 0..col.len() {
                got.push(if col.is_null(i) {
                    None
                } else {
                    Some(col.value(i).to_string())
                });
            }
        }
        got
    }

    #[tokio::test]
    async fn json_get_extracts_string_values_and_nulls_on_absence() {
        let labels = r#"{"scope":"fleet","region":"us-east"}"#;
        assert_eq!(
            eval_str(
                "json_get",
                vec![Some(labels), Some(labels), Some(labels)],
                vec![Some("scope"), Some("region"), Some("missing")],
            )
            .await,
            vec![
                Some("fleet".to_string()),
                Some("us-east".to_string()),
                None
            ]
        );
        // NULL propagation, non-object input, and malformed JSON all yield NULL
        // rather than erroring the scan.
        assert_eq!(
            eval_str(
                "json_get",
                vec![None, Some("[1,2,3]"), Some("not json")],
                vec![Some("scope"), Some("scope"), Some("scope")],
            )
            .await,
            vec![None, None, None]
        );
    }

    #[tokio::test]
    async fn json_get_returns_compact_json_for_nonstring_values() {
        // Numbers, bools, and nested values come back as their JSON text, and a
        // JSON `null` value is SQL NULL.
        let obj = r#"{"n":42,"ok":true,"nested":{"a":1},"z":null}"#;
        assert_eq!(
            eval_str(
                "json_get",
                vec![Some(obj), Some(obj), Some(obj), Some(obj)],
                vec![Some("n"), Some("ok"), Some("nested"), Some("z")],
            )
            .await,
            vec![
                Some("42".to_string()),
                Some("true".to_string()),
                Some(r#"{"a":1}"#.to_string()),
                None
            ]
        );
    }

    #[tokio::test]
    async fn json_get_groups_by_extracted_label() {
        // The motivating query: GROUP BY json_get(labels, 'region').
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        let labels = vec![
            Some(r#"{"region":"us-east"}"#),
            Some(r#"{"region":"us-east"}"#),
            Some(r#"{"region":"eu-west"}"#),
        ];
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("labels", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..3)) as ArrayRef,
                Arc::new(StringArray::from(labels)) as ArrayRef,
            ],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_compat_udfs(&ctx);
        ctx.register_batch("t", batch).unwrap();
        let out = ctx
            .sql(
                "SELECT json_get(labels, 'region') AS region, count(*) AS c \
                 FROM t GROUP BY region ORDER BY region",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for b in &out {
            let region = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let count = b
                .column(1)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            for i in 0..b.num_rows() {
                rows.push((region.value(i).to_string(), count.value(i)));
            }
        }
        assert_eq!(
            rows,
            vec![("eu-west".to_string(), 1), ("us-east".to_string(), 2)]
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
