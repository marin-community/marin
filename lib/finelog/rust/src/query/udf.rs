//! Scalar UDFs for the finelog read engine.
//!
//! DataFusion lacks functions the finelog query corpus + FetchLogs rely on;
//! register Rust equivalents. Three match DuckDB predicate semantics; the
//! `json_*` family extracts values from a JSON-string column (e.g. infra/probes'
//! `labels`, and any EAV label set) so they are first-class in SQL — filterable
//! and groupable — instead of forcing raw-string `regexp_matches`/`contains`
//! workarounds:
//!
//! - `prefix(text, prefix) -> bool` — DuckDB's literal-prefix predicate
//!   (`text` starts with `prefix`).
//! - `regexp_matches(text, pattern) -> bool` — DuckDB's partial-match regex
//!   (the pattern matches *somewhere* in `text`, not anchored).
//! - `contains(text, sub) -> bool` — literal substring containment (DuckDB's
//!   `contains` treats `sub` literally; `%`/`_` are NOT wildcards).
//! - `json_get(text, key) -> text` — top-level object key `key` as text (a JSON
//!   string unquoted; other scalars and nested arrays/objects as their compact
//!   JSON form).
//! - `json_get_int(text, key) -> bigint` — `key` when its value is a JSON
//!   integer. A string-encoded number is NOT coerced (that is
//!   `CAST(json_get(...) AS BIGINT)`).
//! - `json_get_float(text, key) -> double` — `key` when its value is any JSON
//!   number.
//! - `json_get_bool(text, key) -> boolean` — `key` when its value is a JSON
//!   boolean.
//! - `json_contains(text, key) -> boolean` — whether `text` is a JSON object
//!   with a top-level `key` (a present key whose value is JSON `null` counts).
//! - `json_length(text) -> bigint` — element count of a top-level JSON array or
//!   object.
//!
//! The `json_*` document argument is duck-typed: it may be a `Utf8` column of
//! JSON text or a native `Map<Utf8,Utf8>` column, so a query keeps working
//! unchanged if a column migrates from a JSON string to a map (a map value is
//! treated as the equivalent JSON string value).
//!
//! NULL semantics: every UDF returns NULL when any argument is NULL (DuckDB's
//! scalar NULL propagation). The `json_get*` extractors additionally return NULL
//! when `text` is not a JSON object, the key is absent, or the value's JSON type
//! does not match the getter (and `json_get` maps an explicit JSON `null` to
//! NULL); `json_length` returns NULL when `text` is not a JSON array or object.
//! `json_contains` is a total predicate — false, never NULL, for a non-object or
//! an absent key. A regex that fails to compile yields a DataFusion execution
//! error (surfaced to the client as `invalid_argument`, mirroring DuckDB's
//! parse-error path).

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Builder, Int64Builder, MapArray,
    StringArray, StringBuilder,
};
use arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{
    create_udf, ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use regex::Regex;
use serde_json::Value as JsonValue;

/// Register the finelog scalar UDFs (`prefix`, `regexp_matches`, `contains`, and
/// the `json_*` extraction family) on `ctx`.
pub fn register_scalar_udfs(ctx: &datafusion::prelude::SessionContext) {
    ctx.register_udf(prefix_udf());
    ctx.register_udf(regexp_matches_udf());
    ctx.register_udf(contains_udf());
    for kind in [
        JsonKind::Get,
        JsonKind::GetInt,
        JsonKind::GetFloat,
        JsonKind::GetBool,
        JsonKind::Contains,
        JsonKind::Length,
    ] {
        ctx.register_udf(ScalarUDF::from(JsonUdf::new(kind)));
    }
}

/// Cast an already-materialized array to `Utf8` (a cheap clone when it already
/// is). A non-castable type (e.g. `Map`) is a caller error.
fn array_as_utf8(arr: &ArrayRef) -> DFResult<ArrayRef> {
    if arr.data_type() == &DataType::Utf8 {
        Ok(Arc::clone(arr))
    } else {
        arrow::compute::cast(arr, &DataType::Utf8)
            .map_err(|e| DataFusionError::Execution(format!("expected string argument: {e}")))
    }
}

/// Coerce a `ColumnarValue` to a `Utf8` array of length `n` (scalars broadcast).
fn to_string_array(value: &ColumnarValue, n: usize) -> DFResult<ArrayRef> {
    array_as_utf8(&value.clone().into_array(n)?)
}

/// Borrow an `ArrayRef` known to hold `Utf8` as a `StringArray`.
fn as_str(arr: &ArrayRef) -> &StringArray {
    arr.as_any()
        .downcast_ref::<StringArray>()
        .expect("cast to Utf8 yields StringArray")
}

/// Resolve a 2-arg UDF call to its row count and both arguments as `Utf8` arrays
/// of that length (scalars broadcast). Shared prologue for the binary string
/// kernels below.
fn resolve_binary_string_args(
    args: &[ColumnarValue],
    name: &str,
) -> DFResult<(usize, ArrayRef, ArrayRef)> {
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
    Ok((n, lhs, rhs))
}

fn binary_string_bool(
    args: &[ColumnarValue],
    name: &str,
    op: impl Fn(&str, &str) -> DFResult<bool>,
) -> DFResult<ColumnarValue> {
    let (n, lhs, rhs) = resolve_binary_string_args(args, name)?;
    let (lhs, rhs) = (as_str(&lhs), as_str(&rhs));
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

/// The raw JSON value at top-level object key `key` in the document `text`, if
/// `text` parses as a JSON object that contains `key`.
///
/// `None` when `text` is not a JSON object (invalid JSON, or a top-level
/// scalar/array) or the key is absent; a key present with an explicit JSON
/// `null` value returns `Some(JsonValue::Null)`.
fn json_object_value(text: &str, key: &str) -> Option<JsonValue> {
    match serde_json::from_str::<JsonValue>(text).ok()? {
        JsonValue::Object(mut map) => map.remove(key),
        _ => None,
    }
}

/// Render a resolved JSON value as text for `json_get`: a JSON string unquoted,
/// other scalars and nested arrays/objects as their compact JSON form, an
/// explicit JSON `null` as SQL NULL.
fn json_value_as_text(value: JsonValue) -> Option<String> {
    match value {
        JsonValue::Null => None,
        JsonValue::String(s) => Some(s),
        other => Some(other.to_string()),
    }
}

/// Element count of a top-level JSON array (its length) or object (its key
/// count); `None` for a scalar, an explicit `null`, or non-JSON input.
fn json_length_value(text: &str) -> Option<i64> {
    match serde_json::from_str::<JsonValue>(text).ok()? {
        JsonValue::Array(a) => Some(a.len() as i64),
        JsonValue::Object(o) => Some(o.len() as i64),
        _ => None,
    }
}

/// Per-row outcome of a `json_*` document + key lookup. Distinguishes a NULL
/// input cell (NULL output for every function) from an input that is present but
/// has no value at the key (NULL for the getters, `false` for `json_contains`).
enum Resolved {
    /// The document or the key cell is SQL NULL.
    InputNull,
    /// Inputs are present, but the document is not a JSON object / the key is
    /// absent from the object or map.
    Absent,
    /// The value at the key. For a `Map<Utf8,Utf8>` it is a `String` (or `Null`
    /// for a null map value); for a JSON document it is the raw JSON value.
    Value(JsonValue),
}

impl Resolved {
    fn from_object_value(v: Option<JsonValue>) -> Self {
        match v {
            Some(jv) => Resolved::Value(jv),
            None => Resolved::Absent,
        }
    }
}

/// Look up `key` in row `row` of a `Map<Utf8,Utf8>` column.
fn map_lookup(
    keys: &StringArray,
    values: &StringArray,
    offsets: &[i32],
    row: usize,
    key: &str,
) -> Resolved {
    let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
    for j in start..end {
        if !keys.is_null(j) && keys.value(j) == key {
            return Resolved::Value(if values.is_null(j) {
                JsonValue::Null
            } else {
                JsonValue::String(values.value(j).to_string())
            });
        }
    }
    Resolved::Absent
}

/// Which `json_*` function a [`JsonUdf`] implements; fixes its name, arity, and
/// Arrow return type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum JsonKind {
    Get,
    GetInt,
    GetFloat,
    GetBool,
    Contains,
    Length,
}

impl JsonKind {
    fn udf_name(self) -> &'static str {
        match self {
            JsonKind::Get => "json_get",
            JsonKind::GetInt => "json_get_int",
            JsonKind::GetFloat => "json_get_float",
            JsonKind::GetBool => "json_get_bool",
            JsonKind::Contains => "json_contains",
            JsonKind::Length => "json_length",
        }
    }

    fn return_type(self) -> DataType {
        match self {
            JsonKind::Get => DataType::Utf8,
            JsonKind::GetInt | JsonKind::Length => DataType::Int64,
            JsonKind::GetFloat => DataType::Float64,
            JsonKind::GetBool | JsonKind::Contains => DataType::Boolean,
        }
    }

    fn arity(self) -> usize {
        match self {
            JsonKind::Length => 1,
            _ => 2,
        }
    }
}

/// A duck-typed `json_*` scalar UDF. The document argument may be a `Utf8` column
/// of JSON text or a native `Map<Utf8,Utf8>` column, so a query keeps working if
/// a column's physical type migrates from a JSON string to a map. The `any`
/// signature admits either; the actual type is inspected per call.
#[derive(Debug, PartialEq, Eq, Hash)]
struct JsonUdf {
    kind: JsonKind,
    signature: Signature,
}

impl JsonUdf {
    fn new(kind: JsonKind) -> Self {
        Self {
            kind,
            signature: Signature::any(kind.arity(), Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for JsonUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        self.kind.udf_name()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(self.kind.return_type())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let n = args.number_rows;
        let args = args.args;
        if args.len() != self.kind.arity() {
            return Err(DataFusionError::Execution(format!(
                "{} expects {} argument(s), got {}",
                self.kind.udf_name(),
                self.kind.arity(),
                args.len()
            )));
        }
        if let JsonKind::Length = self.kind {
            return json_length_column(&args[0], n);
        }
        let doc = args[0].clone().into_array(n)?;
        let key = to_string_array(&args[1], n)?;
        json_get_column(self.kind, &doc, as_str(&key), n)
    }
}

/// Evaluate a 2-arg `json_*` accessor over `doc` (a `Utf8` JSON or
/// `Map<Utf8,Utf8>` column) and `key`, dispatching on the document's physical
/// type and building the kind's output array.
fn json_get_column(
    kind: JsonKind,
    doc: &ArrayRef,
    key: &StringArray,
    n: usize,
) -> DFResult<ColumnarValue> {
    let out = match doc.data_type() {
        DataType::Map(_, _) => {
            let map = doc
                .as_any()
                .downcast_ref::<MapArray>()
                .expect("Map DataType downcasts to MapArray");
            let keys = as_str(map.keys());
            let values = as_str(map.values());
            let offsets = map.value_offsets();
            build_json_output(kind, n, |i| {
                if map.is_null(i) || key.is_null(i) {
                    Resolved::InputNull
                } else {
                    map_lookup(keys, values, offsets, i, key.value(i))
                }
            })
        }
        _ => {
            let text = array_as_utf8(doc)?;
            let text = as_str(&text);
            build_json_output(kind, n, |i| {
                if text.is_null(i) || key.is_null(i) {
                    Resolved::InputNull
                } else {
                    Resolved::from_object_value(json_object_value(text.value(i), key.value(i)))
                }
            })
        }
    };
    Ok(ColumnarValue::Array(out))
}

/// Build the output array for a 2-arg `json_*` kind from a per-row [`Resolved`].
fn build_json_output(kind: JsonKind, n: usize, resolve: impl Fn(usize) -> Resolved) -> ArrayRef {
    match kind {
        JsonKind::Get => {
            let mut b = StringBuilder::new();
            for i in 0..n {
                match resolve(i) {
                    Resolved::Value(jv) => b.append_option(json_value_as_text(jv)),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        JsonKind::GetInt => {
            let mut b = Int64Builder::with_capacity(n);
            for i in 0..n {
                match resolve(i) {
                    Resolved::Value(jv) => b.append_option(jv.as_i64()),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        JsonKind::GetFloat => {
            let mut b = Float64Builder::with_capacity(n);
            for i in 0..n {
                match resolve(i) {
                    Resolved::Value(jv) => b.append_option(jv.as_f64()),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        JsonKind::GetBool => {
            let mut b = BooleanBuilder::with_capacity(n);
            for i in 0..n {
                match resolve(i) {
                    Resolved::Value(jv) => b.append_option(jv.as_bool()),
                    _ => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        JsonKind::Contains => {
            let mut b = BooleanBuilder::with_capacity(n);
            for i in 0..n {
                match resolve(i) {
                    Resolved::InputNull => b.append_null(),
                    Resolved::Absent => b.append_value(false),
                    Resolved::Value(_) => b.append_value(true),
                }
            }
            Arc::new(b.finish())
        }
        JsonKind::Length => unreachable!("json_length is unary; handled by json_length_column"),
    }
}

/// Evaluate `json_length` over a `Utf8` JSON or `Map<Utf8,Utf8>` column.
fn json_length_column(arg: &ColumnarValue, n: usize) -> DFResult<ColumnarValue> {
    let arr = arg.clone().into_array(n)?;
    let mut b = Int64Builder::with_capacity(n);
    match arr.data_type() {
        DataType::Map(_, _) => {
            let map = arr
                .as_any()
                .downcast_ref::<MapArray>()
                .expect("Map DataType downcasts to MapArray");
            let offsets = map.value_offsets();
            for i in 0..n {
                if map.is_null(i) {
                    b.append_null();
                } else {
                    b.append_value(i64::from(offsets[i + 1] - offsets[i]));
                }
            }
        }
        _ => {
            let text = array_as_utf8(&arr)?;
            let text = as_str(&text);
            for i in 0..n {
                if text.is_null(i) {
                    b.append_null();
                } else {
                    b.append_option(json_length_value(text.value(i)));
                }
            }
        }
    }
    Ok(ColumnarValue::Array(Arc::new(b.finish())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;

    /// Register a `(id, l, r)` batch as table `t` and evaluate
    /// `SELECT {proj} AS m FROM t ORDER BY id` through a real `SessionContext`,
    /// returning the result batches.
    ///
    /// Going through SQL exercises the registered UDFs exactly as the query path
    /// does (the raw `invoke_with_args` API is verbose and version-fragile); the
    /// `id` column pins result order for the assertions.
    async fn run_udf(
        proj: &str,
        lhs: Vec<Option<&str>>,
        rhs: Vec<Option<&str>>,
    ) -> Vec<RecordBatch> {
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
        register_scalar_udfs(&ctx);
        ctx.register_batch("t", batch).unwrap();
        ctx.sql(&format!("SELECT {proj} AS m FROM t ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
    }

    /// Collect column 0 of single-column result `batches` (downcast to `A`) into
    /// `Vec<Option<V>>`, mapping each non-null row through `get` and NULLs to
    /// `None`.
    fn column<A: Array + 'static, V>(
        batches: &[RecordBatch],
        get: impl Fn(&A, usize) -> V,
    ) -> Vec<Option<V>> {
        let mut got = Vec::new();
        for b in batches {
            let col = b.column(0).as_any().downcast_ref::<A>().unwrap();
            for i in 0..col.len() {
                got.push(col.is_valid(i).then(|| get(col, i)));
            }
        }
        got
    }

    /// Evaluate a 2-arg `(l, r)` UDF returning `Boolean`.
    async fn eval(name: &str, lhs: Vec<Option<&str>>, rhs: Vec<Option<&str>>) -> Vec<Option<bool>> {
        column::<BooleanArray, _>(
            &run_udf(&format!("{name}(l, r)"), lhs, rhs).await,
            |c, i| c.value(i),
        )
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

    /// Evaluate a 2-arg `(l, r)` UDF returning `Utf8`.
    async fn eval_str(
        name: &str,
        lhs: Vec<Option<&str>>,
        rhs: Vec<Option<&str>>,
    ) -> Vec<Option<String>> {
        column::<StringArray, _>(
            &run_udf(&format!("{name}(l, r)"), lhs, rhs).await,
            |c, i| c.value(i).to_string(),
        )
    }

    /// Evaluate a 2-arg `(l, r)` UDF returning `Int64`.
    async fn eval_int(
        name: &str,
        lhs: Vec<Option<&str>>,
        rhs: Vec<Option<&str>>,
    ) -> Vec<Option<i64>> {
        column::<Int64Array, _>(
            &run_udf(&format!("{name}(l, r)"), lhs, rhs).await,
            |c, i| c.value(i),
        )
    }

    /// Evaluate a 2-arg `(l, r)` UDF returning `Float64`.
    async fn eval_float(
        name: &str,
        lhs: Vec<Option<&str>>,
        rhs: Vec<Option<&str>>,
    ) -> Vec<Option<f64>> {
        column::<Float64Array, _>(
            &run_udf(&format!("{name}(l, r)"), lhs, rhs).await,
            |c, i| c.value(i),
        )
    }

    fn some(s: &str) -> Option<String> {
        Some(s.to_string())
    }

    #[tokio::test]
    async fn json_get_extracts_top_level_string_value() {
        let doc = r#"{"scope":"fleet","region":"us-east"}"#;
        assert_eq!(
            eval_str(
                "json_get",
                vec![Some(doc), Some(doc)],
                vec![Some("scope"), Some("region")],
            )
            .await,
            vec![some("fleet"), some("us-east")]
        );
    }

    #[tokio::test]
    async fn json_get_absent_key_is_null() {
        assert_eq!(
            eval_str(
                "json_get",
                vec![Some(r#"{"scope":"fleet"}"#)],
                vec![Some("region")],
            )
            .await,
            vec![None]
        );
    }

    #[tokio::test]
    async fn json_get_non_object_input_is_null() {
        // A JSON array, a JSON scalar, and non-JSON text all yield NULL —
        // `json_get` extracts keys only from a top-level object.
        assert_eq!(
            eval_str(
                "json_get",
                vec![Some("[1,2,3]"), Some("42"), Some("not json")],
                vec![Some("0"), Some("x"), Some("x")],
            )
            .await,
            vec![None, None, None]
        );
    }

    #[tokio::test]
    async fn json_get_null_propagation() {
        // NULL in either argument propagates to NULL.
        assert_eq!(
            eval_str(
                "json_get",
                vec![None, Some(r#"{"a":"b"}"#)],
                vec![Some("a"), None],
            )
            .await,
            vec![None, None]
        );
    }

    #[tokio::test]
    async fn json_get_non_string_values_render_as_json_text() {
        // Numbers, booleans, and nested structures come back as their compact
        // JSON form; an explicit JSON `null` value is SQL NULL.
        let doc = r#"{"n":5,"b":true,"nested":{"a":1},"arr":[1,2],"z":null}"#;
        assert_eq!(
            eval_str(
                "json_get",
                vec![Some(doc), Some(doc), Some(doc), Some(doc), Some(doc)],
                vec![Some("n"), Some("b"), Some("nested"), Some("arr"), Some("z")],
            )
            .await,
            vec![
                some("5"),
                some("true"),
                some(r#"{"a":1}"#),
                some("[1,2]"),
                None
            ]
        );
    }

    /// The motivating use case: filter and group a JSON-string `labels` column in
    /// SQL, end-to-end through the query engine.
    #[tokio::test]
    async fn json_get_filters_and_groups_labels() {
        let labels = vec![
            Some(r#"{"scope":"fleet","region":"us-east"}"#),
            Some(r#"{"scope":"fleet","region":"us-west"}"#),
            Some(r#"{"scope":"fleet","region":"us-east"}"#),
            Some(r#"{"scope":"local","region":"us-east"}"#),
        ];
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "labels",
            DataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(labels)) as ArrayRef],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_scalar_udfs(&ctx);
        ctx.register_batch("probes", batch).unwrap();

        // WHERE + GROUP BY over the extracted key, ordered for a stable assert.
        let out = ctx
            .sql(
                "SELECT json_get(labels, 'region') AS region, count(*) AS n \
                 FROM probes \
                 WHERE json_get(labels, 'scope') = 'fleet' \
                 GROUP BY json_get(labels, 'region') \
                 ORDER BY region",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let mut rows = Vec::new();
        for b in &out {
            let region = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            let n = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                rows.push((region.value(i).to_string(), n.value(i)));
            }
        }
        // The two 'fleet' us-east rows collapse; the 'local' row is filtered out.
        assert_eq!(
            rows,
            vec![("us-east".to_string(), 2), ("us-west".to_string(), 1)]
        );
    }

    #[tokio::test]
    async fn json_get_int_extracts_only_json_integers() {
        // A JSON integer extracts; a float, a string-encoded number, and a bool
        // all yield NULL (`json_get_int` does not coerce — that is a CAST).
        let doc = r#"{"i":42,"f":1.5,"s":"7","b":true}"#;
        assert_eq!(
            eval_int(
                "json_get_int",
                vec![Some(doc), Some(doc), Some(doc), Some(doc)],
                vec![Some("i"), Some("f"), Some("s"), Some("b")],
            )
            .await,
            vec![Some(42), None, None, None]
        );
    }

    #[tokio::test]
    async fn json_get_float_extracts_any_json_number() {
        // Both an integer and a float come back as `f64`; a string and a bool are
        // NULL.
        let doc = r#"{"i":42,"f":1.5,"s":"7","b":true}"#;
        assert_eq!(
            eval_float(
                "json_get_float",
                vec![Some(doc), Some(doc), Some(doc), Some(doc)],
                vec![Some("i"), Some("f"), Some("s"), Some("b")],
            )
            .await,
            vec![Some(42.0), Some(1.5), None, None]
        );
    }

    #[tokio::test]
    async fn json_get_bool_extracts_only_json_booleans() {
        // Booleans extract; a number and a string-encoded bool are NULL.
        let doc = r#"{"t":true,"f":false,"n":1,"s":"true"}"#;
        assert_eq!(
            eval(
                "json_get_bool",
                vec![Some(doc), Some(doc), Some(doc), Some(doc)],
                vec![Some("t"), Some("f"), Some("n"), Some("s")],
            )
            .await,
            vec![Some(true), Some(false), None, None]
        );
    }

    #[tokio::test]
    async fn json_contains_is_a_total_key_existence_predicate() {
        // Present keys (including one with a JSON `null` value) are true; an
        // absent key, a non-object, and non-JSON are all false (never NULL).
        assert_eq!(
            eval(
                "json_contains",
                vec![
                    Some(r#"{"a":"x","z":null}"#),
                    Some(r#"{"a":"x","z":null}"#),
                    Some(r#"{"a":"x"}"#),
                    Some("[1,2,3]"),
                    Some("not json"),
                ],
                vec![Some("a"), Some("z"), Some("missing"), Some("0"), Some("k")],
            )
            .await,
            vec![
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
    }

    #[tokio::test]
    async fn json_length_counts_array_and_object_elements() {
        // Array length and object key count; a scalar, a number, and non-JSON are
        // NULL, and a NULL input propagates.
        let docs = vec![
            Some("[1,2,3]"),
            Some(r#"{"a":1,"b":2}"#),
            Some(r#""scalar""#),
            Some("5"),
            Some("not json"),
            None,
        ];
        let n = docs.len();
        let got = column::<Int64Array, _>(
            &run_udf("json_length(l)", docs, vec![None; n]).await,
            |c, i| c.value(i),
        );
        assert_eq!(got, vec![Some(3), Some(2), None, None, None, None]);
    }

    #[tokio::test]
    async fn json_typed_getters_propagate_null_arguments() {
        // A NULL document or NULL key is NULL for every typed getter, and even the
        // total `json_contains` NULL-propagates a NULL argument.
        assert_eq!(
            eval_int("json_get_int", vec![None], vec![Some("k")]).await,
            vec![None]
        );
        assert_eq!(
            eval_float("json_get_float", vec![Some(r#"{"k":1}"#)], vec![None]).await,
            vec![None]
        );
        assert_eq!(
            eval("json_get_bool", vec![None], vec![Some("k")]).await,
            vec![None]
        );
        assert_eq!(
            eval("json_contains", vec![Some(r#"{"k":1}"#)], vec![None]).await,
            vec![None]
        );
    }

    /// One `Map<Utf8,Utf8>` cell for [`run_map_udf`]: `None` is a NULL map cell,
    /// `Some(entries)` is a map of `(key, Option<value>)` pairs (a `None` value
    /// is a null map value).
    type MapRow<'a> = Option<Vec<(&'a str, Option<&'a str>)>>;

    /// Register `(id, m, r)` with a native `Map<Utf8,Utf8>` column `m` and run
    /// `SELECT {proj} AS out FROM t ORDER BY id`.
    async fn run_map_udf(
        proj: &str,
        docs: Vec<MapRow<'_>>,
        keys: Vec<Option<&str>>,
    ) -> Vec<RecordBatch> {
        use arrow::array::MapBuilder;
        let n = docs.len();
        let mut mb = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        for row in docs {
            match row {
                None => mb.append(false).unwrap(),
                Some(entries) => {
                    for (k, v) in entries {
                        mb.keys().append_value(k);
                        match v {
                            Some(val) => mb.values().append_value(val),
                            None => mb.values().append_null(),
                        }
                    }
                    mb.append(true).unwrap();
                }
            }
        }
        let map = mb.finish();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("m", map.data_type().clone(), true),
            Field::new("r", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..n as i64)) as ArrayRef,
                Arc::new(map) as ArrayRef,
                Arc::new(StringArray::from(keys)) as ArrayRef,
            ],
        )
        .unwrap();
        let ctx = SessionContext::new();
        register_scalar_udfs(&ctx);
        ctx.register_batch("t", batch).unwrap();
        ctx.sql(&format!("SELECT {proj} AS out FROM t ORDER BY id"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn json_get_reads_a_native_map_column() {
        // Duck-typed over a Map<Utf8,Utf8>: json_get returns the value string; an
        // absent key, a NULL map cell, and a null map value all yield NULL.
        let m = Some(vec![("scope", Some("fleet")), ("region", Some("us-east"))]);
        let got = column::<StringArray, _>(
            &run_map_udf(
                "json_get(m, r)",
                vec![
                    m.clone(),
                    m.clone(),
                    Some(vec![("scope", Some("local"))]),
                    None,
                    Some(vec![("region", None)]),
                ],
                vec![
                    Some("scope"),
                    Some("region"),
                    Some("region"),
                    Some("scope"),
                    Some("region"),
                ],
            )
            .await,
            |c, i| c.value(i).to_string(),
        );
        assert_eq!(got, vec![some("fleet"), some("us-east"), None, None, None]);
    }

    #[tokio::test]
    async fn json_contains_on_a_native_map_column() {
        let got = column::<BooleanArray, _>(
            &run_map_udf(
                "json_contains(m, r)",
                vec![
                    Some(vec![("a", Some("x")), ("z", None)]),
                    Some(vec![("a", Some("x")), ("z", None)]),
                    Some(vec![("a", Some("x"))]),
                    None,
                ],
                vec![Some("a"), Some("z"), Some("missing"), Some("a")],
            )
            .await,
            |c, i| c.value(i),
        );
        // Present key -> true (a null value still counts); absent -> false; NULL
        // map cell -> NULL.
        assert_eq!(got, vec![Some(true), Some(true), Some(false), None]);
    }

    #[tokio::test]
    async fn json_length_on_a_native_map_column() {
        let got = column::<Int64Array, _>(
            &run_map_udf(
                "json_length(m)",
                vec![
                    Some(vec![("a", Some("1")), ("b", Some("2"))]),
                    Some(vec![]),
                    None,
                ],
                vec![None, None, None],
            )
            .await,
            |c, i| c.value(i),
        );
        assert_eq!(got, vec![Some(2), Some(0), None]);
    }

    #[tokio::test]
    async fn json_get_int_on_a_map_is_null_because_values_are_strings() {
        // A Map<Utf8,Utf8> value is a JSON string, so the typed getters (which do
        // not coerce strings) are NULL — cast `json_get(...)` for a number.
        let got = column::<Int64Array, _>(
            &run_map_udf(
                "json_get_int(m, r)",
                vec![Some(vec![("n", Some("5"))])],
                vec![Some("n")],
            )
            .await,
            |c, i| c.value(i),
        );
        assert_eq!(got, vec![None]);
    }

    #[tokio::test]
    async fn json_get_is_transparent_across_string_and_map_columns() {
        // The point of duck-typing: identical results whether the column is a
        // JSON string or a native map, so a query survives the migration.
        let from_string = eval_str(
            "json_get",
            vec![
                Some(r#"{"region":"us-east"}"#),
                Some(r#"{"region":"us-west"}"#),
            ],
            vec![Some("region"), Some("region")],
        )
        .await;
        let from_map = column::<StringArray, _>(
            &run_map_udf(
                "json_get(m, r)",
                vec![
                    Some(vec![("region", Some("us-east"))]),
                    Some(vec![("region", Some("us-west"))]),
                ],
                vec![Some("region"), Some("region")],
            )
            .await,
            |c, i| c.value(i).to_string(),
        );
        assert_eq!(from_string, from_map);
        assert_eq!(from_string, vec![some("us-east"), some("us-west")]);
    }
}
