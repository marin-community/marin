//! DataFusion read engine.
//!
//! `make_ctx()` builds a `SessionContext` configured to match DuckDB's result
//! shape (Utf8 strings, DuckDB parsing dialect) with the compat UDFs registered.
//! `run_query_over()` registers every live namespace as a `TableProvider`, runs
//! the user SQL verbatim, collects the result, and deregisters — the body of the
//! `StatsService::Query` handler.
//!
//! Query visibility = sealed parquet segments ONLY (see `provider.rs`). The
//! durability contract makes written rows visible because they are sealed before
//! WriteRows/PushLogs ack.

pub mod provider;
pub mod udf;

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::config::Dialect;
use datafusion::common::TableReference;
use datafusion::error::Result as DFResult;
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::query::provider::NamespaceProvider;

/// A namespace ready to register: its exact name (used verbatim in `FROM`) and
/// its provider over the snapshotted sealed segments.
pub struct RegisteredProvider {
    pub name: String,
    pub provider: NamespaceProvider,
}

/// Build a read-only `SessionContext` matching DuckDB's externally-observable
/// result shape.
///
/// - `map_string_types_to_utf8view = false`: SQL `VARCHAR`/`STRING` plan as
///   `Utf8` (not `Utf8View`), matching the pyarrow/DuckDB result schema.
/// - `dialect = "DuckDB"`: DuckDB sqlparser sugar (parsing only).
/// - `enable_ident_normalization` left at the DF53 default (true): the corpus
///   quotes dotted identifiers (`"iris.worker"`), which are preserved verbatim;
///   lowercase unquoted column names are unaffected.
///
/// The compat UDFs (`prefix`/`regexp_matches`/`contains`) are registered so the
/// corpus and FetchLogs resolve them.
pub fn make_ctx() -> SessionContext {
    let mut cfg = SessionConfig::new();
    cfg.options_mut().sql_parser.map_string_types_to_utf8view = false;
    cfg.options_mut().sql_parser.dialect = Dialect::DuckDB;
    let ctx = SessionContext::new_with_config(cfg);
    udf::register_compat_udfs(&ctx);
    ctx
}

/// A collected query result: its arrow schema (always present, even for an
/// empty result so the IPC stream can carry it) and the result batches.
pub struct QueryResult {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

/// Relax every field in `schema` to `nullable = true` and re-stamp `batches`
/// with the relaxed schema.
///
/// DuckDB (the Python backend) returns ALL result columns as nullable, while
/// DataFusion propagates source non-nullability (e.g. the store-form `seq`
/// column is non-nullable), so a result column would carry `nullable = false`
/// and the decoded-Arrow result schema would diverge on the wire. Relaxing here
/// makes the QueryResponse IPC schema match DuckDB exactly. Widening
/// non-nullable -> nullable is always valid (a non-null array satisfies a
/// nullable field), so no array data is touched.
fn relax_result_nullability(
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
) -> DFResult<(SchemaRef, Vec<RecordBatch>)> {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    let relaxed: SchemaRef = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ));
    let mut out = Vec::with_capacity(batches.len());
    for b in batches {
        out.push(RecordBatch::try_new(
            Arc::clone(&relaxed),
            b.columns().to_vec(),
        )?);
    }
    Ok((relaxed, out))
}

/// Register every namespace in `providers`, run `sql` verbatim, collect, and
/// deregister. Returns the result schema + batches.
///
/// Registration is per-call (a fresh `ctx`): names are used exactly as the
/// catalog records them, so `FROM "iris.worker"` resolves. An unknown namespace
/// in the FROM clause surfaces as a DataFusion plan error (the caller maps it to
/// `invalid_argument`, the DuckDB CatalogException slot). The schema is captured
/// from the planned `DataFrame` BEFORE deregistration, so an empty result still
/// carries the correct typed schema without re-planning.
pub async fn run_query_over(
    ctx: &SessionContext,
    providers: Vec<RegisteredProvider>,
    sql: &str,
) -> DFResult<QueryResult> {
    let names: Vec<String> = providers.iter().map(|p| p.name.clone()).collect();
    for rp in providers {
        // `TableReference::bare` keeps a dotted name (`iris.worker`) as ONE
        // table identifier rather than a `schema.table` split, so the user's
        // quoted `FROM "iris.worker"` resolves to exactly this registration.
        ctx.register_table(TableReference::bare(rp.name), Arc::new(rp.provider))?;
    }
    let result = async {
        let df = ctx.sql(sql).await?;
        let schema = Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await?;
        // Match DuckDB's all-nullable result schema (the captured plan schema
        // keeps source non-nullability that DuckDB would have dropped).
        let (schema, batches) = relax_result_nullability(&schema, batches)?;
        Ok(QueryResult { schema, batches })
    }
    .await;
    for name in &names {
        // Best-effort cleanup; a deregister failure must not mask the query
        // result/error.
        let _ = ctx.deregister_table(TableReference::bare(name.as_str()));
    }
    result
}

/// Read `log`-namespace rows matching `where_parts`, ordered by seq.
///
/// Registers `provider` (over the sealed `log` segments) under a fixed internal
/// table name, runs `SELECT seq, [key,] source, data, epoch_ms, level FROM ...
/// WHERE <where_parts> ORDER BY seq [DESC] [LIMIT]`, and decodes the result
/// batches into `LogRow`s. `tail && max_lines > 0` orders `seq DESC` (the caller
/// reverses); otherwise `seq ASC`. `max_lines <= 0` means no LIMIT.
pub async fn fetch_log_rows(
    ctx: &SessionContext,
    provider: NamespaceProvider,
    where_parts: &[String],
    include_key: bool,
    tail: bool,
    max_lines: i32,
) -> DFResult<Vec<crate::store::log_read::LogRow>> {
    use datafusion::arrow::array::{Int32Array, Int64Array, StringArray};

    const LOG_TABLE: &str = "__finelog_log";
    ctx.register_table(TableReference::bare(LOG_TABLE), Arc::new(provider))?;

    let select_cols = if include_key {
        "seq, key, source, data, epoch_ms, level"
    } else {
        "seq, source, data, epoch_ms, level"
    };
    let order = if tail && max_lines > 0 {
        "ORDER BY seq DESC"
    } else {
        "ORDER BY seq"
    };
    let limit = if max_lines > 0 {
        format!("LIMIT {max_lines}")
    } else {
        String::new()
    };
    let where_clause = if where_parts.is_empty() {
        "TRUE".to_string()
    } else {
        where_parts.join(" AND ")
    };
    let sql =
        format!("SELECT {select_cols} FROM \"{LOG_TABLE}\" WHERE {where_clause} {order} {limit}");

    let collected = async {
        let df = ctx.sql(&sql).await?;
        df.collect().await
    }
    .await;
    let _ = ctx.deregister_table(TableReference::bare(LOG_TABLE));
    let batches = collected?;

    let mut rows = Vec::new();
    for b in &batches {
        let seq = b
            .column_by_name("seq")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let key = if include_key {
            b.column_by_name("key")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        } else {
            None
        };
        let source = b
            .column_by_name("source")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let data = b
            .column_by_name("data")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let epoch_ms = b
            .column_by_name("epoch_ms")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
        let level = b
            .column_by_name("level")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let (Some(seq), Some(source), Some(data), Some(epoch_ms), Some(level)) =
            (seq, source, data, epoch_ms, level)
        else {
            return Err(datafusion::error::DataFusionError::Internal(
                "log read result missing an expected column".to_string(),
            ));
        };
        for i in 0..b.num_rows() {
            rows.push(crate::store::log_read::LogRow {
                seq: seq.value(i),
                key: key.map(|k| k.value(i).to_string()),
                source: source.value(i).to_string(),
                data: data.value(i).to_string(),
                epoch_ms: epoch_ms.value(i),
                level: level.value(i),
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::ipc::{decode_one_record_batch, encode_ipc};
    use datafusion::arrow::array::Int64Array;

    #[tokio::test]
    async fn select_one_roundtrips() {
        let ctx = make_ctx();
        let batches = ctx
            .sql("SELECT 1 AS n")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let schema = batches[0].schema();
        // Encode -> decode through the wire IPC codec.
        let buf = encode_ipc(&schema, &batches).unwrap();
        let decoded = decode_one_record_batch(&buf).unwrap();
        assert_eq!(decoded.num_rows(), 1);
        assert_eq!(decoded.schema().field(0).name(), "n");
        let col = decoded
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 1);
    }

    #[tokio::test]
    async fn string_literal_is_utf8_not_utf8view() {
        // map_string_types_to_utf8view=false must make string results Utf8.
        let ctx = make_ctx();
        let batches = ctx
            .sql("SELECT 'hello' AS s")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            batches[0].schema().field(0).data_type(),
            &datafusion::arrow::datatypes::DataType::Utf8
        );
    }

    #[tokio::test]
    async fn compat_udfs_resolve_in_sql() {
        let ctx = make_ctx();
        let batches = ctx
            .sql("SELECT prefix('/a/b', '/a') AS p, regexp_matches('/x/y', 'x/.*') AS r, contains('100% done', '100%') AS c")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        use datafusion::arrow::array::BooleanArray;
        for (i, expected) in [true, true, true].iter().enumerate() {
            let col = batches[0]
                .column(i)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            assert_eq!(col.value(0), *expected, "col {i}");
        }
    }
}
