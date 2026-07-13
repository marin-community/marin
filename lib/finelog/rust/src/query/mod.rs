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

pub mod optimizer;
pub mod provider;
pub mod sidecar;
pub mod trigram_prune;
pub mod udf;

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::{Field, Schema as ArrowSchema, SchemaRef};
use datafusion::common::config::Dialect;
use datafusion::common::TableReference;
use datafusion::error::Result as DFResult;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion::prelude::{SessionConfig, SessionContext};

use crate::query::provider::NamespaceProvider;

/// A namespace ready to register: its exact name (used verbatim in `FROM`) and
/// its provider over the snapshotted sealed segments.
pub struct RegisteredProvider {
    pub name: String,
    pub provider: NamespaceProvider,
}

/// Floor for the query memory pool so a tiny/misreported cgroup can't strangle
/// every query (256 MiB).
const MIN_QUERY_POOL_BYTES: usize = 256 * 1024 * 1024;

/// Fraction of the container/host memory the query engine may use for
/// pool-tracked operators (sorts, joins, aggregations). The remainder is
/// headroom for non-pool allocations (parquet decode scratch, IPC encode,
/// tokio/allocator overhead).
const QUERY_POOL_FRACTION: f64 = 0.7;

/// Best-effort detect the process memory ceiling: the cgroup v2 limit
/// (`memory.max`, i.e. the container's `--memory`) if set, else `/proc/meminfo`
/// `MemTotal`. `None` when neither is readable/finite.
fn detect_memory_limit_bytes() -> Option<usize> {
    if let Ok(raw) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        let raw = raw.trim();
        if raw != "max" {
            if let Ok(v) = raw.parse::<usize>() {
                return Some(v);
            }
        }
    }
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            return rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<usize>()
                .ok()
                .map(|kb| kb.saturating_mul(1024));
        }
    }
    None
}

/// The byte ceiling for the shared query memory pool.
///
/// `FINELOG_QUERY_MEMORY_LIMIT_MB` overrides everything (explicit ops control);
/// otherwise `QUERY_POOL_FRACTION` of the detected container/host memory, floored
/// at `MIN_QUERY_POOL_BYTES`. When memory can't be detected the pool is left
/// effectively unbounded (no regression vs. an un-pooled context).
fn query_pool_bytes() -> usize {
    if let Ok(raw) = std::env::var("FINELOG_QUERY_MEMORY_LIMIT_MB") {
        if let Ok(mb) = raw.trim().parse::<usize>() {
            return mb.saturating_mul(1024 * 1024).max(MIN_QUERY_POOL_BYTES);
        }
    }
    match detect_memory_limit_bytes() {
        Some(total) => (((total as f64) * QUERY_POOL_FRACTION) as usize).max(MIN_QUERY_POOL_BYTES),
        None => usize::MAX / 2,
    }
}

/// A process-wide `RuntimeEnv` whose `GreedyMemoryPool` bounds total query
/// memory. Shared across every `make_ctx` so concurrent queries compete for one
/// budget (bounding the SERVER, not each query independently): a runaway query
/// — e.g. an `ORDER BY <blob> DESC LIMIT n` whose TopK can't push a pruning
/// filter through an intervening join, so it materializes the whole blob column
/// — hits the ceiling and fails with `ResourcesExhausted` instead of
/// OOM-killing the process (which surfaces to clients as a dropped connection /
/// 502).
fn shared_runtime_env() -> Arc<RuntimeEnv> {
    static RT: OnceLock<Arc<RuntimeEnv>> = OnceLock::new();
    RT.get_or_init(|| {
        let bytes = query_pool_bytes();
        tracing::info!(
            limit_mb = bytes / (1024 * 1024),
            "query engine: bounded memory pool (GreedyMemoryPool)"
        );
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(bytes)))
            .build_arc()
            .expect("build query RuntimeEnv")
    })
    .clone()
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
/// - `parquet.pushdown_filters = true` + `parquet.reorder_filters = true`: apply
///   scan predicates *inside* the parquet decoder via a row selection, so other
///   projected columns are read only for surviving rows. Without this, DataFusion
///   reads every projected column for all rows and filters afterward — fatal for
///   namespaces with a large blob column (e.g. `iris.profile.profile_data`):
///   `SELECT length(profile_data) ... WHERE source = ?` would otherwise decode
///   the entire ~GB blob column before the `source` filter drops the rows, where
///   DuckDB's late materialization reads zero blobs for a non-matching key. This
///   is the dominant cost in the dashboard's profile-history query.
///
/// The compat UDFs (`prefix`/`regexp_matches`/`contains`) are registered so the
/// corpus and FetchLogs resolve them, and the [`PrefixRangeRewrite`] analyzer
/// rule rewrites starts-with predicates to expose a prunable key range to the
/// planner (so both FetchLogs and the generic Query API prune row groups on the
/// `[key, seq]`-sorted segments — see [`crate::query::optimizer`]).
///
/// The context runs on a shared, memory-bounded `RuntimeEnv` (see
/// [`shared_runtime_env`]) so a pathological query fails cleanly rather than
/// OOM-killing the server.
pub fn make_ctx() -> SessionContext {
    let mut cfg = SessionConfig::new();
    cfg.options_mut().sql_parser.map_string_types_to_utf8view = false;
    cfg.options_mut().sql_parser.dialect = Dialect::DuckDB;
    cfg.options_mut().execution.parquet.pushdown_filters = true;
    cfg.options_mut().execution.parquet.reorder_filters = true;
    let ctx = SessionContext::new_with_config_rt(cfg, shared_runtime_env());
    ctx.add_analyzer_rule(Arc::new(crate::query::optimizer::PrefixRangeRewrite));
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
/// DuckDB returns ALL result columns as nullable, while
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

/// Threshold (ms) at or above which a completed query is logged at WARN with its
/// SQL — the diagnostic the RPC-level slow-warn can't provide (the interceptor
/// only sees the encoded request, never the SQL). Defaults to the same bar as the
/// RPC warn ([`crate::server::interceptors::DEFAULT_SLOW_RPC_THRESHOLD_MS`]);
/// `FINELOG_SLOW_QUERY_LOG_MS` overrides it (set it low to capture more while
/// debugging a specific slow shape).
fn slow_query_log_ms() -> u128 {
    static MS: OnceLock<u128> = OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("FINELOG_SLOW_QUERY_LOG_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u128>().ok())
            .unwrap_or(crate::server::interceptors::DEFAULT_SLOW_RPC_THRESHOLD_MS as u128)
    })
}

/// Cap arbitrary (possibly user-supplied) SQL for a single log line. Truncates on
/// a char boundary — never mid-codepoint — so non-ASCII SQL can't panic the
/// logger, in a single pass over at most `MAX_CHARS + 1` chars.
fn truncate_sql_for_log(sql: &str) -> String {
    const MAX_CHARS: usize = 4000;
    let mut chars = sql.chars();
    let head: String = chars.by_ref().take(MAX_CHARS).collect();
    if chars.next().is_some() {
        format!("{head} …[truncated]")
    } else {
        head
    }
}

/// Emit one WARN carrying the SQL when a query's execution time reached the slow
/// threshold. `rows` is the result row count on success, `None` when the query
/// errored — a slow *failed* query (e.g. `ResourcesExhausted` after a long scan)
/// is exactly the case worth seeing.
fn log_slow_query(elapsed: Duration, kind: &str, sql: &str, rows: Option<usize>) {
    let elapsed_ms = elapsed.as_millis();
    if elapsed_ms < slow_query_log_ms() {
        return;
    }
    let preview = truncate_sql_for_log(sql);
    let rows_str = rows.map_or_else(|| "ERR".to_string(), |n| n.to_string());
    tracing::warn!(
        kind,
        elapsed_ms = elapsed_ms as u64,
        rows = %rows_str,
        sql = %preview,
        "slow {kind}: {elapsed_ms}ms rows={rows_str} sql={preview}",
    );
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
    let started = Instant::now();
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
    let elapsed = started.elapsed();
    for name in &names {
        // Best-effort cleanup; a deregister failure must not mask the query
        // result/error.
        let _ = ctx.deregister_table(TableReference::bare(name.as_str()));
    }
    let rows = result
        .as_ref()
        .ok()
        .map(|r| r.batches.iter().map(|b| b.num_rows()).sum());
    log_slow_query(elapsed, "Query", sql, rows);
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

    let started = Instant::now();
    let collected = async {
        let df = ctx.sql(&sql).await?;
        df.collect().await
    }
    .await;
    let elapsed = started.elapsed();
    let _ = ctx.deregister_table(TableReference::bare(LOG_TABLE));
    let rows = collected
        .as_ref()
        .ok()
        .map(|b| b.iter().map(|x| x.num_rows()).sum());
    log_slow_query(elapsed, "FetchLogs", &sql, rows);
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
    use crate::test_support::unique_dir;
    use datafusion::arrow::array::Int64Array;

    #[test]
    fn truncate_sql_caps_on_char_boundary() {
        // Short SQL is returned unchanged.
        let short = "SELECT 1";
        assert_eq!(truncate_sql_for_log(short), short);
        // A long multibyte string must truncate WITHOUT panicking mid-codepoint
        // and gain a marker. 5000 '✓' (3 bytes each) exceeds the 4000-char cap;
        // byte-indexed truncation would panic here.
        let long: String = "✓".repeat(5000);
        let out = truncate_sql_for_log(&long);
        assert!(out.ends_with("…[truncated]"));
        assert_eq!(out.chars().filter(|&c| c == '✓').count(), 4000);
    }

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

    #[tokio::test]
    async fn prefix_fetch_returns_exactly_the_prefix_rows() {
        // End-to-end guard for the PREFIX key-range rewrite: keys chosen to
        // straddle the half-open range [P, succ(P)) for P = "/a/". A wrong
        // successor would drop "/a/*" rows or leak "/ab/1" / "/b/1"; "/a" sits
        // just below the lower bound. The result set (not the SQL) is asserted.
        use crate::proto::finelog::logging::MatchScope;
        use crate::query::provider::NamespaceProvider;
        use crate::store::log_read::build_log_predicates;
        use crate::store::segment::{discover_segments, write_segment_to_dir};
        use datafusion::arrow::array::{Int32Array, Int64Array, StringArray};
        use datafusion::arrow::datatypes::DataType;

        let dir = std::env::temp_dir().join(format!(
            "finelog_prefix_fetch_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let keys = ["/a", "/a/1", "/a/2", "/ab/1", "/b/1"];
        let schema: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("source", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
            Field::new("epoch_ms", DataType::Int64, false),
            Field::new("level", DataType::Int32, false),
        ]));
        let n = keys.len() as i64;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(1..=n)),
                Arc::new(StringArray::from(keys.to_vec())),
                Arc::new(StringArray::from(vec!["stdout"; keys.len()])),
                Arc::new(StringArray::from(vec!["line"; keys.len()])),
                Arc::new(Int64Array::from_iter_values(1..=n)),
                Arc::new(Int32Array::from(vec![2; keys.len()])),
            ],
        )
        .unwrap();
        write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let paths: Vec<String> = discover_segments(&dir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let provider = NamespaceProvider::build(schema, &paths).unwrap();
        let preds = build_log_predicates("/a/", 0, MatchScope::MATCH_SCOPE_PREFIX).unwrap();
        let ctx = make_ctx();
        let rows = fetch_log_rows(
            &ctx,
            provider,
            &preds.where_parts,
            preds.include_key,
            true,
            100,
        )
        .await
        .unwrap();
        let mut got: Vec<String> = rows.into_iter().filter_map(|r| r.key).collect();
        got.sort();
        assert_eq!(got, vec!["/a/1".to_string(), "/a/2".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The real store-form `log` arrow schema (single source of truth), so these
    /// tests track any column change instead of hand-copying the layout.
    fn log_arrow() -> SchemaRef {
        crate::store::schema::schema_to_arrow(&crate::store::schema::with_implicit_seq(
            crate::store::store::log_registered_schema(),
        ))
    }

    /// The same schema minus the `cluster` column — the layout of a segment
    /// written before the column was added, expressed as a derivation rather than
    /// a duplicated literal.
    fn log_arrow_pre_cluster() -> SchemaRef {
        let fields: Vec<Field> = log_arrow()
            .fields()
            .iter()
            .filter(|f| f.name() != "cluster")
            .map(|f| f.as_ref().clone())
            .collect();
        Arc::new(ArrowSchema::new(fields))
    }

    #[tokio::test]
    async fn cluster_filter_namespaces_mixed_old_and_new_segments() {
        // The read filter namespaces a global finelog by server-stamped origin
        // across the realistic production mix: new segments carry the `cluster`
        // column; a pre-evolution segment lacks it and null-fills on read. A
        // `cluster = <peer>` filter returns exactly that peer's rows (excluding
        // other peers and the null-filled legacy rows); an empty filter returns
        // every origin — the local single-cluster read behavior.
        use crate::proto::finelog::logging::MatchScope;
        use crate::query::provider::NamespaceProvider;
        use crate::store::log_read::{add_cluster_filter, build_log_predicates};
        use crate::store::segment::{discover_segments, write_segment_to_dir};
        use datafusion::arrow::array::{Int32Array, Int64Array, StringArray};

        let dir = unique_dir("cluster_mixed");
        let full = log_arrow();

        // A post-evolution segment stamped for two federated peers.
        let new_batch = RecordBatch::try_new(
            Arc::clone(&full),
            vec![
                Arc::new(Int64Array::from_iter_values(1..=2)),
                Arc::new(StringArray::from(vec!["/job/alpha", "/job/bravo"])),
                Arc::new(StringArray::from(vec!["stdout"; 2])),
                Arc::new(StringArray::from(vec!["line"; 2])),
                Arc::new(Int64Array::from_iter_values(1..=2)),
                Arc::new(Int32Array::from(vec![2; 2])),
                Arc::new(StringArray::from(vec!["alpha", "bravo"])),
            ],
        )
        .unwrap();
        write_segment_to_dir(&dir, 1, 1, &new_batch).unwrap();

        // A pre-evolution segment with no `cluster` column at all.
        let legacy_batch = RecordBatch::try_new(
            log_arrow_pre_cluster(),
            vec![
                Arc::new(Int64Array::from_iter_values(3..=3)),
                Arc::new(StringArray::from(vec!["/job/legacy"])),
                Arc::new(StringArray::from(vec!["stdout"; 1])),
                Arc::new(StringArray::from(vec!["line"; 1])),
                Arc::new(Int64Array::from_iter_values(3..=3)),
                Arc::new(Int32Array::from(vec![2; 1])),
            ],
        )
        .unwrap();
        write_segment_to_dir(&dir, 1, 3, &legacy_batch).unwrap();

        let paths: Vec<String> = discover_segments(&dir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let ctx = make_ctx();

        // Read every `/job/` key visible under the given cluster filter, sorted.
        let read_keys = |cluster: &str| {
            let mut preds =
                build_log_predicates("/job/", 0, MatchScope::MATCH_SCOPE_PREFIX).unwrap();
            add_cluster_filter(&mut preds.where_parts, cluster);
            NamespaceProvider::build(Arc::clone(&full), &paths)
                .map(|provider| (provider, preds))
                .unwrap()
        };
        let sorted_keys = |mut rows: Vec<crate::store::log_read::LogRow>| {
            rows.sort_by(|a, b| a.key.cmp(&b.key));
            rows.into_iter()
                .filter_map(|r| r.key)
                .collect::<Vec<String>>()
        };

        // Filtered to one peer → only that peer's row.
        let (provider, preds) = read_keys("alpha");
        let rows = fetch_log_rows(
            &ctx,
            provider,
            &preds.where_parts,
            preds.include_key,
            false,
            100,
        )
        .await
        .unwrap();
        assert_eq!(sorted_keys(rows), vec!["/job/alpha".to_string()]);

        // Empty filter → every origin, including the null-filled legacy row.
        let (provider, preds) = read_keys("");
        let rows = fetch_log_rows(
            &ctx,
            provider,
            &preds.where_parts,
            preds.include_key,
            false,
            100,
        )
        .await
        .unwrap();
        assert_eq!(
            sorted_keys(rows),
            vec![
                "/job/alpha".to_string(),
                "/job/bravo".to_string(),
                "/job/legacy".to_string(),
            ]
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
