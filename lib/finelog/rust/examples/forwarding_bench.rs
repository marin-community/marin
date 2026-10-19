//! Compare the former SQL source read with native forwarding over identical files.
//! Run on a disposable local directory; both hub variants use real loopback RPCs.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{AsArray, Int32Array, Int64Array, RecordBatch, StringArray};
use clap::Parser;
use connectrpc::client::ServiceTransport;
use finelog::indices::cache::DEFAULT_INDEX_CACHE_MB;
use finelog::proto::finelog::stats::{StatsServiceClient, WriteRowsRequest};
use finelog::query::{make_ctx, run_query_over, RegisteredProvider};
use finelog::server::forwarding::{chunk_by_bytes, forward_client_config, ship_batch};
use finelog::server::{build_app_with_config, ServerConfig};
use finelog::store::forwarding::{read_forwarding, ReadBudget};
use finelog::store::schema::{schema_to_arrow, stored_form};
use finelog::store::segment::write_segment_to_dir;
use finelog::store::store::log_registered_schema;
use finelog::store::{ServeMode, Store};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::json;

const ROWS_PER_READ: usize = 1_000_000;
const READ_BYTES: usize = 256 << 20;
const REQUEST_BYTES: usize = 150 << 20;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Clone, Copy)]
enum Reader {
    Sql,
    Native,
}

#[derive(Clone, Copy)]
enum Delivery {
    Encode,
    Hub,
}

#[derive(Parser)]
struct Args {
    /// A new, disposable directory on local disk.
    directory: PathBuf,
    #[arg(long, default_value_t = 1_000_000)]
    rows: usize,
    #[arg(long, default_value_t = 128)]
    segments: usize,
    #[arg(long, default_value_t = 160)]
    body_bytes: usize,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
}

fn generate(directory: &Path, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let path = directory.join("log");
    std::fs::create_dir_all(&path)?;
    let schema = schema_to_arrow(&stored_form(log_registered_schema()));
    let per_segment = args.rows.div_ceil(args.segments);
    for start in (0..args.rows).step_by(per_segment) {
        let count = per_segment.min(args.rows - start);
        let bodies: Vec<String> = (start..start + count)
            .map(|row| {
                let mut body = format!(
                    "host{}:{} NCCL INFO channel {} Input tuning row {row} ",
                    row % 64,
                    row % 8,
                    row % 32
                );
                body.push_str(&"x".repeat(args.body_bytes.saturating_sub(body.len())));
                body
            })
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (start + 1..start + count + 1).map(|seq| seq as i64),
                )),
                Arc::new(StringArray::from(
                    (start..start + count)
                        .map(|row| format!("/user/job/task{}:0", row % 64))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(vec!["stdout"; count])),
                Arc::new(StringArray::from(bodies)),
                Arc::new(Int64Array::from_iter_values(
                    (start..start + count).map(|row| 1_792_300_000_000 + row as i64 / 100),
                )),
                Arc::new(Int32Array::from(vec![2; count])),
                Arc::new(StringArray::from(vec![None::<&str>; count])),
            ],
        )?;
        write_segment_to_dir(&path, 0, start as i64 + 1, &batch)?;
    }
    Ok(())
}

async fn trial(
    source: Arc<Store>,
    directory: &Path,
    reader_kind: Reader,
    delivery: Delivery,
    repetition: usize,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = match reader_kind {
        Reader::Sql => "sql",
        Reader::Native => "native",
    };
    let mode = match delivery {
        Delivery::Encode => "read_encode",
        Delivery::Hub => "durable_hub",
    };
    let hub = Arc::new(Store::new(
        Some(directory.join(format!("hub-{reader}-{mode}-{repetition}"))),
        "".into(),
        DEFAULT_INDEX_CACHE_MB,
        ServeMode::Live,
    )?);
    hub.bootstrap_maintenance();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let app = build_app_with_config(Arc::clone(&hub), ServerConfig::default());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
    });
    let url = format!("http://{address}");
    let transport =
        ServiceTransport::new(Client::builder(TokioExecutor::new()).build(HttpConnector::new()));
    let client = StatsServiceClient::new(transport, forward_client_config(url.parse()?));
    let tip = source.namespace_persisted_seq("log")?;
    let started = Instant::now();
    let mut cursor = 0;
    let mut rows = 0;
    let mut encoded_bytes = 0;
    let mut read_time = Duration::ZERO;
    let mut encode_time = Duration::ZERO;
    let mut push_time = Duration::ZERO;
    let mut settle_time = Duration::ZERO;
    while cursor < tip {
        let stage = Instant::now();
        let (schema, batches, through) = if matches!(reader_kind, Reader::Native) {
            let read = read_forwarding(
                Arc::clone(&source),
                "log",
                cursor,
                tip,
                ReadBudget {
                    rows: ROWS_PER_READ,
                    bytes: READ_BYTES,
                    segments: 128,
                    timeout: Duration::from_secs(60),
                },
            )
            .await?;
            (read.schema, read.batches, read.scanned_through)
        } else {
            let snapshot = source.query_snapshot("log")?;
            let provider = source.namespace_provider("log", snapshot)?;
            let sql = format!("SELECT * FROM log WHERE seq > {cursor} AND seq <= {tip} AND (cluster IS NULL OR cluster = '') ORDER BY seq LIMIT {ROWS_PER_READ}");
            let result = run_query_over(
                &make_ctx(),
                vec![RegisteredProvider {
                    name: "log".into(),
                    provider,
                }],
                &sql,
            )
            .await?;
            let through = result
                .batches
                .last()
                .unwrap()
                .column_by_name("seq")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(result.batches.last().unwrap().num_rows() - 1);
            (result.schema, result.batches, through)
        };
        read_time += stage.elapsed();
        assert!(through > cursor);
        let stage = Instant::now();
        let (batch, sequences) = ship_batch(schema, batches, "benchmark")?.unwrap();
        rows += batch.num_rows();
        let chunks = chunk_by_bytes(&batch, &sequences, REQUEST_BYTES)?;
        encoded_bytes += chunks.iter().map(|(bytes, _)| bytes.len()).sum::<usize>();
        drop(batch);
        encode_time += stage.elapsed();
        let stage = Instant::now();
        if matches!(delivery, Delivery::Hub) {
            for (bytes, _) in chunks {
                client
                    .write_rows(
                        WriteRowsRequest::default()
                            .with_namespace("log")
                            .with_arrow_ipc(bytes),
                    )
                    .await?;
            }
        }
        push_time += stage.elapsed();
        if matches!(delivery, Delivery::Hub) {
            let stage = Instant::now();
            source.settle_forwarding(&url, "log", through).await?;
            settle_time += stage.elapsed();
        }
        cursor = through;
    }
    let elapsed = started.elapsed();
    if matches!(delivery, Delivery::Hub) {
        let stored_rows = hub
            .list_namespaces_with_stats()?
            .into_iter()
            .find(|(name, _, _, _)| name == "log")
            .unwrap()
            .2
            .row_count;
        assert_eq!(stored_rows, rows as i64);
    }
    println!(
        "{}",
        json!({ "reader": reader, "mode": mode, "repetition": repetition, "rows": rows, "encoded_bytes": encoded_bytes, "read_ms": read_time.as_secs_f64() * 1000.0, "encode_ms": encode_time.as_secs_f64() * 1000.0, "push_ms": push_time.as_secs_f64() * 1000.0, "settle_ms": settle_time.as_secs_f64() * 1000.0, "elapsed_ms": elapsed.as_secs_f64() * 1000.0, "rows_per_second": rows as f64 / elapsed.as_secs_f64(), "profile": if cfg!(debug_assertions) { "debug" } else { "release" }, "source_durability": "local_disk", "hub_durability": "local_disk", "transport": "loopback_zstd", "source_segments": source.query_snapshot("log")?.paths.len(), "body_bytes": args.body_bytes, "cache": "warm_local_files", "allocator": "production_default" })
    );
    server.abort();
    hub.shutdown(Duration::from_secs(5)).await;
    Ok(())
}

#[tokio::main(worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    // Keep the reference's one SQL window identical to the production fanout
    // limit; the native reader may split it further to respect decoded bytes.
    assert!((1..=ROWS_PER_READ).contains(&args.rows));
    assert!((1..=128).contains(&args.segments));
    assert!(args.repetitions > 0);
    std::fs::create_dir(&args.directory)?;
    let directory = std::fs::canonicalize(&args.directory)?;
    let source_dir = directory.join("source");
    generate(&source_dir, &args)?;
    let source = Arc::new(Store::new(
        Some(source_dir),
        "".into(),
        DEFAULT_INDEX_CACHE_MB,
        ServeMode::Shadow,
    )?);
    for repetition in 0..args.repetitions {
        for delivery in [Delivery::Encode, Delivery::Hub] {
            let readers = if repetition % 2 == 0 {
                [Reader::Sql, Reader::Native]
            } else {
                [Reader::Native, Reader::Sql]
            };
            for reader in readers {
                trial(
                    Arc::clone(&source),
                    &directory,
                    reader,
                    delivery,
                    repetition,
                    &args,
                )
                .await?;
            }
        }
    }
    source.shutdown(Duration::from_secs(5)).await;
    Ok(())
}
