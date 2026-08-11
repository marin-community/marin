use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, AsArray};
use bytes::Bytes;
use connectrpc::client::{full_body, ClientBody};
use connectrpc::compression::{CompressionProvider, ZstdProvider};
use http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;
use serde_json::{json, Value};
use tokio::sync::{oneshot, Semaphore};

use crate::proto::finelog::stats::ColumnType;
use crate::query::{make_ctx, run_query_over};
use crate::server::auth::AuthPolicy;
use crate::server::ingest_health::{IngestHealth, HEALTH_OK};
use crate::server::test_support::{disk_store, serve, PUB_A};
use crate::server::{build_app_with_config, ServerConfig};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{Column, Schema};
use crate::store::Store;
use crate::test_support::unique_dir;

type TestHttpClient = HyperClient<HttpConnector, ClientBody>;

struct TestResponse {
    status: StatusCode,
    headers: HeaderMap,
    payload: Value,
}

#[tokio::test]
async fn cancelled_waiter_does_not_release_owned_work_or_skip_completion() {
    let admission = Arc::new(Semaphore::new(1));
    let permit = Arc::clone(&admission).acquire_owned().await.unwrap();
    let (started_tx, started_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let (completed_tx, completed_rx) = oneshot::channel();

    let waiter = tokio::spawn(super::telemetry::run_to_completion(async move {
        started_tx.send(()).unwrap();
        release_rx.await.unwrap();
        // This send stands in for the successful cache publication that follows
        // Store::write_rows in the production future.
        completed_tx.send(()).unwrap();
        drop(permit);
    }));
    started_rx.await.unwrap();
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    assert!(admission.try_acquire().is_err());

    release_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), completed_rx)
        .await
        .unwrap()
        .unwrap();
    assert!(admission.try_acquire().is_ok());
}

fn http_client() -> TestHttpClient {
    HyperClient::builder(TokioExecutor::new()).build(HttpConnector::new())
}

/// GET a route, asserting 200 and returning the body.
async fn get_text(client: &TestHttpClient, addr: SocketAddr, path: &str) -> String {
    let response = client
        .request(
            Request::get(format!("http://{addr}{path}"))
                .body(full_body(Bytes::new()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "GET {path}");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn serve_with_config(store: Arc<Store>, config: ServerConfig) -> SocketAddr {
    let app = build_app_with_config(store, config);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    addr
}

async fn post_encoded(
    client: &TestHttpClient,
    addr: SocketAddr,
    body: Vec<u8>,
    batch_id: Option<&str>,
    content_type: Option<&str>,
    bearer: Option<&str>,
    content_encoding: Option<&str>,
) -> TestResponse {
    let mut request = Request::post(format!("http://{addr}/v1/telemetry"));
    if let Some(batch_id) = batch_id {
        request = request.header("idempotency-key", batch_id);
    }
    if let Some(content_type) = content_type {
        request = request.header("content-type", content_type);
    }
    if let Some(bearer) = bearer {
        request = request.header("authorization", format!("Bearer {bearer}"));
    }
    if let Some(content_encoding) = content_encoding {
        request = request.header("content-encoding", content_encoding);
    }
    let response = client
        .request(request.body(full_body(Bytes::from(body))).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(bytes.len() <= 1 << 20);
    let payload = serde_json::from_slice(&bytes).unwrap();
    TestResponse {
        status,
        headers,
        payload,
    }
}

async fn post(
    client: &TestHttpClient,
    addr: SocketAddr,
    body: Vec<u8>,
    batch_id: Option<&str>,
    content_type: Option<&str>,
    bearer: Option<&str>,
) -> TestResponse {
    post_encoded(client, addr, body, batch_id, content_type, bearer, None).await
}

fn zstd_body(body: &[u8]) -> Vec<u8> {
    ZstdProvider::with_level(1).compress(body).unwrap().to_vec()
}

fn batch(batch_id: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "version": 1,
        "batch_id": batch_id,
        "resource": {
            "service": "trainer",
            "attributes": {"role": "worker", "root_run_uid": "run-1"}
        },
        "records": [
            {
                "timestamp_ms": 1_700_000_000_000_i64,
                "kind": "counter",
                "name": "tokens",
                "value": 32.0,
                "unit": "token",
                "attributes": {"stage": "train"}
            },
            {
                "timestamp_ms": 1_700_000_000_001_i64,
                "kind": "event",
                "name": "checkpoint",
                "body": {"step": 10},
                "attributes": {}
            }
        ]
    }))
    .unwrap()
}

async fn query(store: &Store, sql: &str) -> Vec<arrow::array::RecordBatch> {
    let _guard = store.query_visibility().read().await;
    let providers = store.query_providers().unwrap();
    run_query_over(&make_ctx(), providers, sql)
        .await
        .unwrap()
        .batches
}

#[tokio::test]
async fn router_registers_index_policy_before_first_telemetry_request() {
    let store = disk_store("telemetry-startup-registration");
    let health = Arc::new(IngestHealth::new());
    let _router = super::telemetry::router(
        Arc::clone(&store),
        Arc::new(AuthPolicy::allow_localhost()),
        1,
        1,
        Arc::clone(&health),
    );
    assert!(
        health.health_body().contains("registration pending"),
        "the telemetry namespace is reported unavailable until it registers",
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        while health.health_body() != HEALTH_OK {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("startup registration did not complete");
    let schema = store.get_table_schema("telemetry_v1").unwrap();

    for name in ["service", "kind", "name"] {
        let column = schema
            .columns
            .iter()
            .find(|column| column.name == name)
            .unwrap();
        assert!(column.index.value_counts);
    }
    let name = schema
        .columns
        .iter()
        .find(|column| column.name == "name")
        .unwrap();
    assert!(name.index.trigram);
    assert_eq!(
        name.index.exact_values,
        [
            "global_step",
            "gpu_memory_temperature_celsius",
            "gpu_memory_total_bytes",
            "gpu_memory_used_bytes",
            "gpu_nvlink_receive_bytes_per_second",
            "gpu_nvlink_transmit_bytes_per_second",
            "gpu_pcie_receive_bytes_per_second",
            "gpu_pcie_replay_errors",
            "gpu_pcie_transmit_bytes_per_second",
            "gpu_power_watts",
            "gpu_row_remap_failures",
            "gpu_sm_active_ratio",
            "gpu_temperature_celsius",
            "gpu_tensor_active_ratio",
            "gpu_utilization_percent",
            "gpu_xid_error_code",
            "hardware_inventory",
            "node_cpu_utilization_percent",
            "node_disk_total_bytes",
            "node_disk_used_bytes",
            "node_memory_total_bytes",
            "node_memory_used_bytes",
            "node_network_receive_bytes",
            "node_network_transmit_bytes",
            "phase",
            "progress_time_seconds",
            "step",
        ]
    );
    let projections: BTreeMap<_, _> = schema
        .projections
        .iter()
        .map(|projection| (projection.name.as_str(), projection))
        .collect();
    assert_eq!(
        projections.keys().copied().collect::<Vec<_>>(),
        [
            "accelerator-faults",
            "accelerator-interconnect",
            "accelerator-inventory",
            "accelerator-memory",
            "accelerator-power",
            "accelerator-sm-activity",
            "accelerator-temperature",
            "accelerator-tensor-activity",
            "accelerator-utilization",
            "node-host-network",
            "node-host-utilization",
            "training-run-attribution",
            "training-status",
        ]
    );
    let power = projections["accelerator-power"];
    assert_eq!(power.predicate_column, "name");
    assert_eq!(power.predicate_values, ["gpu_power_watts"]);
    assert!(power.columns.iter().any(|column| column == "value"));
    assert!(power
        .columns
        .iter()
        .any(|column| column == "attributes_json"));
    assert_eq!(schema.grouped_extrema.len(), 1);
    let grouped = &schema.grouped_extrema[0];
    assert_eq!(grouped.filter_column, "service");
    assert_eq!(grouped.json_column, "resource_attributes_json");
    assert_eq!(grouped.json_key, "job_id");
    assert_eq!(grouped.extrema_column, "timestamp_ms");
}

#[tokio::test]
async fn a_registration_the_catalog_rejects_shows_up_in_health_and_server_info() {
    let store = disk_store("telemetry-wedged-registration");
    // A `name` column of the wrong type: no additive merge reconciles it.
    store
        .register_table(
            "telemetry_v1",
            Schema::new(
                vec![
                    Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                    Column::new("name", ColumnType::COLUMN_TYPE_INT64, false),
                ],
                "timestamp_ms",
            ),
            StoragePolicy::default(),
        )
        .unwrap();

    let addr = serve_with_config(Arc::clone(&store), ServerConfig::default()).await;
    let client = http_client();
    // `get_text` asserts 200: /health stays 200 while degraded so the Kubernetes
    // probes do not crashloop or de-endpoint the pod.
    let health = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let body = get_text(&client, addr, "/health").await;
            if body.contains("registration failed") {
                break body;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the wedged registration never reached /health");
    assert!(health.contains("telemetry_v1"), "{health}");

    let info: Value = serde_json::from_str(&get_text(&client, addr, "/api/server").await).unwrap();
    let namespace = &info["ingest"][0];
    assert_eq!(namespace["namespace"], "telemetry_v1");
    assert_eq!(namespace["state"], "failed");
    assert!(namespace["sinceUnix"].as_i64().unwrap() > 0);
    assert!(namespace["error"].as_str().unwrap().contains("name"));

    let posted = post(
        &client,
        addr,
        batch("11111111-1111-4111-8111-111111111111"),
        Some("11111111-1111-4111-8111-111111111111"),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(posted.status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn accepted_batch_is_queryable_through_normal_store_rows() {
    let remote_dir = unique_dir("telemetry-query-remote");
    let store = Arc::new(
        Store::new(
            Some(unique_dir("telemetry-query")),
            remote_dir.to_string_lossy().into_owned(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
            crate::store::ServeMode::Live,
        )
        .unwrap(),
    );
    store.bootstrap_maintenance();
    let (addr, _) = serve(Arc::clone(&store), AuthPolicy::allow_localhost()).await;
    let client = http_client();
    let batch_id = "f47ac10b-58cc-4372-a567-0e02b2c3d479";

    let response = post(
        &client,
        addr,
        batch(batch_id),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;

    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(response.headers["accept-encoding"], "zstd");
    assert_eq!(response.payload["status"], "accepted");
    assert_eq!(response.payload["deduplicated"], false);
    assert_eq!(response.payload["record_count"], 2);
    store
        .await_persisted("telemetry_v1", 1, Duration::from_secs(5))
        .await
        .unwrap();
    let rows = query(
        &store,
        "SELECT name, value, body_json, resource_attributes_json, attributes_json \
         FROM telemetry_v1 ORDER BY record_index",
    )
    .await;
    assert_eq!(rows.iter().map(|batch| batch.num_rows()).sum::<usize>(), 2);
    let names = rows[0].column(0).as_string::<i32>();
    let values = rows[0]
        .column(1)
        .as_primitive::<arrow::datatypes::Float64Type>();
    let bodies = rows[0].column(2).as_string::<i32>();
    assert_eq!(names.value(0), "tokens");
    assert_eq!(values.value(0), 32.0);
    assert!(bodies.is_null(0));
    assert_eq!(names.value(1), "checkpoint");
    assert_eq!(bodies.value(1), r#"{"step":10}"#);
    assert_eq!(
        rows[0].column(3).as_string::<i32>().value(0),
        r#"{"role":"worker","root_run_uid":"run-1"}"#
    );
    assert_eq!(
        rows[0].column(4).as_string::<i32>().value(0),
        r#"{"stage":"train"}"#
    );

    let bounded = query(
        &store,
        "SELECT name, to_timestamp_millis(timestamp_ms) AS observed_at, \
         json_get(resource_attributes_json, 'root_run_uid') AS run \
         FROM telemetry_v1 \
         WHERE timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2023-11-14 22:13:20') * 1000 AS BIGINT) \
         AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '2023-11-14 22:13:21') * 1000 AS BIGINT)",
    )
    .await;
    assert_eq!(
        bounded.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        2
    );
    assert_eq!(bounded[0].column(2).as_string::<i32>().value(0), "run-1");
}

#[tokio::test]
async fn zstd_batch_is_accepted_and_queryable() {
    let store = disk_store("telemetry-zstd");
    let (addr, _) = serve(Arc::clone(&store), AuthPolicy::allow_localhost()).await;
    let client = http_client();
    let batch_id = "9d159e5a-2f32-4d50-8e31-6d8522147520";
    let body = batch(batch_id);
    let compressed = zstd_body(&body);

    let response = post_encoded(
        &client,
        addr,
        compressed,
        Some(batch_id),
        Some("application/json"),
        None,
        Some("zstd"),
    )
    .await;

    assert_eq!(response.status, StatusCode::OK);
    store
        .await_persisted("telemetry_v1", 1, Duration::from_secs(5))
        .await
        .unwrap();
    let rows = query(&store, "SELECT count(*) AS n FROM telemetry_v1").await;
    assert_eq!(
        rows[0]
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>()
            .value(0),
        2
    );
}

#[tokio::test]
async fn repeated_and_concurrent_requests_append_once_but_changed_content_conflicts() {
    let store = disk_store("telemetry-dedupe");
    let (addr, _) = serve(Arc::clone(&store), AuthPolicy::allow_localhost()).await;
    let client = http_client();
    let batch_id = "7d444840-9dc0-11d1-b245-5ffdce74fad2";
    let body = batch(batch_id);

    let first = post(
        &client,
        addr,
        body.clone(),
        Some(batch_id),
        Some("application/json"),
        None,
    );
    let second = post(
        &client,
        addr,
        body.clone(),
        Some(batch_id),
        Some("application/json"),
        None,
    );
    let (first, second) = tokio::join!(first, second);
    assert_eq!(first.status, StatusCode::OK);
    assert_eq!(second.status, StatusCode::OK);
    assert_ne!(
        first.payload["deduplicated"],
        second.payload["deduplicated"]
    );

    let mut changed: Value = serde_json::from_slice(&body).unwrap();
    changed["records"][0]["value"] = json!(33.0);
    let response = post(
        &client,
        addr,
        serde_json::to_vec(&changed).unwrap(),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::CONFLICT);
    assert_eq!(response.payload["error"]["code"], "idempotency_conflict");

    store
        .await_persisted("telemetry_v1", 1, Duration::from_secs(5))
        .await
        .unwrap();
    let rows = query(&store, "SELECT count(*) AS n FROM telemetry_v1").await;
    assert_eq!(
        rows[0]
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>()
            .value(0),
        2
    );
}

#[tokio::test]
async fn malformed_headers_and_records_have_stable_json_errors() {
    let store = disk_store("telemetry-errors");
    let (addr, _) = serve(store, AuthPolicy::allow_localhost()).await;
    let client = http_client();
    let batch_id = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";

    let response = post(
        &client,
        addr,
        batch(batch_id),
        Some(batch_id),
        Some("text/plain"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.payload["error"]["code"], "invalid_request");

    let response = post_encoded(
        &client,
        addr,
        b"not a zstd frame".to_vec(),
        Some(batch_id),
        Some("application/json"),
        None,
        Some("zstd"),
    )
    .await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.payload["error"]["code"], "invalid_request");

    let mut unknown: Value = serde_json::from_slice(&batch(batch_id)).unwrap();
    unknown["records"][0]["unexpected"] = json!(true);
    let response = post(
        &client,
        addr,
        serde_json::to_vec(&unknown).unwrap(),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.payload["error"]["code"], "invalid_request");

    let response = post(
        &client,
        addr,
        batch(batch_id),
        None,
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.payload["error"]["code"], "invalid_request");
}

#[tokio::test]
async fn body_and_normalized_amplification_limits_return_413() {
    let store = disk_store("telemetry-limits");
    let (addr, _) = serve(store, AuthPolicy::allow_localhost()).await;
    let client = http_client();
    let batch_id = "550e8400-e29b-41d4-a716-446655440000";

    let oversized = vec![b' '; (4 << 20) + 1];
    let response = post(
        &client,
        addr,
        oversized.clone(),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(response.payload["error"]["code"], "request_too_large");

    let response = post_encoded(
        &client,
        addr,
        zstd_body(&oversized),
        Some(batch_id),
        Some("application/json"),
        None,
        Some("zstd"),
    )
    .await;
    assert_eq!(response.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(response.payload["error"]["code"], "request_too_large");

    let repeated_records = (0..5_000)
        .map(|index| {
            json!({
                "timestamp_ms": index,
                "kind": "gauge",
                "name": "load",
                "value": 1.0,
                "attributes": {}
            })
        })
        .collect::<Vec<_>>();
    let amplified = serde_json::to_vec(&json!({
        "version": 1,
        "batch_id": batch_id,
        "resource": {"service": "x".repeat(4_096), "attributes": {}},
        "records": repeated_records
    }))
    .unwrap();
    assert!(amplified.len() < 4 << 20);
    let response = post(
        &client,
        addr,
        amplified,
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(response.payload["error"]["code"], "request_too_large");
}

#[tokio::test]
async fn record_attribute_and_string_bounds_are_checked_before_storage() {
    let store = disk_store("telemetry-structural-limits");
    let (addr, _) = serve(store, AuthPolicy::allow_localhost()).await;
    let client = http_client();
    let minimal_record = json!({
        "timestamp_ms": 1,
        "kind": "gauge",
        "name": "x",
        "value": 1.0,
        "attributes": {}
    });
    let many_attributes = (0..65)
        .map(|i| (format!("k{i}"), json!("v")))
        .collect::<serde_json::Map<_, _>>();
    let cases = [
        (
            Value::Array(vec![minimal_record; 10_001]),
            json!({}),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            json!([{
                "timestamp_ms": 1,
                "kind": "gauge",
                "name": "x",
                "value": 1.0,
                "attributes": many_attributes
            }]),
            json!({}),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            json!([{
                "timestamp_ms": 1,
                "kind": "gauge",
                "name": "x".repeat(4_097),
                "value": 1.0,
                "attributes": {}
            }]),
            json!({}),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            json!([{
                "timestamp_ms": 1,
                "kind": "gauge",
                "name": "",
                "value": 1.0,
                "attributes": {}
            }]),
            json!({}),
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (index, (records, attributes, expected)) in cases.into_iter().enumerate() {
        let batch_id = format!("00000000-0000-4000-8000-{index:012}");
        let body = serde_json::to_vec(&json!({
            "version": 1,
            "batch_id": batch_id,
            "resource": {"service": "test", "attributes": attributes},
            "records": records
        }))
        .unwrap();
        let response = post(
            &client,
            addr,
            body,
            Some(&batch_id),
            Some("application/json"),
            None,
        )
        .await;
        assert_eq!(response.status, expected);
    }
}

#[tokio::test]
async fn admission_and_store_unavailability_are_retryable_json_errors() {
    let client = http_client();
    let batch_id = "123e4567-e89b-12d3-a456-426614174001";
    let store = disk_store("telemetry-admission");
    let addr = serve_with_config(
        store,
        ServerConfig {
            max_concurrent_telemetry: 0,
            ..ServerConfig::default()
        },
    )
    .await;
    let response = post(
        &client,
        addr,
        batch(batch_id),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers["retry-after"], "1");
    assert_eq!(response.payload["error"]["code"], "admission_limited");

    let store = disk_store("telemetry-schema-conflict");
    store
        .register_table(
            "telemetry_v1",
            Schema::new(
                vec![Column::new("other", ColumnType::COLUMN_TYPE_STRING, false)],
                "other",
            ),
            StoragePolicy::default(),
        )
        .unwrap();
    let (addr, _) = serve(store, AuthPolicy::allow_localhost()).await;
    let response = post(
        &client,
        addr,
        batch(batch_id),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers["retry-after"], "1");
    assert_eq!(response.payload["error"]["code"], "storage_unavailable");
}

#[tokio::test]
async fn telemetry_route_uses_existing_default_deny_auth_policy() {
    let store = disk_store("telemetry-auth");
    let policy_json = serde_json::to_string(&json!([{
        "type": "jwt",
        "keys": [{"cluster": "test", "public_keys": [PUB_A]}]
    }]))
    .unwrap();
    let (addr, _) = serve(store, AuthPolicy::parse(&policy_json).unwrap()).await;
    let client = http_client();
    let batch_id = "123e4567-e89b-12d3-a456-426614174000";

    let response = post(
        &client,
        addr,
        batch(batch_id),
        Some(batch_id),
        Some("application/json"),
        None,
    )
    .await;

    assert_eq!(response.status, StatusCode::UNAUTHORIZED);
    assert_eq!(response.payload["error"]["code"], "unauthorized");
}
