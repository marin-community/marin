// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Native-proxy request metrics.
//!
//! Two families, both recorded by the native listener as it handles a request
//! and exposed to Python as JSON snapshots:
//!
//! - `iris_rpc_*` — Connect-RPC counters keyed by service/method/upstream, for
//!   every Iris RPC the proxy handles (controller RPCs included).
//! - `iris_proxy_*` — transport load keyed by endpoint/method/route_kind, plus
//!   request and response byte volume, for every request the proxy *forwards*
//!   (plain HTTP included; controller RPCs excluded).
//!
//! A proxied Connect call contributes to both — transport here, semantics in
//! `iris_rpc_*` — so the two families must not be summed. Counters reset with
//! the proxy process; histogram buckets are cumulative.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::Request;
use axum::http::{Method, StatusCode};
use axum::response::Response;
use serde::Serialize;
use tokio_stream::StreamExt;

use crate::{
    parse_proxy_route, proxy_subdomain, PROXY_METRICS_LOCK_POISONED, PROXY_PATH_PREFIX,
    RPC_METRICS_LOCK_POISONED,
};

const RPC_LATENCY_BUCKETS_SECONDS: [f64; 11] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

// Bound on distinct proxy endpoints tracked at once; overflow folds into
// PROXY_OTHER_ENDPOINT and idle endpoints are evicted after the window below.
const PROXY_ENDPOINT_CAP: usize = 128;
const PROXY_ENDPOINT_IDLE_EVICTION: Duration = Duration::from_secs(3600);
const PROXY_OTHER_ENDPOINT: &str = "__other__";

/// Cumulative latency buckets shared by the RPC and proxy metric families.
/// `record` folds one observation into every bucket whose upper bound it meets
/// (Prometheus `le` semantics); `buckets` renders the cumulative list with a
/// trailing `+Inf` bucket carrying the series' total count.
#[derive(Default)]
struct LatencyHistogram {
    bucket_counts: [u64; RPC_LATENCY_BUCKETS_SECONDS.len()],
    sum_seconds: f64,
}

impl LatencyHistogram {
    fn record(&mut self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        self.sum_seconds += seconds;
        for (index, upper_bound) in RPC_LATENCY_BUCKETS_SECONDS.iter().enumerate() {
            if seconds <= *upper_bound {
                self.bucket_counts[index] += 1;
            }
        }
    }

    fn buckets(&self, total: u64) -> Vec<(String, u64)> {
        RPC_LATENCY_BUCKETS_SECONDS
            .iter()
            .zip(self.bucket_counts)
            .map(|(bound, count)| (bound.to_string(), count))
            .chain(std::iter::once(("+Inf".to_string(), total)))
            .collect()
    }
}

// ===== iris_rpc_* : Connect-RPC metrics =====

/// Lifetime RPC counters and latency observations from the native listener.
///
/// Counters reset with the proxy process. Histogram buckets are cumulative.
#[derive(Debug, Serialize)]
pub struct RpcMetricsSnapshot {
    pub series: Vec<RpcMetricSeries>,
}

#[derive(Debug, Serialize)]
pub struct RpcMetricSeries {
    pub service: String,
    pub method: String,
    pub upstream: String,
    pub requests: u64,
    pub responses: BTreeMap<u16, u64>,
    pub in_flight: u64,
    pub latency_buckets: Vec<(String, u64)>,
    pub latency_count: u64,
    pub latency_sum_seconds: f64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct RpcMetricKey {
    service: String,
    method: String,
    upstream: &'static str,
}

#[derive(Default)]
struct RpcMetricState {
    requests: u64,
    responses: BTreeMap<u16, u64>,
    in_flight: u64,
    latency: LatencyHistogram,
}

#[derive(Default)]
pub(crate) struct RpcMetrics {
    series: BTreeMap<RpcMetricKey, RpcMetricState>,
}

impl RpcMetrics {
    fn begin(&mut self, key: RpcMetricKey) {
        let state = self.series.entry(key).or_default();
        state.requests += 1;
        state.in_flight += 1;
    }

    fn finish(&mut self, key: &RpcMetricKey, status: StatusCode, elapsed: Duration) {
        let Some(state) = self.series.get_mut(key) else {
            return;
        };
        state.in_flight = state.in_flight.saturating_sub(1);
        *state.responses.entry(status.as_u16()).or_default() += 1;
        state.latency.record(elapsed);
    }

    pub(crate) fn snapshot(&self) -> RpcMetricsSnapshot {
        RpcMetricsSnapshot {
            series: self
                .series
                .iter()
                .map(|(key, state)| {
                    let latency_count = state.responses.values().sum();
                    RpcMetricSeries {
                        service: key.service.clone(),
                        method: key.method.clone(),
                        upstream: key.upstream.to_string(),
                        requests: state.requests,
                        responses: state.responses.clone(),
                        in_flight: state.in_flight,
                        latency_buckets: state.latency.buckets(latency_count),
                        latency_count,
                        latency_sum_seconds: state.latency.sum_seconds,
                    }
                })
                .collect(),
        }
    }
}

/// Times one Connect RPC: increments the in-flight gauge on `begin` and records
/// status and latency on `finish`. Attribution comes from the request path only.
pub(crate) struct RpcRequestTimer {
    key: RpcMetricKey,
    metrics: Arc<Mutex<RpcMetrics>>,
    started: Instant,
}

impl RpcRequestTimer {
    pub(crate) fn begin(metrics: &Arc<Mutex<RpcMetrics>>, request: &Request) -> Option<Self> {
        let key = rpc_metric_key(request)?;
        metrics
            .lock()
            .expect(RPC_METRICS_LOCK_POISONED)
            .begin(key.clone());
        Some(Self {
            key,
            metrics: Arc::clone(metrics),
            started: Instant::now(),
        })
    }

    pub(crate) fn finish(self, status: StatusCode) {
        self.metrics
            .lock()
            .expect(RPC_METRICS_LOCK_POISONED)
            .finish(&self.key, status, self.started.elapsed());
    }
}

fn rpc_metric_key(request: &Request) -> Option<RpcMetricKey> {
    let segments = request
        .uri()
        .path()
        .split('/')
        .filter(|segment| !segment.is_empty());
    let mut segments = segments.skip_while(|segment| !segment.starts_with("iris."));
    let service = segments.next()?;
    let method = segments.next()?;
    if segments.next().is_some() {
        return None;
    }
    Some(RpcMetricKey {
        service: service.to_string(),
        method: method.to_string(),
        upstream: if request.uri().path().starts_with(PROXY_PATH_PREFIX) {
            "endpoint"
        } else {
            "controller"
        },
    })
}

// ===== iris_proxy_* : proxy transport metrics =====
//
// Distinct from the `iris_rpc_*` Connect metrics: these count every request the
// native proxy forwards to an endpoint or relays to a peer — plain HTTP included
// (vLLM/OpenAI upstreams, capability URLs, relays) — keyed by the resolved
// endpoint name, normalized HTTP method, and route kind, plus the request and
// response byte volume streamed through. A proxied Connect call contributes to
// both views (transport here, semantics in `iris_rpc_*`); they are not summable.
//
// Per-endpoint series are bounded: at most `PROXY_ENDPOINT_CAP` distinct
// endpoints, idle ones (no activity for `PROXY_ENDPOINT_IDLE_EVICTION` and no
// in-flight request) evicted, and any overflow folded into `PROXY_OTHER_ENDPOINT`
// so an adversarial path cannot grow the label set without bound. The aggregate
// is exact and never evicted, so the true total survives eviction.

fn normalize_method(method: &Method) -> &'static str {
    match method.as_str() {
        "GET" => "GET",
        "POST" => "POST",
        "PUT" => "PUT",
        "PATCH" => "PATCH",
        "DELETE" => "DELETE",
        "HEAD" => "HEAD",
        "OPTIONS" => "OPTIONS",
        _ => "OTHER",
    }
}

/// Attribute a request to a proxy series, or `None` when it is not a proxy
/// request (controller RPCs are covered by `iris_rpc_*`). Attribution comes from
/// the typed proxy route, never a raw path scan, so tokens and cluster tags never
/// leak into labels.
fn proxy_metric_key(request: &Request) -> Option<ProxyMetricKey> {
    let (endpoint, route_kind) = if request.uri().path().starts_with(PROXY_PATH_PREFIX) {
        match parse_proxy_route(request.uri()) {
            Ok(route) => (
                route.encoded_name,
                if route.relay_peer.is_some() {
                    "relay"
                } else {
                    "endpoint"
                },
            ),
            // A malformed /proxy/ path still forwards a decision and should count;
            // bucket it so the aggregate stays exact without a per-path label.
            Err(_) => ("__unparsed__".to_string(), "endpoint"),
        }
    } else {
        let name = proxy_subdomain(request.headers())?;
        (name, "endpoint")
    };
    Some(ProxyMetricKey {
        endpoint,
        method: normalize_method(request.method()),
        route_kind,
    })
}

#[derive(Clone, Copy)]
enum ProxyByteDirection {
    Request,
    Response,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ProxyMetricKey {
    endpoint: String,
    method: &'static str,
    route_kind: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ProxyMetricSeries {
    pub endpoint: String,
    pub method: String,
    pub route_kind: String,
    pub requests: u64,
    pub responses: BTreeMap<u16, u64>,
    pub in_flight: u64,
    pub latency_buckets: Vec<(String, u64)>,
    pub latency_count: u64,
    pub latency_sum_seconds: f64,
    pub request_bytes: u64,
    pub response_bytes: u64,
}

#[derive(Default)]
struct ProxyMetricState {
    requests: u64,
    responses: BTreeMap<u16, u64>,
    in_flight: u64,
    latency: LatencyHistogram,
    request_bytes: u64,
    response_bytes: u64,
    last_activity: Option<Instant>,
}

impl ProxyMetricState {
    fn begin(&mut self, now: Instant) {
        self.requests += 1;
        self.in_flight += 1;
        self.last_activity = Some(now);
    }

    fn finish(&mut self, status: StatusCode, elapsed: Duration, now: Instant) {
        self.in_flight = self.in_flight.saturating_sub(1);
        *self.responses.entry(status.as_u16()).or_default() += 1;
        self.latency.record(elapsed);
        self.last_activity = Some(now);
    }

    fn add_bytes(&mut self, direction: ProxyByteDirection, bytes: u64, now: Instant) {
        match direction {
            ProxyByteDirection::Request => self.request_bytes += bytes,
            ProxyByteDirection::Response => self.response_bytes += bytes,
        }
        self.last_activity = Some(now);
    }

    fn series(&self, endpoint: String, method: String, route_kind: String) -> ProxyMetricSeries {
        let latency_count = self.responses.values().sum();
        ProxyMetricSeries {
            endpoint,
            method,
            route_kind,
            requests: self.requests,
            responses: self.responses.clone(),
            in_flight: self.in_flight,
            latency_buckets: self.latency.buckets(latency_count),
            latency_count,
            latency_sum_seconds: self.latency.sum_seconds,
            request_bytes: self.request_bytes,
            response_bytes: self.response_bytes,
        }
    }
}

#[derive(Default)]
pub(crate) struct ProxyMetrics {
    aggregate: ProxyMetricState,
    by_endpoint: BTreeMap<ProxyMetricKey, ProxyMetricState>,
}

impl ProxyMetrics {
    /// Record a request start and return the key it landed on (the caller's key,
    /// or the overflow bucket when the endpoint cap is full). `finish`/`add_bytes`
    /// must use the returned key so a request accounts to one series throughout.
    fn begin(&mut self, key: ProxyMetricKey, now: Instant) -> ProxyMetricKey {
        self.aggregate.begin(now);
        self.evict_idle(now);
        let resolved =
            if self.by_endpoint.contains_key(&key) || self.by_endpoint.len() < PROXY_ENDPOINT_CAP {
                key
            } else {
                ProxyMetricKey {
                    endpoint: PROXY_OTHER_ENDPOINT.to_string(),
                    method: "",
                    route_kind: "",
                }
            };
        self.by_endpoint
            .entry(resolved.clone())
            .or_default()
            .begin(now);
        resolved
    }

    fn finish(&mut self, key: &ProxyMetricKey, status: StatusCode, elapsed: Duration) {
        let now = Instant::now();
        self.aggregate.finish(status, elapsed, now);
        if let Some(state) = self.by_endpoint.get_mut(key) {
            state.finish(status, elapsed, now);
        }
    }

    fn add_bytes(&mut self, key: &ProxyMetricKey, direction: ProxyByteDirection, bytes: u64) {
        let now = Instant::now();
        self.aggregate.add_bytes(direction, bytes, now);
        if let Some(state) = self.by_endpoint.get_mut(key) {
            state.add_bytes(direction, bytes, now);
        }
    }

    fn evict_idle(&mut self, now: Instant) {
        self.by_endpoint.retain(|_, state| {
            state.in_flight > 0
                || state
                    .last_activity
                    .is_none_or(|last| now.duration_since(last) < PROXY_ENDPOINT_IDLE_EVICTION)
        });
    }

    pub(crate) fn snapshot(&self) -> ProxyMetricsSnapshot {
        ProxyMetricsSnapshot {
            aggregate: self
                .aggregate
                .series(String::new(), String::new(), String::new()),
            series: self
                .by_endpoint
                .iter()
                .map(|(key, state)| {
                    state.series(
                        key.endpoint.clone(),
                        key.method.to_string(),
                        key.route_kind.to_string(),
                    )
                })
                .collect(),
        }
    }
}

/// Lifetime proxy-transport counters from the native listener. `aggregate` is the
/// exact total across all proxied traffic; `series` is the bounded per-endpoint
/// breakdown. Counters reset with the proxy process.
#[derive(Debug, Serialize)]
pub struct ProxyMetricsSnapshot {
    pub aggregate: ProxyMetricSeries,
    pub series: Vec<ProxyMetricSeries>,
}

/// Records streamed body bytes into a proxy series on drop, so a client abort or
/// stream error still books the bytes transferred before it (counted as data
/// frames are polled, never buffered).
struct ProxyByteFlush {
    counter: Arc<AtomicU64>,
    metrics: Arc<Mutex<ProxyMetrics>>,
    key: ProxyMetricKey,
    direction: ProxyByteDirection,
}

impl Drop for ProxyByteFlush {
    fn drop(&mut self) {
        let bytes = self.counter.load(Ordering::Relaxed);
        if bytes == 0 {
            return;
        }
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.add_bytes(&self.key, self.direction, bytes);
        }
    }
}

/// Times one forwarded request: increments the in-flight gauge on `begin`,
/// wraps the request and response bodies to count bytes, and records status and
/// latency on `finish`.
pub(crate) struct ProxyRequestTimer {
    key: ProxyMetricKey,
    metrics: Arc<Mutex<ProxyMetrics>>,
    started: Instant,
}

impl ProxyRequestTimer {
    pub(crate) fn begin(metrics: &Arc<Mutex<ProxyMetrics>>, request: &Request) -> Option<Self> {
        let key = proxy_metric_key(request)?;
        let resolved = metrics
            .lock()
            .expect(PROXY_METRICS_LOCK_POISONED)
            .begin(key, Instant::now());
        Some(Self {
            key: resolved,
            metrics: Arc::clone(metrics),
            started: Instant::now(),
        })
    }

    pub(crate) fn wrap_request_body(&self, request: Request) -> Request {
        let (parts, body) = request.into_parts();
        Request::from_parts(parts, self.counting_body(body, ProxyByteDirection::Request))
    }

    pub(crate) fn finish(self, response: Response<Body>) -> Response<Body> {
        let (parts, body) = response.into_parts();
        self.metrics
            .lock()
            .expect(PROXY_METRICS_LOCK_POISONED)
            .finish(&self.key, parts.status, self.started.elapsed());
        let body = self.counting_body(body, ProxyByteDirection::Response);
        Response::from_parts(parts, body)
    }

    /// Wrap a body so each polled data frame's length accrues to `counter`, which
    /// the held `ProxyByteFlush` books to the series when the stream is dropped.
    fn counting_body(&self, body: Body, direction: ProxyByteDirection) -> Body {
        let counter = Arc::new(AtomicU64::new(0));
        let flush = ProxyByteFlush {
            counter: Arc::clone(&counter),
            metrics: Arc::clone(&self.metrics),
            key: self.key.clone(),
            direction,
        };
        let stream = Body::new(body).into_data_stream().map(move |item| {
            // Force the move closure to own `flush` so it drops (and books bytes)
            // exactly when the wrapped stream is dropped.
            let _flush = &flush;
            if let Ok(bytes) = &item {
                counter.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            item
        });
        Body::from_stream(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::to_bytes;

    fn proxy_request(method: &str, uri: &str) -> Request {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn rpc_metrics_track_started_and_completed_controller_requests() {
        let metrics = Arc::new(Mutex::new(RpcMetrics::default()));
        let request = Request::builder()
            .uri("/iris.cluster.ControllerService/ListJobs")
            .body(Body::empty())
            .unwrap();
        let timer = RpcRequestTimer::begin(&metrics, &request).unwrap();

        let snapshot = metrics.lock().unwrap().snapshot();
        let series = snapshot.series.first().unwrap();
        assert_eq!(series.requests, 1);
        assert_eq!(series.in_flight, 1);
        assert_eq!(series.latency_count, 0);
        assert_eq!(
            series.latency_buckets.last(),
            Some(&("+Inf".to_string(), 0))
        );

        timer.finish(StatusCode::INTERNAL_SERVER_ERROR);

        let snapshot = metrics.lock().unwrap().snapshot();
        let series = snapshot.series.first().unwrap();
        assert_eq!(series.service, "iris.cluster.ControllerService");
        assert_eq!(series.method, "ListJobs");
        assert_eq!(series.upstream, "controller");
        assert_eq!(series.requests, 1);
        assert_eq!(series.responses.get(&500), Some(&1));
        assert_eq!(series.in_flight, 0);
        assert_eq!(series.latency_count, 1);
        assert_eq!(
            series.latency_buckets.last(),
            Some(&("+Inf".to_string(), 1))
        );
    }

    #[test]
    fn proxy_metric_key_attributes_by_typed_route_and_skips_controller_rpcs() {
        let endpoint =
            proxy_metric_key(&proxy_request("POST", "/proxy/serve.model/v1/chat")).unwrap();
        assert_eq!(endpoint.endpoint, "serve.model");
        assert_eq!(endpoint.method, "POST");
        assert_eq!(endpoint.route_kind, "endpoint");

        let relay = proxy_metric_key(&proxy_request(
            "GET",
            "/proxy/t/cluster=cw-rno2a/tok/serve.model/v1/models",
        ))
        .unwrap();
        assert_eq!(relay.endpoint, "serve.model");
        assert_eq!(relay.route_kind, "relay");

        // A non-proxy controller RPC is covered by iris_rpc_*, not counted here.
        assert!(proxy_metric_key(&proxy_request(
            "POST",
            "/iris.cluster.ControllerService/ListJobs"
        ))
        .is_none());
    }

    #[test]
    fn proxy_metrics_track_requests_and_keep_an_exact_aggregate() {
        let metrics = Arc::new(Mutex::new(ProxyMetrics::default()));
        let timer =
            ProxyRequestTimer::begin(&metrics, &proxy_request("GET", "/proxy/svc/a")).unwrap();

        let snapshot = metrics.lock().unwrap().snapshot();
        assert_eq!(snapshot.aggregate.requests, 1);
        assert_eq!(snapshot.aggregate.in_flight, 1);
        let series = snapshot
            .series
            .iter()
            .find(|s| s.endpoint == "svc")
            .unwrap();
        assert_eq!(series.method, "GET");
        assert_eq!(series.route_kind, "endpoint");
        assert_eq!(series.in_flight, 1);

        timer.finish(Response::builder().status(200).body(Body::empty()).unwrap());

        let snapshot = metrics.lock().unwrap().snapshot();
        assert_eq!(snapshot.aggregate.requests, 1);
        assert_eq!(snapshot.aggregate.in_flight, 0);
        assert_eq!(snapshot.aggregate.responses.get(&200), Some(&1));
        let series = snapshot
            .series
            .iter()
            .find(|s| s.endpoint == "svc")
            .unwrap();
        assert_eq!(series.in_flight, 0);
        assert_eq!(series.responses.get(&200), Some(&1));
    }

    #[test]
    fn proxy_endpoint_cap_folds_overflow_into_other() {
        let mut metrics = ProxyMetrics::default();
        let now = Instant::now();
        for index in 0..(PROXY_ENDPOINT_CAP + 5) {
            metrics.begin(
                ProxyMetricKey {
                    endpoint: format!("svc-{index}"),
                    method: "GET",
                    route_kind: "endpoint",
                },
                now,
            );
        }
        // Cap distinct endpoints; the overflow lands in the __other__ bucket, and
        // the aggregate still counts every request.
        assert!(metrics.by_endpoint.len() <= PROXY_ENDPOINT_CAP + 1);
        assert!(metrics
            .by_endpoint
            .keys()
            .any(|key| key.endpoint == PROXY_OTHER_ENDPOINT));
        assert_eq!(metrics.aggregate.requests, (PROXY_ENDPOINT_CAP + 5) as u64);
    }

    #[test]
    fn proxy_metrics_evict_idle_endpoints_without_losing_the_aggregate() {
        let mut metrics = ProxyMetrics::default();
        let key = ProxyMetricKey {
            endpoint: "svc".to_string(),
            method: "GET",
            route_kind: "endpoint",
        };
        let resolved = metrics.begin(key.clone(), Instant::now());
        metrics.finish(&resolved, StatusCode::OK, Duration::from_millis(1));
        // Simulate the endpoint going idle past the window (finish leaves it live).
        metrics.by_endpoint.get_mut(&key).unwrap().last_activity =
            Some(Instant::now() - PROXY_ENDPOINT_IDLE_EVICTION - Duration::from_secs(1));
        // A later request on a different endpoint triggers eviction of the idle one.
        metrics.begin(
            ProxyMetricKey {
                endpoint: "other".to_string(),
                method: "GET",
                route_kind: "endpoint",
            },
            Instant::now(),
        );
        assert!(!metrics.by_endpoint.contains_key(&key));
        assert_eq!(metrics.aggregate.requests, 2);
    }

    #[tokio::test]
    async fn proxy_counting_body_books_streamed_request_and_response_bytes() {
        let metrics = Arc::new(Mutex::new(ProxyMetrics::default()));
        let request = Request::builder()
            .method("POST")
            .uri("/proxy/svc/echo")
            .body(Body::from("request-body"))
            .unwrap();
        let timer = ProxyRequestTimer::begin(&metrics, &request).unwrap();

        let request = timer.wrap_request_body(request);
        let read = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        assert_eq!(read.len(), "request-body".len());

        let response = timer.finish(
            Response::builder()
                .status(200)
                .body(Body::from("response-body"))
                .unwrap(),
        );
        let delivered = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(delivered.len(), "response-body".len());

        let snapshot = metrics.lock().unwrap().snapshot();
        assert_eq!(
            snapshot.aggregate.request_bytes,
            "request-body".len() as u64
        );
        assert_eq!(
            snapshot.aggregate.response_bytes,
            "response-body".len() as u64
        );
        let series = snapshot
            .series
            .iter()
            .find(|s| s.endpoint == "svc")
            .unwrap();
        assert_eq!(series.request_bytes, "request-body".len() as u64);
        assert_eq!(series.response_bytes, "response-body".len() as u64);
    }
}
