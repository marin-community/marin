// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Native public listener for the Iris controller.
//!
//! The listener keeps client and endpoint bodies on Rust's Tokio runtime. Python
//! remains the control plane behind a private controller listener. Rust parses
//! and authenticates every public endpoint request, resolves local endpoints,
//! and streams request bodies. Federation uses a capability-authenticated,
//! normalized control-plane decision. Non-proxy requests are forwarded unchanged.

mod auth;

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use axum::response::Response;
use axum::routing::any;
use axum::Router;
use hyper_rustls::HttpsConnector;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_stream::StreamExt;

pub const DECISION_PATH: &str = "/_iris/internal/proxy-decision";
pub const DECISION_SECRET_HEADER: &str = "x-iris-decision-secret";
pub const UPSTREAM_URL_HEADER: &str = "x-iris-upstream-url";
pub const UPSTREAM_AUTHORIZATION_HEADER: &str = "x-iris-upstream-authorization";
pub const PROXY_PREFIX_HEADER: &str = "x-iris-proxy-prefix";
pub const PROXY_TIMEOUT_HEADER: &str = "x-iris-proxy-timeout";
pub const DEFAULT_PROXY_TIMEOUT_SECONDS: u64 = 120;
pub const PROXY_METHODS: [&str; 7] = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"];

pub use auth::NativeAuthConfig;
use auth::{CacheStats, NativeVerifier, VerifyOutcome, IAP_ASSERTION_HEADER};

const MAX_DECISION_BODY_BYTES: usize = 64 * 1024;
const PROXY_PATH_PREFIX: &str = "/proxy/";
const REGISTRY_LOCK_POISONED: &str = "native proxy registry lock is poisoned";
const DEFAULT_PROXY_TIMEOUT: Duration = Duration::from_secs(DEFAULT_PROXY_TIMEOUT_SECONDS);
const X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");
const X_FORWARDED_HOST: HeaderName = HeaderName::from_static("x-forwarded-host");
const X_FORWARDED_PREFIX: HeaderName = HeaderName::from_static("x-forwarded-prefix");
const X_FORWARDED_PROTO: HeaderName = HeaderName::from_static("x-forwarded-proto");

type HttpsClient = Client<HttpsConnector<HttpConnector>, Body>;
type DecisionResult = Result<ProxyDecision, Box<Response<Body>>>;

#[derive(Clone)]
pub struct ProxyConfig {
    pub controller_url: String,
    pub decision_secret: String,
    pub auth: NativeAuthConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EndpointMapping {
    pub endpoint_id: String,
    pub name: String,
    pub address: String,
    pub link_access: bool,
    pub peer_id: Option<String>,
    pub task_id: Option<String>,
    pub timeout_seconds: Option<f64>,
    pub lease_deadline_epoch_ms: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct RegistrySnapshot {
    pub generation: u64,
    pub endpoints: Vec<EndpointMapping>,
}

#[derive(Debug, Deserialize)]
pub struct MappingDelta {
    pub base_generation: u64,
    pub next_generation: u64,
    pub upserts: Vec<EndpointMapping>,
    pub deletes: Vec<String>,
}

#[derive(Default)]
struct RegistryState {
    generation: u64,
    by_id: HashMap<String, EndpointMapping>,
    by_name: HashMap<String, HashSet<String>>,
}

#[derive(Clone, Default)]
pub struct ProxyControl {
    registry: Arc<RwLock<RegistryState>>,
    ready: Arc<AtomicBool>,
    stats: Arc<CacheStats>,
}

#[derive(Debug, Serialize)]
pub struct ProxyStats {
    pub registry_generation: u64,
    pub endpoint_count: usize,
    pub jwt_cache_hits: u64,
    pub jwt_cache_misses: u64,
}

impl ProxyControl {
    pub fn stats(&self) -> Result<ProxyStats, String> {
        let registry = self
            .registry
            .read()
            .map_err(|_| REGISTRY_LOCK_POISONED.to_string())?;
        let (jwt_cache_hits, jwt_cache_misses) = self.stats.snapshot();
        Ok(ProxyStats {
            registry_generation: registry.generation,
            endpoint_count: registry.by_id.len(),
            jwt_cache_hits,
            jwt_cache_misses,
        })
    }

    pub fn pause_registry(&self) {
        self.ready.store(false, Ordering::Release);
    }

    pub fn replace_registry(&self, snapshot: RegistrySnapshot) -> Result<(), String> {
        let mut next = RegistryState {
            generation: snapshot.generation,
            ..RegistryState::default()
        };
        for mapping in snapshot.endpoints {
            validate_mapping(&mapping)?;
            if next.by_id.contains_key(&mapping.endpoint_id) {
                return Err(format!(
                    "duplicate endpoint id in registry snapshot: {}",
                    mapping.endpoint_id
                ));
            }
            index_mapping(&mut next, mapping);
        }
        *self
            .registry
            .write()
            .map_err(|_| REGISTRY_LOCK_POISONED.to_string())? = next;
        self.ready.store(true, Ordering::Release);
        Ok(())
    }

    pub fn update_mappings(&self, delta: MappingDelta) -> Result<(), String> {
        if delta.next_generation != delta.base_generation + 1 {
            return Err(format!(
                "mapping delta must advance exactly one generation: {} -> {}",
                delta.base_generation, delta.next_generation
            ));
        }
        let mut touched = HashSet::new();
        for endpoint_id in &delta.deletes {
            if !touched.insert(endpoint_id.as_str()) {
                return Err(format!(
                    "duplicate endpoint id in mapping delta: {endpoint_id}"
                ));
            }
        }
        for mapping in &delta.upserts {
            validate_mapping(mapping)?;
            if !touched.insert(mapping.endpoint_id.as_str()) {
                return Err(format!(
                    "duplicate endpoint id in mapping delta: {}",
                    mapping.endpoint_id
                ));
            }
        }

        let mut registry = self
            .registry
            .write()
            .map_err(|_| REGISTRY_LOCK_POISONED.to_string())?;
        if registry.generation != delta.base_generation {
            return Err(format!(
                "mapping generation mismatch: native={}, delta base={}",
                registry.generation, delta.base_generation
            ));
        }
        for endpoint_id in delta.deletes {
            remove_mapping(&mut registry, &endpoint_id);
        }
        for mapping in delta.upserts {
            remove_mapping(&mut registry, &mapping.endpoint_id);
            index_mapping(&mut registry, mapping);
        }
        registry.generation = delta.next_generation;
        Ok(())
    }

    fn resolve(&self, encoded_name: &str) -> Result<Option<EndpointMapping>, String> {
        let decoded = encoded_name.replace('.', "/");
        let candidates = [format!("/{decoded}"), decoded];
        let now = unix_time_millis();
        let registry = self
            .registry
            .read()
            .map_err(|_| REGISTRY_LOCK_POISONED.to_string())?;
        for name in candidates {
            let Some(endpoint_ids) = registry.by_name.get(&name) else {
                continue;
            };
            let mut mappings = endpoint_ids
                .iter()
                .filter_map(|endpoint_id| registry.by_id.get(endpoint_id))
                .filter(|mapping| {
                    mapping
                        .lease_deadline_epoch_ms
                        .is_none_or(|deadline| deadline > now)
                })
                .collect::<Vec<_>>();
            if mappings.is_empty() {
                continue;
            }
            if mappings
                .iter()
                .map(|mapping| mapping.peer_id.as_deref())
                .collect::<HashSet<_>>()
                .len()
                > 1
            {
                return Err(format!("endpoint {name:?} resolves to multiple peers"));
            }
            mappings.sort_unstable_by_key(|mapping| mapping.endpoint_id.as_str());
            return Ok(Some(mappings[0].clone()));
        }
        Ok(None)
    }

    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

fn unix_time_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn validate_mapping(mapping: &EndpointMapping) -> Result<(), String> {
    if mapping.endpoint_id.is_empty() {
        return Err("endpoint id must not be empty".to_string());
    }
    if mapping.name.is_empty() {
        return Err(format!(
            "endpoint {} has an empty name",
            mapping.endpoint_id
        ));
    }
    let address = endpoint_uri(&mapping.address, "", None).map_err(|error| {
        format!(
            "endpoint {} has an invalid address: {error}",
            mapping.endpoint_id
        )
    })?;
    if !matches!(address.scheme_str(), Some("http" | "https")) || address.authority().is_none() {
        return Err(format!(
            "endpoint {} address must be an absolute HTTP(S) URL",
            mapping.endpoint_id
        ));
    }
    if mapping
        .timeout_seconds
        .is_some_and(|timeout| timeout <= 0.0)
    {
        return Err(format!(
            "endpoint {} timeout must be positive",
            mapping.endpoint_id
        ));
    }
    Ok(())
}

fn endpoint_uri(address: &str, sub_path: &str, query: Option<&str>) -> Result<Uri, String> {
    let base = if address.contains("://") {
        address.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", address.trim_end_matches('/'))
    };
    let mut upstream = format!("{base}/{sub_path}");
    if let Some(query) = query {
        upstream.push('?');
        upstream.push_str(query);
    }
    upstream
        .parse()
        .map_err(|error| format!("invalid endpoint upstream URL: {error}"))
}

fn index_mapping(registry: &mut RegistryState, mapping: EndpointMapping) {
    registry
        .by_name
        .entry(mapping.name.clone())
        .or_default()
        .insert(mapping.endpoint_id.clone());
    registry.by_id.insert(mapping.endpoint_id.clone(), mapping);
}

fn remove_mapping(registry: &mut RegistryState, endpoint_id: &str) {
    let Some(previous) = registry.by_id.remove(endpoint_id) else {
        return;
    };
    let remove_name = if let Some(endpoint_ids) = registry.by_name.get_mut(&previous.name) {
        endpoint_ids.remove(endpoint_id);
        endpoint_ids.is_empty()
    } else {
        false
    };
    if remove_name {
        registry.by_name.remove(&previous.name);
    }
}

struct AppState {
    controller_url: String,
    decision_secret: HeaderValue,
    decision_client: HttpsClient,
    controller_client: HttpsClient,
    upstream_client: HttpsClient,
    control: ProxyControl,
    verifier: NativeVerifier,
}

struct ProxyDecision {
    upstream: Uri,
    proxy_prefix: Option<HeaderValue>,
    upstream_authorization: Option<HeaderValue>,
    timeout: Duration,
}

struct ProxyRoute {
    encoded_name: String,
    sub_path: String,
    token: Option<String>,
    proxy_prefix: String,
    redirect: bool,
}

enum NativeDecision {
    Proxy(ProxyDecision),
    Response(Box<Response<Body>>),
    Federation(FederationDecision),
    ControllerProxy,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum FederationDirection {
    Inbound,
    Outbound,
}

#[derive(Serialize)]
struct FederationDecision {
    direction: FederationDirection,
    encoded_name: String,
    sub_path: String,
    query: String,
    proxy_prefix: String,
    peer_id: String,
    task_id: Option<String>,
    local_upstream: Option<String>,
    timeout_seconds: Option<f64>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

fn https_connector() -> HttpsConnector<HttpConnector> {
    let mut connector = HttpConnector::new();
    connector.enforce_http(false);
    connector.set_connect_timeout(Some(Duration::from_secs(30)));
    hyper_rustls::HttpsConnectorBuilder::new()
        .with_webpki_roots()
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .wrap_connector(connector)
}

fn client(max_idle_per_host: usize) -> HttpsClient {
    Client::builder(TokioExecutor::new())
        .pool_max_idle_per_host(max_idle_per_host)
        .build(https_connector())
}

fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    let body = serde_json::to_vec(&ErrorBody { error: message })
        .expect("serializing a string error cannot fail");
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static error response is valid")
}

fn controller_uri(controller_url: &str, request_uri: &Uri) -> Result<Uri, Box<Response<Body>>> {
    format!("{controller_url}{request_uri}")
        .parse()
        .map_err(|error| {
            Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid private controller URI: {error}"),
            ))
        })
}

fn decision_uri(controller_url: &str) -> Result<Uri, Box<Response<Body>>> {
    format!("{controller_url}{DECISION_PATH}")
        .parse()
        .map_err(|error| {
            Box::new(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("invalid proxy decision URI: {error}"),
            ))
        })
}

fn remove_connection_headers(headers: &mut HeaderMap) {
    let named_by_connection = headers
        .get(header::CONNECTION)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(',')
                .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for name in named_by_connection {
        headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        HeaderName::from_static("keep-alive"),
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ] {
        headers.remove(name);
    }
}

fn remove_internal_headers(headers: &mut HeaderMap) {
    for name in [
        DECISION_SECRET_HEADER,
        UPSTREAM_URL_HEADER,
        UPSTREAM_AUTHORIZATION_HEADER,
        PROXY_PREFIX_HEADER,
        PROXY_TIMEOUT_HEADER,
    ] {
        headers.remove(name);
    }
}

fn prepare_public_headers(headers: &mut HeaderMap, peer: SocketAddr) {
    remove_internal_headers(headers);
    if !headers.contains_key(&X_FORWARDED_FOR) {
        headers.insert(
            X_FORWARDED_FOR,
            HeaderValue::from_str(&peer.ip().to_string())
                .expect("IP addresses are valid header values"),
        );
    }
}

async fn decision(state: &AppState, decision: &FederationDecision) -> DecisionResult {
    let uri = decision_uri(&state.controller_url)?;
    let payload = serde_json::to_vec(decision).map_err(|error| {
        Box::new(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to encode federation decision: {error}"),
        ))
    })?;
    let mut decision_request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(payload))
        .expect("decision request is valid");
    decision_request
        .headers_mut()
        .insert(DECISION_SECRET_HEADER, state.decision_secret.clone());

    let response = state
        .decision_client
        .request(decision_request)
        .await
        .map_err(|error| {
            Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                &format!("proxy decision service error: {error}"),
            ))
        })?;
    if !response.status().is_success() {
        let (mut parts, response_body) = response.into_parts();
        remove_connection_headers(&mut parts.headers);
        remove_internal_headers(&mut parts.headers);
        let body = to_bytes(Body::new(response_body), MAX_DECISION_BODY_BYTES)
            .await
            .map_err(|error| {
                Box::new(error_response(
                    StatusCode::BAD_GATEWAY,
                    &format!("proxy decision response error: {error}"),
                ))
            })?;
        return Err(Box::new(Response::from_parts(parts, Body::from(body))));
    }

    let upstream = response
        .headers()
        .get(UPSTREAM_URL_HEADER)
        .ok_or_else(|| {
            Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                "proxy decision omitted its upstream URL",
            ))
        })?
        .to_str()
        .map_err(|_| {
            Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                "proxy decision returned a non-ASCII upstream URL",
            ))
        })?
        .parse::<Uri>()
        .map_err(|error| {
            Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                &format!("proxy decision returned an invalid upstream URL: {error}"),
            ))
        })?;
    if !matches!(upstream.scheme_str(), Some("http" | "https")) || upstream.authority().is_none() {
        return Err(Box::new(error_response(
            StatusCode::BAD_GATEWAY,
            "proxy decision returned a non-HTTP upstream URL",
        )));
    }
    Ok(ProxyDecision {
        upstream,
        proxy_prefix: response.headers().get(PROXY_PREFIX_HEADER).cloned(),
        upstream_authorization: response
            .headers()
            .get(UPSTREAM_AUTHORIZATION_HEADER)
            .cloned(),
        timeout: response
            .headers()
            .get(PROXY_TIMEOUT_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(Duration::from_secs_f64)
            .unwrap_or(DEFAULT_PROXY_TIMEOUT),
    })
}

fn prepare_proxy_request(
    headers: &mut HeaderMap,
    proxy_prefix: Option<HeaderValue>,
    upstream_authorization: Option<HeaderValue>,
) {
    let forwarded_host = headers
        .get(&X_FORWARDED_HOST)
        .cloned()
        .or_else(|| headers.get(header::HOST).cloned());
    remove_connection_headers(headers);
    headers.remove(header::AUTHORIZATION);
    headers.remove(header::COOKIE);
    headers.remove(IAP_ASSERTION_HEADER);
    headers.remove(header::HOST);
    remove_internal_headers(headers);

    if let Some(host) = forwarded_host {
        headers.insert(X_FORWARDED_HOST, host);
    }
    if !headers.contains_key(&X_FORWARDED_PROTO) {
        headers.insert(X_FORWARDED_PROTO, HeaderValue::from_static("http"));
    }
    if let Some(prefix) = proxy_prefix {
        if prefix.is_empty() {
            headers.remove(&X_FORWARDED_PREFIX);
        } else {
            headers.insert(X_FORWARDED_PREFIX, prefix);
        }
    }
    if let Some(authorization) = upstream_authorization {
        headers.insert(header::AUTHORIZATION, authorization);
    }
}

fn prepare_proxy_response(
    headers: &mut HeaderMap,
    upstream: &Uri,
    proxy_prefix: Option<&HeaderValue>,
) {
    remove_connection_headers(headers);
    headers.remove(header::SET_COOKIE);
    let prefix = proxy_prefix
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    for name in [
        header::LOCATION,
        HeaderName::from_static("content-location"),
    ] {
        let Some(value) = headers.get(&name).and_then(|value| value.to_str().ok()) else {
            continue;
        };
        if let Some(rewritten) = rewrite_location(value, upstream, prefix) {
            if let Ok(rewritten) = HeaderValue::from_str(&rewritten) {
                headers.insert(name, rewritten);
            }
        }
    }
}

fn rewrite_location(location: &str, upstream: &Uri, proxy_prefix: &str) -> Option<String> {
    if location.starts_with('/') && !location.starts_with("//") {
        return Some(format!("{proxy_prefix}{location}"));
    }
    let protocol_relative = location.starts_with("//");
    let parsed = if protocol_relative {
        url::Url::parse(&format!("{}:{location}", upstream.scheme_str()?)).ok()?
    } else {
        url::Url::parse(location).ok()?
    };
    let upstream_authority = upstream.authority()?.as_str();
    let parsed_authority = match parsed.port() {
        Some(port) => format!("{}:{port}", parsed.host_str()?),
        None => parsed.host_str()?.to_string(),
    };
    if parsed_authority != upstream_authority
        || (!protocol_relative && Some(parsed.scheme()) != upstream.scheme_str())
    {
        return None;
    }
    let mut rewritten = format!("{proxy_prefix}{}", parsed.path());
    if let Some(query) = parsed.query() {
        rewritten.push('?');
        rewritten.push_str(query);
    }
    if let Some(fragment) = parsed.fragment() {
        rewritten.push('#');
        rewritten.push_str(fragment);
    }
    Some(rewritten)
}

fn prepare_controller_request(headers: &mut HeaderMap) {
    remove_connection_headers(headers);
    headers.remove(header::HOST);
    remove_internal_headers(headers);
}

fn parse_proxy_route(uri: &Uri) -> Result<ProxyRoute, Box<Response<Body>>> {
    let remainder = uri.path().strip_prefix(PROXY_PATH_PREFIX).ok_or_else(|| {
        Box::new(error_response(
            StatusCode::BAD_REQUEST,
            "invalid proxy route",
        ))
    })?;
    let (token, endpoint_and_path, token_prefix) =
        if let Some(token_route) = remainder.strip_prefix("t/") {
            let (token, rest) = token_route.split_once('/').ok_or_else(|| {
                Box::new(error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid token proxy route",
                ))
            })?;
            (
                Some(decode_path_segment(token)?),
                rest,
                format!("/proxy/t/{token}"),
            )
        } else {
            (None, remainder, "/proxy".to_string())
        };
    let (encoded_name, sub_path, redirect) =
        if let Some((name, path)) = endpoint_and_path.split_once('/') {
            (name, path, false)
        } else {
            (endpoint_and_path, "", true)
        };
    if encoded_name.is_empty() {
        return Err(Box::new(error_response(
            StatusCode::BAD_REQUEST,
            "proxy endpoint name is empty",
        )));
    }
    let encoded_name = decode_path_segment(encoded_name)?;
    Ok(ProxyRoute {
        proxy_prefix: format!("{token_prefix}/{encoded_name}"),
        encoded_name,
        sub_path: sub_path.to_string(),
        token,
        redirect,
    })
}

fn proxy_subdomain(headers: &HeaderMap) -> Option<String> {
    let host = headers
        .get(&X_FORWARDED_HOST)
        .or_else(|| headers.get(header::HOST))?
        .to_str()
        .ok()?
        .split(',')
        .next()?
        .split(':')
        .next()?
        .trim()
        .to_ascii_lowercase();
    let labels = host.split('.').collect::<Vec<_>>();
    let proxy_index = labels.iter().position(|label| *label == "proxy")?;
    (proxy_index > 0).then(|| labels[..proxy_index].join("."))
}

fn subdomain_proxy_route(headers: &HeaderMap, uri: &Uri) -> Option<ProxyRoute> {
    let encoded_name = proxy_subdomain(headers)?;
    Some(ProxyRoute {
        encoded_name,
        sub_path: uri.path().trim_start_matches('/').to_string(),
        token: None,
        proxy_prefix: String::new(),
        redirect: false,
    })
}

fn decode_path_segment(value: &str) -> Result<String, Box<Response<Body>>> {
    percent_decode_str(value)
        .decode_utf8()
        .map(|value| value.into_owned())
        .map_err(|_| {
            Box::new(error_response(
                StatusCode::BAD_REQUEST,
                "proxy route is not valid UTF-8",
            ))
        })
}

fn method_allowed(method: &Method) -> bool {
    PROXY_METHODS.contains(&method.as_str())
}

fn proxy_route_for_request(
    headers: &HeaderMap,
    uri: &Uri,
) -> Result<Option<ProxyRoute>, Box<Response<Body>>> {
    if uri.path().starts_with(PROXY_PATH_PREFIX) {
        return parse_proxy_route(uri).map(Some);
    }
    Ok(subdomain_proxy_route(headers, uri))
}

async fn verified_identity(
    state: &AppState,
    headers: &HeaderMap,
    route: &ProxyRoute,
    peer: SocketAddr,
    direct_connection: bool,
) -> Result<auth::VerifiedIdentity, Box<Response<Body>>> {
    match state
        .verifier
        .verify_request(
            headers,
            route.token.as_deref(),
            peer.ip(),
            direct_connection,
        )
        .await
    {
        VerifyOutcome::Verified(identity) => Ok(identity),
        VerifyOutcome::Anonymous => Ok(auth::VerifiedIdentity {
            endpoint: None,
            federation_peer: None,
            expires_at: u64::MAX,
        }),
        VerifyOutcome::Invalid => Err(Box::new(error_response(
            StatusCode::UNAUTHORIZED,
            "authentication required",
        ))),
    }
}

fn authorize_mapping(
    mapping: &EndpointMapping,
    identity: &auth::VerifiedIdentity,
) -> Result<(), Box<Response<Body>>> {
    if mapping.link_access {
        if identity
            .endpoint
            .as_ref()
            .is_some_and(|endpoint| endpoint != &mapping.name)
        {
            return Err(Box::new(error_response(
                StatusCode::FORBIDDEN,
                "token not valid for this endpoint",
            )));
        }
    } else if identity.endpoint.is_some() {
        return Err(Box::new(error_response(
            StatusCode::FORBIDDEN,
            "endpoint-scoped token cannot access this endpoint",
        )));
    }
    Ok(())
}

fn redirect_decision(route: &ProxyRoute, query: Option<&str>) -> NativeDecision {
    let query = query.map(|value| format!("?{value}")).unwrap_or_default();
    let location = format!("{}/{query}", route.proxy_prefix);
    let response = Response::builder()
        .status(StatusCode::TEMPORARY_REDIRECT)
        .header(header::LOCATION, location)
        .body(Body::empty())
        .expect("proxy redirect is valid");
    NativeDecision::Response(Box::new(response))
}

fn federation_decision(
    direction: FederationDirection,
    route: ProxyRoute,
    mapping: EndpointMapping,
    peer_id: String,
    query: Option<&str>,
    local_upstream: Option<String>,
) -> NativeDecision {
    NativeDecision::Federation(FederationDecision {
        direction,
        encoded_name: route.encoded_name,
        sub_path: route.sub_path,
        query: query.unwrap_or_default().to_string(),
        proxy_prefix: route.proxy_prefix,
        peer_id,
        task_id: mapping.task_id,
        local_upstream,
        timeout_seconds: mapping.timeout_seconds,
    })
}

fn local_proxy_decision(
    route: ProxyRoute,
    mapping: EndpointMapping,
    query: Option<&str>,
) -> NativeDecision {
    let upstream = match endpoint_uri(&mapping.address, &route.sub_path, query) {
        Ok(upstream) => upstream,
        Err(error) => {
            return NativeDecision::Response(Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                &error,
            )))
        }
    };
    let proxy_prefix = match HeaderValue::from_str(&route.proxy_prefix) {
        Ok(proxy_prefix) => proxy_prefix,
        Err(error) => {
            return NativeDecision::Response(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid proxy prefix: {error}"),
            )))
        }
    };
    NativeDecision::Proxy(ProxyDecision {
        upstream,
        proxy_prefix: Some(proxy_prefix),
        upstream_authorization: None,
        timeout: mapping
            .timeout_seconds
            .filter(|timeout| timeout.is_finite() && *timeout > 0.0)
            .map(Duration::from_secs_f64)
            .unwrap_or(DEFAULT_PROXY_TIMEOUT),
    })
}

fn mapping_decision(
    route: ProxyRoute,
    mapping: EndpointMapping,
    identity: auth::VerifiedIdentity,
    query: Option<&str>,
) -> NativeDecision {
    if let Some(peer_id) = identity.federation_peer {
        if mapping.peer_id.is_some() {
            return NativeDecision::Response(Box::new(error_response(
                StatusCode::FORBIDDEN,
                "federation peers cannot traverse a mirrored endpoint",
            )));
        }
        let local_upstream = match endpoint_uri(&mapping.address, &route.sub_path, query) {
            Ok(upstream) => Some(upstream.to_string()),
            Err(error) => {
                return NativeDecision::Response(Box::new(error_response(
                    StatusCode::BAD_GATEWAY,
                    &error,
                )))
            }
        };
        return federation_decision(
            FederationDirection::Inbound,
            route,
            mapping,
            peer_id,
            query,
            local_upstream,
        );
    }
    if let Err(response) = authorize_mapping(&mapping, &identity) {
        return NativeDecision::Response(response);
    }
    if route.redirect {
        return redirect_decision(&route, query);
    }
    if let Some(peer_id) = mapping.peer_id.clone() {
        if route.proxy_prefix.is_empty() {
            return NativeDecision::Response(Box::new(error_response(
                StatusCode::BAD_GATEWAY,
                "federated endpoints require a path-style proxy URL",
            )));
        }
        return federation_decision(
            FederationDirection::Outbound,
            route,
            mapping,
            peer_id,
            query,
            None,
        );
    }
    local_proxy_decision(route, mapping, query)
}

async fn native_decision(
    state: &AppState,
    request_headers: &HeaderMap,
    request_method: &Method,
    request_uri: &Uri,
    peer: SocketAddr,
    direct_connection: bool,
) -> NativeDecision {
    if !state.control.is_ready() {
        return if request_uri.path().starts_with(PROXY_PATH_PREFIX)
            || proxy_subdomain(request_headers).is_some()
        {
            NativeDecision::Response(Box::new(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "endpoint registry unavailable",
            )))
        } else {
            NativeDecision::ControllerProxy
        };
    }
    if !method_allowed(request_method) {
        return NativeDecision::Response(Box::new(error_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "proxy method not allowed",
        )));
    }
    let route = match proxy_route_for_request(request_headers, request_uri) {
        Ok(Some(route)) => route,
        Ok(None) => return NativeDecision::ControllerProxy,
        Err(response) => return NativeDecision::Response(response),
    };
    let resolved = match state.control.resolve(&route.encoded_name) {
        Ok(resolved) => resolved,
        Err(error) => {
            return NativeDecision::Response(Box::new(error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                &error,
            )))
        }
    };
    let identity =
        match verified_identity(state, request_headers, &route, peer, direct_connection).await {
            Ok(identity) => identity,
            Err(response) => return NativeDecision::Response(response),
        };
    let Some(mapping) = resolved else {
        return NativeDecision::Response(Box::new(error_response(
            StatusCode::NOT_FOUND,
            &format!("No endpoint '{}'", route.encoded_name),
        )));
    };
    mapping_decision(route, mapping, identity, request_uri.query())
}

async fn send(
    client: &HttpsClient,
    mut request: Request,
    uri: Uri,
    proxy_prefix: Option<Option<HeaderValue>>,
    upstream_authorization: Option<HeaderValue>,
    timeout: Option<Duration>,
) -> Response<Body> {
    let upstream_uri = uri.clone();
    *request.uri_mut() = uri;
    let is_proxy = proxy_prefix.is_some();
    let response_proxy_prefix = proxy_prefix.as_ref().and_then(Option::as_ref).cloned();
    if let Some(prefix) = proxy_prefix {
        prepare_proxy_request(request.headers_mut(), prefix, upstream_authorization);
    } else {
        prepare_controller_request(request.headers_mut());
    }

    let response = if let Some(timeout) = timeout {
        match tokio::time::timeout(timeout, client.request(request)).await {
            Ok(response) => response,
            Err(_) => {
                return error_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    &format!("upstream timeout after {}s", timeout.as_secs_f64()),
                )
            }
        }
    } else {
        client.request(request).await
    };
    match response {
        Ok(response) => {
            if is_proxy && response.status() == StatusCode::UNAUTHORIZED {
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    "upstream refused the controller (401)",
                );
            }
            let (mut parts, body) = response.into_parts();
            if is_proxy {
                prepare_proxy_response(
                    &mut parts.headers,
                    &upstream_uri,
                    response_proxy_prefix.as_ref(),
                );
            } else {
                remove_connection_headers(&mut parts.headers);
            }
            let body = if let Some(timeout) = timeout {
                let stream = Body::new(body)
                    .into_data_stream()
                    .timeout(timeout)
                    .map(move |item| match item {
                        Ok(Ok(bytes)) => Ok(bytes),
                        Ok(Err(error)) => Err(std::io::Error::other(error)),
                        Err(_) => Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!("upstream stream idle for {}s", timeout.as_secs_f64()),
                        )),
                    });
                Body::from_stream(stream)
            } else {
                Body::new(body)
            };
            Response::from_parts(parts, body)
        }
        Err(error) => error_response(
            StatusCode::BAD_GATEWAY,
            &format!("upstream transport error: {error}"),
        ),
    }
}

async fn ingress(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut request: Request,
) -> Response<Body> {
    let direct_connection = !request.headers().contains_key(&X_FORWARDED_FOR);
    prepare_public_headers(request.headers_mut(), peer);
    if request.uri().path() == DECISION_PATH {
        return error_response(StatusCode::NOT_FOUND, "route not found");
    }

    let proxy_request = request.uri().path().starts_with(PROXY_PATH_PREFIX)
        || proxy_subdomain(request.headers()).is_some();
    if proxy_request {
        match native_decision(
            &state,
            request.headers(),
            request.method(),
            request.uri(),
            peer,
            direct_connection,
        )
        .await
        {
            NativeDecision::Proxy(decision) => {
                return send(
                    &state.upstream_client,
                    request,
                    decision.upstream,
                    Some(decision.proxy_prefix),
                    decision.upstream_authorization,
                    Some(decision.timeout),
                )
                .await;
            }
            NativeDecision::Response(response) => return *response,
            NativeDecision::ControllerProxy => {
                let uri = match controller_uri(&state.controller_url, request.uri()) {
                    Ok(uri) => uri,
                    Err(response) => return *response,
                };
                return send(&state.controller_client, request, uri, None, None, None).await;
            }
            NativeDecision::Federation(federation) => {
                let decision = decision(&state, &federation).await;
                return match decision {
                    Ok(decision) => {
                        send(
                            &state.upstream_client,
                            request,
                            decision.upstream,
                            Some(decision.proxy_prefix),
                            decision.upstream_authorization,
                            Some(decision.timeout),
                        )
                        .await
                    }
                    Err(response) => *response,
                };
            }
        }
    }

    let uri = match controller_uri(&state.controller_url, request.uri()) {
        Ok(uri) => uri,
        Err(response) => return *response,
    };
    send(&state.controller_client, request, uri, None, None, None).await
}

pub fn app(config: ProxyConfig, control: ProxyControl) -> Result<Router, String> {
    let decision_secret = HeaderValue::from_str(&config.decision_secret)
        .map_err(|error| format!("decision secret is not a valid header value: {error}"))?;
    let verifier = NativeVerifier::new(config.auth, Arc::clone(&control.stats))?;
    let state = Arc::new(AppState {
        controller_url: config.controller_url.trim_end_matches('/').to_string(),
        decision_secret,
        decision_client: client(8),
        controller_client: client(64),
        upstream_client: client(0),
        control,
        verifier,
    });
    Ok(Router::new().fallback(any(ingress)).with_state(state))
}

pub async fn serve(
    listener: TcpListener,
    config: ProxyConfig,
    control: ProxyControl,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), String> {
    let app = app(config, control)?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown)
    .await
    .map_err(|error| format!("native proxy server failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(endpoint_id: &str, name: &str) -> EndpointMapping {
        EndpointMapping {
            endpoint_id: endpoint_id.to_string(),
            name: name.to_string(),
            address: "http://127.0.0.1:8080".to_string(),
            link_access: false,
            peer_id: None,
            task_id: Some("/job/task".to_string()),
            timeout_seconds: None,
            lease_deadline_epoch_ms: Some(4_000_000_000_000),
        }
    }

    #[test]
    fn mapping_delta_is_atomic_and_generation_checked() {
        let control = ProxyControl::default();
        control
            .replace_registry(RegistrySnapshot {
                generation: 7,
                endpoints: vec![mapping("a", "/a"), mapping("b", "/b")],
            })
            .unwrap();

        control
            .update_mappings(MappingDelta {
                base_generation: 6,
                next_generation: 7,
                upserts: vec![mapping("c", "/c")],
                deletes: vec!["a".to_string()],
            })
            .unwrap_err();
        assert_eq!(control.stats().unwrap().registry_generation, 7);
        assert_eq!(control.resolve("a").unwrap().unwrap().endpoint_id, "a");
        assert!(control.resolve("c").unwrap().is_none());

        control
            .update_mappings(MappingDelta {
                base_generation: 7,
                next_generation: 8,
                upserts: vec![mapping("c", "/c")],
                deletes: vec!["a".to_string()],
            })
            .unwrap();
        assert_eq!(control.stats().unwrap().registry_generation, 8);
        assert!(control.resolve("a").unwrap().is_none());
        assert_eq!(control.resolve("b").unwrap().unwrap().endpoint_id, "b");
        assert_eq!(control.resolve("c").unwrap().unwrap().endpoint_id, "c");
    }

    #[test]
    fn mapping_delta_rejects_duplicate_ids_without_mutation() {
        let control = ProxyControl::default();
        control
            .replace_registry(RegistrySnapshot {
                generation: 1,
                endpoints: vec![mapping("a", "/a")],
            })
            .unwrap();
        control
            .update_mappings(MappingDelta {
                base_generation: 1,
                next_generation: 2,
                upserts: vec![mapping("a", "/new")],
                deletes: vec!["a".to_string()],
            })
            .unwrap_err();
        assert_eq!(control.stats().unwrap().registry_generation, 1);
        assert_eq!(control.resolve("a").unwrap().unwrap().name, "/a");
    }
}
