// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixtures for tests that drive the server over a real socket.
//!
//! A handler's contract is what a client observes across the wire — the auth stack,
//! the codec, and the dispatcher all sit between the two — so these tests speak
//! Connect to an in-process server rather than calling handlers directly.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use connectrpc::client::{ClientBody, ClientConfig, ServiceTransport};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;

use crate::proto::finelog::logging::LogServiceClient;
use crate::proto::finelog::stats::StatsServiceClient;
use crate::server::auth::AuthPolicy;
use crate::server::forwarding::forward_client_config;
use crate::server::{build_app_with_config, ServerConfig};
use crate::store::Store;
use crate::test_support::unique_dir;

/// Fixed Ed25519 test keypairs (PKCS8 private + SPKI public PEM), generated once with
/// `openssl genpkey -algorithm ed25519`. `A` is the keypair a verifier is configured to
/// trust; `B` is a second trusted one, for tests that must tell two clusters apart;
/// `UNTRUSTED` is a keypair no verifier holds the public half of.
pub const PRIV_A: &str = "-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIMD3AX82bVpf0SoIIVssOXbemV9PNWzwtiJhuA61/AeG\n-----END PRIVATE KEY-----\n";
pub const PUB_A: &str = "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAqwwvfFvyRQ+8Dhh0li8h2HtCT4yP40s0pzBwwSAkK5s=\n-----END PUBLIC KEY-----\n";
pub const PRIV_B: &str = "-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIBmJ8qWzlhzFbTWMHs8snOv+rGewn4IUj+ZNPMKTdCtn\n-----END PRIVATE KEY-----\n";
pub const PUB_B: &str = "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEANlmOBl+nfp+EBodU+vEmzW1UBGhLsN2MC2YjSBjnBGg=\n-----END PUBLIC KEY-----\n";
pub const PRIV_UNTRUSTED: &str = "-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIILe2LqkmmgNBtRgBZNAy/OdPM1jlvKsAkD2/0PkHTty\n-----END PRIVATE KEY-----\n";

/// A plaintext Connect transport. Production speaks TLS; everything above the
/// transport is the same code.
pub type TestTransport = ServiceTransport<HyperClient<HttpConnector, ClientBody>>;

#[derive(Default)]
pub struct RequestStats {
    total: AtomicUsize,
    zstd: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl RequestStats {
    fn begin(&self, request: &axum::extract::Request) -> bool {
        if request.method() != axum::http::Method::POST {
            return false;
        }
        self.total.fetch_add(1, Ordering::SeqCst);
        if request
            .headers()
            .get(axum::http::header::CONTENT_ENCODING)
            .is_some_and(|encoding| encoding == "zstd")
        {
            self.zstd.fetch_add(1, Ordering::SeqCst);
        }
        let in_flight = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(in_flight, Ordering::SeqCst);
        true
    }

    fn finish(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn total(&self) -> usize {
        self.total.load(Ordering::SeqCst)
    }

    pub fn zstd_requests(&self) -> usize {
        self.zstd.load(Ordering::SeqCst)
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }
}

/// A disk-backed store with its flush/maintenance tasks running, so a push becomes
/// query-visible exactly as it does in production.
pub fn disk_store(tag: &str) -> Arc<Store> {
    let store = Arc::new(
        Store::new(
            Some(unique_dir(tag)),
            String::new(),
            crate::query::index_cache::DEFAULT_INDEX_CACHE_MB,
        )
        .unwrap(),
    );
    store.bootstrap_maintenance();
    store
}

fn client_config(addr: SocketAddr) -> (TestTransport, ClientConfig) {
    let uri: http::Uri = format!("http://{addr}").parse().unwrap();
    let transport = ServiceTransport::new(
        HyperClient::builder(TokioExecutor::new()).build(HttpConnector::new()),
    );
    // Match production forwarding clients so integration tests exercise the
    // server's level-1 zstd request path for large WriteRows payloads.
    let config = forward_client_config(uri);
    (transport, config)
}

pub fn client(addr: SocketAddr) -> LogServiceClient<TestTransport> {
    let (transport, config) = client_config(addr);
    LogServiceClient::new(transport, config)
}

pub fn stats_client(addr: SocketAddr) -> StatsServiceClient<TestTransport> {
    let (transport, config) = client_config(addr);
    StatsServiceClient::new(transport, config)
}

/// Serve `store` on an ephemeral loopback port under `policy`, counting the RPC
/// requests that reach it. Returns the address and that counter.
pub async fn serve(store: Arc<Store>, policy: AuthPolicy) -> (SocketAddr, Arc<RequestStats>) {
    let requests = Arc::new(RequestStats::default());
    let counted = Arc::clone(&requests);
    let app = build_app_with_config(store, ServerConfig::default().with_auth(policy)).layer(
        axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| {
                let counted = Arc::clone(&counted);
                async move {
                    let observed = counted.begin(&req);
                    let response = next.run(req).await;
                    if observed {
                        counted.finish();
                    }
                    response
                }
            },
        ),
    );
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
    (addr, requests)
}

/// A hub that answers every RPC with `invalid_argument`, as the real one does for a
/// structurally malformed entry (an empty key). Retrying such a request can never
/// succeed, so this fixture lets a test prove the forwarder skips the batch instead of
/// livelocking on it. Returns the address and a count of the requests it served.
pub async fn serve_rejecting() -> (SocketAddr, Arc<RequestStats>) {
    let requests = Arc::new(RequestStats::default());
    let counted = Arc::clone(&requests);
    let app = axum::Router::new().fallback(axum::routing::any(
        move |request: axum::extract::Request| {
            let counted = Arc::clone(&counted);
            async move {
                let observed = counted.begin(&request);
                let error = connectrpc::ConnectError::new(
                    connectrpc::ErrorCode::InvalidArgument,
                    "empty key",
                );
                let response = (
                    axum::http::StatusCode::BAD_REQUEST,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    error.to_json(),
                );
                if observed {
                    counted.finish();
                }
                response
            }
        },
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, requests)
}
