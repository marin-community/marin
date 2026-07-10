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
use crate::server::auth::AuthPolicy;
use crate::server::{build_app_with_config, ServerConfig, MAX_MESSAGE_BYTES};
use crate::store::Store;

/// A plaintext Connect transport. Production speaks TLS; everything above the
/// transport is the same code.
pub type TestTransport = ServiceTransport<HyperClient<HttpConnector, ClientBody>>;

/// A fresh directory under the system temp dir, unique per call.
pub fn unique_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "finelog_{tag}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A disk-backed store with its flush/maintenance tasks running, so a push becomes
/// query-visible exactly as it does in production.
pub fn disk_store(tag: &str) -> Arc<Store> {
    let store = Arc::new(Store::new(Some(unique_dir(tag)), String::new()).unwrap());
    store.bootstrap_maintenance();
    store
}

pub fn client(addr: SocketAddr) -> LogServiceClient<TestTransport> {
    let uri: http::Uri = format!("http://{addr}").parse().unwrap();
    let transport = ServiceTransport::new(
        HyperClient::builder(TokioExecutor::new()).build(HttpConnector::new()),
    );
    LogServiceClient::new(
        transport,
        ClientConfig::new(uri)
            .proto()
            .with_default_max_message_size(MAX_MESSAGE_BYTES),
    )
}

/// Serve `store` on an ephemeral loopback port under `policy`, counting the RPC
/// requests that reach it. Returns the address and that counter.
pub async fn serve(store: Arc<Store>, policy: AuthPolicy) -> (SocketAddr, Arc<AtomicUsize>) {
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&requests);
    let app = build_app_with_config(store, ServerConfig::default().with_auth(policy)).layer(
        axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| {
                let counted = Arc::clone(&counted);
                async move {
                    if req.method() == axum::http::Method::POST {
                        counted.fetch_add(1, Ordering::SeqCst);
                    }
                    next.run(req).await
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
