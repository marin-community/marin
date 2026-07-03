// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! PyO3 surface for embedding the finelog server in a Python process.
//!
//! Exposes [`EmbeddedServer`]: a self-contained tokio runtime serving the same
//! axum app as the `finelog-server` binary, bound to a local port. Iris's
//! controller uses it as the in-process log-server fallback when no external
//! `/system/log-server` endpoint is configured; callers then talk to it over
//! the normal RPC contract (`LogClient` / proxies), so there is exactly one
//! server implementation.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use finelog::server::{build_app_with_config, AuthPolicy, ServerConfig};
use finelog::store::Store;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// How long `stop()` waits for in-flight requests to drain and the store's
/// per-namespace flush/maintenance tasks to finish before forcing the runtime
/// down. Generous: writes already ack only after L0 persist, so durability does
/// not depend on this bound.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(12);

/// An in-process finelog server.
///
/// Construction opens the store, binds the listener (an ephemeral port when
/// `port=0`), and spawns the axum server on an owned multi-threaded tokio
/// runtime. The server keeps running on the runtime's worker threads until
/// [`EmbeddedServer::stop`] (or drop) triggers graceful shutdown.
#[pyclass]
struct EmbeddedServer {
    runtime: Option<Runtime>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    serve: Option<JoinHandle<()>>,
    addr: SocketAddr,
}

#[pymethods]
impl EmbeddedServer {
    #[new]
    #[pyo3(signature = (log_dir=None, remote_log_dir=String::new(), host=String::from("127.0.0.1"), port=0, debug_admin=false, auth_policy=None))]
    fn new(
        log_dir: Option<String>,
        remote_log_dir: String,
        host: String,
        port: u16,
        debug_admin: bool,
        auth_policy: Option<String>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build tokio runtime: {e}")))?;

        let bind: SocketAddr = format!("{host}:{port}").parse().map_err(|e| {
            PyRuntimeError::new_err(format!("invalid host/port {host}:{port}: {e}"))
        })?;

        // Open the store, bootstrap maintenance, and bind the listener INSIDE
        // the runtime: `Store::new` / `bootstrap_maintenance` spawn tokio tasks
        // and so require an ambient runtime context (the binary gets this from
        // `#[tokio::main]`). Binding here also means the chosen (ephemeral) port
        // is known before returning to Python. The spawned maintenance tasks
        // keep running on the runtime's worker threads after `block_on` returns.
        let (store, app, listener, addr) = runtime.block_on(async {
            let store = Arc::new(
                Store::new(log_dir.map(PathBuf::from), remote_log_dir)
                    .map_err(|e| PyRuntimeError::new_err(format!("failed to open store: {e}")))?,
            );
            store.bootstrap_maintenance();
            // No policy → the private allow-localhost default (loopback only); a
            // JSON layer list configures a global finelog's cidr+jwt stack.
            let auth = match auth_policy.as_deref() {
                Some(json) if !json.trim().is_empty() => AuthPolicy::parse(json)
                    .map_err(|e| PyRuntimeError::new_err(format!("invalid auth_policy: {e}")))?,
                _ => AuthPolicy::allow_localhost(),
            };
            let config = ServerConfig::with_debug_admin(debug_admin).with_auth(auth);
            let app = build_app_with_config(Arc::clone(&store), config);
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("failed to bind {bind}: {e}")))?;
            let addr = listener
                .local_addr()
                .map_err(|e| PyRuntimeError::new_err(format!("failed to read local addr: {e}")))?;
            Ok::<_, PyErr>((store, app, listener, addr))
        })?;

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let serve = runtime.spawn(async move {
            let shutdown = async move {
                let _ = shutdown_rx.await;
            };
            // `into_make_service_with_connect_info` records each connection's peer
            // address so the auth CIDR rule can read it.
            let _ = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(shutdown)
            .await;
            store.shutdown(Duration::from_secs(5)).await;
        });

        Ok(Self {
            runtime: Some(runtime),
            shutdown_tx: Some(shutdown_tx),
            serve: Some(serve),
            addr,
        })
    }

    /// The bound port (the ephemeral port chosen by the OS when `port=0`).
    #[getter]
    fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Base URL of the server, e.g. `http://127.0.0.1:54321`.
    #[getter]
    fn address(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Trigger graceful shutdown and join the server. Idempotent.
    fn stop(&mut self, py: Python<'_>) {
        py.detach(|| self.shutdown());
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type, _exc_value, _traceback))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        py.detach(|| self.shutdown());
        false
    }
}

impl EmbeddedServer {
    fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let serve = self.serve.take();
        if let Some(runtime) = self.runtime.take() {
            if let Some(serve) = serve {
                let _ =
                    runtime.block_on(async { tokio::time::timeout(DRAIN_TIMEOUT, serve).await });
            }
            runtime.shutdown_timeout(Duration::from_secs(2));
        }
    }
}

impl Drop for EmbeddedServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[pymodule]
fn finelog_server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmbeddedServer>()?;
    Ok(())
}
