// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::time::Duration;

use iris_proxy::{
    serve, MappingDelta, NativeAuthConfig, ProxyConfig, ProxyControl, RegistrySnapshot,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const LISTEN_BACKLOG: u32 = 2048;

#[pyclass]
struct NativeProxy {
    runtime: Option<Runtime>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    serve: Option<JoinHandle<Result<(), String>>>,
    address: SocketAddr,
    control: ProxyControl,
}

#[pymethods]
impl NativeProxy {
    #[new]
    #[pyo3(signature = (
        public_host,
        public_port,
        controller_url,
        decision_secret,
        auth_config_json,
        worker_threads=0,
    ))]
    fn new(
        public_host: String,
        public_port: u16,
        controller_url: String,
        decision_secret: String,
        auth_config_json: String,
        worker_threads: usize,
    ) -> PyResult<Self> {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder.enable_all();
        if worker_threads > 0 {
            builder.worker_threads(worker_threads);
        }
        let runtime = builder.build().map_err(|error| {
            PyRuntimeError::new_err(format!("failed to build proxy runtime: {error}"))
        })?;
        let bind: SocketAddr = format!("{public_host}:{public_port}")
            .parse()
            .map_err(|error| {
                PyRuntimeError::new_err(format!("invalid proxy bind address: {error}"))
            })?;
        let (listener, address) = runtime.block_on(async {
            let socket = if bind.is_ipv4() {
                tokio::net::TcpSocket::new_v4()
            } else {
                tokio::net::TcpSocket::new_v6()
            }
            .map_err(|error| {
                PyRuntimeError::new_err(format!("failed to create proxy socket: {error}"))
            })?;
            socket.set_reuseaddr(true).map_err(|error| {
                PyRuntimeError::new_err(format!("failed to configure proxy socket: {error}"))
            })?;
            socket.bind(bind).map_err(|error| {
                PyRuntimeError::new_err(format!("failed to bind {bind}: {error}"))
            })?;
            let listener = socket.listen(LISTEN_BACKLOG).map_err(|error| {
                PyRuntimeError::new_err(format!("failed to listen on {bind}: {error}"))
            })?;
            let address = listener.local_addr().map_err(|error| {
                PyRuntimeError::new_err(format!("failed to inspect proxy listener: {error}"))
            })?;
            Ok::<_, PyErr>((listener, address))
        })?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let control = ProxyControl::default();
        let auth: NativeAuthConfig = serde_json::from_str(&auth_config_json)
            .map_err(|error| PyValueError::new_err(format!("invalid auth config: {error}")))?;
        let serve = runtime.spawn(serve(
            listener,
            ProxyConfig {
                controller_url,
                decision_secret,
                auth,
            },
            control.clone(),
            async move {
                let _ = shutdown_rx.await;
            },
        ));
        Ok(Self {
            runtime: Some(runtime),
            shutdown_tx: Some(shutdown_tx),
            serve: Some(serve),
            address,
            control,
        })
    }

    #[getter]
    fn port(&self) -> u16 {
        self.address.port()
    }

    #[getter]
    fn address(&self) -> String {
        format!("http://{}", self.address)
    }

    #[getter]
    fn stats_json(&self) -> PyResult<String> {
        let stats = self.control.stats().map_err(PyRuntimeError::new_err)?;
        serde_json::to_string(&stats).map_err(|error| {
            PyRuntimeError::new_err(format!("failed to encode proxy stats: {error}"))
        })
    }

    fn replace_registry(&self, snapshot_json: &str) -> PyResult<()> {
        let snapshot: RegistrySnapshot = serde_json::from_str(snapshot_json).map_err(|error| {
            PyValueError::new_err(format!("invalid registry snapshot: {error}"))
        })?;
        self.control
            .replace_registry(snapshot)
            .map_err(PyValueError::new_err)
    }

    fn pause_registry(&self) {
        self.control.pause_registry();
    }

    fn update_mappings(&self, delta_json: &str) -> PyResult<()> {
        let delta: MappingDelta = serde_json::from_str(delta_json)
            .map_err(|error| PyValueError::new_err(format!("invalid mapping delta: {error}")))?;
        self.control
            .update_mappings(delta)
            .map_err(PyValueError::new_err)
    }

    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.shutdown())
            .map_err(PyRuntimeError::new_err)
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
    ) -> PyResult<bool> {
        py.detach(|| self.shutdown())
            .map_err(PyRuntimeError::new_err)?;
        Ok(false)
    }
}

impl NativeProxy {
    fn shutdown(&mut self) -> Result<(), String> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        let serve = self.serve.take();
        if let Some(runtime) = self.runtime.take() {
            let result = if let Some(serve) = serve {
                match runtime.block_on(async { tokio::time::timeout(DRAIN_TIMEOUT, serve).await }) {
                    Ok(Ok(result)) => result,
                    Ok(Err(error)) => Err(format!("native proxy task failed: {error}")),
                    Err(_) => Err(format!(
                        "native proxy did not drain within {}s",
                        DRAIN_TIMEOUT.as_secs()
                    )),
                }
            } else {
                Ok(())
            };
            runtime.shutdown_timeout(Duration::from_secs(2));
            result?;
        }
        Ok(())
    }
}

impl Drop for NativeProxy {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            tracing::error!("{error}");
        }
    }
}

#[pymodule]
fn iris_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeProxy>()?;
    module.add("PROXY_DECISION_PATH", iris_proxy::DECISION_PATH)?;
    module.add("DECISION_SECRET_HEADER", iris_proxy::DECISION_SECRET_HEADER)?;
    module.add("UPSTREAM_URL_HEADER", iris_proxy::UPSTREAM_URL_HEADER)?;
    module.add(
        "UPSTREAM_AUTHORIZATION_HEADER",
        iris_proxy::UPSTREAM_AUTHORIZATION_HEADER,
    )?;
    module.add("PROXY_PREFIX_HEADER", iris_proxy::PROXY_PREFIX_HEADER)?;
    module.add("PROXY_TIMEOUT_HEADER", iris_proxy::PROXY_TIMEOUT_HEADER)?;
    module.add(
        "DEFAULT_PROXY_TIMEOUT_SECONDS",
        iris_proxy::DEFAULT_PROXY_TIMEOUT_SECONDS,
    )?;
    module.add("PROXY_METHODS", iris_proxy::PROXY_METHODS.to_vec())?;
    Ok(())
}
