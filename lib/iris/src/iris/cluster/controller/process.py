# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller process host for RPC, HTTP, dashboard, and native proxy surfaces."""

import asyncio
import json
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import uvicorn
from rigging import telemetry
from rigging.server_auth import IAP_ISSUER, IAP_PUBLIC_KEYS_URL
from rigging.timing import Duration, ExponentialBackoff

from iris.cluster.controller.auth import (
    CONTROL_PLANE_AUDIENCE,
    DEFAULT_USER_ROLE,
    ENDPOINT_TOKEN_SCOPE,
    FEDERATION_AUDIENCE,
    NATIVE_PROXY_JWT_CACHE_CAPACITY,
    NATIVE_PROXY_JWT_CACHE_TTL_SECONDS,
    NATIVE_PROXY_JWT_LEEWAY_SECONDS,
    PROXY_PLANE_AUDIENCE,
    NativeProxyAuthConfig,
    NativeProxyAuthMode,
)
from iris.cluster.controller.controller import Controller
from iris.cluster.controller.endpoint_registry import (
    EndpointRegistry,
    ProxyMappingDelta,
    ProxyRegistryReset,
)
from iris.cluster.controller.native_proxy import NativeProxy, NativeProxyStats
from iris.cluster.controller.native_proxy_metrics import (
    NativeProxyTelemetry,
    install_native_proxy_metrics,
    uninstall_native_proxy_metrics,
)
from iris.cluster.controller.runtime import ControllerConfig, ControllerRuntime
from iris.cluster.endpoints import TELEMETRY_ENDPOINT_PATH
from iris.cluster.platforms.types import resolve_external_host
from iris.managed_thread import ManagedThread, ThreadContainer
from iris.rpc.dashboard import ControllerDashboard
from iris.rpc.endpoint_service import EndpointServiceImpl
from iris.rpc.legacy.controller_service import LegacyControllerService
from iris.rpc.resource_service import ResourceServiceImpl

logger = logging.getLogger(__name__)

# Connect RPC handlers use the event loop's default executor. A wider pool keeps
# slow launch handlers from starving worker heartbeats on small controller VMs.
_RPC_HANDLER_THREADS = 64
_CONTROLLER_KEEPALIVE = 120
_PRIVATE_CONTROLLER_HOST = "127.0.0.1"


def _install_rpc_executor(server: uvicorn.Server, *, max_workers: int) -> None:
    """Run Uvicorn with a named, explicitly sized default executor."""

    def run_with_executor(sockets: list[socket.socket] | None = None) -> None:
        with asyncio.Runner(loop_factory=server.config.get_loop_factory()) as runner:
            runner.get_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rpc-handler")
            )
            runner.run(server.serve(sockets=sockets))

    server.run = run_with_executor


class ControllerProcess:
    """Host one controller application around a control runtime.

    The runtime owns durable state and maintenance/control loops. This host owns
    the ASGI server, RPC/resource adapters, dashboard, native listener, and their
    externally visible address.
    """

    def __init__(
        self,
        *,
        config: ControllerConfig,
        runtime: ControllerRuntime,
        controller: Controller,
        controller_service: LegacyControllerService,
        resource_service: ResourceServiceImpl,
        endpoint_registry: EndpointRegistry,
        endpoint_service: EndpointServiceImpl,
        dashboard: ControllerDashboard,
        external_auth_allows_anonymous: bool,
        threads: ThreadContainer,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._controller = controller
        self._controller_service = controller_service
        self._resource_service = resource_service
        self._endpoint_registry = endpoint_registry
        self._endpoint_service = endpoint_service
        self._dashboard = dashboard
        self._external_auth_allows_anonymous = external_auth_allows_anonymous
        self._threads = threads
        self._server: uvicorn.Server | None = None
        self._server_thread: ManagedThread | None = None
        self._native_proxy: NativeProxy | None = None
        self._native_proxy_metrics: NativeProxyTelemetry | None = None
        self._started = False
        self._stopped = False
        endpoint_registry.subscribe_proxy_updates(self._publish_native_proxy_update)

    @property
    def runtime(self) -> ControllerRuntime:
        return self._runtime

    @property
    def controller(self) -> Controller:
        return self._controller

    @property
    def controller_service(self) -> LegacyControllerService:
        return self._controller_service

    @property
    def resource_service(self) -> ResourceServiceImpl:
        return self._resource_service

    @property
    def endpoint_service(self) -> EndpointServiceImpl:
        return self._endpoint_service

    @property
    def dashboard(self) -> ControllerDashboard:
        return self._dashboard

    def start(self) -> None:
        """Start the private ASGI server, control runtime, and public listener."""
        if self._started:
            return
        self._started = True

        server_config = uvicorn.Config(
            self._dashboard.app,
            host=_PRIVATE_CONTROLLER_HOST,
            port=0,
            log_level="warning",
            log_config=None,
            timeout_keep_alive=_CONTROLLER_KEEPALIVE,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        self._server = uvicorn.Server(server_config)
        _install_rpc_executor(self._server, max_workers=_RPC_HANDLER_THREADS)
        self._server_thread = self._threads.spawn_server(self._server, name="controller-server")

        # Register endpoints before the first autoscale tick can provision a
        # worker that immediately resolves /system/log-server.
        for name, url in self._config.endpoints.items():
            self._endpoint_registry.register_system_endpoint(name, url)
            logger.info("Registered system endpoint %s -> %s", name, url)
        self._endpoint_registry.register_system_endpoint("/system/log-server", self._runtime.log_service_address)
        self._runtime.start()

        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server is not None and self._server.started,
            timeout=Duration.from_seconds(5.0),
        )
        assert self._server is not None
        assert self._server.servers
        private_port = self._server.servers[0].sockets[0].getsockname()[1]
        self._native_proxy = NativeProxy(
            self._config.host,
            self._config.port,
            f"http://{_PRIVATE_CONTROLLER_HOST}:{private_port}",
            self._dashboard.proxy_decision_secret,
            json.dumps(asdict(self._native_proxy_auth_config())),
        )
        telemetry.configure(
            endpoint=self._runtime.log_service_address.rstrip("/") + TELEMETRY_ENDPOINT_PATH,
            service="iris-controller",
            attributes={"role": "controller"},
        )
        self._native_proxy_metrics = install_native_proxy_metrics(self._native_proxy)
        self._replace_native_proxy_registry()

    def stop(self) -> None:
        """Stop public ingress, control loops, and the private server."""
        if self._stopped:
            return
        self._stopped = True
        if self._native_proxy_metrics is not None and self._native_proxy is not None:
            uninstall_native_proxy_metrics(self._native_proxy)
            self._native_proxy_metrics = None
        if self._native_proxy is not None:
            self._native_proxy.stop()
        if self._server_thread is not None:
            self._server_thread.stop()
            self._server_thread.join(timeout=Duration.from_seconds(5))
        self._runtime.stop()
        self._threads.stop()

    def _publish_native_proxy_update(self, update: ProxyMappingDelta | ProxyRegistryReset) -> None:
        if self._native_proxy is None:
            return
        if isinstance(update, ProxyRegistryReset):
            self._recover_native_proxy_registry()
            return
        try:
            self._native_proxy.update_mappings(json.dumps(asdict(update)))
        except ValueError:
            logger.exception(
                "Native proxy rejected endpoint mapping generation %d -> %d; replacing registry",
                update.base_generation,
                update.next_generation,
            )
            self._recover_native_proxy_registry()

    def _recover_native_proxy_registry(self) -> None:
        assert self._native_proxy is not None
        try:
            self._replace_native_proxy_registry()
        except ValueError:
            logger.exception("Native proxy registry replacement failed; pausing native routing")
            self._native_proxy.pause_registry()

    def _replace_native_proxy_registry(self) -> None:
        if self._native_proxy is None:
            return
        self._native_proxy.pause_registry()
        snapshot = self._endpoint_registry.proxy_registry_snapshot()
        self._native_proxy.replace_registry(json.dumps(asdict(snapshot)))

    def _native_proxy_auth_config(self) -> NativeProxyAuthConfig:
        auth = self._config.auth
        if auth is None or auth.provider is None:
            mode = NativeProxyAuthMode.PERMISSIVE
        elif self._external_auth_allows_anonymous:
            mode = NativeProxyAuthMode.OPTIONAL
        else:
            mode = NativeProxyAuthMode.ENFORCING
        if auth is not None and auth.jwt_manager is not None:
            issuers, jwks = auth.jwt_manager.native_proxy_verification_material()
        else:
            issuers, jwks = (), {"keys": []}
        return NativeProxyAuthConfig(
            mode=mode,
            issuers=issuers,
            jwks=jwks,
            leeway_seconds=NATIVE_PROXY_JWT_LEEWAY_SECONDS,
            cache_capacity=NATIVE_PROXY_JWT_CACHE_CAPACITY,
            cache_ttl_seconds=NATIVE_PROXY_JWT_CACHE_TTL_SECONDS,
            trusted_cidrs=auth.trusted_cidrs if auth is not None else (),
            control_audience=CONTROL_PLANE_AUDIENCE,
            proxy_audience=PROXY_PLANE_AUDIENCE,
            proxy_scope=ENDPOINT_TOKEN_SCOPE,
            federation_audience=FEDERATION_AUDIENCE,
            iap_public_keys_url=IAP_PUBLIC_KEYS_URL,
            iap_issuer=IAP_ISSUER,
            iap_audience=auth.iap_audience if auth is not None else None,
            federation_keys=auth.federation_keys if auth is not None else {},
            admin_users=tuple(sorted(auth.role_policy.admins)) if auth is not None and auth.role_policy else (),
            default_user_role=(
                auth.role_policy.default_role if auth is not None and auth.role_policy else DEFAULT_USER_ROLE
            ),
        )

    @property
    def port(self) -> int:
        """Actual bound port, including an automatically assigned port."""
        return self._native_proxy.port if self._native_proxy is not None else self._config.port

    @property
    def native_proxy_stats(self) -> NativeProxyStats | None:
        if self._native_proxy is None:
            return None
        return NativeProxyStats.from_json(self._native_proxy.stats_json)

    @property
    def external_host(self) -> str:
        return resolve_external_host(self._config.host)

    @property
    def url(self) -> str:
        return f"http://{self.external_host}:{self.port}"
