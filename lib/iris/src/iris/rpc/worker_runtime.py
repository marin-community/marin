# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect adapters that host and register a native worker daemon."""

from dataclasses import dataclass

import uvicorn
from finelog.client import LogClient
from rigging.auth import BearerTokenInjector, StaticTokenProvider
from rigging.timing import Duration, ExponentialBackoff

from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.worker.control import (
    WorkerController,
    WorkerRegistration,
    WorkerRegistrationResult,
    WorkerServer,
    WorkerTaskProvider,
)
from iris.cluster.worker.worker import WorkerConfig
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.resource_client import ResourceRpcClient
from iris.rpc.worker_codec import worker_metadata_to_proto
from iris.rpc.worker_dashboard import WorkerDashboard
from iris.rpc.worker_service import WorkerServiceImpl


class ConnectWorkerController:
    """Connect-backed implementation of the worker's controller port."""

    def __init__(self, address: str, auth_token: str = "") -> None:
        interceptors: tuple[BearerTokenInjector, ...] = ()
        if auth_token:
            interceptors = (BearerTokenInjector(StaticTokenProvider(auth_token), "authorization"),)
        self._controller = ControllerServiceClientSync(
            address=address,
            timeout_ms=10_000,
            interceptors=interceptors,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )
        self._resources = ResourceRpcClient(
            address,
            timeout_ms=10_000,
            interceptors=interceptors,
        )

    def register(self, request: WorkerRegistration) -> WorkerRegistrationResult:
        response = self._controller.register(
            controller_pb2.Controller.RegisterRequest(
                address=request.address,
                metadata=worker_metadata_to_proto(request.metadata),
                worker_id=request.worker_id,
                slice_id=request.slice_id,
                scale_group=request.scale_group,
            )
        )
        return WorkerRegistrationResult(accepted=response.accepted, worker_id=response.worker_id)

    def resolve_endpoint(self, name: str) -> str:
        endpoints = self._resources.resolve_endpoints(name)
        if not endpoints:
            raise ConnectionError(f"No {name!r} endpoint registered on controller")
        return endpoints[0].address

    def close(self) -> None:
        self._controller.close()
        self._resources.close()


class ConnectWorkerServer:
    """Uvicorn host for the worker's Connect service and dashboard."""

    def __init__(self, *, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None

    def start(self, provider: WorkerTaskProvider, threads: ThreadContainer) -> None:
        dashboard = WorkerDashboard(WorkerServiceImpl(provider), host=self._host, port=self._port)
        server = uvicorn.Server(
            uvicorn.Config(
                dashboard.app,
                host=self._host,
                port=self._port,
                log_level="error",
                log_config=None,
                timeout_keep_alive=120,
            )
        )
        self._server = server
        threads.spawn_server(server, name="worker-server")
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: server.started,
            timeout=Duration.from_seconds(5.0),
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True


@dataclass(frozen=True, slots=True)
class WorkerRpcBindings:
    """Transport adapters injected into one worker process."""

    controller: WorkerController | None
    server: WorkerServer
    log_client: LogClient | None


def worker_rpc_bindings(config: WorkerConfig) -> WorkerRpcBindings:
    """Build the worker's Connect controller, server, and logging adapters."""
    server = ConnectWorkerServer(host=config.host, port=config.port)
    if config.controller_address is None:
        return WorkerRpcBindings(controller=None, server=server, log_client=None)

    controller = ConnectWorkerController(config.controller_address, config.auth_token)
    interceptors: tuple[BearerTokenInjector, ...] = ()
    if config.auth_token:
        interceptors = (BearerTokenInjector(StaticTokenProvider(config.auth_token), "authorization"),)
    log_client = LogClient.connect(
        LOG_SERVER_ENDPOINT_NAME,
        interceptors=interceptors,
        resolver=controller.resolve_endpoint,
    )
    return WorkerRpcBindings(controller=controller, server=server, log_client=log_client)
