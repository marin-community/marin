# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import TracebackType
from typing import cast

import requests
from fray.actor import ActorFuture, ActorGroup, ActorHandle, current_actor
from fray.client import Client
from fray.types import ResourceConfig

from marin.inference.types import ModelDeployment, OpenAIEndpoint, RunningModel

logger = logging.getLogger(__name__)

MARIN_REQUEST_ID_HEADER = "X-Marin-Inference-Request-Id"
JSON_CONTENT_TYPE = "application/json"
DEFAULT_LEASE_TIMEOUT = 30.0
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_WORKER_LEASE_WAIT_TIMEOUT = 1.0
DEFAULT_WORKER_READY_TIMEOUT = 900.0
DEFAULT_CLEANUP_TIMEOUT = 10.0


class OpenAIEndpointKind(StrEnum):
    """OpenAI-compatible endpoint routed by the Iris inference proxy."""

    COMPLETIONS = "completions"
    CHAT_COMPLETIONS = "chat_completions"

    @property
    def http_path(self) -> str:
        match self:
            case OpenAIEndpointKind.COMPLETIONS:
                return "/v1/completions"
            case OpenAIEndpointKind.CHAT_COMPLETIONS:
                return "/v1/chat/completions"

    @property
    def api_path(self) -> str:
        match self:
            case OpenAIEndpointKind.COMPLETIONS:
                return "completions"
            case OpenAIEndpointKind.CHAT_COMPLETIONS:
                return "chat/completions"

    @staticmethod
    def from_http_path(path: str) -> OpenAIEndpointKind | None:
        if path == OpenAIEndpointKind.COMPLETIONS.http_path:
            return OpenAIEndpointKind.COMPLETIONS
        if path == OpenAIEndpointKind.CHAT_COMPLETIONS.http_path:
            return OpenAIEndpointKind.CHAT_COMPLETIONS
        return None


@dataclass(frozen=True)
class OpenAIHttpRequestEnvelope:
    """Opaque OpenAI-compatible HTTP request body plus logical request identity."""

    request_id: str
    endpoint: OpenAIEndpointKind
    payload_json: str


@dataclass(frozen=True)
class OpenAIHttpResponseEnvelope:
    """Opaque OpenAI-compatible HTTP response body returned through the broker."""

    status_code: int
    payload_json: str
    content_type: str = JSON_CONTENT_TYPE


@dataclass(frozen=True)
class InferenceLease:
    """A broker lease for one request."""

    lease_id: str
    worker_id: str
    request: OpenAIHttpRequestEnvelope
    expires_at: float


@dataclass(frozen=True)
class LeaseResult:
    """Lease RPC result.

    ``lease=None`` means no work was available before the lease call timed out.
    ``stopped=True`` means the broker is shutting down and workers should exit.
    """

    lease: InferenceLease | None
    stopped: bool = False


class BrokerRequestStatus(StrEnum):
    """Broker-visible request lifecycle."""

    PENDING = "pending"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {BrokerRequestStatus.SUCCEEDED, BrokerRequestStatus.FAILED}


@dataclass
class _BrokerEntry:
    request: OpenAIHttpRequestEnvelope
    status: BrokerRequestStatus = BrokerRequestStatus.PENDING
    response: OpenAIHttpResponseEnvelope | None = None
    lease_id: str | None = None
    worker_id: str | None = None
    lease_expires_at: float | None = None


class IrisInferenceBroker:
    """In-memory request broker for one eval-scoped Iris inference service."""

    def __init__(self, lease_timeout: float = DEFAULT_LEASE_TIMEOUT, now: Callable[[], float] = time.monotonic) -> None:
        if lease_timeout <= 0:
            raise ValueError("lease_timeout must be positive.")
        self._lease_timeout = lease_timeout
        self._now = now
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._entries: dict[str, _BrokerEntry] = {}
        self._stopped = False

    def submit(self, request: OpenAIHttpRequestEnvelope) -> bool:
        """Submit a logical request.

        Returns True for a new request and False when the same request was
        already known. Re-submitting a known request id with different contents
        is rejected because it would break replay dedupe.
        """
        with self._condition:
            if self._stopped:
                raise RuntimeError("IrisInferenceBroker is stopped.")

            existing = self._entries.get(request.request_id)
            if existing is not None:
                if existing.request != request:
                    raise ValueError(f"request_id {request.request_id!r} was already submitted with different data.")
                return False

            self._entries[request.request_id] = _BrokerEntry(request=request)
            self._condition.notify_all()
            return True

    def lease(self, worker_id: str, wait_timeout: float | None = None) -> LeaseResult:
        """Lease pending work, requeueing any expired leases first."""
        if wait_timeout is not None and wait_timeout < 0:
            raise ValueError("wait_timeout must be non-negative.")

        deadline = None if wait_timeout is None else self._now() + wait_timeout
        with self._condition:
            while True:
                if self._stopped:
                    return LeaseResult(lease=None, stopped=True)

                self._expire_leases_locked()
                lease = self._next_lease_locked(worker_id)
                if lease is not None:
                    return LeaseResult(lease=lease)

                if wait_timeout == 0:
                    return LeaseResult(lease=None)

                if deadline is None:
                    self._condition.wait()
                    continue

                remaining = deadline - self._now()
                if remaining <= 0:
                    return LeaseResult(lease=None)
                self._condition.wait(timeout=remaining)

    def complete(self, request_id: str, response: OpenAIHttpResponseEnvelope) -> bool:
        """Store a successful terminal response if no terminal result exists."""
        return self._finish(request_id, BrokerRequestStatus.SUCCEEDED, response)

    def fail(self, request_id: str, response: OpenAIHttpResponseEnvelope) -> bool:
        """Store a failed terminal response if no terminal result exists."""
        return self._finish(request_id, BrokerRequestStatus.FAILED, response)

    def poll(self, request_id: str) -> OpenAIHttpResponseEnvelope | None:
        """Return a terminal response if the request has one."""
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or not entry.status.terminal:
                return None
            return entry.response

    def wait(self, request_id: str, timeout: float | None = None) -> OpenAIHttpResponseEnvelope | None:
        """Wait for a terminal response, returning None on timeout."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative.")

        deadline = None if timeout is None else self._now() + timeout
        with self._condition:
            while True:
                response = self.poll(request_id)
                if response is not None:
                    return response

                if request_id not in self._entries:
                    return None

                if timeout == 0:
                    return None

                if deadline is None:
                    self._condition.wait()
                    continue

                remaining = deadline - self._now()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def status(self, request_id: str) -> BrokerRequestStatus | None:
        """Return the broker lifecycle state for a request."""
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            return entry.status

    def stop(self) -> None:
        """Wake blocked workers and stop accepting new submissions."""
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _finish(
        self,
        request_id: str,
        status: BrokerRequestStatus,
        response: OpenAIHttpResponseEnvelope,
    ) -> bool:
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                raise KeyError(f"Unknown request_id {request_id!r}.")
            if entry.status.terminal:
                return False

            entry.status = status
            entry.response = response
            entry.lease_id = None
            entry.worker_id = None
            entry.lease_expires_at = None
            self._condition.notify_all()
            return True

    def _expire_leases_locked(self) -> None:
        now = self._now()
        for entry in self._entries.values():
            if entry.status != BrokerRequestStatus.LEASED:
                continue
            if entry.lease_expires_at is None or entry.lease_expires_at > now:
                continue
            entry.status = BrokerRequestStatus.PENDING
            entry.lease_id = None
            entry.worker_id = None
            entry.lease_expires_at = None

    def _next_lease_locked(self, worker_id: str) -> InferenceLease | None:
        for entry in self._entries.values():
            if entry.status != BrokerRequestStatus.PENDING:
                continue

            lease_id = uuid.uuid4().hex
            expires_at = self._now() + self._lease_timeout
            entry.status = BrokerRequestStatus.LEASED
            entry.lease_id = lease_id
            entry.worker_id = worker_id
            entry.lease_expires_at = expires_at
            return InferenceLease(
                lease_id=lease_id,
                worker_id=worker_id,
                request=entry.request,
                expires_at=expires_at,
            )
        return None


class IrisInferenceWorker:
    """Worker actor that forwards leased requests to a local OpenAI-compatible engine."""

    def __init__(
        self,
        broker: ActorHandle,
        engine_base_url: str,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        lease_wait_timeout: float = DEFAULT_WORKER_LEASE_WAIT_TIMEOUT,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive.")
        if lease_wait_timeout <= 0:
            raise ValueError("lease_wait_timeout must be positive.")

        self._broker = broker
        self._engine_base_url = engine_base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._lease_wait_timeout = lease_wait_timeout
        self._worker_id = self._resolve_worker_id()

    def run(self, max_requests: int | None = None) -> int:
        """Process broker work until stopped or ``max_requests`` is reached."""
        if max_requests is not None and max_requests < 0:
            raise ValueError("max_requests must be non-negative.")

        processed = 0
        while max_requests is None or processed < max_requests:
            result = self._broker.lease(self._worker_id, wait_timeout=self._lease_wait_timeout)
            if result.stopped:
                return processed
            if result.lease is None:
                continue

            self._forward(result.lease.request)
            processed += 1

        return processed

    def _forward(self, request: OpenAIHttpRequestEnvelope) -> None:
        url = f"{self._engine_base_url}/{request.endpoint.api_path}"
        try:
            response = requests.post(
                url,
                data=request.payload_json.encode("utf-8"),
                headers={"Content-Type": JSON_CONTENT_TYPE},
                timeout=self._request_timeout,
            )
        except requests.RequestException as exc:
            self._broker.fail(
                request.request_id,
                OpenAIHttpResponseEnvelope(
                    status_code=502,
                    payload_json=json.dumps(
                        {
                            "error": "engine request failed",
                            "request_id": request.request_id,
                            "detail": str(exc),
                        }
                    ),
                ),
            )
            return

        self._broker.complete(
            request.request_id,
            OpenAIHttpResponseEnvelope(
                status_code=response.status_code,
                payload_json=response.text,
                content_type=response.headers.get("Content-Type", JSON_CONTENT_TYPE),
            ),
        )

    def _resolve_worker_id(self) -> str:
        try:
            actor = current_actor()
        except RuntimeError:
            return f"worker-{uuid.uuid4().hex[:8]}"
        return f"{actor.group_name}-{actor.index}"


@dataclass(frozen=True)
class RunningIrisInferenceProxy:
    """Handle for a running local proxy."""

    base_url: str
    request_id_header: str = MARIN_REQUEST_ID_HEADER


class _IrisInferenceProxyServer(HTTPServer):
    broker: ActorHandle
    request_timeout: float


class _IrisInferenceProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("Iris inference proxy: " + format, *args)

    def do_POST(self) -> None:
        endpoint = OpenAIEndpointKind.from_http_path(self.path)
        if endpoint is None:
            self._write_json(404, {"error": "not found"})
            return

        request_id = self.headers.get(MARIN_REQUEST_ID_HEADER) or uuid.uuid4().hex
        request = OpenAIHttpRequestEnvelope(
            request_id=request_id,
            endpoint=endpoint,
            payload_json=self._read_body(),
        )

        try:
            self._proxy_server.broker.submit(request)
        except ValueError as exc:
            self._write_json(409, {"error": str(exc), "request_id": request_id})
            return

        future = self._proxy_server.broker.wait.submit(request_id, self._proxy_server.request_timeout)
        try:
            response = future.result(timeout=self._proxy_server.request_timeout + DEFAULT_CLEANUP_TIMEOUT)
        except TimeoutError:
            self._write_json(504, {"error": "inference request timed out", "request_id": request_id})
            return

        if response is None:
            self._write_json(504, {"error": "inference request timed out", "request_id": request_id})
            return

        self._write_response(response, request_id)

    @property
    def _proxy_server(self) -> _IrisInferenceProxyServer:
        return cast(_IrisInferenceProxyServer, self.server)

    def _read_body(self) -> str:
        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length).decode("utf-8")

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        self._write_response(
            OpenAIHttpResponseEnvelope(status_code=status, payload_json=json.dumps(payload)),
            request_id=str(payload.get("request_id", "")),
        )

    def _write_response(self, response: OpenAIHttpResponseEnvelope, request_id: str) -> None:
        body = response.payload_json.encode("utf-8")
        self.send_response(response.status_code)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(body)))
        if request_id:
            self.send_header(MARIN_REQUEST_ID_HEADER, request_id)
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def serve_iris_inference_proxy(
    broker: ActorHandle,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Iterator[RunningIrisInferenceProxy]:
    """Run a local OpenAI-compatible proxy backed by a broker actor."""
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive.")

    server = _IrisInferenceProxyServer((host, port), _IrisInferenceProxyHandler)
    server.broker = broker
    server.request_timeout = request_timeout
    context = contextvars.copy_context()
    thread = threading.Thread(target=context.run, args=(server.serve_forever,))
    thread.start()
    try:
        yield RunningIrisInferenceProxy(base_url=f"http://{host}:{server.server_port}/v1")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@dataclass(frozen=True)
class IrisInferenceLauncherConfig:
    """Launcher wiring for broker, proxy, and workers.

    The engine itself is deliberately external. Each worker forwards to
    ``engine_base_url`` from the worker process, so Iris deployments should
    start the OpenAI-compatible engine beside the worker before using this
    routing layer.
    """

    engine_base_url: str
    worker_count: int = 1
    lease_timeout: float = DEFAULT_LEASE_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    worker_lease_wait_timeout: float = DEFAULT_WORKER_LEASE_WAIT_TIMEOUT
    worker_ready_timeout: float = DEFAULT_WORKER_READY_TIMEOUT
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 0
    service_name: str | None = None
    broker_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig.with_cpu(cpu=0, ram="512m", disk="1g", preemptible=False)
    )
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_cpu(cpu=1, ram="1g"))

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        if self.lease_timeout <= 0:
            raise ValueError("lease_timeout must be positive.")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive.")
        if self.worker_lease_wait_timeout <= 0:
            raise ValueError("worker_lease_wait_timeout must be positive.")
        if self.worker_ready_timeout <= 0:
            raise ValueError("worker_ready_timeout must be positive.")


@dataclass(frozen=True)
class IrisInferenceLauncher:
    """ModelLauncher-compatible MVP for Iris inference routing."""

    client: Client
    config: IrisInferenceLauncherConfig

    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]:
        """Launch broker, proxy, and workers around an already-running engine."""
        return _IrisInferenceServiceContext(self.client, self.config, deployment)


class _IrisInferenceServiceContext(AbstractContextManager[RunningModel]):
    def __init__(self, client: Client, config: IrisInferenceLauncherConfig, deployment: ModelDeployment) -> None:
        self._client = client
        self._config = config
        self._deployment = deployment
        self._broker: ActorHandle | None = None
        self._worker_group: ActorGroup | None = None
        self._worker_futures: list[ActorFuture] = []
        self._proxy_context: AbstractContextManager[RunningIrisInferenceProxy] | None = None

    def __enter__(self) -> RunningModel:
        service_name = self._config.service_name or f"iris-inference-{uuid.uuid4().hex[:8]}"
        self._broker = self._client.create_actor(
            IrisInferenceBroker,
            self._config.lease_timeout,
            name=f"{service_name}-broker",
            resources=self._config.broker_resources,
        )
        self._worker_group = self._client.create_actor_group(
            IrisInferenceWorker,
            self._broker,
            self._config.engine_base_url,
            name=f"{service_name}-workers",
            count=self._config.worker_count,
            resources=self._config.worker_resources,
            request_timeout=self._config.request_timeout,
            lease_wait_timeout=self._config.worker_lease_wait_timeout,
        )
        workers = self._worker_group.wait_ready(timeout=self._config.worker_ready_timeout)
        self._worker_futures = [worker.run.submit(max_requests=None) for worker in workers]

        self._proxy_context = serve_iris_inference_proxy(
            self._broker,
            host=self._config.proxy_host,
            port=self._config.proxy_port,
            request_timeout=self._config.request_timeout,
        )
        proxy = self._proxy_context.__enter__()
        return RunningModel(
            endpoint=OpenAIEndpoint(base_url=proxy.base_url, model=self._deployment.model_name),
            tokenizer=self._deployment.tokenizer,
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._proxy_context is not None:
            self._proxy_context.__exit__(exc_type, exc_value, traceback)
            self._proxy_context = None

        if self._broker is not None:
            self._broker.stop.remote().result(timeout=DEFAULT_CLEANUP_TIMEOUT)

        for future in self._worker_futures:
            future.result(timeout=DEFAULT_CLEANUP_TIMEOUT)
        self._worker_futures = []

        if self._worker_group is not None:
            self._worker_group.shutdown()
            self._worker_group = None

        return None
