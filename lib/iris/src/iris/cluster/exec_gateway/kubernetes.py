# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes v5 remotecommand transport, multiplexed on the gateway event loop."""

import asyncio
import json
import logging
import ssl
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from kubernetes.client import CoreV1Api
from kubernetes.client.exceptions import ApiException
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from iris.cluster.exec_gateway.auth import ExecPodTarget
from iris.cluster.exec_gateway.service import MAX_CHUNK_BYTES, ExecOutput, ExecTransport
from iris.cluster.platforms.k8s.service import CloudK8sService

_CHANNEL_PROTOCOL = "v5.channel.k8s.io"
_SETUP_PARALLELISM = 8
logger = logging.getLogger(__name__)
_ATTEMPT_UID_LABEL = "iris.attempt_uid"


@dataclass(frozen=True)
class _ConnectionConfig:
    url: str
    authorization: str
    ssl_context: ssl.SSLContext | None
    proxy: str | None
    server_hostname: str | None


class KubernetesExecConnector:
    """Open binary process pipes to one verified live Kubernetes container."""

    def __init__(self, service: CloudK8sService):
        self._service = service
        self._executor = ThreadPoolExecutor(max_workers=_SETUP_PARALLELISM, thread_name_prefix="exec-setup")
        self._slots = asyncio.Semaphore(_SETUP_PARALLELISM)

    def _prepare(self, target: ExecPodTarget, command: tuple[str, ...]) -> _ConnectionConfig:
        with self._service.create_api_client() as api:
            self._validate_pod(CoreV1Api(api), target)
            config = api.configuration
            authorization = config.get_api_key_with_prefix("authorization") or ""
            parts = urlsplit(config.host)
            path = f"/api/v1/namespaces/{quote(target.namespace, safe='')}/pods/{quote(target.pod_name, safe='')}/exec"
            query = [
                ("container", target.container_name),
                ("stdin", "true"),
                ("stdout", "true"),
                ("stderr", "true"),
                ("tty", "false"),
            ]
            query.extend(("command", argument) for argument in command)
            url = urlunsplit(
                (
                    "wss" if parts.scheme == "https" else "ws",
                    parts.netloc,
                    parts.path.rstrip("/") + path,
                    urlencode(query),
                    "",
                )
            )
            context = None
            if parts.scheme == "https":
                context = ssl.create_default_context(cafile=config.ssl_ca_cert)
                if not config.verify_ssl:
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                if config.cert_file:
                    context.load_cert_chain(config.cert_file, config.key_file)
            return _ConnectionConfig(url, authorization, context, config.proxy, config.tls_server_name)

    def _validate_pod(self, api: CoreV1Api, target: ExecPodTarget) -> None:
        if target.namespace != self._service.namespace:
            raise ConnectError(Code.PERMISSION_DENIED, "Execution target is outside the gateway namespace")
        try:
            pod = api.read_namespaced_pod(target.pod_name, target.namespace, _request_timeout=10)
        except ApiException as error:
            raise ConnectError(Code.UNAVAILABLE, "Execution target Pod is unavailable") from error
        if pod.metadata.uid != target.pod_uid or pod.metadata.deletion_timestamp is not None:
            raise ConnectError(Code.FAILED_PRECONDITION, "Execution target Pod has been replaced or terminated")
        if (pod.metadata.labels or {}).get(_ATTEMPT_UID_LABEL) != target.attempt_uid:
            raise ConnectError(Code.FAILED_PRECONDITION, "Execution target Pod belongs to another attempt")
        declared = {container.name for container in pod.spec.containers}
        statuses = pod.status.container_statuses or []
        running = any(status.name == target.container_name and status.state.running is not None for status in statuses)
        if target.container_name not in declared or not running:
            raise ConnectError(Code.FAILED_PRECONDITION, "Execution target container is not running")

    def _revalidate(self, target: ExecPodTarget) -> None:
        with self._service.create_api_client() as api:
            self._validate_pod(CoreV1Api(api), target)

    async def open(self, target: ExecPodTarget, command: tuple[str, ...]) -> ExecTransport:
        async with self._slots:
            loop = asyncio.get_running_loop()
            prepared = await loop.run_in_executor(self._executor, self._prepare, target, command)
            kwargs = {"server_hostname": prepared.server_hostname} if prepared.server_hostname else {}
            try:
                socket = await connect(
                    prepared.url,
                    subprotocols=[_CHANNEL_PROTOCOL],
                    additional_headers={"Authorization": prepared.authorization} if prepared.authorization else {},
                    ssl=prepared.ssl_context,
                    proxy=prepared.proxy,
                    max_size=MAX_CHUNK_BYTES + 1,
                    max_queue=4,
                    write_limit=MAX_CHUNK_BYTES,
                    open_timeout=10,
                    close_timeout=2,
                    **kwargs,
                )
            except (OSError, WebSocketException) as error:
                raise ConnectError(Code.UNAVAILABLE, "Kubernetes exec connection failed") from error
            transport = _KubernetesExecTransport(socket)
            try:
                if socket.subprotocol != _CHANNEL_PROTOCOL:
                    raise ConnectError(Code.UNAVAILABLE, "Kubernetes server does not support v5 execution channels")
                # The exec subresource has no UID precondition. Fence name reuse
                # across connection establishment by checking the UID again.
                await loop.run_in_executor(self._executor, self._revalidate, target)
                return transport
            except BaseException:
                await transport.close()
                raise

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class _KubernetesExecTransport:
    def __init__(self, socket: ClientConnection):
        self._socket = socket
        self._stdin_closed = False
        self._closed = False

    async def send(self, data: bytes, *, eof: bool) -> None:
        try:
            if data:
                await self._socket.send(b"\x00" + data)
            if eof and not self._stdin_closed:
                await self._socket.send(b"\xff\x00")
                self._stdin_closed = True
        except ConnectionClosed as error:
            raise OSError("Kubernetes exec stdin closed") from error

    async def receive(self) -> ExecOutput:
        while True:
            try:
                frame = await self._socket.recv()
            except ConnectionClosed as error:
                raise EOFError("Kubernetes exec closed without exit status") from error
            if not isinstance(frame, bytes) or not frame:
                raise ValueError("Invalid Kubernetes exec channel frame")
            channel, data = frame[0], frame[1:]
            if channel == 1:
                return ExecOutput(stdout=data)
            if channel == 2:
                return ExecOutput(stderr=data)
            if channel == 3:
                status = json.loads(data)
                if status.get("status") == "Success":
                    return ExecOutput(exit_code=0)
                for cause in status.get("details", {}).get("causes", []):
                    if cause.get("reason") == "ExitCode":
                        return ExecOutput(exit_code=int(cause["message"]))
                raise ValueError(f"Kubernetes exec failed: {status.get('message', 'unknown error')}")
            if channel != 255:
                raise ValueError(f"Unexpected Kubernetes exec channel: {channel}")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not self._stdin_closed:
                await self.send(b"", eof=True)
        except OSError as error:
            # Exec can disappear with the Pod; disconnect still owns cleanup.
            logger.debug("Kubernetes exec stdin already closed during cleanup (%s)", type(error).__name__)
        finally:
            await self._socket.close()
