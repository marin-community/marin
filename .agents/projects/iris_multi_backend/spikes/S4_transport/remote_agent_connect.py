"""Hand-written Connect binding for RemoteAgentService.

Mirrors the structure of iris's buf-generated ``*_connect.py`` (e.g.
``lib/iris/src/iris/rpc/worker_connect.py``) so the spike exercises the real
connectrpc server/client primitives (``ConnectASGIApplication`` + ``Endpoint`` on
the server, ``ConnectClient``/``ConnectClientSync.execute_*`` on the client). We
write it by hand only to avoid pulling the buf remote-plugin toolchain into a
throwaway spike; in-tree this file is generated.
"""

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping

from connectrpc.client import ConnectClient, ConnectClientSync
from connectrpc.compression import Compression
from connectrpc.errors import ConnectError
from connectrpc.code import Code
from connectrpc.interceptor import Interceptor, InterceptorSync
from connectrpc.method import IdempotencyLevel, MethodInfo
from connectrpc.request import Headers, RequestContext
from connectrpc.server import ConnectASGIApplication, Endpoint

import remote_agent_pb2 as pb

_SVC = "s4.remote_agent.RemoteAgentService"

_POLL = MethodInfo(
    name="Poll", service_name=_SVC,
    input=pb.PollRequest, output=pb.PollResponse,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)
_STREAM = MethodInfo(
    name="CommandStream", service_name=_SVC,
    input=pb.StreamRequest, output=pb.InteractiveCommand,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)
_REPORT = MethodInfo(
    name="ReportResult", service_name=_SVC,
    input=pb.CommandResult, output=pb.ReportAck,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)


class RemoteAgentService:
    """Service protocol. Root implements this; the agent dials it."""

    async def poll(self, request: pb.PollRequest, ctx: RequestContext) -> pb.PollResponse:
        raise ConnectError(Code.UNIMPLEMENTED, "Not implemented")

    async def command_stream(self, request: pb.StreamRequest, ctx: RequestContext) -> AsyncIterator[pb.InteractiveCommand]:
        raise ConnectError(Code.UNIMPLEMENTED, "Not implemented")
        yield  # pragma: no cover - marks this an async generator

    async def report_result(self, request: pb.CommandResult, ctx: RequestContext) -> pb.ReportAck:
        raise ConnectError(Code.UNIMPLEMENTED, "Not implemented")


class RemoteAgentServiceASGIApplication(ConnectASGIApplication[RemoteAgentService]):
    def __init__(
        self,
        service: RemoteAgentService,
        *,
        interceptors: Iterable[Interceptor] = (),
        compressions: Iterable[Compression] | None = None,
    ) -> None:
        super().__init__(
            service=service,
            endpoints=lambda svc: {
                f"/{_SVC}/Poll": Endpoint.unary(method=_POLL, function=svc.poll),
                f"/{_SVC}/CommandStream": Endpoint.server_stream(method=_STREAM, function=svc.command_stream),
                f"/{_SVC}/ReportResult": Endpoint.unary(method=_REPORT, function=svc.report_result),
            },
            interceptors=interceptors,
            compressions=compressions,
        )

    @property
    def path(self) -> str:
        return f"/{_SVC}"


class RemoteAgentServiceClientSync(ConnectClientSync):
    def poll(self, request: pb.PollRequest, *, headers: Headers | Mapping[str, str] | None = None,
             timeout_ms: int | None = None) -> pb.PollResponse:
        return self.execute_unary(request=request, method=_POLL, headers=headers, timeout_ms=timeout_ms)

    def command_stream(self, request: pb.StreamRequest, *, headers: Headers | Mapping[str, str] | None = None,
                       timeout_ms: int | None = None) -> Iterator[pb.InteractiveCommand]:
        return self.execute_server_stream(request=request, method=_STREAM, headers=headers, timeout_ms=timeout_ms)

    def report_result(self, request: pb.CommandResult, *, headers: Headers | Mapping[str, str] | None = None,
                      timeout_ms: int | None = None) -> pb.ReportAck:
        return self.execute_unary(request=request, method=_REPORT, headers=headers, timeout_ms=timeout_ms)


class RemoteAgentServiceClient(ConnectClient):
    async def poll(self, request: pb.PollRequest, *, headers: Headers | Mapping[str, str] | None = None,
                   timeout_ms: int | None = None) -> pb.PollResponse:
        return await self.execute_unary(request=request, method=_POLL, headers=headers, timeout_ms=timeout_ms)
