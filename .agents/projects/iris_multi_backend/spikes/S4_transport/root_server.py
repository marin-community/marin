"""Root side of the S4 spike: the RPC *server* the agent dials home to.

Holds the authoritative desired-state and the interactive-command plumbing. The
root-side ``RemoteTaskBackend.exec_in_container`` (modeled by ``RootState.exec_*``)
is a *blocking* call made from the controller/CLI thread; it parks on a
``threading.Event`` until the matching ``CommandResult`` returns over a later Poll
(piggyback path) or a ``ReportResult`` unary (held-stream path).
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from dataclasses import dataclass, field

import uvicorn
from connectrpc.request import RequestContext
from starlette.applications import Starlette
from starlette.routing import Mount

import remote_agent_pb2 as pb
from remote_agent_connect import RemoteAgentService, RemoteAgentServiceASGIApplication

# Auth on the dial-home wire: the agent carries a static "system:controller"
# bearer token (spec.md §6). Real deployments mint a JWT (auth.py:462) or present
# an IAP service-account ID token; here a static pair proves the attach/verify path.
from rigging.server_auth import AuthInterceptor, StaticTokenVerifier

AGENT_TOKEN = "s4-system-controller-token"  # noqa: S105 - spike-local secret
AGENT_IDENTITY = "system:controller"


@dataclass
class _Inflight:
    t_issue: float
    event: threading.Event = field(default_factory=threading.Event)
    result: pb.CommandResult | None = None
    t_result: float = 0.0


class RootState:
    """Authoritative root state + interactive-command dispatch for one backend."""

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self._lock = threading.Lock()
        self._epoch = 1
        self._sync_id = 0
        self._upserts: dict[str, pb.DesiredAttempt] = {}
        # Poll-piggyback delivery queue (commands waiting to ride DOWN).
        self._pending_commands: collections.deque[pb.InteractiveCommand] = collections.deque()
        # Held-stream delivery queue (asyncio, drained by command_stream handler).
        self._stream_queue: asyncio.Queue[pb.InteractiveCommand] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inflight: dict[str, _Inflight] = {}
        self._command_seq = 0

    # ---- root-side TaskBackend surface (called from the controller/CLI thread) ----

    def _new_command(self, argv: list[str]) -> tuple[str, pb.InteractiveCommand, _Inflight]:
        with self._lock:
            self._command_seq += 1
            cid = f"cmd-{self._command_seq}"
            cmd = pb.InteractiveCommand(
                command_id=cid,
                target_task_id="task-0",
                origin_user="alice",
                exec=pb.ExecRequest(command=argv, timeout_seconds=30),
            )
            inflight = _Inflight(t_issue=time.perf_counter())
            self._inflight[cid] = inflight
        return cid, cmd, inflight

    def exec_piggyback(self, argv: list[str], timeout: float) -> tuple[pb.ExecResponse, float]:
        """spec.md §1.1 path: command rides DOWN in a PollResponse, result UP next Poll."""
        cid, cmd, inflight = self._new_command(argv)
        with self._lock:
            self._pending_commands.append(cmd)
        return self._await_result(cid, inflight, timeout)

    def exec_stream(self, argv: list[str], timeout: float) -> tuple[pb.ExecResponse, float]:
        """Held-stream path: command pushed immediately, result UP via ReportResult."""
        cid, cmd, inflight = self._new_command(argv)
        loop = self._wait_for_loop()
        loop.call_soon_threadsafe(self._stream_queue.put_nowait, cmd)
        return self._await_result(cid, inflight, timeout)

    def _await_result(self, cid: str, inflight: _Inflight, timeout: float) -> tuple[pb.ExecResponse, float]:
        if not inflight.event.wait(timeout):
            raise TimeoutError(f"interactive command {cid} timed out after {timeout}s")
        latency = inflight.t_result - inflight.t_issue
        assert inflight.result is not None
        return inflight.result.exec, latency

    def _wait_for_loop(self) -> asyncio.AbstractEventLoop:
        deadline = time.monotonic() + 5.0
        while self._loop is None:
            if time.monotonic() > deadline:
                raise RuntimeError("server event loop never came up")
            time.sleep(0.01)
        return self._loop

    # ---- shared helpers ----

    def _resolve(self, results: list[pb.CommandResult]) -> None:
        now = time.perf_counter()
        with self._lock:
            for cr in results:
                inflight = self._inflight.pop(cr.command_id, None)
                if inflight is None:
                    continue
                inflight.result = cr
                inflight.t_result = now
                inflight.event.set()

    def add_desired(self, attempt_uid: str, spec_json: str) -> None:
        with self._lock:
            self._sync_id += 1
            self._upserts[attempt_uid] = pb.DesiredAttempt(
                attempt_uid=attempt_uid, desired_generation=1, spec_json=spec_json
            )


class RemoteAgentServiceImpl(RemoteAgentService):
    def __init__(self, root: RootState) -> None:
        self._root = root

    async def poll(self, request: pb.PollRequest, ctx: RequestContext) -> pb.PollResponse:
        r = self._root
        if r._loop is None:
            r._loop = asyncio.get_running_loop()
        # 1. Drain results that rode UP (resolves blocked exec_piggyback callers).
        if request.command_results:
            r._resolve(list(request.command_results))
        # 2. Pack desired state + any pending interactive commands to ride DOWN.
        with r._lock:
            commands = list(r._pending_commands)
            r._pending_commands.clear()
            snapshot = request.last_sync_id == 0
            upserts = list(r._upserts.values()) if snapshot else []
            resp = pb.PollResponse(
                root_epoch=r._epoch,
                new_sync_id=r._sync_id,
                snapshot=snapshot,
                lease_duration_ms=30_000,
                upserts=upserts,
                pending_commands=commands,
            )
        return resp

    async def command_stream(self, request: pb.StreamRequest, ctx):
        r = self._root
        if r._loop is None:
            r._loop = asyncio.get_running_loop()
        while True:
            cmd = await r._stream_queue.get()
            yield cmd

    async def report_result(self, request: pb.CommandResult, ctx: RequestContext) -> pb.ReportAck:
        self._root._resolve([request])
        return pb.ReportAck(ok=True)


def build_app(root: RootState, *, auth: bool) -> Starlette:
    interceptors = []
    if auth:
        interceptors.append(AuthInterceptor(StaticTokenVerifier({AGENT_TOKEN: AGENT_IDENTITY})))
    rpc_app = RemoteAgentServiceASGIApplication(RemoteAgentServiceImpl(root), interceptors=interceptors)
    return Starlette(routes=[Mount(rpc_app.path, app=rpc_app)])


class ServerHandle:
    def __init__(self, server: uvicorn.Server, thread: threading.Thread, port: int) -> None:
        self.server = server
        self.thread = thread
        self.port = port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


def start_server(root: RootState, *, auth: bool, port: int = 0) -> ServerHandle:
    app = build_app(root, auth=auth)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", timeout_keep_alive=120)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="s4-root-server", daemon=True)
    thread.start()
    # Wait for uvicorn to bind and expose the chosen port.
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server failed to start")
        time.sleep(0.02)
    bound_port = server.servers[0].sockets[0].getsockname()[1]
    return ServerHandle(server, thread, bound_port)
