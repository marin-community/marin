"""Agent side of the S4 spike: the RPC *client* that dials home.

The agent never exposes an inbound endpoint (NAT/IAP-friendly). It only dials the
root: a steady cadence of ``Poll`` for the piggyback path, or a single held
``CommandStream`` plus unary ``ReportResult`` for the held-stream variant.

The "stub" executor stands in for the real worker ``exec_in_container``: it does
trivial work so the measured number is *transport* latency, not exec cost (real
exec adds the same fixed cost to both transports).
"""

from __future__ import annotations

import threading
import time

from connectrpc.compression import Compression

import remote_agent_pb2 as pb
from remote_agent_connect import RemoteAgentServiceClientSync

from rigging.auth import BearerTokenInjector, StaticTokenProvider
from root_server import AGENT_TOKEN


def _client(url: str, *, auth: bool) -> RemoteAgentServiceClientSync:
    interceptors = []
    if auth:
        interceptors.append(BearerTokenInjector(StaticTokenProvider(AGENT_TOKEN), "authorization"))
    # send_compression=None: tiny payloads, keep the wire honest for latency.
    return RemoteAgentServiceClientSync(
        address=url, timeout_ms=30_000, interceptors=interceptors,
        accept_compression=(), send_compression=None,
    )


def _run_stub_exec(cmd: pb.InteractiveCommand) -> pb.CommandResult:
    argv = list(cmd.exec.command)
    return pb.CommandResult(
        command_id=cmd.command_id,
        exec=pb.ExecResponse(exit_code=0, stdout="ran: " + " ".join(argv)),
    )


class PollAgent:
    """Dial-home Poll loop (spec.md §1.1 piggyback path).

    fast_follow: after executing pending commands, immediately issue the next Poll
    to return results instead of waiting a full cadence (collapses the UP leg).
    """

    def __init__(self, url: str, backend_id: str, cadence: float, *, auth: bool, fast_follow: bool) -> None:
        self._client = _client(url, auth=auth)
        self._backend_id = backend_id
        self._cadence = cadence
        self._fast_follow = fast_follow
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="s4-poll-agent", daemon=True)
        self._last_sync_id = 0
        self._pending_results: list[pb.CommandResult] = []
        self.poll_count = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._client.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            results = self._pending_results
            self._pending_results = []
            req = pb.PollRequest(
                backend_id=self._backend_id,
                last_sync_id=self._last_sync_id,
                command_results=results,
            )
            resp = self._client.poll(req)
            self.poll_count += 1
            self._last_sync_id = resp.new_sync_id
            for cmd in resp.pending_commands:
                self._pending_results.append(_run_stub_exec(cmd))
            # fast-follow: if we have results to return, poll again immediately.
            if self._fast_follow and self._pending_results:
                continue
            self._stop.wait(self._cadence)


class StreamAgent:
    """Held-stream variant: one long-lived CommandStream + unary ReportResult."""

    def __init__(self, url: str, backend_id: str, *, auth: bool) -> None:
        self._stream_client = _client(url, auth=auth)
        self._report_client = _client(url, auth=auth)
        self._backend_id = backend_id
        self._thread = threading.Thread(target=self._loop, name="s4-stream-agent", daemon=True)
        self._started = threading.Event()

    def start(self) -> None:
        self._thread.start()
        self._started.wait(5)

    def stop(self) -> None:
        self._stream_client.close()
        self._report_client.close()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        stream = self._stream_client.command_stream(pb.StreamRequest(backend_id=self._backend_id))
        self._started.set()
        for cmd in stream:
            result = _run_stub_exec(cmd)
            self._report_client.report_result(result)
