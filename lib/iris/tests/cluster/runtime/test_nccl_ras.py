# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from iris.cluster.runtime.nccl_ras import (
    CollectiveCountSkew,
    NcclRasFormat,
    capture_nccl_ras,
    collective_count_skews,
    parse_json_response,
    query_nccl_ras,
)


@dataclass(frozen=True)
class RasTestServer:
    port: int
    requests: list[bytes]
    thread: threading.Thread


class PartialTimeoutConnection:
    def __init__(self) -> None:
        self.responses: list[bytes | BaseException] = [b"OK\n", TimeoutError()]
        self.sent = b""

    def __enter__(self) -> "PartialTimeoutConnection":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def sendall(self, request: bytes) -> None:
        self.sent += request

    def shutdown(self, _how: int) -> None:
        pass

    def settimeout(self, _timeout: float) -> None:
        pass

    def recv(self, _size: int) -> bytes:
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _serve_responses(responses: Sequence[bytes]) -> RasTestServer:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    requests: list[bytes] = []

    def serve() -> None:
        with listener:
            for response in responses:
                connection, _ = listener.accept()
                with connection:
                    request = b""
                    while chunk := connection.recv(4096):
                        request += chunk
                    requests.append(request)
                    connection.sendall(response)

    thread = threading.Thread(target=serve)
    thread.start()
    return RasTestServer(port=listener.getsockname()[1], requests=requests, thread=thread)


def test_query_nccl_ras_sends_verbose_status_with_timeout_and_format() -> None:
    server = _serve_responses([b"OK\nstatus"])

    response = query_nccl_ras(
        host="127.0.0.1",
        port=server.port,
        timeout=1.2,
        response_format=NcclRasFormat.JSON,
    )
    server.thread.join()

    assert response == b"OK\nstatus"
    assert server.requests == [b"TIMEOUT 2\nSET FORMAT json\nVERBOSE STATUS\n"]


def test_query_nccl_ras_propagates_timeout_after_partial_response(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = PartialTimeoutConnection()
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: connection)

    with pytest.raises(TimeoutError):
        query_nccl_ras(
            host="127.0.0.1",
            port=28028,
            timeout=1,
            response_format=NcclRasFormat.JSON,
        )

    assert connection.sent == b"TIMEOUT 1\nSET FORMAT json\nVERBOSE STATUS\n"


def test_parse_json_response_skips_command_acknowledgements() -> None:
    report = {"nccl_version": "2.28.9", "communicators": []}

    assert parse_json_response(b"OK\nOK\n" + json.dumps(report).encode()) == report


def test_capture_nccl_ras_falls_back_to_text_when_json_is_unavailable() -> None:
    server = _serve_responses([b"ERROR unknown format\n", b"OK\nJob summary\n"])

    snapshot = capture_nccl_ras(host="127.0.0.1", port=server.port, timeout=1)
    server.thread.join()

    assert snapshot.response_format is NcclRasFormat.TEXT
    assert snapshot.report is None
    assert snapshot.raw_response == "OK\nJob summary\n"
    assert server.requests == [
        b"TIMEOUT 1\nSET FORMAT json\nVERBOSE STATUS\n",
        b"TIMEOUT 1\nSET FORMAT text\nVERBOSE STATUS\n",
    ]


def test_collective_count_skews_names_lagging_ranks() -> None:
    report = {
        "communicators": [
            {
                "hash": "0xabc",
                "ranks": [
                    {"rank": 0, "collective_counts": {"AllReduce": 8, "AllGather": 3}},
                    {"rank": 1, "collective_counts": {"AllReduce": 7, "AllGather": 3}},
                    {"rank": 2, "collective_counts": {"AllReduce": 7, "AllGather": 3}},
                ],
            }
        ]
    }

    assert collective_count_skews(report) == [
        CollectiveCountSkew(
            communicator_hash="0xabc",
            collective="AllReduce",
            minimum=7,
            maximum=8,
            lagging_ranks=(1, 2),
        )
    ]
