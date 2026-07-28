# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
import time
from collections.abc import Sequence

import pytest
from iris.cluster.runtime.nccl_ras import (
    CollectiveCountSkew,
    NcclRasFormat,
    capture_nccl_ras,
    collective_count_skews,
    parse_json_response,
    query_nccl_ras,
)


def _serve(
    responses: Sequence[Sequence[bytes]], *, pause: float = 0, hold_open: float = 0
) -> tuple[int, list[bytes], threading.Thread]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    requests: list[bytes] = []

    def serve() -> None:
        with listener:
            for chunks in responses:
                connection, _ = listener.accept()
                with connection:
                    request = b""
                    while chunk := connection.recv(4096):
                        request += chunk
                    requests.append(request)
                    for chunk in chunks:
                        connection.sendall(chunk)
                        if pause:
                            time.sleep(pause)
                    if hold_open:
                        time.sleep(hold_open)

    thread = threading.Thread(target=serve)
    thread.start()
    return listener.getsockname()[1], requests, thread


def test_query_nccl_ras_reads_partial_response_and_sends_documented_request():
    port, requests, server = _serve([[b"OK\n", b"status"]])
    assert query_nccl_ras(host="127.0.0.1", port=port, timeout=1, response_format=NcclRasFormat.JSON) == b"OK\nstatus"
    server.join()
    assert requests == [b"TIMEOUT 1\nSET FORMAT json\nVERBOSE STATUS\n"]


def test_query_nccl_ras_times_out_before_any_response():
    port, _, server = _serve([[]], hold_open=0.2)
    with pytest.raises(TimeoutError):
        query_nccl_ras(host="127.0.0.1", port=port, timeout=0.01, response_format=NcclRasFormat.JSON)
    server.join()


def test_capture_nccl_ras_falls_back_to_text_when_json_is_unavailable():
    port, requests, server = _serve([[b"ERROR unknown format\n"], [b"OK\nJob summary\n"]])
    snapshot = capture_nccl_ras(host="127.0.0.1", port=port, timeout=1)
    server.join()
    assert snapshot.response_format is NcclRasFormat.TEXT
    assert snapshot.raw_response == "OK\nJob summary\n"
    assert requests[1] == b"TIMEOUT 1\nSET FORMAT text\nVERBOSE STATUS\n"


def test_parse_json_response_skips_acknowledgements_and_skew_names_lagging_ranks():
    report = {
        "communicators": [
            {
                "hash": "0xabc",
                "ranks": [
                    {"rank": 0, "collective_counts": {"AllReduce": 8}},
                    {"rank": 1, "collective_counts": {"AllReduce": 7}},
                ],
            }
        ]
    }
    assert parse_json_response(b"OK\nOK\n" + json.dumps(report).encode()) == report
    assert collective_count_skews(report) == [CollectiveCountSkew("0xabc", "AllReduce", 7, 8, (1,))]
