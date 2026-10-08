# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.weight_merging.record_requests import RecordRequests


@pytest.mark.asyncio
async def test_records_chunked_request_and_preserves_streamed_response(tmp_path):
    request = {"model": "parent", "messages": [{"role": "user", "content": "17 + 24 ="}], "stream": True}
    body = json.dumps(request).encode()
    incoming = iter(
        [
            {"type": "http.request", "body": body[:17], "more_body": True},
            {"type": "http.request", "body": body[17:], "more_body": False},
        ]
    )
    emitted = []
    response = [
        {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]},
        {"type": "http.response.body", "body": b'data: {"token_ids":[41]}\n', "more_body": True},
        {"type": "http.response.body", "body": b"\ndata: [DONE]\n\n", "more_body": False},
    ]

    async def receive():
        return next(incoming)

    async def send(message):
        emitted.append(message)

    async def app(scope, receive, send):
        message = await receive()
        assert json.loads(message["body"]) == request | {"return_token_ids": True}
        assert dict(scope["headers"])[b"content-length"] == str(len(message["body"])).encode()
        for event in response:
            await send(event)

    await RecordRequests(app, tmp_path)(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-length", str(len(body)).encode())],
        },
        receive,
        send,
    )
    assert emitted == response
    (saved,) = tmp_path.iterdir()
    assert (saved / "request.json").read_bytes() == body
    assert (saved / "response.bin").read_bytes() == b'data: {"token_ids":[41]}\n\ndata: [DONE]\n\n'
    assert json.loads((saved / "complete.json").read_text())["path"] == "/v1/chat/completions"
