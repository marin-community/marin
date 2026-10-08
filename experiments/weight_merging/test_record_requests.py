# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import tarfile

import pytest

from experiments.weight_merging.record_requests import RecordRequests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_persists_exchange_before_final_delivery_and_preserves_response(tmp_path, stream):
    request = {"model": "parent", "messages": [{"role": "user", "content": "17 + 24 ="}], "stream": stream}
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

    if not stream:
        response = [
            {"type": "http.response.start", "status": 200, "headers": []},
            {"type": "http.response.body", "body": b'{"choices":[]}', "more_body": False},
        ]
    else:
        # SSE clients finish on DONE, before ASGI sends its final empty body.
        response[-1]["more_body"] = True
        response.append({"type": "http.response.body", "body": b"", "more_body": False})
    captures = tmp_path / "captures"
    durable = tmp_path / "durable"
    durable.mkdir()

    async def receive():
        return next(incoming)

    async def send(message):
        if message["type"] == "http.response.body" and (
            b"[DONE]" in message["body"] or not message.get("more_body", False)
        ):
            (archive_path,) = durable.iterdir()
            with tarfile.open(archive_path) as archive:
                assert json.load(archive.extractfile("complete.json"))["path"] == "/v1/chat/completions"
                assert archive.extractfile("request.json").read() == body
                assert archive.extractfile("response.bin").read() == b"".join(
                    event.get("body", b"") for event in response
                )
        emitted.append(message)

    async def app(scope, receive, send):
        message = await receive()
        assert json.loads(message["body"]) == request | {"return_token_ids": True}
        assert dict(scope["headers"])[b"content-length"] == str(len(message["body"])).encode()
        for event in response:
            await send(event)

    await RecordRequests(app, captures, str(durable))(
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
    (saved,) = captures.iterdir()
    assert (saved / "request.json").read_bytes() == body
    assert (saved / "response.bin").read_bytes() == b"".join(event.get("body", b"") for event in response)
    assert json.loads((saved / "complete.json").read_text())["path"] == "/v1/chat/completions"
