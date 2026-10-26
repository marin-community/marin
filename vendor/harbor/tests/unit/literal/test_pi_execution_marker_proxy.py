"""Pi's execution marker survives both literal-proxy request paths."""

import httpx
import pytest

from harbor.literal.proxy import PI_EXECUTED_TEACHER_HEADER, RecordProxy


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_pi_execution_marker_is_forwarded_only_when_explicit(tmp_path, stream):
    seen: list[str | None] = []

    async def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(PI_EXECUTED_TEACHER_HEADER))
        if stream:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n")
        return httpx.Response(200, json={"error": "fixture response"})

    proxy = RecordProxy("http://model", tmp_path / "literal.jsonl")
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        proxy._client = upstream
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app()), base_url="http://proxy") as client:
            for marker in (None, "0", "1"):
                headers = {} if marker is None else {PI_EXECUTED_TEACHER_HEADER: marker}
                await client.post("/v1/chat/completions", json={"model": "teacher", "stream": stream}, headers=headers)
            rejected = await client.post(
                "/v1/chat/completions",
                json={"model": "teacher", "stream": stream},
                headers={PI_EXECUTED_TEACHER_HEADER: "true"},
            )
    assert seen == [None, "0", "1"]
    assert rejected.status_code == 400
