# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip

import httpx
import pytest
from marin.inference.auxiliary import auxiliary_response
from marin.inference.types import InferenceRequest


@pytest.mark.asyncio
async def test_auxiliary_request_tries_another_endpoint_after_timeout():
    requests = []
    blocked = asyncio.Event()

    async def upstream(request):
        requests.append(request)
        if len(requests) == 1:
            await blocked.wait()
        return httpx.Response(200, stream=httpx.ByteStream(b'{"data":[{"id":"available-model"}]}'))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response = await auxiliary_response(
            client, ("http://one", "http://two"), InferenceRequest("models", "GET", "/v1/models", b""), 0.2
        )

    assert response.status_code == 200
    assert len(requests) == 2
    assert requests[0].url.host != requests[1].url.host
    assert b"available-model" in response.body


@pytest.mark.asyncio
async def test_auxiliary_request_preserves_compression_query_and_headers():
    compressed = gzip.compress(b"metric_total 17\n")
    requests = []

    def upstream(request):
        requests.append(request)
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(compressed))

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response = await auxiliary_response(
            client,
            ("http://worker",),
            InferenceRequest(
                "metrics",
                "GET",
                "/metrics",
                b"",
                query_string="format=prometheus",
                headers=(("x-trace-id", "trace"), ("proxy-authorization", "private")),
            ),
            5,
        )

    assert response.body == compressed
    assert response.headers["content-encoding"] == "gzip"
    [request] = requests
    assert str(request.url) == "http://worker/metrics?format=prometheus"
    assert request.headers["x-trace-id"] == "trace"
    assert "proxy-authorization" not in request.headers
