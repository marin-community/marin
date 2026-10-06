# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from rigging.timing import ExponentialBackoff

from taskforge.llm.web import EXTRACT_BETA, FETCH_MAX_AGE, MAX_ATTEMPTS, web_tools

SEARCH = {"objective": "find x", "search_queries": ["x"]}
FETCH = {"urls": ["https://example.com"]}


@dataclass
class FakeParallel:
    base_url: str = ""
    replies: deque = field(default_factory=deque)
    requests: list[tuple[str, dict[str, str], dict]] = field(default_factory=list)


@pytest.fixture
def fake_parallel() -> Iterator[FakeParallel]:
    fake = FakeParallel()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append((self.path, dict(self.headers), body))
            status, text = fake.replies.popleft()
            self.send_response(status)
            self.end_headers()
            self.wfile.write(text.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    fake.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()
    server.server_close()


def call(fake: FakeParallel, *calls: tuple[int, dict]) -> list[str]:
    """Run ``(tool_index, arguments)`` calls in order through one shared client."""

    async def go() -> list[str]:
        async with httpx.AsyncClient() as http:
            backoff = ExponentialBackoff(initial=0.001, maximum=0.001)
            tools = web_tools(http, "test-key", base_url=fake.base_url, backoff=backoff)
            return [await tools[index].handler(arguments) for index, arguments in calls]

    return asyncio.run(go())


def test_search_and_fetch_post_to_parallel_and_return_its_body(fake_parallel):
    fake_parallel.replies.extend([(200, '{"search_id": "s1", "results": []}'), (200, '{"extract_id": "e1"}')])

    searched, fetched = call(fake_parallel, (0, SEARCH), (1, FETCH))

    assert (searched, fetched) == ('{"search_id": "s1", "results": []}', '{"extract_id": "e1"}')
    (search_path, search_headers, search_body), (fetch_path, fetch_headers, fetch_body) = fake_parallel.requests
    assert (search_path, search_headers["x-api-key"]) == ("/v1/search", "test-key")
    assert search_body == SEARCH
    assert (fetch_path, fetch_headers["parallel-beta"]) == ("/v1beta/extract", EXTRACT_BETA)
    assert fetch_body == {**FETCH, "fetch_policy": {"max_age_seconds": FETCH_MAX_AGE}}


def test_rejected_arguments_go_back_to_the_model_and_other_client_errors_raise_unretried(fake_parallel):
    fake_parallel.replies.extend([(422, '{"detail": "bad url"}'), (404, "no such route")])

    (rejected,) = call(fake_parallel, (1, {"urls": ["not a url"]}))

    assert rejected.startswith("error:") and "bad url" in rejected
    with pytest.raises(httpx.HTTPStatusError):
        call(fake_parallel, (1, FETCH))
    assert len(fake_parallel.requests) == 2


def test_transient_errors_are_retried_until_success_or_attempts_run_out(fake_parallel):
    fake_parallel.replies.extend([(429, "slow down"), (503, "busy"), (200, '{"extract_id": "e1"}')])

    assert call(fake_parallel, (1, FETCH)) == ['{"extract_id": "e1"}']
    assert len(fake_parallel.requests) == 3

    fake_parallel.replies.extend([(500, "boom")] * MAX_ATTEMPTS)
    with pytest.raises(httpx.HTTPStatusError):
        call(fake_parallel, (1, FETCH))
    assert len(fake_parallel.requests) == 3 + MAX_ATTEMPTS
