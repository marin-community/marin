# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for resolver functionality."""

from fluster.actor import FixedResolver
from fluster.actor.client import ActorClient
from fluster.actor.server import ActorServer


class Echo:
    def echo(self, msg: str) -> str:
        return f"echo: {msg}"


def test_fixed_resolver_single():
    resolver = FixedResolver({"svc": "http://localhost:8080"})
    result = resolver.resolve("svc")
    assert len(result.endpoints) == 1
    assert result.first().url == "http://localhost:8080"


def test_fixed_resolver_multiple():
    resolver = FixedResolver({"svc": ["http://h1:8080", "http://h2:8080"]})
    result = resolver.resolve("svc")
    assert len(result.endpoints) == 2


def test_fixed_resolver_missing():
    resolver = FixedResolver({})
    result = resolver.resolve("missing")
    assert result.is_empty


def test_client_with_resolver():
    server = ActorServer(host="127.0.0.1")
    server.register("echo", Echo())
    port = server.serve_background()

    resolver = FixedResolver({"echo": f"http://127.0.0.1:{port}"})
    client = ActorClient(resolver, "echo")

    assert client.echo("hello") == "echo: hello"
