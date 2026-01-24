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

"""Tests for resolver functionality.

This module tests all resolver implementations in iris.actor.resolver:
- FixedResolver: Static endpoint mapping
- GcsResolver: Discovery via GCP VM instance metadata
"""

from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver, GcsResolver, MockGcsApi
from iris.actor.server import ActorServer


class Echo:
    def echo(self, msg: str) -> str:
        return f"echo: {msg}"


# FixedResolver tests


def test_client_with_resolver():
    server = ActorServer(host="127.0.0.1")
    server.register("echo", Echo())
    port = server.serve_background()

    resolver = FixedResolver({"echo": f"http://127.0.0.1:{port}"})
    client = ActorClient(resolver, "echo")

    assert client.echo("hello") == "echo: hello"


def test_gcs_resolver_finds_actors():
    """GcsResolver finds actors via metadata tags."""
    api = MockGcsApi(
        [
            {
                "name": "worker-1",
                "internal_ip": "10.0.0.1",
                "status": "RUNNING",
                "metadata": {
                    "iris_actor_inference": "8080",
                },
            },
        ]
    )
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert "10.0.0.1:8080" in result.first().url
    assert result.first().actor_id == "gcs-worker-1-inference"
    assert result.first().metadata == {"instance": "worker-1"}


def test_gcs_resolver_ignores_non_running():
    """GcsResolver only considers RUNNING instances."""
    api = MockGcsApi(
        [
            {
                "name": "worker-1",
                "internal_ip": "10.0.0.1",
                "status": "TERMINATED",
                "metadata": {
                    "iris_actor_inference": "8080",
                },
            },
        ]
    )
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert result.is_empty


def test_gcs_resolver_multiple_instances():
    """GcsResolver finds actors across multiple instances."""
    api = MockGcsApi(
        [
            {
                "name": "worker-1",
                "internal_ip": "10.0.0.1",
                "status": "RUNNING",
                "metadata": {
                    "iris_actor_inference": "8080",
                },
            },
            {
                "name": "worker-2",
                "internal_ip": "10.0.0.2",
                "status": "RUNNING",
                "metadata": {
                    "iris_actor_inference": "8080",
                },
            },
        ]
    )
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 2
    urls = {ep.url for ep in result.endpoints}
    assert "http://10.0.0.1:8080" in urls
    assert "http://10.0.0.2:8080" in urls


def test_gcs_resolver_no_matching_actor():
    """GcsResolver returns empty when no actor matches."""
    api = MockGcsApi(
        [
            {
                "name": "worker-1",
                "internal_ip": "10.0.0.1",
                "status": "RUNNING",
                "metadata": {
                    "iris_actor_training": "8080",
                },
            },
        ]
    )
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert result.is_empty


def test_gcs_resolver_missing_internal_ip():
    """GcsResolver skips instances without internal IP."""
    api = MockGcsApi(
        [
            {
                "name": "worker-1",
                "internal_ip": None,
                "status": "RUNNING",
                "metadata": {
                    "iris_actor_inference": "8080",
                },
            },
        ]
    )
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert result.is_empty
