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

"""Actor client for making RPC calls to actor servers."""

from typing import Any

import cloudpickle
import httpx

from fluster import actor_pb2


class ActorClient:
    """Simple actor client with hardcoded URL (Stage 1)."""

    def __init__(self, url: str, actor_name: str = ""):
        """Initialize the actor client.

        Args:
            url: Direct URL to actor server (e.g., "http://localhost:8080")
            actor_name: Name of actor on the server
        """
        self._url = url.rstrip("/")
        self._actor_name = actor_name
        self._timeout = 30.0

    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)


class _RpcMethod:
    """Represents a single RPC method call."""

    def __init__(self, client: ActorClient, method_name: str):
        self._client = client
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the RPC call."""
        call = actor_pb2.ActorCall(
            method_name=self._method_name,
            actor_name=self._client._actor_name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        response = httpx.post(
            f"{self._client._url}/fluster.actor.ActorService/Call",
            content=call.SerializeToString(),
            headers={"Content-Type": "application/proto"},
            timeout=self._client._timeout,
        )
        response.raise_for_status()

        resp = actor_pb2.ActorResponse()
        resp.ParseFromString(response.content)

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)
