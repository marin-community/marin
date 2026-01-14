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

"""Actor server implementation for hosting actor instances."""

import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
import uvicorn

from fluster import actor_pb2
from fluster.actor.types import ActorContext, ActorId, _set_actor_context
from fluster.actor_connect import ActorServiceASGIApplication
from connectrpc.request import RequestContext


@dataclass
class RegisteredActor:
    """An actor registered with the server."""

    name: str
    actor_id: ActorId
    instance: Any
    methods: dict[str, Callable]
    registered_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class ActorServer:
    """Server for hosting actor instances and handling RPC calls."""

    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        """Initialize the actor server.

        Args:
            host: Host address to bind to
            port: Port to bind to (0 for auto-assign)
        """
        self._host = host
        self._port = port
        self._actors: dict[str, RegisteredActor] = {}
        self._context: ActorContext | None = None
        self._app: ActorServiceASGIApplication | None = None
        self._actual_port: int | None = None

    @property
    def address(self) -> str:
        """Get the server address as host:port."""
        port = self._actual_port or self._port
        return f"{self._host}:{port}"

    def register(self, name: str, actor: Any) -> ActorId:
        """Register an actor instance with the server.

        Args:
            name: Name for actor discovery
            actor: Actor instance with public methods

        Returns:
            Unique actor ID
        """
        actor_id = ActorId(f"{name}-{uuid.uuid4().hex[:8]}")
        methods = {m: getattr(actor, m) for m in dir(actor) if not m.startswith("_") and callable(getattr(actor, m))}
        self._actors[name] = RegisteredActor(
            name=name,
            actor_id=actor_id,
            instance=actor,
            methods=methods,
        )
        return actor_id

    async def call(self, request: actor_pb2.ActorCall, ctx: RequestContext) -> actor_pb2.ActorResponse:
        """Handle actor RPC call."""
        # Find actor
        actor_name = request.actor_name or next(iter(self._actors), "")
        actor = self._actors.get(actor_name)
        if not actor:
            error = actor_pb2.ActorError(
                error_type="NotFound",
                message=f"Actor '{actor_name}' not found",
            )
            return actor_pb2.ActorResponse(error=error)

        method = actor.methods.get(request.method_name)
        if not method:
            error = actor_pb2.ActorError(
                error_type="NotFound",
                message=f"Method '{request.method_name}' not found",
            )
            return actor_pb2.ActorResponse(error=error)

        try:
            args = cloudpickle.loads(request.serialized_args) if request.serialized_args else ()
            kwargs = cloudpickle.loads(request.serialized_kwargs) if request.serialized_kwargs else {}

            # Set context for this call
            _set_actor_context(self._context)
            try:
                result = method(*args, **kwargs)
            finally:
                _set_actor_context(None)

            return actor_pb2.ActorResponse(serialized_value=cloudpickle.dumps(result))

        except Exception as e:
            error = actor_pb2.ActorError(
                error_type=type(e).__name__,
                message=str(e),
                serialized_exception=cloudpickle.dumps(e),
            )
            return actor_pb2.ActorResponse(error=error)

    async def health_check(self, request: actor_pb2.Empty, ctx: RequestContext) -> actor_pb2.HealthResponse:
        """Handle health check."""
        return actor_pb2.HealthResponse(healthy=True)

    def _create_app(self) -> ActorServiceASGIApplication:
        """Create the Connect RPC ASGI application for the server."""
        return ActorServiceASGIApplication(service=self)

    def serve_background(self, context: ActorContext | None = None) -> int:
        """Start server in background thread.

        Args:
            context: ActorContext to inject into actor method calls

        Returns:
            Actual port the server is listening on
        """
        self._context = context
        self._app = self._create_app()

        # Find available port if port=0
        if self._port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self._actual_port = s.getsockname()[1]
        else:
            self._actual_port = self._port

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._actual_port,
            log_level="error",
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server to be ready
        for _ in range(50):
            try:
                import httpx

                httpx.get(f"http://{self._host}:{self._actual_port}/", timeout=0.1)
            except Exception:
                pass
            time.sleep(0.1)
            if server.started:
                break

        return self._actual_port
