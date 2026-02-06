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

"""Actor server implementation for hosting actor instances.

Example:
    server = ActorServer()
    server.register("my-actor", MyActorClass())
    port = server.serve_background()
"""

import asyncio
import inspect
import logging
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NewType

import cloudpickle
import uvicorn
from connectrpc.request import RequestContext

from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc import actor_pb2
from iris.rpc.actor_connect import ActorServiceASGIApplication
from iris.time_utils import Duration, ExponentialBackoff, Timestamp

logger = logging.getLogger(__name__)

# Type aliases
ActorId = NewType("ActorId", str)


@dataclass
class RegisteredActor:
    name: str
    actor_id: ActorId
    instance: Any
    methods: dict[str, Callable]
    registered_at: Timestamp = field(default_factory=Timestamp.now)


class ActorServer:
    """Server for hosting actor instances and handling RPC calls."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        threads: ThreadContainer | None = None,
    ):
        """Initialize the actor server.

        Args:
            host: Host address to bind to
            port: Port to bind to. If None or 0, auto-assigns a free port.
            threads: ThreadContainer for managing server threads. If None, uses the default registry.
        """
        self._host = host
        self._port = port
        self._actors: dict[str, RegisteredActor] = {}
        self._app: ActorServiceASGIApplication | None = None
        self._actual_port: int | None = None
        self._threads = threads if threads is not None else get_thread_container()
        self._server: uvicorn.Server | None = None
        # Create dedicated executor for running actor methods
        # This avoids relying on asyncio's default executor which can be shut down prematurely
        self._executor = self._threads.spawn_executor(max_workers=32, prefix="actor-method")

    @property
    def address(self) -> str:
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

            # Run the method in our dedicated thread pool to avoid blocking the event loop.
            # This allows actors to make outgoing RPC calls without deadlocking.
            # We use our own executor instead of asyncio.to_thread() to avoid issues
            # when asyncio's default executor is shut down during process cleanup.
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, method, *args, **kwargs)

            return actor_pb2.ActorResponse(serialized_value=cloudpickle.dumps(result))

        except Exception as e:
            error = actor_pb2.ActorError(
                error_type=type(e).__name__,
                message=str(e),
                serialized_exception=cloudpickle.dumps(e),
            )
            return actor_pb2.ActorResponse(error=error)

    async def health_check(self, request: actor_pb2.Empty, ctx: RequestContext) -> actor_pb2.HealthResponse:
        return actor_pb2.HealthResponse(healthy=True)

    async def list_methods(
        self, request: actor_pb2.ListMethodsRequest, ctx: RequestContext
    ) -> actor_pb2.ListMethodsResponse:
        """List all methods available on an actor.

        Returns method names, signatures, and docstrings for debugging.
        """
        actor_name = request.actor_name or next(iter(self._actors), "")
        actor = self._actors.get(actor_name)
        if not actor:
            return actor_pb2.ListMethodsResponse()

        methods = []
        for name, method in actor.methods.items():
            try:
                sig = str(inspect.signature(method))
            except (ValueError, TypeError):
                sig = "()"

            docstring = inspect.getdoc(method) or ""

            methods.append(
                actor_pb2.MethodInfo(
                    name=name,
                    signature=sig,
                    docstring=docstring,
                )
            )

        return actor_pb2.ListMethodsResponse(methods=methods)

    async def list_actors(
        self, request: actor_pb2.ListActorsRequest, ctx: RequestContext
    ) -> actor_pb2.ListActorsResponse:
        """List all actors registered with this server.

        Returns actor names, IDs, and registration timestamps for debugging.
        """
        actors = []
        for actor in self._actors.values():
            actors.append(
                actor_pb2.ActorInfo(
                    name=actor.name,
                    actor_id=actor.actor_id,
                    registered_at_ms=actor.registered_at.epoch_ms(),
                    metadata={},
                )
            )

        return actor_pb2.ListActorsResponse(actors=actors)

    def _create_app(self) -> ActorServiceASGIApplication:
        return ActorServiceASGIApplication(service=self)

    def serve_background(self, port: int | None = None) -> int:
        """Start server in background thread.

        Args:
            port: Port to bind to. If None, uses self._port from __init__.
                  If that's also None or 0, auto-assigns a free port.

        Returns:
            Actual port the server is listening on
        """
        if port is not None:
            bind_port = port
        elif self._port is not None:
            bind_port = self._port
        else:
            bind_port = 0  # Auto-assign

        self._app = self._create_app()

        if bind_port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self._actual_port = s.getsockname()[1]
        else:
            self._actual_port = bind_port

        assert self._actual_port is not None
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._actual_port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)

        self._threads.spawn_server(self._server, name=f"actor-server-{self._actual_port}")

        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: self._server.started,
            timeout=Duration.from_seconds(5.0),
        )

        return self._actual_port

    def wait(self) -> None:
        """Block until the server exits."""
        self._threads.wait()

    def stop(self) -> None:
        """Stop the actor server and wait for threads to exit."""
        self._threads.stop()
