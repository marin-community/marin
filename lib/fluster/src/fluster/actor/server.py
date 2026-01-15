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

ActorServer hosts actor instances and handles RPC calls. It integrates with
FlusterContext for:
- Getting allocated ports via ctx.get_port("actor")
- Registering endpoints via ctx.controller.endpoint_registry

Example:
    # In a job entrypoint:
    server = ActorServer()
    server.register("my-actor", MyActorClass())
    server.serve_and_register("my-actor")  # Uses fluster_ctx() for port + registration

    # Actor methods can access context via fluster_ctx():
    class MyActor:
        def my_method(self):
            ctx = fluster_ctx()
            print(f"Running in job {ctx.job_id}")
"""

import inspect
import logging
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NewType

import cloudpickle
import uvicorn

from fluster.rpc import actor_pb2
from fluster.rpc.actor_connect import ActorServiceASGIApplication
from fluster.context import FlusterContext, fluster_ctx_scope, get_fluster_ctx
from connectrpc.request import RequestContext

logger = logging.getLogger(__name__)

# Type aliases
ActorId = NewType("ActorId", str)


@dataclass
class RegisteredActor:
    """An actor registered with the server."""

    name: str
    actor_id: ActorId
    instance: Any
    methods: dict[str, Callable]
    registered_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class ActorServer:
    """Server for hosting actor instances and handling RPC calls.

    Integrates with FlusterContext for port allocation and endpoint registration.
    Actor methods can access the context via fluster_ctx().
    """

    def __init__(self, host: str = "0.0.0.0", port: int | None = None):
        """Initialize the actor server.

        Args:
            host: Host address to bind to
            port: Port to bind to. If None, uses ctx.get_port("actor") from
                  the FlusterContext when serve_and_register() is called.
                  Use 0 for auto-assign without context.
        """
        self._host = host
        self._port = port
        self._actors: dict[str, RegisteredActor] = {}
        self._fluster_context: FlusterContext | None = None
        self._app: ActorServiceASGIApplication | None = None
        self._actual_port: int | None = None
        self._endpoint_id: str | None = None

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

            # Run method with FlusterContext if available
            if self._fluster_context:
                with fluster_ctx_scope(self._fluster_context):
                    result = method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)

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
                    registered_at_ms=actor.registered_at_ms,
                    metadata={},
                )
            )

        return actor_pb2.ListActorsResponse(actors=actors)

    def _create_app(self) -> ActorServiceASGIApplication:
        """Create the Connect RPC ASGI application for the server."""
        return ActorServiceASGIApplication(service=self)

    def serve_background(self, port: int | None = None) -> int:
        """Start server in background thread.

        Args:
            port: Port to bind to. If None, uses self._port (which may have
                  been set in __init__ or will use ctx.get_port("actor")).

        Returns:
            Actual port the server is listening on
        """
        # Get context if available
        self._fluster_context = get_fluster_ctx()

        # Determine port: explicit > __init__ > context > auto-assign
        if port is not None:
            bind_port = port
        elif self._port is not None:
            bind_port = self._port
        elif self._fluster_context:
            try:
                bind_port = self._fluster_context.get_port("actor")
            except KeyError:
                bind_port = 0  # Auto-assign
        else:
            bind_port = 0  # Auto-assign

        self._app = self._create_app()

        # Find available port if port=0
        if bind_port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self._actual_port = s.getsockname()[1]
        else:
            self._actual_port = bind_port

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

    def shutdown(self) -> None:
        """Unregister endpoint from controller.

        Call this when shutting down the actor server cleanly.
        Note: The controller also automatically cleans up endpoints when jobs terminate.
        """
        if self._endpoint_id and self._fluster_context and self._fluster_context.controller:
            try:
                self._fluster_context.controller.endpoint_registry.unregister(self._endpoint_id)
                logger.info(f"Unregistered endpoint {self._endpoint_id}")
            except Exception as e:
                logger.warning(f"Failed to unregister endpoint: {e}")
            self._endpoint_id = None
