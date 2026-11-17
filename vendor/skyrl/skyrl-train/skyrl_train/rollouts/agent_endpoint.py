"""Leased Iris routes for Daytona agents, without container exec RPCs."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from iris.client.client import IrisClient, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess
from rigging.timing import Duration


@dataclass(frozen=True)
class AgentEndpoint:
    name: str
    client: IrisClient

    async def api_base(self, *, required_lifetime: float) -> str:
        """Mint a fresh route for one trial; never log or persist this credential."""
        response = await asyncio.to_thread(
            self.client.mint_endpoint_token,
            self.name,
            ttl=Duration.from_seconds(required_lifetime + 60),
        )
        if not response.capability_url:
            raise RuntimeError("The Iris controller has no public route for installed agents")
        if response.expires_at.epoch_ms / 1000 - time.time() < required_lifetime:
            raise ValueError("The Iris endpoint token lifetime is shorter than the configured agent attempt")
        return response.capability_url.rstrip("/") + "/v1"


@asynccontextmanager
async def registered_agent_endpoint(port: int, *, request_timeout: float) -> AsyncIterator[AgentEndpoint]:
    """Keep the controller route renewed until the HTTP service stops."""
    context = iris_ctx()
    info = get_job_info()
    if info is None or context.client is None:
        raise RuntimeError("Installed-agent ingress requires an Iris task and controller client")
    if info.advertise_host in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}:
        raise ValueError("Installed-agent ingress requires a controller-reachable advertise host")
    name = f"/native-agent-{uuid4().hex}"
    registry = context.registry
    registration = asyncio.create_task(
        asyncio.to_thread(
            registry.register,
            name,
            f"http://{info.advertise_host}:{port}",
            metadata={PROXY_TIMEOUT_METADATA_KEY: str(request_timeout)},
            access=EndpointAccess.ENDPOINT_ACCESS_LINK,
        )
    )
    try:
        await asyncio.shield(registration)
        yield AgentEndpoint(name, context.client)
    finally:
        # Cancellation cannot stop the synchronous RPC. Wait for its result so
        # a registration that completes after cancellation is still removed.
        endpoint_id = await registration
        await asyncio.to_thread(registry.unregister, endpoint_id)
