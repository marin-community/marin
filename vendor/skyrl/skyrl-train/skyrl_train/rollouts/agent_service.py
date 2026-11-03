"""HTTP service lifetime for installed-agent completion callbacks."""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI


class _ReadyServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config):
        super().__init__(config)
        self.ready = asyncio.Event()

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        if self.started:
            self.ready.set()


@asynccontextmanager
async def serve_agent_app(
    app: FastAPI, *, host: str, startup_timeout: float, shutdown_timeout: float
) -> AsyncIterator[int]:
    """Serve on a reserved port in the caller's loop, then release connections.

    Agent completion callbacks share queues and futures with the rollout engine;
    a separate HTTP-server thread would put those objects on different loops.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        listener.listen()
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = _ReadyServer(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
                timeout_graceful_shutdown=shutdown_timeout,
            )
        )
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        ready = asyncio.create_task(server.ready.wait())
        try:
            async with asyncio.timeout(startup_timeout):
                done, _ = await asyncio.wait((serving, ready), return_when=asyncio.FIRST_COMPLETED)
                if serving in done:
                    await serving
                    raise RuntimeError("Agent HTTP server stopped before startup completed")
            yield port
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)
            server.should_exit = True
            try:
                async with asyncio.timeout(shutdown_timeout):
                    await asyncio.shield(serving)
            except TimeoutError:
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)
                raise
