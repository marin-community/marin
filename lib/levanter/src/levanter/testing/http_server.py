# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a local TCP test server with an explicit readiness signal."""

import asyncio
import socket
from contextlib import asynccontextmanager

import uvicorn


class _ReadyHttpServer(uvicorn.Server):
    def __init__(self, app, ready):
        super().__init__(uvicorn.Config(app, log_level="error", lifespan="off"))
        self.ready = ready

    async def startup(self, sockets=None):
        await super().startup(sockets)
        self.ready.set()


@asynccontextmanager
async def live_http_server(app):
    ready = asyncio.Event()
    http_server = _ReadyHttpServer(app, ready)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        serving = asyncio.create_task(http_server.serve(sockets=[listener]))
        try:
            await asyncio.wait_for(ready.wait(), 10)
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            http_server.should_exit = True
            await serving
