# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

import uvicorn
from iris.cluster.controller.controller import _install_rpc_executor
from iris.managed_thread import ThreadContainer
from rigging.timing import Duration, ExponentialBackoff
from starlette.types import Receive, Scope, Send


class _MarkedEventLoop(asyncio.SelectorEventLoop):
    pass


def _marked_loop_factory() -> asyncio.AbstractEventLoop:
    return _MarkedEventLoop()


def test_controller_server_honors_configured_loop_and_rpc_executor() -> None:
    observed: list[tuple[str, str]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["type"] == "lifespan"
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                executor_thread = await asyncio.to_thread(lambda: threading.current_thread().name)
                observed.append((type(asyncio.get_running_loop()).__name__, executor_thread))
                await send({"type": "lifespan.startup.complete"})
                continue
            await send({"type": "lifespan.shutdown.complete"})
            return

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        lifespan="on",
        loop="tests.cluster.controller.test_controller_server:_marked_loop_factory",
        log_config=None,
    )
    server = uvicorn.Server(config)
    _install_rpc_executor(server, max_workers=2)
    threads = ThreadContainer()
    try:
        threads.spawn_server(server, name="controller-server-test")
        ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
            lambda: bool(observed),
            timeout=Duration.from_seconds(5),
        )
    finally:
        threads.stop()

    assert observed == [("_MarkedEventLoop", "rpc-handler_0")]
