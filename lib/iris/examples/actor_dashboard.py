# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one actor with Connect RPC and an HTTP dashboard on one port."""

import threading

from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Route
from starlette.types import ASGIApp


class CounterActor:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()
        self._web_application = Starlette(
            routes=[
                Route("/", self._dashboard),
                Route("/increment", self._increment_from_http, methods=["POST"]),
            ]
        )

    @property
    def web_application(self) -> ASGIApp:
        return self._web_application

    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value

    def counter_value(self) -> int:
        with self._lock:
            return self._value

    async def _dashboard(self, _request: Request) -> HTMLResponse:
        value = self.counter_value()
        return HTMLResponse(
            f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Iris actor dashboard</title></head>
<body>
  <h1>Iris actor dashboard</h1>
  <p>Counter value: {value}</p>
  <form action="./increment" method="post"><button type="submit">Increment</button></form>
</body>
</html>
"""
        )

    async def _increment_from_http(self, _request: Request) -> RedirectResponse:
        self.increment()
        return RedirectResponse("./", status_code=303)


def main() -> None:
    actor_name = "counter"
    server = ActorServer(host="127.0.0.1")
    server.register(actor_name, CounterActor())
    port = server.serve_background()
    address = f"http://127.0.0.1:{port}"
    client = ActorClient(FixedResolver({actor_name: address}), actor_name)

    try:
        print(f"RPC increment result: {client.increment(2)}")
        print(f"HTTP dashboard: {address}/")
        print("Push Ctrl+C to stop.")
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
