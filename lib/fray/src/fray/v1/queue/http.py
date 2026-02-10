# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Queue implemented using a FastAPI server and HTTP requests.
"""

import logging
import pickle
import threading
import time
from typing import Any

import httpx
from fastapi import Body, FastAPI, Request, Response
from fray.v1.queue.base import Lease, MemoryQueue

logging.getLogger("httpx").setLevel(logging.WARNING)


class ServerThread:
    """Helper class to run uvicorn server in a background thread."""

    def __init__(self, server, host: str, port: int):
        self.server = server
        self.host = host
        self.port = port
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 5.0):
        """Wait for server to be ready to accept connections."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with httpx.Client() as client:
                    client.get(f"http://{self.host}:{self.port}/docs", timeout=1.0)
                return
            except (httpx.ConnectError, httpx.TimeoutException):
                time.sleep(1.0)

    def shutdown(self):
        self.server.should_exit = True


class HttpQueueServer:
    """HTTP server that manages multiple named queues.

    Example:
        with HttpQueueServer(host="0.0.0.0", port=9999) as server:
            queue_a = server.new_queue("tasks")
            queue_b = server.new_queue("results")
            queue_a.push("task1")
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9999):
        self.host = host
        self.port = port
        self.queues: dict[str, MemoryQueue] = {}
        self.app = self._create_app()

        import uvicorn

        config = uvicorn.Config(self.app, host=host, port=port, log_level="error", access_log=False)
        self.server = uvicorn.Server(config)
        self.server_thread: ServerThread | None = None

    def _create_app(self) -> FastAPI:
        """Create FastAPI app with namespaced queue endpoints."""
        app = FastAPI()

        @app.post("/queues/{queue_name}/push")
        async def push(queue_name: str, request: Request):
            if queue_name not in self.queues:
                return Response(status_code=404)
            payload = await request.body()
            item = pickle.loads(payload)
            self.queues[queue_name].push(item)
            return {"status": "ok"}

        @app.post("/queues/{queue_name}/pop")
        def pop(queue_name: str, lease_timeout: float = Body(default=60.0, embed=True)):
            if queue_name not in self.queues:
                return Response(status_code=404)
            lease = self.queues[queue_name].pop(lease_timeout)
            if lease is None:
                return Response(status_code=204)
            return {"lease_id": lease.lease_id, "timestamp": lease.timestamp, "payload": pickle.dumps(lease.item).hex()}

        @app.post("/queues/{queue_name}/done")
        def done(queue_name: str, lease_id: str = Body(...), timestamp: float = Body(...)):
            if queue_name not in self.queues:
                return Response(status_code=404)
            lease = Lease(item=None, lease_id=lease_id, timestamp=timestamp)
            self.queues[queue_name].done(lease)
            return {"status": "ok"}

        @app.post("/queues/{queue_name}/release")
        def release(queue_name: str, lease_id: str = Body(...), timestamp: float = Body(...)):
            if queue_name not in self.queues:
                return Response(status_code=404)
            lease = Lease(item=None, lease_id=lease_id, timestamp=timestamp)
            self.queues[queue_name].release(lease)
            return {"status": "ok"}

        return app

    def get_client_host(self) -> str:
        """Get the hostname/IP that clients should use to connect.

        When server binds to 0.0.0.0, clients need a specific hostname/IP.
        Returns the actual IP address using default route.
        """
        if self.host == "0.0.0.0":
            import socket

            # Get the IP address that clients should use by checking default route
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))  # doesn't actually send anything
                ip = s.getsockname()[0]
                s.close()
                return ip
            except Exception:
                # Fall back to localhost for local testing
                return "127.0.0.1"
        return self.host

    def new_queue(self, name: str) -> "HttpQueue":
        """Create or get a named queue, returns client."""
        if name not in self.queues:
            self.queues[name] = MemoryQueue()
        # Use client-accessible host instead of bind host
        client_host = self.get_client_host()
        return HttpQueue(host=client_host, port=self.port, queue_name=name)

    def __enter__(self):
        self.server_thread = ServerThread(self.server, self.host, self.port)
        self.server_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.server_thread:
            self.server_thread.shutdown()
        return False


class HttpQueue:
    """Thin HTTP queue client that connects to an HttpQueueServer."""

    def __init__(self, host: str, port: int, queue_name: str):
        self.host = host
        self.port = port
        self.queue_name = queue_name

    def __getstate__(self):
        return {"host": self.host, "port": self.port, "queue_name": self.queue_name}

    def __setstate__(self, state):
        self.host = state["host"]
        self.port = state["port"]
        self.queue_name = state["queue_name"]

    def push(self, item: Any) -> None:
        with httpx.Client() as client:
            client.post(f"http://{self.host}:{self.port}/queues/{self.queue_name}/push", content=pickle.dumps(item))

    def peek(self) -> Any | None:
        with httpx.Client() as client:
            response = client.get(f"http://{self.host}:{self.port}/queues/{self.queue_name}/peek")
            if response.status_code == 200:
                return pickle.loads(response.json()["payload"])
            return None

    def pop(self, lease_timeout: float = 60.0) -> Lease[Any] | None:
        with httpx.Client() as client:
            response = client.post(
                f"http://{self.host}:{self.port}/queues/{self.queue_name}/pop", json={"lease_timeout": lease_timeout}
            )
            if response.status_code == 200:
                data = response.json()
                return Lease(
                    item=pickle.loads(bytes.fromhex(data["payload"])),
                    lease_id=data["lease_id"],
                    timestamp=data["timestamp"],
                )
            return None

    def done(self, lease: Lease[Any]) -> None:
        with httpx.Client() as client:
            client.post(
                f"http://{self.host}:{self.port}/queues/{self.queue_name}/done",
                json={"lease_id": lease.lease_id, "timestamp": lease.timestamp},
            )

    def release(self, lease: Lease[Any]) -> None:
        with httpx.Client() as client:
            client.post(
                f"http://{self.host}:{self.port}/queues/{self.queue_name}/release",
                json={"lease_id": lease.lease_id, "timestamp": lease.timestamp},
            )
