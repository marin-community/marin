# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface of the broker.

Sync handlers run on one of three bounded thread pools, chosen by what they
wait on -- never on anyio's shared default pool:

  control   in-memory only: heartbeats, capacity, snapshot metadata, planning
  host_io   one bounded host call each: get/delete/list sandboxes
  create    the create call to a host, which can take minutes on a cold or sick host

On 2026-09-29 every sync handler shared the default pool. Creates to hosts whose
``nerdctl run`` hung for 600 s pinned it, and host heartbeats -- microseconds of
work -- queued behind them for longer than the hosts' 15 s timeout. Every host
then looked dead and placement collapsed to 0-2 of 42 hosts. Heartbeats now run
on the control pool, which no host I/O ever enters.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import anyio
import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from silo.auth import HEADER_API_TOKEN, bearer, verify_bearer
from silo.broker.core import PLACEMENT_POLL_SECONDS, RETRY_AFTER_SECONDS, Broker, HostClient, PlaceAgainLater
from silo.errors import SiloError
from silo.http import HttpClient
from silo.model import ResourceProfile
from silo.wire import error_body

logger = logging.getLogger(__name__)


class HttpHostClient:
    """``HostClient`` over a host's HTTP API, authenticated as the broker."""

    def __init__(self, url: str, host_secret: str) -> None:
        self._http = HttpClient(url, headers={HEADER_API_TOKEN: bearer(host_secret)}, timeout=120)

    def create_sandbox(self, body: Mapping[str, Any]) -> dict[str, Any]:
        # Image pulls happen inside create on a cold host; allow for them.
        return self._http.post("/sandboxes", dict(body), timeout=1800)

    def get_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        return self._http.get(f"/sandboxes/{sandbox_id}")

    def list_sandboxes(self) -> list[dict[str, Any]]:
        return self._http.get("/sandboxes")["items"]

    def delete_sandbox(self, sandbox_id: str) -> None:
        self._http.delete(f"/sandboxes/{sandbox_id}")

    def ensure_image(self, ref: str) -> None:
        self._http.post("/images/ensure", {"ref": ref}, timeout=1800)

    def build_image(self, tag: str, dockerfile: str) -> None:
        self._http.post("/images/build", {"tag": tag, "dockerfile": dockerfile}, timeout=3600)

    def capacity(self) -> dict[str, Any]:
        # Short: a refresh runs while creates wait, and a slow host must not
        # hold up the others.
        return self._http.get("/capacity", timeout=5)


@dataclass(frozen=True)
class PoolSizes:
    """Thread-pool sizes. The sum bounds the broker's request threads."""

    control: int = 32
    host_io: int = 128
    # At most max_inflight_creates_per_host per host are ever in flight, so this
    # only binds with more than create/8 hosts.
    create: int = 384

    ENV: ClassVar[dict[str, str]] = {
        "control": "SILO_BROKER_CONTROL_THREADS",
        "host_io": "SILO_BROKER_HOST_IO_THREADS",
        "create": "SILO_BROKER_CREATE_THREADS",
    }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PoolSizes:
        environ = os.environ if environ is None else environ
        defaults = cls()
        return cls(**{attr: max(1, int(environ.get(name) or getattr(defaults, attr))) for attr, name in cls.ENV.items()})


class _Pools:
    """Capacity limiters, created lazily inside the server's event loop."""

    def __init__(self, sizes: PoolSizes) -> None:
        self.sizes = sizes
        self._limiters: dict[str, anyio.CapacityLimiter] = {}

    def limiter(self, name: str) -> anyio.CapacityLimiter:
        limiter = self._limiters.get(name)
        if limiter is None:
            limiter = anyio.CapacityLimiter(getattr(self.sizes, name))
            self._limiters[name] = limiter
        return limiter

    async def run(self, name: str, func: Callable[..., Any], *args: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(func, *args), limiter=self.limiter(name))

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            name: {"total": limiter.total_tokens, "borrowed": limiter.borrowed_tokens}
            for name, limiter in self._limiters.items()
        }


def _error_response(error: SiloError) -> JSONResponse:
    status = error.status_code or 500
    headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if status == 503 else None
    return JSONResponse(error_body(error), status_code=status, headers=headers)


def build_app(
    broker: Broker,
    *,
    api_token: str,
    host_secret: str,
    self_url: Callable[[], str],
    pool_sizes: PoolSizes | None = None,
) -> Starlette:
    pools = _Pools(pool_sizes or PoolSizes())

    def guard(kind: str, body: bool = False, pool: str = "control"):
        def decorate(handler: Callable[..., Response]):
            async def wrapped(request: Request) -> Response:
                secret = api_token if kind == "api" else host_secret
                if not verify_bearer(secret, request.headers.get(HEADER_API_TOKEN)):
                    return JSONResponse({"error": "Unauthorized", "message": f"{kind} credential required"}, 401)
                payload = await request.json() if body else None
                try:
                    return await pools.run(pool, handler, request, payload)
                except SiloError as error:
                    return _error_response(error)
                except Exception as error:
                    logger.exception("unhandled error in %s", handler.__name__)
                    return JSONResponse(
                        {"error": type(error).__name__, "message": str(error)[:2000], "status_code": 500}, 500
                    )

            wrapped.__name__ = handler.__name__
            return wrapped

        return decorate

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "pools": pools.stats()})

    async def whoami(_: Request) -> Response:
        # Reached through the Iris proxy by NAME, this tells a worker the broker's
        # current direct address -- how a worker recovers after a broker restart
        # instead of being stranded on a dead address.
        return JSONResponse({"url": self_url()})

    @guard("api")
    def capacity(_: Request, __: Any) -> Response:
        return JSONResponse({**broker.capacity(), "pools": pools.stats()})

    @guard("host", body=True, pool="control")
    def heartbeat(_: Request, body: dict[str, Any]) -> Response:
        broker.heartbeat(
            body["host_id"],
            body["url"],
            body["capacity"],
            body.get("sandbox_ids") or [],
            images=body.get("images"),
            heartbeat_seconds=body.get("heartbeat_seconds"),
            heartbeat_timeout_seconds=body.get("heartbeat_timeout_seconds"),
        )
        return JSONResponse({"ok": True})

    @guard("api", body=True)
    def create_snapshot(_: Request, body: dict[str, Any]) -> Response:
        profile = ResourceProfile(cpu=int(body["cpu"]), memory_gb=int(body["memory_gb"]), disk_gb=int(body["disk_gb"]))
        record = broker.create_snapshot(body["name"], body["dockerfile_content"], profile)
        return JSONResponse(record.to_dict(), status_code=201)

    @guard("api")
    def get_snapshot(request: Request, _: Any) -> Response:
        return JSONResponse(broker.get_snapshot(request.path_params["name"]).to_dict())

    @guard("api")
    def list_snapshots(request: Request, _: Any) -> Response:
        page = int(request.query_params.get("page", 1))
        limit = int(request.query_params.get("limit", 100))
        return JSONResponse(broker.list_snapshots(page, limit))

    @guard("api")
    def delete_snapshot(request: Request, _: Any) -> Response:
        broker.delete_snapshot(request.path_params["name"])
        return Response(status_code=204)

    @guard("api")
    def snapshot_logs(request: Request, _: Any) -> Response:
        return PlainTextResponse(broker.snapshot_build_logs(request.path_params["name"]))

    async def create_sandbox(request: Request) -> Response:
        """Create, waiting for room with an async sleep -- never a held thread.

        Written out rather than via ``guard`` because the wait must not run on
        a thread pool: a queue of waiting creates would otherwise fill it and
        block the deletes that free room (width test 01). The create call itself
        runs on the ``create`` pool, so however long hosts take, heartbeats and
        deletes are never behind it.
        """
        if not verify_bearer(api_token, request.headers.get(HEADER_API_TOKEN)):
            return JSONResponse({"error": "Unauthorized", "message": "api credential required"}, 401)
        body = await request.json()
        waiting = False
        plan = None
        try:
            plan = await pools.run(
                "control",
                lambda: broker.plan_create(
                    snapshot=body["snapshot"],
                    labels=body.get("labels") or {},
                    ttl_minutes=int(body.get("ttl_minutes") or 180),
                    network_block_all=bool(body.get("network_block_all", True)),
                    runtime=body.get("runtime"),
                ),
            )
            timeout = float(body.get("timeout") or 600)
            deadline = broker.placement_deadline(timeout)
            last_failure: PlaceAgainLater | None = None
            while True:
                placed = broker.try_place(plan)  # lock-only, no I/O: fine on the loop
                if placed is not None:
                    try:
                        created = await pools.run("create", broker.attempt_create, plan, *placed)
                    except PlaceAgainLater as failure:
                        # A sick host: the failure is counted against it; pause, place again.
                        last_failure = failure
                        if broker.now() >= deadline:
                            raise broker.capacity_exhausted(timeout, last_failure) from None
                        await anyio.sleep(PLACEMENT_POLL_SECONDS)
                        continue
                    if created is not None:
                        return JSONResponse(created, status_code=201)
                    if broker.now() >= deadline:
                        raise broker.capacity_exhausted(timeout, last_failure)
                    continue
                if not waiting:
                    waiting = True
                    broker.note_waiting(+1, plan.record.profile)
                if broker.now() >= deadline:
                    raise broker.capacity_exhausted(timeout, last_failure)
                broker.kick_refresh()  # starts a background refresh if due; never blocks
                await anyio.sleep(PLACEMENT_POLL_SECONDS)
        except SiloError as error:
            return _error_response(error)
        except Exception as error:
            logger.exception("unhandled error in create_sandbox")
            return JSONResponse({"error": type(error).__name__, "message": str(error)[:2000], "status_code": 500}, 500)
        finally:
            if waiting:
                broker.note_waiting(-1)

    @guard("api", pool="host_io")
    def list_sandboxes(_: Request, __: Any) -> Response:
        return JSONResponse({"items": broker.list_sandboxes()})

    @guard("api", pool="host_io")
    def get_sandbox(request: Request, _: Any) -> Response:
        return JSONResponse(broker.get_sandbox(request.path_params["sandbox_id"]))

    @guard("api", pool="host_io")
    def delete_sandbox(request: Request, _: Any) -> Response:
        broker.delete_sandbox(request.path_params["sandbox_id"])
        return JSONResponse({"deleting": True}, status_code=202)

    routes = [
        Route("/health", health),
        Route("/whoami", whoami),
        Route("/capacity", capacity),
        Route("/hosts/heartbeat", heartbeat, methods=["POST"]),
        Route("/snapshots", create_snapshot, methods=["POST"]),
        Route("/snapshots", list_snapshots, methods=["GET"]),
        Route("/snapshots/{name}", get_snapshot, methods=["GET"]),
        Route("/snapshots/{name}", delete_snapshot, methods=["DELETE"]),
        Route("/snapshots/{name}/build-logs", snapshot_logs, methods=["GET"]),
        Route("/sandboxes", create_sandbox, methods=["POST"]),
        Route("/sandboxes", list_sandboxes, methods=["GET"]),
        Route("/sandboxes/{sandbox_id}", get_sandbox, methods=["GET"]),
        Route("/sandboxes/{sandbox_id}", delete_sandbox, methods=["DELETE"]),
    ]
    return Starlette(routes=routes)


def host_client_factory(host_secret: str) -> Callable[[str], HostClient]:
    return lambda url: HttpHostClient(url, host_secret)
