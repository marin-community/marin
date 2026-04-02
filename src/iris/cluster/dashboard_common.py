# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared dashboard components for controller and worker dashboards.

Serves Vue-built assets from the dashboard/dist directory. All dist
lookups are deferred to request time so the server can start even when
the frontend hasn't been built yet (e.g. in tests or local dev).
"""

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


def on_shutdown(
    *callbacks: Callable[[], Awaitable[None]],
) -> Callable[[Starlette], AbstractAsyncContextManager[None]]:
    """Build a Starlette ``lifespan`` that runs *callbacks* on shutdown."""

    @asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
        yield
        for cb in callbacks:
            await cb()

    return _lifespan


# Vue dashboard build output. The path from this file (cluster/dashboard_common.py)
# up to lib/iris/ is four parent directories, then down into dashboard/dist.
VUE_DIST_DIR = Path(__file__).parent.parent.parent.parent / "dashboard" / "dist"
DOCKER_VUE_DIST_DIR = Path("/app/dashboard/dist")

# Allow browsers to cache static assets for up to 10 minutes before revalidating.
STATIC_MAX_AGE_SECONDS = 600


class _CacheControlStaticFiles:
    """Wraps a StaticFiles app to inject a Cache-Control header on every response."""

    def __init__(self, app: ASGIApp, max_age: int = STATIC_MAX_AGE_SECONDS) -> None:
        self._app = app
        self._cache_header = f"public, max-age={max_age}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_cache(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", self._cache_header))
                message["headers"] = headers
            await send(message)

        await self._app(scope, receive, send_with_cache)


def _vue_dist_dir() -> Path | None:
    """Return the Vue dist directory if it exists, or None."""
    for candidate in [VUE_DIST_DIR, DOCKER_VUE_DIST_DIR]:
        if candidate.is_dir():
            return candidate
    return None


class _LazyStaticFiles:
    """Serves static files from the Vue dist, resolving the path at request time.

    Returns 503 if the dist hasn't been built yet rather than crashing at startup.
    """

    def __init__(self, max_age: int = STATIC_MAX_AGE_SECONDS) -> None:
        self._max_age = max_age
        self._inner: ASGIApp | None = None
        self._checked = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope["type"] != "http":
            return

        if not self._checked:
            dist = _vue_dist_dir()
            if dist and (dist / "static").is_dir():
                self._inner = _CacheControlStaticFiles(StaticFiles(directory=dist / "static"), self._max_age)
            self._checked = True

        inner = self._inner
        if inner is None:
            resp = PlainTextResponse("Dashboard assets not built. Run `iris build dashboard`.", status_code=503)
            await resp(scope, receive, send)
            return

        await inner(scope, receive, send)


def static_files_mount() -> Mount:
    """Mount for serving static JS/CSS assets from the Vue dashboard build.

    Resolution is deferred to request time so the server can start without
    the dashboard dist being present.
    """
    return Mount("/static", app=_LazyStaticFiles(), name="static")


_NOT_BUILT_HTML = """\
<!doctype html>
<html><body>
<h1>Dashboard not built</h1>
<p>Run <code>iris build dashboard</code> to build the frontend assets.</p>
</body></html>
"""


def html_shell(title: str, dashboard_type: str = "controller") -> str:
    """Return the pre-built HTML page for a dashboard.

    Vue Router handles all client-side routing, so every route within
    a dashboard type serves the same HTML. Returns a placeholder page
    if the dist hasn't been built.
    """
    dist = _vue_dist_dir()
    if dist is None:
        logger.warning("Vue dashboard dist not found; serving placeholder page")
        return _NOT_BUILT_HTML
    index_path = dist / f"{dashboard_type}.html"
    if not index_path.exists():
        logger.warning("Dashboard HTML %s not found; serving placeholder", index_path)
        return _NOT_BUILT_HTML
    return index_path.read_text()
