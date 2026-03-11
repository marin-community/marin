# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared dashboard components for controller and worker dashboards.

Serves Vue-built assets from the dashboard/dist directory.
"""

from pathlib import Path
from typing import Any

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

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


def _vue_dist_dir() -> Path:
    """Return the Vue dist directory, checking local dev and Docker paths."""
    for candidate in [VUE_DIST_DIR, DOCKER_VUE_DIST_DIR]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Vue dashboard dist not found at {VUE_DIST_DIR} or {DOCKER_VUE_DIST_DIR}. "
        "Run `iris build dashboard` to build the frontend assets."
    )


def static_files_mount() -> Mount:
    """Mount for serving static JS/CSS assets from the Vue dashboard build."""
    static_dir = _vue_dist_dir() / "static"
    return Mount("/static", app=_CacheControlStaticFiles(StaticFiles(directory=static_dir)), name="static")


def html_shell(title: str, dashboard_type: str = "controller") -> str:
    """Return the pre-built HTML page for a dashboard.

    Vue Router handles all client-side routing, so every route within
    a dashboard type serves the same HTML.
    """
    index_path = _vue_dist_dir() / f"{dashboard_type}.html"
    return index_path.read_text()
