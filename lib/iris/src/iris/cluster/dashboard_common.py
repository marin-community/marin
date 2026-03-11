# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared dashboard components for controller and worker dashboards.

Serves Vue-built assets from the dashboard/dist directory when available,
falling back to legacy Preact assets from the static/ directory.
"""

from pathlib import Path
from typing import Any

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

STATIC_DIR = Path(__file__).parent / "static"

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


def _resolve_vue_dist() -> Path | None:
    """Return the Vue dist directory if it exists and contains built assets."""
    for candidate in [VUE_DIST_DIR, DOCKER_VUE_DIST_DIR]:
        if candidate.is_dir() and (candidate / "controller.html").exists():
            return candidate
    return None


def static_files_mount() -> Mount:
    """Mount for serving static JS/CSS assets.

    Serves from the Vue dist/static directory when the Vue build output
    is available, otherwise falls back to the legacy Preact static directory.
    """
    vue_dist = _resolve_vue_dist()
    if vue_dist:
        static_dir = vue_dist / "static"
    else:
        static_dir = STATIC_DIR
    return Mount("/static", app=_CacheControlStaticFiles(StaticFiles(directory=static_dir)), name="static")


def html_shell(title: str, dashboard_type: str = "controller") -> str:
    """Return the HTML page for a dashboard.

    When the Vue build output is available, returns the pre-built HTML file
    (controller.html or worker.html). Vue Router handles all client-side
    routing, so every route within a dashboard type serves the same HTML.

    Falls back to the legacy Preact importmap shell when no Vue build exists.
    """
    vue_dist = _resolve_vue_dist()
    if vue_dist:
        index_path = vue_dist / f"{dashboard_type}.html"
        return index_path.read_text()
    app_script = f"/static/{dashboard_type}/app.js"
    return _legacy_html_shell(title, app_script)


def _legacy_html_shell(title: str, app_script: str) -> str:
    """Preact importmap HTML shell used when no Vue build is available."""
    return f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"><title>{title}</title>
  <link rel="stylesheet" href="/static/shared/styles.css">
</head><body>
  <div id="root"></div>
  <script type="importmap">{{"imports": {{
    "preact": "/static/vendor/preact.mjs",
    "preact/hooks": "/static/vendor/preact-hooks.mjs",
    "htm": "/static/vendor/htm.mjs"
  }}}}</script>
  <script type="module" src="{app_script}"></script>
</body></html>"""
