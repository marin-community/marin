# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared dashboard components for controller and worker dashboards."""

from pathlib import Path
from typing import Any

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

STATIC_DIR = Path(__file__).parent / "static"

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


def static_files_mount() -> Mount:
    """Mount for serving static JS/CSS assets (vendor libs, shared utils, app components)."""
    return Mount("/static", app=_CacheControlStaticFiles(StaticFiles(directory=STATIC_DIR)), name="static")


def html_shell(title: str, app_script: str) -> str:
    """Return an HTML shell that loads a Preact app via ES module importmap."""
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
