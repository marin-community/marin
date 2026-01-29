# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared dashboard components for controller and worker dashboards."""

from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from iris.logging import LogBuffer

STATIC_DIR = Path(__file__).parent / "static"


def static_files_mount() -> Mount:
    """Mount for serving static JS/CSS assets (vendor libs, shared utils, app components)."""
    return Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static")


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


def logs_page_response(_request: Request) -> HTMLResponse:
    """Starlette route handler returning the logs HTML page."""
    return HTMLResponse(html_shell("Iris Logs", "/static/logs/app.js"))


def logs_api_response(request: Request, buffer: LogBuffer | None) -> JSONResponse:
    """Starlette route handler returning log records as JSON."""
    if not buffer:
        return JSONResponse([])
    prefix = request.query_params.get("prefix") or None
    try:
        limit = int(request.query_params.get("limit", "200"))
    except (ValueError, TypeError):
        limit = 200
    records = buffer.query(prefix=prefix, limit=limit)
    return JSONResponse(
        [
            {"timestamp": r.timestamp, "level": r.level, "logger_name": r.logger_name, "message": r.message}
            for r in records
        ]
    )
