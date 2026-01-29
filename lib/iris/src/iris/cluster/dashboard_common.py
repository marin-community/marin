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

from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

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
