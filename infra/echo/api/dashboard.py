# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the compiled Echo dashboard from the API process."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


def dashboard_dist() -> Path:
    here = Path(__file__).resolve()
    candidates = (here.parent / "dist", here.parents[1] / "dashboard" / "dist")
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def install_dashboard(app: FastAPI, dist: Path) -> None:
    app.mount("/static", StaticFiles(directory=dist / "static", check_dir=False), name="static")

    @app.get("/{_full_path:path}", include_in_schema=False, response_model=None)
    def dashboard(_full_path: str) -> FileResponse | HTMLResponse:
        index = dist / "index.html"
        if not index.is_file():
            return HTMLResponse(
                "<h1>Echo</h1><p>Dashboard not built. Run npm --prefix infra/echo/dashboard run build.</p>",
                status_code=503,
            )
        return FileResponse(index)
