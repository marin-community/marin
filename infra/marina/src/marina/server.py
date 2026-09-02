# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The kernel process: discover apps, serve their frontends, expose the shared surface.

One FastAPI application. Each app is mounted under ``/<name>/``: files in its ``dist``
are served verbatim and every other path under the prefix answers ``index.html`` so a
client-side route survives a reload. ``/api/marina/*`` is the surface shared by every
app (the app directory and the caller's identity); ``/`` lists the apps. A per-app
Content-Security-Policy restricts what the page may fetch to itself plus the manifest's
``connect_src``.
"""

import html
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from rigging.server_auth import RouteAuthMiddleware, public, requires_auth

from marina.auth import build_policy, identity_for
from marina.manifest import AppManifest, discover_apps

INDEX_FILE = "index.html"
# A file committed as `x.gz` is served at `x` with a Content-Encoding header, so a large
# static asset can live in the repository compressed.
PRECOMPRESSED_SUFFIX = ".gz"
# Cloud Run sets K_SERVICE in every container; its presence means IAP is the front door.
CLOUD_RUN_SERVICE_ENV = "K_SERVICE"
APPS_DIR_ENV = "MARINA_APPS_DIR"
IAP_AUDIENCE_ENV = "MARINA_IAP_AUDIENCE"


@dataclass(frozen=True)
class MarinaConfig:
    apps_dir: Path
    # The IAP JWT audience for this service; None admits loopback callers only.
    iap_audience: str | None

    @classmethod
    def from_env(cls, default_apps_dir: Path) -> "MarinaConfig":
        """Resolve the process configuration once; refuse to start on Cloud Run without IAP."""
        apps_dir = Path(os.environ.get(APPS_DIR_ENV) or default_apps_dir)
        audience = os.environ.get(IAP_AUDIENCE_ENV) or None
        if os.environ.get(CLOUD_RUN_SERVICE_ENV) and not audience:
            raise ValueError(f"{IAP_AUDIENCE_ENV} must be set when running on Cloud Run")
        return cls(apps_dir=apps_dir, iap_audience=audience)


def content_security_policy(app: AppManifest) -> str:
    connect = " ".join(("'self'", *app.connect_src))
    return (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        f"img-src 'self' data:; font-src 'self' data:; connect-src {connect}; frame-ancestors 'none'"
    )


def app_directory(apps: list[AppManifest]) -> list[dict[str, str]]:
    return [{"name": app.name, "title": app.title, "description": app.description, "path": app.path} for app in apps]


def landing_page(apps: list[AppManifest]) -> str:
    rows = "".join(
        f'<li><a href="{app.path}">{html.escape(app.title)}</a>' f"<span>{html.escape(app.description)}</span></li>"
        for app in apps
    )
    return (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Marina</title>"
        "<style>body{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem}"
        "h1{font-size:1.2rem}ul{list-style:none;padding:0}li{padding:.6rem 0;border-top:1px solid #ddd}"
        "li a{font-weight:600;display:block}li span{color:#666}"
        "@media(prefers-color-scheme:dark){body{background:#15171a;color:#ece9e1}"
        "li{border-color:#333}li span{color:#999}}"
        "</style><h1>Marina</h1><ul>" + rows + "</ul>"
    )


def serve_app_file(app: AppManifest, path: str) -> Response:
    """A file from the app's dist, or index.html for a client-side route."""
    dist = app.dist.resolve()
    if not (dist / INDEX_FILE).is_file():
        return HTMLResponse(
            f"<h1>{html.escape(app.title)}</h1><p>Frontend not built. Run <code>marina build</code>.</p>",
            status_code=503,
        )
    headers = {"Content-Security-Policy": content_security_policy(app)}
    candidate = (dist / path).resolve() if path else dist / INDEX_FILE
    inside = candidate == dist or dist in candidate.parents
    if not inside:
        return FileResponse(dist / INDEX_FILE, headers=headers)
    if candidate.is_file():
        return FileResponse(candidate, headers=headers)
    compressed = candidate.with_name(candidate.name + PRECOMPRESSED_SUFFIX)
    if compressed.is_file():
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        return FileResponse(compressed, media_type=media_type, headers={**headers, "Content-Encoding": "gzip"})
    return FileResponse(dist / INDEX_FILE, headers=headers)


def install_app_routes(api: FastAPI, app: AppManifest) -> None:
    prefix = app.path.rstrip("/")

    @api.get(prefix, include_in_schema=False)
    @requires_auth
    def app_root() -> RedirectResponse:
        return RedirectResponse(app.path)

    @api.api_route(prefix + "/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    @requires_auth
    def app_file(path: str) -> Response:
        return serve_app_file(app, path)


def create_app(config: MarinaConfig) -> RouteAuthMiddleware:
    apps = discover_apps(config.apps_dir)
    policy = build_policy(config.iap_audience)
    api = FastAPI(title="Marina", docs_url=None, redoc_url=None)

    @api.get("/healthz", include_in_schema=False)
    @public
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "apps": [app.name for app in apps]})

    @api.get("/api/marina/apps")
    @requires_auth
    def list_apps() -> JSONResponse:
        return JSONResponse({"apps": app_directory(apps)})

    @api.get("/api/marina/me")
    @requires_auth
    def me(request: Request) -> JSONResponse:
        identity = identity_for(request, policy)
        return JSONResponse({"user": identity.user_id, "role": identity.role})

    @api.get("/", include_in_schema=False)
    @requires_auth
    def landing() -> HTMLResponse:
        return HTMLResponse(landing_page(apps))

    for app in apps:
        install_app_routes(api, app)

    return RouteAuthMiddleware(api, policy)


def asgi() -> RouteAuthMiddleware:
    """Uvicorn factory entry point: ``uvicorn --factory marina.server:asgi``."""
    return create_app(MarinaConfig.from_env(Path.cwd() / "apps"))
