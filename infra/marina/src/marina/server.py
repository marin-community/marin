# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The kernel process: discover apps, serve their frontends, expose the shared surface.

One FastAPI application. Each app is mounted under ``/<name>/``: files in its ``dist``
are served verbatim and every other path under the prefix answers ``index.html`` so a
client-side route survives a reload. ``/<name>/data/*`` reads from the app's directory
under the data root (a GCS bucket in production, a local directory in development), so
large or changing files stay out of the image and the repository. A Python app's API is
mounted at ``/<name>/api/`` behind the same authentication, with the caller's identity
bound for its handlers. ``/api/marina/*`` is the surface shared by every
app (the app directory and the caller's identity); ``/`` lists the apps. A per-app
Content-Security-Policy restricts what the page may fetch to itself plus the manifest's
``connect_src``.
"""

import html
import mimetypes
import os
import posixpath
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from rigging.filesystem.factory import url_to_fs
from rigging.server_auth import (
    RequestAuthPolicy,
    RouteAuthMiddleware,
    extract_bearer_token,
    identity_scope,
    public,
    requires_auth,
    scope_client_address,
    scope_headers,
)
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from marina.apps import create_api, data_url_for, is_python_app, services_for
from marina.auth import build_policy, identity_for
from marina.db import DatabaseSpec, database_from_env
from marina.manifest import AppManifest, discover_apps

INDEX_FILE = "index.html"
# A file committed as `x.gz` is served at `x` with a Content-Encoding header, so a large
# static asset can live in the repository compressed.
PRECOMPRESSED_SUFFIX = ".gz"
# Cloud Run sets K_SERVICE in every container; its presence means IAP is the front door.
CLOUD_RUN_SERVICE_ENV = "K_SERVICE"
APPS_DIR_ENV = "MARINA_APPS_DIR"
DATA_ROOT_ENV = "MARINA_DATA_ROOT"
IAP_AUDIENCE_ENV = "MARINA_IAP_AUDIENCE"
# `host=app,host=app`: a vanity host that used to be one app's own origin (echo.oa.dev)
# redirects into that app's prefix on the canonical origin, so links from before the move
# keep resolving.
HOST_APPS_ENV = "MARINA_HOST_APPS"
# The origin the apps are served from. Aliased hosts send every request here, so one URL
# space holds the apps and a link from one app to another resolves against this origin.
CANONICAL_ORIGIN_ENV = "MARINA_CANONICAL_ORIGIN"
DATA_PREFIX = "data/"
API_PREFIX = "/api"
DATA_CACHE_CONTROL = "private, max-age=300"


@dataclass(frozen=True)
class MarinaConfig:
    apps_dir: Path
    # An fsspec URL (gs://bucket/prefix or a local directory) holding one directory per app.
    data_root: str
    # The IAP JWT audience for this service; None admits loopback callers only.
    iap_audience: str | None
    # None serves static apps only; a Python app that asks for its engine then fails.
    database: DatabaseSpec | None = None
    # Hosts that redirect into one app's prefix, by host name.
    host_apps: dict[str, str] = field(default_factory=dict)
    # Scheme and host the aliased hosts redirect to, e.g. https://marina.oa.dev.
    canonical_origin: str | None = None

    @classmethod
    def from_env(cls, default_apps_dir: Path) -> "MarinaConfig":
        """Resolve the process configuration once; refuse to start on Cloud Run without IAP."""
        apps_dir = Path(os.environ.get(APPS_DIR_ENV) or default_apps_dir)
        data_root = os.environ.get(DATA_ROOT_ENV)
        if not data_root:
            raise ValueError(f"{DATA_ROOT_ENV} is not set")
        audience = os.environ.get(IAP_AUDIENCE_ENV) or None
        if os.environ.get(CLOUD_RUN_SERVICE_ENV) and not audience:
            raise ValueError(f"{IAP_AUDIENCE_ENV} must be set when running on Cloud Run")
        return cls(
            apps_dir=apps_dir,
            data_root=data_root,
            iap_audience=audience,
            database=database_from_env(os.environ),
            host_apps=parse_host_apps(os.environ.get(HOST_APPS_ENV, "")),
            canonical_origin=(os.environ.get(CANONICAL_ORIGIN_ENV) or "").rstrip("/") or None,
        )


def parse_host_apps(spec: str) -> dict[str, str]:
    """Parse ``host=app,host=app``; blank means no aliases."""
    pairs = [item.strip() for item in spec.split(",") if item.strip()]
    result: dict[str, str] = {}
    for pair in pairs:
        host, sep, app = pair.partition("=")
        if not sep or not host or not app:
            raise ValueError(f"{HOST_APPS_ENV} entry {pair!r} is not host=app")
        result[host.strip().lower()] = app.strip()
    return result


def host_redirect(host: str, path: str, host_apps: dict[str, str], canonical_origin: str) -> str | None:
    """The canonical URL for a request on an aliased host, or None on any other host.

    Every path on an aliased host moves to the canonical origin. Serving the app on the alias
    too would put the same pages at two origins, and a root-relative link to another app
    would take this host's prefix with it: ``/evaldash/`` on echo.oa.dev became
    ``/echo/evaldash/``.

    A path already inside the app keeps the prefix it has, so a link that was written or
    cached against the alias's own prefix does not collect a second copy of it.
    """
    app = host_apps.get(host.split(":")[0].lower())
    if app is None:
        return None
    if path == f"/{app}" or path.startswith(f"/{app}/"):
        return f"{canonical_origin}{path}"
    return f"{canonical_origin}/{app}{'' if path == '/' else path}"


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


def clean_relative_path(path: str) -> str | None:
    """A normalized relative path, or None when it would escape its root."""
    normalized = posixpath.normpath(path)
    if not path or normalized.startswith(("../", "/")) or normalized in ("..", "."):
        return None
    return normalized


async def serve_data_file(app: AppManifest, data_root: str, path: str) -> Response:
    """A file from the app's data directory, gzip-encoded when only ``x.gz`` exists."""
    relative = clean_relative_path(path)
    if relative is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    fs, root = url_to_fs(data_url_for(data_root, app.name))
    target = f"{root}/{relative}"
    headers = {"Content-Security-Policy": content_security_policy(app), "Cache-Control": DATA_CACHE_CONTROL}
    media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    if await run_in_threadpool(fs.isfile, target):
        body = await run_in_threadpool(fs.cat_file, target)
        return Response(body, media_type=media_type, headers=headers)
    if await run_in_threadpool(fs.isfile, target + PRECOMPRESSED_SUFFIX):
        body = await run_in_threadpool(fs.cat_file, target + PRECOMPRESSED_SUFFIX)
        return Response(body, media_type=media_type, headers={**headers, "Content-Encoding": "gzip"})
    return JSONResponse({"error": "not found"}, status_code=404)


class AuthenticatedMount:
    """Wrap a mounted app so every request is authenticated and its identity is bound.

    The route middleware passes mounts through untouched; this is the gate for an app's
    API. Handlers read the caller with ``rigging.server_auth.get_verified_identity``.
    """

    def __init__(self, app: ASGIApp, policy: RequestAuthPolicy):
        self._app = app
        self._policy = policy

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self._app(scope, receive, send)
        headers = scope_headers(scope)
        try:
            identity = self._policy.resolve(
                extract_bearer_token(headers), client_address=scope_client_address(scope), headers=headers
            )
        except ValueError:
            response = JSONResponse({"error": "authentication required"}, status_code=401)
            return await response(scope, receive, send)
        with identity_scope(identity):
            return await self._app(scope, receive, send)


def install_app_routes(api: FastAPI, app: AppManifest, data_root: str) -> None:
    prefix = app.path.rstrip("/")

    @api.get(prefix, include_in_schema=False)
    @requires_auth
    def app_root() -> RedirectResponse:
        return RedirectResponse(app.path)

    @api.api_route(prefix + "/" + DATA_PREFIX + "{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    @requires_auth
    async def app_data(path: str) -> Response:
        return await serve_data_file(app, data_root, path)

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

    if config.host_apps:
        origin = config.canonical_origin
        if origin is None:
            raise ValueError(f"{HOST_APPS_ENV} needs {CANONICAL_ORIGIN_ENV} to redirect to")
        for host, app_name in config.host_apps.items():
            if app_name not in {app.name for app in apps}:
                raise ValueError(f"{HOST_APPS_ENV}: {host} points at unknown app {app_name!r}")
            if origin.endswith(f"//{host}"):
                raise ValueError(f"{HOST_APPS_ENV}: {host} is the canonical origin and would redirect to itself")

        @api.middleware("http")
        async def redirect_aliased_hosts(request: Request, call_next):
            target = host_redirect(request.headers.get("host", ""), request.url.path, config.host_apps, origin)
            if target is None:
                return await call_next(request)
            query = f"?{request.url.query}" if request.url.query else ""
            # 307, not 308: a permanent redirect is cached hard, and a browser that kept an
            # earlier target would keep following it after this mapping changes.
            return RedirectResponse(target + query, status_code=307, headers={"Cache-Control": "no-store"})

    for app in apps:
        if is_python_app(app):
            services = services_for(app, config.data_root, config.database)
            api.mount(app.path.rstrip("/") + API_PREFIX, AuthenticatedMount(create_api(app, services), policy))
        install_app_routes(api, app, config.data_root)

    return RouteAuthMiddleware(api, policy)


def asgi() -> RouteAuthMiddleware:
    """Uvicorn factory entry point: ``uvicorn --factory marina.server:asgi``."""
    return create_app(MarinaConfig.from_env(Path.cwd() / "apps"))
