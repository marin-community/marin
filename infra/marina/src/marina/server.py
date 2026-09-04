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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import sqlalchemy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join
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

from marina.applets import (
    MAX_ARCHIVE_BYTES,
    AppletBackendUnavailable,
    AppletConflict,
    AppletForbidden,
    AppletNotFound,
    AppletRuntime,
    AppletStore,
    InvalidQuery,
    QueryLimitExceeded,
    read_applet_package,
    validate_backend_import,
)
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
# `host=app,host=app`: a vanity host that used to be one app's own origin (echo.oa.dev).
# Legacy API paths keep serving that app; pages redirect into its canonical prefix.
HOST_APPS_ENV = "MARINA_HOST_APPS"
# The origin the apps are served from. Aliased hosts send every request here, so one URL
# space holds the apps and a link from one app to another resolves against this origin.
CANONICAL_ORIGIN_ENV = "MARINA_CANONICAL_ORIGIN"
APPLET_ORIGIN_ENV = "MARINA_APPLET_ORIGIN"
APPLET_OPERATORS_ENV = "MARINA_APPLET_OPERATORS"
DATA_PREFIX = "data/"
API_PREFIX = "/api"
# The first path segment the kernel answers itself. An app named for one of these would
# register the same route and lose it: FastAPI keeps the first match, and the kernel's
# routes are installed before any app's.
KERNEL_PREFIXES = frozenset({"a", "api", "healthz"})
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
    # Legacy hosts assigned to one app, by host name.
    host_apps: dict[str, str] = field(default_factory=dict)
    # Scheme and host the aliased hosts redirect to, e.g. https://marina.oa.dev.
    canonical_origin: str | None = None
    # A separate IAP-gated origin that exposes only /a/* applet routes.
    applet_origin: str | None = None
    applet_operators: frozenset[str] = frozenset()

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
            applet_origin=(os.environ.get(APPLET_ORIGIN_ENV) or "").rstrip("/") or None,
            applet_operators=frozenset(
                item.strip() for item in os.environ.get(APPLET_OPERATORS_ENV, "").split(",") if item.strip()
            ),
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


def legacy_api_app(host: str, path: str, host_apps: dict[str, str]) -> str | None:
    """The app serving a root-relative API path on its legacy host, if any."""
    app = host_apps.get(host.split(":")[0].lower())
    if app is None or not (path == API_PREFIX or path.startswith(f"{API_PREFIX}/")):
        return None
    return app


def content_security_policy(app: AppManifest) -> str:
    connect = " ".join(("'self'", *app.connect_src))
    return (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        f"img-src 'self' data:; font-src 'self' data:; connect-src {connect}; frame-ancestors 'none'"
    )


def app_directory(apps: list[AppManifest]) -> list[dict[str, str]]:
    return [{"name": app.name, "title": app.title, "description": app.description, "path": app.path} for app in apps]


def applet_directory(store: AppletStore | None, applet_origin: str | None) -> list[dict[str, object]]:
    if store is None:
        return []
    return [
        {
            "name": str(applet.id),
            "title": applet.title,
            "description": applet.description,
            "path": (applet_origin or "") + applet.path,
            "kind": "applet",
            "published_by": applet.owner,
            "version": applet.current_version,
        }
        for applet in store.apps()
    ]


def applet_content_security_policy(connect_src: tuple[str, ...]) -> str:
    connect = " ".join(("'self'", *connect_src))
    return (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        f"img-src 'self' data:; font-src 'self' data:; connect-src {connect}; frame-ancestors 'none'"
    )


def accepts_encoding(header: str, encoding: str) -> bool:
    """Whether an HTTP Accept-Encoding value permits one named encoding."""
    qualities: dict[str, float] = {}
    for item in header.split(","):
        name, *parameters = item.split(";")
        name = name.strip().lower()
        if name not in {encoding.lower(), "*"}:
            continue
        quality = next((value for value in parameters if value.strip().lower().startswith("q=")), "q=1")
        try:
            qualities[name] = float(quality.split("=", 1)[1])
        except ValueError:
            qualities[name] = 0
    return qualities.get(encoding.lower(), qualities.get("*", 0)) > 0


def accepts_html(header: str) -> bool:
    accepted = {item.split(";", 1)[0].strip().lower() for item in header.split(",")}
    return not accepted or "text/html" in accepted or "*/*" in accepted


def sqlstate(error: sqlalchemy.exc.SQLAlchemyError) -> str | None:
    original = getattr(error, "orig", None)
    if original is None or not original.args or not isinstance(original.args[0], dict):
        return None
    value = original.args[0].get("C")
    return str(value) if value is not None else None


async def request_json(request: Request) -> object:
    try:
        return await request.json()
    except ValueError as error:
        raise HTTPException(status_code=400, detail="request body is not valid JSON") from error


def landing_page(apps: list[AppManifest], applets: list[dict[str, object]] | None = None) -> str:
    entries = [
        *({"title": app.title, "description": app.description, "path": app.path} for app in apps),
        *(applets or []),
    ]
    rows = "".join(
        f'<li><a href="{html.escape(str(app["path"]))}">{html.escape(str(app["title"]))}</a>'
        f'<span>{html.escape(str(app["description"]))}</span></li>'
        for app in entries
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
    target = prefix_join(root, relative)
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


async def call_applet_api(app: ASGIApp, request: Request, path: str) -> Response:
    """Forward one request to a revision-specific in-process ASGI app."""
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in {"content-length", "host", "transfer-encoding"}
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://applet") as client:
        response = await client.request(
            request.method,
            f"/{path}",
            params=request.query_params,
            headers=headers,
            content=await request.body(),
        )
    forwarded = {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
    }
    return Response(response.content, status_code=response.status_code, headers=forwarded)


def create_app(config: MarinaConfig) -> RouteAuthMiddleware:
    apps = discover_apps(config.apps_dir)
    shadowed = sorted(app.name for app in apps if app.name in KERNEL_PREFIXES)
    if shadowed:
        raise ValueError(f"app {shadowed[0]!r} is named for a kernel route; rename it")
    policy = build_policy(config.iap_audience)
    api = FastAPI(title="Marina", docs_url=None, redoc_url=None)
    applet_store = AppletStore(config.database) if config.database is not None else None
    applet_runtime = AppletRuntime(applet_store) if applet_store is not None else None

    @api.get("/healthz", include_in_schema=False)
    @public
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "apps": [app.name for app in apps]})

    @api.get("/api/marina/apps")
    @requires_auth
    def list_apps() -> JSONResponse:
        return JSONResponse({"apps": [*app_directory(apps), *applet_directory(applet_store, config.applet_origin)]})

    @api.get("/api/marina/applets")
    @requires_auth
    def list_applets() -> JSONResponse:
        return JSONResponse({"applets": applet_directory(applet_store, config.applet_origin)})

    @api.get("/api/marina/me")
    @requires_auth
    def me(request: Request) -> JSONResponse:
        identity = identity_for(request, policy)
        return JSONResponse({"user": identity.user_id, "role": identity.role})

    @api.get("/", include_in_schema=False)
    @requires_auth
    def landing() -> HTMLResponse:
        return HTMLResponse(landing_page(apps, applet_directory(applet_store, config.applet_origin)))

    @api.post("/api/marina/applets")
    @requires_auth
    async def publish_applet(request: Request) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=503, detail="Marina has no database configured")
        payload = await request.body()
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise HTTPException(status_code=400, detail=f"archive exceeds {MAX_ARCHIVE_BYTES} bytes")
        try:
            package = await run_in_threadpool(read_applet_package, payload)
            await run_in_threadpool(validate_backend_import, package)
            owner = identity_for(request, policy).user_id
            published = await run_in_threadpool(applet_store.publish, package, owner)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JSONResponse(
            {
                "id": str(published.applet_id),
                "version": published.version,
                "path": published.path,
                "url": (config.applet_origin or "") + published.path,
            },
            status_code=201,
        )

    @api.post("/api/marina/applets/{applet_id}")
    @requires_auth
    async def update_applet(applet_id: uuid.UUID, base_version: int, request: Request) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=503, detail="Marina has no database configured")
        payload = await request.body()
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise HTTPException(status_code=400, detail=f"archive exceeds {MAX_ARCHIVE_BYTES} bytes")
        try:
            package = await run_in_threadpool(read_applet_package, payload)
            await run_in_threadpool(validate_backend_import, package)
            owner = identity_for(request, policy).user_id
            published = await run_in_threadpool(
                applet_store.publish,
                package,
                owner,
                applet_id,
                base_version,
                config.applet_operators,
            )
            retained = await run_in_threadpool(applet_store.versions, applet_id)
            assert applet_runtime is not None
            await run_in_threadpool(
                applet_runtime.retain_versions,
                applet_id,
                {item.version for item in retained},
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        except AppletForbidden as error:
            raise HTTPException(status_code=403, detail="only the publisher may update this applet") from error
        except AppletConflict as error:
            raise HTTPException(status_code=409, detail="applet has a newer current version") from error
        return JSONResponse(
            {
                "id": str(published.applet_id),
                "version": published.version,
                "path": published.path,
                "url": (config.applet_origin or "") + published.path,
            },
            status_code=201,
        )

    @api.get("/api/marina/applets/{applet_id}")
    @requires_auth
    async def applet_details(applet_id: uuid.UUID) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            current = await run_in_threadpool(applet_store.current_version, applet_id)
            versions = await run_in_threadpool(applet_store.versions, applet_id)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        return JSONResponse(
            {
                "id": str(applet_id),
                "current_version": current,
                "url": f"{config.applet_origin or ''}/a/{applet_id}/",
                "versions": [
                    {
                        "version": item.version,
                        "published_by": item.published_by,
                        "published_at": item.published_at.isoformat(),
                        "byte_size": item.byte_size,
                    }
                    for item in versions
                ],
            }
        )

    @api.put("/api/marina/applets/{applet_id}/current")
    @requires_auth
    async def rollback_applet(applet_id: uuid.UUID, request: Request) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        body = await request_json(request)
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("version"), int)
            or not isinstance(body.get("base_version"), int)
        ):
            raise HTTPException(status_code=400, detail="body must contain integer version and base_version")
        actor = identity_for(request, policy).user_id
        try:
            await run_in_threadpool(
                applet_store.rollback,
                applet_id,
                body["version"],
                actor,
                body["base_version"],
                config.applet_operators,
            )
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet or revision not found") from error
        except AppletForbidden as error:
            raise HTTPException(
                status_code=403, detail="only the publisher or an operator may roll back this applet"
            ) from error
        except AppletConflict as error:
            raise HTTPException(status_code=409, detail="applet has a newer current version") from error
        return JSONResponse({"id": str(applet_id), "version": body["version"]})

    @api.delete("/api/marina/applets/{applet_id}", status_code=204)
    @requires_auth
    async def archive_applet(applet_id: uuid.UUID, request: Request) -> Response:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        actor = identity_for(request, policy).user_id
        try:
            await run_in_threadpool(applet_store.archive, applet_id, actor, config.applet_operators)
            assert applet_runtime is not None
            await run_in_threadpool(applet_runtime.retain_versions, applet_id, set())
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        except AppletForbidden as error:
            raise HTTPException(
                status_code=403, detail="only the publisher or an operator may archive this applet"
            ) from error
        return Response(status_code=204)

    @api.get("/a/{applet_id}", include_in_schema=False)
    @requires_auth
    def applet_without_slash(applet_id: uuid.UUID) -> RedirectResponse:
        return RedirectResponse(f"/a/{applet_id}/")

    @api.get("/a/{applet_id}/", include_in_schema=False)
    @requires_auth
    def current_applet(applet_id: uuid.UUID) -> RedirectResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            version = applet_store.current_version(applet_id)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        return RedirectResponse(
            f"/a/{applet_id}/v/{version}/",
            status_code=307,
            headers={"Cache-Control": "no-cache"},
        )

    async def execute_applet_query(applet_id: uuid.UUID, request: Request) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        body = await request_json(request)
        if not isinstance(body, dict) or not isinstance(body.get("sql"), str):
            raise HTTPException(status_code=400, detail="body must contain a SQL string")
        parameters = body.get("parameters", {})
        if not isinstance(parameters, dict):
            raise HTTPException(status_code=400, detail="parameters must be an object")
        try:
            result = await run_in_threadpool(applet_store.query, applet_id, body["sql"], parameters)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        except QueryLimitExceeded as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except InvalidQuery as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except sqlalchemy.exc.SQLAlchemyError as error:
            if sqlstate(error) == "57014":
                raise HTTPException(status_code=504, detail="query exceeded its statement timeout") from error
            raise HTTPException(status_code=422, detail=str(error)) from error
        return JSONResponse(result)

    @api.post("/a/{applet_id}/query")
    @requires_auth
    async def query_applet(applet_id: uuid.UUID, request: Request) -> JSONResponse:
        return await execute_applet_query(applet_id, request)

    @api.post("/a/{applet_id}/v/{version}/query")
    @requires_auth
    async def versioned_applet_query(applet_id: uuid.UUID, version: int, request: Request) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            await run_in_threadpool(applet_store.version, applet_id, version)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet or revision not found") from error
        return await execute_applet_query(applet_id, request)

    @api.api_route(
        "/a/{applet_id}/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    @requires_auth
    async def current_applet_api(applet_id: uuid.UUID, path: str, request: Request) -> Response:
        if applet_store is None or applet_runtime is None:
            raise HTTPException(status_code=404, detail="applet API not found")
        try:
            version = await run_in_threadpool(applet_store.current_version, applet_id)
            applet_api = await run_in_threadpool(applet_runtime.api, applet_id, version)
            return await call_applet_api(applet_api, request, path)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet API not found") from error
        except AppletBackendUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @api.api_route(
        "/a/{applet_id}/v/{version}/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    @requires_auth
    async def versioned_applet_api(applet_id: uuid.UUID, version: int, path: str, request: Request) -> Response:
        if applet_runtime is None:
            raise HTTPException(status_code=404, detail="applet API not found")
        try:
            applet_api = await run_in_threadpool(applet_runtime.api, applet_id, version)
            return await call_applet_api(applet_api, request, path)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet API not found") from error
        except AppletBackendUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    def applet_file_response(applet_id: uuid.UUID, version: int, path: str, request: Request) -> Response:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            record = applet_store.version(applet_id, version)
            stored = applet_store.file(
                applet_id,
                version,
                path,
                accept_gzip=accepts_encoding(request.headers.get("accept-encoding", ""), "gzip"),
                accept_html=accepts_html(request.headers.get("accept", "")),
            )
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet file not found") from error
        etag = f'"{stored.digest.hex()}"'
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "Content-Security-Policy": applet_content_security_policy(record.manifest.connect_src),
            "ETag": etag,
            "Vary": "Accept-Encoding",
        }
        if stored.content_encoding is not None:
            headers["Content-Encoding"] = stored.content_encoding
        if etag in {item.strip() for item in request.headers.get("if-none-match", "").split(",")}:
            return Response(status_code=304, headers=headers)
        return Response(stored.body, media_type=stored.media_type, headers=headers)

    @api.api_route(
        "/a/{applet_id}/v/{version}/",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    @requires_auth
    def applet_root(applet_id: uuid.UUID, version: int, request: Request) -> Response:
        return applet_file_response(applet_id, version, "", request)

    @api.api_route(
        "/a/{applet_id}/v/{version}/{path:path}",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    @requires_auth
    def applet_file(applet_id: uuid.UUID, version: int, path: str, request: Request) -> Response:
        return applet_file_response(applet_id, version, path, request)

    if config.applet_origin is not None:
        parsed_applet_origin = urlparse(config.applet_origin)
        applet_host = parsed_applet_origin.hostname
        if (
            applet_host is None
            or parsed_applet_origin.scheme not in {"http", "https"}
            or parsed_applet_origin.path not in {"", "/"}
        ):
            raise ValueError(f"{APPLET_ORIGIN_ENV} must be an absolute URL")

        @api.middleware("http")
        async def isolate_applet_host(request: Request, call_next):
            host = request.headers.get("host", "").split(":", 1)[0].lower()
            path = request.url.path
            is_applet_path = path == "/a" or path.startswith("/a/")
            if host == applet_host:
                if not is_applet_path:
                    return JSONResponse({"error": "not found"}, status_code=404)
                return await call_next(request)
            if is_applet_path and request.method in {"GET", "HEAD"}:
                query = f"?{request.url.query}" if request.url.query else ""
                return RedirectResponse(
                    config.applet_origin + path + query,
                    status_code=307,
                    headers={"Cache-Control": "no-store"},
                )
            if is_applet_path:
                return JSONResponse({"error": "not found"}, status_code=404)
            return await call_next(request)

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
            host = request.headers.get("host", "")
            path = request.url.path
            app_name = legacy_api_app(host, path, config.host_apps)
            if app_name is not None:
                prefix = f"/{app_name}"
                request.scope["path"] = prefix + path
                request.scope["raw_path"] = prefix.encode() + request.scope["raw_path"]
                return await call_next(request)
            target = host_redirect(host, path, config.host_apps, origin)
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
