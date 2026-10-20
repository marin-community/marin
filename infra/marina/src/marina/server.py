# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve released Postgres packages and expose Marina's shared surface.

One FastAPI application. Each app is mounted under ``/<name>/``: files in its ``dist``
are served verbatim and every other path under the prefix answers ``index.html`` so a
client-side route survives a reload. ``/<name>/data/*`` reads from the app's declared
data URL or its directory under the deployment data root, so large or changing files
stay outside the package. A Python app's API is
mounted at ``/<name>/api/`` behind the same authentication, with the caller's identity
bound for its handlers. ``/api/marina/*`` is the surface shared by every
app (the app directory and the caller's identity); ``/`` lists the apps. A per-app
Content-Security-Policy restricts what the page may fetch to itself, the manifest's
``connect_src``, and the configured Loom origin for apps that enable the agent panel.
The public-applet deployment uses the same kernel without named application routes and
admits only GET and HEAD routes for applets whose stored mode is public.
"""

import html
import mimetypes
import os
import posixpath
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from urllib.parse import urlencode, urlparse

import httpx
import sqlalchemy
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from rigging.server_auth import (
    RequestAuthPolicy,
    RouteAuthMiddleware,
    VerifiedIdentity,
    extract_bearer_token,
    identity_scope,
    public,
    requires_auth,
    scope_client_address,
    scope_headers,
)
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from marina.applet_mcp import ReleasedApplications
from marina.applets import (
    MAX_ARCHIVE_BYTES,
    AppletBackendUnavailable,
    AppletConflict,
    AppletForbidden,
    AppletMode,
    AppletNotFound,
    AppletPackage,
    AppletRuntime,
    AppletStore,
    InvalidQuery,
    PublishResult,
    QueryLimitExceeded,
    read_applet_package,
    validate_backend_import,
)
from marina.auth import build_policy, identity_for
from marina.db import DatabaseSpec, database_from_env
from marina.host_bindings import normalize_hostname
from marina.mcp import marina_mcp
from marina.runtime import RuntimeConfig, load_runtime_config

INDEX_FILE = "index.html"
DIRECTORY_PAGE_SIZE = 50
DIRECTORY_CACHE_CONTROL = "private, no-store"
# A file committed as `x.gz` is served at `x` with a Content-Encoding header, so a large
# static asset can live in the repository compressed.
PRECOMPRESSED_SUFFIX = ".gz"
CLOUD_RUN_SERVICE_ENV = "K_SERVICE"
DATA_ROOT_ENV = "MARINA_DATA_ROOT"
IAP_AUDIENCE_ENV = "MARINA_IAP_AUDIENCE"
# `host=app,host=app`: a vanity host that used to be one app's own origin (echo.oa.dev).
# Legacy API paths keep serving that app; pages redirect into its canonical prefix.
HOST_APPS_ENV = "MARINA_HOST_APPS"
# The origin the apps are served from. Aliased hosts send every request here, so one URL
# space holds the apps and a link from one app to another resolves against this origin.
CANONICAL_ORIGIN_ENV = "MARINA_CANONICAL_ORIGIN"
APPLET_ORIGIN_ENV = "MARINA_APPLET_ORIGIN"
PUBLIC_APPLET_ORIGIN_ENV = "MARINA_PUBLIC_APPLET_ORIGIN"
MARINA_SURFACE_ENV = "MARINA_SURFACE"
APPLET_DOMAINS_ENV = "MARINA_APPLET_DOMAINS"
APPLET_OPERATORS_ENV = "MARINA_APPLET_OPERATORS"
AGENT_ORIGIN_ENV = "MARINA_AGENT_ORIGIN"
DATA_PREFIX = "data/"
API_PREFIX = "/api"
MCP_PATH = "/api/marina/mcp"
MCP_READ_PATH = "/api/marina/mcp/read"
NAMED_APPLET_HOST_STATE = "named_applet_host"
# The first path segment the kernel answers itself. An app named for one of these would
# register the same route and lose it: FastAPI keeps the first match, and the kernel's
# routes are installed before any app's.
KERNEL_PREFIXES = frozenset({"a", "api", "healthz"})
DATA_CACHE_CONTROL = "private, max-age=300"
BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


@dataclass(frozen=True)
class AgentPanelService:
    """Deployment-specific location of the Loom service."""

    origin: str


class MarinaSurface(StrEnum):
    AUTHENTICATED = "authenticated"
    PUBLIC_APPLETS = "public_applets"


@dataclass(frozen=True)
class MarinaConfig:
    # An fsspec URL (gs://bucket/prefix or a local directory) holding one directory per app.
    data_root: str
    # The IAP JWT audience for this service; None admits loopback callers only.
    iap_audience: str | None
    # Without a database, only kernel routes are available.
    database: DatabaseSpec | None = None
    # Legacy hosts assigned to one app, by host name.
    host_apps: dict[str, str] = field(default_factory=dict)
    # Scheme and host the aliased hosts redirect to, e.g. https://marina.oa.dev.
    canonical_origin: str | None = None
    # A separate IAP-gated origin that exposes only /a/* applet routes.
    applet_origin: str | None = None
    # An unauthenticated origin that exposes GET/HEAD routes for public applets only.
    public_applet_origin: str | None = None
    surface: MarinaSurface = MarinaSurface.AUTHENTICATED
    applet_domains: frozenset[str] = frozenset()
    applet_operators: frozenset[str] = frozenset()
    agent_panel: AgentPanelService | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    require_registered_apps: bool = False
    local_publisher: str | None = None

    @classmethod
    def from_env(cls) -> "MarinaConfig":
        """Resolve process configuration and require IAP on the authenticated Cloud Run surface."""
        data_root = os.environ.get(DATA_ROOT_ENV)
        if not data_root:
            raise ValueError(f"{DATA_ROOT_ENV} is not set")
        try:
            surface = MarinaSurface(os.environ.get(MARINA_SURFACE_ENV, MarinaSurface.AUTHENTICATED))
        except ValueError as error:
            raise ValueError(f"{MARINA_SURFACE_ENV} must be authenticated or public_applets") from error
        audience = os.environ.get(IAP_AUDIENCE_ENV) or None
        if os.environ.get(CLOUD_RUN_SERVICE_ENV) and not audience and surface is MarinaSurface.AUTHENTICATED:
            raise ValueError(f"{IAP_AUDIENCE_ENV} must be set when running on Cloud Run")
        return cls(
            data_root=data_root,
            iap_audience=audience,
            database=database_from_env(os.environ),
            host_apps=parse_host_mapping(os.environ.get(HOST_APPS_ENV, ""), HOST_APPS_ENV),
            canonical_origin=(os.environ.get(CANONICAL_ORIGIN_ENV) or "").rstrip("/") or None,
            applet_origin=(os.environ.get(APPLET_ORIGIN_ENV) or "").rstrip("/") or None,
            public_applet_origin=(os.environ.get(PUBLIC_APPLET_ORIGIN_ENV) or "").rstrip("/") or None,
            surface=surface,
            applet_domains=frozenset(
                normalize_hostname(host.strip())
                for host in os.environ.get(APPLET_DOMAINS_ENV, "").split(",")
                if host.strip()
            ),
            applet_operators=frozenset(
                item.strip() for item in os.environ.get(APPLET_OPERATORS_ENV, "").split(",") if item.strip()
            ),
            agent_panel=(
                AgentPanelService(origin=os.environ[AGENT_ORIGIN_ENV].rstrip("/"))
                if os.environ.get(AGENT_ORIGIN_ENV)
                else None
            ),
            runtime=load_runtime_config(),
            require_registered_apps=bool(os.environ.get(CLOUD_RUN_SERVICE_ENV))
            and surface is MarinaSurface.AUTHENTICATED,
        )


def parse_host_mapping(spec: str, env_name: str) -> dict[str, str]:
    """Parse comma-separated host assignments; blank means no assignments."""
    pairs = [item.strip() for item in spec.split(",") if item.strip()]
    result: dict[str, str] = {}
    for pair in pairs:
        host, sep, target = pair.partition("=")
        if not sep or not host or not target:
            raise ValueError(f"{env_name} entry {pair!r} is not host=target")
        result[host.strip().lower()] = target.strip()
    return result


def named_applet_hosts(app: ASGIApp, store: AppletStore, domains: frozenset[str]) -> ASGIApp:
    """Route each named host to one applet before route authentication runs."""

    async def serve(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await app(scope, receive, send)
        host_header = scope_headers(scope).get("host", "")
        hostname = host_header.split(":", 1)[0].lower().rstrip(".")
        if hostname not in domains:
            return await app(scope, receive, send)
        try:
            applet_id = await run_in_threadpool(store.host_applet, hostname)
        except AppletNotFound:
            return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
        if applet_id is None:
            return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
        if scope["path"] == "/a" or scope["path"].startswith("/a/"):
            return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)
        prefix = f"/a/{applet_id}"
        rewritten = {
            **scope,
            "path": prefix + scope["path"],
            "raw_path": prefix.encode() + scope.get("raw_path", scope["path"].encode()),
            "state": {**scope.get("state", {}), NAMED_APPLET_HOST_STATE: True},
        }

        async def send_response(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # Another applet can reuse /v/1/ on this hostname after a rebind.
                if "immutable" in headers.get("cache-control", ""):
                    headers["cache-control"] = "private, no-cache"
                location = headers.get("location")
                if location is not None:
                    target = urlparse(location)
                    if target.netloc in ("", host_header) and target.path.startswith(prefix + "/"):
                        headers["location"] = target._replace(path=target.path.removeprefix(prefix)).geturl()
            await send(message)

        return await app(rewritten, receive, send_response)

    return serve


def public_applet_surface(app: ASGIApp) -> ASGIApp:
    """Expose health and read-only applet routes from an unauthenticated service."""

    async def serve(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await app(scope, receive, send)
        path = scope["path"]
        method = scope["method"]
        if method in {"GET", "HEAD"} and (path == "/healthz" or path == "/a" or path.startswith("/a/")):
            return await app(scope, receive, send)
        return await JSONResponse({"error": "not found"}, status_code=404)(scope, receive, send)

    return serve


def validate_agent_panel(service: AgentPanelService | None, iap_audience: str | None) -> None:
    """Reject origins that cannot safely receive credentialed browser requests."""
    if service is None:
        return
    parsed = urlparse(service.origin)
    absolute_origin = (
        parsed.scheme in {"http", "https"}
        and parsed.hostname is not None
        and parsed.path in {"", "/"}
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )
    if not absolute_origin:
        raise ValueError(f"{AGENT_ORIGIN_ENV} must be an absolute HTTP origin without a path")
    if parsed.scheme == "https":
        return
    if iap_audience is not None or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise ValueError(f"{AGENT_ORIGIN_ENV} must use HTTPS outside a loopback development kernel")


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


def content_security_policy(connect_src: tuple[str, ...]) -> str:
    connect = " ".join(("'self'", *connect_src))
    return (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        f"img-src 'self' data:; font-src 'self' data:; connect-src {connect}; frame-ancestors 'none'"
    )


def applet_url(
    path: str,
    mode: AppletMode,
    applet_origin: str | None,
    public_applet_origin: str | None,
) -> str:
    origin = public_applet_origin if mode is AppletMode.PUBLIC else applet_origin
    return (origin or "") + path


def applet_directory(
    store: AppletStore | None,
    applet_origin: str | None,
    public_applet_origin: str | None,
) -> list[dict[str, object]]:
    if store is None:
        return []
    return [
        {
            "name": applet.name or str(applet.id),
            "title": applet.title,
            "description": applet.description,
            "path": (
                applet.path if applet.name else applet_url(applet.path, applet.mode, applet_origin, public_applet_origin)
            ),
            "kind": "applet",
            "published_by": applet.published_by,
            "published_at": applet.published_at.isoformat(),
            "version": applet.current_version,
            "mode": applet.mode.value,
        }
        for applet in store.active_applets()
    ]


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


def directory_url(sort: str, order: str, page: int = 1) -> str:
    return "/?" + html.escape(urlencode({"sort": sort, "order": order, "page": page}))


def landing_page(
    entries: list[dict[str, object]],
    sort: Literal["name", "released"] = "released",
    order: Literal["asc", "desc"] = "desc",
    page: int = 1,
) -> str:
    sort_field = "title" if sort == "name" else "published_at"
    entries = sorted(
        entries,
        key=lambda entry: (str(entry[sort_field]).casefold(), str(entry["name"])),
        reverse=order == "desc",
    )
    total = len(entries)
    pages = max(1, (total + DIRECTORY_PAGE_SIZE - 1) // DIRECTORY_PAGE_SIZE)
    page = min(page, pages)
    start = (page - 1) * DIRECTORY_PAGE_SIZE
    rows = []
    for app in entries[start : start + DIRECTORY_PAGE_SIZE]:
        released = datetime.fromisoformat(str(app["published_at"])).astimezone(UTC)
        rows.append(
            f'<tr><td><a class="app-name" href="{html.escape(str(app["path"]))}">'
            f'{html.escape(str(app["title"]))}</a>'
            f'<p class="description">{html.escape(str(app["description"]))}</p></td>'
            f'<td class="released"><time datetime="{released.isoformat()}">'
            f'{released.strftime("%Y-%m-%d %H:%M")}</time></td>'
            f'<td class="version">v{app["version"]}</td>'
            f'<td>{html.escape(str(app["mode"]))}</td>'
            f'<td class="publisher">{html.escape(str(app["published_by"]))}</td></tr>'
        )
    headings = []
    for column, label in (("name", "Name"), ("released", "Released (UTC)")):
        active = sort == column
        next_order = "desc" if active and order == "asc" else "asc"
        if not active and column == "released":
            next_order = "desc"
        aria_sort = ("ascending" if order == "asc" else "descending") if active else "none"
        arrow = (" ↑" if order == "asc" else " ↓") if active else " ↕"
        headings.append(
            f'<th scope="col" aria-sort="{aria_sort}"><a href="{directory_url(column, next_order)}">'
            f'{label}<span aria-hidden="true">{arrow}</span></a></th>'
        )
    directory = (
        '<div class="table-scroll" role="region" aria-label="Apps" tabindex="0">'
        "<table><thead><tr>" + "".join(headings) + '<th scope="col">Version</th>'
        '<th scope="col">Access</th><th scope="col">Published by</th></tr></thead>'
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
        if entries
        else '<p class="empty">No apps published yet.</p>'
    )
    previous = (
        f'<a rel="prev" href="{directory_url(sort, order, page - 1)}">← Previous</a>'
        if page > 1
        else '<span aria-disabled="true">← Previous</span>'
    )
    next_link = (
        f'<a rel="next" href="{directory_url(sort, order, page + 1)}">Next →</a>'
        if page < pages
        else '<span aria-disabled="true">Next →</span>'
    )
    pagination = (
        f'<nav class="pagination" aria-label="App pages">{previous}'
        f"<span>Page {page} of {pages} · {DIRECTORY_PAGE_SIZE} per page</span>{next_link}</nav>"
        if total
        else ""
    )
    count = f"Showing {start + 1}&ndash;{min(start + DIRECTORY_PAGE_SIZE, total)} of {total} apps" if total else "0 apps"
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marina — Apps</title>
<style>
:root { color-scheme: light dark; --ink:#17181c; --paper:#fcfbf8; --panel:#fff;
  --edge:#e2ded2; --muted:#6d6a60; --mark:#1f5c4a; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,sans-serif; background:var(--paper); color:var(--ink); }
header { padding:1rem max(1.5rem,calc((100% - 82rem)/2)); border-bottom:1px solid var(--edge); background:var(--panel); }
.wordmark { color:var(--ink); text-decoration:none; font-size:1rem; font-weight:650; letter-spacing:.02em; }
main { max-width:85rem; margin:0 auto; padding:2rem 1.5rem; }
h1 { font-size:1.75rem; line-height:1.2; letter-spacing:-.03em; margin:0; font-weight:600; }
.intro { color:var(--muted); margin:.5rem 0 1.5rem; }
.list-heading { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.count { color:var(--muted); }
.table-scroll { overflow-x:auto; border:1px solid var(--edge); border-radius:.5rem; background:var(--panel); }
table { width:100%; border-collapse:collapse; text-align:left; }
th { color:var(--muted); font-size:.75rem; font-weight:600; white-space:nowrap; }
th, td { padding:.65rem .85rem; border-bottom:1px solid var(--edge); vertical-align:top; }
th a { color:inherit; text-decoration:none; display:block; }
th[aria-sort="ascending"] a, th[aria-sort="descending"] a { color:var(--mark); }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover { background:color-mix(in srgb,var(--mark) 4%,var(--panel)); }
a { color:var(--mark); text-underline-offset:.2em; }
a:hover { text-decoration:underline; }
a:focus-visible, .table-scroll:focus-visible { outline:3px solid var(--mark); outline-offset:3px; }
.app-name { font-weight:600; text-decoration:none; overflow-wrap:anywhere; }
td:first-child { width:52%; min-width:16rem; }
.description { color:var(--muted); font-size:.8rem; margin:.2rem 0 0; overflow-wrap:anywhere; }
.released, .version { white-space:nowrap; font-variant-numeric:tabular-nums; }
.publisher { overflow-wrap:anywhere; font-size:.8rem; }
.pagination { display:flex; align-items:center; justify-content:space-between; gap:1rem;
  margin-top:1rem; font-size:.8rem; }
.pagination span { color:var(--muted); }
.pagination a { text-decoration:none; }
[aria-disabled="true"] { opacity:.5; }
.empty { color:var(--muted); padding:2rem 0; }
@media (prefers-color-scheme:dark) { :root { --ink:#ece9e1; --paper:#15171a; --panel:#1c1f22;
  --edge:#2f3336; --muted:#aaa79e; --mark:#6cc4a5; } }
@media (max-width:40rem) { main { padding:1.5rem 1rem; } header { padding:1rem; }
  .pagination { gap:.5rem; flex-wrap:wrap; } }
</style></head><body><header><a class="wordmark" href="/">Marina</a></header>
<main><div class="list-heading"><h1>All apps</h1><span class="count">"""
        + count
        + '</span></div><p class="intro">Your team&#39;s tools, together in Marina.</p>'
        + directory
        + pagination
        + "</main></body></html>"
    )


def clean_relative_path(path: str) -> str | None:
    """A normalized relative path, or None when it would escape its root."""
    normalized = posixpath.normpath(path)
    if not path or normalized.startswith(("../", "/")) or normalized in ("..", "."):
        return None
    return normalized


def requested_byte_range(header: str, size: int) -> tuple[int, int] | None:
    """Return the requested half-open byte range, or None for an invalid range."""
    match = BYTE_RANGE.fullmatch(header)
    if match is None or not any(match.groups()):
        return None
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last) + 1, size) if last else size
    else:
        length = min(int(last), size)
        start, end = size - length, size
    return (start, end) if start < size and start < end else None


async def storage_file_response(fs, target: str, request: Request, media_type: str, headers: dict[str, str]) -> Response:
    """Serve one existing object, honoring HTTP byte ranges and HEAD requests."""
    size = await run_in_threadpool(fs.size, target)
    headers["Accept-Ranges"] = "bytes"
    range_header = request.headers.get("range")
    if range_header is None:
        headers["Content-Length"] = str(size)
        body = b"" if request.method == "HEAD" else await run_in_threadpool(fs.cat_file, target)
        return Response(body, media_type=media_type, headers=headers)

    byte_range = requested_byte_range(range_header, size)
    if byte_range is None:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    start, end = byte_range
    headers.update({"Content-Length": str(end - start), "Content-Range": f"bytes {start}-{end - 1}/{size}"})
    body = b"" if request.method == "HEAD" else await run_in_threadpool(fs.cat_file, target, start=start, end=end)
    return Response(body, status_code=206, media_type=media_type, headers=headers)


async def serve_data_file(
    data_url: str,
    path: str,
    request: Request,
    connect_src: tuple[str, ...] = (),
) -> Response:
    """A file from the app's data directory, including byte ranges for large files."""
    relative = clean_relative_path(path)
    if relative is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    fs, root = filesystem_for(data_url)
    target = prefix_join(root, relative)
    headers = {
        "Content-Security-Policy": content_security_policy(connect_src),
        "Cache-Control": DATA_CACHE_CONTROL,
    }
    media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
    if await run_in_threadpool(fs.isfile, target):
        return await storage_file_response(fs, target, request, media_type, headers)
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


async def uploaded_applet_package(request: Request) -> AppletPackage:
    payload = await request.body()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise HTTPException(status_code=400, detail=f"archive exceeds {MAX_ARCHIVE_BYTES} bytes")
    try:
        package = await run_in_threadpool(read_applet_package, payload)
        await run_in_threadpool(validate_backend_import, package)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return package


def publish_result_response(
    published: PublishResult,
    applet_origin: str | None,
    public_applet_origin: str | None,
) -> JSONResponse:
    return JSONResponse(
        {
            "id": str(published.applet_id),
            "version": published.version,
            "mode": published.mode.value,
            "path": published.path,
            "url": applet_url(
                published.path,
                published.mode,
                applet_origin,
                public_applet_origin,
            ),
        },
        status_code=201,
    )


async def dispatch_applet_api(
    store: AppletStore | None,
    runtime: AppletRuntime | None,
    applet_id: uuid.UUID,
    version: int | None,
    path: str,
    request: Request,
    identity: VerifiedIdentity | None,
) -> Response:
    if store is None or runtime is None:
        raise HTTPException(status_code=404, detail="applet API not found")
    try:
        selected_version = version
        if selected_version is None:
            selected_version = await run_in_threadpool(store.current_version, applet_id)
        applet_api = await run_in_threadpool(runtime.api, applet_id, selected_version)
        with identity_scope(identity):
            return await call_applet_api(applet_api, request, path)
    except AppletNotFound as error:
        raise HTTPException(status_code=404, detail="applet API not found") from error
    except AppletBackendUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def reserved_applet_hosts(config: MarinaConfig) -> set[str]:
    """Hosts serving Marina or legacy apps cannot be bound to applets."""
    reserved_hosts = set(config.host_apps)
    reserved_hosts.update(
        host
        for origin in (config.canonical_origin, config.applet_origin, config.public_applet_origin)
        if origin and (host := urlparse(origin).hostname) is not None
    )
    return reserved_hosts


def validate_applet_origin(origin: str | None, environment_name: str) -> None:
    if origin is None:
        return
    parsed = urlparse(origin)
    if parsed.hostname is None or parsed.scheme not in {"http", "https"} or parsed.path not in {"", "/"}:
        raise ValueError(f"{environment_name} must be an absolute URL")


def create_app(config: MarinaConfig) -> ASGIApp:
    validate_agent_panel(config.agent_panel, config.iap_audience)
    validate_applet_origin(config.applet_origin, APPLET_ORIGIN_ENV)
    validate_applet_origin(config.public_applet_origin, PUBLIC_APPLET_ORIGIN_ENV)
    applet_store = AppletStore(config.database) if config.database is not None else None
    applet_runtime = AppletRuntime(applet_store, config.data_root) if applet_store is not None else None
    if config.require_registered_apps:
        if applet_store is None:
            raise ValueError("Runtime cutover requires the application registry")
        config.runtime.require_registrations({app.name: app.id for app in applet_store.active_applets() if app.name})
    if config.local_publisher is not None and config.iap_audience is not None:
        raise ValueError("A local publisher is available only to a loopback operator kernel")
    policy = build_policy(config.iap_audience)
    providers = (
        []
        if applet_store is None or applet_runtime is None
        else [ReleasedApplications(applet_store, applet_runtime, read_only=False)]
    )
    read_providers = (
        []
        if applet_store is None or applet_runtime is None
        else [ReleasedApplications(applet_store, applet_runtime, read_only=True)]
    )
    mcp = marina_mcp({}, providers=providers)
    read_mcp = marina_mcp({}, providers=read_providers)
    mcp_app = mcp.http_app(path="/", json_response=True, stateless_http=True)
    read_mcp_app = read_mcp.http_app(path="/", json_response=True, stateless_http=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with mcp_app.lifespan(_app):
            async with read_mcp_app.lifespan(_app):
                yield

    api = FastAPI(title="Marina", docs_url=None, redoc_url=None, lifespan=lifespan)
    reserved_hosts = reserved_applet_hosts(config)
    if reserved_hosts.intersection(config.applet_domains):
        raise ValueError(f"{APPLET_DOMAINS_ENV} must not reuse a Marina or legacy app host")

    def validate_release(package: AppletPackage, applet_id: uuid.UUID | None) -> None:
        if package.manifest.jobs:
            if applet_id is None:
                raise ValueError("Scheduled applications require a declared UUID")
            config.runtime.validate_jobs(applet_id, package.manifest.jobs)
        name = package.manifest.name
        if name in config.runtime.required_apps and config.runtime.required_apps[name] != applet_id:
            raise ValueError(f"Application {name!r} must use its declared UUID")
        reserved_name = next(
            (name for name, reserved_id in config.runtime.required_apps.items() if reserved_id == applet_id), None
        )
        if reserved_name is not None and name != reserved_name:
            raise ValueError(f"UUID {applet_id} is reserved for application {reserved_name!r}")
        if package.manifest.data_url is not None and not any(
            package.manifest.data_url.startswith(prefix) for prefix in config.runtime.data_urls.get(name or "", ())
        ):
            raise ValueError("Application data_url is outside its configured storage prefixes")

    def request_identity(request: Request) -> VerifiedIdentity:
        try:
            return identity_for(request, policy)
        except ValueError as error:
            raise HTTPException(status_code=401, detail="authentication required") from error

    def host_operator(request: Request) -> AppletStore:
        if identity_for(request, policy).user_id not in config.applet_operators:
            raise HTTPException(status_code=403, detail="only an applet operator may manage host bindings")
        if applet_store is None:
            raise HTTPException(status_code=503, detail="Marina has no database configured")
        return applet_store

    def require_applet_read(applet_id: uuid.UUID, request: Request) -> None:
        if config.surface is MarinaSurface.AUTHENTICATED:
            request_identity(request)
            return
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            mode = applet_store.access_mode(applet_id)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        if mode is not AppletMode.PUBLIC:
            raise HTTPException(status_code=404, detail="applet not found")

    @api.get("/healthz", include_in_schema=False)
    @public
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    @api.get("/api/marina/apps")
    @requires_auth
    def list_apps() -> JSONResponse:
        return JSONResponse(
            {
                "apps": [
                    *applet_directory(applet_store, config.applet_origin, config.public_applet_origin),
                ]
            },
            headers={"Cache-Control": DIRECTORY_CACHE_CONTROL},
        )

    @api.get("/api/marina/applets")
    @requires_auth
    def list_applets() -> JSONResponse:
        return JSONResponse(
            {"applets": applet_directory(applet_store, config.applet_origin, config.public_applet_origin)},
            headers={"Cache-Control": DIRECTORY_CACHE_CONTROL},
        )

    @api.get("/api/marina/me")
    @requires_auth
    def me(request: Request) -> JSONResponse:
        identity = identity_for(request, policy)
        return JSONResponse({"user": identity.user_id, "role": identity.role})

    @api.get("/api/marina/agent/config")
    @requires_auth
    def agent_config(app: str) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="app not found")
        try:
            selected = applet_store.named_applet(app)
            manifest = applet_store.version(selected.id, selected.current_version).manifest
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="app not found") from error
        if config.agent_panel is None or manifest.agent is None:
            return JSONResponse({"enabled": False})
        return JSONResponse(
            {
                "enabled": True,
                "origin": config.agent_panel.origin,
                "profile": manifest.agent.profile,
                "repository": manifest.agent.repository,
                "starters": list(manifest.agent.starters),
            }
        )

    @api.get("/api/marina/applet-hosts")
    @requires_auth
    def list_applet_hosts(request: Request) -> JSONResponse:
        return JSONResponse({"hosts": host_operator(request).host_bindings()})

    @api.put("/api/marina/applet-hosts/{hostname}")
    @requires_auth
    async def bind_applet_host(hostname: str, request: Request) -> JSONResponse:
        store = host_operator(request)
        body = await request_json(request)
        try:
            hostname = normalize_hostname(hostname)
            if hostname in reserved_hosts:
                raise ValueError("hostname is reserved for Marina or a legacy app")
            if hostname not in config.applet_domains:
                raise ValueError("hostname has not been provisioned for applets")
            if not isinstance(body, dict) or not isinstance(body.get("applet_id"), str):
                raise ValueError("body must contain an applet_id")
            applet_id = uuid.UUID(body["applet_id"])
            await run_in_threadpool(store.bind_host, hostname, applet_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        return JSONResponse({"hostname": hostname, "applet_id": str(applet_id)})

    @api.delete("/api/marina/applet-hosts/{hostname}", status_code=204)
    @requires_auth
    async def unbind_applet_host(hostname: str, request: Request) -> Response:
        store = host_operator(request)
        try:
            await run_in_threadpool(store.unbind_host, hostname)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="hostname binding not found") from error
        return Response(status_code=204)

    @api.get("/", include_in_schema=False)
    @requires_auth
    def landing(
        sort: Literal["name", "released"] = "released",
        order: Literal["asc", "desc"] = "desc",
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        return HTMLResponse(
            landing_page(
                applet_directory(applet_store, config.applet_origin, config.public_applet_origin), sort, order, page
            ),
            headers={"Cache-Control": DIRECTORY_CACHE_CONTROL},
        )

    @api.post("/api/marina/applets")
    @requires_auth
    async def publish_applet(request: Request, mode: AppletMode = AppletMode.PRIVATE) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=503, detail="Marina has no database configured")
        package = await uploaded_applet_package(request)
        try:
            publisher = config.local_publisher or identity_for(request, policy).user_id
            validate_release(package, None)
            published = await run_in_threadpool(
                applet_store.publish, package, publisher, operators=config.applet_operators, mode=mode
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except AppletForbidden as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        return publish_result_response(published, config.applet_origin, config.public_applet_origin)

    @api.post("/api/marina/applets/{applet_id}")
    @requires_auth
    async def update_applet(
        applet_id: uuid.UUID,
        base_version: int,
        request: Request,
        mode: AppletMode | None = None,
    ) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=503, detail="Marina has no database configured")
        package = await uploaded_applet_package(request)
        try:
            publisher = config.local_publisher or identity_for(request, policy).user_id
            validate_release(package, applet_id)
            operators = config.applet_operators | ({publisher} if config.iap_audience is None else set())
            published = await run_in_threadpool(
                applet_store.publish,
                package,
                publisher,
                applet_id,
                base_version,
                frozenset(operators),
                mode,
                reader_roles=(
                    config.runtime.named_readers
                    if config.runtime.required_apps.get(package.manifest.name or "") == applet_id
                    else ()
                ),
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
            raise HTTPException(status_code=403, detail=str(error)) from error
        except AppletConflict as error:
            raise HTTPException(status_code=409, detail="applet version or registered name conflicts") from error
        return publish_result_response(published, config.applet_origin, config.public_applet_origin)

    @api.get("/api/marina/applets/{applet_id}")
    @requires_auth
    async def applet_details(applet_id: uuid.UUID) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            current = await run_in_threadpool(applet_store.current_version, applet_id)
            mode = await run_in_threadpool(applet_store.access_mode, applet_id)
            versions = await run_in_threadpool(applet_store.versions, applet_id)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        return JSONResponse(
            {
                "id": str(applet_id),
                "current_version": current,
                "mode": mode.value,
                "url": applet_url(
                    f"/a/{applet_id}/",
                    mode,
                    config.applet_origin,
                    config.public_applet_origin,
                ),
                "authenticated_url": f"{config.applet_origin or ''}/a/{applet_id}/",
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

    @api.put("/api/marina/applets/{applet_id}/mode")
    @requires_auth
    async def set_applet_mode(applet_id: uuid.UUID, request: Request) -> JSONResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        body = await request_json(request)
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("mode"), str)
            or not isinstance(body.get("base_version"), int)
        ):
            raise HTTPException(status_code=400, detail="body must contain mode and integer base_version")
        try:
            mode = AppletMode(body["mode"])
        except ValueError as error:
            raise HTTPException(status_code=400, detail="mode must be private or public") from error
        try:
            await run_in_threadpool(
                applet_store.set_mode,
                applet_id,
                mode,
                body["base_version"],
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        except AppletConflict as error:
            raise HTTPException(status_code=409, detail="applet has a newer current version") from error
        path = f"/a/{applet_id}/"
        return JSONResponse(
            {
                "id": str(applet_id),
                "version": body["base_version"],
                "mode": mode.value,
                "url": applet_url(path, mode, config.applet_origin, config.public_applet_origin),
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
        try:
            await run_in_threadpool(
                applet_store.rollback,
                applet_id,
                body["version"],
                body["base_version"],
            )
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet or revision not found") from error
        except AppletConflict as error:
            raise HTTPException(status_code=409, detail="applet has a newer current version") from error
        return JSONResponse({"id": str(applet_id), "version": body["version"]})

    @api.delete("/api/marina/applets/{applet_id}", status_code=204)
    @requires_auth
    async def archive_applet(applet_id: uuid.UUID) -> Response:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="applet not found")
        try:
            await run_in_threadpool(applet_store.archive, applet_id)
            assert applet_runtime is not None
            await run_in_threadpool(applet_runtime.retain_versions, applet_id, set())
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="applet not found") from error
        return Response(status_code=204)

    @api.get("/a/{applet_id}", include_in_schema=False)
    @public
    def applet_without_slash(applet_id: uuid.UUID, request: Request) -> RedirectResponse:
        require_applet_read(applet_id, request)
        return RedirectResponse(f"/a/{applet_id}/")

    @api.get("/a/{applet_id}/", include_in_schema=False)
    @public
    def current_applet(applet_id: uuid.UUID, request: Request) -> RedirectResponse:
        require_applet_read(applet_id, request)
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
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    @public
    async def current_applet_api_read(applet_id: uuid.UUID, path: str, request: Request) -> Response:
        if config.surface is MarinaSurface.PUBLIC_APPLETS:
            await run_in_threadpool(require_applet_read, applet_id, request)
        identity = None if config.surface is MarinaSurface.PUBLIC_APPLETS else request_identity(request)
        return await dispatch_applet_api(applet_store, applet_runtime, applet_id, None, path, request, identity)

    @api.api_route(
        "/a/{applet_id}/v/{version}/api/{path:path}",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    @public
    async def versioned_applet_api_read(applet_id: uuid.UUID, version: int, path: str, request: Request) -> Response:
        if config.surface is MarinaSurface.PUBLIC_APPLETS:
            await run_in_threadpool(require_applet_read, applet_id, request)
        identity = None if config.surface is MarinaSurface.PUBLIC_APPLETS else request_identity(request)
        return await dispatch_applet_api(applet_store, applet_runtime, applet_id, version, path, request, identity)

    @api.api_route(
        "/a/{applet_id}/api/{path:path}",
        methods=["POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    @requires_auth
    async def current_applet_api_write(applet_id: uuid.UUID, path: str, request: Request) -> Response:
        return await dispatch_applet_api(
            applet_store,
            applet_runtime,
            applet_id,
            None,
            path,
            request,
            identity_for(request, policy),
        )

    @api.api_route(
        "/a/{applet_id}/v/{version}/api/{path:path}",
        methods=["POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    @requires_auth
    async def versioned_applet_api_write(applet_id: uuid.UUID, version: int, path: str, request: Request) -> Response:
        return await dispatch_applet_api(
            applet_store,
            applet_runtime,
            applet_id,
            version,
            path,
            request,
            identity_for(request, policy),
        )

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
            "Cache-Control": (
                "public, no-cache"
                if config.surface is MarinaSurface.PUBLIC_APPLETS
                else "private, max-age=31536000, immutable"
            ),
            "Content-Security-Policy": content_security_policy(record.manifest.connect_src),
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
    @public
    def applet_root(applet_id: uuid.UUID, version: int, request: Request) -> Response:
        require_applet_read(applet_id, request)
        return applet_file_response(applet_id, version, "", request)

    @api.api_route(
        "/a/{applet_id}/v/{version}/{path:path}",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    @public
    def applet_file(applet_id: uuid.UUID, version: int, path: str, request: Request) -> Response:
        require_applet_read(applet_id, request)
        return applet_file_response(applet_id, version, path, request)

    if config.applet_origin is not None:
        parsed_applet_origin = urlparse(config.applet_origin)
        applet_host = parsed_applet_origin.hostname
        assert applet_host is not None

        @api.middleware("http")
        async def isolate_applet_host(request: Request, call_next):
            host = request.headers.get("host", "").split(":", 1)[0].lower()
            path = request.url.path
            if request.scope.get("state", {}).get(NAMED_APPLET_HOST_STATE):
                return await call_next(request)
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
        for host in config.host_apps:
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

    api.mount(MCP_READ_PATH, AuthenticatedMount(read_mcp_app, policy))
    api.mount(MCP_PATH, AuthenticatedMount(mcp_app, policy))

    @api.api_route("/{name}", methods=["GET", "HEAD"], include_in_schema=False)
    @requires_auth
    def named_without_slash(name: str) -> RedirectResponse:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="app not found")
        try:
            applet_store.named_applet(name)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="app not found") from error
        return RedirectResponse(f"/{name}/")

    @api.api_route(
        "/{name}/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )
    @requires_auth
    async def named_application(name: str, path: str, request: Request) -> Response:
        if applet_store is None:
            raise HTTPException(status_code=404, detail="app not found")
        try:
            selected = await run_in_threadpool(applet_store.named_applet, name)
            record = await run_in_threadpool(applet_store.version, selected.id, selected.current_version)
        except AppletNotFound as error:
            raise HTTPException(status_code=404, detail="app not found") from error
        if path == "api" or path.startswith("api/"):
            return await dispatch_applet_api(
                applet_store,
                applet_runtime,
                selected.id,
                selected.current_version,
                path.removeprefix("api").lstrip("/"),
                request,
                identity_for(request, policy),
            )
        if request.method not in {"GET", "HEAD"}:
            raise HTTPException(status_code=405, detail="method not allowed")
        manifest = record.manifest
        connect_src = manifest.connect_src
        if manifest.agent is not None and config.agent_panel is not None:
            connect_src += (config.agent_panel.origin,)
        if path.startswith(DATA_PREFIX):
            data_url = manifest.data_url or prefix_join(config.data_root, name)
            return await serve_data_file(data_url, path.removeprefix(DATA_PREFIX), request, connect_src)
        versions = [selected.current_version]
        if path and "." in path.rsplit("/", 1)[-1]:
            # Old pages can request a hashed chunk after a newer frontend becomes current.
            versions += [
                item.version
                for item in await run_in_threadpool(applet_store.versions, selected.id)
                if item.version != selected.current_version
            ]
        for version in versions:
            try:
                response = await run_in_threadpool(applet_file_response, selected.id, version, path, request)
                break
            except HTTPException as error:
                if error.status_code != 404:
                    raise
        else:
            raise HTTPException(status_code=404, detail="application file not found")
        response.headers["Cache-Control"] = "private, no-cache"
        response.headers["Content-Security-Policy"] = content_security_policy(connect_src)
        return response

    authenticated = RouteAuthMiddleware(api, policy)
    if config.surface is MarinaSurface.PUBLIC_APPLETS:
        return public_applet_surface(authenticated)
    if config.applet_domains:
        if applet_store is None:
            raise ValueError(f"{APPLET_DOMAINS_ENV} requires a database")
        return named_applet_hosts(authenticated, applet_store, config.applet_domains)
    return authenticated
