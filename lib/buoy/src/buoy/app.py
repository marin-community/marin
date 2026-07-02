# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Starlette app: a thin JSON API over the cache + an xprof reverse proxy.

Two slow actions are async to stay under the controller proxy's 30s cap:
``POST /api/mirror`` starts a background mirror and returns 202; the SPA polls
``GET /api/mirror_status`` until ``done``, then reads the cache. The first
profile open downloads the artifact + launches xprof lazily on the first
``/xprof`` request.

xprof embeds itself as a TensorBoard plugin and writes its UI state into
``window.parent`` history/location. ``/wrap`` interposes a same-origin frame so
those writes don't clobber the SPA's own URL (see design.md). The real
cross-origin fix would serve xprof from a subdomain endpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import posixpath
import re
from pathlib import Path

import httpx
import wandb
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from buoy import cache
from buoy.config import BuoyConfig
from buoy.mirror import MirrorManager, RunRef
from buoy.xprof import XprofCapacityError, XprofManager

logger = logging.getLogger("buoy.app")

# Hop-by-hop headers only. Content-Encoding/Content-Length are forwarded so the
# browser can decode xprof's responses — xprof serves some assets pre-gzipped
# regardless of Accept-Encoding, and we stream the raw body verbatim, so its
# encoding/length headers must travel with it (mirrors the controller proxy).
_DROP_RESPONSE_HEADERS = frozenset({"transfer-encoding", "connection", "keep-alive"})
XPROF_PROXY_TIMEOUT = 120.0


def _ref(request: Request, *, from_query: bool = True) -> RunRef:
    src = request.query_params if from_query else request.path_params
    return RunRef(src["entity"], src["project"], src.get("run") or src.get("run_id"))


def _prefix(cfg: BuoyConfig, ref: RunRef) -> str:
    return cache.run_prefix(cfg.cache_root, ref.entity, ref.project, ref.run_id)


# --------------------------------------------------------------------------- API


async def defaults(request: Request) -> JSONResponse:
    cfg: BuoyConfig = request.app.state.cfg
    return JSONResponse({"entity": cfg.default_entity})


async def list_entities(request: Request) -> JSONResponse:
    """The viewer's own entity plus any team entities, for the entity picker."""

    def _fetch() -> list[str]:
        viewer = wandb.Api().viewer
        entities: list[str] = []
        own = getattr(viewer, "entity", None)
        if own:
            entities.append(own)
        for team in getattr(viewer, "teams", None) or []:
            name = team if isinstance(team, str) else getattr(team, "name", None)
            if name and name not in entities:
                entities.append(name)
        return entities

    return JSONResponse({"entities": await asyncio.to_thread(_fetch)})


async def list_projects(request: Request) -> JSONResponse:
    entity = request.query_params.get("entity", request.app.state.cfg.default_entity)

    def _fetch() -> list[str]:
        projects = list(wandb.Api().projects(entity))
        # Most-recently-updated first when wandb exposes a timestamp; else API order.
        projects.sort(key=lambda p: str(getattr(p, "updated_at", "") or getattr(p, "lastUpdated", "")), reverse=True)
        return [p.name for p in projects]

    return JSONResponse({"projects": await asyncio.to_thread(_fetch)})


async def list_users(request: Request) -> JSONResponse:
    """Distinct run authors in a project, for the user filter dropdown.

    Scans a bounded window of recent runs (the recent-N run list alone misses
    authors whose latest run is older). The field is free-text regardless, so an
    unlisted user can still be typed.
    """
    entity = request.query_params.get("entity", request.app.state.cfg.default_entity)
    project = request.query_params.get("project", "")
    # Authors only (no per-run dict), so a deep scan is cheap (~3s for 1000 runs on
    # marin_moe) and captures the full contributor set, not just recent authors.
    limit = int(request.query_params.get("limit", "1000"))
    if not project:
        return JSONResponse({"error": "project required"}, status_code=400)

    def _fetch() -> list[str]:
        runs = wandb.Api().runs(f"{entity}/{project}", per_page=200, order="-created_at")
        users: set[str] = set()
        for i, run in enumerate(runs):
            author = getattr(run, "user", None)
            name = getattr(author, "username", None) or getattr(author, "name", None)
            if name:
                users.add(name)
            if i + 1 >= limit:
                break
        return sorted(users)

    return JSONResponse({"users": await asyncio.to_thread(_fetch)})


async def list_runs(request: Request) -> JSONResponse:
    entity = request.query_params.get("entity", request.app.state.cfg.default_entity)
    project = request.query_params.get("project", "")
    # Each run lazily fetches its author, so keep the page small. user/search filter
    # server-side so a run is findable regardless of recency (the recent-N window
    # otherwise hides an older author's runs).
    limit = int(request.query_params.get("limit", "50"))
    user = request.query_params.get("user", "").strip()
    search = request.query_params.get("search", "").strip()
    if not project:
        return JSONResponse({"error": "project required"}, status_code=400)

    def _fetch() -> list[dict]:
        api = wandb.Api()
        clauses: list[dict] = []
        if user:
            clauses.append({"username": user})
        if search:
            clauses.append({"display_name": {"$regex": re.escape(search)}})
        filters = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
        runs = api.runs(f"{entity}/{project}", filters=filters, per_page=min(limit, 100), order="-created_at")
        out = []
        for run in runs:
            author = getattr(run, "user", None)
            out.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "user": getattr(author, "username", None) or getattr(author, "name", None),
                    "state": run.state,
                    "created_at": str(run.created_at),
                }
            )
            if len(out) >= limit:
                break
        return out

    return JSONResponse({"runs": await asyncio.to_thread(_fetch)})


async def start_mirror(request: Request) -> JSONResponse:
    body = await request.json()
    ref = RunRef(body["entity"], body["project"], body["run_id"])
    request.app.state.mirror.start(ref, refresh=bool(body.get("refresh", False)))
    return JSONResponse({"state": "running"}, status_code=202)


async def mirror_status(request: Request) -> JSONResponse:
    return JSONResponse(request.app.state.mirror.status(_ref(request)))


async def get_manifest(request: Request) -> JSONResponse:
    cfg: BuoyConfig = request.app.state.cfg
    ref = _ref(request)
    manifest = cache.read_manifest(_prefix(cfg, ref))
    if manifest is None:
        return JSONResponse({"error": "not mirrored"}, status_code=404)
    return JSONResponse(manifest)


async def get_metrics(request: Request) -> JSONResponse:
    cfg: BuoyConfig = request.app.state.cfg
    ref = _ref(request)
    prefix = _prefix(cfg, ref)
    manifest = cache.read_manifest(prefix)
    if manifest is None:
        return JSONResponse({"error": "not mirrored"}, status_code=404)
    available = set(manifest["history"]["columns"])
    requested = [k for k in request.query_params.get("keys", "").split(",") if k]
    keys = [k for k in requested if k in available] or sorted(available)

    def _read() -> dict:
        # Columnar {x, y} (not a dict per point): a fraction of the JSON, and Plotly
        # consumes the arrays directly — no per-point object churn on either side.
        frame = cache.read_history(prefix, ["_step", *keys])
        series: dict[str, dict] = {}
        for key in keys:
            col = frame[["_step", key]].dropna()
            series[key] = {"x": col["_step"].astype("int64").tolist(), "y": col[key].astype(float).tolist()}
        return series

    return JSONResponse({"metrics": await asyncio.to_thread(_read)})


async def get_config(request: Request) -> JSONResponse:
    cfg: BuoyConfig = request.app.state.cfg
    config = cache.read_json(posixpath.join(_prefix(cfg, _ref(request)), "config.json"))
    if config is None:
        return JSONResponse({"error": "not mirrored"}, status_code=404)
    return JSONResponse(config)


async def get_summary(request: Request) -> JSONResponse:
    """The run's summary metrics (final logged values), for the summary tab."""
    cfg: BuoyConfig = request.app.state.cfg
    summary = cache.read_json(posixpath.join(_prefix(cfg, _ref(request)), "summary.json"))
    if summary is None:
        return JSONResponse({"error": "not mirrored"}, status_code=404)
    return JSONResponse(summary)


# ------------------------------------------------------------------- xprof embed


async def prepare_profile(request: Request) -> JSONResponse:
    """Start (download + launch) xprof in the background; 202, poll prepare_status.

    Kept off the request path because a cold profile (hundreds of MB to download
    + xprof boot) can exceed the controller proxy's 30s cap.
    """
    body = await request.json()
    ref = RunRef(body["entity"], body["project"], body["run_id"])
    manifest = cache.read_manifest(_prefix(request.app.state.cfg, ref))
    if manifest is None:
        return JSONResponse({"error": "not mirrored"}, status_code=404)
    if not manifest.get("profile"):
        return JSONResponse({"error": "run has no profile"}, status_code=409)
    request.app.state.xprof.start_prepare(ref.key, manifest["profile"]["logdir"])
    return JSONResponse({"state": "preparing"}, status_code=202)


async def profile_status(request: Request) -> JSONResponse:
    ref = _ref(request)
    return JSONResponse(request.app.state.xprof.prepare_status(ref.key))


async def xprof_wrap(request: Request) -> Response:
    """Same-origin frame that hosts the real xprof iframe (see module docstring)."""
    return Response(
        "<!doctype html><html><head><meta charset=utf-8>"
        "<style>html,body{margin:0;height:100%}iframe{border:0;width:100%;height:100%}</style>"
        "</head><body><script>"
        "var p=location.pathname.replace('/wrap/','/xprof/');"
        "if(!p.endsWith('/'))p+='/';"
        "var f=document.createElement('iframe');f.src=p+'data/plugin/profile/';"
        "document.body.appendChild(f);"
        "</script></body></html>",
        media_type="text/html",
    )


async def xprof_proxy(request: Request) -> Response:
    cfg: BuoyConfig = request.app.state.cfg
    ref = _ref(request, from_query=False)
    sub = request.path_params["sub"]
    xprof = request.app.state.xprof
    # Warm path: a live xprof serves every asset/data request, so skip the GCS
    # manifest read entirely once the process is up (the common case for a trace).
    port = xprof.port_if_running(ref.key)
    if port is None:
        manifest = cache.read_manifest(_prefix(cfg, ref))
        if manifest is None:
            return Response("not mirrored", status_code=404)
        if not manifest.get("profile"):
            return Response("run has no profile", status_code=409)
        try:
            port = await asyncio.to_thread(xprof.ensure, ref.key, manifest["profile"]["logdir"])
        except XprofCapacityError as exc:
            return Response(str(exc), status_code=503)

    url = "/" + sub + (("?" + request.url.query) if request.url.query else "")
    client: httpx.AsyncClient = request.app.state.http
    # Forward the client's Accept-Encoding and stream the raw body back unchanged,
    # so whatever encoding xprof picks stays consistent with the headers we forward.
    accept_encoding = request.headers.get("accept-encoding", "identity")
    upstream = client.build_request("GET", f"http://127.0.0.1:{port}{url}", headers={"accept-encoding": accept_encoding})
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.TimeoutException:
        return Response("xprof upstream timeout", status_code=504)
    except httpx.HTTPError as exc:
        logger.warning("xprof upstream error for %s: %s", ref.key, exc)
        return Response(f"xprof upstream error: {exc!r}", status_code=502)
    headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=headers,
        media_type=resp.headers.get("content-type"),
        background=BackgroundTask(resp.aclose),
    )


# The dashboard is a bundled Vue SPA built into dashboard/dist by `npm run build`
# (gitignored; shipped in the Iris bundle via GENERATED_ARTIFACT_GLOBS). Resolve its
# dist dir: env override → the in-repo build output next to this package.
STATIC_MAX_AGE = 31_536_000  # 1 year; rsbuild asset filenames are content-hashed (immutable)


class _CacheControlStatic:
    """Wrap StaticFiles to add a long immutable Cache-Control on the hashed assets."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self._header = f"public, max-age={STATIC_MAX_AGE}, immutable".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_cached(message: dict) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"cache-control", self._header)]
            await send(message)

        await self._app(scope, receive, send_cached)


def _dashboard_dist() -> Path:
    override = os.environ.get("BUOY_DASHBOARD_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "dashboard" / "dist"


_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>buoy</title>"
    "<body style='font-family:system-ui;margin:3rem'><h1>buoy</h1>"
    "<p>Dashboard not built — run "
    "<code>npm --prefix lib/buoy/dashboard install &amp;&amp; npm --prefix lib/buoy/dashboard run build</code>.</p>"
)


def _index_html(dist: Path, forwarded_prefix: str) -> HTMLResponse:
    """Serve dist/index.html, rewriting ``<base href="/">`` to the proxy sub-path.

    The controller proxy sets ``X-Forwarded-Prefix`` (e.g. ``/proxy/buoy``) in
    path-style mode; rewriting the base makes the SPA's relative asset and API URLs
    resolve under it. Empty prefix (direct access) leaves the base at ``/``.
    """
    index_path = dist / "index.html"
    if not index_path.is_file():
        return HTMLResponse(_NOT_BUILT_HTML, status_code=503)
    html = index_path.read_text(encoding="utf-8")
    prefix = forwarded_prefix.rstrip("/")
    if prefix:
        html = html.replace('<base href="/"', f'<base href="{prefix}/"', 1)
    return HTMLResponse(html)


def build_app(cfg: BuoyConfig) -> Starlette:
    dist = _dashboard_dist()

    async def index(request: Request) -> HTMLResponse:
        return _index_html(dist, request.headers.get("x-forwarded-prefix", ""))

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.cfg = cfg
        app.state.mirror = MirrorManager(cfg)
        app.state.xprof = XprofManager(cfg)
        # max_keepalive_connections=0 disables connection reuse: the browser fires a
        # burst of xprof asset/trace requests and cancels some on refresh, leaving a
        # pooled connection half-read; the next request on it dies mid-stream with a
        # RemoteProtocolError (intermittent 500s). Same fix as the controller proxy.
        app.state.http = httpx.AsyncClient(
            timeout=XPROF_PROXY_TIMEOUT,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=0),
        )
        try:
            yield
        finally:
            app.state.xprof.shutdown()
            await app.state.http.aclose()

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/defaults", defaults),
            Route("/api/entities", list_entities),
            Route("/api/projects", list_projects),
            Route("/api/users", list_users),
            Route("/api/runs", list_runs),
            Route("/api/mirror", start_mirror, methods=["POST"]),
            Route("/api/mirror_status", mirror_status),
            Route("/api/manifest", get_manifest),
            Route("/api/metrics", get_metrics),
            Route("/api/config", get_config),
            Route("/api/summary", get_summary),
            Route("/api/profile_prepare", prepare_profile, methods=["POST"]),
            Route("/api/profile_status", profile_status),
            Route("/wrap/{entity}/{project}/{run_id}", xprof_wrap),
            Route("/xprof/{entity}/{project}/{run_id}/{sub:path}", xprof_proxy, methods=["GET", "HEAD"]),
            Mount(
                "/static", _CacheControlStatic(StaticFiles(directory=dist / "static", check_dir=False)), name="static"
            ),
        ],
        lifespan=lifespan,
    )
