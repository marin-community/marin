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
import posixpath
from pathlib import Path

import httpx
import wandb
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from buoy import cache
from buoy.config import BuoyConfig
from buoy.mirror import MirrorManager, RunRef
from buoy.xprof import XprofCapacityError, XprofManager

logger = logging.getLogger("buoy.app")

# Read the SPA at import so it travels with the (cloudpickle-by-value) deploy
# rather than depending on a static file being present on the worker.
INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text()
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


async def list_runs(request: Request) -> JSONResponse:
    entity = request.query_params.get("entity", request.app.state.cfg.default_entity)
    project = request.query_params.get("project", "")
    limit = int(request.query_params.get("limit", "50"))
    if not project:
        return JSONResponse({"error": "project required"}, status_code=400)

    def _fetch() -> list[dict]:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", per_page=limit)
        out = []
        for run in runs:
            out.append({"id": run.id, "name": run.name, "state": run.state, "created_at": str(run.created_at)})
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
    request.app.state.mirror.refresh_if_running(ref, manifest)
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
        frame = cache.read_history(prefix, ["_step", *keys])
        series: dict[str, list[dict]] = {}
        for key in keys:
            col = frame[["_step", key]].dropna()
            series[key] = [{"step": int(s), "value": float(v)} for s, v in zip(col["_step"], col[key], strict=True)]
        return series

    return JSONResponse({"metrics": await asyncio.to_thread(_read)})


async def get_config(request: Request) -> JSONResponse:
    cfg: BuoyConfig = request.app.state.cfg
    config = cache.read_json(posixpath.join(_prefix(cfg, _ref(request)), "config.json"))
    if config is None:
        return JSONResponse({"error": "not mirrored"}, status_code=404)
    return JSONResponse(config)


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
    manifest = cache.read_manifest(_prefix(cfg, ref))
    if manifest is None:
        return Response("not mirrored", status_code=404)
    if not manifest.get("profile"):
        return Response("run has no profile", status_code=409)

    try:
        port = await asyncio.to_thread(request.app.state.xprof.ensure, ref.key, manifest["profile"]["logdir"])
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


async def index(request: Request) -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


def build_app(cfg: BuoyConfig) -> Starlette:
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
            Route("/api/runs", list_runs),
            Route("/api/mirror", start_mirror, methods=["POST"]),
            Route("/api/mirror_status", mirror_status),
            Route("/api/manifest", get_manifest),
            Route("/api/metrics", get_metrics),
            Route("/api/config", get_config),
            Route("/api/profile_prepare", prepare_profile, methods=["POST"]),
            Route("/api/profile_status", profile_status),
            Route("/wrap/{entity}/{project}/{run_id}", xprof_wrap),
            Route("/xprof/{entity}/{project}/{run_id}/{sub:path}", xprof_proxy, methods=["GET", "HEAD"]),
        ],
        lifespan=lifespan,
    )
