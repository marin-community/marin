#!/usr/bin/env python3
"""Async round-robin proxy for vLLM backends, built on aiohttp.

Reuses TCP connections per backend, handles many concurrent in-flight
chat-completion requests cleanly. Reloads the backends file every few seconds.

Usage:
    python vllm_router_async.py --port 8001 --backends backends.txt
"""

import argparse
import asyncio
import itertools
import os
import time

import aiohttp
from aiohttp import web

REQUEST_TIMEOUT = aiohttp.ClientTimeout(
    total=None,  # no hard cap on full request
    connect=10,
    sock_connect=10,
    sock_read=1800,  # 30 min for a slow read between bytes
)

_backends = []
_cycler = None
_backends_lock = asyncio.Lock()
_backends_path = ""
_last_load = 0.0
_RELOAD_INTERVAL = 5.0


async def _load_backends():
    global _backends, _cycler, _last_load
    try:
        with open(_backends_path) as f:
            new = [
                ln.strip().rstrip("/")
                for ln in f
                if ln.strip() and not ln.startswith("#")
            ]
    except FileNotFoundError:
        new = []
    async with _backends_lock:
        if new != _backends:
            _backends = new
            _cycler = itertools.cycle(_backends) if _backends else None
            print(
                f"[router] backends reloaded: {len(_backends)} entries: {_backends}",
                flush=True,
            )
        _last_load = time.time()


async def _next_backend():
    if time.time() - _last_load > _RELOAD_INTERVAL:
        await _load_backends()
    async with _backends_lock:
        if not _backends:
            return None
        return next(_cycler)


async def _proxy(request: web.Request):
    backend = await _next_backend()
    if not backend:
        return web.json_response({"error": "no backends available"}, status=503)

    # Compose downstream URL: strip /v1 from backend if request already has it
    path = request.rel_url.path
    if backend.endswith("/v1") and path.startswith("/v1"):
        upstream = backend[:-3] + path
    else:
        upstream = backend + path
    if request.rel_url.query_string:
        upstream += "?" + request.rel_url.query_string

    # Forward body; drop hop-by-hop headers
    body = await request.read() if request.body_exists else None
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower()
        not in ("host", "content-length", "connection", "transfer-encoding")
    }

    session = request.app["session"]
    try:
        async with session.request(
            request.method,
            upstream,
            data=body,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
        ) as upstream_resp:
            resp_headers = {
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower()
                not in ("transfer-encoding", "connection", "content-encoding")
            }
            resp = web.StreamResponse(
                status=upstream_resp.status,
                headers=resp_headers,
            )
            await resp.prepare(request)
            async for chunk in upstream_resp.content.iter_chunked(64 * 1024):
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return web.json_response(
            {"error": f"{type(e).__name__}: {e}", "backend": backend, "url": upstream},
            status=502,
        )


async def _status(request: web.Request):
    await _load_backends()
    async with _backends_lock:
        return web.json_response({"backends": list(_backends)})


async def _on_startup(app: web.Application):
    timeout = aiohttp.ClientTimeout(connect=10)
    connector = aiohttp.TCPConnector(
        limit=0, limit_per_host=0, keepalive_timeout=600, force_close=False
    )
    app["session"] = aiohttp.ClientSession(
        connector=connector, timeout=timeout, auto_decompress=False
    )
    await _load_backends()


async def _on_cleanup(app: web.Application):
    await app["session"].close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--backends", required=True, help="path to backends file (one URL per line)"
    )
    args = p.parse_args()

    global _backends_path
    _backends_path = os.path.abspath(args.backends)

    app = web.Application(client_max_size=1024 * 1024 * 1024)  # 1 GiB upload cap
    app.router.add_get("/router/status", _status)
    app.router.add_route("*", "/{tail:.*}", _proxy)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    print(
        f"[router] listening on {args.host}:{args.port} (backends file: {_backends_path})",
        flush=True,
    )
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
