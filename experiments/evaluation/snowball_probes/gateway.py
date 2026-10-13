# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward scoped external requests to the private, cluster-local Ortet relay."""

import argparse
import asyncio
import contextlib
import threading

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import EndpointAccess, JobName
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background

from experiments.evaluation.snowball_probes.config import ORTET_RELAYS


def run(cluster: str) -> None:
    ctx = iris_ctx()
    info = get_job_info()
    assert info is not None
    assert ctx.client is not None
    client = ctx.client
    api = FastAPI()

    @api.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> StreamingResponse:
        token = request.headers.get("x-ortet-api-token")
        if not token:
            raise HTTPException(401, "Ortet API token required")
        resolution = await asyncio.to_thread(
            client.resolver_for_job(JobName.from_string(ORTET_RELAYS[cluster])).resolve, "glm-5.3"
        )
        if not resolution.endpoints:
            raise HTTPException(503, "Ortet relay is not registered on this cluster")
        base = resolution.endpoints[0].url.rstrip("/")
        http = httpx.AsyncClient(timeout=httpx.Timeout(1200, connect=15))
        headers = {"authorization": f"Bearer {token}", "content-type": "application/json"}
        try:
            upstream = await http.send(
                http.build_request(request.method, f"{base}/v1/{path}", headers=headers, content=await request.body()),
                stream=True,
            )
        except BaseException:
            await http.aclose()
            raise

        async def content():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await http.aclose()

        return StreamingResponse(
            content(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type")
        )

    with contextlib.ExitStack() as stack:
        sock = stack.enter_context(bind_serving_socket(info.advertise_host, 0))
        server = stack.enter_context(serve_app_background(api, sock, name="ortet-probe-gateway"))
        stack.enter_context(
            ctx.registry.registered(
                "ortet",
                f"http://{info.advertise_host}:{sock.getsockname()[1]}",
                metadata={"kind": "ortet-probe-gateway"},
                access=EndpointAccess.ENDPOINT_ACCESS_LINK,
            )
        )
        stop = threading.Event()
        while not stop.wait(15):
            if not server.is_alive():
                raise RuntimeError("Ortet probe gateway stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=ORTET_RELAYS, required=True)
    run(parser.parse_args().cluster)


if __name__ == "__main__":
    main()
