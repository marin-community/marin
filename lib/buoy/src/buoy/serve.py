# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris job entrypoint: install xprof, serve the app, register the endpoint, block.

Reachable through the controller proxy at ``/proxy/<endpoint, '/'->'.'>/``. xprof
is installed into a workdir venv at startup — its heavy, version-pinned deps
would fight the workspace env, and the task container mounts ``/tmp`` noexec so
the binary must live on an exec-allowed mount.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import subprocess
import threading

import uvicorn
from iris.client import iris_ctx
from iris.cluster.backends.types import wait_for_port
from iris.cluster.client.job_info import get_job_info

from buoy.app import build_app
from buoy.config import BuoyConfig

logger = logging.getLogger("buoy.serve")


def install_xprof() -> str:
    """Install xprof into a workdir venv at startup; return its binary path."""
    venv = os.path.join(os.getcwd(), ".xprof-venv")
    uv = shutil.which("uv") or "uv"
    subprocess.run([uv, "venv", venv, "--python", "3.11"], check=True)
    subprocess.run([uv, "pip", "install", "--python", f"{venv}/bin/python", "xprof"], check=True)
    return f"{venv}/bin/xprof"


def serve_in_job(endpoint_name: str = "/buoy") -> None:
    logging.basicConfig(level=logging.INFO, format="[buoy] %(asctime)s %(message)s")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY was not injected into the buoy job")
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("serve_in_job must run inside an Iris job")
    ctx = iris_ctx()
    port = ctx.get_port("http")
    advertise_host = job_info.advertise_host

    cfg = BuoyConfig.from_env()
    if not cfg.xprof_bin:
        logger.info("installing xprof (one-time)…")
        cfg = dataclasses.replace(cfg, xprof_bin=install_xprof())
    logger.info("cache_root=%s xprof_bin=%s", cfg.cache_root, cfg.xprof_bin)

    server = uvicorn.Server(uvicorn.Config(build_app(cfg), host=advertise_host, port=port, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    if not wait_for_port(port, advertise_host, 30.0):
        raise RuntimeError(f"buoy app did not bind on {advertise_host}:{port}")

    address = f"http://{advertise_host}:{port}"
    endpoint_id = ctx.registry.register(endpoint_name, address, {"kind": "buoy"})
    logger.info("registered buoy endpoint name=%s address=%s id=%s", endpoint_name, address, endpoint_id)
    try:
        thread.join()
    finally:
        try:
            ctx.registry.unregister(endpoint_id)
        except Exception:
            logger.warning("failed to unregister endpoint id=%s", endpoint_id, exc_info=True)
