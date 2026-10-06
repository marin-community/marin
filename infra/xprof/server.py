# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import signal
import subprocess
import time
from http.client import HTTPConnection
from pathlib import Path

from cheroot import wsgi
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.platforms.types import find_free_port
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY
from rigging.filesystem.s3_compat import configure_coreweave_s3

from infra.xprof.config import BACKEND_HOST, ENDPOINT_NAME, PORT_NAME, PROXY_TIMEOUT_SECONDS, PUBLIC_PATH
from infra.xprof.gateway import ProfileCache, ProfileStageManager, XprofGateway
from infra.xprof.release import XPROF_RS_BINARY_PATH
from infra.xprof.rust_proxy import RustProxy

logger = logging.getLogger(__name__)
STARTUP_TIMEOUT = 30


def _wait_for_backend(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"xprof-rs exited with status {process.returncode}")
        connection = HTTPConnection(BACKEND_HOST, port, timeout=1)
        try:
            connection.request("GET", "/version")
            if connection.getresponse().status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.1)
    raise TimeoutError("xprof-rs did not start")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()

    workdir = Path(os.environ["IRIS_WORKDIR"])
    cache_dir = workdir / "xprof-cache"
    cache = ProfileCache(cache_dir)
    backend_port = find_free_port()
    binary = Path.cwd() / XPROF_RS_BINARY_PATH
    # Iris extracts workspace zips with zipfile, which drops executable permissions.
    binary.chmod(0o755)
    backend = subprocess.Popen(
        [
            str(binary),
            "--logdir",
            str(cache_dir),
            "--host",
            BACKEND_HOST,
            "--port",
            str(backend_port),
            "--hide_capture_profile_button",
            "--enable_tab_name_label",
        ]
    )
    try:
        _wait_for_backend(backend, backend_port)
        app = XprofGateway(RustProxy(backend_port), ProfileStageManager(cache), PUBLIC_PATH)
        try:
            ctx = iris_ctx()
            job_info = get_job_info()
            if job_info is None:
                raise RuntimeError("No Iris job info available; XProf must run inside an Iris job")
            port = ctx.get_port(PORT_NAME) or find_free_port()
            address = f"http://{job_info.advertise_host}:{port}"
            endpoint_id = ctx.registry.register(
                ENDPOINT_NAME,
                address,
                {"job_id": ctx.job_id.to_wire(), PROXY_TIMEOUT_METADATA_KEY: str(PROXY_TIMEOUT_SECONDS)},
            )
            try:
                http_server = wsgi.Server((os.environ["IRIS_BIND_HOST"], port), app)

                def stop_server(_signum, _frame) -> None:
                    http_server.stop()

                signal.signal(signal.SIGTERM, stop_server)
                signal.signal(signal.SIGINT, stop_server)
                logger.info("XProf registered as %s at %s", ENDPOINT_NAME, address)
                http_server.start()
            finally:
                ctx.registry.unregister(endpoint_id)
        finally:
            app.shutdown()
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait()


if __name__ == "__main__":
    main()
