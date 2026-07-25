# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import signal
from pathlib import Path

from cheroot import wsgi
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.platforms.types import find_free_port
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY
from rigging.filesystem.s3_compat import configure_coreweave_s3
from xprof import server as xprof_server
from xprof.convert import _pywrap_profiler_plugin

from infra.xprof.config import ENDPOINT_NAME, PORT_NAME, PROXY_TIMEOUT_SECONDS, PUBLIC_PATH
from infra.xprof.gateway import ProfileCache, ProfileStageManager, XprofGateway

logger = logging.getLogger(__name__)


def _xprof_application(grpc_port: int):
    worker_address = f"0.0.0.0:{grpc_port}"
    _pywrap_profiler_plugin.initialize_stubs(worker_address)
    _pywrap_profiler_plugin.start_grpc_server(grpc_port, 1)
    context = xprof_server.TBContext(None, xprof_server.DataProvider(None), xprof_server.TBContext.Flags(False))
    context.hide_capture_profile_button = True
    context.enable_tab_name_label = True
    context.src_prefix = None
    plugin = xprof_server.ProfilePluginLoader().load(context)
    if plugin is None:
        raise RuntimeError("XProf profile plugin failed to load")
    return xprof_server.make_wsgi_app(plugin)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()

    workdir = Path(os.environ["IRIS_WORKDIR"])
    grpc_port = int(os.environ["XPROF_GRPC_PORT"])
    profiles = ProfileStageManager(ProfileCache(workdir / "xprof-cache"))
    app = XprofGateway(_xprof_application(grpc_port), profiles, PUBLIC_PATH)

    ctx = iris_ctx()
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("No Iris job info available; XProf must run inside an Iris job")
    port = ctx.get_port(PORT_NAME)
    if port == 0:
        port = find_free_port()
    address = f"http://{job_info.advertise_host}:{port}"
    endpoint_id = ctx.registry.register(
        ENDPOINT_NAME,
        address,
        {"job_id": ctx.job_id.to_wire(), PROXY_TIMEOUT_METADATA_KEY: str(PROXY_TIMEOUT_SECONDS)},
    )
    http_server = wsgi.Server((os.environ["IRIS_BIND_HOST"], port), app)

    def stop_server(_signum, _frame) -> None:
        http_server.stop()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    logger.info("XProf registered as %s at %s", ENDPOINT_NAME, address)
    try:
        http_server.start()
    finally:
        app.shutdown()
        ctx.registry.unregister(endpoint_id)


if __name__ == "__main__":
    main()
