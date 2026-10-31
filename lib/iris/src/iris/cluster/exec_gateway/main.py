# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone Connect gateway application."""

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Mount

from iris.cluster.exec_gateway.service import ExecGatewayService
from iris.rpc.exec_gateway_connect import ExecGatewayServiceASGIApplication


def gateway_application(service: ExecGatewayService) -> Starlette:
    """Serve typed execution RPCs and close every transport during shutdown."""

    @asynccontextmanager
    async def lifespan(app: Starlette):
        del app
        await service.start()
        try:
            yield
        finally:
            await service.shutdown()

    rpc = ExecGatewayServiceASGIApplication(service, read_max_bytes=256 * 1024)
    return Starlette(routes=[Mount(rpc.path, app=rpc)], lifespan=lifespan)
