# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP discovery follows the selected Postgres revision on every request."""

import logging
from collections.abc import Sequence

from fastmcp import FastMCP
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool
from fastmcp.utilities.versions import VersionSpec
from starlette.concurrency import run_in_threadpool

from marina.applets import AppletBackendUnavailable, AppletNotFound, AppletRuntime, AppletStore

logger = logging.getLogger(__name__)


class ReleasedApplications(Provider):
    def __init__(self, store: AppletStore, runtime: AppletRuntime, *, read_only: bool):
        super().__init__()
        self.store = store
        self.runtime = runtime
        self.read_only = read_only

    def current_server(self) -> FastMCP:
        server = FastMCP("Released applications")
        for applet in self.store.active_applets():
            try:
                version = self.store.version(applet.id, applet.current_version)
                if version.manifest.python_entrypoint is None:
                    continue
                registered = self.runtime.registered_api(applet.id, applet.current_version)
            except (AppletNotFound, AppletBackendUnavailable):
                # One unavailable backend must not remove discovery for every application.
                logger.exception("Cannot load MCP operations for %s v%s", applet.id, applet.current_version)
                continue
            if registered is not None:
                server.mount(
                    registered.read_mcp if self.read_only else registered.mcp, namespace=applet.name or str(applet.id)
                )
        return server

    async def _list_tools(self) -> Sequence[Tool]:
        server = await run_in_threadpool(self.current_server)
        return await server.list_tools()

    async def _get_tool(self, name: str, version: VersionSpec | None = None) -> Tool | None:
        server = await run_in_threadpool(self.current_server)
        return await server.get_tool(name, version)
