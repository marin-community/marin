# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Python apps: what an app package may define and what the kernel hands it.

A Python app is a package at ``apps/<name>/`` (it has an ``__init__.py``). Its ``app``
module defines ``create_api(services) -> RegisteredApi``. The kernel mounts its ASGI app
at ``/<name>/api/`` behind the same authentication as every other route and generates
MCP tools from its explicitly marked OpenAPI operations. The module may also define
``migrate(engine)``, which ``marina migrate`` runs against the app's schema before a
deploy serves traffic. A static app has no package and only a ``dist``.
"""

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI
from fastmcp import FastMCP
from rigging.filesystem.storage_path import prefix_join
from sqlalchemy.engine import Engine
from starlette.types import ASGIApp

from marina.db import DatabaseSpec, engine_for
from marina.manifest import AppManifest
from marina.mcp import OperationRisk, mcp_for_api

APP_MODULE = "app"
CREATE_API = "create_api"
MIGRATE = "migrate"


@dataclass(frozen=True)
class Services:
    """What the kernel offers a Python app."""

    name: str
    # The app's directory under the data root, as an fsspec URL.
    data_url: str
    database: DatabaseSpec | None

    def engine(self) -> Engine:
        """A new engine on the app's own schema, created on first use so a kernel without a
        database still serves the apps that need none."""
        if self.database is None:
            raise RuntimeError(f"app {self.name} needs a database but the kernel has none configured")
        return engine_for(self.database, self.name)


@dataclass(frozen=True)
class RegisteredApi:
    """A checked-in app's HTTP service and full and read-only MCP projections."""

    app: ASGIApp
    mcp: FastMCP
    read_mcp: FastMCP


def registered_api(api: FastAPI, *, mounted_app: ASGIApp | None = None) -> RegisteredApi:
    """Generate MCP tools while optionally mounting a lifecycle wrapper around the API."""
    app = api if mounted_app is None else mounted_app
    return RegisteredApi(
        app=app,
        mcp=mcp_for_api(api, app),
        read_mcp=mcp_for_api(api, app, frozenset({OperationRisk.READ})),
    )


def is_python_app(manifest: AppManifest) -> bool:
    return (manifest.root / "__init__.py").is_file()


def _module(manifest: AppManifest):
    apps_dir = str(manifest.root.parent)
    if apps_dir not in sys.path:
        sys.path.insert(0, apps_dir)
    return importlib.import_module(f"{manifest.name}.{APP_MODULE}")


def create_api(manifest: AppManifest, services: Services) -> RegisteredApi:
    factory: Callable[[Services], RegisteredApi] = getattr(_module(manifest), CREATE_API)
    api = factory(services)
    if not isinstance(api, RegisteredApi):
        raise TypeError(f"{manifest.name}.{APP_MODULE}.{CREATE_API} must return RegisteredApi")
    return api


def migration(manifest: AppManifest) -> Callable[[Engine], None] | None:
    return getattr(_module(manifest), MIGRATE, None)


def data_url_for(data_root: str, app: str) -> str:
    """The app's directory under the data root, whether that root is local or ``gs://``."""
    return prefix_join(data_root, app)


def services_for(manifest: AppManifest, data_root: str, database: DatabaseSpec | None) -> Services:
    return Services(name=manifest.name, data_url=data_url_for(data_root, manifest.name), database=database)
