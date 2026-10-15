# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend services and MCP projections, plus adapters for source-level unit tests.

Production backends are loaded from immutable Postgres packages by AppletRuntime.
Application migrations receive the release transaction's Connection.
"""

import importlib
import importlib.util
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
    """An application's HTTP service and full and read-only MCP projections."""

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


def _module(manifest: AppManifest):
    apps_dir = str(manifest.root.parent)
    if apps_dir not in sys.path:
        sys.path.insert(0, apps_dir)
    backend = manifest.root / "backend"
    if not (backend / "__init__.py").is_file():
        return importlib.import_module(f"{manifest.name}.{APP_MODULE}")

    # Keep the URL name free for an installed package, while retaining relative imports.
    package_name = f"_marina_app_{manifest.name.replace('-', '_')}"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            backend / "__init__.py",
            submodule_search_locations=[str(backend)],
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            del sys.modules[package_name]
            raise
    return importlib.import_module(f"{package_name}.{APP_MODULE}")


def create_api(manifest: AppManifest, services: Services) -> RegisteredApi:
    factory: Callable[[Services], RegisteredApi] = getattr(_module(manifest), CREATE_API)
    api = factory(services)
    if not isinstance(api, RegisteredApi):
        raise TypeError(f"{manifest.name}.{APP_MODULE}.{CREATE_API} must return RegisteredApi")
    return api


def data_url_for(manifest: AppManifest, data_root: str) -> str:
    """The app's explicit data URL, or its directory under the deployment data root."""
    return manifest.data_url or prefix_join(data_root, manifest.name)


def services_for(manifest: AppManifest, data_root: str, database: DatabaseSpec | None) -> Services:
    return Services(name=manifest.name, data_url=data_url_for(manifest, data_root), database=database)
