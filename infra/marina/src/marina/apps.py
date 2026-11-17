# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend services and MCP projections for released application packages.

Production backends are loaded from immutable Postgres packages by AppletRuntime.
Application migrations receive the release transaction's Connection.
"""

from dataclasses import dataclass
from typing import Protocol

from fastapi import FastAPI
from fastmcp import FastMCP
from sqlalchemy.engine import Engine
from starlette.types import ASGIApp

from marina.mcp import OperationRisk, mcp_for_api


class Services(Protocol):
    """What the kernel offers a Python app."""

    name: str
    data_url: str

    def engine(self) -> Engine: ...


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
