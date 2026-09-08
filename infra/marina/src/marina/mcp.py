# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP tools generated from checked-in Marina application APIs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import httpx2
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.server.providers.openapi import (
    MCPType,
    OpenAPIResource,
    OpenAPIResourceTemplate,
    OpenAPITool,
    RouteMap,
)
from fastmcp.server.transforms.search import BM25SearchTransform
from fastmcp.utilities.openapi import HTTPRoute, parse_openapi_to_http_routes
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict
from starlette.types import ASGIApp

MARINA_EXTENSION = "x-marina"
OPERATION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class OperationRisk(StrEnum):
    """The maximum side effect an operation may have."""

    READ = "read"
    WRITE = "write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class OperationExtension(BaseModel):
    """Marina metadata attached to an agent-visible OpenAPI operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk: OperationRisk


def operation_extension(risk: OperationRisk) -> dict[str, object]:
    """Return FastAPI ``openapi_extra`` for an agent-visible operation."""
    extension = OperationExtension(risk=risk)
    return {MARINA_EXTENSION: extension.model_dump(mode="json")}


def _registered_operations(document: dict[str, Any]) -> dict[str, OperationExtension]:
    result: dict[str, OperationExtension] = {}
    for route in parse_openapi_to_http_routes(document):
        raw_extension = route.extensions.get(MARINA_EXTENSION)
        if raw_extension is None:
            continue
        extension = OperationExtension.model_validate(raw_extension)
        operation_id = route.operation_id
        if not isinstance(operation_id, str) or OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
            raise ValueError(
                f"agent-visible operation {route.method} {route.path} needs an operationId matching "
                f"{OPERATION_ID_PATTERN.pattern}"
            )
        if operation_id in result:
            raise ValueError(f"duplicate agent-visible operation id {operation_id!r}")
        if route.description is None or not route.description.strip():
            raise ValueError(f"agent-visible operation {operation_id!r} needs a description")
        result[operation_id] = extension
    return result


def _annotations(risk: OperationRisk) -> ToolAnnotations:
    if risk is OperationRisk.READ:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    if risk is OperationRisk.WRITE:
        return ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    if risk is OperationRisk.EXTERNAL_WRITE:
        return ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
    return ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)


def mcp_for_api(api: FastAPI, mounted_app: ASGIApp) -> FastMCP:
    """Generate an executable, fail-closed MCP server from one application API."""
    openapi = api.openapi()
    operations = _registered_operations(openapi)
    client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=mounted_app), base_url="http://marina")

    def route_type(route: HTTPRoute, _default: MCPType) -> MCPType | None:
        if route.operation_id in operations:
            return MCPType.TOOL
        return None

    def annotate(
        route: HTTPRoute,
        component: OpenAPITool | OpenAPIResource | OpenAPIResourceTemplate,
    ) -> None:
        if not isinstance(component, OpenAPITool):
            return
        assert route.operation_id is not None
        risk = operations[route.operation_id].risk
        component.annotations = _annotations(risk)
        component.meta = {**(component.meta or {}), "marina": {"risk": risk.value}}

    return FastMCP.from_openapi(
        openapi_spec=openapi,
        client=client,
        name=api.title,
        route_maps=[RouteMap(mcp_type=MCPType.EXCLUDE)],
        route_map_fn=route_type,
        mcp_component_fn=annotate,
    )


def marina_mcp(servers: Mapping[str, FastMCP]) -> FastMCP:
    """Compose application MCP servers behind on-demand tool discovery."""
    result = FastMCP(
        "Marina",
        instructions="Use find_tool to discover Marina application operations, then call_tool to execute one.",
    )
    for namespace, server in sorted(servers.items()):
        result.mount(server, namespace=namespace)
    result.add_transform(BM25SearchTransform(search_tool_name="find_tool", call_tool_name="call_tool"))
    return result
