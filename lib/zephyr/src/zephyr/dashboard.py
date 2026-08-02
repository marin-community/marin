# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator-owned Connect service and browser dashboard."""

import html
import importlib.resources
from typing import Protocol

from connectrpc.request import RequestContext
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp

from zephyr.plan import Join, PhysicalPlan
from zephyr.rpc import dashboard_pb2
from zephyr.rpc.dashboard_connect import CoordinatorDashboardServiceASGIApplication
from zephyr.worker_context import Aggregation, CounterEntry

DEFAULT_COUNTER_LIMIT = 100
MAX_COUNTER_LIMIT = 500
DEFAULT_WORKER_LIMIT = 50
MAX_WORKER_LIMIT = 200
DEFAULT_METRIC_POINTS = 200
MAX_METRIC_POINTS = 500

_DASHBOARD_SERVICE_PATH = "zephyr.dashboard.v1.CoordinatorDashboardService"
_BASE_ELEMENT = '<base href="/"'
_COUNTER_AGGREGATIONS = {
    Aggregation.SUM: dashboard_pb2.COUNTER_AGGREGATION_SUM,
    Aggregation.AVERAGE: dashboard_pb2.COUNTER_AGGREGATION_AVERAGE,
    Aggregation.MAX: dashboard_pb2.COUNTER_AGGREGATION_MAX,
    Aggregation.MIN: dashboard_pb2.COUNTER_AGGREGATION_MIN,
}
_NOT_BUILT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><base href="/"><title>Zephyr Dashboard</title></head>
<body><h1>Zephyr Dashboard</h1><p>The dashboard asset is not available.</p></body></html>
"""


class CoordinatorDashboardData(Protocol):
    """Internal data boundary between the Connect service and coordinator."""

    def _dashboard_pipelines_response(self) -> dashboard_pb2.ListPipelinesResponse: ...
    def _dashboard_plan_response(self, execution_id: str) -> dashboard_pb2.GetPlanResponse: ...
    def _dashboard_status_response(self, execution_id: str) -> dashboard_pb2.GetStatusResponse: ...
    def _dashboard_metrics_response(self, execution_id: str, max_points: int) -> dashboard_pb2.GetMetricsResponse: ...
    def _dashboard_counters_response(
        self, request: dashboard_pb2.ListCountersRequest
    ) -> dashboard_pb2.ListCountersResponse: ...
    def _dashboard_workers_response(
        self, request: dashboard_pb2.ListWorkersRequest
    ) -> dashboard_pb2.ListWorkersResponse: ...


class CoordinatorDashboardService:
    """Read-only Connect service over a coordinator state snapshot."""

    def __init__(self, coordinator: CoordinatorDashboardData) -> None:
        self._coordinator = coordinator

    async def list_pipelines(
        self, _request: dashboard_pb2.ListPipelinesRequest, _ctx: RequestContext
    ) -> dashboard_pb2.ListPipelinesResponse:
        return await run_in_threadpool(self._coordinator._dashboard_pipelines_response)

    async def get_plan(
        self, request: dashboard_pb2.GetPlanRequest, _ctx: RequestContext
    ) -> dashboard_pb2.GetPlanResponse:
        return await run_in_threadpool(self._coordinator._dashboard_plan_response, request.execution_id)

    async def get_status(
        self, request: dashboard_pb2.GetStatusRequest, _ctx: RequestContext
    ) -> dashboard_pb2.GetStatusResponse:
        return await run_in_threadpool(self._coordinator._dashboard_status_response, request.execution_id)

    async def get_metrics(
        self, request: dashboard_pb2.GetMetricsRequest, _ctx: RequestContext
    ) -> dashboard_pb2.GetMetricsResponse:
        max_points = bounded_limit(request.max_points, DEFAULT_METRIC_POINTS, MAX_METRIC_POINTS)
        return await run_in_threadpool(
            self._coordinator._dashboard_metrics_response,
            request.execution_id,
            max_points,
        )

    async def list_counters(
        self, request: dashboard_pb2.ListCountersRequest, _ctx: RequestContext
    ) -> dashboard_pb2.ListCountersResponse:
        return await run_in_threadpool(self._coordinator._dashboard_counters_response, request)

    async def list_workers(
        self, request: dashboard_pb2.ListWorkersRequest, _ctx: RequestContext
    ) -> dashboard_pb2.ListWorkersResponse:
        return await run_in_threadpool(self._coordinator._dashboard_workers_response, request)


def bounded_limit(value: int, default: int, maximum: int) -> int:
    """Return a positive page limit that does not exceed the service maximum."""
    if value <= 0:
        return default
    return min(value, maximum)


def counter_value(name: str, entry: CounterEntry) -> dashboard_pb2.CounterValue:
    value = dashboard_pb2.CounterValue(
        name=name,
        aggregation=_COUNTER_AGGREGATIONS[entry.aggregation],
        stage=entry.stage or "",
        observations=entry.count,
    )
    if isinstance(entry.value, int):
        value.int_value = entry.value
    else:
        value.double_value = entry.value
    return value


def source_node_id(prefix: str) -> str:
    return f"{prefix}/source"


def stage_node_id(prefix: str, stage_index: int) -> str:
    return f"{prefix}/stage/{stage_index}"


def join_right_prefix(parent_node_id: str, operation_index: int) -> str:
    return f"{parent_node_id}/join/{operation_index}/right"


def pipeline_plan(
    plan: PhysicalPlan,
    *,
    pipeline_name: str,
    pipeline_id: int,
    execution_id: str,
) -> dashboard_pb2.GetPlanResponse:
    """Build a safe graph that contains no source values or callable representations."""
    response = dashboard_pb2.GetPlanResponse(
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        execution_id=execution_id,
        source_item_count=len(plan.source_items),
        source_shard_count=plan.num_shards,
    )

    def add_plan(
        nested_plan: PhysicalPlan,
        *,
        prefix: str,
        parent_node_id: str = "",
        auxiliary: bool = False,
        target_node_id: str = "",
    ) -> None:
        plan_source_node_id = source_node_id(prefix)
        response.nodes.append(
            dashboard_pb2.PlanNode(
                node_id=plan_source_node_id,
                label=f"Source ({nested_plan.num_shards} shards)",
                stage_type="SOURCE",
                output_shards=nested_plan.num_shards,
                stage_index=-1,
                parent_node_id=parent_node_id,
                auxiliary=auxiliary,
            )
        )
        previous_node_id = plan_source_node_id
        current_shards = nested_plan.num_shards
        for stage_index, stage in enumerate(nested_plan.stages):
            node_id = stage_node_id(prefix, stage_index)
            output_shards = stage.output_shards or current_shards
            response.nodes.append(
                dashboard_pb2.PlanNode(
                    node_id=node_id,
                    label=stage.stage_name(),
                    stage_type=stage.stage_type.value.upper(),
                    operation_types=[type(operation).__name__ for operation in stage.operations],
                    output_shards=output_shards,
                    stage_index=stage_index,
                    parent_node_id=parent_node_id,
                    auxiliary=auxiliary,
                )
            )
            response.edges.append(
                dashboard_pb2.PlanEdge(source_node_id=previous_node_id, target_node_id=node_id)
            )
            previous_node_id = node_id
            current_shards = output_shards

            for operation_index, operation in enumerate(stage.operations):
                if not isinstance(operation, Join) or operation.right_plan is None:
                    continue
                add_plan(
                    operation.right_plan,
                    prefix=join_right_prefix(node_id, operation_index),
                    parent_node_id=node_id,
                    auxiliary=True,
                    target_node_id=node_id,
                )

        if target_node_id:
            response.edges.append(
                dashboard_pb2.PlanEdge(
                    source_node_id=previous_node_id,
                    target_node_id=target_node_id,
                    label="join input",
                )
            )

    add_plan(plan, prefix="main")
    return response


def _dashboard_html() -> str:
    resource = importlib.resources.files("zephyr").joinpath("dashboard.html")
    try:
        dashboard_html = resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _NOT_BUILT_HTML
    if _BASE_ELEMENT not in dashboard_html:
        raise RuntimeError("The Zephyr dashboard HTML does not contain the proxy base element")
    return dashboard_html


def _html_with_base(raw_html: str, forwarded_prefix: str) -> str:
    prefix = forwarded_prefix.rstrip("/")
    if not prefix:
        return raw_html
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    safe_prefix = html.escape(prefix, quote=True)
    return raw_html.replace(_BASE_ELEMENT, f'<base href="{safe_prefix}/"', 1)


def create_dashboard_application(coordinator: CoordinatorDashboardData) -> ASGIApp:
    """Create the coordinator dashboard app with Connect routes before SPA routes."""
    service_app = CoordinatorDashboardServiceASGIApplication(CoordinatorDashboardService(coordinator))
    raw_html = _dashboard_html()

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(_html_with_base(raw_html, request.headers.get("x-forwarded-prefix", "")))

    return Starlette(
        routes=[
            Mount(service_app.path, app=service_app),
            Route("/", index),
            Route("/{path:path}", index),
        ]
    )


def service_path(method: str) -> str:
    """Return the relative Connect path used by the browser client."""
    return f"{_DASHBOARD_SERVICE_PATH}/{method}"
