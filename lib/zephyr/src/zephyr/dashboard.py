# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only JSON API and standalone HTML dashboard."""

import enum
import html
import importlib.resources
from dataclasses import dataclass, field
from typing import Protocol

import msgspec
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp

from zephyr.plan import Join, PhysicalPlan
from zephyr.stats import PipelineMetricPoint
from zephyr.worker_context import Aggregation, CounterEntry

DEFAULT_COUNTER_LIMIT = 100
MAX_COUNTER_LIMIT = 500
DEFAULT_WORKER_LIMIT = 50
MAX_WORKER_LIMIT = 200
DEFAULT_METRIC_POINTS = 200
MAX_METRIC_POINTS = 500
SOURCE_STAGE_TYPE = "SOURCE"

_BASE_ELEMENT = '<base href="/"'


class PipelinePhase(enum.StrEnum):
    """Coarse pipeline state that the dashboard header shows."""

    UNKNOWN = "unknown"
    WAITING_FOR_WORKERS = "waiting_for_workers"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PlanNodeState(enum.StrEnum):
    """State of one plan node in the selected execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineSummary:
    execution_id: str
    pipeline_name: str
    current_stage: str


@dataclass(frozen=True)
class PipelineList:
    pipelines: tuple[PipelineSummary, ...]


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    label: str
    stage_type: str
    output_shards: int
    stage_index: int
    parent_node_id: str
    auxiliary: bool
    operation_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePlan:
    pipeline_name: str
    execution_id: str
    source_item_count: int = 0
    nodes: tuple[PlanNode, ...] = ()


@dataclass(frozen=True)
class PlanNodeStatus:
    node_id: str
    state: PlanNodeState


@dataclass(frozen=True)
class WorkerStateCount:
    state: str
    count: int


@dataclass(frozen=True)
class ResourceUsage:
    cpu_cores: float = 0.0
    cpu_utilization: float = 0.0
    memory_bytes: int = 0
    memory_utilization: float = 0.0


@dataclass(frozen=True)
class PipelineStatus:
    execution_id: str
    phase: PipelinePhase = PipelinePhase.UNKNOWN
    current_stage: str = ""
    completed_shards: int = 0
    total_shards: int = 0
    in_flight_shards: int = 0
    queued_shards: int = 0
    retries: int = 0
    started_at_ms: int = 0
    finished_at_ms: int = 0
    fatal_error: str = ""
    coordinator_task_id: str = ""
    expected_workers: int = 0
    worker_states: tuple[WorkerStateCount, ...] = ()
    resources: ResourceUsage = field(default_factory=ResourceUsage)
    node_statuses: tuple[PlanNodeStatus, ...] = ()


@dataclass(frozen=True)
class PipelineMetrics:
    points: tuple[PipelineMetricPoint, ...] = ()
    warning: str = ""


@dataclass(frozen=True)
class CounterValue:
    name: str
    value: float
    aggregation: Aggregation
    stage: str
    observations: int


@dataclass(frozen=True)
class CounterPage:
    counters: tuple[CounterValue, ...]
    total: int
    stages: tuple[str, ...]


@dataclass(frozen=True)
class WorkerAssignment:
    execution_id: str
    shard: int


@dataclass(frozen=True)
class WorkerStatus:
    worker_id: str
    task_id: str
    state: str
    last_seen_age_seconds: float
    assignments: tuple[WorkerAssignment, ...]
    cpu_percent: float
    memory_bytes: int


@dataclass(frozen=True)
class WorkerPage:
    workers: tuple[WorkerStatus, ...]
    total: int


@dataclass(frozen=True)
class CounterQuery:
    """Normalized ``/api/counters`` query parameters."""

    execution_id: str
    stage: str
    search: str
    sort_field: str
    sort_descending: bool
    offset: int
    limit: int


@dataclass(frozen=True)
class WorkerQuery:
    """Normalized ``/api/workers`` query parameters."""

    search: str
    sort_field: str
    sort_descending: bool
    offset: int
    limit: int


class DashboardData(Protocol):
    """Queries used by the dashboard HTTP routes."""

    def pipelines(self) -> PipelineList: ...
    def plan(self, execution_id: str) -> PipelinePlan: ...
    def status(self, execution_id: str) -> PipelineStatus: ...
    def metrics(self, execution_id: str, max_points: int) -> PipelineMetrics: ...
    def counters(self, query: CounterQuery) -> CounterPage: ...
    def workers(self, query: WorkerQuery) -> WorkerPage: ...


def bounded_limit(value: int, default: int, maximum: int) -> int:
    """Return a positive page limit that does not exceed the service maximum."""
    if value <= 0:
        return default
    return min(value, maximum)


def counter_value(name: str, entry: CounterEntry) -> CounterValue:
    return CounterValue(
        name=name,
        value=entry.value,
        aggregation=entry.aggregation,
        stage=entry.stage or "",
        observations=entry.count,
    )


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
    execution_id: str,
) -> PipelinePlan:
    """Build a safe graph that contains no source values or callable representations."""
    nodes: list[PlanNode] = []

    def add_plan(
        nested_plan: PhysicalPlan,
        *,
        prefix: str,
        parent_node_id: str = "",
        auxiliary: bool = False,
    ) -> None:
        nodes.append(
            PlanNode(
                node_id=source_node_id(prefix),
                label=f"Source ({nested_plan.num_shards} shards)",
                stage_type=SOURCE_STAGE_TYPE,
                output_shards=nested_plan.num_shards,
                stage_index=-1,
                parent_node_id=parent_node_id,
                auxiliary=auxiliary,
            )
        )
        current_shards = nested_plan.num_shards
        for stage_index, stage in enumerate(nested_plan.stages):
            node_id = stage_node_id(prefix, stage_index)
            output_shards = stage.output_shards or current_shards
            nodes.append(
                PlanNode(
                    node_id=node_id,
                    label=stage.stage_name(),
                    stage_type=stage.stage_type.value.upper(),
                    operation_types=tuple(type(operation).__name__ for operation in stage.operations),
                    output_shards=output_shards,
                    stage_index=stage_index,
                    parent_node_id=parent_node_id,
                    auxiliary=auxiliary,
                )
            )
            current_shards = output_shards

            for operation_index, operation in enumerate(stage.operations):
                if not isinstance(operation, Join) or operation.right_plan is None:
                    continue
                add_plan(
                    operation.right_plan,
                    prefix=join_right_prefix(node_id, operation_index),
                    parent_node_id=node_id,
                    auxiliary=True,
                )

    add_plan(plan, prefix="main")
    return PipelinePlan(
        pipeline_name=pipeline_name,
        execution_id=execution_id,
        source_item_count=len(plan.source_items),
        nodes=tuple(nodes),
    )


def _dashboard_html() -> str:
    resource = importlib.resources.files("zephyr").joinpath("dashboard.html")
    dashboard_html = resource.read_text(encoding="utf-8")
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


def _json_response(payload: object) -> Response:
    return Response(msgspec.json.encode(payload), media_type="application/json")


def _integer_parameter(request: Request, name: str) -> int:
    """Return an integer query parameter, or 0 when the browser omits it."""
    raw = request.query_params.get(name, "")
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"Query parameter {name!r} must be an integer") from error


def create_dashboard_application(data: DashboardData) -> ASGIApp:
    """Serve the dashboard and its read-only JSON API."""
    raw_html = _dashboard_html()

    async def pipelines(_request: Request) -> Response:
        return _json_response(await run_in_threadpool(data.pipelines))

    async def plan(request: Request) -> Response:
        execution_id = request.query_params.get("execution_id", "")
        return _json_response(await run_in_threadpool(data.plan, execution_id))

    async def status(request: Request) -> Response:
        execution_id = request.query_params.get("execution_id", "")
        return _json_response(await run_in_threadpool(data.status, execution_id))

    async def metrics(request: Request) -> Response:
        execution_id = request.query_params.get("execution_id", "")
        max_points = bounded_limit(_integer_parameter(request, "max_points"), DEFAULT_METRIC_POINTS, MAX_METRIC_POINTS)
        return _json_response(await run_in_threadpool(data.metrics, execution_id, max_points))

    async def counters(request: Request) -> Response:
        query = CounterQuery(
            execution_id=request.query_params.get("execution_id", ""),
            stage=request.query_params.get("stage", ""),
            search=request.query_params.get("search", ""),
            sort_field=request.query_params.get("sort_field", "name"),
            sort_descending=request.query_params.get("sort_descending") == "true",
            offset=max(_integer_parameter(request, "offset"), 0),
            limit=bounded_limit(_integer_parameter(request, "limit"), DEFAULT_COUNTER_LIMIT, MAX_COUNTER_LIMIT),
        )
        return _json_response(await run_in_threadpool(data.counters, query))

    async def workers(request: Request) -> Response:
        query = WorkerQuery(
            search=request.query_params.get("search", ""),
            sort_field=request.query_params.get("sort_field", "worker_id"),
            sort_descending=request.query_params.get("sort_descending") == "true",
            offset=max(_integer_parameter(request, "offset"), 0),
            limit=bounded_limit(_integer_parameter(request, "limit"), DEFAULT_WORKER_LIMIT, MAX_WORKER_LIMIT),
        )
        return _json_response(await run_in_threadpool(data.workers, query))

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(_html_with_base(raw_html, request.headers.get("x-forwarded-prefix", "")))

    return Starlette(
        routes=[
            Route("/api/pipelines", pipelines),
            Route("/api/plan", plan),
            Route("/api/status", status),
            Route("/api/metrics", metrics),
            Route("/api/counters", counters),
            Route("/api/workers", workers),
            Route("/", index),
        ]
    )
