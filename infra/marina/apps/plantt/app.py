# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Database API for the checked-in Plantt application."""

import json
import uuid
from datetime import UTC, date, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from marina.apps import RegisteredApi, Services, registered_api
from marina.mcp import OperationRisk, operation_extension
from pydantic import BaseModel, ConfigDict
from rigging.server_auth import get_verified_user
from sqlalchemy import JSON, BigInteger, Column, DateTime, MetaData, String, Table, Uuid, delete, select, update
from sqlalchemy.engine import Engine, RowMapping

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024

metadata = MetaData()
charts = Table(
    "charts",
    metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column("title", String(300), nullable=False),
    Column("document", JSON, nullable=False),
    Column("revision", BigInteger, nullable=False),
    Column("created_by", String, nullable=False),
    Column("updated_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
CHART_SUMMARY_COLUMNS = (
    charts.c.id,
    charts.c.title,
    charts.c.revision,
    charts.c.created_by,
    charts.c.updated_by,
    charts.c.created_at,
    charts.c.updated_at,
)


class ChartWrite(BaseModel):
    document: dict[str, Any]


class ChartUpdate(ChartWrite):
    revision: int


class ChartDelete(BaseModel):
    revision: int


class ChartSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    revision: int
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime


class ChartRecord(ChartSummary):
    document: dict[str, Any]


class ChartSchema(BaseModel):
    description: str
    fields: dict[str, str]
    task: dict[str, str]
    milestone: dict[str, str]


def migrate(engine: Engine) -> None:
    """Create the app-owned chart table."""
    metadata.create_all(engine)


def _validate_document(document: dict[str, Any]) -> str:
    encoded = json.dumps(document, separators=(",", ":")).encode()
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Chart document exceeds 2 MiB")

    title = document.get("title")
    if not isinstance(title, str) or not title.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Chart title must be a non-empty string")
    if len(title.strip()) > 300:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Chart title cannot exceed 300 characters")

    workstreams = document.get("workstreams")
    if not isinstance(workstreams, list):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Chart workstreams must be an array")

    names: set[str] = set()
    task_names: set[str] = set()
    items: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for workstream in workstreams:
        if not isinstance(workstream, dict) or not isinstance(workstream.get("name"), str):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Every workstream needs a name")
        workstream_tasks = workstream.get("tasks")
        if not isinstance(workstream_tasks, list):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Workstream '{workstream['name']}' must have a tasks array",
            )
        milestones = workstream.get("milestones", [])
        if not isinstance(milestones, list):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Workstream '{workstream['name']}' milestones must be an array",
            )
        for item in [*workstream_tasks, *milestones]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Every task and milestone needs a name")
            name = item["name"]
            if name in names:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"Duplicate task or milestone name: '{name}'",
                )
            names.add(name)
            items.append(item)
        for task in workstream_tasks:
            assert isinstance(task, dict)
            task_names.add(task["name"])
            tasks.append(task)
        for milestone in milestones:
            assert isinstance(milestone, dict)
            _validate_date(milestone.get("date"), f"Milestone '{milestone['name']}' date")

    for item in items:
        dependencies = item.get("deps", [])
        if not isinstance(dependencies, list) or not all(isinstance(name, str) for name in dependencies):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"'{item['name']}' dependencies must be an array of names",
            )
        unknown = next((name for name in dependencies if name not in names), None)
        if unknown is not None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"'{item['name']}' depends on unknown item '{unknown}'",
            )

    start_dependencies: dict[str, str | None] = {}
    for task in tasks:
        name = task["name"]
        start = task.get("start")
        dependency: str | None = None
        if isinstance(start, str):
            dependency = start
        elif isinstance(start, list) and len(start) == 2 and start[0] == "date":
            _validate_date(start[1], f"Task '{name}' start")
        elif isinstance(start, list) and len(start) == 3 and start[0] == "after":
            dependency = start[1] if isinstance(start[1], str) else None
            _validate_duration(start[2], f"Task '{name}' start lag", allow_date=False)
        else:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Task '{name}' has an invalid start")
        if dependency is not None and dependency not in task_names:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Task '{name}' starts from unknown task '{dependency}'",
            )
        _validate_duration(task.get("end"), f"Task '{name}' end", allow_date=True)
        start_dependencies[name] = dependency

    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Task scheduling cycle includes '{name}'",
            )
        visiting.add(name)
        dependency = start_dependencies[name]
        if dependency is not None:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in start_dependencies:
        visit(name)

    capacity = document.get("capacity", [])
    if not isinstance(capacity, list):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Chart capacity must be an array")
    for pool in capacity:
        if not isinstance(pool, dict) or not isinstance(pool.get("name"), str):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Every compute pool needs a name")
        _validate_date(pool.get("from"), f"Compute pool '{pool['name']}' start")
        if pool.get("to") is not None:
            _validate_date(pool["to"], f"Compute pool '{pool['name']}' end")
        growth = pool.get("grows", [])
        if not isinstance(growth, list):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Compute pool '{pool['name']}' growth must be an array",
            )
        for event in growth:
            if not isinstance(event, dict):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"Compute pool '{pool['name']}' has an invalid growth event",
                )
            _validate_date(event.get("date"), f"Compute pool '{pool['name']}' growth date")

    annotations = document.get("annotations", [])
    if not isinstance(annotations, list):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Chart annotations must be an array")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Every annotation must be an object")
        _validate_date(annotation.get("date"), "Annotation date")

    return title.strip()


def _validate_date(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{field} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{field} must be YYYY-MM-DD") from error


def _validate_duration(value: object, field: str, *, allow_date: bool) -> None:
    if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{field} must be a duration tuple")
    unit, amount = value
    if unit == "date" and allow_date:
        _validate_date(amount, field)
        return
    if unit not in {"days", "weeks", "months"} or isinstance(amount, bool) or not isinstance(amount, int | float):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{field} has an invalid duration")
    if amount <= 0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"{field} must be positive")


def _user_id() -> str:
    return get_verified_user() or "anonymous"


def _record(row: RowMapping) -> ChartRecord:
    return ChartRecord.model_validate(dict(row))


def _chart(engine: Engine, chart_id: uuid.UUID) -> ChartRecord:
    with engine.connect() as connection:
        row = connection.execute(select(charts).where(charts.c.id == chart_id)).mappings().one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chart not found")
    return _record(row)


def create_api(services: Services) -> RegisteredApi:
    api = FastAPI(title="Plantt API", docs_url=None, redoc_url=None, openapi_url=None)
    engine = services.engine()

    @api.get(
        "/schema",
        operation_id="describe_chart_schema",
        description="Describe the complete Plantt chart document accepted by create and update operations.",
        response_model=ChartSchema,
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def describe_chart_schema() -> ChartSchema:
        return ChartSchema(
            description="A project plan containing named workstreams, scheduled tasks, milestones, and compute pools.",
            fields={
                "title": "Required chart title.",
                "note": "Optional subtitle.",
                "workstreams": "Workstreams with name, optional note, tasks, and optional milestones.",
                "capacity": "Optional compute pools with name, chip, chips, from, optional to/grows, and color.",
                "annotations": "Optional dated labels with text, date, target, edge, and optional color.",
            },
            task={
                "name": "Globally unique task name.",
                "start": "['date', YYYY-MM-DD], another task name, or ['after', task name, duration].",
                "end": "['days'|'weeks'|'months', number] or ['date', YYYY-MM-DD].",
                "optional": "cluster, chips, significance, link, tooltip, and deps (item names).",
            },
            milestone={
                "name": "Globally unique milestone name.",
                "date": "YYYY-MM-DD.",
                "optional": "emoji, line color, tooltip, and deps (item names).",
            },
        )

    @api.get(
        "/charts",
        operation_id="list_charts",
        description="List shared Plantt charts in most-recently-edited order.",
        response_model=list[ChartSummary],
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def list_charts() -> list[ChartSummary]:
        with engine.connect() as connection:
            rows = connection.execute(select(*CHART_SUMMARY_COLUMNS).order_by(charts.c.updated_at.desc())).mappings()
            return [ChartSummary.model_validate(dict(row)) for row in rows]

    @api.post(
        "/charts",
        operation_id="create_chart",
        description="Create a shared Plantt chart from a complete chart document.",
        response_model=ChartRecord,
        status_code=status.HTTP_201_CREATED,
        openapi_extra=operation_extension(OperationRisk.WRITE),
    )
    def create_chart(body: ChartWrite) -> ChartRecord:
        title = _validate_document(body.document)
        now = datetime.now(UTC)
        chart_id = uuid.uuid4()
        user_id = _user_id()
        with engine.begin() as connection:
            connection.execute(
                charts.insert().values(
                    id=chart_id,
                    title=title,
                    document=body.document,
                    revision=1,
                    created_by=user_id,
                    updated_by=user_id,
                    created_at=now,
                    updated_at=now,
                )
            )
        return _chart(engine, chart_id)

    @api.get(
        "/charts/{chart_id}",
        operation_id="read_chart",
        description="Read one Plantt chart and its optimistic-lock revision.",
        response_model=ChartRecord,
        openapi_extra=operation_extension(OperationRisk.READ),
    )
    def get_chart(chart_id: uuid.UUID) -> ChartRecord:
        return _chart(engine, chart_id)

    @api.put(
        "/charts/{chart_id}",
        operation_id="update_chart",
        description="Replace a Plantt chart when its revision still matches.",
        response_model=ChartRecord,
        openapi_extra=operation_extension(OperationRisk.WRITE),
    )
    def update_chart(chart_id: uuid.UUID, body: ChartUpdate) -> ChartRecord:
        title = _validate_document(body.document)
        with engine.begin() as connection:
            result = connection.execute(
                update(charts)
                .where(charts.c.id == chart_id, charts.c.revision == body.revision)
                .values(
                    title=title,
                    document=body.document,
                    revision=body.revision + 1,
                    updated_by=_user_id(),
                    updated_at=datetime.now(UTC),
                )
            )
        if result.rowcount == 0:
            with engine.connect() as connection:
                exists = connection.execute(select(charts.c.id).where(charts.c.id == chart_id)).scalar_one_or_none()
            if exists is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Chart not found")
            raise HTTPException(status.HTTP_409_CONFLICT, "Chart changed since it was loaded")
        return _chart(engine, chart_id)

    @api.delete(
        "/charts/{chart_id}",
        operation_id="delete_chart",
        description="Delete a Plantt chart when its revision still matches.",
        status_code=status.HTTP_204_NO_CONTENT,
        openapi_extra=operation_extension(OperationRisk.DESTRUCTIVE),
    )
    def delete_chart(chart_id: uuid.UUID, body: ChartDelete) -> Response:
        with engine.begin() as connection:
            result = connection.execute(
                delete(charts).where(charts.c.id == chart_id, charts.c.revision == body.revision)
            )
        if result.rowcount == 0:
            with engine.connect() as connection:
                exists = connection.execute(select(charts.c.id).where(charts.c.id == chart_id)).scalar_one_or_none()
            if exists is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Chart not found")
            raise HTTPException(status.HTTP_409_CONFLICT, "Chart changed since it was loaded")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return registered_api(api)
