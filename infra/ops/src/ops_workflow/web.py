# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Starlette API and static dashboard host for the ops workflow."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from rigging.log_setup import LogBuffer
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles

from ops_workflow.grafana_source import GrafanaAlertSource
from ops_workflow.repository import ArchiveResult, OpsRepository, TurnPendingError
from ops_workflow.service import OpsService
from ops_workflow.slack import SlackDispatcher

MAX_QUESTION_BYTES = 16 * 1024
DIAGNOSTIC_LOG_LIMIT = 500
DIAGNOSTIC_POLL_LIMIT = 60
DIAGNOSTIC_ESCALATION_LIMIT = 60


@dataclass(frozen=True)
class WebConfig:
    auth_mode: str
    static_dir: Path | None
    reconcile_interval: float = 0.5
    poll_interval: float = 60.0


def create_app(
    service: OpsService,
    repository: OpsRepository,
    grafana_source: GrafanaAlertSource,
    log_buffer: LogBuffer,
    slack_dispatcher: SlackDispatcher | None,
    config: WebConfig,
) -> Starlette:
    """Create the API, coordinator loop, and optional built Vue host."""

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        coordinator = asyncio.create_task(
            _supervisor_loop(service.reconcile, config.reconcile_interval, "ops coordinator iteration failed")
        )
        poller = asyncio.create_task(
            _supervisor_loop(
                partial(_poll_grafana, repository, grafana_source),
                config.poll_interval,
                "Grafana polling iteration failed",
            )
        )
        tasks = [coordinator, poller]
        if slack_dispatcher is not None:
            tasks.append(
                asyncio.create_task(
                    _supervisor_loop(
                        slack_dispatcher.reconcile,
                        config.reconcile_interval,
                        "Slack escalation iteration failed",
                    )
                )
            )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await service.gateway.close()
            await repository.close()
            if slack_dispatcher is not None:
                await slack_dispatcher.close()

    async def health(_: Request) -> Response:
        return _json({"ok": True})

    async def overview(request: Request) -> Response:
        _actor(request, config)
        return _json(await repository.overview())

    async def cases(request: Request) -> Response:
        _actor(request, config)
        include_archived = request.query_params.get("archived") == "true"
        return _json({"cases": await repository.list_cases(include_archived=include_archived)})

    async def diagnostics(request: Request) -> Response:
        _actor(request, config)
        logs = [asdict(record) for record in reversed(log_buffer.query(limit=DIAGNOSTIC_LOG_LIMIT))]
        return _json(
            {
                "buffer_scope": "process",
                "resets_on_restart": True,
                "polls": await repository.recent_grafana_polls(limit=DIAGNOSTIC_POLL_LIMIT),
                "escalations": await repository.recent_slack_escalations(limit=DIAGNOSTIC_ESCALATION_LIMIT),
                "logs": logs,
            }
        )

    async def case_detail(request: Request) -> Response:
        _actor(request, config)
        detail = await service.case_with_chat(request.path_params["case_id"])
        if detail is None:
            return _json({"error": "not_found"}, status_code=404)
        return _json(detail)

    async def question(request: Request) -> Response:
        actor = _actor(request, config)
        body = await _object_body(request)
        text = _question_text(body)
        case_id = await repository.create_question(text=text, actor=actor)
        return _json({"case_id": case_id}, status_code=202)

    async def follow_up(request: Request) -> Response:
        actor = _actor(request, config)
        body = await _object_body(request)
        text = _question_text(body)
        try:
            turn_id = await repository.enqueue_follow_up(
                case_id=request.path_params["case_id"],
                text=text,
                actor=actor,
            )
        except KeyError:
            return _json({"error": "not_found"}, status_code=404)
        except TurnPendingError as error:
            return _json({"error": "turn_pending", "turn_id": str(error)}, status_code=409)
        return _json({"turn_id": turn_id}, status_code=202)

    async def archive(request: Request) -> Response:
        actor = _actor(request, config)
        try:
            result = await service.archive(case_id=request.path_params["case_id"], actor=actor)
        except KeyError:
            return _json({"error": "not_found"}, status_code=404)
        if result == ArchiveResult.ACTIVE_TURN:
            return _json({"error": "turn_active"}, status_code=409)
        return _json({"archived": True, "already_archived": result == ArchiveResult.ALREADY_ARCHIVED})

    async def spa(_: Request) -> Response:
        if config.static_dir is None:
            return _json({"service": "marin-ops", "dashboard": "not_built"})
        index = config.static_dir / "index.html"
        if not index.exists():
            return _json({"error": "dashboard_not_built"}, status_code=503)
        return FileResponse(index)

    routes: list[BaseRoute] = [Route("/healthz", health)]
    routes.extend(
        (
            Route("/api/overview", overview),
            Route("/api/diagnostics", diagnostics),
            Route("/api/cases", cases),
            Route("/api/cases/{case_id}", case_detail),
            Route("/api/cases/{case_id}/messages", follow_up, methods=["POST"]),
            Route("/api/cases/{case_id}/archive", archive, methods=["POST"]),
            Route("/api/questions", question, methods=["POST"]),
        )
    )
    if config.static_dir is not None and (config.static_dir / "static").exists():
        routes.append(Mount("/static", StaticFiles(directory=config.static_dir / "static"), name="static"))
    routes.extend((Route("/", spa), Route("/{path:path}", spa)))
    return Starlette(routes=routes, lifespan=lifespan)


async def _supervisor_loop(step: Callable[[], Awaitable[None]], interval: float, failure_message: str) -> None:
    while True:
        try:
            await step()
        except Exception:
            # Background steps persist item-level failures themselves. This guard keeps a
            # transient dependency outage from terminating the process-level supervisor.
            logging.getLogger(__name__).exception(failure_message)
        await asyncio.sleep(interval)


async def _poll_grafana(repository: OpsRepository, source: GrafanaAlertSource) -> None:
    snapshot = await source.snapshot()
    results = await repository.reconcile_grafana_snapshot(snapshot)
    queued = sum(len(result.queued_case_ids) for result in results)
    logging.getLogger(__name__).info(
        "reconciled Grafana snapshot: alerts=%d groups=%d queued=%d",
        len(snapshot.alerts),
        len(results),
        queued,
    )


def _actor(request: Request, config: WebConfig) -> str:
    if config.auth_mode == "local":
        return "local-operator"
    if config.auth_mode != "iap":
        raise RuntimeError(f"unsupported auth mode {config.auth_mode}")
    principal = request.headers.get("x-goog-authenticated-user-email", "")
    prefix = "accounts.google.com:"
    if not principal.startswith(prefix) or len(principal) == len(prefix):
        raise HTTPException(401, "IAP identity required")
    return principal.removeprefix(prefix).lower()


async def _object_body(request: Request) -> Mapping[str, object]:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "request body must be JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be an object")
    return body


def _question_text(body: Mapping[str, object]) -> str:
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "text must be a non-empty string")
    text = text.strip()
    if len(text.encode()) > MAX_QUESTION_BYTES:
        raise HTTPException(413, f"text exceeds {MAX_QUESTION_BYTES} bytes")
    return text


def _json(value: object, *, status_code: int = 200) -> JSONResponse:
    serializable = json.loads(json.dumps(value, default=str))
    return JSONResponse(serializable, status_code=status_code)
