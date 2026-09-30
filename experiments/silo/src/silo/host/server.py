# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface of a host agent.

Two audiences, two credentials (see ``silo.auth``):

  broker  -> host   create/list/capacity/images      Bearer SILO_HOST_SECRET
  worker  -> host   exec, sessions, files, get/delete X-Silo-Sandbox-Token

Handlers are sync on purpose: every operation shells out to nerdctl, and
Starlette runs sync handlers in a threadpool. The pool is widened at startup,
since one host serves tens of sandboxes each with a poll loop running.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from silo.auth import HEADER_API_TOKEN, HEADER_SANDBOX_TOKEN, verify_bearer, verify_sandbox_capability
from silo.errors import SiloError
from silo.host.agent import HostAgent
from silo.model import ImagePlan, ResourceProfile
from silo.wire import error_body

logger = logging.getLogger(__name__)


def _profile(body: dict[str, Any]) -> ResourceProfile:
    return ResourceProfile(cpu=int(body["cpu"]), memory_gb=int(body["memory_gb"]), disk_gb=int(body["disk_gb"]))


def _plan(body: dict[str, Any]) -> ImagePlan:
    return ImagePlan(
        base_ref=body["base_ref"],
        entrypoint=tuple(body["entrypoint"]) if body.get("entrypoint") else None,
        cmd=tuple(body["cmd"]) if body.get("cmd") else None,
        user=body.get("user"),
        workdir=body.get("workdir"),
        env=body.get("env") or {},
        run_steps=tuple(body.get("run_steps") or ()),
    )


def _command_view(entry: Any) -> dict[str, Any]:
    return {"cmd_id": entry.cmd_id, "command": entry.command, "exit_code": entry.exit_code}


def build_app(agent: HostAgent, host_secret: str) -> Starlette:
    def guard(kind: str, body: str | None = None):
        """Authenticate, read the body, run the sync handler on the threadpool,
        and map SiloError to its HTTP status with its message intact."""

        def decorate(handler: Callable[..., Response]):
            async def wrapped(request: Request) -> Response:
                is_broker = verify_bearer(host_secret, request.headers.get(HEADER_API_TOKEN))
                if kind == "broker" and not is_broker:
                    return JSONResponse({"error": "Unauthorized", "message": "broker credential required"}, 401)
                if kind == "sandbox" and not is_broker:
                    sandbox_id = request.path_params.get("sandbox_id", "")
                    if not verify_sandbox_capability(host_secret, sandbox_id, request.headers.get(HEADER_SANDBOX_TOKEN)):
                        return JSONResponse({"error": "Unauthorized", "message": "sandbox capability required"}, 401)
                payload: Any = None
                if body == "json":
                    payload = await request.json()
                elif body == "raw":
                    payload = await request.body()
                try:
                    return await run_in_threadpool(handler, request, payload)
                except SiloError as error:
                    return JSONResponse(error_body(error), status_code=error.status_code or 500)
                except Exception as error:
                    logger.exception("unhandled error in %s", handler.__name__)
                    return JSONResponse(
                        {"error": type(error).__name__, "message": str(error)[:2000], "status_code": 500}, 500
                    )

            wrapped.__name__ = handler.__name__
            return wrapped

        return decorate

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "host_id": agent.config.host_id})

    @guard("broker")
    def capacity(_: Request, __: Any) -> Response:
        # Sandbox ids let the broker tell which of its placements this report
        # already counts (see HostState.apply_report).
        return JSONResponse({**agent.capacity(), "sandbox_ids": [r.id for r in agent.list_sandboxes()]})

    @guard("broker", body="json")
    def create_sandbox(request: Request, body: dict[str, Any]) -> Response:
        record = agent.create_sandbox(
            sandbox_id=body["sandbox_id"],
            snapshot_name=body["snapshot_name"],
            image_ref=body["image_ref"],
            plan=_plan(body["plan"]),
            profile=_profile(body["profile"]),
            runtime=body.get("runtime"),
            labels=body.get("labels") or {},
            ttl_minutes=int(body.get("ttl_minutes", 180)),
        )
        return JSONResponse(record.to_dict(), status_code=201)

    @guard("broker")
    def list_sandboxes(_: Request, __: Any) -> Response:
        return JSONResponse({"items": [r.to_dict() for r in agent.list_sandboxes()]})

    @guard("sandbox")
    def get_sandbox(request: Request, _: Any) -> Response:
        return JSONResponse(agent.get_sandbox(request.path_params["sandbox_id"]).to_dict())

    @guard("sandbox")
    def delete_sandbox(request: Request, _: Any) -> Response:
        agent.delete_sandbox(request.path_params["sandbox_id"])
        return JSONResponse({"deleting": True}, status_code=202)

    @guard("sandbox", body="json")
    def exec_(request: Request, body: dict[str, Any]) -> Response:
        outcome = agent.exec(
            request.path_params["sandbox_id"],
            body["command"],
            cwd=body.get("cwd"),
            env=body.get("env"),
            timeout=body.get("timeout"),
        )
        return JSONResponse(
            {
                "exit_code": outcome.exit_code,
                "result": outcome.output.decode(errors="replace"),
                "timed_out": outcome.timed_out,
            }
        )

    @guard("sandbox", body="json")
    def create_session(request: Request, body: dict[str, Any]) -> Response:
        agent.create_session(request.path_params["sandbox_id"], body["session_id"])
        return JSONResponse({"session_id": body["session_id"]}, status_code=201)

    @guard("sandbox")
    def delete_session(request: Request, _: Any) -> Response:
        agent.delete_session(request.path_params["sandbox_id"], request.path_params["session_id"])
        return Response(status_code=204)

    @guard("sandbox", body="json")
    def execute_session_command(request: Request, body: dict[str, Any]) -> Response:
        entry = agent.execute_session_command(
            request.path_params["sandbox_id"],
            request.path_params["session_id"],
            body["command"],
            run_async=bool(body.get("run_async", True)),
        )
        return JSONResponse(_command_view(entry))

    @guard("sandbox")
    def get_session_command(request: Request, _: Any) -> Response:
        entry = agent.get_session_command(
            request.path_params["sandbox_id"], request.path_params["session_id"], request.path_params["cmd_id"]
        )
        return JSONResponse(_command_view(entry))

    @guard("sandbox")
    def get_session_command_logs(request: Request, _: Any) -> Response:
        stdout, stderr = agent.get_session_command_logs(
            request.path_params["sandbox_id"], request.path_params["session_id"], request.path_params["cmd_id"]
        )
        return JSONResponse({"stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")})

    @guard("sandbox", body="raw")
    def put_file(request: Request, data: bytes) -> Response:
        path = request.query_params["path"]
        agent.upload_file(request.path_params["sandbox_id"], path, data)
        return Response(status_code=204)

    @guard("sandbox")
    def get_file(request: Request, _: Any) -> Response:
        path = request.query_params["path"]
        data = agent.download_file(request.path_params["sandbox_id"], path)
        return Response(data, media_type="application/octet-stream")

    @guard("broker", body="json")
    def ensure_image(request: Request, body: dict[str, Any]) -> Response:
        agent.ensure_image(body["ref"])
        return JSONResponse({"ref": body["ref"], "present": True})

    @guard("broker", body="json")
    def build_image(request: Request, body: dict[str, Any]) -> Response:
        agent.build_image(body["tag"], body["dockerfile"])
        return JSONResponse({"tag": body["tag"], "built": True})

    sb = "/sandboxes/{sandbox_id}"
    ss = sb + "/sessions/{session_id}"
    routes = [
        Route("/health", health),
        Route("/capacity", capacity),
        Route("/sandboxes", create_sandbox, methods=["POST"]),
        Route("/sandboxes", list_sandboxes, methods=["GET"]),
        Route(sb, get_sandbox, methods=["GET"]),
        Route(sb, delete_sandbox, methods=["DELETE"]),
        Route(sb + "/exec", exec_, methods=["POST"]),
        Route(sb + "/sessions", create_session, methods=["POST"]),
        Route(ss, delete_session, methods=["DELETE"]),
        Route(ss + "/exec", execute_session_command, methods=["POST"]),
        Route(ss + "/commands/{cmd_id}", get_session_command, methods=["GET"]),
        Route(ss + "/commands/{cmd_id}/logs", get_session_command_logs, methods=["GET"]),
        Route(sb + "/files", put_file, methods=["PUT"]),
        Route(sb + "/files", get_file, methods=["GET"]),
        Route("/images/ensure", ensure_image, methods=["POST"]),
        Route("/images/build", build_image, methods=["POST"]),
    ]
    return Starlette(routes=routes)
