# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry a native script's judge request through the owning host HTTP client."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.machine import Machine
from verifyit.modes.grade_judge import JudgeConnection

SCRIPT_JUDGE_PORT = 8790
SCRIPT_JUDGE_BASE_URL = f"http://127.0.0.1:{SCRIPT_JUDGE_PORT}/v1"
SCRIPT_JUDGE_GUEST_KEY = "machine-local-judge-relay"
SCRIPT_JUDGE_PATHS = {"/v1/chat/completions", "/v1/responses"}


@asynccontextmanager
async def script_judge_service(machine: Machine, connection: JudgeConnection) -> AsyncIterator[None]:
    """Relay native request bytes without giving the verifier the upstream credential."""
    async with httpx.AsyncClient(
        timeout=120,
        trust_env=False,
        headers={"Authorization": "Bearer " + connection.api_key, "Content-Type": "application/json"},
    ) as client:

        async def forward(request: HttpRequest) -> HttpResponse:
            if request.method != "POST" or request.path not in SCRIPT_JUDGE_PATHS:
                return HttpResponse(404, b"{}")
            response = await client.post(
                connection.base_url.rstrip("/") + request.path.removeprefix("/v1"), content=request.body
            )
            return HttpResponse(response.status_code, response.content)

        async with asyncio.TaskGroup() as group:
            async with serve_http(machine, forward, port=SCRIPT_JUDGE_PORT, task_group=group):
                yield
