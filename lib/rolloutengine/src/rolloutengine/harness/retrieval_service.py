# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve pinned search results and complete documents to network-denied task actors."""

import asyncio
import json
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from pydantic import JsonValue
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.machine import Command, Machine
from taskcompendium.browsecomp import browsecomp_tools
from taskcompendium.models import FunctionDefinition, ProviderRequirement
from taskcompendium.openresearcher import openresearcher_state

from rolloutengine.harness.browsecomp import INVALID_ARGUMENTS, retrieval_output
from rolloutengine.harness.shell import ToolCallResult, ToolContext

RETRIEVAL_RELAY_PORT = 8789
RETRIEVAL_TIMEOUT = 300
GUEST_CLIENT = """import base64, json, sys, urllib.request
request = json.load(sys.stdin)
endpoint, target = request.pop("endpoint"), request.pop("target")
timeout = request.pop("timeout")
def call(path, body):
    req = urllib.request.Request(
        endpoint + path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)
if request["name"] == "search":
    result = call("/search", request)
    with open(target, "w") as output:
        json.dump(result, output, ensure_ascii=False)
else:
    offset = 0
    with open(target, "wb") as output:
        while True:
            chunk = call("/document", {**request, "offset":offset})
            content = base64.b64decode(chunk["data"], validate=True)
            if not content and not chunk["done"]:
                raise ValueError("Empty nonterminal retrieval chunk")
            output.write(content)
            offset += len(content)
            if chunk["done"]:
                break
"""


@dataclass
class RetrievalServiceSession:
    machine: Machine
    requirement: ProviderRequirement
    endpoint: str
    client: httpx.AsyncClient
    cleanup: AsyncExitStack

    async def _relay(self, request: HttpRequest) -> HttpResponse:
        if request.method != "POST" or request.path not in {"/search", "/document"}:
            return HttpResponse(404, b"{}")
        response = await self.client.post(self.endpoint + request.path, content=request.body)
        response.raise_for_status()
        return HttpResponse(response.status_code, response.content)

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        parameter = "query" if name == "search" else "docid"
        if (
            name not in {"search", "get_document"}
            or set(arguments) != {parameter}
            or not isinstance(arguments[parameter], str)
        ):
            return ToolCallResult(INVALID_ARGUMENTS)
        # The tool call owns pump failures, so they reach the engine's normal rollout error boundary.
        async with asyncio.TaskGroup() as group:
            async with serve_http(self.machine, self._relay, port=RETRIEVAL_RELAY_PORT, task_group=group) as endpoint:
                return await self._retrieve(name, arguments, endpoint)

    async def _retrieve(self, name: str, arguments: dict[str, JsonValue], endpoint: str) -> ToolCallResult:
        remote = f"/tmp/retrieval-{uuid.uuid4().hex}.json"
        payload = {
            "endpoint": endpoint,
            "target": remote,
            "timeout": RETRIEVAL_TIMEOUT,
            "name": name,
            "arguments": arguments,
            "state": self.requirement.initial_state,
        }
        try:
            retrieval_output(
                await self.machine.run(
                    Command(
                        ("python3", "-c", GUEST_CLIENT),
                        stdin=json.dumps(payload).encode(),
                        timeout=RETRIEVAL_TIMEOUT,
                    )
                )
            )
            with TemporaryDirectory(prefix="retrieval-result-") as temporary:
                local = Path(temporary) / "result.json"
                await self.machine.download(remote, local)
                result = json.loads(local.read_bytes())
            return ToolCallResult(json.dumps(result, ensure_ascii=False))
        finally:
            retrieval_output(await self.machine.run(Command(("rm", "-f", remote), timeout=30)))

    async def close(self) -> None:
        await self.cleanup.aclose()


@dataclass(frozen=True)
class OpenResearcherServiceProvider:
    endpoint: str | None

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        state = openresearcher_state(requirement)
        if requirement.network_access != "service_only":
            raise ValueError("Retrieval service requires a service-only task network contract")
        if self.endpoint is None:
            raise ValueError("OpenResearcher requires a run-owned retrieval service")
        return browsecomp_tools(state)

    async def open(self, context: ToolContext) -> RetrievalServiceSession:
        self.tools(context.requirement)
        if context.machine is None or self.endpoint is None:
            raise ValueError("Retrieval relay requires its thin actor machine and a service endpoint")
        cleanup = AsyncExitStack()
        try:
            client = await cleanup.enter_async_context(httpx.AsyncClient(timeout=RETRIEVAL_TIMEOUT, trust_env=False))
            health = await client.get(self.endpoint + "/health")
            health.raise_for_status()
            if health.json()["state"] != context.requirement.initial_state:
                raise ValueError("Retrieval service state differs from the frozen task")

            return RetrievalServiceSession(context.machine, context.requirement, self.endpoint, client, cleanup)
        except BaseException:
            await cleanup.aclose()
            raise
