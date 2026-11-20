# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve pinned retrieval through the same private process relay used by judges."""

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

from rolloutengine.harness.browsecomp import retrieval_output
from rolloutengine.harness.shell import ToolCallResult, ToolContext

RETRIEVAL_RELAY_PORT = 8789
# Each base64 frame stays under the existing relay's 4 MiB response cap.
DOCUMENT_CHUNK_BYTES = 1024 * 1024
GUEST_CLIENT = """import base64, json, sys, urllib.request
request = json.load(sys.stdin)
endpoint, target = request.pop("endpoint"), request.pop("target")
def call(path, body):
    req = urllib.request.Request(
        endpoint + path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
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
    cleanup: AsyncExitStack

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        parameter = "query" if name == "search" else "docid"
        if (
            name not in {"search", "get_document"}
            or set(arguments) != {parameter}
            or not isinstance(arguments[parameter], str)
        ):
            return ToolCallResult(json.dumps({"error": "Invalid offline search tool or arguments"}))
        remote = f"/tmp/retrieval-{uuid.uuid4().hex}.json"
        payload = {
            "endpoint": self.endpoint,
            "target": remote,
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
                        timeout=300,
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
            client = await cleanup.enter_async_context(httpx.AsyncClient(timeout=300, trust_env=False))
            health = await client.get(self.endpoint + "/health")
            health.raise_for_status()
            if health.json()["state"] != context.requirement.initial_state:
                raise ValueError("Retrieval service state differs from the frozen task")

            async def relay(request: HttpRequest) -> HttpResponse:
                if request.method != "POST" or request.path not in {"/search", "/document"}:
                    return HttpResponse(404, b"{}")
                response = await client.post(self.endpoint + request.path, content=request.body)
                response.raise_for_status()
                return HttpResponse(response.status_code, response.content)

            group = await cleanup.enter_async_context(asyncio.TaskGroup())
            endpoint = await cleanup.enter_async_context(
                serve_http(
                    context.machine,
                    relay,
                    port=RETRIEVAL_RELAY_PORT,
                    task_group=group,
                )
            )
            return RetrievalServiceSession(context.machine, context.requirement, endpoint, cleanup)
        except BaseException:
            await cleanup.aclose()
            raise
