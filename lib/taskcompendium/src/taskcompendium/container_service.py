# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one isolated JSON-lines tool service for the duration of a trial."""

import asyncio
import hashlib
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

MAX_PROTOCOL_BYTES = 16 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024
PROTOCOL_TIMEOUT = 30


class ContainerService(BaseModel):
    """An image prepared before launch; trials never pull images or mount host paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(pattern=r"^[^\s@]+@sha256:[0-9a-f]{64}$")
    command: tuple[str, ...] = Field(min_length=1)


IMAGE_LABELS = {
    "action_interface": "org.marin.taskcompendium.action-interface",
    "seed_sha256": "org.marin.taskcompendium.seed-sha256",
    "provider_revision": "org.marin.taskcompendium.provider-revision",
    "tools_sha256": "org.marin.taskcompendium.tools-sha256",
}


def validate_container_image(
    runtime: ContainerService, identity: dict[str, str], definitions: list[dict[str, Any]]
) -> None:
    """Check a previously staged image's immutable metadata without pulling it."""
    inspection = subprocess.run(
        ("docker", "image", "inspect", runtime.image),
        capture_output=True,
        text=True,
        check=True,
        timeout=PROTOCOL_TIMEOUT,
    )
    images = json.loads(inspection.stdout)
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Expected one staged provider image")
    labels = images[0]["Config"]["Labels"] or {}
    digest = hashlib.sha256(
        json.dumps(definitions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    expected = {**identity, "tools_sha256": digest}
    if any(labels.get(IMAGE_LABELS[key]) != value for key, value in expected.items()):
        raise ValueError("Provider image metadata differs from its declared identity or tool schemas")


class ProviderProtocolError(RuntimeError):
    """A service, transport, or protocol failure; the trial remains ungraded."""


class ContainerToolProvider:
    """Dispatch each request once to a persistent provider process in a container."""

    def __init__(
        self,
        *,
        runtime: ContainerService,
        action_interface: str,
        seed_sha256: str,
        provider_revision: str,
        tool_definitions: list[dict[str, Any]],
        trace_path: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.identity = {
            "action_interface": action_interface,
            "seed_sha256": seed_sha256,
            "provider_revision": provider_revision,
        }
        self.tool_definitions = tool_definitions
        self.trace_path = trace_path
        self.container_name = f"taskcompendium-provider-{uuid.uuid4().hex}"
        self.process: asyncio.subprocess.Process | None = None
        self.stderr_task: asyncio.Task[None] | None = None
        self.diagnostics = bytearray()
        self.lock = asyncio.Lock()
        self.sequence = 0
        self.call_ids: set[str] = set()
        self.failed = False

    def _record(self, event: dict[str, Any]) -> None:
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")

    async def _read_diagnostics(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(8192):
            remaining = MAX_DIAGNOSTIC_BYTES - len(self.diagnostics)
            self.diagnostics.extend(chunk[:remaining])

    async def start(self) -> None:
        """Start with no network, workspace mounts, capabilities, or writable root."""
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--name",
            self.container_name,
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--pids-limit",
            "128",
            "--memory",
            "1g",
            "--cpus",
            "1",
            "-i",
            self.runtime.image,
            *self.runtime.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=MAX_PROTOCOL_BYTES + 1,
        )
        self.stderr_task = asyncio.create_task(self._read_diagnostics())
        response = await self._request("initialize", self.identity)
        if not isinstance(response, dict) or any(response.get(key) != value for key, value in self.identity.items()):
            raise ProviderProtocolError("Container provider identity differs from the selected binding")
        if response.get("tools") != self.tool_definitions:
            raise ProviderProtocolError("Container provider schemas differ from the selected binding")

    async def _request(self, method: str, params: dict[str, Any]) -> JsonValue:
        async with self.lock:
            if self.failed:
                raise ProviderProtocolError("Provider transport already failed; requests cannot be retried")
            process = self.process
            if process is None or process.stdin is None or process.stdout is None:
                raise ProviderProtocolError("Provider has not started")
            self.sequence += 1
            request = {"id": str(self.sequence), "method": method, "params": params}
            encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            if len(encoded) > MAX_PROTOCOL_BYTES:
                raise ProviderProtocolError("Provider request exceeds the protocol byte limit")
            self._record({"request": request})
            try:
                async with asyncio.timeout(PROTOCOL_TIMEOUT):
                    process.stdin.write(encoded)
                    await process.stdin.drain()
                    line = await process.stdout.readline()
                if not line or not line.endswith(b"\n") or len(line) > MAX_PROTOCOL_BYTES:
                    raise ProviderProtocolError("Provider returned an absent or oversized protocol response")
                response = json.loads(line, parse_constant=lambda value: _reject_constant(value))
                if not isinstance(response, dict) or response.get("id") != request["id"]:
                    raise ProviderProtocolError("Provider response ID differs from its request")
                if set(response) == {"id", "error"}:
                    self._record({"response": response})
                    raise ProviderProtocolError(f"Provider operation failed: {response['error']}")
                if set(response) != {"id", "result"}:
                    raise ProviderProtocolError("Provider returned malformed protocol fields")
                self._record({"response": response})
                return response["result"]
            except (TimeoutError, ValueError, ConnectionError, ProviderProtocolError) as error:
                self.failed = True
                self._record({"failure": type(error).__name__, "message": str(error)})
                raise ProviderProtocolError(f"Provider {method} failed: {error}") from error

    async def native_tool_definitions(self) -> list[dict[str, Any]]:
        return self.tool_definitions

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        if call_id in self.call_ids:
            raise ProviderProtocolError("Provider call IDs must be unique")
        self.call_ids.add(call_id)
        result = await self._request("call", {"name": name, "arguments": arguments, "call_id": call_id})
        if not isinstance(result, str):
            self.failed = True
            raise ProviderProtocolError("Provider call observation must be a string")
        return result

    async def canonical_state(self) -> JsonValue:
        return await self._request("state", {})

    async def stop(self) -> None:
        """Remove this trial's container, including after a failed handshake."""
        process = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "--force",
            self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with asyncio.timeout(PROTOCOL_TIMEOUT):
            _stdout, stderr = await process.communicate()
        if process.returncode:
            raise ProviderProtocolError(f"Provider container cleanup failed: {stderr.decode(errors='replace')}")
        if self.process is not None:
            async with asyncio.timeout(PROTOCOL_TIMEOUT):
                await self.process.wait()
        if self.stderr_task is not None:
            await self.stderr_task
        self._record({"diagnostics": self.diagnostics.decode(errors="replace")})


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON scalar: {value}")
