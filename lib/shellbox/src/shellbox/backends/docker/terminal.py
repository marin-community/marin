# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded host-side terminal reads from one Docker machine."""

import asyncio
import base64
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx

from shellbox.machine import InvalidWorkspaceFile

COLLECTION_TIMEOUT = 30
MAX_CONTROL_OUTPUT_BYTES = 16 * 1024
MAX_STAT_HEADER_BYTES = 8 * 1024
TAR_ENVELOPE_BYTES = 64 * 1024
# Docker PathStat uses Go's os.FileMode type bits, not POSIX st_mode.
GO_MODE_DIR = 1 << 31
GO_MODE_TYPE = GO_MODE_DIR | (1 << 27) | (1 << 26) | (1 << 25) | (1 << 24) | (1 << 21) | (1 << 19)


class DockerMachineState(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"
    STOP_FAILED = "stop_failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class DockerControlPlane:
    socket: str
    api_version: str


@dataclass(frozen=True)
class DockerPathStat:
    name: str
    size: int
    mode: int
    link_target: str


async def _docker_output(*arguments: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        stdin=asyncio.subprocess.DEVNULL,
    )
    assert process.stdout is not None
    output = bytearray()
    try:
        async with asyncio.timeout(COLLECTION_TIMEOUT):
            while chunk := await process.stdout.read(MAX_CONTROL_OUTPUT_BYTES):
                if len(output) + len(chunk) > MAX_CONTROL_OUTPUT_BYTES:
                    raise RuntimeError("Docker control-plane output exceeds its byte limit")
                output.extend(chunk)
            if await process.wait() != 0:
                raise RuntimeError("Cannot resolve Docker control-plane configuration")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    return output.decode("utf-8").strip()


async def docker_control_plane() -> DockerControlPlane:
    """Resolve the local Docker CLI context and installed Engine API version."""
    selected = os.environ.get("DOCKER_HOST") if not os.environ.get("DOCKER_CONTEXT") else None
    if selected is None:
        endpoint = await _docker_output("context", "inspect", "--format", "{{json .Endpoints.docker}}")
        selected = json.loads(endpoint)["Host"]
    transport = urlsplit(selected)
    if transport.scheme != "unix" or transport.netloc or not transport.path.startswith("/"):
        raise NotImplementedError("Harbor file collection requires a local Unix-socket Docker context")
    version = await _docker_output("version", "--format", "{{.Server.APIVersion}}")
    if re.fullmatch(r"[0-9]+\.[0-9]+", version) is None:
        raise RuntimeError("Docker daemon returned an invalid API version")
    return DockerControlPlane(transport.path, version)


async def validate_linux_image(reference: str) -> None:
    """Reject a daemon or local image that cannot run the POSIX adapter."""
    daemon = await _docker_output("info", "--format", "{{.OSType}}")
    image = await _docker_output("inspect", "--format", "{{.Os}}", reference)
    if daemon != "linux" or image != "linux":
        raise NotImplementedError("Shellbox Harbor Docker requires a Linux daemon and image")


def _path_stat(response: httpx.Response) -> DockerPathStat:
    response.raise_for_status()
    header = response.headers["X-Docker-Container-Path-Stat"]
    if len(header) > MAX_STAT_HEADER_BYTES:
        raise RuntimeError("Docker path metadata exceeds its byte limit")
    value = json.loads(base64.b64decode(header, validate=True))
    if (
        not isinstance(value["name"], str)
        or type(value["size"]) is not int
        or value["size"] < 0
        or type(value["mode"]) is not int
        or value["mode"] < 0
        or not isinstance(value["linkTarget"], str)
    ):
        raise RuntimeError("Docker returned invalid path metadata")
    return DockerPathStat(value["name"], value["size"], value["mode"], value["linkTarget"])


def _file_content(payload: bytes, path: str, size: int, max_bytes: int) -> bytes:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            member = archive.next()
            if member is None or member.name != path or not member.isfile():
                raise InvalidWorkspaceFile("Submission archive must contain the declared regular file")
            if member.size > max_bytes:
                raise InvalidWorkspaceFile(f"Submission file {path!r} exceeds its byte limit")
            if member.size != size:
                raise RuntimeError("Docker file metadata differs from its archive")
            source = archive.extractfile(member)
            assert source is not None
            with source:
                content = source.read(max_bytes + 1)
            if len(content) != size or archive.next() is not None:
                raise RuntimeError("Docker returned an incomplete or multi-file archive")
            return content
    except tarfile.TarError as error:
        raise RuntimeError("Docker returned a malformed final-file archive") from error


class DockerTerminalReader:
    """Acquire a single regular file while the selected machine is stable."""

    def __init__(self, container: str, workdir: str, control: DockerControlPlane, state: DockerMachineState):
        workspace = PurePosixPath(workdir)
        if workspace.name in {"", ".", ".."} or workspace.parent != PurePosixPath("/") or str(workspace) != workdir:
            raise ValueError("Terminal file collection requires one normalized root-child workdir")
        self.container = container
        self.workdir = workdir
        self.archive_socket = control.socket
        self.archive_api_version = control.api_version
        self.state = state

    def _archive_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self.archive_socket),
            base_url=f"http://docker/v{self.archive_api_version}/",
            timeout=COLLECTION_TIMEOUT,
            trust_env=False,
        )

    async def _metadata(self, client: httpx.AsyncClient, path: str) -> DockerPathStat | None:
        endpoint = f"containers/{self.container}/archive"
        response = await client.head(endpoint, params={"path": path})
        if response.status_code == 404:
            # HEAD has no error body. Check the container root before assigning zero for a missing file.
            root = await client.head(endpoint, params={"path": "/"})
            _path_stat(root)
            return None
        return _path_stat(response)

    async def _validate_workspace_directory(self, client: httpx.AsyncClient) -> None:
        metadata = await self._metadata(client, self.workdir)
        if metadata is None:
            raise InvalidWorkspaceFile("Submission workspace is missing")
        if metadata.link_target or metadata.mode & GO_MODE_TYPE != GO_MODE_DIR:
            raise InvalidWorkspaceFile("Submission workspace must be a real, symlink-free directory")

    async def read_file(self, path: str, max_bytes: int) -> bytes | None:
        """Acquire one terminal file through Docker's host-side archive API."""
        if path in {"", ".", ".."} or "/" in path or "\\" in path or "\x00" in path or max_bytes <= 0:
            raise ValueError("Docker file collection requires a flat filename and bounded byte limit")
        target = str(PurePosixPath(self.workdir) / path)
        endpoint = f"containers/{self.container}"
        async with self._archive_client() as client:
            pause_requested = False
            try:
                async with asyncio.timeout(COLLECTION_TIMEOUT):
                    state = await self._state(client, endpoint)
                    if state["Restarting"] or state["Paused"]:
                        raise RuntimeError("Docker machine is not available for terminal collection")
                    if state["Running"] != (self.state is DockerMachineState.RUNNING):
                        raise RuntimeError("Docker machine state differs from the collection cutoff")
                    if state["Running"]:
                        pause_requested = True
                        response = await client.post(f"{endpoint}/pause")
                        response.raise_for_status()
                    await self._validate_workspace_directory(client)
                    metadata = await self._metadata(client, target)
                    if metadata is None:
                        return None
                    if metadata.link_target or metadata.mode & GO_MODE_TYPE:
                        raise InvalidWorkspaceFile(f"Submission file {path!r} is not a regular, symlink-free file")
                    if metadata.size > max_bytes:
                        raise InvalidWorkspaceFile(f"Submission file {path!r} exceeds its byte limit")
                    payload = bytearray()
                    async with client.stream(
                        "GET", f"{endpoint}/archive", params={"path": target}, headers={"Accept-Encoding": "identity"}
                    ) as response:
                        archive_stat = _path_stat(response)
                        if response.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
                            raise RuntimeError("Docker returned an unsupported compressed final-file archive")
                        if archive_stat != metadata:
                            raise RuntimeError("Docker path metadata changed during terminal collection")
                        async for chunk in response.aiter_raw(chunk_size=TAR_ENVELOPE_BYTES):
                            if len(payload) + len(chunk) > max_bytes + TAR_ENVELOPE_BYTES:
                                raise InvalidWorkspaceFile("Submission archive exceeds its byte limit")
                            payload.extend(chunk)
                    return _file_content(bytes(payload), path, metadata.size, max_bytes)
            finally:
                if pause_requested:
                    cleanup = asyncio.create_task(self._unpause(client, endpoint))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await cleanup
                        raise

    async def _unpause(self, client: httpx.AsyncClient, endpoint: str) -> None:
        response = await client.post(f"{endpoint}/unpause")
        # The pause request may have failed before freezing the container.
        if response.status_code == 409:
            return
        response.raise_for_status()

    async def _state(self, client: httpx.AsyncClient, endpoint: str) -> dict[str, bool]:
        payload = bytearray()
        async with client.stream("GET", f"{endpoint}/json", headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity").strip().lower() != "identity":
                raise RuntimeError("Docker returned compressed container state")
            async for chunk in response.aiter_raw(chunk_size=TAR_ENVELOPE_BYTES):
                if len(payload) + len(chunk) > TAR_ENVELOPE_BYTES:
                    raise RuntimeError("Docker container state exceeds its byte limit")
                payload.extend(chunk)
        state = json.loads(payload)["State"]
        if any(type(state[key]) is not bool for key in ("Running", "Paused", "Restarting")):
            raise RuntimeError("Docker returned invalid container state")
        return {key: state[key] for key in ("Running", "Paused", "Restarting")}
