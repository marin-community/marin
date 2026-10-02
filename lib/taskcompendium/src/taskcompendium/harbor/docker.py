# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned Docker workspace for caller-selected Harbor agents."""

import asyncio
import base64
import io
import json
import os
import re
import shlex
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from harbor.environments.docker.docker import DockerEnvironment
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.path_validation import validate_relative_file_path

from taskcompendium.lowering import DOCKER_DEFINITION_FILES
from taskcompendium.models import DOCKER_IMAGE_PATTERN, validate_workspace_path
from taskcompendium.submission import MAX_SUBMISSION_FILE_BYTES, SubmissionFailure

COLLECTION_TIMEOUT = 30
MAX_CONTROL_OUTPUT_BYTES = 16 * 1024
MAX_STAT_HEADER_BYTES = 8 * 1024
TAR_ENVELOPE_BYTES = 64 * 1024
# Docker PathStat uses Go's os.FileMode type bits, not POSIX st_mode.
GO_MODE_DIR = 1 << 31
GO_MODE_TYPE = GO_MODE_DIR | (1 << 27) | (1 << 26) | (1 << 25) | (1 << 24) | (1 << 21) | (1 << 19)


@dataclass(frozen=True)
class DockerControlPlane:
    socket: str
    api_version: str


class DockerPathStat(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    size: int = Field(ge=0)
    mode: int = Field(ge=0)
    link_target: str = Field(alias="linkTarget")


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


def _path_stat(response: httpx.Response) -> DockerPathStat:
    response.raise_for_status()
    header = response.headers["X-Docker-Container-Path-Stat"]
    if len(header) > MAX_STAT_HEADER_BYTES:
        raise RuntimeError("Docker path metadata exceeds its byte limit")
    return DockerPathStat.model_validate_json(base64.b64decode(header, validate=True))


def _file_content(payload: bytes, path: str, size: int, max_bytes: int) -> bytes:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            member = archive.next()
            if member is None or member.name != path or not member.isfile():
                raise SubmissionFailure("Submission archive must contain the declared regular file")
            if member.size > max_bytes:
                raise SubmissionFailure(f"Submission file {path!r} exceeds its byte limit")
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


class DockerWorkspaceEnvironment(DockerEnvironment):
    """Use Harbor's isolated Docker lifecycle for one trial workspace."""

    def __init__(self, *args, archive_socket: str, archive_api_version: str, mounts=None, **kwargs) -> None:
        self.archive_socket = archive_socket
        self.archive_api_version = archive_api_version
        self.container_id: str | None = None
        # Harbor normally shares verifier logs with the worker. Keep grading private.
        visible = [mount for mount in mounts or [] if mount["target"] != "/logs/verifier"]
        super().__init__(*args, mounts=visible, **kwargs)

    def _validate_definition(self) -> None:
        super()._validate_definition()
        image = self.task_env_config.docker_image
        match = re.fullmatch(DOCKER_IMAGE_PATTERN, image or "")
        if match is None:
            raise ValueError("Docker task requires its SHA256-pinned image")
        if self.task_env_config.workdir is None:
            raise ValueError("Docker task requires a declared workdir")
        validate_workspace_path(self.task_env_config.workdir)
        allowed_mounts = {"/logs/agent": self.trial_paths.agent_dir, "/logs/artifacts": self.trial_paths.artifacts_dir}
        for mount in self._mounts:
            if (
                mount["target"] not in allowed_mounts
                or Path(mount["source"]).resolve() != Path(str(allowed_mounts[mount["target"]])).resolve()
            ):
                raise ValueError("Docker workspace cannot mount private or caller-selected host paths")
        if any((self.environment_dir / name).exists() for name in DOCKER_DEFINITION_FILES):
            raise ValueError("Pinned Docker tasks cannot override the image with a build or compose file")
        if self.extra_docker_compose_paths:
            raise ValueError("Pinned Docker tasks cannot use extra compose files")

    async def _run_docker_compose_command(self, command: list[str], check: bool = True, timeout_sec: int | None = None):
        if command and command[0] == "up":
            command = ["up", "--pull", "never", *command[1:]]
        return await super()._run_docker_compose_command(command, check=check, timeout_sec=timeout_sec)

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Pinned Docker tasks cannot force a build")
        await super().start(force_build=False)
        result = await self._run_docker_compose_command(["ps", "-q", "main"])
        container_id = (result.stdout or "").strip()
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            raise RuntimeError("Expected one Docker trial container")
        self.container_id = container_id
        async with self._archive_client() as client:
            await self._validate_workspace_directory(client)

    async def _upload_environment_dir_after_start(self) -> None:
        workdir = validate_workspace_path(self.task_env_config.workdir or "")
        identity = await self.exec("id -u; id -g", cwd="/")
        identifiers = (identity.stdout or "").splitlines()
        if identity.return_code != 0 or len(identifiers) != 2 or not all(value.isdecimal() for value in identifiers):
            raise RuntimeError("Cannot resolve the Docker worker's numeric UID and GID")
        owner = ":".join(identifiers)
        result = await self.exec(f"mkdir -p {shlex.quote(str(workdir))}", cwd="/", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot create Docker workspace: {result.stderr or result.stdout}")
        await super()._upload_environment_dir_after_start()
        # Compose cp creates root-owned inputs. Transfer only declared public paths,
        # then restore modes because chown can clear set-id bits.
        modes: dict[int, list[str]] = {0o755: [str(workdir)]}
        for source in sorted(self.environment_dir.rglob("*")):
            target = str(workdir / source.relative_to(self.environment_dir))
            mode = 0o755 if source.is_dir() else source.stat().st_mode & 0o7777
            modes.setdefault(mode, []).append(target)
        paths = " ".join(shlex.quote(path) for group in modes.values() for path in group)
        result = await self.exec(f"chown -- {owner} {paths}", cwd="/", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot assign Docker workspace ownership: {result.stderr or result.stdout}")
        for mode, group in modes.items():
            paths = " ".join(shlex.quote(path) for path in group)
            result = await self.exec(f"chmod {mode:o} {paths}", cwd="/", user="root")
            if result.return_code != 0:
                raise RuntimeError(f"Cannot restore Docker resource modes: {result.stderr or result.stdout}")

    def _archive_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=self.archive_socket),
            base_url=f"http://docker/v{self.archive_api_version}/",
            timeout=COLLECTION_TIMEOUT,
            trust_env=False,
        )

    async def _metadata(self, client: httpx.AsyncClient, path: str) -> DockerPathStat | None:
        endpoint = f"containers/{self.container_id}/archive"
        response = await client.head(endpoint, params={"path": path})
        if response.status_code == 404:
            # HEAD has no error body. Check the container root before assigning zero for a missing file.
            root = await client.head(endpoint, params={"path": "/"})
            _path_stat(root)
            return None
        return _path_stat(response)

    async def _validate_workspace_directory(self, client: httpx.AsyncClient) -> None:
        metadata = await self._metadata(client, self.task_env_config.workdir or "")
        if metadata is None:
            raise SubmissionFailure("Submission workspace is missing")
        if metadata.link_target or metadata.mode & GO_MODE_TYPE != GO_MODE_DIR:
            raise SubmissionFailure("Submission workspace must be a real, symlink-free directory")

    async def read_workspace_file(self, path: str, max_bytes: int) -> bytes | None:
        """Acquire one terminal file through Docker's host-side archive API."""
        validate_relative_file_path(path)
        if "/" in path or not 0 < max_bytes <= MAX_SUBMISSION_FILE_BYTES:
            raise ValueError("Docker file collection requires a flat filename and bounded byte limit")
        if self.container_id is None:
            raise RuntimeError("Docker trial container has not started")
        target = str(validate_workspace_path(self.task_env_config.workdir or "") / path)
        endpoint = f"containers/{self.container_id}"
        async with self._archive_client() as client:
            pause_requested = False
            try:
                async with asyncio.timeout(COLLECTION_TIMEOUT):
                    pause_requested = True
                    response = await client.post(f"{endpoint}/pause")
                    response.raise_for_status()
                    await self._validate_workspace_directory(client)
                    metadata = await self._metadata(client, target)
                    if metadata is None:
                        return None
                    if metadata.link_target or metadata.mode & GO_MODE_TYPE:
                        raise SubmissionFailure(f"Submission file {path!r} is not a regular, symlink-free file")
                    if metadata.size > max_bytes:
                        raise SubmissionFailure(f"Submission file {path!r} exceeds its byte limit")
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
                                raise SubmissionFailure("Submission archive exceeds its byte limit")
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

    async def stop(self, delete: bool) -> None:
        """Clean trial containers and volumes while retaining the shared pinned image."""
        try:
            try:
                await self.prepare_logs_for_host()
            finally:
                if self._keep_containers:
                    await self._run_docker_compose_command(["stop"])
                elif delete:
                    await self._run_docker_compose_command(["down", "--volumes", "--remove-orphans"])
                else:
                    await self._run_docker_compose_command(["down"])
        finally:
            self._cleanup_mounts_compose_file()
            self._cleanup_resources_compose_file()
