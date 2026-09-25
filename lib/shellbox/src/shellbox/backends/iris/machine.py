# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a registry image as a gVisor Iris job and exec into its container."""

import asyncio
import base64
import math
import re
import shlex
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

from iris.cli.connect import ControllerEndpoint, connect_controller
from iris.client import IrisClient, Job
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from rigging.timing import Duration

from shellbox.image import RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec, NetworkPolicy, Result, UnsupportedMachineSpec

TRANSFER_CHUNK_BYTES = 128 * 1024
DEFAULT_MEMORY_MB = 2048
DEFAULT_DISK_MB = 10240
DEFAULT_SCHEDULING_TIMEOUT = 600
DEFAULT_JOB_TTL = 6 * 60 * 60
RPC_PADDING_SECONDS = 60


class IrisMachine:
    """One Iris task container; file transfer uses bounded base64 exec calls."""

    def __init__(
        self,
        endpoint: ControllerEndpoint,
        client: IrisClient,
        rpc: ControllerServiceClientSync,
        job: Job,
        task_id: str,
        spec: MachineSpec,
    ):
        self.endpoint = endpoint
        self.client = client
        self.rpc = rpc
        self.job = job
        self.task_id = task_id
        self.spec = spec
        self._closed = False

    def _exec_sync(
        self, argv: list[str], timeout: float | None = None
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        if self._closed:
            raise RuntimeError("Machine is closed")
        seconds = math.ceil(timeout) if timeout is not None else -1
        response = self.rpc.exec_in_container(
            controller_pb2.Controller.ExecInContainerRequest(
                task_id=self.task_id, command=argv, timeout_seconds=seconds
            ),
            timeout_ms=(seconds + RPC_PADDING_SECONDS) * 1000 if seconds >= 0 else DEFAULT_JOB_TTL * 1000,
        )
        if response.error:
            raise RuntimeError(f"Iris exec failed: {response.error}")
        return response

    async def _script(
        self, script: str, timeout: float | None = None
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        return await asyncio.to_thread(self._exec_sync, ["sh", "-c", script], timeout)

    async def _checked(self, script: str) -> str:
        response = await self._script(script)
        if response.exit_code:
            raise RuntimeError(f"Iris command failed ({response.exit_code}): {response.stderr or response.stdout}")
        return response.stdout

    async def _upload_bytes(self, data: bytes, target: str) -> None:
        quoted = shlex.quote(target)
        await self._checked(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))} && : > {quoted}")
        for offset in range(0, len(data), TRANSFER_CHUNK_BYTES):
            encoded = base64.b64encode(data[offset : offset + TRANSFER_CHUNK_BYTES]).decode("ascii")
            await self._checked(f"printf '%s' {encoded} | base64 -d >> {quoted}")

    async def _download_bytes(self, source: str, limit: int | None = None) -> tuple[bytes, bool]:
        quoted = shlex.quote(source)
        size = int((await self._checked(f"wc -c < {quoted}")).strip())
        count = size if limit is None else min(size, limit)
        chunks = []
        for offset in range(0, count, TRANSFER_CHUNK_BYTES):
            length = min(TRANSFER_CHUNK_BYTES, count - offset)
            encoded = await self._checked(f"tail -c +{offset + 1} {quoted} | head -c {length} | base64")
            chunks.append(base64.b64decode(encoded))
        return b"".join(chunks), size > count

    async def run(self, command: Command) -> Result:
        if self._closed:
            raise RuntimeError("Machine is closed")
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.output_limit_bytes < 0:
            raise ValueError("Output limit must be nonnegative")
        prefix = f"/tmp/.shellbox-{uuid.uuid4().hex}"
        stdin_path, stdout_path, stderr_path = (f"{prefix}-{part}" for part in ("in", "out", "err"))
        if command.stdin:
            await self._upload_bytes(command.stdin, stdin_path)
        exports = "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in {**self.spec.env, **command.env}.items()
        )
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in {**self.spec.env, **command.env}):
            raise ValueError("Environment variable names must be shell identifiers")
        script = (
            f"{exports}\ncd {shlex.quote(command.cwd or self.spec.workdir)} || exit 1\n"
            f"{shlex.join(command.argv)} < {shlex.quote(stdin_path) if command.stdin else '/dev/null'} "
            f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        try:
            response = await asyncio.wait_for(self._script(script), timeout=command.timeout)
            stdout, stdout_truncated = await self._download_bytes(stdout_path, command.output_limit_bytes)
            stderr, stderr_truncated = await self._download_bytes(stderr_path, command.output_limit_bytes)
            return Result(response.exit_code, stdout, stderr, stdout_truncated, stderr_truncated, ExitReason.EXITED)
        except TimeoutError:
            await self.close()
            return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
        except asyncio.CancelledError:
            await self.close()
            raise
        finally:
            if not self._closed:
                await self._script(
                    f"rm -f {shlex.quote(stdin_path)} {shlex.quote(stdout_path)} {shlex.quote(stderr_path)}"
                )

    async def upload(self, source: Path, target: str) -> None:
        if source.is_dir():
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                with tarfile.open(archive.name, "w:gz") as tar:
                    tar.add(source, arcname=".")
                remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
                await self._upload_bytes(Path(archive.name).read_bytes(), remote_archive)
            try:
                await self._checked(
                    f"mkdir -p {shlex.quote(target)} && tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target)}"
                )
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        await self._upload_bytes(source.read_bytes(), target)

    async def download(self, source: str, target: Path) -> None:
        probe = await self._script(f"test -d {shlex.quote(source)}")
        if probe.exit_code == 0:
            remote_archive = f"/tmp/.shellbox-{uuid.uuid4().hex}.tar.gz"
            await self._checked(f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source)} .")
            try:
                data, _ = await self._download_bytes(remote_archive)
                target.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
                    Path(archive.name).write_bytes(data)
                    with tarfile.open(archive.name, "r:gz") as tar:
                        tar.extractall(target, filter="data")
            finally:
                await self._script(f"rm -f {shlex.quote(remote_archive)}")
            return
        data, _ = await self._download_bytes(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.to_thread(self.job.terminate)
        finally:
            try:
                await asyncio.to_thread(self.client.shutdown)
            finally:
                self.endpoint.close()


class IrisMachineFactory:
    """Submit a CPU-only gVisor job from a registry image."""

    def __init__(
        self,
        *,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout: int = DEFAULT_SCHEDULING_TIMEOUT,
        job_ttl: int = DEFAULT_JOB_TTL,
        disk_mb: int = DEFAULT_DISK_MB,
    ):
        if (cluster is None) == (controller_url is None):
            raise ValueError("Specify exactly one Iris cluster or controller URL")
        self.cluster = cluster
        self.controller_url = controller_url
        self.scheduling_timeout = scheduling_timeout
        self.job_ttl = job_ttl
        self.disk_mb = disk_mb

    async def create(self, spec: MachineSpec) -> IrisMachine:
        if not isinstance(spec.source, RegistryImage):
            raise UnsupportedMachineSpec("Iris requires a registry image reference")
        if spec.network is NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("Iris does not provide per-job network denial; select NetworkPolicy.ALLOW")
        return await asyncio.to_thread(self._create_sync, spec)

    def _create_sync(self, spec: MachineSpec) -> IrisMachine:
        endpoint = connect_controller(cluster_name=self.cluster, controller_url=self.controller_url)
        url, credentials = endpoint.url, endpoint.credentials
        client = None
        job = None
        try:
            client = IrisClient.remote(url, workspace=None, credentials=credentials)
            rpc = ControllerServiceClientSync(
                address=url,
                timeout_ms=RPC_PADDING_SECONDS * 1000,
                interceptors=credentials.interceptors() if credentials is not None else [],
                accept_compression=IRIS_RPC_COMPRESSIONS,
                send_compression=None,
            )
            job = client.submit(
                entrypoint=Entrypoint.from_command("sleep", "infinity"),
                name=f"shellbox-{uuid.uuid4().hex}",
                environment=EnvironmentSpec(setup_scripts=[]),
                resources=ResourceSpec(
                    cpu=1, memory=(spec.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024, disk=self.disk_mb * 1024 * 1024
                ),
                task_image=spec.source.reference,
                container_profile=job_pb2.CONTAINER_PROFILE_GVISOR,
                scheduling_timeout=Duration.from_seconds(self.scheduling_timeout),
                timeout=Duration.from_seconds(self.job_ttl),
                max_retries_failure=0,
                max_retries_preemption=0,
            )
            deadline = time.monotonic() + self.scheduling_timeout
            while time.monotonic() < deadline:
                tasks = job.tasks()
                if tasks:
                    status = tasks[0].status()
                    if status.state == job_pb2.TASK_STATE_RUNNING:
                        machine = IrisMachine(endpoint, client, rpc, job, tasks[0].task_id.to_wire(), spec)
                        created = machine._exec_sync(["mkdir", "-p", spec.workdir])
                        if created.exit_code:
                            raise RuntimeError(f"Failed to create Iris workdir {spec.workdir}: {created.stderr}")
                        return machine
                    if status.state not in (
                        job_pb2.TASK_STATE_PENDING,
                        job_pb2.TASK_STATE_BUILDING,
                        job_pb2.TASK_STATE_ASSIGNED,
                    ):
                        raise RuntimeError(f"Iris sandbox task failed before running: {status.error}")
                time.sleep(2)
            raise TimeoutError(f"Iris sandbox did not start within {self.scheduling_timeout} seconds")
        except BaseException:
            try:
                if job is not None:
                    job.terminate()
            finally:
                try:
                    if client is not None:
                        client.shutdown()
                finally:
                    endpoint.close()
            raise
