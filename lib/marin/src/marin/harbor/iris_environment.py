# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor environment backend that runs each sandbox as an Iris job.

The sandbox *is* the Iris task container: the task's prebuilt ``docker_image``
is submitted as the job's ``task_image`` with a ``sleep infinity`` entrypoint,
and all interaction goes through the controller's ``ExecInContainer`` RPC.
No docker-in-docker and no elevated container profile is required. CPU-only
resource requests carry no device or preemptible constraint, so sandboxes
bin-pack onto spare host CPUs of preemptible TPU workers.

Only prebuilt-image tasks (``[environment] docker_image = ...``) are
supported; Dockerfile/compose builds are not.

Usage:
    harbor trials start -p <task-dir> -a oracle \\
        --environment-import-path marin.harbor.iris_environment:IrisEnvironment \\
        --environment-kwarg cluster=marin
"""

import asyncio
import base64
import re
import shlex
import tarfile
import tempfile
import time
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from iris.cli.connect import open_controller_endpoint
from iris.client import IrisClient, Job
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from rigging.timing import Duration

ENVIRONMENT_TYPE = "iris"

UPLOAD_CHUNK_BYTES = 256 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_SCHEDULING_TIMEOUT = 600
# Padding on the RPC deadline so the in-container timeout fires first.
EXEC_RPC_PADDING = 60
# RPC deadline for execs with no in-container timeout.
UNLIMITED_EXEC_RPC_TIMEOUT_MS = 24 * 60 * 60 * 1000

DEFAULT_CPUS = 1
DEFAULT_MEMORY_MB = 2048
DEFAULT_STORAGE_MB = 10240


def _owned_by_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


class IrisEnvironment(BaseEnvironment):
    """Runs the task's prebuilt image as a long-lived Iris job and execs into it."""

    def __init__(
        self,
        *args,
        cluster: str | None = None,
        controller_url: str | None = None,
        scheduling_timeout_sec: int | str = DEFAULT_SCHEDULING_TIMEOUT,
        **kwargs,
    ):
        """
        Args:
            cluster: Iris cluster name resolved via the standard config search
                path (e.g. "marin"). Mutually exclusive with controller_url.
            controller_url: Direct controller URL, bypassing cluster config.
            scheduling_timeout_sec: How long to wait for the sandbox task to
                be scheduled and reach RUNNING.
        """
        super().__init__(*args, **kwargs)
        if cluster is None and controller_url is None:
            raise ValueError("IrisEnvironment requires a `cluster` or `controller_url` kwarg.")
        self._cluster = cluster
        self._controller_url = controller_url
        self._scheduling_timeout = int(scheduling_timeout_sec)
        self._iris: IrisClient | None = None
        self._rpc: ControllerServiceClientSync | None = None
        self._job: Job | None = None
        self._task_id: str | None = None

    @staticmethod
    def type() -> str:
        return ENVIRONMENT_TYPE

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # cpu/memory are scheduler reservations (bin-packing accounting), not cgroup limits.
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    def _validate_definition(self):
        if not self.task_env_config.docker_image:
            raise ValueError(
                "IrisEnvironment only supports prebuilt-image tasks "
                "([environment] docker_image = ...); Dockerfile builds are not supported."
            )

    def _job_name(self) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", self.session_id).strip("-")
        return f"harbor-{sanitized}"[:120]

    async def start(self, force_build: bool) -> None:
        del force_build  # Prebuilt images only; nothing to build.
        await asyncio.to_thread(self._start_sync)
        # Non-mounting environments use the trial's mount targets as mkdir
        # hints (e.g. /logs/agent, /logs/verifier); see BaseEnvironment.mounts.
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    def _start_sync(self) -> None:
        # Resolve the controller URL and credentials, then close the endpoint
        # context immediately: it pushes a thread-local click.Context, which
        # must be popped by the same thread that pushed it, and start/stop run
        # in different asyncio.to_thread workers. The URL outlives the context
        # for IAP-fronted and direct-URL clusters (no local tunnel).
        with open_controller_endpoint(cluster_name=self._cluster, controller_url=self._controller_url) as endpoint:
            url = endpoint.url
            credentials = endpoint.credentials
        self._iris = IrisClient.remote(url, workspace=None, credentials=credentials)
        self._rpc = ControllerServiceClientSync(
            address=url,
            timeout_ms=EXEC_RPC_PADDING * 1000,
            interceptors=credentials.interceptors() if credentials is not None else [],
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )
        self._job = self._iris.submit(
            entrypoint=Entrypoint.from_command("sleep", "infinity"),
            name=self._job_name(),
            # setup_scripts=[] means no setup phase: run the task image as-is
            # (sandbox images have no uv/iris toolchain).
            environment=EnvironmentSpec(setup_scripts=[]),
            resources=ResourceSpec(
                cpu=float(self.task_env_config.cpus or DEFAULT_CPUS),
                memory=(self.task_env_config.memory_mb or DEFAULT_MEMORY_MB) * 1024 * 1024,
                disk=(self.task_env_config.storage_mb or DEFAULT_STORAGE_MB) * 1024 * 1024,
            ),
            task_image=self.task_env_config.docker_image,
            scheduling_timeout=Duration.from_seconds(self._scheduling_timeout),
            # A restarted sandbox is a fresh container with all trial state lost,
            # so never retry: fail the trial and let harbor-level resume handle it.
            max_retries_failure=0,
            max_retries_preemption=0,
        )
        self._task_id = self._wait_for_running()
        self.logger.info(f"Sandbox task {self._task_id} running (job {self._job.job_id})")

    def _wait_for_running(self) -> str:
        assert self._job is not None
        deadline = time.monotonic() + self._scheduling_timeout
        while time.monotonic() < deadline:
            tasks = self._job.tasks()
            if tasks:
                status = tasks[0].status()
                if status.state == job_pb2.TASK_STATE_RUNNING:
                    return tasks[0].task_id.to_wire()
                if status.state not in (
                    job_pb2.TASK_STATE_PENDING,
                    job_pb2.TASK_STATE_BUILDING,
                    job_pb2.TASK_STATE_ASSIGNED,
                ):
                    raise RuntimeError(
                        f"Sandbox task {tasks[0].task_id} entered "
                        f"{job_pb2.TaskState.Name(status.state)} before running: {status.error or 'no error'}"
                    )
            time.sleep(2)
        raise TimeoutError(f"Sandbox job {self._job.job_id} not running after {self._scheduling_timeout}s")

    async def stop(self, delete: bool):
        del delete  # A terminated Iris job cannot be resumed; stop always deletes.
        await asyncio.to_thread(self._stop_sync)

    def _stop_sync(self) -> None:
        if self._job is not None:
            self._job.terminate()
            self._job = None
        self._task_id = None
        self._rpc = None
        if self._iris is not None:
            self._iris.shutdown()
            self._iris = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        script = self._build_script(command, cwd=cwd, env=env, user=user)
        return await asyncio.to_thread(self._exec_sync, script, timeout_sec)

    def _build_script(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
    ) -> str:
        lines = []
        merged_env = self._merge_env(env)
        if merged_env:
            lines.extend(f"export {key}={shlex.quote(value)}" for key, value in merged_env.items())
        if cwd:
            lines.append(f"cd {shlex.quote(cwd)} || exit 1")
        lines.append(command)
        script = "\n".join(lines)

        resolved_user = self._resolve_user(user)
        if resolved_user not in (None, "root", 0, "0"):
            script = f"su -s /bin/sh -c {shlex.quote(script)} {shlex.quote(str(resolved_user))}"
        return script

    def _exec_sync(self, script: str, timeout_sec: int | None) -> ExecResult:
        if self._rpc is None or self._task_id is None:
            raise RuntimeError("IrisEnvironment is not started")
        if timeout_sec is None:
            container_timeout, rpc_timeout_ms = -1, UNLIMITED_EXEC_RPC_TIMEOUT_MS
        else:
            container_timeout = timeout_sec
            rpc_timeout_ms = (timeout_sec + EXEC_RPC_PADDING) * 1000
        response = self._rpc.exec_in_container(
            controller_pb2.Controller.ExecInContainerRequest(
                task_id=self._task_id,
                command=["sh", "-c", script],
                timeout_seconds=container_timeout,
            ),
            timeout_ms=rpc_timeout_ms,
        )
        if response.error:
            raise RuntimeError(f"exec in {self._task_id} failed: {response.error}")
        return ExecResult(stdout=response.stdout, stderr=response.stderr, return_code=response.exit_code)

    async def upload_file(self, source_path: Path | str, target_path: str):
        data = Path(source_path).read_bytes()
        quoted = shlex.quote(target_path)
        parent = shlex.quote(str(Path(target_path).parent))
        await self._check_exec(f"mkdir -p {parent} && rm -f {quoted} && touch {quoted}")
        for offset in range(0, len(data), UPLOAD_CHUNK_BYTES):
            encoded = base64.b64encode(data[offset : offset + UPLOAD_CHUNK_BYTES]).decode()
            await self._check_exec(f"printf '%s' {encoded} | base64 -d >> {quoted}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        remote_tar = f"/tmp/.hb-upload-{Path(str(source_dir)).name}.tar.gz"
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as local_tar:
            with tarfile.open(local_tar.name, "w:gz") as tf:
                # Strip local ownership so in-container extraction never chowns
                # (the sandbox user may be non-root or unable to chown).
                tf.add(str(source_dir), arcname=".", filter=_owned_by_root)
            await self.upload_file(local_tar.name, remote_tar)
        quoted_target = shlex.quote(target_dir)
        quoted_tar = shlex.quote(remote_tar)
        await self._check_exec(
            f"mkdir -p {quoted_target} && tar xzf {quoted_tar} -C {quoted_target} && rm -f {quoted_tar}"
        )

    async def download_file(self, source_path: str, target_path: Path | str):
        quoted = shlex.quote(source_path)
        await self._check_exec(f"test -f {quoted}", error=f"remote file not found: {source_path}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as out:
            offset = 0
            while True:
                result = await self._check_exec(
                    f"tail -c +{offset + 1} {quoted} | head -c {DOWNLOAD_CHUNK_BYTES} | base64"
                )
                chunk = base64.b64decode(result.stdout or "")
                out.write(chunk)
                offset += len(chunk)
                if len(chunk) < DOWNLOAD_CHUNK_BYTES:
                    break

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        remote_tar = f"/tmp/.hb-download-{Path(source_dir).name}.tar.gz"
        await self._check_exec(
            f"tar czf {shlex.quote(remote_tar)} -C {shlex.quote(source_dir)} .",
            error=f"failed to archive remote dir: {source_dir}",
        )
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as local_tar:
            await self.download_file(remote_tar, local_tar.name)
            with tarfile.open(local_tar.name, "r:gz") as tf:
                tf.extractall(path=target, filter="data")
        await self._check_exec(f"rm -f {shlex.quote(remote_tar)}")

    async def _check_exec(self, command: str, error: str | None = None) -> ExecResult:
        result = await self.exec(command, user="root")
        if result.return_code != 0:
            detail = result.stderr or result.stdout or "no output"
            raise RuntimeError(f"{error or f'command failed: {command}'} (rc={result.return_code}): {detail}")
        return result
