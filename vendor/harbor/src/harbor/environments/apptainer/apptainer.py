"""Bridge-backed Apptainer environment for HPC clusters.

``ApptainerEnvironment`` is a Harbor :class:`BaseEnvironment` that does **not**
run Apptainer locally. Instead it delegates every lifecycle operation (start,
exec, file transfer, stop) over HTTP to a *bridge server* which dispatches the
work to long-lived worker processes running on HPC compute nodes (e.g.
Jusuf/JUWELS). Those workers manage the actual ``apptainer`` instances.

This makes it a drop-in replacement for cloud backends (Daytona, Modal) on
clusters where the Harbor driver process cannot itself reach the compute nodes
or run containers — only the bridge can. The bridge server, worker, and the
sbatch launch scripts live alongside this module as reference deployment
artifacts; they run on the cluster, not in the Harbor process.

Configuration (via ``environment.kwargs`` in the trial config, i.e. forwarded
as constructor kwargs):

    bridge_url: Base URL of the bridge server (e.g. ``http://10.128.1.1:9910``).
        Falls back to the ``APPTAINER_BRIDGE_URL`` environment variable.
    sif_cache: Path (on the cluster) to the prebuilt ``.sif`` cache directory.
"""

import asyncio
import base64
import json
import logging
import os
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.container_cache import dockerfile_hash_truncated

logger = logging.getLogger(__name__)

# Default time to keep waiting for HPC workers to come back after they go down
# (e.g. a JUWELS job hits its wall-clock limit and SLURM restarts it). Long
# enough to ride out a job restart instead of failing every in-flight trial.
_DEFAULT_WORKERS_DEAD_TIMEOUT_SEC = 1800.0
_WORKERS_DEAD_TIMEOUT_ENV = "BRIDGE_WORKERS_DEAD_TIMEOUT"
_BRIDGE_URL_ENV = "APPTAINER_BRIDGE_URL"


class BridgeOutageError(RuntimeError):
    """Raised when the bridge or its HPC workers are unavailable.

    Distinct from a task failure: it signals an infrastructure outage (the
    bridge is unreachable, or no workers are alive). RL training configs can
    mask this exception so trials that died on infrastructure are excluded
    from gradient computation rather than scored as zero reward.
    """


def _http_post(url: str, data: dict, timeout: float = 60) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _http_get(url: str, timeout: float = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


async def _async_http_post(url: str, data: dict, timeout: float = 60) -> dict:
    """Non-blocking HTTP POST — runs in a thread to keep the event loop free."""
    return await asyncio.to_thread(_http_post, url, data, timeout)


async def _async_http_get(url: str, timeout: float = 60) -> dict:
    """Non-blocking HTTP GET — runs in a thread to keep the event loop free."""
    return await asyncio.to_thread(_http_get, url, timeout)


async def _poll_job(
    bridge_url: str,
    job_id: str,
    timeout_sec: float = 600,
    poll_interval: float = 0.05,
) -> dict:
    """Poll the bridge for a job result with adaptive backoff.

    Polls fast (50ms) for quick operations, backing off gently to 500ms for
    long-running ones. Never long-polls — each request returns immediately.

    Raises:
        BridgeOutageError: After ``max_consecutive_errors`` consecutive poll
            failures (the bridge has gone away).
        RuntimeError: If the bridge reports the job itself errored.
        TimeoutError: If the job does not complete within ``timeout_sec``.
    """
    deadline = time.time() + timeout_sec
    consecutive_errors = 0
    max_consecutive_errors = 20
    current_interval = poll_interval
    max_interval = 0.5
    backoff_factor = 1.3

    while time.time() < deadline:
        try:
            result = await _async_http_get(
                f"{bridge_url}/job/result/{job_id}", timeout=10
            )
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            logger.warning(
                f"[bridge] poll error for job {job_id} "
                f"({consecutive_errors}/{max_consecutive_errors}): {e}"
            )
            if consecutive_errors >= max_consecutive_errors:
                raise BridgeOutageError(
                    f"Bridge unreachable after {consecutive_errors} consecutive "
                    f"failures. Last error: {e}. Failing fast to avoid wasting "
                    "GPU time."
                )
            await asyncio.sleep(min(current_interval * 4, max_interval))
            continue

        state = result.get("state")
        if state == "done":
            return result
        if state == "error":
            raise RuntimeError(f"Bridge job {job_id} failed: {result.get('error')}")
        await asyncio.sleep(current_interval)
        current_interval = min(current_interval * backoff_factor, max_interval)

    raise TimeoutError(f"Bridge job {job_id} timed out after {timeout_sec}s")


class ApptainerEnvironment(BaseEnvironment):
    """Harbor environment backed by remote Apptainer instances via a bridge.

    Lifecycle operations are submitted to the bridge server, which dispatches
    them to workers running on HPC compute nodes. Workers own the actual
    ``apptainer`` instances; this class only speaks HTTP to the bridge.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        bridge_url: str = "",
        sif_cache: str = "",
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )

        self._bridge_url = bridge_url or os.environ.get(_BRIDGE_URL_ENV, "")
        if not self._bridge_url:
            raise ValueError(
                "ApptainerEnvironment requires a bridge_url (via "
                "environment.kwargs.bridge_url or the "
                f"{_BRIDGE_URL_ENV} environment variable)."
            )

        self._sif_cache = sif_cache
        self._env_id: str | None = None
        # Timestamp (monotonic wall clock) when workers were first seen down.
        self._workers_dead_since: float | None = None
        self._workers_dead_timeout = float(
            os.environ.get(_WORKERS_DEAD_TIMEOUT_ENV, _DEFAULT_WORKERS_DEAD_TIMEOUT_SEC)
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.APPTAINER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # CPU-only HPC nodes; the bridge can run isolated (no-internet)
        # containers. Not "mounted": files live on the remote cluster, so the
        # verifier must download results rather than read a host bind mount.
        return EnvironmentCapabilities(gpus=False, disable_internet=True)

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self) -> None:
        """Bridge environments resolve their image on the worker.

        There is no local file to validate beyond the environment directory
        itself; the worker resolves the ``.sif`` (from the cache or by building
        from the Dockerfile) at start time.
        """

    def _check_transfer_result(self, op_name: str, result: dict) -> None:
        return_code = result.get("return_code")
        if return_code is not None and return_code != 0:
            stderr = (result.get("stderr") or "").strip()
            stdout = (result.get("stdout") or "").strip()
            details = stderr or stdout or str(result)
            raise RuntimeError(
                f"Bridge {op_name} failed with return_code={return_code}: {details}"
            )

    def _collect_task_files(self) -> dict[str, str]:
        """Collect the environment directory as a {relpath: base64} mapping."""
        files_b64: dict[str, str] = {}
        env_dir = self.environment_dir
        if not env_dir.exists():
            return files_b64

        for root, _dirs, files in os.walk(env_dir):
            for fname in files:
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(env_dir)
                files_b64[str(rel_path)] = base64.b64encode(fpath.read_bytes()).decode()
        return files_b64

    def _base_image_from_dockerfile(self) -> str:
        if not self._dockerfile_path.exists():
            return ""
        for line in self._dockerfile_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                return stripped[5:].split(" AS ")[0].strip()
        return ""

    def _check_bridge_health(self) -> None:
        """Wait (up to the dead-timeout) for the bridge workers to be alive.

        When HPC workers are between job allocations (e.g. a 24h job restart),
        this rides out the gap rather than failing immediately. Polls the
        bridge ``/status`` endpoint and raises once workers have been down
        longer than ``self._workers_dead_timeout``.

        Raises:
            BridgeOutageError: If the bridge is unreachable, or workers stay
                down past the timeout.
        """
        try:
            status = _http_get(f"{self._bridge_url}/status", timeout=10)
        except Exception as e:
            raise BridgeOutageError(
                f"Bridge server unreachable at {self._bridge_url}: {e}. "
                "Check that the bridge and SSH tunnels are running."
            )

        if status.get("workers_alive"):
            if self._workers_dead_since is not None:
                recovered = time.time() - self._workers_dead_since
                self.logger.info(
                    f"[bridge] Workers back alive after {recovered:.0f}s downtime"
                )
            self._workers_dead_since = None
            return

        now = time.time()
        if self._workers_dead_since is None:
            self._workers_dead_since = now
            self.logger.warning(
                "[bridge] No workers alive — waiting for worker restart "
                f"(timeout: {self._workers_dead_timeout:.0f}s)"
            )

        dead_duration = now - self._workers_dead_since
        if dead_duration >= self._workers_dead_timeout:
            raise BridgeOutageError(
                f"No bridge workers alive for {dead_duration:.0f}s "
                f"(timeout {self._workers_dead_timeout:.0f}s). "
                "HPC workers may have failed to restart."
            )
        self.logger.warning(
            f"[bridge] Waiting for workers... {dead_duration:.0f}s / "
            f"{self._workers_dead_timeout:.0f}s"
        )

    async def start(self, force_build: bool) -> None:
        """Create a remote Apptainer instance via the bridge.

        ``force_build`` is forwarded to the worker, which decides whether to
        rebuild the ``.sif`` or reuse a cached one.
        """
        self._check_bridge_health()

        df_hash = ""
        if self._dockerfile_path.exists():
            df_hash = dockerfile_hash_truncated(self._dockerfile_path)

        sif_path = ""
        if self._sif_cache and df_hash:
            sif_path = f"{self._sif_cache}/build_{self.environment_name}-{df_hash}.sif"

        resp = await _async_http_post(
            f"{self._bridge_url}/env/create",
            {
                "task_name": self.environment_name,
                "dockerfile_hash": df_hash,
                "sif_path": sif_path,
                "base_image": self._base_image_from_dockerfile(),
                "environment_dir": str(self.environment_dir),
                "force_build": force_build,
                "task_env_config": {
                    "memory_mb": self.task_env_config.memory_mb,
                    "cpus": self.task_env_config.cpus,
                    "allow_internet": self.task_env_config.allow_internet,
                },
                "files_b64": self._collect_task_files(),
            },
        )

        self._env_id = resp["env_id"]
        job_id = resp["job_id"]
        self.logger.info(
            f"[bridge] Creating env {self._env_id} for task {self.environment_name}"
        )

        result = await _poll_job(self._bridge_url, job_id, timeout_sec=1200)
        self.logger.info(f"[bridge] Env {self._env_id} started: {result}")
        if isinstance(result, dict) and result.get("fakeroot") is False:
            self.logger.warning(
                f"[bridge] Env {self._env_id} ({self.environment_name}) is NOT "
                "running as (fake) root: the worker's fakeroot start failed for "
                "this image and it fell back to a plain start. Anything the task "
                "needs root for at task time (apt-get, /root, ...) will fail; a "
                "zero reward may reflect this environment mismatch. Set "
                "BRIDGE_USE_FAKEROOT=1 on the worker to turn this into an error."
            )

    async def stop(self, delete: bool) -> None:
        """Tear down the remote Apptainer instance."""
        if not self._env_id:
            return
        try:
            resp = await _async_http_post(
                f"{self._bridge_url}/env/stop",
                {"env_id": self._env_id, "delete": delete},
            )
            job_id = resp.get("job_id")
            if job_id:
                await _poll_job(self._bridge_url, job_id, timeout_sec=60)
        except Exception as e:
            self.logger.warning(f"[bridge] Error stopping env {self._env_id}: {e}")
        finally:
            self._env_id = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a command in the remote Apptainer instance via the bridge."""
        if not self._env_id:
            raise RuntimeError("Environment not started")

        resolved_user = self._resolve_user(user)
        merged_env = self._merge_env(env)
        effective_timeout = timeout_sec or 600

        resp = await _async_http_post(
            f"{self._bridge_url}/env/exec",
            {
                "env_id": self._env_id,
                "command": command,
                "cwd": cwd,
                "env_vars": merged_env or {},
                "user": resolved_user,
                "timeout_sec": effective_timeout,
            },
        )

        job_id = resp.get("job_id")
        if not job_id:
            raise RuntimeError(f"Exec submit failed: {resp}")

        result = await _poll_job(
            self._bridge_url, job_id, timeout_sec=effective_timeout + 30
        )
        stdout_truncated = bool(result.get("stdout_truncated"))
        stderr_truncated = bool(result.get("stderr_truncated"))
        if stdout_truncated or stderr_truncated:
            self.logger.warning(
                "Apptainer exec output retained only the most recent worker limit "
                "(stdout_truncated=%s, stderr_truncated=%s)",
                stdout_truncated,
                stderr_truncated,
            )
        return ExecResult(
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
            return_code=result.get("return_code", -1),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a single file to the remote instance (base64 over HTTP)."""
        content_b64 = base64.b64encode(Path(source_path).read_bytes()).decode()
        resp = await _async_http_post(
            f"{self._bridge_url}/env/upload",
            {
                "env_id": self._env_id,
                "file_b64": content_b64,
                "target_path": target_path,
                "is_dir": False,
            },
        )
        job_id = resp.get("job_id")
        if job_id:
            result = await _poll_job(self._bridge_url, job_id, timeout_sec=120)
            self._check_transfer_result("upload_file", result)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a directory to the remote instance as a tar (base64 over HTTP)."""
        source = Path(source_dir)
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
            tar_path = tf.name
        try:
            with tarfile.open(tar_path, "w", dereference=True) as tar:
                tar.add(str(source), arcname=".")
            content_b64 = base64.b64encode(Path(tar_path).read_bytes()).decode()
            resp = await _async_http_post(
                f"{self._bridge_url}/env/upload",
                {
                    "env_id": self._env_id,
                    "file_b64": content_b64,
                    "target_path": target_dir,
                    "is_dir": True,
                },
            )
            job_id = resp.get("job_id")
            if job_id:
                result = await _poll_job(self._bridge_url, job_id, timeout_sec=120)
                self._check_transfer_result("upload_dir", result)
        finally:
            os.unlink(tar_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a single file from the remote instance."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        resp = await _async_http_post(
            f"{self._bridge_url}/env/download",
            {"env_id": self._env_id, "source_path": source_path, "is_dir": False},
        )
        job_id = resp.get("job_id")
        if not job_id:
            raise RuntimeError(f"Download submit failed: {resp}")

        result = await _poll_job(self._bridge_url, job_id, timeout_sec=120)
        file_b64 = result.get("file_b64", "")
        if not file_b64:
            raise FileNotFoundError(f"Download returned no data for {source_path}")
        target.write_bytes(base64.b64decode(file_b64))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the remote instance (tar over HTTP)."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        resp = await _async_http_post(
            f"{self._bridge_url}/env/download",
            {"env_id": self._env_id, "source_path": source_dir, "is_dir": True},
        )
        job_id = resp.get("job_id")
        if not job_id:
            raise RuntimeError(f"Download submit failed: {resp}")

        result = await _poll_job(self._bridge_url, job_id, timeout_sec=120)
        file_b64 = result.get("file_b64", "")
        if not file_b64:
            raise FileNotFoundError(f"Download returned no data for {source_dir}")

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
            tar_path = tf.name
        try:
            Path(tar_path).write_bytes(base64.b64decode(file_b64))
            with tarfile.open(tar_path) as tar:
                tar.extractall(str(target))
        finally:
            os.unlink(tar_path)

    async def attach(self) -> None:
        raise NotImplementedError("Cannot attach to a remote bridge environment.")
