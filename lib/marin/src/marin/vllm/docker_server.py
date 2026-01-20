# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

DEFAULT_VLLM_DOCKER_IMAGE: str = "vllm/vllm-tpu:nightly-20260104-4a1e25b-0d4044e"

_SENSITIVE_ENV_KEYS = frozenset(
    {
        "HF_TOKEN",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
    }
)

_VLLM_SIDECAR_LABEL_KEY = "marin.vllm_sidecar"
_VLLM_SIDECAR_LABEL_VALUE = "1"


@dataclass(frozen=True)
class VllmDockerServerConfig:
    """Configuration for launching `vllm serve` in a sibling Docker container."""

    image: str
    model_name_or_path: str
    port: int | None = None
    host: str = "127.0.0.1"
    container_name: str | None = None

    env: dict[str, str] = field(default_factory=dict)
    volumes: list[tuple[str, str]] = field(default_factory=list)

    extra_vllm_args: list[str] = field(default_factory=list)
    docker_run_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VllmDockerServerHandle:
    """A handle for a running vLLM Docker sidecar."""

    container_name: str
    host: str
    port: int
    image: str

    @property
    def server_url(self) -> str:
        """OpenAI-compatible base URL (includes `/v1`)."""

        return f"http://{self.host}:{self.port}/v1"


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def build_docker_run_command(config: VllmDockerServerConfig, *, port: int, container_name: str) -> list[str]:
    """Build `docker run ... vllm serve ...` as argv.

    This is split out for unit testing without invoking Docker.
    """

    # Avoid `--rm` so that if the container exits immediately we can still collect logs/inspect output.
    # We explicitly clean up with `docker rm -f` in `stop_vllm_docker_server_by_name`.
    cmd: list[str] = ["docker", "run", "-d", "--net=host", "--name", container_name]
    cmd.extend(["--label", f"{_VLLM_SIDECAR_LABEL_KEY}={_VLLM_SIDECAR_LABEL_VALUE}"])
    cmd.extend(["--label", f"marin.vllm_port={port}"])

    cmd.extend(config.docker_run_args)

    # Volumes
    for src, dst in config.volumes:
        cmd.extend(["-v", f"{src}:{dst}"])

    # Env
    for key, value in sorted(config.env.items()):
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(config.image)
    cmd.extend(
        [
            "vllm",
            "serve",
            config.model_name_or_path,
            "--host",
            config.host,
            "--port",
            str(port),
            "--trust-remote-code",
        ]
    )
    cmd.extend(config.extra_vllm_args)
    return cmd


def _require_docker_available() -> None:
    if not os.path.exists("/var/run/docker.sock"):
        raise RuntimeError(
            "Docker socket not available at /var/run/docker.sock. "
            "This job requires docker-alongside-docker (mount the socket into the Ray container)."
        )
    if shutil.which("docker") is None:
        raise RuntimeError(
            "`docker` CLI not found on PATH. Install the Docker client in the Ray image to run vLLM as a sidecar."
        )


def _docker_container_status(container_name: str) -> str | None:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _docker_logs_tail(container_name: str, *, max_lines: int = 200) -> str:
    result = subprocess.run(
        ["docker", "logs", "--tail", str(max_lines), container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return f"<failed to read docker logs for {container_name}: {stderr}>"
    return result.stdout


def _docker_inspect(container_name: str) -> str:
    result = subprocess.run(
        ["docker", "inspect", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return f"<failed to inspect container {container_name}: {stderr}>"
    return result.stdout


def _redact_docker_run_command(cmd: list[str]) -> str:
    redacted = list(cmd)
    i = 0
    while i < len(redacted):
        if redacted[i] == "-e" and i + 1 < len(redacted):
            kv = redacted[i + 1]
            if "=" in kv:
                key, value = kv.split("=", 1)
                if key in _SENSITIVE_ENV_KEYS and value:
                    redacted[i + 1] = f"{key}=<redacted>"
        i += 1
    return shlex.join(redacted)


def _disk_usage_line(path: str) -> str | None:
    try:
        usage = shutil.disk_usage(path)
    except FileNotFoundError:
        return None
    except OSError:
        return f"{path}: <failed to stat>"

    total_gib = usage.total / (1024**3)
    used_gib = usage.used / (1024**3)
    free_gib = usage.free / (1024**3)
    used_pct = (usage.used / usage.total * 100.0) if usage.total else 0.0
    return f"{path}: total={total_gib:.1f}GiB used={used_gib:.1f}GiB free={free_gib:.1f}GiB used_pct={used_pct:.1f}%"


def _disk_diagnostics() -> str:
    tmpdir = os.environ.get("TMPDIR") or "/tmp"
    paths = ["/", tmpdir, "/tmp/ray", "/dev/shm", "/var/lib/docker"]
    session_latest = os.path.join("/tmp/ray", "session_latest")
    if os.path.exists(session_latest):
        paths.append(session_latest)
        real_session = os.path.realpath(session_latest)
        if real_session != session_latest:
            paths.append(real_session)

    lines = ["Disk diagnostics:"]
    lines.append(f"  TMPDIR={tmpdir!r} RAY_TMPDIR={os.environ.get('RAY_TMPDIR')!r}")
    for path in paths:
        line = _disk_usage_line(path)
        if line is not None:
            lines.append(f"  {line}")

    docker_root: str | None = None
    try:
        result = subprocess.run(
            ["docker", "info", "-f", "{{.DockerRootDir}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            docker_root = (result.stdout or "").strip() or None
    except OSError:
        docker_root = None
    except subprocess.TimeoutExpired:
        docker_root = None

    if docker_root:
        line = _disk_usage_line(docker_root)
        if line is not None:
            lines.append(f"  docker_root={docker_root!r} {line}")

    return "\n".join(lines)


def _docker_image_present(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _ensure_docker_image_present(image: str) -> None:
    if _docker_image_present(image):
        return
    print(f"vLLM sidecar: Docker image {image!r} not present locally; pulling...")
    subprocess.run(["docker", "pull", image], check=True)


def _docker_list_sidecar_containers() -> list[str]:
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label={_VLLM_SIDECAR_LABEL_KEY}={_VLLM_SIDECAR_LABEL_VALUE}",
            "--format",
            "{{.Names}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _docker_list_marin_vllm_containers_by_prefix() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip().startswith("marin-vllm-")]


def _cleanup_stale_sidecar_containers(*, kill_running: bool) -> list[str]:
    # Best effort cleanup; never fails the job.
    sidecars = sorted(set(_docker_list_sidecar_containers() + _docker_list_marin_vllm_containers_by_prefix()))
    if not sidecars:
        return []

    removed: list[str] = []
    kept_running: list[str] = []
    for name in sidecars:
        status = _docker_container_status(name)
        if status is None:
            continue
        if status == "running" and not kill_running:
            kept_running.append(name)
            continue
        if status in {"running", "exited", "dead", "created", "removing"}:
            stop_vllm_docker_server_by_name(name)
            removed.append(name)

    if removed:
        print(f"vLLM sidecar: cleaned up {len(removed)} prior sidecar container(s): {', '.join(removed)}")
    return kept_running


def _guess_sidecar_port(container_name: str) -> int | None:
    # Back-compat for older names like `marin-vllm-<uuid>-<port>`
    maybe = container_name.rsplit("-", 1)[-1]
    if maybe.isdigit():
        return int(maybe)
    return None


def _is_sidecar_ready(port: int) -> bool:
    try:
        response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_vllm_docker_server(
    config: VllmDockerServerConfig,
    *,
    timeout_seconds: int = 3600,
    poll_interval_seconds: int = 5,
) -> VllmDockerServerHandle:
    """Start a vLLM server in a sibling Docker container and wait until it is ready.

    Raises RuntimeError/TimeoutError with helpful Docker logs on failure.
    """

    _require_docker_available()

    # Use print so this always shows up in Ray logs even if Python logging isn't configured.
    print(_disk_diagnostics())

    # Fray is expected to schedule TPU jobs with exclusive access per TPU VM. If we find a
    # running prior sidecar, it's most likely a leaked container from a crashed worker.
    # Proactively removing it helps avoid TPU vfio "device busy" errors on subsequent runs.
    _cleanup_stale_sidecar_containers(kill_running=True)

    _ensure_docker_image_present(config.image)

    port = config.port if config.port is not None else _pick_free_port(config.host)
    container_name = config.container_name or f"marin-vllm-{uuid.uuid4().hex[:10]}-{port}"

    cmd = build_docker_run_command(config, port=port, container_name=container_name)
    print(f"vLLM sidecar: starting container {container_name!r}")
    print(f"vLLM sidecar: docker command: {_redact_docker_run_command(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Best-effort cleanup in case `docker run` partially created the container.
        stop_vllm_docker_server_by_name(container_name)
        raise RuntimeError(
            "Failed to start vLLM Docker sidecar.\n"
            f"Image: {config.image}\n"
            f"Command: {_redact_docker_run_command(cmd)}\n"
            f"Exit code: {e.returncode}\n"
            f"--- stdout ---\n{(e.stdout or '').strip()}\n"
            f"--- stderr ---\n{(e.stderr or '').strip()}\n"
            f"--- diagnostics ---\n{_disk_diagnostics()}"
        ) from e

    handle = VllmDockerServerHandle(container_name=container_name, host=config.host, port=port, image=config.image)

    server_models_url = f"{handle.server_url}/models"
    start_time = time.time()
    elapsed_time = 0.0
    last_progress_log_time = 0.0

    try:
        while True:
            now = time.time()
            status = _docker_container_status(container_name)
            if status is None or status == "exited" or status == "dead":
                logs = _docker_logs_tail(container_name)
                inspect = _docker_inspect(container_name)
                stop_vllm_docker_server_by_name(container_name)
                raise RuntimeError(
                    "vLLM Docker sidecar exited before becoming ready.\n"
                    f"Container: {container_name}\n"
                    f"Image: {config.image}\n"
                    f"Command: {_redact_docker_run_command(cmd)}\n"
                    f"Status: {status}\n"
                    f"--- diagnostics ---\n{_disk_diagnostics()}\n"
                    f"--- docker logs (tail) ---\n{logs}\n"
                    f"--- docker inspect ---\n{inspect[:8000]}"
                )

            elapsed_time = now - start_time
            if elapsed_time - last_progress_log_time >= 30:
                # Periodic progress log so we don't look "hung" while pulling/initializing/compiling.
                print("vLLM sidecar: waiting for readiness:")
                print(f"  status={status!r} elapsed={elapsed_time:.1f}s url={server_models_url}")
                last_progress_log_time = elapsed_time

            try:
                response = requests.get(server_models_url, timeout=5)
                if response.status_code == 200:
                    return handle
            except requests.ConnectionError:
                # Server not ready yet.
                pass
            except requests.Timeout:
                # Server not ready yet.
                pass

            if elapsed_time > timeout_seconds:
                logs = _docker_logs_tail(container_name)
                stop_vllm_docker_server_by_name(container_name)
                raise TimeoutError(
                    "Failed to start vLLM Docker sidecar within timeout period.\n"
                    f"Container: {container_name}\n"
                    f"Image: {config.image}\n"
                    f"Endpoint: {server_models_url}\n"
                    f"Elapsed seconds: {elapsed_time:.1f}\n"
                    f"--- diagnostics ---\n{_disk_diagnostics()}\n"
                    f"--- docker logs (tail) ---\n{logs}"
                )

            time.sleep(poll_interval_seconds)
    except BaseException:
        # If anything unexpected happens during polling, ensure we don't leak a container that might
        # hold TPU devices. This is especially important when running under Ray.
        stop_vllm_docker_server_by_name(container_name)
        raise


def stop_vllm_docker_server(handle: VllmDockerServerHandle) -> None:
    """Stop (force remove) the Docker sidecar container."""

    stop_vllm_docker_server_by_name(handle.container_name)


def stop_vllm_docker_server_by_name(container_name: str) -> None:
    """Stop (force remove) the Docker sidecar container by name."""

    try:
        subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        # Best effort; we'll try again below.
        pass

    deadline = time.time() + 30
    while time.time() < deadline:
        if _docker_container_status(container_name) is None:
            return
        subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True, text=True)
        time.sleep(1)

    status = _docker_container_status(container_name)
    if status is not None:
        print(
            f"vLLM sidecar: failed to fully remove container {container_name!r} (status={status!r}); TPU may remain busy"
        )
