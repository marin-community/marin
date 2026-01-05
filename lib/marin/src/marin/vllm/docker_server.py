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

import os
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field

import requests

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

    port = config.port if config.port is not None else _pick_free_port(config.host)
    container_name = config.container_name or f"marin-vllm-{uuid.uuid4().hex[:10]}-{port}"

    cmd = build_docker_run_command(config, port=port, container_name=container_name)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    handle = VllmDockerServerHandle(container_name=container_name, host=config.host, port=port, image=config.image)

    server_models_url = f"{handle.server_url}/models"
    start_time = time.time()
    elapsed_time = 0.0

    while True:
        status = _docker_container_status(container_name)
        if status is None or status == "exited" or status == "dead":
            logs = _docker_logs_tail(container_name)
            inspect = _docker_inspect(container_name)
            raise RuntimeError(
                "vLLM Docker sidecar exited before becoming ready.\n"
                f"Container: {container_name}\n"
                f"Image: {config.image}\n"
                f"Command: {_redact_docker_run_command(cmd)}\n"
                f"Status: {status}\n"
                f"--- docker logs (tail) ---\n{logs}\n"
                f"--- docker inspect ---\n{inspect[:8000]}"
            )

        try:
            response = requests.get(server_models_url, timeout=5)
            if response.status_code == 200:
                return handle
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            logs = _docker_logs_tail(container_name)
            stop_vllm_docker_server_by_name(container_name)
            raise TimeoutError(
                "Failed to start vLLM Docker sidecar within timeout period.\n"
                f"Container: {container_name}\n"
                f"Image: {config.image}\n"
                f"Endpoint: {server_models_url}\n"
                f"Elapsed seconds: {elapsed_time:.1f}\n"
                f"--- docker logs (tail) ---\n{logs}"
            )

        time.sleep(poll_interval_seconds)


def stop_vllm_docker_server(handle: VllmDockerServerHandle) -> None:
    """Stop (force remove) the Docker sidecar container."""

    subprocess.run(["docker", "rm", "-f", handle.container_name], check=False, capture_output=True, text=True)


def stop_vllm_docker_server_by_name(container_name: str) -> None:
    """Stop (force remove) the Docker sidecar container by name."""

    subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True, text=True)
