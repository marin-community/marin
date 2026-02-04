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

import dataclasses
import glob
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import requests

from marin.evaluation.evaluators.evaluator import ModelConfig

logger = logging.getLogger(__name__)
DEFAULT_VLLM_TPU_DOCKER_IMAGE: str = "vllm/vllm-tpu:nightly-20260104-4a1e25b-0d4044e"
DEFAULT_VLLM_GPU_DOCKER_IMAGE: str = "nvcr.io/nvidia/vllm:25.12.post1-py3"
VLLM_NATIVE_PIP_PACKAGES: tuple[str, ...] = ("vllm-tpu",)

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
class VllmServerHandle:
    """A handle for a running vLLM server (native or Docker sidecar)."""

    server_url: str
    port: int
    process: subprocess.Popen[str] | None = None
    log_dir: str | None = None
    docker_container_name: str | None = None
    docker_image: str | None = None
    docker_run_cmd: str | None = None

    @property
    def mode(self) -> Literal["native", "docker"]:
        if self.docker_container_name:
            return "docker"
        if self.process is not None or self.log_dir is not None:
            return "native"
        raise RuntimeError("Unable to infer vLLM server mode from handle state.")


def resolve_model_name_or_path(model: ModelConfig) -> tuple[str, ModelConfig]:
    """Resolve the `model` argument to pass to vLLM."""
    model = _maybe_enable_streaming(model)
    model_name_or_path = model.path if model.path is not None else model.name
    return model_name_or_path, model


def _tail_file(path: str, max_lines: int) -> str:
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception as exc:
        return f"<failed to read {path}: {exc}>"


def _native_logs_tail(log_dir: str | None, *, max_lines: int = 200) -> str:
    if not log_dir:
        return "<no log directory available for native vLLM server>"
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    return (
        "--- stdout (tail) ---\n"
        f"{_tail_file(stdout_path, max_lines)}\n"
        "--- stderr (tail) ---\n"
        f"{_tail_file(stderr_path, max_lines)}"
    )


def _docker_container_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    output = result.stdout.strip().lower()
    if output == "true":
        return True
    if output == "false":
        return False
    raise RuntimeError(f"Unexpected output from docker inspect: {output!r}")


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
        return f"failed to inspect container {container_name}: {stderr}"
    return result.stdout


class VllmServerBackend(ABC):
    @abstractmethod
    def start(
        self,
        *,
        model_name_or_path: str,
        host: str,
        port: int | None,
        timeout_seconds: int,
        extra_cli_args: list[str] | None,
    ) -> VllmServerHandle:
        raise NotImplementedError

    @abstractmethod
    def logs_tail(self, handle: VllmServerHandle, *, max_lines: int = 200) -> str:
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self, handle: VllmServerHandle, *, max_lines: int = 200) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def stop(self, handle: VllmServerHandle) -> None:
        raise NotImplementedError


class DockerVllmServerBackend(VllmServerBackend):
    def __init__(self, docker_image: str | None, docker_run_args: list[str] | None) -> None:
        self._docker_image = docker_image
        self._docker_run_args = docker_run_args

    def start(
        self,
        *,
        model_name_or_path: str,
        host: str,
        port: int | None,
        timeout_seconds: int,
        extra_cli_args: list[str] | None,
    ) -> VllmServerHandle:
        return _start_vllm_docker_server(
            model_name_or_path=model_name_or_path,
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            extra_cli_args=extra_cli_args,
            docker_image=self._docker_image,
            docker_run_args=self._docker_run_args,
        )

    def logs_tail(self, handle: VllmServerHandle, *, max_lines: int = 200) -> str:
        return _docker_logs_tail(handle.docker_container_name, max_lines=max_lines)

    def diagnostics(self, handle: VllmServerHandle, *, max_lines: int = 200) -> dict[str, str]:
        diagnostics: dict[str, str] = {}
        if handle.docker_run_cmd:
            diagnostics["vLLM Docker run command (redacted)"] = handle.docker_run_cmd
        diagnostics["vLLM Docker logs (tail)"] = self.logs_tail(handle, max_lines=max_lines)
        diagnostics["vLLM Docker inspect"] = _docker_inspect(handle.docker_container_name)
        return diagnostics

    def stop(self, handle: VllmServerHandle) -> None:
        container_name = handle.docker_container_name
        subprocess.run(["docker", "rm", "-f", container_name], check=False, capture_output=True, text=True)


class NativeVllmServerBackend(VllmServerBackend):
    def start(
        self,
        *,
        model_name_or_path: str,
        host: str,
        port: int | None,
        timeout_seconds: int,
        extra_cli_args: list[str] | None,
    ) -> VllmServerHandle:
        return _start_vllm_native_server(
            model_name_or_path=model_name_or_path,
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            extra_cli_args=extra_cli_args,
        )

    def logs_tail(self, handle: VllmServerHandle, *, max_lines: int = 200) -> str:
        return _native_logs_tail(handle.log_dir, max_lines=max_lines)

    def diagnostics(self, handle: VllmServerHandle, *, max_lines: int = 200) -> dict[str, str]:
        diagnostics: dict[str, str] = {}
        if handle.log_dir:
            diagnostics["vLLM native log dir"] = handle.log_dir
        diagnostics["vLLM native logs (tail)"] = self.logs_tail(handle, max_lines=max_lines)
        return diagnostics

    def stop(self, handle: VllmServerHandle) -> None:
        if handle.process is not None:
            handle.process.kill()


def resolve_vllm_mode(mode: Literal["native", "docker"] | None) -> Literal["native", "docker"]:
    mode_str = (mode if mode is not None else os.environ.get("MARIN_VLLM_MODE", "docker")).lower()
    if mode_str not in ("native", "docker"):
        raise ValueError(f"Unknown MARIN_VLLM_MODE={mode_str!r}; expected 'native' or 'docker'.")
    return mode_str  # type: ignore[return-value]


def _resolve_vllm_backend(
    mode: Literal["native", "docker"],
    *,
    docker_image: str | None,
    docker_run_args: list[str] | None,
) -> VllmServerBackend:
    if mode == "docker":
        return DockerVllmServerBackend(docker_image, docker_run_args)
    if mode == "native":
        return NativeVllmServerBackend()
    raise ValueError(f"Unknown vLLM mode {mode!r}; expected 'native' or 'docker'.")


def _is_object_store_path(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"gs", "s3"}


def _maybe_enable_streaming(model: ModelConfig) -> ModelConfig:
    if model.path is None:
        return model
    if not _is_object_store_path(model.path):
        return model
    if "load_format" in model.engine_kwargs:
        return model

    engine_kwargs = dict(model.engine_kwargs)
    # Default to the non-sharded streamer for maximum compatibility.
    # `runai_streamer_sharded` only works for checkpoints that are already sharded
    # into `model-rank-*-part-*.safetensors`.
    engine_kwargs["load_format"] = "runai_streamer"
    return dataclasses.replace(model, engine_kwargs=engine_kwargs)


def _engine_kwargs_to_cli_args(engine_kwargs: dict) -> list[str]:
    args: list[str] = []
    load_format = engine_kwargs.get("load_format")
    if load_format is not None:
        args.extend(["--load-format", load_format])
    max_model_len = engine_kwargs.get("max_model_len")
    if max_model_len is not None:
        args.extend(["--max-model-len", str(max_model_len)])
    gpu_memory_utilization = engine_kwargs.get("gpu_memory_utilization")
    if gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    return args


def _get_first_model_id(server_url: str) -> str:
    response = requests.get(f"{server_url}/models", timeout=30)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"No models returned from {server_url}/models: {str(payload)[:2000]}")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"Missing model id in {server_url}/models response: {str(payload)[:2000]}")
    return str(model_id)


class VllmEnvironment:
    """Manage vLLM server lifecycle and lm-eval configuration."""

    def __init__(
        self,
        model: ModelConfig,
        *,
        mode: Literal["native", "docker"] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_seconds: int = 3600,
        docker_image: str | None = None,
        docker_run_args: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.model_name_or_path, self.model = resolve_model_name_or_path(model)
        self.mode = resolve_vllm_mode(mode)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.docker_run_args = docker_run_args
        self.extra_cli_args = [*_engine_kwargs_to_cli_args(self.model.engine_kwargs), *(extra_args or [])]
        self._backend = _resolve_vllm_backend(
            self.mode,
            docker_image=self.docker_image,
            docker_run_args=self.docker_run_args,
        )

        self.vllm_server: VllmServerHandle | None = None
        self.model_id: str | None = None

    def __enter__(self) -> "VllmEnvironment":
        if self.vllm_server is None:
            logger.info(
                "Starting vLLM environment",
                extra={
                    "mode": self.mode,
                    "model_name_or_path": self.model_name_or_path,
                    "host": self.host,
                    "port": self.port,
                    "docker_image": self.docker_image,
                    "docker_run_args": self.docker_run_args,
                },
            )
            try:
                self.vllm_server = self._backend.start(
                    model_name_or_path=self.model_name_or_path,
                    host=self.host,
                    port=self.port,
                    timeout_seconds=self.timeout_seconds,
                    extra_cli_args=self.extra_cli_args,
                )
                self.model_id = _get_first_model_id(self.vllm_server.server_url)
                logger.info(
                    "vLLM environment ready",
                    extra={
                        "server_url": self.vllm_server.server_url,
                        "container": self.vllm_server.docker_container_name,
                        "model_id": self.model_id,
                    },
                )
            except Exception:
                logger.exception("Failed to start vLLM environment", extra=self.debug_snapshot())
                if self.vllm_server is not None:
                    try:
                        diagnostics = self._backend.diagnostics(self.vllm_server)
                        for label, value in diagnostics.items():
                            logger.error("%s:\n%s", label, value)
                    except Exception:
                        logger.exception("Failed to collect vLLM diagnostics")
                raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.vllm_server is not None:
            self._backend.stop(self.vllm_server)
            self.vllm_server = None

    @property
    def server_url(self) -> str:
        if self.vllm_server is None:
            raise RuntimeError("vLLM server is not running in this environment.")
        return self.vllm_server.server_url

    def debug_snapshot(self) -> dict[str, str | int | None]:
        return {
            "mode": self.mode,
            "model_name_or_path": self.model_name_or_path,
            "host": self.host,
            "port": self.port,
            "server_url": self.vllm_server.server_url if self.vllm_server else None,
            "docker_container_name": self.vllm_server.docker_container_name if self.vllm_server else None,
            "docker_image": self.vllm_server.docker_image if self.vllm_server else self.docker_image,
            "docker_run_cmd": self.vllm_server.docker_run_cmd if self.vllm_server else None,
        }

    def logs_tail(self, *, max_lines: int = 200) -> str:
        if self.vllm_server is None:
            raise RuntimeError("vLLM server is not running in this environment.")
        return self._backend.logs_tail(self.vllm_server, max_lines=max_lines)

    def diagnostics(self, *, max_lines: int = 200) -> dict[str, str]:
        if self.vllm_server is None:
            return {}
        return self._backend.diagnostics(self.vllm_server, max_lines=max_lines)


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _detect_tpu_environment() -> bool:
    """Detects whether the TPU environment variable TPU_NAME is set and non-empty."""
    return bool(os.environ.get("TPU_NAME"))


def _detect_nvidia_gpu_environment() -> bool:
    for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(key)
        if not value:
            continue
        if value:
            return True
    return bool(glob.glob("/dev/nvidia[0-9]*"))


def _detect_resource_type() -> Literal["tpu", "gpu", "unknown"]:
    if _detect_tpu_environment():
        return "tpu"
    if _detect_nvidia_gpu_environment():
        return "gpu"
    return "unknown"


def _resolve_docker_gpu_arg() -> str | None:
    raw_value = os.environ.get("NVIDIA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is not None:
        value = raw_value.strip()
        if not value:
            return None
        lowered = value.lower()
        if lowered == "all":
            return "all"
        devices = [device.strip() for device in value.split(",") if device.strip()]
        if devices:
            return f"device={','.join(devices)}"
        return None

    if _detect_nvidia_gpu_environment():
        return "all"
    return None


def _docker_run_args_for_resource(
    *,
    resource_type: Literal["tpu", "gpu", "unknown"],
    docker_run_args: list[str] | None,
) -> list[str]:
    docker_args: list[str] = []
    existing_args = docker_run_args or []

    has_privileged = any(arg == "--privileged" or arg.startswith("--privileged=") for arg in existing_args)
    has_gpus = any(arg == "--gpus" or arg.startswith("--gpus=") for arg in existing_args)

    if resource_type == "tpu" and not has_privileged:
        docker_args.append("--privileged")
    if resource_type == "gpu" and not has_gpus:
        gpu_arg = _resolve_docker_gpu_arg()
        if gpu_arg is not None:
            docker_args.extend(["--gpus", gpu_arg])

    if not any(arg.startswith("--shm-size") for arg in existing_args):
        docker_args.append("--shm-size=100gb")
    docker_args.extend(existing_args)
    return docker_args


def _build_docker_run_command(
    *,
    image: str,
    model_name_or_path: str,
    host: str,
    port: int,
    container_name: str,
    env: dict[str, str],
    volumes: list[tuple[str, str]],
    extra_vllm_args: list[str],
    docker_run_args: list[str],
) -> list[str]:
    """Build `docker run ... vllm serve ...` as argv.

    This is split out for unit testing without invoking Docker.
    """

    # Avoid `--rm` so that if the container exits immediately we can still collect logs/inspect output.
    # We explicitly clean up with `docker rm -f` via `VllmServerHandle.stop()`.
    cmd: list[str] = ["docker", "run", "-d", "--net=host", "--name", container_name]

    cmd.extend(docker_run_args)

    # Volumes
    for src, dst in volumes:
        cmd.extend(["-v", f"{src}:{dst}"])

    # Env
    for key, value in sorted(env.items()):
        cmd.extend(["-e", f"{key}={value}"])

    cmd.append(image)
    cmd.extend(
        [
            "vllm",
            "serve",
            model_name_or_path,
            "--host",
            host,
            "--port",
            str(port),
            "--trust-remote-code",
        ]
    )
    cmd.extend(extra_vllm_args)
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


def _start_vllm_docker_server(
    *,
    model_name_or_path: str,
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_seconds: int = 3600,
    poll_interval_seconds: int = 5,
    extra_cli_args: list[str] | None = None,
    docker_image: str | None = None,
    docker_run_args: list[str] | None = None,
    container_name: str | None = None,
) -> VllmServerHandle:
    """Start a vLLM server in a sibling Docker container and wait until it is ready.

    Raises RuntimeError/TimeoutError with helpful Docker logs on failure.
    """

    _require_docker_available()

    resource_type = _detect_resource_type()
    if docker_image is None:
        if resource_type == "tpu":
            docker_image = DEFAULT_VLLM_TPU_DOCKER_IMAGE
        elif resource_type == "gpu":
            docker_image = DEFAULT_VLLM_GPU_DOCKER_IMAGE
        else:
            raise RuntimeError(
                f"Cannot determine default docker image for unknown resource type {resource_type!r}. "
                "Please explicitly specify docker_image parameter."
            )
        logger.info(f"No docker_image specified; defaulting to {docker_image} for {resource_type}.")

    env: dict[str, str] = _vllm_jax_env()
    explain_cache_misses = os.environ.get("JAX_EXPLAIN_CACHE_MISSES")
    if explain_cache_misses is not None:
        env["JAX_EXPLAIN_CACHE_MISSES"] = explain_cache_misses
    env["TPU_MIN_LOG_LEVEL"] = os.environ.get("TPU_MIN_LOG_LEVEL", "3")
    env["TPU_STDERR_LOG_LEVEL"] = os.environ.get("TPU_STDERR_LOG_LEVEL", "3")
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(key)
        if value:
            env[key] = value

    docker_args = _docker_run_args_for_resource(
        resource_type=resource_type,
        docker_run_args=docker_run_args,
    )

    if resource_type == "tpu":
        logger.info(
            "Starting vLLM Docker sidecar with "
            f"TPU_MIN_LOG_LEVEL={env.get('TPU_MIN_LOG_LEVEL')} "
            f"TPU_STDERR_LOG_LEVEL={env.get('TPU_STDERR_LOG_LEVEL')}"
        )
    else:
        logger.info(f"Starting vLLM Docker sidecar for {resource_type} resources.")

    resolved_port = port if port is not None else _pick_free_port(host)
    resolved_name = container_name or f"marin-vllm-{uuid.uuid4().hex[:10]}-{resolved_port}"

    cmd = _build_docker_run_command(
        image=docker_image,
        model_name_or_path=model_name_or_path,
        host=host,
        port=resolved_port,
        container_name=resolved_name,
        env=env,
        volumes=[("/tmp", "/tmp")],
        extra_vllm_args=list(extra_cli_args or []),
        docker_run_args=docker_args,
    )
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            "Failed to start vLLM Docker sidecar.\n"
            f"Image: {docker_image}\n"
            f"Command: {_redact_docker_run_command(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )

    server_url = f"http://{host}:{resolved_port}/v1"
    handle = VllmServerHandle(
        server_url=server_url,
        port=resolved_port,
        docker_container_name=resolved_name,
        docker_image=docker_image,
        docker_run_cmd=_redact_docker_run_command(cmd),
    )

    server_models_url = f"{handle.server_url}/models"
    start_time = time.time()

    try:
        while True:
            running = _docker_container_running(resolved_name)
            if not running:
                logs = _docker_logs_tail(resolved_name)
                inspect = _docker_inspect(resolved_name)
                raise RuntimeError(
                    "vLLM Docker sidecar exited before becoming ready.\n"
                    f"Container: {resolved_name}\n"
                    f"Image: {docker_image}\n"
                    f"Command: {_redact_docker_run_command(cmd)}\n"
                    f"--- docker logs (tail) ---\n{logs}\n"
                    f"--- docker inspect ---\n{inspect[:8000]}"
                )

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

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                logs = _docker_logs_tail(resolved_name)
                subprocess.run(["docker", "rm", "-f", resolved_name], check=False, capture_output=True, text=True)
                raise TimeoutError(
                    "Failed to start vLLM Docker sidecar within timeout period.\n"
                    f"Container: {resolved_name}\n"
                    f"Image: {docker_image}\n"
                    f"Endpoint: {server_models_url}\n"
                    f"Elapsed seconds: {elapsed_time:.1f}\n"
                    f"--- docker logs (tail) ---\n{logs}"
                )

            time.sleep(poll_interval_seconds)
    except BaseException:
        subprocess.run(["docker", "rm", "-f", resolved_name], check=False, capture_output=True, text=True)
        raise


def _vllm_jax_env() -> dict[str, str | Any]:
    return {
        "TOKENIZERS_PARALLELISM": "false",
        # See `_vllm_env`.
        "MODEL_IMPL_TYPE": os.environ.get("MODEL_IMPL_TYPE", "vllm"),
        "JAX_ENABLE_COMPILATION_CACHE": os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "1"),
        "JAX_COMPILATION_CACHE_DIR": os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            _default_jax_compilation_cache_dir(),
        ),
        "VLLM_XLA_CACHE_PATH": os.environ.get(
            "VLLM_XLA_CACHE_PATH",
            os.environ.get("JAX_COMPILATION_CACHE_DIR", _default_jax_compilation_cache_dir()),
        ),
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": os.environ.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1"),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2"),
    }


def _default_jax_compilation_cache_dir() -> str:
    marin_prefix = os.environ.get("MARIN_PREFIX")
    if marin_prefix:
        return os.path.join(marin_prefix, "compilation-cache")
    return "/tmp/marin-jax-compilation-cache"


def _vllm_env() -> dict[str, str]:
    env = dict(os.environ)
    # tpu_inference defaults MODEL_IMPL_TYPE=auto, which selects flax_nnx for many
    # architectures. flax_nnx currently fails without an auto mesh context, so
    # default to the vllm implementation unless the user overrides it.
    env.setdefault("MODEL_IMPL_TYPE", "vllm")
    # Reduce TPU runtime logging noise by default (match training defaults).
    env.setdefault("TPU_MIN_LOG_LEVEL", "3")
    env.setdefault("TPU_STDERR_LOG_LEVEL", "3")
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", _default_jax_compilation_cache_dir())
    # vllm-tpu uses XLA compilation caches; this env var is the one it keys off.
    env.setdefault("VLLM_XLA_CACHE_PATH", env["JAX_COMPILATION_CACHE_DIR"])
    # Cache aggressively for iterative bring-up workflows.
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2")
    return env


def _start_vllm_native_server(
    *,
    model_name_or_path: str,
    host: str = "127.0.0.1",
    port: int | None = None,
    timeout_seconds: int = 3600,
    extra_cli_args: list[str] | None = None,
) -> VllmServerHandle:
    """Start `vllm serve` in-process and wait until `/v1/models` responds."""

    resolved_port = port if port is not None else 8000

    vllm_bin = shutil.which("vllm") or "vllm"
    cmd: list[str] = [
        vllm_bin,
        "serve",
        model_name_or_path,
        "--trust-remote-code",
        "--host",
        host,
        "--port",
        str(resolved_port),
        *(extra_cli_args or []),
    ]

    log_dir = tempfile.mkdtemp(prefix="vllm_server_")
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")
    native_env = _vllm_env()
    logger.info(
        "Starting vLLM native server with "
        f"TPU_MIN_LOG_LEVEL={native_env.get('TPU_MIN_LOG_LEVEL')} "
        f"TPU_STDERR_LOG_LEVEL={native_env.get('TPU_STDERR_LOG_LEVEL')}"
    )
    process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True, env=native_env)

    server_url: str = f"http://{host}:{resolved_port}/v1"
    start_time: float = time.time()

    try:
        while True:
            if process.poll() is not None:
                stdout_f.close()
                stderr_f.close()
                logs = _native_logs_tail(log_dir)
                raise RuntimeError(
                    "vLLM server process exited before becoming ready.\n"
                    f"Command: {cmd}\n"
                    f"Exit code: {process.returncode}\n"
                    f"Logs: {log_dir}\n"
                    f"{logs}"
                )

            try:
                response = requests.get(f"{server_url}/models", timeout=5)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                # Server not ready yet.
                pass
            except requests.Timeout:
                # Server not ready yet.
                pass

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                process.kill()
                stdout_f.close()
                stderr_f.close()
                logs = _native_logs_tail(log_dir)
                raise TimeoutError(
                    "Failed to start vLLM server within timeout period.\n" f"Logs: {log_dir}\n" f"{logs}"
                )

            time.sleep(5)
    except BaseException:
        if process.poll() is None:
            process.kill()
        stdout_f.close()
        stderr_f.close()
        raise

    stdout_f.close()
    stderr_f.close()
    return VllmServerHandle(
        server_url=server_url,
        port=resolved_port,
        process=process,
        log_dir=log_dir,
    )
