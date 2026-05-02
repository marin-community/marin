# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from rigging.filesystem import marin_prefix

from marin.evaluation.evaluators.evaluator import ModelConfig

logger = logging.getLogger(__name__)
_REMOVED_VLLM_MODE_MESSAGE = (
    "MARIN_VLLM_MODE no longer selects a vLLM backend; the Docker sidecar implementation was removed. "
    "Unset MARIN_VLLM_MODE or set it to 'native'."
)


@dataclass(frozen=True)
class VllmServerHandle:
    """A handle for a running native vLLM server."""

    server_url: str
    port: int
    process: subprocess.Popen[str]
    log_dir: str


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


def validate_vllm_mode_env() -> None:
    mode = os.environ.get("MARIN_VLLM_MODE")
    if mode is None or mode.strip().lower() in {"", "native"}:
        return
    raise ValueError(_REMOVED_VLLM_MODE_MESSAGE)


def _native_diagnostics(handle: VllmServerHandle, *, max_lines: int = 200) -> dict[str, str]:
    return {
        "vLLM native log dir": handle.log_dir,
        "vLLM native logs (tail)": _native_logs_tail(handle.log_dir, max_lines=max_lines),
    }


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
    max_num_batched_tokens = engine_kwargs.get("max_num_batched_tokens")
    if max_num_batched_tokens is not None:
        args.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    return args


def _poll_until_ready(
    server_url: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: float = 5,
    check_alive: Callable[[], None] | None = None,
) -> None:
    """Block until ``GET {server_url}/models`` returns 200.

    Args:
        server_url: The vLLM ``/v1`` base URL (e.g. ``http://127.0.0.1:8000/v1``).
        timeout_seconds: Maximum seconds to wait before raising ``TimeoutError``.
        poll_interval_seconds: Seconds between consecutive polls.
        check_alive: Optional callable invoked each iteration *before* the HTTP
            probe. Should raise if the underlying server process is
            no longer alive (the exception propagates directly to the caller).
    """
    models_url = f"{server_url}/models"
    start_time = time.time()

    while True:
        if check_alive is not None:
            check_alive()

        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                return
        except (requests.ConnectionError, requests.Timeout):
            pass  # Server not ready yet.

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"vLLM server at {models_url} did not become ready within {timeout_seconds}s "
                f"(elapsed {elapsed:.1f}s)."
            )

        time.sleep(poll_interval_seconds)


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
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_seconds: int = 3600,
        extra_args: list[str] | None = None,
    ) -> None:
        validate_vllm_mode_env()
        self.model_name_or_path, self.model = resolve_model_name_or_path(model)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.extra_cli_args = [*_engine_kwargs_to_cli_args(self.model.engine_kwargs), *(extra_args or [])]

        self.vllm_server: VllmServerHandle | None = None
        self.model_id: str | None = None

    def __enter__(self) -> "VllmEnvironment":
        if self.vllm_server is None:
            logger.info(
                "Starting vLLM environment",
                extra={
                    "model_name_or_path": self.model_name_or_path,
                    "host": self.host,
                    "port": self.port,
                },
            )
            try:
                self.vllm_server = _start_vllm_native_server(
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
                        "model_id": self.model_id,
                    },
                )
            except Exception:
                logger.exception("Failed to start vLLM environment", extra=self.debug_snapshot())
                if self.vllm_server is not None:
                    try:
                        diagnostics = _native_diagnostics(self.vllm_server)
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
            self.vllm_server.process.kill()
            self.vllm_server = None

    @property
    def server_url(self) -> str:
        if self.vllm_server is None:
            raise RuntimeError("vLLM server is not running in this environment.")
        return self.vllm_server.server_url

    def debug_snapshot(self) -> dict[str, str | int | None]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "host": self.host,
            "port": self.port,
            "server_url": self.vllm_server.server_url if self.vllm_server else None,
            "log_dir": self.vllm_server.log_dir if self.vllm_server else None,
        }

    def logs_tail(self, *, max_lines: int = 200) -> str:
        if self.vllm_server is None:
            raise RuntimeError("vLLM server is not running in this environment.")
        return _native_logs_tail(self.vllm_server.log_dir, max_lines=max_lines)

    def diagnostics(self, *, max_lines: int = 200) -> dict[str, str]:
        if self.vllm_server is None:
            return {}
        return _native_diagnostics(self.vllm_server, max_lines=max_lines)


def _default_jax_compilation_cache_dir() -> str:
    return f"{marin_prefix()}/compilation-cache"


# Canonical vLLM environment defaults for the native subprocess.
# Each (key, default) pair is resolved from the current environment at call time.
_VLLM_ENV_DEFAULTS: tuple[tuple[str, str], ...] = (
    # tpu_inference defaults MODEL_IMPL_TYPE=auto, which selects flax_nnx for many
    # architectures. flax_nnx currently fails without an auto mesh context, so
    # default to the vllm implementation unless the user overrides it.
    ("MODEL_IMPL_TYPE", "vllm"),
    ("TPU_MIN_LOG_LEVEL", "3"),
    ("TPU_STDERR_LOG_LEVEL", "3"),
    ("JAX_ENABLE_COMPILATION_CACHE", "1"),
)


def _vllm_env() -> dict[str, str]:
    """Build the vLLM environment for the native (subprocess) backend.

    Starts from ``os.environ`` and applies the canonical defaults.
    """
    env = dict(os.environ)
    cache_dir = env.get("JAX_COMPILATION_CACHE_DIR", _default_jax_compilation_cache_dir())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    # vllm-tpu uses XLA compilation caches; this env var is the one it keys off.
    env.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    # Cache aggressively for iterative bring-up workflows.
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2")
    for key, default in _VLLM_ENV_DEFAULTS:
        env.setdefault(key, default)
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

    def _check_process_alive() -> None:
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
        _poll_until_ready(
            server_url,
            timeout_seconds=timeout_seconds,
            check_alive=_check_process_alive,
        )
    except Exception:
        process.kill()
        raise
    finally:
        stdout_f.close()
        stderr_f.close()
    return VllmServerHandle(
        server_url=server_url,
        port=resolved_port,
        process=process,
        log_dir=log_dir,
    )
