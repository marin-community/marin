# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import glob
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

import requests
from iris.marin_fs import marin_prefix

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_async import (
    AsyncVllmRuntime,
    start_async_vllm_server,
    stop_async_vllm_server,
)
from marin.inference.vllm_inprocess import (
    InProcessEligibility,
    evaluate_inprocess_eligibility,
)

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
    async_runtime: AsyncVllmRuntime | None = None

    @property
    def mode(self) -> Literal["native", "docker"]:
        if self.docker_container_name:
            return "docker"
        if self.async_runtime is not None:
            return "native"
        if self.process is not None or self.log_dir is not None:
            return "native"
        raise RuntimeError("Unable to infer vLLM server mode from handle state.")


def resolve_model_name_or_path(model: ModelConfig) -> tuple[str, ModelConfig]:
    """Resolve the `model` argument to pass to vLLM."""
    model = _maybe_enable_streaming(model)
    model_name_or_path = model.path if model.path is not None else model.name
    return model_name_or_path, model


def _model_name_or_path_without_streaming(model: ModelConfig) -> str:
    return model.path if model.path is not None else model.name


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


# ---------------------------------------------------------------------------
# Live log streaming for vLLM subprocess → Iris dashboard
#
# Iris determines log levels by parsing a single-letter prefix from each line:
#   I20260314 12:34:56 thread source message  →  INFO
#   W...  →  WARNING,  E...  →  ERROR,  D...  →  DEBUG
#
# vLLM logs to files (not the parent's stdout), so without streaming, the
# Iris dashboard shows nothing during XLA compilation.  These helpers start
# daemon threads that tail the log files and re-emit each line to stdout in
# Iris-compatible format.
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_VLLM_LEVEL_PREFIXES = [
    ("CRITICAL", "E"),
    ("ERROR", "E"),
    ("WARNING", "W"),
    ("DEBUG", "D"),
    ("INFO", "I"),
]


def _vllm_line_level(line: str) -> str:
    """Parse a vLLM log line and return the Iris single-letter level prefix.

    vLLM emits lines like ``INFO 03-14 04:01:46 [module.py:59] message``
    or ANSI-colored variants like ``\\033[0;36m(APIServer pid=91)\\033[0;0m INFO ...``.
    """
    clean = _ANSI_RE.sub("", line).lstrip()
    for level_str, prefix in _VLLM_LEVEL_PREFIXES:
        if clean.startswith(level_str):
            return prefix
    return "I"


def _iris_emit(level_char: str, source: str, message: str) -> None:
    """Emit a single log line to stdout in Iris-compatible format.

    The Iris worker captures task stdout, parses the level prefix, and stores
    the entry with the correct ``LogLevel`` so dashboard filtering works.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    tid = threading.current_thread().ident or 0
    print(f"{level_char}{ts} {tid} {source} {message}", flush=True)


def _emit_exception_to_iris(*, source: str, header: str, exc: BaseException, max_lines: int = 200) -> None:
    """Emit exception details to Iris-visible stdout logs."""
    _iris_emit("E", source, header)
    _iris_emit("E", source, f"{type(exc).__name__}: {exc}")

    emitted = 0
    for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
        for rendered_line in line.rstrip().splitlines():
            if not rendered_line:
                continue
            _iris_emit("E", source, rendered_line)
            emitted += 1
            if emitted >= max_lines:
                _iris_emit("E", source, f"Traceback truncated after {max_lines} lines")
                return


def _tail_vllm_log(path: str, source: str, process: subprocess.Popen) -> None:
    """Tail a vLLM log file and re-emit lines to stdout for Iris capture.

    Runs as a daemon thread.  Stops when the vLLM process exits, then drains
    any remaining buffered lines.
    """
    try:
        with open(path) as f:
            while process.poll() is None:
                line = f.readline()
                if line:
                    line = line.rstrip()
                    if line:
                        _iris_emit(_vllm_line_level(line), f"vllm.{source}", line)
                else:
                    time.sleep(0.5)
            # Drain remaining lines after process exits.
            for line in f:
                line = line.rstrip()
                if line:
                    _iris_emit(_vllm_line_level(line), f"vllm.{source}", line)
    except Exception:
        # Daemon thread — swallow errors silently.
        pass


def _start_vllm_log_tailers(log_dir: str, process: subprocess.Popen) -> tuple[threading.Thread, threading.Thread]:
    """Start daemon threads that tail vLLM stdout/stderr log files."""
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")

    t_out = threading.Thread(target=_tail_vllm_log, args=(stdout_path, "stdout", process), daemon=True)
    t_err = threading.Thread(target=_tail_vllm_log, args=(stderr_path, "stderr", process), daemon=True)
    t_out.start()
    t_err.start()
    return t_out, t_err


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


class ManagedAsyncVllmServerBackend(VllmServerBackend):
    def __init__(self, *, model: ModelConfig, mapping_model_name: str) -> None:
        self._model = model
        self._mapping_model_name = mapping_model_name

    def start(
        self,
        *,
        model_name_or_path: str,
        host: str,
        port: int | None,
        timeout_seconds: int,
        extra_cli_args: list[str] | None,
    ) -> VllmServerHandle:
        resolved_port = port if port is not None else 8000
        runtime = start_async_vllm_server(
            model=self._model,
            model_name_or_path=model_name_or_path,
            mapping_model_name=self._mapping_model_name,
            host=host,
            port=resolved_port,
            timeout_seconds=timeout_seconds,
            extra_cli_args=extra_cli_args,
        )
        return VllmServerHandle(
            server_url=runtime.server_url,
            port=runtime.port,
            async_runtime=runtime,
        )

    def logs_tail(self, handle: VllmServerHandle, *, max_lines: int = 200) -> str:
        runtime = handle.async_runtime
        if runtime is None:
            return "<async runtime unavailable>"
        return runtime.logs_tail(max_lines=max_lines)

    def diagnostics(self, handle: VllmServerHandle, *, max_lines: int = 200) -> dict[str, str]:
        runtime = handle.async_runtime
        if runtime is None:
            return {"vLLM async-native diagnostics": "<runtime unavailable>"}
        return {
            "vLLM async-native model id": str(runtime.model_id),
            "vLLM async-native events": runtime.logs_tail(max_lines=max_lines),
        }

    def stop(self, handle: VllmServerHandle) -> None:
        runtime = handle.async_runtime
        if runtime is None:
            return
        stop_async_vllm_server(runtime)


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
    model_loader_extra_config = engine_kwargs.get("model_loader_extra_config")
    if model_loader_extra_config is not None:
        args.extend(["--model-loader-extra-config", json.dumps(model_loader_extra_config)])
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
        native_startup_failure_mode: Literal["fallback", "raise"] = "fallback",
        docker_image: str | None = None,
        docker_run_args: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.mode = resolve_vllm_mode(mode)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.native_startup_failure_mode = native_startup_failure_mode
        self.docker_image = docker_image
        self.docker_run_args = docker_run_args
        self._extra_args = list(extra_args or [])

        self.model = model
        self.model_name_or_path = _model_name_or_path_without_streaming(self.model)
        self.extra_cli_args = [*_engine_kwargs_to_cli_args(self.model.engine_kwargs), *self._extra_args]

        self._fallback_backend: VllmServerBackend | None = None
        self._fallback_model: ModelConfig | None = None
        self._fallback_extra_cli_args: list[str] | None = None
        self._inprocess_eligibility: InProcessEligibility | None = None
        self._backend: VllmServerBackend

        if self.mode == "native":
            # Pass only raw extra_args (not engine_kwargs converted to CLI flags)
            # to the eligibility check.  engine_kwargs are read directly by
            # the native async backend; passing them as CLI flags would look
            # "supported" but raw extra_args need explicit handling there.
            self._inprocess_eligibility = evaluate_inprocess_eligibility(
                model=self.model,
                model_name_or_path=self.model_name_or_path,
                extra_cli_args=self._extra_args or None,
            )
            if self._inprocess_eligibility.eligible:
                assert self._inprocess_eligibility.mapping_model_name is not None
                self._backend = ManagedAsyncVllmServerBackend(
                    model=self.model,
                    mapping_model_name=self._inprocess_eligibility.mapping_model_name,
                )

                self._fallback_backend = NativeVllmServerBackend()
                self._fallback_model = _maybe_enable_streaming(self.model)
                self._fallback_extra_cli_args = [
                    *_engine_kwargs_to_cli_args(self._fallback_model.engine_kwargs),
                    *self._extra_args,
                ]
            else:
                assert self._inprocess_eligibility is not None
                logger.info(
                    "Async native vLLM not selected; using subprocess native backend",
                    extra={"reason": self._inprocess_eligibility.reason},
                )
                self.model = _maybe_enable_streaming(self.model)
                self.model_name_or_path = _model_name_or_path_without_streaming(self.model)
                self.extra_cli_args = [*_engine_kwargs_to_cli_args(self.model.engine_kwargs), *self._extra_args]
                self._backend = NativeVllmServerBackend()
        else:
            self.model = _maybe_enable_streaming(self.model)
            self.model_name_or_path = _model_name_or_path_without_streaming(self.model)
            self.extra_cli_args = [*_engine_kwargs_to_cli_args(self.model.engine_kwargs), *self._extra_args]
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
                self._start_with_current_backend()
            except Exception as exc:
                if self._fallback_backend is not None and self.native_startup_failure_mode == "fallback":
                    _emit_exception_to_iris(
                        source="vllm.async",
                        header="Async native vLLM startup failed; falling back to subprocess native backend",
                        exc=exc,
                    )
                    logger.warning(
                        "Async native vLLM startup failed; falling back to subprocess native backend",
                        exc_info=True,
                    )
                    self._activate_fallback_backend()
                    try:
                        self._start_with_current_backend()
                    except Exception:
                        logger.exception("Failed to start fallback vLLM backend", extra=self.debug_snapshot())
                        if self.vllm_server is not None:
                            try:
                                diagnostics = self._backend.diagnostics(self.vllm_server)
                                for label, value in diagnostics.items():
                                    logger.error("%s:\n%s", label, value)
                            except Exception:
                                logger.exception("Failed to collect vLLM diagnostics")
                        raise
                else:
                    if self._fallback_backend is not None:
                        _emit_exception_to_iris(
                            source="vllm.async",
                            header="Async native vLLM startup failed; not falling back",
                            exc=exc,
                        )
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

    def _start_with_current_backend(self) -> None:
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

    def _activate_fallback_backend(self) -> None:
        if self._fallback_backend is None or self._fallback_model is None:
            raise RuntimeError("Fallback backend requested but not configured.")

        self._backend = self._fallback_backend
        self._fallback_backend = None

        self.model = self._fallback_model
        self._fallback_model = None
        self.model_name_or_path = _model_name_or_path_without_streaming(self.model)
        self.extra_cli_args = self._fallback_extra_cli_args or [*_engine_kwargs_to_cli_args(self.model.engine_kwargs)]
        self._fallback_extra_cli_args = None

    def close(self) -> None:
        if self.vllm_server is not None:
            self._backend.stop(self.vllm_server)
            self.vllm_server = None
            self.model_id = None

    @property
    def server_url(self) -> str:
        if self.vllm_server is None:
            raise RuntimeError("vLLM server is not running in this environment.")
        return self.vllm_server.server_url

    def debug_snapshot(self) -> dict[str, str | int | None]:
        return {
            "mode": self.mode,
            "backend": type(self._backend).__name__,
            "model_name_or_path": self.model_name_or_path,
            "host": self.host,
            "port": self.port,
            "server_url": self.vllm_server.server_url if self.vllm_server else None,
            "docker_container_name": self.vllm_server.docker_container_name if self.vllm_server else None,
            "docker_image": self.vllm_server.docker_image if self.vllm_server else self.docker_image,
            "docker_run_cmd": self.vllm_server.docker_run_cmd if self.vllm_server else None,
            "inprocess_eligibility": self._inprocess_eligibility.reason if self._inprocess_eligibility else None,
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
    """Return True when running on TPU hardware.

    Ray TPU pods do not consistently set `TPU_NAME`, so we also detect the
    presence of TPU device nodes and other TPU-related environment variables.
    """

    if os.environ.get("TPU_NAME"):
        return True

    # GKE TPU device plugin exposes /dev/accel* device nodes.
    if glob.glob("/dev/accel*"):
        return True

    # Heuristic fallbacks for TPU pods / libtpu environments.
    for key in (
        "TPU_ACCELERATOR_TYPE",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_MESH_CONTROLLER_ADDRESS",
        "TPU_VISIBLE_DEVICES",
    ):
        if os.environ.get(key):
            return True

    # Ray TPU workers may not expose TPU env vars/device nodes to the driver
    # container, but Ray's TPUAcceleratorManager can still report topology.
    try:
        from ray._private.accelerators import TPUAcceleratorManager

        pod_type = None
        if hasattr(TPUAcceleratorManager, "_get_current_node_tpu_pod_type"):
            pod_type = TPUAcceleratorManager._get_current_node_tpu_pod_type()
        elif hasattr(TPUAcceleratorManager, "get_current_node_tpu_pod_type"):
            pod_type = TPUAcceleratorManager.get_current_node_tpu_pod_type()
        if pod_type:
            return True

        tpu_name = None
        if hasattr(TPUAcceleratorManager, "get_current_node_tpu_name"):
            tpu_name = TPUAcceleratorManager.get_current_node_tpu_name()
        if tpu_name:
            return True
    except Exception:
        pass

    return False


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
    except Exception:
        subprocess.run(["docker", "rm", "-f", resolved_name], check=False, capture_output=True, text=True)
        raise


def _vllm_jax_env() -> dict[str, str | Any]:
    env: dict[str, str | Any] = {
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

    # Pass through vLLM knobs when callers set them in the environment.
    for key in (
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
        "VLLM_TPU_SKIP_PRECOMPILE",
    ):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    return env


def _default_jax_compilation_cache_dir() -> str:
    return f"{marin_prefix()}/compilation-cache"


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

    # Stream vLLM logs to stdout in real time so they appear in the Iris
    # dashboard.  Without this, XLA compilation is a silent black box.
    _start_vllm_log_tailers(log_dir, process)

    server_url: str = f"http://{host}:{resolved_port}/v1"
    start_time: float = time.time()
    last_heartbeat: float = start_time
    _HEARTBEAT_INTERVAL: float = 30.0

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
            pass
        except requests.Timeout:
            pass

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            process.kill()
            stdout_f.close()
            stderr_f.close()
            logs = _native_logs_tail(log_dir)
            raise TimeoutError("Failed to start vLLM server within timeout period.\n" f"Logs: {log_dir}\n" f"{logs}")

        if time.time() - last_heartbeat >= _HEARTBEAT_INTERVAL:
            _iris_emit("I", "vllm.startup", f"Waiting for vLLM server... ({int(elapsed_time)}s elapsed)")
            last_heartbeat = time.time()

        time.sleep(5)

    _iris_emit("I", "vllm.startup", f"vLLM server ready after {int(time.time() - start_time)}s")
    stdout_f.close()
    stderr_f.close()
    return VllmServerHandle(
        server_url=server_url,
        port=resolved_port,
        process=process,
        log_dir=log_dir,
    )
