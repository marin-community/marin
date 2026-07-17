# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import requests
from rigging.filesystem import marin_prefix

from marin.evaluation.evaluators.evaluator import ModelConfig

logger = logging.getLogger(__name__)
# Bounded tail for the failure path and diagnostics(); the full stream reaches the job log, so
# this is only a convenience snapshot, capped because vLLM logs can be large.
_NATIVE_LOG_TAIL_LINES = 1000
_REMOVED_VLLM_MODE_MESSAGE = (
    "MARIN_VLLM_MODE no longer selects a vLLM backend; the Docker sidecar implementation was removed. "
    "Unset MARIN_VLLM_MODE or set it to 'native'."
)
# The worker interpreter marin-serve provisions everywhere it controls one: the checkout-free
# venv and the isolated uvx vLLM envs. Kept single so they cannot drift — cloudpickle needs the
# worker venv to match the launching CLI, and the uvx env to match the venv. Marin pins 3.12.
WORKER_PYTHON_VERSION = "3.12"


class VllmLauncher(Protocol):
    """Builds the argv and extra environment that run the ``vllm`` CLI.

    vLLM always runs as a subprocess, so a launcher is the command prefix (before its
    ``serve …`` args) plus any environment it needs. Implementations run either the
    ``vllm`` already on ``PATH`` or one provisioned in a throwaway uv-managed env.
    """

    def command(self) -> list[str]: ...

    def env(self) -> dict[str, str]:
        """Extra environment variables to overlay on the vLLM subprocess env."""
        ...


@dataclass(frozen=True)
class WorkspaceVllm:
    """Run the ``vllm`` installed in the active workspace venv (the TPU-vLLM stack)."""

    def command(self) -> list[str]:
        return [shutil.which("vllm") or "vllm"]

    def env(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class IsolatedCudaVllm:
    """Run CUDA vLLM from a throwaway uv-managed environment via ``uvx``.

    GPU serving pins CUDA vLLM here instead of in the Marin workspace lockfile:
    vLLM is only ever a ``vllm serve`` subprocess, so ``uvx`` provisions it — and
    its torch/CUDA wheel tree — in a cached, isolated environment that never
    enters Marin's own resolution. Bumping the version is therefore just a string,
    with no workspace re-lock. The ``[runai]`` extra keeps gs://-checkpoint
    streaming working, at parity with the TPU path.
    """

    version: str
    # Match the workspace interpreter so cloudpickled entrypoints stay compatible.
    python_version: str = WORKER_PYTHON_VERSION
    # uv's PyTorch index selector; stock vLLM (>=0.25) targets torch 2.11 / CUDA 13.
    torch_backend: str = "cu128"

    def command(self) -> list[str]:
        return [
            "uvx",
            "--from",
            f"vllm[runai]=={self.version}",
            "--python",
            self.python_version,
            "--torch-backend",
            self.torch_backend,
            "vllm",
        ]

    def env(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class IsolatedTpuVllm:
    """Run Marin's forked TPU vLLM from a throwaway uv-managed environment via ``uvx``.

    The TPU counterpart to :class:`IsolatedCudaVllm`. ``vllm`` and its ``tpu-inference``
    runtime are two git forks pinned by SHA (see ``marin.inference.tpu_vllm_pins``); this
    provisions them in an isolated uv-tool env rather than the workspace lock, so
    ``marin-serve --tpu`` runs from outside a checkout.
    """

    vllm_ref: str
    """``uvx --from`` spec for the vLLM fork, e.g.
    ``vllm @ git+https://github.com/marin-community/vllm.git@<sha>``."""
    tpu_inference_ref: str
    """``uvx --with`` spec for the tpu-inference fork (vLLM's TPU runtime dependency)."""
    # Match the workspace interpreter so cloudpickled entrypoints stay compatible.
    python_version: str = WORKER_PYTHON_VERSION
    # torch is only a dependency here (jax/libtpu do TPU compute), so resolve it from the
    # CPU index rather than dragging in a CUDA tree.
    torch_backend: str = "cpu"

    def command(self) -> list[str]:
        return [
            "uvx",
            "--from",
            self.vllm_ref,
            "--with",
            self.tpu_inference_ref,
            "--python",
            self.python_version,
            "--torch-backend",
            self.torch_backend,
            "vllm",
        ]

    def env(self) -> dict[str, str]:
        # vLLM targets CUDA unless VLLM_TARGET_DEVICE is set; the uvx build subprocess
        # inherits this from the launch environment.
        return {"VLLM_TARGET_DEVICE": "tpu"}


# Forwarded lines route to the parent's stderr (finelog tags it ERROR) or stdout (INFO) by their
# own level, not by source stream: vLLM writes all levels to its stderr.
_ERROR_LEVEL_MARKERS = ("ERROR", "CRITICAL")


def _looks_like_error(line: str) -> bool:
    """Coarse severity check — a substring, not a format parse; a misroute only mislabels the level."""
    return any(marker in line for marker in _ERROR_LEVEL_MARKERS)


class _LogPump:
    """Forward a vLLM subprocess's stdout/stderr to the parent's fds and to on-disk logs.

    One daemon reader thread per pipe drains the child and, per line, appends it to a capped
    on-disk log (which backs the failure tail and ``diagnostics()``) and re-emits it to the
    parent's stdout/stderr by severity. Forwarding goes to the fds directly, not through the
    logger, so it does not depend on ``rigging.configure_logging`` having run — several callers of
    this module never call it. A reader must never stall while the child lives: a full pipe blocks
    the child.
    """

    def __init__(self, process: subprocess.Popen[str], stdout_path: str, stderr_path: str) -> None:
        self._process = process
        # Open for the server's lifetime (closed by close()); line-buffered so the tail stays current.
        self._stdout_file = open(stdout_path, "w", buffering=1)  # noqa: SIM115
        self._stderr_file = open(stderr_path, "w", buffering=1)  # noqa: SIM115
        # Both readers may write to the parent's stdout; serialize so lines don't interleave.
        self._sink_lock = threading.Lock()
        assert process.stdout is not None and process.stderr is not None
        self._threads = (
            threading.Thread(
                target=self._pump, args=(process.stdout, self._stdout_file), name="vllm-stdout", daemon=True
            ),
            threading.Thread(
                target=self._pump, args=(process.stderr, self._stderr_file), name="vllm-stderr", daemon=True
            ),
        )

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def _pump(self, stream, log_file) -> None:
        # A stalled reader deadlocks the child, so a failed write must not break the drain loop:
        # guard the disk and parent-fd writes independently.
        try:
            for line in iter(stream.readline, ""):
                try:
                    log_file.write(line)
                except Exception:
                    logger.warning("Failed to persist a vLLM log line to %s", log_file.name, exc_info=True)
                sink = sys.stderr if _looks_like_error(line) else sys.stdout
                try:
                    with self._sink_lock:
                        sink.write(line.rstrip("\r\n") + "\n")
                        sink.flush()
                except Exception:
                    logger.warning("Failed to forward a vLLM log line to the parent process", exc_info=True)
        finally:
            # At EOF: flush so a newline-less final fragment (a crash mid-write) reaches the tail,
            # which reads right after join(); then close the read end so serves don't leak pipe fds.
            try:
                log_file.flush()
            except Exception:
                logger.debug("Failed to flush a vLLM native log file", exc_info=True)
            stream.close()

    def join(self, timeout: float | None = None) -> None:
        """Wait for both readers to drain and exit, bounded by ``timeout``."""
        for thread in self._threads:
            thread.join(timeout=timeout)

    def close(self) -> None:
        for log_file in (self._stdout_file, self._stderr_file):
            try:
                log_file.close()
            except Exception:
                # Best-effort during teardown; a close failure must not mask the caller's shutdown.
                logger.debug("Failed to close a vLLM native log file", exc_info=True)


@dataclass(frozen=True)
class VllmServerHandle:
    """A handle for a running native vLLM server."""

    server_url: str
    port: int
    process: subprocess.Popen[str]
    process_group_id: int | None
    log_dir: str
    # Owns the reader threads and on-disk log files.
    log_pump: _LogPump | None = None

    def stop(self, *, timeout_seconds: float = 10) -> None:
        self._signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            self.process.wait(timeout=timeout_seconds)

        if self._process_group_exists():
            # The API parent can exit before EngineCore does, so check the group after wait().
            self._signal(signal.SIGKILL)

        # Child and group are gone, so the pipes are at EOF; join the readers (bounded, so a
        # descendant holding a pipe cannot hang teardown) and close the logs.
        if self.log_pump is not None:
            self.log_pump.join(timeout=timeout_seconds)
            self.log_pump.close()

    def _signal(self, sig: signal.Signals) -> None:
        if self.process_group_id is not None:
            try:
                os.killpg(self.process_group_id, sig)
            except ProcessLookupError:
                pass
            return

        if self.process.poll() is None:
            logger.warning(
                "vLLM process group unavailable; signaling only parent process pid=%s signal=%s",
                self.process.pid,
                sig.name,
            )
            self.process.send_signal(sig)

    def _process_group_exists(self) -> bool:
        if self.process_group_id is None:
            return False
        try:
            os.killpg(self.process_group_id, 0)
            return True
        except ProcessLookupError:
            return False


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


def _native_logs_tail(log_dir: str | None, *, max_lines: int = _NATIVE_LOG_TAIL_LINES) -> str:
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


def _native_diagnostics(handle: VllmServerHandle, *, max_lines: int = _NATIVE_LOG_TAIL_LINES) -> dict[str, str]:
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
                f"vLLM server at {models_url} did not become ready within {timeout_seconds}s (elapsed {elapsed:.1f}s)."
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
    """Manage vLLM server lifecycle and eval-client configuration."""

    def __init__(
        self,
        model: ModelConfig,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_seconds: int = 3600,
        extra_args: list[str] | None = None,
        launcher: VllmLauncher | None = None,
    ) -> None:
        validate_vllm_mode_env()
        self.model_name_or_path, self.model = resolve_model_name_or_path(model)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.extra_cli_args = [*_engine_kwargs_to_cli_args(self.model.engine_kwargs), *(extra_args or [])]
        # Default to the workspace vLLM (TPU stack); GPU serving passes IsolatedCudaVllm.
        self.launcher: VllmLauncher = launcher or WorkspaceVllm()

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
                    launcher=self.launcher,
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
            self.vllm_server.stop()
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

    def logs_tail(self, *, max_lines: int = _NATIVE_LOG_TAIL_LINES) -> str:
        if self.vllm_server is None:
            raise RuntimeError("vLLM server is not running in this environment.")
        return _native_logs_tail(self.vllm_server.log_dir, max_lines=max_lines)

    def diagnostics(self, *, max_lines: int = _NATIVE_LOG_TAIL_LINES) -> dict[str, str]:
        if self.vllm_server is None:
            return {}
        return _native_diagnostics(self.vllm_server, max_lines=max_lines)


# Cache aggressively for iterative bring-up workflows: every compilation is worth keeping, and
# a serve of the same model on the same slice should not pay for the compile twice. Both serving
# backends key off these — vLLM through its subprocess environment, Levanter through jax.config.
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = -1
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS = 2


def default_jax_compilation_cache_dir() -> str:
    """Persistent XLA/JAX compilation cache shared by every serving backend on this slice."""
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
    cache_dir = env.get("JAX_COMPILATION_CACHE_DIR", default_jax_compilation_cache_dir())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    # TPU vLLM uses XLA compilation caches; this env var is the one it keys off.
    env.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", str(JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES))
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", str(JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS))
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
    launcher: VllmLauncher | None = None,
) -> VllmServerHandle:
    """Start `vllm serve` as a subprocess and wait until `/v1/models` responds."""

    resolved_port = port if port is not None else 8000
    launcher = launcher or WorkspaceVllm()

    cmd: list[str] = [
        *launcher.command(),
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
    native_env = _vllm_env()
    # A launcher (e.g. the isolated TPU build) may require extra env, such as the
    # vLLM build target; overlay it after the canonical defaults so it wins.
    native_env.update(launcher.env())
    logger.info(
        "Starting vLLM native server (output streams to the job log). "
        f"TPU_MIN_LOG_LEVEL={native_env.get('TPU_MIN_LOG_LEVEL')} "
        f"TPU_STDERR_LOG_LEVEL={native_env.get('TPU_STDERR_LOG_LEVEL')}"
    )
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=native_env,
        # vLLM can leave EngineCore children alive after the API parent exits; a process group lets cleanup
        # release the TPU instead of leaving libtpu held by a stale child.
        start_new_session=True,
    )
    # Pump before readiness polling: vLLM logs heavily during the long weight-load/compile, and an
    # undrained pipe would block the child mid-startup.
    log_pump = _LogPump(process, stdout_path, stderr_path)
    log_pump.start()
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group_id = None

    server_url: str = f"http://{host}:{resolved_port}/v1"

    def _check_process_alive() -> None:
        if process.poll() is not None:
            # Child has exited; drain the readers before reading the tail so it has the final lines.
            log_pump.join(timeout=5)
            logs = _native_logs_tail(log_dir)
            raise RuntimeError(
                "vLLM server process exited before becoming ready.\n"
                f"Command: {cmd}\n"
                f"Exit code: {process.returncode}\n"
                f"Logs: {log_dir}\n"
                f"{logs}"
            )

    handle = VllmServerHandle(
        server_url=server_url,
        port=resolved_port,
        process=process,
        process_group_id=process_group_id,
        log_dir=log_dir,
        log_pump=log_pump,
    )

    try:
        _poll_until_ready(
            server_url,
            timeout_seconds=timeout_seconds,
            check_alive=_check_process_alive,
        )
    except Exception:
        handle.stop()
        raise
    return handle
