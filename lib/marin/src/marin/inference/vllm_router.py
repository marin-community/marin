# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own a released vLLM Router process without importing its Python package."""

import contextlib
import logging
import os
import signal
import socket
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import requests
from rigging.timing import Duration, ExponentialBackoff

from marin.inference.backend import OPENAI_API_SUFFIX
from marin.inference.config import WORKER_PYTHON_VERSION, VllmRouterConfig

logger = logging.getLogger(__name__)
VLLM_ROUTER_VERSION = "0.1.16"
_ROUTER_SHUTDOWN_TIMEOUT = 10


def available_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class RunningVllmRouter:
    """Routed OpenAI endpoint and the process that owns it."""

    server_url: str
    process: subprocess.Popen
    log_dir: Path
    command: tuple[str, ...]

    def logs_tail(self) -> str:
        return "\n".join(path.read_text()[-16000:] for path in sorted(self.log_dir.glob("vllm-router*"))[-2:])

    def check_alive(self) -> None:
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM Router exited with code {code}; logs: {self.log_dir}\n{self.logs_tail()}")


@contextlib.contextmanager
def vllm_router(
    config: VllmRouterConfig,
    worker_urls: Sequence[str],
    *,
    data_parallel_size: int,
    host: str = "127.0.0.1",
    port: int | None = None,
    startup_timeout_seconds: float = 600,
    check_dependencies: Callable[[], None] = lambda: None,
) -> Iterator[RunningVllmRouter]:
    """Start the pinned router and yield only after workers and routes are ready.

    Worker URLs are HTTP roots, without ``/v1``. Each exposes ``data_parallel_size``
    native ranks, including remote ranks behind a global DP frontend.
    The caller owns the workers; this context owns only the router.
    Logs flow to the job and remain in the reported directory after shutdown.
    """
    if not worker_urls:
        raise ValueError("vLLM Router requires at least one worker URL")
    if data_parallel_size < 1:
        raise ValueError("router data parallel size must be positive")
    port = available_port(host) if port is None else port
    log_dir = Path(tempfile.mkdtemp(prefix="vllm_router_"))
    command = (
        "uvx",
        "--python",
        WORKER_PYTHON_VERSION,
        "--from",
        f"vllm-router=={VLLM_ROUTER_VERSION}",
        "vllm-router",
        "--host",
        host,
        "--port",
        str(port),
        "--worker-urls",
        *worker_urls,
        "--policy",
        "consistent_hash",
        "--intra-node-data-parallel-size",
        str(data_parallel_size),
        "--request-timeout-secs",
        str(config.request_timeout_seconds),
        "--disable-retries",
        "--worker-startup-timeout-secs",
        str(max(1, int(startup_timeout_seconds))),
        "--worker-startup-check-interval",
        "1",
        # Router metrics are separate from vLLM worker metrics. Avoid the upstream
        # fixed default port so simultaneous local services cannot collide.
        "--prometheus-port",
        str(available_port(host)),
        "--prometheus-host",
        host,
        "--log-dir",
        str(log_dir),
    )
    logger.info("Starting vLLM Router %s: %s; logs: %s", VLLM_ROUTER_VERSION, command, log_dir)
    environment = dict(os.environ)
    # A caller's vLLM/plugin constraints must not enter the router's environment.
    environment.pop("UV_CONSTRAINT", None)
    environment.pop("UV_OVERRIDE", None)
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    router = RunningVllmRouter(f"http://{host}:{port}{OPENAI_API_SUFFIX}", process, log_dir, command)
    try:

        def ready() -> bool:
            check_dependencies()
            router.check_alive()
            try:
                root = router.server_url.removesuffix(OPENAI_API_SUFFIX)
                return (
                    all(requests.get(f"{worker}/health", timeout=5).status_code == 200 for worker in worker_urls)
                    and requests.get(f"{root}/health", timeout=5).status_code == 200
                    and requests.get(f"{router.server_url}/models", timeout=5).status_code == 200
                )
            except (requests.ConnectionError, requests.Timeout):
                return False

        try:
            ExponentialBackoff(initial=0.1, maximum=2).wait_until_or_raise(
                ready,
                timeout=Duration.from_seconds(startup_timeout_seconds),
                error_message=f"vLLM Router did not become ready; logs: {log_dir}",
            )
        except Exception:
            logger.error("vLLM Router startup failed; logs: %s\n%s", log_dir, router.logs_tail())
            raise
        yield router
    finally:
        # Signal the session even if uvx's parent already exited. A surviving
        # Python/Rust child must not retain the listener after a failed launch.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=_ROUTER_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM Router did not stop gracefully; killing its process group")
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=_ROUTER_SHUTDOWN_TIMEOUT)
