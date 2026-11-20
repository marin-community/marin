# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one task-local LMCache RAM server for CUDA vLLM workers."""

import contextlib
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import requests
from rigging.timing import Duration, ExponentialBackoff
from rigging.tunnel import terminate_process_group

from marin.inference.vllm_router import available_ports
from marin.inference.vllm_server import _LogPump, _native_logs_tail

logger = logging.getLogger(__name__)
LMCACHE_REQUIREMENT = "lmcache==0.5.7.post1"
LMCACHE_KV_TRANSFER_FLAG = "--kv-transfer-config"
_HOST = "127.0.0.1"
_SHUTDOWN_GRACE_SECONDS = 30
_LMCACHE_ENTRYPOINT = """\
import torch
# LMCache otherwise hides native ABI import errors behind a Torch fallback.
try:
    import lmcache.cuda_ops
except ImportError as error:
    raise RuntimeError(
        f"RAM LMCache requires a native CUDA extension compatible with torch {torch.__version__}. "
        "Use a matching LMCache build or the validated Marin vLLM release."
    ) from error
from lmcache.cli.main import main

main()
"""


@dataclass(frozen=True)
class RunningLmcache:
    process: subprocess.Popen[str]
    log_dir: Path
    port: int

    def check_alive(self) -> None:
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(
                f"LMCache exited with code {code}; {self.log_dir}\n{_native_logs_tail(str(self.log_dir))}"
            )

    @property
    def vllm_args(self) -> tuple[str, ...]:
        return (
            LMCACHE_KV_TRANSFER_FLAG,
            json.dumps(
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "lmcache.mp.host": f"tcp://{_HOST}",
                        "lmcache.mp.port": self.port,
                        "lmcache.mp.autostart": False,
                    },
                }
            ),
            "--disable-hybrid-kv-cache-manager",
        )


def _interrupt_serving(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def lmcache_server(
    python_command: Sequence[str],
    ram_gib: float,
    *,
    environment: dict[str, str],
    startup_timeout_seconds: float,
    excluded_ports: Sequence[int] = (),
) -> Iterator[RunningLmcache]:
    """Start RAM storage before yielding; callers stop vLLM before leaving."""
    port, http_port, metrics_port = available_ports(_HOST, 3, exclude=excluded_ports)
    log_dir = Path(tempfile.mkdtemp(prefix="marin_lmcache_"))
    argv = [
        *python_command,
        "-c",
        _LMCACHE_ENTRYPOINT,
        "server",
        "--host",
        _HOST,
        "--port",
        str(port),
        "--http-host",
        _HOST,
        "--http-port",
        str(http_port),
        "--prometheus-port",
        str(metrics_port),
        "--l1-size-gb",
        str(ram_gib),
        "--eviction-policy",
        "LRU",
    ]
    logger.info("Starting task-local LMCache: RAM=%s GiB port=%s HTTP=%s logs=%s", ram_gib, port, http_port, log_dir)
    with contextlib.ExitStack() as stack:
        # Unwind the serving contexts on ordinary task SIGTERM. Applications
        # with their own signal handler retain responsibility for cancellation.
        if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            previous = signal.signal(signal.SIGTERM, _interrupt_serving)
            stack.callback(signal.signal, signal.SIGTERM, previous)
        process = subprocess.Popen(
            argv,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        pump = None
        try:
            pump = _LogPump(process, str(log_dir / "stdout.log"), str(log_dir / "stderr.log"))
            pump.start()
            server = RunningLmcache(process, log_dir, port)

            def ready() -> bool:
                server.check_alive()
                try:
                    return requests.get(f"http://{_HOST}:{http_port}/healthcheck", timeout=5).status_code == 200
                except (requests.ConnectionError, requests.Timeout):
                    return False

            ExponentialBackoff(initial=0.1, maximum=2).wait_until_or_raise(
                ready,
                timeout=Duration.from_seconds(startup_timeout_seconds),
                error_message=f"LMCache did not become ready; logs: {log_dir}",
            )
            yield server
        finally:
            try:
                terminate_process_group(process, grace_period=_SHUTDOWN_GRACE_SECONDS)
            finally:
                # The uvx parent can exit before a server child does. The
                # shared helper only handles a still-running parent.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
                if pump is not None:
                    pump.join(timeout=_SHUTDOWN_GRACE_SECONDS)
                    pump.close()
