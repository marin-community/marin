"""Capture blocked inference processes before readiness-timeout teardown."""

import json
import os
import shutil
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import ray
from iris.cluster.runtime.profile import LocalProfileDispatch, capture_threads
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV, DEBUG_MODE_ENV, DebugMode

STARTUP_DUMP_TIMEOUT = 90
MAX_CONCURRENT_STACK_DUMPS = 8
INFERENCE_PROCESS_NAMES = ("AsyncVLLMInferenceEngine", "EngineCore")


def capture_process_stacks(pid: int, root: str, profiler: str) -> dict:
    """Persist native frames, or capture Python frames and the native unwind error."""
    directory = Path(root) / "stacks"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"startup-timeout.{socket.gethostname()}.{pid}"
    dispatch = LocalProfileDispatch(pyspy_bin=profiler, resume_pid=pid)
    receipt = {"pid": pid, "native": False}
    for native in (True, False):
        try:
            frames = capture_threads(dispatch, pid=str(pid), include_native=native, subprocesses=False)
            if not frames.strip():
                raise RuntimeError("Profiler returned no frames")
            destination = Path(f"{prefix}.{'native' if native else 'python'}.txt")
            destination.write_bytes(frames)
            receipt.update(native=native, stack_path=str(destination))
            break
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            receipt["native_error" if native else "python_error"] = str(error)
    Path(f"{prefix}.json").write_text(json.dumps(receipt, sort_keys=True))
    return receipt


def capture_node_stacks(root: str) -> list[dict]:
    """Dump actor constructors and EngineCore children in this job's task container."""
    pids = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            identity = " ".join([process.info["name"] or "", *(process.info["cmdline"] or [])])
            if any(name in identity for name in INFERENCE_PROCESS_NAMES):
                pids.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    profiler = shutil.which("py-spy") or "py-spy"
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STACK_DUMPS) as executor:
        return list(executor.map(lambda pid: capture_process_stacks(pid, root, profiler), pids))


remote_capture_node_stacks = ray.remote(num_cpus=0)(capture_node_stacks)


def capture_startup_timeout() -> None:
    """Capture all inference hosts while blocked actors and native children still exist."""
    if os.environ.get(DEBUG_MODE_ENV) != DebugMode.DISTRIBUTED:
        return
    try:
        root = os.environ[DEBUG_ARTIFACT_DIR_ENV]
        requests = [
            remote_capture_node_stacks.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
            ).remote(root)
            for node in ray.nodes()
            if node["Alive"]
        ]
        receipts = ray.get(requests, timeout=STARTUP_DUMP_TIMEOUT)
        logger.error("Inference readiness timeout stack receipts: {}", receipts)
    except Exception as error:
        # Diagnostics must preserve the original readiness timeout even if a host is lost.
        logger.error("Failed to collect inference readiness timeout stacks: {}: {}", type(error).__name__, error)
