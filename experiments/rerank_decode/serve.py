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

"""Utilities for launching and managing vLLM servers for two-model rerank decode."""

import logging
import os
import subprocess
import sys
import time

import requests

logger = logging.getLogger(__name__)


def launch_vllm_server(
    model: str,
    port: int,
    gpu_ids: list[int],
    tensor_parallel_size: int | None = None,
) -> subprocess.Popen:
    """Launch a single vLLM OpenAI-compatible server on the specified GPUs.

    Args:
        model: HuggingFace model name or path.
        port: Port to serve on.
        gpu_ids: Which GPUs this server should use.
        tensor_parallel_size: TP degree. Defaults to len(gpu_ids).

    Returns:
        The server subprocess handle.
    """
    if tensor_parallel_size is None:
        tensor_parallel_size = len(gpu_ids)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    logger.info("Launching vLLM server: model=%s port=%d gpus=%s", model, port, gpu_ids)
    proc = subprocess.Popen(cmd, env=env)
    return proc


def wait_for_server(port: int, timeout: float = 300, poll_interval: float = 2.0) -> None:
    """Block until the vLLM server at the given port is healthy.

    Args:
        port: Server port.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between health checks.
    """
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("vLLM server on port %d is ready.", port)
                return
        except requests.RequestException:
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"vLLM server on port {port} did not become healthy within {timeout}s")


def launch_vllm_servers(
    proposal_model: str,
    scoring_model: str,
    proposal_gpus: list[int],
    scoring_gpus: list[int],
    proposal_port: int = 8000,
    scoring_port: int = 8001,
) -> tuple[subprocess.Popen, subprocess.Popen]:
    """Launch two vLLM servers for the proposal and scoring models.

    Returns:
        A tuple of (proposal_process, scoring_process).
    """
    procs = []
    try:
        procs.append(launch_vllm_server(proposal_model, proposal_port, proposal_gpus))
        procs.append(launch_vllm_server(scoring_model, scoring_port, scoring_gpus))
        wait_for_server(proposal_port)
        wait_for_server(scoring_port)
    except:
        shutdown_servers(*procs)
        raise

    return procs[0], procs[1]


def shutdown_servers(*procs: subprocess.Popen) -> None:
    """Terminate the given server processes."""
    for proc in procs:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            pass