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

"""VLLM inference server worker for Fray-based inference pool."""

import logging
import os
import subprocess
import tempfile
import time
from typing import Any

import requests
from fray.isolated_env import TemporaryVenv
from fray.queue.base import Queue
from marin.evaluation.evaluation_config import ModelConfig

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find a free port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def download_model(model: ModelConfig, cache_path: str = "/tmp") -> str:
    """Download the model if it's not already downloaded."""
    downloaded_path: str | None = model.ensure_downloaded(local_path=os.path.join(cache_path, model.name))
    model_name_or_path: str = model.name if downloaded_path is None else downloaded_path
    return model_name_or_path


def start_vllm_server(
    model_name_or_path: str,
    engine_kwargs: dict[str, Any],
    device: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_seconds: int = 3600,
    vllm_cmd: str = "vllm",
    vllm_env: dict[str, Any] | None = None,
    venv: TemporaryVenv | None = None,
) -> str:
    """Start VLLM server in a subprocess and wait for it to be ready.

    Args:
        model_name_or_path: Model name or path to load
        engine_kwargs: Additional VLLM engine arguments
        device: Device to use ("auto", "cpu", "cuda", or "tpu")
        host: Server host
        port: Server port
        timeout_seconds: Maximum time to wait for server to be ready
        vllm_cmd: Path to vllm binary command
        vllm_env: Environment dict for vllm process (deprecated, use venv instead)
        venv: TemporaryVenv instance for process tracking

    Returns:
        Server URL (e.g., "http://127.0.0.1:8000/v1")

    Raises:
        TimeoutError: If server doesn't start within timeout
    """

    chat_template = "{% for message in messages %}{{ message.content }}\n{% endfor %}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
        f.write(chat_template)
        chat_template_file = f.name

    # Build command as list for better security and compatibility
    cmd = [vllm_cmd, "serve", model_name_or_path, "--trust-remote-code", "--host", host, "--port", str(port)]
    cmd.extend(["--chat-template", chat_template_file])

    for key, value in engine_kwargs.items():
        cmd.append(f"--{key.replace('_', '-')}")
        cmd.append(str(value))

    logger.info(f"Starting VLLM server: {' '.join(cmd)}")

    # Set up environment with VLLM-specific variables
    if venv is not None:
        env = venv.get_env()
    elif vllm_env is not None:
        env = vllm_env
    else:
        env = os.environ.copy()

    env["VLLM_TARGET_DEVICE"] = device
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"

    if venv is not None:
        logger.info(f"Using venv at {venv.venv_path}")
        process = venv.run_async(cmd, env=env, stdout=None, stderr=None, text=True)
    else:
        # Fallback for backwards compatibility
        process = subprocess.Popen(cmd, env=env, stdout=None, stderr=None, text=True)

    server_url = f"http://{host}:{port}/v1"
    start_time = time.time()

    while True:
        try:
            health_url = f"{server_url}/models"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                raw_response = response.json()
                loaded_models = [m["id"] for m in raw_response["data"]]

                if model_name_or_path in loaded_models:
                    elapsed = time.time() - start_time
                    logger.info(f"VLLM server ready at {server_url} ({elapsed:.1f}s)")
                    return server_url
                else:
                    logger.info(f"Model {model_name_or_path} not loaded yet. Loaded: {loaded_models}")
        except requests.ConnectionError:
            logger.info("Health check failed (connection error).")
        except Exception as e:
            logger.warning(f"Health check error: {e}")

        exit_code = process.poll()
        if exit_code is not None:
            logger.error(f"vLLM server exited with code {exit_code}")
            raise ValueError(f"vLLM server exited with code {exit_code}")

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            process.kill()
            logger.error(f"VLLM server timed out after {timeout_seconds}s")
            raise TimeoutError(f"VLLM server failed to start within {timeout_seconds}s")

        time.sleep(5)

    logger.info(f"VLLM server started at {server_url}")
    return server_url


def vllm_server_worker(
    model: ModelConfig,
    request_queue: Queue[dict[str, Any]],
    response_queue: Queue[dict[str, Any]],
) -> None:
    """Worker process that runs VLLM server and processes queue requests."""
    with TemporaryVenv(
        venv_args=["--seed"],
        prefix="vllm_venv_",
    ) as venv:
        logger.info(f"Created isolated venv for vLLM at {venv.venv_path}")

        # Install vllm-tpu first, then upgrade libtpu to override vllm-tpu's older libtpu dependency
        logger.info("Installing vllm-tpu==0.11.1 and libtpu==0.0.30")
        env = venv.get_env()
        subprocess.check_call(
            ["uv", "pip", "install", "--python", venv.python_path, "vllm-tpu==0.11.1"],
            env=env,
        )
        # Install newer libtpu to override the older version pulled in by vllm-tpu
        subprocess.check_call(
            ["uv", "pip", "install", "--python", venv.python_path, "--upgrade", "libtpu==0.0.30"],
            env=env,
        )

        vllm_cmd_path = os.path.join(venv.bin_path, "vllm")
        if not os.path.exists(vllm_cmd_path):
            raise FileNotFoundError(f"vLLM binary not found at {vllm_cmd_path} after installation")
        logger.info(f"vLLM binary verified at {vllm_cmd_path}")

        port = find_free_port()
        logger.info(f"Auto-assigned port {port} for vLLM")

        server_url = start_vllm_server(
            model_name_or_path=model.path,
            engine_kwargs=model.engine_kwargs,
            device=model.device,
            port=port,
            vllm_cmd=vllm_cmd_path,
            venv=venv,
        )

        logger.info("VLLM worker ready, polling request queue")

        while True:
            lease = request_queue.pop(lease_timeout=60.0)

            if lease is None:
                time.sleep(1)
                continue

            request = lease.item
            request_id = request.get("_request_id")
            logger.info(f"Processing request {request_id}")

            try:
                endpoint = request.get("_endpoint", "/chat/completions")
                payload = {k: v for k, v in request.items() if not k.startswith("_")}
                if "model" not in payload or payload["model"] is None:
                    payload["model"] = model.name

                url = f"{server_url}{endpoint}"
                logger.info(f"Sending request to vLLM at {url}")
                http_response = requests.post(
                    url,
                    json=payload,
                    timeout=300,
                )
                http_response.raise_for_status()

                response = http_response.json()
                response["_request_id"] = request_id
            except Exception as e:
                logger.error(f"Request {request_id} failed: {e}")
                response = {
                    "_request_id": request_id,
                    "error": str(e),
                }

            response_queue.push(response)
            request_queue.done(lease)
