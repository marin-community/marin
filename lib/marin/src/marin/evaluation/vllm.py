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
import time
from dataclasses import dataclass
from typing import Any

import requests
from fray.queue.base import Queue

from marin.evaluation.evaluation_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for inference via the pool.

    Matches OpenAI Chat Completions API format.
    """

    request_id: str
    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: list[str] | str | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None


@dataclass
class InferenceResponse:
    """Response from inference via the pool.

    Matches OpenAI Chat Completions API response format.
    """

    request_id: str
    id: str | None = None
    object: str = "chat.completion"
    created: int | None = None
    model: str | None = None
    choices: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    error: str | None = None


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
) -> str:
    """Start VLLM server in a subprocess and wait for it to be ready.

    Args:
        model_name_or_path: Model name or path to load
        engine_kwargs: Additional VLLM engine arguments
        device: Device to use ("auto", "cpu", "cuda", or "tpu")
        host: Server host
        port: Server port
        timeout_seconds: Maximum time to wait for server to be ready

    Returns:
        Server URL (e.g., "http://127.0.0.1:8000/v1")

    Raises:
        TimeoutError: If server doesn't start within timeout
    """
    # Build VLLM serve command
    # Create a temporary chat template file for models without one
    import tempfile

    chat_template = "{% for message in messages %}{{ message.content }}\n{% endfor %}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
        f.write(chat_template)
        chat_template_file = f.name

    command = (
        f"vllm serve {model_name_or_path} "
        f"--trust-remote-code "
        f"--host {host} "
        f"--port {port} "
        f"--chat-template {chat_template_file} "
    )

    # Add device specification
    if device != "auto":
        command += f"--device {device} "

    # Add distributed backend for TPU
    if device == "tpu":
        command += "--distributed-executor-backend ray "

    # Add engine kwargs
    for key, value in engine_kwargs.items():
        command += f" --{key.replace('_', '-')} {value}"

    logger.info(f"Starting VLLM server: {command}")
    process = subprocess.Popen(command, shell=True, stdout=None, stderr=None)

    # Wait for server to be ready
    server_url = f"http://{host}:{port}/v1"
    start_time = time.time()

    while True:
        try:
            health_url = f"{server_url}/models"
            logger.info(f"Health check: GET {health_url}")
            response = requests.get(health_url, timeout=5)
            logger.info(f"Health check response: {response.status_code}")
            if response.status_code == 200:
                raw_response = response.json()
                logger.info(f"Health check body: {raw_response}")
                loaded_models = [m["id"] for m in raw_response["data"]]

                if model_name_or_path in loaded_models:
                    elapsed = time.time() - start_time
                    logger.info(f"VLLM server ready at {server_url} ({elapsed:.1f}s)")
                    return server_url
                else:
                    logger.info(f"Model {model_name_or_path} not loaded yet. Loaded: {loaded_models}")
        except requests.ConnectionError as e:
            elapsed = time.time() - start_time
            logger.info(f"Health check connection error: {e}")
            if process.poll() is not None:
                raise ValueError(f"vLLM server exited with code {process.poll()}") from None
        except Exception as e:
            logger.warning(f"Health check error: {e}")

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            process.kill()
            raise TimeoutError(f"VLLM server failed to start within {timeout_seconds}s")

        time.sleep(1)

    logger.info(f"VLLM server started at {server_url}")
    return server_url


def vllm_server_worker(
    model: ModelConfig,
    request_queue: Queue[InferenceRequest],
    response_queue: Queue[InferenceResponse],
    port: int,
) -> None:
    """Worker process that runs VLLM server and processes queue requests."""
    import sys

    print("ENTERING vllm_server_worker", file=sys.stderr, flush=True)
    logger.info(f"Starting VLLM worker on port {port}")

    server_url = start_vllm_server(
        model_name_or_path=model.path,
        engine_kwargs=model.engine_kwargs,
        device=model.device,
        port=port,
    )

    logger.info("VLLM worker ready, polling request queue")

    while True:
        lease = request_queue.pop(lease_timeout=60.0)

        if lease is None:
            time.sleep(1)
            continue

        request = lease.item
        logger.info(f"Processing request {request.request_id}")

        try:
            payload = {
                "model": request.model or model.name,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "n": request.n,
                "stream": request.stream,
                "stop": request.stop,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
            }

            if request.logit_bias:
                payload["logit_bias"] = request.logit_bias
            if request.user:
                payload["user"] = request.user

            payload = {k: v for k, v in payload.items() if v is not None}

            url = f"{server_url}/chat/completions"
            logger.info(f"Making request to {url} with payload: {payload}")
            http_response = requests.post(
                url,
                json=payload,
                timeout=300,  # 5 minute timeout for generation
            )
            logger.info(f"Got response status: {http_response.status_code}")
            logger.info(f"Response headers: {dict(http_response.headers)}")
            logger.info(f"Response body: {http_response.text[:500]}")
            http_response.raise_for_status()

            vllm_response = http_response.json()

            response = InferenceResponse(
                request_id=request.request_id,
                id=vllm_response.get("id"),
                object=vllm_response.get("object", "chat.completion"),
                created=vllm_response.get("created"),
                model=vllm_response.get("model"),
                choices=vllm_response.get("choices"),
                usage=vllm_response.get("usage"),
            )

            logger.info(f"Request {request.request_id} completed successfully")

        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            response = InferenceResponse(
                request_id=request.request_id,
                error=str(e),
            )

        response_queue.push(response)
        request_queue.done(lease)
