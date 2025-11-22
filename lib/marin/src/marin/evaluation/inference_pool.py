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

"""Fray-based inference pool with load-balanced VLLM servers."""

import logging
import threading
import time
import uuid
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fray.cluster.base import Cluster, Entrypoint, JobId, JobRequest, create_environment
from fray.queue.base import Queue

from marin.evaluation.evaluation_config import InferencePoolConfig
from marin.evaluation.vllm import InferenceRequest, InferenceResponse, vllm_server_worker

logger = logging.getLogger(__name__)


class ProxyServerThread(threading.Thread):
    """Thread that runs the OpenAI-compatible proxy server.

    This thread manages a FastAPI server that proxies requests to the inference
    pool via queues. It supports graceful shutdown via uvicorn's should_exit flag.
    """

    def __init__(
        self,
        request_queue: Queue[InferenceRequest],
        response_queue: Queue[InferenceResponse],
        host: str = "127.0.0.1",
        port: int = 9000,
    ):
        """Initialize the proxy server thread.

        Args:
            request_queue: Queue to push inference requests to
            response_queue: Queue to poll inference responses from
            host: Server host
            port: Server port
        """
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.host = host
        self.port = port
        self.server: uvicorn.Server | None = None

    def run(self) -> None:
        """Run the proxy server."""
        app = FastAPI(title="Inference Pool Proxy")

        @app.post("/v1/chat/completions")
        async def chat_completions(request: dict[str, Any]) -> dict[str, Any]:
            """OpenAI-compatible chat completions endpoint."""
            logger.info(f"/chat/completions request: {request}")
            request_id = str(uuid.uuid4())

            messages = request.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="messages field is required")

            inference_request = InferenceRequest(
                request_id=request_id,
                messages=messages,
                model=request.get("model"),
                temperature=request.get("temperature", 1.0),
                max_tokens=request.get("max_tokens"),
                top_p=request.get("top_p", 1.0),
                n=request.get("n", 1),
                stream=request.get("stream", False),
                stop=request.get("stop"),
                presence_penalty=request.get("presence_penalty", 0.0),
                frequency_penalty=request.get("frequency_penalty", 0.0),
                logit_bias=request.get("logit_bias"),
                user=request.get("user"),
            )

            self.request_queue.push(inference_request)
            logger.info(f"Pushed request {request_id} to queue")

            start_time = time.time()
            timeout = 300  # 5 minutes

            while True:
                lease = self.response_queue.pop(lease_timeout=10.0)

                if lease is None:
                    if time.time() - start_time > timeout:
                        raise HTTPException(status_code=504, detail="Request timed out")
                    time.sleep(0.1)
                    continue

                response = lease.item

                if response.request_id == request_id:
                    self.response_queue.done(lease)
                    logger.info(f"Received response for request {request_id}")

                    if response.error:
                        raise HTTPException(status_code=500, detail=response.error)

                    return {
                        "id": response.id or request_id,
                        "object": response.object,
                        "created": response.created or int(time.time()),
                        "model": response.model,
                        "choices": response.choices or [],
                        "usage": response.usage or {},
                    }
                else:
                    self.response_queue.release(lease)
                    time.sleep(0.1)

        @app.get("/v1/models")
        async def list_models() -> dict[str, Any]:
            """List available models (OpenAI compatibility)."""
            logger.info("Received /v1/models request")
            return {
                "object": "list",
                "data": [
                    {
                        "id": "default",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "marin",
                    }
                ],
            }

        @app.get("/health")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "ok"}

        logger.info(f"Starting OpenAI proxy server at http://{self.host}:{self.port}")
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        self.server = uvicorn.Server(config)
        self.server.run()

    def shutdown(self) -> None:
        """Gracefully shutdown the proxy server."""
        logger.info("Shutting down proxy server")
        if self.server:
            self.server.should_exit = True


class InferencePool:
    """Manages a pool of VLLM inference servers via Fray.

    The pool:
    1. Launches a single Fray job that runs multiple VLLM workers
    2. Starts an HTTP proxy server that exposes an OpenAI-compatible API
    3. Uses queues for fault-tolerant request/response dispatch

    Usage:
        with InferencePool(config, cluster, request_queue, response_queue) as pool:
            pool.wait_for_healthy()
            # Use pool.get_base_url() for inference
    """

    def __init__(
        self,
        config: InferencePoolConfig,
        cluster: Cluster,
        request_queue: Queue[InferenceRequest],
        response_queue: Queue[InferenceResponse],
    ):
        """Initialize the inference pool.

        Args:
            config: Pool configuration
            cluster: Fray cluster for job launching
            request_queue: Queue for inference requests
            response_queue: Queue for inference responses
        """
        self.config = config
        self.cluster = cluster
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.job_id: JobId | None = None
        self.proxy_thread: ProxyServerThread | None = None

    def __enter__(self) -> "InferencePool":
        logger.info("Starting inference pool")
        logger.info(f"Launching Fray job with {self.config.num_servers} workers")

        job_request = JobRequest(
            name="vllm-inference-pool",
            entrypoint=Entrypoint(
                callable=lambda: vllm_server_worker(
                    model=self.config.model_config,
                    request_queue=self.request_queue,
                    response_queue=self.response_queue,
                    port=self.config.vllm_port_range[0],
                ),
            ),
            resources=self.config.resource_config,
            environment=create_environment(pip_packages=["vllm==0.11.0"]),
        )

        self.cluster.__enter__()
        self.job_id = self.cluster.launch(job_request)
        logger.info(f"Launched pool job: {self.job_id}")

        self.proxy_thread = ProxyServerThread(
            self.request_queue, self.response_queue, self.config.proxy_host, self.config.proxy_port
        )
        self.proxy_thread.start()

        # Wait for proxy to be ready
        proxy_url = f"http://{self.config.proxy_host}:{self.config.proxy_port}"
        for _ in range(30):
            try:
                response = requests.get(f"{proxy_url}/health", timeout=1)
                if response.status_code == 200:
                    logger.info(f"Proxy server ready at {proxy_url}")
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            raise TimeoutError("Proxy server failed to start")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info("Shutting down inference pool")
        self.cluster.__exit__(exc_type, exc_val, exc_tb)

        if self.proxy_thread:
            self.proxy_thread.shutdown()
            self.proxy_thread.join(timeout=1)

        return False

    def base_url(self) -> str:
        return f"http://{self.config.proxy_host}:{self.config.proxy_port}/v1"

    def wait_for_healthy(self, timeout: float = 300) -> None:
        start_time = time.time()
        while True:
            info = self.cluster.poll(self.job_id)
            if info.status == "running":
                logger.info("Pool job is running")
                break
            elif info.status in ["failed", "stopped"]:
                raise RuntimeError(f"Pool job failed: {info.error_message}")

            if time.time() - start_time > timeout:
                raise TimeoutError("Pool job failed to start within timeout")

        logger.info("Pool is healthy")
