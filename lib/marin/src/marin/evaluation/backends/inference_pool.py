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
from marin.evaluation.backends.vllm import vllm_server_worker
from marin.evaluation.evaluation_config import InferencePoolConfig

logger = logging.getLogger(__name__)


class ProxyServerThread(threading.Thread):
    """Thread providing a proxy server for inference requests."""

    def __init__(
        self,
        request_queue: Queue[dict[str, Any]],
        response_queue: Queue[dict[str, Any]],
        host: str = "127.0.0.1",
        port: int = 9000,
    ):
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.host = host
        self.port = port
        self.server: uvicorn.Server | None = None

    def run(self) -> None:
        app = FastAPI(title="Inference Pool Proxy")

        async def handle_inference_request(request: dict[str, Any], endpoint: str, method: str) -> dict[str, Any]:
            logger.info(f"{endpoint} request: {request}")

            # Add internal request ID and endpoint to the incoming dict
            request_id = str(uuid.uuid4())
            request["_request_id"] = request_id
            request["_endpoint"] = endpoint
            request["_method"] = method

            self.request_queue.push(request)
            logger.info(f"Pushed request {request_id} to queue")

            start_time = time.time()
            timeout = 300  # 5 minutes

            while True:
                lease = self.response_queue.pop(lease_timeout=10.0)

                if lease is None:
                    if time.time() - start_time > timeout:
                        raise HTTPException(status_code=504, detail="Request timed out")
                    time.sleep(1.0)
                    continue

                response = lease.item

                if response.get("_request_id") == request_id:
                    self.response_queue.done(lease)
                    logger.info(f"Received response for request {request_id}")

                    if response.get("error"):
                        raise HTTPException(status_code=500, detail=response["error"])

                    response_dict = {k: v for k, v in response.items() if not k.startswith("_")}
                    return response_dict
                else:
                    self.response_queue.release(lease)
                    time.sleep(1.0)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: dict[str, Any]) -> dict[str, Any]:
            return await handle_inference_request(request, "/chat/completions", "POST")

        @app.post("/v1/completions")
        async def completions(request: dict[str, Any]) -> dict[str, Any]:
            return await handle_inference_request(request, "/completions", "POST")

        @app.get("/v1/models")
        async def list_models() -> dict[str, Any]:
            return await handle_inference_request({}, "/models", "GET")

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        logger.info(f"Starting OpenAI proxy server at http://{self.host}:{self.port}")
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        self.server.run()

    def shutdown(self) -> None:
        logger.info("Shutting down proxy server")
        if self.server:
            self.server.should_exit = True


class InferencePool:
    """Manages a pool of VLLM inference servers via Fray.

    Usage:
        with InferencePool(config, cluster, request_queue, response_queue) as pool:
            # Use pool.get_base_url() for inference
    """

    def __init__(
        self,
        config: InferencePoolConfig,
        cluster: Cluster,
        request_queue: Queue[dict[str, Any]],
        response_queue: Queue[dict[str, Any]],
    ):
        self.config = config
        self.cluster = cluster
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.job_id: JobId | None = None
        self.proxy_thread: ProxyServerThread | None = None

    def __enter__(self) -> "InferencePool":
        logger.info("Starting inference pool")
        logger.info(f"Launching Fray job with {self.config.resource_config.replicas} workers")

        job_request = JobRequest(
            name="vllm-inference-pool",
            entrypoint=Entrypoint(
                callable=vllm_server_worker,
                function_args={
                    "model": self.config.model_config,
                    "request_queue": self.request_queue,
                    "response_queue": self.response_queue,
                },
            ),
            resources=self.config.resource_config,
            environment=create_environment(),
        )

        self.job_id = self.cluster.launch(job_request)
        logger.info(f"Launched pool job: {self.job_id}")

        self.proxy_thread = ProxyServerThread(
            self.request_queue, self.response_queue, self.config.proxy_host, self.config.proxy_port
        )
        self.proxy_thread.start()

        proxy_url = f"http://{self.config.proxy_host}:{self.config.proxy_port}"
        for _ in range(30):
            try:
                response = requests.get(f"{proxy_url}/health", timeout=10)
                if response.status_code == 200:
                    logger.info(f"Proxy server ready at {proxy_url}")
                    break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("Proxy server failed to start")

        self.wait_for_healthy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info("Shutting down inference pool")

        if self.proxy_thread:
            self.proxy_thread.shutdown()
            self.proxy_thread.join(timeout=1)

        return False

    def base_url(self) -> str:
        return f"http://{self.config.proxy_host}:{self.config.proxy_port}/v1"

    def wait_for_healthy(self, timeout: float = 3600) -> None:
        """Wait for the inference pool to be healthy."""
        start_time = time.time()
        proxy_url = f"http://{self.config.proxy_host}:{self.config.proxy_port}"

        def _time_left():
            return max(0, timeout - (time.time() - start_time))

        while _time_left() > 0:
            info = self.cluster.poll(self.job_id)
            if info.status in ["failed", "stopped"]:
                raise RuntimeError(f"Pool job failed during startup: {info.error_message}")

            try:
                response = requests.get(f"{proxy_url}/health", timeout=1)
                if response.status_code == 200:
                    logger.info("Proxy server is healthy")
                    break
            except requests.RequestException:
                pass

        while _time_left() > 0:
            info = self.cluster.poll(self.job_id)
            if info.status in ["failed", "stopped"]:
                raise RuntimeError(f"Pool job failed during startup: {info.error_message}")

            try:
                response = requests.get(f"{proxy_url}/v1/models", timeout=60)
                if response.status_code == 200:
                    logger.info("VLLM worker is healthy")
                    return
            except requests.RequestException as e:
                logger.info(f"VLLM worker health check failed: {e}")

            time.sleep(30)

        raise RuntimeError("VLLM worker failed to start")
