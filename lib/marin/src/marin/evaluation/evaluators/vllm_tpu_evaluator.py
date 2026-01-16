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

import dataclasses
import os
import subprocess
import time
from abc import ABC
from typing import ClassVar
from urllib.parse import urlparse

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from fray.cluster.ray.deps import build_runtime_env_for_packages

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.utils import kill_process_on_port
from marin.utils import remove_tpu_lockfile_on_exit

class VllmTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    _STREAMING_LOAD_FORMATS: ClassVar[set[str]] = {"runai_streamer", "runai_streamer_sharded"}

    @staticmethod
    def _vllm_env() -> dict[str, str]:
        env = dict(os.environ)
        # tpu_inference defaults MODEL_IMPL_TYPE=auto, which selects flax_nnx for many
        # architectures. flax_nnx currently fails without an auto mesh context, so
        # default to the vllm implementation unless the user overrides it.
        env.setdefault("MODEL_IMPL_TYPE", "vllm")
        return env

    @staticmethod
    def _is_object_store_path(path: str) -> bool:
        parsed = urlparse(path)
        return parsed.scheme in {"gs", "s3"}

    @staticmethod
    def _maybe_enable_streaming(model: ModelConfig) -> ModelConfig:
        if model.path is None:
            return model
        if not VllmTpuEvaluator._is_object_store_path(model.path):
            return model
        if "load_format" in model.engine_kwargs:
            return model

        engine_kwargs = dict(model.engine_kwargs)
        # Default to the non-sharded streamer for maximum compatibility.
        # `runai_streamer_sharded` only works for checkpoints that are already sharded
        # into `model-rank-*-part-*.safetensors`.
        engine_kwargs["load_format"] = "runai_streamer"
        return dataclasses.replace(model, engine_kwargs=engine_kwargs)

    @staticmethod
    def _engine_kwargs_to_cli_args(engine_kwargs: dict) -> list[str]:
        args: list[str] = []
        load_format = engine_kwargs.get("load_format")
        if isinstance(load_format, str):
            args.extend(["--load-format", load_format])
        return args

    @staticmethod
    def resolve_model_name_or_path(model: ModelConfig) -> tuple[str, ModelConfig]:
        """Resolve the `model` argument to pass to vLLM.

        - If `model.path` is set, use it (and auto-enable streaming for `gs://` / `s3://`).
        - Otherwise, fall back to `model.name` (e.g. an HF repo id).
        """
        model = VllmTpuEvaluator._maybe_enable_streaming(model)
        model_name_or_path = model.path if model.path is not None else model.name
        return model_name_or_path, model

    @staticmethod
    def start_vllm_server_in_background(
        model: ModelConfig, host: str = "127.0.0.1", port: int = 8000, timeout_seconds: int = 3600
    ) -> str:
        """
        Serve the model with a local vLLM server in the background.
        Returns the port the server is running on.
        """
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path, model = VllmTpuEvaluator.resolve_model_name_or_path(model)

        # From https://docs.vllm.ai/en/v0.4.0/models/engine_args.html
        command: list[str] = [
            "vllm",
            "serve",
            model_name_or_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--distributed-executor-backend",
            "ray",
            *VllmTpuEvaluator._engine_kwargs_to_cli_args(model.engine_kwargs),
        ]
        process = subprocess.Popen(command, env=VllmTpuEvaluator._vllm_env())

        # Check that the server has started by sending heartbeat checks
        server_url: str = f"http://{host}:{port}/v1"
        start_time: float = time.time()
        elapsed_time: float = 0
        while True:
            try:
                # Attempt to send a request to the server's health endpoint
                response = requests.get(f"{server_url}/models")
                if response.status_code == 200:
                    raw_response: dict = response.json()
                    loaded_models: list[str] = [model["id"] for model in raw_response["data"]]

                    # Can be on a machine with a vLLM server up and running, so also check the model is loaded
                    print(f"vLLM server is up and running at {server_url}: {response.text}")
                    if model_name_or_path in loaded_models:
                        print(f"Model {model_name_or_path} is loaded.")
                        break
                    else:
                        print(f"Model {model_name_or_path} is not loaded yet. Loaded models: {loaded_models}")
            except requests.ConnectionError:
                # If the connection is refused, wait and try again
                print(f"vLLM server is not ready yet. Elapsed time in seconds: {elapsed_time}")

            # Check if the timeout has been reached
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                process.kill()
                raise TimeoutError("Failed to start vLLM server within timeout period.")

            time.sleep(5)  # Wait 5 seconds before retrying

        print(f"vLLM server is ready at {server_url} ({elapsed_time}s).")
        return server_url

    @staticmethod
    def cleanup(model: ModelConfig, vllm_port: int | None = None) -> None:
        """
        Clean up the vLLM server and any other resources.
        """
        print("Cleaning up resources.")
        # Kill the vLLM server
        try:
            if vllm_port is not None:
                kill_process_on_port(vllm_port)
        except Exception as e:
            print(f"Failed to kill vLLM server on port {vllm_port}: {e}")

        # Delete the checkpoint
        model.destroy()

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        return build_runtime_env_for_packages(extra=["eval", "vllm"])

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        resource_config: ResourceConfig | None = None,
        wandb_tags: list[str] | None = None,
        generation_params: dict | None = None,
    ) -> None:
        """
        Launches the evaluation run with Fray.
        """

        def _run():
            with remove_tpu_lockfile_on_exit():
                import logging
                logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
                self.evaluate(model, evals, output_path, max_eval_instances, wandb_tags=wandb_tags, generation_params=generation_params)

        if resource_config is None:
            resource_config = ResourceConfig()

        job_request = JobRequest(
            name="vllm-tpu-evaluation",
            entrypoint=Entrypoint.from_callable(_run),
            resources=resource_config,
            environment=EnvironmentConfig.create(extras=["eval", "vllm"]),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
