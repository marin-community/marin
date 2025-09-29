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

import os
import subprocess
import time
from abc import ABC

import ray
import requests

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.utils import kill_process_on_port
from marin.run.ray_deps import build_runtime_env_for_packages
from marin.utils import remove_tpu_lockfile_on_exit


class VllmTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/tmp"

    @staticmethod
    def download_model(model: ModelConfig) -> str:
        """
        Download the model if it's not already downloaded
        """
        downloaded_path: str | None = model.ensure_downloaded(
            local_path=os.path.join(VllmTpuEvaluator.CACHE_PATH, model.name)
        )
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = model.name if downloaded_path is None else downloaded_path
        return model_name_or_path

    @staticmethod
    def start_vllm_server_in_background(
        model: ModelConfig, host: str = "127.0.0.1", port: int = 8000, timeout_seconds: int = 3600
    ) -> str:
        """
        Serve the model with a local vLLM server in the background.
        Returns the port the server is running on.
        """
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = VllmTpuEvaluator.download_model(model)

        # From https://docs.vllm.ai/en/v0.4.0/models/engine_args.html
        command: str = (
            f"vllm serve {model_name_or_path} "
            f"--trust-remote-code "
            f"--host {host} "
            f"--port {port} "
            f"--device tpu "
            f"--distributed-executor-backend ray"
        )
        process = subprocess.Popen(command, shell=True)

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
        return build_runtime_env_for_packages(extra=["eval", "tpu"])

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        resource_config: ResourceConfig | None = None,
    ) -> None:
        """
        Launches the evaluation run with Ray.
        """

        @ray.remote(
            scheduling_strategy=self._get_scheduling_strategy(resource_config),
            runtime_env=self.get_runtime_env(),
            max_calls=1,
        )
        @remove_tpu_lockfile_on_exit
        def launch(
            model: ModelConfig,
            evals: list[EvalTaskConfig],
            output_path: str,
            max_eval_instances: int | None = None,
        ) -> None:
            self.evaluate(model, evals, output_path, max_eval_instances)

        ray.get(launch.remote(model, evals, output_path, max_eval_instances))
