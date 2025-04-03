import os
import subprocess
import time
from abc import ABC
from typing import ClassVar

import ray
import requests

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, Evaluator, ModelConfig
from marin.evaluation.utils import kill_process_on_port
from marin.generation.ray_utils import scheduling_strategy_fn
from marin.utils import remove_tpu_lockfile_on_exit


class VllmTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    DEFAULT_PIP_PACKAGES: ClassVar[list[Dependency]] = []

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
        command: str = f"vllm serve {model_name_or_path} --trust-remote-code --host {host} --port {port} --device tpu"
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

    _python_version: str = "3.10"
    _pip_packages: ClassVar[list[Dependency]] = DEFAULT_PIP_PACKAGES
    _py_modules: ClassVar[list[Dependency]] = []
    _env_vars: ClassVar[dict[str, str]] = {}

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        # Get the current runtime environment
        current_runtime_env = ray.get_runtime_context().runtime_env or {}

        # Get the current pip packages
        current_pip_packages = []
        if current_runtime_env.get("pip"):
            if isinstance(current_runtime_env["pip"], dict):
                current_pip_packages = current_runtime_env["pip"].get("packages", [])
            else:
                current_pip_packages = current_runtime_env["pip"]

        all_packages = current_pip_packages + [str(package) for package in self._pip_packages]
        runtime_env: dict = {
            "pip": {
                "packages": all_packages,
            },
            "env_vars": self._env_vars,
        }

        # An empty list of py_modules can cause an error in Ray
        if len(self._py_modules) > 0:
            runtime_env["py_modules"] = [str(module) for module in self._py_modules]

        return runtime_env

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
            scheduling_strategy=scheduling_strategy_fn(resource_config.num_tpu, resource_config.strategy),
            runtime_env=self.get_runtime_env(),
        )
        @remove_tpu_lockfile_on_exit
        def launch(
            model: ModelConfig, evals: list[EvalTaskConfig], output_path: str, max_eval_instances: int | None = None
        ) -> None:
            self.evaluate(model, evals, output_path, max_eval_instances)

        ray.get(launch.remote(model, evals, output_path, max_eval_instances))
