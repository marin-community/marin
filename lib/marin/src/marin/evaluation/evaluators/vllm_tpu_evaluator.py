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
import tempfile
import shutil
from abc import ABC

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.utils import kill_process_on_port
from marin.utils import remove_tpu_lockfile_on_exit


class VllmTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/tmp"
    VLLM_PIP_PACKAGES: tuple[str, ...] = ("vllm-tpu",)

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
        model: ModelConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout_seconds: int = 3600,
        extra_args: list[str] | None = None,
    ) -> str:
        """
        Serve the model with a local vLLM server in the background.
        Returns the port the server is running on.
        """
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = VllmTpuEvaluator.download_model(model)

        vllm_bin = shutil.which("vllm")
        if vllm_bin is None:
            raise RuntimeError(
                "`vllm` CLI not found on PATH. Install vLLM in the job runtime environment "
                "(e.g. via `EnvironmentConfig(pip_packages=[...])`) before using VllmTpuEvaluator."
            )

        cmd: list[str] = [
            vllm_bin,
            "serve",
            model_name_or_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--device",
            "tpu",
            "--distributed-executor-backend",
            "ray",
        ]
        if extra_args:
            cmd.extend(extra_args)

        log_dir = tempfile.mkdtemp(prefix="vllm_server_")
        stdout_path = os.path.join(log_dir, "stdout.log")
        stderr_path = os.path.join(log_dir, "stderr.log")
        stdout_f = open(stdout_path, "w")
        stderr_f = open(stderr_path, "w")
        process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True)

        # Check that the server has started by sending heartbeat checks
        server_url: str = f"http://{host}:{port}/v1"
        start_time: float = time.time()
        elapsed_time: float = 0

        def tail(path: str, max_lines: int = 200) -> str:
            try:
                with open(path, "r") as f:
                    lines = f.readlines()
                return "".join(lines[-max_lines:])
            except Exception as e:
                return f"<failed to read {path}: {e}>"

        while True:
            if process.poll() is not None:
                stdout_f.close()
                stderr_f.close()
                raise RuntimeError(
                    "vLLM server process exited before becoming ready.\n"
                    f"Command: {cmd}\n"
                    f"Exit code: {process.returncode}\n"
                    f"Logs: {log_dir}\n"
                    f"--- stdout (tail) ---\n{tail(stdout_path)}\n"
                    f"--- stderr (tail) ---\n{tail(stderr_path)}"
                )
            try:
                # Attempt to send a request to the server's health endpoint
                response = requests.get(f"{server_url}/models", timeout=5)
                if response.status_code == 200:
                    # If the /models endpoint responds, the server is up.
                    print(f"vLLM server is up and running at {server_url}: {response.text}")
                    break
            except requests.ConnectionError:
                # If the connection is refused, wait and try again
                print(f"vLLM server is not ready yet. Elapsed time in seconds: {elapsed_time}")
            except requests.Timeout:
                print(f"vLLM server is not ready yet (timeout). Elapsed time in seconds: {elapsed_time}")

            # Check if the timeout has been reached
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                process.kill()
                stdout_f.close()
                stderr_f.close()
                raise TimeoutError("Failed to start vLLM server within timeout period.")

            time.sleep(5)  # Wait 5 seconds before retrying

        stdout_f.close()
        stderr_f.close()
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

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """
        Launches the evaluation run with Fray.
        """

        def launch(
            model: ModelConfig,
            evals: list[EvalTaskConfig],
            output_path: str,
            max_eval_instances: int | None = None,
            wandb_tags: list[str] | None = None,
        ) -> None:
            import logging

            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            self.evaluate(model, evals, output_path, max_eval_instances, wandb_tags)

        def _run():
            with remove_tpu_lockfile_on_exit():
                launch(model, evals, output_path, max_eval_instances, wandb_tags)

        if resource_config is None:
            resource_config = ResourceConfig()

        job_request = JobRequest(
            name="vllm-tpu-evaluation",
            entrypoint=Entrypoint.from_callable(_run),
            resources=resource_config,
            environment=EnvironmentConfig.create(
                extras=["eval", "tpu"],
                pip_packages=self.VLLM_PIP_PACKAGES,
            ),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
