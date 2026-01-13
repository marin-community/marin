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
import tempfile
import time
from abc import ABC
from typing import ClassVar
from urllib.parse import urlparse

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, GpuConfig, JobRequest, ResourceConfig, TpuConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.evaluation.utils import kill_process_on_port
from marin.utils import remove_tpu_lockfile_on_exit


class BaseVllmEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM."""

    _STREAMING_LOAD_FORMATS: ClassVar[set[str]] = {"runai_streamer", "runai_streamer_sharded"}
    _VLLM_DOCKER_IMAGE_ENV: ClassVar[str] = "MARIN_VLLM_DOCKER_IMAGE"
    _VLLM_GPU_DOCKER_IMAGE_ENV: ClassVar[str] = "MARIN_VLLM_GPU_DOCKER_IMAGE"
    _VLLM_TPU_DOCKER_IMAGE_ENV: ClassVar[str] = "MARIN_VLLM_TPU_DOCKER_IMAGE"
    _VLLM_SIDECAR_CONTAINER_PREFIX: ClassVar[str] = "marin-vllm-sidecar"
    _VLLM_USE_CLI_ENV: ClassVar[str] = "MARIN_VLLM_USE_CLI"
    _VLLM_SIDECAR_SHM_SIZE_ENV: ClassVar[str] = "MARIN_VLLM_SIDECAR_SHM_SIZE"

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
        if not BaseVllmEvaluator._is_object_store_path(model.path):
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
        max_model_len = engine_kwargs.get("max_model_len")
        if isinstance(max_model_len, int):
            args.extend(["--max-model-len", str(max_model_len)])
        gpu_memory_utilization = engine_kwargs.get("gpu_memory_utilization")
        if isinstance(gpu_memory_utilization, (int, float)):
            args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        return args

    @staticmethod
    def _vllm_server_command(
        model_name_or_path: str,
        host: str,
        port: int,
        engine_kwargs: dict,
        *,
        use_cli: bool,
    ) -> list[str]:
        if use_cli:
            return [
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
                *BaseVllmEvaluator._engine_kwargs_to_cli_args(engine_kwargs),
            ]
        return [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name_or_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(port),
            "--distributed-executor-backend",
            "ray",
            *BaseVllmEvaluator._engine_kwargs_to_cli_args(engine_kwargs),
        ]

    @staticmethod
    def _resolve_vllm_docker_image(resource_config: ResourceConfig | None) -> str | None:
        if resource_config is None:
            return os.environ.get(BaseVllmEvaluator._VLLM_DOCKER_IMAGE_ENV)
        if isinstance(resource_config.device, TpuConfig):
            return os.environ.get(BaseVllmEvaluator._VLLM_TPU_DOCKER_IMAGE_ENV) or os.environ.get(
                BaseVllmEvaluator._VLLM_DOCKER_IMAGE_ENV
            )
        if isinstance(resource_config.device, GpuConfig):
            return os.environ.get(BaseVllmEvaluator._VLLM_GPU_DOCKER_IMAGE_ENV) or os.environ.get(
                BaseVllmEvaluator._VLLM_DOCKER_IMAGE_ENV
            )
        return os.environ.get(BaseVllmEvaluator._VLLM_DOCKER_IMAGE_ENV)

    @staticmethod
    def _vllm_sidecar_container_name(port: int) -> str:
        return f"{BaseVllmEvaluator._VLLM_SIDECAR_CONTAINER_PREFIX}-{port}"

    @staticmethod
    def _write_env_file(env: dict[str, str]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            for key, value in env.items():
                if value is None:
                    continue
                handle.write(f"{key}={value}\n")
            return handle.name

    @staticmethod
    def resolve_model_name_or_path(model: ModelConfig) -> tuple[str, ModelConfig]:
        """Resolve the `model` argument to pass to vLLM.

        - If `model.path` is set, use it (and auto-enable streaming for `gs://` / `s3://`).
        - Otherwise, fall back to `model.name` (e.g. an HF repo id).
        """
        model = BaseVllmEvaluator._maybe_enable_streaming(model)
        model_name_or_path = model.path if model.path is not None else model.name
        return model_name_or_path, model

    @staticmethod
    def start_vllm_server_in_background(
        model: ModelConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout_seconds: int = 3600,
        resource_config: ResourceConfig | None = None,
        docker_image: str | None = None,
    ) -> str:
        """
        Serve the model with a local vLLM server in the background.
        Returns the port the server is running on.
        """
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path, model = BaseVllmEvaluator.resolve_model_name_or_path(model)

        # From https://docs.vllm.ai/en/v0.4.0/models/engine_args.html
        use_cli = os.environ.get(BaseVllmEvaluator._VLLM_USE_CLI_ENV, "").lower() in {"1", "true", "yes"}
        if docker_image and "vllm-openai" in docker_image:
            use_cli = True

        command = BaseVllmEvaluator._vllm_server_command(
            model_name_or_path=model_name_or_path,
            host=host,
            port=port,
            engine_kwargs=model.engine_kwargs,
            use_cli=use_cli,
        )
        docker_image = docker_image or BaseVllmEvaluator._resolve_vllm_docker_image(resource_config)
        env = BaseVllmEvaluator._vllm_env()

        sidecar_container_name: str | None = None
        if docker_image:
            container_name = BaseVllmEvaluator._vllm_sidecar_container_name(port)
            docker_command = [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--network",
                "host",
                "--name",
                container_name,
            ]
            shm_size = os.environ.get(BaseVllmEvaluator._VLLM_SIDECAR_SHM_SIZE_ENV, "10g")
            if shm_size:
                docker_command.extend(["--shm-size", shm_size])
            if resource_config is not None:
                if isinstance(resource_config.device, TpuConfig):
                    docker_command.append("--privileged")
                elif isinstance(resource_config.device, GpuConfig):
                    docker_command.extend(["--gpus", str(resource_config.device.count)])

            env_file_path = BaseVllmEvaluator._write_env_file(env)
            try:
                docker_command.extend(["--env-file", env_file_path])
                if model.path and not BaseVllmEvaluator._is_object_store_path(model.path):
                    docker_command.extend(["-v", f"{model.path}:{model.path}"])
                entrypoint = command[0]
                entrypoint_args = command[1:]
                docker_command.extend(["--entrypoint", entrypoint, docker_image, *entrypoint_args])
                result = subprocess.run(docker_command, check=False, capture_output=True, text=True)
                if result.returncode != 0 and entrypoint == "python" and "executable file not found" in result.stderr:
                    docker_command[-(len(entrypoint_args) + 2)] = "python3"
                    result = subprocess.run(docker_command, check=True)
                else:
                    result.check_returncode()
            finally:
                try:
                    os.unlink(env_file_path)
                except OSError:
                    pass
            process = None
            sidecar_container_name = container_name
        else:
            process = subprocess.Popen(command, env=env)

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
                if sidecar_container_name is not None:
                    subprocess.run(["docker", "rm", "-f", sidecar_container_name], check=False)
                if process is not None:
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
                container_name = BaseVllmEvaluator._vllm_sidecar_container_name(vllm_port)
                subprocess.run(["docker", "rm", "-f", container_name], check=False)
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

        extras = ["eval"]
        if isinstance(resource_config.device, TpuConfig):
            extras.append("tpu")
        elif isinstance(resource_config.device, GpuConfig):
            extras.append("gpu")

        env_vars: dict[str, str] = {}
        docker_image = BaseVllmEvaluator._resolve_vllm_docker_image(resource_config)
        if docker_image:
            env_vars[BaseVllmEvaluator._VLLM_DOCKER_IMAGE_ENV] = docker_image

        job_request = JobRequest(
            name="vllm-tpu-evaluation",
            entrypoint=Entrypoint.from_callable(_run),
            resources=resource_config,
            environment=EnvironmentConfig.create(
                extras=extras,
                env_vars=env_vars if env_vars else None,
            ),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
