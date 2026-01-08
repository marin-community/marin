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
import shutil
import subprocess
import tempfile
import time
from abc import ABC
from dataclasses import dataclass
from typing import Literal

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.utils import remove_tpu_lockfile_on_exit
from marin.vllm.docker_server import (
    VllmDockerServerConfig,
    start_vllm_docker_server,
    stop_vllm_docker_server_by_name,
)


@dataclass(frozen=True)
class VllmServerHandle:
    server_url: str
    port: int
    mode: Literal["native", "docker"]
    process: subprocess.Popen[str] | None = None
    log_dir: str | None = None
    docker_container_name: str | None = None


class VllmTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/tmp"
    HOST_GCSFUSE_MOUNT_ENV: str = "MARIN_VLLM_HOST_GCSFUSE_MOUNT"
    VLLM_NATIVE_PIP_PACKAGES: tuple[str, ...] = ("vllm-tpu",)

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
        port: int | None = None,
        timeout_seconds: int = 3600,
        extra_args: list[str] | None = None,
        mode: Literal["native", "docker"] | None = None,
        docker_image: str | None = None,
        docker_run_args: list[str] | None = None,
    ) -> VllmServerHandle:
        """
        Serve the model with a local vLLM server in the background.
        Returns a handle that can be used to stop the server.
        """
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = VllmTpuEvaluator.download_model(model)

        mode_str = (mode if mode is not None else os.environ.get("MARIN_VLLM_MODE", "docker")).lower()
        if mode_str not in ("native", "docker"):
            raise ValueError(f"Unknown MARIN_VLLM_MODE={mode_str!r}; expected 'native' or 'docker'.")
        if mode_str == "docker":
            resolved_image = docker_image or os.environ.get("MARIN_VLLM_DOCKER_IMAGE")
            if not resolved_image:
                raise RuntimeError(
                    "MARIN_VLLM_DOCKER_IMAGE is required when MARIN_VLLM_MODE=docker. "
                    "Set it to a vllm-tpu image tag, e.g. vllm/vllm-tpu:<tag>."
                )

            volumes: list[tuple[str, str]] = [("/tmp", "/tmp")]
            host_gcsfuse_mount = os.environ.get(VllmTpuEvaluator.HOST_GCSFUSE_MOUNT_ENV)
            if not host_gcsfuse_mount and os.path.isdir("/tmp/gcsfuse_mount"):
                # Cluster configs mount GCS once on the host at `/tmp/gcsfuse_mount` and expose it
                # inside the Ray container via a symlink at `/opt/gcsfuse_mount`. Since the Docker
                # daemon sees the host filesystem, we must bind-mount the host path (under `/tmp`),
                # not the in-container symlink path (under `/opt`).
                host_gcsfuse_mount = "/tmp/gcsfuse_mount"
            if host_gcsfuse_mount:
                volumes.append((host_gcsfuse_mount, "/opt/gcsfuse_mount"))
            elif model_name_or_path.startswith("/opt/gcsfuse_mount/"):
                raise RuntimeError(
                    f"Model path {model_name_or_path!r} is under /opt/gcsfuse_mount, but "
                    f"{VllmTpuEvaluator.HOST_GCSFUSE_MOUNT_ENV} is not set. When using docker sidecars, "
                    "any host mounts must be visible to the Docker daemon (host filesystem). "
                    "Either set the env var to a host path that contains the model (and is gcsfuse-mounted on the host),"
                    "or ensure the host has GCS mounted at /tmp/gcsfuse_mount, "
                    "or use an HF repo id and let vLLM download inside the sidecar."
                )

            if (
                os.path.isabs(model_name_or_path)
                and os.path.exists(model_name_or_path)
                and not (
                    model_name_or_path.startswith("/tmp/")
                    or model_name_or_path == "/tmp"
                    or model_name_or_path.startswith("/opt/gcsfuse_mount/")
                    or model_name_or_path == "/opt/gcsfuse_mount"
                )
            ):
                volumes.append((model_name_or_path, model_name_or_path))

            env: dict[str, str] = {"TOKENIZERS_PARALLELISM": "false"}
            for key in ("HF_TOKEN", "WANDB_API_KEY", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
                value = os.environ.get(key)
                if value:
                    env[key] = value

            if host_gcsfuse_mount and "HF_HOME" not in env:
                env["HF_HOME"] = "/opt/gcsfuse_mount/hf-cache"
                env.setdefault("HF_HUB_CACHE", "/opt/gcsfuse_mount/hf-cache/hub")
                env.setdefault("TRANSFORMERS_CACHE", "/opt/gcsfuse_mount/hf-cache")

            docker_args = ["--privileged"]
            if docker_run_args:
                docker_args.extend(docker_run_args)

            config = VllmDockerServerConfig(
                image=resolved_image,
                model_name_or_path=model_name_or_path,
                host=host,
                port=port,
                env=env,
                volumes=volumes,
                extra_vllm_args=(extra_args or []),
                docker_run_args=docker_args,
            )
            docker_handle = start_vllm_docker_server(config, timeout_seconds=timeout_seconds)
            return VllmServerHandle(
                server_url=docker_handle.server_url,
                port=docker_handle.port,
                mode="docker",
                docker_container_name=docker_handle.container_name,
            )

        vllm_bin = shutil.which("vllm")
        if vllm_bin is None:
            raise RuntimeError(
                "`vllm` CLI not found on PATH. Install vLLM in the job runtime environment "
                "(e.g. via `EnvironmentConfig(pip_packages=[...])`) before using VllmTpuEvaluator."
            )

        resolved_port = port if port is not None else 8000

        cmd: list[str] = [
            vllm_bin,
            "serve",
            model_name_or_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(resolved_port),
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
        server_url: str = f"http://{host}:{resolved_port}/v1"
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
        return VllmServerHandle(
            server_url=server_url,
            port=resolved_port,
            mode="native",
            process=process,
            log_dir=log_dir,
        )

    @staticmethod
    def cleanup(
        model: ModelConfig,
        vllm_server: VllmServerHandle | None = None,
    ) -> None:
        """
        Clean up the vLLM server and any other resources.
        """
        print("Cleaning up resources.")
        # Kill the vLLM server
        try:
            if vllm_server is not None and vllm_server.mode == "docker" and vllm_server.docker_container_name:
                stop_vllm_docker_server_by_name(vllm_server.docker_container_name)
            elif vllm_server is not None and vllm_server.mode == "native":
                if vllm_server.process is not None:
                    vllm_server.process.kill()
        except Exception as e:
            print(f"Failed to kill vLLM server: {e}")

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
                pip_packages=(
                    self.VLLM_NATIVE_PIP_PACKAGES
                    if os.environ.get("MARIN_VLLM_MODE", "docker").lower() == "native"
                    else ()
                ),
            ),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
