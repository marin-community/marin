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
import shutil
import subprocess
import tempfile
import time
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Literal
from urllib.parse import urlparse

import requests
from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.utils import remove_tpu_lockfile_on_exit
from marin.vllm.docker_server import (
    DEFAULT_VLLM_DOCKER_IMAGE,
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
    """For `Evaluator`s that runs inference with vLLM on TPUs."""

    _STREAMING_LOAD_FORMATS: ClassVar[set[str]] = {"runai_streamer", "runai_streamer_sharded"}
    VLLM_NATIVE_PIP_PACKAGES: tuple[str, ...] = ("vllm-tpu",)

    @staticmethod
    def _resolve_vllm_mode(mode: Literal["native", "docker"] | None) -> Literal["native", "docker"]:
        mode_str = (mode if mode is not None else os.environ.get("MARIN_VLLM_MODE", "docker")).lower()
        if mode_str not in ("native", "docker"):
            raise ValueError(f"Unknown MARIN_VLLM_MODE={mode_str!r}; expected 'native' or 'docker'.")
        return mode_str  # type: ignore[return-value]

    @staticmethod
    def _default_jax_compilation_cache_dir() -> str:
        marin_prefix = os.environ.get("MARIN_PREFIX")
        if marin_prefix:
            return os.path.join(marin_prefix, "compilation-cache")
        return "/tmp/marin-jax-compilation-cache"

    @staticmethod
    def _vllm_env() -> dict[str, str]:
        env = dict(os.environ)
        # tpu_inference defaults MODEL_IMPL_TYPE=auto, which selects flax_nnx for many
        # architectures. flax_nnx currently fails without an auto mesh context, so
        # default to the vllm implementation unless the user overrides it.
        env.setdefault("MODEL_IMPL_TYPE", "vllm")
        # Reduce TPU runtime logging noise by default (match training defaults).
        env.setdefault("TPU_MIN_LOG_LEVEL", "3")
        env.setdefault("TPU_STDERR_LOG_LEVEL", "3")
        env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
        env.setdefault("JAX_COMPILATION_CACHE_DIR", VllmTpuEvaluator._default_jax_compilation_cache_dir())
        # vllm-tpu uses XLA compilation caches; this env var is the one it keys off.
        env.setdefault("VLLM_XLA_CACHE_PATH", env["JAX_COMPILATION_CACHE_DIR"])
        # Cache aggressively for iterative bring-up workflows.
        env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
        env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2")
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
        max_model_len = engine_kwargs.get("max_model_len")
        if isinstance(max_model_len, int) and max_model_len > 0:
            args.extend(["--max-model-len", str(max_model_len)])
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
    def download_model(model: ModelConfig) -> str:
        """Resolve and (if needed) stage a model for vLLM.

        Historically, Marin evaluators used gcsfuse-backed local paths. In the fuseless
        setup, vLLM can stream weights directly from object storage (`gs://` / `s3://`)
        when `load_format=runai_streamer`. This helper keeps the old API surface for
        callers that expect `download_model(...)` to exist.
        """
        model_name_or_path, model = VllmTpuEvaluator.resolve_model_name_or_path(model)
        model.ensure_downloaded()
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
        """Start `vllm serve` and wait until `/v1/models` responds.

        Defaults to docker mode unless overridden by `mode` or `MARIN_VLLM_MODE`.
        """

        model_name_or_path, model = VllmTpuEvaluator.resolve_model_name_or_path(model)

        mode_str = VllmTpuEvaluator._resolve_vllm_mode(mode)

        engine_cli_args = VllmTpuEvaluator._engine_kwargs_to_cli_args(model.engine_kwargs)
        extra_cli_args = [*engine_cli_args, *(extra_args or [])]

        if mode_str == "docker":
            return VllmTpuEvaluator.start_vllm_docker_sidecar_in_background(
                model_name_or_path=model_name_or_path,
                host=host,
                port=port,
                timeout_seconds=timeout_seconds,
                extra_cli_args=extra_cli_args,
                docker_image=docker_image,
                docker_run_args=docker_run_args,
            )

        return VllmTpuEvaluator.start_vllm_native_server_in_background(
            model_name_or_path=model_name_or_path,
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            extra_cli_args=extra_cli_args,
        )

    @staticmethod
    def start_vllm_docker_sidecar_in_background(
        *,
        model_name_or_path: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_seconds: int = 3600,
        extra_cli_args: list[str] | None = None,
        docker_image: str | None = None,
        docker_run_args: list[str] | None = None,
    ) -> VllmServerHandle:
        resolved_image = docker_image or os.environ.get("MARIN_VLLM_DOCKER_IMAGE") or DEFAULT_VLLM_DOCKER_IMAGE
        if "MARIN_VLLM_DOCKER_IMAGE" not in os.environ and docker_image is None:
            print(f"MARIN_VLLM_DOCKER_IMAGE not set; defaulting to {resolved_image}")

        env: dict[str, str] = {
            "TOKENIZERS_PARALLELISM": "false",
            # See `_vllm_env`.
            "MODEL_IMPL_TYPE": os.environ.get("MODEL_IMPL_TYPE", "vllm"),
            "JAX_ENABLE_COMPILATION_CACHE": os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "1"),
            "JAX_COMPILATION_CACHE_DIR": os.environ.get(
                "JAX_COMPILATION_CACHE_DIR",
                VllmTpuEvaluator._default_jax_compilation_cache_dir(),
            ),
            "VLLM_XLA_CACHE_PATH": os.environ.get(
                "VLLM_XLA_CACHE_PATH",
                os.environ.get("JAX_COMPILATION_CACHE_DIR", VllmTpuEvaluator._default_jax_compilation_cache_dir()),
            ),
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": os.environ.get(
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1"
            ),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": os.environ.get(
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2"
            ),
        }
        explain_cache_misses = os.environ.get("JAX_EXPLAIN_CACHE_MISSES")
        if explain_cache_misses is not None:
            env["JAX_EXPLAIN_CACHE_MISSES"] = explain_cache_misses
        env["TPU_MIN_LOG_LEVEL"] = os.environ.get("TPU_MIN_LOG_LEVEL", "3")
        env["TPU_STDERR_LOG_LEVEL"] = os.environ.get("TPU_STDERR_LOG_LEVEL", "3")
        for key in ("HF_TOKEN", "WANDB_API_KEY"):
            value = os.environ.get(key)
            if value:
                env[key] = value

        docker_args = ["--privileged"]
        if not any(arg.startswith("--shm-size") for arg in docker_run_args or []):
            docker_args.append("--shm-size=200gb")
        if docker_run_args:
            docker_args.extend(docker_run_args)

        config = VllmDockerServerConfig(
            image=resolved_image,
            model_name_or_path=model_name_or_path,
            host=host,
            port=port,
            env=env,
            volumes=[("/tmp", "/tmp")],
            extra_vllm_args=list(extra_cli_args or []),
            docker_run_args=docker_args,
        )
        print(
            "Starting vLLM Docker sidecar with "
            f"TPU_MIN_LOG_LEVEL={env.get('TPU_MIN_LOG_LEVEL')} "
            f"TPU_STDERR_LOG_LEVEL={env.get('TPU_STDERR_LOG_LEVEL')}"
        )
        docker_handle = start_vllm_docker_server(config, timeout_seconds=timeout_seconds)
        return VllmServerHandle(
            server_url=docker_handle.server_url,
            port=docker_handle.port,
            mode="docker",
            docker_container_name=docker_handle.container_name,
        )

    @staticmethod
    def start_vllm_native_server_in_background(
        *,
        model_name_or_path: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_seconds: int = 3600,
        extra_cli_args: list[str] | None = None,
    ) -> VllmServerHandle:
        resolved_port = port if port is not None else 8000

        vllm_bin = shutil.which("vllm") or "vllm"
        cmd: list[str] = [
            vllm_bin,
            "serve",
            model_name_or_path,
            "--trust-remote-code",
            "--host",
            host,
            "--port",
            str(resolved_port),
            *(extra_cli_args or []),
        ]

        log_dir = tempfile.mkdtemp(prefix="vllm_server_")
        stdout_path = os.path.join(log_dir, "stdout.log")
        stderr_path = os.path.join(log_dir, "stderr.log")
        stdout_f = open(stdout_path, "w")
        stderr_f = open(stderr_path, "w")
        native_env = VllmTpuEvaluator._vllm_env()
        print(
            "Starting vLLM native server with "
            f"TPU_MIN_LOG_LEVEL={native_env.get('TPU_MIN_LOG_LEVEL')} "
            f"TPU_STDERR_LOG_LEVEL={native_env.get('TPU_STDERR_LOG_LEVEL')}"
        )
        process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True, env=native_env)

        server_url: str = f"http://{host}:{resolved_port}/v1"
        start_time: float = time.time()
        elapsed_time: float = 0.0

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
                response = requests.get(f"{server_url}/models", timeout=5)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                process.kill()
                stdout_f.close()
                stderr_f.close()
                raise TimeoutError("Failed to start vLLM server within timeout period.")

            time.sleep(5)

        stdout_f.close()
        stderr_f.close()
        return VllmServerHandle(
            server_url=server_url,
            port=resolved_port,
            mode="native",
            process=process,
            log_dir=log_dir,
        )

    @staticmethod
    def cleanup(model: ModelConfig, vllm_server: VllmServerHandle | None = None) -> None:
        """Clean up the vLLM server and any other resources."""

        print("Cleaning up resources.")
        try:
            if vllm_server is not None and vllm_server.mode == "docker" and vllm_server.docker_container_name:
                stop_vllm_docker_server_by_name(vllm_server.docker_container_name)
            elif vllm_server is not None and vllm_server.mode == "native" and vllm_server.process is not None:
                vllm_server.process.kill()
        except Exception as e:
            print(f"Failed to stop vLLM server: {e}")

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
        """Launch the evaluation run with Fray."""

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

        def _run() -> None:
            with remove_tpu_lockfile_on_exit():
                launch(model, evals, output_path, max_eval_instances, wandb_tags)

        if resource_config is None:
            resource_config = ResourceConfig()

        mode_str = os.environ.get("MARIN_VLLM_MODE", "docker").lower()

        job_request = JobRequest(
            name="vllm-tpu-evaluation",
            entrypoint=Entrypoint.from_callable(_run),
            resources=resource_config,
            environment=EnvironmentConfig.create(
                extras=["eval", "tpu"],
                pip_packages=(self.VLLM_NATIVE_PIP_PACKAGES if mode_str == "native" else ()),
            ),
        )

        cluster = current_cluster()
        job_id = cluster.launch(job_request)
        cluster.wait(job_id, raise_on_failure=True)
