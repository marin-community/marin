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
from abc import ABC
from typing import ClassVar, Literal
from urllib.parse import urlparse

from fray.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig
from marin.utils import remove_tpu_lockfile_on_exit
from marin.vllm.vllm_server import (
    VllmServerHandle,
    start_vllm_server_in_background as start_vllm_server_process,
)


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

        return start_vllm_server_process(
            model_name_or_path=model_name_or_path,
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            extra_cli_args=extra_cli_args,
            mode=mode_str,
            docker_image=docker_image,
            docker_run_args=docker_run_args,
        )

    @staticmethod
    def cleanup(model: ModelConfig, vllm_server: VllmServerHandle | None = None) -> None:
        """Clean up the vLLM server and any other resources."""

        print("Cleaning up resources.")
        try:
            if vllm_server is not None:
                vllm_server.stop()
        except Exception as e:
            print(f"Failed to stop vLLM server: {e}")

        model.cleanup()

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
