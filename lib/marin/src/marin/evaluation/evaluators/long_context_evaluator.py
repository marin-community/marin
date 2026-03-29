# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence

import requests
from fray.v1.cluster import ResourceConfig

from marin.evaluation.benchmarks.long_context import evaluate_long_context_tasks, write_long_context_results
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig, launch_evaluate_with_ray
from marin.evaluation.utils import is_remote_path, upload_to_gcs
from marin.inference.vllm_server import VLLM_NATIVE_PIP_PACKAGES, VllmEnvironment, resolve_vllm_mode

logger = logging.getLogger(__name__)


def _env_vars_from_keys(keys: Sequence[str]) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


class LongContextEvaluator(Evaluator):
    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        resource_config: ResourceConfig,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        mode_str = resolve_vllm_mode(None)
        pip_packages = VLLM_NATIVE_PIP_PACKAGES if mode_str == "native" else ()
        env_vars = _env_vars_from_keys(
            [
                "HF_TOKEN",
                "MARIN_PREFIX",
                "MARIN_VLLM_MODE",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
                "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION",
                "VLLM_TPU_SKIP_PRECOMPILE",
            ]
        )
        env_vars.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        env_vars.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")
        env_vars.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")

        launch_evaluate_with_ray(
            evaluator=self,
            job_name="long-context-eval",
            model=model,
            evals=evals,
            output_path=output_path,
            resource_config=resource_config,
            max_eval_instances=max_eval_instances,
            wandb_tags=wandb_tags,
            extras=("eval", "tpu"),
            pip_packages=pip_packages,
            env_vars=env_vars,
        )

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        os.environ.setdefault("VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION", "1")
        os.environ.setdefault("VLLM_TPU_SKIP_PRECOMPILE", "1")

        with tempfile.TemporaryDirectory(prefix="marin-long-context-") as local_output_dir:
            with VllmEnvironment(model) as env:
                if env.model_id is None:
                    raise RuntimeError("Expected vLLM server to expose a model id.")

                results = evaluate_long_context_tasks(
                    evals,
                    completion_fn=lambda prompts, max_gen_toks: self._generate_predictions(
                        env=env,
                        model=model,
                        prompts=prompts,
                        max_gen_toks=max_gen_toks,
                    ),
                    max_eval_instances=max_eval_instances,
                )
                payload = {
                    "model_name": model.name,
                    "model_path": model.path,
                    "tasks": results,
                }
                write_long_context_results(local_output_dir, results)
                with open(os.path.join(local_output_dir, "metadata.json"), "w") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)

            if is_remote_path(output_path):
                upload_to_gcs(local_output_dir, output_path)
            else:
                shutil.copytree(local_output_dir, output_path, dirs_exist_ok=True)

    def _generate_predictions(
        self,
        *,
        env: VllmEnvironment,
        model: ModelConfig,
        prompts: list[str],
        max_gen_toks: int,
    ) -> list[str]:
        generation_params = dict(model.generation_params or {})
        temperature = float(generation_params.get("temperature", 0.0))
        top_p = float(generation_params.get("top_p", 1.0))
        timeout = int(generation_params.get("timeout", 1800))

        predictions: list[str] = []
        for prompt in prompts:
            if model.apply_chat_template:
                response = requests.post(
                    f"{env.server_url}/chat/completions",
                    json={
                        "model": env.model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_gen_toks,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                predictions.append(payload["choices"][0]["message"]["content"])
            else:
                response = requests.post(
                    f"{env.server_url}/completions",
                    json={
                        "model": env.model_id,
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_gen_toks,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                predictions.append(payload["choices"][0]["text"])

        return predictions
