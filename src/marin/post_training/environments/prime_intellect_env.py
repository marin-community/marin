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

"""
Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars
"""
import os
import json
import uuid
import time
from datetime import datetime
import logging
import numpy as np
import importlib
import subprocess
import importlib.util
from datasets import Dataset

import verifiers as vf
from typing import cast
from pathlib import Path
from verifiers.types import Endpoints
from verifiers.utils.client_utils import setup_client
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls

from .marin_env import EnvStep, InferenceContext, MarinEnv

logger = logging.getLogger(__name__)


def eval_environment(
    env: str,
    env_args: dict,
    env_dir_path: str,
    endpoints_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int | None,
    temperature: float | None,
    sampling_args: dict | None,
    verbose: bool,
    save_dataset: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
):
    logger.setLevel("DEBUG" if verbose else "INFO")
    try:
        endpoints_path_obj = Path(endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            logger.debug(f"Loading endpoint registry from {endpoints_file}")
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            # check that module exposes ENDPOINTS
            if not hasattr(endpoints_module, "ENDPOINTS"):
                raise AttributeError(f"Module '{endpoints_file}' does not have a 'ENDPOINTS' attribute")
            ENDPOINTS = cast(Endpoints, endpoints_module.ENDPOINTS)
            logger.debug(f"Successfully loaded {len(ENDPOINTS)} endpoints from registry")
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"No local endpoint registry found at {endpoints_path}. "
            f"Please specify the model name (-m), API host base URL (-b), and API key variable name (-k). "
            f"Error details: {e!s}"
        )
        logger.debug("Using default empty endpoints registry")
        ENDPOINTS: Endpoints = {}

    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]
        logger.debug(f"Using endpoint configuration for model '{model}' from registry")
    else:
        logger.debug(f"Model '{model}' not found in endpoint registry, using command-line arguments")

    # Setup eval client with high limits to prevent API timeout errors
    client = setup_client(
        api_base_url,
        api_key_var,
        timeout=3600.0,  # 1h
        max_connections=28000,  # Number of available ports
        max_keepalive_connections=28000,  # Number of available ports
        max_retries=10,  # 10 retries (w/ exponential backoffs)
    )
    logger.debug(f"Initialized OpenAI client with base_url: {api_base_url}")
    vf_env = vf.load_environment(env_id=env, **env_args)
    # Merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if sampling_args is not None:
        merged_sampling_args.update(sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = temperature

    logger.info(f"Starting evaluation with model: {model}")
    logger.info(
        f"Configuration: num_examples={num_examples}, \
            rollouts_per_example={rollouts_per_example}, \
            max_concurrent={max_concurrent}, \
            max_tokens={max_tokens}, \
            temperature={temperature}"
    )
    start_time = time.time()
    results = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args=merged_sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
    )
    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    print("--- Evaluation ---")
    print(f"Environment: {env}")
    print(f"Model: {model}")
    print(f"Provider: {api_base_url}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print("--- Example ---")
    printable_prompts = [messages_to_printable(p) for p in results.prompt]
    printable_completions = [messages_to_printable(c) for c in results.completion]
    vf.print_prompt_completions_sample(printable_prompts, printable_completions, results.reward, step=0)
    print("--- All ---")
    print("Rewards:")
    print(f"reward: avg - {sum(results.reward) / len(results.reward):.3f}, std - {np.std(results.reward):.3f}")
    r = rollouts_per_example
    n = len(results.reward) // r
    for i in range(r):
        # rounded to 3 decimal places
        trials = [round(results.reward[(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results.metrics:
        v = results.metrics[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)

    data_dict = {}
    if save_dataset or save_to_hf_hub:
        ids = [i // rollouts_per_example for i in range(n * rollouts_per_example)]
        rewards = results.reward
        tasks = results.task
        data_dict = {
            "id": ids,
            "prompt": [sanitize_tool_calls(p) for p in printable_prompts],
            "completion": [sanitize_tool_calls(c) for c in printable_completions],
            "task": tasks,
            "generation_ms": [s["timing"]["generation_ms"] for s in results.state],
            "scoring_ms": [s["timing"]["scoring_ms"] for s in results.state],
            "total_ms": [s["timing"]["total_ms"] for s in results.state],
        }
        if results.info[0] != {}:
            data_dict["info"] = results.info
        if results.answer[0] != "":
            data_dict["answer"] = results.answer
        data_dict["reward"] = rewards
        for k in results.metrics:
            v = results.metrics[k]
            data_dict[k] = v

        dataset = Dataset.from_dict(data_dict)
        metadata = {
            "env": env,
            "model": model,
            "num_examples": n,
            "rollouts_per_example": rollouts_per_example,
            "sampling_args": merged_sampling_args,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_ms": (end_time - start_time) * 1000,
            "avg_reward": sum(results.reward) / len(results.reward),
        }
        for k in results.metrics:
            metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

        uuid_str = str(uuid.uuid4())[:8]
        env_model_str = f"{env}--{model.replace('/', '--')}"
        if save_dataset:
            module_name = env.replace("-", "_")
            local_env_dir = Path(env_dir_path) / module_name
            print(local_env_dir)
            if local_env_dir.exists():
                results_path = local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
            else:
                results_path = Path("./outputs") / "evals" / env_model_str / uuid_str
            results_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_json(results_path / "results.jsonl")
            with open(results_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            logger.info(f"Saved dataset to {results_path}")
        if save_to_hf_hub:
            if hf_hub_dataset_name == "":
                dataset_name = f"{env}_{model.replace('/', '-')}_n{n}_r{rollouts_per_example}"
            else:
                dataset_name = hf_hub_dataset_name
            dataset.push_to_hub(dataset_name)
            logger.info(f"Saved dataset to Hugging Face Hub: {dataset_name}")

    return results, data_dict


class PrimeIntellectEnv(MarinEnv):
    """
    Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
    """

    def __init__(self, tokenizer, output_dir_path: str, **kwargs):
        self.tokenizer = tokenizer
        self._output_dir_path: str = os.path.join(output_dir_path)
        os.makedirs(self._output_dir_path, exist_ok=True)

    def step(
        self,
        env_id: str,
        inference_ctx: InferenceContext | None = None,
        mode: str = "train",
        **kwargs,
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            env_id: The ID of the environment to evaluate
            inference_ctx: The inference context
            mode: The mode to evaluate the environment
            kwargs: The keyword arguments
        """
        # Download the environment
        subprocess.run(["prime", "env", "install", env_id])
        env_id = env_id.split("/", 1)[-1]
        base_url = inference_ctx.inference_server.base_url if inference_ctx is not None else "https://api.openai.com/v1"

        result, data_dict = eval_environment(
            env=env_id,
            env_args=kwargs.get("env_args", {}),
            env_dir_path=kwargs.get("env_dir_path", "./environments"),
            endpoints_path=kwargs.get("endpoints_path", "./configs/endpoints.py"),
            model=kwargs.get("model", "gpt-4.1-mini"),
            api_key_var=kwargs.get("api_key_var", "OPENAI_API_KEY"),
            api_base_url=base_url,
            num_examples=kwargs.get("num_examples", 5),
            rollouts_per_example=kwargs.get("rollouts_per_example", 3),
            max_concurrent=kwargs.get("max_concurrent", 32),
            max_tokens=kwargs.get("max_tokens", None),
            temperature=kwargs.get("temperature", None),
            sampling_args=kwargs.get("sampling_args", None),
            verbose=kwargs.get("verbose", False),
            save_dataset=True,
            save_to_hf_hub=kwargs.get("save_to_hf_hub", False),
            hf_hub_dataset_name=kwargs.get("hf_hub_dataset_name", ""),
        )

        return EnvStep(
            examples=result.prompt, responses=result.completion, rewards=result.reward, metrics=result.metrics
        )
