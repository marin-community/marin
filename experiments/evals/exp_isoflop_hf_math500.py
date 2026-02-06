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
Evaluate all ISOFlop models on MATH-500 using vLLM.

This script evaluates ISOFlop models on MATH-500 accuracy.
"""

import logging
import warnings

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)  # Root logger

import os
import json
import fsspec

from experiments.isoflop_sweep import MARIN_SCALING_SUITES
from marin.execution.executor import executor_main, output_path_of, versioned
from marin.execution.executor import Executor, ExecutorStep, this_output_path
from marin.evaluation.utils import discover_hf_checkpoints
from fray.cluster import ResourceConfig
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
)
from experiments.evals.vllm_math500_eval import Math500EvalConfig, Math500ProcessConfig, run_math500_eval, process_math500_data
from experiments.defaults import default_tokenize
from levanter.data.text import ChatLmDatasetFormat

logger = logging.getLogger(__name__)

DEFAULT_CHAT_TEMPLATE = "{{messages[0]['content']}}{% generation %} {{messages[1]['content']}}{% endgeneration %}"


def build_hf_steps(prompt_format: str = "question_only"):
    steps = []
    for model_config in models:
        model_instance = download_model_step(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )
        directory_friendly_name = get_directory_friendly_name(model_config.model_name)
        name = f"{directory_friendly_name}"
        steps.append(
            ExecutorStep(
                name=f"analysis/math500_rollouts/{name}",
                fn=run_math500_eval,
                config=Math500EvalConfig(
                    model_path=output_path_of(model_instance),
                    output_path=this_output_path(),
                    prompt_format=versioned(prompt_format),
                ),
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "math"],
            )
        )
    
    return steps

def build_isoflop_steps(prompt_format: str = "question_only"):
    isoflop_steps, isoflop_candidates = MARIN_SCALING_SUITES["nemotron"]

    steps = []
    for isoflop_step, candidate in zip(isoflop_steps, isoflop_candidates, strict=False):
        experiment_name = isoflop_step.name.split("/")[-1]
        checkpoint_path = get_isoflop_hf_model(
            isoflop_step=isoflop_step,
            prefix="gs://marin-us-central1"
        )
        name = f"{experiment_name}"
        steps.append(
            ExecutorStep(
                name=f"analysis/math500_rollouts/{name}",
                fn=run_math500_eval,
                config=Math500EvalConfig(
                    model_path=checkpoint_path,
                    output_path=this_output_path(),
                    prompt_format=versioned(prompt_format),
                ),
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "math"],
            )
        )
    
    return steps

def build_steps(model_types: list[str], prompt_format: str = "question_only"):
    steps = []
    if "iso" in model_types:
        isoflop_steps = build_isoflop_steps(prompt_format=prompt_format)
        steps.extend(isoflop_steps)
    
    if "hf" in model_types:
        hf_steps = build_hf_steps(prompt_format=prompt_format)
        steps.extend(hf_steps)

    return steps


def get_step_output_path(step: ExecutorStep, prefix: str) -> str:
    """
    Get the output path for an ExecutorStep.

    Args:
        step: The ExecutorStep to get the output path for
        prefix: The prefix to use. If None, uses MARIN_PREFIX env var.

    Returns:
        The output path as a string.
    """

    executor_info_base_path = os.path.join(prefix, "experiments")
    executor = Executor(
        prefix=prefix,
        executor_info_base_path=executor_info_base_path,
    )
    executor.compute_version(step, is_pseudo_dep=False)
    return executor.output_paths[step]


def get_isoflop_hf_model(isoflop_step: ExecutorStep, prefix: str):
    path = get_step_output_path(isoflop_step, prefix)
    hf_path = os.path.join(path, "hf")

    checkpoints = discover_hf_checkpoints(base_path=hf_path)
    def get_step(checkpoint):
        return int(checkpoint.rsplit("step-", 1)[-1])

    return sorted(checkpoints, key=get_step)[-1]


def exists_correct_and_incorrect(eval_step: ExecutorStep) -> tuple[bool, bool]:
    eval_output_path = get_step_output_path(eval_step, prefix="gs://marin-us-central1")
    results_file = os.path.join(eval_output_path, "results.json.gz")

    with fsspec.open(results_file, "rt", compression="gzip") as f:
        eval_results = json.load(f)

        has_correct = any(
            sample["correct"]
            for result in eval_results["results"]
            for sample in result["samples"]
        )
        has_incorrect = any(
            not sample["correct"]
            for result in eval_results["results"]
            for sample in result["samples"]
        )

        return has_correct, has_incorrect


def math500_rollouts_tokenized(tokenizer: str, prompt_format: str = "question_only") -> dict[str, ExecutorStep]:
    result = {}

    hf_steps = build_hf_steps(prompt_format=prompt_format)
    for eval_step in hf_steps:
        name = eval_step.name.split("/")[-1]

        has_correct, has_incorrect = exists_correct_and_incorrect(eval_step)
        for filter_type in ["all", "correct", "incorrect"]:
            if filter_type == "correct" and not has_correct:
                continue
            if filter_type == "incorrect" and not has_incorrect:
                continue

            process_step = ExecutorStep(
                name=f"documents/math500_rollouts/{name}/{filter_type}",
                fn=process_math500_data,
                config=Math500ProcessConfig(
                    eval_path=output_path_of(eval_step),
                    output_path=this_output_path(),
                    filter=filter_type,
                ),
            )

            tokenized_step = default_tokenize(
                name=f"math500_rollouts/{name}/{filter_type}",
                dataset=output_path_of(process_step),
                tokenizer=tokenizer,
                is_validation=True,
                format=ChatLmDatasetFormat(chat_template=DEFAULT_CHAT_TEMPLATE),
            )

            result[f"{name}_filter_{filter_type}"] = tokenized_step

    return result


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    import warnings
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)

    model_types = ["hf"]
    prompt_format = "standard_fewshot"
    steps = build_steps(model_types=model_types, prompt_format=prompt_format)

    executor_main(
        steps=steps,
        description="ISOFlop and HF model MATH-500 evaluations."
    )


if __name__ == "__main__":
    main()
