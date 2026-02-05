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
Evaluate all ISOFlop models on logprob-based evaluation datasets.

This script evaluates ISOFlop models on paloma + uncheatable_eval datasets (logprobs only, no eval harness).
"""

import logging
import warnings

# Set these BEFORE any executor imports
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)  # Root logger

import os
import math

from levanter.compat.hf_checkpoints import HFCheckpointConverter

from experiments.llama import llama3_tokenizer
from experiments.isoflop_sweep import MARIN_SCALING_SUITES
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, output_path_of
from marin.execution.executor import Executor, ExecutorStep
from marin.evaluation.utils import discover_hf_checkpoints
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from fray.cluster import ResourceConfig
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
    truncate_model_name,
)
from experiments.evals.exp_math_reasoning_data import get_tokenized_data_steps as math_reasoning_tokenized
from experiments.evals.exp_isoflop_hf_math500 import math500_rollouts_tokenized

logger = logging.getLogger(__name__)


def build_steps(model_types: list[str]):
    math_reasoning_dict = math_reasoning_tokenized(tokenizer=llama3_tokenizer)
    math500_rollouts_dict = math500_rollouts_tokenized(tokenizer=llama3_tokenizer, prompt_format="standard_fewshot")
    eval_data = mixture_for_evaluation(math_reasoning_dict | math500_rollouts_dict)

    isoflop_steps, isoflop_candidates = MARIN_SCALING_SUITES["nemotron"]

    steps = []
    if "iso" in model_types:
        for isoflop_step, candidate in zip(isoflop_steps, isoflop_candidates, strict=False):
            experiment_name = isoflop_step.name.split("/")[-1]
            wandb_tags = [
                "model_type=isoflop-hf",
                f"FLOPs={candidate.flops_budget:.1e}",
                f"d={candidate.model_config.hidden_dim}",
                f"L={candidate.model_config.num_layers}",
                f"B={candidate.batch_size}",
                f"steps={candidate.train_steps}",
                "eval=math-reasoning-eval",
            ]
            model_config = isoflop_step.config.train_config.model
            # checkpoint_path = (output_path_of(isoflop_step)).nonblocking()
            checkpoint_path = get_isoflop_hf_model(
                isoflop_step=isoflop_step,
                prefix="gs://marin-us-central1"
            )
            steps.append(
                default_lm_log_probs(
                    checkpoint=checkpoint_path,
                    model=model_config,
                    data=eval_data,
                    resource_config=ResourceConfig.with_tpu("v5p-8"),
                    checkpoint_is_hf=True,
                    per_device_batch_size=4,
                    name=f"{experiment_name}-math-reasoning-eval-logprobs",
                    wandb_tags=wandb_tags,
                )
            )

    if "hf" in model_types:
        for model_config in models:
            tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name
            math_reasoning_dict = math_reasoning_tokenized(tokenizer=tokenizer)
            math500_rollouts_dict = math500_rollouts_tokenized(tokenizer=tokenizer, prompt_format="standard_fewshot")
            eval_data = mixture_for_evaluation(math_reasoning_dict | math500_rollouts_dict)

            model_identifier = f"{model_config.model_name}@{model_config.revision}"
            model_instance = download_model_step(
                HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
            )
            hf_model_config = HFCheckpointConverter.from_hf(model_identifier).config_from_hf_checkpoint(model_identifier)

            directory_friendly_name = get_directory_friendly_name(model_config.model_name)
            wandb_tags = [
                "model_type=hf",
                f"M={truncate_model_name(model_config.model_name)}",
                "eval=math-reasoning-eval",
            ]
            steps.append(
                default_lm_log_probs(
                    checkpoint=output_path_of(model_instance),
                    model=hf_model_config,
                    data=eval_data,
                    resource_config=ResourceConfig.with_tpu("v5p-8"),
                    checkpoint_is_hf=True,
                    per_device_batch_size=4,
                    name=f"{directory_friendly_name}-math-reasoning-eval-logprobs",
                    wandb_tags=wandb_tags,
                )
            )

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
    print(f"path: {path}")
    hf_path = os.path.join(path,"hf")
    print(f"hf_path: {hf_path}")
    checkpoints = discover_hf_checkpoints(base_path=hf_path)
    def get_step(checkpoint):
        return int(checkpoint.rsplit("step-", 1)[-1])

    return sorted(checkpoints, key=get_step)[-1]



def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return
    
    import warnings
    warnings.filterwarnings("ignore")

    import logging
    logging.getLogger("marin.execution.executor").setLevel(logging.ERROR)
    
    model_types = ["iso"] # ["iso", "hf"]
    steps = build_steps(model_types=model_types)
    
    # bsz = 10
    # n_batches = math.ceil(len(steps) / bsz)

    # for i in range(n_batches):
    #     batch = steps[i*bsz:(i+1)*bsz]
    #     executor_main(steps=batch)

    executor_main(
        steps=steps, 
        description="ISOFlop and HF model BPB evaluations."
    )


if __name__ == "__main__":
    main()

