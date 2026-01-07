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
#1600: Perplexity Correlation

This experiment evaluates the correlation between perplexity and other metrics to check the quality of uncheatable evals.
"""

import logging
import os
from functools import lru_cache

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.evals.task_configs import EvalTaskConfig
from experiments.isoflop_sweep import create_isoflop_sweep_steps
from experiments.llama import llama3_tokenizer
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from experiments.paloma import paloma_tokenized
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.scaling_laws.recipe import MARIN_2025_RECIPE

# Import shared components from exp1600_uncheatable_evals
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
    truncate_model_name,
    uncheatable_eval_tokenized,
)

logger = logging.getLogger(__name__)

EVAL_TASKS = [
    EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset,
    EvalTaskConfig("gsm8k_loss", num_fewshot=8, task_alias="gsm8k_loss_8shot"),
    EvalTaskConfig("math_500_loss", num_fewshot=0),
]


@lru_cache(maxsize=1)
def build_steps():
    steps = []
    isoflop_steps, isoflop_candidates = create_isoflop_sweep_steps(
        nemotron_mix,
        experiment_name="nemo-wider-depth-adapt",
        recipe=MARIN_2025_RECIPE,
    )
    for isoflop_step, candidate in zip(isoflop_steps, isoflop_candidates, strict=False):
        experiment_name = isoflop_step.name.split("/")[-1]
        paloma_tokenized_dict = paloma_tokenized(tokenizer=llama3_tokenizer)
        uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=llama3_tokenizer)
        eval_data = mixture_for_evaluation(paloma_tokenized_dict | uncheatable_eval_tokenized_dict)
        wandb_tags = [
            f"FLOPs={candidate.flops_budget:.1e}",
            f"d={candidate.hidden_size}",
            f"L={candidate.num_layers}",
            f"B={candidate.batch_size}",
            f"steps={candidate.train_steps}",
        ]
        model_config = isoflop_step.config.train_config.model
        checkpoint_path = output_path_of(isoflop_step)
        steps.append(
            default_lm_log_probs(
                checkpoint=checkpoint_path,
                model=model_config,
                data=eval_data,
                resource_config=ResourceConfig.with_tpu("v5p-8"),
                checkpoint_is_hf=False,
                per_device_batch_size=4,
                name=f"{experiment_name}-paloma-uncheatable-eval-logprobs-v2",
                wandb_tags=wandb_tags,
            )
        )
        steps.append(
            evaluate_levanter_lm_evaluation_harness(
                model_name=experiment_name,
                model_path=checkpoint_path,
                evals=EVAL_TASKS,
                resource_config=ResourceConfig.with_tpu("v5p-8"),
                # wandb_tags=wandb_tags,
            )
        )

    for model_config in models:
        tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name
        paloma_tokenized_dict = paloma_tokenized(tokenizer=tokenizer)
        uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=tokenizer)
        eval_data = mixture_for_evaluation(paloma_tokenized_dict | uncheatable_eval_tokenized_dict)

        model_identifier = f"{model_config.model_name}@{model_config.revision}"
        model_instance = download_model_step(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )
        hf_model_config = HFCheckpointConverter.from_hf(model_identifier).config_from_hf_checkpoint(model_identifier)

        directory_friendly_name = get_directory_friendly_name(model_config.model_name)
        wandb_tags = [
            f"M={truncate_model_name(model_config.model_name)}",
            "eval=paloma-uncheatable-eval-bpb",
        ]
        steps.append(
            default_lm_log_probs(
                checkpoint=output_path_of(model_instance),
                model=hf_model_config,
                data=eval_data,
                resource_config=ResourceConfig.with_tpu("v5p-8"),
                checkpoint_is_hf=True,
                per_device_batch_size=4,
                name=f"{directory_friendly_name}-paloma-uncheatable-eval-logprobs-v2",
                wandb_tags=wandb_tags,
            )
        )

        steps.append(
            evaluate_levanter_lm_evaluation_harness(
                model_name=f"{directory_friendly_name}-mmlu-5shot-sl",
                model_path=output_path_of(model_instance),
                evals=EVAL_TASKS,
                resource_config=ResourceConfig.with_tpu("v5p-8"),
                # wandb_tags=[f"M={model_config.model_name}", "eval=mmlu-5shot-sl"],
            )
        )

    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    for step in build_steps():
        executor_main(steps=[step])


if __name__ == "__main__":
    main()
