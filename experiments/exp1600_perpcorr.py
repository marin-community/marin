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

from experiments.llama import llama3_tokenizer
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from experiments.paloma import paloma_tokenized

from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from experiments.evals.task_configs import EvalTaskConfig
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.isoflop_sweep import generate_isoflop_sweep
from experiments.tootsie.exp1295_32b import nemotron_mix

# Import shared components from exp1600_uncheatable_evals
from experiments.evals.exp1600_uncheatable_evals import (
    models,
    get_directory_friendly_name,
    uncheatable_eval_tokenized,
)

"""
#1600: Perplexity Correlation

This experiment evaluates the correlation between perplexity and other metrics to check the quality of uncheatable evals.
"""


EVAL_TASKS = [
    EvalTaskConfig("mmlu_sl_verb", num_fewshot=5, task_alias="mmlu_sl_verb_5_shot"),
    EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot"),  # 4-way MCQ commonsense reasoning dataset,
    EvalTaskConfig("gsm8k_loss", num_fewshot=8, task_alias="gsm8k_loss_8shot"),
    EvalTaskConfig("math_500_loss", num_fewshot=0),
]

steps = []
isoflop_steps, isoflop_metadatas = generate_isoflop_sweep(
    nemotron_mix,
    experiment_name="nemo-wider-depth-adapt",
)
for isoflop_step, isoflop_metadata in zip(isoflop_steps, isoflop_metadatas, strict=False):
    experiment_name = isoflop_step.name.split("/")[-1]
    paloma_tokenized_dict = paloma_tokenized(tokenizer=llama3_tokenizer)
    uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=llama3_tokenizer)
    eval_data = mixture_for_evaluation(paloma_tokenized_dict | uncheatable_eval_tokenized_dict)
    budget, hidden_size, num_layers, batch_size, train_steps = isoflop_metadata
    wandb_tags = (
        f"FLOPs={budget:.1e}",
        f"d={hidden_size}",
        f"L={num_layers}",
        f"B={batch_size}",
        f"steps={train_steps}",
    )
    steps.append(
        default_lm_log_probs(
            checkpoint=isoflop_step,
            model=isoflop_metadata,
            data=eval_data,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            checkpoint_is_hf=False,
            per_device_batch_size=4,
            name=f"{experiment_name}-paloma-uncheatable-eval-logprobs-v2",
            wandb_tags=wandb_tags,
        )
    )
    steps.append(
        evaluate_levanter_lm_evaluation_harness(
            model_name=experiment_name,
            model_path=isoflop_step,
            evals=EVAL_TASKS,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            # wandb_tags=wandb_tags,
        )
    )

for model_config in model_with_config:
    paloma_tokenized_dict = paloma_tokenized(tokenizer=model_config.tokenizer)
    uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=model_config.tokenizer)
    eval_data = mixture_for_evaluation(paloma_tokenized_dict | uncheatable_eval_tokenized_dict)

    directory_friendly_name = get_directory_friendly_name(model_config.model_name)
    steps.append(
        default_lm_log_probs(
            checkpoint=model_config.model_name,
            model=model_config.model_config,
            data=eval_data,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            checkpoint_is_hf=True,
            per_device_batch_size=4,
            name=f"{directory_friendly_name}-paloma-uncheatable-eval-logprobs-v2",
            wandb_tags=[f"M={model_config.model_name}", "eval=paloma-uncheatable-eval-bpb"],
        )
    )

    steps.append(
        evaluate_levanter_lm_evaluation_harness(
            model_name=f"{directory_friendly_name}-mmlu-5shot-sl",
            model_path=model_config.model_path,
            evals=EVAL_TASKS,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            # wandb_tags=[f"M={model_config.model_name}", "eval=mmlu-5shot-sl"],
        )
    )


if __name__ == "__main__":
    for step in steps:
        executor_main(steps=[step])
