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

from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import mixture_for_evaluation

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from experiments.models import (
    marin_8b_base_config,
    # olmo_2_base_32b_config,
    # qwen3_32b_config,
    # llama_3_1_8b_config,
    # llama_3_2_1b_config,
    # olmo_2_base_8b_config,
    # qwen3_0_6b_config,
    # qwen3_1_7b_config,
    # qwen3_4b_config,
    # qwen3_8b_config,
    # qwen3_0_6b_base_config,
    # qwen3_1_7b_base_config,
    # qwen3_4b_base_config,
    # qwen3_8b_base_config,
)
from experiments.uncheatable_eval import uncheatable_eval_tokenized


model_steps = [
    marin_8b_base_config,
    # olmo_2_base_32b_config,
    # qwen3_32b_config,
    # llama_3_1_8b_config,
    # llama_3_2_1b_config,
    # olmo_2_base_8b_config,
    # qwen3_0_6b_config,
    # qwen3_1_7b_config,
    # qwen3_4b_config,
    # qwen3_8b_config,
    # qwen3_0_6b_base_config,
    # qwen3_1_7b_base_config,
    # qwen3_4b_base_config,
    # qwen3_8b_base_config,
]


def get_directory_friendly_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(".", "-")


steps = []
for model_step in model_steps:
    uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=model_step.hf_repo_id)
    eval_data = mixture_for_evaluation(uncheatable_eval_tokenized_dict)
    model_config = HFCheckpointConverter.from_hf(f"{model_step.hf_repo_id}@{model_step.hf_revision}").config_from_hf_checkpoint(
                    f"{model_step.hf_repo_id}@{model_step.hf_revision}"
                )
    print(f"model_config: {model_config}")
    directory_friendly_name = get_directory_friendly_name(model_step.hf_repo_id)
    print(f"directory_friendly_name: {directory_friendly_name}")
    steps.append(
        default_lm_log_probs(
            checkpoint=f"{model_step.hf_repo_id}@{model_step.hf_revision}",
            model=model_config,
            data=eval_data,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            checkpoint_is_hf=True,
            per_device_batch_size=1,
            name=f"{directory_friendly_name}-uncheatable-eval-logprobs",
            wandb_tags=[
                f"M={model_step.hf_repo_id[:62] if len(model_step.hf_repo_id) > 62 else model_step.hf_repo_id}",
                "eval=uncheatable-eval",
            ],
        )
    )


if __name__ == "__main__":
    for step in steps:
        executor_main(steps=[step])
