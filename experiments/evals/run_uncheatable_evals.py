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

from experiments.llama import llama_3_2_1b as llama_3_2_1b_config, llama3_tokenizer, llama_8b
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import mixture_for_evaluation

from dataclasses import dataclass
from levanter.models.llama import LmConfig
from levanter.models.olmo import Olmo2Config
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from marin.execution.executor import ExecutorStep
from experiments.models import (
    llama_3_1_8b,
    olmo_2_base_8b,
    olmo_2_base_32b,
    marin_8b_base,
    llama_3_2_1b as llama_3_2_1b_model,
    qwen3_0_6b,
    qwen3_1_7b,
    qwen3_4b,
    qwen3_8b,
    qwen3_0_6b_base,
    qwen3_1_7b_base,
    qwen3_4b_base,
    qwen3_8b_base,
    qwen3_32b,
)
from experiments.qwen3 import (
    qwen3_0_6b as qwen3_0_6b_config,
    qwen3_1_7b as qwen3_1_7b_config,
    qwen3_4b as qwen3_4b_config,
    qwen3_8b as qwen3_8b_config,
    qwen3_32b as qwen3_32b_config,
    marin_32b as marin_32b_config,
)
from experiments.uncheatable_eval import uncheatable_eval_tokenized

UNCHEATABLE_EVAL_TO_FILE_PATTERN = {
    # "wikipedia_arabic": "wikipedia_arabic_*.jsonl.gz",
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    # "wikipedia_french": "wikipedia_french_*.jsonl.gz",
    # "wikipedia_german": "wikipedia_german_*.jsonl.gz",
    # "wikipedia_japanese": "wikipedia_japanese_*.jsonl.gz",
    # "wikipedia_spanish": "wikipedia_spanish_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    # "ao3_chinese": "ao3_chinese_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}

marin_32b_base = (
    "gs://marin-us-central1/checkpoints/tootsie-32b-cooldown-bison-adamc/hf/tootsie-32b-cooldown-bison-adamc/step-192000"
)

olmo_7b = Olmo2Config(
    seq_len=4096,
    hidden_dim=4096,
    intermediate_dim=11008,
    num_heads=32,
    num_kv_heads=32,
    num_layers=32,
)

olmo_32b = Olmo2Config(
    seq_len=4096,
    hidden_dim=5120,
    intermediate_dim=27648,
    num_heads=40,
    num_kv_heads=40,
    num_layers=64,
)


@dataclass
class ModelConfig:
    model_name: str
    model_config: LmConfig
    tokenizer: str
    model_path: ExecutorStep


# Evaluate log probabilities of meta-llama/Llama-3.2-1B on a subset of DCLM baseline
# Uses 1024 samples by default (adjust max_samples_per_dataset as needed)

model_with_config = [
    ModelConfig(
        model_name="marin-community/marin-8b-base",
        model_config=llama_8b,
        tokenizer=llama3_tokenizer,
        model_path=marin_8b_base,
    ),
    ModelConfig(
        model_name="marin-32b",
        model_config=marin_32b_config,
        tokenizer=llama3_tokenizer,
        model_path=marin_32b_base,
    ),
    ModelConfig(
        model_name="allenai/OLMo-2-0325-32B",
        model_config=olmo_32b,
        tokenizer="allenai/OLMo-2-0325-32B",
        model_path=olmo_2_base_32b,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-32B", model_config=qwen3_32b_config, tokenizer="Qwen/Qwen3-32B", model_path=qwen3_32b
    ),
    ModelConfig(
        model_name="meta-llama/Llama-3.1-8B", model_config=llama_8b, tokenizer=llama3_tokenizer, model_path=llama_3_1_8b
    ),
    ModelConfig(
        model_name="meta-llama/Llama-3.2-1B",
        model_config=llama_3_2_1b_config,
        tokenizer=llama3_tokenizer,
        model_path=llama_3_2_1b_model,
    ),
    ModelConfig(
        model_name="allenai/OLMo-2-1124-7B",
        model_config=olmo_7b,
        tokenizer="allenai/OLMo-2-1124-7B",
        model_path=olmo_2_base_8b,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-0.6B", model_config=qwen3_0_6b_config, tokenizer="Qwen/Qwen3-0.6B", model_path=qwen3_0_6b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-1.7B", model_config=qwen3_1_7b_config, tokenizer="Qwen/Qwen3-1.7B", model_path=qwen3_1_7b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-4B", model_config=qwen3_4b_config, tokenizer="Qwen/Qwen3-4B", model_path=qwen3_4b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-8B", model_config=qwen3_8b_config, tokenizer="Qwen/Qwen3-8B", model_path=qwen3_8b
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-0.6B-Base",
        model_config=qwen3_0_6b_config,
        tokenizer="Qwen/Qwen3-0.6B",
        model_path=qwen3_0_6b_base,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-1.7B-Base",
        model_config=qwen3_1_7b_config,
        tokenizer="Qwen/Qwen3-1.7B",
        model_path=qwen3_1_7b_base,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-4B-Base",
        model_config=qwen3_4b_config,
        tokenizer="Qwen/Qwen3-4B",
        model_path=qwen3_4b_base,
    ),
    ModelConfig(
        model_name="Qwen/Qwen3-8B-Base",
        model_config=qwen3_8b_config,
        tokenizer="Qwen/Qwen3-8B",
        model_path=qwen3_8b_base,
    ),
]


def get_directory_friendly_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(".", "-")


steps = []
for model_config in model_with_config:
    uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=model_config.tokenizer)
    eval_data = mixture_for_evaluation(uncheatable_eval_tokenized_dict)

    directory_friendly_name = get_directory_friendly_name(model_config.model_name)
    steps.append(
        default_lm_log_probs(
            checkpoint=model_config.model_path,
            model=model_config.model_config,
            data=eval_data,
            resource_config=SINGLE_TPU_V5p_8_FULL,
            checkpoint_is_hf=True,
            per_device_batch_size=1,
            name=f"{directory_friendly_name}-uncheatable-eval-logprobs",
            wandb_tags=[
                f"M={model_config.model_name[:62] if len(model_config.model_name) > 62 else model_config.model_name}",
                "eval=uncheatable-eval",
            ],
        )
    )


if __name__ == "__main__":
    for step in steps:
        executor_main(steps=[step])
