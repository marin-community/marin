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
#1600: Uncheatable Evals

This experiment evaluates models' perplexity on diverse, high-quality, and fresh datasets
(arXiv, GitHub, news) to better capture raw intelligence without relying on private datasets.

Reference: https://github.com/Jellyfish042/uncheatable_eval
"""

import os.path
from dataclasses import dataclass

from experiments.llama import llama_3_2_1b as llama_3_2_1b_config, llama3_tokenizer, llama_8b
from levanter.models.llama import LmConfig
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, ExecutorStep
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import mixture_for_evaluation, TokenizerStep
from src.marin.download.uncheatable_eval.download import make_uncheatable_eval_step
from experiments.defaults import default_tokenize
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
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
from experiments.olmo2 import (
    olmo_7b,
    olmo_32b,
)
from experiments.tootsie.exp1529_32b_bison_cooldown import tootsie_32b_cooldown_bison as marin_32b_base


# Complete mapping of all available datasets
ALL_UNCHEATABLE_EVAL_DATASETS = {
    "wikipedia_arabic": "wikipedia_arabic_*.jsonl.gz",
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    "wikipedia_french": "wikipedia_french_*.jsonl.gz",
    "wikipedia_german": "wikipedia_german_*.jsonl.gz",
    "wikipedia_japanese": "wikipedia_japanese_*.jsonl.gz",
    "wikipedia_spanish": "wikipedia_spanish_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    "ao3_chinese": "ao3_chinese_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}

# We only include English and code datasets for this experiment
ACTIVE_DATASETS = [
    "wikipedia_english",
    "github_python",
    "github_cpp",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "ao3_english",
]

uncheatable_eval = make_uncheatable_eval_step()


def uncheatable_eval_tokenized(
    *, base_path="tokenized/", tokenizer: str = llama3_tokenizer, uncheatable_eval_raw: ExecutorStep = uncheatable_eval
) -> dict[str, TokenizerStep]:
    uncheatable_eval_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset in ACTIVE_DATASETS:
        path_part = ALL_UNCHEATABLE_EVAL_DATASETS[dataset]
        uncheatable_eval_steps[os.path.join("uncheatable_eval", dataset)] = default_tokenize(
            name=os.path.join("uncheatable_eval", dataset),
            dataset=uncheatable_eval_raw.cd(f"{path_part}"),
            tokenizer=tokenizer,
            is_validation=True,
        )

    return uncheatable_eval_steps


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


def truncate_model_name(model_name: str, max_length: int = 62) -> str:
    """Truncate model name to max_length if it exceeds that length."""
    return model_name[:max_length] if len(model_name) > max_length else model_name


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
                f"M={truncate_model_name(model_config.model_name)}",
                "eval=uncheatable-eval",
            ],
        )
    )


if __name__ == "__main__":
    for step in steps:
        executor_main(steps=[step])
