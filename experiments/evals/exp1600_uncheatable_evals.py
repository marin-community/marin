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

import os
import os.path
import logging
from dataclasses import dataclass
from functools import lru_cache


from experiments.llama import llama3_tokenizer
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, ExecutorStep, output_path_of
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import mixture_for_evaluation, TokenizerStep
from src.marin.download.uncheatable_eval.download import make_uncheatable_eval_step
from experiments.defaults import default_tokenize
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from experiments.models import ModelConfig as HFModelConfig, download_model_step

logger = logging.getLogger(__name__)

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
    revision: str
    tokenizer: str | None = None  # Optional: if None, uses model_name as tokenizer


# Evaluate log probabilities on uncheatable eval datasets
# Format: (model_name, revision, tokenizer)
# If tokenizer is None, uses model_name as tokenizer

models = [
    ModelConfig(model_name="marin-community/marin-8b-base", revision="main", tokenizer=llama3_tokenizer),
    ModelConfig(model_name="allenai/OLMo-2-0325-32B", revision="main"),
    ModelConfig(model_name="Qwen/Qwen3-32B", revision="main"),
    ModelConfig(model_name="meta-llama/Llama-3.1-8B", revision="main", tokenizer=llama3_tokenizer),
    ModelConfig(model_name="meta-llama/Llama-3.2-1B", revision="main", tokenizer=llama3_tokenizer),
    ModelConfig(model_name="allenai/OLMo-2-1124-7B", revision="main"),
    ModelConfig(model_name="Qwen/Qwen3-0.6B", revision="main"),
    ModelConfig(model_name="Qwen/Qwen3-1.7B", revision="main"),
    ModelConfig(model_name="Qwen/Qwen3-4B", revision="main"),
    ModelConfig(model_name="Qwen/Qwen3-8B", revision="main"),
    ModelConfig(model_name="Qwen/Qwen3-0.6B-Base", revision="main", tokenizer="Qwen/Qwen3-0.6B"),
    ModelConfig(model_name="Qwen/Qwen3-1.7B-Base", revision="main", tokenizer="Qwen/Qwen3-1.7B"),
    ModelConfig(model_name="Qwen/Qwen3-4B-Base", revision="main", tokenizer="Qwen/Qwen3-4B"),
    ModelConfig(model_name="Qwen/Qwen3-8B-Base", revision="main", tokenizer="Qwen/Qwen3-8B"),
]


def get_directory_friendly_name(model_name: str) -> str:
    return model_name.replace("/", "--").replace(".", "-")


def truncate_model_name(model_name: str, max_length: int = 62) -> str:
    """Truncate model name to max_length if it exceeds that length."""
    return model_name[:max_length] if len(model_name) > max_length else model_name


@lru_cache(maxsize=1)
def build_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for model_config in models:
        # Use model_name as tokenizer if not specified
        tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name

        uncheatable_eval_tokenized_dict = uncheatable_eval_tokenized(tokenizer=tokenizer)
        eval_data = mixture_for_evaluation(uncheatable_eval_tokenized_dict)

        # Download model and load config dynamically from HuggingFace
        model_identifier = f"{model_config.model_name}@{model_config.revision}"
        model_instance = download_model_step(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )
        hf_model_config = HFCheckpointConverter.from_hf(model_identifier).config_from_hf_checkpoint(model_identifier)

        directory_friendly_name = get_directory_friendly_name(model_config.model_name)
        steps.append(
            default_lm_log_probs(
                checkpoint=output_path_of(model_instance),
                model=hf_model_config,
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
    return steps


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    for step in build_steps():
        executor_main(steps=[step])


if __name__ == "__main__":
    main()
