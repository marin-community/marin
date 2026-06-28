# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
#1600: Uncheatable Evals

This experiment evaluates models' perplexity on diverse, high-quality, and fresh datasets
(arXiv, GitHub, news) to better capture raw intelligence without relying on private datasets.

Reference: https://github.com/Jellyfish042/uncheatable_eval
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from fray.cluster import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner

from experiments.evals.hf_log_probs import default_hf_lm_log_probs
from experiments.evals.uncheatable import uncheatable_validation
from experiments.llama import llama3_tokenizer
from experiments.models import ModelConfig as HFModelConfig
from experiments.models import download_model

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str
    revision: str
    tokenizer: str | None = None  # Optional: if None, uses model_name as tokenizer


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
def build_steps() -> list[ArtifactStep[Artifact]]:
    steps: list[ArtifactStep[Artifact]] = []
    for model_config in models:
        tokenizer = model_config.tokenizer if model_config.tokenizer is not None else model_config.model_name
        validation_datasets = uncheatable_validation(tokenizer=tokenizer)

        model_checkpoint = download_model(
            HFModelConfig(hf_repo_id=model_config.model_name, hf_revision=model_config.revision)
        )
        directory_friendly_name = get_directory_friendly_name(model_config.model_name)
        steps.append(
            default_hf_lm_log_probs(
                hf_repo_id=model_config.model_name,
                hf_revision=model_config.revision,
                checkpoint=model_checkpoint,
                validation_datasets=validation_datasets,
                resource_config=ResourceConfig.with_tpu("v5p-8"),
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

    StepRunner().run([lower(step) for step in build_steps()])


if __name__ == "__main__":
    main()
