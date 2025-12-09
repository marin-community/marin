#!/usr/bin/env python3
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

"""Fine-tune several base models on the social scaling datasets.

This experiment runs short (1000-step) continued-pretraining runs on the
longitudinal, OSF, Psych-101, and WVS social scaling datasets. For each
dataset, we fine-tune:

* Llama 3 8B
* Qwen 2.5 (0.5B, 1.5B, 3B, 7B, 14B)
* OLMo 2 7B

All runs:
* initialize from the corresponding Hugging Face checkpoint
* train with global batch size 16
* train for 1000 steps
* use a small sweep of learning rates
* run on v5p-8"""

import dataclasses
from functools import lru_cache
from itertools import islice

import haliax

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer
from experiments.olmo2 import olmo_7b
from experiments.simple_train_config import SimpleTrainConfig
from experiments.social_scaling.social_scaling_datasets import tokenize_social_scaling
from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import QwenConfig
from marin.execution.executor import ExecutorStep, executor_main

QWEN2_5_HANDLES: dict[str, str] = {
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B",
    "qwen2_5_3b": "Qwen/Qwen2.5-3B",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B",
    #    "qwen2_5_14b": "Qwen/Qwen2.5-14B",
}

QWEN2_5_TOKENIZER = "Qwen/Qwen2.5-7B"

OLMO2_7B_HF = "allenai/OLMo-2-1124-7B"

LEARNING_RATES: tuple[float, ...] = (1e-5, 3e-5, 1e-4)

BATCH_SIZE = 4
NUM_TRAIN_STEPS = 1000
MAX_STEPS_PER_EXECUTOR = 4


@lru_cache(maxsize=1)
def build_model_configs() -> dict[str, tuple[str, object]]:
    """Build Levanter model configs for each HF checkpoint."""
    models: dict[str, tuple[str, object]] = {}

    llama3_8b_cfg = LlamaConfig().hf_checkpoint_converter(ref_checkpoint=llama3_tokenizer).default_config
    llama3_8b_cfg = dataclasses.replace(
        llama3_8b_cfg, gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload")
    )
    # models["llama3_8b"] = (llama3_tokenizer, llama3_8b_cfg)

    for name, handle in QWEN2_5_HANDLES.items():
        qwen_cfg = QwenConfig().hf_checkpoint_converter(ref_checkpoint=handle).default_config
        qwen_cfg = dataclasses.replace(
            qwen_cfg, gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload")
        )
        models[name] = (handle, qwen_cfg)

    olmo2_7b_cfg = dataclasses.replace(
        olmo_7b, gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload")
    )
    # models["olmo2_7b"] = (OLMO2_7B_HF, olmo2_7b_cfg)

    return models


@lru_cache(maxsize=1)
def build_tokenized() -> dict[str, dict[str, ExecutorStep]]:
    """Create tokenization steps for each model family and dataset."""
    tokenized_by_model: dict[str, dict[str, ExecutorStep]] = {}

    # Llama 3 8B uses its own tokenizer.
    tokenized_by_model["llama3_8b"] = tokenize_social_scaling(
        base_path="tokenized/social_scaling/llama3_8b",
        tokenizer=llama3_tokenizer,
    )

    # Qwen 2.5 family: all sizes share a single tokenizer.
    for model_name in QWEN2_5_HANDLES:
        tokenized_by_model[model_name] = tokenize_social_scaling(
            base_path=f"tokenized/social_scaling/{model_name}",
            tokenizer=QWEN2_5_TOKENIZER,
        )

    # OLMo 2 7B.
    tokenized_by_model["olmo2_7b"] = tokenize_social_scaling(
        base_path="tokenized/social_scaling/olmo2_7b",
        tokenizer=OLMO2_7B_HF,
    )

    return tokenized_by_model


def build_social_scaling_runs() -> list[ExecutorStep]:
    """Create fine-tuning runs for all models, datasets, and learning rates."""
    model_configs = build_model_configs()
    tokenized = build_tokenized()

    steps: list[ExecutorStep] = []

    for model_name, (hf_handle, model_cfg) in model_configs.items():
        datasets = tokenized[model_name]
        for dataset_key, tokenized_step in datasets.items():
            dataset_suffix = dataset_key.split("/")[-1]
            for lr in LEARNING_RATES:
                run_name = f"social_scaling/{model_name}/{dataset_suffix}/lr{lr:g}"

                train_cfg = SimpleTrainConfig(
                    resources=ResourceConfig.with_tpu("v5p-8"),
                    train_batch_size=BATCH_SIZE,
                    num_train_steps=NUM_TRAIN_STEPS,
                    learning_rate=lr,
                    initialize_from_hf=hf_handle,
                    steps_per_export=NUM_TRAIN_STEPS,
                )

                step = default_train(
                    name=run_name,
                    tokenized=tokenized_step,
                    model_config=model_cfg,
                    train_config=train_cfg,
                    tags=(
                        "social_scaling",
                        model_name,
                        dataset_key,
                    ),
                    use_default_validation=True,
                    eval_harness_tasks=[],
                )
                steps.append(step)

    return steps


def _chunked(steps: list[ExecutorStep], chunk_size: int) -> list[list[ExecutorStep]]:
    chunks: list[list[ExecutorStep]] = []
    iterator = iter(steps)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        chunks.append(chunk)
    return chunks


def main() -> None:
    """Entry point for the social scaling fine-tuning sweep."""
    steps = build_social_scaling_runs()
    for group in _chunked(steps, MAX_STEPS_PER_EXECUTOR):
        executor_main(steps=group)


if __name__ == "__main__":
    main()
