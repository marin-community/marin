# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SFT on MEnvData-SWE-Trajectory (OpenHands SWE-bench trajectories, Claude Sonnet 4.5).

Dataset: ernie-research/MEnvData-SWE-Trajectory (3,872 trajectories)
Base model: Qwen/Qwen3-8B
Training config: Same as 32K v2 OT-Agent SFT.

Dataset format conversion (handled by convert_openhands_xml_to_tool_calls):
    - XML <functions><function=name>...</function></functions> → structured tool_calls
    - reasoning_content field → passed through for <think> rendering
    - Bare tool messages → assigned tool_call_id/name from preceding assistant

Usage (Iris):
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-32 \
        --region us-central1 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e MARIN_PREFIX gs://marin-us-central1 \
        --no-wait \
        -- python experiments/exp_menvdata_sft_qwen3_8b.py
"""

import dataclasses
import math

from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.qwen3 import qwen3_8b, qwen3_8b_tokenizer
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

DATASET_ID = "ernie-research/MEnvData-SWE-Trajectory"
NUM_SAMPLES = 3872


def build_dataset_specs() -> tuple[dict[str, str], dict[str, float]]:
    datasets: dict[str, str] = {}
    weights: dict[str, float] = {}
    datasets[DATASET_ID] = DATASET_ID
    weights[DATASET_ID] = float(NUM_SAMPLES)
    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str):
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name.split('/')[-1]}_qwen3_8b_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen3_8b_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = int(sum(mixture_weights.values()))

# Training config matching 32K v2 OT-Agent SFT
TARGET_EPOCHS = 7
TRAIN_BATCH_SIZE = 16
MICROBATCH_SIZE = 16
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

RESOURCES = ResourceConfig.with_tpu("v5p-32")

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_8b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-8B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=4e-5,
    max_seq_len=32768,
    seed=42,
    steps_per_checkpoint=(total_examples // TRAIN_BATCH_SIZE) // 4,
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=0.0,
    max_grad_norm=1e-4,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-8,
    pad_tokenizer_to_match_model=True,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=total_examples,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

qwen3_8b_32k = dataclasses.replace(
    qwen3_8b,
    max_seq_len=32768,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
)

RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

exp_menvdata_sft = default_sft(
    name=f"exp_menvdata_sft_qwen3_8b_32768tokens_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_8b_32k,
    sft_config=sft_config,
    tags=["qwen", "menvdata", "sft", "openhands", "swebench", RESOURCE_SUFFIX],
)

exp_menvdata_checkpoint = exp_menvdata_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== MEnvData SFT: Qwen3-8B on OpenHands SWE-bench trajectories ===")
    print(f"Dataset: {DATASET_ID} ({NUM_SAMPLES:,} samples)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}")
    print(f"max_grad_norm: {sft_config.max_grad_norm}")
    executor_main(steps=[exp_menvdata_sft])
