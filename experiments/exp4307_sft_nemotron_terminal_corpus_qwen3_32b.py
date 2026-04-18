# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SFT reproduction of Nemotron-Terminal-32B using the full Terminal-Corpus on Qwen3-32B.

This launcher follows the full-corpus recipe from "On Data Engineering for Scaling
LLM Terminal Capabilities" (arXiv:2602.21193): 2 epochs, batch 128, lr 2e-5,
and 32,768-token SFT. The paper reports 27.4% on Terminal-Bench 2.0 for the
released Nemotron-Terminal-32B checkpoint. Qwen3-32B uses a 40,960-token default
context window at evaluation time; this launcher exports the final HF checkpoint
for downstream TB2 evaluation.

Usage:
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --tpu v5p-256 \
        --zone us-east5-a \
        -e MARIN_PREFIX gs://marin-us-east5 \
        -e WANDB_ENTITY marin-community \
        -e WANDB_PROJECT marin \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e HF_TOKEN ${HF_TOKEN} \
        -e TPU_CI true \
        --no-wait \
        -- python experiments/exp4307_sft_nemotron_terminal_corpus_qwen3_32b.py
"""

import dataclasses
import math
import os

import haliax
from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.qwen3 import qwen3_32b_hf, qwen3_32b_tokenizer, qwen3_8b_tokenizer
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

SUBSET_SIZES = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690,
}

DATASETS = {name: name for name in SUBSET_SIZES}
WEIGHTS = {name: float(size) for name, size in SUBSET_SIZES.items()}
EFFECTIVE_EXAMPLES = int(sum(WEIGHTS.values()))

TARGET_EPOCHS = 2
_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-256")
_NUM_CHIPS = int(_TPU_VARIANT.split("-")[-1]) // 2
TRAIN_BATCH_SIZE = 128
_TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
_NUM_DATA_PARALLEL = _NUM_CHIPS // _TENSOR_PARALLEL_SIZE
MICROBATCH_SIZE = min(128, _NUM_DATA_PARALLEL)
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * EFFECTIVE_EXAMPLES / TRAIN_BATCH_SIZE))

RESOURCES = ResourceConfig.with_tpu(_TPU_VARIANT)
RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"


def create_tokenization_step(dataset_identifier: str, short_name: str):
    # Qwen3-8B and Qwen3-32B share the same tokenizer (vocab_size=151643, identical encoding).
    # Reuse the 8B-tokenized data that already exists and is marked SUCCESS.
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name.split('/')[-1]}_qwen3_8b_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen3_8b_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
    )


tokenized_datasets = {name: create_tokenization_step(dataset_id, name) for name, dataset_id in DATASETS.items()}

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=qwen3_32b_tokenizer,
    initialize_from_hf="Qwen/Qwen3-32B",
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(
        TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES, _TENSOR_PARALLEL_SIZE
    ),
    per_device_eval_parallelism=1,
    tensor_parallel_size=_TENSOR_PARALLEL_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=2e-5,
    max_seq_len=32768,
    seed=42,
    steps_per_checkpoint=max(1, NUM_TRAIN_STEPS // 4),
    lr_schedule="cosine",
    warmup=0.1,
    decay=0.9,
    weight_decay=1e-4,
    beta1=0.9,
    beta2=0.95,
    max_grad_norm=1.0,
    pad_tokenizer_to_match_model=True,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    WEIGHTS,
    shuffle=EFFECTIVE_EXAMPLES,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

qwen3_32b_32k = dataclasses.replace(
    qwen3_32b_hf,
    max_seq_len=32768,
    gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload"),
    cross_entropy_block_size=32000,
)

exp4307_sft = default_sft(
    name=f"exp4307_nemotron_terminal_qwen3_32b_32768tok_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_32b_32k,
    sft_config=sft_config,
    tags=["qwen", "qwen3-32b", "nemotron-terminal", "terminal-corpus", "exp4307", RESOURCE_SUFFIX],
)

exp4307_checkpoint = exp4307_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp4307: Nemotron-Terminal-Corpus Qwen3-32B SFT ===")
    print(f"Datasets: {len(SUBSET_SIZES)} subsets")
    print(f"Effective examples: {EFFECTIVE_EXAMPLES:,}")
    print(f"Training steps: {NUM_TRAIN_STEPS:,}")
    print(f"Epochs: {TARGET_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Resources: {RESOURCES.device.variant}")
    executor_main(steps=[exp4307_sft])
